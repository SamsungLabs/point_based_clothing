import os
import sys
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ["DEBUG"] = ''

import matplotlib as mpl

mpl.use('agg')

import torch
from torch import nn

sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/src')

from models.nstack import NeuralStack, PointNeuralTex

from utils.utils import setup, get_args_and_modules, save_model, save_ntex
from utils.tensorboard_logging import setup_logging
from utils.argparse_utils import MyArgumentParser
from utils.defaults import DEFAULTS
from utils.saver import Saver
from utils import distributed

import logging

import warnings

warnings.filterwarnings("ignore")

torch.autograd.set_detect_anomaly(True)

logging.basicConfig(level=logging.INFO)


parser = MyArgumentParser(conflict_handler='resolve')
parser.add = parser.add_argument

parser.add('--config_name', type=str, default="")

parser.add('--smpl_model_path', type=str, default='./data/smpl_models/SMPL_NEUTRAL.pkl', help='')
parser.add('--generator', type=str, default="", help='')
parser.add('--draping_network', type=str, default="draping_network_wrapper", help='')
parser.add('--discriminator_list', type=str, default="", help='')
parser.add('--criterions', type=str, default="", help='')
parser.add('--dataloader', type=str, default="", help='')
parser.add('--runner', type=str, default="", help='')
parser.add('--wscheduler', type=str, default="", help='')

parser.add('--pcd_size', type=int, default=4096 * 2)

parser.add('--args-to-ignore', type=str,
           default="checkpoint_path,splits_dir,experiments_dir,extension,"
                   "experiment_name,rank,local_rank,world_size,num_gpus,num_workers,batch_size,distributed")
parser.add('--experiments_dir', type=Path, default=DEFAULTS.logdir, help='')
parser.add('--experiment_name', type=str, default="", help='')
parser.add('--splits_dir', default="data/splits", type=str)

parser.add('--vgg_weights_dir', default=DEFAULTS.vgg_weights_dir, type=str)

# Training process
parser.add('--num_epochs', type=int, default=200)
parser.add('--set_eval_mode_in_train', action='store_bool', default=False)
parser.add('--set_eval_mode_in_test', action='store_bool', default=True)
parser.add('--save_frequency', type=int, default=1, help='')
parser.add('--logging', action='store_bool', default=True)
parser.add('--skip_eval', action='store_bool', default=False)
parser.add('--eval_frequency', type=int, default=1)

# Hardware
parser.add('--device', type=str, default='cuda')
parser.add('--num_gpus', type=int, default=1, help='requires apex if > 1, requires horovod if > 8')

parser.add('--local_rank', type=int, default=0, help='global rank of the device, DO NOT SET')
parser.add('--world_size', type=int, default=1, help='number of devices, DO NOT SET')

# Misc
parser.add('--random_seed', type=int, default=123, help='')
parser.add('--checkpoint_path', type=str, default='')
parser.add('--saver', type=str, default='')
parser.add('--safe_kill', action='store_bool', default=True)

args, default_args, m = get_args_and_modules(parser)

# Set random seed, number of threads etc.
setup(args)

args.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
args.num_gpus = args.world_size
args.distributed = args.world_size > 1

# In case of distributed training we first initialize rank and world_size
if args.distributed and 1 < args.num_gpus <= 8:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    distributed.synchronize()

writer = None
# if args.logging and args.local_rank == 0:
args.experiment_dir, writer = setup_logging(
    args, default_args, args.args_to_ignore.split(','))
args.experiment_dir = Path(args.experiment_dir)

if args.local_rank != 0:
    writer = None

dataloader_val = m['dataloader'].get_dataloader(args, part='val', phase='val')

criterion_list = [crit.get_net(args) for crit in m['criterion_list']]
runner = m['runner']

if args.checkpoint_path != '':
    if os.path.exists(args.checkpoint_path):
        raise NotImplementedError
    else:
        raise FileNotFoundError(f"Checkpoint `{args.checkpoint_path}` not found")
else:
    discriminator_list = [disc.get_net(args) for disc in m['discriminator_list']]
    generator = m['generator'].get_net(args)
    # this is a hack for now
    dataset_train = m['dataloader'].dataset.get_dataset(args, part='train')
    args.frames = dataset_train.datalist['id']
    # print(args.frames)
    # print(dataset_train.datalist['seq'].nunique())
    del dataset_train
    ###
    draping_network = m['draping_network'].get_net(args)

    args.wsc_step = 0
    wscheduler = None if m['wscheduler'] is None else m['wscheduler'].get_scheduler(args)

    ndesc_stack = NeuralStack(args.n_people, lambda: PointNeuralTex(args.ntex_channels, args.pcd_size))
    optimizer_G = runner.get_optimizer(generator, ndesc_stack, draping_network, args)
    optimizer_D_list = [wrapper.get_optimizer(disc, args) for wrapper, disc in
                        zip(m['discriminator_list'], discriminator_list)]

training_module = runner.TrainingModule(generator, draping_network, discriminator_list, criterion_list, args)

# If someone tries to terminate the program, let us save the weights first
if args.local_rank == 0 and args.safe_kill:
    import signal, sys, os

    parent_pid = os.getpid()


    def save_last_model_and_exit(_1, _2):
        if os.getpid() == parent_pid:  # otherwise, dataloader workers will try to save the model too!
            save_model(training_module, optimizer_G, optimizer_D_list, epoch - 1, args)
            # protect from Tensorboard's "Unable to get first event timestamp
            # for run `...`: No event timestamp could be found"
            if writer is not None:
                writer.close()
            sys.exit()


    signal.signal(signal.SIGINT, save_last_model_and_exit)
    signal.signal(signal.SIGTERM, save_last_model_and_exit)

if args.distributed and 1 < args.num_gpus <= 8:
    training_module = nn.parallel.DistributedDataParallel(
        training_module,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        broadcast_buffers=False
    )

# Optional results saver
saver = None
if args.saver:
    saver = Saver(save_dir=f'{args.experiment_dir}/validation_results/', save_fn=args.saver)

# Main loop
for epoch in range(0, args.num_epochs):
    # ===================
    #       Train
    # ===================

    training_module.train(not args.set_eval_mode_in_train)
    torch.set_grad_enabled(True)

    dataloader_train = m['dataloader'].get_dataloader(args, part='train', phase='train')

    runner.run_epoch(dataloader_train, training_module, ndesc_stack, optimizer_G, optimizer_D_list,
                     epoch, args, phase='train', writer=writer, saver=saver, wscheduler=wscheduler)

    if not args.skip_eval and (epoch + 1) % args.eval_frequency == 0:
        # ===================
        #       Validate
        # ===================
        training_module.train(not args.set_eval_mode_in_test)
        torch.set_grad_enabled(False)

        dataloader_val = m['dataloader'].get_dataloader(args, part='val', phase='val')
        runner.run_epoch(dataloader_val, training_module, ndesc_stack, None, None,
                         epoch, args, phase='val', writer=writer, saver=saver)

    # Save
    if (epoch + 1) % args.save_frequency == 0:
        save_model(training_module, optimizer_G, optimizer_D_list, epoch, args, wscheduler=wscheduler,
                   name=f'model{args.local_rank}')
        save_ntex(ndesc_stack, epoch, args, name=f'ntex{args.local_rank}')
