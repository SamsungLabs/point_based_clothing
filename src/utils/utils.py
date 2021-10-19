import importlib
import logging
import os
import random
from collections import defaultdict

import cv2
import imgaug as ia
import numpy as np
import torch
import yamlenv
from huepy import green


def setup(args):
    torch.set_num_threads(1)
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    os.environ['OMP_NUM_THREADS'] = '1'

    if args.random_seed is None:
        args.random_seed = random.randint(1, 10000)

    print("Random Seed: ", args.random_seed)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    if args.device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.cuda.manual_seed_all(args.random_seed)

    ia.seed(args.random_seed)


def get_args_and_modules(parser, phase='train', saved_args=None):
    '''
        Gathers args from modules and config
    '''

    args_, _ = parser.parse_known_args()

    if phase == 'test':
        saved_args = load_saved_args(args_.experiment_dir, args_) if saved_args is None else saved_args

    # Update main defaults
    update_defaults_fn = get_update_defaults_fn(saved_args, args_) if phase == 'test' else load_config(
        args_.config_name,
        args_)

    if update_defaults_fn is not None:
        update_defaults_fn(parser)

    # Parse with new defaults
    args_, _ = parser.parse_known_args()

    # Add generator args
    m_generator = load_module('generators', args_.generator).Wrapper
    m_generator.get_args(parser)

    # Add draping network args
    m_draping_network = load_module('models', args_.draping_network).Wrapper
    m_draping_network.get_args(parser)

    # Add runner args
    m_runner = load_module('runners', args_.runner)
    m_runner.get_args(parser)

    # Add discriminator args
    m_discriminator_list = load_discriminator_list(args_.discriminator_list)
    for disc in m_discriminator_list:
        disc.get_args(parser)
    # m_discriminator = load_module('discriminators', args_.discriminator).Wrapper
    # m_discriminator.get_args(parser)

    # Add criterion args
    m_criterion_list = load_criterion_list(args_.criterions)
    for crit in m_criterion_list:
        crit.get_args(parser)

    m_dataloader = load_module('dataloaders', 'dataloader').Dataloader(args_.dataloader)
    m_dataloader.get_args(parser)

    if 'wscheduler' in args_:
        m_wscheduler = load_module('wschedulers', args_.wscheduler).Wrapper
        m_wscheduler.get_args(parser)
    else:
        m_wscheduler = None

    update_defaults_fn = get_update_defaults_fn(saved_args) if phase == 'test' else load_config(args_.config_name,
                                                                                                args_)
    if update_defaults_fn is not None:
        update_defaults_fn(parser)

    # Finally parse everything
    args, default_args = parser.parse_args(), parser.parse_args([])

    return args, default_args, dict(generator=m_generator,
                                    draping_network=m_draping_network,
                                    runner=m_runner,
                                    dataloader=m_dataloader,
                                    discriminator_list=m_discriminator_list,
                                    criterion_list=m_criterion_list,
                                    wscheduler=m_wscheduler)


def load_saved_args(experiment_dir, args):
    yaml_config = f'{experiment_dir}/args.yaml'

    print((f'Using config {green(yaml_config)}'))

    with open(yaml_config, 'r') as stream:
        config = yamlenv.load(stream)

    return config


def get_update_defaults_fn(config, args={}):
    if isinstance(config, str):
        with open(config, 'r') as stream:
            config = yamlenv.load(stream, vars(args))

    def update_defaults_fn(parser):
        parser.set_defaults(**config)
        return parser

    return update_defaults_fn


def load_config(config_name, args):
    config = f'configs/{config_name}.yaml'

    if os.path.exists(config):
        print((f'Using config {green(config)}'))
        return get_update_defaults_fn(config, args)
    else:
        raise Exception(f'Did not find config {green(config)}')


def load_module(module_type, module_name):
    m = importlib.import_module(f'{module_type}.{module_name}')

    return m


def load_discriminator_list(module_name_list):

    if module_name_list == '':
        return []

    discriminator_names = module_name_list.split(',')
    discriminator_names = [d.strip() for d in discriminator_names]

    ms = []
    for dn in discriminator_names:
        m = importlib.import_module(f'discriminators.{dn}')
        ms.append(m.Wrapper)

    return ms


def load_criterion_list(module_name_list):
    criterion_names = module_name_list.split(',')
    criterion_names = [c.strip() for c in criterion_names]

    ms = []
    for cn in criterion_names:
        m = importlib.import_module(f'criterions.{cn}')
        ms.append(m.Wrapper)

    return ms


def load_wscheduler(wsc_name):
    m = importlib.import_module(f'weight_schedulers.{cn}')

    return m


class Meter:
    def __init__(self):
        super().__init__()
        self.data = defaultdict(list)
        self.accumulated = ['topk']

    def update(self, name, val):
        self.data[name].append(val)

    def get_avg(self, name):
        if name in self.accumulated:
            return self.get_last(name)

        return np.mean(self.data[name])

    def get_last(self, name):
        return self.data[name][-1]


def get_state_dict(net, distributed):
    if distributed:
        return net.module.state_dict()
    else:
        return net.state_dict()


def save_model(training_module, optimizer_G, optimizer_D_list, epoch, args, wscheduler=None, ntex_stack=None,
               name='model'):
    if args.num_gpus > 1 and training_module is not None:
        training_module = training_module.module
    # if training_module is not None:
    # training_module.cpu()

    save_dict = {}
    if training_module is not None:
        if training_module.generator is not None:
            save_dict['generator'] = training_module.generator.state_dict()
        if training_module.draping_network is not None:
            save_dict['draping_network'] = training_module.draping_network.state_dict()
        if training_module.discriminator_list is not None:
            save_dict['discriminator_list'] = [disc.state_dict() for disc in training_module.discriminator_list]
    if optimizer_G is not None:
        save_dict['optimizer_G'] = optimizer_G.state_dict()
    if optimizer_D_list is not None:
        save_dict['optimizer_D_list'] = [opt_D.state_dict() for opt_D in optimizer_D_list]
    if wscheduler is not None:
        args.wsc_step = wscheduler.step
    if ntex_stack is not None:
        save_dict['ntex_stack'] = ntex_stack.state_dict()
    if args is not None:
        save_dict['args'] = args

    epoch_string = f'{epoch:04}' if type(epoch) is int else epoch
    save_path = f'{args.experiment_dir}/checkpoints/{name}_{epoch_string}.pth'
    torch.save(save_dict, save_path, pickle_protocol=-1)

    # if training_module is not None:
    # training_module.to(args.device)


def save_ntex(ntex_stack, epoch, args, name='ntex'):
    keys = ntex_stack.pid2ntid.keys()
    tex_modules = {}

    for k in keys:
        tm = ntex_stack.get_texmodule(k).state_dict()
        tex_modules[k] = tm

    epoch_string = f'{epoch:04}' if type(epoch) is int else epoch
    save_path = f'{args.experiment_dir}/checkpoints/{name}_{epoch_string}.pth'
    torch.save(tex_modules, save_path, pickle_protocol=-1)


def load_ntex(generator_cp_path, args):
    cp_path, cp_name = os.path.split(generator_cp_path)
    cp_name = os.path.splitext(cp_name)[0]
    cp_ep = cp_name.split('_')[-1]

    ntexs = os.listdir(cp_path)
    ntexs = [x for x in ntexs if cp_ep in x and 'ntex' in x]

    nstack = ntex.NeuralTexStack(args.n_people, texsegm_path=args.texsegm_path)
    for ntex_batch in ntexs:
        ntex_path = os.path.join(cp_path, ntex_batch)
        sd = torch.load(ntex_path)

        for k in sd.keys():
            tm = nstack.get_texmodule(k)
            tm.load_state_dict(sd[k])

    return nstack


def load_model_from_checkpoint(checkpoint_path, device, inference=False):
    raise NotImplementedError
