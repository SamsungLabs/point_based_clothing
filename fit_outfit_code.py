# Everywhere in our project "glo" stands for "Global Latent Optimization".
# "glo_vec" and "best_z" are the synonims of "outfit code" in the paper.
# ".glo_stack" attribute contains outfit codes for all the people from the current dataset.

import os
import cv2
import sys
import yaml
import pickle
import logging
import imageio
import argparse
import pandas as pd
from copy import deepcopy
from munch import munchify
from datetime import datetime

os.environ["DEBUG"] = ''

# sys.path.append('..')
sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/src')

import torch
from torch.utils.data import DataLoader

torch.autograd.set_detect_anomaly(True)

from dataloaders import internet_images
from dataloaders.loaders.internet_images_loader import Loader
from dataloaders.loaders.utils import make_dataloaders

from models.draping_network_wrapper import Wrapper
from models.nstack import NeuralStack, PointNeuralTex
from models.pcd_converter import PCDConverter
from models.pcd_renderer import Renderer

from runners.cloth import get_optimizer_glo_fitting

from outfit_code.utils import multiple_mean
from outfit_code.fit import multi_fit

from utils.argparse_utils import MyArgumentParser


logging.basicConfig(level=logging.INFO)

parser = MyArgumentParser(description='Fit an outfit code from a single image.', conflict_handler='resolve')
parser.add = parser.add_argument

parser.add('--config_name', type=str, default='outfit_code/psp',
           help='Path to the experiment config file.')

parser.add('--data_name', type=str, default='',
           help='Name of the dataset you wanna use. In current setup, there must exist\
           a loader for particularly this dataset. Create a new one if you use new dataset).')
parser.add('--data_root', type=str, default='',
           help='Path to the folder with segmentations and smpl parameters.\
           See README.md for the required data folder structure.')
parser.add('--rgb_dir', type=str, default='',
           help='Path to the images of clothing (relative to `args.data_root`)')
parser.add('--segm_dir', type=str, default='',
           help='Path to the segmentations of clothing (relative to `args.data_root`)')
parser.add('--smpl_dir', type=str, default='',
           help='Path to the SMPL parameters (relative to `args.data_root`)')

parser.add('--pids', type=str, default='',
           help='Comma-separated list of specific subject ids to fit outfit codes to. \
           Fit to all the dataset if not provided.')

parser.add('--smpl_model_path', type=str, default='./data/smpl_models/SMPL_NEUTRAL.pkl',
           help='Path to the .pkl of the neutral SMPL model.')
parser.add('--out_root', type=str, default='./out/outfit_code',
           help='Path to save the outfit codes and image paths.')
parser.add('--device', type=str, default='cuda:0',
           help='Computing device.')


if __name__ == '__main__':
    args = parser.parse_args()
    
    config_path = f'configs/{args.config_name}.yaml'
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    config = munchify(config)
    config.device = args.device
    
    config_fine = deepcopy(config)
    config_fine.n_people = 1
    
    # data setup
    args.data_root = config.data_root if args.data_root == "" else args.data_root
    args.data_name = args.data_root.split('/')[-1]
    args.rgb_dir = config.rgb_dir if args.rgb_dir == "" else args.rgb_dir
    args.segm_dir = config.segm_dir if args.segm_dir == "" else args.segm_dir
    args.smpl_dir = config.smpl_dir if args.smpl_dir == "" else args.smpl_dir
    
    # NOTE: currently the script supports only the samples with 1 frame to fit to
    if config.get('splits_dir') is None:
        pid_list_orig = list(filter(lambda x: x[0] != '.', os.listdir(f'{args.data_root}/{args.smpl_dir}')))
        pid_list = [x[:-4] for x in pid_list_orig]
        if args.pids != '':
            pid_list = [x for x in pid_list if x in args.pids.split(',')]
        datalists = [[pid] for pid in pid_list]
    else:
        pid_list_orig = list(filter(lambda x: x[0] != '.', os.listdir(f'{args.data_root}/{config.splits_dir}')))
        pid_list = deepcopy(pid_list_orig)  # [x['seq'].values for x in datalists]
        if args.pids != '':
            pid_list = [x for x in pid_list if x in args.pids.split(',')]
        datalists = [pd.read_csv(f'{args.data_root}/{config.splits_dir}/{pid}/fit1.csv') for pid in pid_list]
    
    # models setup
    args.smpl_model_path = config.smpl_model_path if args.smpl_model_path == "" else args.smpl_model_path
    converter = PCDConverter(config.device)
    ## Important: `visibility_thr` might highly affect the quality, consider changing it (see README.md for details)
    renderer = Renderer(height=config.image_size[0], width=config.image_size[1], 
                        pcd_features_dim=config.ntex_dim, visibility_thr=1e-3, device=config.device)
    
    # output files
    os.makedirs(args.out_root, exist_ok=True)
    outfit_codes_file = f'{args.out_root}/outfit_codes_{args.data_name}.pkl'
    imgs_info_file = f'{args.out_root}/image_paths_{args.data_name}.pkl'
    
    # vis setup
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    vis_root = f"{args.out_root}/vis_{args.data_name}"
    os.makedirs(vis_root, exist_ok=True)
    
    # main loop
    for i, pid in enumerate(pid_list):
        print(f'\n> Start fitting to: pid={pid}, frame={datalists[i]}')
        
        # > Coarse fitting stage: init 4 random outfit codes and optimize them independently
        print(f'\n> Coarse fitting stage: {config.max_iter_coarse} iterations\n')
        dataloader, dataloader_val = make_dataloaders(args.data_name, args.data_root, args.rgb_dir, args.segm_dir, 
                                                      args.smpl_dir, datalists[i], args.smpl_model_path, 
                                                      batch_size=config.n_people, max_iter=config.max_iter_coarse)
        
        draping_net = Wrapper.get_net(config)
        print(f'\n> Num outfit code vectors to be optimized in parallel: {len(draping_net.glo_stack.textures)}')
        for t in draping_net.glo_stack.textures:
            print(t.z)
        
        # dummy neural descriptors for Differentiable Rasterizer to work: 
        # they will not be optimized during the outfit code fitting procedure
        ndesc_stack = NeuralStack(1, lambda: PointNeuralTex(config.ntex_dim, config.pcl_size)).to(config.device)
        
        date_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        vid_path = f"{vis_root}/{pid}_coarse_{date_time}.mp4"
        video_imsize = (config.n_people * config.vis_image_size[1], config.vis_image_size[0])
        video_writer = cv2.VideoWriter(vid_path, fourcc, 30, video_imsize)
        
        optimizer, glo_scheduler = get_optimizer_glo_fitting(draping_net, config)
        model_tuple = (draping_net, converter, ndesc_stack, renderer, config.pretrained_ct_path)
        best_coarse, min_val_coarse = multi_fit(model_tuple, dataloader, dataloader_val, optimizer, glo_scheduler,
                                                config.max_iter_coarse, video_writer=video_writer, img_size=config.image_size[0])
        video_writer.release()
        print(f'\n> Best coarse chamfer (val set): {min_val_coarse}')
        
        # slerp mean of 4 found vectors
        glo_vecs = []
        for glo_vec in draping_net.glo_stack.textures:
            glo_vecs.append(glo_vec.z.data)
        glo_vec = torch.from_numpy(multiple_mean(glo_vecs)).cuda()
        
        # > Fine fitting stage: optimize `glo_vec` starting from its best mean coarse value
        print(f'\n> Fine fitting stage: {config.max_iter_fine} iterations\n')
        dataloader, dataloader_val = make_dataloaders(args.data_name, args.data_root, args.rgb_dir, args.segm_dir, 
                                                      args.smpl_dir, datalists[i], args.smpl_model_path, 
                                                      batch_size=config_fine.n_people, max_iter=config.max_iter_fine)
        
        draping_net = Wrapper.get_net(config_fine)
        draping_net.glo_stack.textures[0].z.data = glo_vec
        
        vid_path = f"{vis_root}/{pid}_fine_{date_time}.mp4"
        video_imsize = (config.vis_image_size[1],  config.vis_image_size[0])
        video_writer = cv2.VideoWriter(vid_path, fourcc, 30, video_imsize)

        optimizer, glo_scheduler = get_optimizer_glo_fitting(draping_net, config)
        model_tuple = (draping_net, converter, ndesc_stack, renderer, config.pretrained_ct_path)
        best_z, min_val = multi_fit(model_tuple, dataloader, dataloader_val, optimizer, glo_scheduler,
                                    config.max_iter_fine, video_writer=video_writer, img_size=config.image_size[0])
        video_writer.release()

        # > Save the best found outfit code to the `outfit_codes_file`
        if min_val_coarse < min_val:
            best_z = best_coarse
            print(f'\n> Final after fitting (coarse is better than fine): {min_val_coarse}')
        else:
            print(f'\n> Final after fitting (fine is better than coarse): {min_val}')

        if os.path.exists(outfit_codes_file):
            with open(outfit_codes_file, 'rb') as f:
                outfit_codes_dict = pickle.load(f)
        else:
            outfit_codes_dict = {}
        outfit_codes_dict[pid] = best_z
        with open(outfit_codes_file, 'wb') as f:
            pickle.dump(outfit_codes_dict, f)
            print(f'\n> Outfit code for {pid} saved to {outfit_codes_file}')
            
        if os.path.exists(imgs_info_file):
            with open(imgs_info_file, 'rb') as f:
                imgs_info_dict = pickle.load(f)
        else:
            imgs_info_dict = {}
        imgs_info_dict[pid] = next(iter(dataloader))[0]['rgb_path'][0]
        with open(imgs_info_file, 'wb') as f:
            pickle.dump(imgs_info_dict, f)
            print(f'\n> Image path {imgs_info_dict[pid]} for {pid} saved to {imgs_info_file}')
