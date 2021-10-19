import os
import sys
import time
import math
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from munch import munchify

sys.path.append('..')

import torch

from pytorch3d.loss.chamfer import chamfer_distance

from .utils import verts2cam, sample_visible_points, mask_to_2dpcd
from .utils import sample_tensor, oversample_tensor

from utils.common import dict2device
from utils.vis_utils import visualize_code_fitting


os.environ["DEBUG"] = ''


def forward_pass(data_dict, draping_net, converter, ndesc_stack, renderer, device=torch.device('cuda:0'), outfit_code=None):
    data_dict = dict2device(data_dict, device)
    
    # draping network inference
    data_dict = converter.source_to_normalized_dict(data_dict)
    glo_out = draping_net(data_dict, outfit_code=outfit_code)  # 'cloth_pcd'
    data_dict.update(glo_out)
    data_dict = converter.normalized_result_to_azure_dict(data_dict)

    # differentiable soft rasterizer inference
    pids = data_dict['seq']
    ndesc_stack.move_to(pids, device)
    ndesc = ndesc_stack.generate_batch(pids).permute(0, 2, 1)
    data_dict['ndesc'] = ndesc
    raster_dict = renderer.render_dict(data_dict)
    data_dict.update(raster_dict)

    return data_dict


def calc_chamfer(data_dict, target_dict, video_writer=None, pids=None, img_size=256):
    cloth_pcd = data_dict['cloth_pcd']
    K = data_dict['K']
    visibility_mask = data_dict['visibilty_mask']
    
    cloth_pcd_visible = sample_visible_points(cloth_pcd, visibility_mask)
    cloth_pcd_visible_cam = verts2cam(cloth_pcd_visible, K)[..., :2]

    real_segm = target_dict['real_segm']
    segm_pcds_sampled = mask_to_2dpcd(real_segm)
    
    if video_writer is not None:
        visualize_code_fitting(real_segm, cloth_pcd_visible_cam, video_writer, pids=pids)

    chamfer = chamfer_distance(cloth_pcd_visible_cam / (img_size / 10), segm_pcds_sampled / (img_size / 10))[0]

    return chamfer


def select_best(model, dataloader_val, glo_vecs, device=torch.device('cuda:0'), img_size=256):
    '''
    Select the best GLO vector (outfit code) by evaluation of the Chamfer loss on validation dataset.
    '''
    print('Evaluating on valid set')
    
    draping_net, converter, ndesc_stack, renderer, pretrained_ct_path = model
    min_val = np.inf
    best_z = None

    for glo_id, glo_vec in enumerate(glo_vecs):
        with torch.no_grad():
            cham_val = []

            for data_dict_val, target_dict_val in dataloader_val:
                data_dict_val = dict2device(data_dict_val, device)
                target_dict_val = dict2device(target_dict_val, device)
                data_dict_val = forward_pass(data_dict_val, draping_net, converter, ndesc_stack, renderer, 
                                             device=device, outfit_code=glo_vec)

                cham_v = calc_chamfer(data_dict_val, target_dict_val) * ((img_size / 10) ** 2)
                cham_val.append(cham_v.item())

            if np.mean(cham_val) < min_val:
                min_val = np.max(cham_val)
                best_z = glo_vec.detach().cpu().numpy()

    return best_z, min_val


def multi_fit(model, dataloader, dataloader_val, optimizer, glo_scheduler, 
              max_iter=100, device=torch.device('cuda:0'), video_writer=None, 
              eval_each_iter=20, img_size=256):
    draping_net, converter, ndesc_stack, renderer, pretrained_ct_path = model
    num_epochs = math.ceil(max_iter / len(dataloader))
    global_it = 0

    min_val = np.inf
    best_z = None

    for epoch in range(num_epochs):        
        for it, (data_dict, target_dict) in enumerate(tqdm(dataloader)):
            out_dict = dict(data_dict=data_dict, target_dict=target_dict)
            data_dict = dict2device(data_dict, device)
            target_dict = dict2device(target_dict, device)

            data_dict = forward_pass(data_dict, *model[:-1], device=device)
            
            chamfer = calc_chamfer(data_dict, target_dict, video_writer=video_writer, 
                                   pids=data_dict.get('pids_order'), img_size=img_size)

            loss = chamfer

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_it > max_iter:
                print('multifit ended, min_val', min_val)
                return best_z, min_val

            if global_it % eval_each_iter == 0:
                with torch.no_grad():
                    glo_vecs = []
                    for glo_vec in draping_net.glo_stack.textures:
                        glo_vecs.append(glo_vec.z.data)

                    best_z_c, min_val_c = select_best(model, dataloader_val, glo_vecs, 
                                                      device=device, img_size=img_size)

                    if min_val_c < min_val:
                        best_z = best_z_c
                        min_val = min_val_c
                        print('multifit updated with', min_val)
            global_it += 1
            glo_scheduler.step()

    print('min_val', min_val)

    return best_z, min_val
