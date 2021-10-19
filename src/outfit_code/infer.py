import torch
import numpy as np
from skimage import io

from .fit import forward_pass
from .notebook_utils import draw_pcd_bg, show_nb

from utils.common import tti, to_sigm, itt, dict2device


def infer_pid(dataloader, outfit_codes_dict, image_paths_dict, 
              draping_network, converter, ndesc_stack, renderer,
              pid, device='cuda:0'):
    '''
    Infer the draping network and renderer (rasterizer) to predict the clothing point cloud with visible points.
    '''
    
    # > predict
    outfit_code = torch.from_numpy(outfit_codes_dict[pid]).to(device)
    print(f'Current style: pid={pid}, shape={outfit_code.shape}')
    
    for i, (data_dict, target_dict) in enumerate(dataloader):
        data_dict = dict2device(data_dict, device)
    data_dict['zrotMatrix_c3d'] = None
    
    source_pcd = data_dict['source_pcd'][0]
    
    out_dict = forward_pass(data_dict, draping_network, converter, ndesc_stack, renderer, 
                            device=device, outfit_code=outfit_code)

    # > visualize
    K = data_dict['K'].squeeze(0).to(device).float()
    
    source_pcd = source_pcd @ K.T
    source_pcd[:, :2] /= source_pcd[:, 2:]

    cloth_pcd = out_dict['cloth_pcd'][0]
    cloth_pcd = cloth_pcd[out_dict['visibilty_mask'][0]]
    cloth_pcd = cloth_pcd @ K.T
    cloth_pcd[:, :2] /= cloth_pcd[:, 2:]
    
    cloth_mask = tti(to_sigm(target_dict['real_segm']))
    cloth_mask = np.tile(cloth_mask[:,:,None], (1,1,3))
    
    smpl_img = draw_pcd_bg(cloth_mask, source_pcd[:,:2])
    cloth_img = draw_pcd_bg(cloth_mask, cloth_pcd[:,:2])
    
    rgb_img = io.imread(image_paths_dict[pid])
    
    show_nb([rgb_img, smpl_img, cloth_img], 
            title=f'Outfit point cloud fitted to a single image', 
            titles=['rgb', 'source pcd (cutted smpl)', 'outfit pcd (visible points only)'], n_cols=3)
