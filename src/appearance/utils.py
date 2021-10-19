import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from utils.common import dict2device, itt, tti, to_sigm


def get_smpl_render(smpl_verts, K, renderer):
    KRT = torch.cat([K, torch.zeros_like(K[..., :1])], dim=-1).cuda()
    
    smpl_rast, _, _ = renderer.get_smpl_rast(smpl_verts, renderer.faces, KRT)
    smpl_rast = smpl_rast[..., 2:3]
    
    return smpl_rast


def get_smpl_depth(renderer, data_dict):
    vw = data_dict['smpl_verts_world']
    K = data_dict['K']
    
    KRT = torch.cat([K, torch.zeros_like(K[..., :1])], dim=-1).cuda()
    
    smpl_depth, _, _ = renderer.get_smpl_rast(vw, renderer.faces, KRT)
    smpl_depth = smpl_depth[..., 2:3]
    
    return smpl_depth


def get_cloth_depth(data_dict, max_kernel=31):
    smpl_depth = data_dict['smpl_depth'].clone()
    cloth_depth = data_dict['cloth_depth'].clone()
    dmin = data_dict['dmin'][:, None, None, None]

    smpl_depth[smpl_depth == 0.] = 100.
    cloth_depth[cloth_depth < dmin] = 100
    k = max_kernel

    cloth_depth = -torch.nn.functional.max_pool2d(-cloth_depth, kernel_size=k, stride=1, padding=k//2)

    return cloth_depth
    

def get_flooded_cloth(renderer, data_dict, T=2e-3):
    cloth_depth = get_cloth_depth(data_dict)
    smpl_depth = data_dict['smpl_depth'].clone()

    pcd_masks = data_dict['flood_raster_mask']
    B = pcd_masks.shape[0]

    cloth_masks = []
    for i in range(B):
        pcd_mask_b = pcd_masks[i:i + 1].clone()
        pcdmask_b_np = tti(pcd_mask_b).clip(0., 1.) > 0
        pcdmask_b_np = pcdmask_b_np.astype(np.uint8)
        mask1 = np.zeros((pcdmask_b_np.shape[0] + 2, pcdmask_b_np.shape[1] + 2)).astype(np.uint8)
        _, cloth_mask_np, _, _ = cv2.floodFill(pcdmask_b_np, mask1, (0, 0), 255)
        cloth_mask = itt(cloth_mask_np).unsqueeze(0)
        cloth_masks.append(cloth_mask)
    cloth_masks = torch.cat(cloth_masks).to(pcd_masks.device) / 255.

    cloth_masks_inv = 1. - cloth_masks
    cloth_masks_inv = (cloth_masks_inv > 0).float()
    
    cloth_depth_f = cloth_depth * cloth_masks_inv + 100 * (1. - cloth_masks_inv)
    visibility_flood_mask = (cloth_depth_f - T) < smpl_depth + (smpl_depth == 0)
    
    data_dict['flood_cloth_mask'] = visibility_flood_mask
    data_dict['smpl_depth'] = smpl_depth
    
    return data_dict


def ndesc2pca(ndesc, model=None):
    ndesc = ndesc[0].detach().cpu().numpy()
    
    if model is None:
        model = PCA(3)        
        ndesc_pca = model.fit_transform(ndesc)
    else:
        ndesc_pca = model.transform(ndesc)
    ndesc_pca = torch.FloatTensor(ndesc_pca).unsqueeze(0).cuda()
    
    return ndesc_pca, model


def to_img(img_list):
    return [(x * 255).astype(np.uint8) for x in img_list]


def to_tensor(img_list):
    return [x / 255. for x in img_list]


def show_axes(img_list):
    ncols = 3
    nrows = len(img_list) // ncols + int(len(img_list) % ncols > 0)

    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 10, nrows * 10))

    for i in range(len(img_list)):
        axs[i // ncols][i % ncols].imshow(img_list[i])
        axs[i // ncols][i % ncols].axis('off')

    plt.show();
