import torch
import numpy as np
import matplotlib.pyplot as plt

from .utils import get_smpl_depth, get_flooded_cloth, get_smpl_render, ndesc2pca

from utils.common import dict2device, itt, tti, to_sigm


def infer_geometry(data_dict, converter, draping_network, outfit_code):
    '''
    Predict outfit point cloud given cutted smpl (encodes pose and shape) and outfit code (encodes style).
    
    Args:
        data_dict: contains all the data needed to predict the point cloud.
        outfit_code: latent code that controls the style of an outfit.
    '''
    
    data_dict = converter.source_to_normalized_dict(data_dict)
    
    draping_network.eval()
    with torch.no_grad():
        # the point cloud is in glo_out['cloth_pcd']
        glo_out = draping_network(data_dict, outfit_code=outfit_code)  
    data_dict.update(glo_out)
    
    data_dict = converter.normalized_result_to_azure_dict(data_dict)
    
    return data_dict


def infer_appearance(data_dict, ndesc_stack, renderer, renderer_flood, generator, device):
    '''
    Predict image of an outfit given a point cloud and neural descriptors.
    
    Args:
        data_dict: contains all the data needed to render the clothing image.
    '''
    
    # get neural descriptors for each point
    ndesc_stack.move_to(data_dict['seq'], device)
    ndesc = ndesc_stack.generate_batch(data_dict['seq']).permute(0, 2, 1)
    data_dict['ndesc'] = ndesc
    
    # get rasterization info
    data_dict['smpl_depth'] = get_smpl_depth(renderer_flood, data_dict).permute(0, 3, 1, 2)
    data_dict['cloth_depth'], data_dict['dmin'] = renderer_flood.render_cloth_depth(data_dict)
    
    # infer differentiable rasterizer (`renderer`) and neural renderer (`generator`)
    with torch.no_grad():
        raster_dict = renderer.render_dict(data_dict)
        data_dict.update(raster_dict)
        
        flood_raster_dict = renderer_flood.render_dict(data_dict)
        for k, v in flood_raster_dict.items():
            data_dict['flood_' + k] = v
        
        g_out = generator(data_dict)
    
    return data_dict, g_out


def get_pca_raster_dict(data_dict, renderer):
    pca_dict = dict()
    pca_dict.update(data_dict)

    ndesc_pca, pca_model = ndesc2pca(data_dict['ndesc'], model=None)
    pca_dict['ndesc'] = ndesc_pca

    pca_raster_dict = renderer.render_dict(pca_dict)

    return pca_raster_dict


def postprocess(data_dict, renderer):
    out_dict = dict()
    out_dict.update(data_dict)
    
    smpl_rast_plt = tti(data_dict['smpl_rast'])
    smpl_rast_plt = smpl_rast_plt ** 0.5
    out_dict['smpl_rast_plt'] = smpl_rast_plt

    pca_raster_dict = get_pca_raster_dict(data_dict, renderer)
    raster_features_np = tti(pca_raster_dict['raster_features']).clip(0., 1.)
    raster_mask_np = tti(pca_raster_dict['raster_mask']).clip(0., 1.)
    out_dict['raster_features_np'] = raster_features_np
    out_dict['raster_mask_np'] = raster_mask_np

    rgb_np = tti(to_sigm(data_dict['fake_rgb']))
    segm_np = tti(data_dict['fake_segm']) * tti(data_dict['flood_cloth_mask'])
    out_dict['rgb_np'] = rgb_np
    out_dict['segm_np'] = segm_np
    
    real_rgb_np = tti(to_sigm(data_dict['real_rgb']))
    real_segm_np = tti(data_dict['real_segm'])
    real_rgb_nos_np = tti(to_sigm(data_dict['real_rgb_nos']))
    out_dict['real_rgb_np'] = real_rgb_np
    out_dict['real_segm_np'] = real_segm_np
    out_dict['real_rgb_nos_np'] = real_rgb_nos_np
    
    return out_dict


def do_overlays(data_dict):
    real_rgb_np = data_dict['real_rgb_np']
    real_segm_np = data_dict['real_segm_np']
    rgb_np = data_dict['rgb_np']
    segm_np = data_dict['segm_np']
    raster_mask_np = data_dict['raster_mask_np']
    raster_features_np = data_dict['raster_features_np']
    smpl_rast_plt = data_dict['smpl_rast_plt']
    
    common_mask = (segm_np[..., None] + (smpl_rast_plt[..., None] > 0)).clip(0.,1.)
    rgb_np = rgb_np * segm_np[..., None] + smpl_rast_plt[..., None] * (1. - segm_np[..., None])
    rgb_np = rgb_np * common_mask + 1 * (1 - common_mask)

    raster_common_mask = (raster_mask_np[..., None] + (smpl_rast_plt[..., None] > 0)).clip(0.,1.)
    raster_features_np = raster_features_np*raster_mask_np[...,None]+smpl_rast_plt[...,None]*(1.-raster_mask_np[...,None])
    raster_features_np = raster_features_np * raster_common_mask + 1 * (1-raster_common_mask)

    real_rgb_np = (real_rgb_np * real_segm_np[..., None]) + 1 * (1. - real_segm_np[..., None])
    real_segm_np = 1. - real_segm_np
    real_segm_np = np.stack([real_segm_np] * 3, axis=-1)
    
    out_dict = dict()
    out_dict['rgb_np'] = rgb_np
    out_dict['raster_features_np'] = raster_features_np
    out_dict['real_rgb_np'] = real_rgb_np
    out_dict['real_segm_np'] = real_segm_np
    
    return out_dict


def do_vton_overlays(data_dict):
    out_dict = dict()
    
    real_rgb_nos_np = data_dict['real_rgb_nos_np']
    
    raster_mask_np = data_dict['raster_mask_np']
    raster_features_np = data_dict['raster_features_np']
    vton_pcd_np = real_rgb_nos_np*(1.-raster_mask_np[...,None])+raster_features_np[...,::-1]*raster_mask_np[...,None]
    out_dict['vton_pcd_np'] = vton_pcd_np
    
    rgb_np = data_dict['rgb_np']
    segm_np = data_dict['segm_np']
    vton_rgb_np = real_rgb_nos_np * (1. - segm_np[..., None]) + rgb_np * segm_np[..., None]
    out_dict['vton_rgb_np'] = vton_rgb_np
    
    return out_dict


def infer_pid(style_pid, data_dict, target_dict, outfit_code, ndesc_stack, converter,
              draping_network, renderer, renderer_flood, generator, device, args):
    '''
    Infer outfit geometry and outfit appearance given the input point cloud, 
     style outfit code, point neural descriptors, and models.
    '''
    
    out_dict = dict()
    out_dict.update(target_dict)
    
    data_dict = infer_geometry(data_dict, converter, draping_network, outfit_code)
    data_dict, g_out = infer_appearance(data_dict, ndesc_stack, renderer, renderer_flood, generator, device)
    data_dict = get_flooded_cloth(renderer_flood, data_dict, T=args.visibility_thr)
    out_dict.update(data_dict)
    out_dict.update(g_out)
    
    smpl_rast = get_smpl_render(data_dict['smpl_verts_world'], data_dict['K'], renderer).permute(0,3,1,2)
    out_dict['smpl_rast'] = smpl_rast

    postprocess_dict = postprocess(out_dict, renderer)
    out_dict.update(postprocess_dict)
    
    overlays_dict = do_overlays(out_dict)
    out_dict.update(overlays_dict)

    vton_overlays_dict = do_vton_overlays(out_dict)
    out_dict.update(vton_overlays_dict)
        
    return out_dict
