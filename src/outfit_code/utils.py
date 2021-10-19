import numpy as np
import torch


def verts2cam(verts, K):
    KRT = torch.cat([K, torch.zeros_like(K[..., :1])], dim=-1)
    verts_homo = torch.cat([verts, torch.zeros_like(verts[..., :1])], dim=-1)

    verts_cam = torch.bmm(verts_homo, KRT.transpose(1, 2))

    verts_cam_xy = verts_cam[..., :2] / verts_cam[..., 2:]
    verts_cam = torch.cat([verts_cam_xy, verts_cam[..., 2:]], dim=-1)
    return verts_cam


def make_meshgrid(H, W):
    y = torch.arange(0, H)
    x = torch.arange(0, W)
    grid_y, grid_x = torch.meshgrid(x, y)
    meshgrid = torch.stack([grid_x, grid_y], dim=0)
    return meshgrid


def sample_tensor(tensor, n_items, dim=0):
    if tensor.shape[dim] > n_items:
        perm = torch.randperm(tensor.shape[dim]).to(tensor.device)
        idx = perm[:n_items]
        tensor = torch.index_select(tensor, dim, idx)
    return tensor


def oversample_tensor(tensor, n_items, dim=0):
    if tensor.shape[dim] < n_items:
        idx = torch.randint(low=0, high=tensor.shape[dim], size=(n_items,))
        idx = idx.to(tensor.device)
        tensor = torch.index_select(tensor, dim, idx)
    return tensor


def sample_visible_points(cloth_pcd, visibilty_mask, max_points=np.inf):
    n_visible = visibilty_mask.sum(dim=-1).min().item()
    n_visible = min(n_visible, max_points)

    visible_stack = []
    for i in range(visibilty_mask.shape[0]):
        visible_indices = visibilty_mask[i].nonzero().squeeze()
        visible_indices = sample_tensor(visible_indices, n_visible)

        cloth_visible = cloth_pcd[i, visible_indices, :]
        visible_stack.append(cloth_visible)
    visible_stack = torch.stack(visible_stack, dim=0)
    return visible_stack


def mask_to_2dpcd(mask, max_points=np.inf):
    B, _, H, W = mask.shape
    meshgrid = make_meshgrid(H, W).to(mask.device).float()

    segm_pcds = []
    for i in range(B):
        spcd = meshgrid[:, mask[i, 0] > 0].transpose(1, 0)
        segm_pcds.append(spcd)

    n_spcd = (mask[:, 0] > 0).sum(dim=(1, 2)).min().item()
    n_spcd = min(n_spcd, max_points)

    segm_pcds_sampled = []
    for i in range(B):
        segm_pcd = segm_pcds[i]
        segm_pcd = sample_tensor(segm_pcd, n_spcd)
        segm_pcds_sampled.append(segm_pcd)

    segm_pcds_sampled = torch.stack(segm_pcds_sampled, dim=0)

    return segm_pcds_sampled


def slerp(val, low, high):
    '''
    Spherical linear interpolation: https://en.wikipedia.org/wiki/Slerp
    '''

    omega = np.arccos(np.clip(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high  # L'Hopital's rule / LERP
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def multiple_mean(vectors):
    norms = []
    normed_vecs = []
    for vector in vectors:
        norms.append(np.linalg.norm(vector.detach().cpu().numpy()))
        normed_vecs.append(vector.detach().cpu().numpy() / norms[-1])

    init_slerp = slerp(0.5, normed_vecs[0], normed_vecs[1])
    for next_vec in normed_vecs[2:]:
        init_slerp = slerp(0.5, init_slerp, next_vec)
    
    return np.mean(norms) * init_slerp
