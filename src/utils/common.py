import functools
import numpy as np
import torch


def dict2device(d, device):
    if type(d) == torch.Tensor:
        return d.to(device)

    if type(d) == dict:
        for k, v in d.items():
            d[k] = dict2device(v, device)
    return d


def itt(img):
    tensor = torch.FloatTensor(img)  #
    if len(tensor.shape) == 3:
        tensor = tensor.permute(2, 0, 1)
    else:
        tensor = tensor.unsqueeze(0)
    return tensor


def tti(tensor):
    tensor = tensor.detach().cpu()
    tensor = tensor[0].permute(1, 2, 0)
    image = tensor.numpy()
    if image.shape[-1] == 1:
        image = image[..., 0]
    return image


def accumulate_dict(accum_dict, next_dict):
    for k, v in next_dict.items():
        if k not in accum_dict:
            accum_dict[k] = []
        if hasattr(v, 'item'):
            v = v.item()
        accum_dict[k].append(v)
    return accum_dict


def to_tanh(t):
    return t * 2 - 1.


def to_sigm(t):
    return (t + 1) / 2


def tensor2rgb(tensor):
    if len(tensor.shape) == 4:
        if tensor.shape[1] == 3:
            return tensor
        tensor = tensor.permute(1, 2, 3, 0)

    if len(tensor.shape) == 3 and tensor.shape[0] == 3:
        return tensor

    if len(tensor.shape) == 3:
        tensor = torch.stack([tensor] * 3, dim=0)
        return tensor

    n_channels = tensor.shape[0]

    if n_channels == 1:
        tensor = torch.cat([tensor] * 3, dim=0)
        tensor = tensor - tensor.min()
        tensor = tensor / tensor.max()
    elif n_channels == 2:
        zero_channel = torch.zeros_like(tensor)[:1]
        tensor = torch.cat([tensor, zero_channel], dim=0)
    elif n_channels > 3:
        tensor = tensor[:3]

    if n_channels > 0:
        tensor = tensor.clamp(0., 1.)

    if len(tensor.shape) == 4:
        tensor = tensor.permute(3, 0, 1, 2)

    return tensor


def requires_grad(model, flag=True):
    if type(model) == dict:
        for k, v in model.items():
            requires_grad(v, flag)
    elif type(model) == list:
        for k in model:
            requires_grad(k, flag)
    else:
        for p in model.parameters():
            p.requires_grad = flag


def get_rotation_matrix(angle, axis='x'):
    if axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
    elif axis == 'y':
        return np.array([
            [np.cos(angle), 0, -np.sin(angle)],
            [0, 1, 0],
            [np.sin(angle), 0, np.cos(angle)]
        ])
    elif axis == 'z':
        return np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError(f"Unkown axis {axis}")

        
def rotate_pcd(pcd, angle, axis, mean_point=None, device=torch.device('cuda:0')):
    rotation_matrix = get_rotation_matrix(angle, axis=axis)
    rotation_matrix = torch.from_numpy(rotation_matrix).type(torch.float32).to(device).unsqueeze(0)

    if mean_point is None:
        mean_point = torch.mean(pcd, dim=1)

    pcd_rot = pcd - mean_point
    pcd_rot = pcd_rot.bmm(rotation_matrix.transpose(1, 2))
    pcd_rot = pcd_rot + mean_point

    return pcd_rot, mean_point


def str2param(string):
    param = [int(x) for x in string.split(',')]
    param = np.array(param).reshape(3, 2)
    return param


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def str2array(string):
    return [str(x) for x in string.split(',')]
