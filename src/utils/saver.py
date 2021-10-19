import torch
from pathlib import Path
import shutil
import sys
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import torchvision
from PIL import Image
from huepy import red
from utils.common import tensor2rgb
from utils.common import tti
import threading


def tensor_to_np_recursive(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, dict):
        for k in data.keys():
            data[k] = tensor_to_np_recursive(data[k])

        return data

    elif isinstance(data, (tuple, list)):
        for i in range(len(data)):
            data[i] = tensor_to_np_recursive(data[i])

        return data
    else:
        return data


def make_visual(data, img_names, n_samples=2):
    # img_names = ['target_rgb', 'source_uvi', 'source_segm', 'fake_rgb', 'fake_uvi']

    datalist = []
    for imn in img_names:
        if imn in data:
            item = data[imn][:n_samples]
            item = tensor2rgb(item)
            B, C, H, W = item.shape
            item = item.permute(1,0,2,3)
            item = item.reshape(C, -1, W).unsqueeze(0)
            item = item.clamp(0., 1.)
            datalist.append(item)

    visual = torch.cat(datalist, dim=3)
    return visual


class Saver:
    def __init__(self, save_dir, save_fn='npz_per_batch', clean_dir=False):
        super(Saver, self).__init__()
        self.save_dir = Path(str(save_dir))
        self.need_save = True

        if clean_dir and os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)

        os.makedirs(self.save_dir, exist_ok=True)

        self.save_fn = sys.modules[__name__].__dict__[save_fn]

    def save(self, epoch, **kwargs):
        self.save_fn(save_dir=self.save_dir, epoch=epoch, **kwargs)


def get_saved_name(directory):
    directory = Path(directory)
    files = [x.stem for x in directory.iterdir() if x.stem.isdigit()]
    ids = [int(x) for x in files]

    if len(ids) == 0:
        new_id = 0
    else:
        new_id = max(ids) + 1
    return f'{new_id}'



def visual_per_item(data, save_dir, epoch):
    visual = data['visual']
    visual_np = tti(visual)

    folder = f'{save_dir}/epoch_{epoch:06}'
    os.makedirs(folder, exist_ok=True)

    if 'name' in data:
        name = data['name']
    else:
        name = get_saved_name(folder)

    if type(name) == int or name.isdigit():
        name = f'{int(name):06}'

    name = name + '.png'

    plt.imsave(os.path.join(folder, name), visual_np)

def visual_per_item_tr(data, save_dir, epoch):
    assert 'name' in data
    t1 = threading.Thread(target=visual_per_item, args=(data, save_dir, epoch,))
    t1.start()


def normalize(x):
    maxval = x.max()
    minval = x.min()
    y = (255 * (x - minval) / (maxval - minval))
    return y


def save_images(path, name, out_images, stickmen_images, gt_images):
    images = [out_images, stickmen_images, gt_images]

    if not os.path.exists(path):
        os.makedirs(path)

    out_grid = torchvision.utils.make_grid(normalize(out_images), nrow=1)
    gt_grid = torchvision.utils.make_grid(normalize(gt_images), nrow=1)
    stickmen_grid = torchvision.utils.make_grid(normalize(stickmen_images), nrow=1)

    grid = torchvision.utils.make_grid([out_grid, stickmen_grid, gt_grid], nrow=3)
    grid = grid.permute(1, 2, 0).data.cpu().numpy().astype(np.uint8)

    file = path + name
    im = Image.fromarray(grid)
    im.save(file, "PNG")
