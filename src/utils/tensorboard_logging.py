import os.path
import datetime
from tensorboardX import SummaryWriter
from pathlib import Path
from huepy import *
from torchvision.utils import make_grid
import os.path
from utils.common import tti
import numpy as np
import cv2
import sys

from huepy import yellow


class MySummaryWriter(SummaryWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_dir = args[0]
        self.last_it = 0

    def add_image(self, name, image_tensor):
        grid = make_grid(image_tensor.detach().clamp(0, 1).data.cpu(), nrow=1)
        grid_np = tti(grid.unsqueeze(0))
        grid_np = (grid_np * 255.).astype(np.uint8)
        out_path = os.path.join(self.save_dir, 'visuals', f'{self.last_it:08d}.png')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        cv2.imwrite(out_path, grid_np[..., [2,1,0]])

        super(MySummaryWriter, self).add_image(name, grid, self.last_it)


def get_postfix(args, default_args, args_to_ignore, delimiter='__'):
    s = []

    for arg in sorted(args.keys()):
        if not isinstance(arg, Path) and arg not in args_to_ignore and default_args[arg] != args[arg]:
            s += [f"{arg}^{args[arg]}"]

    return delimiter.join(s).replace('/', '+')  # .replace(';', '+')


def setup_logging(args, default_args, args_to_ignore, exp_name_use_date=True, tensorboard=True, test=False):
    if not args.experiment_name:
        args.experiment_name = get_postfix(vars(args), vars(default_args), args_to_ignore)
    
        if exp_name_use_date:
            time = datetime.datetime.now()
            args.experiment_name = time.strftime(f"%m-%d_%H-%M___{args.experiment_name}")

    save_dir = os.path.join(args.experiments_dir, args.experiment_name)
    os.makedirs(f'{save_dir}/checkpoints', exist_ok=True)

    # sys.stderr = open(os.path.join(save_dir, "stderr"), 'w')
    # sys.stdout = open(os.path.join(save_dir, "stdout"), 'w')

    writer = MySummaryWriter(save_dir, filename_suffix='_train') if tensorboard else None

    return save_dir, writer
