from torch import nn
from criterions.common.perceptual_loss import PerceptualLoss
import torch


class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--rastmask_weight', type=float, default=1e+5)

    @staticmethod
    def get_net(args):
        criterion = Criterion(args.rastmask_weight)
        return criterion.to(args.device)


class Criterion(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        self.i = 0

    def forward(self, data_dict):
        fake_segm = data_dict['fake_segm']
        raster_mask = data_dict['raster_mask']

        missing_mask = (raster_mask - fake_segm).clamp(0., 1.)

        loss = missing_mask.mean() * self.weight
        loss_G_dict = dict(rastmask=loss)

        return loss_G_dict
