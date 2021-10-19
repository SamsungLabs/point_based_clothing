from torch import nn
from criterions.common.perceptual_loss import PerceptualLoss
import torch


class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--ndrgb_weight', type=float, default=1e+5)

    @staticmethod
    def get_net(args):
        criterion = Criterion(args.ndrgb_weight)
        return criterion.to(args.device)


class Criterion(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        self.i = 0

    def forward(self, data_dict):
        fake_rgb = data_dict['fake_rgb'].detach()
        fake_segm = data_dict['fake_segm'].detach()
        raster_mask = data_dict['raster_mask']
        raster_features = data_dict['raster_features']

        raster_features = raster_features[:, :3] * fake_segm
        real_rgb = fake_rgb * fake_segm * raster_mask
        loss = (raster_features - real_rgb).pow(2).mean()*self.weight


        real_rgb = data_dict['real_rgb'].detach()
        real_segm = data_dict['real_segm'].detach()
        raster_mask = data_dict['raster_mask']
        raster_features = data_dict['raster_features']

        raster_features = raster_features[:, :3] * real_segm
        real_rgb = real_rgb * real_segm * raster_mask
        loss += (raster_features - real_rgb).pow(2).mean()*self.weight

        loss_G_dict = dict(ndrgb=loss)

        return loss_G_dict

