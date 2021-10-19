from torch import nn
from criterions.common.perceptual_loss import PerceptualLoss
import torch


class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--smplmask_weight', type=float, default=1e+5)

    @staticmethod
    def get_net(args):
        criterion = Criterion(args.smplmask_weight)
        return criterion.to(args.device)


class Criterion(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        self.i = 0

    def forward(self, data_dict):
        real_segm = data_dict['real_segm']
        fake_segm = data_dict['fake_segm']
        smpl_mask = data_dict['smpl_mask']

        mask_req = smpl_mask * real_segm
        missing_mask = (mask_req - fake_segm).clamp(0., 1.)

        loss = missing_mask.mean() * self.weight
        loss_G_dict = dict(smplmask=loss)

        return loss_G_dict
