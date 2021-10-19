from torch import nn
from criterions.common.perceptual_loss import PerceptualLoss
import torch


class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--dice_weight', type=float, default=1e-2)

    @staticmethod
    def get_net(args):
        criterion = Criterion(args.dice_weight)
        return criterion.to(args.device)


def dice_loss(input, target):
    dims = (1, 2, 3)
    numer = (2 * input * target).sum(dims)
    denom = (input.pow(2) + target.pow(2)).sum(dims)

    dice = numer / (denom + 1e-16)
    loss = -torch.log(dice + 1e-16).mean()
    return loss


class Criterion(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, data_dict):
        real_segm = data_dict['real_segm']
        fake_segm = data_dict['fake_segm']

        loss = dice_loss(fake_segm, real_segm) * self.weight
        loss_G_dict = dict(dice=loss)

        return loss_G_dict
