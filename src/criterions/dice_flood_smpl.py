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
    denom = ((input.pow(2) + target.pow(2))).sum(dims)

    dice = numer / (denom + 1e-16)
    loss = -torch.log(dice + 1e-16).mean()
    return loss


class Criterion(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        self.i = 0

    def forward(self, data_dict):
        real_segm = data_dict['real_segm']
        fake_segm = data_dict['fake_segm']

        smplmask = data_dict['smpl_mask']
        flood_mask = data_dict['flood_cloth_mask']
        smplmask_inv = 1. - smplmask
        flood_mask_inv = 1. - flood_mask

        lossmask = (real_segm + smplmask_inv + flood_mask_inv).clamp(0., 1.).detach()
        fake_segm = fake_segm * lossmask
        real_segm = real_segm * lossmask

        # out_dict = dict(fake_segm=fake_segm, real_segm=real_segm, flood_mask=flood_mask, lossmask=lossmask)
        # torch.save(out_dict, f'/Vol0/user/a.grigorev/temp/diceflood/{self.i:06d}.pth')
        # self.i += 1

        loss = dice_loss(fake_segm, real_segm) * self.weight
        loss_G_dict = dict(dice=loss)

        return loss_G_dict
