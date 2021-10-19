import torch
from torch import nn
from criterions.common.perceptual_loss import PerceptualLoss
from utils.common import to_tanh, to_sigm

class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--perc_weight', type=float, default=1.)
        parser.add('--perc_n_relus', type=int, default=7)

    @staticmethod
    def get_net(args):
        criterion = Criterion(args.perc_weight, args.vgg_weights_dir, n_relus=args.perc_n_relus)
        return criterion.to(args.device)


class Criterion(nn.Module):
    def __init__(self, perc_weight, vgg_weights_dir, n_relus=7):
        super().__init__()

        self.perceptual_crit = PerceptualLoss(perc_weight, vgg_weights_dir, n_relus=n_relus).eval()
        self.i = 0

    def forward(self, inputs):
        fake_rgb = inputs['fake_rgb']
        real_rgb = inputs['real_rgb']

        real_segm = inputs['real_segm']
        smplmask = inputs['smpl_mask']
        flood_mask = inputs['flood_cloth_mask']
        smplmask_inv = 1. - smplmask
        flood_mask_inv = 1. - flood_mask

        lossmask = (real_segm + smplmask_inv + flood_mask_inv).clamp(0., 1.).detach()

        fake_rgb = to_tanh(to_sigm(fake_rgb) * lossmask)
        real_rgb = to_tanh(to_sigm(real_rgb) * lossmask)

        # out_dict = dict(fake_rgb=fake_rgb, real_rgb=real_rgb, real_segm=real_segm, flood_mask_inv=flood_mask_inv, smplmask_inv=smplmask_inv, lossmask=lossmask)
        # torch.save(out_dict, f'/Vol0/user/a.grigorev/temp/percflood/{self.i:06d}.pth')
        # self.i += 1

        loss = self.perceptual_crit(fake_rgb, real_rgb)
        loss_G_dict = dict(perceptual=loss)

        return loss_G_dict
