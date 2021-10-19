from torch import nn
from criterions.common.perceptual_loss import PerceptualLoss


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

    def forward(self, inputs):
        fake_rgb = inputs['fake_rgb']
        real_rgb = inputs['real_rgb']
        real_segm = inputs['real_segm']

        fake_rgb = fake_rgb * real_segm
        real_rgb = real_rgb * real_segm

        if 'lossmask' in inputs:
            mask = inputs['lossmask']
            fake_rgb = fake_rgb * mask
            real_rgb = real_rgb * mask

        loss = self.perceptual_crit(fake_rgb, real_rgb)
        loss_G_dict = dict(perceptual_rs=loss)

        return loss_G_dict

