from torch import nn
from criterions.common.perceptual_loss import PerceptualLoss
import torch


class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--perc_weight', type=float, default=1.)
        parser.add('--perc_n_relus', type=int, default=-1)
        parser.add('--perc_sizes', type=str, default='64,128,256,512')

    @staticmethod
    def get_net(args):

        perc_sizes = args.perc_sizes.split(',')
        perc_sizes = [int(x.strip()) for x in perc_sizes]

        criterion = Criterion(args.perc_weight, args.vgg_weights_dir, n_relus=args.perc_n_relus, perc_sizes=perc_sizes)
        return criterion.to(args.device)


class Criterion(nn.Module):
    def __init__(self, perc_weight, vgg_weights_dir, n_relus=7, perc_sizes=None):
        super().__init__()


        self.perceptual_crit = PerceptualLoss(perc_weight, vgg_weights_dir, n_relus=n_relus).eval()

        self.perc_sizes = [] if perc_sizes is None else perc_sizes 

    def forward(self, inputs):
        fake_rgb = inputs['fake_rgb']
        real_rgb = inputs['real_rgb']

        if 'lossmask' in inputs:
            mask = inputs['lossmask']
            fake_rgb = fake_rgb * mask
            real_rgb = real_rgb * mask

        if len(self.perc_sizes) == 0:
            loss = self.perceptual_crit(fake_rgb, real_rgb)
        else:
            _, _, H, W = fake_rgb.shape
            losses = []
            for size in self.perc_sizes:
                if size != H:
                    fake_rgb_resized = torch.nn.functional.interpolate(fake_rgb, size=(size, size))
                    real_rgb_resized = torch.nn.functional.interpolate(real_rgb, size=(size, size))
                else:
                    fake_rgb_resized = fake_rgb
                    real_rgb_resized = real_rgb

                losses.append(self.perceptual_crit(fake_rgb_resized, real_rgb_resized))
            loss = sum(losses) / len(losses)



        loss_G_dict = dict(perceptual=loss)

        return loss_G_dict
