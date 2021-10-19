import torch
from torch import nn


class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--gan_weight', type=float, default=1.)

    @staticmethod
    def get_net(args):
        criterion = Criterion(args.gan_weight)
        return criterion.to(args.device)


class Criterion(torch.nn.Module):
    def __init__(self, weight=1.):
        super().__init__()
        self.weight = weight


    def forward(self, data_dict):
        fake_score = data_dict['fake_score']
        real_score = data_dict['real_score']

        loss_G = 0.5 * (1. - fake_score).pow(2).mean()  # * 1e-1
        loss_D = (0.5 * fake_score.pow(2).mean() + 0.5 * (1. - real_score).pow(2).mean())  # * 1e-1

        loss_G *= self.weight
        loss_D *= self.weight

        loss_G_dict = dict(adversarial=loss_G)
        loss_D_dict = dict(adversarial=loss_D)

        return loss_G_dict, loss_D_dict
