from torch import nn
from utils.glofit_tools import calc_chamfer
import torch


class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--chamfer_weight', type=float, default=5e+1)

    @staticmethod
    def get_net(args):
        criterion = Criterion(args.chamfer_weight)
        return criterion.to(args.device)


class Criterion(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        self.i = 0

    def forward(self, data_dict):
        loss = calc_chamfer(data_dict, data_dict)
        # print('chamfer loss too!', loss)
        loss_G_dict = dict(chamfer_2d=loss)

        return loss_G_dict
