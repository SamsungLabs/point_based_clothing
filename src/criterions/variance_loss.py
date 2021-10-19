from torch import nn
import torch


class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--var_weight', type=float, default=5e+1)

    @staticmethod
    def get_net(args):
        criterion = Criterion(args.var_weight)
        return criterion.to(args.device)


class Criterion(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        self.i = 0

    def forward(self, data_dict):
        loss = data_dict['glo_batch'].var(dim=0).mean()
        # print('chamfer loss too!', loss)
        loss_G_dict = dict(glo_var_loss=loss)

        return loss_G_dict
