import torch
from torch import nn

from generators.common import npbg_unet_bu
from utils.common import to_tanh, to_sigm


class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--gen_in_channels', type=int, default=17)
        parser.add('--backbone', type=str, default='resnet18')
        parser.add('--segm_channels', type=int, default=1)
        parser.add('--ntex_channels', type=int, default=16)
        parser.add('--n_people', type=int)

    @staticmethod
    def get_net(args):
        net = Generator(args.gen_in_channels, args.segm_channels)
        net = net.to(args.device)
        return net


class Generator(nn.Module):
    def __init__(self, in_channels, segm_channels):
        super().__init__()
        n_out = 16
        self.model = npbg_unet_bu.UNet(
            num_input_channels=in_channels,
            num_output_channels=n_out,
            feature_scale=4,
            more_layers=0,
            upsample_mode='bilinear',
            norm_layer='bn',
        )
        norm_layer = nn.BatchNorm2d

        padding = nn.ZeroPad2d

        self.rgb_head = nn.Sequential(
            norm_layer(n_out, affine=True),
            nn.ReLU(True),
            padding(1),
            nn.Conv2d(n_out, n_out, 3, 1, 0, bias=False),
            norm_layer(n_out, affine=True),
            nn.ReLU(True),
            padding(1),
            nn.Conv2d(n_out, n_out, 3, 1, 0, bias=False),
            norm_layer(n_out, affine=True),
            nn.ReLU(True),
            padding(1),
            nn.Conv2d(n_out, 3, 3, 1, 0, bias=True),
            nn.Tanh())

        self.segm_head = nn.Sequential(
            norm_layer(n_out, affine=True),
            nn.ReLU(True),
            padding(1),
            nn.Conv2d(n_out, segm_channels, 3, 1, 0, bias=True),
            nn.Sigmoid())

    def forward(self, data_dict):
        raster_mask = data_dict['raster_mask']
        raster_features = data_dict['raster_features']

        inp = torch.cat([raster_mask, raster_features], dim=1)
        out = self.model(inp)

        rgb = self.rgb_head(out)
        segm = self.segm_head(out)

        segm_fg = segm[:, :1]

        if 'background' in data_dict:
            background = data_dict['background']
            rgb_segm = to_sigm(rgb) * segm_fg + background * (1. - segm_fg)
        else:
            rgb_segm = to_sigm(rgb) * segm_fg
        rgb_segm = to_tanh(rgb_segm)
        rgb_segm_bbg = to_tanh(to_sigm(rgb) * segm_fg)

        out_dict = dict(fake_rgb=rgb_segm, fake_segm=segm, fake_rgb_bbg=rgb_segm_bbg)

        return out_dict
