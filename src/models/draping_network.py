import os
import sys

parent_dirname = '/'.join(os.path.dirname(__file__).split('/')[:-2])
sys.path.append(parent_dirname)
sys.path.append(parent_dirname + '/cloud_transformers')

from cloud_transformers.layers.multihead_ct_adain import MultiHeadUnionAdaIn, forward_style
from cloud_transformers.layers.utils import AdaIn1dUpd

import torch.nn.functional as F
from torch import nn


def sum_hack(lists):
    start = lists[0]
    for i in range(1, len(lists)):
        start += lists[i]

    return start


class ParamEncoder(nn.Module):
    def __init__(self, input_dim, encoder_depth, output_dim, bn):
        '''
        `input_dim`: dimensionality of an outfit code (defaults to 8 in our experiments)
        '''

        super(ParamEncoder, self).__init__()
        self.dims = [input_dim, 256, 512, 512, 512, 512, 512, 512, 512, 512]  # 10 layers at most
        layers = []
        if encoder_depth == 1:
            layers.append(nn.Linear(input_dim, output_dim))  # mapping
        else:
            i = 0
            while (i < min(encoder_depth - 1, len(self.dims) - 1)):
                layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
                if bn:
                    layers.append(nn.BatchNorm1d(self.dims[i + 1]))
                last_dim = self.dims[i + 1]
                i += 1
            layers.append(nn.Linear(last_dim, output_dim))  # mapping
            if bn:
                layers.append(nn.BatchNorm1d(output_dim))
        self.mlps = nn.ModuleList(layers)

    def forward(self, x):
        for l in self.mlps:
            x = F.relu(l(x), inplace=True)
        return x


class DrapingNetwork(nn.Module):
    def __init__(self, encoder_input_dim, encoder_depth, num_latent=512, encoder_bn=True, ct_heads=None,
                 ct_feats_dims=None):
        super().__init__()

        self.model_dim = 512

        self.encoder = nn.Sequential(ParamEncoder(
            input_dim=encoder_input_dim, encoder_depth=encoder_depth,
            output_dim=num_latent, bn=encoder_bn))

        self.start = nn.Sequential(nn.Conv1d(in_channels=3,
                                             out_channels=self.model_dim, kernel_size=1, bias=False),
                                   AdaIn1dUpd(self.model_dim, num_latent=num_latent),
                                   nn.ReLU(True))

        features_dims_vals = [(4, 4), (4 * 4, 4 * 4), (4 * 4, 4 * 8)] if ct_feats_dims is None else ct_feats_dims
        heads_vals = [(16, 16), (16, 16), (16, 16)] if ct_heads is None else ct_heads
        self.attentions_decoder = nn.ModuleList(sum_hack([[MultiHeadUnionAdaIn(model_dim=self.model_dim,
                                                                               features_dims=features_dims_vals[0],
                                                                               heads=heads_vals[0],
                                                                               tensor_sizes=[128, 32],
                                                                               model_dim_out=self.model_dim,
                                                                               n_latent=num_latent,
                                                                               tensor_dims=[2, 3],
                                                                               scales=True),
                                                           MultiHeadUnionAdaIn(model_dim=self.model_dim,
                                                                               features_dims=features_dims_vals[1],
                                                                               heads=heads_vals[1],
                                                                               tensor_sizes=[64, 16],
                                                                               model_dim_out=self.model_dim,
                                                                               n_latent=num_latent,
                                                                               tensor_dims=[2, 3],
                                                                               scales=True),
                                                           MultiHeadUnionAdaIn(model_dim=self.model_dim,
                                                                               features_dims=features_dims_vals[2],
                                                                               heads=heads_vals[2],
                                                                               tensor_sizes=[16, 8],
                                                                               model_dim_out=self.model_dim,
                                                                               n_latent=num_latent,
                                                                               tensor_dims=[2, 3],
                                                                               scales=True)] for _ in range(4)]))

        self.final = nn.Sequential(nn.Conv1d(in_channels=self.model_dim,
                                             out_channels=self.model_dim, kernel_size=1, bias=False),
                                   AdaIn1dUpd(self.model_dim, num_latent=num_latent),
                                   nn.ReLU(inplace=True),
                                   nn.Conv1d(in_channels=self.model_dim,
                                             out_channels=3, kernel_size=1))

    def forward(self, noise, input, return_lattice=False):
        '''
        `input` (torch.FloatTensor): tensor of outfit codes of shape (B, encoder_input_dim)
        `noise` (torch.FloatTensor): tensor of point clouds of shape (B, 3, num_points)
        '''

        z = self.encoder(input)

        x = forward_style(self.start, noise, z)

        lattices_sizes = []

        for i in range(len(self.attentions_decoder)):
            x, lattice_size = self.attentions_decoder[i](x, z, noise)
            lattices_sizes += lattice_size

        x = forward_style(self.final, x, z)

        return x.unsqueeze(2), lattices_sizes
