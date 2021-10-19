import os
import sys
import torch
import pickle
import numpy as np
from torch import nn

sys.path.append('..')

from .draping_network import DrapingNetwork
from .nstack import NeuralStack, GLOVector

from utils.common import rgetattr, str2param, str2array


class Wrapper:
    @staticmethod
    def get_args(parser):
        parser.add('--glovec_size', type=int, default=8)
        parser.add('--encoder_depth', type=int, default=5)
        parser.add('--ct_num_latent', type=int, default=512)
        parser.add('--encoder_bn', action='store_bool', default=False)
        parser.add('--ct_heads', type=str, default='1,1,1,1,1,1')
        parser.add('--ct_feats_dims', type=str, default='16,16,64,64,128,128')
        parser.add('--pretrained_ct_path', type=str, default=None)
        parser.add('--pretrained_glovecs_path', type=str, default=None)
        parser.add('--pretrained_glovecs_pids', type=str, default=None)
        parser.add('--freeze', type=str, default='encoder,start,attentions_decoder,final')
        return parser

    @staticmethod
    def get_net(args):
        encoder_input_dim = args.glovec_size
        ct_heads = str2param(args.ct_heads)
        ct_feats_dims = str2param(args.ct_feats_dims)
        
        if not hasattr(args, 'pretrained_glovecs_path'):
            args.pretrained_glovecs_path = None

        if not hasattr(args, 'pretrained_glovecs_pids'):
            args.pretrained_glovecs_pids = None

        if args.pretrained_glovecs_pids is None:
            pretrained_glovecs_pids = []
        else:
            pretrained_glovecs_pids = args.pretrained_glovecs_pids.split(',')
            pretrained_glovecs_pids = [x.strip() for x in pretrained_glovecs_pids]
        
        net = DrapingNetworkWrapper(args.n_people, args.glovec_size, encoder_input_dim, 
                                    args.encoder_depth, args.ct_num_latent, args.encoder_bn, ct_heads, ct_feats_dims, 
                                    args.pretrained_glovecs_path, pretrained_glovecs_pids)

        for component in str2array(args.freeze):
            param_name = f'{component}.parameters'
            if 'cloud_transformer' not in component:
                param_name = f'cloud_transformer.{param_name}'
            for i, param in enumerate(rgetattr(net, param_name)()):
                param.requires_grad = False

        print(f'\n> Loading the draping network model from {args.pretrained_ct_path}...')
        if args.pretrained_ct_path is not None and os.path.exists(args.pretrained_ct_path):
            checkpoint = torch.load(args.pretrained_ct_path)
            checkpoint_model = {k[6:]: v for k, v in checkpoint.items() if k.startswith('model')}
            net.cloud_transformer.load_state_dict(checkpoint_model)

        net = net.to(args.device)
        return net


class DrapingNetworkWrapper(nn.Module):
    def __init__(self, n_people, glovec_size, encoder_input_dim, encoder_depth, num_latent, encoder_bn=True,
                 ct_heads=None, ct_feats_dims=None, pretrained_glovecs_path=None, pretrained_glovecs_pids=None):
        super().__init__()
        
        self.cloud_transformer = DrapingNetwork(encoder_input_dim, encoder_depth, num_latent, encoder_bn, 
                                                ct_heads, ct_feats_dims)
        self.n_people = n_people
        self.glo_stack = NeuralStack(self.n_people, lambda: GLOVector(glovec_size))

        if pretrained_glovecs_path is not None:
            load_all = len(pretrained_glovecs_pids) == 0
            glo_fitted = pickle.load(open(pretrained_glovecs_path, 'rb'))
            print(sorted(glo_fitted.keys()))

            for k, v in glo_fitted.items():
                if load_all or k in pretrained_glovecs_pids:
                    ndesc = self.glo_stack.get_texmodule(k)
                    ndesc.z.data = torch.FloatTensor(v).to(ndesc.z.device)

    def forward(self, data_dict, outfit_code=None):
        '''
        Args:
            outfit_code (`torch.FloatTensor`): outfit codes tensor of shape `(1, 8)`
        '''
        
        pids = None
        if outfit_code is None:  # case of outfit code fitting
            pids = data_dict['seq'].copy()
            pids = [str(x) for x in np.random.permutation(np.arange(0, self.n_people))]
            outfit_code = self.glo_stack.generate_batch(pids)
        elif torch.is_tensor(outfit_code):  # case of inference scripts
            if len(outfit_code.shape) == 1:
                outfit_code = outfit_code[None,:]
        elif isinstance(outfit_code, list):  # case of appearance fitting
            outfit_code = self.glo_stack.generate_batch(outfit_code)

        source_pcd = data_dict['source_pcd'].permute(0, 2, 1)
        
        if outfit_code.shape[0] == 1 and source_pcd.shape[0] > 1:
            # one style vector for several point clouds
            outfit_code = outfit_code.expand(source_pcd.shape[0], -1)
        
        pred_pcl, _ = self.cloud_transformer(noise=source_pcd, input=outfit_code)
        pred_pcl = pred_pcl[:, :, 0].permute(0, 2, 1)
        
        out_dict = dict(cloth_pcd=pred_pcl)
        out_dict['glo_vec'] = outfit_code.clone()
        out_dict['pids_order'] = pids
        
        return out_dict
