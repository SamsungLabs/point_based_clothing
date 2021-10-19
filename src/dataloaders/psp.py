import os
import pickle

from .loaders.psp_loader import Loader

from utils.defaults import DEFAULTS
from .common import utils


class Dataset:
    @staticmethod
    def get_args(parser):
        parser.add('--data_root', default=DEFAULTS.psp_data_root, type=str)
        parser.add('--rgb_dir', type=str)
        parser.add('--segm_dir', type=str)
        parser.add('--smpl_dir', type=str)
        parser.add('--imsize', default=512, type=int)
        parser.add('--scale_bbox', default=1.2, type=float)
        parser.add('--n_train_samples', default=-1, type=int)
        return parser

    @staticmethod
    def get_dataset(args, part):
        dirlist = utils.get_part_data(args, part).fillna('')

        if args.world_size > 1 and args.multiperson:
            pids = sorted(list(set(dirlist['seq'])))

            rank = args.local_rank
            WS = args.world_size
            pps = len(pids) // WS
            if rank == (WS - 1):
                pids_selected = pids[pps * rank:]
            else:
                pids_selected = pids[pps * rank:pps * (rank + 1)]
            dirlist = dirlist[dirlist['seq'].isin(pids_selected)]

        elif args.world_size > 1:
            rank = args.local_rank
            WS = args.world_size
            ips = len(dirlist.shape[0]) // WS

            if rank == (WS - 1):
                dirlist = dirlist.iloc[ips * rank:]
            else:
                dirlist = dirlist.iloc[ips * rank:ips * (rank + 1)]

        if part == 'train' and args.n_train_samples > 0:
            dirlist = dirlist.sample(frac=1.)
            dirlist = dirlist.iloc[:args.n_train_samples]

        bg_color = 'rand' if part == 'train' else 'black'
        loader = Loader(args.data_root, args.rgb_dir, args.segm_dir, args.smpl_dir,
                        args.imsize, args.smpl_model_path, args.scale_bbox, bg_color=bg_color)

        dataset = ClothDataset(loader, dirlist)

        return dataset


class ClothDataset:
    def __init__(self, loader, datalist):
        self.datalist = datalist
        self.loader = loader

    def get_sample(self, seq, ind):
        data_dict, target_dict = self.loader.load_sample(seq, ind)

        return data_dict, target_dict

    def __getitem__(self, item):
        row = self.datalist.iloc[item]
        seq = str(row['seq'])
        fid = f"{int(row['id']):06d}"

        data_dict, target_dict = self.get_sample(seq, fid)
        data_dict['index'] = item
        data_dict['fid'] = fid

        return data_dict, target_dict

    def __len__(self):
        return self.datalist.shape[0]
