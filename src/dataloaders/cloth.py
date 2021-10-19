import os
import pickle

from .common import utils
from .loaders.cloth_base_loader import ClothBaseLoader

from utils.defaults import DEFAULTS


class Dataset:
    @staticmethod
    def get_args(parser):
        parser.add('--data_root', default=DEFAULTS.data_root, type=str)
        parser.add('--rgb_dir', type=str)
        parser.add('--segm_dir', type=str)
        parser.add('--smpl_dir', type=str)
        parser.add('--K_file', type=str)
        parser.add('--imsize', default=512, type=int)
        parser.add('--scale_bbox', default=1.2, type=float)
        parser.add('--n_train_samples', default=-1, type=int)
        parser.add('--ndc', action='store_bool', default=False)
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

        with open(os.path.join(args.data_root, args.K_file), 'rb') as f:
            K_dict = pickle.load(f)

        bg_color = 'rand' if part == 'train' else 'black'
        loader = SampleLoader(args.data_root, args.rgb_dir, args.segm_dir, args.smpl_dir, K_dict,
                              args.imsize, args.scale_bbox, bg_color)

        dataset = ClothDataset(loader, dirlist)

        return dataset


class SampleLoader(ClothBaseLoader):
    def __init__(self, data_root, rgb_dir, segm_dir, smpl_dir, K_dict, image_size, scale_bbox, 
                 bg_color, rgb_ex='.png', segm_ex='.png'):
        cam_flip_r = ['000583592412']
        cam_flip_l = ['000230292412']

        super().__init__(data_root, rgb_dir, segm_dir, smpl_dir, K_dict, image_size, scale_bbox, bg_color,
                         rgb_ex=rgb_ex, segm_ex=segm_ex, cam_flip_r=cam_flip_r, cam_flip_l=cam_flip_l)


class ClothDataset:
    def __init__(self, loader, datalist):
        self.datalist = datalist
        self.loader = loader

    def get_sample(self, seq, cam, ind):
        data_dict, target_dict = self.loader.load_sample(seq, cam, ind)

        return data_dict, target_dict

    def __getitem__(self, item):
        row = self.datalist.iloc[item]
        seq = str(row['seq'])
        cam = str(row['cam'])
        fid = str(row['id'])

        # print(cam)

        data_dict, target_dict = self.get_sample(seq, cam, fid)
        data_dict['index'] = item
        data_dict['fid'] = fid
        data_dict['cam'] = cam

        return data_dict, target_dict

    def __len__(self):
        return self.datalist.shape[0]

