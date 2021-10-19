import numpy as np
import cv2
from pathlib import Path
import pickle
from dataloaders.common.bbox import get_bbox_from_verts, get_square_bbox, scale_bbox
from dataloaders.common.utils import update_smplifyx_after_crop_and_resize, ltrb2ijhw, get_masked_mesh
from utils.img import crop_image, resize_image
from utils.common import itt, to_tanh
import torch
from PIL import Image
from utils.defaults import DEFAULTS
from .cloth_base_loader import ClothBaseLoader


class Loader(ClothBaseLoader):
    def __init__(self, data_root, rgb_dir, segm_dir, smpl_dir, image_size, smpl_model_path, scale_bbox=1.2, bg_color='rand',
                 rgb_ex='.png', segm_ex='.png', only_eval=False, verts_mask_path=DEFAULTS.verts_mask_path,
                 smpl_faces_path=DEFAULTS.smpl_faces_path, sample_inds_path=DEFAULTS.sample_inds_path):

        super().__init__(data_root, rgb_dir, segm_dir, smpl_dir, None, image_size, scale_bbox=scale_bbox,
                         bg_color=bg_color, rgb_ex=rgb_ex, segm_ex=segm_ex, cam_flip_r=None, cam_flip_l=None,
                         only_eval=only_eval, verts_mask_path=verts_mask_path, 
                         smpl_faces_path=smpl_faces_path, sample_inds_path=sample_inds_path)

        self.FOCAL = 5000

    def load_smpl(self, seq, cam, i, H, W):
        path = Path(seq) / cam
        verts_path = self.data_root / self.smpl_dir / path / i
        verts_path = verts_path.with_suffix('.pkl')
#         verts_path = verts_path + '/000.pkl'

        with open(verts_path, 'rb') as f:
            smpl_dict = pickle.load(f)

        betas = smpl_dict['betas']
        translation = smpl_dict['transl']
        smpl_pose = smpl_dict['full_pose_aa']
        smpl_pose = np.concatenate([translation, smpl_pose], axis=0)

        smpl_verts_world = smpl_dict['vertices']
        K = self.make_K(H, W)

        smpl_verts = smpl_verts_world @ K.T

        out_dict = dict(smpl_pose=smpl_pose, smpl_verts=smpl_verts, K=K, smpl_verts_world=smpl_verts_world, betas=betas)
        return out_dict

    def make_K(self, H, W):
        K = np.eye(3)
        K[0, 0] = K[1, 1] = self.FOCAL
        K[0, 2] = W // 2
        K[1, 2] = H // 2
        return K

    def load_sample(self, seq, i):
        im_sizes = (self.image_size, self.image_size)
        path = Path(seq)
        cam = ''

        rgb = self.load_rgb(path, i)
        segm = self.load_segm(path, i)
        H, W, _ = rgb.shape

        sample = {}
        smpl_dict = self.load_smpl(seq, cam, i, H, W)


        sample.update(smpl_dict)

        if self.only_eval:
            sample['vertices'], sample['K'] = self.flip_verts(sample['vertices'], sample['K'], cam, H=1536, W=2048)
            return sample

        sample['rgb'] = rgb
        sample['segm'] = segm

        H, W, _ = sample['rgb'].shape
        sample = self.flip_sample(sample, cam, ['rgb', 'segm'], [], [])
        sample['smpl_verts'], sample['K'] = self.flip_verts(sample['smpl_verts'], sample['K'], cam, H, W)

        Hf, Wf, _ = sample['rgb'].shape

        ltrb = self.get_bbox(sample['smpl_verts'])
        rgb, cropped_sizes, _ = self.crop_resize_img(sample['rgb'], ltrb)
        segm, _, _ = self.crop_resize_img(sample['segm'], ltrb)

        verts, K = update_smplifyx_after_crop_and_resize(sample['smpl_verts'], sample['K'], ltrb2ijhw(ltrb),
                                                         cropped_sizes,
                                                         im_sizes)

        K_inv = np.linalg.inv(K)
        verts_world = verts @ K_inv.T

        smpl_mesh_cut = get_masked_mesh(verts_world, self.smpl_faces, self.smpl_cloth_mask)
        smpl_mesh_cut = smpl_mesh_cut.subdivide_loop(number_of_iterations=1)
        source_pcd = np.asarray(smpl_mesh_cut.vertices)[self.sample_inds]
        rgb = rgb / 255.
        segm = segm / 255.

        rgb_pt = itt(rgb)
        segm_pt = itt(segm)

        if self.bg_color == 'rand':
            background = torch.rand_like(rgb_pt)
        else:
            background = torch.zeros_like(rgb_pt)

        rgb_pt_nos = to_tanh(rgb_pt.clone())
        rgb_pt = rgb_pt * segm_pt + background * (1. - segm_pt)
        rgb_pt = to_tanh(rgb_pt)
        segm_pt = segm_pt[:1]

        smpl_pose_pt = torch.FloatTensor(sample['smpl_pose'])
        smpl_transl = smpl_pose_pt[:3]
        smpl_glor = smpl_pose_pt[3:6]

        smpl_verts_cam_pt = torch.FloatTensor(verts)
        K_pt = torch.FloatTensor(K)

        smpl_verts_world_pt = torch.FloatTensor(verts_world)

        source_pcd_pt = torch.FloatTensor(source_pcd)
        
        rgb_path = self.data_root / self.rgb_dir / path / i
        rgb_path = rgb_path.with_suffix(self.rgb_ex)

        data_dict = dict(smpl_pose=smpl_pose_pt, smpl_verts_cam=smpl_verts_cam_pt, smpl_verts_world=smpl_verts_world_pt,
                         K=K_pt, background=background, seq=seq, rgb_path=str(rgb_path),
                         source_pcd=source_pcd_pt, betas=smpl_dict['betas'],
                         smpl_transl=smpl_transl, smpl_glor=smpl_glor)
        target_dict = dict(real_rgb=rgb_pt, real_segm=segm_pt, real_rgb_nos=rgb_pt_nos)

        return data_dict, target_dict
