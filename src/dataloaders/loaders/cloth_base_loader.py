import cv2
import pickle
import numpy as np
from PIL import Image
from pathlib import Path

import torch

from dataloaders.common.bbox import get_bbox_from_verts, get_square_bbox, scale_bbox
from dataloaders.common.utils import update_smplifyx_after_crop_and_resize, ltrb2ijhw, get_masked_mesh
from utils.img import crop_image, resize_image
from utils.common import itt, to_tanh
from utils.defaults import DEFAULTS


class ClothBaseLoader:
    def __init__(self, data_root, rgb_dir, segm_dir, smpl_dir, K_dict, image_size, scale_bbox=1.2, bg_color='rand',
                 rgb_ex='.png', segm_ex='.png', cam_flip_r=None, cam_flip_l=None, only_eval=False,
                 verts_mask_path=DEFAULTS.verts_mask_path, smpl_faces_path=DEFAULTS.smpl_faces_path,
                 sample_inds_path=DEFAULTS.sample_inds_path):
        self.data_root = Path(data_root)
        self.rgb_dir = rgb_dir
        self.segm_dir = segm_dir
        self.smpl_dir = smpl_dir
        self.image_size = image_size
        self.rgb_ex = rgb_ex
        self.segm_ex = segm_ex
        self.K_dict = K_dict
        self.only_eval = only_eval
        self.scale_bbox = scale_bbox
        self.bg_color = bg_color
        self.smpl_faces = np.load(smpl_faces_path)

        with open(verts_mask_path, 'rb') as f:
            self.smpl_cloth_mask = pickle.load(f)['non_masked']
        with open(sample_inds_path, 'rb') as f:
            self.sample_inds = pickle.load(f)

        self.cam_flip_r = [] if cam_flip_r is None else cam_flip_r
        self.cam_flip_l = [] if cam_flip_l is None else cam_flip_l

    def load_rgb(self, path, i):
        rgb_dir = self.rgb_dir
        file_path = self.data_root / rgb_dir / path / i
        file_path = file_path.with_suffix(self.rgb_ex)
        try:
            img = cv2.imread(str(file_path))[..., [2, 1, 0]]
        except TypeError:  # probably wrong suffix
            file_path = self.data_root / rgb_dir / path / i
            file_path = file_path.with_suffix('.jpg' if self.rgb_ex == '.png' else '.png')
            img = cv2.imread(str(file_path))[..., [2, 1, 0]]
        return img  # H x W x 3

    def load_smpl(self, seq, cam, i):
        path = Path(seq) / cam
        verts_path = self.data_root / self.smpl_dir / path / i
        verts_path = verts_path.with_suffix('.pkl')

        with open(verts_path, 'rb') as f:
            smpl_dict = pickle.load(f)

        betas = smpl_dict['betas']
        translation = smpl_dict['transl']
        smpl_pose = smpl_dict['full_pose_aa']
        smpl_pose = np.concatenate([translation, smpl_pose], axis=0)
        
        smpl_verts_world = smpl_dict['vertices']
        K = self.K_dict[seq][cam][:, :3]

        smpl_verts = smpl_verts_world @ K.T
        
        out_dict = dict(smpl_pose=smpl_pose, smpl_verts=smpl_verts, K=K, smpl_verts_world=smpl_verts_world, betas=betas)
        return out_dict

    def load_segm(self, path, i):
        segm_path = self.data_root / self.segm_dir / path / i
        segm_path = segm_path.with_suffix(self.segm_ex)
        segm = Image.open(segm_path)
        segm = np.array(segm)
        segm_full = segm
        segm = segm_full[..., -1] - segm_full[..., 0]

        if '38/' in str(path):
            segm = segm - segm_full[..., 1]
        segm = segm[..., None]
        segm = np.concatenate([segm] * 3, axis=-1)
        return segm

    def flip_img(self, img, cam):
        if cam in self.cam_flip_l:
            img = cv2.flip(img, 1)
            img = np.swapaxes(img, 0, 1)
        elif cam in self.cam_flip_r:
            img = cv2.flip(img, 0)
            img = np.swapaxes(img, 0, 1)
        return img

    def flip_coords(self, coords, cam, H, W):
        channel_perm = [1, 0, 2] if coords.shape[-1] == 3 else [1, 0]

        if cam in self.cam_flip_l:
            coords = coords[:, channel_perm]
            coords[:, 1] = W - coords[:, 1]
        elif cam in self.cam_flip_r:
            coords = coords[:, channel_perm]
            coords[:, 0] = H - coords[:, 0]
        return coords

    def make_flip_K(self, cam, H, W):
        K = np.eye(3)
        if cam in self.cam_flip_l:
            K = K[[1, 0, 2]]
            K[1, 0] *= -1
            K[1, 2] = W
        elif cam in self.cam_flip_r:
            K = K[[1, 0, 2]]
            K[0, 1] *= -1
            K[0, 2] = H

        return K

    def flip_verts(self, verts, K, cam, H, W):
        K_flip = self.make_flip_K(cam, H, W)

        verts = verts @ K_flip.T
        K = K_flip @ K
        return verts, K

    def flip_sample(self, sample, cam, imgs, coords, verts, H=None, W=None):
        if H is None or W is None:
            H, W, _ = sample['rgb'].shape

        for k, v in sample.items():
            if k in imgs:
                sample[k] = self.flip_img(v, cam)
            elif k in coords:
                sample[k] = self.flip_coords(v, cam, H, W)
            elif k in verts:
                sample[k] = self.flip_verts(v, cam, H, W)

        return sample

    def get_bbox(self, verts):
        ltrb = get_bbox_from_verts(verts)
        ltrb = scale_bbox(ltrb, self.scale_bbox)
        ltrb = get_square_bbox(ltrb)
        return ltrb

    def crop_resize_img(self, image, bbox):
        im_sizes = (self.image_size, self.image_size)

        image, crop_mask = crop_image(image, bbox, return_mask=True)
        cropped_sizes = image.shape[:2]
        image = resize_image(image, im_sizes)
        crop_mask = resize_image(crop_mask, im_sizes)
        return image, cropped_sizes, crop_mask

    def load_sample(self, seq, cam, i):
        im_sizes = (self.image_size, self.image_size)
        path = Path(seq) / cam

        sample = {}
        smpl_dict = self.load_smpl(seq, cam, i)

        sample.update(smpl_dict)

        if self.only_eval:
            sample['vertices'], sample['K'] = self.flip_verts(sample['vertices'], sample['K'], cam, H=1536, W=2048)
            return sample

        rgb = self.load_rgb(path, i)

        segm = self.load_segm(path, i)

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

        data_dict = dict(smpl_pose=smpl_pose_pt, smpl_verts_cam=smpl_verts_cam_pt, smpl_verts_world=smpl_verts_world_pt,
                         K=K_pt, background=background, seq=seq,
                         source_pcd=source_pcd_pt, betas=smpl_dict['betas'],
                         frame_id=i,
                         smpl_transl=smpl_transl, smpl_glor=smpl_glor)
        target_dict = dict(real_rgb=rgb_pt, real_segm=segm_pt, real_rgb_nos=rgb_pt_nos)

        return data_dict, target_dict
