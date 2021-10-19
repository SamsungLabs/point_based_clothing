import os
import sys
import cv2
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

import smplx

import torch

parent_dirname = '/'.join(os.path.dirname(__file__).split('/')[:-2])
sys.path.append(parent_dirname)

from dataloaders.common.bbox import get_bbox_from_verts, get_square_bbox, scale_bbox
from dataloaders.common.utils import update_smplifyx_after_crop_and_resize, ltrb2ijhw, get_masked_mesh

from utils.img import crop_image, resize_image
from utils.common import itt, to_tanh
from utils.defaults import DEFAULTS


class Loader():
    def __init__(self, data_root, rgb_dir, segm_dir, smpl_dir, image_size, smpl_model_path, scale_bbox=1.2, 
                 rgb_ex='.png', segm_ex='.png', bg_color='black',
                 verts_mask_path=DEFAULTS.verts_mask_path, smpl_faces_path=DEFAULTS.smpl_faces_path,
                 sample_inds_path=DEFAULTS.sample_inds_path):
        '''
        Dataloader class for random images loaded from the Internet.
        '''
        
        self.data_root = Path(data_root)
        self.rgb_dir = rgb_dir
        self.segm_dir = segm_dir
        self.smpl_dir = smpl_dir
        
        self.image_size = image_size
        self.rgb_ex = rgb_ex
        self.segm_ex = segm_ex
        self.bg_color = bg_color
        
        self.scale_bbox = scale_bbox
        
        self.smpl_faces = np.load(smpl_faces_path)
        self.smpl_model = smplx.body_models.SMPL(smpl_model_path)
        with open(verts_mask_path, 'rb') as f:
            self.smpl_cloth_mask = pickle.load(f)['non_masked']
        with open(sample_inds_path, 'rb') as f:
            self.sample_inds = pickle.load(f)

        self.FOCAL = 5000

    def load_rgb(self, path):
        file_path = self.data_root / self.rgb_dir / path
        file_path = file_path.with_suffix(self.rgb_ex)
        print(file_path)
        try:
            img = cv2.imread(str(file_path))[..., [2, 1, 0]]
        except TypeError:  # probably wrong suffix
            file_path = self.data_root / self.rgb_dir / path
            file_path = file_path.with_suffix('.jpg' if self.rgb_ex == '.png' else '.png')
            img = cv2.imread(str(file_path))[..., [2, 1, 0]]
        return img  # H x W x 3
        
    def load_segm(self, path):
        segm_path = self.data_root / self.segm_dir / path
        segm_path = segm_path.with_suffix(self.segm_ex)
        segm = Image.open(segm_path)
        segm = np.array(segm)
        segm_full = segm
        segm = segm_full[..., -1] - segm_full[..., 0]

        segm = segm[..., None]
        segm = np.concatenate([segm] * 3, axis=-1)
        return segm
    
    def repack_smpl(self, smpl_dict):
        betas = smpl_dict['betas'][0]
        transl = smpl_dict['camera_translation'][0]
        full_pose_aa = np.concatenate([smpl_dict['global_orient'][0], smpl_dict['body_pose'][0].detach().cpu().numpy()])

        betas_pt = torch.FloatTensor(betas).unsqueeze(0)
        transl_pt = torch.FloatTensor(transl).unsqueeze(0)
        pose_pt = torch.FloatTensor(full_pose_aa).unsqueeze(0)

        smpl_output = self.smpl_model(betas=betas_pt, transl=transl_pt, body_pose=pose_pt[:, 3:],
                                      global_orient=pose_pt[:, :3])
        vertices = smpl_output.vertices[0].cpu().numpy()

        out_dict = dict(betas=betas, transl=transl, full_pose_aa=full_pose_aa, vertices=vertices)
        return out_dict
    
    def load_smpl(self, path, H, W):
        path = Path(path)
        verts_path = self.data_root / self.smpl_dir / path
        verts_path = verts_path.with_suffix('.pkl')

        with open(verts_path, 'rb') as f:
            smpl_dict = pickle.load(f)
        smpl_dict = self.repack_smpl(smpl_dict)

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

    def load_sample(self, path):
        im_sizes = (self.image_size, self.image_size)

        seq = path
        path = Path(path)
        
        rgb = self.load_rgb(path)

        segm = self.load_segm(path)
        H, W, _ = segm.shape
        
        smpl_dict = self.load_smpl(path, H, W)

        sample = {'rgb': rgb, 'segm': segm}
        sample.update(smpl_dict)

        ltrb = self.get_bbox(sample['smpl_verts'])
        rgb, cropped_sizes, _ = self.crop_resize_img(sample['rgb'], ltrb)
        segm, cropped_sizes, _ = self.crop_resize_img(sample['segm'], ltrb)

        verts, K = update_smplifyx_after_crop_and_resize(sample['smpl_verts'], sample['K'], ltrb2ijhw(ltrb),
                                                         cropped_sizes,
                                                         im_sizes)

        K_inv = np.linalg.inv(K)
        verts_world = verts @ K_inv.T

        smpl_mesh_cut = get_masked_mesh(verts_world, self.smpl_faces, self.smpl_cloth_mask)
        smpl_mesh_cut = smpl_mesh_cut.subdivide_loop(number_of_iterations=1)
        source_pcd = np.asarray(smpl_mesh_cut.vertices)[self.sample_inds]
        segm = segm / 255.

        segm_pt = itt(segm)
        segm_pt = segm_pt[:1]

        smpl_pose_pt = torch.FloatTensor(sample['smpl_pose'])
        smpl_transl = smpl_pose_pt[:3]
        smpl_glor = smpl_pose_pt[3:6]

        smpl_verts_cam_pt = torch.FloatTensor(verts)
        K_pt = torch.FloatTensor(K)

        smpl_verts_world_pt = torch.FloatTensor(verts_world)

        source_pcd_pt = torch.FloatTensor(source_pcd)
        
        rgb_path = self.data_root / self.rgb_dir / path
        rgb_path = rgb_path.with_suffix(self.rgb_ex)
        
        rgb = rgb / 255.
        rgb_pt = itt(rgb)
        rgb_pt_nos = to_tanh(rgb_pt.clone())
        
        if self.bg_color == 'rand':
            background = torch.rand_like(rgb_pt)
        else:
            background = torch.zeros_like(rgb_pt)
        
        rgb_pt = rgb_pt * segm_pt + background * (1. - segm_pt)
        rgb_pt = to_tanh(rgb_pt)

        data_dict = dict(smpl_pose=smpl_pose_pt, smpl_verts_cam=smpl_verts_cam_pt, smpl_verts_world=smpl_verts_world_pt,
                         K=K_pt, seq=seq, rgb_path=str(rgb_path),
                         source_pcd=source_pcd_pt, betas=smpl_dict['betas'],
                         smpl_transl=smpl_transl, smpl_glor=smpl_glor)
        target_dict = dict(real_rgb=rgb_pt, real_segm=segm_pt, real_rgb_nos=rgb_pt_nos)

        return data_dict, target_dict
