from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import compositing
from pytorch3d.renderer.points import rasterize_points
import numpy as np
import torch
from torch import nn
import nvdiffrast.torch as dr
from utils.defaults import DEFAULTS

import os

import pytorch3d


class RasterizePointsXYsBlending(nn.Module):
    """
    > Taken from https://github.com/facebookresearch/synsin/models/layers/z_buffer_layers.py
    
    Rasterizes a set of points using a differentiable renderer. Points are
    accumulated in a z-buffer using an accumulation function
    defined in opts.accumulation and are normalised with a value M=opts.M.
    Inputs:
    - pts3D: the 3D points to be projected
    - src: the corresponding features
    - C: size of feature
    - learn_feature: whether to learn the default feature filled in when
                     none project
    - radius: where pixels project to (in pixels)
    - size: size of the image being created
    - points_per_pixel: number of values stored in z-buffer per pixel
    - opts: additional options
    Outputs:
    - transformed_src_alphas: features projected and accumulated
        in the new view
    """

    def __init__(
            self,
            C=64,
            learn_feature=True,
            radius=1.5,
            size=256,
            points_per_pixel=8,
            opts=None,
    ):
        super().__init__()
        if learn_feature:
            default_feature = nn.Parameter(torch.randn(1, C, 1))
            self.register_parameter("default_feature", default_feature)
        else:
            default_feature = torch.zeros(1, C, 1)
            self.register_buffer("default_feature", default_feature)

        self.radius = radius
        self.size = size
        self.points_per_pixel = points_per_pixel
        self.opts = opts

    def forward(self, pts3D, src, return_ids=False):

        pts3D = pts3D.clone()
        bs = src.size(0)
        if len(src.size()) > 3:
            bs, c, w, _ = src.size()
            image_size = w

            pts3D = pts3D.permute(0, 2, 1)
            src = src.unsqueeze(2).repeat(1, 1, w, 1, 1).view(bs, c, -1)
        else:
            bs = src.size(0)
            image_size = self.size

        # Make sure these have been arranged in the same way
        assert pts3D.size(2) == 3
        assert pts3D.size(1) == src.size(2)

        pts3D[:, :, 1] = - pts3D[:, :, 1]
        pts3D[:, :, 0] = - pts3D[:, :, 0]

        # Add on the default feature to the end of the src
        # src = torch.cat((src, self.default_feature.repeat(bs, 1, 1)), 2)

        if isinstance(image_size, tuple):
            radius = float(self.radius) / float(min(image_size)) * 2.0
        else:
            radius = float(self.radius) / float(image_size) * 2.0

        pts3D = Pointclouds(points=pts3D, features=src.permute(0, 2, 1))
        points_idx, _, dist = rasterize_points(
            pts3D, image_size, radius, self.points_per_pixel
        )

        if "DEBUG" in os.environ and os.environ["DEBUG"]:
            print("Max dist: ", dist.max(), pow(radius, self.opts.rad_pow))

        dist = dist / pow(radius, self.opts.rad_pow)

        if os.environ["DEBUG"]:
            print("Max dist: ", dist.max())

        alphas = (
            (1 - dist.clamp(max=1, min=1e-3).pow(0.5))
                .pow(self.opts.tau)
                .permute(0, 3, 1, 2)
        )

        if self.opts.accumulation == 'alphacomposite':
            transformed_src_alphas = compositing.alpha_composite(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1, 0),
            )
        elif self.opts.accumulation == 'wsum':
            transformed_src_alphas = compositing.weighted_sum(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1, 0),
            )
        elif self.opts.accumulation == 'wsumnorm':
            transformed_src_alphas = compositing.weighted_sum_norm(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1, 0),
            )

        if return_ids:
            return transformed_src_alphas, points_idx
        else:
            return transformed_src_alphas


def reproject_z_t(world_kps, projection_matrix):
    '''
    Returns reprojected 2D camera points given 3D world points and projection matrix.
    '''
    batch_size, num_pts, _ = world_kps.shape

    padd = torch.ones((batch_size, num_pts, 1),
                      dtype=torch.float, device=world_kps.device)
    homogeneous_coords = torch.cat([world_kps, padd], dim=-1)

    camera_kps = torch.einsum('bpd, bnd->bpn',
                              [homogeneous_coords, projection_matrix])

    camera_kps_norm = camera_kps[:, :, :2] / camera_kps[:, :, 2:]  # X/Z, Y/Z ~ (B, N, 2)
    return torch.cat([camera_kps_norm, camera_kps[:, :, 2:]], dim=-1)


class Renderer:
    def __init__(self, height, width, pcd_features_dim, device, z_far=10.0, z_near=0.0001, radius=1.5,
                 faces_path=DEFAULTS.smpl_faces_path, visibility_thr=0, scale=1.):
        self.glctx = dr.RasterizeGLContext()

        self.height = int(height * scale)
        self.width = int(width * scale)
        self.scale = scale

        self.z_far = z_far
        self.z_near = z_near
        self.z_range = z_far - z_near

        self.visibility_thr = visibility_thr

        self.pcd_feature_dim = pcd_features_dim

        class fake_opt_r:
            def __init__(self):
                self.accumulation = 'alphacomposite'
                self.tau = 1.0
                self.rad_pow = 2

        self.rasterize_pcd = RasterizePointsXYsBlending(C=pcd_features_dim, opts=fake_opt_r(),
                                                        size=(self.height, self.width),
                                                        learn_feature=False, radius=radius)

        faces = np.load(faces_path)
        self.faces = torch.LongTensor(faces).int().to(device)

    def convert_to_ndc_t(self, points, intrinsic):
        '''
        Convert points into the normalized device coordinates
        '''
        points_proj = reproject_z_t(points, intrinsic)

        norm = torch.from_numpy(np.array([self.width, self.height, 1])).float().to(points.device)
        points_norm = points_proj / norm[None, None, :]

        z = points_norm[:, :, 2:]
        z_norm = (-z + self.z_near) / self.z_range

        xy_norm = (points_norm[:, :, :2] - 0.5) * 2

        return xy_norm, z_norm

    def get_smpl_rast(self, vertices, faces, intrinsic):
        xy_norm, z_norm = self.convert_to_ndc_t(vertices, intrinsic)
        z_correction = (1 + z_norm)
        z_correction = z_correction.min(dim=1)[0][..., 0]
        z_correction[z_correction > 0] = 0
        z_correction = z_correction[:, None, None]
        z_norm -= z_correction

        v = torch.cat([xy_norm, -z_norm, torch.ones_like(z_norm)], dim=-1)
        rast, rast_db = dr.rasterize(self.glctx,
                                     pos=v.contiguous(),
                                     tri=faces.contiguous(),
                                     resolution=[self.height, self.width])

        return rast, (xy_norm, z_norm), z_correction

    def get_smpl_render_t(self, vertices, faces, vertices_clothes, intrinsic):
        # project smpl vertices
        rast, (xy_norm, z_norm), z_correction = self.get_smpl_rast(vertices, faces, intrinsic)
        xy_clothes_norm, z_clothes_norm = self.convert_to_ndc_t(vertices_clothes, intrinsic)

        z_clothes_norm -= z_correction

        return rast, (xy_norm, z_norm), (xy_clothes_norm, z_clothes_norm)

    def get_garment_mask(self, vertices, faces, vertices_clothes, intrinsic):
        rast, (xy_norm, z_norm), (xy_clothes_norm, z_clothes_norm) = self.get_smpl_render_t(vertices, faces,
                                                                                            vertices_clothes,
                                                                                            intrinsic)
        depth_feature_map = rast[:, :, :, 2:3].permute(0, 3, 1, 2)
        positions = xy_clothes_norm[:, None]

        smpl_depth_at_closes = torch.nn.functional.grid_sample(depth_feature_map, positions,
                                                               mode='nearest', padding_mode='zeros',
                                                               align_corners=True)
        smpl_depth_at_closes = smpl_depth_at_closes[:, 0, 0]
        
        clothes_depth = -z_clothes_norm[:, :, 0]

        T = self.visibility_thr
        T2 = T * 2.5
        depth_delta = smpl_depth_at_closes - clothes_depth
        visibilty_mask = torch.logical_or((depth_delta > -T2), (smpl_depth_at_closes < 1e-8))
        visibilty_alphas = ((T2 + depth_delta.detach()) / (T2 - T)).clamp(0., 1.)
        visibilty_alphas[smpl_depth_at_closes < 1e-8] = 1.

        visible_pcd = vertices_clothes.clone()

        smpl_depth_at_closes = smpl_depth_at_closes[..., None]

        # mask non-visible
        visible_pcd[~visibilty_mask[:, :, None].expand(-1, -1, 3)] = -10
        xy_clothes_norm[~visibilty_mask[:, :, None].expand(-1, -1, 2)] = -10
        z_clothes_norm[~visibilty_mask[:, :, None]] = -10
        smpl_depth_at_closes[~visibilty_mask[:, :, None]] = -10

        return visible_pcd, visibilty_mask, visibilty_alphas, smpl_depth_at_closes, (
            xy_clothes_norm, z_clothes_norm), rast

    def render_smpl_with_clothes(self, vertices, vertices_clothes, intrinsic, pcd_texture):
        visible_pcd, visibilty_mask, visibilty_alphas, smpl_depth_at_closes, (
            xy_clothes_norm, z_clothes_norm), rast = self.get_garment_mask(vertices,
                                                                           self.faces,
                                                                           vertices_clothes,
                                                                           intrinsic)
        ndc_clothes = torch.cat([xy_clothes_norm,
                                 -z_clothes_norm], dim=-1)
        dummy_features = torch.ones_like(ndc_clothes[:, :, 2:])

        visibilty_alphas = visibilty_alphas[..., None]
        pcd_texture = pcd_texture * visibilty_alphas

        raster_mask = self.rasterize_pcd(ndc_clothes, dummy_features.permute(0, 2, 1))
        raster_features = self.rasterize_pcd(ndc_clothes, pcd_texture.permute(0, 2, 1))

        return raster_mask, raster_features, visibilty_mask, rast

    def render_cloth_depth(self, data_dict):
        smpl_verts = data_dict['smpl_verts_world']
        cloth_pcd = data_dict['cloth_pcd']
        K = data_dict['K'].clone()

        K[:, :2, 2] = K[:, :2, 2] * self.scale
        K[:, :2, :2] = K[:, :2, :2] * self.scale
        smpl_verts = smpl_verts * self.scale
        cloth_pcd = cloth_pcd * self.scale

        KRT = torch.cat([K, torch.zeros_like(K[..., :1])], dim=-1).cuda()

        visible_pcd, visibilty_mask, visibilty_alphas, depth_features, (
            xy_clothes_norm, z_clothes_norm), rast = self.get_garment_mask(smpl_verts,
                                                                           self.faces,
                                                                           cloth_pcd,
                                                                           KRT)

        ndc_clothes = torch.cat([xy_clothes_norm,
                                 -z_clothes_norm], dim=-1)
        visibilty_alphas = visibilty_alphas[..., None]
        depth_render, point_ids = self.rasterize_pcd(ndc_clothes, depth_features.permute(0, 2, 1), return_ids=True)


        depth_renders = []
        dmin = []
        B = depth_render.shape[0]
        for i in range(B):
            pids = point_ids[i, ...,  :1] - i * cloth_pcd.shape[1]
            depths = depth_features[i, ..., 0]

            depths_nn = depths[depths > 0]
            dmin.append(depths_nn.min())

            nbg_mask = pids > -1

            dr = torch.zeros_like(pids).float()

            dr[nbg_mask] = depths[pids[nbg_mask].long()]
            depth_renders.append(dr)

        depth_render = torch.stack(depth_renders, dim=0).permute(0,3,1,2)
        dmin = torch.stack(dmin, dim=0)

        # depth_render[depth_render > -1] = depth_features[depth_render[depth_render > -1]]

        return depth_render, dmin

    def render_dict(self, data_dict):
        smpl_verts = data_dict['smpl_verts_world']
        cloth_pcd = data_dict['cloth_pcd']
        K = data_dict['K'].clone()
        
        K[:, :2, 2] = K[:, :2, 2] * self.scale
        K[:, :2, :2] = K[:, :2, :2] * self.scale
        smpl_verts = smpl_verts * self.scale
        cloth_pcd = cloth_pcd * self.scale

        KRT = torch.cat([K, torch.zeros_like(K[..., :1])], dim=-1).cuda()
        pcd_texture = data_dict['ndesc']

        raster_mask, raster_features, visibilty_mask, smpl_rast = self.render_smpl_with_clothes(smpl_verts, cloth_pcd,
                                                                                                KRT, pcd_texture)
        smpl_rast = smpl_rast.permute(0, 3, 1, 2)
        smpl_mask = (smpl_rast[:, 2:3] > 0).float()
        smpl_rast = smpl_rast[:, 2:3]
        smpl_rast = smpl_rast - smpl_rast.min()
        smpl_rast = smpl_rast / smpl_rast.max()
        
        # raster_mask.shape torch.Size([4, 1, 256, 256]) 
        # raster_features torch.Size([4, 16, 256, 256]) 
        # visibilty_mask torch.Size([4, 8192]) 
        # smpl_mask torch.Size([4, 1, 256, 256]) 
        # smpl_rast torch.Size([4, 1, 256, 256])

        return dict(raster_mask=raster_mask, raster_features=raster_features, visibilty_mask=visibilty_mask,
                    smpl_mask=smpl_mask, smpl_rast=smpl_rast)
