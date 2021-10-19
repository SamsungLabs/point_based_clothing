import pandas as pd
import numpy as np
import math
import torch
import open3d


def get_part_data(args, part):
    data = pd.read_csv(f"{args.splits_dir}/{part}.csv", dtype=str)
    return data


def segm2ijhw(segm, border=5):
    H_nnz = np.nonzero(segm.sum(axis=1))[0]
    W_nnz = np.nonzero(segm.sum(axis=0))[0]

    i = H_nnz[0]
    j = W_nnz[0]

    h = H_nnz[-1] - i
    w = W_nnz[-1] - j

    if h > w:
        d = h - w
        j -= d // 2
        w = h
    elif w > h:
        d = w - h
        i -= d // 2
        h = w

    i -= border
    j -= border
    h += border * 2
    w += border * 2

    return [i, j, h, w]


def verts2ijhw(verts, border=150):
    i = int(verts[:, 1].min())
    j = int(verts[:, 0].min())

    h = int(verts[:, 1].max() + 1) - i
    w = int(verts[:, 0].max() + 1) - j

    if h > w:
        d = h - w
        j -= d // 2
        w = h
    elif w > h:
        d = w - h
        i -= d // 2
        h = w

    i -= border
    j -= border
    h += border * 2
    w += border * 2

    return [i, j, h, w]


def crop_img(image, i, j, sz):
    """Crop and pad image with black pixels so that its' top-left corner is (i,j)"""
    # Step 1: cut
    i_img = max(i, 0)
    j_img = max(j, 0)
    h_img = sz + min(i, 0)
    w_img = sz + min(j, 0)

    image = image[i_img:i_img + h_img, j_img:j_img + w_img]

    # Step 2: pad
    H_new, W_new, _ = image.shape

    pad_top = -min(i, 0)
    pad_left = -min(j, 0)
    pad_bot = sz - H_new - pad_top
    pad_right = sz - W_new - pad_left

    pad_seq = [(pad_top, pad_bot), (pad_left, pad_right)]
    if len(image.shape) == 3:
        pad_seq.append((0, 0))

    if (pad_top + pad_bot + pad_left + pad_right) > 0:
        image = np.pad(image, pad_seq, mode='constant')

    return image


def verts2ndc(verts, calibration_matrix, orig_w, orig_h, near=0.0001, far=10.0):
    device = verts.device

    # unproject verts
    calibration_matrix_inv = torch.inverse(calibration_matrix)

    verts_3d = torch.mm(verts, calibration_matrix_inv.transpose(0, 1))

    fx, fy = calibration_matrix[0, 0], calibration_matrix[1, 1]
    cx, cy = calibration_matrix[0, 2], calibration_matrix[1, 2]

    matrix_ndc = torch.tensor([
        [2 * fx / orig_w, 0.0, (orig_w - 2 * cx) / orig_w, 0.0],
        [0.0, -2 * fy / orig_h, -(orig_h - 2 * cy) / orig_h, 0.0],
        [0.0, 0.0, (-far - near) / (far - near), -2.0 * far * near / (far - near)],
        [0.0, 0.0, -1.0, 0.0]
    ], device=device)

    # convert verts to verts ndc
    verts_3d_homo = torch.cat([verts_3d, torch.ones(*verts_3d.shape[:1], 1, device=device)], dim=-1)
    verts_3d_homo[:, 2] *= -1  # invert z-axis

    verts_ndc = torch.mm(verts_3d_homo, matrix_ndc.transpose(0, 1))

    return verts_ndc, matrix_ndc


def find_ndc_boundaries(verts):
    z_min = verts.min(dim=0)[0][2].item()
    z_max = verts.max(dim=0)[0][2].item()

    far = math.ceil(z_max)
    near = (z_min * far) / (2 * far - z_min)
    near = max(math.floor(near), 1e-4)

    return far, near


def ltrb2ijhw(ltrb):
    l, t, r, b = ltrb
    i, j = t, l
    h, w = b - t, r - l
    return (i, j, h, w)


def update_smplifyx_after_crop_and_resize(verts, K, ijhw, image_shape, new_image_shape):
    # it's supposed that it smplifyx's verts are in trivial camera coordinates
    fx, fy, cx, cy = 1.0, 1.0, 0.0, 0.0
    # crop
    cx, cy = cx - ijhw[1], cy - ijhw[0]
    # scale
    h, w = image_shape
    new_h, new_w = new_image_shape

    h_scale, w_scale = new_w / w, new_h / h

    fx, fy = fx * w_scale, fy * h_scale
    cx, cy = cx * w_scale, cy * h_scale

    # update verts
    K_upd = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ])

    verts = verts @ K_upd.T
    K = K_upd @ K

    return verts, K


def get_masked_mesh(vertices, faces, mesh_mask):
    old_ids = np.arange(len(vertices))[mesh_mask]
    vertices_masked = vertices[mesh_mask]

    # check if face contains a masked vertex
    valid_faces_mask = np.isin(faces, old_ids).all(axis=1)
    valid_faces = faces[valid_faces_mask]

    # map valid faces ids to the new ones
    valid_faces_new_id = np.digitize(valid_faces.ravel(), old_ids, right=True)
    valid_faces_new_id = valid_faces_new_id.reshape(-1, 3)

    cut_mesh = open3d.geometry.TriangleMesh()

    cut_mesh.vertices = open3d.utility.Vector3dVector(vertices_masked)
    cut_mesh.triangles = open3d.utility.Vector3iVector(valid_faces_new_id)

    return cut_mesh
