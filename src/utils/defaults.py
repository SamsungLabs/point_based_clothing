import munch

DEFAULTS = {}

# data paths
DEFAULTS['psp_data_root'] = 'samples/psp'

# train.py
DEFAULTS['logdir'] = 'logs'
DEFAULTS['vgg_weights_dir'] = 'checkpoints'

# cloth dataloader
DEFAULTS['verts_mask_path'] = 'data/vertices_cloth_mask.pkl'
DEFAULTS['smpl_faces_path'] = 'data/smplx_faces.npy'
DEFAULTS['sample_inds_path'] = 'data/smpl_sample_inds.pkl'

DEFAULTS = munch.munchify(DEFAULTS)
