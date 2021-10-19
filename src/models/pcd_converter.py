import torch
from smplx.lbs import batch_rodrigues


class PCDConverter:
    def __init__(self, device):
        self.J0 = torch.FloatTensor([-0.0033, -0.2314, 0.0212]).unsqueeze(0).to(device)
        self.glor_c3d = torch.FloatTensor([1.1553, 1.0915, 1.2142]).unsqueeze(0).to(device)
        self.trans_c3d = torch.FloatTensor([-0.6243, 0.0129, 0.8486]).unsqueeze(0).to(device)

        # self.glor_c3d = torch.FloatTensor([5.46430746e-01, 2.89516700e-01,  2.50626756e-01]).unsqueeze(0).to(device)
        # self.trans_c3d = torch.FloatTensor([-3.17047866e-03, -5.56754512e-04,  9.16602543e-01]).unsqueeze(0).to(device)

        self.rotmat_c3d = batch_rodrigues(self.glor_c3d)
        self.rotmat_c3d_inv = torch.inverse(self.rotmat_c3d)

        self.zRotMatrix = torch.FloatTensor([[0.65664387, -0.7542008, 0.],
                                             [0.7542008, 0.65664387, 0.],
                                             [0., 0., 1.]]).to(device).unsqueeze(0)
        self.zRotMatrix_inv = torch.inverse(self.zRotMatrix)
    
    
    def normalized_to_azure(self, V, transl_az, glor_az, zrotMatrix_c3d=None, 
                            style_dset='cloth3d', pose_dset='azure', device='cuda:0'):
        V = V.clone()
        rotmat_az = batch_rodrigues(glor_az[:, 0])
        if V.shape[0] == 2:
            print(rotmat_az[0, 0, 0].item())  # VERY STRANGE BUG
        rotmat_az_inv = torch.inverse(rotmat_az)
        
        if not (style_dset == 'bcnet' and pose_dset == 'azure'):
            V -= self.J0
            
        V = torch.bmm(V, rotmat_az_inv)
            
        if pose_dset == 'azure' or style_dset == 'cloth3d' and pose_dset == 'psp':
            V += self.J0
        
        V += transl_az
        
        if zrotMatrix_c3d is not None:
            V = torch.bmm(V, zrotMatrix_c3d.to(device).transpose(1, 2))
        
        return V
    
    
    def azure_to_normalized(self, V, transl_az, glor_az, zrotMatrix_c3d=None, 
                            style_dset='cloth3d', pose_dset='azure', device='cuda:0'):
        V = V.clone()

        if zrotMatrix_c3d is not None:
            zRotMatrix_inv = torch.inverse(zrotMatrix_c3d.to(device))
        rotmat_az = batch_rodrigues(glor_az[:, 0])
        
        if zrotMatrix_c3d is not None:
            V = torch.bmm(V, zRotMatrix_inv.transpose(1, 2))

        V -= transl_az
        
        if pose_dset == 'azure' or style_dset == 'cloth3d' and pose_dset == 'psp':
            V -= self.J0
        
        V = torch.bmm(V, rotmat_az)
        
        if not (style_dset == 'bcnet' and pose_dset == 'azure'):
            V += self.J0

        return V
    

    def source_to_normalized_dict(self, data_dict, style_dset='cloth3d', pose_dset='azure'):
        '''
        args:
            dset (str): one of ['azure', 'cloth3d', 'bcnet']
        '''
        
        V = data_dict['source_pcd']

        smpl_pose = data_dict['smpl_pose']
        transl_az = data_dict['smpl_transl']
        glor_az = data_dict['smpl_glor']

        smpl_pose[:, :3] = self.trans_c3d
        smpl_pose[:, 3:6] = self.glor_c3d
        data_dict['smpl_pose'] = smpl_pose
        
        zmat = data_dict.get('zrotMatrix_c3d')
        if zmat is not None:
            zmat = zmat.unsqueeze(0)

        V_norm = self.azure_to_normalized(V, transl_az.unsqueeze(1), glor_az.unsqueeze(1), zrotMatrix_c3d=zmat, 
                                          style_dset=style_dset, pose_dset=pose_dset)

        data_dict['source_pcd'] = V_norm

        return data_dict
    
    
    def normalized_result_to_azure_dict(self, data_dict, style_dset='cloth3d', pose_dset='azure'):
        V = data_dict['cloth_pcd']
        smpl_pose = data_dict['smpl_pose']
        transl_az = data_dict['smpl_transl']
        glor_az = data_dict['smpl_glor']

        smpl_pose[:, :3] = transl_az
        smpl_pose[:, 3:6] = glor_az
        data_dict['smpl_pose'] = smpl_pose
        
        zmat = data_dict.get('zrotMatrix_c3d')
        if zmat is not None:
            zmat = zmat.unsqueeze(0)

        data_dict['cloth_pcd'] = self.normalized_to_azure(V, transl_az.unsqueeze(1), glor_az.unsqueeze(1), zrotMatrix_c3d=zmat, 
                                                          style_dset=style_dset, pose_dset=pose_dset)
        return data_dict
