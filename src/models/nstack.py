import torch
from torch import nn
import numpy as np


def project_l2_ball_v2(z):
    summed = torch.sum(z ** 2, dim=1, keepdim=True)
    return z / torch.max(torch.sqrt(summed), torch.ones_like(summed))


class PointNeuralTex(torch.nn.Module):
    def __init__(self, texchannels=16, pcd_size=4096 * 2):
        super().__init__()

        self.neuraltex = torch.nn.Parameter(torch.randn(texchannels, pcd_size),
                                            requires_grad=True)

    def forward(self):
        return self.neuraltex


class GLOVector(torch.nn.Module):
    def __init__(self, dim_glo):
        super().__init__()

        self.z = nn.Parameter(torch.randn(dim_glo),
                              requires_grad=True)

    def forward(self):
        with torch.no_grad():
            self.z.data = project_l2_ball_v2(self.z[None])[0]

        return self.z


def nstack_from_statedict(texture_factory, state_dict):
    n_people = len(state_dict.keys())
    nstack = NeuralStack(n_people, texture_factory)

    for i, k in enumerate(state_dict.keys()):
        nstack.pid2ntid[k] = nn.Parameter(torch.LongTensor([i]), requires_grad=False)
        nstack.textures[i].load_state_dict(state_dict[k])

    return nstack


class NeuralStack(torch.nn.Module):
    def __init__(self, n_people, texture_factory):

        super().__init__()

        self.textures = nn.ModuleList([])
        for i in range(n_people):
            ntex = texture_factory()
            self.textures.append(ntex)
        self.pid2ntid = nn.ParameterDict()

    def load_state_dict_tex(self, state_dict):
        textures_sd = {k: v for (k, v) in state_dict.items() if 'textures' in k}
        self.load_state_dict(textures_sd, strict=False)

        for k in state_dict.keys():
            if 'pid2ntid' in k:
                pid = k.split('.')[-1]
                ntid = state_dict[k]
                self.pid2ntid[pid] = nn.Parameter(torch.LongTensor([ntid]), requires_grad=False)

    def set_texmodule(self, pid, vector, attr='z'):
        if torch.is_tensor(pid):
            pid = pid.tolist()
        pid = str(pid)

        if pid not in self.pid2ntid:
            if len(self.pid2ntid) == 0:
                next_tid = 0
            else:
                next_tid = len(self.pid2ntid.keys())
            self.pid2ntid[pid] = nn.Parameter(torch.LongTensor([next_tid]), requires_grad=False)

        getattr(self.textures[self.pid2ntid[pid]], attr).data = vector.data

    def get_texmodule(self, pid):
        if torch.is_tensor(pid):
            pid = pid.tolist()
        pid = str(pid)

        if pid not in self.pid2ntid:
            if len(self.pid2ntid) == 0:
                next_tid = 0
            else:
                next_tid = len(self.pid2ntid.keys())
            self.pid2ntid[pid] = nn.Parameter(torch.LongTensor([next_tid]), requires_grad=False)
        return self.textures[self.pid2ntid[pid]]

    def __getitem__(self, pid):
        return self.get_texmodule(pid)

    def move_to(self, pids, device):
        for pid in pids:
            ntex_module = self.get_texmodule(pid)
            ntex_module.to(device)

    def generate_batch(self, pids):
        ntexs = []
        for pid in pids:
            ntex_module = self.get_texmodule(pid)

            nt = ntex_module()
            ntexs.append(nt)

        ntexs = torch.stack(ntexs, dim=0)

        return ntexs
