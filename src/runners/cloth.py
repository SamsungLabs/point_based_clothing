import sys
from time import time
from tqdm import tqdm
from huepy import red

sys.path.append('..')

import numpy as np

import kornia

import torch
from torch import nn
from torch.optim import Adam

from optimizers.tex_optimizer import MyAdam

from models.pcd_converter import PCDConverter
from models.pcd_renderer import Renderer

from utils.common import dict2device, accumulate_dict, to_tanh, to_sigm, tensor2rgb
from utils.defaults import DEFAULTS
from utils.saver import make_visual
from utils.utils import Meter


def get_args(parser):
    parser.add('--log_frequency_loss', type=int, default=1)
    parser.add('--log_frequency_images', type=int, default=100)
    parser.add('--detailed_metrics', action='store_bool', default=True)
    parser.add('--num_visuals_per_img', default=2, type=int)

    parser.add('--lr_gen', default=5e-5, type=float)
    parser.add('--lr_tex', type=float, default=1e-3)
    parser.add('--lr_glo', type=float, default=1e-3)
    parser.add('--beta1', default=0.0, type=float, help='beta1 for Adam')
    parser.add('--fix_g', action='store_bool', default=False)

    return parser


class OptWrapper():
    def __init__(self, optimizer_list):
        self.optimizer_list = optimizer_list

    def zero_grad(self):
        for opt in self.optimizer_list:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizer_list:
            opt.step()

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizer_list]

    def load_state_dict(self, state_dict):
        for i in range(len(self.optimizer_list) - 1):
            self.optimizer_list[i].load_state_dict(state_dict[i])


def get_optimizer(generator, ndesc_stack, glo_model, args):
    model_parameters = generator.parameters()
    glo_decoder_parameters = list(glo_model.cloud_transformer.attentions_decoder.parameters()) + list(glo_model.cloud_transformer.start.parameters()) + list(glo_model.cloud_transformer.final.parameters())

    gen_opt = Adam(model_parameters, lr=args.lr_gen, betas=(args.beta1, 0.999), eps=1e-4)
    tex_opt = MyAdam(ndesc_stack.parameters(), lr=args.lr_tex, betas=(args.beta1, 0.999), eps=1e-4)
    glo_opt = MyAdam(glo_model.glo_stack.parameters(), lr=args.lr_glo, betas=(args.beta1, 0.999), eps=1e-4)
    ct_encoder_opt = Adam(glo_model.cloud_transformer.encoder.parameters(), 
                          lr=args.lr_ct_encoder, betas=(args.beta1, 0.999), eps=1e-4)
    ct_decoder_opt = Adam(glo_decoder_parameters, lr=args.lr_ct_decoder, betas=(args.beta1, 0.999), eps=1e-4)

    if not hasattr(args, 'fix_g'):
        args.fix_g = False

    if args.fix_g:
        wrapper = OptWrapper([tex_opt, glo_opt, ct_decoder_opt, ct_encoder_opt])
    else:
        wrapper = OptWrapper([gen_opt, tex_opt, glo_opt, ct_decoder_opt, ct_encoder_opt])
    return wrapper


def get_optimizer_glo_fitting(draping_network, args, freeze=None):
    opt_list = []
    
    glo_opt = MyAdam(draping_network.glo_stack.parameters(), lr=args.lr_glo, betas=(args.beta1, args.beta2), eps=1e-4)
    glo_scheduler = torch.optim.lr_scheduler.ExponentialLR(glo_opt, gamma=args.gamma)
    opt_list.append(glo_opt)
    
    if freeze == 'except_encoder':
        enc_opt = MyAdam(draping_network.cloud_transformer.encoder.parameters(), 
                         lr=args.lr_enc, betas=(args.beta1, args.beta2), eps=1e-4)
        opt_list.append(enc_opt)

    wrapper = OptWrapper(opt_list)

    return wrapper, glo_scheduler


class TrainingModule(torch.nn.Module):
    def __init__(self, generator, glo_model, discriminator_list, criterion_list, args, augmentations=True):
        super(TrainingModule, self).__init__()
        self.generator = generator
        self.glo_model = glo_model

        self.discriminator_list = nn.ModuleList(discriminator_list)
        self.criterion_list = nn.ModuleList(criterion_list)

        self.renderer = Renderer(args.imsize, args.imsize, args.ntex_channels, device=args.device)

        if augmentations:
            self.augmenter = kornia.augmentation.RandomAffine(degrees=(-45., 45.), translate=(0.3, 0.3),
                                                              scale=(0.7, 1.3), return_transform=True)
        else:
            self.augmenter = None

        self.device = args.device
        self.converter = PCDConverter(self.device)

    def augment(self, items):
        ncs = [x.shape[1] for x in items]
        items_stack = torch.cat(items, dim=1)
        auged_stack, transform = self.augmenter(items_stack)
        auged = []

        st_c = 0
        for nc in ncs:
            item = auged_stack[:, st_c:st_c + nc]
            auged.append(item)
            st_c += nc

        return auged, transform

    def augment_coords(self, coords, transform):
        coords_aug = torch.cat([coords.clone(), torch.ones_like(coords[:, :, :1])],
                               dim=-1)
        coords_aug = torch.bmm(transform, coords_aug.permute(0, 2, 1)).permute(0, 2, 1)
        coords_aug = coords_aug[..., :2]
        coords_aug[coords < 0] = -1
        return coords_aug

    def augment_coords_dict(self, data_dict, keys_to_aug, transform):
        for k in keys_to_aug:
            data_dict[k] = self.augment_coords(data_dict[k], transform)
        return data_dict

    def forward(self, data_dict, target_dict, ndesc_stack, phase):

        # convert input pcd to cloth3d position
        data_dict = self.converter.source_to_c3d_dict(data_dict)

        # infer pointclout
        glo_out = self.glo_model(data_dict)  # 'cloth_pcd'
        data_dict.update(glo_out)

        # convert 'cloth_pcd' bach to original position
        data_dict = self.converter.result_to_azure_dict(data_dict)

        # draw neural descriptors from stack
        pids = data_dict['seq']
        ndesc_stack.move_to(pids, self.device)
        ndesc = ndesc_stack.generate_batch(pids).permute(0, 2, 1)
        data_dict['ndesc'] = ndesc

        # rasterize pointcloud with neural descriptors
        raster_dict = self.renderer.render_dict(data_dict)
        data_dict.update(raster_dict)

        # pass through generator (rendering network) to get 'fake_rgb' and 'fake_segm'
        g_out = self.generator(data_dict)

        disc_in = dict()
        disc_in.update(data_dict)
        disc_in.update(target_dict)
        disc_in.update(g_out)

        if phase == 'test':
            return disc_in, dict(), dict()

        disc_out = {}
        for discriminator in self.discriminator_list:
            dout = discriminator(disc_in)
            disc_out.update(dout)

        criterion_in = dict()
        criterion_in.update(disc_in)
        criterion_in.update(disc_out)

        # calculate losses
        losses_G_dict = dict()
        losses_D_dict = dict()

        for criterion in self.criterion_list:
            crit_out = criterion(criterion_in)

            if len(crit_out) == 2:
                crit_out_G, crit_out_D = crit_out
                losses_G_dict.update(crit_out_G)
                losses_D_dict.update(crit_out_D)
            elif len(crit_out) == 1:
                losses_G_dict.update(crit_out)
            else:
                raise Exception(f'Wrong number of outputs in criterion {type(criterion)}, should be one or two')

        return criterion_in, losses_G_dict, losses_D_dict


def detach_dict(dic):
    for k, v in dic.items():
        if hasattr(v, 'detach'):
            dic[k] = v.detach()
    return dic


def run_epoch(dataloader, training_module, ndesc_stack, optimizer_G, optimizer_D_list, epoch, args,
              phase,
              writer=None,
              saver=None, wscheduler=None):
    if phase == 'train':
        optimizer_G.zero_grad()
        if optimizer_D_list is not None:
            for opt in optimizer_D_list:
                opt.zero_grad()

    pbar = tqdm(dataloader, initial=0, dynamic_ncols=True, smoothing=0.01) if args.local_rank == 0 else dataloader
    for it, data in enumerate(pbar):

        data_dict, target_dict = data

        if phase == 'test':
            data_dict['pid'] = [str(it)]

        data_dict = dict2device(data_dict, args.device)
        target_dict = dict2device(target_dict, args.device)
        out_data, losses_G_dict, losses_D_dict = training_module(data_dict, target_dict, ndesc_stack, phase)

        loss_G_accum = dict()
        loss_D_accum = dict()

        # if phase == 'train':
        if (writer is not None and it % args.log_frequency_images == 0) or phase in ['val', 'test']:
            # make visual
            visual_in = dict(fake_rgb=to_sigm(out_data['fake_rgb']),
                             fake_segm=tensor2rgb(out_data['fake_segm']) * 0.5,
                             real_rgb=to_sigm(out_data['real_rgb']),
                             real_segm=tensor2rgb(out_data['real_segm'] * 0.5),
                             raster_features=tensor2rgb(out_data['raster_features'] * 0.5))

            visual = make_visual(visual_in, visual_in.keys())

        if saver and phase in ['val', 'test']:
            saver.save(epoch, data=dict(visual=visual, name=f"{args.local_rank}_{it:04d}"))
            if phase == 'test':
                continue

        if phase == 'train':
            if wscheduler is not None:
                losses_G_dict_weighted, losses_D_dict_weighted = wscheduler(losses_G_dict, losses_D_dict)
            else:
                losses_G_dict_weighted, losses_D_dict_weighted = losses_G_dict, losses_D_dict

            loss_G_weighted = sum(losses_G_dict_weighted.values())
            loss_D_weighted = sum(losses_D_dict_weighted.values())

            optimizer_G.zero_grad()
            if losses_D_dict:
                [opt.zero_grad() for opt in optimizer_D_list]
            loss_G_weighted.backward(retain_graph=True)
            if losses_D_dict:
                loss_D_weighted.backward()
            optimizer_G.step()
            if losses_D_dict:
                [opt.step() for opt in optimizer_D_list]

        pids = out_data['seq']
        # ndesc_stack.move_to(pids, 'cpu')

        losses_G_dict = detach_dict(losses_G_dict)
        losses_D_dict = detach_dict(losses_D_dict)

        # accumulate loss values
        loss_G_accum = accumulate_dict(loss_G_accum, losses_G_dict)
        loss_D_accum = accumulate_dict(loss_D_accum, losses_D_dict)

        if writer is not None:
            if it % args.log_frequency_images == 0:
                writer.add_image(f'1.Images/{phase}/visual', visual)

            if phase == 'train':
                for k, v in losses_G_dict.items():
                    writer.add_scalar(f'{phase}/G_{k}', v.item(), writer.last_it)
                for k, v in losses_D_dict.items():
                    writer.add_scalar(f'{phase}/D_{k}', v.item(), writer.last_it)

                writer.last_it += 1

                exname = args.experiment_name.split('^')[-1]
                pbar.set_description("{}: {}".format(exname, writer.last_it))

    # write accumulated losses to tensorboard
    if writer is not None and phase == 'val':
        for k, v in loss_G_accum.items():
            writer.add_scalar(f'{phase}/G_{k}', np.mean(v), writer.last_it)
        for k, v in loss_D_accum.items():
            writer.add_scalar(f'{phase}/D_{k}', np.mean(v), writer.last_it)

    print(f' * \n * Epoch {epoch} {red(phase.capitalize())} finished')
