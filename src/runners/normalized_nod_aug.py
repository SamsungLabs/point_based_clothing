from time import time

import kornia
import numpy as np
import torch
from huepy import red
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

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

    parser.add('--visibility_thr', default=0., type=float)

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


def get_optimizer(generator, ndesc_stack, draping_network, args):
    model_parameters = generator.parameters()
    ct_decoder_parameters = list(draping_network.cloud_transformer.attentions_decoder.parameters()) + list(
        draping_network.cloud_transformer.start.parameters()) + list(draping_network.cloud_transformer.final.parameters())

    gen_opt = Adam(model_parameters, lr=args.lr_gen, betas=(args.beta1, 0.999), eps=1e-4)
    tex_opt = MyAdam(ndesc_stack.parameters(), lr=args.lr_tex, betas=(args.beta1, 0.999), eps=1e-4)
    glo_opt = MyAdam(draping_network.glo_stack.parameters(), lr=args.lr_glo, betas=(args.beta1, 0.999), eps=1e-4)
    ct_encoder_opt = Adam(draping_network.cloud_transformer.encoder.parameters(), lr=args.lr_ct_encoder,
                          betas=(args.beta1, 0.999), eps=1e-4)
    ct_decoder_opt = Adam(ct_decoder_parameters, lr=args.lr_ct_decoder, betas=(args.beta1, 0.999), eps=1e-4)

    if not hasattr(args, 'fix_g'):
        args.fix_g = False

    if not hasattr(args, 'visibility_thr'):
        args.visibility_thr = 0

    if args.fix_g:
        wrapper = OptWrapper([tex_opt, glo_opt, ct_decoder_opt, ct_encoder_opt])
    else:
        wrapper = OptWrapper([gen_opt, tex_opt, glo_opt, ct_decoder_opt, ct_encoder_opt])
    return wrapper


class TrainingModule(torch.nn.Module):
    def __init__(self, generator, draping_network, discriminator_list, criterion_list, args, augmentations=True):
        super(TrainingModule, self).__init__()
        self.generator = generator
        self.draping_network = draping_network

        self.discriminator_list = nn.ModuleList(discriminator_list)
        self.criterion_list = nn.ModuleList(criterion_list)

        self.renderer = Renderer(args.imsize, args.imsize, args.ntex_channels, device=args.device,
                                 visibility_thr=args.visibility_thr)

        if augmentations:
            # self.augmenter = kornia.augmentation.RandomAffine(degrees=(-45., 45.), translate=(0.3, 0.3),
            #                                                   scale=(0.7, 1.3), return_transform=True)
            self.augmenter = kornia.augmentation.RandomAffine(degrees=(-20., 20.), translate=(0.15, 0.15),
                                                              scale=(0.8, 1.2), return_transform=True)
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

        # for augmented training we need non-segmented rgb
        target_dict['real_rgb'] = target_dict['real_rgb_nos']

        # convert input pcd to cloth3d position
        data_dict = self.converter.source_to_normalized_dict(data_dict)

        # infer pointclout
        glo_out = self.draping_network(data_dict, outfit_code=data_dict['seq'])  # 'cloth_pcd'
        data_dict.update(glo_out)

        # convert 'cloth_pcd' bach to original position
        data_dict = self.converter.normalized_result_to_azure_dict(data_dict)

        # draw neural descriptors from stack
        pids = data_dict['seq']
        ndesc_stack.move_to(pids, self.device)
        ndesc = ndesc_stack.generate_batch(pids).permute(0, 2, 1)
        data_dict['ndesc'] = ndesc

        # rasterize pointcloud with neural descriptors
        raster_dict = self.renderer.render_dict(data_dict)
        data_dict.update(raster_dict)

        if self.augmenter and phase == 'train':
            raster_mask = data_dict['raster_mask']
            raster_features = data_dict['raster_features']
            smpl_mask = data_dict['smpl_mask']

            real_rgb = target_dict['real_rgb']
            real_segm = target_dict['real_segm']

            [raster_mask, raster_features, real_rgb, real_segm, smpl_mask], transform = self.augment(
                [raster_mask, raster_features, real_rgb, real_segm, smpl_mask])

            data_dict['raster_mask'] = raster_mask
            data_dict['raster_features'] = raster_features
            data_dict['smpl_mask'] = smpl_mask

            target_dict['real_rgb'] = real_rgb
            target_dict['real_segm'] = real_segm

        real_rgb = target_dict['real_rgb']
        real_segm = target_dict['real_segm']
        if 'background' in data_dict:
            background = data_dict['background']
            real_rgb = to_sigm(real_rgb) * real_segm + background * (1. - real_segm)
            real_rgb = to_tanh(real_rgb)
        else:
            real_rgb = to_tanh(to_sigm(real_rgb) * real_segm)

        target_dict['real_rgb'] = real_rgb


        # pass through generator (rendering network) to get 'fake_rgb' and 'fake_segm'
        g_out = self.generator(data_dict)

        disc_in = dict()
        disc_in.update(data_dict)
        disc_in.update(target_dict)
        disc_in.update(g_out)

        if phase == 'test':
            return disc_in, dict(), dict()

        if False:
            pass
            # if self.augmenter and phase == 'train':
            #     [uv, rgb, segm, uv_mask, smpl_stickman, openpose_stickman], transform = self.augment(
            #         [uv, rgb, segm, uv_mask, smpl_stickman, openpose_stickman])
            #     uv = uv * uv_mask + (-10 * (1. - uv_mask))
            #
            #     data_dict['uv'] = uv
            #     data_dict['smpl_stickman'] = smpl_stickman
            #     target_dict['real_rgb'] = rgb
            #     target_dict['real_segm'] = segm
            #     target_dict['openpose_stickman'] = openpose_stickman
            #
            #     target_dict = self.augment_coords_dict(target_dict,
            #                                            ['smpl_lhand_kp', 'smpl_rhand_kp', 'openpose_lhand_kp',
            #                                             'openpose_rhand_kp', 'face_kp'], transform)

        # disc_out = {}
        # for discriminator in self.discriminator_list:
        #     dout = discriminator(disc_in)
        #     disc_out.update(dout)

        criterion_in = dict()
        criterion_in.update(disc_in)
        # criterion_in.update(disc_out)

        # calculate losses
        losses_G_dict = dict()
        # losses_D_dict = dict()

        for criterion in self.criterion_list:
            crit_out = criterion(criterion_in)

            losses_G_dict.update(crit_out)

        return criterion_in, losses_G_dict


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
        out_data, losses_G_dict = training_module(data_dict, target_dict, ndesc_stack, phase)

        out_data = detach_dict(out_data)

        loss_G_accum = dict()

        # if phase == 'train':
        if (writer is not None and it % args.log_frequency_images == 0) or phase in ['val', 'test']:

            # make visual
            visual_in = dict(fake_rgb=to_sigm(out_data['fake_rgb']),
                             fake_segm=tensor2rgb(out_data['fake_segm']),
                             real_rgb=to_sigm(out_data['real_rgb']),
                             real_segm=tensor2rgb(out_data['real_segm']),
                             raster_features=tensor2rgb(out_data['raster_features'] * 0.5))

            visual = make_visual(visual_in, visual_in.keys())

        if saver and phase in ['val', 'test']:
            saver.save(epoch, data=dict(visual=visual, name=f"{args.local_rank}_{it:04d}"))
            if phase == 'test':
                continue

        if phase == 'train':
            # if wscheduler is not None:
            #     losses_G_dict_weighted, losses_D_dict_weighted = wscheduler(losses_G_dict, losses_D_dict)
            # else:
            losses_G_dict_weighted = losses_G_dict

            loss_G_weighted = sum(losses_G_dict_weighted.values())

            optimizer_G.zero_grad()
            loss_G_weighted.backward(retain_graph=True)
            optimizer_G.step()

            losses_G_dict_weighted = detach_dict(losses_G_dict_weighted)
            loss_G_weighted = loss_G_weighted.detach()

        losses_G_dict = detach_dict(losses_G_dict)

        pids = out_data['seq']
        # ndesc_stack.move_to(pids, 'cpu')

        losses_G_dict = detach_dict(losses_G_dict)

        # accumulate loss values
        loss_G_accum = accumulate_dict(loss_G_accum, losses_G_dict)

        if writer is not None:
            if it % args.log_frequency_images == 0:
                writer.add_image(f'1.Images/{phase}/visual', visual)

            if phase == 'train':
                for k, v in losses_G_dict.items():
                    writer.add_scalar(f'{phase}/G_{k}', v.item(), writer.last_it)

                writer.last_it += 1

                exname = args.experiment_name.split('^')[-1]
                pbar.set_description("{}: {}".format(exname, writer.last_it))

    # write accumulated losses to tensorboard
    if writer is not None and phase == 'val':
        for k, v in loss_G_accum.items():
            writer.add_scalar(f'{phase}/G_{k}', np.mean(v), writer.last_it)

    print(f' * \n * Epoch {epoch} {red(phase.capitalize())} finished')
