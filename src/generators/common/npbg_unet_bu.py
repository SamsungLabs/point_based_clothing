import math

import numpy as np
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_1=3, kernel_size_2=3, normalization=nn.BatchNorm2d):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size_1, padding=kernel_size_1//2),
                                   normalization(out_channels),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size_2, padding=kernel_size_2//2),
                                   normalization(out_channels),
                                   nn.ReLU())

    def forward(self, inputs, **kwargs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block=BasicBlock):
        super().__init__()

        self.conv = conv_block(in_channels, out_channels)
        self.down = nn.AvgPool2d(2, 2)

    def forward(self, inputs):
        outputs = self.down(inputs)
        outputs = self.conv(outputs)
        return outputs


class UpsampleBlock(nn.Module):
    def __init__(self, out_channels, upsample_mode, same_num_filt=False, conv_block=BasicBlock):
        super().__init__()

        num_filt = out_channels if same_num_filt else out_channels * 2
        if upsample_mode == 'deconv':
            self.up = nn.ConvTranspose2d(num_filt, out_channels, 4, stride=2, padding=1)
            self.conv = conv_block(out_channels * 2, out_channels, normalization=Identity)
        elif upsample_mode == 'bilinear' or upsample_mode == 'nearest':
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode=upsample_mode),
                                    # Before refactoring, it was a nn.Sequential with only one module.
                                    # Need this for backward compatibility with model checkpoints.
                                    nn.Sequential(
                                        nn.Conv2d(num_filt, out_channels, 3, padding=1)
                                    )
                                    )
            self.conv = conv_block(out_channels * 2, out_channels, normalization=Identity)
        else:
            assert False

    def forward(self, inputs1, inputs2):
        in1_up = self.up(inputs1)

        if (inputs2.size(2) != in1_up.size(2)) or (inputs2.size(3) != in1_up.size(3)):
            diff2 = (inputs2.size(2) - in1_up.size(2)) // 2
            diff3 = (inputs2.size(3) - in1_up.size(3)) // 2
            inputs2_ = inputs2[:, :, diff2: diff2 + in1_up.size(2), diff3: diff3 + in1_up.size(3)]
        else:
            inputs2_ = inputs2

        output = self.conv(torch.cat([in1_up, inputs2_], 1))

        return output


class UpsampleBlock2(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, upsample_mode, conv_block=BasicBlock):
        super().__init__()

        if upsample_mode == 'deconv':
            self.up = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
            self.conv = conv_block(in_channels + skip_channels, out_channels, normalization=Identity)
        elif upsample_mode == 'bilinear' or upsample_mode == 'nearest':
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode=upsample_mode),
                                    # Before refactoring, it was a nn.Sequential with only one module.
                                    # Need this for backward compatibility with model checkpoints.
                                    nn.Sequential(
                                        nn.Conv2d(in_channels, in_channels, 3, padding=1)
                                    )
                                    )
            self.conv = conv_block(in_channels + skip_channels, out_channels, normalization=Identity)
        else:
            assert False

    def forward(self, inputs1, inputs2):
        in1_up = self.up(inputs1)

        if (inputs2.size(2) != in1_up.size(2)) or (inputs2.size(3) != in1_up.size(3)):
            diff2 = (inputs2.size(2) - in1_up.size(2)) // 2
            diff3 = (inputs2.size(3) - in1_up.size(3)) // 2
            inputs2_ = inputs2[:, :, diff2: diff2 + in1_up.size(2), diff3: diff3 + in1_up.size(3)]
        else:
            inputs2_ = inputs2

        output = self.conv(torch.cat([in1_up, inputs2_], 1))

        return output


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx = len(self) + idx

        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class UNet(nn.Module):
    r""" Rendering network with UNet architecture and multi-scale input.
    Args:
        num_input_channels: Number of channels in the input tensor or list of tensors. An integer or a list of integers for each input tensor.
        num_output_channels: Number of output channels.
        feature_scale: Division factor of number of convolutional channels. The bigger the less parameters in the model.
        more_layers: Additional down/up-sample layers.
        upsample_mode: One of 'deconv', 'bilinear' or 'nearest' for ConvTranspose, Bilinear or Nearest upsampling.
        norm_layer: [unused] One of 'bn', 'in' or 'none' for BatchNorm, InstanceNorm or no normalization. Default: 'bn'.
        conv_block: Type of convolutional block, like Convolution-Normalization-Activation. One of 'basic', 'partial' or 'gated'.
    """

    def __init__(
            self,
            num_input_channels=3,
            num_output_channels=3,
            feature_scale=4,
            more_layers=0,
            upsample_mode='bilinear',
            norm_layer='bn',
    ):
        super().__init__()

        self.feature_scale = feature_scale
        self.more_layers = more_layers

        if isinstance(num_input_channels, int):
            num_input_channels = [num_input_channels]

        if len(num_input_channels) < 5:
            num_input_channels += [0] * (5 - len(num_input_channels))

        self.num_input_channels = num_input_channels[:5]

        self.conv_block = BasicBlock

        filters = [64, 128, 256, 512, 1024]
        filters = [x // self.feature_scale for x in filters]

        # norm_layer = get_norm_layer(norm_layer)

        self.start = self.conv_block(self.num_input_channels[0], filters[0])

        self.down1 = DownsampleBlock(filters[0], filters[1] - self.num_input_channels[1], conv_block=self.conv_block)
        self.down2 = DownsampleBlock(filters[1], filters[2] - self.num_input_channels[2], conv_block=self.conv_block)
        self.down3 = DownsampleBlock(filters[2], filters[3] - self.num_input_channels[3], conv_block=self.conv_block)
        self.down4 = DownsampleBlock(filters[3], filters[4] - self.num_input_channels[4], conv_block=self.conv_block)

        if self.more_layers > 0:
            self.more_downs = [
                DownsampleBlock(filters[4], filters[4], conv_block=self.conv_block) for i in range(self.more_layers)]
            self.more_ups = [UpsampleBlock(filters[4], upsample_mode, same_num_filt=True, conv_block=self.conv_block)
                             for i in range(self.more_layers)]

            self.more_downs = ListModule(*self.more_downs)
            self.more_ups = ListModule(*self.more_ups)

        self.up4 = UpsampleBlock(filters[3], upsample_mode, conv_block=self.conv_block)
        self.up3 = UpsampleBlock(filters[2], upsample_mode, conv_block=self.conv_block)
        self.up2 = UpsampleBlock(filters[1], upsample_mode, conv_block=self.conv_block)
        self.up1 = UpsampleBlock(filters[0], upsample_mode, conv_block=self.conv_block)

        # Before refactoring, it was a nn.Sequential with only one module.
        # Need this for backward compatibility with model checkpoints.
        self.final = nn.Sequential(
            nn.Conv2d(filters[0], num_output_channels, 1)
        )

    def forward(self, *inputs, **kwargs):
        inputs = list(inputs)

        n_input = len(inputs)
        n_declared = np.count_nonzero(self.num_input_channels)
        assert n_input == n_declared, f'got {n_input} input scales but declared {n_declared}'

        in64 = self.start(inputs[0])
        down1 = self.down1(in64)

        if self.num_input_channels[1]:
            down1 = torch.cat([down1, inputs[1]], 1)

        down2 = self.down2(down1)

        if self.num_input_channels[2]:
            down2 = torch.cat([down2, inputs[2]], 1)

        down3 = self.down3(down2)

        if self.num_input_channels[3]:
            down3 = torch.cat([down3, inputs[3]], 1)

        down4 = self.down4(down3)
        if self.num_input_channels[4]:
            down4 = torch.cat([down4, inputs[4]], 1)

        if self.more_layers > 0:
            prevs = [down4]
            for kk, d in enumerate(self.more_downs):
                out = d(prevs[-1])
                prevs.append(out)

            up_ = self.more_ups[-1](prevs[-1], prevs[-2])
            for idx in range(self.more_layers - 1):
                l = self.more_ups[self.more - idx - 2]
                up_ = l(up_, prevs[self.more - idx - 2])
        else:
            up_ = down4

        up4 = self.up4(up_, down3)
        up3 = self.up3(up4, down2)
        up2 = self.up2(up3, down1)
        up1 = self.up1(up2, in64)

        return self.final(up1)


class UNet2(nn.Module):
    r""" Rendering network with UNet architecture and multi-scale input.
    Args:
        num_input_channels: Number of channels in the input tensor or list of tensors. An integer or a list of integers for each input tensor.
        num_output_channels: Number of output channels.
        more_layers: Additional down/up-sample layers.
        upsample_mode: One of 'deconv', 'bilinear' or 'nearest' for ConvTranspose, Bilinear or Nearest upsampling.
        norm_layer: [unused] One of 'bn', 'in' or 'none' for BatchNorm, InstanceNorm or no normalization. Default: 'bn'.
        conv_block: Type of convolutional block, like Convolution-Normalization-Activation. One of 'basic', 'partial' or 'gated'.
    """

    def __init__(
            self,
            num_input_channels=3,
            num_output_channels=3,
            channels_list=None,
            n_resblocks=0,
            start_ks=7,
            upsample_mode='bilinear',
            final_funnel=False
    ):
        super().__init__()

        if channels_list is None:
            channels_list = [16, 32, 54, 128, 256]

        if isinstance(num_input_channels, int):
            num_input_channels = [num_input_channels]

        if len(num_input_channels) < 5:
            num_input_channels += [0] * (5 - len(num_input_channels))

        self.num_input_channels = num_input_channels[:5]

        self.conv_block = BasicBlock

        filters = channels_list
        n_levels = len(filters) - 1
        self.start = self.conv_block(self.num_input_channels[0], filters[0], kernel_size_1=start_ks)

        down_blocks = []
        up_blocks = []
        for i in range(n_levels):
            in_channels = filters[i]
            out_channels = filters[i + 1] if len(self.num_input_channels) <= i + 1 else \
                filters[i + 1] - self.num_input_channels[1]

            down = DownsampleBlock(in_channels, out_channels, conv_block=self.conv_block)

            up = UpsampleBlock2(out_channels, in_channels, in_channels, upsample_mode, conv_block=self.conv_block)

            down_blocks.append(down)
            up_blocks.append(up)

        up_blocks = list(reversed(up_blocks))

        self.down_blocks = torch.nn.ModuleList(down_blocks)
        self.up_blocks = torch.nn.ModuleList(up_blocks)

        res_blocks = []
        for i in range(n_resblocks):
            res = self.conv_block(filters[-1], filters[-1])
            res_blocks.append(res)
        self.res_blocks = torch.nn.ModuleList(res_blocks)

        if not final_funnel:
            self.final = nn.Conv2d(filters[0], num_output_channels, 1)
        else:
            final_blocks = []
            filt0_logch = int(math.log2(filters[0]))
            out_logch = int(math.log2(num_output_channels))
            N_funnel_blocks = filt0_logch - out_logch - 1

            in_ch = filters[0]
            out_ch = 2 ** filt0_logch // 2
            for i in range(N_funnel_blocks):
                funn = self.conv_block(in_ch, out_ch)
                final_blocks.append(funn)

                in_ch = out_ch
                out_ch = out_ch // 2

            final_blocks.append(nn.Conv2d(in_ch, num_output_channels, 1))

            self.final = torch.nn.Sequential(*final_blocks)

    def forward(self, *inputs, **kwargs):
        inputs = list(inputs)

        n_input = len(inputs)
        n_declared = np.count_nonzero(self.num_input_channels)
        assert n_input == n_declared, f'got {n_input} input scales but declared {n_declared}'

        out = self.start(inputs[0])

        down_outputs = [out]
        for i in range(len(self.down_blocks)):
            inp = out
            out = self.down_blocks[i](inp)
            down_outputs.append(out)
        down_outputs = list(reversed(down_outputs))

        out_prev = out
        for i in range(len(self.res_blocks)):
            inp = (out + out_prev) / 2
            out = self.res_blocks[i](inp)
            out_prev = inp

        for i in range(len(self.up_blocks)):
            inp = out
            skip = down_outputs[i + 1]

            out = self.up_blocks[i](inp, skip)

        return self.final(out)
