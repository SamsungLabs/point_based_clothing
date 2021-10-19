import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch
from collections import OrderedDict
from os.path import expanduser
import os


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(-1)


class PerceptualLoss(nn.Module):
    def __init__(self, weight, vgg_weights_dir, net='caffe', n_relus=7, relu_start=0):
        super().__init__()
        self.weight = weight
        self.n_relus = n_relus
        self.relu_start = relu_start

        if net == 'pytorch':
            vgg19 = torchvision.models.vgg19(pretrained=True).features
            
            mean = torch.tensor([0.485, 0.456, 0.406])
            std  = torch.tensor([0.229, 0.224, 0.225])


        elif net == 'caffe':
            vgg_weights = torch.load(os.path.join(vgg_weights_dir, 'vgg19-d01eb7cb.pth'))

            map = {'classifier.6.weight': u'classifier.7.weight', 'classifier.6.bias': u'classifier.7.bias'}
            vgg_weights = OrderedDict([(map[k] if k in map else k, v) for k, v in vgg_weights.items()])

            model = torchvision.models.vgg19()
            model.classifier = nn.Sequential(Flatten(), *model.classifier._modules.values())

            model.load_state_dict(vgg_weights)

            vgg19 = model.features

            mean = torch.tensor([103.939, 116.779, 123.680]) / 255.
            std = torch.tensor([1., 1., 1.]) / 255.

        elif net == 'face':
            # Load caffe weights for face vgg
            # from https://media.githubusercontent.com/media/yzhang559/vgg-face/master/VGG_FACE.caffemodel.pth
            vgg19 = torch.load(os.path.join(vgg_weights_dir, 'vgg_face.pth'))

            mean = torch.tensor([103.939, 116.779, 123.680]) / 255.
            std = torch.tensor([1., 1., 1.]) / 255.

        else:
            assert False

        self.register_buffer('mean', mean[None, :, None, None])
        self.register_buffer('std' ,  std[None, :, None, None])

        vgg19_avg_pooling = []

        for weights in vgg19.parameters():
            weights.requires_grad = False

        for module in vgg19.modules():
            if module.__class__.__name__ == 'Sequential':
                continue
            elif module.__class__.__name__ == 'MaxPool2d':
                vgg19_avg_pooling.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            else:
                vgg19_avg_pooling.append(module)

        vgg19_avg_pooling = nn.Sequential(*vgg19_avg_pooling)

        self.vgg19 = vgg19_avg_pooling

    def normalize_inputs(self, x):
        return (x - self.mean) / self.std

    def forward(self, input, target):
        if input.shape[0] != target.shape[0] or input.shape[0] == 0:
            return (input.sum() + target.sum()) * 0 # dirty hack

        input = (input + 1) / 2
        target = (target.detach() + 1) / 2

        loss = 0
        count = 0

        features_input = self.normalize_inputs(input)
        features_target = self.normalize_inputs(target)
        for layer in self.vgg19:
            features_input = layer(features_input)
            features_target = layer(features_target)

            if layer.__class__.__name__ == 'ReLU':
                if count >= self.relu_start:
                    loss = loss + F.l1_loss(features_input, features_target)
                count += 1
                if count == self.n_relus:
                    break

        return loss * self.weight
