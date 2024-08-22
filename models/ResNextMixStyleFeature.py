"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.ModulatedAttLayer import ModulatedAttLayer
from models.mixstyle import MixStyle
from collections import OrderedDict


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, is_last=False):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               groups=groups, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.is_last = is_last

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class StridedConv(nn.Module):
    '''
    downsampling conv layer
    '''
    def __init__(self, in_planes, planes, use_relu=False):
        super(StridedConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=planes,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.use_relu = use_relu
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv(x)
        if self.use_relu:
            out = self.relu(out)

        return out


class ResNextMixstyle(nn.Module):

    def __init__(self, block, layers, groups=1, width_per_group=64,
                 use_modulatedatt=False, use_fc=False, dropout=None,
                 use_glore=False, use_gem=False):
        self.inplanes = 64
        super(ResNextMixstyle, self).__init__()

        self.groups = groups
        self.base_width = width_per_group


        self.p = 0.7
        self.eps = 1e-6
        alpha = 0.1
        self.beta = torch.distributions.Beta(alpha, alpha)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.downsampling = nn.Sequential(
            OrderedDict([(f'StridedConv{k}', StridedConv(in_planes=64*4*(2**k), planes=64*4*(2**(k+1)), use_relu=(k!=1))) for k in range(2)])
        )
        self.decoder = nn.Sequential(
        nn.Conv2d(256*4*2, 256*4, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
        )


        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.use_fc = use_fc
        self.use_dropout = True if dropout else False

        if self.use_fc:
            print('Using fc.')
            self.fc_add = nn.Linear(512 * block.expansion, 512)

        if self.use_dropout:
            print('Using dropout.')
            self.dropout = nn.Dropout(p=dropout)

        self.use_modulatedatt = use_modulatedatt
        if self.use_modulatedatt:
            print('Using self attention.')
            self.modulatedatt = ModulatedAttLayer(in_channels=512 * block.expansion)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, is_last=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            groups=self.groups, base_width=self.base_width))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                groups=self.groups, base_width=self.base_width,
                                is_last=(is_last and i == blocks - 1)))

        return nn.Sequential(*layers)

    def forward(self, x, *args):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        if random.random() > self.p and self.training:
            B = x.size(0)
            perm = torch.randperm(B)

            lmda = self.beta.sample((B, 1, 1, 1))
            lmda = lmda.to(x.device)

            mu_x = x.mean(dim=[2, 3], keepdim=True)
            var_x = x.var(dim=[2, 3], keepdim=True)
            sig_x = (var_x + self.eps).sqrt()
            mu_x, sig_x = mu_x.detach(), sig_x.detach()
            x_norm = (x - mu_x) / sig_x

            mu2, sig2 = mu_x[perm], sig_x[perm]
            mu_mix = mu_x * lmda + mu2 * (1 - lmda)
            sig_mix = sig_x * lmda + sig2 * (1 - lmda)
            x = x_norm * sig_mix + mu_mix

        x_shallow = self.downsampling(x)

        x = self.layer2(x)
        x = self.layer3(x)

        x = self.decoder(torch.cat((x_shallow, x),dim=1))

        x = self.layer4(x)

        if self.use_modulatedatt:
            x, feature_maps = self.modulatedatt(x)
        else:
            feature_maps = None

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        if self.use_fc:
            x = F.relu(self.fc_add(x))

        if self.use_dropout:
            x = self.dropout(x)

        return x, feature_maps