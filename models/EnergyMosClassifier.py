"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""

import torch.nn as nn
from utils import *
from os import path
import torch.nn.init as init


class DotProduct_MOS(nn.Module):

    def __init__(self, num_classes=1000, feat_dim=2048, num_components=3, *args):
        super(DotProduct_MOS, self).__init__()
        self.num_components = num_components

        self.components = nn.ModuleList([
            nn.Linear(feat_dim, num_classes) for _ in range(num_components)
        ])

        # for ablation study
        self.n_experts = num_components
        self.prior = nn.Linear(feat_dim, num_components, bias=False)

    # 还差一个求每个softmax的权重的函数。
    def calculate_weight(self, logits, targets):
        epsilon = 1e-10
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[torch.arange(logits.size(0)), targets] = False
        logits_wo_y = logits[mask].view(logits.shape[0], logits.shape[1]-1, logits.shape[2])

        boundary_energy = -torch.logsumexp(logits_wo_y, dim=1)
        energy = -torch.logsumexp(logits, dim=1)
        energy_y = logits[~mask].view(logits.shape[0], 1, logits.shape[2])

        mixing_weights = torch.clamp(1.0 - boundary_energy / energy, min=epsilon)

        sum_weights = torch.sum(mixing_weights, dim=1, keepdim=True)

        # # 进行归一化计算
        normalized_weights = mixing_weights / sum_weights

        return normalized_weights


    def forward(self, x, targets, *args):
        # for ablation_study
        prior_logit = self.prior(x).contiguous().view(-1, self.n_experts)
        mixing_weights = nn.functional.softmax(prior_logit, -1)

        x = torch.stack([component(x) for component in self.components], dim=2)
        # mixing_weights = self.calculate_weight(x, targets)

        return x, mixing_weights


def create_model(feat_dim, num_classes=1000, num_components=3, stage1_weights=False, dataset=None, log_dir=None, test=False, *args):
    print('Loading Dot Product Mixture of Classifier.')
    clf = DotProduct_MOS(num_classes, feat_dim, num_components)

    if not test:
        if stage1_weights:
            assert (dataset)
            print('Loading %s Stage 1 Classifier Weights.' % dataset)
            if log_dir is not None:
                subdir = log_dir.strip('/').split('/')[-1]
                subdir = subdir.replace('stage2', 'stage1')
                weight_dir = path.join('/'.join(log_dir.split('/')[:-1]), subdir)
                # weight_dir = path.join('/'.join(log_dir.split('/')[:-1]), 'stage1')
            else:
                weight_dir = './logs/%s/stage1' % dataset
            print('==> Loading classifier weights from %s' % weight_dir)
            clf.fc = init_weights(model=clf.fc,
                                  weights_path=path.join(weight_dir, 'final_model_checkpoint.pth'),
                                  classifier=True)
        else:
            print('Random initialized classifier weights.')

    return clf