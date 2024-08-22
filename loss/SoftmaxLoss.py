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

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, targets, *args):
        return self.criterion(logits, targets)


def create_loss(labels):
    print('Loading Softmax Loss.')
    loss = CELoss()
    return loss


