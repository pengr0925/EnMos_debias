import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, Counter

class DebiasLoss(nn.Module):
    def __init__(self, beta, lamda, labels, size_average=True):
        super(DebiasLoss, self).__init__()
        self.beta = beta
        self.lamda = lamda

        self.class_counter = Counter(labels)

        self.C = len(self.class_counter.keys())
        self.class_bias = []
        for i in self.class_counter.keys():
            self.class_bias.append((self.C * self.class_counter[i]) / len(labels))

        # self.base_probs = []
        # for i in self.class_counter.keys():
        #     self.base_probs.append(self.class_counter[i] / len(labels))

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, targets, adaptive_marg_coef, w_norm):

        margin_lf = torch.log(torch.tensor(np.array(self.class_bias) + 1e-12,
                                                 dtype=torch.float32,
                                                 device='cuda')).detach()

        sub_orig = logits - torch.gather(logits, 1, targets[:, None])
        dispersion = torch.log(1 + (torch.gather(logits, 1, targets[:, None]).squeeze()/w_norm[targets] - w_norm[targets]).pow(2))

        margin_2 = torch.zeros_like(logits)
        margin_2.scatter_(1, targets.view(-1, 1), dispersion[:, None])

        for i in range(logits.shape[0]):
            if (sub_orig[i] > 0.0).sum() == 0.0: # hard negtive sample
                margin_2[i] = 0.0

        verify_1 = nn.functional.softmax(logits + self.lamda * margin_lf.detach(), dim=1)
        logits = logits - self.beta * adaptive_marg_coef * margin_2 + self.lamda * margin_lf.detach()
        verify_2 = nn.functional.softmax(logits, dim=1)
        if (verify_1[range(logits.shape[0]), targets] < verify_2[range(logits.shape[0]), targets]).sum() > 0.0:
            print('Warning !!!!')

        return self.criterion(logits, targets)


def create_loss(beta, lamda, labels):
    print('Loading Softmax Loss.')
    loss = DebiasLoss(beta, lamda, labels)
    return loss
