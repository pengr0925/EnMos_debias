import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, Counter

torch.autograd.set_detect_anomaly(True)

class Energy_based_Debias_Loss(nn.Module):
    def __init__(self, beta, lamda, labels, size_average=True):
        super(Energy_based_Debias_Loss, self).__init__()
        self.beta = beta
        self.lamda = lamda

        self.m_above = -7.0
        self.m_below = -25.0

        self.class_counter = Counter(labels)

        C = len(self.class_counter.keys())
        self.class_bias = []
        for i in self.class_counter.keys():
            self.class_bias.append((C * self.class_counter[i]) / len(labels))

        # self.base_probs = []
        # for i in self.class_counter.keys():
        #     self.base_probs.append(self.class_counter[i] / len(labels))

        self.criterion = nn.CrossEntropyLoss()

    def sample_gumble(self, logits, targets):

        logits_wo_y = torch.cat(
            [torch.cat((logits[i][0:j], logits[i][j + 1:])) for i, j in enumerate(targets)]
        ).view(logits.shape[0], logits.shape[1] - 1)

        boundary_energy = -torch.logsumexp(logits_wo_y, dim=1)
        energy = -torch.logsumexp(logits, dim=1)
        energy_y = - logits[torch.arange(logits.size(0)), targets]
        gumble_beta = torch.where(boundary_energy > 0, 0.0*boundary_energy, boundary_energy) / energy

        U = torch.rand_like(logits)
        return -torch.log(-torch.log(U + 1e-10) + 1e-10) * self.beta * gumble_beta.unsqueeze(1)

    def gumble_logits(self, logits, gumble_noises):
        return logits + gumble_noises

    def forward(self, logits, targets, adaptive_marg_coef, w_norm):

        gumble_noises = self.sample_gumble(logits, targets)
        logits = self.gumble_logits(logits, gumble_noises)

        logits = logits + torch.log(torch.tensor(np.array(self.class_bias) + 1e-12,
                                                 dtype=torch.float32,
                                                 device='cuda')).detach()

        return self.criterion(logits, targets)




def create_loss(beta, lamda, labels):
    print('Loading Energy based Debias Loss.')
    loss = Energy_based_Debias_Loss(beta, lamda, labels)
    return loss