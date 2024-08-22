import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, Counter

torch.autograd.set_detect_anomaly(True)

class Energy_based_Debias_Loss(nn.Module):
    def __init__(self, beta, num_components, labels, size_average=True):
        super(Energy_based_Debias_Loss, self).__init__()
        self.beta = beta
        self.num_components = num_components

        self.class_counter = Counter(labels)
        C = len(self.class_counter.keys())
        self.class_bias = []
        for i in sorted(self.class_counter.keys()):
            self.class_bias.append((C * self.class_counter[i]) / len(labels))

        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def calculate_probs(self, logits, mixing_weights):
        mix_probs = F.softmax(logits, dim=1) * mixing_weights.unsqueeze(dim=1)
        return mix_probs.sum(dim=-1)

    def forward(self, logits, targets, mixing_weights, epsilon=1e-10):

        # logits = logits + torch.log(torch.tensor(np.array(self.class_bias) + 1e-12,
        #                                          dtype=torch.float32,
        #                                          device='cuda')).unsqueeze(0).unsqueeze(2).detach()

        probs = self.calculate_probs(logits, mixing_weights)
        if (probs>1.0).sum() > 0 or (probs<=0.0).sum() > 0:
            print(0)
        if torch.isnan(probs).any():
            raise ValueError("Inputs contain NaN values")

        if torch.isinf(probs).any():
            raise ValueError("Inputs contain Inf values")

        return F.nll_loss(torch.log(probs + 1e-12), targets)




def create_loss(beta, lamda, labels):
    print('Loading Energy based Debias Loss.')
    loss = Energy_based_Debias_Loss(beta, lamda, labels)
    return loss