import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import numpy as np

def count_dataset(labels):
    result_count = Counter(labels)
    base_probs = []
    for i in result_count.keys():
        base_probs.append(result_count[i]/len(labels))
    return base_probs


class LALoss(nn.Module):
    def __init__(self, labels):
        super(LALoss, self).__init__()
        self.base_probs = count_dataset(labels)
        self.tau = 1.0

    def forward(self, logits, target,*args):
        logits = logits + torch.log(torch.tensor(np.array(self.base_probs) ** self.tau + 1e-12,
                                    dtype=torch.float32,
                                    device='cuda').detach())
        return F.cross_entropy(logits, target)


def create_loss(labels):
    print('Loading Logit Adjustment Loss.')
    loss = LALoss(labels)
    return loss