import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, Counter

class DebiasLoss(nn.Module):
    def __init__(self, labels, size_average=True):
        super(DebiasLoss, self).__init__()
        self.class_counter = Counter(labels)

        C = len(self.class_counter.keys())
        # self.class_bias = []
        # for i in self.class_counter.keys():
        #     self.class_bias.append((C * self.class_counter[i]) / len(labels))

        self.class_bias_1 = []
        for i in self.class_counter.keys():
            self.class_bias_1.append(len(labels) / (C * self.class_counter[i]))



        self.base_probs = []
        for i in self.class_counter.keys():
            self.base_probs.append(self.class_counter[i] / len(labels))

        self.criterion = nn.CrossEntropyLoss()


    def forward(self, logits, targets, *args):
        logits = logits + torch.log(torch.tensor(np.array(self.class_bias) + 1e-12,
                                                 dtype=torch.float32,
                                                 device='cuda')).detach()

        # post-hoc
        # logits = logits - torch.log(torch.tensor(np.array(self.class_bias_1) + 1e-12,
        #                                          dtype=torch.float32,
        #                                          device='cuda')).detach()

        return self.criterion(logits, targets)




def create_loss(beta, lamda, labels):
    print('Loading Softmax Loss.')
    loss = DebiasLoss(labels)
    return loss
