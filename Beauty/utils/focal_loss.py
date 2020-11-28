# -*- coding: utf-8 -*-
from abc import ABC

import torch
import torch.nn as nn

__all__ = ["FocalLoss"]


class FocalLoss(nn.Module, ABC):
    """
        -α(1-yi)**γ *ce_loss(xi,yi)
    """
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()