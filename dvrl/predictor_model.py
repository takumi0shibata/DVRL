# coding=utf-8
"""Predictor model during dvrl training"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Simple Essay Scorer
    """
    def __init__(self, input_feature: int) -> None:
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_feature, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
