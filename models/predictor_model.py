# coding=utf-8
"""
Predictor model during dvrl training
"""

import torch
import torch.nn as nn


class EssayScorer(nn.Module):
    """
    Essay Scorer
    """
    def __init__(self, input_feature=2048):
        super(EssayScorer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_feature, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
