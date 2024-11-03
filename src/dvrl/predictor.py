# coding=utf-8
"""Predictor model during dvrl training"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Simple Essay Scorer
    """
    def __init__(
        self,
        input_feature: int,
        hidden_dim: int = 512
    ) -> None:
        """
        Args:
            input_feature: Number of input features
            hidden_dim: Number of hidden units
        """
        super(MLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_feature, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
