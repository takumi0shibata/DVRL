"""Loss function for data valuation"""

import torch
import torch.nn as nn


class DvrlLoss(nn.Module):
    def __init__(
        self,
        epsilon: float,
        threshold: float,
        std_penalty_weight: float = None
    ) -> None:
        """
        Args:
            epsilon: Small value to avoid overflow
            threshold: Encourages exploration
            std_penalty_weight: Weight for the standard deviation penalty term
        """
        super().__init__()
        self.epsilon = epsilon
        self.threshold = threshold
        self.std_penalty_weight = std_penalty_weight

    def forward(self, est_data_value, s_input, reward_input):
        """
        Calculate the loss.
        Args:
            est_data_value: Estimated data value
            s_input: data selection array
            reward_input: Reward
        Returns:
            dve_loss: Loss value
        """
        # Generator loss (REINFORCE algorithm)
        one = torch.ones_like(est_data_value, dtype=est_data_value.dtype)
        prob = torch.sum(s_input * torch.log(est_data_value + self.epsilon) + \
                         (one - s_input) * \
                         torch.log(one - est_data_value + self.epsilon))

        zero = torch.Tensor([0.0])
        zero = zero.to(est_data_value.device)

        dve_loss = (-reward_input * prob) + \
                   1e3 * torch.maximum(torch.mean(est_data_value) - self.threshold, zero) + \
                   1e3 * torch.maximum(1 - self.threshold - torch.mean(est_data_value), zero)
        
        # Variance penalty term
        if self.std_penalty_weight is not None:
            variance_penalty = self.std_penalty_weight * torch.var(est_data_value)
            dve_loss -= variance_penalty

        return dve_loss
