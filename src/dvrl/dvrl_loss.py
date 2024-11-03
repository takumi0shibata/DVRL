"""Loss function for data valuation"""

import math
import torch
import torch.nn as nn


class DvrlLoss(nn.Module):
    def __init__(
        self,
        epsilon: float,
        threshold: float,
    ) -> None:
        """
        Args:
            epsilon: Small value to avoid overflow
            threshold: Encourages exploration
        """
        super().__init__()
        self.epsilon = epsilon
        self.threshold = threshold

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

        return dve_loss
    

class DvrlLossV2(nn.Module):
    def __init__(self, epsilon: float = 1e-8, gamma: float = 0.1, penalty=False) -> None:
        """
        Args:
            epsilon: Small value to avoid numerical instability
        """
        super().__init__()
        self.epsilon = epsilon
        self.gamma = gamma
        self.penalty = penalty

    def forward(self, sampling_data_num, estimated_lambda, reward, cluster_size):
        """
        Calculate the loss for the two-stage sampling process.
        Args:
            sampling_data_num: Tensor of sampled data counts from the Poisson distribution per cluster
            estimated_lambda: Tensor of Poisson distribution parameters per cluster (output of the policy network)
            reward: Reward calculated based on model performance
        Returns:
            dve_loss: Loss value
        """
        device = estimated_lambda.device

        # Ensure sampling_data_num is a tensor on the correct device
        if not isinstance(sampling_data_num, torch.Tensor):
            sampling_data_num = torch.tensor(sampling_data_num, dtype=torch.float32, device=device)

        # Compute the log factorial term using lgamma for numerical stability
        log_factorial = torch.lgamma(sampling_data_num + 1)

        # Compute the log probability per cluster
        log_prob = -estimated_lambda + sampling_data_num * torch.log(estimated_lambda + self.epsilon) - log_factorial \
                    #   - torch.lgamma(cluster_size + 1) + torch.lgamma(cluster_size - sampling_data_num + 1) + torch.lgamma(sampling_data_num + 1)

        # Sum log probabilities over all clusters
        total_log_prob = log_prob.sum()

        if self.penalty:
            # Add penalty to encourage higher estimated_lambda
            # lambda_penalty = -self.gamma * estimated_lambda.mean()
            lambda_penalty = -self.gamma * torch.std(estimated_lambda)
            # Compute the loss using the REINFORCE algorithm
            dve_loss = -reward * total_log_prob + lambda_penalty

        # Compute the loss using the REINFORCE algorithm
        dve_loss = -reward * total_log_prob

        return dve_loss
