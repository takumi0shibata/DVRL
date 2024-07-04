import torch
import torch.nn as nn


class DataValueEstimator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        comb_dim: int,
        layer_number: int,
        act_fn: callable
    ) -> None:
        """
        Args:
          input_dim: The dimensionality of the input features (x_input and y_input combined).
          hidden_dim: The dimensionality of the hidden layers.
          comb_dim: The dimensionality of the combined layer.
          layer_number: Total number of layers in the MLP before combining with y_hat.
          act_fn: Activation function to use.
        """

        super(DataValueEstimator, self).__init__()
        
        self.act_fn = act_fn
        # Initial layer
        self.initial_layer = nn.Linear(input_dim, hidden_dim)
        # Intermediate layers
        self.intermediate_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(layer_number - 3)]
        )
        # Layer before combining with y_hat
        self.pre_comb_layer = nn.Linear(hidden_dim, comb_dim)
        # Layer after combining with y_hat
        self.comb_layer = nn.Linear(comb_dim + 1, comb_dim)
        # Output layer
        self.output_layer = nn.Linear(comb_dim, 1)
        
    def forward(
        self,
        x_input: torch.Tensor,
        y_input: torch.Tensor,
        y_hat_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
          x_input: Input features.
          y_input: Target labels.
          y_hat_input: Predicted labels or some representation thereof.
          
        Returns:
          Tensor: The estimated data values.
        """
        inputs = torch.cat((x_input, y_input), dim=1)
        
        # Initial layer
        x = self.act_fn(self.initial_layer(inputs))
        
        # Intermediate layers
        for layer in self.intermediate_layers:
            x = self.act_fn(layer(x))
        
        # Pre-combination layer
        x = self.act_fn(self.pre_comb_layer(x))
        
        # Combining with y_hat_input
        x = torch.cat((x, y_hat_input), dim=1)  # Ensure y_hat_input is properly shaped
        x = self.act_fn(self.comb_layer(x))
        
        # Output layer with sigmoid activation
        x = torch.sigmoid(self.output_layer(x))
        
        return x