'''config of predictor model'''

import torch

class MLPConfig:
    optimizer: torch.optim.Optimizer = torch.optim.Adam
    lr: float = 1e-3
    batch_size: int = 512
    epochs: int = 100
    use_final_epoch_model: bool = True

class FeaturesModelConfig:
    optimizer: torch.optim.Optimizer = torch.optim.Adam
    lr: float = 1e-3
    batch_size: int = 512
    epochs: int = 100
    use_final_epoch_model: bool = True

class PAESModelConfig:
    optimizer: torch.optim.Optimizer = torch.optim.RMSprop
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 50
    use_final_epoch_model: bool = True