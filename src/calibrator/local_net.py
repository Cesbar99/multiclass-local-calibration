import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from omegaconf import ListConfig


class AuxiliaryMLP(pl.LightningModule):
    def __init__(self, hidden_dim=64, latent_dim=2, log_var_initializer=0.01):
        super().__init__()
        self.latent_dim = latent_dim
                
        if isinstance(log_var_initializer, (float, int, torch.Tensor)) and not hasattr(log_var_initializer, '__len__'):
            # Scalar case: fill with the same value
            var_tensor = torch.full((latent_dim,), log_var_initializer)
        elif isinstance(log_var_initializer, (ListConfig, list, tuple, np.ndarray, torch.Tensor)):
            # Vector case: convert to tensor and validate shape
            var_tensor = torch.tensor(log_var_initializer, dtype=torch.float32)
            if var_tensor.shape[0] != latent_dim:
                raise ValueError(f"Length of var_init ({var_tensor.shape[0]}) must match latent_dim ({latent_dim})")
        else:
            raise TypeError("var_init must be a float, int, list, tuple, or tensor")        
        
        # Inverse Softplus
        var_tensor = torch.log(torch.exp(var_tensor) - 1)        
        
        # Small weight initialization
        small_init = nn.init.normal_

        # Define layers
        self.dense1 = nn.Linear(latent_dim, hidden_dim)
        self.dense8 = nn.Linear(hidden_dim, 2 * latent_dim)

        # Initialize weights manually
        small_init(self.dense1.weight, mean=0.0, std=0.01)
        small_init(self.dense8.weight, mean=0.0, std=0.01)

        # Bias initialization: [0.0]*latent_dim + [var_init]*latent_dim
        bias_init = torch.cat([
            torch.zeros(latent_dim),
            var_tensor
        ])
        with torch.no_grad():
            self.dense8.bias.copy_(bias_init)

    def forward(self, z):
        # z: (batch_size, latent_dim)
        z_aug = torch.cat([z, torch.zeros_like(z[:, :self.latent_dim])], dim=1)  # (batch_size, 2*latent_dim)

        x = F.relu(self.dense1(z))
        out = self.dense8(x) + z_aug
        return out
    
    
    