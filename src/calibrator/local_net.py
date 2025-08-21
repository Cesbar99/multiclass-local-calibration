import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class AuxiliaryMLP(pl.LightningModule):
    def __init__(self, hidden_dim=64, latent_dim=2, log_var_initializer=0.01):
        super().__init__()
        self.latent_dim = latent_dim

        # Equivalent to log_var_initializer in TensorFlow
        var_init = torch.log(torch.tensor(log_var_initializer, dtype=torch.float32))

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
            torch.full((latent_dim,), var_init)
        ])
        with torch.no_grad():
            self.dense8.bias.copy_(bias_init)

    def forward(self, z):
        # z: (batch_size, latent_dim)
        z_aug = torch.cat([z, torch.zeros_like(z[:, :self.latent_dim])], dim=1)  # (batch_size, 2*latent_dim)

        x = F.relu(self.dense1(z))
        out = self.dense8(x) + z_aug
        return out
    
    
    