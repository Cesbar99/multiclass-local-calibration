import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from omegaconf import ListConfig


class AuxiliaryMLP(pl.LightningModule):
    def __init__(self, hidden_dim=64, latent_dim=2, log_var_initializer=0.01, dropout_rate=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        
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
        self.dropout1 = nn.Dropout(p=self.dropout_rate)  # 🔹 Dropout layer
        
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(p=self.dropout_rate)  # 🔹 Dropout layer
        
        self.dense3 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout3 = nn.Dropout(p=self.dropout_rate)  # 🔹 Dropout layer
        
        self.dense4 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout4 = nn.Dropout(p=self.dropout_rate)  # 🔹 Dropout layer
        
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
        x = self.dropout1(x)
        
        # x = F.relu(self.dense2(x))
        # x = self.dropout2(x)
        
        # x = F.relu(self.dense3(x))
        # x = self.dropout3(x)

        # x = F.relu(self.dense4(x))
        # x = self.dropout4(x)
        
        out = self.dense8(x) + z_aug
        return out
    
    
class AuxiliaryMLPV2(pl.LightningModule):
    def __init__(self, hidden_dim=64, feature_dim=2048, output_dim=2, similarity_dim=50, log_var_initializer=0.01, dropout_rate=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.similarity_dim = similarity_dim
        self.dropout_rate = dropout_rate
        
        if isinstance(log_var_initializer, (float, int, torch.Tensor)) and not hasattr(log_var_initializer, '__len__'):
            # Scalar case: fill with the same value
            var_tensor = torch.full((similarity_dim,), log_var_initializer)
        elif isinstance(log_var_initializer, (ListConfig, list, tuple, np.ndarray, torch.Tensor)):
            # Vector case: convert to tensor and validate shape
            var_tensor = torch.tensor(log_var_initializer, dtype=torch.float32)
            if var_tensor.shape[0] != similarity_dim:
                raise ValueError(f"Length of var_init ({var_tensor.shape[0]}) must match output_dim ({similarity_dim})")
        else:
            raise TypeError("var_init must be a float, int, list, tuple, or tensor")        
        
        # Inverse Softplus
        var_tensor = torch.log(torch.exp(var_tensor) - 1)        
        
        # Small weight initialization
        small_init = nn.init.normal_

        # Define layers
        self.dense1 = nn.Linear(feature_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p=self.dropout_rate)  # Dropout layer                
        
        self.similarity_head = nn.Linear(hidden_dim, 2 * similarity_dim) #hidden_dim
        self.classifcation_head = nn.Linear(hidden_dim, output_dim) #hidden_dim

        # Initialize weights manually
        small_init(self.dense1.weight, mean=0.0, std=0.01)
        small_init(self.similarity_head.weight, mean=0.0, std=0.01)
        small_init(self.classifcation_head.weight, mean=0.0, std=0.01)

        # Bias initialization: [0.0]*similarity_dim + [var_init]*similarity_dim
        bias_init = torch.cat([
            torch.zeros(similarity_dim),
            var_tensor
        ])
        with torch.no_grad():
            self.similarity_head.bias.copy_(bias_init)
            #self.classifcation_head.bias.copy_(bias_init)
            
    def forward(self, feats, logits, pca):
        # z: (batch_size, latent_dim)
        # z_aug = torch.cat([z, torch.zeros_like(z[:, :self.output_dim])], dim=1)  # (batch_size, 2*latent_dim)
        # logits_aug = torch.cat([logits, torch.zeros_like(logits[:, :self.output_dim])], dim=1)  # (batch_size, 2*latent_dim)
        pca_aug = torch.cat([pca, torch.zeros_like(pca[:, :self.similarity_dim])], dim=1)  # (batch_size, 2*latent_dim)

        x = F.relu(self.dense1(feats))
        x = self.dropout1(x)
        
        similarity_out = self.similarity_head(x) + pca_aug  # (batch_size, 2*similarity_dim)
        classification_out = self.classifcation_head(x) + logits # self.dense8(x) + z_aug
        
        return classification_out, similarity_out
    
    
    