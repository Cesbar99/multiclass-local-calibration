import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from omegaconf import ListConfig

    
class AuxiliaryMLPV2(pl.LightningModule):
    def __init__(self, hidden_dim=64, feature_dim=2048, output_dim=2, similarity_dim=50, 
                 log_var_initializer=0.01, dropout_rate=0.1, linearly_combine_pca=False):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.similarity_dim = similarity_dim
        self.dropout_rate = dropout_rate        
        self.linearly_combine_pca = linearly_combine_pca
        
        if isinstance(log_var_initializer, (float, int, torch.Tensor)) and not hasattr(log_var_initializer, '__len__'):
            # Scalar case: fill with the same value            
            var_tensor = torch.full((1,), log_var_initializer)
        else:
            raise TypeError("var_init must be a float, int, or tensor")        
        
        # Inverse Softplus
        var_tensor = torch.log(torch.exp(var_tensor) - 1)        
        
        # Small weight initialization
        small_init = nn.init.normal_

        # Define layers
        self.dense1 = nn.Linear(feature_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p=self.dropout_rate)  # Dropout layer                                    
        
        self.similarity_head = nn.Linear(hidden_dim, similarity_dim)            
        self.classifcation_head = nn.Linear(hidden_dim, output_dim) 

        # Initialize weights manually
        small_init(self.dense1.weight, mean=0.0, std=0.01)
        small_init(self.similarity_head.weight, mean=0.0, std=0.01)
        small_init(self.classifcation_head.weight, mean=0.0, std=0.01)
            
        if self.linearly_combine_pca:   
            self.alpha_sim = nn.Parameter(torch.tensor(1., dtype=torch.float32)) 
            self.beta_sim = nn.Parameter(torch.empty(1).normal_(mean=0.0, std=0.01)) 
            self.alpha_cls = nn.Parameter(torch.tensor(1., dtype=torch.float32)) 
            self.beta_cls = nn.Parameter(torch.empty(1).normal_(mean=0.0, std=0.01)) 
            
    def forward(self, feats, logits, pca):        
        x = F.relu(self.dense1(feats))
        x = self.dropout1(x)
                
        if self.fixed_var:
            if self.linearly_combine_pca:
                similarity_out =  self.similarity_head(x) + self.alpha_sim*pca + self.beta_sim 
                classification_out = self.classifcation_head(x) + self.alpha_cls*logits + self.beta_cls.unsqueeze(0)
            else:
                similarity_out = self.similarity_head(x) + pca 
                classification_out = self.classifcation_head(x) + logits 
        
        return classification_out, similarity_out
    
    
    
