import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from utils.utils import *
from calibrator.local_net import *


def categorical_cross_entropy(probs, targets, eps=1e-8):
    probs = torch.clamp(probs, eps, 1.0 - eps)  # avoid log(0)
    return -torch.sum(targets * torch.log(probs), dim=1).mean()

def compute_multiclass_js_dist(p2, p1, num_classes, eps=1e-4):
    # Clip values to avoid log(0)
    p2 = torch.clamp(p2, eps, 1 - eps)
    p1 = torch.clamp(p1, eps, 1 - eps)
    m = 0.5 * (p1 + p2)
    m = torch.clamp(m, eps, 1 - eps)

    # KL divergence terms
    kl_p1_m = p1 * torch.log(p1 / m) + (1 - p1) * torch.log((1 - p1) / (1 - m))
    kl_p2_m = p2 * torch.log(p2 / m) + (1 - p2) * torch.log((1 - p2) / (1 - m))

    # Jensen-Shannon divergence
    js_divergence = 0.5 * (kl_p1_m + kl_p2_m)
    js_divergence = torch.maximum(js_divergence, torch.tensor(eps, device=js_divergence.device))

    # JS distance
    js_dist = torch.sqrt(js_divergence)

    if num_classes == 2:
        to_ret = torch.sum(js_dist, dim=0, keepdim=True)[0]
    else:
        to_ret = torch.mean(torch.mean(js_dist, dim=0, keepdim=True), dim=1)[0]

    return to_ret

def multiclass_neighborhood_class0_prob(means, z_hat, sigma, y, eps=1e-6, ret_weights=False):
    B = z_hat.shape[0]

    # Expand dims for broadcasting
    means_i = means.unsqueeze(1)  # (B, 1, C)
    z_j = z_hat.unsqueeze(0)      # (1, B, C)

    # Compute squared differences
    diffs_squared = (z_j - means_i) ** 2  # (B, B, C)

    # Expand sigma for broadcasting
    sigma_i = sigma.unsqueeze(1)  # (B, 1, C)

    # Gaussian kernel exponent
    exponent = -diffs_squared / (2.0 * sigma_i + eps)  # (B, B, C)

    # Sum over dimensions
    sq_dists = torch.sum(exponent, dim=-1, keepdim=False)  # (B, B)

    # Unnormalized similarity scores
    p_unnorm = torch.exp(sq_dists)  # (B, B)

    # Mask out self-contributions
    mask = 1.0 - torch.eye(B, device=p_unnorm.device)
    p_unnorm = p_unnorm * mask

    # Normalize across neighbors
    p_norm = p_unnorm / (torch.sum(p_unnorm, dim=1, keepdim=True) + eps)
    p_norm = torch.clamp(p_norm, eps, 1.0 - eps)

    # Weighted sum over neighbors
    probs_classes = torch.matmul(p_norm, y)  # (B, C)
    if ret_weights:
        return probs_classes, p_norm
    else:
        return probs_classes

class AuxTrainerV2(pl.LightningModule):
    def __init__(self, kwargs, num_classes, feature_dim, similarity_dim):              
        super().__init__()                            
        self.num_classes = num_classes        
        self.optimizer_cfg = kwargs.optimizer        
        self.alpha1 = kwargs.alpha1        
        self.lambda_js = kwargs.lambda_js
        self.log_var_initializer = kwargs.log_var_initializer        
        self.smoothing = kwargs.smoothing                
        self.predict_labels = kwargs.predict_labels
        self.use_empirical_freqs = kwargs.use_empirical_freqs
        self.js_distance = kwargs.js_distance           
        self.feature_dim = feature_dim
        self.similarity_dim = similarity_dim                
        self.linearly_combine_pca = kwargs.linearly_combine_pca
        self.device_name = 'cuda'

        self.model = AuxiliaryMLPV2(hidden_dim=kwargs.hidden_dim, feature_dim=feature_dim, output_dim=num_classes, 
                                    similarity_dim=similarity_dim, log_var_initializer=kwargs.log_var_initializer, 
                                    dropout_rate=kwargs.dropout, linearly_combine_pca=kwargs.linearly_combine_pca)
        
    def forward(self, init_feats, init_logits, init_pca):
        return self.model(init_feats, init_logits, init_pca)

    def training_step(self, batch):                
        init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch        

        # Optional label smoothing        
        noisy_y_one_hot = label_smoothing(y_one_hot, self.smoothing) if self.smoothing else y_one_hot 

        # Forward pass
        latents_class, latents_sim = self(init_feats, init_logits, init_pca)
        means = latents_sim[:, :self.similarity_dim]               
        var_tensor = torch.full((1,), self.log_var_initializer)
        var_tensor = torch.log(torch.exp(var_tensor) - 1)       
        stddev = F.softplus(var_tensor).to(self.device_name)        
            
        sigma = stddev ** 2        
        z_hat = means

        # Probabilities
        probs_hat = F.softmax(latents_class, dim=1)
        p2 = probs_hat

        # Neighborhood-based probabilities
        p1, weights = multiclass_neighborhood_class0_prob(means, z_hat, sigma=sigma, y=noisy_y_one_hot, ret_weights=True) 

        # KL divergence
        js_loss = compute_multiclass_js_dist(p2, p1, self.num_classes)
        
        target = y_one_hot        
        scores = p1
            
        constraint_loss = categorical_cross_entropy(scores, target)
            
        total_loss = (self.lambda_js * js_loss +
                      self.alpha1 * constraint_loss)

        self.log("train_total", total_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train_js", kl_loss, on_epoch=True, on_step=False, prog_bar=False)
        self.log("train_con_loss", constraint_loss, on_epoch=True, on_step=False, prog_bar=True)

        return total_loss
    
    def validation_step(self, batch):
        init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch                

        # Optional label smoothing        
        noisy_y_one_hot = label_smoothing(y_one_hot, self.smoothing) if self.smoothing else y_one_hot

        # Forward pass
        latents_class, latents_sim = self(init_feats, init_logits, init_pca)
        means = latents_sim[:, :self.similarity_dim]        
        var_tensor = torch.full((1,), self.log_var_initializer)
        var_tensor = torch.log(torch.exp(var_tensor) - 1)       
        stddev = F.softplus(var_tensor).to(self.device_name)       
        sigma = stddev ** 2

        z_hat = means

        # Probabilities
        probs_hat = F.softmax(latents_class, dim=1)
        p2 = probs_hat

        # Neighborhood-based probabilities
        p1, weights = multiclass_neighborhood_class0_prob(means, z_hat, sigma=sigma, y=noisy_y_one_hot, ret_weights=True) 

        # KL divergence
        js_loss = compute_multiclass_js_dist(p2, p1, self.num_classes)

        target = y_one_hot 
        scores = p1    
            
        constraint_loss = categorical_cross_entropy(scores, target) 
        
        total_loss = (self.lambda_kl * kl_loss +
                      self.alpha1 * constraint_loss) 
        optuna_loss = kl_loss + constraint_loss 
            
        self.log("val_total", total_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("optuna_loss", optuna_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_kl", kl_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_con_loss", constraint_loss, on_epoch=True, on_step=False, prog_bar=False)

    def configure_optimizers(self):
        opt_name = self.optimizer_cfg.name
        opt_name = opt_name[0].upper() + opt_name[1:]
        lr = self.optimizer_cfg.lr
        wd = self.optimizer_cfg.get("weight_decay", 0.0)        
        
        # Dynamically get the optimizer class
        if 'dam' in opt_name:
            optimizer_class = getattr(torch.optim, opt_name, None)
        elif 'gd' in opt_name:
            optimizer_class = getattr(torch.optim, 'SGD', None)
        
        if optimizer_class is None:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        # Build kwargs dynamically
        optimizer_kwargs = {"lr": lr, "weight_decay": wd}
        if opt_name == "SGD":
            optimizer_kwargs["momentum"] = self.optimizer_cfg.get("momentum", 0.9)

        optimizer = optimizer_class(self.parameters(), **optimizer_kwargs)
        return optimizer
    
    def predict_step(self, batch):
        init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch                 
        
        # Forward pass
        latents_class, latents_sim = self(init_feats, init_logits, init_pca)
        means = latents_sim[:, :self.similarity_dim] 
                
        new_logits = latents_class
        
        # Probabilities and Labels      
        preds = torch.argmax(latents_class, dim=-1).view(-1,1)
        target = torch.argmax(y_one_hot, dim=-1).view(-1,1)
                
        return {
            "preds": preds,            
            "true": target,
            "logits": new_logits,
        }
        
    def extract_pca(self, batch):
        init_feats, init_logits, init_pca, y_one_hot, _, _ = batch 
        
        # Forward pass
        logits, latents_sim = self(init_feats, init_logits, init_pca)
        means = latents_sim[:, :self.similarity_dim] #init_pca        
        var_tensor = torch.full((1,), self.log_var_initializer)
        var_tensor = torch.log(torch.exp(var_tensor) - 1)       
        stddev = F.softplus(var_tensor).to(self.device_name)        
        sigma = stddev**2
                
        preds = torch.argmax(logits, dim=-1).view(-1,1)  # predicted class
        # Create dict in the same format as predict outputs            
        out = {
            "features": means, #means, #init_pca       
            "logits": logits,
            "preds": preds,     
            "true": torch.argmax(y_one_hot, dim=-1).view(-1,1)
        }
        return out

       
