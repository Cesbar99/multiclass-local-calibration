import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from utils.utils import *
from calibrator.local_net import *

def categorical_cross_entropy(probs, targets, eps=1e-8):
    probs = torch.clamp(probs, eps, 1.0 - eps)  # avoid log(0)
    return -torch.sum(targets * torch.log(probs), dim=1).mean()

def compute_multiclass_kl_divergence(p2, p1, y_one_hot, num_classes, js_distance=False,
                                     entropy_factor=1, eps=1e-4):
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
    kl = torch.sqrt(js_divergence)

    # Entropy-based weighting (currently unused, but placeholder kept)
    weights = 1.0

    if num_classes == 2:
        to_ret = torch.sum(kl * weights, dim=0, keepdim=True)[0]
    else:
        to_ret = torch.mean(torch.mean(kl * weights, dim=0, keepdim=True), dim=1)[0]

    return to_ret

def multiclass_neighborhood_class0_prob(means, z_hat, sigma, y, eps=1e-6):
    """
    means: (B, C)
    z_hat: (B, C)
    sigma: (B, C) — variance per dimension (not stddev)
    y: (B, C) — one-hot encoded label vector

    Returns:
        probs_classes: (B, C) — estimated P(y=c | Neigh_i) for each sample i
    """
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

    return probs_classes


class AuxTrainer(pl.LightningModule):
    def __init__(self, kwargs, num_classes):              
        super().__init__()
        self.model = AuxiliaryMLP(hidden_dim=kwargs.hidden_dim, latent_dim=num_classes, log_var_initializer=kwargs.log_var_initializer)
        self.num_classes = num_classes        
        self.optimizer_cfg = kwargs.optimizer
        self.alpha1 = kwargs.alpha1
        self.alpha2 = kwargs.alpha2
        self.lambda_kl = kwargs.lambda_kl
        self.entropy_factor = kwargs.entropy_factor
        self.noise = kwargs.noise
        self.smoothing = kwargs.smoothing
        self.logits_scaling = kwargs.logits_scaling
        self.sampling = kwargs.sampling
        self.predict_labels = kwargs.predict_labels
        self.use_empirical_freqs = kwargs.use_empirical_freqs
        self.js_distance = kwargs.js_distance        

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        init_logits, y_one_hot, init_preds, init_preds_one_hot = batch        

        # Add noise
        epsilon = torch.randn_like(init_logits)
        noisy_logits = init_logits + self.noise * epsilon

        # Optional label smoothing        
        noisy_y_one_hot = random_label_smoothing(y_one_hot, self.smoothing) if self.smoothing else y_one_hot

        # Forward pass
        latents = self(noisy_logits)
        means = latents[:, :self.num_classes]
        log_std = latents[:, self.num_classes:]
        stddev = F.softplus(log_std)
        sigma = stddev ** 2
        avg_variance = torch.mean(sigma) #dim=0

        # Reparameterization
        epsilon = torch.randn_like(means)
        z_hat = means + stddev * epsilon if self.sampling else means

        # Scaled probabilities
        probs_hat = F.softmax(z_hat / self.logits_scaling, dim=1)
        p2 = probs_hat

        # Neighborhood-based probabilities
        p1 = multiclass_neighborhood_class0_prob(means, z_hat, sigma=sigma, y=noisy_y_one_hot) 

        # KL divergence
        kl_loss = compute_multiclass_kl_divergence(p2, p1, y_one_hot, self.num_classes,
                                                   self.js_distance, self.entropy_factor)

        # Constraint loss
        new_preds = torch.argmax(probs_hat, dim=1)
        prediction_mask = (new_preds == init_preds).float()
        constraint = torch.mean(1.0 - prediction_mask)

        if self.predict_labels:
            target = y_one_hot 
        else:
            target = init_preds_one_hot 
        if self.use_empirical_freqs:
            scores = p1
        else:
            scores = p2
        constraint_loss = categorical_cross_entropy(scores, target) #F.cross_entropy(quantity, torch.argmax(target, dim=1))

        total_loss = (self.lambda_kl * kl_loss +
                      self.alpha1 * constraint_loss +
                      self.alpha2 * avg_variance ** 2)

        self.log("train_total", total_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train_kl", kl_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train_con_loss", constraint_loss, on_epoch=False, on_step=True, prog_bar=False)
        self.log("train_constraint", constraint, on_epoch=False, on_step=True, prog_bar=True)

        return total_loss
    
    def validation_step(self, batch):
        init_logits, y_one_hot, init_preds, init_preds_one_hot = batch

        # Add noise
        epsilon = torch.randn_like(init_logits)
        noisy_logits = init_logits + self.noise * epsilon

        # Optional label smoothing        
        noisy_y_one_hot = random_label_smoothing(y_one_hot, self.smoothing) if self.smoothing else y_one_hot

        # Forward pass
        latents = self(noisy_logits)
        means = latents[:, :self.num_classes]
        log_std = latents[:, self.num_classes:]
        stddev = F.softplus(log_std)
        sigma = stddev ** 2
        avg_variance = torch.mean(sigma)

        # Reparameterization
        epsilon = torch.randn_like(means)
        z_hat = means + stddev * epsilon if self.sampling else means

        # Scaled probabilities
        probs_hat = F.softmax(z_hat / self.logits_scaling, dim=1)
        p2 = probs_hat

        # Neighborhood-based probabilities
        p1 = multiclass_neighborhood_class0_prob(means, z_hat, sigma=sigma, y=noisy_y_one_hot) 

        # KL divergence
        kl_loss = compute_multiclass_kl_divergence(p2, p1, y_one_hot, self.num_classes,
                                                   self.js_distance, self.entropy_factor)

        # Constraint loss
        new_preds = torch.argmax(probs_hat, dim=1)
        prediction_mask = (new_preds == init_preds).float()
        constraint = torch.mean(1.0 - prediction_mask)

        if self.predict_labels:
            target = y_one_hot 
        else:
            target = init_preds_one_hot 
        if self.use_empirical_freqs:
            scores = p1
        else:
            scores = p2
        constraint_loss = categorical_cross_entropy(scores, target) #F.cross_entropy(quantity, torch.argmax(target, dim=1))

        total_loss = (self.lambda_kl * kl_loss +
                      self.alpha1 * constraint_loss +
                      self.alpha2 * avg_variance ** 2)

        self.log("val_total", total_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_kl", kl_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_con_loss", constraint_loss, on_epoch=True, on_step=False, prog_bar=False)
        self.log("val_constraint", constraint, on_epoch=True, on_step=False, prog_bar=True)        

    def configure_optimizers(self):
        opt_name = self.optimizer_cfg.name.lower()
        lr = self.optimizer_cfg.lr
        wd = self.optimizer_cfg.get("weight_decay", 0.0)

        # Dynamically get the optimizer class
        optimizer_class = getattr(torch.optim, opt_name.capitalize(), None)
        if optimizer_class is None:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        # Build kwargs dynamically
        optimizer_kwargs = {"lr": lr, "weight_decay": wd}
        if opt_name == "sgd":
            optimizer_kwargs["momentum"] = self.optimizer_cfg.get("momentum", 0.9)

        optimizer = optimizer_class(self.parameters(), **optimizer_kwargs)
        return optimizer
    
    def predict_step(self, batch):
        """
        Prediction step for a batch of data.
        :param batch:
        A tuple containing the input data `x`, target labels `y`, and feedback `h`.
        :param batch_idx:
        Index of the batch in the current epoch.
        :param dataloader_idx:
        Index of the dataloader (default is 0).
        :return:
            A dictionary containing the predictions, probabilities, feedback, true labels, rejection score, and selection status.
        """
        init_logits, y_one_hot, _, _ = batch        
        
        # Add noise
        epsilon = torch.randn_like(init_logits)
        noisy_logits = init_logits + self.noise * epsilon
        
        # Forward pass
        latents = self(noisy_logits)
        means = latents[:, :self.num_classes]
        log_std = latents[:, self.num_classes:]
        stddev = F.softplus(log_std)
        
        # Reparameterization
        epsilon = torch.randn_like(means)
        new_logits = means + stddev * epsilon if self.sampling else means
        
        # Scaled probabilities        
        preds = torch.argmax(new_logits, dim=-1).view(-1,1)
        target = torch.argmax(y_one_hot, dim=-1).view(-1,1)
        return {
            "preds": preds,            
            "true": target,
            "logits": new_logits,
        }
    
    