import torch
import torch.nn.functional as F
import os
from os.path import join
import csv
import pandas as pd 
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torchvision import transforms
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import re
import glob

import pytorch_lightning as pl
import torch

import pytorch_lightning as pl
import torch

from data_sets.dataset import CalibrationDatasetv2
from torch.utils.data import Dataset, DataLoader

class CalibrationPlotCallback(pl.Callback):
    def __init__(self, kwargs, dataloader, every_n_epochs=10, device="cuda", type='train'):
        super().__init__()
        self.dataloader = dataloader
        self.every_n_epochs = every_n_epochs
        self.device = device
        self.data = kwargs.data
        self.num_classes = kwargs.dataset.num_classes
        self.num_features = kwargs.dataset.num_features
        self.save_path = kwargs.save_path_calibration_plots
        self.type=type
        self.verion = kwargs.calibrator_version
        self.quantize = kwargs.quantize
        if not self.quantize:
            self.lambda_kl = kwargs.models.lambda_kl
            self.kernel_only = kwargs.models.kernel_only        
        
    def on_train_epoch_end(self, trainer, pl_module):
        # Only run every N epochs
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        pl_module.eval()
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch in self.dataloader:
                if self.verion == 'v2':
                    init_feats, init_logits, init_pca, y_one_hot, _, _ = batch # it is actually init_feats!!!!!
                    init_feats = init_feats.to(self.device)
                    init_pca = init_pca.to(self.device)
                else:
                    init_logits, y_one_hot, _, _ = batch
                init_logits = init_logits.to(self.device)
                y_one_hot = y_one_hot.to(self.device)

                if self.quantize:
                    probs, _, _, _, _ = pl_module(init_feats)
                else:
                    # Add noise
                    epsilon = torch.randn_like(init_logits)
                    noisy_logits = init_logits + pl_module.noise * epsilon

                    # Optional label smoothing        
                    y_one_hot = label_smoothing(y_one_hot, pl_module.smoothing) if pl_module.smoothing else y_one_hot

                    # Forward pass                
                    if self.verion != 'v2':
                        latents = pl_module(noisy_logits)
                        means = latents[:, :pl_module.num_classes]
                        log_std = latents[:, pl_module.num_classes:]
                        stddev = F.softplus(log_std)

                        # Reparameterization
                        epsilon = torch.randn_like(means)
                        z_hat = means + stddev * epsilon if pl_module.sampling else means
                    else:
                        latents, _ = pl_module(init_feats, init_logits, init_pca)
                        z_hat = latents
                    
                    # Scaled probabilities
                    probs = F.softmax(z_hat / pl_module.logits_scaling, dim=1)                               

                all_probs.append(probs.cpu())
                all_targets.append(torch.argmax(y_one_hot.cpu(), dim=1))
               
        all_probs = torch.cat(all_probs)
        all_targets = torch.cat(all_targets)

        if self.quantize:   
            appendix = f"quantcal_{self.data}_{self.num_classes}_classes_{self.num_features}_features/in_training/"
        else:
            appendix = f"calibrate_{self.data}_{self.num_classes}_classes_{self.num_features}_features/in_training/"
            if self.lambda_kl == 0:
                appendix = f"reference_kernel_{self.data}_{self.num_classes}_classes_{self.num_features}_features/in_training/"
            if self.kernel_only:
                appendix = f"kernel_only_{self.data}_{self.num_classes}_classes_{self.num_features}_features/in_training/"
        # Call your function
        multiclass_calibration_plot(all_targets, all_probs, 
                                    save_path=self.save_path + appendix, #f"calibrate_{self.data}_{self.num_classes}_classes_{self.num_features}_features/in_training/", 
                                    filename=f"multiclass_calibration_{self.type}_cal_ep{trainer.current_epoch}.png")

        # Switch back to training mode
        pl_module.train()
        

class ClearCacheCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.empty_cache()


class VerboseModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_best_score = None

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)

        # Check if a new best model was saved
        if self.best_model_score is not None:
            if self._last_best_score is None or self.best_model_score < self._last_best_score:
                current_epoch = trainer.current_epoch
                print(f"\nNew best model saved at epoch {current_epoch}: {self.best_model_path} with val_total = {self.best_model_score:.4f}\n")
                self._last_best_score = self.best_model_score
       
    
def estimate_bandwidth_silverman(z_cal_full: torch.Tensor):
        """
        z_cal_full: (N_cal, D) tensor
        Returns per-dimension stddev h: (D,)
        """
        N, D = z_cal_full.shape
        # empirical std per dimension
        std = z_cal_full.std(dim=0, unbiased=True)  # (D,)
        factor = (4.0 / (D + 2.0)) ** (1.0 / (D + 4.0)) * (N ** (-1.0 / (D + 4.0)))
        h = factor * std  # (D,)
        return h
  

def load_optuna_config(csv_path, kwargs, pretrain=False):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with open(csv_path, mode="r") as f:
        reader = csv.DictReader(f)
        row = next(reader)  # only one row assumed

    print("Loaded Optuna config from CSV")

    for key, value in row.items():
        key = key.strip()
        value = value.strip()
        
        if key in ["study_name", "value", "optuna_epochs", "train_epochs", "max_iter"]:
            continue

        # type parsing
        if value in ["True", "False"]:
            parsed_value = value == "True"
        else:
            try:
                parsed_value = int(value)
            except ValueError:
                try:
                    parsed_value = float(value)
                except ValueError:
                    parsed_value = value  # string (e.g. optimizer)
        if pretrain:
            kwargs[key] = parsed_value                    
        else:
            kwargs.models[key] = parsed_value
             
                    
def get_raw_res(raws, features=False, adabw=False, reduced_dim=None, fit_pca=None, quantize=False):
    
    preds = torch.cat([raws[j]["preds"].cpu() for j in range(len(raws))])
    #probs = torch.cat([raws[j]["probs"].cpu() for j in range(len(raws))])
    logits = torch.cat([raws[j]["logits"].cpu() for j in range(len(raws))])
    feats = torch.cat([raws[j]["features"].cpu() for j in range(len(raws))]) if features else None
    true = torch.cat([raws[j]["true"].cpu() for j in range(len(raws))])
    if quantize:
        indices = torch.cat([raws[j]["indices"].cpu() for j in range(len(raws))])
        alpha = torch.cat([raws[j]["alpha"].cpu() for j in range(len(raws))])
    if adabw:
        sigma = torch.cat([raws[j]["bandwidth"].cpu() for j in range(len(raws))])
    
    raw_res = pd.DataFrame()
    raw_res["true"] = true.numpy().flatten()
    raw_res["preds"] = preds.numpy()
    
    #raw_res["logits"] = logits.numpy()
    #raw_res["probs"] = probs.numpy()
    #logits_tmp = pd.DataFrame()
    #feats_tmp = pd.DataFrame()

    #for i in range(logits.shape[1]):
        #tmp["class_probs_{}".format(i)] = probs[:, i].cpu().numpy()
    #    logits_tmp["logits_{}".format(i)] = logits[:, i].cpu().numpy()
    logits_np = logits.cpu().numpy()
    logits_tmp = pd.DataFrame(
        logits_np, columns=[f"logits_{i}" for i in range(logits_np.shape[1])]
    )    
    if quantize:
        #raw_res["indices"] = indices.numpy()
        indices_np = indices.cpu().numpy()
        indices_tmp = pd.DataFrame(
            indices_np, columns=[f"indices_{i}" for i in range(indices_np.shape[1])]
        )    
        alpha_np = alpha.cpu().numpy()
        alpha_tmp = pd.DataFrame(
            alpha_np, columns=[f"alpha_{i}" for i in range(alpha_np.shape[1])]
        ) 
    if features:        
        #for i in range(feats.shape[1]):
        #    feats_tmp["features_{}".format(i)] = feats[:, i].cpu().numpy()
        feats_np = feats.cpu().numpy()
        feats_tmp = pd.DataFrame(
            feats_np, columns=[f"features_{i}" for i in range(feats_np.shape[1])]
        )        
        if reduced_dim is not None and reduced_dim > 0:
            print('RUNNING PCA ON FEATURES TO REDUCE DIM TO: ', reduced_dim)
            # Standardization            
            scaler = StandardScaler()
            feats_scaled = scaler.fit_transform(feats_tmp.values)
            if fit_pca is not None:
                pca = fit_pca
            else:                                
                # PCA
                print('FITTING PCA...')
                pca = PCA(n_components=reduced_dim)
                pca = pca.fit(feats_scaled)
                
            feats_pca = pca.transform(feats_scaled)
            feats_pca_tmp = pd.DataFrame(
                feats_pca, columns=[f"pca_{i}" for i in range(feats_pca.shape[1])]
            )
            feats_tmp = pd.concat([feats_tmp, feats_pca_tmp], axis=1)
        else:
            pca = None
    else:
        pca = None
            
    if adabw:
        sigma_np = sigma.cpu().numpy()
        sigma_tmp = pd.DataFrame(
            sigma_np, columns=[f"bandwidth" for i in range(sigma_np.shape[1])]
        )                

    raw_res = pd.concat([raw_res, logits_tmp], axis=1)    
    if quantize:
        raw_res = pd.concat([raw_res, indices_tmp], axis=1)        
        raw_res = pd.concat([raw_res, alpha_tmp], axis=1)
    if features:        
        raw_res = pd.concat([raw_res, feats_tmp], axis=1)    
    if adabw:
        raw_res = pd.concat([raw_res, sigma_tmp], axis=1)
    return raw_res, pca

def create_logdir(name: str, resume_training: bool, wandb_logger):
    basepath = os.path.dirname(os.path.abspath(sys.argv[0]))
    basepath = os.path.join(os.path.dirname(os.path.dirname(basepath)), 'result')
    basepath = join(basepath, 'runs', name)
    # basepath = join(os.path.dirname(os.path.abspath(sys.argv[0])),'runs', name)
    run_name = wandb_logger.experiment.name
    logdir = join(basepath,run_name)
    if os.path.exists(logdir) and not resume_training:
        raise Exception(f'Run {run_name} already exists. Please delete the folder {logdir} or choose a different run name.')
    os.makedirs(logdir,exist_ok=True)
    return logdir

def subsample_triplets(probs: torch.Tensor, 
                       y_true: torch.Tensor, 
                       pca: torch.Tensor, 
                       n_samples: int = 10000):
    """
    Randomly subsample n_samples elements from (probs, y_true, pca)
    while keeping them aligned.

    Args:
        probs: (N, C) tensor of predicted probabilities
        y_true: (N,) tensor of labels
        pca: (N, d) tensor of features
        n_samples: number of elements to sample

    Returns:
        probs_sub, y_true_sub, pca_sub
    """
    N = probs.shape[0]
    n_samples = min(n_samples, N)  # safety if N < 10k

    idx = torch.randperm(N)[:n_samples]  # random permutation
    probs_sub = probs[idx]
    y_true_sub = y_true[idx]
    pca_sub = pca[idx]

    return probs_sub, y_true_sub, pca_sub

def compute_multiclass_calibration_metrics_w_lce_adabw(
    probs: torch.Tensor,
    y_true: torch.Tensor,
    pca: torch.Tensor,
    bw: torch.Tensor,    
    n_bins: int = 15,    
    gamma: float = 0.1,
    full_ece: bool = False,
    bin_strategy: str = 'default'
):
    """
    Computes:
      - ECCE (per-class then averaged)
      - ECE (per-class then averaged or list if full_ece)
      - MCE (averaged across classes)
      - Brier score (averaged across classes)
      - NLL
      - LCE metrics (average absolute LCE and average MLCE across classes)

    Parameters:
      probs: (N, C) predicted probabilities
      y_true: (N,) true labels (long)
      pca: (N, d) feature vectors (used to compute kernel similarities)
      n_bins: number of confidence bins
      gamma: bandwidth for Gaussian kernel (float)
      full_ece: if True, returns per-class ECE/ECCE lists (and LCE-list)

    Returns:
      avg_ecce, avg_ece, avg_mce, avg_brier, nll, avg_lce, avg_mlce
      (If full_ece=True, avg_ece / avg_ecce / lce_list are lists of per-class values)
    """    
    device = probs.device
    N, n_classes = probs.shape
    
    # Confidence and correctness
    #conf, preds = torch.max(probs, dim=1)            # (N,)
    #correct = (preds == y_true).float().to(device)   # (N,)

    # Metrics containers
    ecces = []
    eces = []
    mces = []
    briers = []
    per_class_lce_avg = []   # mean abs LCE for each class
    per_class_mlce = []      # max abs LCE for each class

    # Negative log-likelihood
    log_probs = torch.log(probs + 1e-12)  # numerical stability
    loss = F.nll_loss(log_probs, y_true, reduction='mean')
    nll = loss.item()

    # Precompute kernel matrix K from pca features (Gaussian kernel).
    # Ensure pca shape is (N, d) and on same device
    pca = pca.to(device).float()
    if pca.dim() == 1:
        pca = pca.view(N, 1)
    if pca.shape[0] != N:
        raise ValueError("pca must have same first dimension as probs (N samples).")

    # pairwise squared distances (N x N)
    #diff = pca.unsqueeze(1) - pca.unsqueeze(0)         # (N, N, d)
    #D = (diff * diff).sum(dim=2)                        # (N, N)
    # pca: (N, d)

    # Gaussian kernel: k_ij = exp(-||phi_i - phi_j||^2 / (2 * gamma^2))
    if gamma <= 0:
        raise ValueError("gamma must be > 0")    
    #K = torch.exp(-D / (2.0 * (gamma ** 2)))            # (N, N)
    eps = 1e-12

    for class_idx in range(n_classes):
        print('Class ', class_idx)
        # One-vs-all labels & class probabilities
        labels_binary = (y_true == class_idx).float().to(device)   # (N,)
        probs_class = probs[:, class_idx].to(device)              # (N,)

        # Brier Score for this class
        brier = torch.mean((probs_class - labels_binary) ** 2).item()
        if bin_strategy == 'quantile':
            # probs_class: (N,) tensor
            arr = probs_class.detach().cpu().numpy()

            # Compute quantile edges: [0%, 100%] split into n_bins intervals
            bin_edges_np = np.quantile(arr, np.linspace(0.0, 1.0, n_bins + 1))
            bin_edges_np[0] = 0.0    # force exact 0
            bin_edges_np[-1] = 1.0   # force exact 1

            # Convert back to torch tensor on correct device
            bin_edges = torch.tensor(bin_edges_np, dtype=torch.float32, device=device)

            # Assign samples to bins (like np.digitize, left-inclusive, right-exclusive)
            bin_indices = torch.bucketize(probs_class, bin_edges, right=False) - 1
            bin_indices = bin_indices.clamp(0, n_bins - 1)  # keep in range [0, n_bins-1]
        else:
            # Bin predictions (confidence bins)
            bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=device)
            # bucketize returns integer bin indices in [0..n_bins]; we will iterate 1..n_bins
            bin_indices = torch.bucketize(probs_class, bin_edges, right=True)
            #bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=device)
            #bin_indices = torch.bucketize(probs_class, bin_edges, right=False) - 1
            #bin_indices = bin_indices.clamp(0, n_bins - 1)  # ensure within [0, n_bins-1]                      

        total_count = probs_class.numel()
        ece = 0.0
        mce = 0.0

        # store bin stats for ECCE
        bin_accs = []
        bin_confs = []
        bin_weights = []

        # For LCE: we will compute per-sample LCE values (initialized to zero)
        lce_vals = torch.zeros(N, device=device)
        # keep track of all indices you computed LCE for
        valid_idx = []

        # iterate bins (1..n_bins) like your original code
        for b in range(1, n_bins + 1):
            idx_in_bin = (bin_indices == b).nonzero(as_tuple=True)[0]
            #print(idx_in_bin)
            if idx_in_bin.numel() == 0:
                continue

            # ECE / MCE computation (bin-level)
            bin_probs = probs_class[idx_in_bin]
            bin_labels = labels_binary[idx_in_bin]
            bin_accuracy = torch.mean(bin_labels).item()
            bin_confidence = torch.mean(bin_probs).item()
            bin_error = abs(bin_accuracy - bin_confidence)

            bin_accs.append(bin_accuracy)
            bin_confs.append(bin_confidence)
            bin_weights.append(idx_in_bin.numel() / total_count)

            ece += bin_error * idx_in_bin.numel() / total_count
            mce = max(mce, bin_error)
            
            print(idx_in_bin.numel())
            # ---------- LCE computation for all samples in this bin ----------
            if idx_in_bin.numel() > 20: #and idx_in_bin.numel() < 17000
                #bin_indices = torch.where(idx_in_bin)[0]   # indices of samples in this bin
                pca_bin = pca[idx_in_bin]               # (n_b, d)
                gamma_bin = bw[idx_in_bin]          # (n_b,) bandwidths for samples in this bin
                
                # expand gamma so that each row i uses gamma_i
                gamma_sq = (gamma_bin) #(gamma_bin ** 2)                # (n_b, 1)
                denom_gamma = 2.0 * gamma_sq               # (n_b, 1)

                # squared distance matrix (n_b × n_b)
                sq_norms = (pca_bin ** 2).sum(dim=1, keepdim=True)
                D_bin = sq_norms + sq_norms.t() - 2 * pca_bin @ pca_bin.t()
                D_bin = torch.clamp(D_bin, min=0)
                
                # kernel matrix; K_sub: (s, s) kernel submatrix with points in this bin
                Ksub = torch.exp(- D_bin / denom_gamma)  # (n_b, n_b)
                           
                #Ksub = K_bin[idx_in_bin][:, idx_in_bin]              # (s, s)
                # local calibration residual per sample in bin: (p_j - 1[f(xj)==yj])
                e_sub = (probs_class[idx_in_bin] - labels_binary[idx_in_bin]).to(device)  # (s,)
                # numerator for each sample i in this bin: sum_j K[i,j] * e_j
                numer = Ksub.matmul(e_sub)                       # (s,)
                # denominator for each sample i: sum_j K[i,j]
                denom = Ksub.sum(dim=1)                          # (s,)

                # avoid division by zero: if denom==0, set LCE to 0 for those samples
                denom_safe = denom.clone()
                zero_mask = denom_safe <= eps
                denom_safe[zero_mask] = 1.0  # temporary to compute ratios
                lce_sub = numer / (denom_safe + eps)             # (s,)
                if zero_mask.any():
                    lce_sub[zero_mask] = 0.0

                # write back into global per-sample LCE array
                lce_vals[idx_in_bin] = lce_sub
                valid_idx.append(idx_in_bin)

        # ECCE computation (CDF difference) for this class
        if len(bin_accs) > 0:
            bin_accs_t = torch.tensor(bin_accs, device=device, dtype=torch.float32)
            bin_confs_t = torch.tensor(bin_confs, device=device, dtype=torch.float32)
            bin_weights_t = torch.tensor(bin_weights, device=device, dtype=torch.float32)

            cum_pred = torch.cumsum(bin_weights_t * bin_confs_t, dim=0)
            cum_true = torch.cumsum(bin_weights_t * bin_accs_t, dim=0)
            ecce_val = torch.sum(torch.abs(cum_pred - cum_true)).item()
            # normalize by number of non-empty bins (keeps ECCE in [0,1])
            ecce_val /= len(bin_accs)
        else:
            ecce_val = 0.0

        # aggregate per-class metrics
        ecces.append(ecce_val)
        eces.append(ece)
        mces.append(mce)
        briers.append(brier)

        # flatten valid indices
        if len(valid_idx) > 0:
            valid_idx = torch.cat(valid_idx)
            lce_abs = torch.abs(lce_vals[valid_idx])
            per_class_lce_avg.append(float(lce_abs.mean().item()))
            per_class_mlce.append(float(lce_abs.max().item()))
        else:
            per_class_lce_avg.append(0.0)
            per_class_mlce.append(0.0)  

    # Final aggregation
    if full_ece:
        avg_ece = [round(x, 4) for x in eces]
        avg_ecce = [round(x, 4) for x in ecces]
        lce_list = [round(x, 4) for x in per_class_lce_avg]   # per-class mean-abs-LCE
        mlce_list = [round(x, 4) for x in per_class_mlce]      # per-class max-abs-LCE
    else:
        avg_ece = sum(eces) / len(eces)
        avg_ecce = sum(ecces) / len(ecces)
        lce_list = None

    avg_mce = sum(mces) / len(mces)
    avg_brier = sum(briers) / len(briers)

    # LCE aggregated across classes
    avg_lce = sum(per_class_lce_avg) / len(per_class_lce_avg)
    avg_mlce = sum(per_class_mlce) / len(per_class_mlce)

    # Return order:
    # avg_ecce, avg_ece, avg_mce, avg_brier, nll, avg_lce, avg_mlce
    if full_ece:
        # also return per-class LCE list when full_ece requested
        return avg_ecce, avg_ece, avg_mce, avg_brier, nll, lce_list, mlce_list
    else:
        return avg_ecce, avg_ece, avg_mce, avg_brier, nll, avg_lce, avg_mlce

"""
def compute_multiclass_calibration_metrics_w_lce(
    probs: torch.Tensor,
    y_true: torch.Tensor,
    pca: torch.Tensor,
    class_freqs: list,
    n_bins: int = 15,
    gamma: float = 0.1,
    full_ece: bool = False,
    bin_strategy: str = 'default', # quantile
    data: str = 'cifar10',
    model_type: str = 'resnet'
):
    '''
    Computes:
      - ECCE (per-class then averaged)
      - ECE (per-class then averaged or list if full_ece)
      - MCE (averaged across classes)
      - Brier score (averaged across classes)
      - NLL
      - LCE metrics (average absolute LCE and average MLCE across classes)

    Parameters:
      probs: (N, C) predicted probabilities
      y_true: (N,) true labels (long)
      pca: (N, d) feature vectors (used to compute kernel similarities)
      n_bins: number of confidence bins
      gamma: bandwidth for Gaussian kernel (float)
      full_ece: if True, returns per-class ECE/ECCE lists (and LCE-list)

    Returns:
      avg_ecce, avg_ece, avg_mce, avg_brier, nll, avg_lce, avg_mlce
      (If full_ece=True, avg_ece / avg_ecce / lce_list are lists of per-class values)
    '''    
    device = probs.device
    N, n_classes = probs.shape
    
    # Confidence and correctness
    #conf, preds = torch.max(probs, dim=1)            # (N,)
    #correct = (preds == y_true).float().to(device)   # (N,)

    # Metrics containers
    ecces = []
    eces = []
    mces = []
    # briers = []
    per_class_lce_avg = []   # mean abs LCE for each class
    per_class_mlce = []      # max abs LCE for each class
        
    if data == 'food101':
        filter = 10 # less samples for classes so reduce
    else:
        filter = 10 if model_type == "vit" else 20# filter for too little ESS when computing local metrics

    # Negative log-likelihood
    log_probs = torch.log(probs + 1e-12)  # numerical stability
    loss = F.nll_loss(log_probs, y_true, reduction='mean')
    nll = loss.item()
    
    # Classic multiclass Brier score
    y_onehot = F.one_hot(y_true, num_classes=n_classes).float().to(device)
    avg_brier = torch.mean(torch.sum((probs - y_onehot) ** 2, dim=1)).item()

    # Precompute kernel matrix K from pca features (Gaussian kernel).
    # Ensure pca shape is (N, d) and on same device
    pca = pca.to(device).float()
    if pca.dim() == 1:
        pca = pca.view(N, 1)
    if pca.shape[0] != N:
        raise ValueError("pca must have same first dimension as probs (N samples).")

    # pairwise squared distances (N x N)
    #diff = pca.unsqueeze(1) - pca.unsqueeze(0)         # (N, N, d)
    #D = (diff * diff).sum(dim=2)                        # (N, N)
    # pca: (N, d)

    # Gaussian kernel: k_ij = exp(-||phi_i - phi_j||^2 / (2 * gamma^2))
    if gamma <= 0:
        raise ValueError("gamma must be > 0")    
    #K = torch.exp(-D / (2.0 * (gamma ** 2)))            # (N, N)
    eps = 1e-12

    for class_idx in range(n_classes):
        print('Class ', class_idx)
        # One-vs-all labels & class probabilities
        labels_binary = (y_true == class_idx).float().to(device)   # (N,)
        probs_class = probs[:, class_idx].to(device)              # (N,)

        # Brier Score for this class
        #brier = torch.mean((probs_class - labels_binary) ** 2).item()
        if bin_strategy == 'quantile':
            # probs_class: (N,) tensor
            arr = probs_class.detach().cpu().numpy()

            # Compute quantile edges: [0%, 100%] split into n_bins intervals
            bin_edges_np = np.quantile(arr, np.linspace(0.0, 1.0, n_bins + 1))
            bin_edges_np[0] = 0.0    # force exact 0
            bin_edges_np[-1] = 1.0   # force exact 1

            # Convert back to torch tensor on correct device
            bin_edges = torch.tensor(bin_edges_np, dtype=torch.float32, device=device)

            # Assign samples to bins (like np.digitize, left-inclusive, right-exclusive)
            bin_indices = torch.bucketize(probs_class, bin_edges, right=False) - 1
            bin_indices = bin_indices.clamp(0, n_bins - 1)  # keep in range [0, n_bins-1]
        else:
            # Bin predictions (confidence bins)
            bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=device)
            # bucketize returns integer bin indices in [0..n_bins]; we will iterate 1..n_bins
            bin_indices = torch.bucketize(probs_class, bin_edges, right=True)
            bin_indices = bin_indices.clamp(1, n_bins)
            #bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=device)
            #bin_indices = torch.bucketize(probs_class, bin_edges, right=False) - 1
            #bin_indices = bin_indices.clamp(0, n_bins - 1)  # ensure within [0, n_bins-1]                                  

        total_count = probs_class.numel()
        ece = 0.0
        mce = 0.0

        # store bin stats for ECCE
        bin_accs = []
        bin_confs = []
        bin_weights = []

        # For LCE: we will compute per-sample LCE values (initialized to zero)
        lce_vals = torch.zeros(N, device=device)
        # keep track of all indices you computed LCE for
        valid_idx = []

        # iterate bins (1..n_bins) 
        for b in range(1, n_bins+1): #range(1, n_bins + 1): #range(n_bins): #
            idx_in_bin = (bin_indices == b).nonzero(as_tuple=True)[0]
            #print(idx_in_bin)
            if idx_in_bin.numel() == 0:
                continue

            # ECE / MCE computation (bin-level)
            bin_probs = probs_class[idx_in_bin]
            bin_labels = labels_binary[idx_in_bin]
            bin_accuracy = torch.mean(bin_labels).item()
            bin_confidence = torch.mean(bin_probs).item()
            bin_error = abs(bin_accuracy - bin_confidence)

            bin_accs.append(bin_accuracy)
            bin_confs.append(bin_confidence)
            bin_weights.append(idx_in_bin.numel() / total_count)

            ece += bin_error * idx_in_bin.numel() / total_count
            mce = max(mce, bin_error)
            
            print(idx_in_bin.numel())
            # ---------- LCE computation for all samples in this bin ----------
            if idx_in_bin.numel() > filter: #and idx_in_bin.numel() < 17000
                #bin_indices = torch.where(idx_in_bin)[0]   # indices of samples in this bin
                pca_bin = pca[idx_in_bin]               # (n_b, d)

                # squared distance matrix (n_b × n_b)
                sq_norms = (pca_bin ** 2).sum(dim=1, keepdim=True)
                D_bin = sq_norms + sq_norms.t() - 2 * pca_bin @ pca_bin.t()
                D_bin = torch.clamp(D_bin, min=0)

                # kernel matrix; K_sub: (s, s) kernel submatrix with points in this bin
                Ksub = torch.exp(-D_bin / (2.0 * gamma**2))  # (n_b, n_b)
                
                # w = Ksub
                # row_sum = w.sum(dim=1, keepdim=True) + 1e-12
                # w_norm = w / row_sum
                # ess = 1.0 / (w_norm.pow(2).sum(dim=1) + 1e-12)  # (n_b,)

                # print("ESS mean:", ess.mean().item(), " / bin size:", w.size(0))
                # print("max weight mean:", w_norm.max(dim=1).values.mean().item())
                                        
                #Ksub = K_bin[idx_in_bin][:, idx_in_bin]              # (s, s)
                # local calibration residual per sample in bin: (p_j - 1[f(xj)==yj])
                e_sub = (probs_class[idx_in_bin] - labels_binary[idx_in_bin]).to(device)  # (s,)
                # numerator for each sample i in this bin: sum_j K[i,j] * e_j
                numer = Ksub.matmul(e_sub)                       # (s,)
                # denominator for each sample i: sum_j K[i,j]
                denom = Ksub.sum(dim=1)                          # (s,)

                # avoid division by zero: if denom==0, set LCE to 0 for those samples
                denom_safe = denom.clone()
                zero_mask = denom_safe <= eps
                denom_safe[zero_mask] = 1.0  # temporary to compute ratios
                lce_sub = numer / (denom_safe + eps)             # (s,)
                if zero_mask.any():
                    lce_sub[zero_mask] = 0.0

                # write back into global per-sample LCE array
                lce_vals[idx_in_bin] = lce_sub
                valid_idx.append(idx_in_bin)

        # ECCE computation (CDF difference) for this class
        if len(bin_accs) > 0:
            bin_accs_t = torch.tensor(bin_accs, device=device, dtype=torch.float32)
            bin_confs_t = torch.tensor(bin_confs, device=device, dtype=torch.float32)
            bin_weights_t = torch.tensor(bin_weights, device=device, dtype=torch.float32)

            cum_pred = torch.cumsum(bin_weights_t * bin_confs_t, dim=0)
            cum_true = torch.cumsum(bin_weights_t * bin_accs_t, dim=0)
            ecce_val = torch.sum(torch.abs(cum_pred - cum_true)).item()
            # normalize by number of non-empty bins (keeps ECCE in [0,1])
            ecce_val /= len(bin_accs)
        else:
            ecce_val = 0.0

        # aggregate per-class metrics
        ecces.append(ecce_val)
        eces.append(ece)
        mces.append(mce)
        # briers.append(brier)

        # flatten valid indices
        if len(valid_idx) > 0:
            valid_idx = torch.cat(valid_idx)
            lce_abs = torch.abs(lce_vals[valid_idx])
            per_class_lce_avg.append(float(lce_abs.mean().item()))
            per_class_mlce.append(float(lce_abs.max().item()))
        else:
            per_class_lce_avg.append(0.0)
            per_class_mlce.append(0.0)  

    # Final aggregation
    if full_ece:
        avg_ece = [round(x, 4) for x in eces]
        avg_ecce = [round(x, 4) for x in ecces]
        lce_list = [round(x, 4) for x in per_class_lce_avg]   # per-class mean-abs-LCE
        mlce_list = [round(x, 4) for x in per_class_mlce]      # per-class max-abs-LCE
    else:
        avg_ece = np.dot(np.array(eces).T, np.array(class_freqs)) #sum(eces) / len(eces)
        avg_ecce = np.dot(np.array(ecces).T, np.array(class_freqs)) #sum(ecces) / len(ecces)
        lce_list = None

    avg_mce = np.dot(np.array(mces).T, np.array(class_freqs)) #sum(mces) / len(mces)
    # avg_brier = np.dot(np.array(briers).T, np.array(class_freqs)) #sum(briers) / len(briers)

    # LCE aggregated across classes
    avg_lce = np.dot(np.array(per_class_lce_avg).T, np.array(class_freqs)) # sum(per_class_lce_avg) / len(per_class_lce_avg) 
    avg_mlce = np.dot(np.array(per_class_mlce).T, np.array(class_freqs)) # sum(per_class_mlce) / len(per_class_mlce) 

    # Return order:
    # avg_ecce, avg_ece, avg_mce, avg_brier, nll, avg_lce, avg_mlce
    if full_ece:
        # also return per-class LCE list when full_ece requested
        return avg_ecce, avg_ece, avg_mce, avg_brier, nll, lce_list, mlce_list
    else:
        return avg_ecce, avg_ece, avg_mce, avg_brier, nll, avg_lce, avg_mlce
"""

def compute_multiclass_calibration_metrics_w_lce(
    probs: torch.Tensor,
    y_true: torch.Tensor,
    pca: torch.Tensor,
    class_freqs: list,
    n_bins: int = 15,
    n_bins_esse: int = 15,
    gamma: float = 0.1,
    full_ece: bool = False,
    bin_strategy: str = 'default',  # 'quantile' or default
    data: str = 'cifar10',
    model_type: str = 'resnet'
):
    """
    Computes:
      - ECCE (per-class then averaged)
      - ECE (per-class then averaged or list if full_ece)
      - MCE (averaged across classes)
      - Brier score
      - NLL
      - LCE metrics (average absolute LCE and average MLCE across classes)
      - NEW: ESS-binned LCE profile, where support is measured by ESS
             computed from kernel weights within the same confidence bin
             and excluding self-contribution.

    Parameters:
      probs: (N, C) predicted probabilities
      y_true: (N,) true labels (long)
      pca: (N, d) feature vectors
      class_freqs: class frequencies for weighted aggregation
      n_bins: number of bins used both for confidence and ESS profile
      gamma: Gaussian kernel bandwidth
      full_ece: if True, return per-class ECE/ECCE/LCE metrics and ESS profiles
      bin_strategy: 'quantile' or default uniform confidence bins

    Returns:
      If full_ece=False:
        avg_ecce, avg_ece, avg_mce, avg_brier, nll, avg_lce, avg_mlce, ess_lce_profile

      If full_ece=True:
        avg_ecce, avg_ece, avg_mce, avg_brier, nll, lce_list, mlce_list, ess_lce_profile

      where ess_lce_profile is a dict with:
        - 'avg_abs_lce_per_ess_bin': weighted average abs LCE in each ESS bin
        - 'avg_ess_per_bin': weighted average ESS in each ESS bin
        - 'count_per_bin': total number of valid samples in each ESS bin
        - 'per_class': list with the same information for each class
    """
    device = probs.device
    N, n_classes = probs.shape
    eps = 1e-12
    n_bins_ess = n_bins_esse
    
    # Metrics containers
    ecces = []
    eces = []
    mces = []
    per_class_lce_avg = []
    per_class_mlce = []

    # NEW: store per-class ESS-binned profiles
    per_class_ess_profiles = []

    if data == 'food101':
        filter_thr = 10
    else:
        filter_thr = 10 if model_type == "vit" else 20

    # Negative log-likelihood
    log_probs = torch.log(probs + 1e-12)
    loss = F.nll_loss(log_probs, y_true, reduction='mean')
    nll = loss.item()

    # Multiclass Brier
    y_onehot = F.one_hot(y_true, num_classes=n_classes).float().to(device)
    avg_brier = torch.mean(torch.sum((probs - y_onehot) ** 2, dim=1)).item()

    # PCA checks
    pca = pca.to(device).float()
    if pca.dim() == 1:
        pca = pca.view(N, 1)
    if pca.shape[0] != N:
        raise ValueError("pca must have same first dimension as probs (N samples).")

    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    for class_idx in range(n_classes):
        print('Class ', class_idx)

        labels_binary = (y_true == class_idx).float().to(device)   # (N,)
        probs_class = probs[:, class_idx].to(device)               # (N,)

        # Confidence bins
        if bin_strategy == 'quantile':
            arr = probs_class.detach().cpu().numpy()
            bin_edges_np = np.quantile(arr, np.linspace(0.0, 1.0, n_bins + 1))
            bin_edges_np[0] = 0.0
            bin_edges_np[-1] = 1.0
            bin_edges = torch.tensor(bin_edges_np, dtype=torch.float32, device=device)

            # indices in [0, n_bins-1]
            bin_indices = torch.bucketize(probs_class, bin_edges, right=False) - 1
            bin_indices = bin_indices.clamp(0, n_bins - 1)
            bin_loop = range(n_bins)
        else:
            bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=device)
            # indices in [1, n_bins]
            bin_indices = torch.bucketize(probs_class, bin_edges, right=True)
            bin_indices = bin_indices.clamp(1, n_bins)
            bin_loop = range(1, n_bins + 1)

        total_count = probs_class.numel()
        ece = 0.0
        mce = 0.0

        # For ECCE
        bin_accs = []
        bin_confs = []
        bin_weights = []

        # Per-sample LCE and ESS
        lce_vals = torch.zeros(N, device=device)
        ess_vals = torch.zeros(N, device=device)

        valid_idx = []

        for b in bin_loop:
            idx_in_bin = (bin_indices == b).nonzero(as_tuple=True)[0]
            if idx_in_bin.numel() == 0:
                continue

            # ECE / MCE
            bin_probs = probs_class[idx_in_bin]
            bin_labels = labels_binary[idx_in_bin]
            bin_accuracy = torch.mean(bin_labels).item()
            bin_confidence = torch.mean(bin_probs).item()
            bin_error = abs(bin_accuracy - bin_confidence)

            bin_accs.append(bin_accuracy)
            bin_confs.append(bin_confidence)
            bin_weights.append(idx_in_bin.numel() / total_count)

            ece += bin_error * idx_in_bin.numel() / total_count
            mce = max(mce, bin_error)

            print(idx_in_bin.numel())

            # ---------- LCE + ESS computation in this confidence bin ----------
            if idx_in_bin.numel() > filter_thr:
                pca_bin = pca[idx_in_bin]  # (s, d)

                sq_norms = (pca_bin ** 2).sum(dim=1, keepdim=True)
                D_bin = sq_norms + sq_norms.t() - 2 * pca_bin @ pca_bin.t()
                D_bin = torch.clamp(D_bin, min=0)

                # Gaussian kernel
                Ksub = torch.exp(-D_bin / (2.0 * gamma ** 2))  # (s, s)

                # ---------- LCE ----------
                e_sub = (probs_class[idx_in_bin] - labels_binary[idx_in_bin]).to(device)
                numer = Ksub.matmul(e_sub)
                denom = Ksub.sum(dim=1)

                denom_safe = denom.clone()
                zero_mask_lce = denom_safe <= eps
                denom_safe[zero_mask_lce] = 1.0

                lce_sub = numer / (denom_safe + eps)
                if zero_mask_lce.any():
                    lce_sub[zero_mask_lce] = 0.0

                # ---------- ESS (exclude self-contribution) ----------
                Ksupport = Ksub.clone()
                Ksupport.fill_diagonal_(0.0)

                row_sum = Ksupport.sum(dim=1, keepdim=True)  # (s, 1)
                row_sum_safe = row_sum.clone()
                zero_mask_ess = row_sum_safe.squeeze(1) <= eps
                row_sum_safe[row_sum_safe <= eps] = 1.0

                w_norm = Ksupport / (row_sum_safe + eps)
                ess_sub = 1.0 / (w_norm.pow(2).sum(dim=1) + eps)

                # no neighbor support after removing diagonal
                ess_sub[zero_mask_ess] = 0.0

                # write back
                lce_vals[idx_in_bin] = lce_sub
                ess_vals[idx_in_bin] = ess_sub
                valid_idx.append(idx_in_bin)

        # ECCE
        if len(bin_accs) > 0:
            bin_accs_t = torch.tensor(bin_accs, device=device, dtype=torch.float32)
            bin_confs_t = torch.tensor(bin_confs, device=device, dtype=torch.float32)
            bin_weights_t = torch.tensor(bin_weights, device=device, dtype=torch.float32)

            cum_pred = torch.cumsum(bin_weights_t * bin_confs_t, dim=0)
            cum_true = torch.cumsum(bin_weights_t * bin_accs_t, dim=0)
            ecce_val = torch.sum(torch.abs(cum_pred - cum_true)).item()
            ecce_val /= len(bin_accs)
        else:
            ecce_val = 0.0

        ecces.append(ecce_val)
        eces.append(ece)
        mces.append(mce)

        # ---------- Per-class LCE aggregation ----------
        if len(valid_idx) > 0:
            valid_idx = torch.cat(valid_idx)
            lce_abs = torch.abs(lce_vals[valid_idx])
            ess_valid = ess_vals[valid_idx]

            per_class_lce_avg.append(float(lce_abs.mean().item()))
            per_class_mlce.append(float(lce_abs.max().item()))

            # ---------- NEW: ESS-binned LCE profile ----------
            # Keep only samples with positive ESS
            positive_mask = ess_valid > 0
            ess_valid_pos = ess_valid[positive_mask]
            lce_abs_pos = lce_abs[positive_mask]
            
            if ess_valid_pos.numel() > 0:
                ess_np = ess_valid_pos.detach().cpu().numpy()                

                # Quantile ESS bins
                ess_edges_np = np.quantile(ess_np, np.linspace(0.0, 1.0, n_bins_ess + 1))
                ess_edges_np[0] = ess_np.min() - 1e-12
                ess_edges_np[-1] = ess_np.max() + 1e-12

                # Handle degenerate case where many ESS values are identical
                ess_edges_np = np.maximum.accumulate(ess_edges_np)

                ess_edges = torch.tensor(ess_edges_np, dtype=torch.float32, device=device)

                ess_bin_idx = torch.bucketize(ess_valid_pos, ess_edges, right=False) - 1
                ess_bin_idx = ess_bin_idx.clamp(0, n_bins_ess - 1)

                class_bin_lce = []
                class_bin_ess = []
                class_bin_count = []

                for eb in range(n_bins_ess):
                    idx_ess_bin = (ess_bin_idx == eb).nonzero(as_tuple=True)[0]
                    if idx_ess_bin.numel() == 0:
                        class_bin_lce.append(np.nan)
                        class_bin_ess.append(np.nan)
                        class_bin_count.append(0)
                    else:
                        class_bin_lce.append(float(lce_abs_pos[idx_ess_bin].mean().item()))
                        class_bin_ess.append(float(ess_valid_pos[idx_ess_bin].mean().item()))
                        class_bin_count.append(int(idx_ess_bin.numel()))
            else:
                class_bin_lce = [np.nan] * n_bins_ess
                class_bin_ess = [np.nan] * n_bins_ess
                class_bin_count = [0] * n_bins_ess

        else:
            per_class_lce_avg.append(0.0)
            per_class_mlce.append(0.0)

            class_bin_lce = [np.nan] * n_bins_ess
            class_bin_ess = [np.nan] * n_bins_ess
            class_bin_count = [0] * n_bins_ess

        per_class_ess_profiles.append({
            "avg_abs_lce_per_ess_bin": class_bin_lce,
            "avg_ess_per_bin": class_bin_ess,
            "count_per_bin": class_bin_count
        })

    # ---------- Final aggregation ----------
    if full_ece:
        avg_ece = [round(x, 4) for x in eces]
        avg_ecce = [round(x, 4) for x in ecces]
        lce_list = [round(x, 4) for x in per_class_lce_avg]
        mlce_list = [round(x, 4) for x in per_class_mlce]
    else:
        avg_ece = np.dot(np.array(eces).T, np.array(class_freqs))
        avg_ecce = np.dot(np.array(ecces).T, np.array(class_freqs))
        lce_list = None

    avg_mce = np.dot(np.array(mces).T, np.array(class_freqs))
    avg_lce = np.dot(np.array(per_class_lce_avg).T, np.array(class_freqs))
    avg_mlce = np.dot(np.array(per_class_mlce).T, np.array(class_freqs))

    # ---------- NEW: aggregate ESS-binned profile across classes ----------
    agg_bin_lce = []
    agg_bin_ess = []
    agg_bin_count = []

    for eb in range(n_bins_ess):
        vals_lce = []
        vals_ess = []
        vals_counts = []

        for c in range(n_classes):
            count_c = per_class_ess_profiles[c]["count_per_bin"][eb]
            lce_c = per_class_ess_profiles[c]["avg_abs_lce_per_ess_bin"][eb]
            ess_c = per_class_ess_profiles[c]["avg_ess_per_bin"][eb]

            if count_c > 0 and not np.isnan(lce_c):
                vals_lce.append(lce_c * count_c)
                vals_counts.append(count_c)

            if count_c > 0 and not np.isnan(ess_c):
                vals_ess.append(ess_c * count_c)

        total_bin_count = int(np.sum(vals_counts)) if len(vals_counts) > 0 else 0

        if total_bin_count > 0:
            agg_bin_lce.append(float(np.sum(vals_lce) / total_bin_count))
            agg_bin_ess.append(float(np.sum(vals_ess) / total_bin_count))
            agg_bin_count.append(total_bin_count)
        else:
            agg_bin_lce.append(np.nan)
            agg_bin_ess.append(np.nan)
            agg_bin_count.append(0)

    ess_lce_profile = {
        "avg_abs_lce_per_ess_bin": agg_bin_lce,
        "avg_ess_per_bin": agg_bin_ess,
        "count_per_bin": agg_bin_count,
        "per_class": per_class_ess_profiles
    }

    if full_ece:
        return avg_ecce, avg_ece, avg_mce, avg_brier, nll, lce_list, mlce_list, ess_lce_profile
    else:
        return avg_ecce, avg_ece, avg_mce, avg_brier, nll, avg_lce, avg_mlce, ess_lce_profile
    
def compute_multiclass_calibration_metrics(probs: torch.Tensor, y_true: torch.Tensor, n_bins: int = 15, class_freqs: list = None, full_ece: bool = False):
    """
    Computes ECE, MCE, and Brier Score for a multiclass classifier in a one-vs-all manner.
    Then averages the metrics across all classes.

    Parameters:
    - probs: Tensor of shape (n_samples, n_classes), predicted probabilities
    - y_true: Tensor of shape (n_samples,), true class labels
    - n_bins: int, number of bins for calibration metrics

    Returns:
    - avg_ece: averaged Expected Calibration Error over classes
    - avg_mce: averaged Maximum Calibration Error over classes
    - avg_brier: averaged Brier Score over classes
    """
    n_classes = probs.shape[1]

    ecces = []
    eces = []
    mces = []
    briers = []
    
    # Convert probabilities to log-probabilities
    log_probs = torch.log(probs + 1e-12)  # Add small value to avoid log(0)
    # Gather the log-probabilities corresponding to the true labels
    loss = F.nll_loss(log_probs, y_true, reduction='mean')    
    nll = loss.item()
    
    if isinstance(n_bins, torch.Tensor):
        n_bins = int(n_bins.item())
    else:
        n_bins = int(n_bins)

    for class_idx in range(n_classes):
        # One-vs-all labels
        labels_binary = (y_true == class_idx).float()
        probs_class = probs[:, class_idx]

        # Brier Score
        brier = torch.mean((probs_class - labels_binary) ** 2).item()

        # Bin predictions
        bin_edges = torch.linspace(
            0.0, 1.0, n_bins + 1,
            device=probs_class.device,
            dtype=probs_class.dtype
        )
        bin_indices = torch.bucketize(probs_class, bin_edges, right=True)
        bin_indices = bin_indices.clamp(1, n_bins)
        
        total_count = probs_class.numel()
        ece = 0.0
        mce = 0.0
        
        bin_accs = []
        bin_confs = []
        bin_weights = []

        for i in range(1, n_bins + 1):
            bin_mask = bin_indices == i
            if torch.any(bin_mask):
                bin_probs = probs_class[bin_mask]
                bin_labels = labels_binary[bin_mask]
                bin_accuracy = torch.mean(bin_labels).item()
                bin_confidence = torch.mean(bin_probs).item()
                bin_error = abs(bin_accuracy - bin_confidence)
                
                # Store for ECCE
                bin_accs.append(bin_accuracy)
                bin_confs.append(bin_confidence)
                bin_weights.append(bin_probs.numel() / total_count)
                
                # Standard ECE & MCE
                ece += bin_error * bin_probs.numel() / total_count
                mce = max(mce, bin_error)   
                             
        # --- ECCE computation (CDF difference) ---
        bin_accs = torch.tensor(bin_accs)
        bin_confs = torch.tensor(bin_confs)
        bin_weights = torch.tensor(bin_weights)

        if len(bin_accs) > 0:  # avoid empty case
            cum_pred = torch.cumsum(bin_weights * bin_confs, dim=0)
            cum_true = torch.cumsum(bin_weights * bin_accs, dim=0)
            ecce = torch.sum(torch.abs(cum_pred - cum_true)).item()
            ecce /= len(bin_accs)  # normalize to [0,1]
        else:
            ecce = 0.0

        #ecce = torch.sum(torch.abs(cum_pred - cum_true)).item()
        
        ecces.append(ecce)
        eces.append(ece)
        mces.append(mce)
        briers.append(brier)

    if full_ece:
        avg_ece = [round(x, 4) for x in eces]
        avg_ecce = [round(x, 4) for x in ecces]
    else:
        avg_ece = np.dot(np.array(eces).T, np.array(class_freqs)) #sum(eces) / len(eces)
        avg_ecce = np.dot(np.array(ecces).T, np.array(class_freqs)) #sum(ecces) / len(ecces)
        
    avg_mce = sum(mces) / len(mces)
    avg_brier = sum(briers) / len(briers)

    return avg_ecce, avg_ece, avg_mce, avg_brier, nll

def multiclass_calibration_plot(
    y_true, probs, n_bins=15, bin_strategy='uniform',
    save_path="calibration_plots", filename="multiclass_calibration.pdf"
):
    """
    Create clean, publication-quality calibration plots (bar style) for multiclass classification.
    """
    # CIFAR10_CLASSES = [
    #     "airplane", "automobile", "bird", "cat", "deer",
    #     "dog", "frog", "horse", "ship", "truck"
    # ]   
    # Styling
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2,
    })

    n_classes = probs.shape[1]
    if n_classes > 10:
        rng = np.random.default_rng(seed=None)
        class_indices = rng.choice(n_classes, size=10, replace=False)
        n_classes = 10
    else:
        class_indices = range(n_classes)

    n_cols = min(n_classes, 5)
    n_rows = (n_classes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows))
    axes = np.array(axes).reshape(-1)

    palette = sns.color_palette("tab10", n_classes)

    for idx, class_idx in enumerate(class_indices):
        ax = axes[idx]
        y_true_binary = (y_true == class_idx).int()
        y_prob_class = probs[:, class_idx]

        # Calibration curve
        strategy = 'quantile' if bin_strategy == 'quantile' else 'uniform'
        frac_pos, mean_pred = calibration_curve(
            y_true_binary, y_prob_class, n_bins=n_bins, strategy=strategy
        )

        # Bin edges & counts
        if strategy == 'quantile':
            bin_edges = np.quantile(y_prob_class, np.linspace(0, 1, n_bins + 1))
            bin_edges[0], bin_edges[-1] = 0.0, 1.0
        else:
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

        bin_counts, _ = np.histogram(y_prob_class, bins=bin_edges)
        bin_idx_for_text = np.digitize(mean_pred, bin_edges) - 1
        bin_idx_for_text = np.clip(bin_idx_for_text, 0, n_bins - 1)

        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.2)

        # Bars + calibration curve
        color = palette[idx % len(palette)]
        for j, (mp, fp) in enumerate(zip(mean_pred, frac_pos)):
            count = bin_counts[bin_idx_for_text[j]]
            ax.bar(mp, fp, width=0.06, color=color, alpha=0.4, edgecolor="none")
            ax.text(mp, fp + 0.03, f"{count}", ha="center", va="bottom", fontsize=9, color="black")

        ax.plot(mean_pred, frac_pos, "o-", color=color, markersize=5)

        # Axes formatting
        ax.set_title(f"Class {class_idx}", pad=10) #ax.set_title(CIFAR10_CLASSES[class_idx], pad=10, weight="bold") #
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Empirical frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")

    # Remove unused axes
    for j in range(n_classes, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, filename)
    plt.savefig(full_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Calibration plot saved to: {full_path}")
    
'''    
def multiclass_calibration_plot(y_true, probs, n_bins=15, bin_strategy='default', save_path="calibration_plots", filename="multiclass_calibration.png"):
    """
    Saves a grid of calibration plots, one per class, to a specified directory.

    Parameters:
    - y_true: array-like of shape (n_samples,), true class labels
    - probs: array-like of shape (n_samples, n_classes), predicted probabilities
    - n_bins: int, number of bins for calibration curve
    - save_path: str, directory to save the plot
    - filename: str, name of the output image file
    """
    n_classes = probs.shape[1]
    print('NUMBER OF CLASSES: ', n_classes)
    if n_classes > 10:
        print("Warning: More than 10 classes in dataset! Random sample 10 classes for plotting.")    
        rng = np.random.default_rng(seed=None)  # `None` means use OS entropy
        class_indices = rng.choice(n_classes, size=10, replace=False)
        n_classes = 10
    else:
        class_indices = range(n_classes)
        

    # Grid layout
    n_cols = min(n_classes, 5)
    n_rows = (n_classes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i in range(n_classes):
        ax = axes[i] #class_idx
        class_idx = class_indices[i]
        y_true_binary = (y_true == class_idx).int()
        y_prob_class = probs[:, class_idx]

        # Calibration curve
        strategy = 'quantile' if bin_strategy == 'quantile' else 'uniform'
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true_binary, y_prob_class, n_bins=n_bins, strategy=strategy
        )
        if bin_strategy == 'quantile':
            # Get actual quantile-based bin edges
            bin_edges = np.quantile(y_prob_class, np.linspace(0, 1, n_bins + 1))
            bin_edges[0], bin_edges[-1] = 0.0, 1.0
        else:
            # Bin sample counts
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        
        bin_counts, _ = np.histogram(y_prob_class, bins=bin_edges)            
        # Map predicted means to closest bin centers
        bin_idx_for_text = np.digitize(mean_predicted_value, bin_edges) - 1
        bin_idx_for_text = np.clip(bin_idx_for_text, 0, n_bins - 1)

        # Plot
        ax.plot([0, 1], [0, 1], "k:", label="Perfect calibration")
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"Class {class_idx}")

        for i, (j, k) in enumerate(zip(mean_predicted_value, fraction_of_positives)):
            count = bin_counts[bin_idx_for_text[i]]
            ax.bar(j, k, width=0.07, color="blue", alpha=0.3)
            ax.text(j, k + 0.03, f"{count}", ha='center', va='bottom', fontsize=8)

        ax.set_xlabel(f"Predicted probability of class {class_idx}")
        ax.set_ylabel("Empirical frequency")
        ax.set_title(f"Calibration Plot\nClass {class_idx}")
        ax.legend()
        ax.grid(True)

    # Hide unused axes
    for i in range(n_classes, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, filename)
    plt.savefig(full_path)
    plt.close(fig)
    print(f"Calibration plot saved to: {full_path}")
'''

def label_smoothing(one_hot_labels: torch.Tensor, smoothing: float) -> torch.Tensor:
    '''
    Applies label smoothing to one-hot encoded labels.

    Args:
        one_hot_labels (torch.Tensor): Tensor of shape (batch_size, num_classes)
        smoothing (float): Smoothing factor between 0 and 1

    Returns:
        torch.Tensor: Smoothed labels of same shape
    '''
    assert 0.0 <= smoothing < 1.0, "Smoothing must be in [0, 1)"
    num_classes = one_hot_labels.size(1)
    smooth_labels = one_hot_labels * (1.0 - smoothing) + smoothing / num_classes
    return smooth_labels


def random_label_smoothing(one_hot_labels, smoothing=0.1):
    """
    Applies random label smoothing to one-hot encoded labels using PyTorch.

    Parameters:
    - one_hot_labels: Tensor of shape (batch_size, num_classes)
    - smoothing: float, maximum amount of random noise to add

    Returns:
    - smoothed_labels: Tensor of shape (batch_size, num_classes)
    """
    # Generate uniform random noise in [0, smoothing)
    noise = torch.rand_like(one_hot_labels) * smoothing

    # Apply smoothing
    smoothed_labels = one_hot_labels * (1.0 - smoothing) + noise

    # Optional: Normalize so each row sums to ~1
    smoothed_labels = smoothed_labels / smoothed_labels.sum(dim=1, keepdim=True)

    return smoothed_labels


######### WHEN TRAINING OV-RIDE DEFUALT CHECKPOINT DICT WITH ACTUAL USED VALUES #########   
def fix_default_checkpoint(kwargs):
    if kwargs.pretrain:
        to_ret = {}
        for key in kwargs.checkpoint:
            print(key)
            if key ==  'epochs_bw':
                continue
            else:
                if key in kwargs:
                    to_ret[key] = kwargs[key]
                else:
                    if key in kwargs.models:
                        to_ret[key] = kwargs.models[key]
                    else:
                        if key in kwargs.dataset:
                            to_ret[key] = kwargs.dataset[key]
                        else:
                            raise ValueError(f'Key: {key} present in config.checkpoint not found in main, dataset and models config structure!')
    else:
        to_ret = kwargs.checkpoint
        for key in kwargs.checkpoint:
            if key ==  'epochs':
                to_ret['epochs'] = kwargs.models.epochs               
    return to_ret


def apply_transform(example):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    example['image'] = transform(example['image'])
    return example


def stratified_split(dataset, train_ratio=0.45, val_ratio=0.1, seed=42):
    labels = [example['label'] for example in dataset]
    indices = list(range(len(dataset)))

    # First split: train vs temp (val + eval_cal)
    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices, labels, stratify=labels, test_size=(1 - train_ratio), random_state=seed
    )

    # Second split: val vs eval_cal
    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    val_idx, eval_cal_idx = train_test_split(
        temp_idx, stratify=temp_labels, test_size=(1 - val_ratio_adjusted), random_state=seed
    )

    return train_idx, val_idx, eval_cal_idx


def print_class_distribution(name, labels_tensor):
    label_counts = Counter(labels_tensor.tolist())
    print(f"\n📊 Class distribution in {name}:")
    for cls in sorted(label_counts):
        print(f"  Class {cls}: {label_counts[cls]} samples")
        

def extract_method_label(folder_name: str) -> str:
    """
    Extract method label from folder name.

    Rules:
    - general case: method name is the first token before '_'
      example: 'temperature_xxx_yyy' -> 'temperature'
    - special case: if first token is 'competition', also include the next token
      example: 'competition_dirichlet_xxx_yyy' -> 'competition_dirichlet'
    """
    parts = folder_name.split('_')
    if len(parts) == 0:
        return folder_name

    if parts[0] == "competition":
        if len(parts) > 1:
            return f"{parts[0]}_{parts[1]}"
        return "competition"

    return parts[0]


def collect_ess_profiles(metrics_root: str):
    """
    Collect all ESS profile CSV files grouped by method label.

    Returns
    -------
    method_to_runs : dict
        {
            method_label: [
                DataFrame with columns [ess_bin, avg_abs_lce, avg_ess, count, run_file],
                ...
            ]
        }
    """
    method_to_runs = {}

    for entry in os.listdir(metrics_root):
        folder_path = os.path.join(metrics_root, entry)
        if not os.path.isdir(folder_path):
            continue

        method_label = extract_method_label(entry)

        # only aggregated ESS profile files, not per-class ones
        pattern = os.path.join(folder_path, "ess_profile_seed_*.csv")
        files = sorted(glob.glob(pattern))

        if len(files) == 0:
            continue

        run_dfs = []
        for f in files:
            try:
                df = pd.read_csv(f)
            except Exception as e:
                print(f"Skipping {f}: {e}")
                continue

            required_cols = {"ess_bin", "avg_abs_lce"}
            if not required_cols.issubset(df.columns):
                print(f"Skipping {f}: missing required columns {required_cols}")
                continue

            df = df.copy()
            df["run_file"] = os.path.basename(f)
            run_dfs.append(df)

        if len(run_dfs) > 0:
            method_to_runs[method_label] = run_dfs

    return method_to_runs


def aggregate_method_runs(method_to_runs):
    """
    Aggregate ESS profiles across runs for each method.

    Returns
    -------
    agg_dict : dict
        {
            method_label: DataFrame with columns:
                ess_bin, mean_lce, std_lce, sem_lce, n_runs
        }
    """
    agg_dict = {}
    method_map = {'competition_DC': 'DC', 'competition_PC': 'PC', 'competition_IR': 'IR', 
                  'competition_TS': 'TS', 'competition_SMS': 'SM', 'competition_PS': 'PS', 
                  'calibrate': 'LC', 'pre-train': 'NC', 'reference': 'KC', 'quantize': 'VQ'}

    for method, run_dfs in method_to_runs.items():
        aligned = []

        for run_idx, df in enumerate(run_dfs):
            tmp = df[["ess_bin", "avg_abs_lce"]].copy()
            tmp = tmp.rename(columns={"avg_abs_lce": f"run_{run_idx}"})
            aligned.append(tmp)

        merged = aligned[0]
        for df_next in aligned[1:]:
            merged = pd.merge(merged, df_next, on="ess_bin", how="outer")

        merged = merged.sort_values("ess_bin").reset_index(drop=True)

        run_cols = [c for c in merged.columns if c.startswith("run_")]
        values = merged[run_cols].to_numpy(dtype=float)

        mean_lce = np.nanmean(values, axis=1)
        std_lce = np.nanstd(values, axis=1, ddof=1) if values.shape[1] > 1 else np.zeros(values.shape[0])
        n_runs = np.sum(~np.isnan(values), axis=1)
        sem_lce = std_lce / np.sqrt(np.maximum(n_runs, 1))

        agg_df = pd.DataFrame({
            "ess_bin": merged["ess_bin"].to_numpy(),
            "mean_lce": mean_lce,
            "std_lce": std_lce,
            "sem_lce": sem_lce,
            "n_runs": n_runs
        })

        agg_dict[method_map[method]] = agg_df

    return agg_dict


def method_sort_key(method_name: str):
    """
    Sort methods so that:
    1. uncalibrated first
    2. competition_* methods next
    3. quantize last
    4. everything else in between alphabetically
    """
    lower = method_name.lower()

    if lower in {"uncalibrated", "baseline", "raw", "none"}:
        return (0, lower)
    if lower.startswith("competition_"):
        return (2, lower)
    if lower == "quantize":
        return (4, lower)
    return (3, lower)


def plot_ess_profiles(
    agg_dict,
    save_path=None,
    title="Average absolute LCE vs density bin",
    interval="std"
):
    """
    Plot mean avg_abs_lce across ESS bins for all methods with variability bands.

    Parameters
    ----------
    agg_dict : dict
        output of aggregate_method_runs
    save_path : str or None
        if provided, save figure there
    interval : str
        'std' for mean ± std
        'sem95' for mean ± 1.96*SEM
    """
    plt.figure(figsize=(10, 6))

    sorted_methods = sorted(agg_dict.keys(), key=method_sort_key)

    for method in sorted_methods:
        df = agg_dict[method]

        x = df["ess_bin"].to_numpy()
        y = df["mean_lce"].to_numpy()

        if interval == "sem95":
            band = 1.96 * df["sem_lce"].to_numpy()
        else:
            band = df["std_lce"].to_numpy()

        lower = y - band
        upper = y + band

        plt.plot(x, y, marker='o', linewidth=2, label=method)
        plt.fill_between(x, lower, upper, alpha=0.2)

    plt.xlabel("Density bin")
    plt.ylabel("Average absolute LCE")
    plt.title(title)
    plt.xticks(sorted(df["ess_bin"].unique()))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    metrics_root = "metrics"  # change if needed

    method_to_runs = collect_ess_profiles(metrics_root)
    agg_dict = aggregate_method_runs(method_to_runs)

    print("Methods found:")
    for method, runs in method_to_runs.items():
        print(f"  {method}: {len(runs)} runs")

    plot_ess_profiles(
        agg_dict,
        save_path=os.path.join(metrics_root, "ess_profile_comparison.png"),
        title="Average absolute LCE across density bins",
        interval="std"   # use "sem95" for 95% confidence band
    )