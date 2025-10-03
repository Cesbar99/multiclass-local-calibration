import torch
import torch.nn.functional as F
import os
from os.path import join
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

import pytorch_lightning as pl
import torch

import pytorch_lightning as pl
import torch

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

                # Add noise
                epsilon = torch.randn_like(init_logits)
                noisy_logits = init_logits + pl_module.noise * epsilon

                # Optional label smoothing        
                noisy_y_one_hot = label_smoothing(y_one_hot, pl_module.smoothing) if pl_module.smoothing else y_one_hot

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
                all_targets.append(torch.argmax(noisy_y_one_hot.cpu(), dim=1))
               
        all_probs = torch.cat(all_probs)
        all_targets = torch.cat(all_targets)

        # Call your function
        multiclass_calibration_plot(all_targets, all_probs, 
                                    save_path=self.save_path+f"calibrate_{self.data}_{self.num_classes}_classes_{self.num_features}_features/in_training/", 
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
       
                
def get_raw_res(raws, features=False, adabw=False, reduced_dim=None):
    
    preds = torch.cat([raws[j]["preds"].cpu() for j in range(len(raws))])
    logits = torch.cat([raws[j]["logits"].cpu() for j in range(len(raws))])
    feats = torch.cat([raws[j]["features"].cpu() for j in range(len(raws))]) if features else None
    true = torch.cat([raws[j]["true"].cpu() for j in range(len(raws))])
    
    raw_res = pd.DataFrame()
    raw_res["true"] = true.numpy().flatten()
    raw_res["preds"] = preds.numpy()

    logits_np = logits.cpu().numpy()
    logits_tmp = pd.DataFrame(
        logits_np, columns=[f"logits_{i}" for i in range(logits_np.shape[1])]
    )    
    if features:        
        feats_np = feats.cpu().numpy()
        feats_tmp = pd.DataFrame(
            feats_np, columns=[f"features_{i}" for i in range(feats_np.shape[1])]
        )        
        if reduced_dim is not None and reduced_dim > 0:
            print('RUNNING PCA ON FEATURES TO REDUCE DIM TO: ', reduced_dim)
            # Standardization            
            scaler = StandardScaler()
            feats_scaled = scaler.fit_transform(feats_tmp.values)

            # PCA
            pca = PCA(n_components=reduced_dim)
            feats_pca = pca.fit_transform(feats_scaled)

            feats_pca_tmp = pd.DataFrame(
                feats_pca, columns=[f"pca_{i}" for i in range(feats_pca.shape[1])]
            )
            feats_tmp = pd.concat([feats_tmp, feats_pca_tmp], axis=1)
                
    raw_res = pd.concat([raw_res, logits_tmp], axis=1)    
    if features:        
        raw_res = pd.concat([raw_res, feats_tmp], axis=1)    
    return raw_res

def create_logdir(name: str, resume_training: bool, wandb_logger):
    basepath = os.path.dirname(os.path.abspath(sys.argv[0]))
    basepath = os.path.join(os.path.dirname(os.path.dirname(basepath)), 'result')
    basepath = join(basepath, 'runs', name)
    run_name = wandb_logger.experiment.name
    logdir = join(basepath,run_name)
    if os.path.exists(logdir) and not resume_training:
        raise Exception(f'Run {run_name} already exists. Please delete the folder {logdir} or choose a different run name.')
    os.makedirs(logdir,exist_ok=True)
    return logdir

def compute_multiclass_calibration_metrics_w_lce(
    probs: torch.Tensor,
    y_true: torch.Tensor,
    pca: torch.Tensor,
    class_freqs: list,
    n_bins: int = 15,
    gamma: float = 0.1,
    full_ece: bool = False,
    bin_strategy: str = 'default'
):
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

    if gamma <= 0:
        raise ValueError("gamma must be > 0")    
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

                # squared distance matrix (n_b × n_b)
                sq_norms = (pca_bin ** 2).sum(dim=1, keepdim=True)
                D_bin = sq_norms + sq_norms.t() - 2 * pca_bin @ pca_bin.t()
                D_bin = torch.clamp(D_bin, min=0)

                # kernel matrix; K_sub: (s, s) kernel submatrix with points in this bin
                Ksub = torch.exp(-D_bin / (2.0 * gamma**2))  # (n_b, n_b)
                           
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
        avg_ece = np.dot(np.array(eces).T, np.array(class_freqs)) #sum(eces) / len(eces)
        avg_ecce = np.dot(np.array(ecces).T, np.array(class_freqs)) #sum(ecces) / len(ecces)
        lce_list = None

    avg_mce = np.dot(np.array(mces).T, np.array(class_freqs)) #sum(mces) / len(mces)
    avg_brier = np.dot(np.array(briers).T, np.array(class_freqs)) #sum(briers) / len(briers)

    # LCE aggregated across classes
    avg_lce = np.dot(np.array(per_class_lce_avg).T, np.array(class_freqs)) #sum(per_class_lce_avg) / len(per_class_lce_avg)
    avg_mlce = np.dot(np.array(per_class_mlce).T, np.array(class_freqs)) #sum(per_class_mlce) / len(per_class_mlce)

    # Return order:
    # avg_ecce, avg_ece, avg_mce, avg_brier, nll, avg_lce, avg_mlce
    if full_ece:
        # also return per-class LCE list when full_ece requested
        return avg_ecce, avg_ece, avg_mce, avg_brier, nll, lce_list, mlce_list
    else:
        return avg_ecce, avg_ece, avg_mce, avg_brier, nll, avg_lce, avg_mlce

def multiclass_calibration_plot(
    y_true, probs, n_bins=15, bin_strategy='uniform',
    save_path="calibration_plots", filename="multiclass_calibration.pdf"
):

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

def label_smoothing(one_hot_labels: torch.Tensor, smoothing: float) -> torch.Tensor:
    assert 0.0 <= smoothing < 1.0, "Smoothing must be in [0, 1)"
    num_classes = one_hot_labels.size(1)
    smooth_labels = one_hot_labels * (1.0 - smoothing) + smoothing / num_classes
    return smooth_labels


def random_label_smoothing(one_hot_labels, smoothing=0.1):
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

def print_class_distribution(name, labels_tensor):
    label_counts = Counter(labels_tensor.tolist())
    print(f"\n📊 Class distribution in {name}:")
    for cls in sorted(label_counts):
        print(f"  Class {cls}: {label_counts[cls]} samples")

