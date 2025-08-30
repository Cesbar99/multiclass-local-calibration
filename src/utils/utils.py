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

    def on_train_epoch_end(self, trainer, pl_module):
        # Only run every N epochs
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        pl_module.eval()
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch in self.dataloader:
                init_logits, y_one_hot, _, _ = batch
                init_logits = init_logits.to(self.device)
                y_one_hot = y_one_hot.to(self.device)

                # Add noise
                epsilon = torch.randn_like(init_logits)
                noisy_logits = init_logits + pl_module.noise * epsilon

                # Optional label smoothing        
                noisy_y_one_hot = label_smoothing(y_one_hot, pl_module.smoothing) if pl_module.smoothing else y_one_hot

                # Forward pass
                latents = pl_module(noisy_logits)
                means = latents[:, :pl_module.num_classes]
                log_std = latents[:, pl_module.num_classes:]
                stddev = F.softplus(log_std)

                # Reparameterization
                epsilon = torch.randn_like(means)
                z_hat = means + stddev * epsilon if pl_module.sampling else means

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
       
                
def get_raw_res(raws):
    preds = torch.cat([raws[j]["preds"].cpu() for j in range(len(raws))])
    #probs = torch.cat([raws[j]["probs"].cpu() for j in range(len(raws))])
    logits = torch.cat([raws[j]["logits"].cpu() for j in range(len(raws))])
    true = torch.cat([raws[j]["true"].cpu() for j in range(len(raws))])
    
    raw_res = pd.DataFrame()
    raw_res["true"] = true.numpy().flatten()
    raw_res["preds"] = preds.numpy()
    #raw_res["logits"] = logits.numpy()
    #raw_res["probs"] = probs.numpy()
    tmp = pd.DataFrame()

    for i in range(logits.shape[1]):
        #tmp["class_probs_{}".format(i)] = probs[:, i].cpu().numpy()
        tmp["logits_{}".format(i)] = logits[:, i].cpu().numpy()
        
    raw_res = pd.concat([raw_res, tmp], axis=1)    
    return raw_res


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


def compute_multiclass_calibration_metrics(probs: torch.Tensor, y_true: torch.Tensor, n_bins: int = 15, full_ece: bool = False):
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

    eces = []
    mces = []
    briers = []

    for class_idx in range(n_classes):
        # One-vs-all labels
        labels_binary = (y_true == class_idx).float()
        probs_class = probs[:, class_idx]

        # Brier Score
        brier = torch.mean((probs_class - labels_binary) ** 2).item()

        # Bin predictions
        bin_edges = torch.linspace(0, 1, n_bins + 1)
        bin_indices = torch.bucketize(probs_class, bin_edges, right=True)

        ece = 0.0
        mce = 0.0

        for i in range(1, n_bins + 1):
            bin_mask = bin_indices == i
            if torch.any(bin_mask):
                bin_probs = probs_class[bin_mask]
                bin_labels = labels_binary[bin_mask]
                bin_accuracy = torch.mean(bin_labels).item()
                bin_confidence = torch.mean(bin_probs).item()
                bin_error = abs(bin_accuracy - bin_confidence)
                ece += bin_error * bin_probs.numel() / probs_class.numel()
                mce = max(mce, bin_error)

        eces.append(ece)
        mces.append(mce)
        briers.append(brier)

    if full_ece:
        avg_ece = [round(ece, 4) for ece in eces]
    else:
        avg_ece = sum(eces) / len(eces)
    avg_mce = sum(mces) / len(mces)
    avg_brier = sum(briers) / len(briers)

    return avg_ece, avg_mce, avg_brier


def multiclass_calibration_plot(y_true, probs, n_bins=15, save_path="calibration_plots", filename="multiclass_calibration.png"):
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

    # Grid layout
    n_cols = min(n_classes, 5)
    n_rows = (n_classes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for class_idx in range(n_classes):
        ax = axes[class_idx]
        y_true_binary = (y_true == class_idx).int()
        y_prob_class = probs[:, class_idx]

        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true_binary, y_prob_class, n_bins=n_bins, strategy='uniform'
        )

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

