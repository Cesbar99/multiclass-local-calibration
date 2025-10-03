import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

class TemperatureScaler(nn.Module):
    def __init__(self, max_iter=50, lr=0.01):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  
        self.max_iter = max_iter
        self.lr = lr

    def forward(self, logits):
        return logits / self.temperature

    def predict_proba(self, logits):
        scaled_logits = self.forward(logits)
        return F.softmax(scaled_logits, dim=1)

    def fit(self, val_loader, device="cuda"):
        """
        Optimize temperature using validation set.
        """
        self.to(device)

        logits_list = []
        labels_list = []

        with torch.no_grad():
            for init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot in val_loader:
                init_logits = init_logits.to(device)
                y = y_one_hot.to(device).argmax(dim=1)
                logits_list.append(init_logits)
                labels_list.append(y)

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        # Optimize temperature
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=self.lr, max_iter=self.max_iter)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        print(f"Optimal temperature: {self.temperature.item():.4f}")
        return self

    def calibrated_predictions(self, batch, device="cuda"):
        """Return calibrated probabilities."""        
        with torch.no_grad():
            init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch
            init_logits = init_logits.to(device)                                              
            init_pca = init_pca.to(device)
            init_preds = init_preds.to(device)

            new_logits = init_logits / self.temperature
        
            out = {    
                "features": init_pca, 
                "logits": new_logits,        
                "preds": torch.argmax(new_logits, dim=-1).view(-1,1),        
                "true": torch.argmax(y_one_hot, dim=-1).view(-1,1)
            }
        return out        
    

class IsotonicCalibrator():
    """
    Multi-class isotonic regression calibration (one-vs-rest).
    Fits an isotonic regressor per class.
    """
    def __init__(self, out_dim: int):
        self.out_dim = out_dim
        self.models = [IsotonicRegression(out_of_bounds="clip") for _ in range(out_dim)]

    def fit(self, val_loader, device="cuda"):
        logits_list = []
        labels_list = []

        # collect validation data
        with torch.no_grad():
            for init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot in val_loader:
                init_logits = init_logits.to(device)
                y = y_one_hot.to(device).argmax(dim=1)
                logits_list.append(init_logits)
                labels_list.append(y)

        logits = torch.cat(logits_list).cpu().numpy()   
        labels = torch.cat(labels_list).cpu().numpy()   

        # convert to probabilities
        probs = F.softmax(torch.tensor(logits), dim=1).numpy()  

        # one-vs-rest isotonic regression per class
        for c in range(self.out_dim):
            y_binary = (labels == c).astype(float)
            self.models[c].fit(probs[:, c], y_binary)

        return self

    def predict_proba(self, logits, device="cuda"):
        logits = logits.to(device)
        probs = F.softmax(logits, dim=1).cpu().numpy()  
        
        calibrated = np.zeros_like(probs)
        for c, model in enumerate(self.models):
            calibrated[:, c] = model.predict(probs[:, c])

        # renormalize (per sample, so rows sum to 1)
        calibrated /= calibrated.sum(axis=1, keepdims=True)
        return torch.tensor(calibrated, device=device, dtype=torch.float32)

    def calibrated_predictions(self, batch, device="cuda"):
        """Return calibrated probabilities."""        
        with torch.no_grad():
            init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch
            init_logits = init_logits.to(device)                                              
            init_pca = init_pca.to(device)
            init_preds = init_preds.to(device)

            new_logits = self.predict_proba(init_logits)
        
            out = {    
                "features": init_pca, 
                "logits": new_logits,        
                "preds": torch.argmax(new_logits, dim=-1).view(-1,1),        
                "true": torch.argmax(y_one_hot, dim=-1).view(-1,1)
            }
        return out    
    

class PlattScaler():
    """
    Multi-class Platt scaling via one-vs-rest logistic regression.
    """
    def __init__(self, out_dim: int):
        self.out_dim = out_dim
        self.models = [LogisticRegression(solver="lbfgs") for _ in range(out_dim)]

    def fit(self, val_loader, device="cuda"):
        logits_list = []
        labels_list = []

        # collect validation data
        with torch.no_grad():
            for init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot in val_loader:
                init_logits = init_logits.to(device)
                y = y_one_hot.to(device).argmax(dim=1)
                logits_list.append(init_logits)
                labels_list.append(y)

        logits = torch.cat(logits_list).cpu().numpy()   # (N, C)
        labels = torch.cat(labels_list).cpu().numpy()   # (N,)

        # use softmax probs as features
        probs = F.softmax(torch.tensor(logits), dim=1).numpy()  # (N, C)

        # one-vs-rest logistic regression per class
        for c in range(self.out_dim):
            y_binary = (labels == c).astype(int)
            self.models[c].fit(probs, y_binary)

        return self

    def predict_proba(self, logits, device="cuda"):
        logits = logits.to(device)
        probs = F.softmax(logits, dim=1).cpu().numpy()  # (B, C)

        calibrated = np.zeros_like(probs)
        for c, model in enumerate(self.models):
            calibrated[:, c] = model.predict_proba(probs)[:, 1]
        
        calibrated /= calibrated.sum(axis=1, keepdims=True)
        return torch.tensor(calibrated, device=device, dtype=torch.float32)
    
    def calibrated_predictions(self, batch, device="cuda"):              
        with torch.no_grad():
            init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch
            init_logits = init_logits.to(device)                                              
            init_pca = init_pca.to(device)
            init_preds = init_preds.to(device)

            new_logits = self.predict_proba(init_logits)
        
            out = {    
                "features": init_pca, 
                "logits": new_logits,        
                "preds": torch.argmax(new_logits, dim=-1).view(-1,1),        
                "true": torch.argmax(y_one_hot, dim=-1).view(-1,1)
            }
        return out    
    

class DirichletCalibrator(nn.Module):
    def __init__(self, n_classes, lr=0.01, max_iter=100):
        super().__init__()
        self.n_classes = n_classes        
        self.feature_dim = 2 * n_classes + 1
        self.W = nn.Parameter(torch.zeros(self.n_classes, self.feature_dim))
        self.lr = lr
        self.max_iter = max_iter

    def _features(self, probs):
        log_probs = torch.log(probs + 1e-12)
        ones = torch.ones(probs.shape[0], 1, device=probs.device)
        return torch.cat([log_probs, probs, ones], dim=1)

    def forward(self, probs):
        features = self._features(probs)  # (N, 2C+1)
        logits = features @ self.W.t()    # (N, C)
        return F.softmax(logits, dim=1)

    def fit(self, val_loader, device="cuda"):
        self.to(device)
        probs_list, labels_list = [], []

        # Collect validation set outputs
        with torch.no_grad():
            for batch in val_loader:
                # Adjust depending on your dataloader format
                init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch
                init_logits = init_logits.to(device)
                probs = F.softmax(init_logits, dim=1)
                y = y_one_hot.to(device).argmax(dim=1)

                probs_list.append(probs)
                labels_list.append(y)

        probs = torch.cat(probs_list)     
        labels = torch.cat(labels_list)   

        optimizer = optim.LBFGS([self.W], lr=self.lr, max_iter=self.max_iter)
        nll_criterion = nn.NLLLoss()

        def closure():
            optimizer.zero_grad()
            q = self.forward(probs)
            loss = nll_criterion(torch.log(q + 1e-12), labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            q = self.forward(probs)
            loss = nll_criterion(torch.log(q + 1e-12), labels)
            print(f"Dirichlet calibration training done. Final NLL: {loss.item():.4f}")

        return self

    def calibrated_predictions(self, batch, device="cuda"):
        """Return calibrated probabilities for a batch."""
        with torch.no_grad():
            init_feats, init_logits, init_pca, y_one_hot, init_preds, init_preds_one_hot = batch
            init_logits = init_logits.to(device)
            probs = F.softmax(init_logits, dim=1)
            calibrated_probs = self.forward(probs)
            init_preds = init_preds.to(device)

            return {
                "features": init_pca.to(device),
                "logits": calibrated_probs,
                "preds": torch.argmax(calibrated_probs, dim=-1).view(-1, 1), 
                "true": torch.argmax(y_one_hot, dim=-1).view(-1, 1)
            }
