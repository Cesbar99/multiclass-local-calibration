import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class TemperatureScaler(nn.Module):
    def __init__(self, max_iter=50, lr=0.01):
        super(TemperatureScaler, self).__init__()
        #self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # init T > 1
        self.max_iter = max_iter
        self.lr = lr

    def forward(self, logits):
        # Scale logits by temperature
        return logits / self.temperature

    def predict_proba(self, logits):
        scaled_logits = self.forward(logits)
        return F.softmax(scaled_logits, dim=1)

    def fit(self, val_loader, device="cuda"):
        """
        Optimize temperature using validation set.
        """
        self.to(device)
        #self.model.to(device)
        #self.model.eval()

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
        
            out = {    
                "features": init_pca, 
                "logits": init_logits / self.temperature,        
                "preds": init_preds,        
                "true": torch.argmax(y_one_hot, dim=-1).view(-1,1)
            }
        return out        

