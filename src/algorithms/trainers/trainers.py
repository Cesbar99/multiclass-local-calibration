import torch
import torchmetrics
from algorithms.networks.networks import *
import pytorch_lightning as pl

class MedMnistModel(pl.LightningModule):
    def __init__(self, kwargs):
        super().__init__()        
        self.temperature = kwargs.temperature
        self.optimizer_cfg = kwargs.optimizer
        self.use_acc = kwargs.use_acc
        
        if kwargs.model == 'tissue_resnet50':            
            self.model = TissueMNISTResNet50(self.temperature)
            num_classes = 8
        else:
            raise ValueError(f"Unsupported {kwargs.model} architecture provided! Only 'tissue_resnet50' is available!"
        
        task = 'multiclass'        
        if self.use_acc:
            self.acc_train = torchmetrics.Accuracy(task=task, num_classes=num_classes) # CHANGE IF NEEDED
            self.acc_val = torchmetrics.Accuracy(task=task, num_classes=num_classes) # CHANGE IF NEEDED
        self.acc_test = torchmetrics.Accuracy(task=task, num_classes=num_classes) # CHANGE IF NEEDED

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        y = y.squeeze(1)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        if self.use_acc:            
            self.acc_train(logits, y)
            self.log('train_acc', self.acc_train, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        y = y.squeeze(1)
        logits = self(x)
        val_loss = F.cross_entropy(logits, y)
        self.log("val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)
        if self.use_acc:                          
            self.acc_val(logits, y)
            self.log('val_acc', self.acc_val, on_epoch=True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        opt_name = self.optimizer_cfg.name.lower()
        sched_name = self.optimizer_cfg.scheduler
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
        
        # Add scheduler here
        if isinstance(sched_name, str) and sched_name:
            scheduler_class = getattr(torch.optim.lr_scheduler, sched_name, None)
            if scheduler_class is None:
                raise ValueError(f"Trying to pass an argument to learning rate scheduler which is inalid! '{sched_name}' was given!")
        else:
            scheduler_class = None

        if scheduler_class is None: 
            return {
            "optimizer": optimizer}
        else:
            scheduler = scheduler_class(optimizer, step_size=self.optimizer_cfg.step_size, gamma=self.optimizer_cfg.gamma)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",  # Step every epoch
                    "frequency": 1        # Apply every epoch
                }
            }                                    
    
    def predict_step(self, batch):
        x, y = batch  
        logits = self(x)        
        preds = torch.argmax(logits, dim=-1).view(-1,1)
        target = y
        return {
            "preds": preds,            
            "true": target,
            "logits": logits,
        }
        
    def extract_features(self, batch):
        x, y = batch        
        feats = self.model.repr(x)  
        logits = self.model.resnet50.fc(feats)
        preds = torch.argmax(logits, dim=-1).view(-1,1)  
        
        out = {
            "features": feats,                  
            "logits": logits,
            "preds": preds,     
            "true": y
        }
        return out

class Cifar10Model(pl.LightningModule):
    def __init__(self, kwargs):
        super().__init__()        
        self.temperature = kwargs.temperature
        self.optimizer_cfg = kwargs.optimizer
        self.use_acc = kwargs.use_acc
        self.model = Cifar10ResNet50(self.temperature) #Cifar10Vit(self.temperature)            
        num_classes = 10
        
        task = 'multiclass'        
        if self.use_acc:
            self.acc_train = torchmetrics.Accuracy(task=task, num_classes=num_classes) # CHANGE IF NEEDED
            self.acc_val = torchmetrics.Accuracy(task=task, num_classes=num_classes) # CHANGE IF NEEDED
        self.acc_test = torchmetrics.Accuracy(task=task, num_classes=num_classes) # CHANGE IF NEEDED

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        if self.use_acc:            
            self.acc_train(logits, y)
            self.log('train_acc', self.acc_train, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        logits = self(x)
        val_loss = F.cross_entropy(logits, y)
        self.log("val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)
        if self.use_acc:                          
            self.acc_val(logits, y)
            self.log('val_acc', self.acc_val, on_epoch=True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        opt_name = self.optimizer_cfg.name.lower()
        sched_name = self.optimizer_cfg.scheduler
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
        #optimizer = optimizer_class(params, weight_decay=wd)
        
        # Add scheduler here
        if isinstance(sched_name, str) and sched_name:
            scheduler_class = getattr(torch.optim.lr_scheduler, sched_name, None)
            if scheduler_class is None:
                raise ValueError(f"Trying to pass an argument to learning rate scheduler which is inalid! '{sched_name}' was given!")
        else:
            scheduler_class = None

        if scheduler_class is None: 
            return {
            "optimizer": optimizer}
        else:
            scheduler = scheduler_class(optimizer, step_size=self.optimizer_cfg.step_size, gamma=self.optimizer_cfg.gamma)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",  # Step every epoch
                    "frequency": 1        # Apply every epoch
                }
            }                                    
    
    def predict_step(self, batch):
        x, y = batch  # assuming batch = (x, y, h)

        logits = self(x)        
        preds = torch.argmax(logits, dim=-1).view(-1,1)
        target = y
        return {
            "preds": preds,            
            "true": target,
            "logits": logits,
        }
        
    def extract_features(self, batch):
        x, y = batch        
        feats = self.model.repr(x)  # feature representation
        logits = self.model.resnet50.fc(feats)
        preds = torch.argmax(logits, dim=-1).view(-1,1) 
        out = {
            "features": feats,                  
            "logits": logits,
            "preds": preds,     
            "true": y
        }
        return out

class Cifar100Model(pl.LightningModule):
    def __init__(self, kwargs):
        super().__init__()     
        self.name = kwargs.model   
        self.temperature = kwargs.temperature
        self.optimizer_cfg = kwargs.optimizer
        self.use_acc = kwargs.use_acc
        self.model = Cifar100ResNet152(self.temperature)
        task = 'multiclass'        
        if self.use_acc:
            self.acc_train = torchmetrics.Accuracy(task=task, num_classes=num_classes) # CHANGE IF NEEDED
            self.acc_val = torchmetrics.Accuracy(task=task, num_classes=num_classes) # CHANGE IF NEEDED
        self.acc_test = torchmetrics.Accuracy(task=task, num_classes=num_classes) # CHANGE IF NEEDED

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        if self.use_acc:            
            self.acc_train(logits, y)
            self.log('train_acc', self.acc_train, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        logits = self(x)
        val_loss = F.cross_entropy(logits, y)
        self.log("val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)
        if self.use_acc:                          
            self.acc_val(logits, y)
            self.log('val_acc', self.acc_val, on_epoch=True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        opt_name = self.optimizer_cfg.name.lower()
        sched_name = self.optimizer_cfg.scheduler
        lr = self.optimizer_cfg.lr     
        wd = self.optimizer_cfg.get("weight_decay", 0.0)

        # Dynamically get the optimizer class
        if 'adam'in opt_name:
            optimizer_class = getattr(torch.optim, opt_name.capitalize(), None)
        else:
            optimizer_class = getattr(torch.optim, opt_name.upper(), None)
            
        if optimizer_class is None:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        # Build kwargs dynamically
        optimizer_kwargs = {"lr": lr, "weight_decay": wd}
        if opt_name == "sgd":
            optimizer_kwargs["momentum"] = self.optimizer_cfg.get("momentum", 0.9)

        optimizer = optimizer_class(self.parameters(), **optimizer_kwargs)
        #optimizer = optimizer_class(params, weight_decay=wd)
        
        # Add scheduler here
        if isinstance(sched_name, str) and sched_name:
            scheduler_class = getattr(torch.optim.lr_scheduler, sched_name, None)
            if scheduler_class is None:
                raise ValueError(f"Trying to pass an argument to learning rate scheduler which is inalid! '{sched_name}' was given!")
        else:
            scheduler_class = None

        if scheduler_class is None: 
            return {
            "optimizer": optimizer}
        else:
            scheduler = scheduler_class(optimizer, step_size=self.optimizer_cfg.step_size, gamma=self.optimizer_cfg.gamma)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",  # Step every epoch
                    "frequency": 1        # Apply every epoch
                }
            }                                    
    
    def predict_step(self, batch):
        x, y = batch  # assuming batch = (x, y, h)

        logits = self(x)        
        preds = torch.argmax(logits, dim=-1).view(-1,1)
        target = y
        return {
            "preds": preds,            
            "true": target,
            "logits": logits,
        }
        
    def extract_features(self, batch):
        x, y = batch        
        feats = self.model.repr(x)  # feature representation         
        logits = self.model.resnet152.fc(feats)
        preds = torch.argmax(logits, dim=-1).view(-1,1)  # predicted class
        out = {
            "features": feats,                  # replace logits with features
            "logits": logits,
            "preds": preds,     # dummy preds
            "true": y
        }
        return out
