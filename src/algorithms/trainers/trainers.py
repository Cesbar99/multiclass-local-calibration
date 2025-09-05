import torch
import torchmetrics
from algorithms.networks.networks import *
import pytorch_lightning as pl

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class SynthTab(pl.LightningModule):
    def __init__(self, input_dim, output_dim, temperature, optimizer_cfg, use_acc):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.temperature = temperature
        self.optimizer_cfg = optimizer_cfg
        self.use_acc = use_acc
                    
        self.model = DeepMLP(input_dim, output_dim, temperature)
        task = 'binary' if self.output_dim == 2 else 'multiclass'
        
        if self.use_acc:
            self.acc_train = torchmetrics.Accuracy(task=task, num_classes=self.output_dim)
            self.acc_val = torchmetrics.Accuracy(task=task, num_classes=self.output_dim)
        self.acc_test = torchmetrics.Accuracy(task=task, num_classes=self.output_dim)

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
        #preds = torch.argmax(logits, dim=1)
        #acc = (preds == y).float().mean()
        self.log("val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)
        if self.use_acc:            
            self.acc_val(logits, y)
            self.log('val_acc', self.acc_val, on_epoch=True, on_step=False, prog_bar=True)

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
        x, y = batch  # assuming batch = (x, y, h)

        logits = self(x)        
        preds = torch.argmax(logits, dim=-1).view(-1,1)
        target = y
        return {
            "preds": preds,            
            "true": target,
            "logits": logits,
        }

    
class MnistModel(pl.LightningModule):
    def __init__(self, kwargs):
        super().__init__()        
        self.temperature = kwargs.temperature
        self.optimizer_cfg = kwargs.optimizer
        self.use_acc = kwargs.use_acc
                    
        self.model = MnistArch(self.temperature)
        task = 'multiclass'
        
        if self.use_acc:
            self.acc_train = torchmetrics.Accuracy(task=task, num_classes=10)
            self.acc_val = torchmetrics.Accuracy(task=task, num_classes=10)
        self.acc_test = torchmetrics.Accuracy(task=task, num_classes=10)

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
        x, y = batch  # assuming batch = (x, y, h)

        logits = self(x)        
        preds = torch.argmax(logits, dim=-1).view(-1,1)
        target = y
        return {
            "preds": preds,            
            "true": target,
            "logits": logits,
        }


class MedMnistModel(pl.LightningModule):
    def __init__(self, kwargs):
        super().__init__()        
        self.temperature = kwargs.temperature
        self.optimizer_cfg = kwargs.optimizer
        self.use_acc = kwargs.use_acc
                    
        if kwargs.model == 'tissue_vit':
            self.model = TissueMnistVit(self.temperature)
            num_classes = 8
        elif kwargs.model == 'path_vit':
            self.model = PathMnistVit(self.temperature)
            num_classes = 9
        
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
        # params = [
        #     {"params": self.model.vit.fc.parameters(), "lr": self.optimizer_cfg.lr_fc},
        #     {"params": self.model.vit.layer4.parameters(), "lr": self.optimizer_cfg.lr_layer4}
        # ]        
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

        #return optimizer
    
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
        x, y = batch  # assuming batch = (x, y, h)

        logits = self(x)        
        preds = torch.argmax(logits, dim=-1).view(-1,1)
        target = y
        return {
            "preds": preds,            
            "true": target,
            "logits": logits,
        }


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
        #y = y.squeeze(1)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        if self.use_acc:            
            self.acc_train(logits, y)
            self.log('train_acc', self.acc_train, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        #y = y.squeeze(1)
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
        # params = [
        #     {"params": self.model.vit.fc.parameters(), "lr": self.optimizer_cfg.lr_fc},
        #     {"params": self.model.vit.layer4.parameters(), "lr": self.optimizer_cfg.lr_layer4}
        # ]        
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

        #return optimizer
    
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
        x, y = batch  # assuming batch = (x, y, h)

        logits = self(x)        
        preds = torch.argmax(logits, dim=-1).view(-1,1)
        target = y
        return {
            "preds": preds,            
            "true": target,
            "logits": logits,
        }


class Cifar10LongTailModel(pl.LightningModule):
    def __init__(self, kwargs):
        super().__init__()        
        self.temperature = kwargs.temperature
        self.optimizer_cfg = kwargs.optimizer
        self.use_acc = kwargs.use_acc
        self.model = Cifar10LTVit(self.temperature)            
        num_classes = 10 # CHANGE IF NEEDED
        
        task = 'multiclass'        
        if self.use_acc:
            self.acc_train = torchmetrics.Accuracy(task=task, num_classes=num_classes) # CHANGE IF NEEDED
            self.acc_val = torchmetrics.Accuracy(task=task, num_classes=num_classes) # CHANGE IF NEEDED
        self.acc_test = torchmetrics.Accuracy(task=task, num_classes=num_classes) # CHANGE IF NEEDED

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        #y = y.squeeze(1)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        if self.use_acc:            
            self.acc_train(logits, y)
            self.log('train_acc', self.acc_train, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        #y = y.squeeze(1)
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
        # params = [
        #     {"params": self.model.vit.fc.parameters(), "lr": self.optimizer_cfg.lr_fc},
        #     {"params": self.model.vit.layer4.parameters(), "lr": self.optimizer_cfg.lr_layer4}
        # ]        
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

        #return optimizer
    
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
        x, y = batch  # assuming batch = (x, y, h)

        logits = self(x)        
        preds = torch.argmax(logits, dim=-1).view(-1,1)
        target = y
        return {
            "preds": preds,            
            "true": target,
            "logits": logits,
        }
    

class CovTypeModel(pl.LightningModule):
    def __init__(self, kwargs, numerical_features, category_counts):
        super().__init__()          
        self.loss_name = kwargs.loss.name
        if self.loss_name == 'focal':
            self.loss_fn = FocalLoss(gamma=kwargs.loss.gamma)                    
        self.temperature = kwargs.temperature
        self.optimizer_cfg = kwargs.optimizer
        self.use_acc = kwargs.use_acc
        num_classes = 7 # CHANGE IF NEEDED
        if kwargs.model == 'covtype_resmlp':
            self.model = ResMLP(input_dim=12, hidden_dim=128, num_classes=num_classes, depth=kwargs.depth, dropout=kwargs.dropout) #
        else:
            self.model = CovType_FTT(numerical_features=numerical_features, category_counts=category_counts, temperature=self.temperature, num_labels=num_classes)                   
            
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
        if self.loss_name == 'focal':
            loss = self.loss_fn(logits, y)
        else:
            loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        if self.use_acc:            
            self.acc_train(logits, y)
            self.log('train_acc', self.acc_train, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, y = batch        
        logits = self(x)
        if self.loss_name == 'focal':
            val_loss = self.loss_fn(logits, y)
        else:
            val_loss = F.cross_entropy(logits, y)
        self.log("val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)
        if self.use_acc:                          
            self.acc_val(logits, y)
            self.log('val_acc', self.acc_val, on_epoch=True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        opt_name = self.optimizer_cfg.name.lower()
        sched_name = self.optimizer_cfg.scheduler
        lr = self.optimizer_cfg.lr
        # params = [
        #     {"params": self.model.vit.fc.parameters(), "lr": self.optimizer_cfg.lr_fc},
        #     {"params": self.model.vit.layer4.parameters(), "lr": self.optimizer_cfg.lr_layer4}
        # ]        
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

        #return optimizer
    
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
        x, y = batch  # assuming batch = (x, y, h)

        logits = self(x)        
        preds = torch.argmax(logits, dim=-1).view(-1,1)
        target = y
        return {
            "preds": preds,            
            "true": target,
            "logits": logits,
        }
            
            
class OttoModel(pl.LightningModule):
    def __init__(self, kwargs, numerical_features):
        super().__init__()        
        self.loss_name = kwargs.loss.name
        if self.loss_name == 'focal':
            self.loss_fn = FocalLoss(gamma=kwargs.loss.gamma)      
        self.temperature = kwargs.temperature
        self.optimizer_cfg = kwargs.optimizer
        self.use_acc = kwargs.use_acc
        num_classes = 9 # CHANGE IF NEEDED
        if kwargs.model == 'otto_resmlp':
            self.model = ResMLP(input_dim=93, hidden_dim=128, num_classes=num_classes, depth=kwargs.depth, dropout=kwargs.dropout) #
        else:
            self.model = Otto_FTT(numerical_features=numerical_features, temperature=self.temperature, num_labels=num_classes)                   
            
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
        if self.loss_name == 'focal':
            loss = self.loss_fn(logits, y)
        else:
            loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        if self.use_acc:            
            self.acc_train(logits, y)
            self.log('train_acc', self.acc_train, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, y = batch        
        logits = self(x)
        if self.loss_name == 'focal':
            val_loss = self.loss_fn(logits, y)
        else:
            val_loss = F.cross_entropy(logits, y)
        self.log("val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)
        if self.use_acc:                          
            self.acc_val(logits, y)
            self.log('val_acc', self.acc_val, on_epoch=True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        opt_name = self.optimizer_cfg.name.lower()
        sched_name = self.optimizer_cfg.scheduler
        lr = self.optimizer_cfg.lr
        # params = [
        #     {"params": self.model.vit.fc.parameters(), "lr": self.optimizer_cfg.lr_fc},
        #     {"params": self.model.vit.layer4.parameters(), "lr": self.optimizer_cfg.lr_layer4}
        # ]        
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

        #return optimizer
    
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
        x, y = batch  # assuming batch = (x, y, h)

        logits = self(x)        
        preds = torch.argmax(logits, dim=-1).view(-1,1)
        target = y
        return {
            "preds": preds,            
            "true": target,
            "logits": logits,
        }            
            
class Cifar10OODModel(nn.Module):
    def __init__():
        super().__init__()
        pass
    
class Cifar100Model(nn.Module):
    def __init__():
        super().__init__()
        pass
    
class Cifar100LongTailModel(nn.Module):
    def __init__():
        super().__init__()
        pass
    
class ImagenetModel(nn.Module):
    def __init__():
        super().__init__()
        pass
    
class ImagenetOODModel(nn.Module):
    def __init__():
        super().__init__()
        pass
    
class ImagenetLongTailModel(nn.Module):
    def __init__():
        super().__init__()
        pass


  