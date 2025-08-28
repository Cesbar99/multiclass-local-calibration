import optuna
import pytorch_lightning as pl
from calibrator.cal_trainer import *
from calibrator.local_net import *
from pytorch_lightning.loggers import CSVLogger


def objective(trial, kwargs, train_loader, val_loader, wandb_logger):
    cuda_device = kwargs.cuda_device
    total_epochs = kwargs.models.epochs   
    
    # Suggest hyperparameters
    lambda_kl = trial.suggest_float("lambda_kl", 0.0, 1.0)
    alpha1 = trial.suggest_float("alpha1", 0.0, 1.0)
    log_var_init = trial.suggest_float("log_var_init", 0.0, 10.0)
    
    kwargs.models.lambda_kl = lambda_kl
    kwargs.models.alpha1 = alpha1
    kwargs.models.log_var_init = log_var_init

    # Build your model with these hyperparameters
    model = AuxTrainer(kwargs, num_classes=kwargs.dataset.checkpoint.num_classes)    

    trainer = pl.Trainer(
            max_epochs=total_epochs,
            accelerator="cuda",
            devices=[cuda_device],
            logger=wandb_logger,
            enable_progress_bar=False,
            enable_model_summary=False,
            determinisitc=True
    )
    
    # Train and validate
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Use final validation loss as objective
    val_loss = trainer.callback_metrics["val_total"].item()
    return val_loss
