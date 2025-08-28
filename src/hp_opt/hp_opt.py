import optuna
import pytorch_lightning as pl
from calibrator.cal_trainer import *
from calibrator.local_net import *
from pytorch_lightning.loggers import CSVLogger


def objective(trial, kwargs, train_loader, val_loader, wandb_logger):
    cuda_device = kwargs.cuda_device
    total_epochs = kwargs.optuna_epochs
    
    csv_logger = CSVLogger(
        save_dir="optuna_logs",
        name=f"trial_{trial.number}"
    )

    # Suggest hyperparameters
    lambda_kl = trial.suggest_float("lambda_kl", 1.0, 10.0)
    alpha1 = trial.suggest_float("alpha1", 1.0, 10.0)
    log_var_initializer = trial.suggest_float("log_var_initializer", 0.001, 10.0)
    #smoothing = trial.suggest_float("smoothing", 0.001, 0.2)
    
    kwargs.models.lambda_kl = lambda_kl
    kwargs.models.alpha1 = alpha1
    kwargs.models.log_var_initializer = log_var_initializer
    #kwargs.models.smoothing = smoothing

    # Build your model with these hyperparameters
    model = AuxTrainer(kwargs.models, num_classes=kwargs.dataset.num_classes)    

    trainer = pl.Trainer(
            max_epochs=total_epochs,
            accelerator="cuda",
            devices=[cuda_device],
            logger=csv_logger,
            enable_progress_bar=False,
            enable_model_summary=False,
            deterministic=True
    )
    
    # Train and validate
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Use final validation loss as objective
    optuna_loss = trainer.callback_metrics["optuna_loss"].item()
    val_kl = trainer.callback_metrics["val_kl"].item()
    val_con_loss = trainer.callback_metrics["val_con_loss"].item()
    if kwargs.multi_obj:
        return val_kl, val_con_loss
    else:    
        return optuna_loss

def print_callback(study, trial):
    print(f"Trial {trial.number} finished with value: {trial.value}")
    print(f"  Params: {trial.params}\n")

def multi_obj_print_callback(study, trial):
    print(f"Trial {trial.number} finished with values: {trial.values}")
    print(f"  Params: {trial.params}\n")