from random import random
import optuna
import pytorch_lightning as pl
from calibrator.cal_trainer import *
from calibrator.local_net import *
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping
from calibrator.replicator import *


def objective(trial, kwargs, train_loader, val_loader, wandb_logger):
    cuda_device = kwargs.cuda_device
    total_epochs = kwargs.optuna_epochs
    
    csv_logger = CSVLogger(
        save_dir="optuna_logs",
        name=f"trial_{trial.number}"
    )

    # Suggest hyperparameters
    # lambda_kl = trial.suggest_float("lambda_kl", 1.0, 10.0)
    # alpha1 = trial.suggest_float("alpha1", 1.0, 10.0)
    # log_var_initializer = trial.suggest_float("log_var_initializer", 0.001, 10.0)
    alpha_sim = trial.suggest_float("alpha_sim", 0.05, 1.0)
    alpha_cls = trial.suggest_float("alpha_cls", 0.05, 1.0)
    #smoothing = trial.suggest_float("smoothing", 0.001, 0.2)
    
    # kwargs.models.lambda_kl = lambda_kl
    # kwargs.models.alpha1 = alpha1
    # kwargs.models.log_var_initializer = log_var_initializer
    kwargs.models.alpha_sim = alpha_sim
    kwargs.models.alpha_cls = alpha_cls
    #kwargs.models.smoothing = smoothing

    # Build your model with these hyperparameters
    model = AuxTrainerV2(kwargs.models, num_classes=kwargs.dataset.num_classes, feature_dim=kwargs.dataset.feature_dim, similarity_dim=kwargs.similarity_dim) #AuxTrainer(kwargs.models, num_classes=kwargs.dataset.num_classes)    

    trainer = pl.Trainer(
            max_epochs=total_epochs,
            accelerator="cuda",
            devices=[cuda_device],
            logger=csv_logger,
            enable_progress_bar=False,
            enable_model_summary=False,
            deterministic=True,
            callback=[EarlyStopping(monitor="val_total", #val_kl
                                      patience=5, #5
                                      mode="min", 
                                      verbose=True, 
                                      min_delta=0.0)]
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
    
    
def replicator_objective(trial, kwargs, train_loader, val_loader, test_loader, wandb_logger):
    cuda_device = kwargs.cuda_device
    total_epochs = kwargs.optuna_epochs   
    
    # Reproducibility
    seed = 1234 + trial.number  # Different seed for each trial
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

    # Suggest hyperparameters        
    lin_comb = trial.suggest_categorical("lin_comb", [0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99])
    ceiling = trial.suggest_categorical("ceiling", [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5])
    hidden_dim = trial.suggest_categorical("feature_dim", [8, 10, 16, 32, 64, 128])

    # kwargs.models.lin_comb = lin_comb
    # kwargs.models.ceiling = ceiling
    # kwargs.models.feature_dim = hidden_dim

    # Build your model with these hyperparameters
    calibrator = PotentialReplicatorCalibrator(                      
        n_classes =             kwargs.dataset.num_classes,
        data =                  kwargs.data,
        n_steps =               kwargs.models.n_steps,
        hidden =                hidden_dim,
        lin_comb =              lin_comb,
        ceiling =               ceiling,                     
        lr =                    kwargs.models.lr,
        weight_decay=           kwargs.models.weight_decay,
        epochs =                total_epochs,        
        eta =                   kwargs.models.step_size,        
        eps =                   1e-8,        
        kl_reg =                kwargs.models.kl_reg, 
        l2_reg =                kwargs.models.l2_reg,
        fit_stage =             kwargs.models.fit_stage,
        potential=              kwargs.models.potential
        )   
    
    # Train and validate
    calibrator.fit(train_loader, val_loader, test_loader, optuna_=True, trial=trial, device=cuda_device)

    # Use final validation loss as objective
    optuna_loss = calibrator.best_val_nll 
    
    return optuna_loss    


def print_callback(study, trial):
    print(f"Trial {trial.number} finished with value: {trial.value}")
    print(f"  Params: {trial.params}\n")


def multi_obj_print_callback(study, trial):
    print(f"Trial {trial.number} finished with values: {trial.values}")
    print(f"  Params: {trial.params}\n")
    
    