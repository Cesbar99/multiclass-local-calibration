from competitors.competitors import *
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from algorithms.networks.networks import *
from algorithms.trainers.trainers import *
import hydra
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict, OmegaConf
import time
from utils.utils import *
from data_sets.dataset import *
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from actions.test import test
from calibrator.cal_trainer import *
from calibrator.local_net import *
import optuna
from optuna.samplers import NSGAIISampler
from hp_opt.hp_opt import *
from pytorch_lightning.loggers import WandbLogger
import json
import csv 
import os
from tqdm import tqdm
from calibrator.replicator import *


def replicate(kwargs, wandb_logger=None):
    seed = kwargs.seed    
    cuda_device = kwargs.cuda_device
    pl.seed_everything(seed, workers=True)  
    
    if kwargs.data == 'synthetic':
        dataset = SynthData(kwargs, experiment=kwargs.exp_name)  
    elif kwargs.data == 'tissue':
        dataset = MedMnistData(kwargs, experiment=kwargs.exp_name)   
    elif kwargs.data == 'cifar10':
        dataset = Cifar10Data(kwargs, experiment=kwargs.exp_name)
    elif kwargs.data == 'cifar100':
        dataset = Cifar100Data(kwargs, experiment=kwargs.exp_name)  
    elif kwargs.data == 'cubic':
        dataset = CubicData(kwargs)
    
    if kwargs.data == 'cubic':
        data = kwargs.dataset.warp_type
    else:
        data = kwargs.data
    
    # calibrator = BoostReplicatorCalibrator(n_classes = kwargs.dataset.num_classes,
    #     lr =                kwargs.models.lr,
    #     max_iter =          kwargs.models.max_iter,
    #     n_steps =           kwargs.models.n_steps,
    #     step_size =         kwargs.models.step_size,
    #     eps =               1e-12,        
    #     kl_reg =            kwargs.models.kl_reg,        
    #     state_dependent =   kwargs.models.state_dependent,
    #     feature_dim =       kwargs.models.feature_dim,
    #     weight_decay =      kwargs.models.weight_decay)       
    #     # fitness_clip =      kwargs.models.fitness_clip) 
    
    # calibrator = StepwiseReplicatorCalibrator(                                          
    #     n_classes =             kwargs.dataset.num_classes,
    #     data =                  data,
    #     feature_dim =           kwargs.models.feature_dim,                       
    #     weak_lr =               kwargs.models.lr,
    #     weak_epochs =           kwargs.models.max_iter,
    #     n_steps =               kwargs.models.n_steps,
    #     eta =                   kwargs.models.step_size,
    #     alpha =                 kwargs.models.alpha,
    #     eps =                   1e-12,        
    #     kl_reg =                kwargs.models.kl_reg,        
    #     weight_decay =          kwargs.models.weight_decay,
    #     finetune_grad_clip =    kwargs.models.finetune_grad_clip,
    #     finetune_epochs =       kwargs.models.finetune_epochs
    #     )
        #state_dependent =   kwargs.models.state_dependent,
        
        # fitness_clip =      kwargs.models.fitness_clip) 
      
    # calibrator = DirCalibrator(                                             
    #     n_classes =             kwargs.dataset.num_classes,
    #     data =                  data,                            
    #     weak_lr =               kwargs.models.lr,
    #     weak_epochs =           kwargs.models.max_iter,        
    #     eps =                   1e-12,        
    #     kl_reg =                kwargs.models.kl_reg,        
    #     weight_decay =          kwargs.models.weight_decay        
    # )
        
    calibrator = PotentialReplicatorCalibrator(                      
        n_classes =             kwargs.dataset.num_classes,
        data =                  data,
        n_steps =               kwargs.models.n_steps,
        hidden =                kwargs.models.feature_dim,  
        lin_comb =              kwargs.models.lin_comb,
        ceiling =               kwargs.models.ceiling, 
        optimizer =             kwargs.models.optimizer,                    
        lr =                    kwargs.models.lr,
        weight_decay=           kwargs.models.weight_decay,
        epochs =                kwargs.models.max_iter,        
        eta =                   kwargs.models.step_size,        
        eps =                   1e-8,        
        kl_reg =                kwargs.models.kl_reg, 
        l2_reg =                kwargs.models.l2_reg,
        fit_stage =             kwargs.models.fit_stage,
        potential=              kwargs.models.potential
        )        
        
                
    ############## OPTUNA TIME ##############    
    if kwargs.use_optuna:     
        csv_path = f"optuna_logs/optuna_best_configs_new_{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features.csv"
        
        if seed == 42: # only run optuna for one seed to save time
            print(f'STARTING OPTUNA HYPERPARAMETER SEARCH FOR {kwargs.n_trials} TRIALS OF {kwargs.optuna_epochs} EPOCHS FORREPLICATOR CALIBRATOR...\n')       
            
            # search_space = {
            # "lin_comb": [0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99],
            # "ceiling": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5],
            # "feature_dim": [8, 10, 16, 32, 64, 128]
            # }   
            # sampler = optuna.samplers.GridSampler(search_space)
            study = optuna.create_study(direction="minimize",study_name="hyper_params_4_replicator", 
                                        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5))
            calls = [print_callback]
            study.optimize(
                lambda trial: replicator_objective(trial, kwargs, dataset.data_train_cal_loader, 
                                                dataset.data_val_cal_loader, dataset.data_test_cal_loader, wandb_logger),
                n_trials=kwargs.n_trials,
                show_progress_bar=True,
                callbacks=calls
            )
            
            # Print best result
            print("Best trial:")
            print(f"  Value: {study.best_trial.value}")
            for key, value in study.best_trial.params.items():
                print(f"    {key}: {value}")
                kwargs.models[key] = value   
                
            # ---- SAVE BEST CONFIG TO CSV ----            

            file_exists = os.path.isfile(csv_path)

            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "study_name",                        
                        "value",
                        "lin_comb",
                        "ceiling",
                        "feature_dim",
                        "n_steps",
                        "step_size",
                        "optuna_epochs",
                        "max_iter",
                        "optimizer",
                        "lr",
                        "weight_decay",
                        "kl_reg",
                        "l2_reg"                        
                    ]
                )

                # write header only once
                if not file_exists:
                    writer.writeheader()

                writer.writerow({
                    "study_name": study.study_name,                                                            
                    "value": study.best_trial.value,
                    "lin_comb": study.best_trial.params["lin_comb"], # kwargs.models.lin_comb, 
                    "ceiling": study.best_trial.params["ceiling"],
                    "feature_dim": study.best_trial.params["feature_dim"],
                    "n_steps": study.best_trial.params["n_steps"], # kwargs.models.n_steps, 
                    "step_size": study.best_trial.params["step_size"],
                    "optuna_epochs": kwargs.optuna_epochs,
                    "max_iter": kwargs.models.max_iter,
                    "optimizer": study.best_trial.params["optimizer"], # kwargs.models.optimizer,
                    "lr": study.best_trial.params["lr"],
                    "weight_decay": kwargs.models.weight_decay,
                    "kl_reg": kwargs.models.kl_reg,
                    "l2_reg": kwargs.models.l2_reg                    
                })

            print(f"Saved best hyperparameters to {csv_path}")  
                    
        else:
            print("Loading hyperparameters from CSV...")
            load_optuna_config(csv_path, kwargs)   
            
            
    ############## DEFINE PATH ##############         
    name = kwargs.exp_name
    name += f'{kwargs.models.n_steps}'
    if kwargs.models.kl_reg > 0:
        name += '_KL'
    if kwargs.models.state_dependent:
        name += '_DEP'
        
    kwargs.method = 'RC' #REPLICATOR CALIBRATION

    path = f"checkpoints/{name}/{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features/"    
    os.makedirs(path, exist_ok=True) 
    
    result_path = f"results/{name}/{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features"    
    os.makedirs(result_path, exist_ok=True)    
    
    raw_results_path_test_cal = "results/{}/{}_{}_classes_{}_features/raw_results_test_replicate_seed-{}_ep-{}.csv".format(
            name, 
            kwargs.data,
            kwargs.dataset.num_classes,
            kwargs.dataset.num_features,
            seed,
            kwargs.models.max_iter           
        )
    raw_results_path_train_cal = "results/{}/{}_{}_classes_{}_features/raw_results_train_replicate_seed-{}_ep-{}.csv".format(
        name, 
        kwargs.data,
        kwargs.dataset.num_classes,
        kwargs.dataset.num_features,
        seed,
        kwargs.models.max_iter,                       
    )                 
    
    
    ################ FITTING TIME ##############                
    calibrator = PotentialReplicatorCalibrator(                      
        n_classes =             kwargs.dataset.num_classes,
        data =                  data,
        n_steps =               kwargs.models.n_steps,
        hidden =                kwargs.models.feature_dim,  
        lin_comb =              kwargs.models.lin_comb,
        ceiling =               kwargs.models.ceiling,                     
        optimizer =             kwargs.models.optimizer,
        lr =                    kwargs.models.lr,
        weight_decay=           kwargs.models.weight_decay,
        epochs =                kwargs.models.max_iter,        
        eta =                   kwargs.models.step_size,        
        eps =                   1e-8,        
        kl_reg =                kwargs.models.kl_reg, 
        l2_reg =                kwargs.models.l2_reg,
        fit_stage =             kwargs.models.fit_stage,
        potential=              kwargs.models.potential
        )                   
                
    # Fit on validation set
    if kwargs.models.fit_stage:
        print('Boosting')
        calibrator.fit_stagewise(dataset.data_train_cal_loader, dataset.data_test_cal_loader, device=cuda_device)
    else:
        if calibrator.name == 'REPLICATOR':
            print('Fitting Replicator Calibrator')        
            calibrator.fit(dataset.data_train_cal_loader, dataset.data_val_cal_loader, dataset.data_test_cal_loader, device=cuda_device)
        elif calibrator.name == 'DIRICHLET':
            print('Fitting Dirichlet Calibrator')
            #calibrator.fit(dataset.data_train_cal_loader, dataset.data_test_cal_loader, device=cuda_device)
            calibrator.fit(dataset.data_train_cal_loader, device=cuda_device)

    raws = []
    calibrator.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calibrator.to(device)

    with torch.no_grad():
        for batch in tqdm(dataset.data_test_cal_loader, desc="Extracting Replicator Calibration logits"):
            batch = [b.to(device) for b in batch]                
            raw = calibrator.calibrated_predictions(batch)
            raws.append(raw)
            
    res, pca = get_raw_res(raws, features=True, reduced_dim=None)
    res.to_csv(raw_results_path_test_cal, index=False)
    
    
    print(f"\nSTART TESTING {kwargs.method}!")        
    test(kwargs)
    
