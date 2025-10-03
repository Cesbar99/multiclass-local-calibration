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
from tqdm import tqdm

def calibrate(kwargs, wandb_logger):
    
    seed = kwargs.seed
    total_epochs = kwargs.models.epochs    
    cuda_device = kwargs.cuda_device

    if kwargs.data == 'tissue':
        dataset = MedMnistData(kwargs, experiment=kwargs.exp_name)   
    elif kwargs.data == 'cifar10':
        dataset = Cifar10Data(kwargs, experiment=kwargs.exp_name)
    elif kwargs.data == 'cifar100':
        dataset = Cifar10Data(kwargs, experiment=kwargs.exp_name) 
           
    path = f"checkpoints/{kwargs.exp_name}/{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features/"
    os.makedirs(path, exist_ok=True) 
    os.makedirs(f"results/{kwargs.exp_name}/{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features", exist_ok=True)    
    
    if kwargs.use_optuna:  
        if kwargs.multi_obj:
            study = optuna.create_study(
                directions=["minimize", "minimize"],  
                sampler=NSGAIISampler(),
                study_name="multi_objective"
            )
            calls = [multi_obj_print_callback]
        else:  
            study = optuna.create_study(direction="minimize", study_name="standard")
            calls = [print_callback]
        study.optimize(
            lambda trial: objective(trial, kwargs, dataset.data_train_cal_loader, dataset.data_val_cal_loader, wandb_logger),
            n_trials=kwargs.n_trials,
            show_progress_bar=True,
            callbacks=calls
        )
        if kwargs.multi_obj:
            fig = optuna.visualization.plot_pareto_front(study)    
            appendix = kwargs.exp_name + '_' + kwargs.data + '_' + f'{kwargs.dataset.num_classes}_classes_' + f'{kwargs.dataset.num_features}_features'        
            fig.write_html(f"results/plots/{appendix}/pare_front.html")
            pareto_trials = study.best_trials  # List of Pareto-optimal trials
            for trial in pareto_trials:
                print(f"Trial {trial.number}: KL={trial.values[0]}, Constraint={trial.values[1]}")
            best = min(pareto_trials, key=lambda t: t.values[0] + t.values[1])  # Simple sum             
            for key, value in best.params.items():
                print(f"    {key}: {value}")
                kwargs.models[key] = value  
        else:
            # Print best result
            print("Best trial:")
            print(f"  Value: {study.best_trial.value}")
            for key, value in study.best_trial.params.items():
                print(f"    {key}: {value}")
                kwargs.models[key] = value
                
    if kwargs.calibrator_version == 'v2':
        pl_model = AuxTrainerV2(kwargs.models, num_classes=kwargs.dataset.num_classes, 
                                feature_dim=kwargs.dataset.feature_dim, similarity_dim=kwargs.similarity_dim)     
    
    raw_results_path_test_cal = "results/{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}.csv".format(
            kwargs.exp_name,
            kwargs.data,
            kwargs.dataset.num_classes,
            kwargs.dataset.num_features,
            seed,
            total_epochs           
        )
    raw_results_path_train_cal = "results/{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}.csv".format(
        kwargs.exp_name,
        kwargs.data,
        kwargs.dataset.num_classes,
        kwargs.dataset.num_features,
        seed,
        total_epochs,                       
    )
        
    print(F'BEGIN CALIBRATION FOR {total_epochs} EPOCHS WITH SEED {seed}!')        
    trainer = pl.Trainer(
            max_epochs=total_epochs,
            accelerator="cuda",
            devices=[cuda_device],
            logger=wandb_logger,
            check_val_every_n_epoch=1,
            #gradient_clip_val=5,
            deterministic=False)   
    start = time.time()
    trainer.fit(pl_model, dataset.data_train_cal_loader,
                    dataset.data_val_cal_loader)
    train_time = time.time() - start
    print(train_time)
    
    path_model = join(path, f"classifier_seed-{seed}_ep-{total_epochs}")
    torch.save(pl_model.model.state_dict(), path_model)

    raws = []
    pl_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl_model.to(device)

    with torch.no_grad():
        for batch in tqdm(dataset.data_train_cal_loader, desc="Extracting pca features"):
            batch = [b.to(device) for b in batch]                
            raw = pl_model.extract_pca(batch)
            raws.append(raw)

    print('pca shape: ', raws[1]['features'].shape)
    res = get_raw_res(raws, features=True, adabw=kwargs.models.adabw, reduced_dim=None)
    
    res.to_csv(raw_results_path_train_cal, index=False)
    
    raws = []
    pl_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pl_model.to(device)

    with torch.no_grad():
        for batch in tqdm(dataset.data_test_cal_loader, desc="Extracting pca features"):
            batch = [b.to(device) for b in batch]                
            raw = pl_model.extract_pca(batch)
            raws.append(raw)

    print('pca shape: ', raws[1]['features'].shape)
    res = get_raw_res(raws, features=True, adabw=kwargs.models.adabw, reduced_dim=None)
    res.to_csv(raw_results_path_test_cal, index=False)

    print("CALIBRATION OVER!")
    
    # Save hp to JSON file
    path = f"hyperparams/{kwargs.data}_{kwargs.checkpoint.num_classes}_classes_{kwargs.checkpoint.num_features}_features"
    os.makedirs(path, exist_ok=True)    
    path_hp = join(path, f"classifier_seed-{seed}_ep-{total_epochs}.json")        
    hp_dict = OmegaConf.to_container(kwargs.models, resolve=True)    
    with open(path_hp, "w") as f:
        json.dump(hp_dict, f, indent=4)

    print("\nSTART TESTING!")        
    test(kwargs)
        
    
    
    
