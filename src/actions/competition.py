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
from tqdm import tqdm


def competition(kwargs, wandb_logger=None):
    seed = kwargs.seed    
    cuda_device = kwargs.cuda_device
    pl.seed_everything(seed, workers=True)  
    
    elif kwargs.data == 'tissue':
        dataset = MedMnistData(kwargs, experiment=kwargs.exp_name)   
    elif kwargs.data == 'cifar10':
        dataset = Cifar10Data(kwargs, experiment=kwargs.exp_name)
    elif kwargs.data == 'cifar100':
        dataset = Cifar100Data(kwargs, experiment=kwargs.exp_name)  

    if 'DC' in kwargs.methods:
        kwargs.method = 'DC' #DIRICHLET CALIBRATION
        #num_classes}_classes_{kwargs.dataset.num_features}_features/"
        #os.makedirs(path, exist_ok=True) 
        os.makedirs(f"results/{kwargs.exp_name}_{kwargs.method}/{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features", exist_ok=True)    
        raw_results_path_test_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}.csv".format(
                    kwargs.exp_name,
                    kwargs.method,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    seed,
                    kwargs.models.max_iter           
                )
        raw_results_path_train_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}.csv".format(
                kwargs.exp_name,
                kwargs.method,
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                seed,
                kwargs.models.max_iter                      
            )
        # Assume you already trained `model`
        scaler = DirichletCalibrator(n_classes=kwargs.dataset.num_classes, max_iter=kwargs.models.max_iter, lr=kwargs.models.temp_lr)
        # Fit on validation set
        scaler.fit(dataset.data_train_cal_loader, device=cuda_device)
        
        raws = []
        scaler.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scaler.to(device)

        with torch.no_grad():
            for batch in tqdm(dataset.data_test_cal_loader, desc="Extracting Dirichlet Calibration logits"):
                batch = [b.to(device) for b in batch]                
                raw = scaler.calibrated_predictions(batch)
                raws.append(raw)
                
        res = get_raw_res(raws, features=True, reduced_dim=None)
        res.to_csv(raw_results_path_test_cal, index=False)
        
        raws = []
        scaler.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scaler.to(device)

        with torch.no_grad():
            for batch in tqdm(dataset.data_train_cal_loader, desc="Extracting Dirichlet Calibration logits"):
                batch = [b.to(device) for b in batch]                
                raw = scaler.calibrated_predictions(batch)
                raws.append(raw)
                
        res = get_raw_res(raws, features=True, reduced_dim=None)
        res.to_csv(raw_results_path_train_cal, index=False)
        
        print(f"\nSTART TESTING {kwargs.method}!")        
        test(kwargs)
    
    if 'TS' in kwargs.methods:
        kwargs.method = 'TS' #TEMPERATURE SCALING
        #num_classes}_classes_{kwargs.dataset.num_features}_features/"
        #os.makedirs(path, exist_ok=True) 
        os.makedirs(f"results/{kwargs.exp_name}_{kwargs.method}/{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features", exist_ok=True)    
        raw_results_path_test_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}.csv".format(
                    kwargs.exp_name,
                    kwargs.method,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    seed,
                    kwargs.models.max_iter           
                )
        raw_results_path_train_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}.csv".format(
                kwargs.exp_name,
                kwargs.method,
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                seed,
                kwargs.models.max_iter                      
            )
        # Assume you already trained `model`
        scaler = TemperatureScaler(kwargs.models.max_iter, kwargs.models.temp_lr)
        # Fit on validation set
        scaler.fit(dataset.data_train_cal_loader, device=cuda_device)
        
        raws = []
        scaler.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scaler.to(device)

        with torch.no_grad():
            for batch in tqdm(dataset.data_test_cal_loader, desc="Extracting Temeperature scaling logits"):
                batch = [b.to(device) for b in batch]                
                raw = scaler.calibrated_predictions(batch)
                raws.append(raw)
                
        res = get_raw_res(raws, features=True, reduced_dim=None)
        res.to_csv(raw_results_path_test_cal, index=False)
        
        raws = []
        scaler.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scaler.to(device)

        with torch.no_grad():
            for batch in tqdm(dataset.data_train_cal_loader, desc="Extracting Temeperature Scaling logits"):
                batch = [b.to(device) for b in batch]                
                raw = scaler.calibrated_predictions(batch)
                raws.append(raw)
                
        res = get_raw_res(raws, features=True, reduced_dim=None)
        res.to_csv(raw_results_path_train_cal, index=False)
        
        print(f"\nSTART TESTING {kwargs.method}!")        
        test(kwargs)

    if 'IR' in kwargs.methods:    
        kwargs.method = 'IR' #ISOTONIC REGRESSION
        #num_classes}_classes_{kwargs.dataset.num_features}_features/"
        #os.makedirs(path, exist_ok=True) 
        os.makedirs(f"results/{kwargs.exp_name}_{kwargs.method}/{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features", exist_ok=True)    
        raw_results_path_test_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}.csv".format(
                    kwargs.exp_name,
                    kwargs.method,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    seed,
                    kwargs.models.max_iter           
                )
        raw_results_path_train_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}.csv".format(
                kwargs.exp_name,
                kwargs.method,
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                seed,
                kwargs.models.max_iter                      
            )
        # Assume you already trained `model`
        scaler = IsotonicCalibrator(out_dim=kwargs.dataset.num_classes)
        # Fit on validation set
        scaler.fit(dataset.data_train_cal_loader, device=cuda_device)
        
        raws = []        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        

        with torch.no_grad():
            for batch in tqdm(dataset.data_test_cal_loader, desc="Extracting Isotonic Regression logits"):
                batch = [b.to(device) for b in batch]                
                raw = scaler.calibrated_predictions(batch)
                raws.append(raw)
                
        res = get_raw_res(raws, features=True, reduced_dim=None)
        res.to_csv(raw_results_path_test_cal, index=False)
        
        raws = []        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        

        with torch.no_grad():
            for batch in tqdm(dataset.data_train_cal_loader, desc="Extracting Isotonic Regression logits"):
                batch = [b.to(device) for b in batch]                
                raw = scaler.calibrated_predictions(batch)
                raws.append(raw)
                
        res = get_raw_res(raws, features=True, reduced_dim=None)
        res.to_csv(raw_results_path_train_cal, index=False)
        
        print(f"\nSTART TESTING {kwargs.method}!")        
        test(kwargs)

    if 'PS' in kwargs.methods:    
        kwargs.method = 'PS' #PLATT SCALING
        #num_classes}_classes_{kwargs.dataset.num_features}_features/"
        #os.makedirs(path, exist_ok=True) 
        os.makedirs(f"results/{kwargs.exp_name}_{kwargs.method}/{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features", exist_ok=True)    
        raw_results_path_test_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}.csv".format(
                    kwargs.exp_name,
                    kwargs.method,
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    seed,
                    kwargs.models.max_iter           
                )
        raw_results_path_train_cal = "results/{}_{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}.csv".format(
                kwargs.exp_name,
                kwargs.method,
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                seed,
                kwargs.models.max_iter                      
            )
        # Assume you already trained `model`
        scaler = PlattScaler(out_dim=kwargs.dataset.num_classes)
        # Fit on validation set
        scaler.fit(dataset.data_train_cal_loader, device=cuda_device)
        
        raws = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        

        with torch.no_grad():
            for batch in tqdm(dataset.data_test_cal_loader, desc="Extracting Platt scaling logits"):
                batch = [b.to(device) for b in batch]                
                raw = scaler.calibrated_predictions(batch)
                raws.append(raw)
                
        res = get_raw_res(raws, features=True, reduced_dim=None)
        res.to_csv(raw_results_path_test_cal, index=False)
        
        raws = []        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        

        with torch.no_grad():
            for batch in tqdm(dataset.data_train_cal_loader, desc="Extracting Platt Scaling logits"):
                batch = [b.to(device) for b in batch]                
                raw = scaler.calibrated_predictions(batch)
                raws.append(raw)
                
        res = get_raw_res(raws, features=True, reduced_dim=None)
        res.to_csv(raw_results_path_train_cal, index=False)
        
        print(f"\nSTART TESTING {kwargs.method}!")        
        test(kwargs)
    

    
    
    
    
    
