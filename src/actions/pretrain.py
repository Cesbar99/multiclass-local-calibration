import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
from tqdm import tqdm

def pretrain(kwargs, wandb_logger):
    
    seed = kwargs.seed
    #pl.seed_everything(seed, workers=True)  
    total_epochs = kwargs.models.epochs
    cuda_device = kwargs.cuda_device
    
    if kwargs.data == 'synthetic':
        dataset = SynthData(kwargs.dataset, experiment=kwargs.exp_name)
        pl_model = SynthTab(input_dim=kwargs.dataset.num_features,            
                            output_dim=kwargs.dataset.num_classes,
                            temperature=kwargs.models.temperature,
                            optimizer_cfg=kwargs.models.optimizer,
                            use_acc=kwargs.models.use_acc
                        )

    if kwargs.data == 'tissue':
        dataset = MedMnistData(kwargs.dataset, experiment=kwargs.exp_name, name=kwargs.data)
        pl_model = MedMnistModel(kwargs.models)    

    elif kwargs.data == 'cifar10':
        dataset = Cifar10Data(kwargs.dataset, experiment=kwargs.exp_name, name=kwargs.data)
        pl_model = Cifar10Model(kwargs.models)
        
    elif kwargs.data == 'cifar100':
        dataset = Cifar100Data(kwargs.dataset, experiment=kwargs.exp_name, name=kwargs.data)
        pl_model = Cifar100Model(kwargs.models)   

    path = f"checkpoints/{kwargs.exp_name}/{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features/"    
    os.makedirs(path, exist_ok=True) 
    os.makedirs(f"results/{kwargs.exp_name}/{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features", exist_ok=True)    
    path_model = "checkpoints/{}/{}_{}_classes_{}_features/classifier_seed-{}_ep-{}_tmp_{}.pt".format(
            kwargs.exp_name,
            kwargs.data,
            kwargs.dataset.num_classes,
            kwargs.dataset.num_features,
            seed,
            total_epochs,
            kwargs.models.temperature
        )
    raw_results_path_train_cal = "results/{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}_tmp_{}.csv".format(
            kwargs.exp_name,
            kwargs.data,
            kwargs.dataset.num_classes,
            kwargs.dataset.num_features,
            seed,
            total_epochs,
            kwargs.models.temperature            
        )
    raw_results_path_eval_cal = "results/{}/{}_{}_classes_{}_features/raw_results_eval_cal_seed-{}_ep-{}_tmp_{}.csv".format(
            kwargs.exp_name,
            kwargs.data,
            kwargs.dataset.num_classes,
            kwargs.dataset.num_features,
            seed,
            total_epochs,
            kwargs.models.temperature            
        )
        
    print(F'BEGIN PRE-TRAINING FOR {total_epochs} EPOCHS WITH SEED {seed} AND {kwargs.models.temperature} TEMPERATURE!')        
    trainer = pl.Trainer(
            max_epochs=total_epochs,
            accelerator="cuda",
            devices=[cuda_device],
            logger=wandb_logger,
            check_val_every_n_epoch=1,            
            deterministic=True,
            callbacks=[ ClearCacheCallback()]
         )
    start = time.time()
    trainer.fit(pl_model, dataset.data_train_loader,
                    dataset.data_val_loader)
    train_time = time.time() - start
    print(train_time)
    torch.save(pl_model.model.state_dict(), path_model)
    
    if kwargs.return_features:
        raws = []
        pl_model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pl_model.to(device)

        with torch.no_grad():
            for batch in tqdm(dataset.data_train_cal_loader, desc="Extracting features"):
                batch = [b.to(device) for b in batch]                
                raw = pl_model.extract_features(batch)
                raws.append(raw)

        print('features shape: ', raws[1]['features'].shape, raws[1]['preds'].shape, raws[1]['true'].shape)
        res = get_raw_res(raws, features=True, reduced_dim=kwargs.similarity_dim)
    else:
        raws = trainer.predict(pl_model, dataset.data_train_cal_loader) #dataset.data_train_cal_loader
        res = get_raw_res(raws)
    res.to_csv(raw_results_path_train_cal, index=False)
    
    if kwargs.return_features:
        raws = []
        pl_model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pl_model.to(device)

        with torch.no_grad():
            for batch in tqdm(dataset.data_eval_cal_loader, desc="Extracting features"):
                batch = [b.to(device) for b in batch]                
                raw = pl_model.extract_features(batch)
                raws.append(raw)

        print('features shape: ', raws[1]['features'].shape)
        res = get_raw_res(raws, features=True, reduced_dim=kwargs.similarity_dim)
    else:
        raws = trainer.predict(pl_model, dataset.data_eval_cal_loader) #dataset.data_eval_cal_loader
        res = get_raw_res(raws)
    res.to_csv(raw_results_path_eval_cal, index=False)
    
    print("PRE-TRAINING OVER!")
    print("START TESTING!")
    test(kwargs)
    
    
    
    
