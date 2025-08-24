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
from datasets.dataset import *
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from actions.test import test

def pretrain(kwargs, wandb_logger):
    
    seed = kwargs.seed
    pl.seed_everything(seed, workers=True)  
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
        
    elif kwargs.data == 'mnist':
        if kwargs.dataset.variant:
            kwargs.data = kwargs.data + '_' + kwargs.dataset.variant            
        dataset = MnistData(kwargs.dataset, experiment=kwargs.exp_name)
        pl_model = MnistModel(kwargs.models)
    
    elif kwargs.data == 'tissue':
        dataset = MedMnistData(kwargs.dataset, experiment=kwargs.exp_name, name=kwargs.data)
        pl_model = MedMnistModel(kwargs.models)    
        
    elif kwargs.data == 'path':
        dataset = MedMnistData(kwargs.dataset, experiment=kwargs.exp_name, name=kwargs.data)
        pl_model = MedMnistModel(kwargs.models)
        
    elif kwargs.data == 'cifar10':
        dataset = Cifar10Data()
        pl_model = Cifar10Model()
        
    elif kwargs.data == 'cifar10_ood':
        dataset = Cifar10OODData()
        pl_model = Cifar10OODModel()
        
    elif kwargs.data == 'cifar10_longtail':
        dataset = Cifar10LongTailData()
        pl_model = Cifar10LongTailModel()
        
    elif kwargs.data == 'cifar100':
        dataset = Cifar100Data()        
        pl_model = Cifar100Model()    
        
    elif kwargs.data == 'cifar100_longtail':
        dataset = Cifar100LongTailData()
        pl_model = Cifar100LongTailModel()
        
    elif kwargs.data == 'Imagenet':
        dataset = ImagenetData()
        pl_model = ImagenetModel()
        
    elif kwargs.data == 'imagenet_ood':
        dataset = ImagenetOODData()
        pl_model = ImagenetOODModel()
        
    elif kwargs.data == 'imagenet_longtail':
        dataset = ImagenetLongTailData()  
        pl_model = ImagenetLongTailModel()    
    
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
            callbacks=[
                 EarlyStopping(
                     monitor="val_loss",
                     patience=5,
                     mode="min",
                     verbose=True,
                     min_delta=0.0,
                 ),
                 ModelCheckpoint(
                    monitor="val_loss",                                                                                             # Metric to track
                    mode="min",                                                                                                     # Lower is better
                    save_top_k=1,                                                                                                   # Only keep the best model
                    filename=f"classifier_seed-{seed}_ep-{total_epochs}_tmp_{kwargs.models.temperature}.pt",                        # Static filename (no epoch suffix)
                    dirpath=path,                                                                                                   # Save in your existing checkpoint folder
                    save_weights_only=True,                                                                                         # Save only weights (not full LightningModule)
                    auto_insert_metric_name=False,                                                                                  # Prevent metric name in filename
                    every_n_epochs=1,                                                                                               # Run every epoch                    
                    enable_version_counter=False,
                    verbose=True
                ) ,
                ClearCacheCallback()  
            ]
         )
    start = time.time()
    trainer.fit(pl_model, dataset.data_train_loader,
                    dataset.data_val_loader)
    train_time = time.time() - start
    print(train_time)
    #torch.save(pl_model.model.state_dict(), path_model)
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(F'LOADING CHECKPOINT FILE {best_model_path}')
    checkpoint = torch.load(best_model_path)
    pl_model.load_state_dict(checkpoint['state_dict'])
    
    raws = trainer.predict(pl_model, dataset.data_train_cal_loader)
    res = get_raw_res(raws)
    res.to_csv(raw_results_path_train_cal, index=False)
    
    raws = trainer.predict(pl_model, dataset.data_eval_cal_loader)
    res = get_raw_res(raws)
    res.to_csv(raw_results_path_eval_cal, index=False)
    
    print("PRE-TRAINING OVER!")
    print("START TESTING!")
    test(kwargs)
    
    