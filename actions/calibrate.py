import torch 
import pytorch_lightning as pl
from models.networks import * 
from models.trainers import * 
import hydra
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict, OmegaConf
import time
import os
from utils.utils import get_raw_res
from calibrator.local_net import *
from calibrator.cal_trainer import *
from datasets.dataset import * 
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

def calibrate(kwargs, wandb_logger):
    
    seed = kwargs.seed
    total_epochs = kwargs.total_epochs    
    cuda_device = kwargs.cuda_device
    pl.seed_everything(seed, workers=True)  
    
    if kwargs.experiment == 'synthetic':
        dataset = SynthData(calibration=kwargs.calibration)
    elif kwargs.experiment == 'mnist':
        dataset = MnistData(calibration=kwargs.calibration)
    elif kwargs.experiment == 'cifar10':
        dataset = Cifar10Data(calibration=kwargs.calibration)
    elif kwargs.experiment == 'cifar10_ood':
        dataset = Cifar10OODData(calibration=kwargs.calibration)
    elif kwargs.experiment == 'cifar10_longtail':
        dataset = Cifar10LongTailData(calibration=kwargs.calibration)
    elif kwargs.experiment == 'cifar100':
        dataset = Cifar100Data(calibration=kwargs.calibration)    
    elif kwargs.experiment == 'cifar100_longtail':
        dataset = Cifar100LongTailData(calibration=kwargs.calibration)
    elif kwargs.experiment == 'Imagenet':
        dataset = ImagenetData(calibration=kwargs.calibration)
    elif kwargs.experiment == 'imagenet_ood':
        dataset = ImagenetOODData(calibration=kwargs.calibration)
    elif kwargs.experiment == 'imagenet_longtail':
        dataset = ImagenetLongTailData(calibration=kwargs.calibration)    
    
    os.makedirs(f"results/{kwargs.exp_name}/{kwargs.data}", exist_ok=True)
    path_model = "models/{}/{}/model_s_{}_seed-{}_ep-{}_tmp_{}.pt".format(
            kwargs.exp_name,
            kwargs.data,
            seed,
            total_epochs,
            kwargs.temperature
        )
    raw_results_path_test = "results/{}/{}/raw_results_test_seed-{}_ep-{}_tmp_{}.csv".format(
            kwargs.exp_name,
            kwargs.data,
            seed,
            total_epochs,
            kwargs.temperature            
        )
    raw_results_path_cal = "results/{}/{}/raw_results_cal_seed-{}_ep-{}_tmp_{}.csv".format(
            kwargs.exp_name,
            kwargs.data,
            seed,
            total_epochs,
            kwargs.temperature            
        )
    
    calibrator_model = AuxiliaryMLP()
    model = AuxTrainer(calibrator_model, dim=kwargs.dataset.num_classes, num_classes=kwargs.num_classes, lr=kwargs.lr, alpha1=kwargs.alpha1, alpha2=kwargs.alpha2,
                 lambda_kl=kwargs.lambda_kl, entropy_factor=kwargs.entropy_factor, noise=kwargs.noise, smoothing=kwargs.smoothing,
                 logits_scaling=kwargs.logits_scaling, sampling=kwargs.sampling, predict_labels=kwargs.predict_labels,
                 use_empirical_freqs=kwargs.use_empirical_freqs, js_distance=kwargs.js_distance, model_confident=kwargs.model_confident)    
    
    print(F'BEGIN CALIBRATION FOR {total_epochs} EPOCHS WITH SEED {seed}!')        
    trainer = pl.Trainer(
            max_epochs=total_epochs,
            accelerator="cuda",
            devices=[cuda_device],
            logger=wandb_logger,
            check_val_every_n_epoch=5,
            #gradient_clip_val=5,
            deterministic=True,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    mode="min",
                    verbose=False,
                    min_delta=0.0,
                )]
        )
    start = time.time()
    trainer.fit(model, dataset.data_train_loader,
                    dataset.data_val_loader)
    train_time = time.time() - start
    time_to_fit = train_time
    print(train_time)

    torch.save(model.model.state_dict(), path_model)
    
    raws = trainer.predict(model, dataset.data_test_loader)
    
    res = get_raw_res(raws)
    os.makedirs("results/{}".format(kwargs.data), exist_ok=True)
    res.to_csv(raw_results_path_test, index=False)
    
    print("CALIBRATION OVER!")
    
    
    