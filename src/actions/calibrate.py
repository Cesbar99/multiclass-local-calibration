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
from calibrator.cal_trainer import *
from calibrator.local_net import *

def calibrate(kwargs, wandb_logger):
    
    seed = kwargs.seed
    total_epochs = kwargs.models.epochs    
    cuda_device = kwargs.cuda_device
    pl.seed_everything(seed, workers=True)  
    
    if kwargs.data == 'synthetic':
        dataset = SynthData(kwargs, experiment=kwargs.exp_name)
    elif kwargs.data == 'mnist':
        dataset = MnistData(calibration=kwargs.calibration)
    elif kwargs.data == 'cifar10':
        dataset = Cifar10Data(calibration=kwargs.calibration)
    elif kwargs.data == 'cifar10_ood':
        dataset = Cifar10OODData(calibration=kwargs.calibration)
    elif kwargs.data == 'cifar10_longtail':
        dataset = Cifar10LongTailData(calibration=kwargs.calibration)
    elif kwargs.data == 'cifar100':
        dataset = Cifar100Data(calibration=kwargs.calibration)    
    elif kwargs.data == 'cifar100_longtail':
        dataset = Cifar100LongTailData(calibration=kwargs.calibration)
    elif kwargs.data == 'Imagenet':
        dataset = ImagenetData(calibration=kwargs.calibration)
    elif kwargs.data == 'imagenet_ood':
        dataset = ImagenetOODData(calibration=kwargs.calibration)
    elif kwargs.data == 'imagenet_longtail':
        dataset = ImagenetLongTailData(calibration=kwargs.calibration)    
    
    os.makedirs(f"checkpoints/{kwargs.exp_name}/{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features", exist_ok=True)    
    os.makedirs(f"results/{kwargs.exp_name}/{kwargs.data}_{kwargs.checkpoint.num_classes}_classes_{kwargs.checkpoint.num_features}_features", exist_ok=True)    
    path_model = "checkpoints/{}/{}_{}_classes_{}_features/calibrator_seed-{}_ep-{}.pt".format(
            kwargs.exp_name,
            kwargs.data,
            kwargs.checkpoint.num_classes,
            kwargs.checkpoint.num_features,
            seed,
            total_epochs
        )
    raw_results_path_test_cal = "results/{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}.csv".format(
            kwargs.exp_name,
            kwargs.data,
            kwargs.checkpoint.num_classes,
            kwargs.checkpoint.num_features,
            seed,
            total_epochs           
        )
    
    pl_model = AuxTrainer(kwargs.models, num_classes=kwargs.dataset.num_classes)    
    
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
                    monitor="val_total_loss",
                    patience=10,
                    mode="min",
                    verbose=False,
                    min_delta=0.0,
                )]
        )
    start = time.time()
    trainer.fit(pl_model, dataset.data_train_cal_loader,
                    dataset.data_val_cal_loader)
    train_time = time.time() - start
    print(train_time)
    torch.save(pl_model.model.state_dict(), path_model)
    
    raws = trainer.predict(pl_model, dataset.data_test_cal_loader)
    res = get_raw_res(raws)
    res.to_csv(raw_results_path_test_cal, index=False)

    print("CALIBRATION OVER!")
    print("START TESTING!")
    test(kwargs)
    
    
    
    