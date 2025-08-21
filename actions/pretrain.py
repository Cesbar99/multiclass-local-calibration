import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from models.networks.networks import *
from models.trainers.trainers import *
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
    total_epochs = kwargs.epochs
    cuda_device = kwargs.cuda_device
    
    if kwargs.data == 'synthetic':
        dataset = SynthData(kwargs.dataset)
        pl_model = SynthTab(input_dim=kwargs.dataset.num_features,            
                            output_dim=kwargs.dataset.num_classes,
                            temperature=kwargs.models.temperature,
                            optimizer_cfg=kwargs.models.optimizer,
                            use_acc=kwargs.models.use_acc
                        )
        
    elif kwargs.data == 'mnist':
        dataset = MnistData()
        pl_model = MnistModel()
        
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
        
    os.makedirs(f"models/{kwargs.exp_name}/{kwargs.data}", exist_ok=True)    
    os.makedirs(f"results/{kwargs.exp_name}/{kwargs.data}", exist_ok=True)    
    path_model = "models/{}/{}/model_seed-{}_ep-{}_tmp_{}.pt".format(
            kwargs.exp_name,
            kwargs.data,
            seed,
            total_epochs,
            kwargs.models.temperature
        )
    raw_results_path_test = "results/{}/{}/raw_results_test_seed-{}_ep-{}_tmp_{}.csv".format(
            kwargs.exp_name,
            kwargs.data,
            seed,
            total_epochs,
            kwargs.models.temperature            
        )
    raw_results_path_cal = "results/{}/{}/raw_results_cal_seed-{}_ep-{}_tmp_{}.csv".format(
            kwargs.exp_name,
            kwargs.data,
            seed,
            total_epochs,
            kwargs.models.temperature            
        )
                
    print(F'BEGIN TRAINING FOR {total_epochs} EPOCHS WITH SEED {seed}!')        
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
                ClearCacheCallback()]
        )
    start = time.time()
    trainer.fit(pl_model, dataset.data_train_loader,
                    dataset.data_val_loader)
    train_time = time.time() - start
    print(train_time)
    torch.save(pl_model.model.state_dict(), path_model)
    
    raws = trainer.predict(pl_model, dataset.data_test_loader)
    res = get_raw_res(raws)
    res.to_csv(raw_results_path_test, index=False)
    
    raws = trainer.predict(pl_model, dataset.data_cal_loader)
    res = get_raw_res(raws)
    res.to_csv(raw_results_path_cal, index=False)
    
    print("PRE-TRAINING OVER!")
    print("START TESTING!")
    test(kwargs)
    
    