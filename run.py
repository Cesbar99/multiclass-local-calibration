import pytorch_lightning as pl
from src.models import networks, trainers
import hydra
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict, OmegaConf
from src.actions.pretrain import *
from src.actions.test import *
from src.actions.calibrate import *
from pytorch_lightning.loggers import WandbLogger
import time
from datetime import datetime
import os
import sys
import wandb

def main(cfg):
    
    kwargs = OmegaConf.create(cfg)  
    
    now = datetime.now()
    start = time.time()
    pl.seed_everything(kwargs.seed)
    
    base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    base_dir = os.path.join(os.path.dirname(os.path.dirname(base_dir)), 'result')
    
    exp_name = f'{kwargs.exp_name}_{kwargs.data}_{now.strftime("%m%d_%H%M")}' #target
    if kwargs.use_wandb:
        if kwargs.resume_training and kwargs.wandb_id:
             wandb_logger = WandbLogger(name=exp_name, project=kwargs.wandb_project, entity=kwargs.wandb_entity, save_dir=base_dir, offline=kwargs.offline, id=kwargs.wandb_id, resume='allow')
        else:
            wandb_logger = WandbLogger(name=exp_name, project=kwargs.wandb_project, entity=kwargs.wandb_entity, save_dir=base_dir, offline=kwargs.offline)
    else:
        wandb_logger = WandbLogger(name=exp_name, project='Test', entity=kwargs.wandb_entity, save_dir=base_dir, offline=kwargs.offline)
    kwargs.wandb_id = wandb_logger.version
    
    default_batch_sizes = {"pre-training": 32, "calibration": 1024}    
    if kwargs.pretrain:
        kwargs.exp_name = 'pre-train'
        if kwargs.dataset.batch_size is None:
            kwargs.dataset.batch_size = kwargs.default_batch_sizes.get(kwargs.exp_name, 32)  # fallback default
        print("Pretraining model...")
        pretrain(kwargs, wandb_logger)
        # Pretrain the model here if needed
        # This is a placeholder for pretraining logic
    elif kwargs.test:
        #kwargs.exp_name = 'test'
        print("Testing model...")
        test(kwargs)
        # Logic to resume training from a checkpoint
        # This is a placeholder for resuming logic
    elif kwargs.calibrate:
        kwargs.exp_name = 'calibrate'
        if kwargs.dataset.batch_size is None:
            kwargs.dataset.batch_size = default_batch_sizes.get(kwargs.exp_name, 512)  # fallback default
        print("Calibrating model with {kwargs.calibration_method} technique...")
        calibrate(kwargs, wandb_logger)
        # Logic to calibrate the model
        # This is a placeholder for calibration logic

    wandb.finish()
    del wandb_logger

    end = time.time()
    time_elapsed = end-start
    print('Total running time: {:.0f}h {:.0f}m'.
        format(time_elapsed // 3600, (time_elapsed % 3600)//60))
    
@hydra.main(config_path='./configs', config_name='config_local', version_base=None)
def main_entry(cfg: DictConfig):                      
    main(cfg) #main(cfg, split) #main(**OmegaConf.to_container(cfg, resolve=True) )
    

if __name__ == "__main__":
    main_entry()
    
    
    