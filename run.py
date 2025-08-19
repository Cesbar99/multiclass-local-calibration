import pytorch_lightning as pl
from models import networks, trainers
import hydra
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict, OmegaConf
from actions.pretrain import *
from actions.test import *
from actions.calibrate import *

def main(cfg):
    
    kwargs = OmegaConf.create(cfg)  
    if kwargs.pretrain:
        print("Pretraining model...")
        pretrain(kwargs)
        # Pretrain the model here if needed
        # This is a placeholder for pretraining logic
        pass
    elif kwargs.test:
        print("Testing model...")
        test(kwargs)
        # Logic to resume training from a checkpoint
        # This is a placeholder for resuming logic
        pass
    elif kwargs.calibrate:
        print("Calibrating model with {kwargs.calibration_method} technique...")
        calibrate(kwargs)
        # Logic to calibrate the model
        # This is a placeholder for calibration logic
        pass


@hydra.main(config_path='./configs', config_name='config_local', version_base=None)
def main_entry(cfg: DictConfig):
    kwargs = OmegaConf.create(cfg)                       
    main(cfg) #main(cfg, split) #main(**OmegaConf.to_container(cfg, resolve=True) )
    

if __name__ == "__main__":
    main_entry()
    
    
    