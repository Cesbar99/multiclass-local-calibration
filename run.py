from matplotlib.pylab import rint
import pytorch_lightning as pl
import hydra
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict, OmegaConf
from src.actions.pretrain import *
from src.actions.test import *
from src.actions.calibrate import *
from src.actions.quantize import *
from src.actions.replicate import *
from src.actions.competition import *
from src.actions.viz_and_test import *
from pytorch_lightning.loggers import WandbLogger
import time
from datetime import datetime
import os
import sys
import wandb
    
def main(cfg: DictConfig):
    kwargs = cfg #OmegaConf.create(cfg)  
    
    now = datetime.now()
    start = time.time()    
        
    dataset_name = kwargs.data
    model_name = kwargs.models_map[dataset_name]
    epochs = kwargs.checkpoint.epochs
    if epochs == 9:
        model_class = 'resnet'
    elif kwargs.checkpoint.epochs == 5:
        model_class = 'vit'
    else:
        if not kwargs.data == 'weather':
            raise ValueError(
                f'Checkpoint not corresponding to a trained modl! {kwargs.checkpoint.epochs} was given but only 9 and 20 are supported')

    base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    base_dir = os.path.join(os.path.dirname(os.path.dirname(base_dir)), 'result')
    
    exp_name = f'{kwargs.exp_name}_{kwargs.data}_{now.strftime("%m%d_%H%M")}' #target
    if kwargs.use_optuna:
        exp_name = 'optuna_'+ exp_name #{kwargs.data}_{now.strftime("%m%d_%H%M")}' 
    if kwargs.use_wandb:
        if kwargs.resume_training and kwargs.wandb_id:
             wandb_logger = WandbLogger(name=exp_name, project=kwargs.wandb_project, entity=kwargs.wandb_entity, save_dir=base_dir, offline=kwargs.offline, id=kwargs.wandb_id, resume='allow')
        else:
            wandb_logger = WandbLogger(name=exp_name, project=kwargs.wandb_project, entity=kwargs.wandb_entity, save_dir=base_dir, offline=kwargs.offline)        
    else:
        wandb_logger = WandbLogger(name=exp_name, project='Test', entity=kwargs.wandb_entity, save_dir=base_dir, offline=kwargs.offline)
    #if kwargs.use_optuna:
    #    wandb_optuna_logger = WandbLogger(name=optuna_exp_name, project=kwargs.wandb_project, entity=kwargs.wandb_entity, save_dir=base_dir, offline=kwargs.offline)
    #else: 
    #    wandb_optuna_logger = None
    kwargs.wandb_id = wandb_logger.version
    
    if kwargs.pretrain:
        kwargs.exp_name = 'pre-train'
        if kwargs.dataset.batch_size is None:
            kwargs.dataset.batch_size = kwargs.batch_size_map.get(kwargs.exp_name, 32)  # fallback default        
            print('Using default batch_size set to: ', kwargs.dataset.batch_size)
        for seed in kwargs.seeds:   
            pl.seed_everything(seed)    
            kwargs.seed = seed
            kwargs.checkpoint = fix_default_checkpoint(kwargs)
            print("Pretraining model...")
            pretrain(kwargs, wandb_logger)
        
    elif kwargs.test:
        print("Testing model...")        
        #for seed in kwargs.seeds:   
        if 'competition' in exp_name:    
            kwargs.exp_name = 'competition'                        
            for method in kwargs.methods:       
                if kwargs.only_test:
                    print(f'Using method: {method}')             
                pl.seed_everything(seed)       
                kwargs.seed = seed
                kwargs.checkpoint.seed = seed
                kwargs.method = method                    
                test(kwargs)
        elif 'calibrate' in exp_name:    
            kwargs.exp_name = 'calibrate'
            pl.seed_everything(seed)       
            kwargs.seed = seed
            kwargs.checkpoint.seed = seed
            test(kwargs)     
        elif 'quantize' in exp_name:    
            kwargs.exp_name = 'quantize'
            for slot in kwargs.models.slots:
                kwargs.models.S = slot
                repr_dim = 2048 if model_class == 'resnet' else 768
                kwargs.models.d = int(repr_dim/slot)
                print(f'Testing model with {kwargs.models.d} dimensions per slot...')
                print(f'Testing model with {kwargs.models.S} slots...')
                for kappa in kwargs.models.kappas:
                    kwargs.models.K = kappa               
                    print(f'Testing model with {kwargs.models.K} codewords...')
                    pl.seed_everything(seed)       
                    kwargs.seed = seed
                    kwargs.checkpoint.seed = seed
                    test(kwargs)    
        elif 'pre-train' in exp_name:    
            kwargs.exp_name = 'pre-train'
            for seed in kwargs.seeds:   
                pl.seed_everything(seed)     
                kwargs.seed = seed
                test(kwargs)       
        elif 'replicate' in exp_name:    
            kwargs.exp_name = 'replicate'                            
            pl.seed_everything(seed)       
            kwargs.seed = seed
            kwargs.checkpoint.seed = seed
            test(kwargs)   
        elif 'ess_plot' in exp_name:
            kwargs.exp_name = 'ess_plot'                            
            # pl.seed_everything(seed)       
            # kwargs.seed = seed
            # kwargs.checkpoint.seed = seed
            test(kwargs)                

    elif kwargs.calibrate:                
        kwargs.exp_name = 'calibrate'
        if kwargs.dataset.batch_size is None:
            kwargs.dataset.batch_size = kwargs.batch_size_map.get(kwargs.exp_name, 512)  # fallback default        
            print('Using default batch_size set to: ', kwargs.dataset.batch_size)
            print("Calibrating model with {kwargs.calibration_method} technique...")
        for seed in kwargs.seeds:   
            pl.seed_everything(seed)     
            kwargs.seed = seed
            kwargs.checkpoint.seed = seed
            calibrate(kwargs, wandb_logger)
            
    elif kwargs.competition:        
        kwargs.exp_name = 'competition'
        if kwargs.dataset.batch_size is None:
            kwargs.dataset.batch_size = kwargs.batch_size_map.get(kwargs.exp_name, 512)  # fallback default        
            print('Using default batch_size set to: ', kwargs.dataset.batch_size)
            print("Testing peroformance of competitors...")
        for seed in kwargs.seeds:            
            pl.seed_everything(seed) 
            kwargs.seed = seed
            kwargs.checkpoint.seed = seed
            competition(kwargs, wandb_logger)
            
    elif kwargs.viz_and_test:
        print("Computing visualisations and computing aggreagting metricss...")                                 
        viz_and_test(kwargs)
        
    elif kwargs.quantize:                
        kwargs.exp_name = 'quantize'

        if kwargs.models.quadratic:
            print("Using quadratic calibration model...")
        else:
            print("Using linear calibration model...")
            
        if kwargs.dataset.batch_size is None:
            kwargs.dataset.batch_size = kwargs.batch_size_map.get(kwargs.exp_name, 512)  # fallback default        
            print('Using default batch_size set to: ', kwargs.dataset.batch_size) 
                       
        for slot in kwargs.models.slots:
            kwargs.models.S = slot
            repr_dim = 2048 if model_class == 'resnet' else 768
            kwargs.models.d = int(repr_dim / slot)
            print(f'Testing model with {kwargs.models.d} dimensions per slot...')
            print(f'Testing model with {kwargs.models.S} slots...')
            print(f'Quantizing model with {kwargs.models.S} slots...')
            
            for kappa in kwargs.models.kappas:
                kwargs.models.K = kappa               
                print(f'Quantizing model with {kwargs.models.K} codewords...')
                
                for seed in kwargs.seeds:               
                    pl.seed_everything(seed)     
                    kwargs.seed = seed
                    kwargs.checkpoint.seed = seed
                    quantize(kwargs, wandb_logger)

    elif kwargs.replicate:                
        kwargs.exp_name = 'replicate'        
            
        if kwargs.dataset.batch_size is None:
            kwargs.dataset.batch_size = kwargs.batch_size_map.get(kwargs.exp_name, 512)  # fallback default        
            print('Using default batch_size set to: ', kwargs.dataset.batch_size)                 
                
        for seed in kwargs.seeds: 
            print('Starting replicator calibration')              
            pl.seed_everything(seed)     
            kwargs.seed = seed
            kwargs.checkpoint.seed = seed
            replicate(kwargs, wandb_logger)
    
    wandb.finish()
    del wandb_logger

    end = time.time()
    time_elapsed = end-start
    print('Total running time: {:.0f}h {:.0f}m'.
        format(time_elapsed // 3600, (time_elapsed % 3600)//60))
    
#@hydra.main(config_path='./src/configs', config_name='config_local', version_base=None)
def main_entry():            
    
    #cli_overrides = [arg for arg in sys.argv[1:] if "=" in arg]
    excluded_keys = {"dataset", "models"}
    init_overrides = [
        arg for arg in sys.argv[1:]
        if "=" in arg and arg.split(".")[0] not in excluded_keys
    ]
    second_overrides = [
        arg for arg in sys.argv[1:]
        if "=" in arg and arg.split(".")[0] in excluded_keys
    ]
    with initialize(config_path="./src/configs", version_base=None):
                
        cfg = compose(config_name="config_local", overrides=init_overrides)
        
        dataset_name = cfg.data
        if cfg.pretrain:
            model_name = cfg.models_map[cfg.data].strip()            
            full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides
            cfg = compose(config_name="config_local", overrides=full_overrides)
            
        elif cfg.calibrate:
            model_name = 'calibrator'
            full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides            
            cfg = compose(config_name="config_local", overrides=full_overrides)
        
        elif cfg.quantize:
            model_name = 'quantizer'
            full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides            
            cfg = compose(config_name="config_local", overrides=full_overrides)
            
        elif cfg.replicate:
            model_name = 'replicator'
            full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides            
            cfg = compose(config_name="config_local", overrides=full_overrides)
            
        elif cfg.test:
            if cfg.exp_name not in ['pre-train', 'calibrate', 'competition', 'quantize', 'replicate', 'ess_plot']:
                raise ValueError(f"Explicitly provide 'exp_name' argument from CLI when testing! Allowed values are 'pre-train', 'calibrate', 'competition', 'quantize', 'replicate'. Instead '{cfg.exp_name}' was given!")                 
            
            elif cfg.exp_name == 'pre-train':
                model_name = cfg.models_map[cfg.data].strip() 
                full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides
                model_name = cfg.models_map[cfg.data]
                cfg = compose(config_name="config_local", overrides=full_overrides)
                
            elif cfg.exp_name == 'calibrate':
                model_name = 'calibrator'
                full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides
                model_name = 'calibrator'
                cfg = compose(config_name="config_local", overrides=full_overrides)
                
            elif cfg.exp_name == 'quantize':
                model_name = 'quantizer'
                full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides
                model_name = 'quantizer'
                cfg = compose(config_name="config_local", overrides=full_overrides)
            
            elif cfg.exp_name == 'replicate':
                model_name = 'replicator'
                full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides
                model_name = 'replicator'
                cfg = compose(config_name="config_local", overrides=full_overrides)            
                
            elif cfg.exp_name == 'competition':
                model_name = 'competition'
                full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides
                model_name = 'competition'
                cfg = compose(config_name="config_local", overrides=full_overrides)
                
            elif cfg.exp_name == 'ess_plot':
                model_name = 'competition'
                full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides
                model_name = 'competition'
                cfg = compose(config_name="config_local", overrides=full_overrides)
                
        elif cfg.competition: #.exp_name == 'competition':
            model_name = 'competition'
            full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides
            cfg = compose(config_name="config_local", overrides=full_overrides)
            
        elif cfg.viz_and_test: #.exp_name == 'competition':
            model_name = 'competition'
            full_overrides = init_overrides + [f"dataset={dataset_name}", f"models={model_name}"] + second_overrides
            cfg = compose(config_name="config_local", overrides=full_overrides)
                                      
    main(cfg) #main(cfg, split) #main(**OmegaConf.to_container(cfg, resolve=True) )
    

if __name__ == "__main__":
    main_entry()
    
    
    