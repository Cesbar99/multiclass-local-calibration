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


class VerboseModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_best_score = None

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)

        # Check if a new best model was saved
        if self.best_model_score is not None:
            if self._last_best_score is None or self.best_model_score < self._last_best_score:
                current_epoch = trainer.current_epoch
                print(f"\nNew best model saved at epoch {current_epoch}: {self.best_model_path} with val_total = {self.best_model_score:.4f}\n")
                self._last_best_score = self.best_model_score


def calibrate(kwargs, wandb_logger):
    
    seed = kwargs.seed
    total_epochs = kwargs.models.epochs    
    cuda_device = kwargs.cuda_device
    pl.seed_everything(seed, workers=True)  
    
    if kwargs.data == 'synthetic':
        dataset = SynthData(kwargs, experiment=kwargs.exp_name)        
    elif kwargs.data == 'mnist':
        if kwargs.dataset.variant:
            kwargs.data = kwargs.data + '_' + kwargs.dataset.variant                        
        dataset = MnistData(kwargs, experiment=kwargs.exp_name)
    elif kwargs.data == 'tissue':
        dataset = MedMnistData(kwargs, experiment=kwargs.exp_name)        
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
    
    
    if kwargs.data == 'synthetic':
        os.makedirs(f"checkpoints/{kwargs.exp_name}/{kwargs.data}_{kwargs.checkpoint.num_classes}_classes_{kwargs.checkpoint.num_features}_features", exist_ok=True)    
        os.makedirs(f"results/{kwargs.exp_name}/{kwargs.data}_{kwargs.checkpoint.num_classes}_classes_{kwargs.checkpoint.num_features}_features", exist_ok=True)    
        
        pl_model = AuxTrainer(kwargs.models, num_classes=kwargs.checkpoint.num_classes)    
        
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
    else:        
        os.makedirs(f"checkpoints/{kwargs.exp_name}/", exist_ok=True) #/{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features", )    
        os.makedirs(f"results/{kwargs.exp_name}/{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features", exist_ok=True)    
        
        pl_model = AuxTrainer(kwargs.models, num_classes=kwargs.dataset.num_classes)    
        
        path_model = "checkpoints/{}/{}_{}_classes_{}_features/calibrator_seed-{}_ep-{}.pt".format(
                kwargs.exp_name,
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                seed,
                total_epochs
            )
        raw_results_path_test_cal = "results/{}/{}_{}_classes_{}_features/raw_results_test_cal_seed-{}_ep-{}.csv".format(
                kwargs.exp_name,
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                seed,
                total_epochs           
            )
        
    print(F'BEGIN CALIBRATION FOR {total_epochs} EPOCHS WITH SEED {seed}!')        
    trainer = pl.Trainer(
            max_epochs=total_epochs,
            accelerator="cuda",
            devices=[cuda_device],
            logger=wandb_logger,
            check_val_every_n_epoch=1,
            #gradient_clip_val=5,
            deterministic=True,
            callbacks=[
                 EarlyStopping(
                     monitor="val_total",
                     patience=5,
                     mode="min",
                     verbose=True,
                     min_delta=0.0,
                 ),
                 VerboseModelCheckpoint(
                    monitor="val_total",                                                                                            # Metric to track
                    mode="min",                                                                                                     # Lower is better
                    save_top_k=1,                                                                                                   # Only keep the best model
                    filename=f"{kwargs.data}_{kwargs.dataset.num_classes}_classes_{kwargs.dataset.num_features}_features",          # Static filename (no epoch suffix)
                    dirpath=os.path.dirname(f'checkpoints/{kwargs.exp_name}/'),                                                     # Save in your existing checkpoint folder
                    save_weights_only=True,                                                                                         # Save only weights (not full LightningModule)
                    auto_insert_metric_name=False,                                                                                  # Prevent metric name in filename
                    every_n_epochs=1,                                                                                               # Run every epoch                    
                )   
            ]
         )
    start = time.time()
    trainer.fit(pl_model, dataset.data_train_cal_loader,
                    dataset.data_val_cal_loader)
    train_time = time.time() - start
    print(train_time)
    #torch.save(pl_model.model.state_dict(), path_model)
    best_model_path = trainer.checkpoint_callback.best_model_path
    checkpoint = torch.load(best_model_path)
    pl_model.load_state_dict(checkpoint['state_dict'])

    raws = trainer.predict(pl_model, dataset.data_test_cal_loader)
    res = get_raw_res(raws)
    res.to_csv(raw_results_path_test_cal, index=False)

    print("CALIBRATION OVER!")
    print("START TESTING!")        
    test(kwargs)
    
    
    
    