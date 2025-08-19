import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from localibration.models import networks, trainers
import hydra
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict, OmegaConf
import time
from localibration.utils.utils import get_raw_res
from localibration.calibrator.local_net import AuxiliaryMLP

def pretrain():
    
    seed = kwargs.seed
    pl.seed_everything(seed, workers=True)  
    
    if kwargs.experiment == 'synthetic':
        data = SynthData(calibration=kwargs.calibration)
    elif kwargs.experiment == 'mnist':
        data = MnistData(calibration=kwargs.calibration)
    elif kwargs.experiment == 'cifar10':
        data = Cifar10Data(calibration=kwargs.calibration)
    elif kwargs.experiment == 'cifar10_ood':
        data = Cifar10OODData(calibration=kwargs.calibration)
    elif kwargs.experiment == 'cifar10_longtail':
        data = Cifar10LongTailData(calibration=kwargs.calibration)
    elif kwargs.experiment == 'cifar100':
        data = Cifar100Data(calibration=kwargs.calibration)    
    elif kwargs.experiment == 'cifar100_longtail':
        data = Cifar100LongTailData(calibration=kwargs.calibration)
    elif kwargs.experiment == 'Imagenet':
        data = ImagenetData(calibration=kwargs.calibration)
    elif kwargs.experiment == 'imagenet_ood':
        data = ImagenetOODData(calibration=kwargs.calibration)
    elif kwargs.experiment == 'imagenet_longtail':
        data = ImagenetLongTailData(calibration=kwargs.calibration)    
    
    calibrator_model = AuxiliaryMLP()
    model = AuxTrainer(calibrator_model, dim, num_classes=kwargs.num_classes, lr=kwargs.lr, alpha1=kwargs.alpha1, alpha2=kwargs.alpha2,
                 lambda_kl=kwargs.lambda_kl, entropy_factor=kwargs.entropy_factor, noise=kwargs.noise, smoothing=kwargs.smoothing,
                 logits_scaling=kwargs.logits_scaling, sampling=kwargs.sampling, predict_labels=kwargs.predict_labels,
                 use_empirical_freqs=kwargs.use_empirical_freqs, js_distance=kwargs.js_distance, model_confident=kwargs.model_confident)
    logger = TensorBoardLogger(
            "tb_logs", name="{}".format(kwargs.experiment)
        )
    total_epochs = kwargs.total_epochs    
    cuda_device = kwargs.cuda_device
    print(F'BEGIN CALIBRATION FOR {total_epochs} EPOCHS WITH SEED {seed}!')        
    trainer = pl.Trainer(
            max_epochs=total_epochs,
            accelerator="cuda",
            devices=[cuda_device],
            logger=logger,
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
    torch.save(model.model.state_dict(), path_model_f)
    
    raws = trainer.predict(model_joint, dataset.data_test_loader)
    
    res = get_raw_res(raws, paradigm)
    os.makedirs("results/{}".format(data), exist_ok=True)
    res.to_csv(raw_results_path_test, index=False)
    
    print("CALIBRATION OVER!")
    
    
    