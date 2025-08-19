import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from models import networks, trainers
import hydra
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict, OmegaConf
import time
from utils.utils import get_raw_res

def pretrain():
    
    seed = kwargs.seed
    pl.seed_everything(seed, workers=True)  
    
    if kwargs.experiment == 'synthetic':
        data = SynthData()
        model = SynthTab(input_dim=kwargs.mlp_input_dim, 
                           output_dim=kwargs.num_classes, 
                           temperature=kwargs.temperature)
    elif kwargs.experiment == 'mnist':
        data = MnistData()
        model = MnistModel(input_dim=kwargs.mlp_input_dim, 
                           output_dim=kwargs.num_classes, 
                           temperature=kwargs.temperature)
    elif kwargs.experiment == 'cifar10':
        data = Cifar10Data()
        model = Cifar10Model(input_dim=kwargs.mlp_input_dim, 
                             output_dim=kwargs.num_classes, 
                             temperature=kwargs.temperature)
    elif kwargs.experiment == 'cifar10_ood':
        data = Cifar10OODData()
        model = Cifar10OODModel(input_dim=kwargs.mlp_input_dim, 
                                output_dim=kwargs.num_classes, 
                                temperature=kwargs.temperature)
    elif kwargs.experiment == 'cifar10_longtail':
        data = Cifar10LongTailData()
        model = Cifar10LongTailModel(input_dim=kwargs.mlp_input_dim, 
                                    output_dim=kwargs.num_classes, 
                                    temperature=kwargs.temperature)
    elif kwargs.experiment == 'cifar100':
        data = Cifar100Data()        
        model = Cifar100Model(input_dim=kwargs.mlp_input_dim, 
                              output_dim=kwargs.num_classes, 
                              temperature=kwargs.temperature)    
    elif kwargs.experiment == 'cifar100_longtail':
        data = Cifar100LongTailData()
        model = Cifar100LongTailModel(input_dim=kwargs.mlp_input_dim, 
                                      output_dim=kwargs.num_classes, 
                                      temperature=kwargs.temperature)
    elif kwargs.experiment == 'Imagenet':
        data = ImagenetData()
        model = ImagenetModel(input_dim=kwargs.mlp_input_dim, 
                              output_dim=kwargs.num_classes, 
                              temperature=kwargs.temperature)
    elif kwargs.experiment == 'imagenet_ood':
        data = ImagenetOODData()
        model = ImagenetOODModel(input_dim=kwargs.mlp_input_dim, 
                                 output_dim=kwargs.num_classes, 
                                 temperature=kwargs.temperature)
    elif kwargs.experiment == 'imagenet_longtail':
        data = ImagenetLongTailData()  
        model = ImagenetLongTailModel(input_dim=kwargs.mlp_input_dim, 
                                      output_dim=kwargs.num_classes, 
                                      temperature=kwargs.temperature)    
    
    logger = TensorBoardLogger(
            "tb_logs", name="{}".format(kwargs.experiment)
        )
    total_epochs = kwargs.total_epochs    
    cuda_device = kwargs.cuda_device
    print(F'BEGIN TRAINING FOR {total_epochs} EPOCHS WITH SEED {seed}!')        
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
    
    print("PRE-TRAINING OVER!")
    
    
    