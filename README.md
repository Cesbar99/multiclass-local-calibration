# LOCAL CALIBRATION WITH JENSEN SHANNON DISTANCE

data types: 'cifar10', 'cifar100', 'tissue'

num classes: 10, 100, 8

TO PERFORM LCN CALIBRATION RUN:

CIFAR10: 

CUDA_VISIBLE_DEVICES=1 python -u run.py exp_name=calibrate pretrain=False calibrate=True data=cifar10 checkpoint.epochs=9 models.log_var_initializer=10 models.alpha1=1 models.alpha2=0. models.lambda_kl=1 models.sampling=False models.smoothing=0. models.dropout=0.3 models.hidden_dim=64 models.epochs=100 use_optuna=False n_trials=100 models.optimizer.weight_decay=0. dataset.batch_size=1024 dataset.feature_dim=2048 similarity_dim=50 calibrator_version=v2 gamma=10 models.adabw=False models.fixed_var=True models.linearly_combine_pca=True models.kernel_only=False models.alpha_sim=1 models.alpha_cls=1 extract_embeddings=True

CIFAR100:

CUDA_VISIBLE_DEVICES=1 python -u run.py exp_name=calibrate pretrain=False calibrate=True data=cifar100 checkpoint.epochs=9 models.log_var_initializer=10 models.alpha1=1 models.alpha2=0. models.lambda_kl=1 models.sampling=False models.smoothing=0. models.dropout=0.3 models.hidden_dim=64 models.epochs=100 use_optuna=False n_trials=100 models.optimizer.weight_decay=0. dataset.batch_size=1024 dataset.feature_dim=2048 similarity_dim=50 calibrator_version=v2 gamma=10 models.adabw=False models.fixed_var=True models.linearly_combine_pca=True models.kernel_only=False models.alpha_sim=1 models.alpha_cls=1 extract_embeddings=True

Tissue:

CUDA_VISIBLE_DEVICES=1 python -u run.py exp_name=calibrate pretrain=False calibrate=True data=tissue checkpoint.epochs=9 models.log_var_initializer=10 models.alpha1=1 models.alpha2=0. models.lambda_kl=1 models.sampling=False models.smoothing=0. models.dropout=0.3 models.hidden_dim=64 models.epochs=100 use_optuna=False n_trials=100 models.optimizer.weight_decay=0. dataset.batch_size=1024 dataset.feature_dim=2048 similarity_dim=50 calibrator_version=v2 gamma=10 models.adabw=False models.fixed_var=True models.linearly_combine_pca=True models.kernel_only=False models.alpha_sim=1 models.alpha_cls=1 extract_embeddings=True

TO PERFORM KERNEL SIMPLY RUN THE SAME COMMAND BUT SET:

models.lambda_kl=0

TO EVALUATE COMPETITORS RUN (SMS, DC, TS, PS, IR):

CUDA_VISIBLE_DEVICES=1 python -u run.py pretrain=False calibrate=False quantize=False replicate=False test=False competition=True exp_name=competition data=cifar10 gamma=10 dataset.batch_size=128 return_features=True similarity_dim=50 models.max_iter=2000 checkpoint.num_classes=10 checkpoint.epochs=9 models.temp_lr=1e-3 n_bins_calibration_metrics=15 models.num_neighbors=50

ALERT THIS REQUIRES PRE-TRAINED CLASSIFIER WHICH IS MISSING FROM THE REPO!

To solve this problem deo the following:
1) in the cloned repo, create a new folder named results
2) create another new folder named data. Inside create a folder named CIFAR10 (or CIFAR100 or TISSUE) and ther store: cifar-10-python.tar.gz
3) Run the following command:
   
CUDA_VISIBLE_DEVICES=1 python -u run.py pretrain=True calibrate=False quantize=False replicate=False test=False competition=True exp_name=pre-train data=cifar10 gamma=10 dataset.batch_size=128 return_features=True similarity_dim=50 models.epochs=9 checkpoint.epochs=9 n_bins_calibration_metrics=15 

Now the code can access the calibration and test data!

DON'T FORGET TO CHECK THE CONFIG FILE (config_local.yaml) AND CHANGE PATHS ACCORDINGLY!

TO RUN A NEW MODEL OR A NEW DATASET:

1) Create a config file for your dataset (use configs/dataset/cifar10.yaml as a template). 
2) Create a config file for your model (inside configs/models/). 
3) Go to data_sets/dataset.py and write your dataloader there (use cifar10 as a template).
4) Go to algorithms/networks.py and add ther your new architecture or model there
5) Go to algorithms/trainers.py and add the training for your custom architecture
6) Go to config.yaml and edit models_map accordingly.
7) 6) Go to pretrain.py and add at the top your dataset.
8) Go to calibrate.py and add at the top (line 65 circa) your dataset.
9) If your base-model to pre-train does not support pytorch lightning, go to actions/pretrain.py and add your training code there 

REMEMBER THAT IF YOUR MODEL DOES NOT SUPPORT FEATURE REPRESENTATIONS, LOCAL METRICS CANNOT BE COMPUTED!
