# replicator_calibration
oops! I calibrated again!

data types: 'cifar10', 'cifar100', 'tissue'

num classes: 10, 100, 8

TO PERFORM REPLICATOR CALIBRATION RUN:

CUDA_VISIBLE_DEVICES=1 python -u run.py pretrain=False calibrate=False quantize=False replicate=True test=False competition=False exp_name=replicate data={one_of_the_possible_data_types} n_bins_calibration_metrics=15 gamma=10 dataset.batch_size=64 return_features=True similarity_dim=50 use_optuna=True optuna_epochs=20 n_trials=60 models.max_iter=100 checkpoint.num_classes={one_of_the-above_values} checkpoint.epochs=9 models.lr=1e-3 models.n_steps=100 models.step_size=0.01 models.kl_reg=0. models.state_dependent=True models.weight_decay=0 models.alpha=1 models.finetune_epochs=30 models.finetune_grad_clip=0 models.feature_dim=10 models.fit_stage=False models.potential=False

TO EVALUATE COMPETITORS RUN (SMS, DC, TS, PS, IR):

CUDA_VISIBLE_DEVICES=1 python -u run.py pretrain=False calibrate=False quantize=False replicate=False test=False competition=True exp_name=competition data={one_of_the_possible_data_types} gamma=10 dataset.batch_size=128 return_features=True similarity_dim=50 models.max_iter=1000 checkpoint.num_classes={one_of_the-above_values} checkpoint.epochs=9 models.temp_lr=1e-3 n_bins_calibration_metrics=15

ALERT THIS REQUIRES PRE-TRAINED CLASSIFIER WHICH IS MISSING FROM THE REPO!

To solve this problem deo the following:
1) in the cloned repo, create a new folder named results
2) there keep a folder named pre-train
3) inside keep the raw_results for each model (labels, predictios=ns, logits, features)
4) Have a separate file for the calibration data and a file for the test data

Now the code can access the calibration and test data!

DON'T FORGET TO CHECK THE CONFIG FILE (config_local.yaml) AND CHANGE PATHS ACCORDINGLY!

TO RUN A NEW MODEL OR A NEW DATASET:

1) Go to config.yaml and edit models_map accordingly.
2) Go to replicate.py and add at the top (line 65 circa) your dataset.
3) Create a config file for your dataset (use configs/dataset/cifar10.yaml as a template. Either manually add the correct class priors or if you have a lot of classes override them loading the dataset as per the following step)
4) Go to data_sets/dataset.py and write your dataloader there (use cifar10 as a template). 
