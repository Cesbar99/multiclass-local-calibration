import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, scale
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from utils.utils import *
from scipy.io import loadmat
import medmnist
from medmnist import TissueMNIST, PathMNIST, INFO
from torch.utils.data import random_split
import os
from tiny_imagenet_torch import TinyImageNet
from datasets import load_dataset
from PIL import Image
import io
import numpy as np





def generateCalibrationData(kwargs, dataname=None):
    #temperature = str(int(kwargs.checkpoint.temperature))

    if kwargs.data == 'synthetic':   
        test_results = "results/{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}_tmp_{}.csv".format(
        'pre-train',
        kwargs.data,
        kwargs.checkpoint.num_classes,
        kwargs.checkpoint.num_features,
        kwargs.checkpoint.seed,
        kwargs.checkpoint.epochs,
        kwargs.checkpoint.temperature       
        )
        cal_results = "results/{}/{}_{}_classes_{}_features/raw_results_eval_cal_seed-{}_ep-{}_tmp_{}.csv".format(
            'pre-train',
            kwargs.data,
            kwargs.checkpoint.num_classes,
            kwargs.checkpoint.num_features,
            kwargs.checkpoint.seed,
            kwargs.checkpoint.epochs,
            kwargs.checkpoint.temperature
        )           
    else:        
        test_results = "results/{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}_tmp_{}.csv".format(
            'pre-train',
            kwargs.data,
            kwargs.dataset.num_classes,
            kwargs.dataset.num_features,
            kwargs.checkpoint.seed,
            kwargs.checkpoint.epochs,
            kwargs.checkpoint.temperature       
        )
        cal_results = "results/{}/{}_{}_classes_{}_features/raw_results_eval_cal_seed-{}_ep-{}_tmp_{}.csv".format(
            'pre-train',
            kwargs.data,
            kwargs.dataset.num_classes,
            kwargs.dataset.num_features,
            kwargs.checkpoint.seed,
            kwargs.checkpoint.epochs,
            kwargs.checkpoint.temperature
        )    
    
    # Load your data
    df_train_calibration_data = pd.read_csv(test_results)
    df_eval_calibration_data = pd.read_csv(cal_results)    

    # Extract features and labels
    X_train_cal = df_train_calibration_data.drop(columns=["true", "preds"]).values
    std_per_column = np.std(X_train_cal, axis=0)
    print("Standard deviation per column:")
    print(std_per_column)
    print("Mean standard deviation per column:")
    print(np.mean(std_per_column))
    
    y_train_cal = df_train_calibration_data["true"].values
    p_train_cal = df_train_calibration_data["preds"].values

    X_eval_cal_full = df_eval_calibration_data.drop(columns=["true", "preds"]).values
    y_eval_cal_full = df_eval_calibration_data["true"].values
    p_eval_cal_full = df_eval_calibration_data["preds"].values

    # Split into 90% test and 10% val
    X_test_cal, X_val_cal, y_test_cal, y_val_cal, p_test_cal, p_val_cal = train_test_split(X_eval_cal_full,
                                                                                            y_eval_cal_full,
                                                                                            p_eval_cal_full,
                                                                                            test_size=0.1,
                                                                                            random_state=kwargs.seed,
                                                                                            stratify=y_eval_cal_full
                                                                                            )

    
    print(f'Learn Calibration shape: {X_train_cal.shape}, Validation shape: {X_val_cal.shape}, Test Calibration shape: {X_test_cal.shape}')
    # Convert to PyTorch tensors
    X_train_cal = torch.tensor(X_train_cal, dtype=torch.float32)
    y_train_cal = torch.tensor(y_train_cal, dtype=torch.long)
    p_train_cal = torch.tensor(p_train_cal, dtype=torch.long)

    X_test_cal = torch.tensor(X_test_cal, dtype=torch.float32)
    y_test_cal = torch.tensor(y_test_cal, dtype=torch.long)
    p_test_cal = torch.tensor(p_test_cal, dtype=torch.long)
    
    X_val_cal = torch.tensor(X_val_cal, dtype=torch.float32)
    y_val_cal = torch.tensor(y_val_cal, dtype=torch.long)
    p_val_cal = torch.tensor(p_val_cal, dtype=torch.long)

    # Create datasets
    train_cal_set = CalibrationDataset(X_train_cal, y_train_cal,p_train_cal, num_classes=kwargs.dataset.num_classes)
    test_cal_set = CalibrationDataset(X_test_cal, y_test_cal, p_test_cal, num_classes=kwargs.dataset.num_classes)
    val_cal_set = CalibrationDataset(X_val_cal, y_val_cal, p_val_cal, num_classes=kwargs.dataset.num_classes)
    
    # Create data loaders
    data_train_cal_loader = DataLoader(
        train_cal_set, batch_size=kwargs.dataset.batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    data_test_cal_loader = DataLoader(
        test_cal_set, batch_size=kwargs.dataset.batch_size, shuffle=False, num_workers=8, pin_memory=True
    )
    data_val_cal_loader = DataLoader(
        val_cal_set, batch_size=kwargs.dataset.batch_size, shuffle=False, num_workers=8, pin_memory=True
    )

    return data_train_cal_loader, data_test_cal_loader, data_val_cal_loader 

def generateCalibrationDatav2(kwargs, dataname=None):
    #temperature = str(int(kwargs.checkpoint.temperature))
    
    corruptions = [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "fog",
        "snow",
        "frost", # try        
        "brightness", # good
        "contrast",
        "pixelate",        
    ]
    
    if (kwargs.corruption_type) and (kwargs.corruption_type not in corruptions):
        raise ValueError(f'Unknown corruption type! {kwargs.corruption_type} was given.')

    if kwargs.data == 'synthetic':   
        test_results = "results/{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}_tmp_{}.csv".format(
        'pre-train',
        kwargs.data,
        kwargs.checkpoint.num_classes,
        kwargs.checkpoint.num_features,
        kwargs.checkpoint.seed,
        kwargs.checkpoint.epochs,
        kwargs.checkpoint.temperature       
        )
        cal_results = "results/{}/{}_{}_classes_{}_features/raw_results_eval_cal_seed-{}_ep-{}_tmp_{}.csv".format(
            'pre-train',
            kwargs.data,
            kwargs.checkpoint.num_classes,
            kwargs.checkpoint.num_features,
            kwargs.checkpoint.seed,
            kwargs.checkpoint.epochs,
            kwargs.checkpoint.temperature
        )           
    else:        
        if kwargs.corruption_type:
            cal_results = "results/{}/{}_{}_classes_{}_features/raw_results_train_cal_corrupt_{}_seed-{}_ep-{}_tmp_{}.csv".format(
                'pre-train',
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                kwargs.corruption_type+f"_severity_{kwargs.severity}",
                kwargs.checkpoint.seed,
                kwargs.checkpoint.epochs,
                kwargs.checkpoint.temperature       
            )
            test_results = "results/{}/{}_{}_classes_{}_features/raw_results_eval_cal_corrupt_{}_seed-{}_ep-{}_tmp_{}.csv".format(
                'pre-train',
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                kwargs.corruption_type+f"_severity_{kwargs.severity}",
                kwargs.checkpoint.seed,
                kwargs.checkpoint.epochs,
                kwargs.checkpoint.temperature
            )  
        else:
            cal_results = "results/{}/{}_{}_classes_{}_features/raw_results_train_cal_seed-{}_ep-{}_tmp_{}.csv".format(
                'pre-train',
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                kwargs.checkpoint.seed,
                kwargs.checkpoint.epochs,
                kwargs.checkpoint.temperature       
            )
            test_results = "results/{}/{}_{}_classes_{}_features/raw_results_eval_cal_seed-{}_ep-{}_tmp_{}.csv".format(
                'pre-train',
                kwargs.data,
                kwargs.dataset.num_classes,
                kwargs.dataset.num_features,
                kwargs.checkpoint.seed,
                kwargs.checkpoint.epochs,
                kwargs.checkpoint.temperature
            )    
            if kwargs.data == 'weather':
                test_shift_results = "results/{}/{}_{}_classes_{}_features/raw_results_eval_cal_shift_seed-{}_ep-{}_tmp_{}.csv".format(
                    'pre-train',
                    kwargs.data,
                    kwargs.dataset.num_classes,
                    kwargs.dataset.num_features,
                    kwargs.checkpoint.seed,
                    kwargs.checkpoint.epochs,
                    kwargs.checkpoint.temperature
                )
        
        # if kwargs.data == 'food101':
        #     val_results = "results/{}/{}_{}_classes_{}_features/raw_results_val_cal_seed-{}_ep-{}_tmp_{}.csv".format(
        #         'pre-train',
        #         kwargs.data,
        #         kwargs.dataset.num_classes,
        #         kwargs.dataset.num_features,
        #         kwargs.checkpoint.seed,
        #         kwargs.checkpoint.epochs,
        #         kwargs.checkpoint.temperature
        #     )    
    
    # Load your data
    df_train_calibration_data = pd.read_csv(cal_results)
    if kwargs.data == 'weather' and kwargs.dataset.shift:
        df_eval_calibration_data = pd.read_csv(test_shift_results)        
        df_in_test = pd.read_csv(test_results)
    else:
        df_eval_calibration_data = pd.read_csv(test_results)
    # if kwargs.data == 'food101':
    #     df_val_calibration_data = pd.read_csv(val_results)    

    # Extract features and labels
    cols = df_train_calibration_data.columns
    # Single pass grouping
    features_cols = [c for c in cols if c.startswith("features")]
    logits_cols   = [c for c in cols if c.startswith("logits")]
    pca_cols      = [c for c in cols if c.startswith("pca")]
    # Extract values
    feats_train_cal  = df_train_calibration_data[features_cols].values
    logits_train_cal = df_train_calibration_data[logits_cols].values
    pca_train_cal    = df_train_calibration_data[pca_cols].values
    #feats_train_cal = df_train_calibration_data.filter(regex=r'^features').values #df_train_calibration_data.drop(columns=["true", "preds"]).values
    #logits_train_cal = df_train_calibration_data.filter(regex=r'^logits').values
    #pca_train_cal = df_train_calibration_data.filter(regex=r'^pca').values
    y_train_cal = df_train_calibration_data["true"].values
    p_train_cal = df_train_calibration_data["preds"].values
    # std_per_column = np.std(feats_train_cal, axis=0)
    # print("Standard deviation per column:")
    # print(std_per_column)
    # print("Mean standard deviation per column:")
    # print(np.mean(std_per_column))
    
    cols = df_eval_calibration_data.columns
    # Single pass grouping
    features_cols = [c for c in cols if c.startswith("features")]
    logits_cols   = [c for c in cols if c.startswith("logits")]
    pca_cols      = [c for c in cols if c.startswith("pca")]
    # Extract values
    feats_eval_cal  = df_eval_calibration_data[features_cols].values
    logits_eval_cal = df_eval_calibration_data[logits_cols].values
    pca_eval_cal    = df_eval_calibration_data[pca_cols].values
    #feats_eval_cal = df_eval_calibration_data.filter(regex=r'^features').values #df_train_calibration_data.drop(columns=["true", "preds"]).values
    #logits_eval_cal = df_eval_calibration_data.filter(regex=r'^logits').values
    #pca_eval_cal = df_eval_calibration_data.filter(regex=r'^pca').values
    y_eval_cal = df_eval_calibration_data["true"].values
    p_eval_cal = df_eval_calibration_data["preds"].values
    
    if kwargs.data == 'weather' and kwargs.dataset.shift:
        cols = df_in_test.columns
        # Single pass grouping
        features_cols = [c for c in cols if c.startswith("features")]
        logits_cols   = [c for c in cols if c.startswith("logits")]
        pca_cols      = [c for c in cols if c.startswith("pca")]
        # Extract values
        feats_in_test  = df_in_test[features_cols].values
        logits_in_test = df_in_test[logits_cols].values
        pca_in_test    = df_in_test[pca_cols].values
        #feats_eval_cal_full = df_in_test.filter(regex=r'^features').values #df_train_calibration_data.drop(columns=["true", "preds"]).values
        #logits_eval_cal_full = df_in_test.filter(regex=r'^logits').values
        #pca_eval_cal_full = df_in_test.filter(regex=r'^pca').values
        y_in_test = df_in_test["true"].values
        p_in_test = df_in_test["preds"].values
    
    # if kwargs.data == 'food101':
    #     cols = df_val_calibration_data.columns
    #     # Single pass grouping
    #     features_cols = [c for c in cols if c.startswith("features")]
    #     logits_cols   = [c for c in cols if c.startswith("logits")]
    #     pca_cols      = [c for c in cols if c.startswith("pca")]
    #     # Extract values
    #     feats_val_cal  = df_val_calibration_data[features_cols].values
    #     logits_val_cal = df_val_calibration_data[logits_cols].values
    #     pca_val_cal    = df_val_calibration_data[pca_cols].values
    #     #feats_eval_cal = df_eval_calibration_data.filter(regex=r'^features').values #df_train_calibration_data.drop(columns=["true", "preds"]).values
    #     #logits_eval_cal = df_eval_calibration_data.filter(regex=r'^logits').values
    #     #pca_eval_cal = df_eval_calibration_data.filter(regex=r'^pca').values
    #     y_val_cal = df_val_calibration_data["true"].values
    #     p_val_cal = df_val_calibration_data["preds"].values
    # else:
    
    # Split into 90% test and 10% val
    if kwargs.data == 'weather' or kwargs.data == 'tissue':                 
        if kwargs.data == 'weather' and kwargs.dataset.shift:
            (feats_in_test, feats_val,
            logits_in_test, logits_val,
            pca_in_test, pca_val,
            y_in_test, y_val,
            p_in_test, p_val) = train_test_split(
                feats_in_test,
                logits_in_test,
                pca_in_test,
                y_in_test,
                p_in_test,
                test_size=0.1, #0.1  # 10% for validation
                random_state=kwargs.seed, # for reproducibility
                shuffle=True)    
            
            feats_test = feats_eval_cal
            logits_test = logits_eval_cal
            pca_test = pca_eval_cal
            y_test = y_eval_cal
            p_test = p_eval_cal     
        else:
            (feats_test, feats_val,
            logits_test, logits_val,
            pca_test, pca_val,
            y_test, y_val,
            p_test, p_val) = train_test_split(
                feats_eval_cal,
                logits_eval_cal,
                pca_eval_cal,
                y_eval_cal,
                p_eval_cal,
                test_size=0.1, #0.1  # 10% for validation
                random_state=kwargs.seed, # for reproducibility
                shuffle=True)  
        
        if kwargs.dataset.subsample < 1.0:
            discard_size = 1 - kwargs.dataset.subsample
            
            (feats_train_cal, _,
            logits_train_cal, _,
            pca_train_cal, _,
            y_train_cal, _,
            p_train_cal, _) = train_test_split(
                feats_train_cal,
                logits_train_cal,
                pca_train_cal,
                y_train_cal,
                p_train_cal,
                test_size=discard_size, #0.1  # 10% for validation
                random_state=kwargs.seed, # for reproducibility
                shuffle=True)                        
                                            
            (feats_val, _,
            logits_val, _,
            pca_val, _,
            y_val, _,
            p_val, _) = train_test_split(
                feats_val,
                logits_val,
                pca_val,
                y_val,
                p_val,
                test_size=discard_size, #0.1  # 10% for validation
                random_state=kwargs.seed, # for reproducibility
                shuffle=True)                              
    else:
        (feats_test, feats_val,
        logits_test, logits_val,
        pca_test, pca_val,
        y_test, y_val,
        p_test, p_val) = train_test_split(
            feats_eval_cal,
            logits_eval_cal,
            pca_eval_cal,
            y_eval_cal,
            p_eval_cal,
            test_size=0.1, #0.1  # 10% for validation
            random_state=kwargs.seed, # for reproducibility
            shuffle=True)        
    
    print(f'Learn Calibration shape: {feats_train_cal.shape}, Validation shape: {feats_val.shape}, Test Calibration shape: {feats_test.shape}')
    # Convert to PyTorch tensors
    feats_train_cal = torch.tensor(feats_train_cal, dtype=torch.float32)
    logits_train_cal = torch.tensor(logits_train_cal, dtype=torch.float32)
    pca_train_cal = torch.tensor(pca_train_cal, dtype=torch.float32)
    y_train_cal = torch.tensor(y_train_cal, dtype=torch.long)
    p_train_cal = torch.tensor(p_train_cal, dtype=torch.long)

    feats_test_cal = torch.tensor(feats_test, dtype=torch.float32)
    logits_test_cal = torch.tensor(logits_test, dtype=torch.float32)
    pca_test_cal = torch.tensor(pca_test, dtype=torch.float32)
    y_test_cal = torch.tensor(y_test, dtype=torch.long)
    p_test_cal = torch.tensor(p_test, dtype=torch.long)
    
    feats_val_cal = torch.tensor(feats_val, dtype=torch.float32)
    logits_val_cal = torch.tensor(logits_val, dtype=torch.float32)
    pca_val_cal = torch.tensor(pca_val, dtype=torch.float32)
    y_val_cal = torch.tensor(y_val, dtype=torch.long)
    p_val_cal = torch.tensor(p_val, dtype=torch.long)

    # Create datasets
    train_cal_set = CalibrationDatasetv2(feats_train_cal, logits_train_cal, pca_train_cal, y_train_cal,p_train_cal, num_classes=kwargs.dataset.num_classes)
    test_cal_set = CalibrationDatasetv2(feats_test_cal, logits_test_cal, pca_test_cal, y_test_cal, p_test_cal, num_classes=kwargs.dataset.num_classes)
    val_cal_set = CalibrationDatasetv2(feats_val_cal, logits_val_cal, pca_val_cal, y_val_cal, p_val_cal, num_classes=kwargs.dataset.num_classes)
    
    # Create data loaders
    data_train_cal_loader = DataLoader(
        train_cal_set, batch_size=kwargs.dataset.batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    data_test_cal_loader = DataLoader(
        test_cal_set, batch_size=kwargs.dataset.batch_size, shuffle=False, num_workers=8, pin_memory=True
    )
    data_val_cal_loader = DataLoader(
        val_cal_set, batch_size=kwargs.dataset.batch_size, shuffle=False, num_workers=8, pin_memory=True
    )

    return data_train_cal_loader, data_test_cal_loader, data_val_cal_loader 

class CalibrationDataset(Dataset):
    def __init__(self, X, y, p, num_classes, transforms_fn=None):
        self.num_classes = num_classes
        self.transforms_fn = transforms_fn
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)  # one-hot encoded
        self.p = torch.tensor(p, dtype=torch.long)  # one-hot encoded
        self.y_onehot = F.one_hot(self.y, num_classes=self.num_classes).float()
        self.p_onehot = F.one_hot(self.p, num_classes=self.num_classes).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.transforms_fn is not None:
            x = self.transforms_fn(self.X[idx])
        else:
            x = self.X[idx]
        y_onehot = self.y_onehot[idx]
        p_onehot = self.p_onehot[idx]
        p = self.p[idx]
        return x, y_onehot, p, p_onehot
    
class CalibrationDatasetv2(Dataset):
    def __init__(self, feats, logits, pca, y, p, num_classes, transforms_fn=None):
        self.num_classes = num_classes
        self.transforms_fn = transforms_fn
        self.feats = torch.tensor(feats, dtype=torch.float32)
        self.logits = torch.tensor(logits, dtype=torch.float32)
        self.pca = torch.tensor(pca, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)  # one-hot encoded
        self.p = torch.tensor(p, dtype=torch.long)  # one-hot encoded
        self.y_onehot = F.one_hot(self.y, num_classes=self.num_classes).float()
        self.p_onehot = F.one_hot(self.p, num_classes=self.num_classes).float()        

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        if self.transforms_fn is not None:
            feats = self.transforms_fn(self.feats[idx])
        else:
            feats = self.feats[idx]        
        logits = self.logits[idx]
        pca = self.pca[idx]
        y_onehot = self.y_onehot[idx]
        p_onehot = self.p_onehot[idx]
        p = self.p[idx]
        return feats, logits, pca, y_onehot, p, p_onehot


class ClassificationDataset(Dataset):
    def __init__(self, X, y, transforms_fn=None, as_array=False):
        if as_array:
            self.X = X #torch.tensor(X, dtype=torch.float32)        
        else:
            self.X = torch.tensor(X, dtype=torch.float32)        
        self.y = torch.tensor(y, dtype=torch.long)  # one-hot encoded
        self.transforms_fn = transforms_fn
        self.as_array = as_array

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]   
        if self.as_array:                         
            x = Image.fromarray((x * 255).astype(np.uint8)) if x.max() <= 1.0 else Image.fromarray(x.astype(np.uint8))

        if self.transforms_fn is not None:
            x = self.transforms_fn(x)
            
        y = self.y[idx]
        return x, y

class WeatherClassificationDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)  
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)         
        self.y = torch.tensor(y, dtype=torch.long)  # one-hot encoded                

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_num = self.X_num[idx]           
        x_cat = self.X_cat[idx]           
        y = self.y[idx]
        return (x_cat, x_num), y 

class SynthData(Dataset):
    def __init__(self, kwargs, experiment=None, name='synthetic'):        
        self.dataname = name
        if experiment == 'pre-train':        
            self.generatePretrainingSynthData(num_features = kwargs.num_features,
                                    num_classes = kwargs.num_classes,
                                    n_samples = kwargs.num_samples,
                                    n_redundant = kwargs.n_redundant,
                                    n_clusters_per_class = kwargs.n_clusters_per_class,
                                    class_sep = kwargs.class_sep,
                                    flip_y = kwargs.flip_y,
                                    batch_size = kwargs.batch_size,
                                    random_state = kwargs.random_state)      
        elif experiment == 'calibrate' or experiment == 'competition':
            self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generateCalibrationData(kwargs, dataname=self.dataname) 
            print("Loading synthetic data for calibration complete")   

    def generatePretrainingSynthData(self, num_features,
                                num_classes,
                                n_samples,
                                n_redundant,
                                n_clusters_per_class,
                                class_sep,
                                flip_y,
                                batch_size,
                                random_state):        
        
        X, y = make_classification(
        n_samples=n_samples, #240000
        n_features=num_features,        # 2D input
        n_informative=num_features,     # both features informative
        n_redundant=n_redundant, #0
        n_clusters_per_class=n_clusters_per_class, #1
        n_classes=num_classes,
        class_sep=class_sep, #0.5
        flip_y=flip_y, #0.1
        random_state=random_state #42
    )

        X = StandardScaler().fit_transform(X)  # normalize input

        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=random_state)
        X_train, X_train_cal, y_train, y_train_cal = train_test_split(X, y, test_size=0.5, random_state=random_state)
        #X_cal, X_test, y_cal, y_test = train_test_split(X_test, y_test, test_size=0.1668, random_state=random_state)        
        X_eval_cal, X_train_cal, y_eval_cal, y_train_cal = train_test_split(X_train_cal, y_train_cal, test_size=0.1668, random_state=random_state)        
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1668, random_state=random_state)
        #y_train_oh = F.one_hot(y_train, num_classes=num_classes) #to_categorical(y_train, num_classes)
      
        print(f'Train shape: {X_train.shape}, Learn Calibration shape: {X_train_cal.shape}, Validation shape: {X_val.shape}, Eval Calibration shape: {X_eval_cal.shape}')
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long) 
        X_train_cal = torch.tensor(X_train_cal, dtype=torch.float32)
        y_train_cal = torch.tensor(y_train_cal, dtype=torch.long) 
        X_eval_cal = torch.tensor(X_eval_cal, dtype=torch.float32)
        y_eval_cal = torch.tensor(y_eval_cal, dtype=torch.long) 
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long) 
        
        train_set = ClassificationDataset(X_train, y_train)
        eval_cal_set   = ClassificationDataset(X_eval_cal, y_eval_cal)
        val_set   = ClassificationDataset(X_val, y_val)
        train_cal_set  = ClassificationDataset(X_train_cal, y_train_cal)       

        self.data_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,)
        self.data_eval_cal_loader   = DataLoader(eval_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_train_cal_loader  = DataLoader(train_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        
        print("Loading synthetic data for pre-training complete")            
            
class WeatherData(Dataset):
    def __init__(self, kwargs, experiment=None, name='wheather', seed=42):        
        self.dataname = name
        if experiment == 'pre-train':   
            self.generatePretrainingWheatherData(batch_size=kwargs.batch_size, seed=seed, exp_name=experiment)      
        elif experiment == 'xg_debug':
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_train_cal, self.y_train_cal, self.X_eval_cal, self.y_eval_cal, self.X_eval_cal_shift, self.y_eval_cal_shift = self.generatePretrainingWheatherData(batch_size=kwargs.batch_size, seed=seed, exp_name=experiment)      
        else:
            if kwargs.calibrator_version == 'v2':
                self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generateCalibrationDatav2(kwargs)
            else:
                self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generateCalibrationData(kwargs) 

    def generatePretrainingWheatherData(self, seed, batch_size, exp_name='pre-train'):        
        
        scaler = StandardScaler()
        ############  FOR ALL CONTINUOUS ############ 
        self.category_counts = ()        
        
        ######## TRAINING SET ########
        weather = pd.read_csv(f'./data/WEATHER/shifts_canonical_train_seed_{seed}.csv')  # 60k obs        
        train_df, val_df = train_test_split(weather, test_size=0.1, stratify=weather["fact_cwsm_class"], random_state=seed)                
        
        X = train_df.iloc[:,6:] #weather.drop(columns=['id', 'target'])  
        feature_cols = X.columns.tolist()    
        self.numerical_features = feature_cols  
        X_train = scaler.fit_transform(X[feature_cols])                            
        # self.categorical_features, self.numerical_features, _ = find_categorical_columns(X, target_col='fact_cwsm_class',
        #                                                                                            threshold_for_categorical=20)                    
        # X_num_raw = X.drop(columns=self.categorical_features) #+['fact_cwsm_class'])
        # X_cat_raw = X.drop(columns=self.numerical_features) #+['fact_cwsm_class'])
        
        # cat_maps = self.fit_category_maps(X_cat_raw, self.categorical_features)
        label_map = self.fit_label_map(train_df["fact_cwsm_class"])
        
        # self.cat_levels = {}
        # for col in self.categorical_features:
        #     self.cat_levels[col] = pd.Categorical(X_cat[col]).categories
        # for col in X_cat.columns:
        #     try:
        #         X_cat[col] = X_cat[col].astype(int)
        #     except (ValueError, TypeError):
        #         pass
        # X_cat = X_cat.astype('category').apply(lambda col: col.cat.codes)
        # weather['target_int'] = weather['fact_cwsm_class'].astype(int).astype('category').cat.codes
        # y_train = weather['target_int'].values #- 1 # from 0 to 8 not from 1 to 9!!!!                 
        
        # X_num = scaler.fit_transform(X_num_raw)  # normalize input   
        # X_cat, self.category_counts = self.transform_categories(X_cat_raw, self.categorical_features, cat_maps, use_unknown=True)
        y_train = self.transform_labels(train_df["fact_cwsm_class"], label_map)      
        self.class_counts = np.bincount(y_train, minlength=5)   
        # X_train = np.concatenate([X_num, X_cat], axis=1)
        
        ######## VALIDATION SET ########
        X_val_raw = val_df.iloc[:, 6:].copy()
        X_val = scaler.transform(X_val_raw[feature_cols])

        # X_val_num = scaler.transform(X_val_raw[self.numerical_features].copy())
        # X_val_cat, _ = self.transform_categories(
        #     X_val_raw[self.categorical_features].copy(),
        #     self.categorical_features,
        #     cat_maps,
        #     use_unknown=True
        # )
        y_val = self.transform_labels(val_df["fact_cwsm_class"], label_map)

        # X_val = np.concatenate([X_val_num, X_val_cat], axis=1)
        
        ######## TEST SET (IN-DISTRIBUTION) ########        
        eval_in_df = pd.read_csv(f'./data/WEATHER/shifts_canonical_eval_in_seed_{seed}.csv')  # 60k obs   
        
        X = eval_in_df.iloc[:,6:] #weather.drop(columns=['id', 'target'])     
        X_eval_cal = scaler.transform(X[feature_cols])      

        # X_num = scaler.transform(X[self.numerical_features].copy())
        # X_cat, _ = self.transform_categories(
        #     X[self.categorical_features].copy(),
        #     self.categorical_features,
        #     cat_maps,
        #     use_unknown=True
        # )
        y_eval_cal = self.transform_labels(eval_in_df["fact_cwsm_class"], label_map)

        # X_eval_cal = np.concatenate([X_num, X_cat], axis=1)                                     
        
        # X_eval_cal = weather.iloc[:,6:] #weather.drop(columns=['id', 'target'])               
        # X_num = X_eval_cal.drop(columns=self.categorical_features) #+['fact_cwsm_class'])
        # X_cat = X_eval_cal.drop(columns=self.numerical_features) #+['fact_cwsm_class'])
        # for col in X_cat.columns:
        #     try:
        #         X_cat[col] = X_cat[col].astype(int)
        #     except (ValueError, TypeError):
        #         pass
        # X_cat = X_cat.astype('category').apply(lambda col: col.cat.codes)
        # weather['target_int'] = weather['fact_cwsm_class'].astype(int).astype('category').cat.codes
        # y_eval_cal = weather['target_int'].values #- 1 # from 0 to 8 not from 1 to 9!!!!            
        
        # X_eval_cal = scaler.transform(X_num)  # normalize input   
        # X_eval_cal = np.concatenate([X_eval_cal, X_cat.values], axis=1)
        
        ######## TEST SET (SHIFT) ########        
        eval_out_df = pd.read_csv(f'./data/WEATHER/shifts_canonical_eval_out_seed_{seed}.csv')  # 60k obs     
        
        X = eval_out_df.iloc[:,6:] #weather.drop(columns=['id', 'target'])        
        X_eval_cal_shift = scaler.transform(X[feature_cols])      

        # X_num = scaler.transform(X[self.numerical_features].copy())
        # X_cat, _ = self.transform_categories(
        #     X[self.categorical_features].copy(),
        #     self.categorical_features,
        #     cat_maps,
        #     use_unknown=True
        # )
        y_eval_cal_shift = self.transform_labels(eval_out_df["fact_cwsm_class"], label_map)

        # X_eval_cal_shift = np.concatenate([X_num, X_cat], axis=1)      
        
        # X_eval_cal_shift = weather.iloc[:,6:] #weather.drop(columns=['id', 'target'])               
        # X_eval_cal_shift = filter_unseen_rows(X_eval_cal_shift, self.categorical_features, self.cat_levels)
        # weather = weather.loc[X_eval_cal_shift.index].copy()
        # X_num = X_eval_cal_shift[self.numerical_features].copy()
        # X_cat = X_eval_cal_shift[self.categorical_features].copy()
              
        # for col in X_cat.columns:
        #     try:
        #         X_cat[col] = pd.Categorical(X_cat[col], categories=self.cat_levels[col]).codes # X_cat[col] = X_cat[col].astype(int)
        #     except (ValueError, TypeError):
        #         pass
        # X_cat = X_cat.astype('category').apply(lambda col: col.cat.codes)
        # weather['target_int'] = weather['fact_cwsm_class'].astype(int).astype('category').cat.codes
        # y_eval_cal_shift = weather['target_int'].values #- 1 # from 0 to 8 not from 1 to 9!!!!        
        
        # X_eval_cal_shift = scaler.transform(X_num)  # normalize input   
        # X_eval_cal_shift = np.concatenate([X_eval_cal_shift, X_cat.values], axis=1)
        
        ######## CALIBRATION SET ########
        cal_df = pd.read_csv(f'./data/WEATHER/shifts_canonical_dev_in_processed.csv')  # 60k obs                    
        
        X = cal_df.iloc[:,6:] #weather.drop(columns=['id', 'target'])        
        X_train_cal = scaler.transform(X[feature_cols])      

        # X_num = scaler.transform(X[self.numerical_features].copy())
        # X_cat, _ = self.transform_categories(
        #     X[self.categorical_features].copy(),
        #     self.categorical_features,
        #     cat_maps,
        #     use_unknown=True
        # )
        y_train_cal = self.transform_labels(cal_df["fact_cwsm_class"], label_map)

        # X_train_cal = np.concatenate([X_num, X_cat], axis=1)                            
        
        # X_train_cal = weather.iloc[:,6:] #weather.drop(columns=['id', 'target'])                       
        # X_num = X_train_cal.drop(columns=self.categorical_features) #+['fact_cwsm_class'])
        # X_cat = X_train_cal.drop(columns=self.numerical_features) #+['fact_cwsm_class'])              
        # for col in X_cat.columns:
        #     try:
        #         X_cat[col] = X_cat[col].astype(int)
        #     except (ValueError, TypeError):
        #         pass
        # X_cat = X_cat.astype('category').apply(lambda col: col.cat.codes)
        # weather['target_int'] = weather['fact_cwsm_class'].astype(int).astype('category').cat.codes
        # y_train_cal = weather['target_int'].values #- 1 # from 0 to 8 not from 1 to 9!!!!        
        
        # X_train_cal = scaler.transform(X_num)  # normalize input               
        # X_train_cal = np.concatenate([X_train_cal, X_cat.values], axis=1)

        ######## VALIDATION SET (PRE-TRAINING) ########
        # X_train, X_train_cal, y_train, y_train_cal = train_test_split(X, y, test_size=0.5, random_state=random_state)
        # X_eval_cal, X_train_cal, y_eval_cal, y_train_cal = train_test_split(X_train_cal, y_train_cal, test_size=0.16, random_state=random_state)        
        # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=seed, stratify=y_train)   
        print(f'Train shape: {X_train.shape}, Learn Calibration shape: {X_train_cal.shape}, Validation shape: {X_val.shape}, Eval Calibration shape: {X_eval_cal.shape}')
                
        # X_train = torch.tensor(X_train, dtype=torch.float32)
        # y_train = torch.tensor(y_train, dtype=torch.long) 
        # X_train_cal = torch.tensor(X_train_cal, dtype=torch.float32)
        # y_train_cal = torch.tensor(y_train_cal, dtype=torch.long) 
        # X_eval_cal = torch.tensor(X_eval_cal, dtype=torch.float32)
        # y_eval_cal = torch.tensor(y_eval_cal, dtype=torch.long) 
        # X_val = torch.tensor(X_val, dtype=torch.float32)
        # y_val = torch.tensor(y_val, dtype=torch.long) 
        
        # train_set = WeatherClassificationDataset(X_train, y_train)
        # eval_cal_set   = WeatherClassificationDataset(X_eval_cal, y_eval_cal)
        # val_set   = WeatherClassificationDataset(X_val, y_val)
        # train_cal_set  = WeatherClassificationDataset(X_train_cal, y_train_cal)          

        # self.data_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,)
        # self.data_eval_cal_loader   = DataLoader(eval_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        # self.data_val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        # self.data_train_cal_loader  = DataLoader(train_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        n_num = len(self.numerical_features)

        X_train_num = X_train[:, :n_num]
        X_train_cat = X_train[:, n_num:]

        X_val_num = X_val[:, :n_num]
        X_val_cat = X_val[:, n_num:]

        X_train_cal_num = X_train_cal[:, :n_num]
        X_train_cal_cat = X_train_cal[:, n_num:]

        X_eval_cal_num = X_eval_cal[:, :n_num]
        X_eval_cal_cat = X_eval_cal[:, n_num:]
        
        X_eval_cal_num_shift = X_eval_cal_shift[:, :n_num]
        X_eval_cal_cat_shift = X_eval_cal_shift[:, n_num:]
        
        # for split_name, X_cat_arr in [
        #     ("train", X_train_cat),
        #     ("val", X_val_cat),
        #     ("train_cal", X_train_cal_cat),
        #     ("eval", X_eval_cal_cat),
        #     ("eval_shift", X_eval_cal_cat_shift),
        # ]:
        #     print(f"\n{split_name}")
        #     for i in range(X_cat_arr.shape[1]):
        #         print(
        #             i,
        #             "min=", X_cat_arr[:, i].min(),
        #             "max=", X_cat_arr[:, i].max(),
        #             "allowed_max=", self.category_counts[i] - 1
        #         )
            
        # X_train_num = X_train[:, :-len(self.numerical_features)]
        # X_train_cat = X_train[:, -len(self.numerical_features):]
        # X_val_num = X_val[:, :-len(self.numerical_features)]
        # X_val_cat = X_val[:, -len(self.numerical_features):]
        # X_train_cal_num = X_train_cal[:, :-len(self.numerical_features)]
        # X_train_cal_cat = X_train_cal[:, -len(self.numerical_features):]
        # X_eval_cal_num = X_eval_cal[:, :-len(self.numerical_features)]
        # X_eval_cal_cat = X_eval_cal[:, -len(self.numerical_features):]
                  
        X_train_num = torch.tensor(X_train_num, dtype=torch.float32)
        X_train_cat = torch.tensor(X_train_cat, dtype=torch.long)
        y_train = torch.tensor(y_train, dtype=torch.long) 
        
        X_train_cal_num = torch.tensor(X_train_cal_num, dtype=torch.float32)
        X_train_cal_cat = torch.tensor(X_train_cal_cat, dtype=torch.long)
        y_train_cal = torch.tensor(y_train_cal, dtype=torch.long) 
        
        X_eval_cal_num = torch.tensor(X_eval_cal_num, dtype=torch.float32)
        X_eval_cal_cat = torch.tensor(X_eval_cal_cat, dtype=torch.long)
        y_eval_cal = torch.tensor(y_eval_cal, dtype=torch.long)         
        
        X_eval_cal_num_shift = torch.tensor(X_eval_cal_num_shift, dtype=torch.float32)
        X_eval_cal_cat_shift = torch.tensor(X_eval_cal_cat_shift, dtype=torch.long)
        y_eval_cal_shift = torch.tensor(y_eval_cal_shift, dtype=torch.long) 
        
        X_val_num = torch.tensor(X_val_num, dtype=torch.float32)
        X_val_cat = torch.tensor(X_val_cat, dtype=torch.long)
        y_val = torch.tensor(y_val, dtype=torch.long) 
        
        train_set = WeatherClassificationDataset(X_train_num, X_train_cat, y_train)
        eval_cal_set   = WeatherClassificationDataset(X_eval_cal_num, X_eval_cal_cat, y_eval_cal)
        eval_cal_set_shift   = WeatherClassificationDataset(X_eval_cal_num_shift, X_eval_cal_cat_shift, y_eval_cal_shift)
        val_set   = WeatherClassificationDataset(X_val_num, X_val_cat, y_val)
        train_cal_set  = WeatherClassificationDataset(X_train_cal_num, X_train_cal_cat, y_train_cal)             

        self.data_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,)
        self.data_eval_cal_loader   = DataLoader(eval_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_eval_cal_shift_loader   = DataLoader(eval_cal_set_shift, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_train_cal_loader  = DataLoader(train_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)            
        
        print("Loading weather data for pre-training complete")       
        
        if exp_name == 'xg_debug':
            return X_train, y_train, X_val, y_val, X_train_cal, y_train_cal, X_eval_cal, y_eval_cal, X_eval_cal_shift, y_eval_cal_shift        
        
    def fit_category_maps(self, df, categorical_features):
        cat_maps = {}
        for col in categorical_features:
            cat_maps[col] = pd.Index(pd.Categorical(df[col]).categories)
        return cat_maps

    def transform_categories(self, df, categorical_features, cat_maps, use_unknown=True):
        cols = []
        category_counts = []

        for col in categorical_features:
            categories = cat_maps[col]
            codes = pd.Categorical(df[col], categories=categories).codes

            if use_unknown:
                unknown_idx = len(categories)
                codes = np.where(codes < 0, unknown_idx, codes)
                category_counts.append(len(categories) + 1)
            else:
                if (codes < 0).any():
                    raise ValueError(f"Unseen category in {col}")
                category_counts.append(len(categories))

            cols.append(codes.reshape(-1, 1))

        X_cat = np.concatenate(cols, axis=1) if cols else np.empty((len(df), 0), dtype=np.int64)
        return X_cat, category_counts

    def fit_label_map(self, y):
        return pd.Index(pd.Categorical(y.astype(int)).categories)

    def transform_labels(self, y, label_categories):
        codes = pd.Categorical(y.astype(int), categories=label_categories).codes
        if (codes < 0).any():
            raise ValueError("Unseen label found.")
        return codes       
        
def print_class_frequencies(dataset):  
    class_freqs = []  
    labels = np.array(dataset.labels).squeeze()   # dataset.labels is usually (N,1)
    counts = Counter(labels)
    total = len(labels)
    print("Class frequencies:")
    for cls, count in sorted(counts.items()):
        print(f"  Class {cls}: {count} ({count/total:.2%})")
        class_freqs.append(count/total)
    return class_freqs


class MedMnistData(Dataset):    
    def __init__(self, kwargs, experiment=None, name='path', model_class=None):
        self.name = name
        self.model_class = model_class
        if experiment == 'pre-train':                     
            self.generatePretrainingMedMnistData(size=kwargs.size,
                                batch_size = kwargs.batch_size,
                                random_state = kwargs.random_state)    
            kwargs.class_freqs = self.class_freqs  
        elif experiment == 'calibrate' or experiment == 'competition' or experiment == 'quantize'  or experiment == 'replicate':
                # kwargs.dataset.class_freqs = [0.321, 0.047, 0.035, 0.093, 0.071, 0.047, 0.237, 0.149]
                if kwargs.calibrator_version == 'v2':
                    self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generateCalibrationDatav2(kwargs)
                else:
                    self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generateCalibrationData(kwargs) 
        print("Loading synthetic data for calibration complete")   
        
    def generatePretrainingMedMnistData(self, size,
                                batch_size,
                                random_state):
        if self.model_class == 'convnext':
            # Use ImageNet normalization for ConvNeXt, as it was pre-trained on ImageNet
            l_transform = [ #transforms.Compose(
                transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                        ),
            ]
        else:
            l_transform = [transforms.Grayscale(num_output_channels=3), transforms.ToTensor()]
        #if self.name == 'tissue':
        #    l_transforms.append()  # Convert to 3-channel RGB)
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(10),
            #transforms.Resize((224, 224)) if size < 224 else transforms.Lambda(lambda x: x),            
            #transforms.ToTensor(), 
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                    std=[0.229, 0.224, 0.225])
        #]

        # Remove any None or identity transforms
        #l_transforms = [t for t in l_transforms if not isinstance(t, transforms.Lambda)]
        
        # l_transform = transforms.Compose([ 
        #         transforms.Grayscale(num_output_channels=3),                            
        #         transforms.Resize((224, 224)),
        #         transforms.RandomCrop(32, padding=4),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        #         ),
        # ])

        trans = transforms.Compose(l_transform)


        print(f"Dataset source information : MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")
        info = INFO['tissuemnist']
        print(info['description'])
        os.makedirs(f"./data/{self.name.upper()}/data_train", exist_ok=True)
        os.makedirs(f"./data/{self.name.upper()}/data_val", exist_ok=True)
        os.makedirs(f"./data/{self.name.upper()}/data_test", exist_ok=True)
        generator = torch.Generator().manual_seed(random_state)
        
        if self.name == 'tissue':
            train_set = TissueMNIST(root="./data/TISSUE/data_train", split="train", transform=trans, download=True, size=size) #, as_rgb=self.as_rgb) #165k        
            self.class_freqs = print_class_frequencies(train_set)
            train_size = len(train_set)
            half_size = train_size // 2                    
            train_set, eval_cal_set = random_split(train_set, [half_size, train_size - half_size], generator=generator) #82.5k BOTH  
            val_set = TissueMNIST(root="./data/TISSUE/data_val", split="val", transform=trans, download=True, size=size) #, as_rgb=self.as_rgb) #20k
            train_cal_set = TissueMNIST(root="./data/TISSUE/data_test", split="test", transform=trans, download=True, size=size) #, as_rgb=self.as_rgb) #40k
        elif self.name == 'path':
            train_set = PathMNIST(root="./data/PATH/data_train", split="train", transform=trans, download=True, size=size) #, as_rgb=self.as_rgb) #89kk        
            train_size = len(train_set)
            half_size = train_size // 2                    
            train_set, eval_cal_set = random_split(train_set, [half_size, train_size - half_size], generator=generator) #45k BOTH  
            val_set = PathMNIST(root="./data/PATH/data_test", split="test", transform=trans, download=True, size=size) #, as_rgb=self.as_rgb) #7k
            train_cal_set = PathMNIST(root="./data/PATH/data_val", split="val", transform=trans, download=True, size=size) #, as_rgb=self.as_rgb) #10k
                
        print(f'Train shape: {len(train_set)}, Learn Calibration shape: {len(train_cal_set)}, Validation shape: {len(val_set)}, Eval Calibration shape: {len(eval_cal_set)}')

        self.data_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,)
        self.data_eval_cal_loader = DataLoader(eval_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_train_cal_loader = DataLoader(train_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        
        print(f"Loading {self.name} mnist data for pre-training complete.")         


class Cifar10Data(Dataset):    
    def __init__(self, kwargs, experiment=None, name='cifar10', model_class=None):
        self.name = name
        self.model_class = model_class
        if experiment == 'pre-train':  
            kwargs.class_freqs = [1/kwargs.num_classes]*kwargs.num_classes                        
            self.generatePretrainingCifar10Data(
                                    batch_size = kwargs.batch_size,
                                    random_state = kwargs.random_state)      
        elif experiment == 'calibrate' or experiment == 'competition' or experiment == 'quantize' or experiment == 'replicate':
            kwargs.dataset.class_freqs = [1/kwargs.dataset.num_classes]*kwargs.dataset.num_classes
            if kwargs.calibrator_version == 'v2':
                self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generateCalibrationDatav2(kwargs)
            else:
                self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generateCalibrationData(kwargs) 
        #print("Loading synthetic data for calibration complete")   

    def generatePretrainingCifar10Data(self, 
                                batch_size,
                                random_state):  
        data_dir =  f"./data/{self.name.upper()}"                     
        os.makedirs(data_dir, exist_ok=True)
        generator = torch.Generator().manual_seed(random_state)
        if self.model_class == 'convnext':
            # Use ImageNet normalization for ConvNeXt, as it was pre-trained on ImageNet
            l_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                        ),
            ])
        else:
            l_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        
        # l_transform = transforms.Compose([ 
        #         transforms.Resize((224, 224)),
        #         transforms.RandomCrop(32, padding=4),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        #         ),
        # ])
        full_train = datasets.CIFAR10(root=data_dir, train=True, 
                                        transform=l_transform, download=True)
        
        train_size = int(0.45 * len(full_train))
        val_size = int(0.1 * len(full_train))
        eval_cal_size = len(full_train) - train_size - val_size
        train_set, val_set, eval_cal_set = random_split(full_train, [train_size, val_size, eval_cal_size], generator=generator)

        train_cal_set = datasets.CIFAR10(root=data_dir, train=False, 
                                        transform=l_transform, download=True)
        
        print(f'Train shape: {len(train_set)}, Learn Calibration shape: {len(train_cal_set)}, Validation shape: {len(val_set)}, Eval Calibration shape: {len(eval_cal_set)}')

        self.data_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,)
        self.data_eval_cal_loader   = DataLoader(eval_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_train_cal_loader  = DataLoader(train_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        
        print("Loading CIFAR10 data for pre-training complete") 

class Cifar100Data(Dataset):    
    def __init__(self, kwargs, experiment=None, name='cifar100', model_class=None):
        self.name = name
        self.model_class = model_class
        if experiment == 'pre-train':    
            kwargs.class_freqs = [1/kwargs.num_classes]*kwargs.num_classes
            self.generatePretrainingCifar100Data(
                                    batch_size = kwargs.batch_size,
                                    random_state = kwargs.random_state)      
        elif experiment == 'calibrate' or experiment == 'competition' or experiment == 'quantize'  or experiment == 'replicate':
            kwargs.dataset.class_freqs = [1/kwargs.dataset.num_classes]*kwargs.dataset.num_classes
            if kwargs.calibrator_version == 'v2':
                self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generateCalibrationDatav2(kwargs)
            else:
                self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generateCalibrationData(kwargs) 
        print("Loading synthetic data for calibration complete")   

    def generatePretrainingCifar100Data(self, 
                                batch_size,
                                random_state):  
        data_dir =  f"./data/{self.name.upper()}"                     
        os.makedirs(data_dir, exist_ok=True)
        generator = torch.Generator().manual_seed(random_state)
        
        l_transform = transforms.Compose([
             transforms.Resize((224, 224)),
             transforms.ToTensor(),
             transforms.Normalize(                        # use ImageNet mean/std for ViT
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])
        # l_transform = transforms.Compose([ 
        #         transforms.Resize((224, 224)),                          
        #         transforms.RandomCrop(32, padding=4),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        #         ),
        # ])
        # train_transform = transforms.Compose([
        #     transforms.Resize((224, 224)),               # match ViT input size
        #     transforms.RandomHorizontalFlip(p=0.5),      # common for CIFAR
        #     transforms.RandomCrop(224, padding=4),       # adds spatial jitter
        #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),  # brightness, contrast, saturation, hue
        #     transforms.RandomRotation(15),               # small rotations
        #     transforms.RandomGrayscale(p=0.1),           # sometimes grayscale
        #     transforms.ToTensor(), 
        #     transforms.Normalize(                        # use ImageNet mean/std for ViT
        #         mean=[0.485, 0.456, 0.406], 
        #         std=[0.229, 0.224, 0.225]
        #     ),                       
        # ])
        # val_transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(), 
        #     transforms.Normalize(                        # use ImageNet mean/std for ViT
        #         mean=[0.485, 0.456, 0.406], 
        #         std=[0.229, 0.224, 0.225]
        #     ),            
        # ])
        
        full_train = datasets.CIFAR100(root=data_dir, train=True, 
                                        transform=l_transform, download=True)
        
        train_size = int(0.45 * len(full_train))
        val_size = int(0.1 * len(full_train))
        eval_cal_size = len(full_train) - train_size - val_size
        train_set, val_set, eval_cal_set = random_split(full_train, [train_size, val_size, eval_cal_size], generator=generator)

        train_cal_set = datasets.CIFAR100(root=data_dir, train=False, 
                                        transform=l_transform, download=True)
        
        print(f'Train shape: {len(train_set)}, Learn Calibration shape: {len(train_cal_set)}, Validation shape: {len(val_set)}, Eval Calibration shape: {len(eval_cal_set)}')

        self.data_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        self.data_eval_cal_loader   = DataLoader(eval_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_train_cal_loader  = DataLoader(train_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        
        print("Loading CIFAR100 data for pre-training complete") 

def find_categorical_columns(train_df, target_col='fact_cwsm_class', threshold_for_categorical=20):             

    # obvious types
    categorical_cols = train_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    numerical_cols = train_df.select_dtypes(include=['number']).columns.tolist()

    # remove target
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)

    # heuristic: numeric columns with few unique values are likely categorical
    numeric_as_categorical = []
    for col in numerical_cols:
        n_unique = train_df[col].nunique(dropna=True)
        if n_unique <= threshold_for_categorical:
            numeric_as_categorical.append(col)

    # move them
    for col in numeric_as_categorical:
        numerical_cols.remove(col)
        categorical_cols.append(col)

    # optional: convert categorical columns to integer codes
    for col in categorical_cols:
        train_df[col] = train_df[col].astype('category')

    category_counts = [train_df[col].nunique() for col in categorical_cols]    

    # print("Categorical columns:", categorical_cols, '\n')
    # print("Numerical columns:", numerical_features)
    # print("Category counts:", category_counts)
    
    return categorical_cols, numerical_cols, category_counts 

def filter_unseen_rows(df, categorical_features, cat_levels):
    mask = pd.Series(True, index=df.index)

    for col in categorical_features:
        allowed = cat_levels[col]
        col_mask = df[col].isin(allowed)

        # if you want to keep NaNs too, use:
        # col_mask = df[col].isin(allowed) | df[col].isna()

        mask &= col_mask

    dropped = (~mask).sum()
    print(f"Dropping {dropped} rows with unseen categorical values out of {len(df)}")

    return df.loc[mask].copy()
