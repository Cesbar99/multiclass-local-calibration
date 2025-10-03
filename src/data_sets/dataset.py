import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from utils.utils import *
from scipy.io import loadmat
import medmnist
from medmnist import TissueMNIST, INFO
from torch.utils.data import random_split
import os
from tiny_imagenet_torch import TinyImageNet
from datasets import load_dataset
from PIL import Image
import io

def generateCalibrationDatav2(kwargs, dataname=None):         
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
    cols = df_train_calibration_data.columns
    # Single pass grouping
    features_cols = [c for c in cols if c.startswith("features")]
    logits_cols   = [c for c in cols if c.startswith("logits")]
    pca_cols      = [c for c in cols if c.startswith("pca")]
    # Extract values
    feats_train_cal  = df_train_calibration_data[features_cols].values
    logits_train_cal = df_train_calibration_data[logits_cols].values
    pca_train_cal    = df_train_calibration_data[pca_cols].values
    y_train_cal = df_train_calibration_data["true"].values
    p_train_cal = df_train_calibration_data["preds"].values
    
    cols = df_eval_calibration_data.columns
    # Single pass grouping
    features_cols = [c for c in cols if c.startswith("features")]
    logits_cols   = [c for c in cols if c.startswith("logits")]
    pca_cols      = [c for c in cols if c.startswith("pca")]
    # Extract values
    feats_eval_cal  = df_eval_calibration_data[features_cols].values
    logits_eval_cal = df_eval_calibration_data[logits_cols].values
    pca_eval_cal    = df_eval_calibration_data[pca_cols].values
    
    y_eval_cal = df_eval_calibration_data["true"].values
    p_eval_cal = df_eval_calibration_data["preds"].values

    # Split into 90% test and 10% val
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
        test_size=0.1,   
        random_state=kwargs.seed, 
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
        
class MedMnistData(Dataset):    
    def __init__(self, kwargs, experiment=None, name='tissue'):         
        self.name = name 
        if experiment == 'pre-train':                     
            kwargs.class_freqs = self.generatePretrainingMedMnistData(size=kwargs.size,
                                batch_size = kwargs.batch_size,
                                random_state = kwargs.random_state)      
        elif experiment == 'calibrate' or experiment == 'competition':
                kwargs.class_freqs = [0.321, 0.047, 0.035, 0.093, 0.071, 0.047, 0.237, 0.149]
                self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generateCalibrationDatav2(kwargs)
        print("Loading synthetic data for calibration complete")   
        
    def generatePretrainingMedMnistData(self, size,
                                batch_size,
                                random_state):                        
        l_transforms = [transforms.Grayscale(num_output_channels=3), transforms.ToTensor()]
        trans = transforms.Compose(l_transforms)

        print(f"Dataset source information : MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")
        info = INFO['tissuemnist']
        print(info['description'])
        os.makedirs(f"./data/{self.name.upper()}/data_train", exist_ok=True)
        os.makedirs(f"./data/{self.name.upper()}/data_val", exist_ok=True)
        os.makedirs(f"./data/{self.name.upper()}/data_test", exist_ok=True)
        generator = torch.Generator().manual_seed(random_state)
        
        if self.name == 'tissue':
            train_set = TissueMNIST(root="./data/TISSUE/data_train", split="train", transform=trans, download=True, size=size) #, as_rgb=self.as_rgb) #165k        
            class_freqs = print_class_frequencies(train_set)
            train_size = len(train_set)
            half_size = train_size // 2                    
            train_set, eval_cal_set = random_split(train_set, [half_size, train_size - half_size], generator=generator) #82.5k BOTH  
            val_set = TissueMNIST(root="./data/TISSUE/data_val", split="val", transform=trans, download=True, size=size) #, as_rgb=self.as_rgb) #20k
            train_cal_set = TissueMNIST(root="./data/TISSUE/data_test", split="test", transform=trans, download=True, size=size) #, as_rgb=self.as_rgb) #40k
        
        print(f'Train shape: {len(train_set)}, Learn Calibration shape: {len(train_cal_set)}, Validation shape: {len(val_set)}, Eval Calibration shape: {len(eval_cal_set)}')

        self.data_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,)
        self.data_eval_cal_loader = DataLoader(eval_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_train_cal_loader = DataLoader(train_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        
        print(f"Loading {self.name} mnist data for pre-training complete.") 
        return class_freqs

class Cifar10Data(Dataset):    
    def __init__(self, kwargs, experiment=None, name='cifar10'):       
        self.name = name   
        if experiment == 'pre-train':  
            kwargs.class_freqs = [1/kwargs.num_classes]*kwargs.num_classes                        
            self.generatePretrainingCifar10Data(
                                    batch_size = kwargs.batch_size,
                                    random_state = kwargs.random_state)      
        elif experiment == 'calibrate' or experiment == 'competition':
            kwargs.dataset.class_freqs = [1/kwargs.dataset.num_classes]*kwargs.dataset.num_classes
            self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generateCalibrationDatav2(kwargs)
        print("Loading synthetic data for calibration complete")   

    def generatePretrainingCifar10Data(self, 
                                batch_size,
                                random_state):  
        data_dir =  f"./data/{self.name.upper()}"                     
        os.makedirs(data_dir, exist_ok=True)
        generator = torch.Generator().manual_seed(random_state)
        
        l_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
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
    def __init__(self, kwargs, experiment=None, name='cifar100'):       
        self.name = name   
        if experiment == 'pre-train':    
            kwargs.class_freqs = [1/kwargs.num_classes]*kwargs.num_classes
            self.generatePretrainingCifar100Data(
                                    batch_size = kwargs.batch_size,
                                    random_state = kwargs.random_state)      
        elif experiment == 'calibrate' or experiment == 'competition':
            kwargs.dataset.class_freqs = [1/kwargs.dataset.num_classes]*kwargs.dataset.num_classes
            self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generateCalibrationDatav2(kwargs)
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
             transforms.Normalize(                        
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        full_train = datasets.CIFAR100(root=data_dir, train=True, 
                                        transform=l_transform, download=True)
        
        train_size = int(0.45 * len(full_train))
        val_size = int(0.1 * len(full_train))
        eval_cal_size = len(full_train) - train_size - val_size
        train_set, val_set, eval_cal_set = random_split(full_train, [train_size, val_size, eval_cal_size], generator=generator)

        train_cal_set = datasets.CIFAR100(root=data_dir, train=False, 
                                        transform=l_transform, download=True)
        
        print(f'Train shape: {len(train_set)}, Learn Calibration shape: {len(train_cal_set)}, Validation shape: {len(val_set)}, Eval Calibration shape: {len(eval_cal_set)}')

        self.data_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,)
        self.data_eval_cal_loader   = DataLoader(eval_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_train_cal_loader  = DataLoader(train_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        
        print("Loading CIFAR100 data for pre-training complete") 
