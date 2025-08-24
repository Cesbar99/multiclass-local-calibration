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
from medmnist import TissueMNIST, PathMNIST, INFO
from torch.utils.data import random_split
import os
from tiny_imagenet_torch import TinyImageNet

def generateCalibrationData(kwargs):
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


class ClassificationDataset(Dataset):
    def __init__(self, X, y, transforms_fn=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)  # one-hot encoded
        self.transforms_fn = transforms_fn

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.transforms_fn is not None:
            x = self.transforms_fn(self.X[idx])
        else:
            x = self.X[idx]
        y = self.y[idx]
        return x, y

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
        elif experiment == 'calibrate':
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
            
            
class MnistData(Dataset):    
    def __init__(self, kwargs, experiment=None, name='mnist'):          
        if experiment == 'pre-train':   
            if kwargs.variant:                  
                self.generatePretrainingNMnistData(variant = kwargs.variant,
                                    batch_size = kwargs.batch_size,
                                    random_state = kwargs.random_state)        
            else:
                self.generatePretrainingMnistData(
                                    batch_size = kwargs.batch_size,
                                    random_state = kwargs.random_state)      
        elif experiment == 'calibrate':
                self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generateCalibrationData(kwargs) 
        print("Loading synthetic data for calibration complete")   

    def generatePretrainingMnistData(self, 
                                batch_size,
                                random_state):                        
        
        transform = transforms.ToTensor()
        mnist_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

        # Convert to NumPy arrays
        X = mnist_dataset.data.numpy().astype(np.float32) / 255.0  # Normalize to [0,1]
        X = np.expand_dims(X, axis=1)  # Add channel dimension: (N, 1, 28, 28)
        y = np.array(mnist_dataset.targets)
        
        X_train, X_train_cal, y_train, y_train_cal = train_test_split(X, y, test_size=0.5, random_state=random_state)        
        X_eval_cal, X_train_cal, y_eval_cal, y_train_cal = train_test_split(X_train_cal, y_train_cal, test_size=0.1668, random_state=random_state)        
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1668, random_state=random_state)
        
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
        
        print("Loading MNIS data for pre-training complete") 
        
    def generatePretrainingNMnistData(self, variant,
                                batch_size,
                                random_state):                        
        variant_map = {
                'awgn': 'mnist-with-awgn.mat',
                'motion_blur': 'mnist-with-motion-blur.mat',
                'reduced_contrast': 'mnist-with-reduced-contrast-and-awgn.mat'
        }
        mat_path = f"./data/NMNIST/{variant_map[variant]}"
        data = loadmat(mat_path)

        # Load and normalize images
        X = data['train_x'].astype(np.float32) / 255.0  # shape: (60000, 784)
        X = X.reshape(-1, 1, 28, 28)  # shape: (60000, 1, 28, 28)
        X_train_cal = data['test_x'].astype(np.float32) / 255.0  # shape: (60000, 784)
        X_train_cal = X_train_cal.reshape(-1, 1, 28, 28)  # shape: (60000, 1, 28, 28)

        # Decode one-hot labels
        y = np.argmax(data['train_y'], axis=1)  # shape: (60000,)
        y_train_cal = np.argmax(data['test_y'], axis=1)  # shape: (60000,)

        # Split into training, calibration, validation, and eval sets
        X_train, X_eval_cal, y_train, y_eval_cal = train_test_split(X, y, test_size=0.829, random_state=random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=random_state)

        # Convert to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_train_cal = torch.tensor(X_train_cal, dtype=torch.float32)
        y_train_cal = torch.tensor(y_train_cal, dtype=torch.long)
        X_eval_cal = torch.tensor(X_eval_cal, dtype=torch.float32)
        y_eval_cal = torch.tensor(y_eval_cal, dtype=torch.long)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)

        # Wrap in datasets
        train_set = ClassificationDataset(X_train, y_train)
        eval_cal_set = ClassificationDataset(X_eval_cal, y_eval_cal)
        val_set = ClassificationDataset(X_val, y_val)
        train_cal_set = ClassificationDataset(X_train_cal, y_train_cal)

        # Create DataLoaders
        self.data_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        self.data_eval_cal_loader = DataLoader(eval_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_train_cal_loader = DataLoader(train_cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        
        print(f"Train shape: {X_train.shape}, Learn Calibration shape: {X_train_cal.shape}, Validation shape: {X_val.shape}, Eval Calibration shape: {X_eval_cal.shape}")
        print(f"Loading n-MNIST variant '{variant}' for pre-training completed.")
                        

class MedMnistData(Dataset):    
    def __init__(self, kwargs, experiment=None, name='path'):         
        self.name = name 
        if experiment == 'pre-train':                     
            self.generatePretrainingMedMnistData(size=kwargs.size,
                                batch_size = kwargs.batch_size,
                                random_state = kwargs.random_state)      
        elif experiment == 'calibrate':
                self.data_train_cal_loader, self.data_test_cal_loader, self.data_val_cal_loader = generateCalibrationData(kwargs) 
        print("Loading synthetic data for calibration complete")   
        
    def generatePretrainingMedMnistData(self, size,
                                batch_size,
                                random_state):                        
        l_transforms = [transforms.ToTensor()]
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(10),
            #transforms.Resize((224, 224)) if size < 224 else transforms.Lambda(lambda x: x),
            #transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel RGB
            #transforms.ToTensor(), 
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                    std=[0.229, 0.224, 0.225])
        #]

        # Remove any None or identity transforms
        #l_transforms = [t for t in l_transforms if not isinstance(t, transforms.Lambda)]

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
        
        print("Loading TissueMnist data for pre-training complete") 


def Cifar10Data():
    pass

def Cifar10OODData():
    pass

def Cifar10LongTailData():
    pass

def Cifar100Data():
    pass

def Cifar100LongTailData():
    pass

def ImagenetData():
    pass

def ImagenetOODData():
    pass

def ImagenetLongTailData():
    pass

