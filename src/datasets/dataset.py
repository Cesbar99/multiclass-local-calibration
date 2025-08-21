import torch
import torch.nn.functional as F
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd

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
    def __init__(self, kwargs, experiment=None):        
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
            self.generateCalibrationSynthData(kwargs)    

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
        
    def generateCalibrationSynthData(self, kwargs):
        test_results = "results/{}/{}/raw_results_train_cal_seed-{}_ep-{}_tmp_{}.csv".format(
            'pre-train',
            kwargs.checkpoint.data,
            kwargs.checkpoint.seed,
            kwargs.checkpoint.epochs,
            kwargs.checkpoint.temperature            
        )
        cal_results = "results/{}/{}/raw_results_eval_cal_seed-{}_ep-{}_tmp_{}.csv".format(
            'pre-train',
            kwargs.checkpoint.data,
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
        self.data_train_cal_loader = DataLoader(
            train_cal_set, batch_size=kwargs.dataset.batch_size, shuffle=True, num_workers=8, pin_memory=True
        )
        self.data_test_cal_loader = DataLoader(
            test_cal_set, batch_size=kwargs.dataset.batch_size, shuffle=False, num_workers=8, pin_memory=True
        )
        self.data_val_cal_loader = DataLoader(
            val_cal_set, batch_size=kwargs.dataset.batch_size, shuffle=False, num_workers=8, pin_memory=True
        )

        print("Loading synthetic data for calibration complete")
            
def MnistData():
    pass

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

