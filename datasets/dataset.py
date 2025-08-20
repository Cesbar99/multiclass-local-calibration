import torch
import torch.nn.functional as F
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader



class ClassificationDataset(Dataset):
    def __init__(self, X, y, transforms_fn=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)  # one-hot encoded
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
    def __init__(self, kwargs):
                
        self.generateSynthData(num_features = kwargs.num_features,
                                num_classes = kwargs.num_classes,
                                n_samples = kwargs.num_samples,
                                n_redundant = kwargs.n_redundant,
                                n_clusters_per_class = kwargs.n_clusters_per_class,
                                class_sep = kwargs.class_sep,
                                flip_y = kwargs.flip_y,
                                batch_size = kwargs.batch_size,
                                random_state = kwargs.random_state)                              

    def generateSynthData(self, num_features,
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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=random_state)
        X_cal, X_test, y_cal, y_test = train_test_split(X_test, y_test, test_size=0.1, random_state=random_state)
        #y_train_oh = F.one_hot(y_train, num_classes=num_classes) #to_categorical(y_train, num_classes)

        print(X_train.shape, X_test.shape, X_cal.shape)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long) 
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long) 
        X_cal = torch.tensor(X_cal, dtype=torch.float32)
        y_cal = torch.tensor(y_cal, dtype=torch.long) 
        
        train_set = ClassificationDataset(X_train, y_train)
        cal_set   = ClassificationDataset(X_cal, y_cal)
        test_set  = ClassificationDataset(X_test, y_test)

        self.data_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,)
        self.data_cal_loader   = DataLoader(cal_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        self.data_test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        
        print("loading synthetic data complete")
        
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

