import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledLogits(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, logits):
        return logits / self.temperature
    
##### DEEP MLP FOR SYNTHETIC TABULAR DATA #####
class DeepMLP(nn.Module):
    def __init__(self, input_dim, output_dim, temperature=1.0):
        super().__init__()
        self.temperature = temperature

        self.main = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            #nn.Linear(128, 128),
            #nn.ReLU(),
            #nn.Linear(128, 128),
            #nn.ReLU(),
            #nn.Linear(128, 256),
            #nn.ReLU(),
            #nn.Linear(256, 128),
            #nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.output_layer = nn.Linear(64, output_dim)
        self.scaler = ScaledLogits(temperature)

    def forward(self, x):
        latent = self.main(x)
        logits = self.output_layer(latent)        
        return self.scaler(logits)
    


class MnistArch(nn.Module):
    def __init__(self, temperature=1.0):        
        super(MnistArch, self).__init__()
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10

        layers = [
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 200),  # After convs and pooling, image size is reduced
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 10)
        ]        
        
        self.model = nn.Sequential(*layers)
        self.scaler = ScaledLogits(temperature)

    def forward(self, x):
        logits = self.model(x) 
        return self.scaler(logits)   
    
class Cifar10Arch(nn.Module):
    def __init__():
        super().__init__()
        pass
    
class Cifar10OODArch(nn.Module):
    def __init__():
        super().__init__()
        pass

class Cifar10LongTailArch(nn.Module):
    def __init__():
        super().__init__()
        pass
    
class Cifar100Arch(nn.Module):
    def __init__():
        super().__init__()
        pass
    
class Cifar100LongTailArch(nn.Module):
    def __init__():
        super().__init__()
        pass
    
class ImagenetArch(nn.Module):
    def __init__():
        super().__init__()
        pass
    
class ImagenetOODArch(nn.Module):
    def __init__():
        super().__init__()
        pass
    
class ImagenetLongTailArch(nn.Module):
    def __init__():
        super().__init__()
        pass
