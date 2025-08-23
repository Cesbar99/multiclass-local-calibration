import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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

    
class TissueMNISTResNet50(nn.Module):
    """Model for just classification.
    The architecture of our model is the same as standard DenseNet121
    """

    def __init__(self, temperature=1.0, num_labels=8):
        super(TissueMNISTResNet50, self).__init__()      
        self.scaler = ScaledLogits(temperature)
        self.resnet50 = models.resnet50(weights='IMAGENET1K_V2')
        self.num_features = self.resnet50.fc.in_features
        self.resnet50.layer4 = nn.Sequential(
            self.resnet50.layer4,
            nn.Dropout(p=0.2)
        )
        self.resnet50.fc = nn.Sequential(
                nn.Dropout(p=0.1),                
                nn.Linear(self.num_features, num_labels))
        self.n_classes = num_labels
        
        for param in self.resnet50.parameters():
            param.requires_grad = False        
        for name, param in self.resnet50.named_parameters():
            if 'layer4' in name or 'fc' in name: #
                param.requires_grad = True

    def repr(self, x):
        # See note [TorchScript super()]
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)

        x = self.resnet50.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        logits = self.resnet50(x)
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
