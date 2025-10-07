import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import timm
from tab_transformer_pytorch import FTTransformer

class ScaledLogits(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, logits):
        return logits / self.temperature
    
class TissueMNISTResNet50(nn.Module):

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
                #nn.Dropout(p=0.1),                
                nn.Linear(self.num_features, num_labels))
        self.n_classes = num_labels
        
        for param in self.resnet50.parameters():
            param.requires_grad = False        
        for name, param in self.resnet50.named_parameters():
            if 'layer4' in name or 'fc' in name: 
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
    
class Cifar10ResNet50(nn.Module):

    def __init__(self, temperature=1.0, num_labels=10):
        super(Cifar10ResNet50, self).__init__()      
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
            if 'layer4' in name or 'fc' in name: 
                param.requires_grad = True

    def repr(self, x):
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
    
class Cifar100ResNet152(nn.Module):

    def __init__(self, temperature=1.0, num_labels=100):
        super(Cifar100ResNet152, self).__init__()      
        self.scaler = ScaledLogits(temperature)
        self.resnet152 = models.resnet152(weights='IMAGENET1K_V2')
        self.num_features = self.resnet152.fc.in_features
        
        self.resnet152.fc = nn.Sequential(
                nn.Dropout(p=0.5),                
                nn.Linear(self.num_features, num_labels))
        self.n_classes = num_labels
        
        for param in self.resnet152.parameters():
            param.requires_grad = False        
        for name, param in self.resnet152.named_parameters():
            if 'layer4' in name or 'fc' in name: 
                param.requires_grad = True

    def repr(self, x):
        x = self.resnet152.conv1(x)
        x = self.resnet152.bn1(x)
        x = self.resnet152.relu(x)
        x = self.resnet152.maxpool(x)

        x = self.resnet152.layer1(x)
        x = self.resnet152.layer2(x)
        x = self.resnet152.layer3(x)
        x = self.resnet152.layer4(x)

        x = self.resnet152.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        logits = self.resnet152(x)
        return self.scaler(logits)      
 
