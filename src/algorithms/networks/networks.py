import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from tab_transformer_pytorch import FTTransformer

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
         
                        
class TissueMnistVit(nn.Module):
    """Model for just classification.
    The architecture of our model is the same as standard DenseNet121
    """

    def __init__(self, temperature=1.0, num_labels=8):
        super(TissueMnistVit, self).__init__()      
        self.scaler = ScaledLogits(temperature)
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_labels, in_chans=1)
        print(self.vit)
        
        # Freeze all parameters
        for param in self.vit.parameters():
            param.requires_grad = False

        # Unfreeze last transformer block
        for name, param in self.vit.named_parameters():
            if 'blocks.11' in name:
                param.requires_grad = True

        # Unfreeze classifier head
        for param in self.vit.get_classifier().parameters():
            param.requires_grad = True

    def forward(self, x):
        logits = self.vit(x)                
        return self.scaler(logits)          
    
class PathMnistVit(nn.Module):
    """Model for just classification.
    The architecture of our model is the same as standard DenseNet121
    """

    def __init__(self, temperature=1.0, num_labels=9):
        super(PathMnistVit, self).__init__()      
        self.scaler = ScaledLogits(temperature)
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_labels, in_chans=3)
        print(self.vit)
        
        # Freeze all parameters
        for param in self.vit.parameters():
            param.requires_grad = False

        # Unfreeze last transformer block
        for name, param in self.vit.named_parameters():
            if 'blocks.11' in name:
                param.requires_grad = True

        # Unfreeze classifier head
        for param in self.vit.get_classifier().parameters():
            param.requires_grad = True

    def forward(self, x):
        logits = self.vit(x)                
        return self.scaler(logits)       
    
    
class Cifar10Vit(nn.Module):
    """Model for just classification.
    The architecture of our model is the same as standard DenseNet121
    """

    def __init__(self, temperature=1.0, num_labels=10):
        super(Cifar10Vit, self).__init__()      
        self.scaler = ScaledLogits(temperature)
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_labels, in_chans=3)
        print(self.vit)
        
        # Freeze all parameters
        for param in self.vit.parameters():
            param.requires_grad = False

        # Unfreeze last transformer block
        for name, param in self.vit.named_parameters():
            if 'blocks.11' in name:
                param.requires_grad = True

        # Unfreeze classifier head
        for param in self.vit.get_classifier().parameters():
            param.requires_grad = True

    def forward(self, x):
        logits = self.vit(x)                
        return self.scaler(logits)   
    
    
class Cifar10LTVit(nn.Module):
    """Model for just classification.
    The architecture of our model is the same as standard DenseNet121
    """

    def __init__(self, temperature=1.0, num_labels=10):
        super(Cifar10LTVit, self).__init__()      
        self.scaler = ScaledLogits(temperature)
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_labels, in_chans=3)
        print(self.vit)
        
        # Freeze all parameters
        for param in self.vit.parameters():
            param.requires_grad = False

        # Unfreeze last transformer block
        for name, param in self.vit.named_parameters():
            if 'blocks.11' in name:
                param.requires_grad = True

        # Unfreeze classifier head
        for param in self.vit.get_classifier().parameters():
            param.requires_grad = True

    def forward(self, x):
        logits = self.vit(x)                
        return self.scaler(logits)        

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        if self.dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = F.relu(self.fc1(x))
        if self.dropout:
            x = self.dropout(x)
        x = self.norm2(x)
        x = self.fc2(x)
        return x + residual  # Residual connection

class ResMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, depth=4, dropout=0.1):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(depth)])
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x_cat, x_num = x
        x = torch.cat([x_cat, x_num], dim=1) 
        x = F.relu(self.input_layer(x))
        x = self.blocks(x)
        return self.output_layer(x)    
    
    
class CovType_FTT(nn.Module):
    """Model for just classification.
    The architecture of our model is the same as standard DenseNet121
    """
    def __init__(self, category_counts, numerical_features, temperature=1.0, num_labels=10):
        super(CovType_FTT, self).__init__()      
        self.scaler = ScaledLogits(temperature)
        self.ftt = FTTransformer(
            categories = category_counts,                               # tuple containing the number of unique values within each category (10, 5, 6, 5, 8)
            num_continuous = len(numerical_features),                   # number of continuous values
            dim = 32,                                                   # dimension, paper set at 32
            dim_out = num_labels,                                       # binary prediction, but could be anything
            depth = 6,                                                  # depth, paper recommended 6
            heads = 8,                                                  # heads, paper recommends 8
            attn_dropout = 0.1,                                         # post-attention dropout
            ff_dropout = 0.1                                            # feed forward dropout
            #mlp_hidden_mults = (4, 2),                                 # relative multiples of each hidden dimension of the last mlp to logits
            #mlp_act = nn.ReLU()                                        # activation for final mlp, defaults to relu, but could be anything else (selu etc)
            #continuous_mean_std = cont_mean_std # (optional) - normalize the continuous values before layer norm
        )        
        print(self.ftt)

    def forward(self, x):
        x_cat, x_num = x
        logits = self.ftt(x_cat, x_num)                
        return self.scaler(logits)        
    
class Otto_FTT(nn.Module):
    """Model for just classification.
    The architecture of our model is the same as standard DenseNet121
    """
    def __init__(self, numerical_features, temperature=1.0, num_labels=9):
        super(Otto_FTT, self).__init__()      
        self.scaler = ScaledLogits(temperature)
        self.ftt = FTTransformer(
            categories = [],                                            # tuple containing the number of unique values within each category (10, 5, 6, 5, 8)
            num_continuous = numerical_features,                        # number of continuous values
            dim = 32,                                                   # dimension, paper set at 32
            dim_out = num_labels,                                       # binary prediction, but could be anything
            depth = 6,                                                  # depth, paper recommended 6
            heads = 8,                                                  # heads, paper recommends 8
            attn_dropout = 0.1,                                         # post-attention dropout
            ff_dropout = 0.1                                            # feed forward dropout
            #mlp_hidden_mults = (4, 2),                                 # relative multiples of each hidden dimension of the last mlp to logits
            #mlp_act = nn.ReLU()                                        # activation for final mlp, defaults to relu, but could be anything else (selu etc)
            #continuous_mean_std = cont_mean_std # (optional) - normalize the continuous values before layer norm
        )        
        print(self.ftt)

    def forward(self, x):
        x_categ, x_num = x        
        logits = self.ftt(x_categ, x_num)                
        return self.scaler(logits)        
    
class Cifar10OODArch(nn.Module):
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
