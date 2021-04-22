import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class Net(nn.Module):
    def __init__(self):
        """
        Initializer of the network class. Define each layer of the network.

        Args:
            A () : ...
        """
        super(Net, self).__init__()
        self.Resnet = models.resnet50(pretrained=True)
        self.Resnet.fc = nn.Linear(2048, 18)

    def forward(self, x):
        """
        Forward pass of my network.

        Args:
            x () : input of my network

        Returns:
            x () : return of my network 
        """
        return self.Resnet(x)
    
    
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        
        
    def forward(self, x):
        return x
    
    
    def residual_block(self, filter_size):
        layers = []
        
    def make_layer(self, in_dim, mid_dim, out_dim, repeats, starting=False):
        layers = []
        layers.append(Bottleneck(in_dim, mid_dim, out_dim, down=True, starting=starting))
        for _ in range(1, repeats):
            layers.append(Bottleneck(out_dim, mid_dim, out_dim, down=False))
        return nn.Sequential(*layers)
