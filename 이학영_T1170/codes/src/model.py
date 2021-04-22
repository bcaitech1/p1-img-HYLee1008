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