# import the libs
import torch
import torch.nn as nn
import torch.nn.functional as F
# from tqdm.notebook import tqdm
import numpy as np
from torchvision import datasets,transforms
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=2)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(2)
        
        self.fc1 = nn.Linear(32*8*8, 600)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(600, 10)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu1(x)
        x = self.maxpool2(x)
        
        x = x.view(-1, 32*8*8)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
        
    