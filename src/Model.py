import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Backgammon import Backgammon

class ResNet(nn.Module):
    def __init__(self, game, num_blocks, num_hidden, device):
        super(ResNet, self).__init__()
        self.device = device
        num_input_channels = 4
        
        self.startBlock = nn.Sequential(
            nn.Conv2d(num_input_channels, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for _ in range(num_blocks)]
        )
        
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 4 * 6,  game.action_size)
        )
        
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * 4 * 6, 1),
            nn.Tanh()
        )
        
        self.to(device)
    
    def forward(self, x):
        x = self.startBlock(x)
        for block in self.backBone:
            x = block(x)
        
        policy = self.policyHead(x)
        value = self.valueHead(x)
        
        return policy, value
    
class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
