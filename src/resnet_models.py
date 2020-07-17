

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from utils import cyclical_lr

import torch.nn.functional as F


class ResBlock(nn.Module):

    def __init__(self, inlayer, outlayer, stride=1, batch_norm=False):
        super(ResBlock, self).__init__()

        if batch_norm:
            self.conv_block1 = nn.Sequential(
                nn.Conv2d(inlayer, outlayer, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(outlayer),
                nn.ReLU(inplace=True)
            )

            self.conv_block2 = nn.Sequential(
                nn.Conv2d(outlayer, outlayer, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(outlayer)
            )

        else:
            self.conv_block1 = nn.Sequential(
                nn.Conv2d(inlayer, outlayer, kernel_size=3, stride=stride, padding=1),
                nn.ReLU(inplace=True)
            )

            self.conv_block2 = nn.Sequential(
                nn.Conv2d(outlayer, outlayer, kernel_size=3, stride=stride, padding=1),
            )


        self.relu = nn.ReLU()

    
    def forward(self, x):

        residual = x

        x = self.conv_block1(x)
        x = self.conv_block2(x)

        x = x + residual
        
        # Not specified in the paper
        x =  self.relu(x)

        return x 


class SWRCAB(nn.Module):

    def __init__(self, inlayer=64, outlayer=64):
        super(SWRCAB, self).__init__()

        self.inlayer = inlayer
        self.outlayer = outlayer

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(inlayer, outlayer, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.attention_block = nn.Sequential(
            nn.Conv2d(inlayer, outlayer, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(inlayer, outlayer, kernel_size=3, stride=1, padding=1)
        )
                

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(outlayer, outlayer)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):


        residual = x

        x = self.conv_block1(x)

        # Calculate attention
        attention = self.attention_block(x)

        x = self.conv_block2(x)

        # Channel weights
        weights = x * attention

        # Seres block
        weights = self.gap(weights).view(-1, self.outlayer)
        weights = self.fc(weights)
        weights = self.sigmoid(weights)



        # Expand dimension
        weights = weights.view(-1, self.outlayer, 1, 1)

        # Channel wise excitation
        x = x * weights

        x = x + residual

        return x




class ResidualInResiduals(nn.Module):


    def __init__(self, inner_channels=64, block_count=3):
        super(ResidualInResiduals, self).__init__()


        self.res_blocks = nn.ModuleList([SWRCAB(inner_channels, inner_channels) for i in range(block_count)])
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):

        residual = x


        for i, _ in enumerate(self.res_blocks):
            x = self.res_blocks[i](x)

        x = self.conv_block1(x)
        
        x = x + residual

        return x