#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Hongwei Yi (hongweiyi@pku.edu.cn)

import torch.nn as nn
import torch
import numpy as np


def conv(in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
        nn.LeakyReLU(0.0,inplace=True)
    )

def convbn(in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.0,inplace=True)
    )

# def conv3d(in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True):
#     return nn.Sequential(
#         nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
#         nn.LeakyReLU(0.0,inplace=True)
#     )


def conv3dgn(in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
        nn.GroupNorm(1, 1), 
        nn.LeakyReLU(0.0,inplace=True)
    )

def conv3d(in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
        nn.BatchNorm3d(out_channels),
        nn.LeakyReLU(0.0,inplace=True)
    )

def resnet_block(in_channels,  kernel_size=3, dilation=[1,1], bias=True):
    return ResnetBlock(in_channels, kernel_size, dilation, bias=bias)

def resnet_block_bn(in_channels,  kernel_size=3, dilation=[1,1], bias=True):
    return ResnetBlockBn(in_channels, kernel_size, dilation, bias=bias)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias):
        super(ResnetBlock, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0], padding=((kernel_size-1)//2)*dilation[0], bias=bias),
            nn.LeakyReLU(0.0, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding=((kernel_size-1)//2)*dilation[1], bias=bias),
        )
    def forward(self, x):
        out = self.stem(x) + x
        return out

class ResnetBlockBn(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias):
        super(ResnetBlockBn, self).__init__()
        self.stem = nn.Sequential(
            convbn(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0], bias=bias),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding=((kernel_size-1)//2)*dilation[1], bias=bias),
        )
    def forward(self, x):
        out = self.stem(x) + x
        return out

##### Define weightnet-3d
def volumegatelight(in_channels, kernel_size=3, dilation=[1,1], bias=True):
    return nn.Sequential(
        #MSDilateBlock3D(in_channels, kernel_size, dilation, bias),
        conv3d(in_channels, 1, kernel_size=1, stride=1, bias=bias),
        conv3d(1, 1, kernel_size=1, stride=1)
     )

def volumegatelightgn(in_channels, kernel_size=3, dilation=[1,1], bias=True):
    return nn.Sequential(
        #MSDilateBlock3D(in_channels, kernel_size, dilation, bias),
        conv3dgn(in_channels, 1, kernel_size=1, stride=1, bias=bias),
        conv3dgn(1, 1, kernel_size=1, stride=1)
     )



