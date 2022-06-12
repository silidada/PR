#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/12 16:13
# @Author  : Chen HanJie
# @FileName: model.py
# @Software: PyCharm
import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock1(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock1, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels // 2, (1, 1), padding=0, stride=(1, 1))
        self.conv2 = nn.Conv2d(channels // 2, channels // 2, (3, 3), padding=1, stride=(1, 1))
        self.conv3 = nn.Conv2d(channels // 2, channels, (1, 1), padding=0, stride=(1, 1))
        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.batch_norm(out)
        out = torch.relu(out + x)
        return out


class ResidualBlock2(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock2, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels // 2, (1, 1), padding=0)
        self.conv2 = nn.Conv2d(channels // 2, channels // 2, (3, 3), padding=1, stride=(2, 2))
        self.conv3 = nn.Conv2d(channels // 2, channels * 2, (1, 1), padding=0)
        self.batch_norm = nn.BatchNorm2d(channels * 2)

        self.conv4 = nn.Conv2d(channels, channels * 2, (1, 1), padding=0, stride=(2, 2))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.batch_norm(out)
        x = self.conv4(x)
        out = torch.relu(out + x)
        return out


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # input (128*128*1) -> (128,128,32)
        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding=1, stride=(1,1))
        self.batchNorm1 = nn.BatchNorm2d(32)
        # input (64*64*32) -> (64,64,64)
        self.conv2 = nn.Conv2d(32, 64, (3, 3), padding=1, stride=(2, 2))
        self.batchNorm2 = nn.BatchNorm2d(64)
        # input (64*64*64) -> (64*64*64)
        self.block1 = ResidualBlock1(64)
        # input (64*64*64) -> (32*32*128)
        self.block2 = ResidualBlock2(64)
        # input (32*32*128) -> (32*32*128)
        self.block3 = ResidualBlock1(128)
        # input (32*32*128) -> (16*16*256)
        self.block4 = ResidualBlock2(128)
        # input (16*16*256) -> (16*16*256)
        self.block5 = ResidualBlock1(256)
        # input (16*16*256) -> (8*8*512)
        self.block6 = ResidualBlock2(256)
        # input (8*8*512) -> (8*8*256)
        self.conv3 = nn.Conv2d(512, 256, (1, 1), padding=0, stride=(1, 1))
        self.batchNorm3 = nn.BatchNorm2d(256)
        # input (8*8*256) -> (8*8*256)
        self.block7 = ResidualBlock1(256)
        # input (8*8*256) -> (4*4*512)
        self.block8 = ResidualBlock2(256)
        # input (4*4*512) -> (4*4*512)
        self.block9 = ResidualBlock1(512)
        # input (4*4*512) -> (2*2*1024)
        self.block10 = ResidualBlock2(512)
        # input (2*2*1024) -> (2*2*512)
        self.conv4 = nn.Conv2d(1024, 512, (1, 1), padding=0, stride=(1, 1))
        self.batchNorm4 = nn.BatchNorm2d(512)
        # input (2*2*512) -> (1*1*256)
        self.conv5 = nn.Conv2d(512, 256, (2, 2), padding=0, stride=(1, 1))
        self.batchNorm5 = nn.BatchNorm2d(256)
        # input (1*1*256) -> (1*1*19)
        self.conv6 = nn.Conv2d(256, 19, (1, 1), padding=0, stride=(1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = F.relu(self.batchNorm1(self.conv1(x)))
        out = F.relu(self.batchNorm2(self.conv2(out)))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = F.relu(self.batchNorm3(self.conv3(out)))
        out = self.block7(out)
        out = self.block8(out)
        out = self.block9(out)
        out = self.block10(out)
        out = F.relu(self.batchNorm4(self.conv4(out)))
        out = F.relu(self.batchNorm5(self.conv5(out)))
        out = self.sigmoid(self.conv6(out))
        return out



if __name__ == '__main__':
    test_tensor = torch.randn((5,1,128,128))
    model = Model()
    result = model(test_tensor)