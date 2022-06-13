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
        self.conv6 = nn.Conv2d(256, 2, (1, 1), padding=0, stride=(1, 1))
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
        out = self.conv6(out)
        out = self.sigmoid(out)
        # out = torch.softmax(out, 1)
        return out


class Model_simple(nn.Module):
    def __init__(self):
        super(Model_simple, self).__init__()
        # (1*128*128)
        self.conv1 = nn.Conv2d(1, 16, (3,3), stride=(2,2), padding=1)
        # self.conv1 = nn.Conv2d(1, 16, (3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(16, 16, (3, 3), stride=(2, 2), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)
        # (16*16*16)
        self.conv3 = nn.Conv2d(16, 32, (3,3), stride=(1,1), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)
        # (32*8*8)
        self.conv4 = nn.Conv2d(32, 64, (3,3), stride=(1,1), padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)
        # self.conv5 = nn.Conv2d(64, 64, (3,3), stride=(1,1), padding=1)
        # (64*8*8)
        self.conv6 = nn.Conv2d(64, 16, (1,1), stride=(1,1), padding=0)
        self.conv7 = nn.Conv2d(16, 4, (1, 1), stride=(1, 1), padding=0)

        # (8*8*8)
        # self.fc = nn.Linear(128, 2)
        self.fc = nn.Linear(64, 16)
        self.dp = nn.Dropout()
        self.out = nn.Linear(16,2)
        # self.out = nn.Linear(256, 2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        input_size = x.size(0)
        x = F.relu(self.pool1(self.bn1(self.conv2(self.conv1(x)))))
        x = F.relu(self.pool2(self.bn2(self.conv3(x))))
        x = F.relu(self.pool3(self.bn3(self.conv4(x))))
        # x = self.conv6(self.conv5(x))
        # x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = x.view(input_size, -1)

        x = self.fc(x)
        x = self.dp(x)
        x = self.out(x)
        # x = self.out(x)
        # return torch.softmax(x, 1)
        return self.sigmoid(x)


class Model_simpleFCN(nn.Module):
    def __init__(self):
        super(Model_simpleFCN, self).__init__()
        # (1*128*128)
        self.conv1 = nn.Conv2d(1, 16, (3,3), stride=(2,2), padding=1)
        # self.conv1 = nn.Conv2d(1, 16, (3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(16, 16, (3, 3), stride=(2, 2), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)
        # (16*16*16)
        self.conv3 = nn.Conv2d(16, 32, (3,3), stride=(1,1), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)
        # (32*8*8)
        self.conv4 = nn.Conv2d(32, 64, (3,3), stride=(1,1), padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)
        # self.conv5 = nn.Conv2d(64, 64, (3,3), stride=(1,1), padding=1)
        # (64*8*8)
        self.conv5 = nn.Conv2d(64, 64, (3,3),stride=(2,2), padding=1)
        self.conv51 = nn.Conv2d(64, 64, (3,3),stride=(2,2), padding=1)
        self.conv6 = nn.Conv2d(64, 16, (1,1), stride=(1,1), padding=0)
        self.conv7 = nn.Conv2d(16, 2, (1, 1), stride=(1, 1), padding=0)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        input_size = x.size(0)
        x = F.relu(self.pool1(self.bn1(self.conv2(self.conv1(x)))))
        x = F.relu(self.pool2(self.bn2(self.conv3(x))))
        x = F.relu(self.pool3(self.bn3(self.conv4(x))))
        # x = self.conv6(self.conv5(x))
        x = self.conv5(x)
        x = self.conv51(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = x.view(input_size, -1)

        return self.sigmoid(x)


class Model_simple64(nn.Module):
    def __init__(self):
        super(Model_simple64, self).__init__()
        # (1*64*64)
        # self.conv1 = nn.Conv2d(1, 16, (3,3), stride=(2,2), padding=1)
        self.conv1 = nn.Conv2d(1, 3, (5, 5), stride=(2, 2), padding=2)
        # self.conv2 = nn.Conv2d(16, 16, (3, 3), stride=(2, 2), padding=1)
        self.bn1 = nn.BatchNorm2d(3)
        self.pool1 = nn.MaxPool2d(2)
        # (3*16*16)
        self.conv3 = nn.Conv2d(3, 32, (3,3), stride=(1,1), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)
        # (32*8*8)
        self.conv4 = nn.Conv2d(32, 64, (3,3), stride=(1,1), padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)
        # self.conv5 = nn.Conv2d(64, 64, (3,3), stride=(1,1), padding=1)
        # (64*4*4)
        # self.conv6 = nn.Conv2d(64, 16, (1,1), stride=(1,1), padding=0)
        # self.conv7 = nn.Conv2d(16, 4, (1, 1), stride=(1, 1), padding=0)

        # (4*8*8)
        # self.fc = nn.Linear(128, 2)
        self.dp = nn.Dropout()
        self.fc = nn.Linear(1024, 256)
        self.dp1 = nn.Dropout()
        self.out = nn.Linear(256,2)
        # self.out = nn.Linear(256, 2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        input_size = x.size(0)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # print(x.size())
        x = self.pool2(F.relu(self.bn2(self.conv3(x))))
        x = self.pool3(F.relu(self.bn3(self.conv4(x))))
        # x = F.relu(self.pool1(self.bn1((self.conv1(x)))))
        # x = F.relu(self.pool2(self.bn2(self.conv3(x))))
        # x = F.relu(self.pool3(self.bn3(self.conv4(x))))
        # x = self.conv6(self.conv5(x))
        # x = self.conv5(x)
        # x = self.conv6(x)
        # x = self.conv7(x)
        x = x.view(input_size, -1)

        x = self.fc(x)
        x = self.dp(x)
        x = self.out(x)
        # x = self.out(x)
        # return torch.softmax(x, 1)
        return self.sigmoid(x)


if __name__ == '__main__':
    test_tensor = torch.randn((254,1,64,64))
    model = Model_simpleFCN()
    result = model(test_tensor)
    print(result.shape)