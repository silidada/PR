<!--
  -*- coding: utf-8 -*-
 @DATE      : 2022/6/13
 @Author    : Chen HanJie
 @FileName  : README.md
 @Project   : PR
 -->

# 基于CNN性别识别
- [基于CNN性别识别](#基于cnn性别识别)
  - [数据集](#数据集)
    - [数据集处理](#数据集处理)
    - [数据集分配](#数据集分配)
  - [神经网络模型](#神经网络模型)
    - [1. 原始模型](#1-原始模型)
    - [2. 简化模型](#2-简化模型)
    - [3. 拓宽模型](#3-拓宽模型)
  - [训练策略](#训练策略)
    - [损失函数选择](#损失函数选择)
    - [优化器选择](#优化器选择)
  - [How to use](#how-to-use)
    - [predict](#predict)
    - [evaluate](#evaluate)
    - [train](#train)
## 数据集
总数居4000张人脸图片，男女比例大约6：4，如下:
![](imgs/dataset.png)

### 数据集处理
由于样本分配稍微有点不均匀，我们采取重采样方式，增加女性的数据，从而使得样本比例基本为1：1；
经过样本的重采样我们最终得到4860个数据。但是存在数据标签缺失的情况，我们实际可使用数据为4500个左右。

4500个数据中，存在相同的人的不同表情等的人脸数据，而且数据相似度较高，于是我们采用了数据增强的方式来提高训练效果。
在实践中发现在数据增强前，数据容易存在过拟合现象，进行了数据增强后，过拟合现象明显减缓。

### 数据集分配
训练集：验证集：测试集 = 8：1：1

训练集和验证集进行k=10的k折交叉验证

## 神经网络模型
我们搭建了原始模型(Model), 加宽模型(Model_depth_wise), 简化模型(Model_simple)，等等；其他作用不大，有兴趣自行前往[model](model.py)查看。

### 1. 原始模型
我们开始想法是通过深度较深的卷积模型来识别图片，为了防止深度神经网络（DNN）隐藏层过多时的网络退化问题，我们使用了残差网络。
并且使用输出层使用(1*1)的卷积核来实现降维，从而代替全连接层来减少模型的参数。

综上所述，我们使用了全卷积深度残差网络，模型输入为（n\*128\*128\*1），输出为（n\*2）。

具体模型结果如下：
[原图](imgs/model.png)

<img src="./imgs/model.png" alt="model" style="zoom: 50%;" />
数据增强：
![](imgs/Model_acc.png)
![](imgs/Model_loss.png)
原始数据：

![](imgs/Model_no_enhance_loss.png)
![](imgs/Model_no_enhance_acc.png)
### 2. 简化模型
[原图](imgs/Model_simple.png)
<img src="./imgs/Model_simple.png" alt="model" style="zoom: 50%;" />
数据增强：
![](imgs/result_loss_Model_simple.png)
![](imgs/result_acc_Model_simple.png)
原始数据：

![](imgs/Model_simple_no_enhance_loss.png)
![](imgs/Model_simple_no_enhance_acc.png)
### 3. 拓宽模型

为了增加模型的鲁棒性，我们将原始模型的残差通道拓展至三条。
具体如下图：
[原图](imgs/model_deep_wise.png)

<img src="imgs/model_deep_wise.png" alt="model" style="zoom: 50%;" />

数据增强：
![](imgs/Model_depth_wise_acc.png)
![](imgs/Model_depth_wise_loss.png)

原始数据：
![](imgs/Model_depth_wise_no_enhance_acc.png)
![](imgs/Model_depth_wise_no_enhance_loss.png)


## 训练策略

### 损失函数选择

本任务为二分类任务，故选择交叉熵损失函数。

### 优化器选择

在比较了adam和sgd后，发现使用adam优化时，模型经常难以收敛，而使用sgd收敛比较稳定，不会存在模型不收敛的情况。

## How to use
安装torch和opencv；

我们在github中提供了三个预训练模型，请打开model文件夹查看。
### predict
```text
python .\predict.py --Model Model  --model_path .\output\Model_epoch1000\best.pt --image test

或者运行
python .\predict.py --help
来了解更多参数
```
### evaluate
```text
 python .\evaluate.py --Model Model --model_path .\model\Model.pt --label_input ./face/faceDR

或者运行
python .\evaluate.py --help
来了解更多参数
```
### train
```text
python train.py  --Model Model_simple --epochs 1000 --cuda --data_enhance --cuda_id 0

或者运行
python .\train.py --help
来了解更多参数
```