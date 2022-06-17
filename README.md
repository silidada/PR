<!--
  -*- coding: utf-8 -*-
 @DATE      : 2022/6/17
 @Author    : Chen HanJie
 @FileName  : README.md
 @Project   : PR
 -->

# 模式识别课程作业

## 简介
使用给定人脸数据集来进行性别识别。数据包含两个卷标文件：faceDR和faceDS。
![](face_recognition/imgs/dataset.png)
卷标文件格式如下：
```text
 1223 (_sex  male) (_age  child) (_race white) (_face smiling) (_prop '())
 1224 (_sex  male) (_age  child) (_race white) (_face serious) (_prop '())
 1225 (_sex  male) (_age  child) (_race white) (_face smiling) (_prop '())
 1226 (_sex  male) (_age  child) (_race white) (_face smiling) (_prop '())
 1227 (_sex  male) (_age  child) (_race white) (_face serious) (_prop '())
 1228 (_missing descriptor)
```
## 文件目录
```text
---face_recignition_classic               基于传统算法的性别识别
---face_recognition                       基于深度学习的性别识别
---Fisher Fisher                          人脸验证算法（与本项目无关）
```

## 实现概要
本次课程作业我们主要使用了Pytorch 和 sklearn两个库；

在基于深度学习的性别识别中，我们使用了pytorch搭建了多个神经网络来进行识别。

在基于传统算法的性别识别中，我们使用了sklearn实现了KNN、SVM、决策、BP等多种算法来进行性别识别。
