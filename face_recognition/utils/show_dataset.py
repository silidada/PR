#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/12 12:04
# @Author  : Chen HanJie
# @FileName: show_dataset.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import matplotlib.image as mping
import os
import numpy as np
import cv2
import math
img_list = []
img_dir = os.listdir("../face/rawdata")
img_path_list = []
for n in img_dir:
    img_path_list.append(os.path.join("../face/rawdata", n))

for n in img_path_list:
    img = np.fromfile(n, dtype='uint8')
    nn = int(math.sqrt(img.shape[0]))
    img = img.reshape(nn, nn, 1)
    img = cv2.resize(img, (128, 128))
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_list.append(imgRGB)

m=10
n=20
img_temp = []
for i in range(0,m*n,n):
    img_temp.append(np.concatenate(img_list[i:i+n],axis=1))
img_end = np.concatenate(img_temp,axis=0)

mping.imsave(f"../imgs/dataset.png",img_end)