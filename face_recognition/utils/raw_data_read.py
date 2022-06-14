#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/12 15:48
# @Author  : Chen HanJie
# @FileName: raw_data_read.py
# @Software: PyCharm

import numpy as np
import os
from label_pre import label_pre_one_hot
import math
import cv2
import random


# 添加椒盐噪声
def spiced_salt_noise(img,prob):
    output = np.zeros(img.shape,np.uint8)
    thres = 1 - prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0 # 椒盐噪声由纯黑和纯白的像素点随机组成
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = img[i][j]
    return output


# 镜像变换
def mirror(img,mode):
    img = cv2.flip(img, mode)  # mode = 1 水平翻转 mode = 0 垂直翻
    return img


# 旋转
def rotation(img,angle,scale):
    rows = img.shape[0]
    cols = img.shape[1]
    # 这里的第一个参数为旋转中心，第二个为旋转角度，第三个为旋转后的缩放因子
    # 可以通过设置旋转中心，缩放因子，以及窗口大小来防止旋转后超出边界的问题
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale) # 向左旋转angle度并缩放为原来的scale倍
    img = cv2.warpAffine(img, M, (cols, rows)) # 第三个参数是输出图像的尺寸中心
    return img


# 模糊
def blur(img,scale):
    img = cv2.blur(img,(scale, scale)) # scale越大越模糊
    return img


def read_rawdata(channels, path, image_name_list, dsize, add_noisy=False):
    assert os.path.exists(path)

    img_list = []

    num_noisy = len(image_name_list) // 8
    ii = 0

    for name in image_name_list:
        full_path = os.path.join(path, name)
        if not os.path.exists(full_path):
            # print("数据缺失：", name)
            img_list.append(None)
            continue
        img = np.fromfile(full_path, dtype='uint8')
        n = int(math.sqrt(img.shape[0]))
        img = img.reshape(n, n, channels)
        img = cv2.resize(img, (dsize,dsize))

        if ii < num_noisy and add_noisy:
            mean = 0
            sigma = 30
            gauss = np.random.normal(mean, sigma, (dsize, dsize))
            # print(gauss.shape, img.shape)
            img = img + gauss
            img = np.clip(img, a_min=0, a_max=255)

        if num_noisy <= ii < num_noisy*2 and add_noisy:
            img = spiced_salt_noise(img, 0.1)

        if num_noisy*2 <= ii < num_noisy*3 and add_noisy:
            img = mirror(img,1)

        if num_noisy * 3 <= ii < num_noisy * 4 and add_noisy:
            angle = random.randint(0,360)
            img = rotation(img,angle,1)

        if num_noisy * 4 <= ii < num_noisy * 5 and add_noisy:
            img = blur(img, 2)

        ii += 1

        img_list.append(img)
    return img_list


if __name__ == '__main__':
    channels = 1  # 图像的通道数，灰度图为1
    path_rawdata = r"../face/rawdata"
    path_label1 = r"../face/faceDR"
    path_label2 = r"../face/faceDS"
    label_one_hot, img_name, label_name = label_pre_one_hot(path_label1, path_label2)
    img_list = read_rawdata(channels, path_rawdata, img_name)
    print(len(img_list))



