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


def read_rawdata(channels, path, image_name_list, dsize):
    assert os.path.exists(path)

    img_list = []

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



