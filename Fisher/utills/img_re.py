#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/12 12:04
# @Author  : Chen HanJie
# @FileName: img_re.py
# @Software: PyCharm

import os
import cv2

img_names = os.listdir(r"..\jaffe")
i = 0
for name in img_names:
    n = os.path.join(r"..\jaffe", name)
    img = cv2.imread(n)
    p = name.split(".")[0]

    save_path = os.path.join(r"..\faces", p)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, str(i)+".bmp")
    # print(save_path)
    cv2.imwrite(save_path, img)
    i += 1