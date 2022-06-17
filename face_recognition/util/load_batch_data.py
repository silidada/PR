#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/17 20:14
# @Author  : Chen HanJie
# @FileName: load_batch_data.py
# @Software: PyCharm
import torch
import random
from raw_data_read import read_rawdata
import numpy as np

def load_batch_data(label, path_rawdata, img_name, batch_size, step, steps, dsize, add_noisy=False):
    if step == steps - 1:
        batch_label = label[step * batch_size:-1]
        name = img_name[step * batch_size:-1]
    else:
        batch_label = label[step * batch_size:(step + 1) * batch_size]
        name = img_name[step * batch_size:(step + 1) * batch_size]

    ii = random.randint(0, 200)

    random.seed(ii)
    random.shuffle(label)
    random.seed(ii)
    random.shuffle(img_name)

    img_list = read_rawdata(1, path_rawdata, name, dsize, add_noisy=add_noisy)

    img_list_ = []
    label_list = []

    for i in range(len(img_list)):
        if img_list[i] is None:
            # print("img_list[i] is None")
            continue
        if batch_label[i] is None:
            # print(batch_label[i] is None)
            continue
        img_list_.append(img_list[i])
        label_list.append(batch_label[i])

    # print(len(img_list_))

    img_tensor = torch.tensor(np.array(img_list_), dtype=torch.float) / 255.
    label_tensor = torch.tensor(np.array(label_list), dtype=torch.float)
    return img_tensor, label_tensor

if __name__ == '__main__':
    pass
