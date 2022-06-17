#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/17 19:54
# @Author  : Chen HanJie
# @FileName: load_model.py
# @Software: PyCharm

import sys
sys.path.append("..")
from model import Model, Model_simple, Model_simple64, Model_simpleFCN, Model_depth_wise, Model_depth_wise_fc


def load_model(opt):
    img_size = 128
    if opt.Model == 'Model':
        model = Model()
    elif opt.Model == 'Model_depth_wise':
        model = Model_depth_wise()
    elif opt.Model == 'Model_simple':
        model = Model_simple()
    elif opt.Model == 'Model_simple64':
        model = Model_simple64()
        img_size = 64
    elif opt.Model == 'Model_simpleFCN':
        model = Model_simpleFCN()
    elif opt.Model == 'Model_depth_wise_fc':
        model = Model_depth_wise_fc()
    else:
        print("no such model:", opt.Model)
        exit(0)
    return model, img_size

if __name__ == '__main__':
    pass
