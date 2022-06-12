#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/12 12:12
# @Author  : Chen HanJie
# @FileName: label_pre.py
# @Software: PyCharm

import re

def read_label(path):
    with open(path, "r") as f:
        r = f.readline()
        while r:
            t = re.search('_missing descriptor', r)
            print(t)


if __name__ == '__main__':
    read_label("../face/faceDR")
