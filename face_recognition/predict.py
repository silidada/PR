#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/17 20:46
# @Author  : Chen HanJie
# @FileName: predict.py
# @Software: PyCharm

import sys
sys.path.append("util")
import torch
import argparse
from util.load_model import load_model
import numpy as np
import math
import cv2
import os


def read_img(path):
    if path.endswith(".png") or path.endswith(".jpg") or path.endswith(".bmp"):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = np.fromfile(opt.image, dtype='uint8')
        n = int(math.sqrt(img.shape[0]))
        img = img.reshape(n, n, 1)
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gender recognize evaluate.')
    parser.add_argument('--Model', type=str, default='Model',
                        help='The Model you want to use, such as Model, Model_wise, Model_simple etc.')
    parser.add_argument('--model_path', type=str, default="./model/best.pt",
                        help='Model parameter save path.')
    parser.add_argument('--cuda', action="store_true",
                        help='Do you want to use the cuda?')
    parser.add_argument('--cuda_id', type=int, default=0,
                        help='Do you want to use the cuda?')
    parser.add_argument('--image', type=str, default="./test",
                        help='test image path')
    parser.add_argument('--image_list', action="store_true",
                        help='do you want to predict a list of image')
    parser.add_argument('--dir', type=str, default="./face/rawdata",
                        help='do you want to predict a list of image')
    opt = parser.parse_args()

    print(opt)

    if opt.cuda:
        if torch.cuda.is_available():
            device = torch.device('cuda'+':'+str(opt.cuda_id))
        else:
            print("cuda is not available. Now use the cpu or you can press 'ctrl+C' to pulse it")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    model, img_size = load_model(opt)
    model = model.to(device)
    model.eval()

    state_dict = torch.load(opt.model_path, map_location=device)
    model.load_state_dict(state_dict)

    if not opt.image_list:
        img = read_img(opt.image)
        img = cv2.resize(img, (img_size, img_size))

        img_tensor = torch.tensor(np.array([img]), dtype=torch.float)

        img_tensor = img_tensor.unsqueeze(1)
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            result = model(img_tensor)

        result = result.squeeze()
        pred1 = result.argmax(dim=0)

        img = cv2.putText(img, 'female' if pred1.item() else 'male', (0,10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (255, 255, 255), thickness = 1)
        cv2.imshow('PR demo', img)
        cv2.waitKey(0)
    else:
        img_list = os.listdir(opt.dir)

        for img_path in img_list:
            img = read_img(img_path)
            img = cv2.resize(img, (img_size, img_size))

            img_tensor = torch.tensor(np.array([img]), dtype=torch.float)

            img_tensor = img_tensor.unsqueeze(1)
            img_tensor = img_tensor.to(device)
            with torch.no_grad():
                result = model(img_tensor)

            result = result.squeeze()
            pred1 = result.argmax(dim=0)

            img = cv2.putText(img, 'female' if pred1.item() else 'male', (0, 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                              fontScale=0.5, color=(255, 255, 255), thickness=1)
            cv2.imshow('PR demo', img)

            cv2.waitKey(1)
            # print(img_path)
