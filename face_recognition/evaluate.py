#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/17 19:51
# @Author  : Chen HanJie
# @FileName: evaluate.py
# @Software: PyCharm
import sys
sys.path.append("util")
import torch
import argparse
from util.load_model import load_model
from util.label_pre import label_pre_one_hot
from util.load_batch_data import load_batch_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gender recognize evaluate.')
    parser.add_argument('--Model', type=str, default='Model',
                        help='The Model you want to use, such as Model, Model_wise, Model_simple etc.')
    parser.add_argument('--label_input', type=str, default="./face/faceDR",
                        help='The dataset origin path.')
    parser.add_argument('--raw_data_dir', type=str, default="./face/rawdata",
                        help='The raw data directory.')
    parser.add_argument('--model_path', type=str, default="./model/best.pt",
                        help='Model parameter save path.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='The training batch size, it is up to your memory.')
    parser.add_argument('--cuda', action="store_true",
                        help='Do you want to use the cuda?')
    parser.add_argument('--cuda_id', type=int, default=0,
                        help='Do you want to use the cuda?')
    opt = parser.parse_args()

    print(opt)

    path_rawdata = opt.raw_data_dir
    path_label = opt.label_input
    batch_size = opt.batch_size

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

    state_dict = torch.load(opt.model_path, map_location=device)
    model.load_state_dict(state_dict)

    label_one_hot, img_name, label_name = label_pre_one_hot([path_label])
    print(label_name)

    data_len = len(label_one_hot)
    steps = data_len // batch_size
    if not data_len % batch_size == 0:
        steps += 1

    corr_eval = 0
    eval_size = 0
    for step in range(steps):
        img_tensor, label_tensor = load_batch_data(label_one_hot, path_rawdata, img_name, batch_size, step,
                                                   steps, img_size)
        img_tensor = img_tensor.unsqueeze(1)
        img_tensor = img_tensor.to(device)
        label_tensor = label_tensor.to(device)
        with torch.no_grad():
            result = model(img_tensor)
        result = result.squeeze()

        x1 = result[:, 0:2]
        y1 = label_tensor[:, 0:2]

        pred1 = x1.argmax(dim=1)
        real1 = y1.argmax(dim=1)

        corr_eval += torch.eq(pred1, real1).sum().float().item()
        eval_size += img_tensor.size(0)
    acc = corr_eval / eval_size
    print(acc)
