#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/12 17:38
# @Author  : Chen HanJie
# @FileName: train.py
# @Software: PyCharm
import sys

sys.path.append(".")
sys.path.append("./utils")
import torch
from torch import nn
from utils.label_pre import label_pre_one_hot
from utils.raw_data_read import read_rawdata
from model import Model, Model_simple, Model_simple64
from torch import optim
from tqdm import tqdm
import numpy as np
import cv2


def load_batch_data(label, path_rawdata, img_name, batch_size, step, steps, dsize):
    if step == steps - 1:
        batch_label = label[step * batch_size:-1]
        name = img_name[step * batch_size:-1]
    else:
        batch_label = label[step * batch_size:(step + 1) * batch_size]
        name = img_name[step * batch_size:(step + 1) * batch_size]
    img_list = read_rawdata(1, path_rawdata, name, dsize)

    img_list_ = []
    label_list = []

    for i in range(len(img_list)):
        if img_list[i] is None:
            continue
        if batch_label[i] is None:
            continue
        img_list_.append(img_list[i])
        label_list.append(batch_label[i])

    img_list_ = np.array(img_list_)
    label_list = np.array(label_list)
    np.random.seed(5)
    np.random.shuffle(img_list_)
    np.random.seed(5)
    np.random.shuffle(label_list)
    np.random.seed(11)
    np.random.shuffle(img_list_)
    np.random.seed(11)
    np.random.shuffle(label_list)
    # np.random.seed(2)
    # np.random.shuffle(img_list_)
    # np.random.seed(2)
    # np.random.shuffle(label_list)
    # np.random.seed(3)
    # np.random.shuffle(img_list_)
    # np.random.seed(3)
    # np.random.shuffle(label_list)

    img_tensor = torch.tensor(img_list_, dtype=torch.float) / 255.
    label_tensor = torch.tensor(label_list, dtype=torch.float)
    return img_tensor, label_tensor


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    lr = 0.1
    epochs = 500
    batch_size = 256
    img_szie = 128

    path_rawdata = "./face/rawdata"
    path_label1 = "./face/faceDR"
    path_label2 = "./face/faceDS"
    label_one_hot, img_name, label_name = label_pre_one_hot(path_label1, path_label2)
    train_label_all = label_one_hot[:3600]
    train_img_name_all = img_name[:3600]
    train_label_all_list = []
    train_img_name_all_list = []
    for i in range(10):
        train_img_name_all_list.append(train_img_name_all[360*i:360*(i+1)])
        train_label_all_list.append(train_label_all[360*i:360*(i+1)])
    # train_label = label_one_hot[:3200]
    # val_label = label_one_hot[3200:3600]
    test_label = label_one_hot[3600:4000]
    # train_img_name = img_name[:3200]
    # val_img_name = img_name[3200:3600]
    test_img_name = img_name[3600:4000]
    train_data_size = len(train_label_all)*9/10
    steps = train_data_size // batch_size
    if not train_data_size % batch_size == 0:
        steps += 1
    steps = int(steps)

    val_steps = (len(train_label_all)*1/10) // batch_size
    if not (len(train_label_all)*1/10) % batch_size == 0:
        val_steps += 1
    val_steps = int(val_steps)

    eval_steps = (len(test_label)) // batch_size
    if not len(test_label) % batch_size == 0:
        eval_steps += 1
    eval_steps = int(eval_steps)

    model = Model_simple().to(device)
    opt = optim.SGD(model.parameters(), lr=lr)
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.98)
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = nn.MSELoss()
    loss_fn1 = nn.MSELoss(reduction='none')
    best_acc, best_epoch = 0, 0
    global_step = 0

    acc = 0
    loss_ = 0
    corr_val = 0
    corr_train = 0
    acc_train = 0
    acc_eval = 0

    rooo = 0
    corr_eval=0
    for epoch in range(epochs):
        val_label = []
        val_img_name = []
        train_label = []
        train_img_name = []
        corr_train = 0
        for i in range(10):
            if i == rooo:
                val_label = train_label_all_list[i]
                val_img_name = train_img_name_all_list[i]
            else:
                train_label += train_label_all_list[i]
                train_img_name += train_img_name_all_list[i]
        rooo += 1
        if rooo == 10:
            rooo=0
        # train_label = label_one_hot[:3200]
        # val_label = label_one_hot[3200:3600]
        # test_label = label_one_hot[3600:4000]
        # train_img_name = img_name[:3200]
        # val_img_name = img_name[3200:3600]
        # print(len(val_label), len(train_label))
        with tqdm(total=len(range(steps))) as _tqdm:  # 使用需要的参数对tqdm进行初始化

            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, epochs))
            for step in range(steps):
                img_tensor, label_tensor = load_batch_data(train_label, path_rawdata, train_img_name, batch_size, step,
                                                           steps, img_szie)
                img_tensor = img_tensor.unsqueeze(1)
                img_tensor = img_tensor.to(device)
                label_tensor = label_tensor.to(device)
                # print(label_tensor[:,:2])
                # print(img_tensor.size())
                result = model(img_tensor)
                result = result.squeeze()

                result1 = result[:, 0:2]
                # result2 = result[:, 2:6]
                # result3 = result[:, 6:11]
                # result4 = result[:, 11:14]
                # result5 = result[:, 14:-1]
                label1 = label_tensor[:, 0:2]
                # label1[label1[:,1] == 1] += 5
                # label2 = label_tensor[:, 2:6]
                # label3 = label_tensor[:, 6:11]
                # label4 = label_tensor[:, 11:14]
                # label5 = label_tensor[:, 14:-1]

                loss1 = loss_fn(result1, label1)
                # loss2 = loss_fn(result2, label2)
                # loss3 = loss_fn(result3, label3)
                # loss4 = loss_fn(result4, label4)
                # loss5 = loss_fn1(result5, label5)
                loss_total = loss1# + loss2 + loss3 + loss4 + loss5
                # print(loss_fn1(result1, label1))
                opt.zero_grad()
                loss_total.backward()
                opt.step()
                global_step += 1

                pred1 = result1.argmax(dim=1)
                real1 = label1.argmax(dim=1)

                corr_train += torch.eq(pred1, real1).sum().float().item()

                loss_ = loss_total.data
                # if loss_ < 0.05:
                #     print(result)
                #     print(label1)
                _tqdm.set_postfix({"train_acc":acc_train, "val_acc": corr_val, "eval_acc": corr_eval,"losss": loss_})
                _tqdm.update(1)

        acc_train = corr_train / len(train_img_name)
        corr_val = 0
        for step in range(val_steps):
            img_tensor, label_tensor = load_batch_data(val_label, path_rawdata, val_img_name, batch_size, step,
                                                       val_steps, img_szie)
            img_tensor = img_tensor.unsqueeze(1)
            img_tensor = img_tensor.to(device)
            label_tensor = label_tensor.to(device)
            with torch.no_grad():
                result = model(img_tensor)
            result = result.squeeze()

            # result[result < 0.5] = 0
            # result[result >= 0.5] = 1

            x1 = result[:, 0:2]
            y1 = label_tensor[:, 0:2]

            pred1 = x1.argmax(dim=1)
            real1 = y1.argmax(dim=1)

            corr_val += torch.eq(pred1, real1).sum().float().item()

        if epoch%5==0:
            corr_eval = 0.
            for step in range(eval_steps):
                img_tensor, label_tensor = load_batch_data(test_label, path_rawdata, test_img_name, batch_size, step,
                                                           eval_steps, img_szie)
                img_tensor = img_tensor.unsqueeze(1)
                img_tensor = img_tensor.to(device)
                label_tensor = label_tensor.to(device)
                with torch.no_grad():
                    result = model(img_tensor)
                result = result.squeeze()

                # result[result < 0.5] = 0
                # result[result >= 0.5] = 1

                x1 = result[:, 0:2]
                y1 = label_tensor[:, 0:2]

                pred1 = x1.argmax(dim=1)
                real1 = y1.argmax(dim=1)

                corr_eval += torch.eq(pred1, real1).sum().float().item()
            corr_eval /= len(test_img_name)
        corr_val /= len(val_img_name)

        lr_sch.step()
