#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/12 17:38
# @Author  : Chen HanJie
# @FileName: train.py
# @Software: PyCharm
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.append("../..")
sys.path.append("./utils")
import torch
from torch import nn
from utils.label_pre import label_pre_one_hot
from utils.raw_data_read import read_rawdata
from model import Model, Model_simple, Model_simple64, Model_simpleFCN, Model_depth_wise, Model_depth_wise_fc
from torch import optim
from tqdm import tqdm
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random


def load_batch_data(label, path_rawdata, img_name, batch_size, step, steps, dsize, add_noisy=False):
    if step == steps - 1:
        batch_label = label[step * batch_size:-1]
        name = img_name[step * batch_size:-1]
    else:
        batch_label = label[step * batch_size:(step + 1) * batch_size]
        name = img_name[step * batch_size:(step + 1) * batch_size]

    ii = random.randint(0,200)

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
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    lr = 0.2
    epochs = 1000
    batch_size = 256
    img_szie = 128
    model_save_path = "./model"
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    path_rawdata = "./face/rawdata"
    path_label1 = "./face/faceDR"
    path_label2 = "./face/faceDS"
    label_one_hot, img_name, label_name = label_pre_one_hot(path_label1, path_label2)

    random.seed(100)
    random.shuffle(label_one_hot)
    random.seed(100)
    random.shuffle(img_name)

    label_size_factor = int(len(label_one_hot) // 10)
    train_label_all = label_one_hot[:label_size_factor*9]
    train_img_name_all = img_name[:label_size_factor*9]
    train_label_all_list = []
    train_img_name_all_list = []

    train_label_size_factor = int(len(train_label_all) // 10)

    # print(train_label_size_factor, len(train_img_name_all))

    for i in range(10):
        train_img_name_all_list.append(train_img_name_all[train_label_size_factor*i:train_label_size_factor*(i+1)])
        train_label_all_list.append(train_label_all[train_label_size_factor*i:train_label_size_factor*(i+1)])
    # train_label = label_one_hot[:3200]
    # val_label = label_one_hot[3200:3600]
    test_label = label_one_hot[label_size_factor*9:]
    # train_img_name = img_name[:3200]
    # val_img_name = img_name[3200:3600]
    test_img_name = img_name[label_size_factor*9:]
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
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)
    # loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss()
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
    eval_acc_list = []
    val_acc_list = []
    train_acc_list = []
    train_loss = []

    # print(len(train_img_name_all_list[-1]))


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
            train_size = 0
            for step in range(steps):
                img_tensor, label_tensor = load_batch_data(train_label, path_rawdata, train_img_name, batch_size, step,
                                                           steps, img_szie, add_noisy=False)
                # print(train_label)
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
                train_size += img_tensor.size(0)

                loss_ = loss_total.data

                # if loss_ < 0.05:
                #     print(result)
                #     print(label1)
                _tqdm.set_postfix({"train_acc":acc_train, "val_acc": corr_val, "eval_acc": corr_eval,"losss": loss_})
                _tqdm.update(1)
        train_loss.append(loss_.item())
        acc_train = corr_train / train_size
        train_acc_list.append(acc_train)
        corr_val = 0
        val_size = 0
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
            val_size += img_tensor.size(0)

        corr_eval = 0.
        eval_size = 0
        for step in range(eval_steps):
            img_tensor, label_tensor = load_batch_data(test_label, path_rawdata, test_img_name, batch_size, step,
                                                       eval_steps, img_szie)
            img_tensor = img_tensor.unsqueeze(1)
            img_tensor = img_tensor.to(device)
            # print(img_tensor.shape, label_tensor.shape)
            label_tensor = label_tensor.to(device)
            with torch.no_grad():
                result = model(img_tensor)
            result = result.squeeze()

            # result[result < 0.5] = 0
            # result[result >= 0.5] = 1

            x1 = result[:, 0:2]
            y1 = label_tensor[:, 0:2]

            # print(y1)

            pred1 = x1.argmax(dim=1)
            real1 = y1.argmax(dim=1)

            corr_eval += torch.eq(pred1, real1).sum().float().item()
            eval_size += img_tensor.size(0)

        corr_eval /= eval_size
        if corr_eval > best_acc:
            torch.save(model.state_dict(), os.path.join(model_save_path, "best0.pt"))
        eval_acc_list.append(corr_eval)

        corr_val /= val_size
        val_acc_list.append(corr_val)
        lr_sch.step()


    x = [i for i in range(len(train_loss))]
    fig = plt.figure(figsize=(10,5))
    plt.plot(x, train_loss, c='blue', label='train loss')
    plt.legend(loc='best')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("train")
    plt.savefig("./result1.png")
    plt.clf()
    # fig = plt.figure(figsize=(10, 5))
    x = [i for i in range(len(train_acc_list))]
    plt.plot(x, train_acc_list, c='red', label='train accuracy')

    x = [i for i in range(len(val_acc_list))]
    plt.plot(x, val_acc_list, c='blue', label='valid accuracy')

    plt.plot(x, eval_acc_list, c='green', label='evaluate accuracy')
    plt.legend(loc='best')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title("train")
    # fig.set_xlabel('epoch')
    plt.savefig("./result2.png")

    data = torch.randn(2, 1, img_szie, img_szie).to(device)

    # 导出为onnx格式
    torch.onnx.export(
        model,
        data,
        'model.onnx',
        export_params=True,
        opset_version=8,
    )
