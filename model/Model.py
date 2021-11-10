# Copyright (c) 2021 Dai HBG

"""
该文档定义BiLSTM模型

开发日志
2021-11-10
-- 定义模型
"""

import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, chord, predict_chord, mask):  # 计算被mask掉的地方的预测的分类损失
        return torch.mean(-torch.log(chord * predict_chord))  # 需要改


class AutoChordNet(nn.Module):
    def __init__(self, melody_keys=128, chord_num=96, hidden_size=128, num_layers=2, device='cpu'):
        super(AutoChordNet, self).__init__()

        self.device = device
        self.melody_keys = melody_keys  # melody_keys设置为128，也就是128个维度来表征一个曲子；实际上可以缩减为12或者24之类的以降维
        self.chord_num = chord_num  # chord_num是使用one_hot编码的和弦数量
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_1 = nn.LSTM(input_size=self.melody_keys+self.chord_num+1,  # 加上一个mask的维度
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              batch_first=True,
                              dropout=0.2,
                              bidirectional=True)

        self.lstm_2 = nn.LSTM(input_size=self.hidden_size*2,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              batch_first=True,
                              dropout=0.2,
                              bidirectional=True)

        self.linear = nn.Linear(self.hidden_size * 2 * self.num_layers + self.melody_keys + self.chord_num + 1, self.chord_num)

    def forward(self, x):  # 提前拼接好所需的东西
        y, (h_n, c_n) = self.lstm_1(x)
        h_n = h_n.flatten(start_dim=1)
        y, (h_n, c_n) = self.lstm_2(h_n)
        y = h_n.flatten(start_dim=1)  # 取出最后一个时间步
        z = torch.cat([x, y], dim=-1)
        z = self.linear(z)
        return z


def trainer(epochs=1000):  # 训练函数，需要生成掩盖函数
    pass


