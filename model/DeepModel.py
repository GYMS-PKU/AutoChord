# Copyright (c) 2021 Dai HBG

"""
该文档定义BiLSTM模型

开发日志
2021-11-10
-- 定义模型
2021-11-23
-- 定义Loss，更改模型结构为给定已有melody和chord预测下一个chord
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

    def forward(self, predict_chord, chord, mask):  # 计算被mask掉的地方的预测的分类损失
        """
        :param predict_chord: 预测的chord，一个chord数量维度的输出
        :param chord: 真实的chord，是一个long类型的分类
        :param mask: 是一个列表，表示需要计算loss的那些
        :return:
        """
        return F.nll_loss(predict_chord[mask], chord[mask])


class ChordLstmNet(nn.Module):  # 根据时序chord和melody预测下一个chord的分类
    def __init__(self, chord_num=42, melody_keys=12, hidden_size=128, num_layers=2, bidirectional=False,
                 alpha=0.2, device='cpu'):
        """
        :param chord_num: 和弦个数
        :param melody_keys: 旋律表示的维度
        :param hidden_size:
        :param num_layers:
        :param bidirectional: 是否双向
        :param alpha: LeakyRelu参数
        :param device:
        """
        super(ChordLstmNet, self).__init__()
        self.device = device
        self.melody_keys = melody_keys  # melody_keys设置为12，也就是12个维度来表征一个曲子
        self.chord_num = chord_num  # chord_num是使用one_hot编码的和弦数量
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_1 = nn.LSTM(input_size=self.melody_keys + self.chord_num + 1,  # 加上一个mask的维度
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              batch_first=True,
                              dropout=0.2,
                              bidirectional=bidirectional)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.Dense1 = nn.Linear(self.hidden_size * 2 * self.num_layers + self.melody_keys + self.chord_num,
                                self.chord_num * 2)
        self.Dense2 = nn.Linear(self.chord_num * 2, self.chord_num)

    def forward(self, chord, melody, x):
        """
        :param chord: chord序列，batch_size * seq_length * chord_num
        :param melody: melody序列，batch_size * seq_length * melody_keys
        :param x: 下一个melody，batch_size * melody_keys
        :return: 下一个chord的分类概率
        """
        ipt = torch.cat([chord, melody], dim=2)  # 得到batch_size * seq_length * (chord_num + melody_keys)
        y, (h_n, c_n) = self.lstm_1(ipt)
        h_n = h_n.transpose(1, 0)
        # h_n是(batch_size, num_layers * direction_num, hidden_size)的矩阵
        ipt = torch.cat([h_n.flatten(start_dim=1), chord[:, -1, :], x], dim=-1)
        # 拼接LSTM最后一个时间步的输出，最后一个chord，melody
        y = self.leakyrelu(self.Dense1(ipt))
        y = self.Dense2(y)
        return F.log_softmax(y, dim=1)


class AutoChordNet(nn.Module):
    def __init__(self, melody_keys=12, chord_num=96, hidden_size=128, num_layers=2, device='cpu'):
        super(AutoChordNet, self).__init__()

        self.device = device
        self.melody_keys = melody_keys  # melody_keys设置为12，也就是12个维度来表征一个曲子
        self.chord_num = chord_num  # chord_num是使用one_hot编码的和弦数量
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_1 = nn.LSTM(input_size=self.melody_keys + self.chord_num + 1,  # 加上一个mask的维度
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              batch_first=True,
                              dropout=0.2,
                              bidirectional=True)

        self.lstm_2 = nn.LSTM(input_size=self.hidden_size * 2,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              batch_first=True,
                              dropout=0.2,
                              bidirectional=True)

        self.linear = nn.Linear(self.hidden_size * 2 * self.num_layers + self.melody_keys + self.chord_num + 1,
                                self.chord_num)

    def forward(self, x):  # 提前拼接好所需的东西
        y, (h_n, c_n) = self.lstm_1(x)
        h_n = h_n.transpose(1, 0)
        h_n = h_n.flatten(start_dim=1)
        y, (h_n, c_n) = self.lstm_2(h_n)
        h_n = h_n.transpose(1, 0)
        y = h_n.flatten(start_dim=1)  # 取出最后一个时间步
        z = torch.cat([x, y], dim=-1)
        z = self.linear(z)
        return z
