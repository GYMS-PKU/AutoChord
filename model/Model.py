# Copyright (c) 2021 Dai HBG

"""
该文档定义BiLSTM模型
"""

import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable


class AutoChordNet(nn.Module):
    def __init__(self, melody_keys=128, chord_num=96, hidden_size=128, num_layers=2, device='cpu'):
        super(AutoChordNet, self).__init__()

        self.device = device
        self.melody_keys = melody_keys  # melody_keys设置为128，也就是128个维度来表征一个曲子；实际上可以缩减为12或者24之类的以降维
        self.chord_num = chord_num  # chord_num是使用one_hot编码的和弦数量
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_1 = nn.LSTM(input_size=self.melody_keys+self.chord_num,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              batch_first=True,
                              dropout=0.2,
                              bidirectional=True)

        self.lstm_1 = nn.LSTM(input_size=self.hidden_size*2,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              batch_first=True,
                              dropout=0.2,
                              bidirectional=True)

        self.linear = nn.Linear(self.hidden_size * 2 + self.melody_keys + self.chord_num, self.chord_num)

    def forward(self):
        pass
