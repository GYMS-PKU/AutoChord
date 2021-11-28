# Copyright (c) 2021 Dai HBG

"""
该文档定义BiLSTM模型

开发日志
2021-11-10
-- 定义模型
2021-11-23
-- 定义Loss，更改模型结构为给定已有melody和chord预测下一个chord
2021-11-27
-- 新增模型父类，统一对外接口，给定已有的序列，给出下一个和弦的概率分布
"""

import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from time import time


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
        self.lstm_1 = nn.LSTM(input_size=self.melody_keys + self.chord_num,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              batch_first=True,
                              dropout=0.2,
                              bidirectional=bidirectional)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dim1 = 2 if bidirectional else 1
        self.Dense1 = nn.Linear(self.hidden_size * self.dim1 * self.num_layers + self.melody_keys + self.chord_num,
                                self.chord_num * 2)
        self.Dense2 = nn.Linear(self.chord_num * 2, self.chord_num)

    def forward(self, chord, melody, x):
        """
        :param chord: chord序列，batch_size * seq_length * chord_num
        :param melody: melody序列，batch_size * seq_length * melody_keys
        :param x: 下一个melody，batch_size * melody_keys
        :return: 下一个chord的分类概率
        """
        ipt = torch.cat([chord, melody], dim=-1)  # 得到batch_size * seq_length * (chord_num + melody_keys)
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


class MyDeepModel:  #
    def __init__(self, chord_num=42, melody_keys=12, hidden_size=128, num_layers=2, bidirectional=False,
                 alpha=0.2, device='cpu', loss='ts'):
        """
        :param chord_num: 和弦个数
        :param melody_keys: 旋律表示的维度
        :param hidden_size:
        :param num_layers:
        :param bidirectional: 是否双向
        :param alpha: LeakyRelu参数
        :param device:
        :param loss: 损失函数
        """
        self.device = device
        self.melody_keys = melody_keys  # melody_keys设置为12，也就是12个维度来表征一个曲子
        self.chord_num = chord_num  # chord_num是使用one_hot编码的和弦数量
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.alpha = alpha
        if loss == 'ts':
            self.loss = F.nll_loss()
        elif loss == 'mask':
            self.loss = MyLoss()
        else:
            self.loss = F.nll_loss()


class MyChordLstmNet(MyDeepModel):  # 使用LSTM来预测下一个和弦
    def __init__(self, chord_num=42, melody_keys=12, hidden_size=128, num_layers=2, bidirectional=False,
                 alpha=0.2, device='cpu'):
        super(MyChordLstmNet, self).__init__(chord_num=chord_num, melody_keys=melody_keys, hidden_size=hidden_size,
                                             num_layers=num_layers, bidirectional=bidirectional,
                                             alpha=alpha, device=device, loss=loss)
        self.model = ChordLstmNet(chord_num=chord_num, melody_keys=melody_keys, hidden_size=hidden_size,
                                  num_layers=num_layers, bidirectional=bidirectional,
                                  alpha=alpha, device=device).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-3)

    def fit(self, train_data, test_data=None, epochs=10, batch_size=1000, verbose=True):
        """
        :param train_data: # 训练数据，为了统一对外接口，考虑到需要用到单一旋律预测，因此格式需要在内部自行处理
        :param test_data:
        :param epochs:
        :param batch_size:
        :param verbose:
        :return:
        """

        # 暂时一次性将所有东西搬到显存中，如果爆了就改成单独进去
        train_data = [(sample[0].unsqueeze(0).to(self.device), sample[1].unsqueeze(0).to(self.device),
                       sample[2].unsqueeze(0).to(self.device), sample[3].unsqueeze(0).to(self.device))
                      for sample in train_data]
        test_data = [(sample[0].unsqueeze(0).to(self.device), sample[1].unsqueeze(0).to(self.device),
                      sample[2].unsqueeze(0).to(self.device), sample[3].unsqueeze(0).to(self.device))
                     for sample in test_data]
        for epoch in range(epochs):
            num = 0
            loss = 0
            t = time()
            self.model.train()
            for sample in train_data:
                if num < batch_size:
                    out = model(sample[0], sample[1], sample[2])
                    loss += F.nll_loss(out, sample[3])
                    num += 1
                else:
                    num = 0
                    loss /= batch_size
                    if verbose:
                        print('loss: {:.4f} time used: {:.4f}s'.format(loss, time() - t))
                    train_loss.append(loss)
                    loss.backward()
                    loss = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    t = time()
            if verbose:
                print('epoch {} testing'.format(epoch))
            if test_data is not None:
                loss = 0
                with torch.no_grad():
                    for sample in test_data:
                        a = sample[0].unsqueeze(0).to(device)
                        b = sample[1].unsqueeze(0).to(device)
                        c = sample[2].unsqueeze(0).to(device)
                        out = model(a, b, c)
                        loss += F.nll_loss(out, sample[3].unsqueeze(0).to(device))

                    loss /= len(test_data)
                    test_loss.append(loss)
                    if verbose:
                        print('test loss: {:.4f} time used: {:.4f}s'.format(loss, time() - t))

    def predict(self, chord, melody, x, single=True):
        """
        :param chord: num * chord_num的二维np.arrray
        :param melody: num * key_num的二维np.array
        :param x: key_num的一维melody
        :param single: 是否返回一维概率
        :return:
        """
        self.model.eval()
        if len(chord.shape) == 2:
            out = self.model(torch.tensor(chord).unsqueeze(0).to(self.device),
                             torch.tensor(melody).unsqueeze(0).to(self.device),
                             torch.tensor(x).unsqueeze(0).to(self.device))
        else:
            out = self.model(torch.tensor(chord).to(self.device),
                             torch.tensor(melody).to(self.device),
                             torch.tensor(x).to(self.device))
        if single:
            return out[0].cpu().numpy()
        else:
            return out.cpu().numpy()


class MyMarkovChain(MyDeepModel):
    def __init__(self):
        super(MyMarkovChain, self).__init__()
        self.transition_matrix = None

    def fit(self, train_data):  # 用训练数据生成转移概率矩阵
        """
        :param train_data: 训练数据
        :return:
        """
        transition_matrix = np.zeros((train_data[0][0].shape[1], train_data[0][0].shape[1]))
        for sample in train_data:
            i = np.argsort(sample[0][-1].numpy())[-1]  # 最后一步的和弦编号
            transition_matrix[i, sample[3][0]] += 1
        for i in range(transition_matrix.shape[0]):
            transition_matrix[i] = transition_matrix[i] / np.sum(transition_matrix[i])
        self.transition_matrix = transition_matrix

    def predict(self, chord_index):
        """
        :param chord_index: 和弦编号
        :return:
        """
        return self.transition_matrix[chord_index]
