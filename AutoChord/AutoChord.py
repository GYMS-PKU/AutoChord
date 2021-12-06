# Copyright (c) 2021 Dai HBG


"""
该代码定义AutoChord类，该类集成所有生成和弦的方法

日志
2021-12-05
-- 新增整合的模型训练
"""

import sys

sys.path.append('../dataloader')
sys.path.append('../generator')
sys.path.append('../model')
from DataLoader import DataLoader
from generator import Generator
from DeepModel import *


class AutoChord:
    def __init__(self, raw_data_path=None, processed_data_path=None, device='cpu'):
        """
        :param raw_data_path: 原始数据路径
        :param processed_data_path: 已处理数据路径
        :param device: 设备
        """
        if raw_data_path is None:
            raw_data_path = '../dataset'
        self.raw_data_path = raw_data_path
        if processed_data_path is None:
            processed_data_path = '../dataset/processed_data'
        self.processed_data_path = processed_data_path
        self.device = device

        self.dataloader = DataLoader(raw_data_path=raw_data_path, processed_data_path=processed_data_path,
                                     device=device)
        self.generator = {'major': Generator.ChordGenerator(global_num_chord_dic=
                                                            self.dataloader.global_num_chord_dic['major'],
                                                            global_num_chord_dic_one_hot=
                                                            self.dataloader.global_num_chord_one_hot_dic['major']),
                          'minor': Generator.ChordGenerator(global_num_chord_dic=
                                                            self.dataloader.global_num_chord_dic['minor'],
                                                            global_num_chord_dic_one_hot=
                                                            self.dataloader.global_num_chord_one_hot_dic['minor'])}

        self.model = None

    def get_model(self, model_name=None, params=None):
        """
        :param model_name: 模型名字
        :param params: 参数
        :return:
        """
        if model_name is None:
            model_name = 'lstm'
        if model_name == 'lstm':
            if params is None:
                params = {'chord_num': 48, 'device': 'cpu'}
            self.model = MyChordLstmNet(chord_num=params['chord_num'], device=params['device'])

    def fit(self, train_data, test_data=None, epochs=10, batch_size=1000, verbose=True, shuffle=True):
        """
        :param train_data: list[tuple]，训练集合
        :param test_data: list[tuple]，测试集合
        :param epochs:
        :param batch_size:
        :param verbose:
        :param shuffle:
        :return:
        """
        train_loss, test_loss = self.model.fit(train_data=train_data, test_data=test_data, epochs=epochs,
                                               batch_size=batch_size, verbose=verbose, shuffle=shuffle)
        return train_loss, test_loss

    def generate(self, melody, tonic='major', method='back'):  # 给定旋律生成chord
        """
        :param melody: 旋律，array，目前限定成模12，也就是0到11之间，不管八度
        :param tonic: 调性
        :param method: 生成方法，默认是回溯，可选global_markov，lstm
        :return: 返回根据和弦字典的编号序列
        """
        if method == 'back':
            return self.generator[tonic].generate(melody, method)
        elif method == 'lstm':
            if self.model is None:
                print('no model yet')
                return
            return self.generator[tonic].generate(melody, method, model=self.model)
