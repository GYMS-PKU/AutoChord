# Copyright (c) 2021 Dai HBG


"""
该代码定义AutoChord类，该类集成所有生成和弦的方法
"""


import sys
sys.path.append('../dataloader')
sys.path.append('../generator')
sys.path.append('../model')
from DataLoader import DataLoader
from ChordGenerator import ChordGenerator


class AutoChord:
    def __init__(self, raw_data_path=None, processed_data_path=None, device='cpu'):
        """
        :param raw_data_path: 原始数据路径
        :param processed_data_path: 已处理数据路径
        :param device: 设备
        """
        if raw_data_path is None:
            raw_data_path = 'E:/Documents/学习资料/AutoChord/datasets'
        self.raw_data_path = raw_data_path
        if processed_data_path is None:
            processed_data_path = 'E:/Documents/学习资料/AutoChord/datasets/processed_data'
        self.processed_data_path = processed_data_path
        self.device = device

        self.dataloader = DataLoader(raw_data_path=raw_data_path, processed_data_path=processed_data_path,
                                     device=device)
        self.ChordGenerator = ChordGenerator()

    def generate(self, melody, method='back'):  # 给定旋律生成chord
        """
        :param melody: 旋律，array，目前限定成模12，也就是0到11之间，不管八度
        :param method: 生成方法，默认是回溯，可选global_markov，lstm
        :return: 返回根据和弦字典的编号序列
        """
        return self.ChordGenerator.generate(melody, methods)

