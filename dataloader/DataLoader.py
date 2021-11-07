# Copyright (c) 2021 Dai HBG

"""
该文档主要定义读取midi数据，转为标准numpy格式并存储
考虑先每一个样本存一个文件，然后所有样本统一存储为一个大文件，以加速读取
"""

import numpy as np
import pickle
import pypianoroll as pr
import xml.etree.ElementTree as et
import os


class DataLoader:
    def __init__(self, raw_data_path=None, processed_data_path=None):
        if raw_data_path is None:
            raw_data_path = 'F:/Documents/学习资料/自动配和弦/datasets'
        self.raw_data_path = raw_data_path
        if processed_data_path is None:
            processed_data_path = 'F:/Documents/学习资料/自动配和弦/datasets/processed_data'
        self.processed_data_path = processed_data_path
        if os.path.isfile(self.processed_data_path):
            os.makedirs('F:/Documents/学习资料/自动配和弦/datasets/processed_data')

    def process_raw_data(self):  # 将已有数据处理成dic，dic['key']为调号，dic['melody']为旋律，dic['chord']为和弦，name为名字
        processed_num = 0
        lst = []
        for i in os.listdir('{}/pianoroll'.format(self.raw_data_path)):
            for j in os.listdir('{}/pianoroll/{}'.format(self.raw_data_path, i)):
                for p in os.listdir('{}/pianoroll/{}/{}'.format(self.raw_data_path, i, j)):
                    for k in os.listdir('{}/pianoroll/{}/{}/{}'.format(self.raw_data_path, i, j, p)):
                        if k[-9:] == 'nokey.mid':
                            try:
                                dic = {}
                                mt = pr.read('{}/pianoroll/{}/{}/{}/{}'.format(self.raw_data_path, i, j, p, k))  # 拉回C或c
                                # print('{}/pianoroll/{}/{}/{}/{}'.format(self.raw_data_path, i, j, p, k))
                                # print(mt)
                                dic['melody'] = mt.tracks[0].pianoroll
                                dic['chord'] = mt.tracks[1].pianoroll
                                tree = et.parse('{}/xml/{}/{}/{}/{}.xml'.format(self.raw_data_path, i, j, p, k[:-10]))
                                root = tree.getroot()
                                meta = root.find('meta')
                                key = meta.find('key')
                                dic['key'] = key.text
                                dic['name'] = k[:-10]
                                processed_num += 1
                                lst.append(dic)
                            except:
                                pass
            print('{} done. total {} samples now'.format(i, processed_num))
        print('writing processed data')
        with open('{}/processed_data.pkl'.format(self.processed_data_path), 'wb') as f:
            pickle.dump(lst, f)
        print('done.')
