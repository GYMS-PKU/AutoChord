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

        if 'processed_data' not in os.listdir(self.raw_data_path):
            os.makedirs('F:/Documents/学习资料/自动配和弦/datasets/processed_data')

        if 'processed_data.pkl' is os.listdir(self.processed_data_path):
            with open('{}/processed_data.pkl', 'rb') as f:
                self.processed_data = pickle.load(f)
        else:
            self.processed_data = None

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

    def get_train_date(self):  # 得到每个和弦对应的旋律音
        if self.processed_data is None:
            print('no processed_data yet')
            return

        dic = {}  # 类型和processed_data一致，但是每一个时间步都有和弦和一个旋律

    @staticmethod
    def select_chord(melody, chord):  # 从一个melody和chord序列筛选出纯的旋律和和弦序列
        chord_s = []  # 记录下需要的chord行指标
        melody_s = []  # 记录下需要的melody行指标
        chord_sum_num = np.sum(chord, axis=1)
        melody_sum_num = np.sum(melody, axis=1)
        i = 0
        while i < len(melody):
            if chord_sum_num[i] > 0:
                if melody_sum_num[i] > 0:
                    chord_s.append(i)
                    melody_s.append(i)
                    i += 1
                else:  # 此时在有和弦的地方没有旋律音，需要向后找一个，找不到就退出
                    i_0 = i  # 记录下离下一个旋律音最近的chord
                    for j in range(i+1, len(melody)):
                        if melody_sum_num[j] > 0:
                            melody_s.append(j)
                            chord_s.append(i_0)
                            i = j+1  # 只要找到就退出
                            break
                        else:
                            if chord_sum_num[j] > 0:
                                i_0 = j  # 更新最近的和弦位置
                    if (j == len(melody)-1) and (i < j):
                        i = j+1
            else:
                i += 1
        if len(chord_s) >= 2:
            return melody[melody_s], chord[chord_s]
        else:
            return None
