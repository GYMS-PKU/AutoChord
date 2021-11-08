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

        if 'train_data.pkl' in os.listdir(self.processed_data_path):
            with open('{}/train_data.pkl'.format(self.processed_data_path), 'rb') as f:
                self.train_data = pickle.load(f)
        else:
            self.processed_data = None
            if 'processed_data.pkl' in os.listdir(self.processed_data_path):
                with open('{}/processed_data.pkl'.format(self.processed_data_path), 'rb') as f:
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

    def get_train_data(self):  # 得到每个和弦对应的旋律音
        if self.processed_data is None:
            print('no processed_data yet')
            return
        n = 0
        lst = []  # 类型和processed_data一致，但是每一个时间步都有和弦和一个旋律
        for value in self.processed_data:
            melody, chord = self.select_chord(value['melody'], value['chord'])
            if melody is None:
                continue
            tmp_dic = {'key': value['key'], 'name': value['name'], 'melody': melody, 'chord': chord}
            lst.append(tmp_dic)
            n += 1
            if n % 1000 == 0:
                print('{} valid samples'.format(n))
        print('total {} valid samples'.format(n))
        with open('{}/train_data.pkl'.format(self.processed_data_path), 'wb') as f:
            pickle.dump(lst, f)
        self.train_data = lst

    @staticmethod
    def select_chord(melody, chord):  # 从一个melody和chord序列筛选出纯的旋律和和弦序列
        chord_s = []  # 记录下需要的chord行指标
        melody_s = []  # 记录下需要的melody行指标
        chord_sum_num = np.sum(chord, axis=1)
        melody_sum_num = np.sum(melody, axis=1)
        i = 0
        still = False  # 标记是否是上一和弦的延续
        while i < len(melody):
            if chord_sum_num[i] > 0:
                if len(chord_s) > 0:  # 如果已经有和弦
                    if (np.sum(np.abs(chord[i] - chord[chord_s[-1]])) == 0) and still:  # 和目前正在延续的和弦相同，则跳过
                        i += 1
                        continue  # 跳过延续的和弦
                if melody_sum_num[i] > 0:  # 此时有和弦且有旋律
                    if len(chord_s) == 0:  # 这是第一个和弦，就直接放进去
                        chord_s.append(i)
                        melody_s.append(i)
                        i += 1
                        still = True  # 同时和弦开始延续
                    else:  # 否则只有新和弦才能进入
                        if np.sum(np.abs(chord[i] - chord[chord_s[-1]])) != 0:  # 如果和上一个和弦不相同就一定是新和弦
                            chord_s.append(i)
                            melody_s.append(i)
                            i += 1
                            still = True
                        else:  # 否则需要判断当前是不是延续状态
                            if not still:  # 虽然相同但不延续
                                chord_s.append(i)
                                melody_s.append(i)
                                i += 1
                                still = True
                            else:  # 此时还在延续
                                i += 1
                else:
                    i += 1
            else:
                i += 1  # 遇到空就一定结束和弦延续
                still = False  # 延续的和弦结束了
        if len(chord_s) >= 2:
            return melody[melody_s], chord[chord_s]
        else:
            return None, None
