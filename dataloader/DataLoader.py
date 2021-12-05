# Copyright (c) 2021 Dai HBG

"""
该文档主要定义读取midi数据，转为标准numpy格式并存储
考虑先每一个样本存一个文件，然后所有样本统一存储为一个大文件，以加速读取

开发日志
2021-11-08
-- 数据读取并存储为一个字典的方法，
2021-11-09
-- 压缩数据，只把有和弦的选出来
2021-11-10
-- 定义训练数据预处理方法，包括随机生成mask以及数据的拼接；需要解决不等长训练数据的问题
2021-11-15
-- 新增训练设备识别
-- 新增和弦识别，统计和弦总数，以及选择需要训练的和弦
2021-11-23
-- 新增构造按照多级和弦的大小调位置，然后统计和弦个数
-- 生成训练集时和弦转为tuple格式
2021-11-24
-- 新增时序训练集的构造方法，从double_compressed_data中构造
-- 更新：chord_dic改为chord_num_dic和num_chord_dic，整合在get_structure_chord_dic中
-- 更新：get_train_data方法更改，默认生成为单向lstm准备的训练数据，使用valid_double_compressed_data，形式为tuple
-- 更新：新增valid_double_compressed_data，需要筛选出structure_chord_dic中收录的部分
2021-12-01
-- 更新：需要生成global_num_chord_dic和global_chord_num_dic，并且保证structure_chord_dic中的数字编号和其一致
"""

import numpy as np
import pickle
import pypianoroll as pr
import xml.etree.ElementTree as et
import os
import torch

import sys
sys.path.append('C:/Users/Administrator/Desktop/Repositories/AutoChord/dataloader')

from chord_dic import *
from time import time


class DataLoader:
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
        self.chord_name_dic = chord_name_dic

        if 'processed_data' not in os.listdir(self.raw_data_path):
            os.makedirs('../dataset/processed_data')

        # 读取缓存训练数据
        if 'train_data.pkl' in os.listdir(self.processed_data_path):
            print('reading train_data')
            with open('{}/train_data.pkl'.format(self.processed_data_path), 'rb') as f:
                self.train_data = pickle.load(f)
        else:
            self.train_data = None

        # 读取chord_num_dic和num_chord_dic，注意这里是tuple表示
        if 'global_chord_num_dic.pkl' in os.listdir(self.processed_data_path):
            print('reading global_chord_num_dic')
            try:
                with open('{}/global_chord_num_dic.pkl'.format(self.processed_data_path), 'rb') as f:
                    self.global_chord_num_dic = pickle.load(f)
            except EOFError:
                print('global_chord_num_dic not found!')
                self.global_chord_num_dic = None
        else:
            self.global_chord_num_dic = None
        if 'global_num_chord_dic.pkl' in os.listdir(self.processed_data_path):
            print('reading global_num_chord_dic')
            try:
                with open('{}/global_num_chord_dic.pkl'.format(self.processed_data_path), 'rb') as f:
                    self.global_num_chord_dic = pickle.load(f)
            except EOFError:
                print('global_num_chord_dic not found!')
                self.global_num_chord_dic = None
        else:
            self.global_num_chord_dic = None
        if 'global_num_chord_one_hot_dic.pkl' in os.listdir(self.processed_data_path):
            print('reading global_num_chord_one_hot_dic')
            try:
                with open('{}/global_num_chord_one_hot_dic.pkl'.format(self.processed_data_path), 'rb') as f:
                    self.global_num_chord_one_hot_dic = pickle.load(f)
            except EOFError:
                print('global_num_chord_one_hot_dic not found!')
                self.global_num_chord_one_hot_dic = None
        else:
            self.global_num_chord_one_hot_dic = None

        # 读取二次压缩数据，其中和弦用tuple表示
        if 'double_compressed_data.pkl' in os.listdir(self.processed_data_path):
            print('reading double_compressed_data')
            with open('{}/double_compressed_data.pkl'.format(self.processed_data_path), 'rb') as f:
                self.double_compressed_data = pickle.load(f)
        else:
            # 因为随机mask不一样，因此无论如何都要保存compressed_data，此后每调用一次get_train_data方法都重置train_data
            if 'compressed_data.pkl' in os.listdir(self.processed_data_path):
                print('reading compressed_data')
                with open('{}/compressed_data.pkl'.format(self.processed_data_path), 'rb') as f:
                    self.compressed_data = pickle.load(f)
            else:
                self.compressed_data = None
                if 'processed_data.pkl' in os.listdir(self.processed_data_path):
                    with open('{}/processed_data.pkl'.format(self.processed_data_path), 'rb') as f:
                        self.processed_data = pickle.load(f)
                else:
                    self.processed_data = None

        self.structure_chord_dic = None  # 结构化和弦，包含三和弦、七和弦、九和弦
        self.get_structure_chord_dic()

        self.valid_compressed_data = None

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

    def compress_data(self):  # 压缩数据，得到每个和弦对应的旋律音
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
        with open('{}/compressed_data.pkl'.format(self.processed_data_path), 'wb') as f:
            pickle.dump(lst, f)
        self.compressed_data = lst

    def double_compress_data(self):  # 二次压缩数据，旋律成为一个范围在0-11的向量，和弦是tuple的列表
        if self.compressed_data is None:
            print('no compressed_data yet')
            return
        n = 0
        lst = []
        for value in self.compressed_data:
            melody = np.argmax(value['melody'], axis=1) % 12  # 旋律
            chord = []
            for i in value['chord']:
                tmp = []
                for j in range(len(i)):
                    if i[j] > 0:
                        tmp.append(j % 12)
                chord.append(tuple(tmp))
            tmp_dic = {'key': value['key'], 'name': value['name'], 'melody': melody, 'chord': chord}
            lst.append(tmp_dic)
            n += 1
            if n % 1000 == 0:
                print('{} valid samples'.format(n))
        print('total {} valid samples'.format(n))
        with open('{}/double_compressed_data.pkl'.format(self.processed_data_path), 'wb') as f:
            pickle.dump(lst, f)
        self.double_compressed_data = lst

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

    def get_train_data(self, min_length=1, m='major', write_cache=False, valid_compressed_data=None):
        # 拼接得到用于训练的数据，转为list形式的torch向量存在self.train_data中
        """
        :param min_length: 最小训练长度，1意味着最短的形式是一个和弦，加两个melody，预测一个chord
        :param m: 大小调
        :param write_cache: 是否写缓存
        :param valid_compressed_data: 传入保证是三和弦的sample
        :return:
        """
        train_data = []
        n = 0
        if valid_compressed_data is None:
            valid_compressed_data = self.valid_compressed_data
        if valid_compressed_data is None:
            print('no valid_compressed_data yet.')

        for c_data in valid_compressed_data:  # 循环做出dataset，以(chord, melody, key, true_chord)形式存储
            melody = c_data['melody'].copy()  # np.array(int)
            chord = c_data['chord'].copy()  # [tuple]

            if len(melody) <= min_length:
                continue

            melody_mat = np.zeros((len(melody), 12))
            chord_mat = np.zeros((len(chord), len(self.global_chord_num_dic[m])))
            for i in range(len(melody)):  # 生成one_hot矩阵
                melody_mat[i, melody[i]] = 1
                chord_mat[i][self.global_chord_num_dic[m][chord[i]]] = 1
            melody_mat = torch.Tensor(melody_mat)
            chord_mat = torch.Tensor(chord_mat)
            for j in range(min_length, len(melody)-1):
                train_data.append((chord_mat[:j], melody_mat[:j], melody_mat[j+1],
                                   torch.tensor(self.global_chord_num_dic[m][chord[j+1]])))
                n += 1
                if n % 10000 == 0:
                    print('{} valid train_data'.format(n))
        print('total {} valid train_data'.format(n))
        if write_cache:
            with open('{}/train_data.pkl'.format(self.processed_data_path), 'wb') as f:
                pickle.dump(train_data, f)
        self.train_data = train_data

    """
    def get_chord_dic(self):  # 扫描compressed_data的所有和弦并以元组的形式表示和弦，依次从最低音到最高音
        if self.train_data is None:
            print('no compressed_data yet!')
        chord_dic = {}
        n = 0
        count = 0
        t = time()
        for data in self.train_data:
            chord = data[1]  # 和弦
            for c in chord:
                lst = []
                j = 0
                while j < len(c):
                    if c[j] > 0:
                        lst.append(j % 12)
                    j += 1
                lst = tuple(lst)
                try:
                    chord_dic[lst]
                except KeyError:
                    chord_dic[lst] = n
                    n += 1
                    if n % 10 == 0:
                        print('{} different chords found'.format(n))
            count += 1
            if count % 1000 == 0:
                print('{} data scanned'.format(count))
                print('time used: {:.4f}s'.format(time() - t))
        self.chord_dic = chord_dic
        with open('{}/chord_dic.pkl'.format(self.processed_data_path), 'wb') as f:
            pickle.dump(chord_dic, f)
    """

    def get_structure_chord_dic(self):  # 直接自定义结构化的和弦字典，以和弦级别为初始的检索chord
        structure_chord_dic = {'triad': {'major': {}, 'minor': {}}, 'seventh': {'major': {}, 'minor': {}},
                               'ninth': {'major': {}, 'minor': {}}}  # 三和弦、七和弦、九和弦

        # 三和弦
        structure_chord_dic['triad']['major'] = major_triad_structure_chord_dic
        structure_chord_dic['triad']['minor'] = minor_triad_structure_chord_dic

        # 七和弦
        structure_chord_dic['seventh']['major'] = major_seventh_structure_chord_dic
        structure_chord_dic['seventh']['minor'] = minor_seventh_structure_chord_dic

        self.structure_chord_dic = structure_chord_dic

        # 生成global_chord_num_dic和global_num_chord_dic，需要分成自然大小调
        if (self.global_chord_num_dic is None) or (self.global_num_chord_dic is None):
            # num_chord_dic和chord_num_dic都分成大小调两个字典
            global_num_chord_dic = {'major': {}, 'minor': {}}
            global_chord_num_dic = {'major': {}, 'minor': {}}
            major_num = 0
            minor_num = 0
            for chord_type in structure_chord_dic.keys():  # 三和弦，七和弦，九和弦，当前仅三和弦和七和弦
                for tonic_type in structure_chord_dic[chord_type]:  # 大小调
                    for chord in structure_chord_dic[chord_type][tonic_type].values():
                        for pos in chord.values():  # 转位
                            try:
                                global_chord_num_dic[tonic_type][pos]
                            except KeyError:
                                if tonic_type == 'major':
                                    global_chord_num_dic[tonic_type][pos] = major_num
                                    global_num_chord_dic[tonic_type][major_num] = pos
                                    major_num += 1
                                else:
                                    global_chord_num_dic[tonic_type][pos] = minor_num
                                    global_num_chord_dic[tonic_type][minor_num] = pos
                                    minor_num += 1
            self.global_num_chord_dic = global_num_chord_dic
            self.global_chord_num_dic = global_chord_num_dic
            global_num_chord_one_hot_dic = {'major': {}, 'minor': {}}
            for m in ['major', 'minor']:
                for num, chord in self.global_num_chord_dic[m].items():
                    tmp = np.zeros(len(self.global_num_chord_dic[m]))
                    tmp[num] = 1
                    global_num_chord_one_hot_dic[m][num] = tmp
            self.global_num_chord_one_hot_dic = global_num_chord_one_hot_dic

            with open('{}/global_chord_num_dic.pkl'.format(self.processed_data_path), 'wb') as f:
                pickle.dump(global_chord_num_dic, f)
            with open('{}/global_num_chord_dic.pkl'.format(self.processed_data_path), 'wb') as f:
                pickle.dump(global_num_chord_dic, f)
            with open('{}/global_num_chord_one_hot_dic.pkl'.format(self.processed_data_path), 'wb') as f:
                pickle.dump(global_num_chord_one_hot_dic, f)
