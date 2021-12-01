# Copyright (c) 2021 Dai HBG


"""
该代码定义根据回溯算法生成一个和弦序列

开发日志
2021-11-21
-- 初始化
2021-11-23
-- 新增根据全局转移矩阵的选取方法
-- 新增根据lstm的block sampler的生成方法
2021-12-01
-- 更新：使用lstm和nn预测时对和弦序列生成one-hot
"""

import numpy as np
import sys
sys.path.append('../rule')
from Rule import Rule


class ChordGenerator:
    def __init__(self, global_num_chord_dic, global_num_chord_dic_one_hot=None,
                 feasible_chord_dic=None, global_transition_matrix=None):
        """
        :param global_num_chord_dic: 全局编号和弦字典，存储所有和弦的全局位置，形式为int->tuple，例如1->(0, 4, 7)表示C大调的原位主和弦
        :param global_num_chord_dic_one_hot: 全局编号到和弦one_hot表示的字典
        :param feasible_chord_dic: 可行和弦字典，形式为int->tuple，例如1->(0, 4, 7)表示C大调的原位主和弦，要求编号和全局字典相同
        :param global_transition_matrix: 全局转移矩阵
        """
        self.global_num_chord_dic = global_num_chord_dic
        self.global_num_chord_dic_one_hot = global_num_chord_dic_one_hot
        if feasible_chord_dic is None:
            feasible_chord_dic = global_num_chord_dic
        self.feasible_chord_dic = feasible_chord_dic
        self.global_transition_matrix = global_transition_matrix
        self.rule = Rule()  # 判定规则，用于判断和弦进行是否合法

    def generate(self, melody, method='back', model=None):  # 给定旋律生成chord
        """
        :param melody: 旋律，array，目前限定成模12，也就是0到11之间，不管八度
        :param method: 生成方法，默认是回溯，可选global_markov，lstm
        :param model: 用于生成下一个chord概率的模型
        :return: 返回根据和弦字典的编号序列
        """
        tt = len(melody)
        chords = np.zeros(tt)  # 生成的和弦序列，注意这里使用的是和弦编号，最后可以用chord_dic转为tuple
        chords[:] = np.nan  # 先填充缺失值
        chords_one_hot = np.zeros((tt, len(self.global_num_chord_dic)))  # 生成的和弦序列，但是使用one-hot编码，适用于使用NN类方法

        melody_mat = np.zeros((len(melody), 12))  # 生成melody的one-hot表示，适用于NN类方法
        for i in range(len(melody)):
            melody_mat[i, melody[i]] = 1

        if method in ['back', 'global_markov', 'lstm', 'nn']:
            t = 0
            # key_frame = []  # 关键帧，用于记录冲突
            # key_frame_feasible_chords = []  # 记录关键帧的可行集
            feasible_chords_record = []  # 记录可行集，如果回溯，则需要剔除；其中元素是和弦编号的list
            while t < tt:
                if len(feasible_chords_record) == t + 1:  # 此时说明是回溯回来的，此时不需要重复计算可行和弦
                    if method == 'back' or t == 0:  # 随机采样，之后可以添加指定和弦，例如主和弦
                        tmp_chord = np.random.choice(feasible_chords_record[-1])  # 直接从这里随机生成下一个和弦
                    elif method == 'global_markov':  # 全局马尔可夫采样
                        p = self.global_transition_matrix[chords[t-1]][feasible_chords_record[-1]].copy()
                        p /= np.sum(p)  # 局部转移概率
                        tmp_chord = np.random.choice(feasible_chords_record[-1], p=p)
                    elif method in ['lstm', 'nn']:  # 此时可行集中的元素是具体的
                        if method == 'nn':  # 此时仅仅使用虽有最后一个chord，melody和下一个melody
                            # 注意默认使用single模式来predict，因此预测就是一维向量
                            # 由于使用log_softmax，因此输出需要取指数
                            p = model.predict(chords_one_hot[t-1:t], melody[t-1:t], melody[t:t+1])
                            p = np.exp(p)[feasible_chords_record[-1]]
                            p /= np.sum(p)
                            tmp_chord = np.random.choice(feasible_chords_record[-1], p=p)
                            chords_one_hot[t] = self.global_num_chord_dic_one_hot[tmp_chord]
                        else:
                            p = model.predict(chords_one_hot[:t], melody[:t], melody[t:t + 1])
                            p = np.exp(p)[feasible_chords_record[-1]]
                            p /= np.sum(p)
                            tmp_chord = np.random.choice(feasible_chords_record[-1], p=p)
                            chords_one_hot[t] = self.global_num_chord_dic_one_hot[tmp_chord]
                    else:
                        tmp_chord = np.random.choice(feasible_chords_record[-1])
                    chords[t] = tmp_chord
                    t += 1
                    continue

                # 否则这是第一次抵达该音
                key = melody[t]  # 当前旋律音
                feasible_chords = []  # 生成可行集，注意可行集用Rule中定义的方法判别

                for i in self.feasible_chord_dic.keys():  # 从可行集中过滤和弦
                    tmp_chord = self.feasible_chord_dic[i]
                    if not self.rule.in_chord(key, tmp_chord):  # 不是旋律内音
                        continue
                    if t == 0:  # 初始第一个和弦不用检查
                        feasible_chords.append(i)
                    else:
                        if self.rule.check_rules(self.feasible_chord_dic[chords[t-1]],
                                                 tmp_chord, melody[t-1], melody[t]):
                            # 自定义的规则都检查通过后才能进入可行集
                            feasible_chords.append(i)
                # print(feasible_chords)
                if not feasible_chords:  # 如果没有可行的和弦，就依次往前回溯。注意此时feasible_chords_record的长度仍然是t
                    t -= 1
                    while len(feasible_chords_record[t]) == 1:  # 表明上一步只有一个可行集，则需要进一步回溯
                        feasible_chords_record.pop()  # 弹出最后一个
                        chords[t] = np.nan  # 将chords该和弦取消
                        if t == 0:  # 最后一个也弹出了
                            print('No chords can satisfy rules! Please check your melody or the rules set.')
                        t -= 1
                    # 现在t的位置是引发矛盾的和弦
                    feasible_chords_record[t].remove(chords[t])  # 可行集删除引发矛盾的和弦
                    if method == 'back' or t == 0:  # 随机采样
                        tmp_chord = np.random.choice(feasible_chords_record[-1])  # 直接从这里随机生成下一个和弦
                    elif method == 'global_markov':  # 全局马尔可夫采样
                        p = self.global_transition_matrix[chords[t - 1]][feasible_chords_record[-1]].copy()
                        p /= np.sum(p)  # 局部转移概率
                        tmp_chord = np.random.choice(feasible_chords_record[-1], p=p)
                    else:
                        tmp_chord = np.random.choice(feasible_chords_record[-1])
                    chords[t] = tmp_chord
                    t += 1
                    continue
                else:
                    feasible_chords_record.append(feasible_chords)  # 可行集记录添加
                    if method == 'back' or t == 0:  # 随机采样
                        tmp_chord = np.random.choice(feasible_chords_record[-1])  # 直接从这里随机生成下一个和弦
                    elif method == 'global_markov':  # 全局马尔可夫采样
                        p = self.global_transition_matrix[chords[t - 1]][feasible_chords_record[-1]].copy()
                        p /= np.sum(p)  # 局部转移概率
                        tmp_chord = np.random.choice(feasible_chords_record[-1], p=p)
                    elif method in ['lstm', 'nn']:  # 此时可行集中的元素是具体的
                        if method == 'nn':  # 此时仅仅使用虽有最后一个chord，melody和下一个melody
                            # 注意默认使用single模式来predict，因此预测就是一维向量
                            # 由于使用log_softmax，因此输出需要取指数
                            p = model.predict(chords_one_hot[t-1:t], melody[t-1:t], melody[t:t+1])
                            p = np.exp(p)[feasible_chords_record[-1]]
                            p /= np.sum(p)
                            tmp_chord = np.random.choice(feasible_chords_record[-1], p=p)
                            chords_one_hot[t] = self.global_num_chord_dic_one_hot[tmp_chord]
                        else:
                            p = model.predict(chords_one_hot[:t], melody[:t], melody[t:t + 1])
                            p = np.exp(p)[feasible_chords_record[-1]]
                            p /= np.sum(p)
                            tmp_chord = np.random.choice(feasible_chords_record[-1], p=p)
                            chords_one_hot[t] = self.global_num_chord_dic_one_hot[tmp_chord]
                    else:
                        tmp_chord = np.random.choice(feasible_chords_record[-1])
                    chords[t] = tmp_chord
                    t += 1
                    # print(tmp_chord)
                    continue
            return chords
