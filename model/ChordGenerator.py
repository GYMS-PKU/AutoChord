# Copyright (c) 2021 Dai HBG


"""
该代码定义根据回溯算法生成一个和弦序列

开发日志
2021-11-21
-- 初始化
"""


import numpy as np
import sys
sys.path.append('../rule')
from Rule import Rule


class ChordGenerator:
    def __init__(self, chord_dic):
        """
        :param chord_dic: 和弦字典，形式为int->tuple，例如1->(0, 4, 7)表示C大调的原位主和弦
        """
        self.chord_dic = chord_dic
        self.rule = Rule  # 判定规则，用于判断和弦进行是否合法

    def generate(self, melody, method='back'):  # 给定旋律生成chord
        """
        :param melody: 旋律，array，目前限定成模12，也就是0到11之间，不管八度
        :param method: 生成方法，默认是回溯，可选markov，lstm
        :return: 返回根据和弦字典的编号序列
        """
        if method == 'back':
            tt = len(melody)
            t = 0
            chords = np.zeros(tt)  # 和弦
            key_frame = []  # 关键帧，用于记录冲突
            key_frame_feasible_chords = []  # 记录关键帧的可行集
            while t < tt:
                key = melody[t]  # 当前旋律音
                feasible_chords = []  # 生成可行集，注意可行集用Rule中定义的方法判别

                for i in self.chord_dic.keys():
                    tmp_chord = self.chord_dic[i]
                    if not self.rule.in_chord(key, tmp_chord):  # 不是旋律内音
                        continue
                    if t == 0:  # 初始第一个和弦不用检查
                        feasible_chords.append(i)
                    else:
                        if self.rule.check_rules(chords[t-1], tmp_chord):  # 自定义的规则都检查通过后才能进入可行集
                            feasible_chords.append(i)

                if not feasible_chords:  # 如果没有可行的和弦
                    if not key_frame:
                        key_frame.append(t-1)  # 上一时刻称为关键帧
