# Copyright (c) 2021 Dai HBG

"""
该代码定义判断和弦进行是否符合规定的规则类
在C大调中

开发日志：
2021-11-22
-- 初始化，定义不良进行的检查
"""


import numpy as np


class Rule:
    def __init__(self):
        self.rules = {}
        self.get_rules()

    @staticmethod
    def in_chord(key, chord):  # 检查旋律音是否是和弦内音
        """
        :param key: 旋律，int
        :param chord: 和弦，tuple
        :return:
        """
        return key in chord

    def check_rules(self, chord_1, chord_2, key_1, key_2):  # 检查相邻和弦
        for key, value in self.rules.items():
            if not value(chord_1, chord_2, key_1, key_2):  # 只要规则集中有一个没有通过就不通过
                return False
        return True

    def get_rules(self):  # 定义规则
        def cross(chord_1, chord_2, key_1=None, key_2=None):  # 检查声部交叉
            if (len(chord_1) == 1) or (len(chord_2) == 1):  # 有一个是单低音
                return True
            return True
        self.rules['cross'] = cross

        def parallel(chord_1, chord_2, key_1, key_2):  # 检查是否有平行八五度以及隐伏八五度
            # 检查平行八度和隐伏八度
            eight_2 = False  # 检查chord_2中是否有八度
            pos = None  # 八度对应的低音声部和高音声部
            low_2 = None  # 八度对应的低声部音
            high_2 = None  # 八度对应的高声部音
            for i in range(len(chord_2)):
                if chord_2[i] == key_2:
                    eight_2 = True
                    pos = (i, len(chord_2))  # 八度声部
                    low_2 = chord_2[i]  # 低声部音
                    high_2 = key_2  # 高声部音
                    break
            if not eight_2:
                for i in range(len(chord_2) - 1):
                    for j in range(i + 1, len(chord_2)):
                        if chord_2[i] == chord_2[j]:
                            eight_2 = True
                            pos = (i, j)  # 八度声部
                            low_2 = chord_2[i]  # 低声部音
                            high_2 = chord_2[j]  # 高声部音
                            break
            if eight_2:
                # 取出第一个和弦对应的声部
                if pos[1] < len(chord_1) + 1:
                    low_1 = chord_1[pos[0]]
                    high_1 = chord_1[pos[1]] if pos[1] < len(chord_1) else key_1
                    if low_1 == high_1:  # 平行八度
                        print(chord_1, key_1, chord_2, key_2, 'p eight')
                        return False
                    if (abs(low_1-low_2) in [1, 2, 10, 11]) or (abs(high_1-high_2) in [1, 2, 10, 11]):  # 隐伏八度
                        print(chord_1, key_1, chord_2, key_2, 'hidden eight')
                        return False

            # 检查平行五度和隐伏五度
            five_2 = False  # 再检查chord_2中是否有五度
            pos = None
            low_2 = None
            high_2 = None
            for i in range(len(chord_2)):
                if (key_2 - chord_2[i]) % 12 == 7:
                    five_2 = True
                    pos = (i, len(chord_2) + 1)  # 记录下五度声部
                    low_2 = chord_2[i]  # 记录声部
                    high_2 = key_2  # 记录声部
                    break
            if not five_2:
                for i in range(len(chord_2) - 1):
                    for j in range(i + 1, len(chord_2)):
                        if (chord_2[j] - chord_2[i]) % 12 == 7:
                            five_2 = True
                            pos = (i, j)  # 记录下声部
                            low_2 = chord_2[i]  # 记录声部
                            high_2 = chord_2[j]  # 记录声部
                            break
            if five_2:
                # 取出第一个和弦对应的声部
                if pos[1] < len(chord_1) + 1:
                    low_1 = chord_1[pos[0]]
                    high_1 = chord_1[pos[1]] if pos[1] < len(chord_1) else key_1
                    if (high_1 - low_1) % 12 == 7:  # 平行五度
                        print(chord_1, key_1, chord_2, key_2, 'p fifth')
                        return False
                    if (abs(low_1 - low_2) in [1, 2, 10, 11]) or (abs(high_1 - high_2) in [1, 2, 10, 11]):  # 隐伏五度
                        print(chord_1, key_1, chord_2, key_2, 'hidden fifth')
                        return False
            return True

        self.rules['parallel'] = parallel

        def repeat_three(chord_1, chord_2, key_1, key_2):  # 检查三音是否重复
            return True

        self.rules['repeat_three'] = repeat_three

        def diagonal(chord_1, chord_2, key_1, key_2):  # 检查声部对斜
            for i in range(len(chord_1)):
                if i > len(chord_2):  # 一般而言现在要求和声长度一致
                    break
                for j in range(len(chord_2)):
                    if i == j:
                        continue
                    if abs(chord_1[i] - chord_2[j]) in [1, 11]:
                        return False
                if abs(chord_1[i] - key_2) in [1, 11]:
                    return False
            for j in range(len(chord_2)):
                if abs(key_1 - chord_2[j]) in [1, 11]:
                    return False
            return True

        self.rules['diagonal'] = diagonal

        def direction(chord_1, chord_2, key_1, key_2):  # 检查四部同向
            if (len(chord_1) != 3) or (len(chord_2) != 3):  # 目前只检查四部和声
                return True
            # 首先按照mod 12拉成一个单调向量再比较
            c_1 = [i for i in chord_1] + [key_1]
            c_2 = [i for i in chord_2] + [key_2]
            for j in range(1, len(c_1)):
                pass
            return True

        self.rules['direction'] = direction
