# Copyright (c) 2021 Dai HBG

"""
该代码定义判断和弦进行是否符合规定的规则类
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

    def check_rules(self, chord_1, chord_2):  # 检查相邻和弦
        for key, value in self.rules:
            if not value(chord_1, chord_2):  # 只要规则集中有一个没有通过就不通过
                return False
        return True

    def get_rules(self):  # 定义规则
        def cross(chord_1, chord_2):  # 检查声部交叉
            return True

        self.rules['cross'] = cross

        def repeat_three(chord_1, chord_2):  # 检查三音是否重复
            return True

        self.rules['repeat_three'] = repeat_three
