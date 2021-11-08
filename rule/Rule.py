# Copyright (c) 2021 Dai HBG

"""
该代码定义判断和弦进行是否符合规定的规则类
"""


import numpy as np


class Rule:
    def __init__(self):
        self.rules = {}
        self.get_rules()

    def get_rules(self):  # 定义规则
        def cross(chord):
            return True
        self.rules['cross'] = cross

