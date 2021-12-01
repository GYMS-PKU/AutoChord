# Copyright (c) 2021 Dai HBG


"""
该文档定义主要和弦的字典

日志：
2021-12-01
-- 初始化
"""

# 三和弦
major_triad_structure_chord_dic = {i: {} for i in range(1, 8)}  # 大调
minor_triad_structure_chord_dic = {i: {} for i in range(1, 8)}  # 小调

major_triad_structure_chord_dic[1] = {1: (0, 4, 7), 2: (4, 7, 0), 3: (7, 0, 4)}
minor_triad_structure_chord_dic[1] = {1: (0, 3, 7), 2: (3, 7, 0), 3: (7, 0, 3)}

major_triad_structure_chord_dic[2] = {1: (2, 5, 9), 2: (5, 9, 2), 3: (9, 2, 5)}
minor_triad_structure_chord_dic[2] = {1: (2, 5, 8), 2: (5, 8, 2), 3: (8, 2, 5)}

major_triad_structure_chord_dic[3] = {1: (4, 7, 11), 2: (7, 11, 4), 3: (11, 4, 7)}
minor_triad_structure_chord_dic[3] = {1: (3, 7, 10), 2: (7, 10, 3), 3: (10, 3, 7)}

major_triad_structure_chord_dic[4] = {1: (5, 9, 0), 2: (9, 0, 5), 3: (0, 5, 9)}
minor_triad_structure_chord_dic[4] = {1: (5, 8, 0), 2: (8, 0, 5), 3: (0, 5, 8)}

major_triad_structure_chord_dic[5] = {1: (7, 11, 2), 2: (11, 2, 7), 3: (2, 7, 11)}
minor_triad_structure_chord_dic[5] = {1: (7, 10, 2), 2: (10, 2, 7), 3: (2, 7, 10)}

major_triad_structure_chord_dic[6] = {1: (9, 0, 4), 2: (0, 4, 9), 3: (4, 9, 0)}
minor_triad_structure_chord_dic[6] = {1: (8, 0, 3), 2: (0, 3, 8), 3: (3, 8, 0)}

major_triad_structure_chord_dic[7] = {1: (11, 2, 5), 2: (2, 5, 11), 3: (5, 11, 2)}
minor_triad_structure_chord_dic[7] = {1: (10, 2, 5), 2: (2, 5, 10), 3: (5, 10, 2)}

# 七和弦，需要更改
major_seventh_structure_chord_dic = {i: {} for i in range(1, 8)}  # 大调
minor_seventh_structure_chord_dic = {i: {} for i in range(1, 8)}  # 小调

major_seventh_structure_chord_dic[1] = {1: (0, 4, 7, 11), 2: (4, 7, 0, 11), 3: (7, 0, 11, 4), 4: (0, 11, 4, 7)}
minor_seventh_structure_chord_dic[1] = {1: (0, 3, 7, 10), 2: (3, 7, 0, 10), 3: (7, 0, 10, 3), 4: (0, 10, 3, 7)}

major_seventh_structure_chord_dic[2] = {1: (2, 5, 9, 0), 2: (5, 9, 0, 2), 3: (9, 0, 2, 5), 4: (0, 2, 5, 9)}
minor_seventh_structure_chord_dic[2] = {1: (2, 5, 8, 0), 2: (5, 8, 0, 2), 3: (8, 0, 2, 5), 4: (0, 2, 5, 8)}

major_seventh_structure_chord_dic[3] = {1: (4, 7, 11, 2), 2: (7, 11, 2, 4), 3: (11, 2, 4, 7), 4: (2, 4, 7, 11)}
minor_seventh_structure_chord_dic[3] = {1: (3, 7, 10, 2), 2: (7, 10, 2, 3), 3: (10, 2, 3, 7), 4: (2, 3, 7, 11)}

major_seventh_structure_chord_dic[4] = {1: (5, 9, 0, 4), 2: (9, 0, 4, 5), 3: (0, 4, 5, 9), 4: (4, 5, 9, 0)}
minor_seventh_structure_chord_dic[4] = {1: (5, 8, 0, 3), 2: (8, 0, 3, 5), 3: (0, 3, 5, 8), 4: (3, 5, 8, 0)}

major_seventh_structure_chord_dic[5] = {1: (7, 11, 2, 5), 2: (11, 2, 5, 7), 3: (2, 5, 7, 11), 4: (5, 7, 11, 2)}
minor_seventh_structure_chord_dic[5] = {1: (7, 10, 2, 5), 2: (10, 2, 5, 7), 3: (2, 5, 7, 10), 4: (5, 7, 10, 2)}

major_seventh_structure_chord_dic[6] = {1: (9, 0, 4, 7), 2: (0, 4, 7, 9), 3: (4, 7, 9, 0), 4: (7, 9, 0, 4)}
minor_seventh_structure_chord_dic[6] = {1: (8, 0, 3, 7), 2: (0, 3, 8, 9), 3: (3, 7, 8, 0), 4: (7, 8, 0, 3)}

major_seventh_structure_chord_dic[7] = {1: (11, 2, 5, 9), 2: (2, 5, 9, 11), 3: (5, 9, 11, 2), 4: (9, 11, 2, 5)}
minor_seventh_structure_chord_dic[7] = {1: (10, 2, 5, 8), 2: (2, 5, 8, 10), 3: (5, 8, 10, 2), 4: (8, 10, 2, 5)}

# 定义和弦到和弦名称的字典
chord_name_dic = {
    # 三和弦
    (0, 4, 7): 'T', (4, 7, 0): 'T_6', (7, 0, 4): 'T_46',
    (0, 3, 7): 't', (3, 7, 0): 't_6', (7, 0, 3): 't_46',
    (2, 5, 9): 'SII', (5, 9, 2): 'SII_6', (9, 2, 5): 'SII_46',
    (2, 5, 9): 'sII', (5, 9, 2): 'sII_6', (9, 2, 5): 'sII_46',
    (4, 7, 11): 'DTIII', (7, 11, 4): 'DTIII_6', (11, 4, 7): 'DTIII_46',
    (3, 7, 10): 'dtIII', (7, 10, 3): 'dtIII_6', (10, 3, 7): 'dtIII_46',
    (5, 9, 0): 'S', (9, 0, 5): 'S_6', (0, 5, 9): 'S_46',
    (5, 8, 0): 's', (8, 0, 5): 's_6', (0, 5, 8): 's_46',
    (7, 11, 2): 'D', (11, 2, 7): 'D_6', (2, 7, 11): 'D_46',
    (7, 10, 2): 'd', (10, 2, 7): 'd_6', (2, 7, 10): 'd_46',
    (9, 0, 4): 'TSVI', (0, 4, 9): 'TSVI_6', (4, 9, 0): 'TSVI_46',
    (8, 0, 3): 'tsVI', (0, 3, 8): 'tsVI_6', (3, 8, 0): 'tsVI_46',
    (11, 2, 5): 'DVII', (2, 5, 11): 'DVII_6', (5, 11, 2): 'DVII_46',
    (10, 2, 5): 'dVII', (2, 5, 10): 'dVII_6', (5, 10, 2): 'dVII_46',

    # 七和弦
    (0, 4, 7, 11): 'T_7', (4, 7, 11, 0): 'T_56', (7, 11, 0, 4): 'T_34', (11, 0, 4, 7): 'T_2',
    (0, 3, 7, 10): 't_7', (3, 7, 10, 0): 't_56', (7, 10, 0, 3): 't_34', (10, 0, 3, 7): 't_2',
    (2, 5, 9, 0): 'SII_7', (5, 9, 0, 2): 'SII_56', (9, 0, 2, 5): 'SII_34', (0, 2, 5, 9): 'S_II_2',
    (2, 5, 8, 0): 'sII_7', (5, 8, 0, 2): 'sII_56', (8, 0, 2, 5): 'sII_34', (0, 2, 5, 8): 's_II_2',
    (4, 7, 11, 2): 'DTIII_7', (7, 11, 2, 4): 'DTIII_56', (11, 2, 4, 7): 'DTIII_34', (2, 4, 7, 11): 'DTIII_2',
    (3, 7, 10, 2): 'dtIII_7', (7, 10, 2, 3): 'dtIII_56', (10, 2, 3, 7): 'dtIII_34', (2, 3, 7, 10): 'dtIII_2',
    (5, 9, 0, 4): 'S_7', (9, 0, 4, 5): 'S_56', (0, 4, 5, 9): 'S_34', (4, 5, 9, 0): 'S_2',
    (5, 8, 0, 3): 's_7', (8, 0, 3, 5): 's_56', (0, 3, 5, 8): 's_34', (3, 5, 8, 0): 's_2',
    (7, 11, 2, 5): 'D_7', (11, 2, 5, 7): 'D_56', (2, 5, 7, 11): 'D_34', (5, 7, 11, 2): 'D_2',
    (7, 10, 2, 5): 'd_7', (10, 2, 5, 7): 'd_56', (2, 5, 7, 10): 'd_34', (5, 7, 10, 2): 'd_2',
    (9, 0, 4, 7): 'TSVI_7', (0, 4, 7, 9): 'TSVI_56', (4, 7, 9, 0): 'TSVI_34', (7, 9, 0, 4): 'TSVI_2',
    (8, 0, 3, 7): 'tsVI_7', (0, 3, 7, 8): 'tsVI_56', (3, 7, 8, 0): 'tsVI_34', (7, 8, 0, 3): 'tsVI_2',
    (11, 2, 5, 9): 'DVII_7', (2, 5, 9, 11): 'DVII_56', (5, 9, 11, 2): 'DVII_34', (9, 11, 2, 5): 'DVII_2',
    (10, 2, 5, 8): 'dVII_7', (2, 5, 8, 10): 'dVII_56', (5, 8, 10, 2): 'dVII_34', (8, 10, 2, 5): 'dVII_2',
}
