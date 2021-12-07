# Copyright (c) 2021 Dai HBG


def vec2chord(chords, tonic='major'):
    """
    :param chords: 和弦序列
    :param tonic: 调性
    :return:
    """
    dic = {'major': {0: 1, 2: 2, 4: 3, 5: 4, 7: 5, 9: 6, 11: 7},
           'minor': {0: 1, 2: 2, 3: 3, 5: 4, 7: 5, 8: 6, 10: 7}}
    new_chords = []
    for chord in chords:
        new_chords.append((dic[tonic][i] for i in chord))
    return new_chords
