# Copyright (c) 2021 Dai HBG


"""
该脚本用于包含三和弦和七和弦的模型训练

日志
2021-12-19
- 三和弦，七和弦
"""

import matplotlib.pyplot as plt
import torch
import argparse

parser = argparse.ArgumentParser(description='train model')
parser.add_argument('-model', type=str, default='lstm', help='type of model')
parser.add_argument('-model_name', type=str, default='lstm_major', help='model_name')
parser.add_argument('-tonic', type=str, default='major', help='tonic')
parser.add_argument('-epochs', type=int, default=100, help='training epochs')
parser.add_argument('-device', type=str, default='cuda', help='device')
parser.add_argument('-batch_size', type=int, default=1000, help='batch size')
parser.add_argument('-verbose', type=bool, default=True, help='verbose')
parser.add_argument('-training_set_ratio', type=float, default=0.7, help='ratio of training set')
parser.add_argument('-plotting_loss', type=bool, default=True, help='saving loss plot')
args = parser.parse_args()

import sys

sys.path.append('..')

from dataloader.DataLoader import DataLoader
from AutoChord.AutoChord import AutoChord

dl = DataLoader(device='cpu')
AC = AutoChord()

total = 0  # 出现的和弦总数
triad_num = 0  # 三和弦总数
triad_num_dic = {'minor': {i: 0 for i in dl.global_chord_num_dic['minor'].keys()},
                 'major': {i: 0 for i in dl.global_chord_num_dic['major'].keys()}}
triad_sample_num = 0  # 全部使用三和弦的样本数
triad_samples = {'major': [], 'minor': []}
for sample in dl.double_compressed_data:
    if sample['key'] in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        m = 'major'
    else:
        m = 'minor'
    sample_copy = sample.copy()
    melody = sample['melody']
    chord = sample['chord']
    t = True
    for i in range(len(melody)):
        total += 1
        try:
            _ = dl.global_chord_num_dic[m][chord[i]]
            triad_num_dic[m][chord[i]] += 1
            triad_num += 1
        except KeyError:
            if len(chord[i]) < 3:
                for c in dl.global_chord_num_dic[m].keys():
                    if melody[i] not in c:
                        continue
                    for k in chord[i]:
                        if k not in c:
                            break
                    else:
                        sample_copy['chord'][i] = c
                        triad_num_dic[m][c] += 1
                        triad_num += 1
                        break
                else:
                    t = False
            else:
                t = False
                pass
    if t:
        triad_samples[m].append(sample_copy)
        triad_sample_num += 1

print('total {} chords.'.format(total))
print('total {} triad and seventh.'.format(triad_num))
print('total {} triad or seventh samples.'.format(triad_sample_num))

# getting train data
dl.get_train_data(min_length=1, write_cache=False,
                  valid_compressed_data=triad_samples[args.tonic],
                  tonic=args.tonic)

print('')
print('start training...')
params = {'chord_num': 48, 'device': args.device}
AC.get_model(model_name=args.model, params=params)
train_data_length = len(dl.train_data)
train_loss, test_loss = AC.fit(dl.train_data[:int(train_data_length * args.training_set_ratio)],
                               dl.train_data[int(train_data_length * args.training_set_ratio):],
                               epochs=args.epochs, batch_size=args.batch_size, verbose=args.verbose)

if args.plotting_loss:
    plt.figure(dpi=150)
    plt.plot(train_loss, label='train')
    plt.plot([i * int(train_data_length * args.training_set_ratio) // args.batch_size
              for i in range(1, args.epochs + 1)], test_loss[:], label='test')
    plt.xlabel('iterations')
    plt.grid(linestyle='--')
    plt.legend()
    plt.ylabel('nll_Loss')
    plt.title('Loss of {}'.format(args.model_name))
    plt.savefig('../pics/{}.jpg'.format(args.model_name))

torch.save(AC.model.model, '../trained_models/{}'.format(args.model_name))
