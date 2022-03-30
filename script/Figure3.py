import os
import csv

import matplotlib.pyplot as plt
import numpy as np


def to_float(list_):
    return [float(da_) for da_ in list_]


def sort_(list_, nums):
    return sorted(list_, key=lambda x: float(x.split('_')[nums]))


def mean_std(list_):
    return [(np.mean(i, axis=0), np.std(i, axis=0)) for i in list_]


def get_csv(list_csv, path1):
    list_csv = sort_(list_csv, -5)
    sp_list, se_list, acc_list, auc_list = [], [], [], []
    for csv_name in list_csv:
        pa_ = os.path.join(path1, csv_name)
        f = csv.reader(open(pa_, 'r'))
        for nums_csv, data_csv in enumerate(f):
            if nums_csv >= 1:
                data_csv = to_float(data_csv[1:])
                try:
                    sp_list.append(data_csv[0])
                    se_list.append(data_csv[1])
                    acc_list.append(data_csv[2])
                    auc_list.append(data_csv[3])
                except:
                    pass
    return sp_list, se_list, acc_list, auc_list


def get_csv_1(path1):
    sp_list, se_list, acc_list, auc_list = [], [], [], []
    f = csv.reader(open(path1, 'r'))
    for nums_csv, data_csv in enumerate(f):
        if nums_csv >= 1:
            data_csv = to_float(data_csv[1:])
            try:
                sp_list.append(data_csv[0])
                se_list.append(data_csv[1])
                acc_list.append(data_csv[2])
                auc_list.append(data_csv[3])
            except:
                pass
    return sp_list, se_list, acc_list, auc_list


def get_alpha(list_, path):
    sp_list_table, se_list_table, acc_list_table, auc_list_table = [], [], [], []
    for kl in list_:
        pa_ = os.path.join(path, kl)
        sp_list_table_row, se_list_table_row, acc_list_table_row, auc_list_table_row = [], [], [], []

        for dims in sort_(os.listdir(pa_), -1):
            p_ = os.path.join(pa_, dims, 'Re_CNN_mf_{}_{}.csv'.format(kl, dims))
            sp_list, se_list, acc_list, auc_list = get_csv_1(p_)
            sp_list_table_row.append(sp_list)
            se_list_table_row.append(se_list)
            acc_list_table_row.append(acc_list)
            auc_list_table_row.append(auc_list)
        sp_list_table.append(sp_list_table_row)
        se_list_table.append(se_list_table_row)
        acc_list_table.append(acc_list_table_row)
        auc_list_table.append(auc_list_table_row)
    sp_list_table, se_list_table, acc_list_table, auc_list_table = np.array(sp_list_table), np.array(
        se_list_table), np.array(acc_list_table), np.array(auc_list_table)
    return sp_list_table, se_list_table, acc_list_table, auc_list_table


def get_seed(list_, path):
    sp_list_seed, se_list_seed, acc_list_seed, auc_list_seed = [], [], [], []
    for i in list_:
        pa_ = os.path.join(path, i, 'Re_CNN_mf')
        sp_list_table, se_list_table, acc_list_table, auc_list_table = get_alpha(sort_(os.listdir(pa_), -1), pa_)
        sp_list_seed.append(sp_list_table)
        se_list_seed.append(se_list_table)
        acc_list_seed.append(acc_list_table)
        auc_list_seed.append(auc_list_table)
    sp_list_seed, se_list_seed, acc_list_seed, auc_list_seed = np.array(sp_list_seed), np.array(se_list_seed), np.array(
        acc_list_seed), np.array(auc_list_seed)
    sp_list_seed, se_list_seed, acc_list_seed, auc_list_seed = mean_std(
        [sp_list_seed, se_list_seed, acc_list_seed, auc_list_seed])

    return sp_list_seed, se_list_seed, acc_list_seed, auc_list_seed


if __name__ == '__main__':
    path2 = r'../outputs'
    metric_ = 'auc'
    sp_list_seed, se_list_seed, acc_list_seed, auc_list_seed = get_seed(sort_(os.listdir(path2), -1), path2)

    dict_ = {
        'auc': (auc_list_seed, (86, 93)),
        'acc': (acc_list_seed, (77, 87)),
        'se': (se_list_seed, (65, 85)),
        'sp': (sp_list_seed, (80, 94)),
    }
    aa = auc_list_seed[1] / 100
    x_axis = np.array([1, 30, 50, 70, 100, 150, 200, 250, 300, 400, 512])
    plt.figure(figsize=(8, 7))
    Color = ['lime', 'blue', 'black', 'brown', 'red', 'pink',
             'chocolate', 'darkred', 'gold', 'blue', 'red',
             'lime', 'greenyellow', 'tomato', 'pink']
    markers = ['o', 's', '^', 'p', '.', 'v',
               ',', 'd', 'h', '2', 'x',
               '4', 'd', '+']
    Linestyle = ['-', '--', '-.', ':', '-', '--',
                 '-.', ':', '-', '--', '-.',
                 ':', '-', '--']
    alpha_ = ['{}=0.1'.format(chr(945)), '{}=0.5'.format(chr(945)), '{}=1'.format(chr(945)), '{}=5'.format(chr(945)),
              '{}=10'.format(chr(945)), '{}=20'.format(chr(945))]
    for nums, i in enumerate(dict_[metric_][0][0]):
        plt.plot(x_axis, i, color=Color[nums], marker='o', linewidth=1.0, linestyle='-')
    plt.ylim(dict_[metric_][1])
    plt.legend(alpha_, loc=0, prop={'size': 8}, borderaxespad=1)
    plt.title('Re-CNN')
    plt.xlabel('dims')
    plt.ylabel('{}'.format(metric_))
    plt.savefig('Figures/Figure3.png')
    plt.close()
