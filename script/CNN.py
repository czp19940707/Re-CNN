import csv
import os

import numpy as np


def mean_std(list_):
    return [(np.mean(i, axis=0) / 100., np.std(i, axis=0) / 100.) for i in list_]


if __name__ == '__main__':
    model_name = 'CNN_mf'
    acc_seeds, sensitivity_seeds, specificity_seeds, auc_seeds = [], [], [], []
    path = r'../outputs'
    for i in os.listdir(path):
        seed = i.split('_')[-1]
        pa_ = os.path.join(path, i, model_name, model_name + '.csv')
        f = csv.reader(open(pa_, 'r'))
        for nums, j in enumerate(f):
            if nums >= 1:
                acc_seeds.append(float(j[-2]))
                sensitivity_seeds.append(float(j[1]))
                specificity_seeds.append(float(j[2]))
                auc_seeds.append(float(j[-1]))
    acc_seeds, sensitivity_seeds, specificity_seeds, auc_seeds = np.array(acc_seeds), np.array(
        sensitivity_seeds), np.array(specificity_seeds), np.array(auc_seeds)
    acc_seeds, sensitivity_seeds, specificity_seeds, auc_seeds = mean_std(
        [acc_seeds, sensitivity_seeds, specificity_seeds, auc_seeds])

    print('model: {}'.format(model_name))
    print('random cross validation mean acc: {} std: {}'.format(acc_seeds[0], acc_seeds[1]))
    print('random cross validation mean auc: {} std: {}'.format(auc_seeds[0], auc_seeds[1]))
    print('random cross validation mean sensitivity: {} std: {}'.format(sensitivity_seeds[0], sensitivity_seeds[1]))
    print('random cross validation mean specificity: {} std: {}'.format(specificity_seeds[0], specificity_seeds[1]))

