import csv

import numpy as np


def print_result(set_, f1):
    male_ad, female_ad, male_cn, female_cn = 0, 0, 0, 0
    age_max_ad = 0.
    age_min_ad = 999.
    age_max_cn = 0.
    age_min_cn = 999.
    age_ad = []
    age_cn = []
    for nums, i in enumerate(f1):
        if nums >= 1:
            if i[2] == 'CN':
                if float(i[4]) >= age_max_cn:
                    age_max_cn = float(i[4])
                if float(i[4]) <= age_min_cn:
                    age_min_cn = float(i[4])
                if i[3] == 'M':
                    male_cn += 1
                if i[3] == 'F':
                    female_cn += 1
                age_cn.append(float(i[4]))
            if i[2] == 'AD':
                if float(i[4]) >= age_max_ad:
                    age_max_ad = float(i[4])
                if float(i[4]) <= age_min_ad:
                    age_min_ad = float(i[4])
                if i[3] == 'M':
                    male_ad += 1
                if i[3] == 'F':
                    female_ad += 1
                age_ad.append(float(i[4]))
    f = open(r'{}_inf.txt'.format(set_), 'w')
    f.write('dataset: {}\ngroup: {}\n'.format(set_, 'AD'))
    f.write('age_min_ad: {} age_max_ad: {}\n'.format(age_min_ad, age_max_ad))
    f.write('subject nums: {} male rate: {} female rate: {}\n'.format(male_ad + female_ad, male_ad, female_ad))
    f.write('age mean: {} std: {}\n'.format(np.array(age_ad).mean(), np.array(age_ad).std()))

    f.write('\ngroup: {}\n'.format('CN'))
    f.write('age_min_ad: {} age_max_ad: {}\n'.format(age_min_cn, age_max_cn))
    f.write('subject nums: {} male: {} female: {}\n'.format(male_cn + female_cn, male_cn, female_cn))
    f.write('age mean: {} std: {}'.format(np.array(age_cn).mean(), np.array(age_cn).std()))


if __name__ == '__main__':
    path1 = r'aibl_information.csv'
    path2 = r'adni_information.csv'
    f1 = csv.reader(open(path1, 'r'))
    f2 = csv.reader(open(path2, 'r'))
    male, female, age_male, age_female = 0, 0, 0., 0.
    print_result('adni', f2)
    print_result('aibl', f1)
