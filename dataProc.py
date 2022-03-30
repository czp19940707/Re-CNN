from torch.utils.data import Dataset
import os
import numpy as np


class dataProc(Dataset):
    def __init__(self, split, if_='hipp', seed=1):
        self.seed = seed
        self.split = split
        self.path = self.open_txt(split)
        self.n_classes = 2
        self.if_ = if_

    def open_txt(self, split):
        if split == 'Training':
            path = r'data/seed_{}/train_{}.txt'.format(self.seed, self.seed)
        elif split == 'Validation':
            path = r'data/seed_{}/valid_{}.txt'.format(self.seed, self.seed)
        else:
            path = r'data/seed_{}/test_{}.txt'.format(self.seed, self.seed)

        try:
            open(path, 'r')
        except:
            print('seed not found or txt file not exist!')
            import sys
            sys.exit(0)

        data = []
        with open(path, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                data.append(line)
        return data

    def __len__(self):
        return len(self.path)

    def z_score_norm(self, study_data):
        study_data = (study_data - np.mean(study_data) + 0.000001) / (np.std(study_data) + 0.000001)
        return study_data

    def hipp_crop(self, npy):
        return npy[40:120, 50:150, 10:90]

    def get_sample_weights(self):
        weights = []
        count_nums = np.arange(0, self.n_classes).astype(np.int64)
        count = float(len(self.path))
        label = self.to_label(self.path)
        count_class_list = [float(label.count(i)) for i in count_nums]
        for i in label:
            for j in count_nums:
                if i == j:
                    weights.append(count / count_class_list[j])
        imbalanced_ratio = [count_class_list[0] / i_r for i_r in count_class_list]
        return weights, imbalanced_ratio

    def to_label(self, path):
        label = []
        for i in path:
            cls = int(i.split(' ')[-1])
            label.append(cls)
        return label

    def __getitem__(self, item):
        sd, cls = self.path[item].split(' ')[0], int(self.path[item].split(' ')[1])
        img = self.path[item].split(' ')[0].replace('Morphological_Metrics', 'image')
        if self.if_ == 'hipp':
            np_image, np_sd = self.hipp_crop(np.load(img)), np.load(sd)
        else:
            np_image, np_sd = np.load(img), np.load(sd)
        np_sd = self.z_score_norm(np_sd)
        np_image = np_image[None, ...]
        return np_image, np_sd, cls, self.path[item].split(' ')[0].split('/')[-1]


if __name__ == '__main__':
    dp = dataProc(split='Training')
    for i in dp:
        pass
