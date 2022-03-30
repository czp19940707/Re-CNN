import os

from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split as ts
import mat4py
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch


class mlp(nn.Module):
    def __init__(self, n_classes=2):
        super(mlp, self).__init__()
        self.linear_1 = nn.Linear(204, 1024)
        self.linear_2 = nn.Linear(1024, 50)
        self.linear_3 = nn.Linear(50, n_classes)

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        return self.linear_3(x)


def proc(list_):
    X, y, name = [], [], []
    for cls in list_:
        for pa in os.listdir(cls):
            if pa.endswith('.npy'):
                pa_ = os.path.join(cls, pa)
                features = np.load(pa_)
                features = z_score(features)
                X.append(features)
                if cls.split('/')[-1] == 'AD':
                    y.append(1)
                elif cls.split('/')[-1] == 'CN':
                    y.append(0)
                name.append(pa.split('_')[-1][:-4])
    return X, y, name


def function1(matrix):
    return matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]


def z_score(npy):
    return (npy - np.mean(npy) + 0.0000001) / (np.std(npy) + 0.0000001)


def mlp_train(X_train, y_train, X_test, y_test, seeds):
    # mlp
    prob_list = []
    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
    net = mlp(n_classes=2).cuda()
    net.load_state_dict(torch.load(os.path.join(r'../mlp_params', seeds, 'mlp_best.pth')))
    net.eval()
    with torch.no_grad():
        for i in range(X_test.shape[0]):
            val_data = torch.FloatTensor(X_test[i, ...][None, ...]).cuda()
            out = net.forward(val_data)
            prob = F.softmax(out)
            # _, preds = out.max(0)
            prob_list.append(prob.cpu().numpy().squeeze())
            # pred_list.append(preds.cpu().numpy().item())
    return np.array(prob_list)


def classifier(model, X_train, y_train, X_test, y_test, seeds):
    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
    if model == 'rf':
        rfc = RandomForestClassifier(n_estimators=25, random_state=0, bootstrap=False)
        rfc = rfc.fit(X_train, y_train)
        predict_results = rfc.predict_proba(X_test)
    elif model == 'SVM':
        clf_poly = svm.SVC(kernel='poly', probability=True)
        clf_poly.fit(X_train, y_train)
        predict_results = clf_poly.predict_proba(X_test)
    elif model == 'Xgboost':
        model = XGBClassifier()
        Xgb = model.fit(X_train, y_train)
        predict_results = Xgb.predict_proba(X_test)
    elif model == 'mlp':
        predict_results = mlp_train(X_train, y_train, X_test, y_test, seeds)
    return predict_results


def mean_std(list_):
    return [(np.mean(i, axis=0), np.std(i, axis=0)) for i in list_]


if __name__ == '__main__':
    model_name = 'Xgboost'
    path1 = r'../data/ADNI/Morphological_Metrics/AD'
    path2 = r'../data/ADNI/Morphological_Metrics/CN'
    path3 = r'../data/AIBL/Morphological_Metrics/AD'
    path4 = r'../data/AIBL/Morphological_Metrics/CN'
    acc_seeds, sensitivity_seeds, specificity_seeds, auc_seeds = [], [], [], []
    for seeds in os.listdir(r'../data'):
        if seeds.startswith('seed'):
            pa_ = os.path.join(r'../data', seeds)
            seed_num = seeds[-1]
            txt1 = os.path.join(pa_, 'train_{}.txt'.format(seed_num))
            txt2 = os.path.join(pa_, 'test_{}.txt'.format(seed_num))
            train_, val_ = [], []
            with open(txt1, 'r') as f:
                for i in f.readlines():
                    lines = i.strip('\n')
                    train_.append(lines.replace('\\', '/').split('/')[-1][:-6])
            with open(txt2, 'r') as f:
                for i in f.readlines():
                    lines = i.strip('\n')
                    val_.append(lines.replace('\\', '/').split('/')[-1][:-6])
            X, y, name = proc([path1, path2, path3, path4])
            X_train, y_train, X_test, y_test = [], [], [], []
            for (a, b, c) in zip(X, y, name):
                if c in train_:
                    X_train.append(a)
                    y_train.append(b)
                elif c in val_:
                    X_test.append(a)
                    y_test.append(b)
            temp = list(zip(X_train, y_train))
            random.shuffle(temp)
            X_train[:], y_train[:] = zip(*temp)

            temp = list(zip(X_test, y_test))
            random.shuffle(temp)
            X_test[:], y_test[:] = zip(*temp)

            predict_results = classifier(model_name, X_train, y_train, X_test, y_test, seeds)
            fpr, tpr, threshold = roc_curve(y_test, predict_results[:, 1])
            matrix_ = confusion_matrix(y_test, np.argmax(predict_results, axis=-1))
            TN, FP, FN, TP = function1(matrix_)
            recall_ = recall_score(y_test, np.argmax(predict_results, axis=-1))
            sensitivity = TP / (TP + FN)
            specificity = TN / (TN + FP)
            acc_ = (TP + TN) / (TP + FP + FN + TN)
            # print(acc_)

            roc_auc = auc(fpr, tpr)
            acc_seeds.append(acc_)
            sensitivity_seeds.append(sensitivity)
            specificity_seeds.append(specificity)
            auc_seeds.append(roc_auc)

    acc_seeds, sensitivity_seeds, specificity_seeds, auc_seeds = mean_std(
        [acc_seeds, sensitivity_seeds, specificity_seeds, auc_seeds])
    print('model: {}'.format(model_name))
    print('random cross validation mean acc: {} std: {}'.format(acc_seeds[0], acc_seeds[1]))
    print('random cross validation mean auc: {} std: {}'.format(auc_seeds[0], auc_seeds[1]))
    print('random cross validation mean sensitivity: {} std: {}'.format(sensitivity_seeds[0], sensitivity_seeds[1]))
    print('random cross validation mean specificity: {} std: {}'.format(specificity_seeds[0], specificity_seeds[1]))
