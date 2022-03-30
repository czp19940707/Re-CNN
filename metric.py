from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score, recall_score
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.decomposition import PCA


class metric(object):
    def __init__(self):
        pass
    @staticmethod
    def accuracy_score(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def auc(y_true, y_pred, pos_label):
        fpr, tpr, threshold = roc_curve(y_true, y_pred, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc

    @staticmethod
    def f_1_score(y_true, y_pred):
        return f1_score(y_true, y_pred, average='micro')

    @staticmethod
    def Confusion_matrix(y_true, y_pred):
        return confusion_matrix(y_true, y_pred)

    @staticmethod
    def recall_score(y_true, y_pred):
        return recall_score(y_true, y_pred, average='micro')

def get_AD_risk(raw):
    x1, x2 = raw[0, :, :, :], raw[1, :, :, :]
    risk = np.exp(x2) / (np.exp(x1) + np.exp(x2))
    return risk

def DPM_statistics(DPMs, Labels):
    shape = DPMs[0].shape[1:]
    voxel_number = shape[0] * shape[1] * shape[2]
    TP, FP, TN, FN = np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape)
    for label, DPM in zip(Labels, DPMs):
        risk_map = get_AD_risk(DPM)
        if label == 0:
            TN += (risk_map < 0.5).astype(np.int)
            FP += (risk_map >= 0.5).astype(np.int)
        elif label == 1:
            TP += (risk_map >= 0.5).astype(np.int)
            FN += (risk_map < 0.5).astype(np.int)
    tn = float("{0:.2f}".format(np.sum(TN) / voxel_number))
    fn = float("{0:.2f}".format(np.sum(FN) / voxel_number))
    tp = float("{0:.2f}".format(np.sum(TP) / voxel_number))
    fp = float("{0:.2f}".format(np.sum(FP) / voxel_number))
    matrix = [[tn, fn], [fp, tp]]
    count = len(Labels)
    TP, TN, FP, FN = TP.astype(np.float)/count, TN.astype(np.float)/count, FP.astype(np.float)/count, FN.astype(np.float)/count
    ACCU = TP + TN
    F1 = 2*TP/(2*TP+FP+FN)
    MCC = (TP*TN-FP*FN)/(np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))+0.00000001*np.ones(shape))
    return matrix, ACCU, F1, MCC

def classifier(data, label, n_classes):
    random_state = np.random.RandomState(0)
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.5,
                                                        random_state=0)
    classifier = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True,
                                             random_state=random_state))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    predict_results = np.eye(n_classes)[np.argmax(y_score, axis=1)]
    # 为每个类别计算ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # fpr, tpr, roc_auc = metric.auc(y_test, y_score)
    Confusion_matrix = metric.Confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predict_results, axis=1))
    recall_score = metric.recall_score(np.argmax(y_test, axis=1), np.argmax(predict_results, axis=1))
    acc = accuracy_score(predict_results, y_test)
    fpr, tpr, threshold = roc_curve(np.argmax(y_test, axis=1), y_score[..., 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    return Confusion_matrix, recall_score, acc, fpr, tpr, roc_auc

def feature_select(x):
    pca = PCA(n_components=100)
    newX = pca.fit_transform(x)
    return newX

def statistics(outs, Labels, n_classes):
    new_data, new_label = [], []
    for features, labels in zip(outs, Labels):
        new_data.append(features.flatten())
        new_label.append(labels)
    new_data = np.array(new_data)
    new_data = np.array([i for i in new_data.T if np.sum(i) != 0.]).T
    # new_data = feature_select(new_data)
    new_label = np.eye(n_classes)[np.array(new_label)].squeeze()
    Confusion_matrix, recall_score, acc, fpr, tpr, roc_auc = classifier(new_data, new_label, n_classes)
    return Confusion_matrix, recall_score, acc, fpr, tpr, roc_auc


if __name__ == '__main__':
    # metric = metric()
    # aa = metric.auc(np.array([0, 0, 1, 1]), np.array([3, 4, 5, 1]))
    # print(aa)
    outs = [torch.randn(1, 1, 33, 33, 33) for _ in range(5)]
    Labels = [torch.tensor(0) for _ in range(5)]
    svm_statistics(outs, Labels)