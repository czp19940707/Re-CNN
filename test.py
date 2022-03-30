import os

from utils import *
from dataProc import dataProc
from torch.utils.data import DataLoader
from global_settings import *
import csv
import numpy as np
import torch.nn.functional as F
from metric import metric
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', type=str, default='checkpoints', help='params save path')
    parser.add_argument('-net', type=str, default='CNN', help='model name')
    parser.add_argument('-b', type=int, default=8, help='batch_size')
    parser.add_argument('-if_', type=str, default='hipp', help='input_format, select image or patch')
    parser.add_argument('-kl', type=float, default=1., help='superparameter kl')
    parser.add_argument('-dims', type=float, default=100, help='reparametrized deep semantic feature dims')
    parser.add_argument('-seed', type=float, default=0, help='random cross-validation seed')
    parser.add_argument('-save', type=str, default='outputs', help='results save path')
    args = parser.parse_args()
    if args.net.startswith('Re'):
        net_name = os.path.join(args.net, 'lk_{}'.format(args.kl), 'dims_{}'.format(args.dims))
    else:
        net_name = args.net

    save_path = args.cp
    model_ = args.net
    batch_size = args.b
    classes = ['CN', 'AD']
    # classes
    n_classes = len(classes)
    checkpoint_path = save_path
    # save acc, sp, se, auc to csv file
    csv_file_path = os.path.join(args.save, 'seed_{}'.format(args.seed), net_name)
    if not os.path.exists(csv_file_path):
        os.makedirs(csv_file_path)

    if args.net.startswith('Re'):
        f = open(os.path.join(csv_file_path, '{}_kl_{}_dims_{}.csv'.format(args.net, args.kl, args.dims)), 'w',
                 encoding='utf-8')
    else:
        f = open(os.path.join(csv_file_path, '{}.csv'.format(args.net)), 'w', encoding='utf-8')

    csv_writer = csv.writer(f)
    csv_writer.writerow(['model', 'specificity', 'sensitivity', 'acc', 'auc'])

    cifar10_test = dataProc(split='Testing', if_=args.if_, seed=args.seed)
    cifar10_test_loader = DataLoader(
        cifar10_test, shuffle=False, num_workers=0, batch_size=batch_size, drop_last=True
    )
    net = get_network(args)
    recent_folder = most_recent_folder(os.path.join(checkpoint_path, 'seed_{}'.format(args.seed), net_name), DATE_FORMAT)
    if not recent_folder:
        raise Exception('no recent folder were found')
        # weights_path = most_recent_weights(os.path.join(checkpoint_path, i, recent_folder))
    weights_path = best_acc_weights(os.path.join(checkpoint_path, 'seed_{}'.format(args.seed), net_name, recent_folder))
    if not weights_path:
        weights_path = most_recent_weights(os.path.join(checkpoint_path, 'seed_{}'.format(args.seed), net_name, recent_folder))
        if not weights_path:
            raise Exception('no recent weights file were found')
    net.load_state_dict(torch.load(os.path.join(checkpoint_path, 'seed_{}'.format(args.seed), net_name, recent_folder, weights_path)))
    net.eval()
    class_correct = list(0. for i in range(n_classes))
    class_total = list(0. for i in range(n_classes))
    pred_list = []
    label_list = []
    prob_list = [[] for i in range(n_classes)]
    with torch.no_grad():
        for data in cifar10_test_loader:
            images, morphological_metrics, labels, name = data

            if torch.cuda.is_available():
                labels = labels.cuda()
                morphological_metrics = morphological_metrics.cuda().float()
                images = images.cuda()

            if args.net.startswith('Re'):
                if args.net.endswith('mf'):
                    pred, _, _ = net.forward(images, morphological_metrics, stage='Testing')
                else:
                    pred, _, _ = net.forward(images, stage='Testing')

            else:
                if args.net.endswith('mf'):
                    pred = net.forward(images, morphological_metrics)
                else:
                    pred = net.forward(images)
            _, predicted = torch.max(pred, 1)
            c = (predicted == labels)
            for b in range(batch_size):
                label = labels[b]
                class_correct[label] += c[b].item()
                class_total[label] += 1
                # label list
                label_list.append(label.cpu().item())
                # softmax prob class all
                pred_list.append(predicted[b].cpu().item())
                for j in range(n_classes):
                    prob_list[j].append(F.softmax(pred[b])[j].cpu().numpy())

    # Confusion_matrix
    Confusion_matrix = metric.Confusion_matrix(label_list, pred_list)
    # draw roc curve
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for m in range(n_classes):
        fpr[m], tpr[m], roc_auc[m] = metric.auc(label_list, np.array(prob_list[m]), pos_label=m)

    acc_result_list = []
    acc_result_list.append(args.net)
    print('Model: {}'.format(args.net))
    for n in range(n_classes):
        acc_result_list.append(str(100 * class_correct[n] / class_total[n]))
        print('Accuracy of %5s : %2f %%' % (
            classes[n], 100 * class_correct[n] / class_total[n]))

    print('Mean Accuracy: %2f %%' % (100 * np.sum(np.array(class_correct)) / np.sum(np.array(class_total))))
    acc_result_list.append(100 * np.sum(np.array(class_correct)) / np.sum(np.array(class_total)))
    acc_result_list.append(100 * roc_auc[0])
    csv_writer.writerow(acc_result_list)
