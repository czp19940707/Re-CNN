import argparse
from utils import get_network, model_loss, WarmUpLR, most_recent_folder, best_acc_weights, most_recent_weights, \
    last_epoch, get_dataset, optimizer_select
from dataProc import dataProc
from torch.utils.data import DataLoader
import torch
from global_settings import *
import os
import time
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score, recall_score
import torch.nn.functional as F
from metric import metric


def train(epoch):
    start = time.time()
    net.train()
    for batch_index, dp_list in enumerate(Data_training_loader):
        images, morphological_metrics, label, name = dp_list
        if args.gpu:
            label = label.cuda()
            morphological_metrics = morphological_metrics.cuda().float()
            images = images.cuda()
        optimizer.zero_grad()
        if args.net.endswith('mf'):
            if args.net.startswith('Re'):
                pred_cls, mu, std = net(images, morphological_metrics, stage='Training')
                train_loss = ml.classification_loss(pred_cls, label) + args.kl * ml.ll_loss(mu, std)
            else:
                pred_cls = net(images, morphological_metrics)
                train_loss = ml.classification_loss(pred_cls, label)
        else:
            if args.net.startswith('Re'):
                pred_cls, mu, std = net(images, stage='Training')
                train_loss = ml.classification_loss(pred_cls, label) + args.kl * ml.ll_loss(mu, std)
            else:
                pred_cls = net(images)
                train_loss = ml.classification_loss(pred_cls, label)
        train_loss_list.append(train_loss.cpu().detach().numpy())
        train_loss.backward()
        optimizer.step()

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            train_loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b,
            total_samples=len(Data_training)
        ))
        if epoch <= args.warm:
            warmup_scheduler.step()

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))


@torch.no_grad()
def eval_training(epoch=0, tb=True):
    start = time.time()
    net.eval()
    test_loss_ = 0.0  # cost function error
    correct = 0.0
    prob_list = np.array([])
    label_list = np.array([])
    for dp_list in Data_test_loader:
        images, morphological_metrics, label, name = dp_list
        if args.gpu:
            label = label.cuda()
            morphological_metrics = morphological_metrics.cuda().float()
            images = images.cuda()

        if args.net.endswith('mf'):
            if args.net.startswith('Re'):
                pred_cls, mu, std = net(images, morphological_metrics, stage='Training')
                test_loss = ml.classification_loss(pred_cls, label) + args.kl * ml.ll_loss(mu, std)
            else:
                pred_cls = net(images, morphological_metrics)
                test_loss = ml.classification_loss(pred_cls, label)
        else:
            if args.net.startswith('Re'):
                pred_cls, mu, std = net(images, stage='Training')
                test_loss = ml.classification_loss(pred_cls, label) + args.kl * ml.ll_loss(mu, std)
            else:
                pred_cls = net(images)
                test_loss = ml.classification_loss(pred_cls, label)
        test_loss_ += test_loss.item()
        _, preds = pred_cls.max(1)
        correct += preds.eq(label).sum()
        prob = F.softmax(pred_cls[:, 0]).cpu().numpy()
        prob_list = np.concatenate((prob_list, prob))
        label_list = np.concatenate((label_list, label.cpu()))

    _, _, auc_value = metric.auc(label_list, prob_list, pos_label=0)
    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print(
        'Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f},  AUC_value: {:.4f}, Time consumed:{:.2f}s'.format(
            epoch,
            test_loss_ / len(Data_test),
            correct.float() / len(Data_test),
            auc_value,
            finish - start
        ))
    print()
    return correct.float() / len(Data_test_loader), auc_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='Re_CNN_mf', help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=32, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=2, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.0005, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-m', type=list, default=[40, 80], help='learning rate decline by milestones')
    parser.add_argument('-epoch', type=int, default=120, help='epoch')
    parser.add_argument('-dn', type=str, default='ADNI', help='The dataset your want to training')
    parser.add_argument('-optim', type=str, default='Adam', help='The optimizer your want to use')
    parser.add_argument('-save', type=str, default='checkpoints', help='save checkpoints')
    parser.add_argument('-if_', type=str, default='hipp', help='input_format, select image or patch')
    parser.add_argument('-kl', type=float, default=1., help='superparameter kl')
    parser.add_argument('-dims', type=float, default=100, help='reparametrized deep semantic feature dims')
    parser.add_argument('-seed', type=int, default=0, help='random cross-validation seed')

    args = parser.parse_args()
    # select model
    net = get_network(args)
    # load dataset
    Data_training, Data_test = get_dataset(args)
    sample_weight, imbalanced_ratio = Data_training.get_sample_weights()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
    Data_training_loader = DataLoader(
        Data_training, sampler=sampler, num_workers=8, batch_size=args.b)

    Data_test_loader = DataLoader(
        Data_test, shuffle=False, num_workers=8, batch_size=args.b
    )

    # loss
    ml = model_loss(class_num=Data_training.n_classes)
    loss_save_file = r'Train loss'
    if not os.path.exists(loss_save_file):
        os.makedirs(loss_save_file)
    # init optimizer
    optimizer, train_scheduler = optimizer_select(args, net)

    iter_per_epoch = len(Data_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    CHECKPOINT_PATH = args.save
    if args.net.startswith('Re'):
        net_name = os.path.join(args.net, 'kl_{}'.format(args.kl), 'dims_{}'.format(args.dims))
    else:
        net_name = args.net
    if args.resume:
        recent_folder = most_recent_folder(os.path.join(CHECKPOINT_PATH, 'seed_{}'.format(args.seed), net_name),
                                           fmt=DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(CHECKPOINT_PATH, 'seed_{}'.format(args.seed), net_name, recent_folder)

    else:
        checkpoint_path = os.path.join(CHECKPOINT_PATH, 'seed_{}'.format(args.seed), net_name, TIME_NOW)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    best_auc = 0.0
    if args.resume:
        best_weights = best_acc_weights(
            os.path.join(CHECKPOINT_PATH, 'seed_{}'.format(args.seed), net_name, recent_folder))
        if best_weights:
            weights_path = os.path.join(CHECKPOINT_PATH, 'seed_{}'.format(args.seed), net_name, recent_folder,
                                        best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(
            os.path.join(CHECKPOINT_PATH, 'seed_{}'.format(args.seed), net_name, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(CHECKPOINT_PATH, 'seed_{}'.format(args.seed), net_name, recent_folder,
                                    recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(CHECKPOINT_PATH, 'seed_{}'.format(args.seed), net_name, recent_folder))

    # 0914
    train_loss_list = []

    for epoch in range(1, args.epoch + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)
        acc, auc_ = eval_training(epoch)

        # start to save best performance model after learning rate decay to 0.01
        if epoch > args.m[0] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            checkpoint_path_ = os.path.split(checkpoint_path)[0]
            for i in os.listdir(checkpoint_path_):
                if i.endswith('best_acc.pth'):
                    os.remove(os.path.join(checkpoint_path_, i))
            print('saving weights file to {}'.format(weights_path[:-4] + '_acc.pth'))
            torch.save(net.state_dict(), weights_path[:-4] + '_acc.pth')
            best_acc = acc
            continue

        if epoch > args.m[0] and best_auc < auc_:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            checkpoint_path_ = os.path.split(checkpoint_path)[0]
            for i in os.listdir(checkpoint_path_):
                if i.endswith('best_auc.pth'):
                    os.remove(os.path.join(checkpoint_path_, i))
            print('saving weights file to {}'.format(weights_path[:-4] + '_auc.pth'))
            torch.save(net.state_dict(), weights_path[:-4] + '_auc.pth')
            best_auc = auc_
            continue

        if not epoch % SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            checkpoint_path_ = os.path.split(checkpoint_path)[0]
            for i in os.listdir(checkpoint_path_):
                if i.endswith('regular.pth'):
                    os.remove(os.path.join(checkpoint_path_, i))
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)


    x_train_loss_list = np.array([i * args.b for i in range(len(train_loss_list))])
    import matplotlib.pyplot as plt

    plt.xlabel('iters')
    plt.ylabel('loss')
    plt.title('Loss\nmodel:{}'.format(args.net))
    plt.plot(x_train_loss_list, train_loss_list, c='blue')
    plt.savefig(os.path.join(loss_save_file, '{}_Loss.png'.format(args.net)))
