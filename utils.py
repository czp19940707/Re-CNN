import sys

import torchvision.models
from torch import nn
import torch
from torch.optim.lr_scheduler import _LRScheduler
import os
import re
import datetime
from torch.nn import init
from torch.utils.data import random_split
from torch.autograd import Variable
import torch.nn.functional as F


def get_network(args):
    if args.net == 'Re_CNN':
        from net.Re_CNN import Re_CNN
        latent_dims = int(args.dims)
        net = Re_CNN(latent_dims=latent_dims)
    elif args.net == 'Re_CNN_mf':
        from net.Re_CNN import Re_CNN_mf
        latent_dims = int(args.dims)
        net = Re_CNN_mf(latent_dims=latent_dims)

    elif args.net == 'CNN':
        from net.CNN import conventional_CNN
        net = conventional_CNN()

    elif args.net == 'CNN_mf':
        from net.CNN import conventional_CNN_mf
        net = conventional_CNN_mf()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if torch.cuda.is_available():  # use_gpu
        net = net.cuda()

    return net


def get_dataset(args):
    if args.dn == 'ADNI':
        from dataProc import dataProc
        return dataProc(split='Training', if_=args.if_, seed=args.seed), dataProc(split='Validation', if_=args.if_, seed=args.seed)


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def optimizer_select(args, net):
    if args.optim == 'SGD':
        import torch.optim as optim
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.m, gamma=0.2)
        return optimizer, train_scheduler
    elif args.optim == 'Ranger':
        from ranger import Ranger
        import torch.optim as optim
        optimizer = Ranger(net.parameters(), lr=args.lr, weight_decay=5e-4)
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.m, gamma=0.2)
        return optimizer, train_scheduler
    elif args.optim == 'Adam':
        import torch.optim as optim
        optimizer = optim.Adam(net.parameters(), lr=args.lr, eps=1e-8, betas=(0.9, 0.99))
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.m, gamma=0.2)
        return optimizer, train_scheduler


class model_loss:
    def __init__(self, class_num=2):
        self.criterion = nn.CrossEntropyLoss()
        self.focalloss = FocalLoss(class_num=class_num)

    def ll_loss(self, z_mean, z_stddev):
        mean_sq = z_mean * z_mean
        stddev_sq = z_stddev * z_stddev
        return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

    def svdd_loss(self, features):
        # mean = torch.mean(features)
        # return torch.div(torch.sum((features - mean).pow(2)), len(features))
        features_nonzero = features[(features != 0)]
        mean = torch.mean(features_nonzero)
        return torch.sum((features_nonzero - mean).pow(2))

    def classification_loss(self, pred, label):
        # return self.focalloss(pred, label)
        return self.criterion(pred, label)

    def Hybrid_loss(self, outputs, label):
        Hybrid_loss_ = 0
        for pred in outputs:
            batch_size = pred.shape[0]
            new_batch = []
            for j in range(batch_size):
                pr = pred[j, ...]
                la = label[j]
                index_ = torch.max(pr, dim=1)[1][la]
                pr_max = pr[..., index_][None, ...]
                new_batch.append(pr_max)
            Hybrid_loss_ += self.criterion(torch.cat(new_batch, dim=0), label)
        return Hybrid_loss_

    def Hybrid_loss_1(self, pred, label):
        Hybrid_loss_ = 0
        for i in pred:
            Hybrid_loss_1 = 0
            len_ = i.shape[-1]
            for j in range(len_):
                Hybrid_loss_1 += self.criterion(i[..., j], label)
            Hybrid_loss_ += Hybrid_loss_1 / len_
        return Hybrid_loss_


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]


def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]


def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    best_files = [w for w in files if w.split('-')[-1] == 'best_acc.pth']
    if len(best_files) == 0:
        return ''
    return best_files[-1]


def best_auc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    best_files = [w for w in files if w.split('-')[-1] == 'best_auc.pth']
    if len(best_files) == 0:
        return ''
    return best_files[-1]


def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
        raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    # print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def reset_desnet_key(pretrained_state):
    import collections
    new_state = collections.OrderedDict()
    for name, module in pretrained_state.items():
        if 'denseblock' in name:
            params = name.split('.')
            name = '.'.join(params[:-3] + [params[-3] + params[-2]] + [params[-1]])
            new_state[name] = module
        elif 'classifier' in name:
            break
        elif 'norm5' in name:
            continue
        else:
            new_state[name] = module
    return new_state


# def fixed_parameters(model):
#     import collections
#     model.requires_grad_(True)
#     fixed_state = collections.OrderedDict()
#     model_state = model.state_dict()
#     fixed_list = ['denseblock4', 'norm5', 'ASPP', 'classification', 'transition4']
#     for name, module in model_state.items():
#         for item in fixed_list:
#             if item in name:
#                 try:
#                     fixed_state[name] = module.requires_grad_(True)
#                 except:
#                     pass
#     model_state.update(fixed_state)
#     model.load_state_dict(model_state)
#     return model
def fixed_parameters(model):
    model.requires_grad_(False)
    fixed_list = ['denseblock4', 'norm5', 'ASPP', 'classification', 'transition4']
    for k, v in model.named_parameters():
        for item in fixed_list:
            if item in k:
                v.requires_grad = True
    return model
