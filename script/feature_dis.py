import torch.cuda
import numpy as np
from utils import *
from dataProc import dataProc
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from global_settings import *
from net.CNN import *
from net.Re_CNN import *


def CNN_forward(net, patch):
    for name, module in net.named_children():
        if name == 'classifier':
            patch = F.adaptive_avg_pool3d(patch, 1)
            patch = patch.view(patch.shape[0], -1)
            break
        patch = module.forward(patch)
    return patch


def Re_CNN_forward(net, patch):
    for name, module in net.named_children():
        if name == '_enc_mu':
            patch_ = patch.view(patch.shape[0], -1)
            patch = module.forward(patch_)
            break
        else:
            patch = module.forward(patch)
    return patch


def normalization(tensor):
    npy = tensor.cpu().detach().numpy().squeeze()
    norm_npy = (npy - np.min(npy) + 0.00000001) / (np.max(npy) - np.min(npy) + 0.00000001)
    return norm_npy


def load_state(model_name, kl, dim, seed):
    checkpoint_path = '../checkpoints'
    if model_name.startswith('Re'):
        model_path = os.path.join('seed_{}'.format(seed), model_name, 'kl_{}'.format(kl), 'dims_{}'.format(dim))
    else:
        model_path = os.path.join('seed_{}'.format(seed), model_name)
    recent_folder = most_recent_folder(os.path.join(checkpoint_path, model_path), DATE_FORMAT)
    if not recent_folder:
        raise Exception('no recent folder were found')
    weights_path = best_acc_weights(os.path.join(checkpoint_path, model_path, recent_folder))
    if not weights_path:
        weights_path = most_recent_weights(os.path.join(checkpoint_path, model_path, recent_folder))
        if not weights_path:
            raise Exception('no recent weights file were found')
    net.load_state_dict(torch.load(os.path.join(checkpoint_path, model_path, recent_folder, weights_path)))


def return_dis(features_):
    x_a = np.linspace(0, 0.95, 20)
    x_b = np.linspace(0.05, 1, 20)
    features_num_list = []
    for a, b in zip(x_a, x_b):
        num = 0
        for i in features_:
            if a <= i < b:
                num += 1
        features_num_list.append(num)
    return features_num_list


if __name__ == '__main__':
    kl, dim, seed = 1.0, 100, 0
    structural_MRI = r'../sample.npy'
    model1, model2 = 'CNN', 'Re_CNN_mf'
    patch = torch.FloatTensor(np.load(structural_MRI)[40:120, 50:150, 10:90][None, None, ...])
    features_list = []
    plt.figure(figsize=(18, 5))
    for nums, (model_name, alpha, color) in enumerate(zip([model1, model2], [1, 0.5], ['g', 'b'])):
        if model_name.startswith('Re'):
            net = Re_CNN_mf(latent_dims=dim)
        else:
            net = conventional_CNN()
        load_state(model_name, kl, dim, seed)
        # data_iter = iter(test_data_loader)
        # patch, sd, label, name = next(data_iter)

        if torch.cuda.is_available():
            patch = patch.cuda()
            net.cuda()
        net.eval()
        if model_name.startswith('Re'):
            features_ = Re_CNN_forward(net, patch)
        else:
            features_ = CNN_forward(net, patch)

        features_ = normalization(features_)[:100]
        features_list.append([features_, model_name, alpha, color])
        x_axis = np.array([i for i in range(len(features_))])

        if model_name.startswith('VAE'):
            model_name = 'Re_CNN'
        plt.subplot(1, 3, nums + 1)
        plt.bar(x_axis, features_)
        plt.title('model: {}'.format(model_name))
        plt.xlabel('n_features')
        plt.ylabel('value')
        # plt.savefig('features_dis/{}_dis.png'.format(model_name))
        # plt.show()
    # plt.close()

    x_ = np.linspace(0.1, 0.95, 20)
    plt.subplot(1, 3, 3)
    for i in features_list:
        features_, model_name, alpha, color = i
        if model_name.startswith('VAE'):
            model_name = 'Re_CNN'
        dis_ = return_dis(features_)
        plt.bar(x_, dis_, width=0.04, color=color, alpha=alpha, label=model_name)
    plt.title('features_dis')
    plt.xlabel('value')
    plt.ylabel('n_features')
    plt.legend()
    plt.savefig('Figures/Figure2.png')
    plt.show()
