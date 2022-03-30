from utils import *
from global_settings import *
import numpy as np
import matplotlib.pyplot as plt
from net.CNN import *
from net.Re_CNN import *


def load_state(model_name, net, kl, dim, seed):
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


def min_max(npy):
    return (npy - np.min(npy) + 0.000001) / (np.max(npy) - np.min(npy) + 0.000001)


if __name__ == '__main__':
    kl, dim, seed = 1.0, 100, 0
    parts = 50
    model_name_1 = 'Re_CNN_mf'
    model_name_2 = r'CNN'
    neuron = 0
    net1 = Re_CNN_mf(latent_dims=dim)
    load_state(model_name_1, net1, kl, dim, seed)
    net2 = conventional_CNN()
    load_state(model_name_2, net2, kl, dim, seed)
    for name, parameters in net1.named_parameters():
        if name == 'classifier.0.weight':
            np_weight_1 = parameters.cpu().detach().numpy()
    for name, parameters in net2.named_parameters():
        if name == 'classifier.0.weight':
            np_weight_2 = parameters.cpu().detach().numpy()
    x_axis1 = np.array([(1 / parts) * i for i in range(parts + 1)])

    aa = min_max(np.mean(np.abs(np_weight_1[..., :dim]), axis=0))
    bb = min_max(np.mean(np.abs(np_weight_2[..., :dim]), axis=0))

    count_aa, count_bb = [], []
    for i in range(parts):
        count_ = 0
        for j in aa:
            if x_axis1[i] < j <= x_axis1[i + 1]:
                count_ += 1
        count_aa.append(count_)
    for i in range(parts):
        count_ = 0
        for j in bb:
            if x_axis1[i] < j <= x_axis1[i + 1]:
                count_ += 1
        count_bb.append(count_)
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.bar(x_axis1[1:], count_aa, width=(1 / parts))
    plt.xlabel('value')
    plt.ylabel('counts')
    plt.title('Re-CNN')
    plt.subplot(1, 3, 2)
    plt.bar(x_axis1[1:], count_bb, width=(1 / parts))
    plt.xlabel('value')
    plt.ylabel('counts')
    plt.title('CNN')
    color = ['g', 'b']
    alpha = [1, 0.5]
    model_name = [model_name_1, model_name_2]
    plt.subplot(1, 3, 3)
    for nums, (i) in enumerate([count_aa, count_bb]):
        plt.bar(x_axis1[1:], i, width=(1 / parts), color=color[nums], alpha=alpha[nums])

    plt.legend(model_name, loc=0, borderaxespad=1)
    plt.xlabel('value')
    plt.ylabel('counts')
    plt.savefig(r'Figures/Figure5.png')
    plt.show()
