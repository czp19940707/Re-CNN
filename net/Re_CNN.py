# coding: utf8
import torch
import torch.nn.functional as F

from net.modules import PadMaxPool3d, Flatten
import torch.nn as nn
from torch.autograd import Variable


class Re_CNN_mf(nn.Module):
    """
    Re_CNN with morphological metrics
    """

    def __init__(self, n_classes=2, latent_dims=100):
        super(Re_CNN_mf, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )

        self.mf_dims = 204
        self._enc_mu = torch.nn.Linear(128 * 3 * 4 * 3, latent_dims)
        self._enc_log_sigma = torch.nn.Linear(128 * 3 * 4 * 3, latent_dims)

        self.classifier = nn.Sequential(

            nn.Linear(latent_dims + self.mf_dims, 1024),
            nn.ReLU(),

            nn.Linear(1024, 50),
            nn.ReLU(),

            nn.Linear(50, n_classes)

        )

    def reparametrize(self, h_enc, stage):
        mu = self._enc_mu(h_enc)
        logvar = self._enc_log_sigma(h_enc)
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)

        if stage == 'Training':
            return eps.mul(std).add_(mu), mu, std
        else:
            return mu, mu, std

    def forward(self, x, mf, stage='Training'):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x, mu, std = self.reparametrize(x, stage)
        x = torch.cat([x, mf], dim=-1)
        x = self.classifier(x)
        return x, mu, std


class Re_CNN(nn.Module):
    """
    Re_CNN with no morphological metrics
    """

    def __init__(self, latent_dims=100):
        super(Re_CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )
        self._enc_mu = torch.nn.Linear(128 * 3 * 4 * 3, latent_dims)
        self._enc_log_sigma = torch.nn.Linear(128 * 3 * 4 * 3, latent_dims)
        self.classifier = nn.Sequential(

            nn.Linear(latent_dims, 1024),
            nn.ReLU(),

            nn.Linear(1024, 50),
            nn.ReLU(),

            nn.Linear(50, 2)

        )

        # self.flattened_shape = [-1, 128, 6, 7, 6]

    def reparametrize(self, h_enc, stage):
        mu = self._enc_mu(h_enc)
        logvar = self._enc_log_sigma(h_enc)
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)

        if stage == 'Training':
            return eps.mul(std).add_(mu), mu, std
        else:
            return mu, mu, std

    def forward(self, x, stage='Training'):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x, mu, std = self.reparametrize(x, stage)
        x = self.classifier(x)
        return x, mu, std


if __name__ == '__main__':
    net = Re_CNN_mf().cuda()
    input_ = torch.randn(2, 1, 80, 100, 80).cuda()
    sd = torch.randn(2, 204).cuda()
    with torch.no_grad():
        out = net.forward(input_, sd)

