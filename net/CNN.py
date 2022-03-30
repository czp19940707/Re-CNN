import torch.nn as nn
from net.modules import PadMaxPool3d, Flatten
import torch.nn.functional as F
import torch


class conventional_CNN(nn.Module):
    """
    Classifier for a binary classification task
    Image level architecture used on Minimal preprocessing
    """

    def __init__(self):
        super(conventional_CNN, self).__init__()

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
        self.classifier = nn.Sequential(

            nn.Linear(128 * 3 * 4 * 3, 1024),
            nn.ReLU(),

            nn.Linear(1024, 50),
            nn.ReLU(),

            nn.Linear(50, 2)

        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


class conventional_CNN_mf(nn.Module):
    """

    """

    def __init__(self):
        super(conventional_CNN_mf, self).__init__()

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

        self.classifier = nn.Sequential(

            nn.Linear(128 * 3 * 4 * 3 + self.mf_dims, 1024),
            nn.ReLU(),

            nn.Linear(1024, 50),
            nn.ReLU(),

            nn.Linear(50, 2)

        )

    def forward(self, x, mf):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = torch.cat([x, mf], dim=-1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    net = conventional_CNN().cuda()
    input_ = torch.randn(2, 1, 80, 100, 80).cuda()
    mf = torch.randn(2, 204).cuda()
    with torch.no_grad():
        out = net.forward(input_)
    print(out)
