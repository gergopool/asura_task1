import torch
import torch.nn as nn

from .general import GeneralNet

__all__ = ['tiny']


class ConvBlock(nn.Module):

    def __init__(self, in_n: int, out_n: int, k: int = 3, s: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_n,
                              out_channels=out_n,
                              kernel_size=k,
                              stride=s,
                              padding=1)
        self.bn = nn.BatchNorm2d(out_n)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TinyNet(GeneralNet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = nn.Sequential(
            ConvBlock(3, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2),
            ConvBlock(128, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(2),
            ConvBlock(256, 512),
        )
        self.fc = nn.Linear(in_features=512, out_features=self.n_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(dim=(-2, -1))
        x = self.fc(x)
        return x


def tiny():
    return TinyNet