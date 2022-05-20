from pyparsing import NotAny
import torch
import torch.nn as nn


class GeneralNet(nn.Module):

    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

    def step(self, progress: float):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError