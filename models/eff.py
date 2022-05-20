import torch
from torch import nn
import timm

from .general import GeneralNet

__all__ = ['eff']


class EfficientNetV2(GeneralNet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = timm.create_model('tf_efficientnetv2_s_in21k', pretrained=True)
        self.classifier = nn.Linear(self.encoder.get_classifier().in_features, self.n_classes)
        self.encoder.reset_classifier(num_classes=0, global_pool="avg")
        self.frozen = True

    def step(self, progress: float):
        if progress >= 0.2 and self.frozen:
            self.frozen = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.frozen:
            with torch.no_grad():
                x = self.encoder(x)
        else:
            x = self.encoder(x)

        x = self.classifier(x)
        return x


def eff():
    return EfficientNetV2
