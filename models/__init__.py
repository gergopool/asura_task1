from torch import nn

from .tiny import *
from .eff import *


def build_network(name: str, n_classes: int) -> nn.Module:
    if name not in globals():
        raise NameError(f"Neural net arch {name} is unknown.")
    return globals()[name]()(n_classes=n_classes)