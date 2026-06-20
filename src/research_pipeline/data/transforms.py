"""Transform slot (preprocessing / tokenizer).

A factory returning a torchvision transform, wired into the DataModule via
``_target_``. Swap preprocessing by pointing the ``transform`` slot at a
different factory in ``configs/data/*.yaml`` -- no code change to the DataModule.
"""
from __future__ import annotations

import torchvision.transforms as T

# CIFAR-10 channel statistics.
_MEAN = (0.4914, 0.4822, 0.4465)
_STD = (0.2470, 0.2435, 0.2616)


def default_cifar_transform() -> T.Compose:
    return T.Compose([T.ToTensor(), T.Normalize(_MEAN, _STD)])
