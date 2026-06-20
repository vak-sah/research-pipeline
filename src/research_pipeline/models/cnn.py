"""Small CNN baseline.

CONTRACT: forward returns a DICT (never a bare tensor). Today it only emits
``logits``; emitting a dict keeps adding heads/losses (e.g. an ``embedding``)
additive rather than breaking. AdaptiveAvgPool makes the head independent of the
input spatial size, so the same model works on CIFAR (32x32) and tiny synthetic
inputs (8x8) used by the smoke test.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3, width: int = 48):
        super().__init__()
        self.features = nn.Sequential(
            _block(in_channels, width),
            _block(width, width * 2),
            _block(width * 2, width * 4),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(width * 4, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return {"logits": self.head(h)}
