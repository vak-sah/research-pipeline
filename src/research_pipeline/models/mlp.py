"""MLP baseline -- alternative model to demonstrate ``model=`` swapping.

CONTRACT: forward returns a DICT, at least {"logits": ...}.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int = 3072, hidden_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"logits": self.net(x)}
