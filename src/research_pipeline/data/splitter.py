"""Splitter slot (custom train/val splitting).

CONTRACT: a callable ``splitter(dataset) -> (train_indices, val_indices)``.
Wired into the DataModule via ``_target_``; swap splitting strategy (stratified,
grouped, temporal, ...) by pointing the ``splitter`` slot at a different class.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


class RandomSplitter:
    def __init__(self, val_fraction: float = 0.1, seed: int = 42):
        self.val_fraction = val_fraction
        self.seed = seed

    def __call__(self, dataset: Sequence) -> Tuple[List[int], List[int]]:
        n = len(dataset)
        generator = torch.Generator().manual_seed(self.seed)
        order = torch.randperm(n, generator=generator).tolist()
        n_val = int(n * self.val_fraction)
        return order[n_val:], order[:n_val]
