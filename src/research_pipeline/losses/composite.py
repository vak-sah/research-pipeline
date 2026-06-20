"""Composite loss.

CONTRACT: ``forward(outputs, targets)`` consumes the model-output dict plus
targets and returns a dict of named, weighted terms plus a ``"total"``.

Today there is a single cross-entropy term, so the output is
``{"ce": ..., "total": ...}``. Adding a term later (e.g. a contrastive term on
an ``embedding`` key) stays additive: read the key(s) you need, add a
``"<name>": weight * value`` entry, and keep ``total`` as their sum. The
canonical way to add a term is to extend this class and expose its weight in
``configs/loss/composite.yaml`` -- the training loop never changes.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class CompositeLoss(nn.Module):
    def __init__(self, ce_weight: float = 1.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        terms: Dict[str, torch.Tensor] = {}
        terms["ce"] = self.ce_weight * self.ce(outputs["logits"], targets)
        # To add a term later, append it here, e.g.:
        #   terms["contrastive"] = self.contrastive_weight * contrastive(outputs["embedding"], targets)
        terms["total"] = sum(terms.values())
        return terms
