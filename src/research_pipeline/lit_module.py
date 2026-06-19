"""LightningModule.

Contract
--------
batch        : (input, target)   -- the default collate of (sample, label) pairs
model(input) : Dict[str, Tensor] -- always a dict, at least {"logits": ...}
loss(out, y) : Dict[str, Tensor] -- named weighted terms + "total"

Every loss term plus the total and accuracy are logged per stage. Optimizer and
scheduler are built from their configs in ``configure_optimizers`` via Hydra
instantiate, so swapping either is a one-file config change.
"""
from __future__ import annotations

from typing import Any, Optional

import lightning.pytorch as pl
from hydra.utils import instantiate


class LitClassifier(pl.LightningModule):
    def __init__(self, model, loss, optimizer_cfg, scheduler_cfg: Optional[Any] = None):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        out = self.model(x)
        terms = self.loss(out, y)
        bs = y.size(0)
        for name, value in terms.items():
            self.log(f"{stage}/{name}", value, on_step=False, on_epoch=True,
                     prog_bar=(name == "total"), batch_size=bs)
        acc = (out["logits"].argmax(dim=1) == y).float().mean()
        self.log(f"{stage}/acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        return terms["total"]

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer_cfg, params=self.parameters())
        if self.scheduler_cfg is None:
            return optimizer
        scheduler = instantiate(self.scheduler_cfg, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
