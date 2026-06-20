"""Structured (typed) configs.

These dataclasses are registered with Hydra's ConfigStore so that bad or
misspelled overrides fail at STARTUP (compose time) with a clear error,
instead of deep inside the training loop. Component groups that are built via
``_target_`` instantiation are typed ``Any`` on purpose -- their own schema is
their constructor signature -- while the orchestration-level knobs (trainer,
mlflow, checkpoint, seed) are strictly typed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class TrainerConfig:
    _target_: str = "lightning.pytorch.Trainer"
    max_epochs: int = 10
    accelerator: str = "auto"
    devices: Any = "auto"
    precision: str = "32-true"
    log_every_n_steps: int = 20
    gradient_clip_val: Optional[float] = None
    limit_train_batches: Optional[Any] = None
    limit_val_batches: Optional[Any] = None
    num_sanity_val_steps: int = 0
    enable_progress_bar: bool = True


@dataclass
class MLflowConfig:
    experiment_name: str = "research-pipeline"
    tracking_uri: str = "file:./mlruns"


@dataclass
class CheckpointConfig:
    monitor: str = "val/acc"
    mode: str = "max"


@dataclass
class Config:
    # Component slots filled by Hydra config groups (built via _target_).
    data: Any = MISSING
    model: Any = MISSING
    loss: Any = MISSING
    optimizer: Any = MISSING
    scheduler: Any = MISSING
    callbacks: Any = MISSING
    # Strictly typed orchestration knobs.
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    seed: int = 42
    run_name: Optional[str] = None


def register() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=Config)


# Register on import so `compose(config_name="config")` always sees the schema.
register()
