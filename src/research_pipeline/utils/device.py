"""Device-aware execution plan.

accelerator=auto, with precision and a few knobs chosen per device:
  CUDA : bf16-mixed, TF32 on, optional torch.compile, pin_memory=True
  MPS  : fp32 (32-true), compile off, pin_memory=False
  CPU  : fp32 (32-true), compile off, pin_memory=False
DataLoader worker counts are defaulted per OS (conservative on macOS, where
fork-based workers are fragile).
"""
from __future__ import annotations

import os
import platform
from dataclasses import dataclass

import torch


@dataclass
class DevicePlan:
    accelerator: str
    precision: str
    compile: bool
    num_workers: int
    pin_memory: bool


def _cpu_count() -> int:
    return os.cpu_count() or 1


def resolve_device_plan(allow_compile: bool = True) -> DevicePlan:
    if torch.cuda.is_available():
        # TF32 for faster fp32 matmuls / convolutions on Ampere+.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        return DevicePlan(
            accelerator="gpu",
            precision="bf16-mixed",
            compile=allow_compile,
            num_workers=min(_cpu_count(), 8),
            pin_memory=True,
        )

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        # macOS: keep worker count conservative (fork-based workers are fragile).
        workers = min(_cpu_count(), 4) if platform.system() == "Darwin" else min(_cpu_count(), 8)
        return DevicePlan(
            accelerator="mps",
            precision="32-true",
            compile=False,
            num_workers=workers,
            pin_memory=False,
        )

    return DevicePlan(
        accelerator="cpu",
        precision="32-true",
        compile=False,
        num_workers=min(_cpu_count(), 4),
        pin_memory=False,
    )
