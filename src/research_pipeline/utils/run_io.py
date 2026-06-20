"""Per-run artifact I/O.

Writes the resolved config, the seed, and the git SHA into the run directory.
The best checkpoint is written separately by a ``ModelCheckpoint`` callback.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # noqa: BLE001 -- not a git repo / git missing
        return "unknown"


def save_run_metadata(out_dir, cfg: DictConfig, seed: int) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=str(out_dir / "config_resolved.yaml"), resolve=True)
    (out_dir / "seed.txt").write_text(f"{seed}\n")
    (out_dir / "git_sha.txt").write_text(f"{get_git_sha()}\n")
