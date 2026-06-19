"""Training entrypoint -- the ONLY training loop.

Adding a model / loss / dataset / optimizer / scheduler / callback never
requires editing this file: every component is built from config via
``hydra.utils.instantiate``.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

# Keep using MLflow's local file backend (required); opt out of the v3 gate.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import hydra
import lightning.pytorch as pl
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from research_pipeline import config_schema  # noqa: F401  (registers structured configs)
from research_pipeline.lit_module import LitClassifier
from research_pipeline.utils.device import resolve_device_plan
from research_pipeline.utils.run_io import save_run_metadata

log = logging.getLogger(__name__)

# Absolute path so the entrypoint works both via `python -m research_pipeline.train`
# and via the installed `rp-train` console script (whose cwd differs).
_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> float:
    pl.seed_everything(cfg.seed, workers=True)
    plan = resolve_device_plan()
    log.info("Device plan: %s", plan)

    out_dir = Path(HydraConfig.get().runtime.output_dir)

    # --- Build components (purely from config) ---
    model = instantiate(cfg.model)
    if plan.compile:
        model = torch.compile(model)
    loss = instantiate(cfg.loss)
    datamodule = instantiate(cfg.data, num_workers=plan.num_workers, pin_memory=plan.pin_memory)
    lit = LitClassifier(model=model, loss=loss, optimizer_cfg=cfg.optimizer, scheduler_cfg=cfg.scheduler)

    # --- Callbacks: best checkpoint (configurable monitor/mode) + config-driven extras ---
    callbacks = [
        ModelCheckpoint(
            dirpath=str(out_dir / "checkpoints"),
            filename="best",
            monitor=cfg.checkpoint.monitor,
            mode=cfg.checkpoint.mode,
            save_top_k=1,
        )
    ]
    if cfg.get("callbacks"):
        callbacks += [instantiate(c) for c in cfg.callbacks.values()]

    # --- Tracking: MLflow local file backend ---
    logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
        run_name=cfg.run_name,
    )
    try:
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    except Exception as exc:  # noqa: BLE001 -- hyperparam logging is best-effort
        log.warning("Could not log hyperparams: %s", exc)

    # --- Per-run artifacts: resolved config, seed, git SHA (checkpoint via callback) ---
    save_run_metadata(out_dir, cfg, cfg.seed)

    trainer = instantiate(
        cfg.trainer,
        accelerator=plan.accelerator,
        precision=plan.precision,
        callbacks=callbacks,
        logger=logger,
    )
    trainer.fit(lit, datamodule)
    # Capture fit metrics before test() replaces callback_metrics with test/*.
    fit_metrics = dict(trainer.callback_metrics)
    test_metrics = trainer.test(lit, datamodule, ckpt_path="best")

    def get(key: str) -> float:
        value = fit_metrics.get(key)
        return float(value) if value is not None else float("nan")

    print("\n=== SUMMARY ===")
    print(f"train_loss(total): {get('train/total'):.4f}")
    print(f"val_acc:           {get('val/acc'):.4f}")
    if test_metrics:
        print(f"test metrics:      {test_metrics[0]}")
    print(f"run_dir:           {out_dir}")
    return get("val/acc")


if __name__ == "__main__":
    main()
