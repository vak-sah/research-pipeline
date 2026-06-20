"""Objective-level tests: verify the pipeline's CONTRACTS, not just "no crash"."""
import glob
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

from conftest import CONFIG_DIR, PROJECT_ROOT


def _compose(overrides):
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="config", overrides=overrides)


def test_model_output_is_dict_and_loss_has_total_plus_terms():
    from research_pipeline.losses.composite import CompositeLoss
    from research_pipeline.models.cnn import SmallCNN

    model = SmallCNN(num_classes=10)
    out = model(torch.randn(4, 3, 32, 32))
    assert isinstance(out, dict), "model must return a dict, never a bare tensor"
    assert "logits" in out

    loss = CompositeLoss()
    terms = loss(out, torch.randint(0, 10, (4,)))
    assert "total" in terms, "composite loss must return a 'total'"
    assert "ce" in terms, "composite loss must expose per-term keys"
    # total equals the sum of the (single) weighted term today.
    assert torch.allclose(terms["total"], terms["ce"])


def test_device_plan_matches_this_machine():
    from research_pipeline.utils.device import resolve_device_plan

    plan = resolve_device_plan()
    if torch.cuda.is_available():
        assert plan.accelerator == "gpu"
        assert plan.precision == "bf16-mixed"
        assert plan.pin_memory is True
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        assert plan.accelerator == "mps"
        assert plan.precision == "32-true"
        assert plan.compile is False
    else:
        assert plan.accelerator == "cpu"
        assert plan.precision == "32-true"
        assert plan.compile is False


def test_overriding_model_and_optimizer_instantiates():
    cfg = _compose(["model=mlp", "optimizer=sgd"])
    model = instantiate(cfg.model)
    out = model(torch.randn(2, 3, 32, 32))
    assert "logits" in out
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    assert optimizer.__class__.__name__ == "SGD"


def test_invalid_override_fails_at_startup():
    # 'trainer.max_epochs' is a typed int in the structured config; a non-int
    # must be rejected at compose (startup) time, not mid-training.
    with pytest.raises(Exception):
        _compose(["trainer.max_epochs=not_an_int"])


def test_misspelled_field_fails_at_startup():
    # struct mode: unknown key on a typed node is rejected at startup.
    with pytest.raises(Exception):
        _compose(["trainer.max_epochsX=3"])


def test_smoke_two_step_fit_on_cpu_no_download():
    import lightning.pytorch as pl

    from research_pipeline.lit_module import LitClassifier

    cfg = _compose(["data=synthetic", "model=cnn"])
    model = instantiate(cfg.model)
    loss = instantiate(cfg.loss)
    datamodule = instantiate(cfg.data)
    lit = LitClassifier(model, loss, cfg.optimizer, cfg.scheduler)

    trainer = pl.Trainer(
        max_steps=2,
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(lit, datamodule)
    assert trainer.global_step >= 2


def test_short_run_writes_artifacts_and_mlflow_metrics():
    """End-to-end: a short real run writes per-run artifacts and an MLflow run."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    outputs = PROJECT_ROOT / "outputs"
    before = set(glob.glob(str(outputs / "*")))

    cmd = [
        sys.executable, "-m", "research_pipeline.train",
        "data=synthetic",
        "trainer.max_epochs=1",
        "trainer.limit_train_batches=2",
        "trainer.limit_val_batches=2",
        "mlflow.experiment_name=pytest-smoke",
        "run_name=pytest_run",
    ]
    result = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, f"train failed:\n{result.stderr[-3000:]}"

    after = set(glob.glob(str(outputs / "*")))
    new_dirs = sorted(after - before)
    assert new_dirs, "no new run directory was created under outputs/"
    run_dir = Path(new_dirs[-1])

    # Per-run artifacts.
    assert (run_dir / "config_resolved.yaml").exists()
    assert (run_dir / "seed.txt").exists()
    assert (run_dir / "git_sha.txt").exists()
    checkpoints = list((run_dir / "checkpoints").glob("*.ckpt"))
    assert checkpoints, "no best checkpoint was saved"

    # MLflow run with logged metrics.
    import mlflow

    mlflow.set_tracking_uri(f"file:{PROJECT_ROOT / 'mlruns'}")
    experiment = mlflow.get_experiment_by_name("pytest-smoke")
    assert experiment is not None, "MLflow experiment was not created"
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) >= 1, "no MLflow run recorded"
    metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
    assert metric_cols, "no metrics were logged to MLflow"
