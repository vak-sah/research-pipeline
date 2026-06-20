"""Auxiliary commands.

  rp-mlflow-ui        -> launch the MLflow UI against ./mlruns
  rp-export-figures   -> export portable PNG figures for one MLflow run

We rely on MLflow for live metric charts and do NOT auto-generate PNGs during
training; figure export is an explicit, separate step for portable artifacts.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Keep using MLflow's local file backend (required); opt out of the v3 gate.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


def mlflow_ui() -> None:
    """Launch the MLflow UI bound to the local file store."""
    cmd = ["mlflow", "ui", "--backend-store-uri", "file:./mlruns"] + sys.argv[1:]
    raise SystemExit(subprocess.call(cmd))


def export_figures() -> None:
    """Render PNG line charts of every logged metric for a single run."""
    parser = argparse.ArgumentParser(description="Export PNG figures for an MLflow run.")
    parser.add_argument("run", help="MLflow run id, or run name (with --experiment)")
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--experiment", default=None, help="experiment name (to resolve run by name)")
    parser.add_argument("--out", default=None, help="output dir (default: outputs/figures/<run_id>)")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)

    run_id = args.run
    # Resolve by name if it doesn't look like a run id.
    try:
        client.get_run(run_id)
    except Exception:
        if args.experiment is None:
            raise SystemExit(f"Run '{run_id}' not found; pass --experiment to resolve by name.")
        exp = client.get_experiment_by_name(args.experiment)
        if exp is None:
            raise SystemExit(f"Experiment '{args.experiment}' not found.")
        matches = client.search_runs(
            [exp.experiment_id], filter_string=f"tags.mlflow.runName = '{run_id}'"
        )
        if not matches:
            raise SystemExit(f"No run named '{run_id}' in experiment '{args.experiment}'.")
        run_id = matches[0].info.run_id

    run = client.get_run(run_id)
    out_dir = Path(args.out) if args.out else Path("outputs/figures") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_keys = list(run.data.metrics.keys())
    if not metric_keys:
        raise SystemExit(f"Run {run_id} has no logged metrics.")

    for key in metric_keys:
        history = client.get_metric_history(run_id, key)
        history = sorted(history, key=lambda m: m.step)
        steps = [m.step for m in history]
        values = [m.value for m in history]
        fig, ax = plt.subplots()
        ax.plot(steps, values, marker="o")
        ax.set_title(key)
        ax.set_xlabel("step")
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)
        safe = key.replace("/", "_")
        fig.savefig(out_dir / f"{safe}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote {len(metric_keys)} figure(s) to {out_dir}")
