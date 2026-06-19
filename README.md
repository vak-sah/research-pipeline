# research-pipeline

Domain-agnostic PyTorch Lightning + Hydra research pipeline for fast, modular
idea-testing. Runs on Apple Silicon (MPS), NVIDIA (CUDA), and CPU, tuned per
device. No internet required at runtime (one-time dataset download is fine).

See [CLAUDE.md](CLAUDE.md) for the full component contracts and how to extend.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # add ".[dev]" for pytest
```

## Train

```bash
rp-train                                  # default config (CIFAR-10, small CNN)
rp-train +experiment=cifar10_baseline     # the named ERM baseline
rp-train model=mlp optimizer=sgd          # swap components from the CLI
```

Each run writes `outputs/<timestamp>/` with the best checkpoint, the resolved
config, the seed, and the git SHA. Metrics go to MLflow (`./mlruns`).

## Track

```bash
rp-mlflow-ui                              # MLflow UI at http://127.0.0.1:5000
rp-export-figures <run_id>                # portable PNGs for one run
```

## Test

```bash
pytest
```
