# research-pipeline

Domain-agnostic PyTorch Lightning + Hydra research pipeline for fast, modular
idea-testing. Runs on Apple Silicon (MPS), NVIDIA (CUDA), and CPU, tuned per
device. No internet required at runtime (one-time dataset download is fine).

See [CLAUDE.md](CLAUDE.md) for the component contracts, how to add each kind of
component, and which changes are safe vs may ripple.

## Install

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e .              # add ".[dev]" for pytest
```

(Plain `python -m venv .venv && pip install -e .` works too.)

## Run

```bash
rp-train                                       # default config (CIFAR-10, small CNN)
rp-train +experiment=cifar10_baseline          # run a named, immutable experiment
rp-train +experiment=cifar10_baseline optimizer.lr=0.005   # CLI-override a hyperparameter
CUDA_VISIBLE_DEVICES= rp-train                 # force the CPU/MPS path (skip CUDA)
```

Device & precision are chosen automatically by `utils/device.py` (`accelerator=auto`:
CUDA→`bf16-mixed`+TF32+optional `torch.compile`, MPS/CPU→`32-true`). They are owned
by the device plan, not the CLI — to force the non-CUDA path, hide CUDA as shown above.

Each run writes `outputs/<timestamp>/` with the best checkpoint, the resolved
config, the seed, and the git SHA. Metrics stream to MLflow (`./mlruns`).

## Swap components (one config + one class — never edit the training loop)

```bash
rp-train model=mlp                             # swap model    (configs/model/mlp.yaml)
rp-train optimizer=sgd scheduler=cosine        # swap optimizer/scheduler
rp-train data=synthetic                        # swap dataset  (configs/data/synthetic.yaml)
```

| Add a… | Do this | Run |
|---|---|---|
| **model** | class returning a dict `{"logits": …}` in `models/<name>.py` + `configs/model/<name>.yaml` (`_target_`) | `rp-train model=<name>` |
| **loss term** | a `terms["<name>"] = weight * value` line in `CompositeLoss` + its weight in `configs/loss/composite.yaml` | (active once weighted) |
| **dataset** | a `LightningDataModule` yielding `(input, target)` + `configs/data/<name>.yaml` | `rp-train data=<name>` |
| **transform** | a factory in `data/transforms.py` + point the `transform:` slot in `configs/data/<name>.yaml` | (used by that data config) |
| **splitter** | a `dataset -> (train_idx, val_idx)` callable in `data/splitter.py` + point the `splitter:` slot | (used by that data config) |

## Track

```bash
rp-mlflow-ui                                   # MLflow UI at http://127.0.0.1:5000
rp-export-figures <run_id>                     # portable PNGs for one run -> outputs/figures/<run_id>/
rp-export-figures <run_name> --experiment cifar10-erm   # resolve a run by name
```

We rely on MLflow for live metric charts; training does **not** auto-generate PNGs.

## Experiments are immutable

`configs/experiment/*.yaml` are named, immutable override bundles. **To change an
experiment, clone it to a new file — never edit one in place.** This keeps past
runs comparable across methods.

```bash
cp configs/experiment/cifar10_baseline.yaml configs/experiment/cifar10_bigger_lr.yaml   # then edit the clone
```

## Test

```bash
pytest                                         # objective test suite (contracts, not just "no crash")
```
