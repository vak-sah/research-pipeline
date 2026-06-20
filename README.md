# research-pipeline

Domain-agnostic PyTorch Lightning + Hydra research pipeline for fast, modular
idea-testing. Runs on Apple Silicon (MPS), NVIDIA (CUDA), and CPU, tuned per
device. No internet required at runtime (one-time dataset download is fine).

See [CLAUDE.md](CLAUDE.md) for the component contracts, how to add each kind of
component, and which changes are safe vs may ripple.

## How it works

The whole repo is built around one idea: **a single training loop you never edit,
fed by swappable components selected from config.**

- **`train.py` is the only training loop.** It builds every piece — model, loss,
  data, optimizer, scheduler, callbacks — from config via Hydra `_target_`
  instantiation. Testing a new idea means adding a component, not editing the loop.
- **Each component is a Hydra config group.** `configs/<group>/<name>.yaml` picks
  one implementation for that slot; the CLI swaps it (`rp-train model=mlp`). The
  groups are `data, model, loss, optimizer, scheduler, trainer, callbacks,
  experiment`.
- **Two contracts are kept deliberately rich; everything else is YAGNI.** The
  model returns a **dict** (`{"logits": …}`, never a bare tensor) and the loss is
  **composite** (named weighted terms + a `total`). This is what makes adding a
  new head or loss term *additive* instead of breaking.
- **Device & precision are automatic** (`accelerator=auto`, resolved by
  `utils/device.py`) — not something you tune per run.
- **Structured (typed) configs** mean a bad or misspelled override fails at
  **startup** with a clear error, not 20 minutes into training.

A typical loop: clone an experiment → tweak the clone or pass CLI overrides →
`rp-train` → watch metrics in MLflow → compare runs. Past experiment configs stay
immutable so comparisons remain honest.

## Project structure

```
configs/                     # one Hydra group per swappable component
  config.yaml                #   root: attaches typed schema + composes all groups
  data/      {cifar10,synthetic}.yaml
  model/     {cnn,mlp}.yaml
  loss/      composite.yaml
  optimizer/ {adamw,sgd}.yaml
  scheduler/ cosine.yaml
  trainer/   default.yaml
  callbacks/ default.yaml
  experiment/cifar10_baseline.yaml   # IMMUTABLE named bundle of overrides

src/research_pipeline/
  train.py                   # @hydra.main entrypoint — the ONLY training loop
  config_schema.py           # typed dataclasses → ConfigStore (startup validation)
  lit_module.py              # LitClassifier: model→dict, loss→dict, logs every term
  cli.py                     # rp-mlflow-ui, rp-export-figures
  models/    {cnn,mlp}.py     # forward(x) → dict {"logits": …}
  losses/    composite.py     # CompositeLoss → {"total", "ce", …}
  data/
    datamodule.py            # CIFAR10DataModule (transform/splitter slots), Synthetic
    transforms.py            #   transform slot
    splitter.py              #   splitter slot
  utils/
    device.py                # accelerator/precision/compile/TF32/workers/pin_memory
    run_io.py                # save resolved config, seed, git SHA

tests/test_objectives.py     # verifies the contracts (not just "it runs")
```

Each `configs/<group>/` mirrors a package under `src/research_pipeline/`: the YAML
picks the class and its args, the class lives in code. Gitignored at runtime:
`outputs/` (per-run artifacts), `mlruns/` (tracking), `data/` (datasets).

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

## Swap & add components (one config + one class — never edit the training loop)

Swap an existing implementation from the CLI:

```bash
rp-train model=mlp                             # swap model    (configs/model/mlp.yaml)
rp-train optimizer=sgd scheduler=cosine        # swap optimizer/scheduler
rp-train data=synthetic                        # swap dataset  (configs/data/synthetic.yaml)
```

Add a new one — a config file pointing (`_target_`) at a class:

| Add a… | Do this | Run |
|---|---|---|
| **model** | class returning a dict `{"logits": …}` in `models/<name>.py` + `configs/model/<name>.yaml` (`_target_`) | `rp-train model=<name>` |
| **loss term** | a `terms["<name>"] = weight * value` line in `CompositeLoss` + its weight in `configs/loss/composite.yaml` | (active once weighted) |
| **dataset** | a `LightningDataModule` yielding `(input, target)` + `configs/data/<name>.yaml` | `rp-train data=<name>` |
| **transform** | a factory in `data/transforms.py` + point the `transform:` slot in `configs/data/<name>.yaml` | (used by that data config) |
| **splitter** | a `dataset -> (train_idx, val_idx)` callable in `data/splitter.py` + point the `splitter:` slot | (used by that data config) |

See **CLAUDE.md** for the full contracts, the deferred extension points (custom
training dynamic, eval/inference strategy, `collate_fn`), and which changes are
safe vs may ripple.

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
