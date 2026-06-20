# research-pipeline

Domain-agnostic PyTorch Lightning + Hydra pipeline for fast, modular idea-testing.
Runs on Apple Silicon (MPS) and NVIDIA (CUDA), tuned per device, and on CPU.
No internet is required at runtime (one-time dataset downloads are fine).

## Core principle

Adding a component = **one config file + one class**, wired by Hydra `_target_`.
You never edit `src/research_pipeline/train.py` or `lit_module.py` to add a model,
loss term, dataset, optimizer, scheduler, or callback.

The design is deliberately YAGNI **except** two things kept rich on purpose so
future methods are additive, not breaking:
- the model returns a **dict** (not a bare tensor), and
- the loss is **composite** (named weighted terms + a total).

## Tree

```
configs/                      # one file per swappable component (Hydra groups)
  config.yaml                 # root: attaches schema, composes all groups
  data/{cifar10,synthetic}.yaml
  model/{cnn,mlp}.yaml
  loss/composite.yaml
  optimizer/{adamw,sgd}.yaml
  scheduler/cosine.yaml
  trainer/default.yaml
  callbacks/default.yaml
  experiment/cifar10_baseline.yaml   # IMMUTABLE named bundle (clone to change)
src/research_pipeline/
  train.py                    # @hydra.main entrypoint -- the ONLY training loop
  config_schema.py            # typed dataclasses -> ConfigStore (startup validation)
  lit_module.py               # LitClassifier(LightningModule)
  cli.py                      # rp-mlflow-ui, rp-export-figures
  models/{cnn,mlp}.py         # return dict {"logits": ...}
  losses/composite.py         # CompositeLoss -> {"total", "ce", ...}
  data/
    datamodule.py             # CIFAR10DataModule (slots), SyntheticDataModule
    transforms.py             # transform slot
    splitter.py               # splitter slot
  utils/
    device.py                 # accelerator/precision/compile/TF32/workers/pin_memory
    run_io.py                 # save resolved config, seed, git SHA
tests/test_objectives.py      # verifies the contracts below
```

## Contracts (inputs -> outputs)

### Model  (`models/*.py`)
`forward(x) -> Dict[str, Tensor]`. ALWAYS a dict, at least `{"logits": ...}`,
never a bare tensor. Adding an output (e.g. `embedding`) is additive.

### Loss  (`losses/composite.py`)
`CompositeLoss(outputs: dict, targets) -> {"total": ..., "<term>": weighted, ...}`.
`total = sum(weighted terms)`. Each term reads only the output keys it needs.
Today: `{"ce", "total"}`.

### DataModule  (`data/datamodule.py`)
Batch is `(input, target)` (default collate). `CIFAR10DataModule` exposes
swappable slots wired via `_target_`:
- `transform`: preprocessing/tokenizer (a callable / torchvision transform)
- `splitter` : `dataset -> (train_idx, val_idx)`
`num_workers` / `pin_memory` are filled at runtime from `utils/device.py`.

### LightningModule  (`lit_module.py`)
`model(input) -> dict`, `loss(out, target) -> dict`; every term + total + accuracy
are logged per stage. Optimizer/scheduler built from config in
`configure_optimizers`.

### Device  (`utils/device.py`)
`accelerator=auto`. CUDA: `bf16-mixed` + TF32 + optional `torch.compile`,
`pin_memory=True`. MPS/CPU: `32-true`, compile off, `pin_memory=False`.
DataLoader workers defaulted per OS.

### Per-run artifacts  (`utils/run_io.py` + `ModelCheckpoint`)
Each run writes to a gitignored `outputs/<timestamp>/`:
- `checkpoints/best.ckpt` (monitor + mode from `configs/callbacks` / `checkpoint`)
- `config_resolved.yaml`, `seed.txt`, `git_sha.txt`

## Structured configs

`config_schema.py` registers typed dataclasses (`Config`, `TrainerConfig`,
`MLflowConfig`, `CheckpointConfig`) in Hydra's ConfigStore. Bad or misspelled
overrides fail at **startup** with a clear error (e.g. `trainer.max_epochs=abc`
or `trainer.max_epochsX=3`), not deep in training.

## How to add a component (what EXISTS today)

Each is **one config file + one class**, wired by `_target_`. The training loop
(`train.py`) is never edited for any of these.

- **Model**: add `models/<name>.py` returning a dict + `configs/model/<name>.yaml`
  (`_target_`). Run `rp-train model=<name>`.
- **Loss term**: add a `terms["<name>"] = weight * value` line in `CompositeLoss`
  (reading only the output keys it needs) + expose its weight in
  `configs/loss/composite.yaml`. (`losses/composite.py` is the one class kept rich
  on purpose, so this stays additive.)
- **Dataset**: add a `LightningDataModule` yielding `(input, target)` batches +
  `configs/data/<name>.yaml`. Run `rp-train data=<name>`.
- **Transform**: add a factory in `data/transforms.py` + point the `transform:`
  slot in a `configs/data/<name>.yaml`.
- **Splitter**: add a `dataset -> (train_idx, val_idx)` callable in
  `data/splitter.py` + point the `splitter:` slot in a `configs/data/<name>.yaml`.
- **Optimizer / Scheduler**: add `configs/{optimizer,scheduler}/<name>.yaml`
  (`_target_` to the torch class + its args).
- **Callback**: add an entry to `configs/callbacks/default.yaml` (`_target_` to a
  Lightning callback), or a new callbacks file.

### Device / precision (not a CLI knob)
`accelerator=auto`; `utils/device.py` resolves the precision/compile/worker plan
per device and `train.py` passes it to the `Trainer` as explicit kwargs — so they
**override** any `trainer.accelerator=` / `trainer.precision=` you pass on the CLI.
To force the non-CUDA path without code, hide CUDA: `CUDA_VISIBLE_DEVICES= rp-train`.
To change the policy itself, edit `resolve_device_plan` (it returns a `DevicePlan`;
keeping that return shape is safe — see below).

## Deferred — add when needed (one config + one class)

These were intentionally **not** built (YAGNI). The architecture already accepts
each as an additive change; none of them exist yet.

- **Custom training dynamic** (extra hooks / `automatic_optimization=False` for
  manual/recursive loops): subclass `LitClassifier`, override `training_step`
  and/or `configure_optimizers`, and set `self.automatic_optimization = False`.
  To select it from config, make the LightningModule `_target_`-instantiated in
  `train.py` instead of hard-constructed.
- **Custom eval / inference strategy** (test-time compute, validation pooling,
  TTA/MC passes): add a small strategy component and inject it into the
  LightningModule, delegating `validation_step` / `test_step` to it.
- **Custom `collate_fn`** (non-default batch construction): add a `collate_fn`
  slot to the DataModule `__init__` and pass it to the `DataLoader`. Today the
  default collate is used and the batch is `(input, target)`.

## Safe vs ripple

**Stays inside a contract (safe — purely additive, nothing downstream changes):**
- A new **model**, **dataset**, **transform**, **splitter**, **optimizer**,
  **scheduler**, **callback**, or **experiment**.
- A new **loss term** in `CompositeLoss`, or a new **model output key** (e.g.
  `embedding`) consumed only by a new term. `total` stays the sum; existing
  consumers of `logits` are untouched.
- Editing the **device plan** policy while keeping the `DevicePlan` return shape.

**Changes a contract shape (may ripple — touches `train.py` / `lit_module.py` /
existing consumers):**
- **Renaming/removing** an existing output key (e.g. `logits`) — breaks the loss,
  accuracy logging, and any term that reads it.
- Changing the **batch structure** away from `(input, target)` — ripples into
  `LitClassifier._shared_step`.
- Making the **LightningModule swappable** or flipping `automatic_optimization`,
  adding an **eval-strategy** injection, or adding a **`collate_fn` slot** — each
  widens a constructor/loop surface (`train.py` and/or the DataModule/LitModule
  `__init__`), so do it deliberately, not as a drop-in config swap.

## Experiments (IMMUTABLE RULE)

`configs/experiment/*.yaml` are immutable, named bundles of overrides. **To
change an experiment, CLONE it to a new file -- never edit one in place.** This
preserves cross-method comparability of past runs.
Run: `rp-train +experiment=cifar10_baseline`.

## Tracking & commands

- MLflow, local file backend (`./mlruns`). Rely on it for live metric charts;
  training does **not** auto-generate PNGs.
- `rp-train [overrides...]`            -- train (e.g. `rp-train +experiment=cifar10_baseline`)
- `rp-mlflow-ui`                       -- launch the MLflow UI on `./mlruns`
- `rp-export-figures <run_id> [--experiment NAME]` -- portable PNGs for one run
- `pytest`                            -- run the objective test suite

## Conventions

- `.gitignore`: `outputs/ mlruns/ data/ envs/`.
- No lockfiles, containers, or determinism flags (YAGNI).
- CIFAR-10 is fetched once from the fast.ai S3 mirror (the canonical host is
  unreachable here); set `RP_DATA_DIR` to relocate the data dir.
