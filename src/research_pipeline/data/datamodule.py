"""DataModules.

``CIFAR10DataModule`` exposes swappable slots wired via ``_target_``:
  - ``transform``: preprocessing / tokenizer (applied to every split)
  - ``splitter`` : custom train/val splitting -> (train_idx, val_idx)
(The default collate is used, per YAGNI.)

The CIFAR-10 images are fetched once (downloads are allowed; runtime is not
required to have internet) from the fast.ai S3 mirror, since the canonical host
is unreachable from this environment. The data lands as an ImageFolder tree.

``SyntheticDataModule`` is a tiny, no-download, in-memory dataset used by the
smoke test.
"""
from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.datasets import ImageFolder

# fast.ai mirror of CIFAR-10 as PNG image folders (train/test x class).
_CIFAR_URL = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"


def _download_cifar(data_dir: Path) -> None:
    root = data_dir / "cifar10"
    if (root / "train").exists() and (root / "test").exists():
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    tgz = data_dir / "cifar10.tgz"
    if not tgz.exists():
        request = urllib.request.Request(_CIFAR_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=120) as response, open(tgz, "wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
    with tarfile.open(tgz, "r:gz") as archive:
        archive.extractall(data_dir)


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        transform: Optional[Callable] = None,
        splitter: Optional[Callable[[Sequence], Tuple[Sequence[int], Sequence[int]]]] = None,
        batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
        download: bool = True,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.splitter = splitter
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.download = download

    def prepare_data(self) -> None:
        if self.download:
            _download_cifar(self.data_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        root = self.data_dir / "cifar10"
        full_train = ImageFolder(str(root / "train"), transform=self.transform)
        train_idx, val_idx = self.splitter(full_train)
        self.train_ds = Subset(full_train, list(train_idx))
        self.val_ds = Subset(full_train, list(val_idx))
        self.test_ds = ImageFolder(str(root / "test"), transform=self.transform)

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_ds, shuffle=False)


class SyntheticDataModule(pl.LightningDataModule):
    def __init__(
        self,
        num_samples: int = 64,
        image_size: int = 8,
        channels: int = 3,
        num_classes: int = 10,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.image_size = image_size
        self.channels = channels
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def _make(self, n: int, seed: int) -> TensorDataset:
        generator = torch.Generator().manual_seed(seed)
        x = torch.randn(n, self.channels, self.image_size, self.image_size, generator=generator)
        y = torch.randint(0, self.num_classes, (n,), generator=generator)
        return TensorDataset(x, y)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_ds = self._make(self.num_samples, seed=0)
        self.val_ds = self._make(self.num_samples // 2, seed=1)
        self.test_ds = self._make(self.num_samples // 2, seed=2)

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self):
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_ds, shuffle=False)
