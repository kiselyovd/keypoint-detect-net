"""Lightning DataModule."""
from __future__ import annotations

from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader, random_split

from .dataset import ImageDataset
from .transforms import build_eval_transforms, build_train_transforms


class ImageDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        val_split: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage: str | None = None) -> None:
        root = Path(self.hparams.data_dir)
        train_tf = build_train_transforms(self.hparams.image_size)
        eval_tf = build_eval_transforms(self.hparams.image_size)
        full_train = ImageDataset(root / "train", transform=train_tf)
        n_val = int(len(full_train) * self.hparams.val_split)
        n_train = len(full_train) - n_val
        import torch

        gen = torch.Generator().manual_seed(self.hparams.seed)
        self.train_ds, self.val_ds = random_split(full_train, [n_train, n_val], generator=gen)
        self.test_ds = ImageDataset(root / "test", transform=eval_tf)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
