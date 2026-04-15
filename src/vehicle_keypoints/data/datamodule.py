"""Lightning DataModule for ViTPose baseline (COCO-style top-down pose)."""
from __future__ import annotations

from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from .coco_dataset import CocoKeypointsDataset


class KeypointsDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_images: str | Path,
        train_annotations: str | Path,
        val_images: str | Path,
        val_annotations: str | Path,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 256,  # unused — kept for Hydra cfg compatibility
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.train_ds: CocoKeypointsDataset | None = None
        self.val_ds: CocoKeypointsDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        self.train_ds = CocoKeypointsDataset(self.hparams.train_images, self.hparams.train_annotations)
        self.val_ds = CocoKeypointsDataset(self.hparams.val_images, self.hparams.val_annotations)

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
