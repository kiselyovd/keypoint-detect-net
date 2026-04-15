"""Training smoke — one-epoch ViTPose fit on data/sample/."""
from __future__ import annotations

from pathlib import Path

import lightning as L
import pytest
import torch

from vehicle_keypoints.data import CocoKeypointsDataset
from vehicle_keypoints.models import KeypointsModule, build_model


def test_vitpose_one_epoch_on_sample(tmp_path: Path) -> None:
    sample = Path("data/sample")
    if not (sample / "annotations.json").exists():
        pytest.skip("data/sample/annotations.json not present")
    ds = CocoKeypointsDataset(sample / "images", sample / "annotations.json")
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True)
    model = build_model("vitpose_s", num_keypoints=14, pretrained=False)
    lit = KeypointsModule(model, num_keypoints=14, lr=1e-4, model_name="vitpose_s")
    trainer = L.Trainer(
        max_epochs=1,
        max_steps=2,
        logger=False,
        enable_checkpointing=False,
        accelerator="cpu",
        enable_progress_bar=False,
    )
    trainer.fit(lit, loader, loader)
    assert "train/loss_epoch" in trainer.callback_metrics
