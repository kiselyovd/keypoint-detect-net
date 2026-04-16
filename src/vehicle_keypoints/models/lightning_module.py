"""Lightning wrapper for ViTPose (heatmap-regression head)."""

from __future__ import annotations

import lightning as L
import torch
from torch import nn, optim


class KeypointsModule(L.LightningModule):
    """Heatmap-regression Lightning module for top-down pose estimation.

    Forward expects a (B, 3, H, W) crop. Output is (B, K, H', W') heatmaps.
    Training target is a set of Gaussian-blobbed heatmaps centered on GT kpt locations.
    Loss is MSE, masked by keypoint visibility.
    """

    def __init__(
        self,
        model: nn.Module,
        num_keypoints: int,
        lr: float = 5e-4,
        model_name: str | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_keypoints = num_keypoints
        self.save_hyperparameters(ignore=["model"])

    def _forward_heatmaps(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out.heatmaps if hasattr(out, "heatmaps") else out

    def _masked_mse(
        self, pred: torch.Tensor, target: torch.Tensor, vis: torch.Tensor
    ) -> torch.Tensor:
        mask = (vis > 0).to(pred.dtype).unsqueeze(-1).unsqueeze(-1)
        diff = (pred - target) * mask
        denom = mask.sum().clamp_min(1.0)
        return (diff**2).sum() / denom

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, target_hm, vis = batch
        pred_hm = self._forward_heatmaps(x)
        if pred_hm.shape[-2:] != target_hm.shape[-2:]:
            pred_hm = nn.functional.interpolate(
                pred_hm, size=target_hm.shape[-2:], mode="bilinear", align_corners=False
            )
        loss = self._masked_mse(pred_hm, target_hm, vis)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        x, target_hm, vis = batch
        pred_hm = self._forward_heatmaps(x)
        if pred_hm.shape[-2:] != target_hm.shape[-2:]:
            pred_hm = nn.functional.interpolate(
                pred_hm, size=target_hm.shape[-2:], mode="bilinear", align_corners=False
            )
        loss = self._masked_mse(pred_hm, target_hm, vis)
        self.log("val/loss", loss, prog_bar=True)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.AdamW(self.parameters(), lr=self.hparams.lr)  # type: ignore[attr-defined]
