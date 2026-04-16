"""ViTPose-Small wrapper returning heatmap predictions for N keypoints."""

from __future__ import annotations

import torch
from torch import nn


class ViTPoseSmall(nn.Module):
    """Thin wrapper around HF VitPose model, re-headed to N keypoints.

    We use `usyd-community/vitpose-small-simple` as the pretrained backbone (ImageNet + MS-COCO
    human pose). The head is replaced to emit `num_keypoints` heatmaps.
    """

    def __init__(self, num_keypoints: int = 14, pretrained: bool = True) -> None:
        super().__init__()
        from transformers import VitPoseConfig, VitPoseForPoseEstimation

        model_id = "usyd-community/vitpose-small-simple"
        if pretrained:
            try:
                self.backbone = VitPoseForPoseEstimation.from_pretrained(
                    model_id, num_labels=num_keypoints, ignore_mismatched_sizes=True
                )
            except Exception:
                cfg = VitPoseConfig(num_labels=num_keypoints)
                self.backbone = VitPoseForPoseEstimation(cfg)
        else:
            cfg = VitPoseConfig(num_labels=num_keypoints)
            self.backbone = VitPoseForPoseEstimation(cfg)
        self.num_keypoints = num_keypoints

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=x)
        return out.heatmaps if hasattr(out, "heatmaps") else out.logits
