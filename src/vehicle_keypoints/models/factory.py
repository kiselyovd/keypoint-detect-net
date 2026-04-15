"""Model factory — returns a torch.nn.Module by name."""
from __future__ import annotations

from torch import nn

def build_model(name: str, num_keypoints: int, pretrained: bool = True) -> nn.Module:
    if name.startswith("yolo26"):
        from ultralytics import YOLO

        return YOLO(f"{name}-pose.pt" if pretrained else f"{name}-pose.yaml")
    raise ValueError(f"Unknown model: {name}")
