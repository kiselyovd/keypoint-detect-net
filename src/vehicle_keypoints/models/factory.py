"""Model factory -- main YOLO26-pose + baseline ViTPose-S."""

from __future__ import annotations

from typing import Any

YOLO_FALLBACKS = ("yolo26n-pose.pt", "yolo11n-pose.pt")


def build_model(name: str, num_keypoints: int, pretrained: bool = True) -> Any:
    """Return either an `ultralytics.YOLO` or a `torch.nn.Module`.

    YOLO path (name startswith "yolo"):
      - loads an Ultralytics `.pt` (pretrained on COCO human pose; we fine-tune the head
        on CarFusion's 14-kpt layout at train time).
    ViTPose path (name startswith "vitpose"):
      - returns a `ViTPoseSmall` nn.Module emitting heatmaps of shape (B, num_keypoints, H', W').
    """
    if name.startswith("yolo"):
        from ultralytics import YOLO

        if pretrained:
            for candidate in (f"{name}-pose.pt", *YOLO_FALLBACKS):
                try:
                    return YOLO(candidate)
                except (FileNotFoundError, Exception):
                    continue
            raise ValueError(f"No pretrained pose checkpoint found for {name} (tried fallbacks)")
        return YOLO(f"{name}-pose.yaml")

    if name.startswith("vitpose"):
        from .vitpose import ViTPoseSmall

        return ViTPoseSmall(num_keypoints=num_keypoints, pretrained=pretrained)

    raise ValueError(f"Unknown model: {name}")
