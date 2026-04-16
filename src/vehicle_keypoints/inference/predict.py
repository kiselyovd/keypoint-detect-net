"""Inference helper: load YOLO pose checkpoint and run detection."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..utils import configure_logging, get_logger

log = get_logger(__name__)


@dataclass
class Detector:
    """Thin wrapper around an ultralytics.YOLO predictor."""

    model: Any

    @classmethod
    def from_checkpoint(cls, ckpt_path: str | Path) -> Detector:
        from ultralytics import YOLO

        return cls(model=YOLO(str(ckpt_path)))

    @classmethod
    def from_pretrained_or_random(cls, base_name: str = "yolo26n") -> Detector:
        """Factory used in tests — loads pretrained pose `.pt` if available, else YAML."""
        from ultralytics import YOLO

        for candidate in (f"{base_name}-pose.pt", "yolo11n-pose.pt", f"{base_name}-pose.yaml"):
            try:
                return cls(model=YOLO(candidate))
            except Exception:
                continue
        raise RuntimeError(f"Could not instantiate YOLO for {base_name}")

    def predict(self, image_path: str | Path, conf: float = 0.25) -> list[dict]:
        results = self.model.predict(source=str(image_path), conf=conf, verbose=False)
        detections: list[dict] = []
        if not results:
            return detections
        r = results[0]
        boxes = getattr(r, "boxes", None)
        keypoints = getattr(r, "keypoints", None)
        if boxes is None or keypoints is None or boxes.data.shape[0] == 0:
            return detections
        bbox_xywh = (
            boxes.xywh.cpu().numpy() if hasattr(boxes, "xywh") else boxes.data.cpu().numpy()[:, :4]
        )
        scores = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else np.ones(len(bbox_xywh))
        kpts_arr = keypoints.data.cpu().numpy() if hasattr(keypoints, "data") else None
        for i, bb in enumerate(bbox_xywh):
            kpts = kpts_arr[i] if kpts_arr is not None else np.zeros((14, 3))
            if kpts.shape[-1] == 2:
                kpts = np.concatenate([kpts, np.ones((kpts.shape[0], 1)) * 2], axis=-1)
            # Normalize to CarFusion 14-keypoint schema: truncate or pad with zeros.
            if kpts.shape[0] > 14:
                kpts = kpts[:14]
            elif kpts.shape[0] < 14:
                pad = np.zeros((14 - kpts.shape[0], kpts.shape[1]))
                kpts = np.concatenate([kpts, pad], axis=0)
            detections.append(
                {
                    "bbox": [
                        float(bb[0] - bb[2] / 2),
                        float(bb[1] - bb[3] / 2),
                        float(bb[2]),
                        float(bb[3]),
                    ],
                    "keypoints": [[float(x), float(y), float(v)] for x, y, v in kpts],
                    "score": float(scores[i]),
                }
            )
        return detections


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--conf", type=float, default=0.25)
    args = p.parse_args()
    configure_logging()
    det = Detector.from_checkpoint(args.checkpoint)
    result = det.predict(args.input, conf=args.conf)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
