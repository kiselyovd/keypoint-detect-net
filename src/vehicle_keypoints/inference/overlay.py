"""Render keypoints + skeleton overlays on input images (CPU, OpenCV)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2

CARFUSION_SKELETON: tuple[tuple[int, int], ...] = (
    (0, 2),
    (1, 3),
    (0, 1),
    (2, 3),
    (9, 11),
    (10, 12),
    (9, 10),
    (11, 12),
    (4, 0),
    (4, 9),
    (4, 5),
    (5, 1),
    (5, 10),
    (6, 2),
    (6, 11),
    (7, 3),
    (7, 12),
    (6, 7),
)

_KPT_COLOR = (0, 255, 0)  # green (BGR)
_EDGE_COLOR = (0, 200, 255)  # amber
_BBOX_COLOR = (255, 0, 0)  # blue


def draw_keypoints(
    image_path: str | Path,
    detections: Sequence[dict],
    out_path: str | Path,
    *,
    kpt_radius: int = 4,
    edge_thickness: int = 2,
) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read {image_path}")
    for det in detections:
        x, y, w, h = [int(v) for v in det["bbox"]]
        cv2.rectangle(img, (x, y), (x + w, y + h), _BBOX_COLOR, 2)
        kpts = det["keypoints"]
        for a, b in CARFUSION_SKELETON:
            if a >= len(kpts) or b >= len(kpts):
                continue
            xa, ya, va = kpts[a]
            xb, yb, vb = kpts[b]
            if va > 0 and vb > 0:
                cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), _EDGE_COLOR, edge_thickness)
        for kx, ky, v in kpts:
            if v > 0:
                cv2.circle(img, (int(kx), int(ky)), kpt_radius, _KPT_COLOR, -1)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def encode_overlay_bytes(image_path: str | Path, detections: Sequence[dict]) -> bytes:
    tmp = Path(image_path).with_suffix(".overlay.png")
    try:
        draw_keypoints(image_path, detections, tmp)
        return tmp.read_bytes()
    finally:
        if tmp.exists():
            tmp.unlink()
