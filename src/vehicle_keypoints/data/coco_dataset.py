"""Top-down COCO-keypoints dataset for ViTPose training.

Yields `(crop_tensor, target_heatmap, visibility)` for each GT car instance.
Crop is extracted from `bbox` (with margin) and resized to a fixed (H, W).
Heatmap targets are Gaussian blobs at GT keypoint locations in the crop.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


NUM_KEYPOINTS = 14
DEFAULT_CROP = (256, 192)  # (H, W) — ViTPose canonical
DEFAULT_HEATMAP = (64, 48)  # (H/4, W/4)


def _gaussian_heatmap(
    kpts_xy: np.ndarray,
    vis: np.ndarray,
    heatmap_hw: tuple[int, int],
    sigma: float = 2.0,
) -> np.ndarray:
    h, w = heatmap_hw
    hm = np.zeros((NUM_KEYPOINTS, h, w), dtype=np.float32)
    for k in range(NUM_KEYPOINTS):
        if vis[k] <= 0:
            continue
        cx, cy = kpts_xy[k]
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        ys = np.arange(h)[:, None]
        xs = np.arange(w)[None, :]
        hm[k] = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2))
    return hm


class CocoKeypointsDataset(Dataset):
    """Iterate per-instance (one crop per annotation)."""

    def __init__(
        self,
        images_root: Path | str,
        annotations_json: Path | str,
        crop_hw: tuple[int, int] = DEFAULT_CROP,
        heatmap_hw: tuple[int, int] = DEFAULT_HEATMAP,
        margin: float = 0.1,
    ) -> None:
        self.images_root = Path(images_root)
        self.crop_hw = crop_hw
        self.heatmap_hw = heatmap_hw
        self.margin = margin
        raw: dict[str, Any] = json.loads(Path(annotations_json).read_text(encoding="utf-8"))
        self.images_by_id = {img["id"]: img for img in raw["images"]}
        self.annotations = [a for a in raw["annotations"] if a["num_keypoints"] > 0]

    def __len__(self) -> int:
        return len(self.annotations)

    def _file_for(self, img_info: dict[str, Any]) -> Path:
        # Template may have scene-prefixed layout or flat layout; try both.
        fn = img_info["file_name"].replace("\\", "/")
        scene_path = self.images_root / fn
        if scene_path.is_file():
            return scene_path
        # Flat fallback — our data/sample/ uses "<scene>__<stem>.jpg" convention.
        from pathlib import PurePosixPath

        stem = PurePosixPath(fn).stem
        scene = PurePosixPath(fn).parts[0] if "/" in fn else ""
        flat_name = f"{scene}__{stem}.jpg" if scene else f"{stem}.jpg"
        flat_path = self.images_root / flat_name
        if flat_path.is_file():
            return flat_path
        # Last resort: basename only
        return self.images_root / Path(fn).name

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ann = self.annotations[idx]
        img_info = self.images_by_id[ann["image_id"]]
        img_path = self._file_for(img_info)
        img = Image.open(img_path).convert("RGB")
        img_arr = np.asarray(img)

        x, y, w, h = ann["bbox"]
        mx, my = w * self.margin, h * self.margin
        x0 = max(int(x - mx), 0)
        y0 = max(int(y - my), 0)
        x1 = min(int(x + w + mx), img_arr.shape[1])
        y1 = min(int(y + h + my), img_arr.shape[0])
        if x1 <= x0 or y1 <= y0:
            crop = np.zeros((self.crop_hw[0], self.crop_hw[1], 3), dtype=np.uint8)
            vis = np.zeros((NUM_KEYPOINTS,), dtype=np.float32)
            target = np.zeros((NUM_KEYPOINTS, *self.heatmap_hw), dtype=np.float32)
            return (
                torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0,
                torch.from_numpy(target),
                torch.from_numpy(vis),
            )

        crop = img_arr[y0:y1, x0:x1]
        crop_img = Image.fromarray(crop).resize(
            (self.crop_hw[1], self.crop_hw[0]), Image.BILINEAR
        )
        scale_x = self.heatmap_hw[1] / (x1 - x0)
        scale_y = self.heatmap_hw[0] / (y1 - y0)

        kpts_raw = np.asarray(ann["keypoints"], dtype=np.float32).reshape(NUM_KEYPOINTS, 3)
        vis = (kpts_raw[:, 2] > 0).astype(np.float32)
        kxy_in_crop = np.stack(
            [(kpts_raw[:, 0] - x0) * scale_x, (kpts_raw[:, 1] - y0) * scale_y], axis=1
        )
        target_hm = _gaussian_heatmap(kxy_in_crop, vis, self.heatmap_hw)

        crop_t = torch.from_numpy(np.asarray(crop_img)).permute(2, 0, 1).float() / 255.0
        return crop_t, torch.from_numpy(target_hm), torch.from_numpy(vis)
