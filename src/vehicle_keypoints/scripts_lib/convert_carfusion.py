"""Convert raw CarFusion (CMU) dumps to COCO keypoints JSON.

Reusable functions (import-friendly for tests); a thin CLI wrapper lives at
`scripts/convert_carfusion_to_coco.py`.

Input layout (raw CarFusion):
    raw_dir/<scene>/gt/<video_id>_<frame_id>.txt  — per-frame keypoint rows
    raw_dir/<scene>/<image_subdir>/<video_id>_<frame_id>.jpg

Each `.txt` row has 5 comma-separated fields:
    x, y, keypoint_id(1..14), instance_id, visibility(1|2|3)

CarFusion visibility convention -> COCO visibility:
    1 (visible)          -> 2 (labeled + visible)
    2 (labeled occluded) -> 1 (labeled but not visible)
    3 (occluded)         -> 2 (labeled + visible)  # legacy script treated 3 as 1
    other                -> 0 (not labeled)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Polygon

from ..utils import get_logger

log = get_logger(__name__)

IMAGE_WIDTH, IMAGE_HEIGHT = 1920, 1080
NUM_KEYPOINTS = 14

_CARFUSION_SKELETON = [
    [0, 2], [1, 3], [0, 1], [2, 3],
    [9, 11], [10, 12], [9, 10], [11, 12],
    [4, 0], [4, 9], [4, 5], [5, 1], [5, 10],
    [6, 2], [6, 11], [7, 3], [7, 12], [6, 7],
]


def _to_int(s: str) -> int:
    s = s.strip()
    try:
        return int(s)
    except ValueError:
        return int(float(s))


def _annotation_from_instance(instance: np.ndarray) -> tuple[list[int], list[list[int]], list[int], int]:
    """Derive COCO bbox + convex-hull segmentation from a single instance's keypoints."""
    visible = instance[:, 2] > 0
    num_keypoints = int(visible.sum())

    bbox: list[int] = [0, 0, 0, 0]
    segmentation: list[list[int]] = []

    if num_keypoints >= 3:
        try:
            hull = Polygon([(x[0], x[1]) for x in instance[visible, :2]]).convex_hull
            frame = Polygon([(0, 0), (IMAGE_WIDTH, 0), (IMAGE_WIDTH, IMAGE_HEIGHT), (0, IMAGE_HEIGHT)])
            hull = hull.intersection(frame).convex_hull
            bounds = hull.bounds
            w, h = bounds[2] - bounds[0], bounds[3] - bounds[1]
            x_o = max(bounds[0] - w / 10, 0)
            y_o = max(bounds[1] - h / 10, 0)
            x_i = min(x_o + (w / 4) + w, IMAGE_WIDTH)
            y_i = min(y_o + (h / 4) + h, IMAGE_HEIGHT)
            bbox = [int(x_o), int(y_o), int(x_i - x_o), int(y_i - y_o)]
            segmentation = [[int(c[0]), int(c[1])] for c in list(hull.exterior.coords)[:-1]]
        except (ValueError, AttributeError):
            bbox = [0, 0, 0, 0]
            segmentation = []

    keypoints_flat: list[int] = instance.reshape(-1).astype(int).tolist()
    return bbox, segmentation, keypoints_flat, num_keypoints


def _parse_txt(path: Path) -> dict[int, np.ndarray]:
    instances: dict[int, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = [_to_int(s) for s in line.split(",") if s.strip()]
        if len(row) < 5:
            continue
        x, y, kpt_id, inst_id, vis = row[:5]
        if not 1 <= kpt_id <= NUM_KEYPOINTS:
            continue
        coco_vis = {1: 2, 2: 1, 3: 2}.get(vis, 0)
        if x <= 0 or y <= 0 or x > IMAGE_WIDTH or y > IMAGE_HEIGHT:
            coco_vis = 0
        arr = instances.setdefault(inst_id, np.zeros((NUM_KEYPOINTS, 3), dtype=np.int32))
        arr[kpt_id - 1] = (x, y, coco_vis)
    return instances


def convert_scene_dir(
    raw_dir: Path | str,
    image_subdir: str,
    out_json: Path | str,
) -> None:
    raw = Path(raw_dir)
    scene_dirs = sorted(p for p in raw.iterdir() if p.is_dir())
    if not scene_dirs:
        raise SystemExit(f"No scene dirs under {raw}")

    data: dict[str, Any] = {
        "info": {
            "url": "https://www.andrew.cmu.edu/user/dnarapur/",
            "year": 2018,
            "date_created": time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()),
            "description": "CarFusion vehicle keypoint dataset (CMU).",
            "version": "1.0",
            "contributor": "CMU",
        },
        "licenses": [{"id": 1, "name": "unknown", "url": "unknown"}],
        "categories": [
            {
                "name": "car",
                "id": 1,
                "skeleton": _CARFUSION_SKELETON,
                "supercategory": "car",
                "keypoints": [str(k) for k in range(NUM_KEYPOINTS)],
            }
        ],
        "images": [],
        "annotations": [],
    }

    ann_id = 0
    for scene_idx, scene in enumerate(scene_dirs, start=1):
        gt_dir = scene / "gt"
        if not gt_dir.is_dir():
            log.warning("scene_missing_gt", scene=scene.name)
            continue
        for txt in sorted(gt_dir.glob("*.txt")):
            stem = txt.stem
            try:
                vid_str, frame_str = stem.split("_")
                video_id = int(vid_str)
                frame_id = int(frame_str)
            except ValueError:
                video_id = hash(stem) & 0xFFFF
                frame_id = 0
            image_id = int(scene_idx * 1e8 + video_id * 1e5 + frame_id)

            data["images"].append(
                {
                    "flickr_url": "unknown",
                    "coco_url": "unknown",
                    "file_name": f"{scene.name}/{image_subdir}/{stem}.jpg",
                    "id": image_id,
                    "license": 1,
                    "date_captured": "unknown",
                    "width": IMAGE_WIDTH,
                    "height": IMAGE_HEIGHT,
                }
            )

            instances = _parse_txt(txt)
            for instance in instances.values():
                bbox, seg, kpts_flat, num_kpts = _annotation_from_instance(instance)
                if num_kpts == 0:
                    continue
                data["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": bbox,
                        "iscrowd": 0,
                        "area": bbox[2] * bbox[3],
                        "keypoints": kpts_flat,
                        "num_keypoints": num_kpts,
                        "segmentation": [sum(seg, [])] if seg else [],
                    }
                )
                ann_id += 1

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(data), encoding="utf-8")
    log.info(
        "convert_done", out=str(out_json),
        images=len(data["images"]), annotations=len(data["annotations"]),
    )
