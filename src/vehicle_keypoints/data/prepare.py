"""Prepare COCO -> Ultralytics YOLO-format dataset layout."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import yaml

from ..utils import get_logger

log = get_logger(__name__)

NUM_KEYPOINTS = 14
CLASS_NAME = "car"
ANN_FILENAMES = {"train": "car_keypoints_train.json", "test": "car_keypoints_test.json"}


def _coco_to_yolo_row(ann: dict[str, Any], img_w: int, img_h: int) -> str:
    x, y, w, h = ann["bbox"]
    # Clip bbox to image bounds, then renormalize.
    x0 = max(0.0, float(x))
    y0 = max(0.0, float(y))
    x1 = min(float(img_w), float(x) + float(w))
    y1 = min(float(img_h), float(y) + float(h))
    bw_px = max(0.0, x1 - x0)
    bh_px = max(0.0, y1 - y0)
    cx = (x0 + bw_px / 2) / img_w
    cy = (y0 + bh_px / 2) / img_h
    bw = bw_px / img_w
    bh = bh_px / img_h
    # Clamp normalized values to [0, 1] for numerical safety (tiny float drift).
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    bw = min(max(bw, 0.0), 1.0)
    bh = min(max(bh, 0.0), 1.0)
    parts: list[str] = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
    kpts = ann["keypoints"]
    for k in range(NUM_KEYPOINTS):
        kx, ky, v = kpts[k * 3 : k * 3 + 3]
        v_int = int(v)
        if v_int <= 0:
            # Non-labeled kpt: emit (0, 0, 0) — YOLO convention.
            parts.append("0.000000")
            parts.append("0.000000")
            parts.append("0")
        else:
            kx_n = min(max(float(kx) / img_w, 0.0), 1.0)
            ky_n = min(max(float(ky) / img_h, 0.0), 1.0)
            parts.append(f"{kx_n:.6f}")
            parts.append(f"{ky_n:.6f}")
            parts.append(str(v_int))
    return " ".join(parts)


def _scene_of(file_name: str) -> str:
    return Path(file_name).parts[0]


def _load_coco(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_img_by_id"] = {img["id"]: img for img in data["images"]}
    return data


def _emit_split(
    split: str,
    image_ids: set[int],
    coco: dict[str, Any],
    raw_split_root: Path,
    out_dir: Path,
) -> int:
    img_dir = out_dir / "images" / split
    lbl_dir = out_dir / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    ann_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        if ann["image_id"] in image_ids:
            ann_by_image.setdefault(ann["image_id"], []).append(ann)

    written = 0
    for image_id in image_ids:
        img = coco["_img_by_id"][image_id]
        src = raw_split_root / img["file_name"]
        if not src.is_file():
            log.warning("image_missing", src=str(src))
            continue
        # Flatten filename — prefix with scene to avoid collisions across scenes.
        flat = f"{_scene_of(img['file_name'])}__{Path(img['file_name']).name}"
        dst_img = img_dir / flat
        shutil.copy2(src, dst_img)

        rows = [
            _coco_to_yolo_row(a, img["width"], img["height"])
            for a in ann_by_image.get(image_id, [])
        ]
        (lbl_dir / (Path(flat).stem + ".txt")).write_text("\n".join(rows) + ("\n" if rows else ""))
        written += 1
    return written


def prepare_yolo_dataset(
    raw_dir: Path | str,
    out_dir: Path | str,
    *,
    val_frac: float = 0.1,
    seed: int = 42,
) -> None:
    raw = Path(raw_dir)
    out = Path(out_dir)

    train_coco = _load_coco(raw / "annotations" / ANN_FILENAMES["train"])
    test_coco = _load_coco(raw / "annotations" / ANN_FILENAMES["test"])

    scenes = sorted({_scene_of(img["file_name"]) for img in train_coco["images"]})
    rng = random.Random(seed)
    rng.shuffle(scenes)
    n_val = max(1, round(len(scenes) * val_frac))
    val_scenes = set(scenes[:n_val]) if len(scenes) > 1 else set()
    train_scenes = set(scenes) - val_scenes
    log.info("split.scenes", train=sorted(train_scenes), val=sorted(val_scenes))

    train_ids = {
        img["id"] for img in train_coco["images"] if _scene_of(img["file_name"]) in train_scenes
    }
    val_ids = {
        img["id"] for img in train_coco["images"] if _scene_of(img["file_name"]) in val_scenes
    }
    test_ids = {img["id"] for img in test_coco["images"]}

    # If only one scene exists (test env with minimal fixtures), carve val at image level.
    if not val_ids and len(train_ids) > 1:
        train_ids_sorted = sorted(train_ids)
        n_val_imgs = max(1, round(len(train_ids_sorted) * val_frac))
        val_ids = set(train_ids_sorted[:n_val_imgs])
        train_ids = set(train_ids_sorted[n_val_imgs:])

    n_tr = _emit_split("train", train_ids, train_coco, raw / "train", out)
    n_vl = _emit_split("val", val_ids, train_coco, raw / "train", out)
    n_te = _emit_split("test", test_ids, test_coco, raw / "test", out)

    data_yaml = {
        "path": str(out.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "kpt_shape": [NUM_KEYPOINTS, 3],
        "names": {0: CLASS_NAME},
        # No horizontal-flip symmetry for cars in CarFusion — flip_idx is identity.
        "flip_idx": list(range(NUM_KEYPOINTS)),
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    log.info("prepare.done", train=n_tr, val=n_vl, test=n_te, data_yaml=str(out / "data.yaml"))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    prepare_yolo_dataset(args.raw, args.out, val_frac=args.val_frac, seed=args.seed)
