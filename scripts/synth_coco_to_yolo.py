"""Convert a COCO-format synthetic dataset to YOLO-pose format (14-pt CarFusion schema).

Usage:
    .venv/Scripts/python.exe scripts/synth_coco_to_yolo.py \\
        --coco  ../ue5-vehicle-synth/captures/phase0/annotations/coco.json \\
        --images ../ue5-vehicle-synth/captures/phase0 \\
        --out   artifacts/phase0_work/synth_yolo \\
        [--num-kpt 14] [--val-frac 0.1] [--seed 42]

Input expectations:
- COCO JSON with `images`, `annotations`, `categories`.
- Each annotation has `bbox` (XYWH pixel) and `keypoints` (flat list x0 y0 v0 ...).
- The synth dataset has 24 keypoints; only the first --num-kpt are kept.
- `file_name` in images is relative to --images root.

Output layout:
    <out>/
        images/train/  images/val/
        labels/train/  labels/val/
        synth_data.yaml

YOLO-pose label format per line:
    class cx cy bw bh  kx0 ky0 v0  kx1 ky1 v1  ...
All coords normalised [0,1].  Visibility: v<=0 -> 0, v>=1 kept as-is (YOLO uses 0/1/2).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Core converter
# ---------------------------------------------------------------------------


def coco_ann_to_yolo_row(ann: dict, img_w: int, img_h: int, num_kpt: int) -> str:
    """Convert one COCO annotation dict to a YOLO-pose label line string."""
    x, y, w, h = ann["bbox"]
    x0 = max(0.0, float(x))
    y0 = max(0.0, float(y))
    x1 = min(float(img_w), float(x) + float(w))
    y1 = min(float(img_h), float(y) + float(h))
    bw_px = max(0.0, x1 - x0)
    bh_px = max(0.0, y1 - y0)
    cx = min(max((x0 + bw_px / 2) / img_w, 0.0), 1.0)
    cy = min(max((y0 + bh_px / 2) / img_h, 0.0), 1.0)
    bw = min(max(bw_px / img_w, 0.0), 1.0)
    bh = min(max(bh_px / img_h, 0.0), 1.0)
    parts: list[str] = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
    kpts = ann["keypoints"]  # flat: x0 y0 v0 ...
    for k in range(num_kpt):
        kx, ky, v = kpts[k * 3], kpts[k * 3 + 1], kpts[k * 3 + 2]
        v_int = int(v)
        if v_int <= 0:
            parts += ["0.000000", "0.000000", "0"]
        else:
            kx_n = min(max(float(kx) / img_w, 0.0), 1.0)
            ky_n = min(max(float(ky) / img_h, 0.0), 1.0)
            parts += [f"{kx_n:.6f}", f"{ky_n:.6f}", str(v_int)]
    return " ".join(parts)


def convert(
    coco_json: Path,
    images_root: Path,
    out_dir: Path,
    num_kpt: int = 14,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Path:
    """Run conversion; return path to generated data.yaml."""
    print(f"Loading {coco_json} ...")
    data = json.loads(coco_json.read_text(encoding="utf-8"))

    images_by_id = {img["id"]: img for img in data["images"]}
    anns_by_img: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        flat = ann["keypoints"]
        trimmed = dict(ann)
        trimmed["keypoints"] = flat[: num_kpt * 3]
        anns_by_img.setdefault(ann["image_id"], []).append(trimmed)

    all_ids = sorted(images_by_id.keys())
    rng = random.Random(seed)
    rng.shuffle(all_ids)
    n_val = max(1, round(len(all_ids) * val_frac))
    val_ids = set(all_ids[:n_val])
    train_ids = set(all_ids[n_val:])
    print(f"Split: {len(train_ids)} train, {len(val_ids)} val  (seed={seed})")

    missing = 0
    for split, ids in [("train", train_ids), ("val", val_ids)]:
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img_id in ids:
            meta = images_by_id[img_id]
            src = images_root / meta["file_name"]
            if not src.is_file():
                print(f"  WARNING: missing {src}")
                missing += 1
                continue
            shutil.copy2(src, img_dir / src.name)
            anns = anns_by_img.get(img_id, [])
            rows = [coco_ann_to_yolo_row(a, meta["width"], meta["height"], num_kpt) for a in anns]
            lbl = lbl_dir / (src.stem + ".txt")
            lbl.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    if missing:
        print(f"WARNING: {missing} source images were missing and skipped.")

    data_yaml: dict = {
        "path": str(out_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "kpt_shape": [num_kpt, 3],
        "names": {0: "car"},
        "flip_idx": list(range(num_kpt)),
    }
    yaml_path = out_dir / "synth_data.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    print(f"Wrote {yaml_path}")
    return yaml_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Convert COCO synth dataset to YOLO-pose format.")
    p.add_argument("--coco", required=True, help="Path to COCO JSON annotation file")
    p.add_argument(
        "--images", required=True, help="Root dir for images (file_name is relative to this)"
    )
    p.add_argument("--out", required=True, help="Output directory for YOLO dataset")
    p.add_argument(
        "--num-kpt", type=int, default=14, help="Number of keypoints to keep (default 14)"
    )
    p.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction (default 0.1)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    args = p.parse_args()

    yaml_path = convert(
        coco_json=Path(args.coco),
        images_root=Path(args.images),
        out_dir=Path(args.out),
        num_kpt=args.num_kpt,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    print(f"Done. data.yaml -> {yaml_path}")


if __name__ == "__main__":
    main()
