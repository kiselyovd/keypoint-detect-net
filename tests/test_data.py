"""Data prep smoke tests: COCO -> YOLO conversion and split."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from vehicle_keypoints.data.prepare import prepare_yolo_dataset


def _minimal_coco(n_images: int, scene: str) -> dict:
    return {
        "categories": [
            {"id": 1, "name": "car", "keypoints": [str(i) for i in range(14)], "skeleton": []}
        ],
        "images": [
            {
                "id": i,
                "file_name": f"{scene}/images/{i}.jpg",
                "width": 1920,
                "height": 1080,
            }
            for i in range(n_images)
        ],
        "annotations": [
            {
                "id": i,
                "image_id": i,
                "category_id": 1,
                "bbox": [100, 200, 300, 400],
                "num_keypoints": 14,
                "iscrowd": 0,
                "area": 300 * 400,
                "keypoints": list(
                    itertools.chain.from_iterable(
                        [100 + k * 10, 200 + k * 10, 2] for k in range(14)
                    )
                ),
            }
            for i in range(n_images)
        ],
    }


def test_prepare_yolo_dataset(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "annotations").mkdir(parents=True)
    for split_name, n in (("train", 20), ("test", 6)):
        scene = "scene_a" if split_name == "train" else "scene_b"
        coco = _minimal_coco(n, scene)
        scene_dir = raw / split_name / scene / "images"
        scene_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (scene_dir / f"{i}.jpg").write_bytes(b"fakejpg")
        (raw / "annotations" / f"car_keypoints_{split_name}.json").write_text(json.dumps(coco))

    out = tmp_path / "processed"
    prepare_yolo_dataset(raw_dir=raw, out_dir=out, val_frac=0.25, seed=0)

    # data.yaml correct
    import yaml

    dy = yaml.safe_load((out / "data.yaml").read_text())
    assert dy["kpt_shape"] == [14, 3]
    assert dy["names"] == {0: "car"}
    for split in ("train", "val", "test"):
        assert (out / "images" / split).is_dir()
        assert (out / "labels" / split).is_dir()

    # label format: <cls> <cx> <cy> <w> <h> <kpt1_x> <kpt1_y> <v1> ... (14 kpts x 3)
    all_labels = list((out / "labels" / "train").glob("*.txt"))
    assert all_labels
    line = all_labels[0].read_text().strip().splitlines()[0]
    parts = line.split()
    assert parts[0] == "0"
    assert len(parts) == 1 + 4 + 14 * 3
    for p in parts[1:]:
        float(p)

    # train + val == 20 (single scene → image-level fallback split)
    n_train_imgs = len(list((out / "images" / "train").glob("*.jpg")))
    n_val_imgs = len(list((out / "images" / "val").glob("*.jpg")))
    assert n_train_imgs + n_val_imgs == 20


def test_coco_to_yolo_row_clips_bounds():
    """Bboxes that extend past image bounds must be clipped to [0, 1]."""
    from vehicle_keypoints.data.prepare import _coco_to_yolo_row

    # Bbox extends past 1920x1080: x=-50, y=-30, w=2000, h=1200
    ann = {
        "bbox": [-50, -30, 2000, 1200],
        "keypoints": [2000, 1200, 2] * 14,  # all past bounds but vis=2
    }
    row = _coco_to_yolo_row(ann, img_w=1920, img_h=1080)
    parts = row.split()
    # Class id
    assert parts[0] == "0"
    # cx, cy, bw, bh all must be in [0, 1]
    for p in parts[1:5]:
        assert 0.0 <= float(p) <= 1.0, f"{p} out of [0,1]"
    # keypoint coords clipped too
    for k in range(14):
        kx, ky = parts[5 + k * 3], parts[5 + k * 3 + 1]
        assert 0.0 <= float(kx) <= 1.0
        assert 0.0 <= float(ky) <= 1.0


def test_coco_to_yolo_row_zeros_invisible_kpts():
    """vis=0 keypoints must emit (0, 0, 0) regardless of stored coords."""
    from vehicle_keypoints.data.prepare import _coco_to_yolo_row

    # All keypoints invisible but with bogus coords stored
    ann = {
        "bbox": [100, 200, 300, 400],
        "keypoints": [999, 999, 0] * 14,
    }
    row = _coco_to_yolo_row(ann, img_w=1920, img_h=1080)
    parts = row.split()
    for k in range(14):
        kx = parts[5 + k * 3]
        ky = parts[5 + k * 3 + 1]
        v = parts[5 + k * 3 + 2]
        assert kx == "0.000000"
        assert ky == "0.000000"
        assert v == "0"
