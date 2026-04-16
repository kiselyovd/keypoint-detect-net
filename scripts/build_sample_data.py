"""Populate data/sample/ with a tiny paired subset for CI.

Copies 6 images from data/processed/images/train + their YOLO labels + writes a
truncated COCO annotations.json subset so ViTPose tests can load real data.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path, PurePosixPath


def build_sample(src: Path, dst: Path, n: int = 6, seed: int = 42) -> None:
    rng = random.Random(seed)
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)

    candidates = sorted((src / "images" / "train").glob("*.jpg"))
    if len(candidates) < n:
        raise SystemExit(
            f"Not enough images in {src / 'images/train'}: need {n}, have {len(candidates)}"
        )
    chosen = rng.sample(candidates, n)

    kept_stems: set[str] = set()
    for img in chosen:
        shutil.copy2(img, dst / "images" / img.name)
        lbl = src / "labels" / "train" / f"{img.stem}.txt"
        if not lbl.exists():
            raise SystemExit(f"Missing label for {img}")
        shutil.copy2(lbl, dst / "labels" / lbl.name)
        kept_stems.add(img.stem)

    raw_annotations = Path("data/raw/annotations/car_keypoints_train.json")
    if raw_annotations.exists():
        coco = json.loads(raw_annotations.read_text(encoding="utf-8"))
        keep_ids: set[int] = set()
        filtered_images: list[dict] = []
        for img in coco["images"]:
            # Handle both forward and back slashes in file_name.
            fn = img["file_name"].replace("\\", "/")
            orig_stem = PurePosixPath(fn).stem
            scene = PurePosixPath(fn).parts[0]
            if f"{scene}__{orig_stem}" in kept_stems:
                filtered_images.append(img)
                keep_ids.add(img["id"])
        filtered_anns = [a for a in coco["annotations"] if a["image_id"] in keep_ids]
        subset = {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "categories": coco["categories"],
            "images": filtered_images,
            "annotations": filtered_anns,
        }
        (dst / "annotations.json").write_text(json.dumps(subset, indent=2), encoding="utf-8")
    print(f"wrote {n} images + {n} labels + annotations.json to {dst}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/processed")
    p.add_argument("--dst", default="data/sample")
    p.add_argument("-n", type=int, default=6)
    args = p.parse_args()
    build_sample(Path(args.src), Path(args.dst), n=args.n)
