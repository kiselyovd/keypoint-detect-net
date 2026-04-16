"""CLI for CarFusion -> COCO keypoints conversion (see scripts_lib for logic)."""

from __future__ import annotations

import argparse
from pathlib import Path

from vehicle_keypoints.scripts_lib.convert_carfusion import convert_scene_dir
from vehicle_keypoints.utils import configure_logging


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--raw-dir",
        required=True,
        help="Directory containing per-scene subdirs with gt/ and images/",
    )
    p.add_argument("--image-subdir", default="images")
    p.add_argument("--out", required=True, help="Output COCO JSON path")
    args = p.parse_args()
    configure_logging()
    convert_scene_dir(Path(args.raw_dir), args.image_subdir, Path(args.out))


if __name__ == "__main__":
    main()
