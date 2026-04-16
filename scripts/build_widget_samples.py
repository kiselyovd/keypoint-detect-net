"""Render 3 overlay samples for the HF widget from data/sample/.

Writes `data/widget/sample{1,2,3}.jpg` - each is an input image with keypoints
and skeleton drawn. HF inference widget shows these as example inputs.

Widget images live OUTSIDE data/sample/ (lesson from M1/M2: Ultralytics' data
loader picks up anything under data/sample/images, which would inflate counts).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from vehicle_keypoints.inference.overlay import draw_keypoints
from vehicle_keypoints.inference.predict import Detector


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None, help="YOLO .pt (optional - random init if absent)")
    p.add_argument("--src", default="data/sample/images")
    p.add_argument("--dst", default="data/widget")
    p.add_argument("-n", type=int, default=3)
    args = p.parse_args()

    det = (
        Detector.from_checkpoint(args.checkpoint)
        if args.checkpoint
        else Detector.from_pretrained_or_random("yolo26n")
    )
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(sorted(Path(args.src).glob("*.jpg"))[: args.n], start=1):
        dets = det.predict(str(img))
        out = dst / f"sample{i}.jpg"
        draw_keypoints(img, dets, out)
        print(f"wrote {out} ({len(dets)} detections)")


if __name__ == "__main__":
    main()
