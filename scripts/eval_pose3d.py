"""Run the monocular 3D pose baseline on ApolloCar3D.

Pipeline per image: v1 detector -> 14 keypoints -> PnP -> predicted 6DoF;
match to GT cars by 2D centre; aggregate rotation/translation error; write a
metrics JSON and save qualitative overlays.

Usage:
  uv run python scripts/eval_pose3d.py \
    --apollo-root "/path/to/apollocar3d/data" \
    --checkpoint artifacts/sota/<run>/weights/best.pt \
    --split val --limit 100 --out reports/pose3d_apollo.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vehicle_keypoints.inference.predict import Detector
from vehicle_keypoints.pose3d.apollo import load_frame_cars, load_intrinsics
from vehicle_keypoints.pose3d.eval_runner import (
    PredPose,
    aggregate_metrics,
    match_by_center,
    pose_error_row,
)
from vehicle_keypoints.pose3d.lift import solve_pose
from vehicle_keypoints.pose3d.model import CanonicalCarModel
from vehicle_keypoints.pose3d.overlay3d import draw_pose3d
from vehicle_keypoints.utils import configure_logging, get_logger

log = get_logger(__name__)


def _gt_center_px(gt, k: np.ndarray) -> tuple[float, float]:
    p = k @ gt.t
    return (float(p[0] / p[2]), float(p[1] / p[2]))


def _camera_id(img_name: str) -> str:
    """Extract the camera id from a frame name like ..._Camera_6.jpg -> '6'."""
    return Path(img_name).stem.rsplit("_Camera_", 1)[-1]


def _intrinsics_for(img_name: str, root: Path, cache: dict[str, np.ndarray]) -> np.ndarray:
    """Load (and cache) the per-camera intrinsics matching this frame."""
    cam_id = _camera_id(img_name)
    if cam_id not in cache:
        cache[cam_id] = load_intrinsics(root / "camera" / f"{cam_id}.cam")
    return cache[cam_id]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apollo-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="reports/pose3d_apollo.json")
    ap.add_argument("--overlay-dir", default="reports/pose3d_overlays")
    args = ap.parse_args()

    configure_logging()
    root = Path(args.apollo_root)
    model = CanonicalCarModel.load_default()
    det = Detector.from_checkpoint(args.checkpoint)

    # Intrinsics are per-camera; the split mixes Camera_5 and Camera_6 frames.
    k_cache: dict[str, np.ndarray] = {}

    names = (root / "split" / f"{args.split}.txt").read_text(encoding="utf-8").split()
    if args.limit:
        names = names[: args.limit]

    rows: list[dict] = []
    n_gt = n_detected = 0
    for idx, img_name in enumerate(names):
        img_path = root / "images" / img_name
        pose_path = root / "car_poses" / (Path(img_name).stem + ".json")
        if not img_path.exists() or not pose_path.exists():
            continue
        k = _intrinsics_for(img_name, root, k_cache)
        gts = load_frame_cars(pose_path)
        n_gt += len(gts)
        dets = det.predict(img_path, conf=args.conf)
        preds: list[PredPose] = []
        for d in dets:
            kpts = np.array(d["keypoints"], dtype=np.float64)
            res = solve_pose(kpts[:, :2], kpts[:, 2], k, model.points)
            if res is None:
                continue
            bb = d["bbox"]
            center = (bb[0] + bb[2] / 2.0, bb[1] + bb[3] / 2.0)
            preds.append(PredPose(center_px=center, r=res.r, t=res.t))
        n_detected += len(preds)
        gt_centers = [_gt_center_px(g, k) for g in gts]
        for pi, gj in match_by_center(preds, gts, gt_centers, max_px=150.0):
            rows.append(pose_error_row(preds[pi], gts[gj]))
        if idx < 20 and preds:
            out_img = Path(args.overlay_dir) / f"{Path(img_name).stem}.png"
            draw_pose3d(img_path, model, preds[0].r, preds[0].t, k, out_img)

    summary = aggregate_metrics(rows, n_gt=n_gt, n_detected=n_detected)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("pose3d_apollo_done", **{k2: round(v, 4) for k2, v in summary.items()})


if __name__ == "__main__":
    main()
