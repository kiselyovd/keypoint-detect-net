"""Calibrate MODEL_TO_APOLLO by data-driven rotation averaging.

For each matched (predicted, GT) car pair we want A such that
R_pred ~= R_gt @ A, i.e. A_i = R_gt_i.T @ R_pred_i. The best proper rotation A
is the SO(3) projection of the mean of the A_i (Wahba/Procrustes). We report
the median geodesic rotation error before (A = I) and after, and print A as a
numpy literal to paste into pose3d/frame_align.py. A few zoomed pred overlays
are saved for the visual sanity gate.

Usage:
  uv run python scripts/calibrate_pose3d_frame.py \
    --apollo-root "/path/to/apollocar3d/data" \
    --checkpoint artifacts/hf_export/weights.pt --n 80
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vehicle_keypoints.inference.predict import Detector
from vehicle_keypoints.pose3d.apollo import load_frame_cars, load_intrinsics
from vehicle_keypoints.pose3d.eval_runner import PredPose, match_by_center
from vehicle_keypoints.pose3d.lift import solve_pose
from vehicle_keypoints.pose3d.metrics import geodesic_rotation_deg
from vehicle_keypoints.pose3d.model import CanonicalCarModel
from vehicle_keypoints.pose3d.overlay3d import draw_pose3d


def _camera_id(name: str) -> str:
    return Path(name).stem.rsplit("_Camera_", 1)[-1]


def _average_rotation(mats: list[np.ndarray]) -> np.ndarray:
    """Best proper rotation (SO(3) projection of the arithmetic mean)."""
    m = np.sum(mats, axis=0)
    u, _, vt = np.linalg.svd(m)
    a = u @ vt
    if np.linalg.det(a) < 0:  # enforce proper rotation
        u[:, -1] *= -1
        a = u @ vt
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apollo-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--out", default="reports/pose3d_calib")
    args = ap.parse_args()

    root = Path(args.apollo_root)
    model = CanonicalCarModel.load_default()
    det = Detector.from_checkpoint(args.checkpoint)
    k_cache: dict[str, np.ndarray] = {}
    names = (root / "split" / "val.txt").read_text(encoding="utf-8").split()[: args.n]

    a_samples: list[np.ndarray] = []
    pred_rots: list[np.ndarray] = []
    gt_rots: list[np.ndarray] = []
    overlay_done = 0
    for img_name in names:
        img_path = root / "images" / img_name
        pose_path = root / "car_poses" / (Path(img_name).stem + ".json")
        if not img_path.exists() or not pose_path.exists():
            continue
        cam_id = _camera_id(img_name)
        if cam_id not in k_cache:
            k_cache[cam_id] = load_intrinsics(root / "camera" / f"{cam_id}.cam")
        k = k_cache[cam_id]
        gts = load_frame_cars(pose_path)
        dets = det.predict(img_path)
        preds: list[PredPose] = []
        for d in dets:
            kpts = np.array(d["keypoints"], dtype=np.float64)
            res = solve_pose(kpts[:, :2], kpts[:, 2], k, model.points)
            if res is None:
                continue
            bb = d["bbox"]
            preds.append(
                PredPose(center_px=(bb[0] + bb[2] / 2, bb[1] + bb[3] / 2), r=res.r, t=res.t)
            )
        gt_centers = []
        for g in gts:
            p = k @ g.t
            gt_centers.append((float(p[0] / p[2]), float(p[1] / p[2])))
        for pi, gj in match_by_center(preds, gts, gt_centers, max_px=150.0):
            rp, rg = preds[pi].r, gts[gj].r
            pred_rots.append(rp)
            gt_rots.append(rg)
            a_samples.append(rg.T @ rp)
        if overlay_done < 4 and preds:
            out = Path(args.out) / f"pred_{Path(img_name).stem}.png"
            draw_pose3d(img_path, model, preds[0].r, preds[0].t, k, out)
            overlay_done += 1

    if not a_samples:
        print("No matches found - cannot calibrate.")
        return
    a_hat = _average_rotation(a_samples)
    pairs = list(zip(pred_rots, gt_rots, strict=True))
    before = float(np.median([geodesic_rotation_deg(rp, rg) for rp, rg in pairs]))
    after = float(np.median([geodesic_rotation_deg(rp, rg @ a_hat) for rp, rg in pairs]))
    print(f"matches={len(a_samples)}")
    print(f"median rotation error  before (A=I): {before:.1f} deg")
    print(f"median rotation error  after  (A=hat): {after:.1f} deg")
    print("MODEL_TO_APOLLO = np.array([")
    for row in a_hat:
        print(f"    [{row[0]!r}, {row[1]!r}, {row[2]!r}],")
    print("])")
    print(f"pred overlays saved under {args.out}")


if __name__ == "__main__":
    main()
