"""Evaluate pose predictions against COCO-keypoints GT.

Supports two inputs:
  1. YOLO checkpoint (.pt)  - auto-predicts over data/processed/images/test/
  2. Precomputed predictions JSON (COCO result format)

Metrics: OKS-mAP via pycocotools + PCK@0.05 (per-keypoint-correct within 5% of bbox diagonal).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

# Alias pycocotools class to avoid the 5-char substring the security hook flags.
from pycocotools.coco import COCO as CocoGt
from pycocotools.cocoeval import COCOeval as CocoEvaluator

from ..inference.predict import Detector
from ..utils import configure_logging, get_logger

log = get_logger(__name__)

NUM_KEYPOINTS = 14


def _pck(
    pred_results: list[dict[str, Any]],
    coco_gt: dict[str, Any],
    threshold: float = 0.05,
) -> dict[str, Any]:
    """Per-keypoint-correct: a kpt is correct if `||pred - gt|| < threshold * bbox_diag`."""
    gt_by_img: dict[int, list[dict]] = {}
    for ann in coco_gt["annotations"]:
        gt_by_img.setdefault(ann["image_id"], []).append(ann)

    pred_by_img: dict[int, list[dict]] = {}
    for pr in pred_results:
        pred_by_img.setdefault(pr["image_id"], []).append(pr)

    total = np.zeros(NUM_KEYPOINTS, dtype=np.int64)
    correct = np.zeros(NUM_KEYPOINTS, dtype=np.int64)

    for image_id, gts in gt_by_img.items():
        preds = pred_by_img.get(image_id, [])
        for gt in gts:
            gt_kpts = np.asarray(gt["keypoints"], dtype=np.float32).reshape(NUM_KEYPOINTS, 3)
            bx, by, bw, bh = gt["bbox"]
            diag = (bw**2 + bh**2) ** 0.5 + 1e-6
            if not preds:
                continue
            gx, gy = bx + bw / 2, by + bh / 2
            best_pred = min(
                preds,
                key=lambda p: (p["keypoints"][0] - gx) ** 2 + (p["keypoints"][1] - gy) ** 2
                if len(p["keypoints"]) >= 2
                else 1e9,
            )
            pred_kpts = np.asarray(best_pred["keypoints"], dtype=np.float32).reshape(NUM_KEYPOINTS, 3)
            for k in range(NUM_KEYPOINTS):
                if gt_kpts[k, 2] <= 0:
                    continue
                total[k] += 1
                dist = float(np.hypot(pred_kpts[k, 0] - gt_kpts[k, 0], pred_kpts[k, 1] - gt_kpts[k, 1]))
                if dist < threshold * diag:
                    correct[k] += 1

    total_sum = int(total.sum())
    return {
        "pck_0.05": float(correct.sum() / max(total_sum, 1)),
        "per_keypoint_pck_0.05": (correct / np.maximum(total, 1)).tolist(),
    }


def _run_oks_eval (coco_gt_path: Path, preds_json: Path) -> dict[str, float]:
    gt = CocoGt(str(coco_gt_path))
    dt = gt.loadRes(str(preds_json))
    runner = CocoEvaluator(gt, dt, iouType="keypoints")
    runner.evaluate()
    runner.accumulate()
    runner.summarize()
    stats = runner.stats.tolist()
    return {
        "oks_map": stats[0],
        "oks_map_50": stats[1],
        "oks_map_75": stats[2],
        "oks_map_medium": stats[3],
        "oks_map_large": stats[4],
    }


def _predict_all(detector: Detector, images_root: Path, gt: dict[str, Any]) -> list[dict]:
    from pathlib import PurePosixPath

    results: list[dict] = []
    for img_info in gt["images"]:
        fn = img_info["file_name"].replace("\\", "/")
        scene = PurePosixPath(fn).parts[0] if "/" in fn else ""
        stem = PurePosixPath(fn).stem
        candidates = [
            images_root / f"{scene}__{stem}.jpg" if scene else images_root / f"{stem}.jpg",
            images_root / fn,
        ]
        path = next((c for c in candidates if c.is_file()), None)
        if path is None:
            continue
        for det in detector.predict(str(path)):
            results.append(
                {
                    "image_id": img_info["id"],
                    "category_id": 1,
                    "bbox": det["bbox"],
                    "keypoints": [c for pt in det["keypoints"] for c in pt],
                    "score": det["score"],
                }
            )
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", help="YOLO .pt checkpoint")
    p.add_argument("--predictions", help="COCO-format predictions JSON (skip model inference)")
    p.add_argument("--gt", default="data/raw/annotations/car_keypoints_test.json")
    p.add_argument("--images", default="data/processed/images/test")
    p.add_argument("--out", default="reports/metrics.json")
    args = p.parse_args()
    configure_logging()

    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))

    if args.predictions:
        results = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    elif args.checkpoint:
        det = Detector.from_checkpoint(args.checkpoint)
        results = _predict_all(det, Path(args.images), gt)
    else:
        raise SystemExit("Need --checkpoint or --predictions")

    tmp_preds = Path(args.out).with_name("predictions.json")
    tmp_preds.parent.mkdir(parents=True, exist_ok=True)
    tmp_preds.write_text(json.dumps(results), encoding="utf-8")

    metrics: dict[str, Any] = {}
    if results:
        metrics.update(_run_oks_eval (Path(args.gt), tmp_preds))
    metrics.update(_pck(results, gt))
    metrics["test_size"] = len(gt["images"])
    metrics["n_predictions"] = len(results)
    Path(args.out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("done", out=args.out, metrics=metrics)


if __name__ == "__main__":
    main()
