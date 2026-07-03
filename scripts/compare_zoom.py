"""Zoomed, per-keypoint comparison of l_real vs l_synthonly on a few large cars.

Picks the large test cars where the two models DISAGREE most (so the synthetic
contribution is actually visible), then renders big GT | l_real | l_synthonly panels
with numbered keypoints so wheels/lights/roof can be judged point by point.

  uv run python scripts/compare_zoom.py --n 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO / "src"))
from vehicle_keypoints.inference.predict import Detector  # noqa: E402

GT_JSON = REPO / "data" / "raw" / "annotations" / "car_keypoints_test.json"
IMG_DIR = REPO / "data" / "processed" / "images" / "test"
OUT = REPO / "reports" / "pred_zoom.png"
CKPTS = {
    "l_real": REPO / "artifacts" / "sweep_runs" / "l_real" / "weights" / "best.pt",
    "l_synthonly": REPO / "artifacts" / "sweep_runs" / "l_synthonly" / "weights" / "best.pt",
}
SKEL = [
    (0, 2),
    (1, 3),
    (0, 1),
    (2, 3),
    (9, 11),
    (10, 12),
    (9, 10),
    (11, 12),
    (4, 0),
    (5, 1),
    (6, 2),
    (7, 3),
    (4, 9),
    (5, 10),
    (6, 11),
    (7, 12),
    (4, 5),
    (6, 7),
]
PANEL = 440


def proc_name(fn: str) -> str:
    parts = fn.replace("\\", "/").split("/")
    return f"{parts[0]}__{parts[-1]}"


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def draw(img, kpts, color):
    p = [(int(x), int(y), v) for x, y, v in kpts]
    for a, b in SKEL:
        if p[a][2] > 0 and p[b][2] > 0:
            cv2.line(img, p[a][:2], p[b][:2], (0, 0, 0), 5, cv2.LINE_AA)
            cv2.line(img, p[a][:2], p[b][:2], color, 2, cv2.LINE_AA)
    for i, (x, y, v) in enumerate(p):
        if v > 0:
            cv2.circle(img, (x, y), 7, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), 5, color, -1, cv2.LINE_AA)
            cv2.putText(
                img,
                str(i),
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )


def panel(img, kpts, bbox, label, color):
    h, w = img.shape[:2]
    x, y, bw, bh = bbox
    cx, cy = x + bw / 2, y + bh / 2
    s = max(bw, bh) * 1.3
    x0, y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    x1, y1 = int(min(w, cx + s / 2)), int(min(h, cy + s / 2))
    sub = img[y0:y1, x0:x1].copy()
    shifted = [[kx - x0, ky - y0, v] for kx, ky, v in kpts]
    draw(sub, shifted, color)
    sub = cv2.resize(sub, (PANEL, PANEL))
    cv2.rectangle(sub, (0, 0), (PANEL, 26), (0, 0, 0), -1)
    cv2.putText(sub, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return sub


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--pool", type=int, default=60)
    args = ap.parse_args()

    coco = json.loads(GT_JSON.read_text(encoding="utf-8"))
    imgs = {im["id"]: im for im in coco["images"]}
    anns_by_img: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    cand = []
    for iid, anns in anns_by_img.items():
        big = max(anns, key=lambda a: a["bbox"][2] * a["bbox"][3])
        vis = sum(1 for k in range(2, len(big["keypoints"]), 3) if big["keypoints"][k] > 0)
        fp = IMG_DIR / proc_name(imgs[iid]["file_name"])
        if vis >= 12 and big["bbox"][2] > 200 and big["bbox"][3] > 150 and fp.exists():
            cand.append((iid, big, fp))
    rng = np.random.default_rng(3)
    rng.shuffle(cand)
    cand = cand[: args.pool]

    dets = {m: Detector.from_checkpoint(str(p)) for m, p in CKPTS.items()}
    for d in dets.values():
        d.model.to("cpu")
    print(f"loaded {list(dets)}; scoring {len(cand)} large cars by inter-model divergence")

    scored = []
    for _iid, gt, fp in cand:
        pk = {}
        for m, d in dets.items():
            pr = d.predict(str(fp), conf=0.15)
            if pr:
                best = max(pr, key=lambda p: iou(p["bbox"], gt["bbox"]))
                pk[m] = np.array(best["keypoints"], float)
            else:
                pk[m] = np.zeros((14, 3))
        both = (pk["l_real"][:, 2] > 0) & (pk["l_synthonly"][:, 2] > 0)
        if both.sum() < 8:
            continue
        diag = (gt["bbox"][2] ** 2 + gt["bbox"][3] ** 2) ** 0.5
        delta = pk["l_real"][:, :2] - pk["l_synthonly"][:, :2]
        div = np.linalg.norm(delta, axis=1)[both].mean() / diag
        scored.append((div, _iid, gt, fp, pk))
    scored.sort(key=lambda t: t[0], reverse=True)
    print(f"top divergence = {scored[0][0]:.3f}")

    rows = []
    for _div, _iid, gt, fp, pk in scored[: args.n]:
        img = cv2.imread(str(fp))
        gtk = [
            [gt["keypoints"][i], gt["keypoints"][i + 1], gt["keypoints"][i + 2]]
            for i in range(0, 14 * 3, 3)
        ]
        panels = [
            panel(img, gtk, gt["bbox"], "CarFusion GT (noisy)", (60, 200, 255)),
            panel(img, pk["l_real"].tolist(), gt["bbox"], "L real-only", (120, 255, 120)),
            panel(img, pk["l_synthonly"].tolist(), gt["bbox"], "L SYNTH-ONLY", (255, 170, 90)),
        ]
        rows.append(np.hstack(panels))
    cv2.imwrite(str(OUT), np.vstack(rows))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
