"""Side-by-side prediction comparison on REAL CarFusion images.

The OKS-mAP numbers are computed against CarFusion's own ground truth, which is
noisy - so a model trained on exact synthetic labels can predict better yet score
worse. This tool puts the noisy GT next to each model's predictions so quality can
be judged by eye instead of trusting the GT-based metric.

Inference runs on CPU on purpose (the GPU is busy training); a handful of images
x a few models is fast enough.

  uv run python scripts/compare_preds.py --n 6 --models baseline armB armA
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
OUT = REPO / "reports" / "pred_compare.png"


def _sweep(name):
    return REPO / "artifacts" / "sweep_runs" / name / "weights" / "best.pt"


CKPTS = {
    "baseline": REPO / "artifacts" / "baseline_flipfix" / "sota_flipfix3" / "weights" / "best.pt",
    "armB": REPO / "artifacts" / "phase0_v6_runs" / "v6_control_realx8" / "weights" / "best.pt",
    "armA": REPO / "artifacts" / "phase0_v6_runs" / "v6_mixed_finetune" / "weights" / "best.pt",
    "n_real": _sweep("n_real"),
    "n_synth": _sweep("n_synth"),
    "s_real": _sweep("s_real"),
    "s_synth": _sweep("s_synth"),
    "m_real": _sweep("m_real"),
    "m_synth": _sweep("m_synth"),
    "l_real": _sweep("l_real"),
    "l_synth": _sweep("l_synth"),
    "n_synthonly": _sweep("n_synthonly"),
    "s_synthonly": _sweep("s_synthonly"),
    "m_synthonly": _sweep("m_synthonly"),
    "l_synthonly": _sweep("l_synthonly"),
}
LABELS = {
    "baseline": "CarFusion-only (base)",
    "armB": "real-only (armB)",
    "armA": "+SYNTH (armA)",
    "n_real": "nano real",
    "l_real": "L real-only (best)",
    "l_synth": "L +SYNTH",
    "s_real": "s real-only",
    "s_synth": "s +SYNTH",
    "n_synthonly": "nano SYNTH-ONLY",
    "l_synthonly": "L SYNTH-ONLY",
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


def proc_name(file_name: str) -> str:
    """CarFusion GT file_name 'seq\\images_jpg\\frame.jpg' -> flat 'seq__frame.jpg'
    as stored under data/processed/images/test."""
    parts = file_name.replace("\\", "/").split("/")
    return f"{parts[0]}__{parts[-1]}"


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


def draw(img, kpts, color):
    p = [(int(x), int(y), v) for x, y, v in kpts]
    for a, b in SKEL:
        if p[a][2] > 0 and p[b][2] > 0:
            cv2.line(img, p[a][:2], p[b][:2], (0, 0, 0), 4, cv2.LINE_AA)
            cv2.line(img, p[a][:2], p[b][:2], color, 2, cv2.LINE_AA)
    for x, y, v in p:
        if v > 0:
            cv2.circle(img, (x, y), 4, color, -1, cv2.LINE_AA)


def crop(img, bbox, pad=0.35):
    h, w = img.shape[:2]
    x, y, bw, bh = bbox
    cx, cy = x + bw / 2, y + bh / 2
    s = max(bw, bh) * (1 + pad)
    x0, y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    x1, y1 = int(min(w, cx + s / 2)), int(min(h, cy + s / 2))
    return img[y0:y1, x0:x1], (x0, y0)


def panel(img, kpts, bbox, label, color):
    sub, (ox, oy) = crop(img, bbox)
    shifted = [[x - ox, y - oy, v] for x, y, v in kpts]
    sub = sub.copy()
    draw(sub, shifted, color)
    sub = cv2.resize(sub, (300, 300))
    cv2.rectangle(sub, (0, 0), (300, 22), (0, 0, 0), -1)
    cv2.putText(sub, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return sub


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--models", nargs="+", default=["baseline", "armB", "armA"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--rank",
        type=int,
        default=0,
        help="rank N sampled frames by model-consensus-vs-GT; show top --n",
    )
    args = ap.parse_args()

    coco = json.loads(GT_JSON.read_text(encoding="utf-8"))
    imgs = {im["id"]: im for im in coco["images"]}
    anns_by_img: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    # images whose biggest GT car has many labelled points and a decent size
    cand = []
    for iid, anns in anns_by_img.items():
        big = max(anns, key=lambda a: a["bbox"][2] * a["bbox"][3])
        vis = sum(1 for k in range(2, len(big["keypoints"]), 3) if big["keypoints"][k] > 0)
        fpath = IMG_DIR / proc_name(imgs[iid]["file_name"])
        if vis >= 10 and big["bbox"][2] * big["bbox"][3] > 120 * 120 and fpath.exists():
            cand.append((iid, big, fpath))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(cand)

    dets = {}
    for m in args.models:
        d = Detector.from_checkpoint(str(CKPTS[m]))
        d.model.to("cpu")  # keep the GPU free for training
        dets[m] = d
        print(f"loaded {m}")

    def model_kpts(fpath, gt):
        out = {}
        for m in args.models:
            preds = dets[m].predict(str(fpath), conf=0.15)
            if preds:
                best = max(preds, key=lambda p: iou(p["bbox"], gt["bbox"]))
                out[m] = np.array(best["keypoints"], float)
            else:
                out[m] = np.zeros((14, 3))
        return out

    if args.rank:
        # Surface frames where the MODELS agree with each other but disagree with the
        # CarFusion GT - the signature of a mislabelled GT (models right, GT wrong).
        pool = cand[: args.rank]
        scored = []
        for k, (_iid, gt, fpath) in enumerate(pool):
            mk = model_kpts(fpath, gt)
            gtk = np.array(
                [
                    [gt["keypoints"][i], gt["keypoints"][i + 1], gt["keypoints"][i + 2]]
                    for i in range(0, 14 * 3, 3)
                ],
                float,
            )
            diag = max(1.0, (gt["bbox"][2] ** 2 + gt["bbox"][3] ** 2) ** 0.5)
            preds = np.stack(list(mk.values()))  # [M,14,3]
            vis_mask = (preds[..., 2] > 0).all(0) & (gtk[:, 2] > 0)
            if vis_mask.sum() < 6:
                continue
            consensus = preds[..., :2].mean(0)
            disagree = (
                np.linalg.norm(preds[..., :2] - consensus, axis=-1)[:, vis_mask].mean() / diag
            )
            gt_dist = np.linalg.norm(consensus - gtk[:, :2], axis=-1)[vis_mask].mean() / diag
            scored.append((gt_dist - disagree, _iid, gt, fpath, mk))
            if k % 10 == 0:
                print(f"  ranked {k}/{len(pool)}")
        scored.sort(key=lambda t: t[0], reverse=True)
        picks = [(t[1], t[2], t[3], t[4]) for t in scored[: args.n]]
        print(f"ranked {len(scored)} frames; top GT-disagreement = {scored[0][0]:.3f}")
    else:
        picks = [(iid, gt, fpath, None) for iid, gt, fpath in cand[: args.n]]
    print(f"{len(cand)} candidates; rendering {len(picks)}")

    rows = []
    for _iid, gt, fpath, mk in picks:
        img = cv2.imread(str(fpath))
        gt_kpts = [
            [gt["keypoints"][i], gt["keypoints"][i + 1], gt["keypoints"][i + 2]]
            for i in range(0, 14 * 3, 3)
        ]
        panels = [panel(img, gt_kpts, gt["bbox"], "CarFusion GT (noisy)", (60, 200, 255))]
        if mk is None:
            mk = model_kpts(fpath, gt)
        for m in args.models:
            panels.append(panel(img, mk[m].tolist(), gt["bbox"], LABELS.get(m, m), (120, 255, 120)))
        rows.append(np.hstack(panels))
    cv2.imwrite(str(OUT), np.vstack(rows))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
