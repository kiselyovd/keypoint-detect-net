"""#1 bootstrap CI on the synthetic PCK delta, and #4 re-evaluation excluding the
two ill-defined keypoints (exhaust=8, center=13). Consumes the dumped v6 arm
predictions. One-off analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval as CocoEvaluator

REPO = Path(__file__).parent.parent
GT = REPO / "data" / "raw" / "annotations" / "car_keypoints_test.json"
NK = 14
DROP = {8, 13}  # exhaust, center - the two ill-defined points
RNG_SEED = 12345


def _load(tag: str) -> list[dict]:
    return json.loads((REPO / "reports" / f"phase0_v6_{tag}_preds.json").read_text())


def _match(gt: dict, preds: list[dict]):
    """For each GT ann, nearest predicted instance (by centre), as in evaluate._pck."""
    pred_by_img: dict[int, list[dict]] = {}
    for pr in preds:
        pred_by_img.setdefault(pr["image_id"], []).append(pr)
    rows = []  # (image_id, gt_kpts[NK,3], bbox_diag, pred_kpts[NK,3] or None)
    for ann in gt["annotations"]:
        g = np.asarray(ann["keypoints"], np.float32).reshape(NK, 3)
        bx, by, bw, bh = ann["bbox"]
        diag = (bw**2 + bh**2) ** 0.5 + 1e-6
        ps = pred_by_img.get(ann["image_id"], [])
        pk = None
        if ps:
            gx, gy = bx + bw / 2, by + bh / 2
            best = min(
                ps, key=lambda p: (p["keypoints"][0] - gx) ** 2 + (p["keypoints"][1] - gy) ** 2
            )
            pk = np.asarray(best["keypoints"], np.float32).reshape(NK, 3)
        rows.append((ann["image_id"], g, diag, pk))
    return rows


def _pck_hits(rows, keep):
    """Per-image (hits, total) over the kept keypoints, for bootstrap."""
    per_img: dict[int, list[int]] = {}
    for img, g, diag, pk in rows:
        h, t = 0, 0
        for k in keep:
            if g[k, 2] <= 0:
                continue
            t += 1
            if pk is not None and np.hypot(pk[k, 0] - g[k, 0], pk[k, 1] - g[k, 1]) < 0.05 * diag:
                h += 1
        if t:
            per_img.setdefault(img, [0, 0])
            per_img[img][0] += h
            per_img[img][1] += t
    return per_img


def _pck(per_img):
    h = sum(v[0] for v in per_img.values())
    t = sum(v[1] for v in per_img.values())
    return h / max(t, 1)


def _oks_subset(drop: set[int]) -> dict:
    """OKS-mAP for each arm with `drop` keypoints zeroed in the GT (excluded)."""
    gt = json.loads(GT.read_text())
    for ann in gt["annotations"]:
        k = ann["keypoints"]
        for i in drop:
            k[3 * i + 2] = 0  # visibility -> 0 excludes from OKS
    tmp = REPO / "reports" / "_gt_subset.json"
    tmp.write_text(json.dumps(gt))
    out = {}
    for tag in ("armB", "armA"):
        preds = REPO / "reports" / f"phase0_v6_{tag}_preds.json"
        cg = COCO(str(tmp))
        dt = cg.loadRes(str(preds))
        ev = CocoEvaluator(cg, dt, iouType="keypoints")
        ev.params.kpt_oks_sigmas = np.ones(NK) * 0.05
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        out[tag] = float(ev.stats[0])
    return out


def main() -> None:
    gt = json.loads(GT.read_text())
    rows = {tag: _match(gt, _load(tag)) for tag in ("armB", "armA")}
    all_k = list(range(NK))
    keepk = [k for k in range(NK) if k not in DROP]

    # ---- #1 bootstrap CI on the full-14 PCK delta (armA - armB), over images ----
    pi_a = _pck_hits(rows["armA"], all_k)
    pi_b = _pck_hits(rows["armB"], all_k)
    imgs = sorted(set(pi_a) | set(pi_b))
    pck_a, pck_b = _pck(pi_a), _pck(pi_b)
    rng = np.random.default_rng(RNG_SEED)
    deltas = []
    arr_imgs = np.array(imgs)
    for _ in range(2000):
        samp = rng.choice(arr_imgs, size=len(arr_imgs), replace=True)
        h_a = t_a = h_b = t_b = 0
        for im in samp:
            if im in pi_a:
                h_a += pi_a[im][0]
                t_a += pi_a[im][1]
            if im in pi_b:
                h_b += pi_b[im][0]
                t_b += pi_b[im][1]
        deltas.append(h_a / max(t_a, 1) - h_b / max(t_b, 1))
    lo, hi = np.percentile(deltas, [2.5, 97.5]) * 100
    print(
        f"[#1] full-14 PCK: armA {pck_a * 100:.1f}  armB {pck_b * 100:.1f}  "
        f"delta {(pck_a - pck_b) * 100:+.2f}pp  95% CI [{lo:+.2f}, {hi:+.2f}]pp (n_boot=2000)"
    )

    # ---- #4 re-eval excluding exhaust+center ----
    pck_a12 = _pck(_pck_hits(rows["armA"], keepk))
    pck_b12 = _pck(_pck_hits(rows["armB"], keepk))
    print(
        f"[#4] PCK excl. exhaust+center: armA {pck_a12 * 100:.1f}  armB {pck_b12 * 100:.1f}  "
        f"delta {(pck_a12 - pck_b12) * 100:+.2f}pp  (full-14 delta {(pck_a - pck_b) * 100:+.2f}pp)"
    )
    oks = _oks_subset(DROP)
    print(
        f"[#4] OKS-mAP excl. exhaust+center: armA {oks['armA']:.4f}  armB {oks['armB']:.4f}  "
        f"delta {(oks['armA'] - oks['armB']) * 100:+.2f}pp  (full-14 delta -2.35pp)"
    )
    with open(REPO / "reports" / "v6_labelqual_analysis.json", "w") as f:
        json.dump(
            {
                "pck_full_delta": (pck_a - pck_b) * 100,
                "pck_ci": [lo, hi],
                "pck12_delta": (pck_a12 - pck_b12) * 100,
                "oks12": oks,
            },
            f,
            indent=2,
        )
    print("DONE_ANALYSIS")


if __name__ == "__main__":
    main()
