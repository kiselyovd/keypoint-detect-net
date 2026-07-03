"""Test whether the synthetic-only sim2real failure is caused by the noisy
background city-vehicle labels (97% of instances, small far cars) rather than the
texture domain gap.

We rebuild the synth->YOLO training set under a label-quality filter, retrain a
synthetic-ONLY detector (val = real CarFusion), and compare to the unfiltered
synth-only run. Filters:

  --mode rig    : keep only the rig (largest, carefully-built instance) per image
  --mode size   : keep instances whose bbox diagonal >= --min-diag px (absolute)
  --mode pct    : per-image, keep instances at/above the --pct percentile of that
                  image's instance sizes (adaptive: drops the small far half of each
                  scene; the rig is the largest so it always survives)
  --mode all    : keep everything (reproduces the original synth-only)

If a clean filter restores real-world transfer, the labels were the poison and the
dataset generator should apply a distance/size cap. If it stays ~0, the failure is
the appearance domain gap and label filtering will not fix it.

  uv run python scripts/synth_clean_experiment.py --mode rig --size s
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO / "scripts"))
from phase0_train import (  # noqa: E402
    CARFUSION_FLIP_IDX,
    NUM_KPT,
    _coco_to_yolo_row,
    _yolo_train,
    log,
    run_eval,
)
from phase0_train_v4 import SYNTH_V4_ROOT  # noqa: E402

SYNTH_COCO = SYNTH_V4_ROOT / "annotations" / "coco.json"
REAL_VAL = (REPO / "data" / "processed" / "images" / "val").resolve()
WORK = REPO / "artifacts" / "phase0_work"
RUN_DIR = REPO / "artifacts" / "sweep_runs"
REPORTS = REPO / "reports"


def _diag(ann):
    bw, bh = ann["bbox"][2], ann["bbox"][3]
    return (bw * bw + bh * bh) ** 0.5


def _edge_ratio(cv2, gray, bbox):
    """Interior/background edge-energy ratio for a bbox. A real rendered car carries
    far more interior edge energy than its surrounding background; a phantom label on
    empty ground (UE5 culled the car at distance but we still projected keypoints) has
    interior ~= background. >>1 = visible car, <~0.6 = phantom on empty ground."""
    ih, iw = gray.shape
    x, y, w, h = bbox
    x0, y0, x1, y1 = int(max(0, x)), int(max(0, y)), int(min(iw, x + w)), int(min(ih, y + h))
    if x1 - x0 < 6 or y1 - y0 < 6:
        return 99.0  # too small to judge; keep (size filters handle these)
    ins = cv2.Laplacian(gray[y0:y1, x0:x1], cv2.CV_32F).var()
    mx, my = int(w * 0.6), int(h * 0.6)
    bx0, by0, bx1, by1 = max(0, x0 - mx), max(0, y0 - my), min(iw, x1 + mx), min(ih, y1 + my)
    ring = gray[by0:by1, bx0:bx1].copy()
    ring[y0 - by0 : y1 - by0, x0 - bx0 : x1 - bx0] = ring.mean()
    return float(ins) / max(float(cv2.Laplacian(ring, cv2.CV_32F).var()), 1e-3)


def build_filtered_yolo(
    mode: str,
    min_diag: float,
    pct: float,
    out_dir: Path,
    realism: bool = False,
    dephantom: float = 0.0,
    min_area: float = 0.0,
) -> Path:
    """Convert the synth COCO to a YOLO train set under the chosen instance filter,
    90/10 split (seed 42). Returns the data.yaml path (val = real CarFusion val).

    realism=True passes each TRAIN image through the domain-randomized camera-realism
    filter (closes part of the synth->real appearance gap); val stays untouched real.
    dephantom>0 drops instances whose interior/background edge ratio is below that
    value (phantom labels on empty ground where UE5 culled the car at distance).
    min_area>0 drops instances whose bbox area (px^2) is below that - a hard floor so
    no instance is so tiny that its keypoints collapse onto a single pixel.
    """
    import random as _random

    import cv2  # needed for dephantom edge measure and/or realism

    rfilter = None
    if realism:
        from realism_filter import apply_random

        rfilter = (cv2, apply_random, _random.Random(7))
    data = json.loads(SYNTH_COCO.read_text())
    imgs = {im["id"]: im for im in data["images"]}
    by_img: dict[int, list[dict]] = {}
    largest: dict[int, float] = {}
    for a in data["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
        area = a["bbox"][2] * a["bbox"][3]
        largest[a["image_id"]] = max(largest.get(a["image_id"], 0.0), area)

    # per-image size percentile cutoff (adaptive filter): keep instances whose
    # diagonal is at/above the pct-th percentile of that image's instance diagonals.
    pct_cut: dict[int, float] = {}
    if mode == "pct":
        for iid, anns in by_img.items():
            pct_cut[iid] = float(np.percentile([_diag(a) for a in anns], pct))

    def keep(a):
        if mode == "all":
            return True
        if mode == "size":
            return _diag(a) >= min_diag
        if mode == "pct":
            return _diag(a) >= pct_cut[a["image_id"]]
        if mode == "rig":  # only the largest (carefully-built) instance per image
            return abs(a["bbox"][2] * a["bbox"][3] - largest[a["image_id"]]) < 1e-6
        raise ValueError(mode)

    import random

    ids = sorted(imgs)
    random.Random(42).shuffle(ids)
    n_val = max(1, round(len(ids) * 0.1))
    splits = {"val": set(ids[:n_val]), "train": set(ids[n_val:])}

    kept = dropped = 0
    for split, sids in splits.items():
        idir = out_dir / "images" / split
        ldir = out_dir / "labels" / split
        idir.mkdir(parents=True, exist_ok=True)
        ldir.mkdir(parents=True, exist_ok=True)
        for iid in sids:
            im = imgs[iid]
            src = SYNTH_V4_ROOT / im["file_name"]
            if not src.is_file():
                continue
            # load grayscale once if we need the edge measure (dephantom on train split)
            gray = None
            if dephantom > 0 and split == "train":
                bgr = cv2.imread(str(src))
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr is not None else None
            rows = []
            for a in by_img.get(iid, []):
                if not keep(a):
                    dropped += 1
                    continue
                if min_area > 0 and a["bbox"][2] * a["bbox"][3] < min_area:
                    dropped += 1  # too tiny - keypoints would collapse onto one pixel
                    continue
                if gray is not None and _edge_ratio(cv2, gray, a["bbox"]) < dephantom:
                    dropped += 1  # phantom label on empty ground (culled car)
                    continue
                rows.append(_coco_to_yolo_row(a, im["width"], im["height"]))
                kept += 1
            # keep the image even with zero rows? no - skip empty to avoid bg-only noise
            if not rows:
                continue
            if rfilter is not None and split == "train":  # realism only on train; val is real
                cv2, apply_random, rng = rfilter
                cv2.imwrite(str(idir / src.name), apply_random(cv2.imread(str(src)), rng))
            else:
                shutil.copy2(src, idir / src.name)
            (ldir / (src.stem + ".txt")).write_text("\n".join(rows) + "\n", encoding="utf-8")
    log(
        f"[{mode}] kept {kept} instances, dropped {dropped} "
        f"({100 * kept / (kept + dropped):.1f}% kept)"
    )

    cfg = {
        "path": str(out_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": str(REAL_VAL).replace("\\", "/"),
        "kpt_shape": [NUM_KPT, 3],
        "names": {0: "car"},
        "flip_idx": CARFUSION_FLIP_IDX,
    }
    yml = out_dir / "data.yaml"
    yml.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return yml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rig", "size", "pct", "all"], required=True)
    ap.add_argument("--size", choices=["n", "s", "m", "l"], default="s")
    ap.add_argument("--min-diag", type=float, default=100.0)
    ap.add_argument("--pct", type=float, default=50.0, help="per-image size percentile cutoff")
    ap.add_argument(
        "--realism",
        action="store_true",
        help="apply the domain-randomized camera-realism filter to train frames",
    )
    ap.add_argument(
        "--dephantom",
        type=float,
        default=0.0,
        help="drop instances with interior/bg edge ratio below this (phantom labels)",
    )
    ap.add_argument(
        "--min-area",
        type=float,
        default=0.0,
        help="drop instances with bbox area (px^2) below this (anti keypoint-collapse)",
    )
    args = ap.parse_args()

    if args.mode == "size":
        suffix = f"_{int(args.min_diag)}"
    elif args.mode == "pct":
        suffix = f"{int(args.pct)}"
    else:
        suffix = ""
    rtag = "_realism" if args.realism else ""
    dtag = f"_dp{args.dephantom:g}" if args.dephantom > 0 else ""
    atag = f"_a{int(args.min_area)}" if args.min_area > 0 else ""
    tag = f"{args.size}_synthclean_{args.mode}{suffix}{rtag}{dtag}{atag}"
    out_dir = WORK / f"synth_yolo_{args.mode}{suffix}{rtag}{dtag}{atag}"
    yml = build_filtered_yolo(
        args.mode,
        args.min_diag,
        args.pct,
        out_dir,
        realism=args.realism,
        dephantom=args.dephantom,
        min_area=args.min_area,
    )

    batch = 10 if args.size == "l" else 16
    log(f"=== train {tag}: yolo26{args.size}-pose synth-only [{args.mode}] -> real val ===")
    best = _yolo_train(
        init_model=f"yolo26{args.size}-pose.pt",
        data_yaml=yml,
        run_dir=RUN_DIR,
        name=tag,
        epochs=30,
        imgsz=480,
        batch=batch,
        lr0=1e-3,
        patience=10,
        workers=4,
    )
    m = run_eval(best, REPORTS / f"sweep_{tag}_metrics.json")
    log(
        f"RESULT {tag}: OKS {m['oks_map']:.4f} mAP50 {m.get('oks_map_50', 0):.4f} "
        f"PCK {m.get('pck_0.05', 0):.4f}"
    )


if __name__ == "__main__":
    main()
