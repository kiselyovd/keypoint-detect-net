"""Phase 0 kill-switch experiment v4: MIXED synth+real training with real-oversampling x8.

v3 diagnosed a new failure: despite joint training, synthetic frames dominated
~88% of batches because the 100 real frames were vastly outnumbered by the 816
synth frames. The v4 fix is to oversample the 100 real frames x8 (producing 800
copies) so the real domain is seen 8x more often in each epoch.

The v4 synth source is the wider multi-venue MRQ dataset (phase0_v4), replacing
the single-scene City Sample phase0_v2.

  arm A (v4): v1 ckpt -> mixed(synth_v4 + 100 real x8) fine-tune
  arm B     : v2's control result reused verbatim (v1 + 100 real, no synth)

Synth v4 dataset: wide multi-venue MRQ captures. Eval = the same module that
produced the v1 baseline. Kill switch: arm A >= v1 + 2pp OKS-mAP.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from datetime import date
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from phase0_train import (  # noqa: E402
    NUM_KPT,
    _yolo_train,
    convert_synth_to_yolo,
    log,
    run_eval,
)

V1_CKPT = REPO_ROOT / "artifacts" / "sota" / "weights" / "best.pt"
RUN_DIR = REPO_ROOT / "artifacts" / "phase0_v4_runs"
REPORTS = REPO_ROOT / "reports"
WORK = REPO_ROOT / "artifacts" / "phase0_work"
SYNTH_V4_ROOT = Path(
    os.environ.get(
        "VK_SYNTH_PHASE0V4_DIR",
        str(REPO_ROOT.parent / "ue5-vehicle-synth" / "captures" / "phase0_v4"),
    )
)


def oversample_real(real_subset_dir: Path, out_dir: Path, factor: int = 8) -> Path:
    """Replicate each real image/label pair `factor` times under out_dir so the
    YOLO sampler sees the real domain `factor`x more often in the mix."""
    for sub in ("images/train", "labels/train"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    imgs = sorted((real_subset_dir / "images" / "train").glob("*"))
    for img in imgs:
        lbl = real_subset_dir / "labels" / "train" / (img.stem + ".txt")
        for k in range(factor):
            shutil.copy2(img, out_dir / "images" / "train" / f"{img.stem}_r{k}{img.suffix}")
            if lbl.exists():
                shutil.copy2(lbl, out_dir / "labels" / "train" / f"{img.stem}_r{k}.txt")
    src_yaml = real_subset_dir / "real_subset_data.yaml"
    if src_yaml.exists():
        shutil.copy2(src_yaml, out_dir / "real_subset_data.yaml")
    return out_dir


def build_mixed_yaml(synth_yolo_dir: Path, real_subset_dir: Path, out_path: Path) -> Path:
    """One YOLO dataset whose train spans synth v4 + the 100-real subset (oversampled)."""
    real_yaml = yaml.safe_load(
        (real_subset_dir / "real_subset_data.yaml").read_text(encoding="utf-8")
    )
    mixed = {
        "train": [
            str((synth_yolo_dir / "images" / "train").resolve()).replace("\\", "/"),
            str((real_subset_dir / "images" / "train").resolve()).replace("\\", "/"),
        ],
        # validate on REAL val only: model selection must track the real domain
        "val": real_yaml["val"],
        "kpt_shape": [NUM_KPT, 3],
        "names": {0: "car"},
        "flip_idx": real_yaml.get("flip_idx", list(range(NUM_KPT))),
    }
    out_path.write_text(yaml.safe_dump(mixed, sort_keys=False), encoding="utf-8")
    log(f"mixed yaml -> {out_path}")
    return out_path


def main() -> None:
    t0 = time.time()
    synth_v4_coco = SYNTH_V4_ROOT / "annotations" / "coco.json"
    for p, name in ((V1_CKPT, "v1 ckpt"), (synth_v4_coco, "synth v4 coco")):
        if not p.exists():
            log(f"FATAL: {name} missing at {p}")
            raise SystemExit(2)

    synth_yolo_v4 = WORK / "synth_yolo_v4"
    convert_synth_to_yolo(SYNTH_V4_ROOT, synth_yolo_v4)
    real_x8 = oversample_real(WORK / "real_subset", WORK / "real_subset_x8", factor=8)
    mixed_yaml = build_mixed_yaml(synth_yolo_v4, real_x8, WORK / "mixed_v4.yaml")

    log("=== v4 arm A: v1 ckpt + MIXED (synth_v4 + 100 real x8) fine-tune ===")
    a_best = _yolo_train(
        init_model=str(V1_CKPT),
        data_yaml=mixed_yaml,
        run_dir=RUN_DIR,
        name="v4_mixed_finetune",
        epochs=25,
        imgsz=480,
        batch=16,
        lr0=2e-4,
        patience=10,
    )

    log("=== v4 eval (full CarFusion test) ===")
    arm_a = run_eval(a_best, REPORTS / "phase0_v4_armA_metrics.json")

    v1 = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
    arm_b = json.loads((REPORTS / "phase0_v2_armB_metrics.json").read_text(encoding="utf-8"))

    v1_oks = v1["oks_map"]
    a_oks = arm_a["oks_map"]
    b_oks = arm_b["oks_map"]
    threshold = v1_oks + 0.02
    if a_oks >= threshold:
        verdict = f"PASS (arm A {a_oks:.4f} >= v1+2pp {threshold:.4f})"
    elif a_oks >= v1_oks:
        verdict = f"MARGINAL (arm A {a_oks:.4f} in [v1, v1+2pp))"
    else:
        verdict = f"FAIL (arm A {a_oks:.4f} < v1 {v1_oks:.4f})"

    def row(label: str, m: dict) -> str:
        return (
            f"| {label} | {m.get('oks_map', 0.0):.4f} | {m.get('oks_map_50', 0.0):.4f} "
            f"| {m.get('pck_0.05', 0.0):.4f} | {(m.get('oks_map', 0.0) - v1_oks) * 100:+.2f}pp |"
        )

    title = (
        "# Phase 0 Kill-Switch Report (v4: mixed training, wide multi-venue MRQ synth v4, real x8)"
    )
    report = f"""{title}

**Date:** {date.today().isoformat()}
**Design:** v1 checkpoint fine-tuned on ONE mixed dataset (synth v4 wide multi-venue MRQ
frames + the same 100 real frames oversampled x8, seed 42), val on real CarFusion val.
Control (arm B) reused from v2: v1 + 100 real, no synth.
**Kill switch:** arm A OKS-mAP >= v1 + 2pp ({threshold:.4f}).

| Run | OKS-mAP | OKS-mAP50 | PCK@0.05 | delta vs v1 |
|---|---|---|---|---|
{row("v1 baseline (full real train)", v1)}
{row("arm B control (v1 + 100 real, no synth)", arm_b)}
{row("arm A v4 (v1 + mixed synth_v4+100real x8)", arm_a)}

**Synth contribution (arm A - arm B): {(a_oks - b_oks) * 100:+.2f}pp OKS-mAP**

## Verdict

**{verdict}**
"""
    out = REPO_ROOT / "docs" / "phase0" / "kill_switch_report_v4.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    log(f"v4 report -> {out}")
    log(f"VERDICT v4: {verdict}")
    log(f"Phase 0 v4 complete in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
