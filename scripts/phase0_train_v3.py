"""Phase 0 kill-switch experiment v3: MIXED synth+real training (no sequential stages).

v2 diagnosed the failure: 30 sequential synth-only epochs catastrophically
forget the real domain (arm A 0.0840 vs control 0.2204). The standard cure is
joint training: one dataset mixing all synth frames with the same 100 real
frames, fine-tuning the v1 checkpoint directly.

  arm A (v3): v1 ckpt -> mixed(synth_v2 + 100 real) fine-tune
  arm B     : v2's control result reused verbatim (v1 + 100 real, no synth)

Synth v2 dataset: 816 clean frames (cabin frames filtered), 9056 multi-vehicle
instances (parked City Sample cars labeled per type - kills the negative
supervision of run #1/#2 data). Eval = the same module that produced the v1
baseline. Kill switch: arm A >= v1 + 2pp OKS-mAP.
"""

from __future__ import annotations

import json
import os
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
RUN_DIR = REPO_ROOT / "artifacts" / "phase0_v3_runs"
REPORTS = REPO_ROOT / "reports"
WORK = REPO_ROOT / "artifacts" / "phase0_work"
SYNTH_V2_ROOT = Path(
    os.environ.get(
        "VK_SYNTH_PHASE0V2_DIR",
        str(REPO_ROOT.parent / "ue5-vehicle-synth" / "captures" / "phase0_v2"),
    )
)


def build_mixed_yaml(synth_yolo_dir: Path, real_subset_dir: Path, out_path: Path) -> Path:
    """One YOLO dataset whose train spans synth v2 + the 100-real subset."""
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
    for p, name in (
        (V1_CKPT, "v1 ckpt"),
        (SYNTH_V2_ROOT / "annotations" / "coco.json", "synth v2 coco"),
    ):
        if not p.exists():
            log(f"FATAL: {name} missing at {p}")
            raise SystemExit(2)

    synth_yolo_v2 = WORK / "synth_yolo_v2"
    convert_synth_to_yolo(SYNTH_V2_ROOT, synth_yolo_v2)
    mixed_yaml = build_mixed_yaml(synth_yolo_v2, WORK / "real_subset", WORK / "mixed_v3.yaml")

    log("=== v3 arm A: v1 ckpt + MIXED (synth_v2 + 100 real) fine-tune ===")
    a_best = _yolo_train(
        init_model=str(V1_CKPT),
        data_yaml=mixed_yaml,
        run_dir=RUN_DIR,
        name="v3_mixed_finetune",
        epochs=25,
        imgsz=480,
        batch=16,
        lr0=2e-4,
        patience=10,
    )

    log("=== v3 eval (full CarFusion test) ===")
    arm_a = run_eval(a_best, REPORTS / "phase0_v3_armA_metrics.json")

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

    report = f"""# Phase 0 Kill-Switch Report (v3: mixed training, multi-vehicle synth v2)

**Date:** {date.today().isoformat()}
**Design:** v1 checkpoint fine-tuned on ONE mixed dataset (816 synth v2 frames
with 9056 multi-vehicle instances + the same 100 real frames, seed 42), val on
real CarFusion val. Control (arm B) reused from v2: v1 + 100 real, no synth.
**Kill switch:** arm A OKS-mAP >= v1 + 2pp ({threshold:.4f}).

| Run | OKS-mAP | OKS-mAP50 | PCK@0.05 | delta vs v1 |
|---|---|---|---|---|
{row("v1 baseline (full real train)", v1)}
{row("arm B control (v1 + 100 real, no synth)", arm_b)}
{row("arm A v3 (v1 + mixed synth_v2+100real)", arm_a)}

**Synth contribution (arm A - arm B): {(a_oks - b_oks) * 100:+.2f}pp OKS-mAP**

## Verdict

**{verdict}**
"""
    out = REPO_ROOT / "docs" / "phase0" / "kill_switch_report_v3.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    log(f"v3 report -> {out}")
    log(f"VERDICT v3: {verdict}")
    log(f"Phase 0 v3 complete in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
