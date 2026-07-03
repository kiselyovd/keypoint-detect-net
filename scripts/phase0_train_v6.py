"""Phase 0 v6: matched two-arm ablation initialized from the STRONG production
checkpoint (flipfix, ~0.50 OKS-mAP) instead of the old 0.22 v1.

Motivation: the honest question "does synthetic data help a CarFusion detector?"
is most meaningful against the strongest available real-trained detector. We
therefore repeat the v5 matched ablation but initialize both arms from the
production flipfix checkpoint. Recipe is identical to v5 (corrected flip_idx,
25 epochs, 480px, batch 16, lr0 2e-4, seed implied, 100 real x8); the only
difference between arms is the synthetic data.

  arm A : flipfix ckpt -> mixed(synth_v4 + 100 real x8) fine-tune
  arm B : flipfix ckpt -> (100 real x8) fine-tune, NO synth   [control]

Synth contribution = arm A - arm B (both from the same strong start).
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from phase0_train import (  # noqa: E402
    CARFUSION_FLIP_IDX,
    NUM_KPT,
    _yolo_train,
    convert_synth_to_yolo,
    log,
    run_eval,
)
from phase0_train_v4 import SYNTH_V4_ROOT, oversample_real  # noqa: E402

# The STRONG production checkpoint (flipfix retrain, ~0.50 OKS-mAP). Verified
# provenance: produced 2026-06-17 alongside reports/baseline_flipfix_metrics.json.
FLIPFIX_CKPT = (
    REPO_ROOT / "artifacts" / "baseline_flipfix" / "sota_flipfix3" / "weights" / "best.pt"
)

WORK = REPO_ROOT / "artifacts" / "phase0_work"
RUN_DIR = REPO_ROOT / "artifacts" / "phase0_v6_runs"
REPORTS = REPO_ROOT / "reports"
REAL_VAL = str(REPO_ROOT / "data" / "processed" / "images" / "val").replace("\\", "/")


def _arm_yaml(train_dirs: list[Path], out_path: Path) -> Path:
    cfg = {
        "train": [str((d / "images" / "train").resolve()).replace("\\", "/") for d in train_dirs],
        "val": REAL_VAL,
        "kpt_shape": [NUM_KPT, 3],
        "names": {0: "car"},
        "flip_idx": CARFUSION_FLIP_IDX,
    }
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    log(f"arm yaml -> {out_path} (train={[str(d.name) for d in train_dirs]})")
    return out_path


def _train_eval(name: str, train_dirs: list[Path], metrics_path: Path) -> dict:
    yml = _arm_yaml(train_dirs, WORK / f"{name}.yaml")
    best = _yolo_train(
        init_model=str(FLIPFIX_CKPT),
        data_yaml=yml,
        run_dir=RUN_DIR,
        name=name,
        epochs=25,
        imgsz=480,
        batch=16,
        lr0=2e-4,
        patience=10,
    )
    return run_eval(best, metrics_path)


def main() -> None:
    t0 = time.time()
    synth_coco = SYNTH_V4_ROOT / "annotations" / "coco.json"
    for p, n in ((FLIPFIX_CKPT, "flipfix ckpt"), (synth_coco, "synth v4 coco")):
        if not p.exists():
            log(f"FATAL: {n} missing at {p}")
            raise SystemExit(2)

    synth_yolo = WORK / "synth_yolo_v6"
    convert_synth_to_yolo(SYNTH_V4_ROOT, synth_yolo)
    real_x8 = oversample_real(WORK / "real_subset", WORK / "real_subset_x8_v6", factor=8)

    log("=== v6 arm B: flipfix ckpt + 100 real x8 (no synth) ===")
    arm_b = _train_eval("v6_control_realx8", [real_x8], REPORTS / "phase0_v6_armB_metrics.json")
    log("=== v6 arm A: flipfix ckpt + MIXED (synth_v4 + 100 real x8) ===")
    arm_a = _train_eval(
        "v6_mixed_finetune", [synth_yolo, real_x8], REPORTS / "phase0_v6_armA_metrics.json"
    )

    base = json.loads((REPORTS / "baseline_flipfix_metrics.json").read_text(encoding="utf-8"))
    b_oks, a_oks, base_oks = arm_b["oks_map"], arm_a["oks_map"], base["oks_map"]

    def row(label: str, m: dict) -> str:
        return (
            f"| {label} | {m.get('oks_map', 0.0):.4f} | {m.get('oks_map_50', 0.0):.4f} "
            f"| {m.get('pck_0.05', 0.0):.4f} | {(m.get('oks_map', 0.0) - base_oks) * 100:+.2f}pp |"
        )

    report = f"""# Phase 0 Ablation v6 (init from the STRONG production checkpoint)

**Date:** {date.today().isoformat()}
**Init:** flipfix production checkpoint (OKS-mAP {base_oks:.4f}). Both arms
fine-tuned identically (corrected flip_idx, 25 ep, 480px, batch 16, lr0 2e-4,
100 real x8); only the synthetic data differs.

| Run | OKS-mAP | OKS-mAP50 | PCK@0.05 | delta vs prod baseline |
|---|---|---|---|---|
{row("production baseline (full real train)", base)}
{row("arm B (prod + 100 real x8, no synth)", arm_b)}
{row("arm A (prod + synth_v4 + 100 real x8)", arm_a)}

**Synth contribution (arm A - arm B): {(a_oks - b_oks) * 100:+.2f}pp OKS-mAP**
"""
    out = REPO_ROOT / "docs" / "phase0" / "ablation_report_v6.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    log(f"v6 report -> {out}")
    log(f"Synth contribution: {(a_oks - b_oks) * 100:+.2f}pp")
    log(f"Phase 0 v6 complete in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
