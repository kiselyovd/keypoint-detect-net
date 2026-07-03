"""Phase 0 kill-switch v5: CORRECTED-augmentation two-arm ablation.

v4 (and earlier) trained with ultralytics' default fliplr=0.5 but an IDENTITY
flip_idx, so horizontal flips mirrored the image without swapping left/right
keypoints - corrupting ~half the augmented samples. v5 fixes flip_idx (proper
L/R swap) and re-runs BOTH arms fresh from the v1 checkpoint so the comparison
is clean (the only difference between arms is the synthetic data):

  arm A : v1 ckpt -> mixed(synth_v4 + 100 real x8) fine-tune
  arm B : v1 ckpt -> (100 real x8) fine-tune, NO synth   [fresh control]

Both validate on the real CarFusion val split and are evaluated with the same
module that produced the v1 baseline. Kill switch: arm A >= v1 + 2pp OKS-mAP.
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
from phase0_train_v4 import SYNTH_V4_ROOT, V1_CKPT, oversample_real  # noqa: E402

WORK = REPO_ROOT / "artifacts" / "phase0_work"
RUN_DIR = REPO_ROOT / "artifacts" / "phase0_v5_runs"
REPORTS = REPO_ROOT / "reports"
REAL_VAL = str(REPO_ROOT / "data" / "processed" / "images" / "val").replace("\\", "/")


def _arm_yaml(train_dirs: list[Path], out_path: Path) -> Path:
    cfg = {
        "train": [str((d / "images" / "train").resolve()).replace("\\", "/") for d in train_dirs],
        "val": REAL_VAL,
        "kpt_shape": [NUM_KPT, 3],
        "names": {0: "car"},
        "flip_idx": CARFUSION_FLIP_IDX,  # the fix
    }
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    log(f"arm yaml -> {out_path} (train={[str(d.name) for d in train_dirs]})")
    return out_path


def _train_eval(name: str, train_dirs: list[Path], metrics_path: Path) -> dict:
    yml = _arm_yaml(train_dirs, WORK / f"{name}.yaml")
    best = _yolo_train(
        init_model=str(V1_CKPT),
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
    for p, n in ((V1_CKPT, "v1 ckpt"), (synth_coco, "synth v4 coco")):
        if not p.exists():
            log(f"FATAL: {n} missing at {p}")
            raise SystemExit(2)

    synth_yolo = WORK / "synth_yolo_v5"
    convert_synth_to_yolo(SYNTH_V4_ROOT, synth_yolo)
    real_x8 = oversample_real(WORK / "real_subset", WORK / "real_subset_x8_v5", factor=8)

    log("=== v5 arm B: v1 ckpt + 100 real x8 (no synth), corrected flip ===")
    arm_b = _train_eval("v5_control_realx8", [real_x8], REPORTS / "phase0_v5_armB_metrics.json")
    log("=== v5 arm A: v1 ckpt + MIXED (synth_v4 + 100 real x8), corrected flip ===")
    arm_a = _train_eval(
        "v5_mixed_finetune", [synth_yolo, real_x8], REPORTS / "phase0_v5_armA_metrics.json"
    )

    v1 = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
    v1_oks, a_oks, b_oks = v1["oks_map"], arm_a["oks_map"], arm_b["oks_map"]
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

    report = f"""# Phase 0 Kill-Switch Report (v5: corrected flip-aug, fresh two-arm ablation)

**Date:** {date.today().isoformat()}
**Fix:** horizontal-flip augmentation now uses the correct L/R `flip_idx`
({CARFUSION_FLIP_IDX}); earlier runs mirrored images with an identity flip_idx,
corrupting keypoints on ~half the augmented samples. Both arms re-run fresh from
the v1 checkpoint with identical settings; the only difference is the synthetic data.
**Kill switch:** arm A OKS-mAP >= v1 + 2pp ({threshold:.4f}).

| Run | OKS-mAP | OKS-mAP50 | PCK@0.05 | delta vs v1 |
|---|---|---|---|---|
{row("v1 baseline (full real train)", v1)}
{row("arm B (v1 + 100 real x8, no synth)", arm_b)}
{row("arm A (v1 + synth_v4 + 100 real x8)", arm_a)}

**Synth contribution (arm A - arm B): {(a_oks - b_oks) * 100:+.2f}pp OKS-mAP**

## Verdict

**{verdict}**
"""
    out = REPO_ROOT / "docs" / "phase0" / "kill_switch_report_v5.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    log(f"v5 report -> {out}")
    log(f"VERDICT v5: {verdict}")
    log(f"Phase 0 v5 complete in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
