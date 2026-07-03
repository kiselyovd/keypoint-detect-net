"""One size-sweep training job: train a YOLO26{s,m,l}-pose detector on either the
full CarFusion real train (arm=real) or CarFusion + UE5 synthetic (arm=synth),
both initialised from the COCO-pretrained checkpoint of that size, then evaluate
on the real CarFusion val set. Written to be launched as an isolated subprocess
by phase0_sweep.py so several sizes can share the GPU under a VRAM budget.

  uv run python scripts/phase0_sweep_job.py --size m --arm synth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO / "scripts"))

from phase0_train import (  # noqa: E402
    CARFUSION_FLIP_IDX,
    NUM_KPT,
    _yolo_train,
    convert_synth_to_yolo,
    log,
    run_eval,
)
from phase0_train_v4 import SYNTH_V4_ROOT  # noqa: E402

RUN_DIR = REPO / "artifacts" / "sweep_runs"
REPORTS = REPO / "reports"
WORK = REPO / "artifacts" / "phase0_work"
SYNTH_YOLO = WORK / "synth_yolo_sweep"  # shared synth->YOLO conversion (built once)

REAL_TRAIN = (REPO / "data" / "processed" / "images" / "train").resolve()
REAL_VAL = (REPO / "data" / "processed" / "images" / "val").resolve()

# Per-size training config. Batch is tuned so each job fits the RTX 3080 (10 GB);
# the larger models leave less room for parallel jobs (the scheduler handles that).
SIZE_CFG = {
    "n": {"batch": 16, "epochs": 30},  # consistent-recipe nano (replaces legacy flipfix)
    "s": {"batch": 16, "epochs": 30},
    "m": {"batch": 16, "epochs": 30},
    "l": {"batch": 10, "epochs": 30},
}


def _synth_combined_yaml(out_path: Path) -> Path:
    """data.yaml whose train is [full real train, synth train] and val is real val."""
    if not (SYNTH_YOLO / "images" / "train").is_dir():
        convert_synth_to_yolo(SYNTH_V4_ROOT, SYNTH_YOLO)
    cfg = {
        "train": [
            str(REAL_TRAIN).replace("\\", "/"),
            str((SYNTH_YOLO / "images" / "train").resolve()).replace("\\", "/"),
        ],
        "val": str(REAL_VAL).replace("\\", "/"),
        "kpt_shape": [NUM_KPT, 3],
        "names": {0: "car"},
        "flip_idx": CARFUSION_FLIP_IDX,
    }
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out_path


def _synth_only_yaml(out_path: Path) -> Path:
    """data.yaml whose train is ONLY the synthetic frames; val stays the real
    CarFusion val so the card reports true sim2real transfer (synthetic -> real)."""
    if not (SYNTH_YOLO / "images" / "train").is_dir():
        convert_synth_to_yolo(SYNTH_V4_ROOT, SYNTH_YOLO)
    cfg = {
        "train": str((SYNTH_YOLO / "images" / "train").resolve()).replace("\\", "/"),
        "val": str(REAL_VAL).replace("\\", "/"),
        "kpt_shape": [NUM_KPT, 3],
        "names": {0: "car"},
        "flip_idx": CARFUSION_FLIP_IDX,
    }
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=["n", "s", "m", "l"])
    ap.add_argument("--arm", required=True, choices=["real", "synth", "synthonly"])
    ap.add_argument("--imgsz", type=int, default=480)
    args = ap.parse_args()

    cfg = SIZE_CFG[args.size]
    name = f"{args.size}_{args.arm}"
    WORK.mkdir(parents=True, exist_ok=True)

    if args.arm == "synth":
        data_yaml = _synth_combined_yaml(WORK / f"sweep_{name}.yaml")
    elif args.arm == "synthonly":
        data_yaml = _synth_only_yaml(WORK / f"sweep_{name}.yaml")
    else:
        data_yaml = REPO / "data" / "processed" / "data.yaml"  # full real train + real val

    log(
        f"=== sweep {name}: yolo26{args.size}-pose + {args.arm} "
        f"({cfg['epochs']} ep, {args.imgsz}px, batch {cfg['batch']}) ==="
    )
    best = _yolo_train(
        init_model=f"yolo26{args.size}-pose.pt",  # ultralytics auto-downloads if absent
        data_yaml=data_yaml,
        run_dir=RUN_DIR,
        name=name,
        epochs=cfg["epochs"],
        imgsz=args.imgsz,
        batch=cfg["batch"],
        lr0=1e-3,  # ignored under optimizer=auto, kept for the wrapper signature
        patience=10,
        # 4-thread loader (cache stays off): enough to stop the single-thread loader
        # starving the GPU, but few enough that one job's workers don't exhaust system
        # RAM (8 workers x 2 parallel jobs hit a cv2 OutOfMemoryError). The driver runs
        # m/l strictly one at a time (MAX_CONC=1), so 4 workers is plenty.
        workers=4,
    )
    REPORTS.mkdir(parents=True, exist_ok=True)
    m = run_eval(best, REPORTS / f"sweep_{name}_metrics.json")
    log(
        f"SWEEP_DONE {name} OKS-mAP {m['oks_map']:.4f} mAP50 {m.get('oks_map_50', 0):.4f} "
        f"PCK {m.get('pck_0.05', 0):.4f} -> {best}"
    )


if __name__ == "__main__":
    main()
