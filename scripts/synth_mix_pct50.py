"""Practical test: does the pct50 instance-filtered synthetic data make a BETTER
additive than the unfiltered synth in the CarFusion+synth mix?

The size-sweep mix (s/m/l_synth) used ALL synth instances and gave +3.35 / +1.66 /
-2.33 pp OKS over the real-only arm - i.e. synth HURT the large model. The pct50
filter (drop each scene's small far half) doubled synth-only transfer; here we test
whether it also fixes the additive case, especially the large-model regression.

Train = [full real CarFusion train, pct50-filtered synth train]; val = real val.
Compare sweep_{size}_synthpct50mix vs sweep_{size}_synth (old mix) and _real.

  uv run python scripts/synth_mix_pct50.py --size l
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO / "scripts"))
from phase0_train import CARFUSION_FLIP_IDX, NUM_KPT, _yolo_train, log, run_eval  # noqa: E402

WORK = REPO / "artifacts" / "phase0_work"
RUN_DIR = REPO / "artifacts" / "sweep_runs"
REPORTS = REPO / "reports"
REAL_TRAIN = (REPO / "data" / "processed" / "images" / "train").resolve()
REAL_VAL = (REPO / "data" / "processed" / "images" / "val").resolve()
SYNTH_PCT50_TRAIN = (WORK / "synth_yolo_pct50" / "images" / "train").resolve()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=["n", "s", "m", "l"], required=True)
    args = ap.parse_args()
    if not SYNTH_PCT50_TRAIN.is_dir():
        raise SystemExit(f"missing pct50 synth dir {SYNTH_PCT50_TRAIN}; run synth_clean_experiment")

    cfg = {
        "train": [str(REAL_TRAIN).replace("\\", "/"), str(SYNTH_PCT50_TRAIN).replace("\\", "/")],
        "val": str(REAL_VAL).replace("\\", "/"),
        "kpt_shape": [NUM_KPT, 3],
        "names": {0: "car"},
        "flip_idx": CARFUSION_FLIP_IDX,
    }
    name = f"{args.size}_synthpct50mix"
    yml = WORK / f"{name}.yaml"
    yml.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    batch = 10 if args.size == "l" else 16
    log(f"=== {name}: yolo26{args.size}-pose + real + pct50-synth (30 ep, 480px) ===")
    best = _yolo_train(
        init_model=f"yolo26{args.size}-pose.pt",
        data_yaml=yml,
        run_dir=RUN_DIR,
        name=name,
        epochs=30,
        imgsz=480,
        batch=batch,
        lr0=1e-3,
        patience=10,
        workers=4,
    )
    m = run_eval(best, REPORTS / f"sweep_{name}_metrics.json")
    log(
        f"RESULT {name}: OKS {m['oks_map']:.4f} mAP50 {m.get('oks_map_50', 0):.4f} "
        f"PCK {m.get('pck_0.05', 0):.4f}"
    )


if __name__ == "__main__":
    main()
