"""Resume the hung large pct50-mix run from its last checkpoint, eval it, then run
the medium and small pct50-mix arms fresh. The large run deadlocked at epoch 10
(dataloader starvation during heavy concurrent disk I/O); resume continues from the
epoch-9 checkpoint so the ~2h already spent isn't lost.

  uv run python scripts/resume_pct50mix.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import ultralytics
from ultralytics import YOLO

REPO = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO / "scripts"))
from phase0_train import log, run_eval  # noqa: E402

RUN_DIR = REPO / "artifacts" / "sweep_runs"
REPORTS = REPO / "reports"
L_RUN = RUN_DIR / "l_synthpct50mix2"  # the active (9-epoch) run that hung


def resume_large():
    last = L_RUN / "weights" / "last.pt"
    if not last.is_file():
        raise SystemExit(f"no last.pt at {last}")
    ultralytics.settings.update({"runs_dir": str(RUN_DIR).replace("\\", "/")})
    log(f"resuming large from {last} (epoch 9 -> 30)")
    model = YOLO(str(last))
    model.train(resume=True)
    best = L_RUN / "weights" / "best.pt"
    m = run_eval(best, REPORTS / "sweep_l_synthpct50mix_metrics.json")
    log(
        f"RESULT l_synthpct50mix: OKS {m['oks_map']:.4f} mAP50 {m.get('oks_map_50', 0):.4f} "
        f"PCK {m.get('pck_0.05', 0):.4f}"
    )


def main():
    if not (REPORTS / "sweep_l_synthpct50mix_metrics.json").exists():
        resume_large()
    for sz in ("m", "s"):
        if (REPORTS / f"sweep_{sz}_synthpct50mix_metrics.json").exists():
            log(f"{sz} already done, skipping")
            continue
        log(f"=== running {sz}_synthpct50mix fresh ===")
        subprocess.run(
            [sys.executable, "scripts/synth_mix_pct50.py", "--size", sz],
            cwd=str(REPO),
            check=True,
        )


if __name__ == "__main__":
    main()
