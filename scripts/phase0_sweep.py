"""Size-sweep driver: train s/m/l YOLO26-pose detectors as matched pairs
(CarFusion-only vs CarFusion+synthetic) and pick the strongest. Each training is
an isolated subprocess (phase0_sweep_job.py); the driver schedules them under a
VRAM budget so small models share the RTX 3080 while large ones run alone.

  uv run python scripts/phase0_sweep.py

Honest hardware note: a 10 GB card cannot hold two medium/large pose trainings at
once, so those serialise; only the small pair truly overlaps. The scheduler reads
nvidia-smi free memory and launches the next job when it fits.
"""

# The report builder emits wide Markdown table rows inside f-strings; MD has no
# column limit here, so the long string literals are intentional.
# ruff: noqa: E501

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

REPO = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO / "scripts"))
from phase0_train import convert_synth_to_yolo, log  # noqa: E402
from phase0_train_v4 import SYNTH_V4_ROOT  # noqa: E402

WORK = REPO / "artifacts" / "phase0_work"
SYNTH_YOLO = WORK / "synth_yolo_sweep"
REPORTS = REPO / "reports"
LOGS = REPO / "logs"

# (size, arm) jobs, ordered cheap->expensive. Nano is included so n/s/m/l share one
# recipe (the consistent retrained baseline, replacing the legacy flipfix reference).
JOBS = [
    ("n", "real"),
    ("n", "synth"),
    ("s", "real"),
    ("s", "synth"),
    ("m", "real"),
    ("m", "synth"),
    ("l", "real"),
    ("l", "synth"),
    # synthetic-only arms (trained on synth alone, evaluated on real CarFusion) for
    # the "Synthetic only" HF collection - the true sim2real transfer number.
    ("n", "synthonly"),
    ("s", "synthonly"),
    ("m", "synthonly"),
    ("l", "synthonly"),
]
# MiB estimates for the launch gate. Capped below total VRAM (10240) + HEADROOM so
# the largest job can actually pass `free - EST >= HEADROOM`; with MAX_CONC=1 only one
# job runs at a time, so a slight under-estimate is harmless (it still fits the card).
EST_VRAM = {"n": 2500, "s": 4500, "m": 7500, "l": 8000}
HEADROOM = 1000  # MiB kept free
# One job at a time: medium/large fill the 10 GB card on their own, and running two
# data loaders at once exhausted system RAM (cv2 OutOfMemoryError). Serial is safe.
MAX_CONC = 1
GRACE_S = 45  # let a launched job claim its VRAM before the next launch


def gpu_free_mib() -> int:
    out = (
        subprocess.run(  # nosec B607 - nvidia-smi resolved from PATH by design
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .splitlines()[0]
    )
    return int(out)


def run() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    # Build the synth->YOLO conversion ONCE up front so parallel synth jobs don't race.
    if not (SYNTH_YOLO / "images" / "train").is_dir():
        log("pre-building shared synth->YOLO conversion ...")
        convert_synth_to_yolo(SYNTH_V4_ROOT, SYNTH_YOLO)

    # Idempotent: skip (size, arm) jobs that already have a metrics file, so a
    # relaunch after a crash only runs what is missing.
    queue = [(s, a) for s, a in JOBS if not (REPORTS / f"sweep_{s}_{a}_metrics.json").exists()]
    log(f"queue (missing jobs): {[f'{s}_{a}' for s, a in queue]}")
    running: dict[str, subprocess.Popen] = {}
    last_launch = 0.0
    while queue or running:
        for name, p in list(running.items()):
            if p.poll() is not None:
                log(f"job {name} exited rc={p.returncode}")
                del running[name]
        if queue and len(running) < MAX_CONC:
            size, arm = queue[0]
            free = gpu_free_mib()
            grace_ok = (not running) or (time.time() - last_launch > GRACE_S)
            if free - EST_VRAM[size] >= HEADROOM and grace_ok:
                queue.pop(0)
                name = f"{size}_{arm}"
                lf = open(LOGS / f"sweep_{name}.log", "w", encoding="utf-8")  # noqa: SIM115
                p = subprocess.Popen(
                    [sys.executable, "scripts/phase0_sweep_job.py", "--size", size, "--arm", arm],
                    cwd=str(REPO),
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                )
                running[name] = p
                last_launch = time.time()
                log(f"launched {name} (free was {free} MiB, running={list(running)})")
        time.sleep(10)

    log("all sweep jobs finished; building report")
    _report()


def _m(size: str, arm: str) -> dict:
    p = REPORTS / f"sweep_{size}_{arm}_metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _report() -> None:
    rows = []
    best_name, best_oks = None, -1.0
    for size in ("n", "s", "m", "l"):
        real, synth = _m(size, "real"), _m(size, "synth")
        r_oks, s_oks = real.get("oks_map", 0.0), synth.get("oks_map", 0.0)
        contrib = (s_oks - r_oks) * 100
        rows.append(
            f"| {size} | {r_oks:.4f} | {real.get('oks_map_50', 0):.4f} | {real.get('pck_0.05', 0):.4f} "
            f"| {s_oks:.4f} | {synth.get('oks_map_50', 0):.4f} | {synth.get('pck_0.05', 0):.4f} "
            f"| {contrib:+.2f}pp |"
        )
        for nm, oks in ((f"{size}_real", r_oks), (f"{size}_synth", s_oks)):
            if oks > best_oks:
                best_name, best_oks = nm, oks

    base = _m_baseline()
    report = f"""# Size-sweep matched ablation (CarFusion vs CarFusion+synthetic)

**Date:** {date.today().isoformat()}
**Init:** COCO-pretrained yolo26{{n,s,m,l}}-pose; both arms per size trained identically
(30 ep, 480px) on the full CarFusion train; only the synthetic data differs.
Eval on the real CarFusion val set ({3474} images).

Reference: nano production baseline (flipfix, full real) OKS-mAP {base:.4f}.

| size | real OKS-mAP | real mAP50 | real PCK | +synth OKS-mAP | +synth mAP50 | +synth PCK | synth contribution |
|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

**Strongest model: {best_name} (OKS-mAP {best_oks:.4f}).**
"""
    out = REPO / "docs" / "phase0" / "size_sweep_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    log(f"sweep report -> {out}")
    log(f"STRONGEST {best_name} OKS-mAP {best_oks:.4f}")


def _m_baseline() -> float:
    p = REPORTS / "baseline_flipfix_metrics.json"
    return json.loads(p.read_text(encoding="utf-8")).get("oks_map", 0.0) if p.exists() else 0.0


if __name__ == "__main__":
    run()
