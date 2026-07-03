"""Phase 0: two-stage synth-pretrain -> real fine-tune for YOLO-pose.

Stage 1 : fine-tune yolo26n-pose.pt on UE5 synthetic data (864 frames, 14-pt).
Stage 2 : fine-tune stage-1 checkpoint on 100 real CarFusion train frames (low LR).
Control  : fine-tune yolo26n-pose.pt directly on same 100 real frames (no synth),
           to isolate the synth contribution.
Eval     : run evaluate.py on CarFusion test split; write reports/phase0_main_metrics.json.

Usage (PowerShell):
    $env:VK_SYNTH_PHASE0_DIR = "../ue5-vehicle-synth/captures/phase0"  # defaults to this if unset
    .venv/Scripts/python.exe -u scripts/phase0_train.py 2>&1 | Tee-Object logs/phase0.log

All intermediates go to artifacts/phase0/.  Checkpoints are gitignored.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# COCO -> YOLO conversion helpers
# ---------------------------------------------------------------------------

NUM_KPT = 14
# Correct left/right swap for horizontal-flip augmentation on the 14-pt CarFusion
# schema (wheels/lights/roof corners swap R<->L; exhaust + center map to themselves).
# Without this, ultralytics' default fliplr=0.5 mirrors the image but NOT the keypoint
# labels -> the "left wheel" lands on the right while still labelled left, corrupting
# ~half the augmented samples. (Bug found 2026-06-17; earlier runs used identity.)
CARFUSION_FLIP_IDX = [1, 0, 3, 2, 5, 4, 7, 6, 8, 10, 9, 12, 11, 13]


def _coco_to_yolo_row(ann: dict, img_w: int, img_h: int) -> str:
    """Convert one COCO annotation to a single YOLO-pose label line.

    Output: class cx cy bw bh kx0 ky0 v0 kx1 ky1 v1 ... (all normalised [0,1]).
    Uses only the first NUM_KPT keypoints from the COCO annotation.
    Visibility mapping: v<=0 -> 0, v==1 -> 1 (labelled not visible), v==2 -> 2.
    """
    x, y, w, h = ann["bbox"]
    x0 = max(0.0, float(x))
    y0 = max(0.0, float(y))
    x1 = min(float(img_w), float(x) + float(w))
    y1 = min(float(img_h), float(y) + float(h))
    bw_px = max(0.0, x1 - x0)
    bh_px = max(0.0, y1 - y0)
    cx = min(max((x0 + bw_px / 2) / img_w, 0.0), 1.0)
    cy = min(max((y0 + bh_px / 2) / img_h, 0.0), 1.0)
    bw = min(max(bw_px / img_w, 0.0), 1.0)
    bh = min(max(bh_px / img_h, 0.0), 1.0)
    parts: list[str] = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
    kpts = ann["keypoints"]  # flat list: x0 y0 v0 x1 y1 v1 ...
    for k in range(NUM_KPT):
        kx, ky, v = kpts[k * 3 : k * 3 + 3]
        v_int = int(v)
        if v_int <= 0:
            parts += ["0.000000", "0.000000", "0"]
        else:
            kx_n = min(max(float(kx) / img_w, 0.0), 1.0)
            ky_n = min(max(float(ky) / img_h, 0.0), 1.0)
            parts += [f"{kx_n:.6f}", f"{ky_n:.6f}", str(v_int)]
    return " ".join(parts)


def convert_synth_to_yolo(synth_root: Path, out_dir: Path) -> Path:
    """Convert synth COCO JSON (24-pt) to YOLO format using only first 14 pts.

    90/10 train/val split, fixed seed 42.
    Returns path to the generated data.yaml.
    """
    log("Converting synth COCO -> YOLO format ...")
    coco_path = synth_root / "annotations" / "coco.json"
    data = json.loads(coco_path.read_text(encoding="utf-8"))

    images_by_id = {img["id"]: img for img in data["images"]}
    # Trim keypoints to first 14 points before storing
    anns_by_img: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        flat_all = ann["keypoints"]
        ann_14 = dict(ann)
        ann_14["keypoints"] = flat_all[: NUM_KPT * 3]
        anns_by_img.setdefault(ann["image_id"], []).append(ann_14)

    all_img_ids = sorted(images_by_id.keys())
    rng = random.Random(42)
    rng.shuffle(all_img_ids)
    n_val = max(1, round(len(all_img_ids) * 0.1))
    val_ids = set(all_img_ids[:n_val])
    train_ids = set(all_img_ids[n_val:])
    log(f"  synth split: {len(train_ids)} train, {len(val_ids)} val")

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img_id in ids:
            img_meta = images_by_id[img_id]
            src = synth_root / img_meta["file_name"]
            if not src.is_file():
                log(f"  WARNING: missing {src}")
                continue
            dst = img_dir / src.name
            shutil.copy2(src, dst)
            anns = anns_by_img.get(img_id, [])
            rows = [_coco_to_yolo_row(a, img_meta["width"], img_meta["height"]) for a in anns]
            lbl = lbl_dir / (src.stem + ".txt")
            lbl.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    out_abs = str(out_dir.resolve()).replace("\\", "/")
    data_yaml = {
        "path": out_abs,
        "train": "images/train",
        "val": "images/val",
        "kpt_shape": [NUM_KPT, 3],
        "names": {0: "car"},
        "flip_idx": CARFUSION_FLIP_IDX,
    }
    data_yaml_path = out_dir / "synth_data.yaml"
    data_yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    log(f"  wrote {data_yaml_path}")
    return data_yaml_path


def build_real_subset_yaml(
    processed_dir: Path,
    out_dir: Path,
    n: int = 100,
    seed: int = 42,
) -> Path:
    """Build a YOLO data YAML pointing to a random n-image subset of the real train split.

    Val split reuses the full processed val (no data leak; test is separate).
    Returns path to the generated data.yaml.
    """
    log(f"Building real subset YAML (n={n}, seed={seed}) ...")
    train_img_dir = processed_dir / "images" / "train"
    train_lbl_dir = processed_dir / "labels" / "train"
    val_img_dir = processed_dir / "images" / "val"

    all_imgs = sorted(train_img_dir.glob("*.jpg")) + sorted(train_img_dir.glob("*.png"))
    rng = random.Random(seed)
    subset = rng.sample(all_imgs, min(n, len(all_imgs)))
    log(f"  sampled {len(subset)} / {len(all_imgs)} real train images (seed={seed})")

    sub_img_dir = out_dir / "images" / "train"
    sub_lbl_dir = out_dir / "labels" / "train"
    sub_img_dir.mkdir(parents=True, exist_ok=True)
    sub_lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_path in subset:
        dst_img = sub_img_dir / img_path.name
        lbl_src = train_lbl_dir / (img_path.stem + ".txt")
        if not img_path.is_file() or not lbl_src.is_file():
            log(f"  WARNING: skipping {img_path.name} (missing image or label)")
            continue
        # Use copy on Windows (symlinks require elevated perms)
        shutil.copy2(img_path, dst_img)
        shutil.copy2(lbl_src, sub_lbl_dir / lbl_src.name)

    # val points to the full processed val (absolute path OK for YOLO)
    data_yaml_path = out_dir / "real_subset_data.yaml"
    data_yaml = {
        "path": str(out_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": str(val_img_dir.resolve()).replace("\\", "/"),
        "kpt_shape": [NUM_KPT, 3],
        "names": {0: "car"},
        "flip_idx": CARFUSION_FLIP_IDX,
    }
    data_yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    log(f"  wrote {data_yaml_path}")
    return data_yaml_path


# ---------------------------------------------------------------------------
# Training stages
# ---------------------------------------------------------------------------


def _yolo_train(
    init_model: str,
    data_yaml: Path,
    run_dir: Path,
    name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    lr0: float,
    lrf: float = 0.01,
    freeze: int | None = None,
    patience: int | None = None,
    workers: int = 0,
    cache: str | bool = False,
) -> Path:
    """Generic YOLO-pose training wrapper. Returns best.pt path.

    workers/cache default to the original single-thread, no-cache behaviour (kept
    so concurrent runs don't race on the disk .npy cache). Callers that run one job
    at a time can pass workers>0 and cache='ram' to stop the GPU starving on the
    single-threaded data loader.
    """
    import ultralytics
    from ultralytics import YOLO

    run_dir.mkdir(parents=True, exist_ok=True)
    ultralytics.settings.update({"runs_dir": str(run_dir).replace("\\", "/")})

    model = YOLO(init_model)
    kwargs: dict = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "patience": patience if patience is not None else max(5, int(epochs * 0.4)),
        "project": str(run_dir).replace("\\", "/"),
        "name": name,
        "cache": cache,
        "verbose": True,
        "plots": True,
        "lr0": lr0,
        "lrf": lrf,
    }
    if freeze is not None:
        kwargs["freeze"] = freeze
    results = model.train(**kwargs)
    best = Path(str(results.save_dir)) / "weights" / "best.pt"
    log(f"  best.pt = {best}")
    return best


def stage1_synth_pretrain(
    synth_yaml: Path,
    run_dir: Path,
    epochs: int = 30,
    imgsz: int = 480,
    batch: int = 16,
) -> Path:
    """Train from yolo26n-pose.pt on synth data."""
    log(f"=== Stage 1: synth pretrain ({epochs} epochs) ===")
    return _yolo_train(
        init_model="yolo26n-pose.pt",
        data_yaml=synth_yaml,
        run_dir=run_dir,
        name="synth_pretrain",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=1e-3,
        patience=max(5, int(epochs * 0.3)),
    )


def stage2_real_finetune(
    init_ckpt: Path,
    real_yaml: Path,
    run_dir: Path,
    name: str = "real_finetune",
    epochs: int = 20,
    imgsz: int = 480,
    batch: int = 16,
) -> Path:
    """Fine-tune checkpoint on 100 real CarFusion frames."""
    log(f"=== Stage 2 ({name}): real fine-tune ({epochs} epochs) ===")
    return _yolo_train(
        init_model=str(init_ckpt),
        data_yaml=real_yaml,
        run_dir=run_dir,
        name=name,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=1e-4,  # 10x lower LR for fine-tuning
        freeze=10,  # freeze first 10 backbone layers
        patience=max(5, int(epochs * 0.4)),
    )


def control_real_finetune(
    real_yaml: Path,
    run_dir: Path,
    epochs: int = 20,
    imgsz: int = 480,
    batch: int = 16,
) -> Path:
    """Control: fine-tune base yolo26n-pose.pt on same 100 real frames (no synth warmup).

    Isolates the synth contribution: compare this to stage2 result.
    """
    log(f"=== Control: base -> real fine-tune ({epochs} epochs, no synth) ===")
    return _yolo_train(
        init_model="yolo26n-pose.pt",
        data_yaml=real_yaml,
        run_dir=run_dir,
        name="control_real_finetune",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=1e-4,
        freeze=10,
        patience=max(5, int(epochs * 0.4)),
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def run_eval(ckpt: Path, out_json: Path) -> dict:
    """Invoke vehicle_keypoints.evaluation.evaluate on the CarFusion test set.

    Uses the same GT file and image path as the v1 baseline to ensure
    methodology parity.
    """
    import subprocess

    out_json.parent.mkdir(parents=True, exist_ok=True)
    # Run from repo root so default paths (data/raw/..., data/processed/...) resolve correctly.
    repo_root = Path(__file__).parent.parent.resolve()
    venv_python = repo_root / ".venv" / "Scripts" / "python.exe"
    python_exe = str(venv_python) if venv_python.is_file() else sys.executable

    cmd = [
        python_exe,
        "-m",
        "vehicle_keypoints.evaluation.evaluate",
        "--checkpoint",
        str(ckpt),
        "--out",
        str(out_json),
    ]
    log(f"eval cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True, cwd=str(repo_root))
    _ = result  # success implicit (check=True raises on failure)
    metrics = json.loads(out_json.read_text(encoding="utf-8"))
    log(
        f"  OKS-mAP={metrics.get('oks_map', 0):.4f} "
        f" mAP50={metrics.get('oks_map_50', 0):.4f} "
        f" PCK@0.05={metrics.get('pck_0.05', 0):.4f}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Kill-switch report
# ---------------------------------------------------------------------------


def write_kill_switch_report(
    v1_metrics: dict,
    synth_only_metrics: dict | None,
    phase0_metrics: dict,
    control_metrics: dict | None,
    out_path: Path,
) -> None:
    from datetime import date

    v1_oks = v1_metrics.get("oks_map", 0.0)
    v1_map50 = v1_metrics.get("oks_map_50", 0.0)
    v1_pck = v1_metrics.get("pck_0.05", 0.0)

    p0_oks = phase0_metrics.get("oks_map", 0.0)
    p0_map50 = phase0_metrics.get("oks_map_50", 0.0)
    p0_pck = phase0_metrics.get("pck_0.05", 0.0)

    d_oks = p0_oks - v1_oks
    d_map50 = p0_map50 - v1_map50
    d_pck = p0_pck - v1_pck

    threshold_pass = 0.2399
    if p0_oks >= threshold_pass:
        verdict = f"PASS (OKS-mAP >= 0.2399: {d_oks * 100:+.2f}pp over v1)"
    elif p0_oks >= v1_oks:
        verdict = f"MARGINAL (0 <= delta < +2pp: {d_oks * 100:+.2f}pp over v1)"
    else:
        verdict = f"FAIL (regression: {d_oks * 100:+.2f}pp vs v1)"

    def _row(label: str, m: dict, baseline_oks: float) -> str:
        oks = m.get("oks_map", 0.0)
        m50 = m.get("oks_map_50", 0.0)
        pck = m.get("pck_0.05", 0.0)
        delta = f"{(oks - baseline_oks) * 100:+.2f}pp" if baseline_oks > 0 else "-"
        return (
            f"| {label} | {oks:.4f} ({oks * 100:.1f}%) "
            f"| {m50:.4f} ({m50 * 100:.1f}%) "
            f"| {pck:.4f} ({pck * 100:.1f}%) "
            f"| {delta} |"
        )

    rows = []
    rows.append(_row("v1 baseline (YOLO26n-pose, full train)", v1_metrics, 0.0))
    if synth_only_metrics is not None:
        rows.append(_row("Phase 0 stage 1 (synth-only, no real ft)", synth_only_metrics, v1_oks))
    if control_metrics is not None:
        rows.append(_row("Control (base -> 100 real, no synth)", control_metrics, v1_oks))
    rows.append(_row("Phase 0 final (synth pretrain -> 100 real ft)", phase0_metrics, v1_oks))

    table = "\n".join(rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = f"""# Phase 0 Kill-Switch Report

**Date:** {date.today().isoformat()}
**Experiment:** Synth-pretrain (864 UE5 frames, 14-pt) then real fine-tune (100 CarFusion frames)
**Kill-switch threshold:** OKS-mAP >= 0.2399 (v1 baseline + 2pp)

## Data

- Synth pretrain: 864 frames (UE5 City Sample, 14 canonical CarFusion keypoints)
- Real fine-tune / control: 100 CarFusion train frames (random seed 42 subset)
- Eval: 12,761 CarFusion test images (identical to v1 baseline run)

## Metrics

| Run | OKS-mAP | OKS-mAP50 | PCK@0.05 | Delta vs v1 |
|---|---|---|---|---|
{table}

## Delta summary (Phase 0 final vs v1)

- OKS-mAP: {d_oks * 100:+.2f}pp
- OKS-mAP50: {d_map50 * 100:+.2f}pp
- PCK@0.05: {d_pck * 100:+.2f}pp

## Verdict

**{verdict}**
"""
    out_path.write_text(report, encoding="utf-8")
    log(f"Kill-switch report written to {out_path}")
    log(f"VERDICT: {verdict}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.time()

    repo_root = Path(__file__).parent.parent.resolve()
    synth_dir = Path(
        os.environ.get(
            "VK_SYNTH_PHASE0_DIR",
            str(repo_root.parent / "ue5-vehicle-synth" / "captures" / "phase0"),
        )
    )
    if not synth_dir.is_dir():
        raise SystemExit(
            "Set VK_SYNTH_PHASE0_DIR to the synth phase0 root "
            "(contains annotations/coco.json and rgb/)."
        )

    artifacts = repo_root / "artifacts"
    processed_dir = repo_root / "data" / "processed"
    work_dir = artifacts / "phase0_work"
    run_dir = artifacts / "phase0_runs"
    work_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Ensure logs dir exists
    (repo_root / "logs").mkdir(exist_ok=True)

    # --- 1. Convert synth data to YOLO format ---
    synth_yolo_dir = work_dir / "synth_yolo"
    synth_yaml = convert_synth_to_yolo(synth_dir, synth_yolo_dir)

    # --- 2. Build real-100 subset YAML (seed 42) ---
    real_subset_dir = work_dir / "real_subset"
    real_yaml = build_real_subset_yaml(processed_dir, real_subset_dir, n=100, seed=42)

    # --- 3. Stage 1: synth pretrain ---
    synth_ckpt = stage1_synth_pretrain(synth_yaml, run_dir, epochs=30, imgsz=480, batch=16)

    synth_ckpt_copy = artifacts / "phase0" / "synth_pretrained.pt"
    synth_ckpt_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(synth_ckpt, synth_ckpt_copy)
    log(f"Stage 1 checkpoint saved to {synth_ckpt_copy}")

    # --- 4. Eval stage-1 (synth-only) for diagnostics ---
    log("=== Eval stage-1 (synth-only) checkpoint ===")
    synth_eval_out = repo_root / "reports" / "phase0_synth_metrics.json"
    try:
        synth_metrics = run_eval(synth_ckpt_copy, synth_eval_out)
    except Exception as exc:
        log(f"WARNING: synth-only eval failed: {exc}")
        synth_metrics = None

    # --- 5. Stage 2: real fine-tune (synth + real) ---
    final_ckpt = stage2_real_finetune(
        synth_ckpt, real_yaml, run_dir, epochs=20, imgsz=480, batch=16
    )
    final_ckpt_copy = artifacts / "phase0" / "main.pt"
    shutil.copy2(final_ckpt, final_ckpt_copy)
    log(f"Phase 0 final checkpoint saved to {final_ckpt_copy}")

    # --- 6. Eval final checkpoint (synth + real) ---
    log("=== Eval Phase 0 final checkpoint ===")
    phase0_eval_out = repo_root / "reports" / "phase0_main_metrics.json"
    phase0_metrics = run_eval(final_ckpt_copy, phase0_eval_out)

    # --- 7. Control: base -> 100 real (no synth) ---
    control_ckpt = control_real_finetune(real_yaml, run_dir, epochs=20, imgsz=480, batch=16)
    control_ckpt_copy = artifacts / "phase0" / "control.pt"
    shutil.copy2(control_ckpt, control_ckpt_copy)
    log(f"Control checkpoint saved to {control_ckpt_copy}")

    log("=== Eval control checkpoint ===")
    control_eval_out = repo_root / "reports" / "phase0_control_metrics.json"
    try:
        control_metrics = run_eval(control_ckpt_copy, control_eval_out)
    except Exception as exc:
        log(f"WARNING: control eval failed: {exc}")
        control_metrics = None

    # --- 8. Load v1 baseline metrics ---
    v1_json = repo_root / "reports" / "metrics.json"
    v1_metrics = json.loads(v1_json.read_text(encoding="utf-8"))

    # --- 9. Kill-switch report ---
    report_path = repo_root / "docs" / "phase0" / "kill_switch_report.md"
    write_kill_switch_report(
        v1_metrics, synth_metrics, phase0_metrics, control_metrics, report_path
    )

    elapsed = (time.time() - t0) / 60
    log(f"Phase 0 complete in {elapsed:.1f} min")
    log(f"Final checkpoint: {final_ckpt_copy}")
    log(f"Main metrics: {phase0_eval_out}")


if __name__ == "__main__":
    main()
