"""Train + validate a 24-point pose model FULLY on synthetic data.

Decouples dataset label quality from sim-to-real transfer: if a model trained
only on the synthetic frames scores well on held-out synthetic frames, the
24-point labels are clean and learnable - independent of the CarFusion gate.

  uv run python scripts/synth_24pt_experiment.py
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import yaml

REPO = Path(__file__).parent.parent.resolve()
SYNTH_ROOT = REPO.parent / "ue5-vehicle-synth" / "captures" / "phase0_v4"
SYNTH_COCO = SYNTH_ROOT / "annotations" / "coco.json"
WORK = REPO / "artifacts" / "phase0_work" / "synth_yolo_24pt"
RUN_DIR = REPO / "artifacts" / "synth24_runs"
NKPT = 24


def _row(ann: dict, w: int, h: int) -> str:
    x, y, bw, bh = ann["bbox"]
    cx = min(max((x + bw / 2) / w, 0.0), 1.0)
    cy = min(max((y + bh / 2) / h, 0.0), 1.0)
    parts = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{min(bw / w, 1.0):.6f}", f"{min(bh / h, 1.0):.6f}"]
    kp = ann["keypoints"]
    for k in range(NKPT):
        kx, ky, v = kp[k * 3 : k * 3 + 3]
        vi = int(v)
        if vi <= 0:
            parts += ["0.000000", "0.000000", "0"]
        else:
            parts += [
                f"{min(max(kx / w, 0.0), 1.0):.6f}",
                f"{min(max(ky / h, 0.0), 1.0):.6f}",
                str(vi),
            ]
    return " ".join(parts)


def convert() -> Path:
    data = json.loads(SYNTH_COCO.read_text(encoding="utf-8"))
    imgs = {i["id"]: i for i in data["images"]}
    anns: dict[int, list[dict]] = {}
    for a in data["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    ids = sorted(imgs)
    random.Random(42).shuffle(ids)
    n_val = max(1, round(len(ids) * 0.1))
    split = {"val": set(ids[:n_val]), "train": set(ids[n_val:])}
    if WORK.exists():
        shutil.rmtree(WORK)
    root = SYNTH_ROOT
    for sp, sids in split.items():
        (WORK / "images" / sp).mkdir(parents=True, exist_ok=True)
        (WORK / "labels" / sp).mkdir(parents=True, exist_ok=True)
        for iid in sids:
            im = imgs[iid]
            src = root / im["file_name"]
            if not src.is_file():
                continue
            shutil.copy2(src, WORK / "images" / sp / src.name)
            rows = [_row(a, im["width"], im["height"]) for a in anns.get(iid, [])]
            (WORK / "labels" / sp / (src.stem + ".txt")).write_text(
                "\n".join(rows) + "\n", encoding="utf-8"
            )
    yml = WORK / "synth24.yaml"
    yml.write_text(
        yaml.safe_dump(
            {
                "path": str(WORK).replace("\\", "/"),
                "train": "images/train",
                "val": "images/val",
                "kpt_shape": [NKPT, 3],
                "names": {0: "car"},
                "flip_idx": list(range(NKPT)),  # identity; hflip disabled below
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(f"converted: {len(split['train'])} train / {len(split['val'])} val (24-pt)")
    return yml


def main() -> None:
    from ultralytics import YOLO

    yml = convert()
    model = YOLO(str(REPO / "yolo26n-pose.pt"))
    model.train(
        data=str(yml),
        epochs=100,
        imgsz=480,
        batch=16,
        patience=30,
        fliplr=0.0,  # no horizontal flip -> no need for a correct L/R flip_idx
        project=str(RUN_DIR),
        name="synth24",
        exist_ok=True,
        workers=0,
        verbose=False,
    )
    r = model.val(data=str(yml), split="val", imgsz=480, workers=0, verbose=False)
    print(
        f"SYNTH24-VAL  box mAP50 {r.box.map50:.4f}  mAP50-95 {r.box.map:.4f} "
        f"| pose mAP50 {r.pose.map50:.4f}  mAP50-95 {r.pose.map:.4f}"
    )
    print("best:", RUN_DIR / "synth24" / "weights" / "best.pt")


if __name__ == "__main__":
    main()
