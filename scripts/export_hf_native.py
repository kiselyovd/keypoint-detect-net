"""Export trained YOLO pose model to HF Hub layout.

Usage:
    python scripts/export_hf_native.py --checkpoint artifacts/best.pt --out artifacts/hf_export
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Export YOLO keypoints model to HF Hub layout.")
    p.add_argument("--checkpoint", default="artifacts/best.pt")
    p.add_argument("--out", default="artifacts/hf_export")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.checkpoint, out / "weights.pt")
    data_yaml = Path("data/processed/data.yaml")
    if data_yaml.is_file():
        shutil.copy2(data_yaml, out / "data.yaml")
    print(f"Saved YOLO weights + data.yaml to {out}")


if __name__ == "__main__":
    main()
