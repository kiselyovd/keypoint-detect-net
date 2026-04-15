"""Inference CLI — load a checkpoint and predict on input(s)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..utils import configure_logging, get_logger

log = get_logger(__name__)


def load_model(checkpoint_path: str | Path):
    """Load a Lightning module from checkpoint, rebuilding the backbone from hparams."""
    from ultralytics import YOLO
    return YOLO(str(checkpoint_path))


def predict(model, input_path: str | Path):
    """Run a single prediction. Returns a task-specific result dict."""
    results = model(str(input_path))
    return [r.tojson() for r in results]
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    configure_logging()
    model = load_model(args.checkpoint)
    result = predict(model, args.input)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
