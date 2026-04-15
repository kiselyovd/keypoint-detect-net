"""Evaluation CLI — runs model on test set, writes reports/metrics.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..utils import configure_logging, get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="reports/metrics.json")
    args = parser.parse_args()
    configure_logging()
    log.info("evaluate.start", checkpoint=args.checkpoint, data=args.data)
    metrics: dict = {"note": "override evaluate.py per project"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(metrics, indent=2))
    log.info("evaluate.done", out=args.out)


if __name__ == "__main__":
    main()
