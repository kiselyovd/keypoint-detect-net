"""Data preparation stage."""
from __future__ import annotations

from pathlib import Path

from ..utils import get_logger

log = get_logger(__name__)


def prepare_data(raw_dir: Path | str, processed_dir: Path | str) -> None:
    """Transform raw data into processed form. Override per project."""
    raw = Path(raw_dir)
    out = Path(processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    log.info("prepare_data.start", raw=str(raw), out=str(out))
    log.info("prepare_data.done")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    prepare_data(args.raw, args.out)
