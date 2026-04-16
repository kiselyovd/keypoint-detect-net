"""FastAPI DI — lazy-loaded Detector singleton."""

from __future__ import annotations

import os
from functools import lru_cache

from ..inference.predict import Detector


@lru_cache(maxsize=1)
def get_detector() -> Detector:
    ckpt = os.getenv("MODEL_CHECKPOINT", "artifacts/checkpoints/best.pt")
    if os.path.isfile(ckpt):
        return Detector.from_checkpoint(ckpt)
    return Detector.from_pretrained_or_random("yolo26n")
