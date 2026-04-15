"""Dependency injection — singleton model loader."""
from __future__ import annotations

import os
from functools import lru_cache

from ..inference.predict import load_model
from ..utils import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_model():
    ckpt = os.environ.get("MODEL_PATH")
    if not ckpt:
        raise RuntimeError("MODEL_PATH env var not set")
    log.info("model.load", path=ckpt)
    return load_model(ckpt)
