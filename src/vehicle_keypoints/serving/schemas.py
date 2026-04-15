"""Pydantic request/response schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    version: str


class KeypointDetection(BaseModel):
    bbox: list[float]
    keypoints: list[list[float]]
    score: float


class DetectionResponse(BaseModel):
    detections: list[KeypointDetection]
