"""Pydantic schemas for the /detect endpoint."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Keypoint(BaseModel):
    x: float
    y: float
    v: float = Field(description="Visibility: 0=absent, 1=occluded, 2=visible")


class Detection(BaseModel):
    bbox: list[float] = Field(description="[x, y, w, h] in image pixels")
    keypoints: list[Keypoint]
    score: float


class DetectionResponse(BaseModel):
    detections: list[Detection]
    image_width: int
    image_height: int
    request_id: str
