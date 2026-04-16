"""FastAPI routes for the pose detection service."""

from __future__ import annotations

import io
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from ..inference.overlay import draw_keypoints
from ..inference.predict import Detector
from .dependencies import get_detector
from .schemas import Detection, DetectionResponse, Keypoint

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/detect")
async def detect(
    request: Request,
    file: UploadFile = File(...),
    overlay: bool = False,
    detector: Detector = Depends(get_detector),
):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    payload = await file.read()
    pil = Image.open(io.BytesIO(payload)).convert("RGB")
    width, height = pil.size

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        pil.save(tmp.name, "JPEG")
        tmp_path = Path(tmp.name)
    try:
        raw_dets = detector.predict(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    dets = [
        Detection(
            bbox=[float(v) for v in d["bbox"]],
            keypoints=[Keypoint(x=k[0], y=k[1], v=k[2]) for k in d["keypoints"]],
            score=d["score"],
        )
        for d in raw_dets
    ]
    resp = DetectionResponse(
        detections=dets, image_width=width, image_height=height, request_id=request_id
    )
    if not overlay:
        return JSONResponse(resp.model_dump())

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_out:
        out_path = Path(tmp_out.name)
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_in:
            pil.save(tmp_in.name, "JPEG")
            draw_keypoints(tmp_in.name, raw_dets, out_path)
            overlay_bytes = out_path.read_bytes()
    finally:
        out_path.unlink(missing_ok=True)

    return Response(
        content=overlay_bytes,
        media_type="image/png",
        headers={
            "x-request-id": request_id,
            "x-detections": str(len(dets)),
        },
    )
