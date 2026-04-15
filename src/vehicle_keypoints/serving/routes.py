"""FastAPI routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, UploadFile

from .. import __version__
from ..inference.predict import predict
from .dependencies import get_model
from .errors import InferenceError
from .schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        get_model()
        loaded = True
    except Exception:
        loaded = False
    return HealthResponse(
        status="ok" if loaded else "degraded", model_loaded=loaded, version=__version__,
    )


@router.post("/predict")
async def predict_endpoint(file: UploadFile, model=Depends(get_model)) -> dict:
    import tempfile

    suffix = "." + (file.filename or "input.bin").split(".")[-1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        return predict(model, tmp_path)
    except Exception as exc:
        raise InferenceError(str(exc)) from exc
