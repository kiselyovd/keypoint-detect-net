"""Exception types and handlers."""
from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse


class ModelNotLoadedError(RuntimeError):
    pass


class InferenceError(RuntimeError):
    pass


async def inference_error_handler(request: Request, exc: InferenceError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "error": "inference_failed",
            "detail": str(exc),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "error": "model_not_ready",
            "detail": str(exc),
            "request_id": getattr(request.state, "request_id", None),
        },
    )
