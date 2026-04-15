"""FastAPI smoke test for /health and /detect."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vehicle_keypoints.serving.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_detect_returns_json(client: TestClient) -> None:
    sample = next(Path("data/sample/images").glob("*.jpg"), None)
    if sample is None:
        pytest.skip("data/sample/images missing")
    with open(sample, "rb") as f:
        r = client.post("/detect", files={"file": (sample.name, f, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert "detections" in body
    assert set(body.keys()) >= {"detections", "image_width", "image_height", "request_id"}


def test_detect_returns_overlay_png(client: TestClient) -> None:
    sample = next(Path("data/sample/images").glob("*.jpg"), None)
    if sample is None:
        pytest.skip("data/sample/images missing")
    with open(sample, "rb") as f:
        r = client.post(
            "/detect", params={"overlay": "true"}, files={"file": (sample.name, f, "image/jpeg")}
        )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8].startswith(b"\x89PNG")
