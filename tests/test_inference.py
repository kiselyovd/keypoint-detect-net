"""End-to-end inference smoke on data/sample/."""

from __future__ import annotations

from pathlib import Path

import pytest

from vehicle_keypoints.inference.overlay import draw_keypoints
from vehicle_keypoints.inference.predict import Detector


def test_detector_predict_on_sample() -> None:
    sample_img = next(Path("data/sample/images").glob("*.jpg"), None)
    if sample_img is None:
        pytest.skip("data/sample/images missing — run scripts/build_sample_data.py first")
    det = Detector.from_pretrained_or_random("yolo26n")
    detections = det.predict(str(sample_img))
    assert isinstance(detections, list)
    for d in detections:
        assert set(d.keys()) >= {"bbox", "keypoints", "score"}
        assert len(d["keypoints"]) == 14
        for kpt in d["keypoints"]:
            assert len(kpt) == 3


def test_draw_keypoints_on_sample(tmp_path: Path) -> None:
    sample_img = next(Path("data/sample/images").glob("*.jpg"), None)
    if sample_img is None:
        pytest.skip("data/sample/images missing — run scripts/build_sample_data.py first")
    detections = [
        {
            "bbox": [10, 10, 200, 200],
            "keypoints": [[20 + i * 10, 30 + i * 10, 2] for i in range(14)],
            "score": 0.95,
        }
    ]
    out = tmp_path / "overlay.png"
    draw_keypoints(str(sample_img), detections, out)
    assert out.is_file()
    assert out.stat().st_size > 0
