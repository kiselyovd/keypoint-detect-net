"""Smoke test for CarFusion → COCO converter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vehicle_keypoints.scripts_lib.convert_carfusion import convert_scene_dir


def test_convert_scene_dir_smoke(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    scene_a = raw / "scene_a" / "gt"
    scene_a.mkdir(parents=True)

    # CarFusion .txt row format: x,y,kpt_id(1-14),instance_id,visibility(1|2|3)
    (scene_a / "1_100.txt").write_text(
        "\n".join(f"{10 + k * 5},{20 + k * 5},{k + 1},1,2" for k in range(14))
    )
    scene_img_dir = raw / "scene_a" / "images"
    scene_img_dir.mkdir(parents=True)
    (scene_img_dir / "1_100.jpg").write_bytes(b"fakejpg")

    out = tmp_path / "out.json"
    convert_scene_dir(raw_dir=raw, image_subdir="images", out_json=out)

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["categories"][0]["keypoints"] == [str(i) for i in range(14)]
    assert len(data["images"]) == 1
    assert len(data["annotations"]) == 1
    ann = data["annotations"][0]
    assert len(ann["keypoints"]) == 14 * 3
    assert ann["num_keypoints"] == 14


def test_convert_scene_dir_empty(tmp_path: Path) -> None:
    raw = tmp_path / "raw_empty"
    raw.mkdir()
    with pytest.raises(SystemExit, match="No scene dirs"):
        convert_scene_dir(raw_dir=raw, image_subdir="images", out_json=tmp_path / "out.json")


def test_convert_visibility_three_maps_to_visible(tmp_path: Path) -> None:
    """CarFusion vis=3 (occluded) is mapped to COCO vis=2 (visible).

    Divergent from the legacy script which treated vis=3 as invisible.
    """
    raw = tmp_path / "raw"
    scene = raw / "scene_v3" / "gt"
    scene.mkdir(parents=True)
    (raw / "scene_v3" / "images").mkdir(parents=True)
    # All 14 keypoints with vis=3 → expect every COCO keypoint visibility byte == 2
    (scene / "1_100.txt").write_text(
        "\n".join(f"{10 + k * 5},{20 + k * 5},{k + 1},1,3" for k in range(14))
    )
    out = tmp_path / "out.json"
    convert_scene_dir(raw_dir=raw, image_subdir="images", out_json=out)
    data = json.loads(out.read_text(encoding="utf-8"))
    ann = data["annotations"][0]
    coco_vis = ann["keypoints"][2::3]  # every third value is visibility
    assert coco_vis == [2] * 14, f"vis=3 should map to COCO vis=2, got {coco_vis}"
    assert ann["num_keypoints"] == 14
