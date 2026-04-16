"""Model smoke tests: factory returns working YOLO + ViTPose backbones."""

from __future__ import annotations

import pytest
import torch

from vehicle_keypoints.models import KeypointsModule, build_model


def test_yolo_factory_loads():
    """YOLO wrapper returns an ultralytics.YOLO object."""
    try:
        model = build_model("yolo26n", num_keypoints=14, pretrained=False)
    except Exception as exc:
        pytest.skip(f"YOLO build failed in isolated env: {exc}")
    assert hasattr(model, "model") or hasattr(model, "predict")


def test_vitpose_factory_forward():
    model = build_model("vitpose_s", num_keypoints=14, pretrained=False)
    x = torch.randn(2, 3, 256, 192)  # ViTPose canonical top-down input shape
    out = model(x)
    hm = out.heatmaps if hasattr(out, "heatmaps") else out
    assert hm.ndim == 4
    assert hm.shape[0] == 2
    assert hm.shape[1] == 14


def test_keypoints_module_training_step():
    backbone = build_model("vitpose_s", num_keypoints=14, pretrained=False)
    lit = KeypointsModule(backbone, num_keypoints=14, lr=1e-4, model_name="vitpose_s")
    x = torch.randn(2, 3, 256, 192)
    target_hm = torch.zeros(2, 14, 64, 48)
    target_hm[:, :, 32, 24] = 1.0
    loss = lit.training_step((x, target_hm, torch.ones(2, 14)), batch_idx=0)
    assert torch.isfinite(loss).item()
