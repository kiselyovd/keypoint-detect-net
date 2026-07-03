"""Re-run inference for the v6 arm checkpoints and save COCO-format predictions,
so we can bootstrap the PCK delta (#1) and re-evaluate on a keypoint subset (#4)
without retraining. One-off analysis helper."""

from __future__ import annotations

import json
from pathlib import Path

from vehicle_keypoints.evaluation.evaluate import _predict_all
from vehicle_keypoints.inference.predict import Detector

REPO = Path(__file__).parent.parent
GT = REPO / "data" / "raw" / "annotations" / "car_keypoints_test.json"
IMAGES = REPO / "data" / "processed" / "images" / "test"
ARMS = {
    "armB": REPO / "artifacts" / "phase0_v6_runs" / "v6_control_realx8" / "weights" / "best.pt",
    "armA": REPO / "artifacts" / "phase0_v6_runs" / "v6_mixed_finetune" / "weights" / "best.pt",
}


def main() -> None:
    gt = json.loads(GT.read_text(encoding="utf-8"))
    for tag, ckpt in ARMS.items():
        print(f"[{tag}] loading {ckpt.name} ...", flush=True)
        det = Detector.from_checkpoint(str(ckpt))
        res = _predict_all(det, IMAGES, gt)
        out = REPO / "reports" / f"phase0_v6_{tag}_preds.json"
        out.write_text(json.dumps(res), encoding="utf-8")
        print(f"[{tag}] wrote {len(res)} predictions -> {out.name}", flush=True)
    print("DONE_DUMP_PREDS", flush=True)


if __name__ == "__main__":
    main()
