"""Export trained model to HuggingFace-native format (safetensors + config.json).

Run BEFORE publish_to_hf.py so the HF repo gets proper pipeline pills / Inference
Providers instead of just a raw Lightning .ckpt.

Usage:
    python scripts/export_hf_native.py \\
        --checkpoint artifacts/checkpoints/best.ckpt \\
        --out artifacts/hf_export \\
        --base-model <HF_BASE_MODEL_ID>
"""
from __future__ import annotations

import argparse
from pathlib import Path



def main() -> None:
    p = argparse.ArgumentParser(description="Export model to HF-native format.")
    p.add_argument("--checkpoint", default="artifacts/checkpoints/best.ckpt")
    p.add_argument("--out", default="artifacts/hf_export")
    p.add_argument("--base-model", default=None, help="HF base model ID for processor/tokenizer.")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from vehicle_keypoints.inference.predict import load_model

    lit = load_model(args.checkpoint)
    backbone = lit.model

    if not hasattr(backbone, "save_pretrained"):
        raise SystemExit(
            "Backbone is not transformers-compatible; cannot export natively."
        )

    backbone.save_pretrained(out)
    print(f"HF-native export complete: {out}")




if __name__ == "__main__":
    main()
