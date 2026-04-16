"""Upload trained artifacts to HuggingFace Hub."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
from jinja2 import Environment, FileSystemLoader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_metrics(metrics: dict) -> str:
    if not metrics:
        return "TBD"
    scalar = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}
    if not scalar:
        return "TBD"
    rows = [f"| {k} | {v} |" for k, v in scalar.items()]
    return "| Metric | Value |\n|---|---|\n" + "\n".join(rows)


def _build_tags(domain_tag: str, extra_csv: str, library_name: str) -> list[str]:
    """Combine domain tag, library name, comma-separated extras into a sorted unique list."""
    tags: set[str] = set()
    if domain_tag:
        tags.add(domain_tag)
    if library_name:
        tags.add(library_name)
    if extra_csv:
        for t in extra_csv.split(","):
            t = t.strip()
            if t:
                tags.add(t)
    if library_name == "transformers":
        tags.add("pytorch")
    return sorted(tags)


def _metric_results_from(
    metrics_json_path: str,
    pipeline_tag: str,
    dataset_name: str,
    dataset_type: str,
) -> list[dict]:
    """Read metrics.json and return model-index metric_results structure."""
    path = Path(metrics_json_path)
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    numeric = [{"type": k, "value": v} for k, v in raw.items() if isinstance(v, (int, float))]
    if not numeric:
        return []
    task_type_map = {
        "image-segmentation": "image-segmentation",
        "image-classification": "image-classification",
        "text-classification": "text-classification",
        "tabular-classification": "tabular-classification",
    }
    task_type = task_type_map.get(pipeline_tag, pipeline_tag or "other")
    return [
        {
            "task_type": task_type,
            "dataset_type": dataset_type or "unknown",
            "dataset_name": dataset_name or "unknown",
            "metrics": numeric,
        }
    ]


def render_model_card(
    template_path: Path,
    metrics: dict,
    out_path: Path,
    **extra,
) -> None:
    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        keep_trailing_newline=True,
    )
    tpl = env.get_template(template_path.name)
    out_path.write_text(
        tpl.render(metrics_table=_format_metrics(metrics), **extra),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish trained artifacts to HuggingFace Hub.")
    parser.add_argument("--repo-id", default="kiselyovd/vehicle-keypoints")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--metrics", default="reports/metrics.json")
    parser.add_argument("--template", default="docs/model_card.md.j2")
    parser.add_argument("--tag", default=None)
    parser.add_argument(
        "--widget-sources",
        default=None,
        metavar="DIR",
        help="Directory of widget PNG examples to upload to samples/ in HF repo.",
    )
    parser.add_argument(
        "--base-model",
        default="",
        help="HF base model ID (e.g. nvidia/segformer-b2-...).",
    )
    parser.add_argument(
        "--hf-dataset",
        default="carfusion",
        help="HF dataset ID (default: carfusion).",
    )
    parser.add_argument(
        "--dataset-name",
        default="",
        help="Human-readable dataset name (defaults to --hf-dataset).",
    )
    parser.add_argument(
        "--domain-tag",
        default="vehicle-keypoints",
        help="Domain tag (default: vehicle-keypoints).",
    )
    parser.add_argument(
        "--pipeline-tag",
        default="keypoint-detection",
        help="HF pipeline tag (default: keypoint-detection).",
    )
    parser.add_argument(
        "--library-name",
        default="ultralytics",
        help="Library name pill on HF (default: ultralytics).",
    )
    parser.add_argument(
        "--tags",
        default="yolo,pose,carfusion",
        help="Comma-separated extra tags (default: yolo,pose,carfusion).",
    )
    parser.add_argument(
        "--hf-export",
        default="artifacts/hf_export",
        metavar="DIR",
        help="Directory produced by export_hf_native.py; contents copied to HF repo root.",
    )
    parser.add_argument(
        "--vitpose-export",
        default=None,
        help="Path to ViTPose safetensors dir (uploaded under baseline/ on the HF repo)",
    )
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts)
    if not artifacts_dir.exists():
        raise SystemExit(f"Artifacts dir not found: {artifacts_dir}")

    metrics: dict = {}
    metrics_path = Path(args.metrics)
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    widget_examples: list[dict] = []
    if args.widget_sources:
        widget_dir = Path(args.widget_sources)
        if widget_dir.is_dir():
            widget_files = sorted(
                list(widget_dir.glob("*.png")) + list(widget_dir.glob("*.jpg"))
            )
            widget_examples = [
                {
                    "src": f"https://huggingface.co/{args.repo_id}/resolve/main/samples/{p.name}",
                    "example_title": p.stem,
                }
                for p in widget_files
            ]

    dataset_name = args.dataset_name or args.hf_dataset

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        for item in artifacts_dir.rglob("*"):
            if item.is_file():
                rel = item.relative_to(artifacts_dir)
                dest = tmp_path / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(item.read_bytes())

        hf_export_dir = Path(args.hf_export)
        if hf_export_dir.is_dir():
            for item in hf_export_dir.rglob("*"):
                if item.is_file():
                    rel = item.relative_to(hf_export_dir)
                    dest = tmp_path / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(item.read_bytes())

        if args.widget_sources:
            widget_dir = Path(args.widget_sources)
            if widget_dir.is_dir():
                samples_dest = tmp_path / "samples"
                samples_dest.mkdir(parents=True, exist_ok=True)
                widget_src_files = sorted(
                    list(widget_dir.glob("*.png")) + list(widget_dir.glob("*.jpg"))
                )
                for img in widget_src_files:
                    shutil.copy2(img, samples_dest / img.name)

        render_model_card(
            template_path=Path(args.template),
            metrics=metrics,
            out_path=tmp_path / "README.md",
            model_description="Production-grade vehicle keypoint detection (14 anatomical car keypoints, CarFusion).",
            github_url=(
                "https://github.com/"
                "kiselyovd/vehicle-keypoints"
            ),
            repo_id=args.repo_id,
            base_model=args.base_model,
            library_name=args.library_name,
            pipeline_tag=args.pipeline_tag,
            tags=_build_tags(args.domain_tag, args.tags, args.library_name),
            datasets=[args.hf_dataset] if args.hf_dataset else [],
            dataset_name=dataset_name,
            hf_dataset=args.hf_dataset,
            widget_examples=widget_examples,
            metric_results=_metric_results_from(
                args.metrics,
                args.pipeline_tag,
                dataset_name,
                args.hf_dataset,
            ),
        )

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(repo_id=args.repo_id, exist_ok=True)
        commit_message = f"Release {args.tag}" if args.tag else "Upload artifacts"
        api.upload_folder(
            repo_id=args.repo_id,
            folder_path=str(tmp_path),
            commit_message=commit_message,
        )

        if getattr(args, "vitpose_export", None):
            api.upload_folder(
                folder_path=args.vitpose_export,
                repo_id=args.repo_id,
                path_in_repo="baseline",
                commit_message="Baseline ViTPose-S weights",
            )
            print(
                f"Uploaded ViTPose baseline from {args.vitpose_export} to "
                f"{args.repo_id}/baseline"
            )

    print(f"Published to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
