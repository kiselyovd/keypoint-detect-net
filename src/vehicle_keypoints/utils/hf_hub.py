"""HuggingFace Hub helpers."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download, upload_folder


def download_model(repo_id: str, cache_dir: Path | str | None = None) -> Path:
    local = snapshot_download(repo_id=repo_id, cache_dir=str(cache_dir) if cache_dir else None)
    return Path(local)


def publish_model(
    repo_id: str,
    folder: Path | str,
    commit_message: str = "Upload model artifacts",
) -> str:
    return upload_folder(
        repo_id=repo_id,
        folder_path=str(folder),
        commit_message=commit_message,
    )
