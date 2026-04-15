#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-}"
if [[ -z "$DATASET" ]]; then
  echo "Usage: $0 <kaggle-dataset-ref>" >&2
  exit 1
fi

if ! command -v kaggle >/dev/null; then
  uv tool install kaggle
fi

mkdir -p data/raw
kaggle datasets download -d "$DATASET" -p data/raw --unzip
echo "Downloaded $DATASET into data/raw/"
