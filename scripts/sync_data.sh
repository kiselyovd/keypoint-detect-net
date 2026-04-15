#!/usr/bin/env bash
set -euo pipefail

SRC="${1:-}"
if [[ -z "$SRC" ]]; then
  echo "Usage: $0 <path-to-local-dataset>" >&2
  exit 1
fi

mkdir -p data/raw
if command -v rsync >/dev/null; then
  rsync -av --progress "$SRC/" data/raw/
else
  cp -r "$SRC/"* data/raw/
fi
echo "Synced from $SRC to data/raw/"
