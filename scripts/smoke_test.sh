#!/usr/bin/env bash
set -euo pipefail

cleanup() { docker compose down -v || true; }
trap cleanup EXIT

docker compose build
docker compose up -d api
for i in $(seq 1 60); do
  if curl -sf http://localhost:8000/health >/dev/null; then
    echo "Health OK after ${i}s"
    exit 0
  fi
  sleep 1
done
echo "Health check failed" >&2
docker compose logs api
exit 1
