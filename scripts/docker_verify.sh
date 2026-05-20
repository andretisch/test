#!/usr/bin/env bash
# Сборка образа и smoke-тест в Docker.
# Если «permission denied» на docker.sock: sudo usermod -aG docker "$USER" && newgrp docker
# или: sudo DOCKER="docker" bash scripts/docker_verify.sh
# (после install: sudo usermod -aG docker "$USER" и перелогиньтесь)

set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-vehicle-counter:test}"
DOCKER="${DOCKER:-docker}"

if [[ ! -f data/test_video.mp4 ]]; then
  echo "ERROR: нужен data/test_video.mp4" >&2
  exit 1
fi

echo "=== docker build ==="
$DOCKER build -t "$IMAGE" .

echo "=== docker smoke ==="
$DOCKER run --rm \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/config:/app/config" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/output:/app/output" \
  "$IMAGE" \
  python scripts/validate_smoke.py

echo "=== OK ==="
