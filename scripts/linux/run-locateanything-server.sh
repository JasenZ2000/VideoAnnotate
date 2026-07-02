#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

HOST="${LOCANY_HOST:-0.0.0.0}"
PORT="${LOCANY_PORT:-9011}"
CACHE_DIR="${LOCANY_CACHE_DIR:-/tmp/video-annotation-locateanything}"
MODEL="${LOCANY_MODEL:-nvidia/LocateAnything-3B}"
DEVICE="${LOCANY_DEVICE:-cuda}"
DTYPE="${LOCANY_DTYPE:-bf16}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ARGS=(
  --host "$HOST"
  --port "$PORT"
  --cache-dir "$CACHE_DIR"
  --model "$MODEL"
  --device "$DEVICE"
  --dtype "$DTYPE"
)

if [[ -n "${LOCANY_ALLOWED_ROOTS:-}" ]]; then
  IFS=',' read -ra ROOTS <<< "$LOCANY_ALLOWED_ROOTS"
  for root in "${ROOTS[@]}"; do
    [[ -n "$root" ]] && ARGS+=(--allowed-root "$root")
  done
fi

exec "$PYTHON_BIN" locateAnything/locateanything_video_server.py "${ARGS[@]}" "$@"
