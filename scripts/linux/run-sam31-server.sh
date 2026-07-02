#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

: "${SAM31_COMFY_ROOT:?Set SAM31_COMFY_ROOT to the ComfyUI directory}"
: "${SAM31_CHECKPOINT:?Set SAM31_CHECKPOINT to the SAM3.1 checkpoint}"

HOST="${SAM31_HOST:-0.0.0.0}"
PORT="${SAM31_PORT:-9001}"
CACHE_DIR="${SAM31_CACHE_DIR:-/tmp/video-annotation-sam31}"
DEVICE="${SAM31_DEVICE:-cuda}"
DTYPE="${SAM31_DTYPE:-fp16}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ARGS=(
  -m sam31.server
  --host "$HOST"
  --port "$PORT"
  --cache-dir "$CACHE_DIR"
  --comfy-root "$SAM31_COMFY_ROOT"
  --checkpoint "$SAM31_CHECKPOINT"
  --device "$DEVICE"
  --dtype "$DTYPE"
)

if [[ -n "${SAM31_ALLOWED_ROOTS:-}" ]]; then
  IFS=',' read -ra ROOTS <<< "$SAM31_ALLOWED_ROOTS"
  for root in "${ROOTS[@]}"; do
    [[ -n "$root" ]] && ARGS+=(--allowed-root "$root")
  done
fi

exec "$PYTHON_BIN" "${ARGS[@]}" "$@"
