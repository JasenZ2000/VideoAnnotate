#!/usr/bin/env bash
set -euo pipefail

LOCANY_PORT="${LOCANY_PORT:-9011}"
LOCANY_CACHE_DIR="${LOCANY_CACHE_DIR:-/tmp/object-reid-locateanything}"
LOCANY_MODEL="${LOCANY_MODEL:-nvidia/LocateAnything-3B}"
LOCANY_DEVICE="${LOCANY_DEVICE:-cuda}"
LOCANY_DTYPE="${LOCANY_DTYPE:-bf16}"
LOCANY_ALLOWED_ROOTS="${LOCANY_ALLOWED_ROOTS:-}"

args=(
  --host "${LOCANY_HOST:-0.0.0.0}"
  --port "${LOCANY_PORT}"
  --cache-dir "${LOCANY_CACHE_DIR}"
  --model "${LOCANY_MODEL}"
  --device "${LOCANY_DEVICE}"
  --dtype "${LOCANY_DTYPE}"
)

if [[ -n "${LOCANY_ALLOWED_ROOTS}" ]]; then
  IFS=',' read -ra roots <<< "${LOCANY_ALLOWED_ROOTS}"
  for root in "${roots[@]}"; do
    [[ -n "${root}" ]] && args+=(--allowed-root "${root}")
  done
fi

python locateAnything/locateanything_video_server.py "${args[@]}"
