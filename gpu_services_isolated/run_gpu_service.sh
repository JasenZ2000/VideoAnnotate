#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${GPU_SERVICE_ENV_ACTIVATE:-}" ]]; then
  source "$GPU_SERVICE_ENV_ACTIVATE"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export GPU_SERVICE_HOST="${GPU_SERVICE_HOST:-0.0.0.0}"
export GPU_SERVICE_PORT="${GPU_SERVICE_PORT:-10115}"
export LOCATEANYTHING_ROOT="${LOCATEANYTHING_ROOT:-/data2/DET_Group/ZZS/locateAnything/eagle_new/Embodied}"
export LOCANY_MODEL="${LOCANY_MODEL:-/data2/DET_Group/ZZS/locateAnything/eagle/Embodied/pretrain/LocateAnything-3B}"
export LOCANY_CACHE_DIR="${LOCANY_CACHE_DIR:-/data2/DET_Group/ZZS/locateAnything/eagle/Embodied/fast_tmp_isolated}"
export LOCANY_ALLOWED_ROOTS="${LOCANY_ALLOWED_ROOTS:-/data2/DET_Group}"
export LOCANY_OUTPUT_ALLOWED_ROOTS="${LOCANY_OUTPUT_ALLOWED_ROOTS:-/data2/DET_Group}"
export LOCANY_DEVICES="${LOCANY_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"
export LOCANY_DTYPE="${LOCANY_DTYPE:-bf16}"

export LOCANY_KEEP_MODEL_LOADED="${LOCANY_KEEP_MODEL_LOADED:-1}"
export LOCANY_PRELOAD_WORKERS="${LOCANY_PRELOAD_WORKERS:-1}"
export LOCANY_WORKER_START_TIMEOUT="${LOCANY_WORKER_START_TIMEOUT:-600}"
export LOCANY_WORKER_RPC_TIMEOUT="${LOCANY_WORKER_RPC_TIMEOUT:-3600}"
export LOCANY_BATCH_SIZE="${LOCANY_BATCH_SIZE:-4}"
export LOCANY_BATCH_ATTN="${LOCANY_BATCH_ATTN:-la_flash}"
export LOCANY_VISION_ATTN="${LOCANY_VISION_ATTN:-auto}"
export LOCANY_BATCH_SCHEDULER="${LOCANY_BATCH_SCHEDULER:-pipeline}"
export LOCANY_BATCH_GROUP_SIZE="${LOCANY_BATCH_GROUP_SIZE:-0}"
export LOCANY_STRICT_ATTN="${LOCANY_STRICT_ATTN:-1}"
export LOCANY_DENSE_BACKEND="${LOCANY_DENSE_BACKEND:-sdpa}"
export LOCANY_MIN_EXPECTED_FPS="${LOCANY_MIN_EXPECTED_FPS:-0.5}"

if [[ ! -f "$LOCATEANYTHING_ROOT/locateanything_worker.py" ]]; then
  echo "[ERROR] LOCATEANYTHING_ROOT must contain locateanything_worker.py: $LOCATEANYTHING_ROOT" >&2
  exit 2
fi
if [[ ! -d "$LOCANY_MODEL/batch_utils" ]]; then
  echo "[ERROR] LOCANY_MODEL must contain batch_utils: $LOCANY_MODEL" >&2
  exit 2
fi
if [[ ! -d "$LOCANY_MODEL/kernel_utils" ]]; then
  echo "[ERROR] LOCANY_MODEL must contain kernel_utils (not kernal_utils): $LOCANY_MODEL" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
echo "Starting process-isolated LocateAnything service at http://${GPU_SERVICE_HOST}:${GPU_SERVICE_PORT}"
echo "Devices: $LOCANY_DEVICES (one child process and one model per device)"
echo "Inference: batch-hybrid-${LOCANY_BATCH_SIZE} ($LOCANY_BATCH_ATTN)"
echo "Preload workers: $LOCANY_PRELOAD_WORKERS"
exec "$PYTHON_BIN" -m gpu_services_isolated.server --host "$GPU_SERVICE_HOST" --port "$GPU_SERVICE_PORT" "$@"

