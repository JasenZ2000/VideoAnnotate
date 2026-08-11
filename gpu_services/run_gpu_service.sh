#!/usr/bin/env bash
set -euo pipefail

# Optional: source the environment that can import the external LocateAnything runtime.
# Example: export GPU_SERVICE_ENV_ACTIVATE=/opt/venvs/locateanything/bin/activate
if [[ -n "${GPU_SERVICE_ENV_ACTIVATE:-}" ]]; then
  source "$GPU_SERVICE_ENV_ACTIVATE"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# One HTTP process serves both /api/sam31 and /api/locateanything.
export GPU_SERVICE_HOST="${GPU_SERVICE_HOST:-0.0.0.0}"
export GPU_SERVICE_PORT="${GPU_SERVICE_PORT:-10114}"

# LocateAnything runs in this process. This external directory must contain
# locateanything_worker.py and its model dependencies must exist in this Python environment.
export LOCATEANYTHING_ROOT="${LOCATEANYTHING_ROOT:-/data2/DET_Group/ZZS/locateAnything/eagle_new/Embodied}"
export LOCANY_MODEL="${LOCANY_MODEL:-/data2/DET_Group/ZZS/locateAnything/eagle/Embodied/pretrain/LocateAnything-3B}"
export LOCANY_CACHE_DIR="${LOCANY_CACHE_DIR:-/data2/DET_Group/ZZS/locateAnything/eagle/Embodied/fast_tmp}"
export LOCANY_ALLOWED_ROOTS="${LOCANY_ALLOWED_ROOTS:-/data2/DET_Group}"
export LOCANY_OUTPUT_ALLOWED_ROOTS="${LOCANY_OUTPUT_ALLOWED_ROOTS:-/data2/DET_Group}"
# Comma-separated devices allow one LocateAnything job per GPU to run in parallel.
# A request without device (or with device="auto") is assigned to the next free GPU.
export LOCANY_DEVICES="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7"
export LOCANY_DTYPE="${LOCANY_DTYPE:-bf16}"
# LocateAnything service inference is fixed to batch hybrid. The Batch Tool
# intentionally does not expose these implementation details to operators.
export LOCANY_KEEP_MODEL_LOADED="${LOCANY_KEEP_MODEL_LOADED:-1}"
export LOCANY_BATCH_SIZE="${LOCANY_BATCH_SIZE:-4}"
export LOCANY_BATCH_ATTN="${LOCANY_BATCH_ATTN:-la_flash}"
export LOCANY_VISION_ATTN="${LOCANY_VISION_ATTN:-auto}"
export LOCANY_BATCH_SCHEDULER="${LOCANY_BATCH_SCHEDULER:-pipeline}"
export LOCANY_BATCH_GROUP_SIZE="${LOCANY_BATCH_GROUP_SIZE:-0}"
export LOCANY_STRICT_ATTN="${LOCANY_STRICT_ATTN:-1}"
export LOCANY_MIN_EXPECTED_FPS="${LOCANY_MIN_EXPECTED_FPS:-0.5}"

# SAM3.1 remains in its existing ComfyUI environment. The unified service starts
# its runner through SAM31_PYTHON, so this does not have to be the current Python.
export SAM31_COMFY_ROOT="${SAM31_COMFY_ROOT:-/data2/DET_Group/ZZS/generate/update/ComfyUI}"
export SAM31_CHECKPOINT="${SAM31_CHECKPOINT:-/data2/DET_Group/ZZS/my_sam3/sam3.1_multiplex_fp16.safetensors}"
export SAM31_PYTHON="${SAM31_PYTHON:-python}"
export SAM31_RUNNER="${SAM31_RUNNER:-$ROOT_DIR/gpu_services/sam31_track.py}"
export SAM31_CACHE_DIR="${SAM31_CACHE_DIR:-/data2/DET_Group/ZZS/my_sam3/tmp}"
export SAM31_ALLOWED_ROOTS="${SAM31_ALLOWED_ROOTS:-/data2/DET_Group}"
export SAM31_DEVICES="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7"
export SAM31_DTYPE="${SAM31_DTYPE:-fp16}"

if [[ ! -f "$LOCATEANYTHING_ROOT/locateanything_worker.py" ]]; then
  echo "[ERROR] LOCATEANYTHING_ROOT must contain locateanything_worker.py: $LOCATEANYTHING_ROOT" >&2
  exit 2
fi
if [[ ! -f "$SAM31_RUNNER" ]]; then
  echo "[ERROR] SAM31_RUNNER does not exist: $SAM31_RUNNER" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
echo "Starting unified GPU service at http://${GPU_SERVICE_HOST}:${GPU_SERVICE_PORT}"
echo "LocateAnything root: $LOCATEANYTHING_ROOT"
echo "LocateAnything devices: $LOCANY_DEVICES (keep model loaded: $LOCANY_KEEP_MODEL_LOADED)"
echo "LocateAnything inference: batch-hybrid-${LOCANY_BATCH_SIZE} ($LOCANY_BATCH_ATTN, vision: $LOCANY_VISION_ATTN)"
echo "SAM3.1 runner Python: $SAM31_PYTHON"
echo "SAM3.1 runner: $SAM31_RUNNER"
echo "SAM3.1 devices: $SAM31_DEVICES"
exec "$PYTHON_BIN" -m gpu_services.server --host "$GPU_SERVICE_HOST" --port "$GPU_SERVICE_PORT" "$@"
