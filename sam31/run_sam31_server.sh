#!/usr/bin/env bash
# Linux launcher for the remote SAM31 FastAPI job server.
set -euo pipefail

HOST="${SAM31_HOST:-0.0.0.0}"
PORT="${SAM31_PORT:-10112}"
SAM31_ALLOWED_ROOTS="${SAM31_ALLOWED_ROOTS:-/data2/DET_Group/ZZS/my_sam3/tmp}"
CACHE_DIR="${SAM31_CACHE_DIR:-/data2/DET_Group/ZZS/my_sam3/tmp}"
COMFY_ROOT="${SAM31_COMFY_ROOT:-/data2/DET_Group/ZZS/generate/update/ComfyUI}"
CHECKPOINT="${SAM31_CHECKPOINT:-/data2/DET_Group/ZZS/my_sam3/sam3.1_multiplex_fp16.safetensors}"
DEVICE="${SAM31_DEVICE:-cuda:2}"
DTYPE="${SAM31_DTYPE:-fp16}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# if [[ -f ".venv/bin/activate" && -z "${VIRTUAL_ENV:-}" ]]; then
#     source ".venv/bin/activate"
#     PYTHON_BIN="python"
# fi

source ~/miniforge3/etc/profile.d/conda.sh
conda activate zzs_comfy_0_11

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[ERROR] Python not found: $PYTHON_BIN"
    exit 1
fi

ARGS=(
    -m server
    --host "$HOST"
    --port "$PORT"
    --cache-dir "$CACHE_DIR"
    --comfy-root "$COMFY_ROOT"
    --checkpoint "$CHECKPOINT"
    --device "$DEVICE"
    --dtype "$DTYPE"
)

if [[ -n "${SAM31_ALLOWED_ROOTS:-}" ]]; then
    IFS=',' read -ra ROOTS <<< "$SAM31_ALLOWED_ROOTS"
    for root in "${ROOTS[@]}"; do
        if [[ -n "$root" ]]; then
            ARGS+=(--allowed-root "$root")
        fi
    done
fi

echo "Starting SAM31 job server on http://${HOST}:${PORT}"
exec "$PYTHON_BIN" "${ARGS[@]}"
