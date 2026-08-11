#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat >&2 <<'EOF'
Usage:
  benchmark-locateanything.sh LOCATE_ROOT MODEL_PATH VIDEO [CUDA_DEVICE] [OUTPUT_DIR] [extra benchmark args...]

Example:
  ./scripts/linux/benchmark-locateanything.sh \
    /data2/DET_Group/ZZS/locateAnything/eagle/Embodied \
    /data2/DET_Group/ZZS/locateAnything/eagle/Embodied/pretrain/LocateAnything-3B \
    /data/test/sample.mp4 \
    cuda:0 \
    /data/test/locany-benchmark \
    --prompt 'person</c>car' --frames 8 --frame-step 25
EOF
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCATE_ROOT="$1"
MODEL_PATH="$2"
VIDEO="$3"
CUDA_DEVICE="${4:-cuda:0}"
OUTPUT_DIR="${5:-$PWD/locany-benchmark-$(date +%Y%m%d-%H%M%S)}"
if [[ $# -ge 5 ]]; then
  shift 5
else
  shift "$#"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "$PYTHON_BIN" "$ROOT_DIR/scripts/linux/benchmark_locateanything.py" \
  --locate-root "$LOCATE_ROOT" \
  --model-path "$MODEL_PATH" \
  --video "$VIDEO" \
  --device "$CUDA_DEVICE" \
  --output "$OUTPUT_DIR" \
  "$@"
