#!/usr/bin/env bash
# Linux launcher for the video annotator.
# Install dependencies first, for example:
#   python3 -m venv .venv
#   source .venv/bin/activate
#   pip install -r requirements.txt
set -euo pipefail

HOST="${ANNOTATOR_HOST:-0.0.0.0}"
PORT="${ANNOTATOR_PORT:-${PORT:-7860}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -f ".venv/bin/activate" && -z "${VIRTUAL_ENV:-}" ]]; then
    # Use the project virtualenv when present, but do not override an active env.
    source ".venv/bin/activate"
    PYTHON_BIN="python"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[ERROR] Python not found: $PYTHON_BIN"
    exit 1
fi

echo "Starting Video Annotator on http://${HOST}:${PORT}"
exec "$PYTHON_BIN" -m annotator --host "$HOST" --port "$PORT"
