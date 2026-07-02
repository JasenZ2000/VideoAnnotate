#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

HOST="${ANNOTATOR_HOST:-127.0.0.1}"
PORT="${ANNOTATOR_PORT:-7860}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

exec "$PYTHON_BIN" -m annotator --host "$HOST" --port "$PORT" "$@"
