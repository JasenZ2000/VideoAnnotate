#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

HOST="${ANNOTATION_PLATFORM_HOST:-0.0.0.0}"
PORT="${ANNOTATION_PLATFORM_PORT:-8088}"
TASKS_DIR="${ANNOTATION_PLATFORM_TASKS_DIR:-$ROOT_DIR/platform_tasks}"
DATABASE="${ANNOTATION_PLATFORM_DB:-$TASKS_DIR/platform.sqlite3}"
CONFIG="${ANNOTATION_PLATFORM_CONFIG:-$ROOT_DIR/config.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

exec "$PYTHON_BIN" -m workflow_platform.server \
  --host "$HOST" \
  --port "$PORT" \
  --tasks-dir "$TASKS_DIR" \
  --database "$DATABASE" \
  --config "$CONFIG" "$@"
