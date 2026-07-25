#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

HOST="${ANNOTATION_PLATFORM_HOST:-0.0.0.0}"
PORT="${ANNOTATION_PLATFORM_PORT:-8088}"
TASKS_DIR="${ANNOTATION_PLATFORM_TASKS_DIR:-$ROOT_DIR/platform_tasks}"
DATABASE="${ANNOTATION_PLATFORM_DB:-$TASKS_DIR/platform.sqlite3}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SSL_CERTFILE="${ANNOTATION_PLATFORM_SSL_CERTFILE:-}"
SSL_KEYFILE="${ANNOTATION_PLATFORM_SSL_KEYFILE:-}"
AUTO_HTTPS="${ANNOTATION_PLATFORM_AUTO_HTTPS:-0}"
TLS_HOSTS="${ANNOTATION_PLATFORM_TLS_HOSTS:-}"
TLS_CERT_DIR="${ANNOTATION_PLATFORM_TLS_CERT_DIR:-}"

TLS_ARGS=()
if [[ -n "$SSL_CERTFILE" || -n "$SSL_KEYFILE" ]]; then
  if [[ -z "$SSL_CERTFILE" || -z "$SSL_KEYFILE" ]]; then
    echo "ANNOTATION_PLATFORM_SSL_CERTFILE and ANNOTATION_PLATFORM_SSL_KEYFILE must be provided together." >&2
    exit 2
  fi
  TLS_ARGS=(--ssl-certfile "$SSL_CERTFILE" --ssl-keyfile "$SSL_KEYFILE")
elif [[ "$AUTO_HTTPS" == "1" ]]; then
  TLS_ARGS=(--auto-https)
  if [[ -n "$TLS_HOSTS" ]]; then
    TLS_ARGS+=(--tls-hosts "$TLS_HOSTS")
  fi
  if [[ -n "$TLS_CERT_DIR" ]]; then
    TLS_ARGS+=(--tls-cert-dir "$TLS_CERT_DIR")
  fi
fi

exec "$PYTHON_BIN" -m workflow_platform.server \
  --host "$HOST" \
  --port "$PORT" \
  --tasks-dir "$TASKS_DIR" \
  --database "$DATABASE" "${TLS_ARGS[@]}" "$@"
