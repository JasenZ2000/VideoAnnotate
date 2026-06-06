#!/usr/bin/env bash
# Stop the video annotator process started by run_annotator.sh.
set -euo pipefail

PORT="${ANNOTATOR_PORT:-${PORT:-7860}}"

if command -v lsof >/dev/null 2>&1; then
    PIDS="$(lsof -ti "tcp:${PORT}" || true)"
elif command -v fuser >/dev/null 2>&1; then
    PIDS="$(fuser "${PORT}/tcp" 2>/dev/null || true)"
else
    echo "[ERROR] Need lsof or fuser to find processes on port ${PORT}."
    exit 1
fi

if [[ -z "${PIDS}" ]]; then
    echo "No process found on port ${PORT}."
    exit 0
fi

echo "Stopping annotator process(es) on port ${PORT}: ${PIDS}"
kill ${PIDS}

sleep 2

STILL_RUNNING=""
for pid in ${PIDS}; do
    if kill -0 "${pid}" 2>/dev/null; then
        STILL_RUNNING="${STILL_RUNNING} ${pid}"
    fi
done

if [[ -n "${STILL_RUNNING}" ]]; then
    echo "Force stopping:${STILL_RUNNING}"
    kill -9 ${STILL_RUNNING}
fi

echo "Stopped annotator on port ${PORT}."
