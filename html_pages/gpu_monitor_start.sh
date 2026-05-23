#!/bin/bash
set -euo pipefail

PORT=9999
PYTHON=/home/keivenc/dynamo/venv/bin/python3
MONITOR=/home/keivenc/dynamo/dynamo1/dev/observability/dynamo_local_resource_monitor.py
OLD_MONITOR=/home/keivenc/dynamo/dynamo-utils.PRODUCTION/html_pages/gpu_monitor.py

# Stop only the monitor processes that are known to own this dashboard port.
pkill -9 -f "${OLD_MONITOR}.*--port ${PORT}" 2>/dev/null || true
pkill -9 -f "${MONITOR}.*--port ${PORT}" 2>/dev/null || true

exec "$PYTHON" "$MONITOR" --port "$PORT" --host 0.0.0.0
