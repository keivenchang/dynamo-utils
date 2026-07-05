#!/bin/bash
set -euo pipefail

PORT=9999
PYTHON=/home/keivenc/bin/Linux.x86_64/venv.3.12/bin/python3
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MONITOR="${SCRIPT_DIR}/gpu_monitor.py"

# Stop only the monitor processes that are known to own this dashboard port.
pkill -9 -f "${MONITOR}.*--port ${PORT}" 2>/dev/null || true

exec "$PYTHON" "$MONITOR" --port "$PORT" --host 0.0.0.0
