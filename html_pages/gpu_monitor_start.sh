#!/bin/bash
set -euo pipefail

PORT=9999
PYTHON=/home/keivenc/bin/Linux.x86_64/venv.3.12/bin/python3
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MONITOR="${SCRIPT_DIR}/dynamo_local_resource_monitor.py"

# Stop only monitor processes that are known to own this dashboard port. The
# old path is included so an in-place upgrade can replace a running monitor.
for monitor in "${SCRIPT_DIR}/gpu_monitor.py" "$MONITOR"; do
    pkill -9 -f "${monitor}.*--port ${PORT}" 2>/dev/null || true
done

exec "$PYTHON" "$MONITOR" --port "$PORT" --host 0.0.0.0
