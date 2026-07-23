#!/bin/bash
set -euo pipefail

# Launcher for the Dynamo local resource monitor (standalone host dashboard on $PORT).
# dynamo_local_resource_monitor.py + .html.j2 are a verbatim MIRROR of ai-dynamo/dynamo
# dev/observability/ -- do NOT edit them locally. To re-sync from a dynamo checkout:
#   cp <dynamo>/dev/observability/{dynamo_local_resource_monitor.py,dynamo_local_resource_monitor.html.j2} .
#   cp <dynamo>/dev/observability/test_dynamo_local_resource_monitor.py . && \
#     sed -i 's/^from dev\.observability\./from observability./' test_dynamo_local_resource_monitor.py
# (the test import path is the only intentional local delta.)

PORT=9999
PYTHON=/home/keivenc/bin/Linux.x86_64/venv.3.12/bin/python3
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MONITOR="${SCRIPT_DIR}/dynamo_local_resource_monitor.py"

# Stop only monitor processes that are known to own this dashboard port. Old paths are
# included so an in-place upgrade (incl. the html_pages/ -> observability/ move) can
# replace a running monitor.
for monitor in "${SCRIPT_DIR}/gpu_monitor.py" "$(dirname "$SCRIPT_DIR")/html_pages/dynamo_local_resource_monitor.py" "$MONITOR"; do
    pkill -9 -f "${monitor}.*--port ${PORT}" 2>/dev/null || true
done

exec "$PYTHON" "$MONITOR" --port "$PORT" --host 0.0.0.0
