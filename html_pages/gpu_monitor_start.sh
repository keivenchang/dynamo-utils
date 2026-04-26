#!/bin/bash
# Kill any existing gpu_monitor
pkill -9 -f "gpu_monitor.py" 2>/dev/null
sleep 2
exec /home/keivenc/dynamo/venv/bin/python3 /home/keivenc/dynamo/dynamo-utils.PRODUCTION/html_pages/gpu_monitor.py --port 9999 --host 0.0.0.0
