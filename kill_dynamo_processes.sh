#!/bin/bash

# Kill all dynamo/vllm/sglang processes
# Usage: ./kill_dynamo_processes.sh [--force]

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if force mode is enabled
FORCE_MODE=false
if [[ "${1:-}" == "--force" ]]; then
    FORCE_MODE=true
fi

echo -e "${YELLOW}🔄 Starting cleanup of dynamo/vllm/sglang processes...${NC}"
echo

# Function to safely kill processes
safe_kill() {
    local pattern="$1"
    local description="$2"
    local pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    
    if [ -n "$pids" ]; then
        echo -e "${RED}Killing $description processes:${NC}"
        echo "$pids" | while read pid; do
            if [ -n "$pid" ]; then
                ps -p "$pid" -o pid,cmd --no-headers 2>/dev/null || true
            fi
        done
        
        if [ "$FORCE_MODE" = true ]; then
            echo "$pids" | xargs kill -9 2>/dev/null || true
        else
            echo "$pids" | xargs kill 2>/dev/null || true
        fi
        echo
    fi
}

# Kill dynamo frontend processes
safe_kill "python.*dynamo\.frontend" "dynamo frontend"

# Kill vllm processes
safe_kill "python.*dynamo\.vllm" "dynamo vllm"
safe_kill "VLLM::EngineCore" "VLLM EngineCore"
safe_kill "vllm\.entrypoints" "vllm entrypoints"

# Kill sglang processes
safe_kill "python.*dynamo\.sglang" "dynamo sglang"
safe_kill "sglang\.srt" "sglang runtime"

# Kill trtllm processes
safe_kill "python.*dynamo\.trtllm" "dynamo trtllm"

# Kill multiprocess workers
echo -e "${RED}Killing multiprocess workers...${NC}"
ps -ef --forest 2>/dev/null | grep multiprocess | grep -v grep | awk '{print $2}' | xargs kill 2>/dev/null || true

# Kill python processes in /tmp (often leftover workers)
echo -e "${RED}Killing python processes in /tmp...${NC}"
ps -ef 2>/dev/null | grep "python3.*\/tmp" | grep -v grep | awk '{print $2}' | xargs kill 2>/dev/null || true

# Kill any processes with --endpoint flag (common in dynamo)
safe_kill "python3.*--endpoint" "endpoint processes"

# Additional cleanup using pkill for any remaining processes
echo -e "${YELLOW}Running final cleanup with pkill...${NC}"
pkill -f "dynamo\.(frontend|vllm|sglang|trtllm)" 2>/dev/null || true
pkill -f "vllm" 2>/dev/null || true
pkill -f "sglang" 2>/dev/null || true

# If force mode, send SIGKILL to ensure everything is dead
if [ "$FORCE_MODE" = true ]; then
    echo -e "${YELLOW}Force mode: Sending SIGKILL to any remaining processes...${NC}"
    pkill -9 -f "dynamo\.(frontend|vllm|sglang|trtllm)" 2>/dev/null || true
    pkill -9 -f "vllm" 2>/dev/null || true
    pkill -9 -f "sglang" 2>/dev/null || true
fi

echo
echo -e "${GREEN}✅ Cleanup complete!${NC}"

# Show remaining python processes (for verification)
echo
echo -e "${YELLOW}Remaining Python processes:${NC}"
ps aux | grep python | grep -v grep | grep -v "\.cursor-server" | grep -E "(dynamo|vllm|sglang)" || echo "None found (good!)"
