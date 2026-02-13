#!/bin/bash

# Kill all dynamo/vllm/sglang processes
# Usage: ./kill_dynamo_processes.sh [--force] [--all] [--ports PORT1,PORT2,...]

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check for flags
FORCE_MODE=false
KILL_ALL=false
KILL_PORTS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            FORCE_MODE=true
            shift
            ;;
        --all)
            KILL_ALL=true
            shift
            ;;
        --ports)
            IFS=',' read -ra KILL_PORTS <<< "$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo -e "${YELLOW}🔄 Starting cleanup of dynamo/vllm/sglang processes...${NC}"
echo

# Function to check if any processes matching pattern are still alive
check_alive() {
    local pattern="$1"
    local pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        return 0  # Processes alive
    else
        return 1  # No processes
    fi
}

# Function to safely kill processes with retry logic
safe_kill() {
    local pattern="$1"
    local description="$2"
    local pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    
    if [ -z "$pids" ]; then
        return  # No processes to kill
    fi
    
    # First attempt: graceful kill (SIGTERM)
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
    
    # Wait 3 seconds and check if any are still alive
    echo -e "${BLUE}Waiting 3 seconds for processes to terminate...${NC}"
    sleep 3
    
    local remaining_pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [ -n "$remaining_pids" ]; then
        echo -e "${YELLOW}⚠ Some $description processes still alive, force killing...${NC}"
        echo "$remaining_pids" | while read pid; do
            if [ -n "$pid" ]; then
                ps -p "$pid" -o pid,cmd --no-headers 2>/dev/null || true
            fi
        done
        echo "$remaining_pids" | xargs kill -9 2>/dev/null || true
        echo
        
        # Final check after force kill
        sleep 1
        local final_pids=$(pgrep -f "$pattern" 2>/dev/null || true)
        if [ -n "$final_pids" ]; then
            echo -e "${RED}✗ Failed to kill some $description processes:${NC}"
            echo "$final_pids" | while read pid; do
                if [ -n "$pid" ]; then
                    ps -p "$pid" -o pid,cmd --no-headers 2>/dev/null || true
                fi
            done
            echo
        else
            echo -e "${GREEN}✓ All $description processes killed${NC}"
            echo
        fi
    else
        echo -e "${GREEN}✓ All $description processes terminated gracefully${NC}"
        echo
    fi
}

# Function to get PIDs listening on a port
get_port_pids() {
    local port=$1
    if command -v ss >/dev/null 2>&1; then
        ss -ltnp 2>/dev/null | grep ":$port " | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | sort -u
    elif command -v lsof >/dev/null 2>&1; then
        lsof -ti:$port 2>/dev/null || true
    elif command -v fuser >/dev/null 2>&1; then
        fuser $port/tcp 2>/dev/null | tr -d ' ' || true
    fi
}

# Function to kill processes on a specific port with retry
safe_kill_port() {
    local port="$1"
    local max_retries=5
    local retry=0

    while [ $retry -lt $max_retries ]; do
        local pids=$(get_port_pids "$port")
        if [ -z "$pids" ]; then
            if [ $retry -gt 0 ]; then
                echo -e "${GREEN}✓ Port $port freed after $retry attempt(s)${NC}"
            fi
            return 0
        fi

        echo -e "${RED}Killing processes on port $port: $pids (attempt $((retry + 1))/$max_retries)${NC}"
        echo "$pids" | while read pid; do
            if [ -n "$pid" ]; then
                ps -p "$pid" -o pid,cmd --no-headers 2>/dev/null || true
            fi
        done
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 2
        retry=$((retry + 1))
    done

    # Final check
    local remaining=$(get_port_pids "$port")
    if [ -n "$remaining" ]; then
        echo -e "${YELLOW}⚠ Port $port still has processes after cleanup: $remaining${NC}"
    else
        echo -e "${GREEN}✓ Port $port freed after $retry attempt(s)${NC}"
    fi
}

# Kill processes on specified ports (if --ports was provided)
if [ ${#KILL_PORTS[@]} -gt 0 ]; then
    echo -e "${YELLOW}Killing processes on ports: ${KILL_PORTS[*]}${NC}"
    for port in "${KILL_PORTS[@]}"; do
        safe_kill_port "$port"
    done
    echo
fi

# Kill dynamo frontend processes
safe_kill "python.*dynamo\.frontend" "dynamo frontend"

# Kill vllm processes
safe_kill "python.*dynamo\.vllm" "dynamo vllm"
safe_kill "VLLM::EngineCore" "VLLM EngineCore"
safe_kill "VLLM::Worker" "VLLM Worker"
safe_kill "vllm\.entrypoints" "vllm entrypoints"

# Kill sglang processes
safe_kill "python.*dynamo\.sglang" "dynamo sglang"
safe_kill "sglang\.srt" "sglang runtime"

# Kill trtllm processes
safe_kill "python.*dynamo\.trtllm" "dynamo trtllm"

# Kill MPI futures server (child processes of trtllm)
safe_kill "mpi4py\.futures\.server" "MPI futures server"

# Kill TRTLLM EngineCore processes
safe_kill "TRTLLM::EngineCore" "TRTLLM EngineCore"

# Kill multiprocess workers
safe_kill "multiprocess" "multiprocess workers"

# Kill python processes in /tmp (often leftover workers)
safe_kill "python3.*\/tmp" "Python /tmp workers"

# Kill any processes with --endpoint flag (common in dynamo)
safe_kill "python3.*--endpoint" "endpoint processes"

# Kill etcd and natsd if --all flag is set
if [ "$KILL_ALL" = true ]; then
    echo -e "${YELLOW}🔧 --all flag detected, killing etcd and natsd...${NC}"
    safe_kill "etcd" "etcd"
    safe_kill "nats-server" "natsd"
fi

# Additional cleanup using pkill for any remaining processes
echo -e "${YELLOW}Running final cleanup with pkill...${NC}"
pkill -f "dynamo\.(frontend|vllm|sglang|trtllm)" 2>/dev/null || true
pkill -f "vllm" 2>/dev/null || true
pkill -f "sglang" 2>/dev/null || true
pkill -f "mpi4py" 2>/dev/null || true

if [ "$KILL_ALL" = true ]; then
    pkill -f "etcd" 2>/dev/null || true
    pkill -f "nats-server" 2>/dev/null || true
fi

# Wait 3 seconds and check for survivors
sleep 3

# Check for any remaining processes and force kill if needed
if [ "$KILL_ALL" = true ]; then
    remaining=$(pgrep -f "dynamo\.(frontend|vllm|sglang|trtllm)|vllm|sglang|mpi4py|etcd|nats-server" 2>/dev/null || true)
else
remaining=$(pgrep -f "dynamo\.(frontend|vllm|sglang|trtllm)|vllm|sglang|mpi4py" 2>/dev/null || true)
fi

if [ -n "$remaining" ]; then
    echo -e "${YELLOW}⚠ Some processes still alive after cleanup, force killing...${NC}"
    echo "$remaining" | while read pid; do
        if [ -n "$pid" ]; then
            ps -p "$pid" -o pid,cmd --no-headers 2>/dev/null || true
        fi
    done
    pkill -9 -f "dynamo\.(frontend|vllm|sglang|trtllm)" 2>/dev/null || true
    pkill -9 -f "vllm" 2>/dev/null || true
    pkill -9 -f "sglang" 2>/dev/null || true
    pkill -9 -f "mpi4py" 2>/dev/null || true
    
    if [ "$KILL_ALL" = true ]; then
        pkill -9 -f "etcd" 2>/dev/null || true
        pkill -9 -f "nats-server" 2>/dev/null || true
    fi
    
    # Final check
    sleep 1
    if [ "$KILL_ALL" = true ]; then
        final_remaining=$(pgrep -f "dynamo\.(frontend|vllm|sglang|trtllm)|vllm|sglang|mpi4py|etcd|nats-server" 2>/dev/null || true)
    else
        final_remaining=$(pgrep -f "dynamo\.(frontend|vllm|sglang|trtllm)|vllm|sglang|mpi4py" 2>/dev/null || true)
    fi
    
    if [ -n "$final_remaining" ]; then
        echo -e "${RED}✗ Some processes could not be killed:${NC}"
        echo "$final_remaining" | while read pid; do
            if [ -n "$pid" ]; then
                ps -p "$pid" -o pid,cmd --no-headers 2>/dev/null || true
            fi
        done
    else
        echo -e "${GREEN}✓ All remaining processes killed${NC}"
    fi
fi

echo
echo -e "${GREEN}✅ Cleanup complete!${NC}"

# Show remaining python processes (for verification)
echo
echo -e "${YELLOW}Remaining Python processes:${NC}"
ps aux | grep python | grep -v grep | grep -v "\.cursor-server" | grep -E "(dynamo|vllm|sglang)" || echo "None found (good!)"

# If --all was used, show remaining etcd/natsd processes
if [ "$KILL_ALL" = true ]; then
    echo
    echo -e "${YELLOW}Remaining etcd/natsd processes:${NC}"
    ps aux | grep -E "(etcd|nats-server)" | grep -v grep || echo "None found (good!)"
fi
