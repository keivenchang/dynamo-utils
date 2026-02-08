#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ==============================================================================
# await_output.sh - Run a command and exit as soon as expected output appears
# ==============================================================================
#
# Replaces the inefficient pattern:
#   sleep 60 && command && grep 'DONE'
#
# Instead, this script streams output in real-time and exits immediately when
# any of the specified sentinel strings appear, OR when the timeout expires.
#
# All output is captured to a log file for later inspection regardless of
# how the script exits.
#
# USAGE:
#   ./await_output.sh [OPTIONS] -- COMMAND [ARGS...]
#
# OPTIONS:
#   -t, --timeout SECONDS   Maximum time to wait (REQUIRED)
#   -s, --sentinel STRING   String to watch for (can be repeated)
#   -l, --log FILE          Log file path (default: /tmp/await_output.log)
#   -q, --quiet             Don't stream output to terminal (only log to file)
#   -h, --help              Show this help message
#
# EXAMPLES:
#   # Wait for "model loaded" or timeout after 90s
#   ./await_output.sh -t 90 -s "model loaded" -- python serve.py
#
#   # Multiple sentinel strings (exits on first match)
#   ./await_output.sh -s "DONE" -s "ready" -s "listening on" -- ./start_server.sh
#
#   # Custom log file, quiet mode
#   ./await_output.sh -t 60 -s "DONE" -l /tmp/build.log -q -- cargo build 2>&1
#
# EXIT CODES:
#   0 - Sentinel string found in output
#   1 - Usage error / argument error
#   2 - Timeout reached without seeing sentinel
#   3 - Command exited before sentinel was found
#
# ==============================================================================

set -euo pipefail

# ==============================================================================
# DEFAULTS
# ==============================================================================

TIMEOUT=""
SENTINELS=()
LOG_FILE="/tmp/await_output.log"
QUIET=false

# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================

usage() {
    cat <<'EOF'
Usage: await_output.sh [OPTIONS] -- COMMAND [ARGS...]

Run a command and exit as soon as expected output appears, or on timeout.

OPTIONS:
  -t, --timeout SECONDS   Maximum time to wait (REQUIRED)
  -s, --sentinel STRING   String to watch for (can be repeated for multiple)
  -l, --log FILE          Log file path (default: /tmp/await_output.log)
  -q, --quiet             Don't stream output to terminal (only log to file)
  -h, --help              Show this help message

EXIT CODES:
  0 - Sentinel string found in output
  1 - Usage error
  2 - Timeout reached without seeing sentinel
  3 - Command exited before sentinel was found

EXAMPLES:
  ./await_output.sh -t 90 -s "model loaded" -- python serve.py
  ./await_output.sh -t 30 -s "DONE" -s "ready" -- ./start_server.sh
  ./await_output.sh -t 60 -s "DONE" -l /tmp/build.log -q -- cargo build 2>&1
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--timeout)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --timeout requires a value" >&2
                exit 1
            fi
            TIMEOUT="$2"
            shift 2
            ;;
        -s|--sentinel)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --sentinel requires a value" >&2
                exit 1
            fi
            SENTINELS+=("$2")
            shift 2
            ;;
        -l|--log)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --log requires a value" >&2
                exit 1
            fi
            LOG_FILE="$2"
            shift 2
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "ERROR: Unknown option: $1" >&2
            echo "Use --help for usage information" >&2
            exit 1
            ;;
        *)
            # No -- separator; treat this and remaining args as the command
            break
            ;;
    esac
done

# Validate arguments
if [[ $# -eq 0 ]]; then
    echo "ERROR: No command specified" >&2
    echo "Use --help for usage information" >&2
    exit 1
fi

if [[ -z "$TIMEOUT" ]]; then
    echo "ERROR: --timeout is required" >&2
    echo "Use --help for usage information" >&2
    exit 1
fi

if [[ ${#SENTINELS[@]} -eq 0 ]]; then
    echo "ERROR: At least one --sentinel string is required" >&2
    echo "Use --help for usage information" >&2
    exit 1
fi

if ! [[ "$TIMEOUT" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --timeout must be a positive integer, got: $TIMEOUT" >&2
    exit 1
fi

# ==============================================================================
# BUILD GREP PATTERN
# ==============================================================================
# Combine all sentinel strings into a single grep -F pattern file (one per line).
# Using grep -F (fixed strings) avoids regex interpretation issues.

PATTERN_FILE=$(mktemp)
trap 'rm -f "$PATTERN_FILE"' EXIT

for s in "${SENTINELS[@]}"; do
    echo "$s" >> "$PATTERN_FILE"
done

# ==============================================================================
# DISPLAY CONFIGURATION
# ==============================================================================

SENTINEL_DISPLAY=$(printf "'%s'" "${SENTINELS[0]}")
for ((i=1; i<${#SENTINELS[@]}; i++)); do
    SENTINEL_DISPLAY="$SENTINEL_DISPLAY, '${SENTINELS[$i]}'"
done

echo "await_output: watching for $SENTINEL_DISPLAY (timeout: ${TIMEOUT}s)" >&2
echo "await_output: log -> $LOG_FILE" >&2
echo "await_output: running: $*" >&2
echo "---" >&2

# ==============================================================================
# RUN COMMAND AND WATCH OUTPUT
# ==============================================================================

# Truncate log file
> "$LOG_FILE"

START_TIME=$(date +%s)
FOUND=false
CMD_PID=""

# Start the command, merge stderr into stdout, stream through a while-read loop.
# We use a named pipe (FIFO) so we can get the command's PID for cleanup.
FIFO=$(mktemp -u)
mkfifo "$FIFO"
trap 'rm -f "$PATTERN_FILE" "$FIFO"; [ -n "$CMD_PID" ] && kill "$CMD_PID" 2>/dev/null || true' EXIT

# Launch the command writing to the FIFO
"$@" > "$FIFO" 2>&1 &
CMD_PID=$!

# Read from the FIFO line by line
while IFS= read -r line; do
    # Append to log file
    echo "$line" >> "$LOG_FILE"

    # Stream to terminal unless quiet
    if [[ "$QUIET" = false ]]; then
        echo "$line"
    fi

    # Check for sentinel match (fixed-string grep, quiet mode)
    if echo "$line" | grep -qFf "$PATTERN_FILE"; then
        FOUND=true
        ELAPSED=$(( $(date +%s) - START_TIME ))
        # Figure out which sentinel matched for the message
        MATCHED=""
        for s in "${SENTINELS[@]}"; do
            if echo "$line" | grep -qF "$s"; then
                MATCHED="$s"
                break
            fi
        done
        echo "---" >&2
        echo "await_output: sentinel '$MATCHED' found after ${ELAPSED}s" >&2
        # Kill the background command (it may be a long-running server)
        kill "$CMD_PID" 2>/dev/null || true
        wait "$CMD_PID" 2>/dev/null || true
        CMD_PID=""
        exit 0
    fi

    # Check timeout
    NOW=$(date +%s)
    ELAPSED=$(( NOW - START_TIME ))
    if [[ "$ELAPSED" -ge "$TIMEOUT" ]]; then
        echo "---" >&2
        echo "await_output: TIMEOUT after ${TIMEOUT}s without seeing sentinel" >&2
        echo "await_output: log saved to $LOG_FILE" >&2
        kill "$CMD_PID" 2>/dev/null || true
        wait "$CMD_PID" 2>/dev/null || true
        CMD_PID=""
        exit 2
    fi
done < "$FIFO"

# If we get here, the command exited without producing a sentinel match
wait "$CMD_PID" 2>/dev/null
CMD_EXIT=$?
CMD_PID=""
ELAPSED=$(( $(date +%s) - START_TIME ))

echo "---" >&2
echo "await_output: command exited (code=$CMD_EXIT) after ${ELAPSED}s without sentinel" >&2
echo "await_output: log saved to $LOG_FILE" >&2
exit 3
