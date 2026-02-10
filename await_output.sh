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
# any of the specified sentinel strings appear, OR when the timeout expires,
# OR when a watched PID exits.
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
#   -p, --pid PID           Exit when this PID exits (e.g. a background build)
#   -l, --log FILE          Log file path (default: /tmp/await_output.log)
#   -q, --quiet             Don't stream output to terminal (only log to file)
#   -h, --help              Show this help message
#
# At least one of --sentinel or --pid is required.
#
# EXAMPLES:
#   # Wait for "model loaded" or timeout after 90s
#   ./await_output.sh -t 90 -s "model loaded" -- python serve.py
#
#   # Multiple sentinel strings (exits on first match)
#   ./await_output.sh -s "DONE" -s "ready" -s "listening on" -- ./start_server.sh
#
#   # Watch a background build PID; exit when it finishes
#   ./await_output.sh -t 1200 -p $BUILD_PID -- tail -f /tmp/build.log
#
#   # Combine: exit on sentinel OR when PID exits (whichever comes first)
#   ./await_output.sh -t 600 -s "ERROR" -p $BUILD_PID -- tail -f /tmp/build.log
#
#   # Custom log file, quiet mode
#   ./await_output.sh -t 60 -s "DONE" -l /tmp/build.log -q -- cargo build 2>&1
#
# EXIT CODES:
#   0 - Sentinel string found in output, or watched PID exited (no sentinels)
#   1 - Usage error / argument error
#   2 - Timeout reached without seeing sentinel
#   3 - Command exited before sentinel was found
#   4 - Watched PID exited but sentinel was not found
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
WATCH_PID=""

# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================

usage() {
    cat <<'EOF'
Usage: await_output.sh [OPTIONS] -- COMMAND [ARGS...]

Run a command and exit as soon as expected output appears, or on timeout,
or when a watched PID exits.

OPTIONS:
  -t, --timeout SECONDS   Maximum time to wait (REQUIRED)
  -s, --sentinel STRING   String to watch for (can be repeated for multiple)
  -p, --pid PID           Exit when this PID exits (e.g. a background build)
  -l, --log FILE          Log file path (default: /tmp/await_output.log)
  -q, --quiet             Don't stream output to terminal (only log to file)
  -h, --help              Show this help message

At least one of --sentinel or --pid is required.

EXIT CODES:
  0 - Sentinel found, or watched PID exited (when no sentinels specified)
  1 - Usage error
  2 - Timeout reached
  3 - Command exited before sentinel was found
  4 - Watched PID exited but sentinel was not found

EXAMPLES:
  ./await_output.sh -t 90 -s "model loaded" -- python serve.py
  ./await_output.sh -t 30 -s "DONE" -s "ready" -- ./start_server.sh
  ./await_output.sh -t 1200 -p $BUILD_PID -- tail -f /tmp/build.log
  ./await_output.sh -t 600 -s "ERROR" -p $PID -- tail -f /tmp/build.log
  ./await_output.sh -t 60 -s "DONE" -l /tmp/build.log -q -- cargo build 2>&1
EOF
}

self_test() {
    local me="$1"
    local pass=0 fail=0 total=0
    local log="/tmp/await_self_test.log"

    run_case() {
        local name="$1" expect="$2"
        shift 2
        total=$(( total + 1 ))
        local rc=0
        "$me" "$@" > /dev/null 2>&1 || rc=$?
        if [[ "$rc" -eq "$expect" ]]; then
            echo "  PASS  [$name] exit=$rc"
            pass=$(( pass + 1 ))
        else
            echo "  FAIL  [$name] expected=$expect got=$rc"
            fail=$(( fail + 1 ))
        fi
    }

    echo "await_output.sh --self-test"
    echo "========================================"

    # --- Validation errors (exit 1) ---
    run_case "no args" 1
    run_case "no command" 1 \
        -t 5 -s "x"
    run_case "no timeout" 1 \
        -s "x" -- true
    run_case "no sentinel or pid" 1 \
        -t 5 -- true
    run_case "invalid timeout" 1 \
        -t abc -s "x" -- true
    run_case "invalid pid (non-numeric)" 1 \
        -t 5 -p abc -- true
    run_case "invalid pid (dead)" 1 \
        -t 5 -p 999999999 -- true

    # --- Sentinel found (exit 0) ---
    run_case "sentinel found" 0 \
        -t 5 -s "hello" -l "$log" -q -- echo "hello world"
    run_case "sentinel found (2nd sentinel)" 0 \
        -t 5 -s "NOMATCH" -s "found" -l "$log" -q -- echo "found it"
    run_case "sentinel found (multi-line)" 0 \
        -t 5 -s "line2" -l "$log" -q -- bash -c 'echo line1; echo line2; echo line3'

    # --- Timeout (exit 2) ---
    run_case "timeout" 2 \
        -t 2 -s "NEVERMATCH" -l "$log" -q -- bash -c 'while true; do echo tick; sleep 0.5; done'

    # --- Command exits before sentinel (exit 3) ---
    run_case "cmd exits, no sentinel" 3 \
        -t 5 -s "NEVERMATCH" -l "$log" -q -- echo "wrong output"

    # --- PID only, PID exits (exit 0) ---
    sleep 2 & _pid=$!
    run_case "pid only, pid exits" 0 \
        -t 10 -p "$_pid" -l "$log" -q -- tail -f /dev/null

    # --- PID + sentinel, PID exits first (exit 4) ---
    sleep 2 & _pid=$!
    run_case "pid+sentinel, pid exits first" 4 \
        -t 10 -s "NEVERMATCH" -p "$_pid" -l "$log" -q -- tail -f /dev/null

    # --- PID + sentinel, sentinel found first (exit 0) ---
    sleep 30 & _pid=$!
    run_case "pid+sentinel, sentinel wins" 0 \
        -t 10 -s "ready" -p "$_pid" -l "$log" -q \
        -- bash -c 'sleep 1; echo ready'
    kill "$_pid" 2>/dev/null || true

    echo "========================================"
    echo "Results: $pass/$total passed, $fail failed"
    rm -f "$log"
    [[ "$fail" -eq 0 ]]
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
        -p|--pid)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --pid requires a value" >&2
                exit 1
            fi
            WATCH_PID="$2"
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
        --self-test)
            self_test "$0"
            exit $?
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

if [[ ${#SENTINELS[@]} -eq 0 && -z "$WATCH_PID" ]]; then
    echo "ERROR: At least one --sentinel or --pid is required" >&2
    echo "Use --help for usage information" >&2
    exit 1
fi

if [[ -n "$WATCH_PID" ]]; then
    if ! [[ "$WATCH_PID" =~ ^[0-9]+$ ]]; then
        echo "ERROR: --pid must be a numeric PID, got: $WATCH_PID" >&2
        exit 1
    fi
    if ! kill -0 "$WATCH_PID" 2>/dev/null; then
        echo "ERROR: PID $WATCH_PID does not exist or is not accessible" >&2
        exit 1
    fi
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

HAS_SENTINELS=false
if [[ ${#SENTINELS[@]} -gt 0 ]]; then
    HAS_SENTINELS=true
fi

# ==============================================================================
# DISPLAY CONFIGURATION
# ==============================================================================

WATCH_DISPLAY=""
if [[ "$HAS_SENTINELS" = true ]]; then
    SENTINEL_DISPLAY=$(printf "'%s'" "${SENTINELS[0]}")
    for ((i=1; i<${#SENTINELS[@]}; i++)); do
        SENTINEL_DISPLAY="$SENTINEL_DISPLAY, '${SENTINELS[$i]}'"
    done
    WATCH_DISPLAY="sentinel $SENTINEL_DISPLAY"
fi
if [[ -n "$WATCH_PID" ]]; then
    if [[ -n "$WATCH_DISPLAY" ]]; then
        WATCH_DISPLAY="$WATCH_DISPLAY + PID $WATCH_PID"
    else
        WATCH_DISPLAY="PID $WATCH_PID"
    fi
fi

echo "await_output: watching for $WATCH_DISPLAY (timeout: ${TIMEOUT}s)" >&2
echo "await_output: log -> $LOG_FILE" >&2
echo "await_output: running: $*" >&2
echo "---" >&2

# ==============================================================================
# RUN COMMAND AND WATCH OUTPUT
# ==============================================================================

# Truncate log file
> "$LOG_FILE"

START_TIME=$(date +%s)
CMD_PID=""
WATCHER_PID=""

# Flag file: created by the PID watcher when the watched PID exits.
# We use a file (not a variable) because the watcher runs in a subshell.
PID_EXITED_FLAG=""
if [[ -n "$WATCH_PID" ]]; then
    PID_EXITED_FLAG=$(mktemp -u "/tmp/await_pid_flag.XXXXXX")
fi

# Start the command, merge stderr into stdout, stream through a while-read loop.
# We use a named pipe (FIFO) so we can get the command's PID for cleanup.
FIFO=$(mktemp -u)
mkfifo "$FIFO"

cleanup() {
    rm -f "$PATTERN_FILE" "$FIFO" "$PID_EXITED_FLAG"
    [ -n "$CMD_PID" ] && kill "$CMD_PID" 2>/dev/null || true
    [ -n "$WATCHER_PID" ] && kill "$WATCHER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Launch the command writing to the FIFO
"$@" > "$FIFO" 2>&1 &
CMD_PID=$!

# Start the PID watcher AFTER we have CMD_PID.
# When the watched PID exits, the watcher waits briefly for output to drain,
# then kills CMD_PID so the read loop terminates.
if [[ -n "$WATCH_PID" ]]; then
    (
        while kill -0 "$WATCH_PID" 2>/dev/null; do
            sleep 1
        done
        # Watched PID is gone. Let final output flush through tail/command.
        touch "$PID_EXITED_FLAG"
        sleep 2
        kill "$CMD_PID" 2>/dev/null || true
    ) &
    WATCHER_PID=$!
fi

# Read from the FIFO line by line
while IFS= read -r line; do
    # Append to log file
    echo "$line" >> "$LOG_FILE"

    # Stream to terminal unless quiet
    if [[ "$QUIET" = false ]]; then
        echo "$line"
    fi

    # Check for sentinel match (fixed-string grep, quiet mode)
    if [[ "$HAS_SENTINELS" = true ]]; then
        if echo "$line" | grep -qFf "$PATTERN_FILE"; then
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

# The read loop ended -- the command (or PID watcher) closed the FIFO.
# Capture exit code without letting set -e abort on non-zero (e.g. SIGTERM=143).
CMD_EXIT=0
wait "$CMD_PID" 2>/dev/null || CMD_EXIT=$?
CMD_PID=""
ELAPSED=$(( $(date +%s) - START_TIME ))

# Check if the watched PID caused the exit
if [[ -n "$WATCH_PID" && -f "$PID_EXITED_FLAG" ]]; then
    echo "---" >&2
    echo "await_output: watched PID $WATCH_PID exited after ${ELAPSED}s" >&2
    echo "await_output: log saved to $LOG_FILE" >&2
    if [[ "$HAS_SENTINELS" = true ]]; then
        # Sentinels were specified but none matched before PID exited
        exit 4
    else
        # No sentinels -- PID exit is the expected completion
        exit 0
    fi
fi

# Command exited on its own (not due to PID watcher)
echo "---" >&2
echo "await_output: command exited (code=$CMD_EXIT) after ${ELAPSED}s without sentinel" >&2
echo "await_output: log saved to $LOG_FILE" >&2
exit 3
