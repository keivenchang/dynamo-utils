#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Cron log wrapper: run a command and capture ALL output to ~/nvidia/logs/YYYY-MM-DD/<job>.log
#
# Usage:
#   cron_log.sh <job_name> <command...>
#
# Notes:
# - Prints timestamps and exit codes for easier debugging from logs.
# - Set DYNAMO_UTILS_TRACE=1 to enable shell tracing (set -x) for debugging.

set -euo pipefail
if [ -n "${DYNAMO_UTILS_TRACE:-}" ]; then
    set -x
fi

if [ $# -lt 2 ]; then
  echo "Usage: $0 <job_name> <command...>" >&2
  exit 2
fi

JOB_NAME="$1"
shift

# Optional maintenance kill-switch:
# - Set `DYNAMO_UTILS_DISABLE_CRON=1` to skip ALL cron-invoked jobs that use this wrapper.
if [ "${DYNAMO_UTILS_DISABLE_CRON:-}" = "1" ]; then
  echo "cron_log.sh: disabled (DYNAMO_UTILS_DISABLE_CRON=1); skipping job=${JOB_NAME}" >&2
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UTILS_DIR="$(dirname "$SCRIPT_DIR")"
DYNAMO_HOME="${DYNAMO_HOME:-$(dirname "$UTILS_DIR")}"

LOGS_DIR="${LOGS_DIR:-$DYNAMO_HOME/logs}"
TODAY="$(date +%Y-%m-%d)"
DAY_LOG_DIR="$LOGS_DIR/$TODAY"

mkdir -p "$DAY_LOG_DIR"

LOG_FILE="$DAY_LOG_DIR/${JOB_NAME}.log"

# Helper function to add timestamp to each line
add_timestamp() {
  while IFS= read -r line; do
    echo "[$(TZ=America/Los_Angeles date '+%Y-%m-%dT%H:%M:%S.%7NPT')] $line"
  done
}

{
  echo "================================================================================"
  echo "[$(TZ=America/Los_Angeles date '+%Y-%m-%dT%H:%M:%S.%7NPT')] START job=${JOB_NAME}"
  echo "cmd: $*"
} >>"$LOG_FILE"

# Run command and prefix each line with timestamp
"$@" 2>&1 | add_timestamp >>"$LOG_FILE"
rc=${PIPESTATUS[0]}

echo "[$(TZ=America/Los_Angeles date '+%Y-%m-%dT%H:%M:%S.%7NPT')] END job=${JOB_NAME} rc=${rc}" >>"$LOG_FILE"
exit "$rc"


