#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Cron log wrapper: run a command and capture ALL output to ~/nvidia/logs/YYYY-MM-DD/<job>.log
#
# Usage:
#   cron_log.sh <job_name> <command...>
#
# Notes:
# - Keeps a stable symlink at ~/nvidia/logs/<job_name>.log pointing at today's log.
# - Prints timestamps and exit codes for easier debugging from logs.

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <job_name> <command...>" >&2
  exit 2
fi

JOB_NAME="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UTILS_DIR="$(dirname "$SCRIPT_DIR")"
NVIDIA_HOME="${NVIDIA_HOME:-$(dirname "$UTILS_DIR")}"

LOGS_DIR="${LOGS_DIR:-$NVIDIA_HOME/logs}"
TODAY="$(date +%Y-%m-%d)"
DAY_LOG_DIR="$LOGS_DIR/$TODAY"

mkdir -p "$DAY_LOG_DIR"

LOG_FILE="$DAY_LOG_DIR/${JOB_NAME}.log"
ln -sfn "$LOG_FILE" "$LOGS_DIR/${JOB_NAME}.log" 2>/dev/null || true

{
  echo "================================================================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START job=${JOB_NAME}"
  echo "cmd: $*"
} >>"$LOG_FILE"

exec >>"$LOG_FILE" 2>&1

"$@"
rc=$?

echo "[$(date '+%Y-%m-%d %H:%M:%S')] END job=${JOB_NAME} rc=${rc}"
exit "$rc"


