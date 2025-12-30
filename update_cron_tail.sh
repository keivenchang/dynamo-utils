#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Update cron-tail.txt with the last 135 lines of cron.log
#
# Default location:
#   $NVIDIA_HOME/logs/YYYY-MM-DD/cron.log (created by update_html_pages.sh runs)
#
# Output:
#   $NVIDIA_HOME/logs/YYYY-MM-DD/cron-tail.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UTILS_DIR="$(dirname "$SCRIPT_DIR")"
NVIDIA_HOME="${NVIDIA_HOME:-$(dirname "$UTILS_DIR")}"

LOGS_DIR="${LOGS_DIR:-$NVIDIA_HOME/logs}"
TODAY="$(date +%Y-%m-%d)"
DAY_LOG_DIR="$LOGS_DIR/$TODAY"
mkdir -p "$DAY_LOG_DIR"

CRON_LOG="$DAY_LOG_DIR/cron.log"
OUTPUT_FILE="$DAY_LOG_DIR/cron-tail.txt"

if [ -f "$CRON_LOG" ]; then
    tail -135 "$CRON_LOG" > "$OUTPUT_FILE"
else
    echo "cron.log not found at $CRON_LOG" > "$OUTPUT_FILE"
fi
