#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Wrapper script to update HTML pages (branch status and commit history)
# This script is designed to be run by cron and requires absolute paths
#
# Environment Variables (optional):
#   NVIDIA_HOME       - Base directory for logs and output files (default: parent of script dir)
#
# Cron Example:
#   */30 * * * * /path/to/update_html_pages.sh
#
# Or with custom path:
#   */30 * * * * NVIDIA_HOME=/var/www/html /path/to/update_html_pages.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Base directory - can be overridden by environment variable
# Default: parent directory of script location
NVIDIA_HOME="${NVIDIA_HOME:-$(dirname "$SCRIPT_DIR")}"

LOGS_DIR="$NVIDIA_HOME/logs"
LOG_FILE="$LOGS_DIR/cron.log"
BRANCHES_OUTPUT_FILE="$NVIDIA_HOME/index.html"
BRANCHES_TEMP_FILE="$NVIDIA_HOME/.index.html.tmp"

# Create logs directory if it doesn't exist
mkdir -p "$LOGS_DIR"

# Cleanup old log directories - keep only last 10 non-empty dated directories
cleanup_old_logs() {
    local logs_dir="$1"
    local keep_count=10

    # Find all dated directories (YYYY-MM-DD format), check if non-empty, sort by date
    local non_empty_dirs=()
    for dir in "$logs_dir"/202*; do
        [ -d "$dir" ] || continue
        # Check if directory has any files (non-empty)
        if [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
            non_empty_dirs+=("$dir")
        fi
    done

    # Sort directories by name (which is date) in descending order
    IFS=$'\n' sorted_dirs=($(sort -r <<<"${non_empty_dirs[*]}"))
    unset IFS

    # If we have more than keep_count directories, delete the older ones
    local total="${#sorted_dirs[@]}"
    if [ "$total" -gt "$keep_count" ]; then
        local to_delete=$((total - keep_count))
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Cleaning up old logs: $total non-empty directories found, keeping last $keep_count, deleting $to_delete oldest" >> "$LOG_FILE"

        # Delete directories beyond the keep_count (they are at the end of sorted array)
        for ((i=keep_count; i<total; i++)); do
            local dir_to_delete="${sorted_dirs[$i]}"
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Deleting old log directory: $dir_to_delete" >> "$LOG_FILE"
            rm -rf "$dir_to_delete"
        done
    fi
}

# Run cleanup
cleanup_old_logs "$LOGS_DIR"

# Update branch status HTML
cd "$NVIDIA_HOME" || exit 1
if python3 "$SCRIPT_DIR/show_dynamo_branches.py" --html --output "$BRANCHES_TEMP_FILE" 2>> "$LOG_FILE"; then
    # Atomic move - only replace if generation succeeded
    mv "$BRANCHES_TEMP_FILE" "$BRANCHES_OUTPUT_FILE"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Updated $BRANCHES_OUTPUT_FILE" >> "$LOG_FILE"
else
    # Script failed - remove temp file and log error
    rm -f "$BRANCHES_TEMP_FILE"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR: Failed to update $BRANCHES_OUTPUT_FILE" >> "$LOG_FILE"
    exit 1
fi

# Generate commit history HTML
DYNAMO_REPO="$NVIDIA_HOME/dynamo_latest"
COMMIT_HISTORY_HTML="$DYNAMO_REPO/index.html"

if [ ! -d "$DYNAMO_REPO/.git" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR: Not a git repository: $DYNAMO_REPO" >> "$LOG_FILE"
    exit 1
fi

cd "$DYNAMO_REPO" || exit 1

# Checkout main and pull latest
echo "$(date '+%Y-%m-%d %H:%M:%S') - Updating $DYNAMO_REPO to latest main" >> "$LOG_FILE"
if git checkout main >> "$LOG_FILE" 2>&1 && git pull origin main >> "$LOG_FILE" 2>&1; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Successfully updated to latest main" >> "$LOG_FILE"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') - WARNING: Failed to update git repository" >> "$LOG_FILE"
    # Continue anyway - use whatever is currently checked out
fi

if python3 "$SCRIPT_DIR/show_commit_history.py" --repo-path . --html --max-commits 50 --output "$COMMIT_HISTORY_HTML" 2>> "$LOG_FILE"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Updated $COMMIT_HISTORY_HTML" >> "$LOG_FILE"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR: Failed to update commit-history.html" >> "$LOG_FILE"
    exit 1
fi
