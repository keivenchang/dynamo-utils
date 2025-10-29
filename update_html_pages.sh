#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Wrapper script to update HTML pages (branch status and commit history)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_FILE="$HOME/nvidia/index.html"
TEMP_FILE="$HOME/nvidia/.index.html.tmp"
LOG_FILE="$HOME/nvidia/dynamo-utils/update_html_pages.log"
COMMIT_HISTORY_FILE="$HOME/nvidia/dynamo_ci/logs/commit-history.html"
COMMIT_HISTORY_TEMP="$HOME/nvidia/dynamo_ci/logs/.commit-history.html.tmp"
LOGS_DIR="$HOME/nvidia/dynamo_ci/logs"

cd "$HOME/nvidia" || exit 1

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
if python3 "$SCRIPT_DIR/show_dynamo_branches.py" --html > "$TEMP_FILE" 2>> "$LOG_FILE"; then
    # Atomic move - only replace if generation succeeded
    mv "$TEMP_FILE" "$OUTPUT_FILE"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Updated $OUTPUT_FILE" >> "$LOG_FILE"
else
    # Script failed - remove temp file and log error
    rm -f "$TEMP_FILE"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR: Failed to update $OUTPUT_FILE" >> "$LOG_FILE"
    exit 1
fi

# Generate commit history HTML
# Note: show_commit_history.py writes HTML directly to commit-history.html file
cd "$HOME/nvidia/dynamo_ci" || exit 1
if python3 "$SCRIPT_DIR/show_commit_history.py" --html --max-commits 50 2>> "$LOG_FILE"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Updated commit-history.html" >> "$LOG_FILE"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR: Failed to update commit-history.html" >> "$LOG_FILE"
    exit 1
fi
