#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Wrapper script to update HTML pages (branch status and commit history)
# This script is designed to be run by cron and requires absolute paths
#
# Environment Variables (optional):
#   NVIDIA_HOME         - Base directory for logs and output files (default: parent of script dir)
#   SKIP_GITLAB_FETCH   - If set, skip fetching from GitLab API, use cached data only (faster).
#                        Note: GitHub fetching is independently capped/cached by the Python scripts.
#   REFRESH_CLOSED_PRS  - If set, refresh cached closed/merged PR mappings (more GitHub API calls)
#
# Cron Example:
#   # Full fetch once per hour
#   0 * * * * NVIDIA_HOME=$HOME/nvidia /path/to/update_html_pages.sh
#   # Use cache every 3 minutes (except minute 0)
#   3-59/3 * * * * NVIDIA_HOME=$HOME/nvidia SKIP_GITLAB_FETCH=1 /path/to/update_html_pages.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Base directory - can be overridden by environment variable
# Default: parent of dynamo-utils/ (i.e. .../nvidia) because this script lives in dynamo-utils/html_pages/
UTILS_DIR="$(dirname "$SCRIPT_DIR")"
NVIDIA_HOME="${NVIDIA_HOME:-$(dirname "$UTILS_DIR")}"

LOGS_DIR="$NVIDIA_HOME/logs"
BRANCHES_OUTPUT_FILE="$NVIDIA_HOME/index.html"
BRANCHES_TEMP_FILE="$NVIDIA_HOME/.index.html.tmp"

# Prevent concurrent runs (cron can overlap if a run takes longer than its interval).
# Use a per-user lock in /tmp.
LOCK_FILE="/tmp/dynamo-utils.update_html_pages.${USER}.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    # Another instance is running; exit quietly to avoid piling up GitHub/GitLab calls.
    exit 0
fi

#
# Timing reference (rough, from one interactive run on keivenc-linux, 2025-12-25):
# - show_dynamo_branches.py: ~15.6s (often the longest)
# - git checkout+pull dynamo_latest: ~1.4s
# - resource_report.py (1 day): ~0.6s
# - mv operations + cleanup: ~0-2ms each
# Notes:
# - show_commit_history.py can dominate if it fetches from GitLab (use SKIP_GITLAB_FETCH=1 for faster runs).
# - Real timings vary with network conditions, repo state, and API responsiveness.

# Create logs directory if it doesn't exist
mkdir -p "$LOGS_DIR"

# Write all logs into a dated directory (YYYY-MM-DD) so cleanup_old_logs can prune old days.
TODAY="$(date +%Y-%m-%d)"
DAY_LOG_DIR="$LOGS_DIR/$TODAY"
mkdir -p "$DAY_LOG_DIR"

# Keep backward-compatible pointers for humans/tools:
# - logs/latest -> logs/YYYY-MM-DD
# - logs/cron.log -> logs/YYYY-MM-DD/cron.log
ln -sfn "$DAY_LOG_DIR" "$LOGS_DIR/latest" 2>/dev/null || true
LOG_FILE="$DAY_LOG_DIR/cron.log"
ln -sfn "$LOG_FILE" "$LOGS_DIR/cron.log" 2>/dev/null || true

# Per-component logs (append-only, rotated by day directory)
BRANCHES_LOG="$DAY_LOG_DIR/show_dynamo_branches.log"
GIT_UPDATE_LOG="$DAY_LOG_DIR/dynamo_latest_git_update.log"
COMMIT_HISTORY_LOG="$DAY_LOG_DIR/show_commit_history.log"
RESOURCE_REPORT_LOG="$DAY_LOG_DIR/resource_report.log"

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
REFRESH_CLOSED_FLAG="${REFRESH_CLOSED_PRS:+--refresh-closed-prs}"
echo "$(date '+%Y-%m-%d %H:%M:%S') - Generating branches dashboard" >> "$LOG_FILE"
if python3 "$SCRIPT_DIR/show_dynamo_branches.py" --repo-path "$NVIDIA_HOME" --html --output "$BRANCHES_TEMP_FILE" $REFRESH_CLOSED_FLAG >> "$BRANCHES_LOG" 2>&1; then
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
# When testing, avoid overwriting the main dashboard
# Usage:
#   DYNAMO_UTILS_TESTING=1 NVIDIA_HOME=$HOME/nvidia /path/to/update_html_pages.sh
COMMIT_HISTORY_BASENAME="${COMMIT_HISTORY_BASENAME:-index.html}"
if [ -n "${DYNAMO_UTILS_TESTING:-}" ]; then
    COMMIT_HISTORY_BASENAME="index2.html"
fi
COMMIT_HISTORY_HTML="$DYNAMO_REPO/$COMMIT_HISTORY_BASENAME"

if [ ! -d "$DYNAMO_REPO/.git" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR: Not a git repository: $DYNAMO_REPO" >> "$LOG_FILE"
    exit 1
fi

cd "$DYNAMO_REPO" || exit 1

# Checkout main and pull latest
echo "$(date '+%Y-%m-%d %H:%M:%S') - Updating $DYNAMO_REPO to latest main" >> "$LOG_FILE"
if git checkout main >> "$GIT_UPDATE_LOG" 2>&1 && git pull origin main >> "$GIT_UPDATE_LOG" 2>&1; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Successfully updated to latest main" >> "$LOG_FILE"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') - WARNING: Failed to update git repository" >> "$LOG_FILE"
    # Continue anyway - use whatever is currently checked out
fi

# Note: --logs-dir defaults to ../dynamo_ci/logs for dynamo_latest repo
# Set flag based on environment variable
SKIP_FLAG="${SKIP_GITLAB_FETCH:+--skip-gitlab-fetch}"

echo "$(date '+%Y-%m-%d %H:%M:%S') - Generating commit history dashboard" >> "$LOG_FILE"
if python3 "$SCRIPT_DIR/show_commit_history.py" --repo-path . --html --max-commits 100 --output "$COMMIT_HISTORY_HTML" $SKIP_FLAG >> "$COMMIT_HISTORY_LOG" 2>&1; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Updated $COMMIT_HISTORY_HTML" >> "$LOG_FILE"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR: Failed to update commit-history.html" >> "$LOG_FILE"
    exit 1
fi

# Generate resource report HTML (max 2 days) and prune older DB rows (best-effort; do not fail the entire cron if DB is missing)
RESOURCE_DB="${RESOURCE_DB:-$HOME/.cache/dynamo-utils/resource_monitor.sqlite}"
# Output to the top-level nvidia directory so nginx can serve it at /
RESOURCE_REPORT_HTML="$NVIDIA_HOME/resource_report.html"
RESOURCE_REPORT_TMP="$NVIDIA_HOME/.resource_report.html.tmp"

if [ -f "$RESOURCE_DB" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Generating resource report" >> "$LOG_FILE"
    if python3 "$SCRIPT_DIR/resource_report.py" \
        --db-path "$RESOURCE_DB" \
        --output "$RESOURCE_REPORT_TMP" \
        --days 2 \
        --prune-db-days 2 \
        --db-checkpoint-truncate \
        --title "keivenc-linux Resource Report" >> "$RESOURCE_REPORT_LOG" 2>&1; then
        mv "$RESOURCE_REPORT_TMP" "$RESOURCE_REPORT_HTML"
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Updated $RESOURCE_REPORT_HTML" >> "$LOG_FILE"
    else
        rm -f "$RESOURCE_REPORT_TMP"
        echo "$(date '+%Y-%m-%d %H:%M:%S') - WARNING: Failed to update $RESOURCE_REPORT_HTML" >> "$LOG_FILE"
    fi
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') - WARNING: Resource DB not found: $RESOURCE_DB" >> "$LOG_FILE"
fi
