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
#   MAX_GITHUB_API_CALLS - If set, pass --max-github-api-calls to the Python generators
#                         (useful to keep cron runs predictable; defaults remain script-side).
#   DYNAMO_UTILS_TRACE  - If set (non-empty), enable shell tracing (set -x). Off by default to avoid noisy logs / secrets.
#   DYNAMO_UTILS_TESTING - (deprecated) If set, behave like --fast-test (write index2.html and fewer commits).
#
# Args (optional; can be combined):
#   --show-local-branches   Update the branches dashboard ($NVIDIA_HOME/index.html)
#   --show-commit-history   Update the commit history dashboard ($NVIDIA_HOME/dynamo_latest/index.html)
#   --show-local-resources  Update resource_report.html
#   --fast-test             Write index2.html outputs and run a smaller/faster commit history (10 commits)
#   --fast                  Alias for --fast-test
#
# Back-compat aliases (deprecated; kept for existing cron):
#   --run-show-dynamo-branches  (alias for --show-local-branches)
#   --run-show-commit-history   (alias for --show-commit-history)
#   --run-resource-report       (alias for --show-local-resources)
#
# Behavior:
# - If no args are provided, ALL tasks run (branches + commit-history + resource-report).
#
# Cron Example:
#   # Full fetch every 30 minutes (minute 0 and 30)
#   0,30 * * * * NVIDIA_HOME=$HOME/nvidia /path/to/update_html_pages.sh
#   # Cache-only between full runs (every 4 minutes from minute 8..56)
#   8-59/4 * * * * NVIDIA_HOME=$HOME/nvidia SKIP_GITLAB_FETCH=1 /path/to/update_html_pages.sh
#   # Resource report every minute
#   * * * * * NVIDIA_HOME=$HOME/nvidia /path/to/update_html_pages.sh --show-local-resources

set -euo pipefail
if [ -n "${DYNAMO_UTILS_TRACE:-}" ]; then
    set -x
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Base directory - can be overridden by environment variable
# Default: parent of dynamo-utils/ (i.e. .../nvidia) because this script lives in dynamo-utils/html_pages/
UTILS_DIR="$(dirname "$SCRIPT_DIR")"
NVIDIA_HOME="${NVIDIA_HOME:-$(dirname "$UTILS_DIR")}"

LOGS_DIR="$NVIDIA_HOME/logs"
FAST_TEST="${FAST_TEST:-false}"
# Back-compat: treat env var like --fast-test/--fast.
if [ -n "${DYNAMO_UTILS_TESTING:-}" ]; then
    FAST_TEST=true
fi

usage() {
    cat <<'EOF' >&2
Usage: update_html_pages.sh [--show-local-branches] [--show-commit-history] [--show-local-resources] [--fast-test|--fast]

If no args are provided, ALL tasks run.
EOF
}

RUN_SHOW_DYNAMO_BRANCHES=false
RUN_SHOW_COMMIT_HISTORY=false
RUN_RESOURCE_REPORT=false
ANY_FLAG=false

USER_NAME="${USER:-${LOGNAME:-}}"
if [ -z "$USER_NAME" ]; then
    USER_NAME="$(id -un 2>/dev/null || echo unknown)"
fi

while [ "$#" -gt 0 ]; do
    case "$1" in
        --show-local-branches)
            RUN_SHOW_DYNAMO_BRANCHES=true; ANY_FLAG=true; shift ;;
        --show-commit-history)
            RUN_SHOW_COMMIT_HISTORY=true; ANY_FLAG=true; shift ;;
        --show-local-resources)
            RUN_RESOURCE_REPORT=true; ANY_FLAG=true; shift ;;
        --fast-test)
            FAST_TEST=true; shift ;;
        --fast)
            FAST_TEST=true; shift ;;

        # Back-compat aliases (deprecated)
        --run-show-dynamo-branches)
            RUN_SHOW_DYNAMO_BRANCHES=true; ANY_FLAG=true; shift ;;
        --run-show-commit-history)
            RUN_SHOW_COMMIT_HISTORY=true; ANY_FLAG=true; shift ;;
        --run-resource-report)
            RUN_RESOURCE_REPORT=true; ANY_FLAG=true; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 2 ;;
    esac
done

if [ "$ANY_FLAG" = false ]; then
    RUN_SHOW_DYNAMO_BRANCHES=true
    RUN_SHOW_COMMIT_HISTORY=true
    RUN_RESOURCE_REPORT=true
fi

# Compute branches output path after parsing flags so `--fast/--fast-test` is honored.
BRANCHES_BASENAME="${BRANCHES_BASENAME:-index.html}"
if [ "$FAST_TEST" = true ]; then
    BRANCHES_BASENAME="index2.html"
fi
BRANCHES_OUTPUT_FILE="$NVIDIA_HOME/$BRANCHES_BASENAME"

# Prevent concurrent runs (cron can overlap if a run takes longer than its interval).
# Use a per-user lock in /tmp.
LOCK_FILE="/tmp/dynamo-utils.update_html_pages.${USER_NAME}.lock"
if [ "$RUN_RESOURCE_REPORT" = true ] && [ "$RUN_SHOW_DYNAMO_BRANCHES" = false ] && [ "$RUN_SHOW_COMMIT_HISTORY" = false ]; then
    # Separate lock so frequent resource updates aren't blocked by dashboard runs.
    LOCK_FILE="/tmp/dynamo-utils.update_resource_report.${USER_NAME}.lock"
fi
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    # Another instance is running; log a warning and exit to avoid piling up GitHub/GitLab calls.
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: update_html_pages.sh is locked (lock=$LOCK_FILE); skipping this run" >&2
    exit 0
fi

#
# Timing reference (rough, from one interactive run on keivenc-linux, 2025-12-25):
# - show_local_branches.py: ~15.6s (often the longest)
# - git checkout+pull dynamo_latest: ~1.4s
# - show_local_resources.py (1 day): ~0.6s
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

# Log file for this run (rotated by day directory).
LOG_FILE="$DAY_LOG_DIR/cron.log"

# Per-component logs (append-only, rotated by day directory)
BRANCHES_LOG="$DAY_LOG_DIR/show_local_branches.log"
GIT_UPDATE_LOG="$DAY_LOG_DIR/dynamo_latest_git_update.log"
COMMIT_HISTORY_LOG="$DAY_LOG_DIR/show_commit_history.log"
RESOURCE_REPORT_LOG="$DAY_LOG_DIR/resource_report.log"

# NOTE: log retention is handled by the dedicated cleanup cron:
#   dynamo-utils/cleanup_log_and_docker.sh

run_resource_report() {
    # Generate resource report HTML and prune older DB rows (best-effort; do not fail if DB is missing)
    RESOURCE_DB="${RESOURCE_DB:-$HOME/.cache/dynamo-utils/resource_monitor.sqlite}"
    # Output to the top-level nvidia directory so nginx can serve it at /
    RESOURCE_REPORT_HTML="${RESOURCE_REPORT_HTML:-$NVIDIA_HOME/resource_report.html}"

    # Default window is 1 day. In --fast mode, shrink to the last 2 hours (requested) so cron/test
    # runs are snappier while still showing recent activity.
    RESOURCE_DAYS="1"
    if [ "$FAST_TEST" = true ]; then
        RESOURCE_DAYS="0.0833333"  # 2h / 24h
    fi

    if [ -f "$RESOURCE_DB" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Generating resource report" >> "$LOG_FILE"
        echo "===== $(date '+%Y-%m-%d %H:%M:%S') run_resource_report start =====" >> "$RESOURCE_REPORT_LOG"
        if python3 "$SCRIPT_DIR/show_local_resources.py" \
            --db-path "$RESOURCE_DB" \
            --output "$RESOURCE_REPORT_HTML" \
            --days "$RESOURCE_DAYS" \
            --prune-db-days 4 \
            --db-checkpoint-truncate \
            --title "keivenc-linux Resource Report" >> "$RESOURCE_REPORT_LOG" 2>&1; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Updated $RESOURCE_REPORT_HTML" >> "$LOG_FILE"
        else
            echo "$(date '+%Y-%m-%d %H:%M:%S') - WARNING: Failed to update $RESOURCE_REPORT_HTML" >> "$LOG_FILE"
            exit 1
        fi
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') - WARNING: Resource DB not found: $RESOURCE_DB" >> "$LOG_FILE"
    fi
}

run_show_local_branches() {
    cd "$NVIDIA_HOME" || exit 1
    REFRESH_CLOSED_FLAG="${REFRESH_CLOSED_PRS:+--refresh-closed-prs}"
    MAX_GH_FLAG=""
    if [ -n "${MAX_GITHUB_API_CALLS:-}" ]; then
        MAX_GH_FLAG="--max-github-api-calls ${MAX_GITHUB_API_CALLS}"
    fi
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Generating branches dashboard" >> "$LOG_FILE"
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') run_show_local_branches start (output=$BRANCHES_OUTPUT_FILE) =====" >> "$BRANCHES_LOG"
    if python3 "$SCRIPT_DIR/show_local_branches.py" --repo-path "$NVIDIA_HOME" --output "$BRANCHES_OUTPUT_FILE" $REFRESH_CLOSED_FLAG $MAX_GH_FLAG >> "$BRANCHES_LOG" 2>&1; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Updated $BRANCHES_OUTPUT_FILE" >> "$LOG_FILE"
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR: Failed to update $BRANCHES_OUTPUT_FILE" >> "$LOG_FILE"
        exit 1
    fi
}

run_show_commit_history() {
    DYNAMO_REPO="$NVIDIA_HOME/dynamo_latest"
    COMMIT_HISTORY_BASENAME="${COMMIT_HISTORY_BASENAME:-index.html}"
    if [ "$FAST_TEST" = true ]; then
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
    MAX_GH_FLAG=""
    if [ -n "${MAX_GITHUB_API_CALLS:-}" ]; then
        MAX_GH_FLAG="--max-github-api-calls ${MAX_GITHUB_API_CALLS}"
    fi

    MAX_COMMITS="${MAX_COMMITS:-100}"
    if [ "$FAST_TEST" = true ]; then
        MAX_COMMITS=10
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S') - Generating commit history dashboard (max_commits=$MAX_COMMITS)" >> "$LOG_FILE"
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') run_show_commit_history start (max_commits=$MAX_COMMITS output=$COMMIT_HISTORY_HTML) =====" >> "$COMMIT_HISTORY_LOG"
    if python3 "$SCRIPT_DIR/show_commit_history.py" --repo-path . --max-commits "$MAX_COMMITS" --output "$COMMIT_HISTORY_HTML" $SKIP_FLAG $MAX_GH_FLAG >> "$COMMIT_HISTORY_LOG" 2>&1; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Updated $COMMIT_HISTORY_HTML" >> "$LOG_FILE"
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR: Failed to update commit-history.html" >> "$LOG_FILE"
        exit 1
    fi
}

if [ "$RUN_SHOW_DYNAMO_BRANCHES" = true ]; then
    run_show_local_branches
fi
if [ "$RUN_SHOW_COMMIT_HISTORY" = true ]; then
    run_show_commit_history
fi
if [ "$RUN_RESOURCE_REPORT" = true ]; then
    run_resource_report
fi

