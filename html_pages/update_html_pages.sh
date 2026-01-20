#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Cron-friendly wrapper to update HTML dashboards (local branches, remote PRs, commit history, resource report).
# This script is designed to be run by cron.
#
# Environment Variables (optional):
#   NVIDIA_HOME         - Base directory for logs and output files
#                        (default: parent of script dir; for this repo layout that is typically ~/dynamo)
#                        Note: GitHub fetching is independently capped/cached by the Python scripts.
#   REFRESH_CLOSED_PRS  - If set, refresh cached closed/merged PR mappings (more GitHub API calls)
#   MAX_GITHUB_API_CALLS - If set, pass --max-github-api-calls to the Python generators.
#                          Useful to keep cron runs predictable; defaults remain script-side.
#   DYNAMO_UTILS_TRACE  - If set (non-empty), enable shell tracing (set -x). Off by default to avoid noisy logs / secrets.
#   DYNAMO_UTILS_TESTING - (deprecated) If set, behave like --output-debug-html (write debug.html and fewer commits).
#   MAX_COMMITS         - If set, cap commits for commit-history (default: 200; overridden by --output-debug-html).
#   DYNAMO_UTILS_CACHE_DIR - If set, overrides ~/.cache/dynamo-utils for the resource report DB lookup.
#   RESOURCE_DB         - If set, explicit SQLite path for resource report (default: $DYNAMO_UTILS_CACHE_DIR/resource_monitor.sqlite).
#
# Remote PRs (optional; used by --show-remote-branches):
#   REMOTE_GITHUB_USERS - Space-separated GitHub usernames to render.
#                         Back-compat: REMOTE_GITHUB_USER
#   REMOTE_PRS_OUT_DIR  - Output directory root for each user.
#                         Default: $HOME/dynamo/speedoflight/dynamo/users/<user>/
#   REMOTE_PRS_OUT_FILE - Full output filename override (rare; if set, used for every user).
#
# Args (optional; can be combined):
#   --show-local-branches   Update the branches dashboard ($NVIDIA_HOME/index.html)
#   --show-commit-history   Update the commit history dashboard ($NVIDIA_HOME/dynamo_latest/index.html)
#   --show-local-resources  Update the resource report ($NVIDIA_HOME/resource_report.html)
#   --show-remote-branches  Update remote PR dashboards for selected GitHub users (IDENTICAL UI to local branches)
#   --output-debug-html            Faster runs: outputs to debug.html instead of index.html, uses smaller commit window (10 commits), shorter resource window
#   --github-token <token>  GitHub token to pass to all show_*.py scripts (preferred).
#   --skip-gitlab-api     Skip fetching from GitLab API (commit-history only); use cached data only (faster).
#   --gitlab-fetch          Explicitly allow GitLab fetching (overrides --output-debug-html default).
#   --dry-run               Print what would be executed without actually running commands
#   --run-ignore-lock        Bypass the /tmp lock (no flock). Useful for manual runs when a stale lock exists.
#
# Back-compat aliases (deprecated; kept for existing cron):
#   --run-show-dynamo-branches  (alias for --show-local-branches)
#   --run-show-commit-history   (alias for --show-commit-history)
#   --run-resource-report       (alias for --show-local-resources)
#   --show-remote-history       (alias for --show-remote-branches)
#
# Behavior:
# - If no args are provided, ALL tasks run (local branches + commit history + resource report + remote PRs).
#
# Cron Example:
#   # Full fetch every 30 minutes (minute 0 and 30)
#   0,30 * * * * NVIDIA_HOME=$HOME/dynamo /path/to/update_html_pages.sh
#   # Cache-only between full runs (every 4 minutes from minute 8..56)
#   8-59/4 * * * * NVIDIA_HOME=$HOME/dynamo /path/to/update_html_pages.sh --skip-gitlab-api
#   # Resource report every minute
#   * * * * * NVIDIA_HOME=$HOME/dynamo /path/to/update_html_pages.sh --show-local-resources

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
FAST_DEBUG="${FAST_DEBUG:-false}"
# Back-compat: treat env var like --output-debug-html.
if [ -n "${DYNAMO_UTILS_TESTING:-}" ]; then
    FAST_DEBUG=true
fi

usage() {
    cat <<'EOF' >&2
Usage: update_html_pages.sh [FLAGS]

If no args are provided, ALL tasks run.

Flags:
  --show-local-branches     Write: $NVIDIA_HOME/index.html (or debug.html in --output-debug-html)
  --show-commit-history     Write: $NVIDIA_HOME/dynamo_latest/index.html (or debug.html in --output-debug-html)
  --show-local-resources    Write: $NVIDIA_HOME/resource_report.html (or resource_report_debug.html in --output-debug-html)
  --show-remote-branches    Write: $HOME/dynamo/speedoflight/dynamo/users/<user>/index.html (or debug.html in --output-debug-html)
  --show-remote-history     Alias for --show-remote-branches (back-compat)

  --output-debug-html              Faster runs: outputs to debug.html instead of index.html, uses smaller commit window (10 commits), shorter resource window
  --enable-success-build-test-logs  Opt-in: cache raw logs for successful *-build-test jobs to parse pytest slowest tests under "Run tests" (slower)
  --skip-gitlab-api       Skip fetching from GitLab API (commit-history only); use cached data only (faster).
  --gitlab-fetch            Explicitly allow GitLab fetching (overrides --output-debug-html default).
                           Default: GitLab fetch is skipped in --output-debug-html.
  --github-token <token>    GitHub token to pass to all show_*.py scripts.

  --dry-run                 Print what would be executed without actually running commands
  --run-ignore-lock         Bypass the /tmp lock (no flock). Useful for manual runs when a stale lock exists.
  -h, --help                Show this help and exit

Notes:
  - Logs are written under: $NVIDIA_HOME/logs/<YYYY-MM-DD>/
    - cron.log (high-level), plus show_local_branches.log, show_commit_history.log, show_remote_branches.log, resource_report.log
  - Lock file defaults to: /tmp/dynamo-utils.update_html_pages.$USER.lock
    - Resource-only runs use a separate lock: /tmp/dynamo-utils.update_resource_report.$USER.lock
EOF
}

RUN_SHOW_DYNAMO_BRANCHES=false
RUN_SHOW_COMMIT_HISTORY=false
RUN_RESOURCE_REPORT=false
RUN_SHOW_REMOTE_BRANCHES=false
ANY_FLAG=false
IGNORE_LOCK=false
DRY_RUN=false
ENABLE_SUCCESS_BUILD_TEST_LOGS=false
GITLAB_FETCH_SKIP_MODE="auto"  # auto|skip|fetch
GITHUB_TOKEN_ARG=""

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
        --show-remote-history)
            RUN_SHOW_REMOTE_BRANCHES=true; ANY_FLAG=true; shift ;;
        --show-local-resources)
            RUN_RESOURCE_REPORT=true; ANY_FLAG=true; shift ;;
        --show-remote-branches)
            RUN_SHOW_REMOTE_BRANCHES=true; ANY_FLAG=true; shift ;;
        --output-debug-html)
            FAST_DEBUG=true; shift ;;
        --enable-success-build-test-logs)
            ENABLE_SUCCESS_BUILD_TEST_LOGS=true; shift ;;
        --github-token)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for --github-token" >&2
                exit 2
            fi
            GITHUB_TOKEN_ARG="$2"; shift 2 ;;
        --skip-gitlab-api)
            GITLAB_FETCH_SKIP_MODE="skip"; shift ;;
        --skip-gitlab-api)
            # Explicitly allow GitLab fetching (overrides --output-debug-html default).
            GITLAB_FETCH_SKIP_MODE="fetch"; shift ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        --run-ignore-lock)
            IGNORE_LOCK=true; shift ;;
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
    RUN_SHOW_REMOTE_BRANCHES=true
fi

# Determine whether to skip GitLab fetching (commit-history only).
# Prefer CLI flags; keep env vars as deprecated aliases.
GITLAB_FETCH_SKIP_EFFECTIVE=false
if [ "$GITLAB_FETCH_SKIP_MODE" = "skip" ]; then
    GITLAB_FETCH_SKIP_EFFECTIVE=true
elif [ "$GITLAB_FETCH_SKIP_MODE" = "fetch" ]; then
    GITLAB_FETCH_SKIP_EFFECTIVE=false
else
    # auto mode: In --output-debug-html, default to skip GitLab fetch (much faster; good for interactive debugging).
    if [ "$FAST_DEBUG" = true ]; then
        GITLAB_FETCH_SKIP_EFFECTIVE=true
    fi
fi

# Compute branches output path after parsing flags so `--output-debug-html` is honored.
BRANCHES_BASENAME="${BRANCHES_BASENAME:-index.html}"
if [ "$FAST_DEBUG" = true ]; then
    BRANCHES_BASENAME="debug.html"
fi
BRANCHES_OUTPUT_FILE="$NVIDIA_HOME/$BRANCHES_BASENAME"

# Prevent concurrent runs (cron can overlap if a run takes longer than its interval).
# Use a per-user lock in /tmp.
LOCK_FILE="/tmp/dynamo-utils.update_html_pages.${USER_NAME}.lock"
if [ "$RUN_RESOURCE_REPORT" = true ] && [ "$RUN_SHOW_DYNAMO_BRANCHES" = false ] && [ "$RUN_SHOW_COMMIT_HISTORY" = false ]; then
    # Separate lock so frequent resource updates aren't blocked by dashboard runs.
    LOCK_FILE="/tmp/dynamo-utils.update_resource_report.${USER_NAME}.lock"
fi
if [ "$IGNORE_LOCK" = true ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: --run-ignore-lock set; bypassing lock (lock=$LOCK_FILE)" >&2
else
    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
        # Another instance is running; log a warning and exit to avoid piling up GitHub/GitLab calls.
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: update_html_pages.sh is locked (lock=$LOCK_FILE); skipping this run" >&2
        exit 0
    fi
fi

#
# Timing reference (rough, from one interactive run on keivenc-linux, 2025-12-25):
# - show_local_branches.py: ~15.6s (often the longest)
# - git checkout+pull dynamo_latest: ~1.4s
# - show_local_resources.py (1 day): ~0.6s
# - mv operations + cleanup: ~0-2ms each
# Notes:
# - show_commit_history.py can dominate if it fetches from GitLab (use --skip-gitlab-api for faster runs).
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
REMOTE_PRS_LOG="$DAY_LOG_DIR/show_remote_branches.log"

# Timestamp helper: prefix every logged line with a PT timestamp.
ts_pt() {
    TZ=America/Los_Angeles date '+%Y-%m-%dT%H:%M:%S.%7NPT'
}

add_timestamp() {
    while IFS= read -r line; do
        echo "[$(ts_pt)] $line"
    done
}

log_line_ts() {
    # Usage: log_line_ts <log_file> <message...>
    local log_file="$1"
    shift
    echo "[$(ts_pt)] $*" >>"$log_file"
}

run_cmd_to_log_ts() {
    # Usage: run_cmd_to_log_ts <log_file> <command...>
    local log_file="$1"
    shift
    if "$@" 2>&1 | add_timestamp >>"$log_file"; then
        return 0
    fi
    return "${PIPESTATUS[0]}"
}

# NOTE: log retention is handled by the dedicated cleanup cron:
#   dynamo-utils/cleanup_log_and_docker.sh

run_resource_report() {
    # Generate resource report HTML and prune older DB rows (best-effort; do not fail if DB is missing)
    # Single source of truth for caches:
    # - $DYNAMO_UTILS_CACHE_DIR (explicit override), else
    # - ~/.cache/dynamo-utils
    CACHE_ROOT="${DYNAMO_UTILS_CACHE_DIR:-$HOME/.cache/dynamo-utils}"
    RESOURCE_DB="${RESOURCE_DB:-$CACHE_ROOT/resource_monitor.sqlite}"
    # Output to the top-level nvidia directory so nginx can serve it at /
    if [ -z "${RESOURCE_REPORT_HTML:-}" ]; then
        if [ "$FAST_DEBUG" = true ]; then
            RESOURCE_REPORT_HTML="$NVIDIA_HOME/resource_report_debug.html"
        else
            RESOURCE_REPORT_HTML="$NVIDIA_HOME/resource_report.html"
        fi
    fi

    # Default window is 1 day. In --fast mode, shrink to the last 2 hours (requested) so cron/test
    # runs are snappier while still showing recent activity.
    RESOURCE_DAYS="1"
    if [ "$FAST_DEBUG" = true ]; then
        RESOURCE_DAYS="0.0833333"  # 2h / 24h
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] Would generate resource report:"
        echo "[DRY-RUN]   Output: $RESOURCE_REPORT_HTML"
        echo "[DRY-RUN]   Resource DB: $RESOURCE_DB"
        echo "[DRY-RUN]   Days window: $RESOURCE_DAYS"
        echo "[DRY-RUN]   Command: python3 $SCRIPT_DIR/show_local_resources.py --db-path $RESOURCE_DB --output $RESOURCE_REPORT_HTML --days $RESOURCE_DAYS --prune-db-days 4 --db-checkpoint-truncate --title 'keivenc-linux Resource Report'"
        return 0
    fi

    if [ -f "$RESOURCE_DB" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Generating resource report" >> "$LOG_FILE"
        log_line_ts "$RESOURCE_REPORT_LOG" "===== run_resource_report start ====="
        if run_cmd_to_log_ts "$RESOURCE_REPORT_LOG" python3 "$SCRIPT_DIR/show_local_resources.py" \
            --db-path "$RESOURCE_DB" \
            --output "$RESOURCE_REPORT_HTML" \
            --days "$RESOURCE_DAYS" \
            --prune-db-days 4 \
            --db-checkpoint-truncate \
            --title "keivenc-linux Resource Report"; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Updated $RESOURCE_REPORT_HTML" >> "$LOG_FILE"
        else
            echo "$(date '+%Y-%m-%d %H:%M:%S') - WARNING: Failed to update $RESOURCE_REPORT_HTML" >> "$LOG_FILE"
            exit 1
        fi
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') - WARNING: Resource DB not found: $RESOURCE_DB" >> "$LOG_FILE"
    fi
}

# Dry-run echo helper
dry_echo() {
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] $*"
    fi
}

run_show_local_branches() {
    cd "$NVIDIA_HOME" || exit 1
    REFRESH_CLOSED_FLAG="${REFRESH_CLOSED_PRS:+--refresh-closed-prs}"
    MAX_GH_FLAG=""
    if [ -n "${MAX_GITHUB_API_CALLS:-}" ]; then
        MAX_GH_FLAG="--max-github-api-calls ${MAX_GITHUB_API_CALLS}"
    fi
    TOKEN_FLAG=""
    if [ -n "${GITHUB_TOKEN_ARG:-}" ]; then
        TOKEN_FLAG="--github-token ${GITHUB_TOKEN_ARG}"
    fi
    SUCCESS_BUILD_TEST_FLAG=""
    # Always enable: fetch/cache successful *-build-test raw logs so we can parse pytest test timings.
    # NOTE: This can be slower; this is intentional per user request.
    SUCCESS_BUILD_TEST_FLAG="--enable-success-build-test-logs"
    
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] Would generate branches dashboard:"
        echo "[DRY-RUN]   Output: $BRANCHES_OUTPUT_FILE"
        echo "[DRY-RUN]   Command: python3 $SCRIPT_DIR/show_local_branches.py --repo-path $NVIDIA_HOME --output $BRANCHES_OUTPUT_FILE $TOKEN_FLAG $REFRESH_CLOSED_FLAG $MAX_GH_FLAG $SUCCESS_BUILD_TEST_FLAG"
        return 0
    fi
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Generating branches dashboard" >> "$LOG_FILE"
    log_line_ts "$BRANCHES_LOG" "===== run_show_local_branches start (output=$BRANCHES_OUTPUT_FILE) ====="
    if run_cmd_to_log_ts "$BRANCHES_LOG" python3 "$SCRIPT_DIR/show_local_branches.py" --repo-path "$NVIDIA_HOME" --output "$BRANCHES_OUTPUT_FILE" $TOKEN_FLAG $REFRESH_CLOSED_FLAG $MAX_GH_FLAG $SUCCESS_BUILD_TEST_FLAG; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Updated $BRANCHES_OUTPUT_FILE" >> "$LOG_FILE"
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR: Failed to update $BRANCHES_OUTPUT_FILE" >> "$LOG_FILE"
        exit 1
    fi
}

run_show_remote_branches() {
    # Optional: generate a "remote PRs for user" page (IDENTICAL UI to local branches).
    # Shows UserNode (GitHub user) → branches with full CI hierarchy.
    # This task is opt-in and ONLY runs when `--show-remote-branches` is passed.
    #
    # Users list:
    #   REMOTE_GITHUB_USERS="keivenchang"
    # Back-compat:
    #   REMOTE_GITHUB_USER=keivenchang
    #
    # Default output root:
    #   ~/dynamo/speedoflight/dynamo/users/<user>/index.html

    USERS_LIST="${REMOTE_GITHUB_USERS:-${REMOTE_GITHUB_USER:-}}"
    if [ -z "${USERS_LIST:-}" ]; then
        # Default: a single user (requested).
        USERS_LIST="keivenchang"
    fi

    MAX_GH_FLAG=""
    if [ -n "${MAX_GITHUB_API_CALLS:-}" ]; then
        MAX_GH_FLAG="--max-github-api-calls ${MAX_GITHUB_API_CALLS}"
    fi
    TOKEN_FLAG=""
    if [ -n "${GITHUB_TOKEN_ARG:-}" ]; then
        TOKEN_FLAG="--github-token ${GITHUB_TOKEN_ARG}"
    fi
    # Always enable: fetch/cache successful *-build-test raw logs so we can parse pytest test timings.
    SUCCESS_BUILD_TEST_FLAG="--enable-success-build-test-logs"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] Would generate remote PRs dashboards for users: $USERS_LIST"
        for U in $USERS_LIST; do
            if [ -z "${U:-}" ]; then
                continue
            fi
            OUT_DIR="${REMOTE_PRS_OUT_DIR:-$HOME/dynamo/speedoflight/dynamo/users/${U}}"
            OUT_FILE="${REMOTE_PRS_OUT_FILE:-$OUT_DIR/index.html}"
            if [ "$FAST_DEBUG" = true ]; then
                OUT_FILE="${REMOTE_PRS_OUT_FILE:-$OUT_DIR/debug.html}"
            fi
            echo "[DRY-RUN]   User: $U"
            echo "[DRY-RUN]     Output: $OUT_FILE"
            echo "[DRY-RUN]     Command: python3 $SCRIPT_DIR/show_remote_branches.py --github-user $U --base-dir $NVIDIA_HOME/dynamo_latest --output $OUT_FILE $TOKEN_FLAG $MAX_GH_FLAG $SUCCESS_BUILD_TEST_FLAG"
        done
        return 0
    fi

    # Iterate space-separated list.
    for U in $USERS_LIST; do
        if [ -z "${U:-}" ]; then
            continue
        fi
        OUT_DIR="${REMOTE_PRS_OUT_DIR:-$HOME/dynamo/speedoflight/dynamo/users/${U}}"
        OUT_FILE="${REMOTE_PRS_OUT_FILE:-$OUT_DIR/index.html}"
        if [ "$FAST_DEBUG" = true ]; then
            OUT_FILE="${REMOTE_PRS_OUT_FILE:-$OUT_DIR/debug.html}"
        fi
        mkdir -p "$(dirname "$OUT_FILE")"
        
        # Copy shared tree-view.css and debug-tree.html to speedoflight
        cp -f "$SCRIPT_DIR/tree-view.css" "$HOME/dynamo/speedoflight/dynamo/tree-view.css" 2>/dev/null || true
        cp -f "$SCRIPT_DIR/debug-tree.html" "$HOME/dynamo/speedoflight/dynamo/users/${U}/debug-tree.html" 2>/dev/null || true

        echo "$(date '+%Y-%m-%d %H:%M:%S') - Generating remote PRs dashboard (user=${U} output=$OUT_FILE)" >> "$LOG_FILE"
        log_line_ts "$REMOTE_PRS_LOG" "===== run_show_remote_branches start (user=${U} output=$OUT_FILE) ====="

        # Use dynamo_latest as base-dir so we can locate the repo clone for workflow YAML inference.
        # Div trees are now default (no flag needed)
        
        if run_cmd_to_log_ts "$REMOTE_PRS_LOG" python3 "$SCRIPT_DIR/show_remote_branches.py" \
            --github-user "${U}" \
            --base-dir "$NVIDIA_HOME/dynamo_latest" \
            --output "$OUT_FILE" \
                $TOKEN_FLAG \
                $MAX_GH_FLAG \
                $SUCCESS_BUILD_TEST_FLAG; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Updated $OUT_FILE" >> "$LOG_FILE"
        else
            echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR: Failed to update $OUT_FILE" >> "$LOG_FILE"
            exit 1
        fi
    done
}

run_show_commit_history() {
    DYNAMO_REPO="$NVIDIA_HOME/dynamo_latest"
    COMMIT_HISTORY_BASENAME="${COMMIT_HISTORY_BASENAME:-index.html}"
    if [ "$FAST_DEBUG" = true ]; then
        COMMIT_HISTORY_BASENAME="debug.html"
    fi
    COMMIT_HISTORY_HTML="$DYNAMO_REPO/$COMMIT_HISTORY_BASENAME"
    # Always enable: fetch/cache successful *-build-test raw logs so we can parse pytest test timings.
    SUCCESS_BUILD_TEST_FLAG="--enable-success-build-test-logs"

    # Flags (shared by dry-run and real-run paths)
    # MAX_COMMITS: default 100, or 25 in --output-debug-html mode (unless overridden by env var)
    if [ "$FAST_DEBUG" = true ]; then
        MAX_COMMITS="${MAX_COMMITS:-25}"
    else
        MAX_COMMITS="${MAX_COMMITS:-100}"
    fi

    SKIP_FLAG=""
    if [ "$GITLAB_FETCH_SKIP_EFFECTIVE" = true ]; then
        SKIP_FLAG="--skip-gitlab-api"
    fi

    MAX_GH_FLAG=""
    if [ -n "${MAX_GITHUB_API_CALLS:-}" ]; then
        MAX_GH_FLAG="--max-github-api-calls ${MAX_GITHUB_API_CALLS}"
    fi

    # Default: parallelize raw log snippet parsing (CPU-heavy) to speed commit history generation.
    # Set PARALLEL_WORKERS=0 to force single-process, or override to tune.
    PARALLEL_WORKERS="${PARALLEL_WORKERS:-32}"
    PARALLEL_FLAG=""
    if [ -n "${PARALLEL_WORKERS:-}" ]; then
        PARALLEL_FLAG="--parallel-workers ${PARALLEL_WORKERS}"
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] Would generate commit history dashboard:"
        echo "[DRY-RUN]   Output: $COMMIT_HISTORY_HTML"
        echo "[DRY-RUN]   Max commits: $MAX_COMMITS"
        echo "[DRY-RUN]   Command: cd $DYNAMO_REPO && git checkout main && git pull origin main"
        echo "[DRY-RUN]   Command: python3 $SCRIPT_DIR/show_commit_history.py --repo-path . --max-commits $MAX_COMMITS --output $COMMIT_HISTORY_HTML $SKIP_FLAG $MAX_GH_FLAG $PARALLEL_FLAG $SUCCESS_BUILD_TEST_FLAG"
        return 0
    fi

    if [ ! -d "$DYNAMO_REPO/.git" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR: Not a git repository: $DYNAMO_REPO" >> "$LOG_FILE"
        exit 1
    fi

    cd "$DYNAMO_REPO" || exit 1

    # Checkout main and pull latest
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Updating $DYNAMO_REPO to latest main" >> "$LOG_FILE"
    log_line_ts "$GIT_UPDATE_LOG" "===== update dynamo_latest start ====="
    if run_cmd_to_log_ts "$GIT_UPDATE_LOG" git checkout main && run_cmd_to_log_ts "$GIT_UPDATE_LOG" git pull origin main; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Successfully updated to latest main" >> "$LOG_FILE"
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') - WARNING: Failed to update git repository" >> "$LOG_FILE"
        # Continue anyway - use whatever is currently checked out
    fi

    # Note: --logs-dir defaults to ../dynamo_ci/logs for dynamo_latest repo

    echo "$(date '+%Y-%m-%d %H:%M:%S') - Generating commit history dashboard (max_commits=$MAX_COMMITS)" >> "$LOG_FILE"
    log_line_ts "$COMMIT_HISTORY_LOG" "===== run_show_commit_history start (max_commits=$MAX_COMMITS output=$COMMIT_HISTORY_HTML) ====="
    if run_cmd_to_log_ts "$COMMIT_HISTORY_LOG" python3 "$SCRIPT_DIR/show_commit_history.py" --repo-path . --max-commits "$MAX_COMMITS" --output "$COMMIT_HISTORY_HTML" $SKIP_FLAG $MAX_GH_FLAG $PARALLEL_FLAG $SUCCESS_BUILD_TEST_FLAG; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Updated $COMMIT_HISTORY_HTML" >> "$LOG_FILE"
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR: Failed to update commit-history.html" >> "$LOG_FILE"
        exit 1
    fi
}

if [ "$RUN_SHOW_REMOTE_BRANCHES" = true ]; then
    run_show_remote_branches
fi
if [ "$RUN_SHOW_DYNAMO_BRANCHES" = true ]; then
    run_show_local_branches
fi
if [ "$RUN_SHOW_COMMIT_HISTORY" = true ]; then
    run_show_commit_history
fi
if [ "$RUN_RESOURCE_REPORT" = true ]; then
    run_resource_report
fi

