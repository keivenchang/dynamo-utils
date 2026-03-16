#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# clean_log_and_dynamo_images.sh
#
# What it does (by default):
# - Dynamo image cleanup:
#   - Runs container/cleanup_old_images.sh --retain <N> --keep-dev-and-local-dev-only
#   - Only affects dynamo:* images (never touches non-dynamo images)
#   - Runs: docker builder prune -a --force
# - Log cleanup:
#   - Deletes YYYY-MM-DD/ directories older than <keep-days> (default: 30) from:
#     - $DYNAMO_HOME/logs/          (cron/dashboard logs)
#     - $DYNAMO_HOME/dynamo_ci/logs/ (build report logs)
#
# Safety:
# - Only deletes directories that match regex: ^20[0-9]{2}-[0-9]{2}-[0-9]{2}$
# - Never deletes today's directory
# - Uses a per-user flock to avoid overlapping runs
#
# Typical cron (recommended):
#   0 2 * * * DYNAMO_HOME=$HOME/nvidia $HOME/nvidia/dynamo-utils.PRODUCTION/cron_log.sh clean_log_and_dynamo_images $HOME/nvidia/dynamo-utils.PRODUCTION/clean_log_and_dynamo_images.sh

set -euo pipefail

KEEP_DAYS=30
RETAIN_DYNAMO_IMAGES=3
DRY_RUN=false

usage() {
  cat <<EOF
Usage: $0 [--keep-days N] [--retain-dynamo-images N] [--dry-run|--dryrun] [--help]

Options:
  --keep-days N              Keep log directories for N days (default: ${KEEP_DAYS})
  --retain-dynamo-images N   Keep top N dynamo:* images per variant (default: ${RETAIN_DYNAMO_IMAGES})
  --dry-run, --dryrun        Print what would be done without deleting/pruning
  -h, --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-days)
      KEEP_DAYS="${2:-}"; shift 2 ;;
    --retain-dynamo-images)
      RETAIN_DYNAMO_IMAGES="${2:-}"; shift 2 ;;
    --dry-run|--dryrun)
      DRY_RUN=true; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2 ;;
  esac
done

if ! [[ "$KEEP_DAYS" =~ ^[0-9]+$ ]] || [ "$KEEP_DAYS" -lt 1 ]; then
  echo "Error: --keep-days must be a positive integer (got: $KEEP_DAYS)" >&2
  exit 2
fi
if ! [[ "$RETAIN_DYNAMO_IMAGES" =~ ^[0-9]+$ ]] || [ "$RETAIN_DYNAMO_IMAGES" -lt 0 ]; then
  echo "Error: --retain-dynamo-images must be a non-negative integer (got: $RETAIN_DYNAMO_IMAGES)" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO_HOME="${DYNAMO_HOME:-$(dirname "$SCRIPT_DIR")}"
LOGS_DIR="${LOGS_DIR:-$DYNAMO_HOME/logs}"

USER_NAME="${USER:-${LOGNAME:-}}"
if [ -z "$USER_NAME" ]; then
  USER_NAME="$(id -un 2>/dev/null || echo unknown)"
fi

LOCK_FILE="/tmp/dynamo-utils.clean_log_and_dynamo_images.${USER_NAME}.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another clean_log_and_dynamo_images.sh is already running; exiting." >&2
  exit 0
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] clean_log_and_dynamo_images.sh starting"
echo "  DYNAMO_HOME=$DYNAMO_HOME"
echo "  LOGS_DIR=$LOGS_DIR"
echo "  KEEP_DAYS=$KEEP_DAYS"
echo "  RETAIN_DYNAMO_IMAGES=$RETAIN_DYNAMO_IMAGES"
echo "  DRY_RUN=$DRY_RUN"

cleanup_dynamo_images() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Dynamo image cleanup..."

  if ! command -v docker >/dev/null 2>&1; then
    echo "docker not found; skipping image cleanup."
    return 0
  fi

  local cleanup_images="$SCRIPT_DIR/container/cleanup_old_images.sh"
  if [ -x "$cleanup_images" ]; then
    local args=(--force --retain "$RETAIN_DYNAMO_IMAGES" --keep-dev-and-local-dev-only)
    if [ "$DRY_RUN" = true ]; then
      args+=(--dry-run)
    fi
    echo "+ $cleanup_images ${args[*]}"
    "$cleanup_images" "${args[@]}"
  else
    echo "WARNING: $cleanup_images not found/executable; skipping cleanup_old_images.sh"
  fi
}

cleanup_logs_in_dir() {
  local target_dir="$1"

  if [ ! -d "$target_dir" ]; then
    echo "  $target_dir does not exist; skipping."
    return 0
  fi

  local today cutoff_epoch
  today="$(date +%F)"
  cutoff_epoch="$(date -d "$today - $KEEP_DAYS days" +%s)"

  local deleted=0 kept=0 skipped=0 failed=0

  shopt -s nullglob
  for dir in "$target_dir"/*; do
    [ -d "$dir" ] || continue
    local base
    base="$(basename "$dir")"

    # Only delete YYYY-MM-DD directories.
    if ! [[ "$base" =~ ^20[0-9]{2}-[0-9]{2}-[0-9]{2}$ ]]; then
      skipped=$((skipped + 1))
      continue
    fi

    # Never delete today's directory.
    if [ "$base" = "$today" ]; then
      kept=$((kept + 1))
      continue
    fi

    # Parse the directory date.
    local dir_epoch
    if ! dir_epoch="$(date -d "$base" +%s 2>/dev/null)"; then
      echo "WARNING: could not parse date directory: $dir (skipping)"
      skipped=$((skipped + 1))
      continue
    fi

    # Delete if strictly older than cutoff.
    if [ "$dir_epoch" -lt "$cutoff_epoch" ]; then
      if [ "$DRY_RUN" = true ]; then
        echo "+ rm -rf $dir"
      else
        if rm -rf "$dir"; then
          :
        else
          echo "WARNING: failed to delete old log directory: $dir (permissions/read-only files?). Skipping." >&2
          failed=$((failed + 1))
          continue
        fi
      fi
      deleted=$((deleted + 1))
    else
      kept=$((kept + 1))
    fi
  done
  shopt -u nullglob

  echo "  $target_dir: deleted=$deleted kept=$kept skipped=$skipped failed=$failed"
}

cleanup_logs() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Log cleanup (keep-days=$KEEP_DAYS)..."
  mkdir -p "$LOGS_DIR"
  cleanup_logs_in_dir "$LOGS_DIR"
  cleanup_logs_in_dir "$DYNAMO_HOME/dynamo_ci/logs"
}

cleanup_dynamo_images
cleanup_logs

echo "[$(date '+%Y-%m-%d %H:%M:%S')] clean_log_and_dynamo_images.sh done"


