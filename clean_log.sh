#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# clean_log.sh
#
# What it does (by default):
# - Deletes YYYY-MM-DD/ directories older than <keep-days> (default: 30) from:
#   - $DYNAMO_HOME/logs/          (cron/dashboard logs)
#   - $DYNAMO_HOME/dynamo_ci/logs/ (build report logs)
#
# Safety:
# - Only deletes directories that match regex: ^20[0-9]{2}-[0-9]{2}-[0-9]{2}$
# - Never deletes today's directory
#
# Typical cron (recommended):
#   0 2 * * * DYNAMO_HOME=$HOME/nvidia $HOME/nvidia/dynamo-utils.PRODUCTION/cron_log.sh clean_log $HOME/nvidia/dynamo-utils.PRODUCTION/clean_log.sh

set -euo pipefail

KEEP_DAYS=30
DRY_RUN=false

usage() {
  cat <<EOF
Usage: $0 [--keep-days N] [--dry-run|--dryrun] [--help]

Options:
  --keep-days N              Keep log directories for N days (default: ${KEEP_DAYS})
  --dry-run, --dryrun        Print what would be done without deleting/pruning
  -h, --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-days)
      KEEP_DAYS="${2:-}"; shift 2 ;;
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO_HOME="${DYNAMO_HOME:-$(dirname "$SCRIPT_DIR")}"
LOGS_DIR="${LOGS_DIR:-$DYNAMO_HOME/logs}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] clean_log.sh starting"
echo "  DYNAMO_HOME=$DYNAMO_HOME"
echo "  LOGS_DIR=$LOGS_DIR"
echo "  KEEP_DAYS=$KEEP_DAYS"
echo "  DRY_RUN=$DRY_RUN"

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

mkdir -p "$LOGS_DIR"
cleanup_logs_in_dir "$LOGS_DIR"
cleanup_logs_in_dir "$DYNAMO_HOME/dynamo_ci/logs"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] clean_log.sh done"
