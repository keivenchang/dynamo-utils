#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# cleanup_log_and_docker.sh
#
# What it does (by default):
# - Docker cleanup:
#   - Runs dynamo-utils/container/cleanup_old_images.sh --retain <N>
#   - Runs: docker builder prune -a --force
# - Log cleanup:
#   - Deletes $NVIDIA_HOME/logs/YYYY-MM-DD/ directories older than <keep-days> (default: 30)
#
# Safety:
# - Only deletes directories directly under $LOGS_DIR that match regex: ^20[0-9]{2}-[0-9]{2}-[0-9]{2}$
# - Never deletes today's directory
# - Uses a per-user flock to avoid overlapping runs
#
# Typical cron (recommended):
#   0 2 * * * NVIDIA_HOME=$HOME/nvidia $HOME/nvidia/dynamo-utils/cron_log.sh cleanup_log_and_docker $HOME/nvidia/dynamo-utils/cleanup_log_and_docker.sh

set -euo pipefail

KEEP_DAYS=30
RETAIN_IMAGES=3
DRY_RUN=false

usage() {
  cat <<EOF
Usage: $0 [--keep-days N] [--retain-images N] [--dry-run|--dryrun] [--help]

Options:
  --keep-days N       Keep log directories for N days (default: ${KEEP_DAYS})
  --retain-images N   Passed to container/cleanup_old_images.sh --retain (default: ${RETAIN_IMAGES})
  --dry-run, --dryrun Print what would be done without deleting/pruning
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-days)
      KEEP_DAYS="${2:-}"; shift 2 ;;
    --retain-images)
      RETAIN_IMAGES="${2:-}"; shift 2 ;;
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
if ! [[ "$RETAIN_IMAGES" =~ ^[0-9]+$ ]] || [ "$RETAIN_IMAGES" -lt 0 ]; then
  echo "Error: --retain-images must be a non-negative integer (got: $RETAIN_IMAGES)" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NVIDIA_HOME="${NVIDIA_HOME:-$(dirname "$SCRIPT_DIR")}"
LOGS_DIR="${LOGS_DIR:-$NVIDIA_HOME/logs}"

LOCK_FILE="/tmp/dynamo-utils.cleanup_log_and_docker.${USER}.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another cleanup_log_and_docker.sh is already running; exiting." >&2
  exit 0
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] cleanup_log_and_docker.sh starting"
echo "  NVIDIA_HOME=$NVIDIA_HOME"
echo "  LOGS_DIR=$LOGS_DIR"
echo "  KEEP_DAYS=$KEEP_DAYS"
echo "  RETAIN_IMAGES=$RETAIN_IMAGES"
echo "  DRY_RUN=$DRY_RUN"

cleanup_docker() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Docker cleanup..."

  if ! command -v docker >/dev/null 2>&1; then
    echo "docker not found; skipping Docker cleanup."
    return 0
  fi

  local cleanup_images="$SCRIPT_DIR/container/cleanup_old_images.sh"
  if [ -x "$cleanup_images" ]; then
    if [ "$DRY_RUN" = true ]; then
      echo "+ $cleanup_images --dry-run --retain $RETAIN_IMAGES"
    else
      "$cleanup_images" --retain "$RETAIN_IMAGES"
    fi
  else
    echo "WARNING: $cleanup_images not found/executable; skipping cleanup_old_images.sh"
  fi

  if [ "$DRY_RUN" = true ]; then
    echo "+ docker builder prune -a --force"
  else
    docker builder prune -a --force || true
  fi
}

cleanup_logs() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Log cleanup..."

  mkdir -p "$LOGS_DIR"

  local today cutoff_epoch
  today="$(date +%F)"
  cutoff_epoch="$(date -d "$today - $KEEP_DAYS days" +%s)"

  local deleted=0 kept=0 skipped=0

  shopt -s nullglob
  for dir in "$LOGS_DIR"/*; do
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
        rm -rf "$dir"
      fi
      deleted=$((deleted + 1))
    else
      kept=$((kept + 1))
    fi
  done
  shopt -u nullglob

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Log cleanup summary: deleted=$deleted kept=$kept skipped=$skipped (keep-days=$KEEP_DAYS)"
}

cleanup_docker
cleanup_logs

echo "[$(date '+%Y-%m-%d %H:%M:%S')] cleanup_log_and_docker.sh done"


