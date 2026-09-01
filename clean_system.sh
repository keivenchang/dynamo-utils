#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# clean_system.sh
#
# This script orchestrates various cleanup tasks by calling other specialized cleanup scripts.
# It handles root-disk pressure, dynamo image cleanup, log cleanup, and optional VSC cleanup.
#
# Usage:
#   ./clean_system.sh [--keep-days N] [--retain-dynamo-images N] [--transcript-keep-days N]
#       [--opencode-keep-days N] [--clean-vsc] [--pressure-only] [--dry-run]
#
# Options are passed through to the relevant sub-scripts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NVIDIA_HOME="${NVIDIA_HOME:-$(dirname "$SCRIPT_DIR")}"

# Default values for arguments (can be overridden by command line)
CLEANUP_OLD_DYNAMO_IMAGES_ARGS=("--force")
CLEAN_LOG_ARGS=()
CLEAN_DISK_ARGS=("--skip-transcripts")
CLEAN_OPENCODE_ARGS=()
OPENCODE_ENABLED=false
PRESSURE_ONLY=false

USER_NAME="${USER:-${LOGNAME:-}}"
if [ -z "$USER_NAME" ]; then
  USER_NAME="$(id -un 2>/dev/null || echo unknown)"
fi

LOCK_FILE="/tmp/dynamo-utils.clean_system.${USER_NAME}.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another clean_system.sh is already running; exiting." >&2
  exit 0
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] clean_system.sh starting"

# Parse arguments and pass them to appropriate scripts
while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-days)
      CLEAN_LOG_ARGS+=("--keep-days" "$2"); shift 2 ;;
    --retain-dynamo-images)
      CLEANUP_OLD_DYNAMO_IMAGES_ARGS+=("--retain" "$2"); shift 2 ;;
    --clean-vsc)
      CLEANUP_OLD_DYNAMO_IMAGES_ARGS+=("--clean-vsc"); shift ;;
    --transcript-keep-days)
      [ "$#" -ge 2 ] || { echo "Error: --transcript-keep-days requires a value" >&2; exit 2; }
      CLEAN_DISK_ARGS=()
      CLEAN_DISK_ARGS+=("--transcript-keep-days" "$2"); shift 2 ;;
    --opencode-keep-days)
      [ "$#" -ge 2 ] || { echo "Error: --opencode-keep-days requires a value" >&2; exit 2; }
      OPENCODE_ENABLED=true
      CLEAN_OPENCODE_ARGS+=("--keep-days" "$2"); shift 2 ;;
    --opencode-db-path)
      [ "$#" -ge 2 ] || { echo "Error: --opencode-db-path requires a value" >&2; exit 2; }
      OPENCODE_ENABLED=true
      CLEAN_OPENCODE_ARGS+=("--db-path" "$2"); shift 2 ;;
    --opencode-vacuum)
      OPENCODE_ENABLED=true
      CLEAN_OPENCODE_ARGS+=("--vacuum"); shift ;;
    --pressure-only)
      PRESSURE_ONLY=true
      CLEAN_DISK_ARGS+=("--pressure-only")
      shift ;;
    --dry-run|--dryrun)
      CLEANUP_OLD_DYNAMO_IMAGES_ARGS+=("--dry-run")
      CLEAN_LOG_ARGS+=("--dry-run")
      CLEAN_DISK_ARGS+=("--dry-run")
      CLEAN_OPENCODE_ARGS+=("--dry-run")
      shift ;;
    -h|--help)
      echo "Usage: $0 [--keep-days N] [--retain-dynamo-images N] [--transcript-keep-days N] [--opencode-keep-days N] [--clean-vsc] [--pressure-only] [--dry-run|--dryrun]"
      echo ""
      echo "Options for root-disk cleanup (passed to clean_disk_pressure.py):"
      echo "  --pressure-only            Skip Docker/log cleanup and exit when root usage is below 90%"
      echo ""
      echo "Options for log cleanup (passed to clean_log.sh):"
      echo "  --keep-days N              Keep log directories for N days (default: 30)"
      echo ""
      echo "Options for transcript cleanup:"
      echo "  --transcript-keep-days N   Keep Claude/Codex JSONL transcripts for N days (disabled by default)"
      echo ""
      echo "Options for OpenCode cleanup:"
      echo "  --opencode-keep-days N     Keep OpenCode sessions updated within N days (disabled by default)"
      echo "  --opencode-db-path PATH    Override the OpenCode database path"
      echo "  --opencode-vacuum          Compact the database after deleting sessions"
      echo ""
      echo "Options for dynamo image cleanup (passed to container/cleanup_old_dynamo_images.sh):"
      echo "  --retain-dynamo-images N   Keep top N *most recent* dynamo:* images per variant (default: 2)"
      echo "  --clean-vsc                Remove all vsc-* containers (stopped+running) and vsc-* images"
      echo ""
      echo "General options:"
      echo "  --dry-run, --dryrun        Print what would be done without deleting/pruning"
      exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2 ;;
  esac
done

# Ensure the necessary scripts exist and are executable
CLEANUP_OLD_DYNAMO_IMAGES_SCRIPT="$SCRIPT_DIR/container/clean_old_local_dynamo_images.sh"
CLEAN_LOG_SCRIPT="$SCRIPT_DIR/clean_log.sh"
CLEAN_DISK_SCRIPT="$SCRIPT_DIR/clean_disk_pressure.py"
CLEAN_OPENCODE_SCRIPT="$SCRIPT_DIR/clean_opencode_db.py"

if [ ! -x "$CLEAN_DISK_SCRIPT" ]; then
  echo "Error: $CLEAN_DISK_SCRIPT not found or not executable." >&2
  exit 1
fi

if ! $PRESSURE_ONLY && [ ! -x "$CLEANUP_OLD_DYNAMO_IMAGES_SCRIPT" ]; then
  echo "Error: $CLEANUP_OLD_DYNAMO_IMAGES_SCRIPT not found or not executable." >&2
  exit 1
fi

if ! $PRESSURE_ONLY && [ ! -x "$CLEAN_LOG_SCRIPT" ]; then
  echo "Error: $CLEAN_LOG_SCRIPT not found or not executable." >&2
  exit 1
fi

if $OPENCODE_ENABLED && [ ! -x "$CLEAN_OPENCODE_SCRIPT" ]; then
  echo "Error: $CLEAN_OPENCODE_SCRIPT not found or not executable." >&2
  exit 1
fi

# Root pressure must run first: Docker may live on another filesystem and a full
# root filesystem can prevent the later cleanup steps from starting.
FINAL_RC=0
run_cleanup_step() {
  local step_rc
  if "$@"; then
    step_rc=0
  else
    step_rc=$?
  fi
  if [ "$FINAL_RC" -eq 0 ] && [ "$step_rc" -ne 0 ]; then
    FINAL_RC=$step_rc
  fi
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Calling $CLEAN_DISK_SCRIPT ${CLEAN_DISK_ARGS[*]}"
run_cleanup_step "$CLEAN_DISK_SCRIPT" "${CLEAN_DISK_ARGS[@]}"

if $PRESSURE_ONLY; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] clean_system.sh done rc=$FINAL_RC"
  exit "$FINAL_RC"
fi

if $OPENCODE_ENABLED; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Calling $CLEAN_OPENCODE_SCRIPT ${CLEAN_OPENCODE_ARGS[*]}"
  run_cleanup_step "$CLEAN_OPENCODE_SCRIPT" "${CLEAN_OPENCODE_ARGS[@]}"
fi

# Run dynamo image cleanup
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Calling $CLEANUP_OLD_DYNAMO_IMAGES_SCRIPT ${CLEANUP_OLD_DYNAMO_IMAGES_ARGS[*]}"
run_cleanup_step "$CLEANUP_OLD_DYNAMO_IMAGES_SCRIPT" "${CLEANUP_OLD_DYNAMO_IMAGES_ARGS[@]}"

# Run log cleanup
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Calling $CLEAN_LOG_SCRIPT ${CLEAN_LOG_ARGS[*]}"
run_cleanup_step "$CLEAN_LOG_SCRIPT" "${CLEAN_LOG_ARGS[@]}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] clean_system.sh done rc=$FINAL_RC"
exit "$FINAL_RC"
