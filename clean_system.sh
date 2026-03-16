#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# clean_system.sh
#
# This script orchestrates various cleanup tasks by calling other specialized cleanup scripts.
# It handles dynamo image cleanup, log cleanup, and can also perform vsc-related cleanup.
#
# Usage:
#   ./clean_system.sh [--keep-days N] [--retain-dynamo-images N] [--clean-vsc] [--dry-run]
#
# Options are passed through to the relevant sub-scripts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO_HOME="${DYNAMO_HOME:-$(dirname "$SCRIPT_DIR")}"

# Default values for arguments (can be overridden by command line)
CLEANUP_OLD_DYNAMO_IMAGES_ARGS=("--force")
CLEAN_LOG_ARGS=()

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
    --dry-run|--dryrun)
      CLEANUP_OLD_DYNAMO_IMAGES_ARGS+=("--dry-run")
      CLEAN_LOG_ARGS+=("--dry-run")
      shift ;;
    -h|--help)
      echo "Usage: $0 [--keep-days N] [--retain-dynamo-images N] [--clean-vsc] [--dry-run|--dryrun]"
      echo ""
      echo "Options for log cleanup (passed to clean_log.sh):"
      echo "  --keep-days N              Keep log directories for N days (default: 30)"
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

if [ ! -x "$CLEANUP_OLD_DYNAMO_IMAGES_SCRIPT" ]; then
  echo "Error: $CLEANUP_OLD_DYNAMO_IMAGES_SCRIPT not found or not executable." >&2
  exit 1
fi

if [ ! -x "$CLEAN_LOG_SCRIPT" ]; then
  echo "Error: $CLEAN_LOG_SCRIPT not found or not executable." >&2
  exit 1
fi

# Run dynamo image cleanup
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Calling $CLEANUP_OLD_DYNAMO_IMAGES_SCRIPT ${CLEANUP_OLD_DYNAMO_IMAGES_ARGS[*]}"
"$CLEANUP_OLD_DYNAMO_IMAGES_SCRIPT" "${CLEANUP_OLD_DYNAMO_IMAGES_ARGS[@]}"

# Run log cleanup
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Calling $CLEAN_LOG_SCRIPT ${CLEAN_LOG_ARGS[*]}"
"$CLEAN_LOG_SCRIPT" "${CLEAN_LOG_ARGS[@]}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] clean_system.sh done"
