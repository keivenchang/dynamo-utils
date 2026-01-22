#!/bin/bash
# Generic backup script using rsync with versioned backup history
# Excludes cache files, build artifacts, and other unnecessary data
# Keeps changed/deleted files in timestamped backup directories

set -e

# Default values
DEFAULT_SOURCE_DIR="$HOME/nvidia"
DEFAULT_DEST_DIR="/mnt/sda/keivenc/dynamo"

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default log location (cron-friendly):
#   $DYNAMO_HOME/logs/YYYY-MM-DD/backup.log
# Can be overridden via LOG_FILE env var.
UTILS_DIR="$(dirname "$SCRIPT_DIR")"
DYNAMO_HOME="${DYNAMO_HOME:-$(dirname "$UTILS_DIR")}"
LOGS_DIR="${LOGS_DIR:-$DYNAMO_HOME/logs}"
TODAY="$(date +%Y-%m-%d)"
DAY_LOG_DIR="$LOGS_DIR/$TODAY"
mkdir -p "$DAY_LOG_DIR"
LOG_FILE="${LOG_FILE:-$DAY_LOG_DIR/backup.log}"

# Create timestamp for this backup run
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Parse command line arguments
SOURCE_DIR=""
DEST_DIR=""

show_usage() {
    cat << EOF
Usage: $0 --input-path <source> --output-path <destination>

Both --input-path and --output-path are required.

This script performs incremental backups using rsync. Changed or deleted files
are preserved in timestamped backup history directories before being updated.

Options:
    --input-path <path>     Source directory to backup
    --output-path <path>    Destination directory for backup
    -h, --help              Show this help message

Backup History:
    Changed/deleted files are stored in: <output-path-parent>/backup_history/YYYYMMDD_HHMMSS/
    Example: If backing up to /mnt/sda/keivenc/dynamo, history is stored in:
             /mnt/sda/keivenc/backup_history/20251217_172633/

Example:
    $0 --input-path $DEFAULT_SOURCE_DIR --output-path $DEFAULT_DEST_DIR
EOF
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-path)
            SOURCE_DIR="$2"
            shift 2
            ;;
        --output-path)
            DEST_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            ;;
        *)
            echo "Error: Unknown option: $1"
            show_usage
            ;;
    esac
done

# Validate that both parameters are provided
if [ -z "$SOURCE_DIR" ] || [ -z "$DEST_DIR" ]; then
    if [ -z "$SOURCE_DIR" ]; then
        echo "Error: --input-path is required"
    fi
    if [ -z "$DEST_DIR" ]; then
        echo "Error: --output-path is required"
    fi
    echo ""
    show_usage
fi

# Validate source path exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory does not exist: $SOURCE_DIR"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Setup backup history directory
DEST_PARENT="$(dirname "$DEST_DIR")"
BACKUP_HISTORY_DIR="$DEST_PARENT/backup_history/$TIMESTAMP"

echo -e "${GREEN}Starting backup: $SOURCE_DIR -> $DEST_DIR${NC}"
echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting backup: $SOURCE_DIR -> $DEST_DIR" >> "$LOG_FILE"
echo -e "${CYAN}Backup history: $BACKUP_HISTORY_DIR${NC}"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Create backup history directory
mkdir -p "$BACKUP_HISTORY_DIR"

# Run rsync with exclusions and backup for changed/deleted files
rsync -av --delete --backup --backup-dir="$BACKUP_HISTORY_DIR" \
  --exclude='*.pyc' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='.mypy_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='*.egg-info/' \
  --exclude='.venv/' \
  --exclude='venv/' \
  --exclude='node_modules/' \
  --exclude='target/' \
  --exclude='build/' \
  --exclude='dist/' \
  --exclude='*.o' \
  --exclude='*.so' \
  --exclude='*.dylib' \
  --exclude='.DS_Store' \
  --exclude='.git/objects/' \
  --exclude='.git/lfs/' \
  --exclude='.git/FETCH_HEAD' \
  --exclude='.git/HEAD' \
  --exclude='.git/index' \
  --exclude='.git/logs/' \
  --exclude='*.log' \
  --exclude='/tmp/' \
  --exclude='*.tmp' \
  --exclude='*.swp' \
  --exclude='*.swo' \
  --exclude='*~' \
  --exclude='*.bak' \
  --exclude='core' \
  --exclude='core.*' \
  "$SOURCE_DIR/" "$DEST_DIR/" 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
  echo -e "${GREEN}Backup completed successfully!${NC}"
  echo "$(date '+%Y-%m-%d %H:%M:%S') - Backup completed successfully" >> "$LOG_FILE"

  # Check if any files were backed up to history
  BACKUP_HISTORY_COUNT=$(find "$BACKUP_HISTORY_DIR" -type f 2>/dev/null | wc -l)
  if [ "$BACKUP_HISTORY_COUNT" -gt 0 ]; then
    BACKUP_HISTORY_SIZE=$(du -sh "$BACKUP_HISTORY_DIR" 2>/dev/null | awk '{print $1}')
    echo -e "${CYAN}Changed/deleted files backed up: $BACKUP_HISTORY_COUNT files ($BACKUP_HISTORY_SIZE)${NC}"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Backup history: $BACKUP_HISTORY_COUNT files in $BACKUP_HISTORY_DIR" >> "$LOG_FILE"
  else
    echo -e "${CYAN}No files changed or deleted - backup history empty${NC}"
    # Remove empty backup history directory
    rmdir "$BACKUP_HISTORY_DIR" 2>/dev/null || true
  fi
else
  echo -e "${RED}Backup failed with exit code $EXIT_CODE${NC}"
  echo "$(date '+%Y-%m-%d %H:%M:%S') - Backup failed with exit code $EXIT_CODE" >> "$LOG_FILE"
  exit $EXIT_CODE
fi

# Show backup size
echo -e "${YELLOW}Backup statistics:${NC}"
du -sh "$DEST_DIR" 2>/dev/null || echo "Could not calculate backup size"
echo -e "${YELLOW}Backup history location: $DEST_PARENT/backup_history/${NC}"
echo "Log file: $LOG_FILE"
