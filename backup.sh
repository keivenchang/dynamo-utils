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
COMPRESS=false
UNCOMPRESS=false
REMOVE_AFTER_DAYS=""
DRY_RUN=false
DRYRUN=""  # Will be set to "echo" in dry-run mode
BACKUP_ONLY=false  # Skip rsync, only do compress/uncompress/remove operations

show_usage() {
    cat << EOF
Usage: $0 --input-path <source> --output-path <destination> [options]

Both --input-path and --output-path are required.

This script performs incremental backups using rsync. Changed or deleted files
are preserved in timestamped backup history directories before being updated.

Options:
    --input-path <path>         Source directory to backup
    --output-path <path>        Destination directory for backup
    --compress                  Compress yesterday and all prior days' backup history to .tgz
    --uncompress                Uncompress all .tgz archives back to directories
    --remove-after-days <N>     Remove backup history older than N days (default: no removal)
    --backup                    Skip rsync, only perform compress/uncompress/remove operations
    --dry-run, --dryrun         Show what would be done without making changes
    -h, --help                  Show this help message

Backup History:
    Changed/deleted files are stored in: <output-path-parent>/backup_history/YYYYMMDD_HHMMSS/
    Example: If backing up to /mnt/sda/keivenc/dynamo, history is stored in:
             /mnt/sda/keivenc/backup_history/20251217_172633/

Example:
    $0 --input-path $DEFAULT_SOURCE_DIR --output-path $DEFAULT_DEST_DIR
    $0 --input-path ~/data --output-path /mnt/backup --compress --remove-after-days 30
    $0 --input-path ~/data --output-path /mnt/backup --uncompress --dry-run
    $0 --output-path /mnt/backup --backup --compress --dry-run
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
        --compress)
            COMPRESS=true
            shift
            ;;
        --uncompress)
            UNCOMPRESS=true
            shift
            ;;
        --backup)
            BACKUP_ONLY=true
            shift
            ;;
        --remove-after-days)
            REMOVE_AFTER_DAYS="$2"
            shift 2
            ;;
        --dry-run|--dryrun)
            DRY_RUN=true
            shift
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

# Validate that both parameters are provided (unless --backup mode)
if [ "$BACKUP_ONLY" = false ]; then
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
else
    # In backup-only mode, only output-path is required
    if [ -z "$DEST_DIR" ]; then
        echo "Error: --output-path is required"
        echo ""
        show_usage
    fi
fi

# Validate --remove-after-days is a positive integer if provided
if [ -n "$REMOVE_AFTER_DAYS" ]; then
    if ! [[ "$REMOVE_AFTER_DAYS" =~ ^[0-9]+$ ]] || [ "$REMOVE_AFTER_DAYS" -le 0 ]; then
        echo "Error: --remove-after-days must be a positive integer"
        exit 1
    fi
fi

# Validate that --compress and --uncompress are not both specified
if [ "$COMPRESS" = true ] && [ "$UNCOMPRESS" = true ]; then
    echo "Error: --compress and --uncompress cannot be used together"
    exit 1
fi

# Validate source path exists (skip in backup-only mode)
if [ "$BACKUP_ONLY" = false ] && [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory does not exist: $SOURCE_DIR"
    exit 1
fi

# Setup backup history directory
DEST_PARENT="$(dirname "$DEST_DIR")"
BACKUP_HISTORY_DIR="$DEST_PARENT/backup_history/$TIMESTAMP"

if [ "$BACKUP_ONLY" = true ]; then
    echo "Backup operations only mode (skipping rsync)"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Backup operations only mode" >> "$LOG_FILE"
else
    echo "Starting backup: $SOURCE_DIR -> $DEST_DIR"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting backup: $SOURCE_DIR -> $DEST_DIR" >> "$LOG_FILE"
    echo "Backup history: $BACKUP_HISTORY_DIR"
fi

if [ "$DRY_RUN" = true ]; then
    echo "*** DRY RUN MODE - No changes will be made ***"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - DRY RUN MODE" >> "$LOG_FILE"
    DRYRUN="echo +"
fi

if [ "$BACKUP_ONLY" = false ]; then
# Create destination directory if it doesn't exist
$DRYRUN mkdir -p "$DEST_DIR"

# Create backup history directory
$DRYRUN mkdir -p "$BACKUP_HISTORY_DIR"

# Build rsync command
RSYNC_CMD="rsync -av --delete --backup --backup-dir=\"$BACKUP_HISTORY_DIR\""
if [ "$DRY_RUN" = true ]; then
    RSYNC_CMD="$RSYNC_CMD --dry-run"
fi

# Show the command that will be executed
echo "+ $RSYNC_CMD \\"
echo "  --exclude='*.pyc' --exclude='__pycache__/' --exclude='.pytest_cache/' \\"
echo "  --exclude='.mypy_cache/' --exclude='.ruff_cache/' --exclude='*.egg-info/' \\"
echo "  --exclude='.venv/' --exclude='venv/' --exclude='node_modules/' \\"
echo "  --exclude='target/' --exclude='build/' --exclude='dist/' \\"
echo "  --exclude='*.o' --exclude='*.so' --exclude='*.dylib' --exclude='.DS_Store' \\"
echo "  --exclude='.git/objects/' --exclude='.git/lfs/' --exclude='.git/FETCH_HEAD' \\"
echo "  --exclude='.git/HEAD' --exclude='.git/index' --exclude='.git/logs/' \\"
echo "  --exclude='*.log' --exclude='/tmp/' --exclude='*.tmp' \\"
echo "  --exclude='*.swp' --exclude='*.swo' --exclude='*~' --exclude='*.bak' \\"
echo "  --exclude='core' --exclude='core.*' \\"
echo "  --exclude='dynamo_latest/*.html' --exclude='dynamo_ci/*.html' --exclude='commits/*.html' \\"
echo "  \"$SOURCE_DIR/\" \"$DEST_DIR/\""

# Run rsync with exclusions and backup for changed/deleted files
eval "$RSYNC_CMD" \
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
  --exclude='dynamo_latest/*.html' \
  --exclude='dynamo_ci/*.html' \
  --exclude='commits/*.html' \
  "$SOURCE_DIR/" "$DEST_DIR/" 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
fi  # End of BACKUP_ONLY check

if [ "$BACKUP_ONLY" = true ] || [ $EXIT_CODE -eq 0 ]; then
  if [ "$BACKUP_ONLY" = false ]; then
    echo "Backup completed successfully!"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Backup completed successfully" >> "$LOG_FILE"

    # Check if any files were backed up to history
    BACKUP_HISTORY_COUNT=$(find "$BACKUP_HISTORY_DIR" -type f 2>/dev/null | wc -l)
    if [ "$BACKUP_HISTORY_COUNT" -gt 0 ]; then
      BACKUP_HISTORY_SIZE=$(du -sh "$BACKUP_HISTORY_DIR" 2>/dev/null | awk '{print $1}')
      echo "Changed/deleted files backed up: $BACKUP_HISTORY_COUNT files ($BACKUP_HISTORY_SIZE)"
      echo "$(date '+%Y-%m-%d %H:%M:%S') - Backup history: $BACKUP_HISTORY_COUNT files in $BACKUP_HISTORY_DIR" >> "$LOG_FILE"
    else
      echo "No files changed or deleted - backup history empty"
      # Remove empty backup history directory
      if [ "$DRY_RUN" = true ]; then
          echo "[DRY RUN] Would remove empty directory: $BACKUP_HISTORY_DIR"
      else
          rmdir "$BACKUP_HISTORY_DIR" 2>/dev/null || true
      fi
    fi
  fi

  # Compress old backup history directories (yesterday and prior)
  if [ "$COMPRESS" = true ]; then
    echo "Compressing old backup history directories by date..."
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting compression of old backups" >> "$LOG_FILE"
    
    # Get today's date in YYYYMMDD format for comparison
    TODAY_DATE=$(date '+%Y%m%d')
    
    # Find all backup history directories
    BACKUP_HISTORY_ROOT="$DEST_PARENT/backup_history"
    if [ -d "$BACKUP_HISTORY_ROOT" ]; then
      # Get list of unique dates from directories (excluding today)
      DATES_TO_COMPRESS=$(find "$BACKUP_HISTORY_ROOT" -maxdepth 1 -type d -name '[0-9]*_[0-9]*' | \
        sed 's|.*/||' | \
        cut -d'_' -f1 | \
        sort -u | \
        while read date; do
          if [ "$date" -lt "$TODAY_DATE" ]; then
            echo "$date"
          fi
        done)
      
      # Compress each date
      for backup_date in $DATES_TO_COMPRESS; do
        # Skip if already compressed for this date
        if [ -f "$BACKUP_HISTORY_ROOT/${backup_date}.tgz" ]; then
          echo "Already compressed: ${backup_date}.tgz"
          continue
        fi
        
        # Find all directories for this date
        DATE_DIRS=$(find "$BACKUP_HISTORY_ROOT" -maxdepth 1 -type d -name "${backup_date}_*" | sed 's|.*/||' | sort)
        
        if [ -z "$DATE_DIRS" ]; then
          continue
        fi
        
        DIR_COUNT=$(echo "$DATE_DIRS" | wc -l)
        
        if [ "$DRY_RUN" = true ]; then
          echo "[DRY RUN] Would compress $DIR_COUNT directories for date $backup_date -> ${backup_date}.tgz"
          echo "$DATE_DIRS" | while read dirname; do
            echo "  - $dirname"
          done
          set -x
          : tar -czf "$BACKUP_HISTORY_ROOT/${backup_date}.tgz.working" -C "$BACKUP_HISTORY_ROOT" $DATE_DIRS
          : mv "$BACKUP_HISTORY_ROOT/${backup_date}.tgz.working" "$BACKUP_HISTORY_ROOT/${backup_date}.tgz"
          : rm -rf $DATE_DIRS
          { set +x; } 2>/dev/null
        else
          # Compress to .tgz.working first, then move to final name when complete
          echo "Compressing $DIR_COUNT directories for date $backup_date:"
          echo "$DATE_DIRS" | while read dirname; do
            echo "  - $dirname"
          done
          
          set -x
          tar -czf "$BACKUP_HISTORY_ROOT/${backup_date}.tgz.working" -C "$BACKUP_HISTORY_ROOT" $DATE_DIRS
          TAR_EXIT=$?
          { set +x; } 2>/dev/null
          
          if [ $TAR_EXIT -eq 0 ]; then
            # Move to final name
            set -x
            mv "$BACKUP_HISTORY_ROOT/${backup_date}.tgz.working" "$BACKUP_HISTORY_ROOT/${backup_date}.tgz"
            { set +x; } 2>/dev/null
            
            # Remove the original directories after successful compression
            for dirname in $DATE_DIRS; do
              set -x
              rm -rf "$BACKUP_HISTORY_ROOT/$dirname"
              { set +x; } 2>/dev/null
            done
            
            COMPRESSED_SIZE=$(du -sh "$BACKUP_HISTORY_ROOT/${backup_date}.tgz" 2>/dev/null | awk '{print $1}')
            echo "Compressed to: ${backup_date}.tgz ($COMPRESSED_SIZE)"
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Compressed $DIR_COUNT directories for $backup_date to ${backup_date}.tgz" >> "$LOG_FILE"
          else
            echo "Failed to compress directories for date: $backup_date"
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Failed to compress $backup_date" >> "$LOG_FILE"
            # Clean up partial archive if it exists
            set -x
            rm -f "$BACKUP_HISTORY_ROOT/${backup_date}.tgz.working"
            { set +x; } 2>/dev/null
          fi
        fi
      done
    fi
  fi

  # Uncompress backup history archives
  if [ "$UNCOMPRESS" = true ]; then
    echo "Uncompressing backup history archives..."
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting uncompression of backup archives" >> "$LOG_FILE"
    
    BACKUP_HISTORY_ROOT="$DEST_PARENT/backup_history"
    if [ -d "$BACKUP_HISTORY_ROOT" ]; then
      # Find all .tgz archives (YYYYMMDD.tgz format)
      for archive in "$BACKUP_HISTORY_ROOT"/[0-9]*.tgz; do
        if [ -f "$archive" ]; then
          archive_filename=$(basename "$archive" .tgz)
          archive_size=$(du -sh "$archive" 2>/dev/null | awk '{print $1}')
          
          if [ "$DRY_RUN" = true ]; then
            echo "[DRY RUN] Would uncompress: $archive ($archive_size)"
            set -x
            : tar -xzf "$archive" -C "$BACKUP_HISTORY_ROOT"
            : rm -f "$archive"
            { set +x; } 2>/dev/null
          else
            echo "Uncompressing: $archive ($archive_size)"
            set -x
            tar -xzf "$archive" -C "$BACKUP_HISTORY_ROOT"
            TAR_EXIT=$?
            { set +x; } 2>/dev/null
            
            if [ $TAR_EXIT -eq 0 ]; then
              # Remove the archive after successful extraction
              set -x
              rm -f "$archive"
              { set +x; } 2>/dev/null
              
              # Count extracted directories
              EXTRACTED_COUNT=$(find "$BACKUP_HISTORY_ROOT" -maxdepth 1 -type d -name "${archive_filename}_*" | wc -l)
              echo "Uncompressed $EXTRACTED_COUNT directories from: $archive_filename.tgz"
              echo "$(date '+%Y-%m-%d %H:%M:%S') - Uncompressed $archive_filename.tgz" >> "$LOG_FILE"
            else
              echo "Failed to uncompress: $archive"
              echo "$(date '+%Y-%m-%d %H:%M:%S') - Failed to uncompress $archive_filename.tgz" >> "$LOG_FILE"
            fi
          fi
        fi
      done
    else
      echo "Backup history directory does not exist: $BACKUP_HISTORY_ROOT"
    fi
  fi

  # Remove old backup history directories/archives
  if [ -n "$REMOVE_AFTER_DAYS" ]; then
    echo "Removing backup history older than $REMOVE_AFTER_DAYS days..."
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Removing backups older than $REMOVE_AFTER_DAYS days" >> "$LOG_FILE"
    
    # Calculate cutoff date
    CUTOFF_DATE=$(date -d "$REMOVE_AFTER_DAYS days ago" '+%Y%m%d' 2>/dev/null || date -v-"${REMOVE_AFTER_DAYS}d" '+%Y%m%d' 2>/dev/null)
    
    BACKUP_HISTORY_ROOT="$DEST_PARENT/backup_history"
    if [ -d "$BACKUP_HISTORY_ROOT" ]; then
      # Remove old directories
      for backup_dir in "$BACKUP_HISTORY_ROOT"/*/; do
        if [ -d "$backup_dir" ]; then
          backup_dirname=$(basename "$backup_dir")
          backup_date=$(echo "$backup_dirname" | cut -d'_' -f1)
          
          if [ "$backup_date" -lt "$CUTOFF_DATE" ]; then
            if [ "$DRY_RUN" = true ]; then
              echo "[DRY RUN] Would remove directory: $backup_dir"
              set -x
              : rm -rf "$backup_dir"
              { set +x; } 2>/dev/null
            else
              echo "Removing old directory: $backup_dir"
              set -x
              rm -rf "$backup_dir"
              { set +x; } 2>/dev/null
              echo "$(date '+%Y-%m-%d %H:%M:%S') - Removed old directory $backup_dirname" >> "$LOG_FILE"
            fi
          fi
        fi
      done
      
      # Remove old .tgz archives
      for backup_archive in "$BACKUP_HISTORY_ROOT"/*.tgz; do
        if [ -f "$backup_archive" ]; then
          backup_filename=$(basename "$backup_archive" .tgz)
          backup_date=$(echo "$backup_filename" | cut -d'_' -f1)
          
          if [ "$backup_date" -lt "$CUTOFF_DATE" ]; then
            if [ "$DRY_RUN" = true ]; then
              echo "[DRY RUN] Would remove archive: $backup_archive"
              set -x
              : rm -f "$backup_archive"
              { set +x; } 2>/dev/null
            else
              ARCHIVE_SIZE=$(du -sh "$backup_archive" 2>/dev/null | awk '{print $1}')
              echo "Removing old archive: $backup_archive ($ARCHIVE_SIZE)"
              set -x
              rm -f "$backup_archive"
              { set +x; } 2>/dev/null
              echo "$(date '+%Y-%m-%d %H:%M:%S') - Removed old archive $backup_filename.tgz" >> "$LOG_FILE"
            fi
          fi
        fi
      done
    fi
  fi
else
  if [ "$BACKUP_ONLY" = false ]; then
    echo "Backup failed with exit code $EXIT_CODE"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Backup failed with exit code $EXIT_CODE" >> "$LOG_FILE"
    exit $EXIT_CODE
  fi
fi

# Show backup size
if [ "$BACKUP_ONLY" = false ]; then
  echo "Backup statistics:"
  du -sh "$DEST_DIR" 2>/dev/null || echo "Could not calculate backup size"
fi
echo "Backup history location: $DEST_PARENT/backup_history/"
echo "Log file: $LOG_FILE"
