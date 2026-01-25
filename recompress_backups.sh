#!/bin/bash
# Script to recompress backup archives while applying .rsyncrules exclusions
# This reduces archive sizes by removing unnecessary files

set -e

# Default values
BACKUP_HISTORY_DIR=""
RSYNCRULES_FILE=""
DRY_RUN=false

show_usage() {
    cat << EOF
Usage: $0 --backup-history <path> --rsyncrules <path> [options]

This script recompresses existing .tgz backup archives while applying exclusion
rules from .rsyncrules. This helps reduce archive sizes by removing unnecessary
files that should have been excluded.

Options:
    --backup-history <path>     Path to backup_history directory (e.g., /mnt/sda/keivenc/backup_history)
    --rsyncrules <path>         Path to .rsyncrules file (e.g., ~/dynamo/.rsyncrules)
    --dry-run, --dryrun         Show what would be done without making changes
    -h, --help                  Show this help message

Example:
    $0 --backup-history /mnt/sda/keivenc/backup_history --rsyncrules ~/dynamo/.rsyncrules
    $0 --backup-history /mnt/sda/keivenc/backup_history --rsyncrules ~/dynamo/.rsyncrules --dry-run
EOF
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --backup-history)
            BACKUP_HISTORY_DIR="$2"
            shift 2
            ;;
        --rsyncrules)
            RSYNCRULES_FILE="$2"
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

# Validate required parameters
if [ -z "$BACKUP_HISTORY_DIR" ] || [ -z "$RSYNCRULES_FILE" ]; then
    if [ -z "$BACKUP_HISTORY_DIR" ]; then
        echo "Error: --backup-history is required"
    fi
    if [ -z "$RSYNCRULES_FILE" ]; then
        echo "Error: --rsyncrules is required"
    fi
    echo ""
    show_usage
fi

# Validate paths exist
if [ ! -d "$BACKUP_HISTORY_DIR" ]; then
    echo "Error: Backup history directory does not exist: $BACKUP_HISTORY_DIR"
    exit 1
fi

if [ ! -f "$RSYNCRULES_FILE" ]; then
    echo "Error: .rsyncrules file does not exist: $RSYNCRULES_FILE"
    exit 1
fi

if [ "$DRY_RUN" = true ]; then
    echo "*** DRY RUN MODE - No changes will be made ***"
fi

# Build tar exclude options from .rsyncrules file
TAR_EXCLUDE_OPTS=""
echo "Reading exclusions from: $RSYNCRULES_FILE"
while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines and comments
    if [[ -n "$line" && ! "$line" =~ ^[[:space:]]*# ]]; then
        # Skip include patterns (lines starting with +) - tar doesn't support them
        if [[ "$line" =~ ^\+ ]]; then
            continue
        fi
        # Handle explicit exclude patterns (lines starting with -)
        if [[ "$line" =~ ^\- ]]; then
            pattern=$(echo "$line" | sed 's/^- *//')
        else
            # Regular exclude pattern (no prefix)
            pattern="$line"
        fi
        # Remove trailing slash for directories (tar uses --exclude pattern)
        pattern="${pattern%/}"
        TAR_EXCLUDE_OPTS="$TAR_EXCLUDE_OPTS --exclude='$pattern'"
    fi
done < "$RSYNCRULES_FILE"

echo "Processing archives in: $BACKUP_HISTORY_DIR"
echo ""

# Find all .tgz archives
ARCHIVE_COUNT=0
PROCESSED_COUNT=0
FAILED_COUNT=0
TOTAL_SIZE_BEFORE=0
TOTAL_SIZE_AFTER=0

for archive in "$BACKUP_HISTORY_DIR"/*.tgz; do
    if [ ! -f "$archive" ]; then
        continue
    fi
    
    ARCHIVE_COUNT=$((ARCHIVE_COUNT + 1))
    archive_name=$(basename "$archive")
    
    # Get original size
    size_before=$(stat -c%s "$archive" 2>/dev/null || stat -f%z "$archive" 2>/dev/null)
    size_before_human=$(du -sh "$archive" 2>/dev/null | awk '{print $1}')
    TOTAL_SIZE_BEFORE=$((TOTAL_SIZE_BEFORE + size_before))
    
    echo "[$ARCHIVE_COUNT] Processing: $archive_name ($size_before_human)"
    
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] Would extract, recompress with exclusions, and replace"
        set -x
        : mkdir -p "/tmp/recompress_$$"
        : tar -I pigz -xf "$archive" -C "/tmp/recompress_$$"
        : tar -I pigz -cf "$archive.new" -C "/tmp/recompress_$$" $TAR_EXCLUDE_OPTS .
        : mv "$archive.new" "$archive"
        : rm -rf "/tmp/recompress_$$"
        { set +x; } 2>/dev/null
        PROCESSED_COUNT=$((PROCESSED_COUNT + 1))
    else
        # Create temporary directory for extraction
        TEMP_DIR="/tmp/recompress_$$_${ARCHIVE_COUNT}"
        mkdir -p "$TEMP_DIR"
        
        # Extract archive (using pigz for parallel decompression)
        set -x
        tar -I pigz -xf "$archive" -C "$TEMP_DIR"
        EXTRACT_EXIT=$?
        { set +x; } 2>/dev/null
        
        if [ $EXTRACT_EXIT -ne 0 ]; then
            echo "  ERROR: Failed to extract $archive_name"
            rm -rf "$TEMP_DIR"
            FAILED_COUNT=$((FAILED_COUNT + 1))
            continue
        fi
        
        # Recompress with exclusions (using pigz for parallel compression)
        set -x
        eval "tar -I pigz -cf \"$archive.new\" -C \"$TEMP_DIR\" $TAR_EXCLUDE_OPTS ."
        TAR_EXIT=$?
        { set +x; } 2>/dev/null
        
        if [ $TAR_EXIT -ne 0 ]; then
            echo "  ERROR: Failed to recompress $archive_name"
            rm -rf "$TEMP_DIR"
            rm -f "$archive.new"
            FAILED_COUNT=$((FAILED_COUNT + 1))
            continue
        fi
        
        # Get new size and compare
        size_after=$(stat -c%s "$archive.new" 2>/dev/null || stat -f%z "$archive.new" 2>/dev/null)
        size_after_human=$(du -sh "$archive.new" 2>/dev/null | awk '{print $1}')
        TOTAL_SIZE_AFTER=$((TOTAL_SIZE_AFTER + size_after))
        
        size_diff=$((size_before - size_after))
        if [ $size_diff -gt 0 ]; then
            size_diff_human=$(numfmt --to=iec $size_diff 2>/dev/null || echo "${size_diff}B")
            percent_saved=$(awk "BEGIN {printf \"%.1f\", ($size_diff / $size_before) * 100}")
            echo "  Reduced: $size_before_human -> $size_after_human (saved $size_diff_human, ${percent_saved}%)"
        elif [ $size_diff -lt 0 ]; then
            size_diff_human=$(numfmt --to=iec $((-size_diff)) 2>/dev/null || echo "$((size_diff))B")
            echo "  Size increased by $size_diff_human (no unnecessary files found)"
        else
            echo "  No size change"
        fi
        
        # Replace original with new archive
        set -x
        mv "$archive.new" "$archive"
        { set +x; } 2>/dev/null
        
        # Clean up temp directory
        rm -rf "$TEMP_DIR"
        
        PROCESSED_COUNT=$((PROCESSED_COUNT + 1))
    fi
    
    echo ""
done

# Summary
echo "========================================="
echo "Recompression Summary:"
echo "  Total archives found: $ARCHIVE_COUNT"
echo "  Successfully processed: $PROCESSED_COUNT"
if [ $FAILED_COUNT -gt 0 ]; then
    echo "  Failed: $FAILED_COUNT"
fi

if [ "$DRY_RUN" = false ] && [ $PROCESSED_COUNT -gt 0 ]; then
    total_before_human=$(numfmt --to=iec $TOTAL_SIZE_BEFORE 2>/dev/null || echo "${TOTAL_SIZE_BEFORE}B")
    total_after_human=$(numfmt --to=iec $TOTAL_SIZE_AFTER 2>/dev/null || echo "${TOTAL_SIZE_AFTER}B")
    total_saved=$((TOTAL_SIZE_BEFORE - TOTAL_SIZE_AFTER))
    
    if [ $total_saved -gt 0 ]; then
        total_saved_human=$(numfmt --to=iec $total_saved 2>/dev/null || echo "${total_saved}B")
        percent_saved=$(awk "BEGIN {printf \"%.1f\", ($total_saved / $TOTAL_SIZE_BEFORE) * 100}")
        echo "  Total size before: $total_before_human"
        echo "  Total size after: $total_after_human"
        echo "  Total space saved: $total_saved_human (${percent_saved}%)"
    else
        echo "  No space saved (archives may have contained only needed files)"
    fi
fi
echo "========================================="
