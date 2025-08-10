#!/bin/bash

# Script to sync cursor-related files (.cursorrules, .cursorignore, etc.) and devcontainer directory to all subdirectories when they change
# Usage: This script should be run via crontab to monitor changes
# Options: --dryrun or --dry-run to show what would be done without actually doing it
#          --force to run sync regardless of MD5 hash changes

# Define cursor-related files to sync (add new files here as needed)
# Format: [source_file]=target_filename
declare -A SRC_CURSOR_FILES=(
    ["$HOME/nvidia/master.devcontainer.json"]=".devcontainer/keiven/devcontainer.json"
    ["$HOME/nvidia/master.cursorrules"]=".cursorrules"
    ["$HOME/nvidia/master.cursorignore"]=".cursorignore"
)

DEVCONTAINER_SOURCE_DIR="$HOME/nvidia"
TEMP_FILE="$HOME/nvidia/.cursorrules.tmp"
LOG_FILE="$HOME/nvidia/.cursorrules_sync.log"

# Check for flags
DRY_RUN=false
FORCE=false
for arg in "$@"; do
    case $arg in
        --dryrun|--dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
    esac
done

# Function to log messages with timestamp
log_message() {
    local message="$(date '+%Y-%m-%d %H:%M:%S') - $1"
    echo "$message" >> "$LOG_FILE"
    if [ "$DRY_RUN" = true ]; then
        echo "[DRYRUN] $1"
    fi
}

# Check if cursor files exist
for source_file in "${!SRC_CURSOR_FILES[@]}"; do
    if [ ! -f "$source_file" ]; then
        log_message "ERROR: Cursor file not found at $source_file"
        exit 1
    fi
done

# Create a hash of all cursor files and devcontainer profile files
SRC_CURSOR_FILES_HASH=""
for source_file in "${!SRC_CURSOR_FILES[@]}"; do
    FILE_HASH=$(md5sum "$source_file" | cut -d' ' -f1)
    SRC_CURSOR_FILES_HASH="${SRC_CURSOR_FILES_HASH}_${FILE_HASH}"
done

CURRENT_HASH="${SRC_CURSOR_FILES_HASH}"

# Check if we have a previous hash stored
if [ -f "$TEMP_FILE" ]; then
    PREVIOUS_HASH=$(cat "$TEMP_FILE")
else
    PREVIOUS_HASH=""
fi

# If hash unchanged and not forced, exit early
if [ "$CURRENT_HASH" = "$PREVIOUS_HASH" ] && [ "$FORCE" = false ]; then
    log_message "DEBUG: Cursor files and devcontainer directory unchanged, no sync needed."
    exit 0
fi

if [ "$FORCE" = true ]; then
    log_message "INFO: Force flag detected, syncing regardless of changes..."
fi

log_message "INFO: Cursor files or devcontainer directory has changed, syncing to subdirectories..."

# Find all dynamo* subdirectories and copy cursor files and devcontainer directory to them
SYNC_COUNT=0
for dir in "$HOME/nvidia"/dynamo*; do
    if [ ! -d "$dir" ]; then
        continue
    fi
    
    # Skip dynamo-utils directory
    DIR_NAME=$(basename "$dir")
    if [ "$DIR_NAME" = "dynamo-utils" ]; then
        log_message "INFO: Skipping dynamo-utils directory"
        continue
    fi
    
    # Copy each cursor file with warning comment
    for source_file in "${!SRC_CURSOR_FILES[@]}"; do
        target_filename="${SRC_CURSOR_FILES[$source_file]}"
        target_path="$dir/$target_filename"
        target_dir=$(dirname "$target_path")
        
        if [ "$DRY_RUN" = true ]; then
            if [ "$target_dir" != "$dir" ]; then
                log_message "  mkdir -p \"$target_dir\""
            fi
            log_message "  cp \"$source_file\" \"$target_path\""
        else
            # Create target directory if needed
            if [ "$target_dir" != "$dir" ]; then
                mkdir -p "$target_dir"
            fi
            
            TEMP_CURSORFILE=$(mktemp)
            
            # Check if this is a JSON file
            if [[ "$source_file" == *.json ]] || [[ "$target_filename" == *.json ]]; then
                # For JSON files: customize name field only
                # Extract directory name for name customization
                DIR_NAME=$(basename "$dir")
                
                # Apply customizations to JSON file
                # 1. Replace display name with directory-specific name
                # 2. Replace container name with directory-specific name
                sed "s|\"name\": \"NVIDIA Dynamo Development\"|\"name\": \"[$DIR_NAME] $USER Dev Container\"|g" "$source_file" | \
                sed "s|\"dynamo-dev-container\"|\"$DIR_NAME-$USER-devcontainer\"|g" > "$TEMP_CURSORFILE"
                log_message "SUCCESS: JSON copied with name customized for $DIR_NAME"
            else
                # For non-JSON files: add header and copy content
                echo "# DO NOT EDIT - THIS IS AN AUTOMATICALLY GENERATED COPY" > "$TEMP_CURSORFILE"
                echo "# Original file: $source_file" >> "$TEMP_CURSORFILE"
                echo "# Last updated: $(date)" >> "$TEMP_CURSORFILE"
                echo "" >> "$TEMP_CURSORFILE"
                cat "$source_file" >> "$TEMP_CURSORFILE"
            fi
            
            # Copy the temporary file to the destination (cp will overwrite existing file)
            cp "$TEMP_CURSORFILE" "$target_path"
            if [ $? -eq 0 ]; then
                log_message "SUCCESS: Copied $target_filename to $dir"
            else
                log_message "ERROR: Failed to copy $target_filename to $dir"
            fi
            
            # Clean up temporary file
            rm -f "$TEMP_CURSORFILE"
        fi
    done

    # Handle build directory management for dynamo* directories
    if [[ "$DIR_NAME" == dynamo* ]]; then
        if [ "$DRY_RUN" = true ]; then
            log_message "  Would manage build directory for $DIR_NAME"
        else
            # Ensure .build/target exists
            mkdir -p "$dir/.build/target"
            
            # If target is a real directory (not symlink), move it and create symlink
            if [ -d "$dir/target" ] && [ ! -L "$dir/target" ]; then
                rm -rf "$dir/.build/target"/* 2>/dev/null
                if [ "$(ls -A "$dir/target" 2>/dev/null)" ]; then
                    mv "$dir/target"/* "$dir/target"/.* "$dir/.build/target/" 2>/dev/null
                fi
                rm -rf "$dir/target"
                ln -s ".build/target" "$dir/target"
                log_message "SUCCESS: Moved target to .build/target and created symlink in $DIR_NAME"
            fi
        fi
    fi
    
    ((SYNC_COUNT++))
done

# Store the new hash
if [ "$DRY_RUN" = true ]; then
    log_message "Would store new hash in $TEMP_FILE"
else
    echo "$CURRENT_HASH" > "$TEMP_FILE"
fi
log_message "INFO: Sync completed. Updated $SYNC_COUNT directories." 