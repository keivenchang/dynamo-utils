#!/bin/bash

# Script to sync development config files (.cursorrules, .cursorignore, devcontainer.json, etc.) to all subdirectories when they change
# Usage: This script should be run via crontab to monitor changes
# Options: --dryrun or --dry-run to show what would be done without actually doing it
#          --force to run sync regardless of MD5 hash changes
#          --silent to suppress all output (useful for cron jobs)

DEST_SOURCE_DIRS="${DEVCONTAINER_SRC_DIR:-$HOME/nvidia/dynamo}"

# Define development-related config files to sync (add new files here as needed)
# Format: [source_file]=target_filename
declare -A SRC_FILES=(
    ["devcontainer.json"]=".devcontainer/$USER/devcontainer.json"
    [".cursorrules"]=".cursorrules"
    [".cursorignore"]=".cursorignore"
)

# Get the absolute path of the script's directory and prepend it to SRC_FILES keys
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
declare -A TEMP_FILES
for key in "${!SRC_FILES[@]}"; do
    TEMP_FILES["$SCRIPT_DIR/$key"]="${SRC_FILES[$key]}"
done
SRC_FILES=()
for key in "${!TEMP_FILES[@]}"; do
    SRC_FILES["$key"]="${TEMP_FILES[$key]}"
done
unset TEMP_FILES

TEMP_FILE="/tmp/dev_configs.tmp"
LOG_FILE="/tmp/dev_configs_sync.log"

# Check for flags
DRYRUN=false
FORCE=false
SILENT=false
for arg in "$@"; do
    case $arg in
        --dryrun|--dry-run)
            DRYRUN=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --silent)
            SILENT=true
            shift
            ;;
    esac
done

# Function to handle dry run output
dry_run_echo() {
    if [ "$SILENT" = true ]; then
        return
    fi
    
    if [ "$DRYRUN" = true ]; then
        echo "[DRYRUN] $*"
    else
        echo "$*"
    fi
}

# Command wrapper that shows commands using set -x format and respects dry-run mode
cmd() {
    if [ "$DRYRUN" = true ]; then
        # Dry run mode: show command but don't execute
        if [ "$SILENT" = false ]; then
            echo "[DRYRUN] $*"
        fi
    else
        # Not dry run: execute the command
        if [ "$SILENT" = false ]; then
            # Show and execute
            ( set -x; "$@" )
        else
            # Execute silently
            "$@"
        fi
    fi
}

# Check if source files exist
for source_file in "${!SRC_FILES[@]}"; do
    if [ ! -f "$source_file" ]; then
        dry_run_echo "ERROR: Source file not found at $source_file"
        exit 1
    fi
done

# Create a hash of all development config files
SRC_FILES_HASH=""
for source_file in "${!SRC_FILES[@]}"; do
    FILE_HASH=$(md5sum "$source_file" | cut -d' ' -f1)
    SRC_FILES_HASH="${SRC_FILES_HASH}_${FILE_HASH}"
done

CURRENT_HASH="${SRC_FILES_HASH}"

# Check if we have a previous hash stored
if [ -f "$TEMP_FILE" ]; then
    PREVIOUS_HASH=$(cat "$TEMP_FILE")
else
    PREVIOUS_HASH=""
fi

# If hash unchanged and not forced, exit early
if [ "$CURRENT_HASH" = "$PREVIOUS_HASH" ] && [ "$FORCE" = false ]; then
    dry_run_echo "DEBUG: Development config files unchanged, no sync needed."
    exit 0
fi

if [ "$FORCE" = true ]; then
    dry_run_echo "INFO: Force flag detected, syncing regardless of changes..."
fi

dry_run_echo "INFO: Development config files have changed, syncing to subdirectories..."

# Find all dynamo* subdirectories and copy development config files to them
SYNC_COUNT=0
for dir in "$DEST_SOURCE_DIRS"*; do
    if [ ! -d "$dir" ]; then
        continue
    fi
    
    # Skip the script's own directory
    if [ "$dir" = "$SCRIPT_DIR" ]; then
        dry_run_echo "INFO: Skipping source directory $SCRIPT_DIR"
        continue
    fi
    
    dry_run_echo "INFO: Syncing $dir"

    # Copy each development config file with warning comment
    for source_file in "${!SRC_FILES[@]}"; do
        target_filename="${SRC_FILES[$source_file]}"
        target_path="$dir/$target_filename"
        target_dir=$(dirname "$target_path")
        
        # Create target directory if needed
        if [ "$target_dir" != "$dir" ]; then
            cmd mkdir -p "$target_dir"
        fi
        
        TEMP_FILE=$(mktemp)
        
        # Check if this is a JSON file
        if [[ "$source_file" == *.json ]] || [[ "$target_filename" == *.json ]]; then
            # For JSON files: customize name field only
            # Extract directory name for name customization
            DIR_NAME=$(basename "$dir")
            
            # Apply customizations to JSON file
            # 1. Replace display name with directory-specific name
            # 2. Replace container name with directory-specific name
            sed "s|\"name\": \"NVIDIA Dynamo.*\"|\"name\": \"[$DIR_NAME] $USER Dev Container\"|g" "$source_file" | \
            sed "s|\"dynamo-dev-container\"|\"$DIR_NAME-$USER-devcontainer\"|g" > "$TEMP_FILE"
        else
            # For non-JSON files: add header and copy content
            echo "# DO NOT EDIT - THIS IS AN AUTOMATICALLY GENERATED COPY" > "$TEMP_FILE"
            echo "# Original file: $source_file" >> "$TEMP_FILE"
            echo "# Last updated: $(date)" >> "$TEMP_FILE"
            echo "" >> "$TEMP_FILE"
            cat "$source_file" >> "$TEMP_FILE"
        fi
        
        # Copy the temporary file to the destination (cp will overwrite existing file)
        cmd cp "$TEMP_FILE" "$target_path"
        if [ $? -ne 0 ]; then
            echo "❌ ERROR: Failed to copy $target_filename to $dir"
        fi
        
        # Clean up temporary file
        rm -f "$TEMP_FILE" 2>/dev/null
    done

    # Handle build directory management for dynamo* directories
    if [[ "$DIR_NAME" == dynamo* ]]; then
        # Ensure .build/target exists
        cmd mkdir -p "$dir/.build/target"
        
        # If target is a real directory (not symlink), move it and create symlink
        if [ -d "$dir/target" ] && [ ! -L "$dir/target" ]; then
            cmd rm -rf "$dir/.build/target"/*
            if [ "$(ls -A "$dir/target" 2>/dev/null)" ]; then
                cmd mv "$dir/target"/* "$dir/target"/.* "$dir/.build/target/"
            fi
            cmd rm -rf "$dir/target"
            cmd ln -s ".build/target" "$dir/target"
            dry_run_echo "SUCCESS: Moved target to .build/target and created symlink in $DIR_NAME"
        fi
    fi
    
    ((SYNC_COUNT++))
done

# Check if no directories were synced when force flag was used
if [ "$SYNC_COUNT" -eq 0 ] && [ "$FORCE" = true ]; then
    echo "❌ ERROR: No directories found to sync despite --force flag. Check if ${DEST_SOURCE_DIRS}* exist"
    exit 1
fi

# Store the new hash
cmd bash -c "echo '$CURRENT_HASH' > '$TEMP_FILE'"
dry_run_echo "INFO: Sync completed. Updated $SYNC_COUNT directories." 
