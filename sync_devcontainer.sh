_file#!/bin/bash

# ==============================================================================
# sync_config_files.sh - Development Configuration Synchronization Tool
# ==============================================================================
#
# PURPOSE:
#   Automatically syncs development configuration files from a master location
#   (dynamo-utils) to all Dynamo project directories, ensuring consistent
#   development environments across multiple working copies.
#
# HOW IT WORKS:
#   1. Scans for directories matching 'dynamo*' pattern in parent directory
#   2. Copies configuration files from dynamo-utils to each found directory
#   3. Customizes devcontainer.json for each directory (unique container names)
#   4. Tracks changes via MD5 hashes to avoid unnecessary syncs
#
# EXAMPLE SCENARIO:
#   Directory structure:
#   ~/nvidia/
#   ├── dynamo-utils/           # This repository (master configs)
#   │   ├── devcontainer.json   # Master container config
#   │   └── sync_config_files.sh
#   ├── dynamo1/                # Clone for feature development
#   ├── dynamo2/                # Clone for bug fixes
#   └── dynamo3/                # Clone for experiments
#
#   Running ./sync_config_files.sh will:
#   - Copy devcontainer.json → dynamo1/.devcontainer/[user]/devcontainer.json
#     with customizations:
#     * Container name: dynamo1-[user]-devcontainer
#     * Display name: [dynamo1] [user] Dev Container
#
# USAGE:
#   ./sync_config_files.sh           # Normal sync operation
#   ./sync_config_files.sh --dry-run # Preview changes without applying
#   ./sync_config_files.sh --force   # Force sync even if no changes detected
#   ./sync_config_files.sh --silent  # No output (for cron jobs)
#
# CRON EXAMPLE:
#   */5 * * * * /home/user/nvidia/dynamo-utils/sync_config_files.sh --silent
#
# ==============================================================================

USER=$(whoami)
DEST_SRC_DIR_GLOB="${DEVCONTAINER_SRC_DIR:-$HOME/nvidia/dynamo}"

# Define list of elements
FRAMEWORKS=("VLLM" "SGLANG" "TRTLLM""NONE")

# Define development-related config files to sync
DEVCONTAINER_SRC="devcontainer.json"
DEVCONTAINER_DEST=".devcontainer/{framework}/devcontainer.json"

# Get the absolute path of the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVCONTAINER_SRC_PATH="${SCRIPT_DIR}/${DEVCONTAINER_SRC}"

TEMP_SHA_FILE="/tmp/.sync_config_files.sha"
LOG_FILE="/tmp/sync_config_files.log"

# Check for flags
DRYRUN=false
FORCE=false
SILENT=false
while [ $# -gt 0 ]; do
    case $1 in
        --dryrun|--dry-run)
            DRYRUN=true
            ;;
        --force)
            FORCE=true
            ;;
        --silent)
            SILENT=true
            ;;
        *)
            # Unknown option, ignore
            ;;
    esac
    shift
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
        if [ "$SILENT" != true ]; then
            echo "[DRYRUN] $*"
        fi
        # Return success in dryrun mode
        return 0
    else
        # Not dry run: execute the command
        if [ "$SILENT" != true ]; then
            # Show and execute
            ( set -x; "$@" )
        else
            # Execute silently
            "$@"
        fi
    fi
}

# Check if source file exists
if [ ! -f "${DEVCONTAINER_SRC_PATH}" ]; then
    dry_run_echo "ERROR: Source file not found at ${DEVCONTAINER_SRC_PATH}"
    exit 1
fi

# Create a hash of the development config file
CURRENT_HASH=$(md5sum "${DEVCONTAINER_SRC_PATH}" | cut -d' ' -f1)

# Check if we have a previous hash stored
if [ -f "$TEMP_SHA_FILE" ]; then
    PREVIOUS_HASH=$(cat "$TEMP_SHA_FILE")
else
    PREVIOUS_HASH=""
fi

# If hash unchanged and not forced, exit early (unless in dryrun mode)
if [ "$CURRENT_HASH" = "$PREVIOUS_HASH" ] && [ "$FORCE" = false ] && [ "$DRYRUN" != true ]; then
    dry_run_echo "DEBUG: Development config files unchanged, no sync needed."
    exit 0
fi

if [ "$FORCE" = true ]; then
    dry_run_echo "INFO: Force flag detected, syncing regardless of changes..."
fi

dry_run_echo "INFO: Development config files have changed, syncing to subdirectories..."

# Find all dynamo* subdirectories and copy development config files to them
SYNC_COUNT=0
for destdir in "$DEST_SRC_DIR_GLOB"*; do
    if [ ! -d "$destdir" ]; then
        continue
    fi

    # Skip the script's own directory
    if [ "$destdir" = "$SCRIPT_DIR" ]; then
        dry_run_echo "INFO: Skipping source directory $SCRIPT_DIR"
        continue
    fi

    dry_run_echo "INFO: Updating ${destdir}/..."

    # Process the devcontainer.json file
    TEMP_OUTPUT_FILE=$(mktemp)

    # For devcontainer.json: loop through frameworks and create framework-specific versions
    DEST_BASE_DIRNAME=$(basename "${destdir}")

    for framework_uppercase in "${FRAMEWORKS[@]}"; do
        # Convert framework to lowercase for path and name consistency
        framework_lowercase="${framework_uppercase,,}"
        # Replace {framework} placeholder with lowercase framework name
        framework_target_filename="${DEVCONTAINER_DEST//\{framework\}/${framework_lowercase}}"
        framework_target_path="${destdir}/${framework_target_filename}"
        framework_target_dir=$(dirname "${framework_target_path}")

        # Create framework-specific target directory if needed
        if [ "${framework_target_dir}" != "${destdir}" ]; then
            cmd mkdir -p "${framework_target_dir}"
        fi

        # Apply customizations to JSON file for this framework
        # 1. Replace display name with directory-specific name and lowercase framework
        # 2. Replace container name with directory-specific name
        # 3. Replace -vllm with -lowercase_framework in the image name
        sed "s|\"name\": \"NVIDIA Dynamo.*\"|\"name\": \"[${DEST_BASE_DIRNAME}-${USER}] ${framework_uppercase}\"|g" "${DEVCONTAINER_SRC_PATH}" | \
        sed "s|\"dynamo-vllm-devcontainer\"|\"${DEST_BASE_DIRNAME}-${USER}-${framework_lowercase}-devcontainer\"|g" | \
        sed "s|-vllm-|-${framework_lowercase}-|g" > "${TEMP_OUTPUT_FILE}"

        # Copy the framework-specific file to the destination
        if ! cmd cp "${TEMP_OUTPUT_FILE}" "${framework_target_path}"; then
            dry_run_echo "❌ ERROR: Failed to copy ${DEVCONTAINER_SRC} to ${framework_target_path}"
        fi
    done

    # Clean up temporary file
    rm -f "${TEMP_OUTPUT_FILE}" 2>/dev/null

    # Handle build directory management for dynamo* directories
    if [[ "$DEST_BASE_DIRNAME" == dynamo* ]]; then
        # Ensure .build/target exists
        cmd mkdir -p "$destdir/.build/target"

        # If target is a real directory (not symlink), move it and create symlink
        if [ -d "$destdir/target" ] && [ ! -L "$destdir/target" ]; then
            cmd rm -rf "$destdir/.build/target"/*
            if [ "$(ls -A "$destdir/target" 2>/dev/null)" ]; then
                cmd mv "$destdir/target"/* "$destdir/target"/.* "$destdir/.build/target/"
            fi
            cmd rm -rf "$destdir/target"
            cmd ln -s ".build/target" "$destdir/target"
            dry_run_echo "SUCCESS: Moved target to .build/target and created symlink in $DEST_BASE_DIRNAME"
        fi
    fi

    ((SYNC_COUNT++))
done

# Check if no directories were synced when force flag was used
if [ "$SYNC_COUNT" -eq 0 ] && [ "$FORCE" = true ]; then
    dry_run_echo "❌ ERROR: No directories found to sync despite --force flag. Check if ${DEST_SRC_DIR_GLOB}* exist"
    exit 1
fi

# Store the new hash
cmd bash -c "echo '$CURRENT_HASH' > '$TEMP_SHA_FILE'"
dry_run_echo "INFO: Sync completed. Updated $SYNC_COUNT directories."
