#!/bin/bash

# ==============================================================================
# devcontainer_sync.sh - Development Configuration Synchronization Tool
# ==============================================================================
#
# PURPOSE:
#   Automatically syncs development configuration files from a master location
#   (dynamo-utils) to all Dynamo project directories, ensuring consistent
#   development environments across multiple working copies.
#
# HOW IT WORKS:
#   1. Scans for directories matching 'dynamo*' pattern in parent directory
#   2. Copies devcontainer.json from dynamo-utils to each found directory
#   3. Creates framework-specific devcontainer configs (VLLM, SGLANG, TRTLLM)
#   4. Customizes each config with directory-specific names and framework tags
#   5. Manages .build/target symlinks for build artifacts
#   6. Tracks changes via MD5 hashes to avoid unnecessary syncs
#
# EXAMPLE SCENARIO:
#   Directory structure:
#   ~/nvidia/
#   ├── dynamo-utils/           # This repository (master configs)
#   │   ├── devcontainer.json   # Master container config
#   │   └── devcontainer_sync.sh
#   ├── dynamo1/                # Clone for feature development
#   ├── dynamo2/                # Clone for bug fixes
#   └── dynamo3/                # Clone for experiments
#
#   Running ./devcontainer_sync.sh will create for each dynamo directory:
#   - .devcontainer/keivenc_vllm/devcontainer.json
#     * Display name: [dynamo1-keivenc] VLLM
#     * Image name: dynamo1-vllm-devcontainer
#   - .devcontainer/keivenc_sglang/devcontainer.json
#     * Display name: [dynamo1-keivenc] SGLANG
#     * Image name: dynamo1-sglang-devcontainer
#   - .devcontainer/keivenc_trtllm/devcontainer.json
#     * Display name: [dynamo1-keivenc] TRTLLM
#     * Image name: dynamo1-trtllm-devcontainer
#
# USAGE:
#   ./devcontainer_sync.sh           # Normal sync operation
#   ./devcontainer_sync.sh --dryrun  # Preview changes without applying
#   ./devcontainer_sync.sh --force   # Force sync even if no changes detected
#   ./devcontainer_sync.sh --silent  # No output (for cron jobs)
#
# CRON EXAMPLE:
#   */5 * * * * /home/user/nvidia/dynamo-utils/devcontainer_sync.sh --silent
#
# ==============================================================================

USER=$(whoami)
DEST_SRC_DIR_GLOB="${DEVCONTAINER_SRC_DIR:-$HOME/nvidia/dynamo}"

# Define supported ML frameworks for devcontainer creation
FRAMEWORKS=("VLLM" "SGLANG" "TRTLLM")

# Define devcontainer config file locations
DEVCONTAINER_SRC="devcontainer.json"
DEVCONTAINER_DEST=".devcontainer/{framework}/devcontainer.json"

# Get the absolute path of the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVCONTAINER_SRC_PATH="${SCRIPT_DIR}/${DEVCONTAINER_SRC}"

TEMP_SHA_FILE="/tmp/.sync_devcontainer.sha"
LOG_FILE="/tmp/sync_devcontainer.log"

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

    # Create framework-specific devcontainer configs for VLLM, SGLANG, and TRTLLM
    DEST_BASE_DIRNAME=$(basename "${destdir}")

    for framework_uppercase in "${FRAMEWORKS[@]}"; do
        # Convert framework to lowercase for path and name consistency (e.g., TRTLLM -> trtllm)
        framework_lowercase="${framework_uppercase,,}"
        # Replace {framework} placeholder with username_framework format (e.g., keivenc_vllm)
        username_framework_target_filename="${DEVCONTAINER_DEST//\{framework\}/${USER}_${framework_lowercase}}"
        framework_target_path="${destdir}/${username_framework_target_filename}"
        framework_target_dir=$(dirname "${framework_target_path}")

        # Create framework-specific target directory if needed
        if [ "${framework_target_dir}" != "${destdir}" ]; then
            cmd mkdir -p "${framework_target_dir}"
        fi

        # Apply customizations to devcontainer.json for this framework
        # 1. Set display name to [dirname-username] FRAMEWORK (e.g., [dynamo3-keivenc] VLLM)
        # 2. Replace -framework- placeholder with -framework_lowercase- in image name
        # 3. Substitute __HF_TOKEN__ and __GITHUB_TOKEN__ placeholders with environment values
        sed "s|\"name\": \"NVIDIA Dynamo.*\"|\"name\": \"[${DEST_BASE_DIRNAME}-${USER}] ${framework_uppercase}\"|g" "${DEVCONTAINER_SRC_PATH}" | \
        sed "s|-framework-|-${framework_lowercase}-|g" | \
        sed "s|__HF_TOKEN__|${HF_TOKEN:-}|g" | \
        sed "s|__GITHUB_TOKEN__|${GITHUB_TOKEN:-}|g" > "${TEMP_OUTPUT_FILE}"

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
