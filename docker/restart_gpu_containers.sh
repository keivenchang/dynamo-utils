#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Monitor GPU-enabled Docker containers and restart if GPU errors detected

# Note: Don't use 'set -e' as docker commands may return non-zero exit codes
# and we want to continue checking other containers

# Default settings
DRY_RUN=false
LOG_FILE="/tmp/gpu_monitor.log"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dryrun|--dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            cat << 'EOF'
Usage: restart_gpu_containers.sh [OPTIONS]

Monitor GPU-enabled Docker containers and restart if GPU errors detected.

Options:
    --dryrun, --dry-run    Show what would be done without actually restarting containers
    --help, -h             Show this help message

Examples:
    ./restart_gpu_containers.sh                # Normal mode - restart containers with GPU errors
    ./restart_gpu_containers.sh --dry-run     # Dry-run mode - only check and report, don't restart

EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

# Command wrapper that shows commands and executes them if not in dry-run mode
# - If DRY_RUN=false: Shows command with shell tracing format (+ prefix) and executes it
# - If DRY_RUN=true: Only shows command with shell tracing format (+ prefix), does not execute
cmd() {
    # Always show what will be executed using shell tracing format
    ( set -x; : "$@" ) 2>&1 | sed 's/^+ : /+ /'

    # Only execute if not in dry-run mode
    if [ "$DRY_RUN" = false ]; then
        "$@"
    fi
}

should_monitor_container() {
    local container=$1

    # Get the image name for this container
    local image
    image=$(docker inspect "$container" --format='{{.Config.Image}}' 2>/dev/null || echo "")

    # Only monitor containers using vsc-dynamo* images
    if [[ "$image" == vsc-dynamo* ]]; then
        return 0
    fi

    # Skip all other containers
    return 1
}

check_gpu_in_container() {
    local container=$1

    # Check if container has GPU access (either via nvidia runtime or --gpus flag)
    local runtime
    runtime=$(docker inspect "$container" --format='{{.HostConfig.Runtime}}' 2>/dev/null || echo "")

    local device_requests
    device_requests=$(docker inspect "$container" --format='{{.HostConfig.DeviceRequests}}' 2>/dev/null || echo "")

    # Skip if container has neither nvidia runtime nor GPU device requests
    if [[ "$runtime" != "nvidia" ]] && [[ ! "$device_requests" =~ gpu ]]; then
        return 1  # Not a GPU container
    fi

    # Try to run nvidia-smi in the container
    local gpu_check
    gpu_check=$(docker exec "$container" nvidia-smi 2>&1 || true)

    # Check for common GPU error patterns
    if echo "$gpu_check" | grep -qi "error\|failed\|unable\|not found\|no devices"; then
        log "ERROR: GPU error detected in container $container"
        log "Output: $gpu_check"
        return 2  # GPU error
    fi

    return 0  # GPU OK
}

restart_container() {
    local container=$1
    if [ "$DRY_RUN" = true ]; then
        log "[DRY RUN] Would restart container: $container"
        cmd docker restart "$container"
        log "[DRY RUN] Would sleep 5 seconds"
        log "[DRY RUN] Would verify GPU after restart"
    else
        log "Restarting container: $container"
        cmd docker restart "$container"
        sleep 5

        # Verify GPU after restart
        if check_gpu_in_container "$container"; then
            log "SUCCESS: Container $container restarted successfully"
        else
            log "WARNING: Container $container still has GPU issues after restart"
        fi
    fi
}

# Show dry-run mode if enabled
if [ "$DRY_RUN" = true ]; then
    log "=== DRY RUN MODE - No containers will be restarted ==="
fi

# Get all running containers
containers=$(docker ps --format '{{.Names}}' 2>/dev/null || true)

if [[ -z "$containers" ]]; then
    log "No running containers found"
    exit 0
fi

log "Checking GPU status for containers..."

for container in $containers; do
    # Skip containers we shouldn't monitor (deploy-*, etc.)
    if ! should_monitor_container "$container"; then
        if [ "$DRY_RUN" = true ]; then
            log "SKIP: $container - Not a monitored container type"
        fi
        continue
    fi

    if [ "$DRY_RUN" = true ]; then
        log "Checking container: $container"
    fi

    check_gpu_in_container "$container"
    status=$?

    case $status in
        0)
            log "OK: $container - GPU healthy"
            ;;
        1)
            # Not a GPU container, skip
            if [ "$DRY_RUN" = true ]; then
                log "SKIP: $container - Not a GPU container"
            fi
            ;;
        2)
            log "FAILED: $container - GPU error detected, restarting..."
            restart_container "$container"
            ;;
    esac
done

log "GPU monitoring complete"
