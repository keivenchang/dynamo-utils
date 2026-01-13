#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# dangerously_wipe_local_docker.sh
#
# DANGER: This script is intentionally destructive.
# It will:
#   - Kill ALL running Docker containers
#   - Remove ALL containers (running or stopped)
#   - Force-remove ALL local Docker images
#   - (Optional) prune builder cache and other unused Docker data
#
# This is meant for quickly reclaiming disk space on a dev machine.
#
# Safety guard:
#   - Requires an explicit opt-in flag: --wipe-out-all-docker-images
#

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_color() {
  local color=$1
  local message=$2
  echo -e "${color}${message}${NC}"
}

usage() {
  cat <<'EOF'
Usage: dangerously_wipe_local_docker.sh [OPTIONS]

DANGER: Kills all running containers and deletes ALL local Docker images.

Required:
  --wipe-out-all-docker-images
      Explicitly acknowledge this script is destructive.

Options:
  --dry-run, --dryrun
      Print actions without executing them.
  --prune-builder-cache
      Also run: docker builder prune -a --force
  --prune-unused
      Also run: docker system prune -a --force
      (Note: this does NOT remove volumes; use --prune-volumes for that.)
  --prune-volumes
      ALSO remove ALL unused volumes via: docker system prune -a --volumes --force
      (This can delete data you care about.)
  -h, --help
      Show this help.

Examples:
  ./dangerously_wipe_local_docker.sh --dry-run --wipe-out-all-docker-images
  ./dangerously_wipe_local_docker.sh --wipe-out-all-docker-images --prune-builder-cache
  ./dangerously_wipe_local_docker.sh --wipe-out-all-docker-images --prune-unused --prune-volumes
EOF
}

DRY_RUN=false
PRUNE_BUILDER_CACHE=false
PRUNE_UNUSED=false
PRUNE_VOLUMES=false
ACK=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wipe-out-all-docker-images|--i-understand|--i-understand-this-will-delete-all-local-docker-images)
      ACK=true; shift ;;
    --dry-run|--dryrun)
      DRY_RUN=true; shift ;;
    --prune-builder-cache)
      PRUNE_BUILDER_CACHE=true; shift ;;
    --prune-unused)
      PRUNE_UNUSED=true; shift ;;
    --prune-volumes)
      PRUNE_VOLUMES=true; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [ "$ACK" != "true" ]; then
  print_color "$RED" "ERROR: Refusing to run without explicit acknowledgment."
  print_color "$YELLOW" "Re-run with:"
  echo "  --wipe-out-all-docker-images"
  exit 2
fi

if ! command -v docker >/dev/null 2>&1; then
  print_color "$RED" "ERROR: docker not found in PATH."
  exit 127
fi

run() {
  if [ "$DRY_RUN" = true ]; then
    echo "+ $*"
    return 0
  fi
  "$@"
}

print_color "$RED" "=== DANGEROUS DOCKER WIPE ==="
print_color "$YELLOW" "This will kill ALL running containers and delete ALL local Docker images."
echo "  DRY_RUN=$DRY_RUN"
echo "  PRUNE_BUILDER_CACHE=$PRUNE_BUILDER_CACHE"
echo "  PRUNE_UNUSED=$PRUNE_UNUSED"
echo "  PRUNE_VOLUMES=$PRUNE_VOLUMES"
echo

running_containers="$(docker ps -q || true)"
all_containers="$(docker ps -aq || true)"
all_images="$(docker images -aq || true)"

if [ -n "$running_containers" ]; then
  print_color "$BLUE" "Killing running containers..."
  # shellcheck disable=SC2086
  run docker kill $running_containers || true
else
  print_color "$GREEN" "No running containers found."
fi

if [ -n "$all_containers" ]; then
  print_color "$BLUE" "Removing all containers..."
  # shellcheck disable=SC2086
  run docker rm -f $all_containers || true
else
  print_color "$GREEN" "No containers found."
fi

if [ -n "$all_images" ]; then
  print_color "$BLUE" "Removing all local images..."
  # shellcheck disable=SC2086
  run docker image rm -f $all_images || true
else
  print_color "$GREEN" "No images found."
fi

if [ "$PRUNE_BUILDER_CACHE" = true ]; then
  print_color "$BLUE" "Pruning builder cache..."
  run docker builder prune -a --force || true
fi

if [ "$PRUNE_UNUSED" = true ] && [ "$PRUNE_VOLUMES" = true ]; then
  print_color "$BLUE" "Pruning unused Docker objects (including volumes)..."
  run docker system prune -a --volumes --force || true
elif [ "$PRUNE_UNUSED" = true ]; then
  print_color "$BLUE" "Pruning unused Docker objects (excluding volumes)..."
  run docker system prune -a --force || true
elif [ "$PRUNE_VOLUMES" = true ]; then
  print_color "$BLUE" "Pruning unused volumes..."
  run docker system prune -a --volumes --force || true
fi

print_color "$GREEN" "Done."

