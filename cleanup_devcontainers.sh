#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

# Parse command line arguments
ALL_MODE=false
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --all)
            ALL_MODE=true
            ;;
        --dry-run|--dryrun)
            DRY_RUN=true
            ;;
    esac
done

echo "=== Dev Container Cleanup Script ==="
if [ "$DRY_RUN" = true ]; then
    echo "Mode: DRY RUN (no changes will be made)"
fi
if [ "$ALL_MODE" = true ]; then
    echo "Mode: Full cleanup (--all)"
else
    echo "Mode: Stragglers only (use --all for full cleanup)"
fi
echo

# Count items before cleanup
vsc_images_count=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep "^vsc-" | wc -l || echo "0")
unnamed_containers=$(docker ps -a --format "{{.Names}}" | grep -E "^(admiring|adoring|affectionate|agitated|amazing|angry|awesome|beautiful|blissful|bold|boring|brave|busy|charming|clever|cool|compassionate|competent|condescending|confident|cranky|crazy|dazzling|determined|distracted|dreamy|eager|ecstatic|elastic|elated|elegant|eloquent|epic|exciting|fervent|festive|flamboyant|focused|friendly|frosty|funny|gallant|gifted|goofy|gracious|great|happy|hardcore|heuristic|hopeful|hungry|infallible|inspiring|intelligent|interesting|jolly|jovial|keen|kind|laughing|loving|lucid|magical|mystifying|modest|musing|naughty|nervous|nice|nifty|nostalgic|objective|optimistic|peaceful|pedantic|pensive|practical|priceless|quirky|quizzical|recursing|relaxed|reverent|romantic|sad|serene|sharp|silly|sleepy|stoic|strange|stupefied|suspicious|sweet|tender|thirsty|trusting|unruffled|upbeat|vibrant|vigilant|vigorous|wizardly|wonderful|xenodochial|youthful|zealous|zen)_[a-z]+$" | wc -l)

echo "Found:"
echo "  - $vsc_images_count vsc- prefixed images"
echo "  - $unnamed_containers unnamed containers (stragglers)"
echo

# Remove unnamed straggler containers - always done (excludes vsc- containers)
echo "=== Removing unnamed straggler containers ==="
# Find containers with Docker's auto-generated names that are NOT vsc- containers
straggler_names=$(docker ps -a --format "{{.Names}}" | grep -E "^(admiring|adoring|affectionate|agitated|amazing|angry|awesome|beautiful|blissful|bold|boring|brave|busy|charming|clever|cool|compassionate|competent|condescending|confident|cranky|crazy|dazzling|determined|distracted|dreamy|eager|ecstatic|elastic|elated|elegant|eloquent|epic|exciting|fervent|festive|flamboyant|focused|friendly|frosty|funny|gallant|gifted|goofy|gracious|great|happy|hardcore|heuristic|hopeful|hungry|infallible|inspiring|intelligent|interesting|jolly|jovial|keen|kind|laughing|loving|lucid|magical|mystifying|modest|musing|naughty|nervous|nice|nifty|nostalgic|objective|optimistic|peaceful|pedantic|pensive|practical|priceless|quirky|quizzical|recursing|relaxed|reverent|romantic|sad|serene|sharp|silly|sleepy|stoic|strange|stupefied|suspicious|sweet|tender|thirsty|trusting|unruffled|upbeat|vibrant|vigilant|vigorous|wizardly|wonderful|xenodochial|youthful|zealous|zen)_[a-z]+$" | grep -v "^vsc-" || true)

if [ "$DRY_RUN" = true ]; then
    if [ -n "$straggler_names" ]; then
        echo "$straggler_names"
        straggler_count=$(echo "$straggler_names" | wc -l)
        echo "[DRY RUN] Would remove $straggler_count straggler containers"
    else
        echo "No straggler containers found"
    fi
else
    if [ -n "$straggler_names" ]; then
        echo "$straggler_names" | xargs -r docker rm -f
        straggler_count=$(echo "$straggler_names" | wc -l)
        echo "Removed $straggler_count straggler containers"
    else
        echo "No straggler containers found"
    fi
fi

# Remove vsc- images - only if --all flag is set
if [ "$ALL_MODE" = true ]; then
    echo
    echo "=== Removing vsc- prefixed images ==="
    if [ "$vsc_images_count" -gt 0 ]; then
        if [ "$DRY_RUN" = true ]; then
            docker images --format "{{.Repository}}:{{.Tag}}" | grep "^vsc-"
            echo "[DRY RUN] Would remove $vsc_images_count vsc- images"
        else
            docker images --format "{{.Repository}}:{{.Tag}}" | grep "^vsc-" | xargs -r docker rmi -f
            echo "Removed $vsc_images_count vsc- images"
        fi
    else
        echo "No vsc- images to remove"
    fi
else
    if [ "$vsc_images_count" -gt 0 ]; then
        echo
        echo "NOTE: $vsc_images_count vsc- images exist but not removed (use --all to remove)"
    fi
fi

echo
echo "=== Cleanup complete ==="
echo
echo "Remaining containers:"
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Image}}" | head -10
echo
echo "Remaining vsc- images:"
docker images --format "{{.Repository}}:{{.Tag}}" | grep "^vsc-" || echo "None"

