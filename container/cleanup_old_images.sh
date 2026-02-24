#!/bin/bash

# Docker Image Cleanup Script
# Keeps :latest tags and top N images for vllm/sglang/trtllm variants
# Removes older images to free up space

set -e

# Default values
DRY_RUN=false
FORCE=false
RETAIN_COUNT=2
KEEP_DEV_AND_LOCAL_DEV_ONLY=false

# Function to show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Docker Image Cleanup Script for dynamo vllm/sglang/trtllm variants

This script will:
- Keep all :latest-* tags (latest-vllm, latest-sglang, latest-trtllm, etc.)
- Keep the top $RETAIN_COUNT most recent images per variant type (runtime, dev, local-dev)
- Remove older images to free up disk space
- Prune Docker build cache

OPTIONS:
    --dry-run, --dryrun    Show what would be deleted without actually deleting
    --force                Use docker rmi -f to force removal of images
    --retain N             Number of recent images to retain per variant (default: $RETAIN_COUNT)
    --keep-dev-and-local-dev-only  Remove only runtime images; keeps BOTH dev and local-dev (both or nothing)
    -h, --help             Show this help message

EXAMPLES:
    $0 --dry-run                         # Show what would be deleted
    $0 --dryrun                          # Same as --dry-run
    $0 --force                           # Force delete images with docker rmi -f
    $0 --dry-run --retain 3             # Retain top 3 images per variant
    $0 --force --retain 1               # Force delete, retain only 1 recent image per variant
    $0 --force --keep-dev-and-local-dev-only   # Remove runtime only; keep BOTH dev and local-dev

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run|--dryrun)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --retain)
            if [[ $# -lt 2 ]] || [[ $2 =~ ^- ]]; then
                echo "Error: --retain requires a number argument"
                usage
                exit 1
            fi
            RETAIN_COUNT="$2"
            shift 2
            ;;
        --retain=*)
            # Still support the old format for backwards compatibility
            RETAIN_COUNT="${1#*=}"
            shift
            ;;
        --keep-dev-and-local-dev-only)
            KEEP_DEV_AND_LOCAL_DEV_ONLY=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Function to get images for a specific framework variant, sorted by creation time (newest first)
# Generic pattern: dynamo:<IMAGE_SHA>.<sha>-<framework>-<anything>
# Examples: dynamo:A1B2C3.98df1a2c5-vllm-dev, dynamo:D4E5F6.ef2583a9d-sglang-cuda12.9-local-dev,
#           dynamo:A1B2C3.d38954c75-none-cuda12.9-dev-orig
# Excludes latest-* tags (those are always kept).
get_variant_images() {
    local variant=$1
    docker images --format "{{.Repository}}:{{.Tag}} {{.ID}} {{.CreatedAt}}" \
        | grep -E "^dynamo:[^ ]+-${variant}-" \
        | grep -v ":latest-" \
        | sort -k3,4 -r
}

# Function to identify images to keep vs delete
process_variant() {
    local variant=$1
    local variant_name=$2
    
    echo "Processing $variant_name images..."
    
    # Get all images for this variant
    local images=$(get_variant_images "$variant")
    
    if [ -z "$images" ]; then
        echo "No $variant_name images found."
        return
    fi
    
    echo "Found images (keeping newest $RETAIN_COUNT, marking older for deletion):"
    
    # Separate local-dev from regular images
    local local_dev_images=$(echo "$images" | grep -E ".*-local-dev " || true)
    local regular_images=$(echo "$images" | grep -v -E ".*-local-dev " || true)
    
    # Process local-dev images
    if [ -n "$local_dev_images" ]; then
        echo "  -local-dev images:"
        local keep_local_dev=$(echo "$local_dev_images" | head -n $RETAIN_COUNT)
        local delete_local_dev=$(echo "$local_dev_images" | tail -n +$((RETAIN_COUNT + 1)))
        
        echo "$keep_local_dev" | while read -r line; do
            local repo_tag=$(echo "$line" | awk '{print $1}')
            echo "    $repo_tag"
        done
        
        if [ -n "$delete_local_dev" ]; then
            echo "$delete_local_dev" | while read -r line; do
                local repo_tag=$(echo "$line" | awk '{print $1}')
                local image_id=$(echo "$line" | awk '{print $2}')
                echo "    D $repo_tag"
                echo "$image_id" >> /tmp/images_to_delete_$$
            done
        fi
        echo
    fi
    
    # Process regular images
    if [ -n "$regular_images" ]; then
        echo "  regular images:"
        local keep_regular=$(echo "$regular_images" | head -n $RETAIN_COUNT)
        local delete_regular=$(echo "$regular_images" | tail -n +$((RETAIN_COUNT + 1)))
        
        echo "$keep_regular" | while read -r line; do
            local repo_tag=$(echo "$line" | awk '{print $1}')
            echo "    $repo_tag"
        done
        
        if [ -n "$delete_regular" ]; then
            echo "$delete_regular" | while read -r line; do
                local repo_tag=$(echo "$line" | awk '{print $1}')
                local image_id=$(echo "$line" | awk '{print $2}')
                echo "    D $repo_tag"
                echo "$image_id" >> /tmp/images_to_delete_$$
            done
        fi
        echo
    fi
}

# Function to delete dangling <none> images
delete_none_images() {
    echo "Cleaning up <none> (dangling) images..."
    
    local dangling_images=$(docker images --filter "dangling=true" -q --no-trunc)
    
    if [ -z "$dangling_images" ]; then
        echo "No <none> images found."
        echo
        return
    fi
    
    local count=$(echo "$dangling_images" | wc -l)
    echo "Found $count <none> images"
    
    if [ "$DRY_RUN" = true ]; then
        echo "DRY RUN: Would batch delete $count <none> images:"
        echo "$dangling_images" | while read -r image_id; do
            echo "  $image_id"
        done
        echo
        return
    fi
    
    echo "Batch deleting $count <none> images..."
    
    if [ "$FORCE" = true ]; then
        if echo "$dangling_images" | xargs docker rmi -f 2>/dev/null; then
            echo "Successfully batch deleted $count <none> images"
        else
            echo "WARNING: Some <none> images may have failed to delete (likely in use)"
        fi
    else
        if echo "$dangling_images" | xargs docker rmi 2>/dev/null; then
            echo "Successfully batch deleted $count <none> images"
        else
            echo "WARNING: Some <none> images may have failed to delete (likely in use)"
        fi
    fi
    echo
}

# Function to perform the actual deletion
delete_images() {
    local temp_file="/tmp/images_to_delete_$$"
    
    if [ ! -f "$temp_file" ] || [ ! -s "$temp_file" ]; then
        echo "No images to delete!"
        return
    fi
    
    local image_count=$(wc -l < "$temp_file")
    
    if [ "$DRY_RUN" = true ]; then
        echo "DRY RUN: Would batch delete $image_count images"
        echo "Image IDs that would be batch deleted:"
        cat "$temp_file" | while read -r image_id; do
            echo "  $image_id"
        done
        rm -f "$temp_file"
        return
    fi
    
    echo "Batch deleting $image_count Docker images..."
    
    if [ "$FORCE" = true ]; then
        if cat "$temp_file" | xargs docker rmi -f 2>/dev/null; then
            echo "Successfully batch deleted $image_count images"
        else
            echo "WARNING: Some images may have failed to delete (likely in use by containers)"
            echo "Attempting individual deletion for failed images..."
            
            while read -r image_id; do
                if docker image inspect "$image_id" >/dev/null 2>&1; then
                    if docker rmi -f "$image_id" 2>/dev/null; then
                        echo "  Deleted: $image_id"
                    else
                        echo "  FAILED: $image_id (may be in use)"
                    fi
                fi
            done < "$temp_file"
        fi
    else
        if cat "$temp_file" | xargs docker rmi 2>/dev/null; then
            echo "Successfully batch deleted $image_count images"
        else
            echo "WARNING: Some images may have failed to delete (likely in use by containers)"
            echo "Attempting individual deletion for failed images..."
            
            while read -r image_id; do
                if docker image inspect "$image_id" >/dev/null 2>&1; then
                    if docker rmi "$image_id" 2>/dev/null; then
                        echo "  Deleted: $image_id"
                    else
                        echo "  FAILED: $image_id (may be in use)"
                    fi
                fi
            done < "$temp_file"
        fi
    fi
    
    rm -f "$temp_file"
    echo "Batch deletion completed"
}

# Function to remove only runtime dynamo images (keep BOTH dev and local-dev; both or nothing)
remove_runtime_images() {
    echo "Removing runtime-only dynamo images (keeping BOTH dev and local-dev)..."

    local targets=()
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        local repo_tag=$(echo "$line" | awk '{print $1}')
        # Keep latest-*; keep BOTH *-dev and *-local-dev; remove only *-runtime* (no -dev in tag)
        case "$repo_tag" in dynamo:latest-*) continue ;; esac
        case "$repo_tag" in *-local-dev) continue ;; esac
        case "$repo_tag" in *-dev) continue ;; esac
        case "$repo_tag" in *-runtime*) ;; *) continue ;; esac
        targets+=("$repo_tag")
    done < <(docker images --format "{{.Repository}}:{{.Tag}} {{.ID}}" | grep "^dynamo:")

    # Sanity: must not remove any dev or local-dev (both or nothing)
    local bad=
    for t in "${targets[@]}"; do
        case "$t" in *-local-dev|*-dev) bad=1; break ;; esac
    done
    if [ -n "$bad" ]; then
        echo "ERROR: Would remove a dev/local-dev image; aborting (keep both or nothing)." >&2
        return 1
    fi

    if [ ${#targets[@]} -eq 0 ]; then
        echo "No runtime images found to remove. All dev and local-dev images are kept."
        echo
        return
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "DRY RUN: Would keep all *-dev and *-local-dev; would remove ${#targets[@]} runtime image(s)."
    fi
    echo "Found ${#targets[@]} runtime images to remove (all dev and local-dev kept):"
    for t in "${targets[@]}"; do
        echo "  D $t"
    done

    if [ "$DRY_RUN" = true ]; then
        echo "DRY RUN: Would delete ${#targets[@]} images"
        echo
        return
    fi

    local deleted=0
    local failed=0
    for t in "${targets[@]}"; do
        if [ "$FORCE" = true ]; then
            if docker rmi -f "$t" 2>/dev/null; then
                deleted=$((deleted + 1))
            else
                echo "  FAILED: $t"
                failed=$((failed + 1))
            fi
        else
            if docker rmi "$t" 2>/dev/null; then
                deleted=$((deleted + 1))
            else
                echo "  FAILED: $t (use --force?)"
                failed=$((failed + 1))
            fi
        fi
    done
    echo "Deleted $deleted, failed $failed"
    echo
}

# Function to prune orphan image layers (unreferenced after image deletions)
prune_orphan_images() {
    echo "Pruning orphan (unused) image layers..."

    local orphan_size=$(docker system df 2>/dev/null | grep "Images" | awk '{print $(NF-1)}')
    if [ -z "$orphan_size" ] || [ "$orphan_size" = "0B" ]; then
        echo "No orphan layers to prune."
        echo
        return
    fi

    echo "Reclaimable image space: $orphan_size"

    if [ "$DRY_RUN" = true ]; then
        echo "DRY RUN: Would run: docker image prune -f"
        echo
        return
    fi

    if docker image prune -f 2>/dev/null; then
        echo "Orphan image layers pruned"
    else
        echo "WARNING: Some orphan layers may have failed to prune"
    fi
    echo
}

# Function to prune Docker build cache
prune_build_cache() {
    echo "Pruning Docker build cache..."

    local cache_size=$(docker system df --format '{{.Type}}\t{{.Size}}' 2>/dev/null | grep "Build Cache" | awk '{print $NF}')
    if [ -z "$cache_size" ] || [ "$cache_size" = "0B" ]; then
        echo "No build cache to prune."
        echo
        return
    fi

    echo "Build cache size: $cache_size"

    if [ "$DRY_RUN" = true ]; then
        echo "DRY RUN: Would run: docker builder prune -af"
        echo
        return
    fi

    if docker builder prune -af 2>/dev/null; then
        echo "Build cache pruned"
    else
        echo "WARNING: Some build cache entries may have failed to prune"
    fi
    echo
}

# Main execution
main() {
    echo "=== Docker Image Cleanup Script ==="
    echo
    
    if [ "$DRY_RUN" = true ]; then
        echo "DRY RUN MODE - No images will be deleted"
        echo
    fi
    
    echo "Configuration: Retaining top $RETAIN_COUNT images per variant"
    echo
    
    # Clean up any existing temp file
    rm -f /tmp/images_to_delete_$$
    
    # First, delete all <none> images
    delete_none_images
    
    # Process each variant (including base "none" images)
    process_variant "none" "Base (none)"
    process_variant "vllm" "VLLM"
    process_variant "sglang" "SGLang"
    process_variant "trtllm" "TensorRT-LLM"
    
    # Delete the identified images
    delete_images

    # Remove only runtime images if --keep-dev-and-local-dev-only was specified
    if [ "$KEEP_DEV_AND_LOCAL_DEV_ONLY" = true ]; then
        remove_runtime_images
    fi

    # Prune orphan image layers (unreferenced after deletions above)
    prune_orphan_images
    
    # Prune Docker build cache (often the largest consumer of disk space)
    prune_build_cache
    
    echo "=== Cleanup Complete ==="
}

# Run main function
main "$@"
