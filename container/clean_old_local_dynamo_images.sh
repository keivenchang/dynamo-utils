#!/bin/bash

# Docker Image Cleanup Script
# Keeps :latest-* tags and the newest N images per (variant × local-dev) bucket.
# Any image whose repository is (anything/)?dynamo and whose tag starts with a
# git-SHA-looking segment is a candidate. Registry prefix doesn't matter
# (local builds, Azure ACR, AWS ECR, etc. all flow through the same logic),
# and framework names (vllm / sglang / trtllm / dynamo-test / future) are
# auto-detected from the tag rather than hardcoded.

set -e

# Default values
DRY_RUN=false
FORCE=false
RETAIN_COUNT=2
KEEP_DEV_AND_LOCAL_DEV_ONLY=false
CLEAN_VSC=false

# Function to show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Docker Image Cleanup Script for dynamo:* images (all registries)

This script will:
- Keep all :latest-* tags (latest-vllm, latest-sglang, latest-trtllm, etc.)
- Bucket every SHA-tagged dynamo:* image by its (variant, local-dev?) fingerprint
  auto-extracted from the tag (e.g. vllm-dev-cuda12 vs sglang-dev-cuda13 vs
  dynamo-test-cuda12-amd64), regardless of registry prefix
- Keep the top $RETAIN_COUNT most recent images per bucket
- Remove older images to free up disk space
- Prune Docker build cache

OPTIONS:
    --dry-run, --dryrun    Show what would be deleted without actually deleting
    --force                Use docker rmi -f to force removal of images
    --retain N             Number of recent images to retain per variant (default: $RETAIN_COUNT)
    --keep-dev-and-local-dev-only  Remove only runtime images; keeps BOTH dev and local-dev (both or nothing)
    --clean-vsc            Remove all vsc-* containers (stopped+running) and vsc-* images
    -h, --help             Show this help message

EXAMPLES:
    $0 --dry-run                         # Show what would be deleted
    $0 --dryrun                          # Same as --dry-run
    $0 --force                           # Force delete images with docker rmi -f
    $0 --dry-run --retain 3             # Retain top 3 images per variant
    $0 --force --retain 1               # Force delete, retain only 1 recent image per variant
    $0 --force --keep-dev-and-local-dev-only   # Remove runtime only; keep BOTH dev and local-dev
    $0 --clean-vsc --dry-run             # Show vsc-* containers/images that would be removed
    $0 --clean-vsc --force               # Remove all vsc-* containers and images

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
        --clean-vsc)
            CLEAN_VSC=true
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

# Enumerate every SHA-tagged dynamo image, newest first.
#
# Matches repos that are `dynamo` or end in `/dynamo` (any registry prefix) and
# whose tag starts with a git-SHA-looking segment (7+ hex chars followed by `-`).
# `latest-*` tags fall off implicitly since their first segment isn't hex.
#
# Covered shapes:
#   dynamo:<sha>-<anything>                                       (local builds)
#   dynamoci.azurecr.io/ai-dynamo/dynamo:<sha>-<anything>         (Azure ACR)
#   210086341041.dkr.ecr.us-west-2.amazonaws.com/ai-dynamo/dynamo:<sha>-<anything> (AWS ECR)
#   <any-registry>/…/dynamo:<sha>-<anything>                      (future)
list_sha_tagged_dynamo_images() {
    docker images --format "{{.Repository}}:{{.Tag}} {{.ID}} {{.CreatedAt}}" \
        | awk '{
            n = index($1, ":")
            if (n == 0) next
            repo = substr($1, 1, n - 1)
            tag  = substr($1, n + 1)
            if (repo != "dynamo" && substr(repo, length(repo) - 6) != "/dynamo") next
            if (tag !~ /^[0-9a-f]{7,}-/) next
            print
        }' \
        | sort -k3,4 -r
}

# Compute a bucket key for a tag: "<variant_signature>|<local-dev|regular>"
#   variant_signature = tag with leading `<sha>-` stripped and trailing `-local-dev` stripped
bucket_key_for_tag() {
    local tag=$1
    local rest="${tag#*-}"   # drop <sha>-
    if [[ "$rest" == *-local-dev ]]; then
        echo "${rest%-local-dev}|local-dev"
    else
        echo "$rest|regular"
    fi
}

# Enumerate all dynamo:<sha>-* images across every registry, group them by
# (variant_signature × local-dev flag), and retain newest $RETAIN_COUNT per group.
process_all_dynamo_images() {
    echo "Processing all SHA-tagged dynamo images (any registry)..."

    local images
    images=$(list_sha_tagged_dynamo_images)

    if [ -z "$images" ]; then
        echo "No SHA-tagged dynamo images found."
        return
    fi

    declare -A bucket_lines=()
    declare -A bucket_counts=()

    local line repo_tag image_id tag key
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        repo_tag=$(echo "$line" | awk '{print $1}')
        image_id=$(echo "$line" | awk '{print $2}')
        tag="${repo_tag#*:}"
        key=$(bucket_key_for_tag "$tag")
        bucket_lines["$key"]+="${repo_tag} ${image_id}"$'\n'
        bucket_counts["$key"]=$(( ${bucket_counts["$key"]:-0} + 1 ))
    done <<< "$images"

    local sorted_keys
    sorted_keys=$(printf '%s\n' "${!bucket_lines[@]}" | sort)

    local key variant_sig flag bucket_body keep delete
    while IFS= read -r key; do
        [ -z "$key" ] && continue
        variant_sig="${key%|*}"
        flag="${key##*|}"
        echo "  bucket: ${variant_sig} [${flag}]  (found ${bucket_counts[$key]}, retaining ${RETAIN_COUNT})"

        bucket_body="${bucket_lines[$key]}"
        # Already newest-first; preserve order.
        keep=$(printf '%s' "$bucket_body" | head -n "$RETAIN_COUNT")
        delete=$(printf '%s' "$bucket_body" | tail -n +$((RETAIN_COUNT + 1)))

        while IFS=' ' read -r kept_repo_tag _kept_id; do
            [ -z "$kept_repo_tag" ] && continue
            echo "    ${kept_repo_tag}"
        done <<< "$keep"

        if [ -n "$delete" ]; then
            while IFS=' ' read -r del_repo_tag del_image_id; do
                [ -z "$del_repo_tag" ] && continue
                echo "    D ${del_repo_tag}"
                echo "$del_image_id" >> /tmp/images_to_delete_$$
            done <<< "$delete"
        fi
        echo
    done <<< "$sorted_keys"
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
        # Keep latest-*; keep BOTH *-dev-* / *-dev and *-local-dev-* / *-local-dev; remove only *-runtime* (no -dev in tag)
        case "$repo_tag" in dynamo:latest-*) continue ;; esac
        case "$repo_tag" in *-local-dev-*|*-local-dev) continue ;; esac
        case "$repo_tag" in *-dev-*|*-dev) continue ;; esac
        case "$repo_tag" in *-runtime*) ;; *) continue ;; esac
        targets+=("$repo_tag")
    done < <(docker images --format "{{.Repository}}:{{.Tag}} {{.ID}}" | awk '{
        n = index($1, ":")
        if (n == 0) next
        repo = substr($1, 1, n - 1)
        if (repo != "dynamo" && substr(repo, length(repo) - 6) != "/dynamo") next
        print
    }')

    # Sanity: must not remove any dev or local-dev (both or nothing)
    local bad=
    for t in "${targets[@]}"; do
        case "$t" in *-local-dev-*|*-local-dev|*-dev-*|*-dev) bad=1; break ;; esac
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

# Remove stopped (exited) containers to reclaim their writable-layer space
prune_stopped_containers() {
    echo "Removing stopped containers..."

    local stopped
    stopped=$(docker ps -a --filter "status=exited" --format "{{.ID}}\t{{.Names}}\t{{.Size}}\t{{.Status}}" || true)
    if [ -z "$stopped" ]; then
        echo "No stopped containers found."
        echo
        return
    fi

    local count
    count=$(echo "$stopped" | wc -l)
    echo "Found $count stopped container(s):"
    echo "$stopped" | while IFS=$'\t' read -r cid cname csize cstatus; do
        echo "  $cname  $csize  ($cstatus)"
    done

    if [ "$DRY_RUN" = true ]; then
        echo "DRY RUN: Would remove $count stopped container(s)"
        echo
        return
    fi

    local cids
    cids=$(echo "$stopped" | awk -F'\t' '{print $1}')
    echo "$cids" | xargs docker rm -f 2>/dev/null || true
    echo "Removed $count stopped container(s)"
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

# Remove all vsc-* containers (running + stopped) and vsc-* images
cleanup_vsc() {
    echo "=== VSC Container/Image Cleanup ==="
    echo

    # Stop and remove vsc-* containers
    local container_lines
    container_lines=$(docker ps -a --format "{{.ID}}\t{{.Names}}\t{{.Status}}" | grep "	vsc-" || true)
    if [ -n "$container_lines" ]; then
        local count
        count=$(echo "$container_lines" | wc -l)
        echo "Found $count vsc-* container(s):"
        echo "$container_lines" | while IFS=$'\t' read -r cid cname cstatus; do
            echo "  $cname ($cstatus)"
        done
        local cids
        cids=$(echo "$container_lines" | awk -F'\t' '{print $1}')
        if [ "$DRY_RUN" = true ]; then
            echo "DRY RUN: Would stop and remove $count container(s)"
        else
            echo "$cids" | xargs docker rm -f 2>/dev/null || true
            echo "Removed $count container(s)"
        fi
    else
        echo "No vsc-* containers found."
    fi
    echo

    # Remove vsc-* images
    local image_lines
    image_lines=$(docker images --format "{{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}" | grep "^vsc-" || true)
    if [ -n "$image_lines" ]; then
        local count
        count=$(echo "$image_lines" | wc -l)
        echo "Found $count vsc-* image(s):"
        echo "$image_lines" | while IFS=$'\t' read -r repo_tag iid isize; do
            echo "  $repo_tag ($isize)"
        done
        local repo_tags
        repo_tags=$(echo "$image_lines" | awk -F'\t' '{print $1}')
        if [ "$DRY_RUN" = true ]; then
            echo "DRY RUN: Would remove $count image(s)"
        else
            echo "$repo_tags" | xargs docker rmi -f 2>/dev/null || true
            echo "Removed $count image(s)"
        fi
    else
        echo "No vsc-* images found."
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
    
    # Bucket and retain across every SHA-tagged dynamo image (all registries, all variants)
    process_all_dynamo_images
    
    # Delete the identified images
    delete_images

    # Remove only runtime images if --keep-dev-and-local-dev-only was specified
    if [ "$KEEP_DEV_AND_LOCAL_DEV_ONLY" = true ]; then
        remove_runtime_images
    fi

    # Remove stopped containers (reclaims writable-layer space)
    prune_stopped_containers

    # Prune orphan image layers (unreferenced after deletions above)
    prune_orphan_images
    
    # Remove vsc-* containers and images if requested
    if [ "$CLEAN_VSC" = true ]; then
        cleanup_vsc
    fi

    # Prune Docker build cache (often the largest consumer of disk space)
    prune_build_cache
    
    echo "=== Cleanup Complete ==="
}

# Run main function
main "$@"
