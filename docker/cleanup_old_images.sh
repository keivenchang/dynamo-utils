#!/bin/bash

# Docker Image Cleanup Script
# Keeps :latest tags and top N images for vllm/sglang/trtllm variants
# Removes older images to free up space

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DRY_RUN=false
FORCE=false
RETAIN_COUNT=2

# Function to print colored output
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Docker Image Cleanup Script for dynamo vllm/sglang/trtllm variants

This script will:
- Keep all :latest-* tags (latest-vllm, latest-sglang, latest-trtllm, etc.)
- Keep all *-local-dev tags for latest variants
- Keep the top $RETAIN_COUNT most recent images for each variant type
- Remove older images to free up disk space

OPTIONS:
    --dry-run, --dryrun    Show what would be deleted without actually deleting
    --force                Use docker rmi -f to force removal of images
    --retain N             Number of recent images to retain per variant (default: $RETAIN_COUNT)
    -h, --help             Show this help message

EXAMPLES:
    $0 --dry-run                     # Show what would be deleted
    $0 --dryrun                      # Same as --dry-run
    $0 --force                       # Force delete images with docker rmi -f
    $0 --dry-run --retain 3         # Retain top 3 images per variant
    $0 --force --retain 1           # Force delete, retain only 1 recent image per variant

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

# Function to get images for a specific variant, sorted by creation time (newest first)
get_variant_images() {
    local variant=$1
    docker images --format "{{.Repository}}:{{.Tag}} {{.ID}} {{.CreatedAt}}" \
        | grep -E "^(dynamo|<none>):.*v.*\.dev.*-${variant}($| |-local-dev)" \
        | sort -k3,4 -r
}

# Function to identify images to keep vs delete
process_variant() {
    local variant=$1
    local variant_name=$2
    
    print_color $BLUE "Processing $variant_name images..."
    
    # Get all images for this variant
    local images=$(get_variant_images "$variant")
    
    if [ -z "$images" ]; then
        print_color $YELLOW "No $variant_name images found."
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
    print_color $BLUE "Cleaning up <none> (dangling) images..."
    
    local dangling_images=$(docker images --filter "dangling=true" -q --no-trunc)
    
    if [ -z "$dangling_images" ]; then
        print_color $GREEN "No <none> images found."
        echo
        return
    fi
    
    local count=$(echo "$dangling_images" | wc -l)
    print_color $YELLOW "Found $count <none> images"
    
    if [ "$DRY_RUN" = true ]; then
        print_color $YELLOW "DRY RUN: Would batch delete $count <none> images:"
        echo "$dangling_images" | while read -r image_id; do
            echo "  $image_id"
        done
        echo
        return
    fi
    
    # Batch delete all dangling images at once
    print_color $BLUE "Batch deleting $count <none> images..."
    
    if [ "$FORCE" = true ]; then
        set -x
        if echo "$dangling_images" | xargs docker rmi -f 2>/dev/null; then
            { set +x; } 2>/dev/null
            print_color $GREEN "✓ Successfully batch deleted $count <none> images"
        else
            { set +x; } 2>/dev/null
            print_color $YELLOW "⚠ Some <none> images may have failed to delete (likely in use)"
        fi
    else
        set -x
        if echo "$dangling_images" | xargs docker rmi 2>/dev/null; then
            { set +x; } 2>/dev/null
            print_color $GREEN "✓ Successfully batch deleted $count <none> images"
        else
            { set +x; } 2>/dev/null
            print_color $YELLOW "⚠ Some <none> images may have failed to delete (likely in use)"
        fi
    fi
    echo
}

# Function to perform the actual deletion
delete_images() {
    local temp_file="/tmp/images_to_delete_$$"
    
    if [ ! -f "$temp_file" ] || [ ! -s "$temp_file" ]; then
        print_color $GREEN "No images to delete!"
        return
    fi
    
    local image_count=$(wc -l < "$temp_file")
    
    if [ "$DRY_RUN" = true ]; then
        print_color $YELLOW "DRY RUN: Would batch delete $image_count images"
        print_color $YELLOW "Image IDs that would be batch deleted:"
        cat "$temp_file" | while read -r image_id; do
            echo "  $image_id"
        done
        rm -f "$temp_file"
        return
    fi
    
    print_color $YELLOW "Batch deleting $image_count Docker images..."
    
    # Batch delete all images at once
    print_color $BLUE "Executing batch deletion..."
    
    if [ "$FORCE" = true ]; then
        set -x
        if cat "$temp_file" | xargs docker rmi -f 2>/dev/null; then
            { set +x; } 2>/dev/null
            print_color $GREEN "✓ Successfully batch deleted $image_count images"
        else
            { set +x; } 2>/dev/null
            print_color $YELLOW "⚠ Some images may have failed to delete (likely in use by containers)"
            print_color $BLUE "Attempting individual deletion for failed images..."
            
            # Fallback: try individual deletion for any remaining images
            while read -r image_id; do
                # Check if image still exists
                if docker image inspect "$image_id" >/dev/null 2>&1; then
                    if docker rmi -f "$image_id" 2>/dev/null; then
                        print_color $GREEN "✓ Individually deleted: $image_id"
                    else
                        print_color $RED "✗ Failed to delete: $image_id (may be in use)"
                    fi
                fi
            done < "$temp_file"
        fi
    else
        set -x
        if cat "$temp_file" | xargs docker rmi 2>/dev/null; then
            { set +x; } 2>/dev/null
            print_color $GREEN "✓ Successfully batch deleted $image_count images"
        else
            { set +x; } 2>/dev/null
            print_color $YELLOW "⚠ Some images may have failed to delete (likely in use by containers)"
            print_color $BLUE "Attempting individual deletion for failed images..."
            
            # Fallback: try individual deletion for any remaining images
            while read -r image_id; do
                # Check if image still exists
                if docker image inspect "$image_id" >/dev/null 2>&1; then
                    if docker rmi "$image_id" 2>/dev/null; then
                        print_color $GREEN "✓ Individually deleted: $image_id"
                    else
                        print_color $RED "✗ Failed to delete: $image_id (may be in use)"
                    fi
                fi
            done < "$temp_file"
        fi
    fi
    
    rm -f "$temp_file"
    print_color $GREEN "Batch deletion completed"
}

# Main execution
main() {
    print_color $BLUE "=== Docker Image Cleanup Script ==="
    echo
    
    if [ "$DRY_RUN" = true ]; then
        print_color $YELLOW "DRY RUN MODE - No images will be deleted"
        echo
    fi
    
    print_color $BLUE "Configuration: Retaining top $RETAIN_COUNT images per variant"
    echo
    
    # Clean up any existing temp file
    rm -f /tmp/images_to_delete_$$
    
    # First, delete all <none> images
    delete_none_images
    
    # Process each variant
    process_variant "vllm" "VLLM"
    process_variant "sglang" "SGLang" 
    process_variant "trtllm" "TensorRT-LLM"
    
    # Delete the identified images
    delete_images
    
    print_color $BLUE "=== Cleanup Complete ==="
}

# Run main function
main "$@"
