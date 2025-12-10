#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Get repository directory (where the script is called from)
# This should be the project's root directory (e.g., dynamo1)
REPO_DIR="$(pwd)"

# Validate we're in a repository directory (has container/build.sh)
if [ ! -f "container/build.sh" ]; then
    echo "ERROR: Must run from a project's root directory (e.g., dynamo1)" >&2
    echo "Current directory: $REPO_DIR" >&2
    echo "container/build.sh not found" >&2
    exit 1
fi

# Function to print messages (to stderr so they don't interfere with command substitution)
print_msg() {
    echo "$@" >&2
}


# Function to build all targets (runtime + dev + local-dev)
build_all_targets() {
    local framework=$1
    local dry_run=$2
    local no_cache=$3
    local log_file="/tmp/${framework,,}.build.log"
    local runtime_log_file="/tmp/${framework,,}.runtime.build.log"
    local dev_log_file="/tmp/${framework,,}.dev.build.log"

    print_msg "\n=== Building $framework runtime, local-dev, and dev pipeline ==="
    print_msg "Runtime log file: $runtime_log_file"
    print_msg "Local-dev log file: $log_file"
    print_msg "Dev log file: $dev_log_file"


    # First, build the runtime image explicitly
    local runtime_build_cmd="container/build.sh --framework \"$framework\" --target runtime --no-tag-latest"
    if [ "$no_cache" = true ]; then
        runtime_build_cmd+=" --no-cache"
    fi
    if [ "$dry_run" = true ]; then
        runtime_build_cmd+=" --dry-run"
    fi

    # Then build dev (no target, which depends on runtime)
    local dev_build_cmd="container/build.sh --framework \"$framework\" --no-tag-latest"
    if [ "$no_cache" = true ]; then
        dev_build_cmd+=" --no-cache"
    fi
    if [ "$dry_run" = true ]; then
        dev_build_cmd+=" --dry-run"
    fi

    # Finally build local-dev (which depends on dev)
    local local_dev_build_cmd="container/build.sh --framework \"$framework\" --target local-dev --no-tag-latest"
    if [ "$no_cache" = true ]; then
        local_dev_build_cmd+=" --no-cache"
    fi
    if [ "$dry_run" = true ]; then
        local_dev_build_cmd+=" --dry-run"
    fi

    if [ "$dry_run" = true ]; then
        # Run build.sh in dry-run mode for both runtime and local-dev
        local temp_runtime_output="/tmp/${framework,,}.runtime.dryrun.txt"
        local temp_output="/tmp/${framework,,}.build.dryrun.txt"

        eval "$runtime_build_cmd" > "$temp_runtime_output" 2>&1
        eval "$local_dev_build_cmd" > "$temp_output" 2>&1

        # Extract image names from the output
        local runtime_image=$(grep "docker build.*--tag.*runtime" "$temp_runtime_output" | grep -oE "\-\-tag [^ ]+" | sed 's/--tag //' | tail -1)
        if [ -z "$runtime_image" ]; then
            # Fallback pattern for runtime
            local version=$(cd "$SCRIPT_DIR" && git describe --tags --always --abbrev=8 --dirty 2>/dev/null || echo "unknown")
            runtime_image="dynamo:v${version}-${framework,,}-runtime"
        fi

        local localdev_image=$(grep "docker build.*--tag.*local-dev" "$temp_output" | grep -oE "\-\-tag [^ ]+" | sed 's/--tag //' | tail -1)
        if [ -z "$localdev_image" ]; then
            # Fallback pattern for local-dev
            local version=$(cd "$SCRIPT_DIR" && git describe --tags --always --abbrev=8 --dirty 2>/dev/null || echo "unknown")
            localdev_image="dynamo:v${version}-${framework,,}-local-dev"
        fi

        # Extract UID/GID from the output
        local user_info=$(grep "User 'dynamo' will have UID:" "$temp_output" | tail -1)

        rm -f "$temp_runtime_output" "$temp_output"

        print_msg "[DRY-RUN] Would execute:"
        print_msg "  1. $runtime_build_cmd"
        print_msg "  2. $dev_build_cmd"
        print_msg "  3. $local_dev_build_cmd"
        print_msg "[DRY-RUN] This will build:"
        print_msg "  - Stage 1: Base image (if needed)"
        print_msg "  - Stage 2: Runtime image: $runtime_image"
        print_msg "  - Stage 3: Dev image (no target)"
        print_msg "  - Stage 4: Local-dev image: $localdev_image"
        print_msg "[DRY-RUN] Would then run sanity checks in container"
        if [ -n "$user_info" ]; then
            print_msg "[DRY-RUN] $user_info"
        fi
        return 0
    fi

    # First build runtime image
    print_msg "Building runtime image..."
    if eval "$runtime_build_cmd" 2>&1 | tee "$runtime_log_file" >&2; then
        # Extract the actual runtime image name from the log
        local runtime_image=$(grep -E "naming to docker.io/library/dynamo:.*-runtime" "$runtime_log_file" | tail -1 | sed 's/.*naming to docker.io\/library\///' | sed 's/ .*//')
        if [ -z "$runtime_image" ]; then
            # Fallback - construct from framework name
            runtime_image="dynamo:latest-${framework,,}-runtime"
        fi
        print_msg "Successfully built runtime image: $runtime_image"
    else
        print_msg "Failed to build $framework runtime image"
        return 1
    fi

    # Then build dev image (no target)
    print_msg "Building dev image..."
    if eval "$dev_build_cmd" 2>&1 | tee "$dev_log_file" >&2; then
        # Extract the actual dev image name from the log
        # Look for pattern like: "#57 naming to docker.io/library/dynamo:v0.7.0.dev.824902f82-vllm 0.0s done"
        local dev_image=$(grep -E "naming to docker.io/library/dynamo:.*-${framework,,}" "$dev_log_file" | grep -v "dynamo-base" | tail -1 | sed 's/.*naming to docker.io\/library\///' | awk '{print $1}')
        if [ -z "$dev_image" ]; then
            # Fallback - construct from framework name
            dev_image="dynamo:latest-${framework,,}"
        fi
        print_msg "Successfully built dev image: $dev_image"
    else
        print_msg "Failed to build $framework dev image"
        return 1
    fi

    # Finally build local-dev image
    print_msg "Building local-dev image..."
    if eval "$local_dev_build_cmd" 2>&1 | tee "$log_file" >&2; then
        # Extract the actual local-dev image name from the log
        local localdev_image=$(grep -E "naming to docker.io/library/dynamo:.*-local-dev" "$log_file" | tail -1 | sed 's/.*naming to docker.io\/library\///' | sed 's/ .*//')
        if [ -z "$localdev_image" ]; then
            # Fallback - construct from framework name
            localdev_image="dynamo:latest-${framework,,}-local-dev"
        fi
        print_msg "Successfully built local-dev image: $localdev_image"

        # Return all three images as "runtime_image dev_image localdev_image"
        echo "$runtime_image $dev_image $localdev_image"
        return 0
    else
        print_msg "Failed to build $framework local-dev image"
        return 1
    fi
}

# Function to verify runtime image
verify_runtime() {
    local framework=$1
    local image=$2
    local sanity_log="/tmp/${framework,,}.runtime.sanity.log"

    print_msg "\n=== Running runtime image verification for $framework ==="
    print_msg "Log file: $sanity_log"


    # Runtime - run sanity check with --runtime-check --thorough-check flags
    if container/run.sh --image "$image" -- python /workspace/deploy/sanity_check.py --runtime-check --thorough-check > "$sanity_log" 2>&1; then
        print_msg "✓ Runtime image verification passed"
        return 0
    else
        print_msg "✗ Runtime image verification failed"
        print_msg "Check log at: $sanity_log"
        return 1
    fi
}

# Function to verify dev or local-dev image
verify_dev_or_localdev() {
    local target=$1  # "dev" or "local-dev"
    local framework=$2
    local image=$3
    local verify_log="/tmp/${framework,,}.${target}.verify.log"

    # Fix workspace ownership before verification (dev runs as root and writes to target directory)
    # Must fix ownership of entire repo including target/ to avoid permission errors during cargo build
    print_msg "\n=== Running chown -R ... $REPO_DIR ==="
    sudo chown -R $(id -u):$(id -g) "$REPO_DIR"
    
    print_msg "\n=== Running ${target} image verification for $framework ==="
    print_msg "Log file: $verify_log"
    
    # Dev/local-dev: run cargo build, maturin, and sanity check with --mount-workspace
    local compilation_test='
    set -e
    cd /workspace
    echo "=== Running cargo build ==="
    cargo build --locked --features dynamo-llm/block-manager --workspace
    echo "=== Running maturin develop ==="
    cd /workspace/lib/bindings/python
    maturin develop --uv
    cd /workspace
    echo "=== Running sanity_check.py --thorough-check ==="
    python /workspace/deploy/sanity_check.py --thorough-check
    '
    
    if container/run.sh --image "$image" --mount-workspace -- bash -c "$compilation_test" > "$verify_log" 2>&1; then
        print_msg "✓ ${target} image verification passed"
        return 0
    else
        print_msg "✗ ${target} image verification failed"
        print_msg "Check log at: $verify_log"
        return 1
    fi
}

# Function to build and verify a single framework
build_and_verify_framework() {
    local fw=$1
    local dry_run=$2
    local no_cache=$3
    
    # Check for unsupported frameworks
    if [[ "$fw" == "NONE" ]]; then
        print_msg "\n=== $fw framework is not supported ==="
        print_msg "The NONE framework (base image without inference backend) cannot have a local-dev variant."
        print_msg "Skipping $fw..."
        return 0
    fi

    # Check if framework is valid
    if [[ "$fw" != "VLLM" && "$fw" != "SGLANG" && "$fw" != "TRTLLM" ]]; then
        print_msg "ERROR: Unknown framework: $fw"
        print_msg "Available frameworks: vllm, sglang, trtllm"
        return 1
    fi

    print_msg "\n========================================="
    print_msg "Processing $fw Framework"
    print_msg "========================================="

    # Build complete pipeline (runtime + dev + local-dev)
    local images=$(build_all_targets "$fw" "$dry_run" "$no_cache")
    local build_result=$?
    if [ $build_result -ne 0 ]; then
        print_msg "Failed to build $fw complete pipeline"
        return 1
    fi

    # Run verification if not dry-run
    if [ "$dry_run" = false ] && [ -n "$images" ]; then
        # Split the images (returns "runtime_image dev_image localdev_image")
        local runtime_image=$(echo "$images" | awk '{print $1}')
        local dev_image=$(echo "$images" | awk '{print $2}')
        local localdev_image=$(echo "$images" | awk '{print $3}')
        
        if [ -n "$runtime_image" ]; then
            verify_runtime "$fw" "$runtime_image"
        fi
        if [ -n "$dev_image" ]; then
            verify_dev_or_localdev "dev" "$fw" "$dev_image"
        fi
        if [ -n "$localdev_image" ]; then
            verify_dev_or_localdev "local-dev" "$fw" "$localdev_image"
        fi
    fi
    
    print_msg "\n✓ Successfully completed $fw framework build pipeline"
    return 0
}

# Main execution
main() {
    local framework=""
    local dry_run=false
    local no_cache=false
    local unknown_args=()

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --framework)
                if [ -z "$2" ] || [[ "$2" == --* ]]; then
                    print_msg "ERROR: --framework requires a value"
                    exit 1
                fi
                framework=$2
                shift 2
                ;;
            --dry-run|--dryrun)
                dry_run=true
                shift
                ;;
            --no-cache)
                no_cache=true
                shift
                ;;
            -h|--help)
                print_msg "Usage: $0 --framework <framework> [OPTIONS]"
                print_msg ""
                print_msg "Required:"
                print_msg "  --framework <name>   Framework to build (vllm, sglang, trtllm, all)"
                print_msg ""
                print_msg "Options:"
                print_msg "  --dry-run, --dryrun  Show what would be executed without actually running builds"
                print_msg "  --no-cache           Force rebuild without using cache"
                print_msg "  -h, --help           Show this help message"
                print_msg ""
                print_msg "Available frameworks:"
                print_msg "  vllm     - Build vLLM runtime and local-dev images"
                print_msg "  sglang   - Build SGLang runtime and local-dev images"
                print_msg "  trtllm   - Build TensorRT-LLM runtime and local-dev images"
                print_msg "  all      - Build all frameworks"
                print_msg ""
                print_msg "Examples:"
                print_msg "  $0 --framework vllm"
                print_msg "  $0 --framework trtllm --dry-run"
                print_msg "  $0 --framework all"
                exit 0
                ;;
            -*)
                print_msg "ERROR: Unknown option: $1"
                print_msg "Use --help for usage information"
                exit 1
                ;;
            *)
                unknown_args+=("$1")
                shift
                ;;
        esac
    done

    # Check for unexpected arguments
    if [ ${#unknown_args[@]} -gt 0 ]; then
        print_msg "ERROR: Unexpected arguments: ${unknown_args[*]}"
        print_msg "Use --help for usage information"
        exit 1
    fi

    # Default to 'all' if framework not specified
    if [ -z "$framework" ]; then
        framework="all"
    fi

    # Convert to uppercase for comparison
    framework=${framework^^}

    # Validate framework before proceeding
    case "$framework" in
        VLLM|SGLANG|TRTLLM|ALL|NONE)
            # Valid framework
            ;;
        *)
            print_msg "ERROR: Invalid framework: '${framework,,}'"
            print_msg "Available frameworks: vllm, sglang, trtllm, all"
            print_msg "Use --help for more information"
            exit 1
            ;;
    esac

    if [ "$dry_run" = true ]; then
        print_msg "\n=== DRY RUN MODE - No actual builds will be performed ==="
    fi

    # Handle 'all' option
    if [[ "$framework" == "ALL" ]]; then
        FRAMEWORKS=("VLLM" "SGLANG" "TRTLLM")
    else
        FRAMEWORKS=("$framework")
    fi

    # Process frameworks
    if [[ "$framework" == "ALL" ]]; then
        # Run builds in parallel for "all" option
        print_msg "\n=== Starting parallel builds for all frameworks ==="

        local pids=()
        local failed_frameworks=()
        declare -A built_images

        for fw in "${FRAMEWORKS[@]}"; do
            # Skip unsupported frameworks
            if [[ "$fw" == "NONE" ]]; then
                continue
            fi

            # Start build in background and save images to temp file
            local temp_file="/tmp/${fw,,}.build.images.txt"
            (
                # Build only (no verification in parallel to avoid workspace conflicts)
                local images=$(build_all_targets "$fw" "$dry_run" "$no_cache")
                if [ $? -eq 0 ] && [ -n "$images" ]; then
                    echo "$images" > "$temp_file"
                    exit 0
                else
                    exit 1
                fi
            ) &

            # Store the PID
            pids+=($!)
            print_msg "Started $fw build with PID: ${pids[-1]}"
        done

        # Wait for all builds to complete and check results
        print_msg "\n=== Waiting for all builds to complete ==="
        local all_success=true
        for i in "${!pids[@]}"; do
            local pid=${pids[$i]}
            local fw=${FRAMEWORKS[$i]}

            wait $pid
            local result=$?

            if [ $result -eq 0 ]; then
                print_msg "✓ $fw build completed successfully"
                # Read the images from temp file
                local temp_file="/tmp/${fw,,}.build.images.txt"
                if [ -f "$temp_file" ]; then
                    built_images["$fw"]=$(cat "$temp_file")
                    rm -f "$temp_file"
                fi
            else
                print_msg "✗ $fw build failed"
                failed_frameworks+=($fw)
                all_success=false
            fi
        done

        if [ "$all_success" = true ]; then
            print_msg "\n========================================="
            print_msg "All builds completed successfully!"
            print_msg "========================================="

            # Run verifications SEQUENTIALLY to avoid workspace conflicts (dev runs as root)
            if [ "$dry_run" = false ]; then
                print_msg "\n=== Running verifications sequentially for all frameworks ==="
                for fw in "${!built_images[@]}"; do
                    local images="${built_images[$fw]}"
                    if [ -n "$images" ]; then
                        # Split the images (returns "runtime_image dev_image localdev_image")
                        local runtime_image=$(echo "$images" | awk '{print $1}')
                        local dev_image=$(echo "$images" | awk '{print $2}')
                        local localdev_image=$(echo "$images" | awk '{print $3}')
                        
                        if [ -n "$runtime_image" ]; then
                            verify_runtime "$fw" "$runtime_image"
                        fi
                        if [ -n "$dev_image" ]; then
                            verify_dev_or_localdev "dev" "$fw" "$dev_image"
                        fi
                        if [ -n "$localdev_image" ]; then
                            verify_dev_or_localdev "local-dev" "$fw" "$localdev_image"
                        fi
                    fi
                done
                print_msg "\n=== All verifications completed ==="
            fi
        else
            print_msg "\n========================================="
            print_msg "Some builds failed: ${failed_frameworks[*]}"
            print_msg "========================================="
            exit 1
        fi
    else
        # Single framework build (sequential)
        for fw in "${FRAMEWORKS[@]}"; do
            if ! build_and_verify_framework "$fw" "$dry_run" "$no_cache"; then
                exit 1
            fi
        done

        print_msg "\n========================================="
        print_msg "Build completed successfully!"
        print_msg "========================================="
    fi

    # Show final images and logs (only if not in dry-run mode)
    if [ "$dry_run" = false ]; then
        print_msg "\nCreated Docker images:"
        docker images | grep -E "dynamo.*runtime|dynamo.*local-dev" | head -10

        print_msg "\nLog files created:"
        for fw in "${FRAMEWORKS[@]}"; do
            if [[ "$fw" != "NONE" ]]; then
                local fw_lower="${fw,,}"
                [ -f "/tmp/${fw_lower}.build.log" ] && print_msg "  Build log: /tmp/${fw_lower}.build.log"
                [ -f "/tmp/${fw_lower}.run.log" ] && print_msg "  Run log:   /tmp/${fw_lower}.run.log"
            fi
        done
    fi
}

# Run main function with all arguments
main "$@"
