#!/usr/bin/env bash
# Test script for build_dev_from_runtime.py
# This script builds dev and local-dev images for all frameworks and tests them

set -euo pipefail

# Parse arguments
IMAGE_TAG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --image)
            IMAGE_TAG="$2"
            shift 2
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            echo "Usage: $0 --image <base-image-tag>"
            echo "Example: $0 --image ca31b3aa6b9efb1afcd4fcad2c98c10552aa4bad-42233876"
            exit 1
            ;;
    esac
done

if [[ -z "${IMAGE_TAG}" ]]; then
    echo "ERROR: --image parameter is required"
    echo "Usage: $0 --image <base-image-tag>"
    echo "Example: $0 --image ca31b3aa6b9efb1afcd4fcad2c98c10552aa4bad-42233876"
    exit 1
fi

# Configuration
BASE_IMAGE_TAG="${IMAGE_TAG}"
FRAMEWORKS=("sglang" "trtllm" "vllm")
REGISTRY="gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo"
ARCH="amd64"
TARGETS=("dev" "local-dev")

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_SCRIPT="${SCRIPT_DIR}/build_dev_from_runtime.py"

# Find the dynamo repo (go up from dynamo-utils.dev)
DYNAMO_UTILS_ROOT="$(dirname "${SCRIPT_DIR}")"
WORKSPACE_ROOT="$(dirname "${DYNAMO_UTILS_ROOT}")"

# Use dynamo4 specifically
DYNAMO_REPO="dynamo4"
RUN_SH="${WORKSPACE_ROOT}/${DYNAMO_REPO}/container/run.sh"
DEPLOY_SANITY="${WORKSPACE_ROOT}/${DYNAMO_REPO}/deploy/sanity_check.py"

if [[ ! -f "${RUN_SH}" ]]; then
    echo "ERROR: Could not find container/run.sh at: ${RUN_SH}"
    exit 1
fi

if [[ ! -f "${DEPLOY_SANITY}" ]]; then
    echo "ERROR: Could not find deploy/sanity_check.py at: ${DEPLOY_SANITY}"
    exit 1
fi

echo "Using run.sh from: ${RUN_SH}"
echo "Using sanity_check.py from: ${DEPLOY_SANITY}"

log_info() {
    echo "[INFO] $*"
}

log_success() {
    echo "[SUCCESS] $*"
}

log_error() {
    echo "[ERROR] $*"
}

log_warning() {
    echo "[WARNING] $*"
}

# Compilation command to run inside container (as a single string to pass after --)
COMPILE_CMD='export DYNAMO_USE_PREBUILT_KERNELS=1 CARGO_PROFILE_DEV_OPT_LEVEL=0 CARGO_BUILD_JOBS=32 CARGO_PROFILE_DEV_CODEGEN_UNITS=256; cd /workspace && cargo build && cd /workspace/lib/bindings/python && maturin develop --uv'

# Test results tracking
declare -A TEST_RESULTS

test_image() {
    local framework="$1"
    local target="$2"
    local image_tag="$3"
    local test_name="${framework}-${target}"
    
    log_info "=========================================="
    log_info "Testing: ${test_name}"
    log_info "Image: ${image_tag}"
    log_info "=========================================="
    
    # For local-dev, test id and sudo id first
    if [[ "${target}" == "local-dev" ]]; then
        log_info "Testing user permissions with 'id' and 'sudo id'..."
        if "${RUN_SH}" --image "${image_tag}" --mount-workspace -- -c "id && sudo id"; then
            log_success "User permissions OK for ${test_name}"
        else
            log_error "User permissions check failed for ${test_name}"
            TEST_RESULTS["${test_name}"]="FAILED (id check)"
            return 1
        fi
    fi
    
    # Run compilation
    log_info "Running compilation inside container..."
    if "${RUN_SH}" --image "${image_tag}" --mount-workspace -- -c "${COMPILE_CMD}"; then
        log_success "Compilation succeeded for ${test_name}"
    else
        log_error "Compilation failed for ${test_name}"
        TEST_RESULTS["${test_name}"]="FAILED (compilation)"
        return 1
    fi
    
    # Run sanity check inside the container
    log_info "Running sanity check inside container..."
    if "${RUN_SH}" --image "${image_tag}" --mount-workspace -- -c "python3 /workspace/deploy/sanity_check.py"; then
        log_success "Sanity check passed for ${test_name}"
        TEST_RESULTS["${test_name}"]="PASSED"
    else
        log_error "Sanity check failed for ${test_name}"
        TEST_RESULTS["${test_name}"]="FAILED (sanity check)"
        return 1
    fi
    
    log_success "All tests passed for ${test_name}"
    return 0
}

main() {
    log_info "Starting build_dev_from_runtime.py test suite"
    log_info "Base image tag: ${BASE_IMAGE_TAG}"
    log_info "Frameworks: ${FRAMEWORKS[*]}"
    log_info "Targets: ${TARGETS[*]}"
    echo ""
    
    # Build and test each combination
    for framework in "${FRAMEWORKS[@]}"; do
        # Set up log file for this framework
        LOG_FILE="/tmp/runtime-${framework}.log"
        log_info "Logging to ${LOG_FILE}"
        
        # Redirect all output for this framework to its log file
        {
        
        # Build dev first
        RUNTIME_IMAGE="${REGISTRY}:${BASE_IMAGE_TAG}-${framework}-${ARCH}"
        OUTPUT_TAG_DEV="dynamo:${BASE_IMAGE_TAG}-${framework}-${ARCH}-dev"
        
        log_info "=========================================="
        log_info "Building ${framework} dev image"
        log_info "=========================================="
        log_info "Runtime image: ${RUNTIME_IMAGE}"
        log_info "Output tag: ${OUTPUT_TAG_DEV}"
        echo ""
        
        # Build the dev image
        if python3 "${BUILD_SCRIPT}" --target "dev" --no-tag-latest "${RUNTIME_IMAGE}"; then
            log_success "Built ${OUTPUT_TAG_DEV}"
            echo ""
            
            # Test the dev image
            if test_image "${framework}" "dev" "${OUTPUT_TAG_DEV}"; then
                log_success "Test passed: ${framework}-dev"
            else
                log_error "Test failed: ${framework}-dev"
            fi
        else
            log_error "Failed to build ${OUTPUT_TAG_DEV}"
            TEST_RESULTS["${framework}-dev"]="FAILED (build)"
        fi
        
        echo ""
        
        # Fix ownership after dev (which runs as root) before local-dev (which runs as user)
        log_info "Fixing ownership of target/ directories after dev, before local-dev..."
        if sudo chown -R "$(id -u):$(id -g)" "${WORKSPACE_ROOT}"/*/target/ 2>/dev/null; then
            log_success "Ownership fixed"
        else
            log_info "No target directories to fix (this is fine)"
        fi
        
        echo ""
        
        # Build local-dev
        OUTPUT_TAG_LOCAL_DEV="dynamo:${BASE_IMAGE_TAG}-${framework}-${ARCH}-local-dev"
        
        log_info "=========================================="
        log_info "Building ${framework} local-dev image"
        log_info "=========================================="
        log_info "Runtime image: ${RUNTIME_IMAGE}"
        log_info "Output tag: ${OUTPUT_TAG_LOCAL_DEV}"
        echo ""
        
        # Build the local-dev image
        if python3 "${BUILD_SCRIPT}" --target "local-dev" --no-tag-latest "${RUNTIME_IMAGE}"; then
            log_success "Built ${OUTPUT_TAG_LOCAL_DEV}"
            echo ""
            
            # Test the local-dev image
            if test_image "${framework}" "local-dev" "${OUTPUT_TAG_LOCAL_DEV}"; then
                log_success "Test passed: ${framework}-local-dev"
            else
                log_error "Test failed: ${framework}-local-dev"
            fi
        else
            log_error "Failed to build ${OUTPUT_TAG_LOCAL_DEV}"
            TEST_RESULTS["${framework}-local-dev"]="FAILED (build)"
        fi
        
        echo ""
        echo ""
        
        } 2>&1 | tee "${LOG_FILE}"
        
        log_info "Completed ${framework}, log saved to ${LOG_FILE}"
        echo ""
    done
    
    # Print summary
    log_info "=========================================="
    log_info "TEST SUMMARY"
    log_info "=========================================="
    
    local passed=0
    local failed=0
    
    for framework in "${FRAMEWORKS[@]}"; do
        for target in "${TARGETS[@]}"; do
            local test_name="${framework}-${target}"
            local result="${TEST_RESULTS[${test_name}]:-SKIPPED}"
            
            if [[ "${result}" == "PASSED" ]]; then
                log_success "${test_name}: ${result}"
                ((passed++))
            else
                log_error "${test_name}: ${result}"
                ((failed++))
            fi
        done
    done
    
    echo ""
    log_info "Total: $((passed + failed)) tests"
    log_success "Passed: ${passed}"
    if [[ ${failed} -gt 0 ]]; then
        log_error "Failed: ${failed}"
        exit 1
    else
        log_success "All tests passed!"
    fi
}

# Run main function
main "$@"
