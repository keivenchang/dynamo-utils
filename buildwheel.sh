#!/bin/bash

# Exit on any error, undefined variables, and pipe failures
set -euo pipefail

# Parse command line arguments
BUILD_TYPE="debug"  # Default to debug build
CLEAN_BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --release)
            BUILD_TYPE="release"
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--release] [--clean]"
            echo "  --release: Build in release mode with optimizations (slower compilation, faster runtime)"
            echo "  --clean: Clean all build artifacts before building"
            echo "  Default: Debug build with incremental compilation"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

set -x

# Debug builds automatically use incremental compilation
if [ "$BUILD_TYPE" = "debug" ]; then
    INCREMENTAL=true
else
    INCREMENTAL=false
fi

# Check if we're running inside a Docker container
if [ ! -e /.dockerenv ]; then
    echo "ERROR: This script must be run inside a Docker container."
    echo "Please run this script from within the Dynamo container."
    exit 1
fi

# Determine workspace directory - check both possible locations (home/ubuntu/dynamo takes precedence)
if [ -d "~/dynamo" ] && [ -f "~/dynamo/README.md" ]; then
    echo "Running inside Docker container with ~/dynamo"
    WORKSPACE_DIR=~/dynamo
elif [ -d "/workspace" ] && [ -f "/workspace/README.md" ]; then
    echo "Running inside Docker container with /workspace"
    WORKSPACE_DIR=/workspace
else
    echo "ERROR: Could not find Dynamo workspace in ~/dynamo or /workspace"
    echo "Please ensure the Dynamo source code is mounted in the container."
    exit 1
fi

# Wheel output directory
WHEEL_OUTPUT_DIR="$WORKSPACE_DIR/dist"

echo "Build configuration:"
echo "  Build type: $BUILD_TYPE"
echo "  Incremental: $INCREMENTAL"
echo "  Clean build: $CLEAN_BUILD"
echo "  Workspace: $WORKSPACE_DIR"
echo "  Wheel output: $WHEEL_OUTPUT_DIR"
echo ""

# Change to workspace directory where the source code is mounted
cd $WORKSPACE_DIR

# Clean build artifacts if requested (before setting incremental config)
if [ "$CLEAN_BUILD" = true ]; then
    echo "Cleaning all build artifacts..."
    cargo clean
    echo "Build artifacts cleaned"
fi
mkdir -p target
USER_ID=$(stat -c "%u" .)
GROUP_ID=$(stat -c "%g" .)
chown -R $USER_ID:$GROUP_ID .

# Configure incremental compilation if requested
if [ "$INCREMENTAL" = true ]; then
    echo "Enabling incremental compilation..."
    export CARGO_INCREMENTAL=1
    echo "CARGO_INCREMENTAL set to: $CARGO_INCREMENTAL"
else
    echo "Using standard compilation (no incremental)"
    export CARGO_INCREMENTAL=0
    echo "CARGO_INCREMENTAL set to: $CARGO_INCREMENTAL"
fi

# Verify environment variable is set
echo "Environment verification: CARGO_INCREMENTAL=$CARGO_INCREMENTAL"

# Build Rust components with block-manager feature
# --features dynamo-llm/block-manager: Enable block manager functionality
# --workspace: Build all workspace members
if [ "$BUILD_TYPE" = "release" ]; then
    echo "Building Rust components in RELEASE mode..."
    cargo build --release --locked --features dynamo-llm/block-manager --workspace
else
    echo "Building Rust components in DEBUG mode..."
    cargo build --locked --features dynamo-llm/block-manager --workspace
fi

# Clean up any existing wheel files from previous builds
if ls $WHEEL_OUTPUT_DIR/*.whl 1> /dev/null 2>&1; then
    echo "Removing existing wheel files..."
    rm -f $WHEEL_OUTPUT_DIR/*.whl
    echo "Existing wheel files removed"
else
    echo "No existing wheel files found"
fi

# Build Python wheel packages using uv
# --wheel: Create wheel distribution format
# --out-dir: Specify output directory for built wheels
uv build --wheel --out-dir $WHEEL_OUTPUT_DIR

# Change to Python bindings directory for native extension build
cd $WORKSPACE_DIR/lib/bindings/python

# Install maturin with patchelf support for building Python extensions (if not already installed)
# maturin: Tool for building Python packages with Rust code
# patchelf: Dependency for handling shared library dependencies
if ! uv pip show maturin >/dev/null 2>&1; then
    echo "Installing maturin with patchelf support..."
    uv pip install maturin[patchelf]
else
    echo "maturin is already installed, skipping installation"
fi

# Build Python native extension with maturin
# --features block-manager: Enable block manager feature
# --out: Specify output directory for the built wheel
if [ "$BUILD_TYPE" = "release" ]; then
    echo "Building Python extension in RELEASE mode..."
    maturin build --release --features block-manager --out $WHEEL_OUTPUT_DIR
else
    echo "Building Python extension in DEBUG mode..."
    maturin build --features block-manager --out $WHEEL_OUTPUT_DIR
fi

# Install the newly built wheels in development mode
# --upgrade: Upgrade packages if already installed
# --force-reinstall: Force reinstallation even if already installed
# --no-deps: Don't install dependencies (assumes they're already available)
uv pip install --upgrade --force-reinstall --no-deps $WHEEL_OUTPUT_DIR/*.whl

set +x

# List the newly built wheel files
echo ""
echo "=== Built Wheel Files ==="
if ls $WHEEL_OUTPUT_DIR/*.whl 1> /dev/null 2>&1; then
    ls -la $WHEEL_OUTPUT_DIR/*.whl
    echo ""
    echo "Total wheel files: $(ls $WHEEL_OUTPUT_DIR/*.whl | wc -l)"
else
    echo "No wheel files found in $WHEEL_OUTPUT_DIR/"
fi
echo "========================"
