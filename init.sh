#! /bin/bash

set -euo pipefail

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Initialize the Dynamo development environment.

OPTIONS:
    -h, --help    Show this help message and exit

DESCRIPTION:
    This script sets up the Dynamo development environment by:
    - Creating symbolic links to built binaries in the CLI bin directory
    - Setting up the PYTHONPATH for the SDK and planner components

    When running inside a Docker container, it uses /workspace as the base directory.
    When running outside a Docker container, it uses \$HOME/dynamo as the base directory.
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

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

cd $WORKSPACE_DIR
if [ ! -f "README.md" ] || [ ! -d "components" ] || [ ! -d "container" ]; then
    echo "Required files or directories (README.md, components, container) are missing."
    exit 1
fi

# Check for .build/target first, otherwise use target
if [ -d "$WORKSPACE_DIR/.build/target" ]; then
    if [ -d "$WORKSPACE_DIR/target" ] && [ ! -L "$WORKSPACE_DIR/target" ]; then
        rm -rf "$WORKSPACE_DIR/target"
        ln -s "$WORKSPACE_DIR/.build/target" "$WORKSPACE_DIR/target"
    fi
    BUILD_TARGET=$WORKSPACE_DIR/.build/target
else
    BUILD_TARGET=$WORKSPACE_DIR/target
fi
mkdir -p $BUILD_TARGET


set -x

# Start the setup process (only runs if not already completed)
pkill -f "tail -f /dev/null" || true
if [ ! -f /tmp/.setupdone ]; then
    # Clean up Python cache and build artifacts
    find $WORKSPACE_DIR -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find $WORKSPACE_DIR -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    find $WORKSPACE_DIR -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" \) -delete 2>/dev/null || true
    #rm -rf /opt/dynamo/venv/lib/python3.12/site-packages/pyright/dist

    sudo apt update -y
    sudo apt install -y git protobuf-compiler libclang-dev clang file curl

    CARGO_INCREMENTAL=1 cargo build --workspace --bin dynamo-run

    if ! uv pip show maturin >/dev/null 2>&1; then
        echo "Installing maturin with patchelf support..."
        sudo uv pip install maturin[patchelf]
    else
        echo "maturin is already installed, skipping installation"
    fi

    # Build Python wheel
    echo "Building Python wheel..."
    mkdir -p $WORKSPACE_DIR/dist
    uv build --wheel --out-dir $WORKSPACE_DIR/dist

    # Build Rust bindings with block-manager feature
    echo "Building Rust bindings with block-manager feature..."
    cd $WORKSPACE_DIR/lib/bindings/python && \
    maturin build --features block-manager --out $WORKSPACE_DIR/dist

    # Conditional release builds for Python 3.11 and 3.10 (if RELEASE_BUILD=true)
    if [ "${RELEASE_BUILD:-false}" = "true" ]; then
        echo "Building release wheels for Python 3.11 and 3.10..."
        cd $WORKSPACE_DIR/lib/bindings/python && \
        uv run --python 3.11 maturin build --out $WORKSPACE_DIR/dist && \
        uv run --python 3.10 maturin build --out $WORKSPACE_DIR/dist
    fi

    # install the python bindings (development mode)
    cd $WORKSPACE_DIR/lib/bindings/python && maturin develop

    # installs overall python packages, grabs binaries from .build/target/debug
    cd $WORKSPACE_DIR && uv pip install -e .

    export PYTHONPATH=$WORKSPACE_DIR/components/planner/src:$PYTHONPATH

    touch /tmp/.setupdone
else
    echo "Setup already completed, skipping apt-get commands."
fi

echo "Ok, looks like the environment is set up correctly."
