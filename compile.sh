#!/bin/bash

# ==============================================================================
# Dynamo Python Package Builder
# ==============================================================================
#
# Builds and installs Python packages for the Dynamo distributed inference framework.
#
# PACKAGES BUILT:
#   ai-dynamo-runtime: Core Rust extensions + Python bindings
#      - Python Import: dynamo._core, dynamo.runtime, dynamo.llm, dynamo.nixl_connect
#      - Contains: Compiled Rust extension (~45MB), Python modules (~16MB)
#   ai-dynamo: Complete framework with all components
#      - Python Import: dynamo.frontend, dynamo.planner, dynamo.vllm, etc.
#      - Contains: Frontend, planner, and backend components (~90KB)
#
# BUILD MODES (mutually exclusive - ensures clean installations):
#   Development (--dev or --development):
#      - Creates editable installation for ai-dynamo-runtime using maturin develop
#      - Python changes are immediately available (hot-reload)
#      - Rust changes require re-running this script
#      - FASTER BUILD: Uses debug compilation with incremental builds
#      - SLOWER RUNTIME: Debug binaries are less optimized
#      - Force-removes existing installations before proceeding
#   Release (--release):
#      - Creates optimized wheel files for both packages
#      - Installs both ai-dynamo-runtime and ai-dynamo from wheels
#      - SLOWER BUILD: Full optimization takes more time to compile
#      - FASTER RUNTIME: Optimized binaries for production performance
#      - Suitable for production deployment and distribution
#      - Force-removes existing installations before proceeding
#
# REQUIREMENTS:
#   - Docker container environment
#   - Python >= 3.10 with maturin and uv
#   - Workspace at $HOME/dynamo or /workspace
#
# PYTHONPATH SETUP:
#   For development, Python searches these paths before site-packages:
#   export PYTHONPATH="$HOME/dynamo/components/*/src:$PYTHONPATH"
#
#   This enables importing components directly from source for hot-reload development.
#
# USAGE:
#   ./compile.sh --dev             # Development mode (fast build, slower runtime)
#   ./compile.sh --release         # Production wheels (slow build, fast runtime)
#   ./compile.sh --python-clean    # Remove all Python packages and exit
#   ./compile.sh --cargo-clean     # Remove Rust build artifacts and exit
#
# ==============================================================================

# Exit on any error, undefined variables, and pipe failures
set -euo pipefail

# ==============================================================================
# COMMAND LINE ARGUMENT PARSING
# ==============================================================================
# Parse and validate command line arguments to determine build configuration

BUILD_TYPE=""              # Build type selection (development or release)
CARGO_CLEAN=false         # Flag for --cargo-clean mode
PYTHON_CLEAN=false        # Flag for --python-clean mode
DEBUG=false               # Flag for --debug mode
BUILD_RUST=true          # Flag for --rust-only/--rust (default: true)
BUILD_PYTHON=true        # Flag for --py-only/--py (default: true)
DRY_RUN=false            # Flag for --dryrun/--dry-run mode

while [[ $# -gt 0 ]]; do
    case $1 in
        --development|--dev)
            BUILD_TYPE="development"
            shift
            ;;
        --release)
            BUILD_TYPE="release"
            shift
            ;;
        --rust-only|--rust)
            BUILD_RUST=true
            BUILD_PYTHON=false
            shift
            ;;
        --py-only|--py)
            BUILD_RUST=false
            BUILD_PYTHON=true
            # If no build type specified, default to development for py-only builds
            if [ -z "$BUILD_TYPE" ]; then
                BUILD_TYPE="development"
            fi
            shift
            ;;
        --cargo-clean|--rust-clean)
            CARGO_CLEAN=true
            shift
            ;;
        --python-clean|--py-clean)
            PYTHON_CLEAN=true
            shift
            ;;
        --clean)
            CARGO_CLEAN=true
            PYTHON_CLEAN=true
            shift
            ;;

        --debug)
            DEBUG=true
            shift
            ;;
        --dryrun|--dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            cat << 'EOF'
Usage: compile.sh [OPTIONS]

Builds and installs Python packages for the Dynamo distributed inference framework.

BUILD MODES (mutually exclusive, --dev is default if none specified):
  --development, --dev    Build ai-dynamo-runtime in development mode (editable install) [DEFAULT]
  --release               Build both ai-dynamo-runtime and ai-dynamo wheels for production

BUILD COMPONENTS (mutually exclusive):
  --rust-only, --rust     Build only Rust components (cargo build)
  --py-only, --py         Build only Python components (wheel/maturin/uv)
                          Default: Build both Rust and Python components

OTHER OPTIONS:
  --clean                 Clean both Rust build artifacts and Python packages (can be used standalone or with build)
  --cargo-clean           Clean Rust build artifacts (can be used standalone or with build)
  --python-clean          Remove all dynamo packages (.pth, .whl, pip/uv) and exit
  --debug                 Show detailed .pth file information during development builds
  --dryrun, --dry-run     Show what would be executed without running (dry run mode)

EXAMPLES:
  Standalone operations:
    • ./compile.sh --clean              # Clean both Rust and Python artifacts
    • ./compile.sh --cargo-clean        # Only clean Rust build artifacts
    • ./compile.sh --python-clean       # Only remove Python packages

  Combined operations:
    • ./compile.sh                      # Build in development mode
    • ./compile.sh --release --cargo-clean  # Clean then build optimized wheels
    • ./compile.sh --rust-only          # Build only Rust components in development mode
    • ./compile.sh --py-only            # Build only Python components in development mode
    • ./compile.sh --dev --dry-run      # Show what would be built in development mode

PACKAGES BUILT:
  ai-dynamo-runtime: Core Rust extensions + Python bindings
     - Python Import: dynamo._core, dynamo.runtime, dynamo.llm, dynamo.nixl_connect
     - Contains: Compiled Rust extension (~45MB), Python modules (~16MB)
  ai-dynamo: Complete framework with all components
     - Python Import: dynamo.frontend, dynamo.planner, dynamo.vllm, etc.
     - Contains: Frontend, planner, and backend components (~90KB)

REQUIREMENTS:
  - Docker container environment
  - Python >= 3.10 with maturin and uv
  - Workspace at $HOME/dynamo or /workspace
EOF
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ==============================================================================
# WORKSPACE DIRECTORY DISCOVERY
# ==============================================================================
# Determine workspace directory early for all operations

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -d "$HOME/dynamo" ] && [ -f "$HOME/dynamo/README.md" ]; then
    WORKSPACE_DIR=$HOME/dynamo
elif [ -d "/workspace" ] && [ -f "/workspace/README.md" ]; then
    WORKSPACE_DIR=/workspace
else
    if [ "$PYTHON_CLEAN" = true ]; then
        WORKSPACE_DIR=""
    else
        echo "❌ ERROR: Could not find Dynamo workspace in expected locations"
        echo "   Checked: $HOME/dynamo and /workspace"
        echo "   Please ensure the Dynamo source code is properly mounted in the container:"
        echo "   docker run -v /path/to/dynamo:/workspace dynamo-container"
        exit 1
    fi
fi

# ==============================================================================
# CARGO TARGET DIRECTORY SETUP
# ==============================================================================
# Set up CARGO_TARGET_DIR early so it's available for all operations

if [ -n "$WORKSPACE_DIR" ]; then
    # Use cargo target directory for wheels (follows Rust conventions)
    CARGO_TARGET_DIR=$(cargo metadata --format-version=1 --no-deps 2>/dev/null | jq -r '.target_directory' 2>/dev/null || echo "$WORKSPACE_DIR/target")
else
    CARGO_TARGET_DIR=""
fi

# ==============================================================================
# FUNCTION DEFINITIONS
# ==============================================================================
# Define all functions used by this script

# Function to handle dry run output
dry_run_echo() {
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] $*"
    else
        echo "$*"
    fi
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

get_dynamo_version() {
    # Extract version from Cargo.toml workspace.package section
    if [ -f "$WORKSPACE_DIR/Cargo.toml" ]; then
        awk '/^\[workspace\.package\]/{f=1; next} /^\[/{f=0} f && /^version = /{print $3}' "$WORKSPACE_DIR/Cargo.toml" | tr -d '"'
    else
        echo "unknown"
    fi
}

cleanup_python() {
    # Force uninstall with uv pip
    if uv pip show ai-dynamo-runtime >/dev/null 2>&1; then
        cmd uv pip uninstall ai-dynamo-runtime
    fi
    if uv pip show ai-dynamo >/dev/null 2>&1; then
        cmd uv pip uninstall ai-dynamo
    fi

    # Clean up any remaining artifacts in site-packages
    local python_site_packages=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "")
    if [ -n "$python_site_packages" ]; then
        # Remove ALL dynamo-related .dist-info directories
        for dist_info in "$python_site_packages"/ai_dynamo*-*.dist-info; do
            if [ -d "$dist_info" ]; then
                cmd rm -rf "$dist_info"
            fi
        done

        # Remove ALL .pth files
        for pth_file in "$python_site_packages"/*dynamo*.pth "$python_site_packages"/_*dynamo*.pth; do
            if [ -f "$pth_file" ]; then
                cmd rm -f "$pth_file"
            fi
        done

        # Remove editables finder
        if [ -f "$python_site_packages/__editables_finder__.py" ]; then
            cmd rm -f "$python_site_packages/__editables_finder__.py"
        fi
    fi

    # Clean up wheel files if in workspace
    if [ -n "${WHEEL_OUTPUT_DIR:-}" ] && [ -d "${WHEEL_OUTPUT_DIR:-}" ]; then
        for wheel_file in "$WHEEL_OUTPUT_DIR"/*.whl; do
            if [ -f "$wheel_file" ]; then
                cmd rm -f "$wheel_file"
            fi
        done
        # Remove wheel directory if empty
        if [ -d "${WHEEL_OUTPUT_DIR:-}" ] && [ -z "$(ls -A "${WHEEL_OUTPUT_DIR:-}")" ]; then
            cmd rmdir "$WHEEL_OUTPUT_DIR"
        fi
    fi
}






# ==============================================================================
# 1. PYTHON CLEAN (if requested, do this first and exit)
# ==============================================================================

if [ "$PYTHON_CLEAN" = true ]; then
    cleanup_python
fi

# ==============================================================================
# 2. CARGO CLEAN (if requested, do this at the start)
# ==============================================================================

if [ "$CARGO_CLEAN" = true ]; then
    if [ -n "$WORKSPACE_DIR" ]; then
        cd "$WORKSPACE_DIR"
        cmd cargo clean
    else
        # If only cargo clean was requested, exit with error
        if [ -z "$BUILD_TYPE" ]; then
            exit 1
        fi
    fi
fi

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================
# Set up paths (needed for both build and check operations)

if [ -n "$WORKSPACE_DIR" ]; then
    WHEEL_OUTPUT_DIR="$CARGO_TARGET_DIR/wheels"
    PYTHON_BINDINGS_PATH="$WORKSPACE_DIR/lib/bindings/python"
    # Get dynamic version from Cargo.toml
    DYNAMO_VERSION=$(get_dynamo_version)
else
    WHEEL_OUTPUT_DIR=""
    PYTHON_BINDINGS_PATH=""
    DYNAMO_VERSION="unknown"
fi

# ==============================================================================
# 3. BUILD (if development or release requested, or if rust/python only flags used)
# ==============================================================================

# Show dry run indicator removed - [DRY RUN] prefix is sufficient

# Default to development mode if no build type specified but build flags are used
if [ -z "$BUILD_TYPE" ] && ([ "$BUILD_RUST" = true ] || [ "$BUILD_PYTHON" = true ]); then
    # If no build type specified and either Rust or Python is enabled, default to development mode
    BUILD_TYPE="development"
    dry_run_echo "   → No build type specified, defaulting to development mode"
fi

if [ -n "$BUILD_TYPE" ]; then

    # Ensure workspace is available for build
    if [ -z "$WORKSPACE_DIR" ]; then
        echo "❌ ERROR: No workspace found for build"
        exit 1
    fi

    # Check if we're running inside a Docker container
    if [ ! -e /.dockerenv ]; then
        echo "❌ ERROR: This script must be run inside a Docker container."
        exit 1
    fi

    dry_run_echo "Build: $BUILD_TYPE mode (Rust=$BUILD_RUST, Python=$BUILD_PYTHON)"

    # ==============================================================================
    # WORKSPACE SETUP
    # ==============================================================================
    # Prepare the workspace for building

    cmd cd $WORKSPACE_DIR

    if [ "$CARGO_CLEAN" = true ]; then
        cmd cargo clean
    fi

    # Fix file permissions due to the run.sh running as root, causing countless headaches
    USER_ID=$(stat -c "%u" .)
    GROUP_ID=$(stat -c "%g" .)
    cmd sudo chown -R $USER_ID:$GROUP_ID . || true

    # ==============================================================================
    # DEFAULT PYTHON CLEANUP
    # ==============================================================================
    # Always clean existing Python packages before any operations
    cleanup_python

    # ==============================================================================
    # COMPILATION CONFIGURATION
    # ==============================================================================
    # Configure Rust compiler settings for optimal build performance

    # Detect number of CPU cores for optimal build performance
    if command -v nproc >/dev/null 2>&1; then
        # Linux systems
        CPU_CORES=$(nproc)
    elif command -v sysctl >/dev/null 2>&1; then
        # macOS systems
        CPU_CORES=$(sysctl -n hw.ncpu)
    else
        # Fallback to a reasonable default
        CPU_CORES=8
    fi

    # Use detected cores for CARGO_BUILD_JOBS, but cap at 32 for memory considerations
    if [ "$CPU_CORES" -gt 32 ]; then
        CARGO_BUILD_JOBS=32
    else
        CARGO_BUILD_JOBS=$CPU_CORES
    fi

    dry_run_echo "Detected $CPU_CORES CPU cores, using CARGO_BUILD_JOBS=$CARGO_BUILD_JOBS"

    if [ "$BUILD_TYPE" = "development" ]; then
        cmd export CARGO_INCREMENTAL=1
    else
        cmd export CARGO_INCREMENTAL=0
    fi

    # ==============================================================================
    # RUST COMPILATION
    # ==============================================================================
    # Build the Rust components that will become the Python extension

    if [ "$BUILD_RUST" = true ]; then
        if [ "$BUILD_TYPE" = "development" ]; then
            cmd bash -c "CARGO_PROFILE_DEV_OPT_LEVEL=0 CARGO_BUILD_JOBS=$CARGO_BUILD_JOBS CARGO_PROFILE_DEV_CODEGEN_UNITS=256 cargo build --locked --features dynamo-llm/block-manager --workspace"
        else
            cmd cargo build --release --locked --features dynamo-llm/block-manager --workspace
        fi
    else
        dry_run_echo "Skipping Rust build (--py-only mode)"
    fi

    # ==============================================================================
    # PYTHON BUILD COMPONENTS
    # ==============================================================================
    # Build Python packages and wheels

    if [ "$BUILD_PYTHON" = true ]; then
        dry_run_echo "========== Building Python components... =========="

        # Clean up any existing wheel files
        if ls $WHEEL_OUTPUT_DIR/*.whl 1> /dev/null 2>&1; then
            cmd rm -f $WHEEL_OUTPUT_DIR/*.whl
            cmd rmdir $WHEEL_OUTPUT_DIR 2>/dev/null || true
        fi

        # Clean up old _core*.so files to ensure fresh build
        if ls $WORKSPACE_DIR/lib/bindings/python/src/dynamo/_core*.so 1> /dev/null 2>&1; then
            cmd rm -f $WORKSPACE_DIR/lib/bindings/python/src/dynamo/_core*.so
        fi

        # Install maturin if needed (skip in dry-run mode)
        if [ "$DRY_RUN" = false ] && ! uv pip show maturin >/dev/null 2>&1; then
            cmd uv pip install maturin[patchelf]
        fi

        dry_run_echo "Building ai-dynamo-runtime package..."

        if [ "$BUILD_TYPE" = "development" ]; then
            cmd bash -c "cd $WORKSPACE_DIR/lib/bindings/python && CARGO_PROFILE_DEV_OPT_LEVEL=0 CARGO_BUILD_JOBS=$CARGO_BUILD_JOBS CARGO_PROFILE_DEV_CODEGEN_UNITS=256 maturin develop --uv"

            dry_run_echo "Installing components package in editable mode..."
            cmd bash -c "cd $WORKSPACE_DIR/ && uv pip install -e ."
        else
            cmd bash -c "cd $WORKSPACE_DIR/lib/bindings/python && maturin build --release --out $WHEEL_OUTPUT_DIR"

            if [ "$DRY_RUN" = false ]; then
                if ls $WHEEL_OUTPUT_DIR/*.whl 1> /dev/null 2>&1; then
                    WHEEL_FILE=$(ls $WHEEL_OUTPUT_DIR/*.whl)
                    cmd uv pip install --upgrade --force-reinstall --no-deps $WHEEL_FILE
                else
                    echo "❌ ERROR: No wheel files found in $WHEEL_OUTPUT_DIR/"
                    exit 1
                fi
            else
                dry_run_echo "Check for wheel files and install ai-dynamo-runtime package"
            fi

            dry_run_echo "Building ai-dynamo package..."
            cmd pip wheel --no-deps --wheel-dir $WHEEL_OUTPUT_DIR .

            if [ "$DRY_RUN" = false ]; then
                if ls $WHEEL_OUTPUT_DIR/ai_dynamo-*.whl 1> /dev/null 2>&1; then
                    AI_DYNAMO_WHEEL=$(ls $WHEEL_OUTPUT_DIR/ai_dynamo-*.whl)
                    cmd uv pip install --upgrade --find-links $WHEEL_OUTPUT_DIR $AI_DYNAMO_WHEEL
                else
                    echo "❌ ERROR: No ai-dynamo wheel files found in $WHEEL_OUTPUT_DIR/"
                    exit 1
                fi
            else
                dry_run_echo "Check for ai-dynamo wheel files and install package"
            fi
        fi
    else
        dry_run_echo "Skipping Python build (--rust-only mode)"
    fi  # End of BUILD_PYTHON conditional block

    if [ "$BUILD_PYTHON" = true ]; then
        cmd python3 -c "import dynamo"
        if [ "$DRY_RUN" = false ]; then
            if python3 -c "import dynamo" 2>/dev/null; then
                dry_run_echo "Rudimentary Python import test PASSED"
            else
                echo "❌ Python import failed"
            fi
        else
            dry_run_echo "Rudimentary Python import test ran (dry-run mode)"
        fi
    fi

    dry_run_echo "✅ Build completed successfully!"

fi  # End of BUILD_TYPE block
