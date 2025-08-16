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
#   ./pybuild.sh --dev             # Development mode (fast build, slower runtime)
#   ./pybuild.sh --development     # Same as --dev (full flag name)
#   ./pybuild.sh --release         # Production wheels (slow build, fast runtime)
#   ./pybuild.sh --check           # Status check only (no build)
#   ./pybuild.sh --python-clean    # Remove all Python packages and exit
#   ./pybuild.sh --cargo-clean     # Remove Rust build artifacts and exit
#   ./pybuild.sh --dev --cargo-clean    # Clean then development build
#   ./pybuild.sh --dev --check     # Development build + status check
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
CHECK=false               # Flag for --check mode
DEBUG=false               # Flag for --debug mode
BUILD_RUST=true          # Flag for --rust-only/--rust (default: true)
BUILD_PYTHON=true        # Flag for --py-only/--py (default: true)

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
        --check)
            CHECK=true
            shift
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--development|--dev|--release] [--rust-only|--rust|--py-only|--py] [--cargo-clean] [--python-clean] [--check] [--debug]"
            echo ""
            echo "Build Type (mutually exclusive, --dev is default if none specified):"
            echo "  --development, --dev: Build ai-dynamo-runtime in development mode (editable install) [DEFAULT]"
            echo "  --release: Build both ai-dynamo-runtime and ai-dynamo wheels for production"
            echo ""
            echo "Build Components (mutually exclusive):"
            echo "  --rust-only, --rust: Build only Rust components (cargo build)"
            echo "  --py-only, --py: Build only Python components (wheel/maturin/uv)"
            echo "  Default: Build both Rust and Python components"
            echo ""
            echo "Other Options:"
            echo "  --cargo-clean: Clean Rust build artifacts (can be used standalone or with build)"
            echo "  --python-clean: Remove all dynamo packages (.pth, .whl, pip/uv) and exit"
            echo "  --check: Check status of dynamo packages (.pth, .whl, pip/uv) and exit"
            echo "  --debug: Show detailed .pth file information during development builds"
            echo ""
            echo "Standalone operations:"
            echo "  • --check: Only show package status"
            echo "  • --cargo-clean: Only clean Rust build artifacts"
            echo "  • --python-clean: Only remove Python packages"
            echo "Combined operations:"
            echo "  • $0 --check: Build in development mode then show status (--dev is default)"
            echo "  • $0 --release --cargo-clean: Clean then build optimized wheels"
            echo "  • $0 --rust-only: Build only Rust components in development mode"
            echo "  • $0 --py-only: Build only Python components in development mode"
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
    if [ "$PYTHON_CLEAN" = true ] || [ "$CHECK" = true ]; then
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
# 1. PYTHON CLEAN (if requested, do this first and exit)
# ==============================================================================

if [ "$PYTHON_CLEAN" = true ]; then
    echo ""
    echo "PYTHON CLEAN MODE: Removing all dynamo installations and artifacts"
    echo "======================================================================"

    if [ -n "$WORKSPACE_DIR" ]; then
        echo "Workspace: $WORKSPACE_DIR"
        # Show cargo target directory for wheels
        echo "Wheel directory: $CARGO_TARGET_DIR/wheels"
    else
        echo "No workspace found, system-wide cleanup only"
    fi

    # Perform comprehensive cleanup
    cleanup_all_dynamo_installations "cleanup mode"

    echo ""
    echo "CLEANUP COMPLETED SUCCESSFULLY!"
    echo "   • All dynamo packages uninstalled"
    echo "   • All .pth files removed"
    echo "   • All .whl files cleaned up"
    echo "   • All build artifacts removed"
    echo "   → System is now in a clean state for fresh installation"
    echo ""
    echo "Next steps:"
    echo "   • For development build: ./dynamo_build.sh --dev"
    echo "   • For production build: ./dynamo_build.sh --release"

    exit 0
fi

# ==============================================================================
# 2. CARGO CLEAN (if requested, do this at the start)
# ==============================================================================

if [ "$CARGO_CLEAN" = true ]; then
    echo ""
    echo "CARGO CLEAN: Removing Rust build artifacts..."
    echo "   Workspace: ${WORKSPACE_DIR:-'Not found'}"

    if [ -n "$WORKSPACE_DIR" ]; then
        cd "$WORKSPACE_DIR"
        cargo clean
        echo "   ✅ Rust build artifacts cleaned successfully"
    else
        echo "   ❌ No workspace found - cannot clean artifacts"
        # If only cargo clean was requested, exit with error
        if [ -z "$BUILD_TYPE" ] && [ "$CHECK" = false ]; then
            exit 1
        fi
    fi
fi

# ==============================================================================
# FUNCTION DEFINITIONS
# ==============================================================================
# Define all functions used by this script

get_dynamo_version() {
    # Extract version from Cargo.toml workspace.package section
    if [ -f "$WORKSPACE_DIR/Cargo.toml" ]; then
        awk '/^\[workspace\.package\]/{f=1; next} /^\[/{f=0} f && /^version = /{print $3}' "$WORKSPACE_DIR/Cargo.toml" | tr -d '"'
    else
        echo "unknown"
    fi
}

cleanup_all_dynamo_installations() {
    local context="${1:-general}"

    # Force uninstall with both pip and uv to handle any inconsistencies
    if pip show ai-dynamo-runtime >/dev/null 2>&1; then
        echo "$ pip uninstall ai-dynamo-runtime"
        pip uninstall ai-dynamo-runtime --yes >/dev/null 2>&1 || true
    fi
    if pip show ai-dynamo >/dev/null 2>&1; then
        echo "$ pip uninstall ai-dynamo"
        pip uninstall ai-dynamo --yes >/dev/null 2>&1 || true
    fi

    if uv pip show ai-dynamo-runtime >/dev/null 2>&1; then
        echo "$ uv pip uninstall ai-dynamo-runtime"
        uv pip uninstall ai-dynamo-runtime --yes >/dev/null 2>&1 || true
    fi
    if uv pip show ai-dynamo >/dev/null 2>&1; then
        echo "$ uv pip uninstall ai-dynamo"
        uv pip uninstall ai-dynamo --yes >/dev/null 2>&1 || true
    fi

    # Clean up any remaining artifacts in site-packages
    local python_site_packages=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "")
    if [ -n "$python_site_packages" ]; then
        # Remove ALL dynamo-related .dist-info directories
        for dist_info in "$python_site_packages"/ai_dynamo*-*.dist-info; do
            if [ -d "$dist_info" ]; then
                rm -rf "$dist_info"
            fi
        done

        # Remove ALL .pth files
        for pth_file in "$python_site_packages"/*dynamo*.pth "$python_site_packages"/_*dynamo*.pth; do
            if [ -f "$pth_file" ]; then
                rm -f "$pth_file"
            fi
        done

        # Remove editables finder
        if [ -f "$python_site_packages/__editables_finder__.py" ]; then
            rm -f "$python_site_packages/__editables_finder__.py"
        fi
    fi

    # Clean up wheel files if in workspace
    if [ -n "$WHEEL_OUTPUT_DIR" ] && [ -d "$WHEEL_OUTPUT_DIR" ]; then
        for wheel_file in "$WHEEL_OUTPUT_DIR"/*.whl; do
            if [ -f "$wheel_file" ]; then
                rm -f "$wheel_file"
            fi
        done
        # Remove wheel directory if empty
        if [ -d "$WHEEL_OUTPUT_DIR" ] && [ -z "$(ls -A "$WHEEL_OUTPUT_DIR")" ]; then
            rmdir "$WHEEL_OUTPUT_DIR"
        fi
    fi
}

get_package_source() {
    local package_name="$1"
    local python_site_packages=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "")

    if [ -z "$python_site_packages" ]; then
        echo "unknown"
        return
    fi

    # Convert package name to distribution name (ai-dynamo-runtime -> ai_dynamo_runtime)
    local dist_name=$(echo "$package_name" | sed 's/-/_/g')

    # Look for .dist-info directory
    local dist_info_dir=""
    for dist_info in "$python_site_packages"/${dist_name}-*.dist-info; do
        if [ -d "$dist_info" ]; then
            dist_info_dir="$dist_info"
            break
        fi
    done

    if [ -n "$dist_info_dir" ]; then
        # Check for direct_url.json which indicates editable install
        if [ -f "$dist_info_dir/direct_url.json" ]; then
            local editable=$(grep -o '"editable"[[:space:]]*:[[:space:]]*true' "$dist_info_dir/direct_url.json" 2>/dev/null || echo "")
            if [ -n "$editable" ]; then
                echo ".pth (editable)"
                return
            fi
        fi

        # Check INSTALLER file to see if it was installed via wheel
        if [ -f "$dist_info_dir/INSTALLER" ]; then
            local installer=$(cat "$dist_info_dir/INSTALLER" 2>/dev/null)
            if [ "$installer" = "pip" ] || [ "$installer" = "uv" ]; then
                # If no direct_url.json with editable=true, likely from wheel
                echo ".whl (wheel)"
                return
            fi
        fi
    fi

    # Check for .pth files as fallback
    for pth_file in "$python_site_packages"/*${package_name}*.pth "$python_site_packages"/_*${package_name}*.pth; do
        if [ -f "$pth_file" ]; then
            echo ".pth (editable)"
            return
        fi
    done

    # Default assumption
    echo ".whl (wheel)"
}

get_package_files() {
    local package_name="$1"
    local python_site_packages="$2"
    local files_found=false

    if [ -z "$python_site_packages" ]; then
        return 1
    fi

    # Convert package name to distribution name (ai-dynamo-runtime -> ai_dynamo_runtime)
    local dist_name=$(echo "$package_name" | sed 's/-/_/g')

        # Check .dist-info directories
    for dist_info in "$python_site_packages"/${dist_name}-*.dist-info; do
            if [ -d "$dist_info" ]; then
            echo "      $dist_info"
            files_found=true
            fi
        done

        # Check .pth files
    for pth_file in "$python_site_packages"/*${package_name}*.pth "$python_site_packages"/_*${package_name}*.pth; do
            if [ -f "$pth_file" ]; then
            local pth_date=$(TZ=America/Los_Angeles date -d "@$(stat -c '%Y' "$pth_file" 2>/dev/null)" +'%Y-%m-%d %H:%M:%S %Z' 2>/dev/null)
            echo "      $pth_file (modified: $pth_date)"
            files_found=true
            fi
        done

    # Check for specific .pth patterns
    for pth_file in "$python_site_packages"/${dist_name}.pth; do
        if [ -f "$pth_file" ]; then
            local pth_date=$(TZ=America/Los_Angeles date -d "@$(stat -c '%Y' "$pth_file" 2>/dev/null)" +'%Y-%m-%d %H:%M:%S %Z' 2>/dev/null)
            echo "      $pth_file (modified: $pth_date)"
            files_found=true
        fi
    done

    if [ "$files_found" = false ]; then
        return 1
    fi
    return 0
}

check_package_status() {
    echo "Package Status:"

    # Show Cargo target directory first
    echo "Cargo target directory: ${CARGO_TARGET_DIR:-'unknown'}"

    # Show last cargo build time if available (stored in a temp file during builds)
    if [ -f "/tmp/dynamo_cargo_build_time" ]; then
        local last_build_time=$(cat /tmp/dynamo_cargo_build_time 2>/dev/null || echo "unknown")
        echo "Last cargo build time: ${last_build_time}s"
    fi

    # Get the common site-packages path
    local python_site_packages=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "")
    if [ -n "$python_site_packages" ]; then
        echo "Site-packages: $python_site_packages"
    fi

    # Check package installations via uv pip (which sees the same packages as pip)
    echo "uv pip:"
    if uv pip show ai-dynamo-runtime >/dev/null 2>&1; then
        local runtime_version=$(uv pip show ai-dynamo-runtime 2>/dev/null | grep "^Version:" | cut -d' ' -f2)
        local runtime_source=$(get_package_source "ai-dynamo-runtime")

        # Get creation date from .dist-info directory
        local dist_name="ai_dynamo_runtime"
        local install_date="unknown"
        for dist_info in "$python_site_packages"/${dist_name}-*.dist-info; do
            if [ -d "$dist_info" ]; then
                install_date=$(TZ=America/Los_Angeles date -d "@$(stat -c '%Y' "$dist_info" 2>/dev/null)" +'%Y-%m-%d %H:%M:%S %Z' 2>/dev/null)
                break
            fi
        done

        echo "   ✅ ai-dynamo-runtime ($runtime_version) - $runtime_source (created: $install_date)"
        get_package_files "ai-dynamo-runtime" "$python_site_packages"
    else
        echo "   ❌ ai-dynamo-runtime"
    fi
    if uv pip show ai-dynamo >/dev/null 2>&1; then
        local dynamo_version=$(uv pip show ai-dynamo 2>/dev/null | grep "^Version:" | cut -d' ' -f2)
        local dynamo_source=$(get_package_source "ai-dynamo")

        # Get creation date from .dist-info directory
        local dist_name="ai_dynamo"
        local install_date="unknown"
        for dist_info in "$python_site_packages"/${dist_name}-*.dist-info; do
            if [ -d "$dist_info" ]; then
                install_date=$(TZ=America/Los_Angeles date -d "@$(stat -c '%Y' "$dist_info" 2>/dev/null)" +'%Y-%m-%d %H:%M:%S %Z' 2>/dev/null)
                break
            fi
        done

        echo "   ✅ ai-dynamo         ($dynamo_version) - $dynamo_source (created: $install_date)"
        get_package_files "ai-dynamo" "$python_site_packages"
    else
        echo "   ❌ ai-dynamo"
    fi

    # Check wheel files
    echo "Wheels:"
    if [ -n "$WHEEL_OUTPUT_DIR" ] && [ -d "$WHEEL_OUTPUT_DIR" ]; then
        local wheel_count=0
        for wheel_file in "$WHEEL_OUTPUT_DIR"/*.whl; do
            if [ -f "$wheel_file" ]; then
                local wheel_size=$(ls -lh "$wheel_file" | awk '{print $5}')
                local wheel_date=$(TZ=America/Los_Angeles date -d "@$(stat -c '%Y' "$wheel_file" 2>/dev/null)" +'%Y-%m-%d %H:%M:%S %Z' 2>/dev/null)
                echo "   $wheel_file ($wheel_size, created: $wheel_date)"
                wheel_count=$((wheel_count + 1))
            fi
        done
        if [ $wheel_count -eq 0 ]; then
            echo "   ❌ No wheels found in $WHEEL_OUTPUT_DIR"
        fi
    else
        echo "   ❌ No wheel directory ($WHEEL_OUTPUT_DIR)"
    fi

    # Discover source paths for PYTHONPATH recommendations
    local source_paths=()
    local site_package_components=()

    # Find all available source components for PYTHONPATH recommendations
    if [ -n "$WORKSPACE_DIR" ]; then
        # Find direct components (frontend, planner, etc.)
        if [ -d "$WORKSPACE_DIR/components" ]; then
            for comp_dir in "$WORKSPACE_DIR/components"/*; do
                if [ -d "$comp_dir/src" ]; then
                    local comp_name=$(basename "$comp_dir")
                    if [ -f "$comp_dir/src/dynamo/$comp_name/__init__.py" ]; then
                        source_paths+=("$comp_dir/src")
                        site_package_components+=("dynamo.$comp_name")
                    fi
                fi
            done
        fi

        # Find backend components (vllm, sglang, etc.)
        if [ -d "$WORKSPACE_DIR/components/backends" ]; then
            for backend_dir in "$WORKSPACE_DIR/components/backends"/*; do
                if [ -d "$backend_dir/src" ]; then
                    local backend_name=$(basename "$backend_dir")
                    if [ -f "$backend_dir/src/dynamo/$backend_name/__init__.py" ]; then
                        source_paths+=("$backend_dir/src")
                        site_package_components+=("dynamo.$backend_name")
                    fi
                fi
            done
        fi
    fi

    # Show PYTHONPATH and source recommendations compactly
    # PYTHONPATH may not be set; check if it is non-empty before using
    if [ -n "${PYTHONPATH:-}" ]; then
        local path_count=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v '^$' | wc -l)
        echo "Current env: PYTHONPATH ($path_count paths) + site-packages"
    else
        echo "Current env: site-packages only"
    fi

    # Generate dynamic PYTHONPATH recommendation based on found components
    if [ ${#source_paths[@]} -gt 0 ]; then
        #echo -n "Source available: "
        #printf '%s ' "${site_package_components[@]}"
        #echo ""

        # Build PYTHONPATH string
        local pythonpath_recommendation=""
        for path in "${source_paths[@]}"; do
            if [ -z "$pythonpath_recommendation" ]; then
                pythonpath_recommendation="$path"
            else
                pythonpath_recommendation="$pythonpath_recommendation:$path"
            fi
        done

        # Replace /home/ubuntu with $HOME for portability
        # local portable_pythonpath=$(echo "$pythonpath_recommendation" | sed "s|/home/ubuntu|\$HOME|g")
        # echo "Source components available for PYTHONPATH:"
        # echo "   export PYTHONPATH=\"$portable_pythonpath:\$PYTHONPATH\""
    fi

    return 0
}

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

# Default to development mode if no build type specified but build flags are used
if [ -z "$BUILD_TYPE" ] && ([ "$BUILD_RUST" = true ] || [ "$BUILD_PYTHON" = true ]); then
    # If no build type specified and either Rust or Python is enabled, default to development
    BUILD_TYPE="development"
    echo "   → No build type specified, defaulting to development mode"
fi

if [ -n "$BUILD_TYPE" ]; then

    # Ensure workspace is available for build
    if [ -z "$WORKSPACE_DIR" ]; then
        echo "❌ ERROR: No workspace found for build"
        exit 1
    fi

    # Development builds automatically use incremental compilation for faster rebuilds
    if [ "$BUILD_TYPE" = "development" ]; then
        INCREMENTAL=true
    else
        INCREMENTAL=false
    fi

    # Check if we're running inside a Docker container
    if [ ! -e /.dockerenv ]; then
        echo "❌ ERROR: This script must be run inside a Docker container."
        exit 1
    fi

    echo "Build: $BUILD_TYPE mode (Rust=$BUILD_RUST, Python=$BUILD_PYTHON)"

    # ==============================================================================
    # WORKSPACE SETUP
    # ==============================================================================
    # Prepare the workspace for building

    echo "$ cd $WORKSPACE_DIR"
    cd $WORKSPACE_DIR

    if [ "$CARGO_CLEAN" = true ]; then
        echo "$ cargo clean"
        cargo clean
    fi

    # Fix file permissions
    USER_ID=$(stat -c "%u" .)
    GROUP_ID=$(stat -c "%g" .)
    echo "$ chown -R $USER_ID:$GROUP_ID ."
    chown -R $USER_ID:$GROUP_ID .

    # ==============================================================================
    # COMPILATION CONFIGURATION
    # ==============================================================================
    # Configure Rust compiler settings for optimal build performance

    if [ "$INCREMENTAL" = true ]; then
        echo "$ export CARGO_INCREMENTAL=1"
        export CARGO_INCREMENTAL=1
    else
        echo "$ export CARGO_INCREMENTAL=0"
        export CARGO_INCREMENTAL=0
    fi

    # ==============================================================================
    # RUST COMPILATION
    # ==============================================================================
    # Build the Rust components that will become the Python extension

    if [ "$BUILD_RUST" = true ]; then
        if [ "$BUILD_TYPE" = "development" ]; then
            start_time=$(date +%s.%N)
            echo "$ CARGO_INCREMENTAL=1 RUSTFLAGS=\"-C opt-level=0 -C codegen-units=256\" cargo build --locked --features dynamo-llm/block-manager --workspace"
            CARGO_INCREMENTAL=1 RUSTFLAGS="-C opt-level=0 -C codegen-units=256" cargo build --locked --features dynamo-llm/block-manager --workspace
            end_time=$(date +%s.%N)
            build_duration=$(echo "$end_time - $start_time" | bc 2>/dev/null || echo "unknown")
            if [ "$build_duration" != "unknown" ]; then
                echo "$build_duration" > /tmp/dynamo_cargo_build_time
            fi
        else
            start_time=$(date +%s.%N)
            echo "$ cargo build --release --locked --features dynamo-llm/block-manager --workspace"
            cargo build --release --locked --features dynamo-llm/block-manager --workspace
            end_time=$(date +%s.%N)
            build_duration=$(echo "$end_time - $start_time" | bc 2>/dev/null || echo "unknown")
            if [ "$build_duration" != "unknown" ]; then
                echo "$build_duration" > /tmp/dynamo_cargo_build_time
            fi
        fi
    else
        echo ""
        echo "Skipping Rust build (--py-only mode)"
    fi

    # ==============================================================================
    # PYTHON BUILD COMPONENTS
    # ==============================================================================
    # Build Python packages and wheels

    if [ "$BUILD_PYTHON" = true ]; then
        echo ""
        echo "========== Building Python components... =========="

        # Clean up any existing wheel files
        if ls $WHEEL_OUTPUT_DIR/*.whl 1> /dev/null 2>&1; then
            rm -f $WHEEL_OUTPUT_DIR/*.whl
            rmdir $WHEEL_OUTPUT_DIR 2>/dev/null || true
        fi

        # Install maturin if needed
        if ! uv pip show maturin >/dev/null 2>&1; then
            echo "$ uv pip install maturin[patchelf]"
            uv pip install maturin[patchelf] 2>&1 | sed 's/^/[uv] /'
        fi

        echo "Building ai-dynamo-runtime package..."

        if [ "$BUILD_TYPE" = "development" ]; then
        echo "$ cd $WORKSPACE_DIR/lib/bindings/python && CARGO_INCREMENTAL=1 RUSTFLAGS=\"-C opt-level=0 -C codegen-units=256\" maturin develop --uv --features block-manager"

        (cd $WORKSPACE_DIR/lib/bindings/python && CARGO_INCREMENTAL=1 RUSTFLAGS="-C opt-level=0 -C codegen-units=256" maturin develop --uv --features block-manager 2>&1 | sed 's/^/[maturin] /')
        echo "Successfully installed ai-dynamo-runtime package (editable mode)"
    else
        echo "$ cd $WORKSPACE_DIR/lib/bindings/python && maturin build --release --features block-manager --out $WHEEL_OUTPUT_DIR"

        (cd $WORKSPACE_DIR/lib/bindings/python && maturin build --release --features block-manager --out $WHEEL_OUTPUT_DIR 2>&1 | sed 's/^/[maturin] /')

        if ls $WHEEL_OUTPUT_DIR/*.whl 1> /dev/null 2>&1; then
            WHEEL_FILE=$(ls $WHEEL_OUTPUT_DIR/*.whl)
            echo "$ uv pip install --upgrade --force-reinstall --no-deps"
            uv pip install --upgrade --force-reinstall --no-deps $WHEEL_FILE 2>&1 | sed 's/^/[uv] /'
            echo "Sucessfully installed ai-dynamo-runtime package"
        else
            echo "❌ ERROR: No wheel files found in $WHEEL_OUTPUT_DIR/"
            exit 1
        fi

        echo "Building ai-dynamo package..."
        echo "$ pip wheel --no-deps --wheel-dir $WHEEL_OUTPUT_DIR ."

        pip wheel --no-deps --wheel-dir $WHEEL_OUTPUT_DIR . 2>&1 | sed 's/^/[pip wheel] /'

        if ls $WHEEL_OUTPUT_DIR/ai_dynamo-*.whl 1> /dev/null 2>&1; then
            AI_DYNAMO_WHEEL=$(ls $WHEEL_OUTPUT_DIR/ai_dynamo-*.whl)
            echo "$ uv pip install --upgrade --find-links $WHEEL_OUTPUT_DIR"
            uv pip install --upgrade --find-links $WHEEL_OUTPUT_DIR $AI_DYNAMO_WHEEL 2>&1 | sed 's/^/[uv] /'
            echo "Sucessfully installed ai-dynamo package"
        else
            echo "❌ ERROR: No ai-dynamo wheel files found in $WHEEL_OUTPUT_DIR/"
            exit 1
        fi
    fi
    else
        echo ""
        echo "Skipping Python build (--rust-only mode)"
    fi  # End of BUILD_PYTHON conditional block

    if [ "$BUILD_PYTHON" = true ]; then
        echo "$ python3 -c \"import dynamo\""
        if python3 -c "import dynamo" 2>/dev/null; then
            echo "Python import test PASSED"
        else
            echo "❌ Python import failed"
        fi
    fi

    echo ""
    echo "✅ Build completed successfully!"

fi  # End of BUILD_TYPE block

# ==============================================================================
# 4. CHECK (if requested, do this at the end)
# ==============================================================================

if [ "$CHECK" = true ]; then
    echo ""
    if [ -n "$BUILD_TYPE" ]; then
        echo "Post-build status check:"
    else
        echo "Package Status Check:"
    fi
    echo "=============================================="
    check_package_status

    # Run Python import check
    echo ""

    echo "$ $SCRIPT_DIR/python_import_check.py --imports"
    python3 "$SCRIPT_DIR/python_import_check.py" --imports
fi
