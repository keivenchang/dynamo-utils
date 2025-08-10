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
        --cargo-clean)
            CARGO_CLEAN=true
            shift
            ;;
        --python-clean)
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
            echo "Usage: $0 [--development|--dev|--release] [--cargo-clean] [--python-clean] [--check] [--debug]"
            echo "  --development, --dev: Build ai-dynamo-runtime in development mode (editable install)"
            echo "  --release: Build both ai-dynamo-runtime and ai-dynamo wheels for production"
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
            echo "  • $0 --dev --check: Build in development mode then show status"
            echo "  • $0 --release --cargo-clean: Clean then build optimized wheels"
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
# 1. PYTHON CLEAN (if requested, do this first and exit)
# ==============================================================================

if [ "$PYTHON_CLEAN" = true ]; then
    echo ""
    echo "PYTHON CLEAN MODE: Removing all dynamo installations and artifacts"
    echo "======================================================================"

    if [ -n "$WORKSPACE_DIR" ]; then
        echo "Workspace: $WORKSPACE_DIR"
        # Show cargo target directory for wheels
        local target_dir=$(cargo metadata --format-version=1 --no-deps 2>/dev/null | jq -r '.target_directory' 2>/dev/null || echo "$WORKSPACE_DIR/target")
        echo "Wheel directory: $target_dir/wheels"
    else
        echo "No workspace found, system-wide cleanup only"
    fi

    # Perform comprehensive cleanup
    cleanup_all_dynamo_installations "cleanup mode"

    echo ""
    echo "CLEANUP COMPLETED!"
    echo "   All dynamo packages, .pth files, .whl files, and artifacts removed"
    echo "   System is now in a clean state for fresh installation"
    echo ""
    echo "Next steps:"
    echo "   • For development: ./bin/pybuild.sh --dev"
    echo "   • For release: ./bin/pybuild.sh --release"

    exit 0
fi

# ==============================================================================
# 2. CARGO CLEAN (if requested, do this at the start)
# ==============================================================================

if [ "$CARGO_CLEAN" = true ]; then
    echo ""
    echo "CARGO CLEAN: Removing Rust build artifacts..."
    echo "Workspace: ${WORKSPACE_DIR:-'Not found'}"

    if [ -n "$WORKSPACE_DIR" ]; then
        cd "$WORKSPACE_DIR"
        cargo clean
        echo "✅ Rust build artifacts cleaned"
    else
        echo "❌ No workspace found - cannot clean artifacts"
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
    echo "   Ensuring clean state: removing ALL existing dynamo installations..."
    local cleanup_performed=false

    # Force uninstall with both pip and uv to handle any inconsistencies
    echo "      • Checking pip-managed installations..."
    if pip show ai-dynamo-runtime >/dev/null 2>&1; then
        echo "        $ pip uninstall ai-dynamo-runtime..."
        pip uninstall ai-dynamo-runtime --yes >/dev/null 2>&1 || true
        cleanup_performed=true
    fi
    if pip show ai-dynamo >/dev/null 2>&1; then
        echo "        $ pip uninstall ai-dynamo..."
        pip uninstall ai-dynamo --yes >/dev/null 2>&1 || true
        cleanup_performed=true
    fi

    echo "      • Checking uv-managed installations..."
    if uv pip show ai-dynamo-runtime >/dev/null 2>&1; then
        echo "        - uv pip uninstall ai-dynamo-runtime..."
        uv pip uninstall ai-dynamo-runtime --yes >/dev/null 2>&1 || true
        cleanup_performed=true
    fi
    if uv pip show ai-dynamo >/dev/null 2>&1; then
        echo "        - uv pip uninstall ai-dynamo..."
        uv pip uninstall ai-dynamo --yes >/dev/null 2>&1 || true
        cleanup_performed=true
    fi

    # Clean up any remaining artifacts in site-packages
    local python_site_packages=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "")
    if [ -n "$python_site_packages" ]; then
        echo "      • Cleaning remaining artifacts in site-packages..."

        # Remove ALL dynamo-related .dist-info directories
        for dist_info in "$python_site_packages"/ai_dynamo*-*.dist-info; do
            if [ -d "$dist_info" ]; then
                echo "        - Removing .dist-info: $(basename $dist_info)"
                rm -rf "$dist_info"
                cleanup_performed=true
            fi
        done

        # Remove ALL .pth files
        for pth_file in "$python_site_packages"/*dynamo*.pth "$python_site_packages"/_*dynamo*.pth; do
            if [ -f "$pth_file" ]; then
                echo "        - Removing .pth file: $(basename $pth_file)"
                rm -f "$pth_file"
                cleanup_performed=true
            fi
        done

        # Remove editables finder
        if [ -f "$python_site_packages/__editables_finder__.py" ]; then
            echo "        - Removing: __editables_finder__.py"
            rm -f "$python_site_packages/__editables_finder__.py"
            cleanup_performed=true
        fi
    fi

    # Clean up wheel files if in workspace
    if [ -n "$WHEEL_OUTPUT_DIR" ] && [ -d "$WHEEL_OUTPUT_DIR" ]; then
        echo "      • Cleaning wheel files..."
        for wheel_file in "$WHEEL_OUTPUT_DIR"/*.whl; do
            if [ -f "$wheel_file" ]; then
                echo "        - Removing wheel: $(basename $wheel_file)"
                rm -f "$wheel_file"
                cleanup_performed=true
            fi
        done
        # Remove wheel directory if empty
        if [ -d "$WHEEL_OUTPUT_DIR" ] && [ -z "$(ls -A "$WHEEL_OUTPUT_DIR")" ]; then
            rmdir "$WHEEL_OUTPUT_DIR"
        fi
    fi

    if [ "$cleanup_performed" = true ]; then
        echo "      ✅ Complete cleanup performed - no ambiguous installations remain"
    else
        echo "      • No existing installations found to remove"
    fi

    # Verify clean state
    echo "      Verifying clean state..."
    local remaining_issues=false
    if pip show ai-dynamo-runtime >/dev/null 2>&1 || uv pip show ai-dynamo-runtime >/dev/null 2>&1; then
        echo "        WARNING: ai-dynamo-runtime still detected"
        remaining_issues=true
    fi
    if pip show ai-dynamo >/dev/null 2>&1 || uv pip show ai-dynamo >/dev/null 2>&1; then
        echo "        WARNING: ai-dynamo still detected"
        remaining_issues=true
    fi

    if [ "$remaining_issues" = false ]; then
        echo "        ✅ State verified clean - ready for $context"
    else
        echo "        ❌ WARNING: Some packages may still be installed"
        return 1
    fi

    return 0
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
            echo "      $pth_file"
            files_found=true
            fi
        done

    # Check for specific .pth patterns
    for pth_file in "$python_site_packages"/${dist_name}.pth; do
        if [ -f "$pth_file" ]; then
            echo "      $pth_file"
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
    if command -v jq >/dev/null 2>&1; then
        local target_dir=$(cargo metadata --format-version=1 --no-deps 2>/dev/null | jq -r '.target_directory' 2>/dev/null || echo "unknown")
        echo "Cargo target directory: $target_dir"
    else
        echo "Cargo target directory: jq not available, cannot retrieve"
    fi

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
        echo "   ✅ ai-dynamo-runtime ($runtime_version) - $runtime_source"
        get_package_files "ai-dynamo-runtime" "$python_site_packages"
    else
        echo "   ❌ ai-dynamo-runtime"
    fi
    if uv pip show ai-dynamo >/dev/null 2>&1; then
        local dynamo_version=$(uv pip show ai-dynamo 2>/dev/null | grep "^Version:" | cut -d' ' -f2)
        local dynamo_source=$(get_package_source "ai-dynamo")
        echo "   ✅ ai-dynamo ($dynamo_version) - $dynamo_source"
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
                echo "   $wheel_file ($wheel_size)"
                wheel_count=$((wheel_count + 1))
            fi
        done
        if [ $wheel_count -eq 0 ]; then
            echo "   ❌ No wheels found in $WHEEL_OUTPUT_DIR"
        fi
    else
        echo "   ❌ No wheel directory ($WHEEL_OUTPUT_DIR)"
    fi

    # Test import capability and show component sources
    echo "Components:"

        # Dynamically discover ai-dynamo-runtime components with paths
    echo "   From ai-dynamo-runtime (e.g. pyo3/Rust bindings):"
    local runtime_components=()

    # Always include _core (compiled Rust module)
    runtime_components+=("dynamo._core")

    # Discover other runtime components from filesystem structure
    if [ -n "$WORKSPACE_DIR" ] && [ -d "$WORKSPACE_DIR/lib/bindings/python/src/dynamo" ]; then
        for runtime_dir in "$WORKSPACE_DIR/lib/bindings/python/src/dynamo"/*; do
            if [ -d "$runtime_dir" ]; then
                local runtime_name=$(basename "$runtime_dir")
                # Check that the component has proper Python module structure
                if [ -f "$runtime_dir/__init__.py" ]; then
                    runtime_components+=("dynamo.$runtime_name")
                fi
            fi
        done
    fi

    # Test discovered runtime components
    for component in "${runtime_components[@]}"; do
        local path_info=$(python3 -c "
try:
    import $component
    import os
    if hasattr($component, '__file__') and $component.__file__:
        print(os.path.dirname($component.__file__))
    else:
        print('built-in/compiled')
except Exception:
    print('failed')
" 2>/dev/null)

        if [ "$path_info" != "failed" ]; then
            echo "      ✅ $component ($path_info)"
        else
            echo "      ❌ $component"
        fi
    done

    # Dynamically discover ai-dynamo components from filesystem
    echo "   From ai-dynamo:"
    local source_paths=()
    local site_package_components=()
    local discovered_components=()

    # Discover components from filesystem structure
if [ -n "$WORKSPACE_DIR" ]; then
                # Find direct components (frontend, planner, etc.)
        if [ -d "$WORKSPACE_DIR/components" ]; then
            for comp_dir in "$WORKSPACE_DIR/components"/*; do
                if [ -d "$comp_dir/src" ]; then
                    local comp_name=$(basename "$comp_dir")
                    # Check that the component has proper Python module structure
                    if [ -f "$comp_dir/src/dynamo/$comp_name/__init__.py" ]; then
                        discovered_components+=("dynamo.$comp_name")
                    fi
                fi
            done
        fi

        # Find backend components (vllm, sglang, etc.)
        if [ -d "$WORKSPACE_DIR/components/backends" ]; then
            for backend_dir in "$WORKSPACE_DIR/components/backends"/*; do
                if [ -d "$backend_dir/src" ]; then
                    local backend_name=$(basename "$backend_dir")
                    # Check that the backend has proper Python module structure
                    if [ -f "$backend_dir/src/dynamo/$backend_name/__init__.py" ]; then
                        discovered_components+=("dynamo.$backend_name")
                    fi
                fi
            done
        fi
    fi

    # Test discovered components
    for component in "${discovered_components[@]}"; do
        local path_info=$(python3 -c "
try:
    import $component
    import os
    if hasattr($component, '__file__') and $component.__file__:
        print(os.path.dirname($component.__file__))
    else:
        print('built-in/compiled')
except Exception:
    print('failed')
" 2>/dev/null)

        if [ "$path_info" != "failed" ]; then
            echo "      ✅ $component ($path_info)"

            # Check if component is loading from site-packages vs source
            if [[ "$path_info" == *"site-packages"* ]]; then
                # Component is from installed package - find corresponding source
                local component_name=$(echo "$component" | cut -d'.' -f2)
                local potential_source=""

                # Look for source in direct components first
                if [ -d "$WORKSPACE_DIR/components/$component_name/src" ]; then
                    potential_source="$WORKSPACE_DIR/components/$component_name/src"
                # Then look in backends
                elif [ -d "$WORKSPACE_DIR/components/backends/$component_name/src" ]; then
                    potential_source="$WORKSPACE_DIR/components/backends/$component_name/src"
                fi

                # Add to recommendations if source directory exists
                if [ -n "$potential_source" ]; then
                    source_paths+=("$potential_source")
                    site_package_components+=("$component")
                fi
            fi
        else
            echo "      ❌ $component"

            # Component failed to import - check if source exists anyway
            local component_name=$(echo "$component" | cut -d'.' -f2)
            local potential_source=""

            # Look for source in direct components first
            if [ -d "$WORKSPACE_DIR/components/$component_name/src" ]; then
                potential_source="$WORKSPACE_DIR/components/$component_name/src"
            # Then look in backends
            elif [ -d "$WORKSPACE_DIR/components/backends/$component_name/src" ]; then
                potential_source="$WORKSPACE_DIR/components/backends/$component_name/src"
            fi

            # Add to recommendations if source directory exists
            if [ -n "$potential_source" ]; then
                source_paths+=("$potential_source")
                site_package_components+=("$component")
            fi
        fi
    done

    # Show PYTHONPATH and source recommendations compactly
    if [ -n "$PYTHONPATH" ]; then
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
        local portable_pythonpath=$(echo "$pythonpath_recommendation" | sed "s|/home/ubuntu|\$HOME|g")
        echo "For hot-reload of Python components, export your PYTHONPATH:"
        echo "   export PYTHONPATH=\"$portable_pythonpath:\$PYTHONPATH\""
    fi

    return 0
}

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================
# Set up paths (needed for both build and check operations)

if [ -n "$WORKSPACE_DIR" ]; then
    # Use cargo target directory for wheels (follows Rust conventions)
    CARGO_TARGET_DIR=$(cargo metadata --format-version=1 --no-deps 2>/dev/null | jq -r '.target_directory' 2>/dev/null || echo "$WORKSPACE_DIR/target")
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
# 3. BUILD (if development or release requested)
# ==============================================================================

if [ -n "$BUILD_TYPE" ]; then

    # Ensure workspace is available for build
    if [ -z "$WORKSPACE_DIR" ]; then
        echo "❌ ERROR: No workspace found for build"
        exit 1
    fi

echo ""
    echo "Setting up build configuration..."

# Development builds automatically use incremental compilation for faster rebuilds
if [ "$BUILD_TYPE" = "development" ]; then
    INCREMENTAL=true
    echo "   → Development mode: Enabling incremental compilation for faster rebuilds"
else
    INCREMENTAL=false
    echo "   → Release mode: Using full compilation for maximum optimization"
fi

# ==============================================================================
# ENVIRONMENT VALIDATION
# ==============================================================================
# Ensure we're running in the correct environment (Docker container)

echo ""
    echo "Validating execution environment..."

# Check if we're running inside a Docker container by looking for .dockerenv file
if [ ! -e /.dockerenv ]; then
    echo "❌ ERROR: This script must be run inside a Docker container."
    echo "   The build process requires specific dependencies and environment setup"
    echo "   that are provided by the Dynamo Docker container."
    echo ""
    echo "   Please run this script from within the Dynamo container:"
        echo "   docker run -it --rm -v \$(pwd):/workspace dynamo-container ./bin/pybuild.sh"
    exit 1
fi

echo "✅ Running inside Docker container - environment validated"

echo ""
    echo "Build configuration summary:"
echo "   Build type: $BUILD_TYPE"
echo "   Incremental compilation: $INCREMENTAL"
    echo "   Cargo clean: $CARGO_CLEAN"
echo "   Workspace directory: ${WORKSPACE_DIR:-'Not found'}"
echo "   Python bindings source: ${PYTHON_BINDINGS_PATH:-'Not available'}"
echo "   Wheel output directory: ${WHEEL_OUTPUT_DIR:-'Not available'}"
echo ""

# ==============================================================================
# WORKSPACE SETUP
# ==============================================================================
# Prepare the workspace for building

echo "Setting up workspace..."

# Change to workspace directory where the source code is mounted
cd $WORKSPACE_DIR
echo "   → Changed to workspace directory: $WORKSPACE_DIR"

# Clean build artifacts if requested (before setting incremental config)
# This ensures a completely fresh build but takes longer
if [ "$CARGO_CLEAN" = true ]; then
    echo "   → Cleaning all build artifacts (this may take a moment)..."
    cargo clean
    echo "   ✅ Build artifacts cleaned"
else
    echo "   → Skipping clean (using incremental build for speed)"
fi

# Create target directory and fix file permissions
# Docker can sometimes create files with wrong ownership
USER_ID=$(stat -c "%u" .)   # Get the owner of the current directory
GROUP_ID=$(stat -c "%g" .)  # Get the group of the current directory
chown -R $USER_ID:$GROUP_ID .
echo "   → Target directory created and permissions fixed (UID: $USER_ID, GID: $GROUP_ID)"

# ==============================================================================
# COMPILATION CONFIGURATION
# ==============================================================================
# Configure Rust compiler settings for optimal build performance

echo ""
echo "Configuring Rust compilation settings..."

# Configure incremental compilation if requested
# Incremental compilation reuses previous build results for faster rebuilds
if [ "$INCREMENTAL" = true ]; then
    echo "   → Enabling incremental compilation for faster development builds"
    export CARGO_INCREMENTAL=1
    echo "   → CARGO_INCREMENTAL=$CARGO_INCREMENTAL (1=enabled)"
else
    echo "   → Using standard compilation for maximum optimization"
    export CARGO_INCREMENTAL=0
    echo "   → CARGO_INCREMENTAL=$CARGO_INCREMENTAL (0=disabled)"
fi

# Verify environment variable is properly set
echo "   ✅ Compilation configuration applied"

# ==============================================================================
# RUST COMPILATION
# ==============================================================================
# Build the Rust components that will become the Python extension

echo ""
echo "Building Rust components..."

# Build Rust components that will be compiled into the Python extension
# This creates the core native code that powers the dynamo._core module
#
# Key compilation flags explained:
# --locked: Use exact dependency versions from Cargo.lock (reproducible builds)
# --features dynamo-llm/block-manager: Enable block manager for efficient memory management
# --workspace: Build all workspace members (runtime, llm, bindings, etc.)
#
# Fast development compilation flags:
# CARGO_INCREMENTAL=1: Enable incremental compilation for faster rebuilds
# RUSTFLAGS="-C opt-level=0 -C codegen-units=256": Minimize optimization (0) and maximize parallelism (256 units)

if [ "$BUILD_TYPE" = "development" ]; then
    echo "   → DEVELOPMENT mode: Fastest build with incremental compilation and debug optimization"
    echo "   $ CARGO_INCREMENTAL=1 RUSTFLAGS=\"-C opt-level=0 -C codegen-units=256\" cargo build --locked --features dynamo-llm/block-manager --workspace"
    echo ""

    # Measure cargo build time
    start_time=$(date +%s.%N)
    CARGO_INCREMENTAL=1 RUSTFLAGS="-C opt-level=0 -C codegen-units=256" cargo build --locked --features dynamo-llm/block-manager --workspace
    end_time=$(date +%s.%N)
    build_duration=$(echo "$end_time - $start_time" | bc 2>/dev/null || echo "unknown")

    echo ""
    echo "   ✅ Rust components built successfully (DEVELOPMENT mode - fast compilation)"
    if [ "$build_duration" != "unknown" ]; then
        echo "   Cargo build time: ${build_duration}s"
        # Store build time for --check to display later
        echo "$build_duration" > /tmp/dynamo_cargo_build_time
    fi
else
    echo "   → RELEASE mode: Full optimization, slower build, faster runtime"
    echo "   $ cargo build --release --locked --features dynamo-llm/block-manager --workspace"
    echo ""

    # Measure cargo build time
    start_time=$(date +%s.%N)
    cargo build --release --locked --features dynamo-llm/block-manager --workspace
    end_time=$(date +%s.%N)
    build_duration=$(echo "$end_time - $start_time" | bc 2>/dev/null || echo "unknown")

    echo ""
    echo "   ✅ Rust components built successfully (RELEASE mode)"
    if [ "$build_duration" != "unknown" ]; then
        echo "   Cargo build time: ${build_duration}s"
        # Store build time for --check to display later
        echo "$build_duration" > /tmp/dynamo_cargo_build_time
    fi
fi

# ==============================================================================
# WHEEL CLEANUP
# ==============================================================================
# Remove any existing wheel files to ensure clean output

echo ""
echo "Cleaning up previous wheel files..."

# Clean up any existing wheel files from previous builds
# Wheel filename format: ai_dynamo_runtime-0.4.0-cp310-cp310-linux_x86_64.whl
if ls $WHEEL_OUTPUT_DIR/*.whl 1> /dev/null 2>&1; then
    echo "   → Found existing wheel files, removing them..."
    ls $WHEEL_OUTPUT_DIR/*.whl | head -3 | while read file; do
        echo "     - Removing: $(basename $file)"
    done
    rm -f $WHEEL_OUTPUT_DIR/*.whl
    rmdir $WHEEL_OUTPUT_DIR
    echo "   ✅ Existing wheel files removed"
else
    echo "   → No existing wheel files found (clean slate)"
    echo "   Expected wheel name: ai_dynamo_runtime-${DYNAMO_VERSION}-cp3XX-cp3XX-linux_x86_64.whl"
fi

# ==============================================================================
# MATURIN TOOL SETUP
# ==============================================================================
# Install maturin - the tool that bridges Rust and Python

# Install maturin with patchelf support for building Python extensions (if not already installed)
# maturin: Specialized tool for building Python packages with Rust code
# patchelf: Handles shared library dependencies and RPATH settings in Linux binaries
if ! uv pip show maturin >/dev/null 2>&1; then
    echo ""
    echo "Setting up Python-Rust build tools..."
    echo "   → maturin not found, installing with patchelf support..."
    uv pip install maturin[patchelf]
    echo "   ✅ maturin installed successfully"
fi

# ==============================================================================
# PYTHON PACKAGE CREATION
# ==============================================================================
# Create the ai-dynamo-runtime Python package with Rust extensions

echo ""
echo "Creating Python package ai-dynamo-runtime with Rust extensions..."

# Build Python package with native Rust extension using maturin
# This creates the ai-dynamo-runtime wheel containing the dynamo Python package with:
#   - dynamo._core.so (45MB Rust extension with runtime, LLM, metrics)
#   - dynamo.runtime (Python APIs and utilities)
#   - dynamo.llm (LLM inference functionality)
#   - dynamo.nixl_connect (NVIDIA connectivity layer)
#
# Key maturin flags explained:
# --features block-manager: Enable block manager feature for memory optimization
# --out: Specify output directory for the built wheel
# --uv: Use uv package manager for faster dependency resolution
# --debug: Include debug symbols for development builds

if [ "$BUILD_TYPE" = "development" ]; then
    # DEVELOPMENT MODE: Install package in editable mode for hot-reload
    echo "   → DEVELOPMENT MODE: Creating editable installation for development"
    echo ""

    # Clean up any existing installations to ensure no ambiguity
    cleanup_all_dynamo_installations "development installation"
    echo ""

    echo "   Files that will be generated:"
    echo "      • .pth files in site-packages (pointing to source)"
    echo "      • Symlinks to source directories"
    echo "      • Debug build of dynamo._core.so"
    echo "      • No wheel file created - direct source linking"
    echo ""
    echo "   Benefits: Python changes live, Rust changes need rebuild"
    echo ""
    echo "   $ CARGO_INCREMENTAL=1 RUSTFLAGS=\"-C opt-level=0 -C codegen-units=256\" maturin develop --uv --features block-manager --quiet"
    echo ""

    (cd $WORKSPACE_DIR/lib/bindings/python && CARGO_INCREMENTAL=1 RUSTFLAGS="-C opt-level=0 -C codegen-units=256" maturin develop --uv --features block-manager --quiet)

    echo ""
    echo "✅ SUCCESS: Development mode package installed!"
    echo "   Type: Editable installation (linked to source)"
    echo "   Python import: import dynamo"
    if [ "$DEBUG" = true ]; then
        echo ""
        echo "   .pth file locations (editable install links):"
        PYTHON_SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
        echo "      Site-packages: $PYTHON_SITE_PACKAGES"
        if [ -f "$PYTHON_SITE_PACKAGES/__editables_finder__.py" ]; then
            echo "      Found: __editables_finder__.py (modern editable install)"
        fi
        if ls "$PYTHON_SITE_PACKAGES"/*.pth 1> /dev/null 2>&1; then
            echo "      .pth files found:"
            ls "$PYTHON_SITE_PACKAGES"/*.pth | grep -E "(dynamo|editables)" | head -3 | while read pth_file; do
                echo "         • $(basename $pth_file)"
                echo "           Content: $(head -1 $pth_file)"
            done
        fi
        echo ""
    fi
    echo "   Tip: Python changes are live, Rust changes need rebuild"
else
    # RELEASE MODE: Build distributable wheel for production deployment
    echo "   → RELEASE MODE: Creating optimized wheel for distribution"
    echo ""

    # Clean up any existing installations to ensure no ambiguity
    cleanup_all_dynamo_installations "release installation"
    echo ""

    echo "   Files that will be generated:"
    echo "      • ai_dynamo_runtime-${DYNAMO_VERSION}-*.whl (~60MB total)"
    echo "      • Contains: dynamo/_core.so (~45MB Rust), Python modules (~16MB)"
    echo "      • Location: $WHEEL_OUTPUT_DIR/"
    echo ""
    echo "   $ maturin build --release --features block-manager --out $WHEEL_OUTPUT_DIR --quiet"
    echo ""

    (cd $WORKSPACE_DIR/lib/bindings/python && maturin build --release --features block-manager --out $WHEEL_OUTPUT_DIR --quiet)

    echo ""
    echo "Installing wheel package..."
    if ls $WHEEL_OUTPUT_DIR/*.whl 1> /dev/null 2>&1; then
        WHEEL_FILE=$(ls $WHEEL_OUTPUT_DIR/*.whl)
        WHEEL_SIZE=$(du -h "$WHEEL_FILE" | cut -f1)
        echo "   → Found wheel: $(basename $WHEEL_FILE) ($WHEEL_SIZE)"
        echo "   $ uv pip install --upgrade --force-reinstall --no-deps"
        uv pip install --upgrade --force-reinstall --no-deps $WHEEL_FILE
        echo ""
        echo "✅ SUCCESS: Wheel package built and installed!"
        echo "   Wheel location: $WHEEL_FILE"
        echo "   Wheel size: $WHEEL_SIZE"
        echo "   Contains: dynamo._core.so (~45MB Rust), Python modules (~16MB)"
        echo "   Python import: import dynamo"
    else
        echo ""
        echo "❌ ERROR: No wheel files found in $WHEEL_OUTPUT_DIR/"
        echo "   Check the maturin build output above for errors"
        exit 1
    fi

    # Build ai-dynamo package (complete framework)
    echo ""
    echo "Building ai-dynamo package (complete framework)..."
    echo "   → Building complete Dynamo framework with all components"
    echo "   → Contains: frontend, planner, backends (vllm, sglang, trtllm, llama_cpp, mocker)"
    echo "   $ pip wheel --no-deps --wheel-dir $WHEEL_OUTPUT_DIR ."
    echo ""

    pip wheel --no-deps --wheel-dir $WHEEL_OUTPUT_DIR .

    echo ""
    echo "Installing ai-dynamo package..."
    if ls $WHEEL_OUTPUT_DIR/ai_dynamo-*.whl 1> /dev/null 2>&1; then
        AI_DYNAMO_WHEEL=$(ls $WHEEL_OUTPUT_DIR/ai_dynamo-*.whl)
        AI_DYNAMO_SIZE=$(du -h "$AI_DYNAMO_WHEEL" | cut -f1)
        echo "   → Found wheel: $(basename $AI_DYNAMO_WHEEL) ($AI_DYNAMO_SIZE)"
        echo "   $ uv pip install --upgrade --find-links $WHEEL_OUTPUT_DIR"
        uv pip install --upgrade --find-links $WHEEL_OUTPUT_DIR $AI_DYNAMO_WHEEL
        echo ""
        echo "✅ SUCCESS: ai-dynamo package built and installed!"
        echo "   Wheel location: $AI_DYNAMO_WHEEL"
        echo "   Wheel size: $AI_DYNAMO_SIZE"
        echo "   Contains: Complete Dynamo framework (frontend, planner, all backends)"
        echo "   Python import: import dynamo.frontend, dynamo.planner, etc."
    else
        echo ""
        echo "❌ ERROR: No ai-dynamo wheel files found in $WHEEL_OUTPUT_DIR/"
        echo "   Check the pip wheel output above for errors"
        exit 1
    fi
fi

echo ""
echo "Final verification of installation state..."

# Show PYTHONPATH information
echo "Python Environment Information:"
if [ -n "$PYTHONPATH" ]; then
    echo "   PYTHONPATH is set - Python will search these directories for modules:"
    echo "$PYTHONPATH" | tr ':' '\n' | sed 's/^/      • /'
    echo "   This allows importing dynamo.frontend, dynamo.planner, etc. directly from source"
else
    echo "   PYTHONPATH: Not set (using default Python paths only)"
    echo "   To import dynamo components from source, set PYTHONPATH to include component directories"
fi
echo "   Python site-packages: $(python3 -c "import site; print(site.getsitepackages()[0])")"
echo ""

# Check final state consistency
PIP_RUNTIME=$(pip show ai-dynamo-runtime >/dev/null 2>&1 && echo "detected" || echo "not found")
UV_RUNTIME=$(uv pip show ai-dynamo-runtime >/dev/null 2>&1 && echo "detected" || echo "not found")
PIP_DYNAMO=$(pip show ai-dynamo >/dev/null 2>&1 && echo "detected" || echo "not found")
UV_DYNAMO=$(uv pip show ai-dynamo >/dev/null 2>&1 && echo "detected" || echo "not found")

echo "   Package detection:"
echo "      ai-dynamo-runtime: pip=$PIP_RUNTIME, uv=$UV_RUNTIME"
echo "      ai-dynamo: pip=$PIP_DYNAMO, uv=$UV_DYNAMO"

# Test import
echo "   Testing Python import..."
if python3 -c "import dynamo; print('✅ dynamo imported successfully')" 2>/dev/null; then
    echo "      ✅ 'import dynamo' works correctly"
else
    echo "      ❌ 'import dynamo' failed"
fi

echo ""
echo "Build completed successfully!"
echo "   You can now use 'import dynamo' in Python to access the runtime"
echo ""
echo "Installation verification:"
if [ "$BUILD_TYPE" = "development" ]; then
    echo "   • Development mode: Changes to Python files are immediately available"
    echo "   • For Rust changes: Re-run ./bin/pybuild.sh --development"
    echo "   • Check .pth files: ls $(python3 -c "import site; print(site.getsitepackages()[0])")/*dynamo*.pth"
else
    echo "   • Release mode: Optimized wheel installed"
    echo "   • Wheel location: $(ls $WHEEL_OUTPUT_DIR/*.whl 2>/dev/null | head -1 || echo 'Not found')"
    echo "   • For updates: Re-run ./bin/pybuild.sh --release"
fi

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
fi
