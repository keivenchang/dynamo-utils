#!/bin/bash

set -euo pipefail

BACKENDS=examples/backends
# Use existing environment variables if set, otherwise use defaults
: ${DYN_FRONTEND_PORT:=8000}
: ${DYN_BACKEND_PORT:=8081}

# GPU memory utilization defaults (can be overridden with --gpu-mem-fraction)
GPU_MEMORY_UTIL_AGG=0.24      # Aggregated mode: single worker
GPU_MEMORY_UTIL_DISAGG=0.24   # Disaggregated mode: per worker (decode + prefill)

# Function to check if ports are already bound
check_port_available() {
    local port="$1"
    local component="$2"

    # Check if port is in use
    if netstat -tuln 2>/dev/null | grep -q ":$port "; then
        return 1
    elif ss -tuln 2>/dev/null | grep -q ":$port "; then
        return 1
    fi

    return 0
}

find_available_backend_port() {
    local base_port="$DYN_BACKEND_PORT"
    local max_attempts=10
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        local test_port=$((base_port + attempt))

        if check_port_available "$test_port" "backend"; then
            echo "✅ Port $test_port (backend) is available"
            if [ $attempt -gt 0 ]; then
                echo "  (original port $base_port was in use, incremented by +$attempt)"
                # Update the backend port for use in the script
                DYN_BACKEND_PORT=$test_port
            fi
            return 0
        fi

        attempt=$((attempt + 1))
    done

    echo "❌ Error: No available backend ports found in range $base_port to $((base_port + max_attempts - 1))"
    echo "Please free up some ports in this range before running the script."
    exit 1
}

# Function to get available frameworks
get_available_frameworks() {
    if [ -d "$BACKENDS" ]; then
        ls $BACKENDS/ 2>/dev/null | tr '\n' ' ' | sed 's/ $//'
    else
        echo "none found"
    fi
}

# Function to check NVIDIA/CUDA availability
check_nvidia_available() {
    # Check if nvidia-smi is available and working - this is the definitive test
    if nvidia-smi >/dev/null 2>&1; then
        return 0
    fi

    # If nvidia-smi fails, GPU is not available regardless of device files
    return 1
}

# Function to detect and validate framework for backend
# Returns 0 if framework is available, 1 if not
detect_and_validate_framework() {
    if [ "$FRAMEWORK_SPECIFIED" = false ]; then
        echo "Auto-detecting framework..."
        if ! FRAMEWORK=$(auto_select_framework 2>/dev/null); then
            echo "❌ Auto-detection failed: No suitable framework found"
            echo "   Available framework directories: $(get_available_frameworks)"
            echo "   Suggestion: Use --framework <name> to specify a framework explicitly"
            return 1
        fi
        echo "✅ Auto-selected framework: $FRAMEWORK"
    else
        echo "Using specified framework: $FRAMEWORK"
    fi

    # Validate that the framework directory exists
    if [ ! -d "$WORKSPACE_DIR/$BACKENDS/$FRAMEWORK" ]; then
        echo "❌ Error: Framework '$FRAMEWORK' not found in $WORKSPACE_DIR/$BACKENDS/"
        echo "   Available frameworks: $(get_available_frameworks)"
        return 1
    fi

    # Validate that required deployment files exist (for aggregation mode)
    if [ ! -f "$WORKSPACE_DIR/$BACKENDS/$FRAMEWORK/deploy/agg.yaml" ]; then
        echo "⚠️  Warning: Framework '$FRAMEWORK' missing deploy/agg.yaml"
        echo "   This framework may not support aggregation mode"
        return 1
    fi

    return 0
}

# Function to detect which frameworks are actually installed
detect_installed_frameworks() {
    local installed_frameworks=""
    local nvidia_available=false

    # Check NVIDIA availability first
    if check_nvidia_available; then
        nvidia_available=true
    else
        echo "❌ Error: NVIDIA GPU not detected or drivers not available" >&2
        echo "   Backend frameworks require NVIDIA GPU support" >&2
        # Don't exit here - let the caller handle the empty result
        echo ""
        return 1
    fi

    # Check for vLLM
    if python -c "import vllm" >/dev/null 2>&1; then
        if [ "$nvidia_available" = true ]; then
            installed_frameworks="$installed_frameworks vllm"
        else
            echo "   vLLM detected but NVIDIA GPU not available - skipping"
        fi
    fi

    # Check for TensorRT-LLM (only if NVIDIA is available)
    if [ "$nvidia_available" = true ] && python -c "import tensorrt_llm" >/dev/null 2>&1; then
        installed_frameworks="$installed_frameworks trtllm"
    elif [ "$nvidia_available" = false ]; then
        echo "   Skipping TensorRT-LLM check (requires NVIDIA GPU)"
    fi

    # Check for SGLang
    if python -c "import sglang" >/dev/null 2>&1; then
        if [ "$nvidia_available" = true ]; then
            installed_frameworks="$installed_frameworks sglang"
        else
            echo "   SGLang detected but NVIDIA GPU not available - skipping"
        fi
    fi

    echo "$installed_frameworks" | sed 's/^ *//'
}

# Function to resolve model type to actual model path
resolve_model() {
    local model_input="$1"

    # Handle empty or unspecified model (default to qwen)
    if [ -z "$model_input" ] || [ "$model_input" = "" ]; then
        model_input="qwen"
    fi

    # Map predefined model shortcuts to actual model paths
    case "$model_input" in
        "deepseek")
            echo "$DEEPSEEK_MODEL"
            ;;
        "tinyllama")
            echo "$TINYLLAMA_MODEL"
            ;;
        "qwen")
            # Use local HF cache to avoid rate limits
            local qwen_cache_dir="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B"
            if [ -d "$qwen_cache_dir/snapshots" ]; then
                # Get the latest snapshot (most recent directory)
                local latest_snapshot=$(ls -t "$qwen_cache_dir/snapshots" 2>/dev/null | head -1)
                if [ -n "$latest_snapshot" ]; then
                    echo "$qwen_cache_dir/snapshots/$latest_snapshot"
                else
                    # Fallback to HF model name if snapshot not found
                    echo "$QWEN_MODEL"
                fi
            else
                # Fallback to HF model name if cache dir doesn't exist
                echo "$QWEN_MODEL"
            fi
            ;;
        *)
            # Use custom model path as specified by user
            echo "$model_input"
            ;;
    esac
}

# Function to validate model availability
validate_model() {
    local model="$1"

    # Try online validation first (allows downloads if needed)
    local online_result
    online_result=$(python -c "
import sys
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('$model', trust_remote_code=True)
    print('Model validation successful')
    sys.exit(0)
except Exception as e:
    print(f'VALIDATION_ERROR: {e}')
    sys.exit(1)
" 2>&1)

    if [ $? -eq 0 ]; then
        echo "✅ Model(s) found: $model"
        return 0
    fi

    # Check if it's a rate limit error or auth error and print it
    if echo "$online_result" | grep -qE "429.*Too Many Requests|401.*Unauthorized|403.*Forbidden"; then
        echo "⚠️  HuggingFace API access failed (rate limit or auth error), falling back to offline validation..."
        local error_line=$(echo "$online_result" | grep -oP '(429|401|403).*?(?= for url)' | head -n 1)
        if [ -n "$error_line" ]; then
            echo "    (Error: $error_line)"
        fi
    else
        echo "⚠️  Online validation failed, falling back to offline validation..."
    fi

    # Fall back to offline validation with HF_HUB_OFFLINE=1
    if HF_HUB_OFFLINE=1 python -c "
import sys
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('$model', trust_remote_code=True, local_files_only=True)
    print('Model validation successful (offline)')
    sys.exit(0)
except Exception as e:
    print(f'Offline validation failed: {e}')
    sys.exit(1)
" >/dev/null 2>&1; then
        echo "✅ Model(s) found in local cache: $model"
        # Set offline mode globally for the rest of the script execution
        export HF_HUB_OFFLINE=1
        echo "   (Set HF_HUB_OFFLINE=1 to avoid rate limits for the rest of this run)"
        return 0
    else
        echo "❌ Error: Model '$model' not found or not accessible"
        echo "   Online validation failed, and model not found in local cache"
        echo "   Please download the model first or check HuggingFace cache at ~/.cache/huggingface/hub/"
        return 1
    fi
}

# Function to auto-select framework
auto_select_framework() {
    local installed=$(detect_installed_frameworks 2>/dev/null)
    local available=$(get_available_frameworks)

    if [ "$installed" = "" ]; then
        echo "❌ Error: No supported frameworks are installed" >&2
        echo "Available framework directories: $available" >&2

        # If NVIDIA is available and trtllm backend exists, suggest using it
        if check_nvidia_available && [ -d "$BACKENDS/trtllm" ]; then
            echo "⚠️  However, NVIDIA GPU detected and trtllm backend directory exists." >&2
            echo "   You can try: $0 --framework trtllm [other options]" >&2
        fi

        echo "Please install one of: vllm, tensorrt_llm, sglang, llama_cpp" >&2
        return 1
    fi

    # Priority order: vllm > trtllm > sglang > llama_cpp
    for framework in vllm trtllm sglang llama_cpp; do
        if echo "$installed" | grep -q "$framework"; then
            # Also check if the backend directory exists
            if [ -d "$BACKENDS/$framework" ]; then
                echo "$framework"
                return 0
            fi
        fi
    done

    # If we get here, frameworks are installed but no backend directories exist
    echo "❌ Error: Frameworks installed ($installed) but no matching backend directories found" >&2
    echo "Available backend directories: $available" >&2
    return 1
}

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

# Function to execute command or show dry run
dry_run_exec() {
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would execute: $*"
    else
        eval "$*"
    fi
}

# Help function
show_help() {
    local available_frameworks=$(get_available_frameworks)
    local installed_frameworks=$(detect_installed_frameworks)
    local auto_selected=$(auto_select_framework 2>/dev/null || echo "none")

    cat << EOF
Usage: $0 [OPTIONS]

Run Dynamo inference in the development environment.

OPTIONS:
    -h, --help    Show this help message and exit

    --model MODEL Specify the model to use
                  Predefined options: "deepseek" (deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
                                     "tinyllama" (TinyLlama/TinyLlama-1.1B-Chat-v1.0)
                                     "qwen" (Qwen/Qwen3-0.6B)
                  Or specify any custom model path (e.g., "microsoft/DialoGPT-medium")
                  Default: "qwen" (Qwen/Qwen3-0.6B)
    --framework FRAMEWORK Specify the framework to use
                         Available directories: $available_frameworks
                         Installed frameworks: $installed_frameworks
                         Auto-selected: $auto_selected
                         Default: auto-detect (or "vllm" if auto-detection fails)
    --disagg              Run in disaggregated mode (prefill + decode workers)
                         Default: aggregated mode (single worker)
    --gpu-mem-fraction FRACTION
                         GPU memory fraction to use (0.0 to 1.0)
                         Default: 0.24 (24% of GPU memory)
    --enable-prefix-caching
                         Enable vLLM automatic prefix caching (prefix cache hits)
                         Default: disabled in this script (uses --no-enable-prefix-caching)
    --enable-local-indexer
                         Enable worker-local KV indexer for tracking worker's KV cache state.
                         Useful for debugging and observability in disaggregated mode.
                         Only applicable to vLLM framework.
    --use-original-model-name
                         Use the original HuggingFace model name instead of local cache path.
                         Example: Use "Qwen/Qwen3-0.6B" instead of
                         "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/..."
                         Useful when you want consistent model naming across systems.
    --lmcache             Enable LMCache KV cache offloading (vLLM only, enabled by default)
    --no-lmcache          Disable LMCache KV cache offloading (vLLM only)
                         Note: LMCache only supported on x86 architecture
                         Aggregated mode: Uses LMCacheConnectorV1
                         Disaggregated mode: Prefill worker uses LMCache + NIXL multi-connector
    --dryrun, --dry-run Show what would be executed without running
                       (dry run mode)
    --frontend             Run the frontend component
    --backend              Run the backend component
                          (if neither specified, both components run by default)

DESCRIPTION:
    This script builds the workspace and runs Dynamo inference.
    By default, it runs in aggregation mode (single worker).
    Use --disagg to run in disaggregated mode (separate prefill and decode workers).
    The script will automatically detect which framework is installed and available.
    Use --framework to override auto-detection.
    Use --model to specify which model to use for inference.
    Use --dryrun or --dry-run to see what would be executed without running.
EOF
}

# Parse command line arguments
DRY_RUN=false
FRAMEWORK_SPECIFIED=false
RUN_FRONTEND=false
RUN_BACKEND=false
DISAGG_MODE=false
ENABLE_LMCACHE=true  # Enabled by default for vLLM (can be disabled with --no-lmcache)
ENABLE_PREFIX_CACHING=false  # vLLM only; default disabled to reduce memory usage
GPU_MEM_FRACTION_OVERRIDE=""  # Will override defaults if set
ENABLE_LOCAL_INDEXER=false  # vLLM only; enable worker-local KV indexer
USE_ORIGINAL_MODEL_NAME=false  # Use original model name instead of cached path
QWEN_MODEL="Qwen/Qwen3-0.6B"
TINYLLAMA_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# deepseek-ai/DeepSeek-R1-Distill-Llama-8B
DEEPSEEK_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_INPUT=""  # Will store the user's model input, empty means default
FRAMEWORK=""  # Will be auto-detected if not specified
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;

        --model)
            MODEL_INPUT="$2"
            shift 2
            ;;
        --framework)
            FRAMEWORK="$2"
            FRAMEWORK_SPECIFIED=true
            shift 2
            ;;
        --disagg)
            DISAGG_MODE=true
            shift
            ;;
        --lmcache)
            ENABLE_LMCACHE=true
            shift
            ;;
        --no-lmcache)
            ENABLE_LMCACHE=false
            shift
            ;;
        --gpu-mem-fraction)
            GPU_MEM_FRACTION_OVERRIDE="$2"
            shift 2
            ;;
        --enable-prefix-caching)
            ENABLE_PREFIX_CACHING=true
            shift
            ;;
        --enable-local-indexer)
            ENABLE_LOCAL_INDEXER=true
            shift
            ;;
        --use-original-model-name)
            USE_ORIGINAL_MODEL_NAME=true
            shift
            ;;
        --dryrun|--dry-run)
            DRY_RUN=true
            shift
            ;;
        --frontend)
            RUN_FRONTEND=true
            shift
            ;;
        --backend)
            RUN_BACKEND=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Set default behavior: if no component flags specified, run both
if [ "$RUN_FRONTEND" = false ] && [ "$RUN_BACKEND" = false ]; then
    RUN_FRONTEND=true
    RUN_BACKEND=true
fi

# Resolve model input to actual model path
MODEL=$(resolve_model "$MODEL_INPUT")

# If --use-original-model-name is set, use the original model name instead of cache path
if [ "$USE_ORIGINAL_MODEL_NAME" = true ]; then
    case "$MODEL_INPUT" in
        ""|"qwen")
            MODEL="$QWEN_MODEL"
            ;;
        "tinyllama")
            MODEL="$TINYLLAMA_MODEL"
            ;;
        "deepseek")
            MODEL="$DEEPSEEK_MODEL"
            ;;
        *)
            # Already using original name for custom models
            MODEL="$MODEL_INPUT"
            ;;
    esac
    echo "Using original model name: $MODEL"
fi

# Framework detection will happen later, right before backend launch

# Validate model availability
if [ "$DRY_RUN" = false ]; then
    echo "Validating model: $MODEL"
    if ! validate_model "$MODEL"; then
        exit 1
    fi
else
    dry_run_echo "Would validate model: $MODEL"
    dry_run_echo "✅ Model(s) found: $MODEL"
fi

# Note: The script now runs in aggregation mode by default, so --model can be used

# Show dry run indicator
if [ "$DRY_RUN" = true ]; then
    echo "=== DRY RUN MODE ==="
    echo "This is a dry run - no actual commands will be executed"
    echo "==================="
    echo
fi

# Function to clean up dynamo processes
# Parameters: cleanup_frontend (true/false), cleanup_backend (true/false)
cleanup_dynamo_processes() {
    local cleanup_frontend="${1:-true}"
    local cleanup_backend="${2:-true}"

    if [ "$DRY_RUN" = false ]; then
        if [ "$cleanup_backend" = true ]; then
            (ps -ef --forest|grep multiprocess|awk '{print $2}'|xargs -r kill) 2>/dev/null || true
            (ps -ef|grep "python3.*\/tmp"|awk '{print $2}'|xargs -r kill) 2>/dev/null || true
            (ps -ef|grep "VLLM::EngineCore"|awk '{print $2}'|xargs -r kill) 2>/dev/null || true
            (ps -ef|grep "python -m dynamo.sglang.main"|awk '{print $2}'|xargs -r kill) 2>/dev/null || true
            pkill -f "python3 -m dynamo.vllm" 2>/dev/null || true
            pkill -f "python3 -m dynamo.sglang" 2>/dev/null || true
            pkill -f "python3 -m dynamo.trtllm" 2>/dev/null || true
        fi
        if [ "$cleanup_frontend" = true ]; then
            (ps -ef|grep "python -m dynamo.frontend"|grep -v grep|awk '{print $2}'|xargs -r kill) 2>/dev/null || true
            pkill -f "python -m dynamo.frontend" 2>/dev/null || true
        fi
    else
        if [ "$cleanup_backend" = true ]; then
            dry_run_echo "Would kill multiprocess processes: \$(ps -ef --forest|grep multiprocess|awk '{print \$2}')"
            dry_run_echo "Would kill python3 temp processes: \$(ps -ef|grep \"python3.*\/tmp\"|awk '{print \$2}')"
            dry_run_echo "Would kill VLLM processes: \$(ps -ef|grep \"VLLM::EngineCore\"|awk '{print \$2}')"
            dry_run_echo "Would kill sglang processes: \$(ps -ef|grep \"python -m dynamo.sglang.main\"|awk '{print \$2}')"
        fi
        if [ "$cleanup_frontend" = true ]; then
            dry_run_echo "Would kill frontend processes: \$(ps -ef|grep \"python -m dynamo.frontend\"|grep -v grep|awk '{print \$2}')"
        fi
    fi
}

if [ ! -e /.dockerenv ]; then
    echo "This script must be run inside a Docker container."
    exit 1
fi

# Clean up any existing processes before starting - only kill what we're about to launch
cleanup_dynamo_processes "$RUN_FRONTEND" "$RUN_BACKEND"
if [ -d "~/dynamo" ]; then
    WORKSPACE_DIR="~/dynamo"
elif [ -d "/workspace" ]; then
    WORKSPACE_DIR="/workspace"
else
    echo "Error: Neither ~/dynamo nor /workspace directory exists."
    exit 1
fi

# Framework directory validation will happen later, right before backend launch

#set -x
cd $WORKSPACE_DIR

#time CARGO_INCREMENTAL=1 cargo build --workspace --bin dynamo-run
#    --bin http --bin llmctl
# time uv pip install -e .

if [ "$DRY_RUN" = false ]; then
    pkill -f "python3.*--endpoint" || true
fi

# Detect and validate framework if backend will be started - do this early
if [ "$RUN_BACKEND" = true ]; then
    if ! detect_and_validate_framework; then
        # Check what the user actually requested
        if [ "$RUN_FRONTEND" = false ]; then
            # User explicitly requested backend only
            echo "❌ Error: Backend was explicitly requested but no suitable framework is available"
            echo "   This system requires NVIDIA GPU support for backend frameworks"
            exit 1
        else
            # User requested both components (default) or frontend + backend
            # If both were requested, fail completely rather than running partial system
            echo "❌ Error: Backend framework not available but both frontend and backend are needed"
            echo "   This system requires NVIDIA GPU support for backend frameworks"
            echo "   Use --frontend to run frontend only, or install GPU support for full system"
            exit 1
        fi
    fi
fi

# Display mode
if [ "$DISAGG_MODE" = true ]; then
    dry_run_echo "Running Dynamo inference in disaggregated mode (prefill + decode workers)"
else
    dry_run_echo "Running Dynamo inference in aggregation mode (single worker)"
fi

# Only check deploy yaml if we have a valid framework
if [ "$RUN_BACKEND" = true ] && [ -n "$FRAMEWORK" ]; then
    if [ "$DISAGG_MODE" = true ]; then
        dry_run_exec "cd $BACKENDS/$FRAMEWORK && grep 'model' deploy/disagg.yaml 2>/dev/null || echo 'Note: deploy/disagg.yaml not found'"
    else
        dry_run_exec "cd $BACKENDS/$FRAMEWORK && grep 'model' deploy/agg.yaml"
    fi
fi

# Set up trap to kill all background processes on exit
cleanup_on_exit() {
    echo "Cleaning up..."
    # Kill jobs started by this script
    kill $(jobs -p) 2>/dev/null || true

    # Kill PIDs we tracked
    [ -n "$FRONTEND_PID" ] && kill -9 $FRONTEND_PID 2>/dev/null || true
    [ -n "$BACKEND_PID" ] && kill -9 $BACKEND_PID 2>/dev/null || true
    [ -n "$PREFILL_PID" ] && kill -9 $PREFILL_PID 2>/dev/null || true

    # Force kill any remaining dynamo processes
    pkill -9 -f "python.*dynamo\.(frontend|trtllm|vllm|sglang)" 2>/dev/null || true

    # Reuse the cleanup function to be thorough
    cleanup_dynamo_processes

    echo "Cleanup complete."
}

if [ "$DRY_RUN" = false ]; then
    trap cleanup_on_exit INT TERM EXIT
else
    cmd trap 'echo "[DRY RUN] Would clean up and kill background processes on exit"; exit' INT TERM EXIT
fi

#cmd export PYTHONPATH="$HOME/dynamo/components/router/src:$HOME/dynamo/components/frontend/src:$HOME/dynamo/components/planner/src:$HOME/dynamo/components/backends/mocker/src:$HOME/dynamo/components/backends/trtllm/src:$HOME/dynamo/components/backends/vllm/src:$HOME/dynamo/components/backends/sglang/src:$HOME/dynamo/components/backends/llama_cpp/src"

# Check torch import and CUDA availability after PYTHONPATH is set (only if backend will be started)
if [ "$RUN_BACKEND" = true ]; then
    dry_run_echo "Checking torch import and CUDA availability..."

    # Check torch import
    if [ "$DRY_RUN" = false ]; then
        if ! python -c "import torch" >/dev/null 2>&1; then
            echo "❌ Error: Cannot import torch. Please ensure PyTorch is installed."
            exit 1
        fi
    else
        cmd python -c "import torch"
    fi

    # Check if CUDA is available
    if [ "$DRY_RUN" = false ]; then
        if ! python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null | grep -q "CUDA available: True"; then
            echo "❌ Error: CUDA is not available. Please ensure CUDA drivers and PyTorch CUDA support are properly installed."
            echo "CUDA status:"
            python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count() if torch.cuda.is_available() else 'N/A')" 2>/dev/null || echo "Failed to get CUDA status"
            exit 1
        fi
        echo "Torch import successful and CUDA is available"
        echo "CUDA device count: $(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 'Unknown')"
    else
        cmd python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
        dry_run_echo "Torch import successful and CUDA is available"
        dry_run_echo "CUDA device count: \$(python -c \"import torch; print(torch.cuda.device_count())\" 2>/dev/null || echo 'Unknown')"
    fi
fi

# Import check for dynamo.frontend before launching (only if frontend will be started)
if [ "$RUN_FRONTEND" = true ]; then
    if [ "$DRY_RUN" = false ]; then
        if ! python -c "import dynamo.frontend" >/dev/null 2>&1; then
            echo "❌ Import check failed: cannot import dynamo.frontend"
            exit 1
        fi
    else
        cmd python -c "import dynamo.frontend"
    fi
fi

# Check if required ports are available before starting services
if [ "$RUN_FRONTEND" = true ]; then
    if [ "$DRY_RUN" = false ]; then
        if ! check_port_available "$DYN_FRONTEND_PORT" "frontend"; then
            echo "❌ Error: Port $DYN_FRONTEND_PORT (frontend) is already bound"
            echo "Please free up this port before running the script."
            exit 1
        fi
        echo "✅ Port $DYN_FRONTEND_PORT (frontend) is available"
    else
        cmd check_port_available "$DYN_FRONTEND_PORT" "frontend"
    fi
fi
if [ "$RUN_BACKEND" = true ]; then
    if [ "$DRY_RUN" = false ]; then
        find_available_backend_port
    else
        cmd find_available_backend_port
    fi
fi

# Apply GPU memory fraction override if specified
if [ -n "$GPU_MEM_FRACTION_OVERRIDE" ]; then
    GPU_MEMORY_UTIL_AGG="$GPU_MEM_FRACTION_OVERRIDE"
    GPU_MEMORY_UTIL_DISAGG="$GPU_MEM_FRACTION_OVERRIDE"
    echo "Using GPU memory fraction: $GPU_MEM_FRACTION_OVERRIDE"
fi

# Set framework-specific arguments
BATCH_SIZE=2
if [ "$FRAMEWORK" = "vllm" ]; then
    if [ "$ENABLE_PREFIX_CACHING" = true ]; then
        PREFIX_CACHING_ARGS="--enable-prefix-caching"
    else
        PREFIX_CACHING_ARGS="--no-enable-prefix-caching"
    fi
    FRAMEWORK_ARGS="--gpu-memory-utilization $GPU_MEMORY_UTIL_AGG --enforce-eager $PREFIX_CACHING_ARGS --max-num-seqs $BATCH_SIZE"
    # Add LMCache connector if enabled
    if [ "$ENABLE_LMCACHE" = true ]; then
        FRAMEWORK_ARGS="$FRAMEWORK_ARGS --connector lmcache"
    fi
    # Add local indexer if enabled
    if [ "$ENABLE_LOCAL_INDEXER" = true ]; then
        FRAMEWORK_ARGS="$FRAMEWORK_ARGS --enable-local-indexer true"
    fi
elif [ "$FRAMEWORK" = "sglang" ]; then
    FRAMEWORK_ARGS="--mem-fraction-static $GPU_MEMORY_UTIL_AGG --max-running-requests $BATCH_SIZE --enable-metrics"
elif [ "$FRAMEWORK" = "trtllm" ]; then
    FRAMEWORK_ARGS="--free-gpu-memory-fraction $GPU_MEMORY_UTIL_AGG --max-num-tokens 512 --max-batch-size $BATCH_SIZE --publish-events-and-metrics"
else
    FRAMEWORK_ARGS=""
fi

# Function to check if a port is available
check_port() {
    local port=$1
    # Try ss first (modern), then netstat (older), then lsof
    if command -v ss >/dev/null 2>&1; then
        if ss -ltn | grep -q ":$port "; then
            return 1  # Port is in use
        fi
    elif command -v netstat >/dev/null 2>&1; then
        if netstat -ltn | grep -q ":$port "; then
            return 1  # Port is in use
        fi
    elif command -v lsof >/dev/null 2>&1; then
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            return 1  # Port is in use
        fi
    else
        echo "Warning: No port checking tool available (ss, netstat, or lsof)"
    fi
    return 0  # Port is available
}

# Function to check all required ports before starting
check_all_ports() {
    local ports_to_check=()
    local port_descriptions=()

    # Frontend port
    if [ "$RUN_FRONTEND" = true ]; then
        ports_to_check+=(8000)
        port_descriptions+=("frontend")
    fi

    # Backend ports
    if [ "$RUN_BACKEND" = true ]; then
        if [ "$DISAGG_MODE" = true ]; then
            ports_to_check+=($DECODE_PORT)
            port_descriptions+=("decode worker")
            ports_to_check+=($PREFILL_PORT)
            port_descriptions+=("prefill worker")
        else
            ports_to_check+=($DYN_BACKEND_PORT)
            port_descriptions+=("backend")
        fi
    fi

    # Check each port
    local all_available=true
    for i in "${!ports_to_check[@]}"; do
        local port=${ports_to_check[$i]}
        local desc=${port_descriptions[$i]}
        if ! check_port $port; then
            echo "❌ Port $port is already in use (needed for $desc)"
            all_available=false
        fi
    done

    if [ "$all_available" = false ]; then
        echo ""
        echo "Please free up the ports above before running this script."
        echo "You can find processes using ports with: lsof -i :<port>"
        echo "Kill processes with: kill -9 <pid>"
        exit 1
    fi
}

# Function to get PIDs using a port
get_port_pids() {
    local port=$1
    # Try different methods to find PIDs
    if command -v ss >/dev/null 2>&1; then
        ss -ltnp 2>/dev/null | grep ":$port " | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | sort -u
    elif command -v lsof >/dev/null 2>&1; then
        lsof -ti:$port 2>/dev/null || true
    elif command -v fuser >/dev/null 2>&1; then
        fuser $port/tcp 2>/dev/null | tr -d ' ' || true
    fi
}

# Function to kill processes on specific ports with retry
kill_port_processes() {
    local ports=("$@")
    for port in "${ports[@]}"; do
        local max_retries=5
        local retry=0
        while [ $retry -lt $max_retries ]; do
            local pids=$(get_port_pids $port)
            if [ -z "$pids" ]; then
                break
            fi
            echo "Killing processes on port $port: $pids (attempt $((retry + 1))/$max_retries)"
            echo "$pids" | xargs kill -9 2>/dev/null || true
            sleep 2
            retry=$((retry + 1))
        done

        # Final check
        local remaining=$(get_port_pids $port)
        if [ -n "$remaining" ]; then
            echo "Warning: Port $port still has processes after cleanup: $remaining"
        fi
    done
}

# Start background processes based on component selection
FRONTEND_PID=""
BACKEND_PID=""
PREFILL_PID=""

# Kill existing processes and check port availability before starting services
if [ "$DRY_RUN" = false ]; then
    # Set port variables for disagg mode
    if [ "$DISAGG_MODE" = true ]; then
        PREFILL_PORT=$DYN_BACKEND_PORT
        DECODE_PORT=$((DYN_BACKEND_PORT + 1))
    fi

    # Kill processes on required ports
    ports_to_kill=()
    if [ "$RUN_FRONTEND" = true ]; then
        ports_to_kill+=(8000)
    fi
    if [ "$RUN_BACKEND" = true ]; then
        if [ "$DISAGG_MODE" = true ]; then
            ports_to_kill+=($PREFILL_PORT $DECODE_PORT)
        else
            ports_to_kill+=($DYN_BACKEND_PORT)
        fi
    fi

    if [ ${#ports_to_kill[@]} -gt 0 ]; then
        echo "Cleaning up processes on ports: ${ports_to_kill[*]}"
        kill_port_processes "${ports_to_kill[@]}"
    fi

    # Now check if ports are available
    check_all_ports
fi

# Framework detection already done earlier

# Start backend FIRST (so it's ready before frontend accepts requests)
if [ "$RUN_BACKEND" = true ]; then
    # Keep HF_TOKEN set so ModelExpress can authenticate with HuggingFace API
    # cmd unset HF_TOKEN  # DO NOT unset - causes 429 rate limit errors
    metrics="true"

    if [ "$DISAGG_MODE" = true ]; then
        # Disaggregated mode: Launch prefill worker (port 8081) and decode worker (port 8082)
        # Both use GPU_MEMORY_UTIL_DISAGG GPU memory to share the same GPU
        # NOTE: SGLang disaggregated mode typically requires 2 separate GPUs (one for prefill, one for decode)
        # Running both on same GPU may cause OOM errors. Consider using aggregated mode for single GPU.
        PREFILL_PORT=$DYN_BACKEND_PORT
        DECODE_PORT=$((DYN_BACKEND_PORT + 1))

        # Set framework-specific memory args for disagg mode
        if [ "$FRAMEWORK" = "vllm" ]; then
            if [ "$ENABLE_PREFIX_CACHING" = true ]; then
                PREFIX_CACHING_ARGS="--enable-prefix-caching"
            else
                PREFIX_CACHING_ARGS="--no-enable-prefix-caching"
            fi
            DISAGG_FRAMEWORK_ARGS="--gpu-memory-utilization $GPU_MEMORY_UTIL_DISAGG --enforce-eager $PREFIX_CACHING_ARGS --max-num-seqs $BATCH_SIZE"
            PREFILL_FLAG="--is-prefill-worker"
            DECODE_FLAG=""

            # Add LMCache support for prefill worker only (uses multi-connector with NIXL)
            if [ "$ENABLE_LMCACHE" = true ]; then
                PREFILL_FLAG="$PREFILL_FLAG --connector lmcache nixl"
            fi
            
            # Add local indexer if enabled (for both prefill and decode)
            if [ "$ENABLE_LOCAL_INDEXER" = true ]; then
                DISAGG_FRAMEWORK_ARGS="$DISAGG_FRAMEWORK_ARGS --enable-local-indexer true"
            fi
        elif [ "$FRAMEWORK" = "sglang" ]; then
            DISAGG_FRAMEWORK_ARGS="--mem-fraction-static $GPU_MEMORY_UTIL_DISAGG --page-size 16 --chunked-prefill-size 4096 --max-prefill-tokens 4096 --enable-memory-saver --delete-ckpt-after-loading --max-running-requests $BATCH_SIZE --enable-metrics --disaggregation-bootstrap-port 12345 --host 0.0.0.0 --disaggregation-transfer-backend nixl"
            PREFILL_FLAG="--disaggregation-mode prefill"
            DECODE_FLAG="--disaggregation-mode decode"
        elif [ "$FRAMEWORK" = "trtllm" ]; then
            # TensorRT-LLM disaggregated mode uses test YAML configs
            PREFILL_YAML="$WORKSPACE_DIR/tests/serve/configs/trtllm/prefill.yaml"
            DECODE_YAML="$WORKSPACE_DIR/tests/serve/configs/trtllm/decode.yaml"
            DISAGG_FRAMEWORK_ARGS="--publish-events-and-metrics"
            PREFILL_FLAG="--disaggregation-mode prefill --extra-engine-args $PREFILL_YAML"
            DECODE_FLAG="--disaggregation-mode decode --extra-engine-args $DECODE_YAML"
        else
            DISAGG_FRAMEWORK_ARGS=""
            PREFILL_FLAG=""
            DECODE_FLAG=""
        fi

        # Convert to percentage for display (use awk instead of bc for portability)
        GPU_MEM_PERCENT=$(awk "BEGIN {printf \"%.0f%%\", $GPU_MEMORY_UTIL_DISAGG * 100}")

        # Assign different NIXL side-channel ports so prefill and decode workers
        # don't collide when running on the same machine (vLLM default is 5600).
        NIXL_PORT_PREFILL=5600
        NIXL_PORT_DECODE=5601

        # Launch prefill worker FIRST so it can register before decode worker tries to connect
        dry_run_echo "Launching prefill worker on port $PREFILL_PORT ($GPU_MEM_PERCENT GPU memory)..."
        if [ "$DRY_RUN" = false ]; then
            ( set -x; VLLM_NIXL_SIDE_CHANNEL_PORT=$NIXL_PORT_PREFILL DYN_LOG=info DYN_SYSTEM_PORT=$PREFILL_PORT python3 -m dynamo.$FRAMEWORK --model "$MODEL" $DISAGG_FRAMEWORK_ARGS $PREFILL_FLAG ) 2>&1 | sed 's/^/[PREFILL] /' &
            PREFILL_PID=$!
        else
            ( set -x; : VLLM_NIXL_SIDE_CHANNEL_PORT=$NIXL_PORT_PREFILL DYN_LOG=info DYN_SYSTEM_PORT=$PREFILL_PORT python3 -m dynamo.$FRAMEWORK --model "$MODEL" $DISAGG_FRAMEWORK_ARGS $PREFILL_FLAG ) 2>&1 | sed 's/^+ : /+ /'
            dry_run_echo "PREFILL_PID=\$!"
        fi

        # Wait for prefill worker to initialize and register with NATS
        if [ "$DRY_RUN" = false ]; then
            echo "Waiting for prefill worker to initialize..."
            sleep 20

            # Poll the prefill worker's metrics endpoint to verify it's ready
            echo "Checking if prefill worker is ready..."
            max_attempts=30
            attempt=0
            while [ $attempt -lt $max_attempts ]; do
                if curl -s http://localhost:$PREFILL_PORT/metrics > /dev/null 2>&1; then
                    echo "Prefill worker is ready!"
                    break
                fi
                attempt=$((attempt + 1))
                if [ $attempt -lt $max_attempts ]; then
                    sleep 2
                fi
            done

            if [ $attempt -eq $max_attempts ]; then
                echo "Warning: Prefill worker may not be fully ready after $max_attempts attempts"
            fi
        else
            dry_run_echo "sleep 20"
            dry_run_echo "# Poll prefill worker metrics endpoint"
        fi

        # Launch decode worker SECOND (after prefill worker is ready)
        dry_run_echo "Launching decode worker on port $DECODE_PORT ($GPU_MEM_PERCENT GPU memory)..."
        if [ "$DRY_RUN" = false ]; then
            ( set -x; VLLM_NIXL_SIDE_CHANNEL_PORT=$NIXL_PORT_DECODE DYN_LOG=info DYN_SYSTEM_PORT=$DECODE_PORT python3 -m dynamo.$FRAMEWORK --model "$MODEL" $DISAGG_FRAMEWORK_ARGS $DECODE_FLAG ) 2>&1 | sed 's/^/[DECODE] /' &
            BACKEND_PID=$!
        else
            ( set -x; : VLLM_NIXL_SIDE_CHANNEL_PORT=$NIXL_PORT_DECODE DYN_LOG=info DYN_SYSTEM_PORT=$DECODE_PORT python3 -m dynamo.$FRAMEWORK --model "$MODEL" $DISAGG_FRAMEWORK_ARGS $DECODE_FLAG ) 2>&1 | sed 's/^+ : /+ /'
            dry_run_echo "BACKEND_PID=\$!"
        fi
    else
        # Aggregated mode: Launch single worker
        if [ "$DRY_RUN" = false ]; then
            ( set -x; DYN_LOG=info DYN_SYSTEM_PORT=$DYN_BACKEND_PORT python -m dynamo.$FRAMEWORK --model "$MODEL" $FRAMEWORK_ARGS ) &
            BACKEND_PID=$!
        else
            ( set -x; : DYN_LOG=info DYN_SYSTEM_PORT=$DYN_BACKEND_PORT python -m dynamo.$FRAMEWORK --model "$MODEL" $FRAMEWORK_ARGS ) 2>&1 | sed 's/^+ : /+ /'
            dry_run_echo "BACKEND_PID=\$!"
        fi
    fi
fi

# Start frontend AFTER backend (so backend is ready when frontend starts accepting requests)
if [ "$RUN_FRONTEND" = true ]; then
    if [ "$DISAGG_MODE" = true ]; then
        cmd env -u DYN_SYSTEM_PORT python -m dynamo.frontend --http-port $DYN_FRONTEND_PORT --router-mode kv &
    else
        cmd env -u DYN_SYSTEM_PORT python -m dynamo.frontend --http-port $DYN_FRONTEND_PORT &
    fi
    if [ "$DRY_RUN" = false ]; then
        FRONTEND_PID=$!
    else
        dry_run_echo "FRONTEND_PID=\$!"
    fi
fi

# Wait for processes based on what was started
if [ "$RUN_FRONTEND" = true ] && [ "$RUN_BACKEND" = false ]; then
    dry_run_echo "Launched frontend process only. Press Ctrl+C to exit."
    if [ "$DRY_RUN" = false ]; then
        wait $FRONTEND_PID
    else
        dry_run_echo "wait \$FRONTEND_PID"
    fi
elif [ "$RUN_BACKEND" = true ] && [ "$RUN_FRONTEND" = false ]; then
    if [ "$DISAGG_MODE" = true ]; then
        dry_run_echo "Launched $FRAMEWORK backend processes (decode + prefill) only. Press Ctrl+C to exit."
        if [ "$DRY_RUN" = false ]; then
            wait $BACKEND_PID $PREFILL_PID
        else
            dry_run_echo "wait \$BACKEND_PID \$PREFILL_PID"
        fi
    else
        dry_run_echo "Launched $FRAMEWORK backend process only. Press Ctrl+C to exit."
        if [ "$DRY_RUN" = false ]; then
            wait $BACKEND_PID
        else
            dry_run_echo "wait \$BACKEND_PID"
        fi
    fi
else
    if [ "$DISAGG_MODE" = true ]; then
        dry_run_echo "Launched frontend and $FRAMEWORK processes (decode + prefill). Press Ctrl+C to exit."
        if [ "$DRY_RUN" = false ]; then
            wait $FRONTEND_PID $BACKEND_PID $PREFILL_PID
        else
            dry_run_echo "wait \$FRONTEND_PID \$BACKEND_PID \$PREFILL_PID"
        fi
    else
        dry_run_echo "Launched frontend and $FRAMEWORK processes. Press Ctrl+C to exit."
        if [ "$DRY_RUN" = false ]; then
            wait $FRONTEND_PID $BACKEND_PID
        else
            dry_run_echo "wait \$FRONTEND_PID \$BACKEND_PID"
        fi
    fi
fi
