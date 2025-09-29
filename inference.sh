#!/bin/bash

set -euo pipefail

BACKENDS=components/backends
# Use existing environment variables if set, otherwise use defaults
: ${DYN_FRONTEND_PORT:=8000}
: ${DYN_BACKEND_PORT:=8081}

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
            echo "$QWEN_MODEL"
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

    # Try to validate the model using Python
    if python -c "
import sys
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('$model', trust_remote_code=True)
    print('Model validation successful')
    sys.exit(0)
except Exception as e:
    print(f'Model validation failed: {e}')
    sys.exit(1)
" >/dev/null 2>&1; then
        echo "✅ Model(s) found: $model"
        return 0
    else
        echo "❌ Error: Model '$model' not found or not accessible"
        echo "Please ensure the model exists and is accessible"
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
    --dryrun, --dry-run Show what would be executed without running
                       (dry run mode)
    --frontend             Run the frontend component
    --backend              Run the backend component
                          (if neither specified, both components run by default)

DESCRIPTION:
    This script builds the workspace and runs Dynamo inference in aggregation mode.
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

if [ ! -e /.dockerenv ]; then
    echo "This script must be run inside a Docker container."
    exit 1
fi

if [ "$DRY_RUN" = false ]; then
    (ps -ef --forest|grep multiprocess|awk '{print $2}'|xargs kill) && true
    (ps -ef|grep "python3.*\/tmp"|awk '{print $2}'|xargs kill) && true
    (ps -ef|grep "VLLM::EngineCore"|awk '{print $2}'|xargs kill) && true
else
    dry_run_echo "Would kill multiprocess processes: \$(ps -ef --forest|grep multiprocess|awk '{print \$2}')"
    dry_run_echo "Would kill python3 temp processes: \$(ps -ef|grep \"python3.*\/tmp\"|awk '{print \$2}')"
    dry_run_echo "Would kill VLLM processes: \$(ps -ef|grep \"VLLM::EngineCore\"|awk '{print \$2}')"
fi
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

# Run aggregation mode by default
dry_run_echo "Running Dynamo inference in aggregation mode"

# Only check deploy/agg.yaml if we have a valid framework
if [ "$RUN_BACKEND" = true ] && [ -n "$FRAMEWORK" ]; then
    dry_run_exec "cd $BACKENDS/$FRAMEWORK && grep 'model' deploy/agg.yaml"
fi

# Set up trap to kill all background processes on exit
if [ "$DRY_RUN" = false ]; then
    cmd trap 'echo Cleaning up...; kill $(jobs -p) 2>/dev/null || true; exit' INT TERM EXIT
else
    cmd trap 'echo "[DRY RUN] Would clean up and kill background processes on exit"; exit' INT TERM EXIT
fi

#cmd export PYTHONPATH="$HOME/dynamo/components/router/src:$HOME/dynamo/components/metrics/src:$HOME/dynamo/components/frontend/src:$HOME/dynamo/components/planner/src:$HOME/dynamo/components/backends/mocker/src:$HOME/dynamo/components/backends/trtllm/src:$HOME/dynamo/components/backends/vllm/src:$HOME/dynamo/components/backends/sglang/src:$HOME/dynamo/components/backends/llama_cpp/src"

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

# Set framework-specific arguments
if [ "$FRAMEWORK" = "vllm" ]; then
    FRAMEWORK_ARGS="--gpu-memory-utilization 0.20 --enforce-eager --no-enable-prefix-caching --max-num-seqs 64"
elif [ "$FRAMEWORK" = "trtllm" ]; then
    FRAMEWORK_ARGS="--free-gpu-memory-fraction 0.20 --max-num-tokens 8192 --max-batch-size 64"
elif [ "$FRAMEWORK" = "sglang" ]; then
    FRAMEWORK_ARGS="--mem-fraction-static 0.20 --max-running-requests 64"
else
    FRAMEWORK_ARGS=""
fi

# Start background processes based on component selection
FRONTEND_PID=""
BACKEND_PID=""

# Start frontend if requested
if [ "$RUN_FRONTEND" = true ]; then
    cmd python -m dynamo.frontend &
    if [ "$DRY_RUN" = false ]; then
        FRONTEND_PID=$!
    else
        dry_run_echo "FRONTEND_PID=\$!"
    fi
fi

# Framework detection already done earlier

# Start backend if requested
if [ "$RUN_BACKEND" = true ]; then
    cmd unset HF_TOKEN
    if [ "$DRY_RUN" = false ]; then
        ( set -x; DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=$DYN_BACKEND_PORT python -m dynamo.$FRAMEWORK --model "$MODEL" $FRAMEWORK_ARGS ) &
        BACKEND_PID=$!
    else
        ( set -x; : DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=$DYN_BACKEND_PORT python -m dynamo.$FRAMEWORK --model "$MODEL" $FRAMEWORK_ARGS ) 2>&1 | sed 's/^+ : /+ /'
        dry_run_echo "BACKEND_PID=\$!"
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
    dry_run_echo "Launched $FRAMEWORK backend process only. Press Ctrl+C to exit."
    if [ "$DRY_RUN" = false ]; then
        wait $BACKEND_PID
    else
        dry_run_echo "wait \$BACKEND_PID"
    fi
else
    dry_run_echo "Launched frontend and $FRAMEWORK processes. Press Ctrl+C to exit."
    if [ "$DRY_RUN" = false ]; then
        wait $FRONTEND_PID $BACKEND_PID
    else
        dry_run_echo "wait \$FRONTEND_PID \$BACKEND_PID"
    fi
fi
