#!/bin/bash

set -euo pipefail

BACKENDS=components/backends
# Use existing environment variables if set, otherwise use defaults
: ${DYN_FRONTEND_PORT:=8000}
: ${DYN_BACKEND_PORT:=8081}

# Function to check if ports are already bound
check_ports_available() {
    local port_frontend_in_use=false
    local port_backend_in_use=false

    # Check if frontend port is in use
    if netstat -tuln 2>/dev/null | grep -q ":$DYN_FRONTEND_PORT "; then
        port_frontend_in_use=true
    elif ss -tuln 2>/dev/null | grep -q ":$DYN_FRONTEND_PORT "; then
        port_frontend_in_use=true
    fi

    # Check if backend port is in use
    if netstat -tuln 2>/dev/null | grep -q ":$DYN_BACKEND_PORT "; then
        port_backend_in_use=true
    elif ss -tuln 2>/dev/null | grep -q ":$DYN_BACKEND_PORT "; then
        port_backend_in_use=true
    fi

    if [ "$port_frontend_in_use" = true ] || [ "$port_backend_in_use" = true ]; then
        echo "❌ Error: Required ports are already in use:"
        if [ "$port_frontend_in_use" = true ]; then
            echo "  - Port $DYN_FRONTEND_PORT (frontend) is already bound"
        fi
        if [ "$port_backend_in_use" = true ]; then
            echo "  - Port $DYN_BACKEND_PORT (backend) is already bound"
        fi
        echo "Please free up these ports before running the script."
        exit 1
    fi

    echo "✅ Ports $DYN_FRONTEND_PORT (frontend) and $DYN_BACKEND_PORT (backend) are available"
}

# Function to get available frameworks
get_available_frameworks() {
    if [ -d "$BACKENDS" ]; then
        ls $BACKENDS/ 2>/dev/null | tr '\n' ' ' | sed 's/ $//'
    else
        echo "none found"
    fi
}

# Function to detect which frameworks are actually installed
detect_installed_frameworks() {
    local installed_frameworks=""

    # Check for vLLM
    if python -c "import vllm" >/dev/null 2>&1; then
        installed_frameworks="$installed_frameworks vllm"
    fi

    # Check for TensorRT-LLM
    if python -c "import tensorrt_llm" >/dev/null 2>&1; then
        installed_frameworks="$installed_frameworks trtllm"
    fi

    # Check for SGLang
    if python -c "import sglang" >/dev/null 2>&1; then
        installed_frameworks="$installed_frameworks sglang"
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
    local installed=$(detect_installed_frameworks)
    local available=$(get_available_frameworks)

    if [ "$installed" = "" ]; then
        echo "❌ Error: No supported frameworks are installed"
        echo "Available framework directories: $available"
        echo "Please install one of: vllm, tensorrt_llm, sglang, llama_cpp"
        exit 1
    fi

    # Priority order: vllm > trtllm > sglang > llama_cpp
    for framework in vllm trtllm sglang llama_cpp; do
        if echo "$installed" | grep -q "$framework"; then
            # Also check if the backend directory exists
            if [ -d "$BACKENDS/$framework" ]; then
                echo "$framework"
                return
            fi
        fi
    done

    # If we get here, frameworks are installed but no backend directories exist
    echo "❌ Error: Frameworks installed ($installed) but no matching backend directories found"
    echo "Available backend directories: $available"
    exit 1
}

# Function to handle dry run output
dry_run_echo() {
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] $*"
    else
        echo "$*"
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
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Resolve model input to actual model path
MODEL=$(resolve_model "$MODEL_INPUT")

# Auto-detect framework if not specified
if [ "$FRAMEWORK_SPECIFIED" = false ]; then
    echo "Auto-detecting framework..."
    FRAMEWORK=$(auto_select_framework)
    echo "✅ Auto-selected framework: $FRAMEWORK"
else
    echo "Using specified framework: $FRAMEWORK"
fi

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

(ps -ef --forest|grep multiprocess|awk '{print $2}'|xargs kill) && true
(ps -ef|grep "python3.*\/tmp"|awk '{print $2}'|xargs kill) && true
(ps -ef|grep "VLLM::EngineCore"|awk '{print $2}'|xargs kill) && true
if [ -d "~/dynamo" ]; then
    WORKSPACE_DIR="~/dynamo"
elif [ -d "/workspace" ]; then
    WORKSPACE_DIR="/workspace"
else
    echo "Error: Neither ~/dynamo nor /workspace directory exists."
    exit 1
fi

# Validate that the framework directory exists
if [ ! -d "$WORKSPACE_DIR/$BACKENDS/$FRAMEWORK" ]; then
    echo "Error: Framework '$FRAMEWORK' not found in $WORKSPACE_DIR/$BACKENDS/"
    echo "Available frameworks: $(get_available_frameworks)"
    exit 1
fi

#set -x
cd $WORKSPACE_DIR

#time CARGO_INCREMENTAL=1 cargo build --workspace --bin dynamo-run
#    --bin http --bin llmctl
# time uv pip install -e .

if [ "$DRY_RUN" = false ]; then
    pkill -f "python3.*--endpoint" || true
fi

# Run aggregation mode by default
dry_run_echo "Running Dynamo inference in aggregation mode"
dry_run_exec "cd $BACKENDS/$FRAMEWORK"
# look at launch/agg.sh
dry_run_exec "grep 'model' deploy/agg.yaml"

if [ "$DRY_RUN" = false ]; then
    # Set up trap to kill all background processes on exit
    trap 'echo Cleaning up...; kill $(jobs -p) 2>/dev/null || true; exit' INT TERM EXIT

    export PYTHONPATH="$HOME/dynamo/components/router/src:$HOME/dynamo/components/metrics/src:$HOME/dynamo/components/frontend/src:$HOME/dynamo/components/planner/src:$HOME/dynamo/components/backends/mocker/src:$HOME/dynamo/components/backends/trtllm/src:$HOME/dynamo/components/backends/vllm/src:$HOME/dynamo/components/backends/sglang/src:$HOME/dynamo/components/backends/llama_cpp/src"

    # Check torch import and CUDA availability after PYTHONPATH is set
    echo "Checking torch import and CUDA availability..."
    if ! python -c "import torch" >/dev/null 2>&1; then
        echo "❌ Error: Cannot import torch. Please ensure PyTorch is installed."
        exit 1
    fi

    # Check if CUDA is available
    if ! python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null | grep -q "CUDA available: True"; then
        echo "❌ Error: CUDA is not available. Please ensure CUDA drivers and PyTorch CUDA support are properly installed."
        echo "CUDA status:"
        python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count() if torch.cuda.is_available() else 'N/A')" 2>/dev/null || echo "Failed to get CUDA status"
        exit 1
    fi

    echo "Torch import successful and CUDA is available"
    echo "CUDA device count: $(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 'Unknown')"

    # Import check for dynamo.frontend before launching
    if ! python -c "import dynamo.frontend" >/dev/null 2>&1; then
        echo "❌ Import check failed: cannot import dynamo.frontend"
        exit 1
    fi

    # Check if required ports are available before starting services
    check_ports_available

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
    # Start background processes
    set -x
    python -m dynamo.frontend &
    FRONTEND_PID=$!
    unset HF_TOKEN

    DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=$DYN_BACKEND_PORT \
    python -m dynamo.$FRAMEWORK --model "$MODEL" $FRAMEWORK_ARGS &
    BACKEND_PID=$!
    set +x > /dev/null

    # Wait for both processes
    echo "Launched frontend and $FRAMEWORK processes. Press Ctrl+C to exit."
    wait $FRONTEND_PID $BACKEND_PID
else
    # Set up trap to kill all background processes on exit
    dry_run_echo "trap 'echo Cleaning up...; kill \$(jobs -p) 2>/dev/null || true; exit' INT TERM EXIT"

    dry_run_echo "export PYTHONPATH=\"$HOME/dynamo/components/router/src:$HOME/dynamo/components/metrics/src:$HOME/dynamo/components/frontend/src:$HOME/dynamo/components/planner/src:$HOME/dynamo/components/backends/mocker/src:$HOME/dynamo/components/backends/trtllm/src:$HOME/dynamo/components/backends/vllm/src:$HOME/dynamo/components/backends/sglang/src:$HOME/dynamo/components/backends/llama_cpp/src\""

    # Check torch import and CUDA availability after PYTHONPATH is set
    dry_run_echo "Checking torch import and CUDA availability..."
    dry_run_echo "python -c \"import torch\""

    # Check if CUDA is available
    dry_run_echo "python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\""
    dry_run_echo "echo \"Torch import successful and CUDA is available\""
    dry_run_echo "echo \"CUDA device count: \$(python -c \"import torch; print(torch.cuda.device_count())\" 2>/dev/null || echo 'Unknown')\""

    # Import check for dynamo.frontend before launching
    dry_run_echo "python -c \"import dynamo.frontend\""

    # Check if required ports are available before starting services
    dry_run_echo "check_ports_available"

    # Start background processes
    dry_run_echo "python -m dynamo.frontend &"
    dry_run_echo "FRONTEND_PID=\$!"

    dry_run_echo "unset HF_TOKEN"
    # Set framework-specific arguments
    if [ "$FRAMEWORK" = "vllm" ]; then
        FRAMEWORK_ARGS="--gpu-memory-utilization 0.20 --enforce-eager --no-enable-prefix-caching"
    elif [ "$FRAMEWORK" = "trtllm" ]; then
        FRAMEWORK_ARGS="--free-gpu-memory-fraction 0.20 --max-batch-size 64 --max-num-tokens 8192"
    elif [ "$FRAMEWORK" = "sglang" ]; then
        FRAMEWORK_ARGS="--mem-fraction-static 0.20"
    else
        FRAMEWORK_ARGS=""
    fi
    dry_run_echo "DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=$DYN_BACKEND_PORT python -m dynamo.$FRAMEWORK --model \"$MODEL\" \$FRAMEWORK_ARGS &"
    dry_run_echo "BACKEND_PID=\$!"

    # Wait for both processes
    dry_run_echo "echo \"Launched frontend and $FRAMEWORK processes. Press Ctrl+C to exit.\""
    dry_run_echo "wait \$FRONTEND_PID \$BACKEND_PID"
fi
