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
    cat << EOF
Usage: $0 [OPTIONS]

Run Dynamo inference in the development environment.

OPTIONS:
    -h, --help    Show this help message and exit

    --model MODEL Specify the model to use
                  Options: "deepseek" (deepseek-ai/DeepSeek-R1-Distill-Llama-8B)
                           "tinyllama" (TinyLlama/TinyLlama-1.1B-Chat-v1.0)
                          Default: "qwen" (Qwen/Qwen3-0.6B)
    --framework FRAMEWORK Specify the framework directory
                         Available: $available_frameworks
                         Default: "vllm"
    --dryrun, --dry-run Show what would be executed without running
                       (dry run mode)

DESCRIPTION:
    This script builds the workspace and runs Dynamo inference in aggregation mode.
    Use --model to specify which model to use for inference.
    Use --dryrun or --dry-run to see what would be executed without running.
EOF
}

# Parse command line arguments
MODEL_SPECIFIED=false
DRY_RUN=false
QWEN_MODEL="Qwen/Qwen3-0.6B"
# deepseek-ai/DeepSeek-R1-Distill-Llama-8B
DEEPSEEK_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL="$QWEN_MODEL"
FRAMEWORK="vllm"
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;

        --model)
            MODEL_SPECIFIED=true
            if [ "$2" = "deepseek" ]; then
                MODEL="$DEEPSEEK_MODEL"
            fi
            shift 2
            ;;
        --framework)
            FRAMEWORK="$2"
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

    # Start background processes
    python -m dynamo.frontend &
    FRONTEND_PID=$!

    unset HF_TOKEN
    DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=$DYN_BACKEND_PORT \
    python -m dynamo.vllm --model Qwen/Qwen3-0.6B --gpu-memory-utilization 0.20 --enforce-eager --no-enable-prefix-caching &
    VLLM_PID=$!

    # Wait for both processes
    echo "Launched frontend and vllm processes. Press Ctrl+C to exit."
    wait $FRONTEND_PID $VLLM_PID
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
    dry_run_echo "DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=$DYN_BACKEND_PORT python -m dynamo.vllm --model Qwen/Qwen3-0.6B --gpu-memory-utilization 0.20 --enforce-eager --no-enable-prefix-caching &"
    dry_run_echo "VLLM_PID=\$!"

    # Wait for both processes
    dry_run_echo "echo \"Launched frontend and vllm processes. Press Ctrl+C to exit.\""
    dry_run_echo "wait \$FRONTEND_PID \$VLLM_PID"
fi
