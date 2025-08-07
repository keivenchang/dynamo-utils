#!/bin/bash

set -euo pipefail

BACKENDS=components/backends

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
    --agg         Run in aggregation mode (dynamo serve)
                  Default: run in standard mode (dynamo run)
    --model MODEL Specify the model to use
                  Options: "deepseek" (DeepSeek-R1-Distill-Qwen-1.5B)
                          Default: "qwen" (Qwen/Qwen3-0.6B)
    --framework FRAMEWORK Specify the framework directory
                         Available: $available_frameworks
                         Default: "vllm"
    --dryrun, --dry-run Show what would be executed without running
                       (dry run mode)

DESCRIPTION:
    This script builds the workspace and runs Dynamo inference.
    When --agg is specified, it runs "dynamo serve" for aggregation mode.
    Otherwise, it runs "dynamo run" for standard inference mode.
    Use --model to specify which model to use for inference.
    Use --dryrun or --dry-run to see what would be executed without running.
EOF
}

# Parse command line arguments
AGG_MODE=false
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
        --agg)
            AGG_MODE=true
            shift
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

# Validate that if --agg is used, then --model cannot be specified
if [ "$AGG_MODE" = true ] && [ "$MODEL_SPECIFIED" = true ]; then
    echo "Error: If --agg is specified, then --model cannot be specified"
    exit 1
fi

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

# Execute init.sh to set up the environment
# dry_run_echo "Setting up environment with init.sh..."
# if [ "$DRY_RUN" = false ]; then
#     $(dirname "$0")/init.sh
# fi

set -x
cd $WORKSPACE_DIR
export PYTHONPATH=$WORKSPACE_DIR/deploy/sdk/src:$WORKSPACE_DIR/components/planner/src
#PYTHONPATH_SRC=$(find $WORKSPACE_DIR -type d -path "*/src/dynamo" -exec dirname {} \; | sort -u | paste -sd: -)
#export PYTHONPATH=$PYTHONPATH_SRC:$PYTHONPATH

#time CARGO_INCREMENTAL=1 cargo build --workspace --bin dynamo-run
#    --bin http --bin llmctl
# time uv pip install -e .

if [ "$DRY_RUN" = false ]; then
    pkill -f "python3.*--endpoint" || true
fi

if [ "$AGG_MODE" = true ]; then
    dry_run_echo "Running in aggregation mode (dynamo serve)"
    dry_run_exec "cd $BACKENDS/$FRAMEWORK"
    # look at launch/agg.sh
    dry_run_exec "grep 'model' deploy/agg.yaml"

    if [ "$DRY_RUN" = false ]; then
        # Set up trap to kill all background processes on exit
        trap 'echo Cleaning up...; kill $(jobs -p) 2>/dev/null || true; exit' INT TERM EXIT

        # Start background processes
        python -m dynamo.frontend &
        FRONTEND_PID=$!

        unset HF_TOKEN
        DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8081 python -m dynamo.vllm --model Qwen/Qwen3-0.6B --enforce-eager --no-enable-prefix-caching &
        VLLM_PID=$!

        # Wait for both processes
        echo "Launched frontend and vllm processes. Press Ctrl+C to exit."
        wait $FRONTEND_PID $VLLM_PID
    else
        dry_run_echo "Would start background processes:"
        dry_run_echo "  - python -m dynamo.frontend"
        dry_run_echo "  - DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8081 python -m dynamo.vllm --model Qwen/Qwen3-0.6B --enforce-eager --no-enable-prefix-caching"
        dry_run_echo "Would wait for both processes to complete"
    fi
else
    dry_run_echo "Running in standard mode (dynamo run)"
    dry_run_echo "Using model: $MODEL"

    if [ "$DRY_RUN" = false ]; then
        # Set up trap to kill all child processes on exit
        trap 'echo Cleaning up...; kill $(jobs -p) 2>/dev/null || true; pkill -P $$ 2>/dev/null || true; exit' INT TERM EXIT

        dynamo run out=vllm $MODEL
    else
        dry_run_echo "Would execute: dynamo run out=vllm $MODEL"
    fi
fi
