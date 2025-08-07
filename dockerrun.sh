#!/bin/bash

set -euo pipefail

# Default values
DYNAMO_IMAGE=${DYNAMO_IMAGE:-gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo:7bc70bc940ea4bf1cf63de868cee9021eceffcff-31415145-vllm_v1-amd64}
DRY_RUN=false
CONTAINER_NAME="dynamo"

# Function to display usage information
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --help              Show this help message and exit
    --dynamo-image IMAGE Specify the Dynamo Docker image to use
    --dryrun, --dry-run Show the docker command without executing it
    --name NAME         Set the docker container name (default: dynamo)

Environment Variables:
    DYNAMO_IMAGE        Default Docker image (can be overridden with --dynamo-image)
    HF_TOKEN           Hugging Face token for model access

Examples:
    $0
    $0 --dynamo-image my-custom-dynamo:latest
    $0 --dry-run
    $0 --name my-dynamo-container
    $0 --help

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h|--h)
            show_help
            exit 0
            ;;
        --dynamo-image)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --dynamo-image requires an argument" >&2
                exit 1
            fi
            DYNAMO_IMAGE="$2"
            shift 2
            ;;
        --name)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --name requires an argument" >&2
                exit 1
            fi
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --dryrun|--dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Error: Unknown option $1" >&2
            echo "Use --help for usage information" >&2
            exit 1
            ;;
    esac
done

# mkdir -p $HOME/hf_models
#    -v $HOME/hf_models:/root/.cache/huggingface \
#    --user $(id -u):$(id -g) \

# Fetch the container's default PATH
echo "Fetching default PATH from container..."
CONTAINER_PATH=$(docker run --rm ${DYNAMO_IMAGE} bash -c 'echo PATH=$PATH' | grep PATH= | cut -d'=' -f2)
echo "Container PATH: $CONTAINER_PATH"

# Add .rustup path to the container PATH
RUSTUP_PATH="/workspace/.build/.rustup/toolchains/1.87.0-x86_64-unknown-linux-gnu/bin"
ENHANCED_PATH="${RUSTUP_PATH}:${CONTAINER_PATH}"
echo "Enhanced PATH: $ENHANCED_PATH"

DOCKER_CMD="docker run --gpus all -it --rm --network host --runtime nvidia \
--shm-size=10G --ulimit memlock=-1 --ulimit stack=67108864 \
--ulimit nofile=65536:65536 \
-e HF_TOKEN \
-v $HOME/nvidia/dynamo/:/workspace \
-v /tmp:/tmp \
-v /mnt/:/mnt \
-v $HOME/.cache/huggingface:/root/.cache/huggingface \
-w /workspace \
--cap-add CAP_SYS_PTRACE \
--ipc host \
--privileged \
--name ${CONTAINER_NAME} \
-e PATH='${ENHANCED_PATH}' \
-e PYTHONPATH='/workspace/deploy/sdk/src:/workspace/components/planner/src' \
-e DYNAMO_IMAGE=${DYNAMO_IMAGE} \
-v /etc/passwd:/etc/passwd:ro \
-v /etc/group:/etc/group:ro \
${DYNAMO_IMAGE}"

# docker run --gpus all -it --rm --network host --runtime nvidia --shm-size=10G --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=65536:65536 -e HF_TOKEN -v ~/nvidia/dynamo/container/..:/workspace -v /tmp:/tmp -v /mnt/:/mnt -v ~/.cache/huggingface:/root/.cache/huggingface -w /workspace --cap-add CAP_SYS_PTRACE --ipc host --privileged dynamo:v0.1.0.dev.22e6c96f-vllm-dev


if [[ "$DRY_RUN" == "true" ]]; then
    echo "Would run:"
    echo "$DOCKER_CMD"
    exit 0
fi

eval $DOCKER_CMD
echo "Exited $DYNAMO_IMAGE with exit code $?"
