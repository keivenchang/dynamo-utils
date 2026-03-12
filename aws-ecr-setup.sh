#!/usr/bin/env bash
# AWS ECR access setup for ai-dynamo/dynamo images
# Registry: 210086341041.dkr.ecr.us-west-2.amazonaws.com/ai-dynamo/dynamo
#
# Usage:
#   ./aws-ecr-setup.sh              # login (skips browser if creds still valid)
#   ./aws-ecr-setup.sh --login      # same as above
#   ./aws-ecr-setup.sh --logout     # stop refresh daemons + clear credentials
#   ./aws-ecr-setup.sh --install    # one-time: install AWS CLI + nvsec
#   ./aws-ecr-setup.sh --docker     # login + docker login for pull/push
#   ./aws-ecr-setup.sh --list-images       # login + list ECR image tags (fetches from API)
#   ./aws-ecr-setup.sh --fast-list-images  # quick ECR check (first 20 tags, no login check)

set -euo pipefail

REGISTRY_ID=210086341041
REGION=us-west-2
REPO=ai-dynamo/dynamo
NVSEC_VENV=~/nvsec-env
ECR_CACHE_DIR=~/.cache/dynamo-utils/ecr

# Cursor's browser helper relies on VSCODE_IPC_HOOK_CLI to talk to the
# Cursor server. Terminals opened long ago may have a stale socket.
# Find the newest valid socket so browser opening works.
if [ -n "${BROWSER:-}" ] && [ -z "${DISPLAY:-}" ]; then
    _newest_sock=$(find /run/user/$(id -u) -maxdepth 1 -name 'vscode-ipc-*.sock' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2) || true
    if [ -n "${_newest_sock:-}" ]; then
        export VSCODE_IPC_HOOK_CLI="$_newest_sock"
    fi
fi

# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------
# a) Request DL Access entitlement: "access-triton-aws-engineer"
#    This grants the CS-Engineer role on the triton-aws account (210086341041).
#    Without it, nvsec auth works but you'll see zero accounts/roles.
#
# b) Download/install the NVSec CLI tool (--install below).
#    Source repo: https://gitlab-master.nvidia.com/security/security-portal/nvsec-tool

install() {
    echo "=== Installing AWS CLI v2 ==="
    if command -v aws &>/dev/null; then
        echo "AWS CLI already installed: $(aws --version)"
    else
        cd /tmp \
            && curl -sS "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
            && unzip -qo awscliv2.zip \
            && sudo ./aws/install
        echo "Installed: $(aws --version)"
    fi

    echo ""
    echo "=== Installing nvsec ==="
    if [ ! -d "$NVSEC_VENV" ]; then
        python3 -m venv "$NVSEC_VENV"
    fi
    source "$NVSEC_VENV/bin/activate"
    python3 -m pip install --upgrade nvsec \
        -i https://urm.nvidia.com/artifactory/api/pypi/sw-cloudsec-pypi/simple \
        --extra-index-url https://urm.nvidia.com/artifactory/api/pypi/sw-cftt-pypi-local/simple
    echo ""
    echo "Done. Run this script without --install to login."
}

# Full nvsec auth + AWS credential configuration flow.
# Cursor handles port forwarding + browser opening automatically.
# For raw SSH, use: nvsec aws auth --no-browser
#   (requires: ssh -L 53682:localhost:53682 user@host)
# NOTE: The browser tab may show "You need to enable JavaScript to run this app."
#   after auth completes. This is harmless -- the CLI already got the token.
#   Just close the tab.
# Full auth flow: nvsec SSO (browser) -> list roles -> configure AWS CLI creds.
authenticate_nvsec_and_aws() {
    source "$NVSEC_VENV/bin/activate"

    echo "=== Authenticating (opens browser) ==="
    nvsec aws auth

    echo ""
    echo "=== Available accounts/roles ==="
    nvsec aws list

    echo ""
    echo "=== Configuring CLI credentials (profile: default) ==="
    nvsec aws configure 0 --profile default

    echo ""
    echo "=== Verifying credentials ==="
    aws sts get-caller-identity --region "$REGION" --output table
}

# Login: skip browser if creds are still valid, otherwise relogin.
login() {
    source "$NVSEC_VENV/bin/activate"

    if aws sts get-caller-identity --region "$REGION" &>/dev/null; then
        echo "=== AWS credentials still valid, skipping browser auth ==="
        aws sts get-caller-identity --region "$REGION" --output table
        return
    fi

    authenticate_nvsec_and_aws
}

logout() {
    source "$NVSEC_VENV/bin/activate"

    echo "=== Stopping nvsec refresh daemons ==="
    nvsec aws stop --all 2>/dev/null || true

    echo ""
    echo "=== Clearing AWS credentials, SSO cache, and CLI cache ==="
    rm -f ~/.aws/credentials
    rm -f ~/.aws/config
    rm -rf ~/.aws/sso/cache/*
    rm -rf ~/.aws/cli/cache/*
    rm -f ~/.nvsec/access.json

    echo ""
    echo "=== Logging out of Docker ECR ==="
    docker logout "${REGISTRY_ID}.dkr.ecr.${REGION}.amazonaws.com" 2>/dev/null || true

    echo ""
    echo "Logged out. Run '$0 --login' to re-authenticate."
}

docker_login() {
    echo "=== Docker login to ECR ==="
    aws ecr get-login-password --region "$REGION" \
        | docker login --username AWS --password-stdin "${REGISTRY_ID}.dkr.ecr.${REGION}.amazonaws.com"
    echo ""
    echo "Docker logged in. Example pull:"
    echo "  docker pull ${REGISTRY_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:<tag>"
}

list_tags() {
    mkdir -p "$ECR_CACHE_DIR"
    local out="$ECR_CACHE_DIR/aws-ecr-ai-dynamo-dynamo-tags.json"
    echo "=== Listing tagged images in ${REPO} ==="
    aws ecr list-images \
        --registry-id "$REGISTRY_ID" \
        --repository-name "$REPO" \
        --region "$REGION" \
        --filter tagStatus=TAGGED \
        --output json \
        > "$out"
    local count
    count=$(python3 -c "import json; d=json.load(open('$out')); print(len(d['imageIds']))")
    echo "Saved $count tagged images to $out"
}

# Fast list: quick ECR call (first page only, no login check, no save).
fast_list() {
    echo "=== Quick ECR check: ${REPO} ==="
    aws ecr list-images \
        --registry-id "$REGISTRY_ID" \
        --repository-name "$REPO" \
        --region "$REGION" \
        --filter tagStatus=TAGGED \
        --max-items 20 \
        --output json
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
case "${1:-}" in
    --install)
        install
        ;;
    --login)
        login
        ;;
    --logout)
        logout
        ;;
    --docker)
        login
        echo ""
        docker_login
        ;;
    --list-images)
        login
        echo ""
        list_tags
        ;;
    --fast-list-images)
        fast_list
        ;;
    "")
        login
        ;;
    *)
        echo "Usage: $0 [--login|--logout|--install|--docker|--list-images|--fast-list-images]"
        exit 1
        ;;
esac
