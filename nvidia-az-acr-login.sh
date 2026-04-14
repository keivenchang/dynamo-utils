#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Azure ACR access for ai-dynamo/dynamo images
# Registry: dynamoci.azurecr.io/ai-dynamo/dynamo
#
# Usage:
#   ./nvidia-az-acr-login.sh                              # login (az acr login)
#   ./nvidia-az-acr-login.sh --login                      # same as above
#   ./nvidia-az-acr-login.sh --show-available-images       # list tags (quick, no save)
#   ./nvidia-az-acr-login.sh --list-available-images-to-cache  # full metadata saved to cache
#
# Prerequisites:
#   - Azure CLI (az) installed
#   - Logged in: az login
#   - Access to dynamoci ACR (DL: access-azure-dynamo-engineer)

set -euo pipefail

ACR_NAME=dynamoci
ACR_REGISTRY=dynamoci.azurecr.io
REPO=ai-dynamo/dynamo
ACR_CACHE_DIR=~/.cache/dynamo-utils/azure-acr-registry-cache

login() {
    echo "=== Logging into Azure ACR: $ACR_NAME ==="
    if ! command -v az &>/dev/null; then
        echo "ERROR: Azure CLI (az) not found. Install: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli"
        exit 1
    fi
    # Check if logged into Azure
    if ! az account show &>/dev/null; then
        echo "Not logged into Azure. Running: az login"
        az login
    fi
    az acr login --name "$ACR_NAME"
    echo "Logged into $ACR_REGISTRY"
}

show_available_images() {
    echo "=== Listing tags in ${REPO} ==="
    az acr repository show-tags \
        --name "$ACR_NAME" \
        --repository "$REPO" \
        --orderby time_desc \
        --top 30 \
        --output table
}

show_available_images_to_cache() {
    mkdir -p "$ACR_CACHE_DIR"
    local out="$ACR_CACHE_DIR/az-acr-ai-dynamo-dynamo-details.json"
    echo "=== Fetching detailed tag metadata for ${REPO} (this may take a while) ==="
    az acr repository show-tags \
        --name "$ACR_NAME" \
        --repository "$REPO" \
        --orderby time_desc \
        --detail \
        --output json \
        | python3 -c "
import json, sys
tags = json.load(sys.stdin)
# Normalize to imageDetails format matching ECR cache structure
details = []
for t in tags:
    details.append({
        'imageTags': [t['name']],
        'imagePushedAt': t.get('createdTime', ''),
        'imageDigest': t.get('digest', ''),
        'imageSizeInBytes': 0,
    })
out = {'imageDetails': details}
json.dump(out, open('$out', 'w'), indent=2, default=str)
print(f'Saved {len(details)} image details to $out')
"
}

case "${1:-}" in
    --login)
        login
        ;;
    --show-available-images)
        show_available_images
        ;;
    --list-available-images-to-cache)
        show_available_images_to_cache
        ;;
    "")
        login
        ;;
    *)
        echo "Usage: $0 [--login|--show-available-images|--list-available-images-to-cache]"
        exit 1
        ;;
esac
