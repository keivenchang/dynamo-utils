#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Scrub the current HEAD commit message:
#   1. Remove lines starting with "Made-with:"
#   2. Remove lines starting with "Co-Authored-By:" (or "Co-authored-by:")
#   3. De-duplicate "Signed-off-by:" lines (keep first occurrence of each)
#   4. Strip trailing blank lines
#
# Usage: clean-commit.sh [--dry-run]

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

original=$(git log -1 --format="%B")

cleaned=$(printf '%s\n' "$original" | awk '
    /^Made-with:/ { next }
    /^[Cc]o-[Aa]uthored-[Bb]y:/ { next }
    /^Signed-off-by:/ {
        if ($0 in seen) next
        seen[$0] = 1
    }
    { lines[++n] = $0 }
    END {
        # trim trailing blank lines
        while (n > 0 && lines[n] ~ /^[[:space:]]*$/) n--
        for (i = 1; i <= n; i++) print lines[i]
    }
')

if [[ "$original" == "$cleaned"$'\n' ]] || [[ "$original" == "$cleaned" ]]; then
    echo "Nothing to clean."
    exit 0
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "--- original ---"
    printf '%s\n' "$original"
    echo "--- cleaned ---"
    printf '%s\n' "$cleaned"
    exit 0
fi

GIT_EDITOR=true git commit --amend -m "$cleaned"
echo "Commit message cleaned."
