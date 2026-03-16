#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Backfill compile/sanity builds for commits that have no log directory
# in LOG_REPO/logs/.
#
# Detection: scans all LOG_REPO/logs/<date>/*.<sha> directories to find
# which commits already have builds. Only commits with no matching log
# directory are backfilled.
#
# Usage:
#   ./backfill.sh [options] <start_sha> [end_sha]
#
# Options:
#   --dry-run           Show what would run without executing
#   --build-repo PATH   Repo to build in (default: ~/dynamo/dynamo_ci2)
#   --log-repo PATH     Repo to move logs to (default: ~/dynamo/dynamo_ci)
#
# Examples:
#   ./backfill.sh 12785247c                              # dynamo_ci2 -> dynamo_ci
#   ./backfill.sh --build-repo ~/dynamo/dynamo_ci 12785247c
#   ./backfill.sh --build-repo ~/dynamo/dynamo_ci2 --log-repo ~/dynamo/dynamo_ci 52b460e4c dcbccbcd2
#   ./backfill.sh --dry-run 12785247c

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO_HOME="${DYNAMO_HOME:-$HOME/dynamo}"
BUILD_REPO="${DYNAMO_HOME}/dynamo_ci2"
LOG_REPO="${DYNAMO_HOME}/dynamo_ci"
BUILD_SCRIPT="$SCRIPT_DIR/build_images.py"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --build-repo)
            BUILD_REPO="$2"
            shift 2
            ;;
        --log-repo)
            LOG_REPO="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,/^$/s/^# \?//p' "$0"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 [--dry-run] [--build-repo PATH] [--log-repo PATH] <start_sha> [end_sha]"
    echo "  start_sha: first commit (inclusive)"
    echo "  end_sha:   last commit (inclusive, default: origin/main)"
    exit 1
fi

START_SHA="$1"
END_SHA="${2:-origin/main}"

echo "Build repo:  $BUILD_REPO"
echo "Log repo:    $LOG_REPO"
echo ""

if [[ ! -d "$BUILD_REPO/.git" ]]; then
    echo "ERROR: $BUILD_REPO is not a git repo" >&2
    exit 1
fi

# If build and log repos differ, verify log repo exists too
if [[ "$BUILD_REPO" != "$LOG_REPO" && ! -d "$LOG_REPO/logs" ]]; then
    echo "WARNING: $LOG_REPO/logs does not exist, will create on first move" >&2
fi

cd "$BUILD_REPO"
git fetch origin --quiet

ALL_SHAS=$(git log --oneline "${START_SHA}^..${END_SHA}" | tac | awk '{print $1}')
TOTAL=$(echo "$ALL_SHAS" | wc -l)

# Build set of commit SHAs that already have log directories in LOG_REPO
# Log dirs are named <ImageSHA>.<commitSHA> under LOG_REPO/logs/<date>/
ALREADY_BUILT=$(
    for date_dir in "$LOG_REPO"/logs/20*; do
        [ -d "$date_dir" ] || continue
        for d in "$date_dir"/*; do
            [ -d "$d" ] || continue
            basename "$d"
        done
    done | grep -oP '\.\K[a-f0-9]+$' | sort -u
)

NOT_RUN=()
for sha in $ALL_SHAS; do
    if ! echo "$ALREADY_BUILT" | grep -q "^${sha}$"; then
        NOT_RUN+=("$sha")
    fi
done

echo "Range: $START_SHA .. $END_SHA"
echo "Total commits: $TOTAL"
echo "Already passed: $((TOTAL - ${#NOT_RUN[@]}))"
echo "Need backfill: ${#NOT_RUN[@]}"
echo ""

if [[ ${#NOT_RUN[@]} -eq 0 ]]; then
    echo "Nothing to backfill."
    exit 0
fi

for sha in "${NOT_RUN[@]}"; do
    echo "  $sha  $(git log --oneline -1 "$sha" | cut -d' ' -f2-)"
done
echo ""

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] Would run ${#NOT_RUN[@]} builds on $BUILD_REPO"
    if [[ "$BUILD_REPO" != "$LOG_REPO" ]]; then
        echo "[dry-run] Logs would be moved to $LOG_REPO"
    fi
    exit 0
fi

PASSED_COUNT=0
FAILED_COUNT=0

for sha in "${NOT_RUN[@]}"; do
    echo "=== [$((PASSED_COUNT + FAILED_COUNT + 1))/${#NOT_RUN[@]}] $sha ==="
    if python3 "$BUILD_SCRIPT" \
        --repo-path "$BUILD_REPO" \
        --commit-sha "$sha" \
        --reuse-dev-if-image-exists \
        --parallel 2>&1; then

        echo "Exit: 0"
        PASSED_COUNT=$((PASSED_COUNT + 1))

        # Move logs to LOG_REPO if it differs from BUILD_REPO
        if [[ "$BUILD_REPO" != "$LOG_REPO" ]]; then
            for d in "$BUILD_REPO"/logs/2026-*/*."$sha"; do
                [ -d "$d" ] || continue
                date_dir=$(basename "$(dirname "$d")")
                mkdir -p "$LOG_REPO/logs/$date_dir"
                \mv -f "$d" "$LOG_REPO/logs/$date_dir/"
                echo "Moved $(basename "$d") -> $LOG_REPO/logs/$date_dir/"
            done
        fi
    else
        rc=$?
        echo "Exit: $rc"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
    echo ""
done

echo "=== SUMMARY ==="
echo "Build repo: $BUILD_REPO"
echo "Log repo:   $LOG_REPO"
echo "Passed: $PASSED_COUNT / ${#NOT_RUN[@]}"
echo "Failed: $FAILED_COUNT / ${#NOT_RUN[@]}"
