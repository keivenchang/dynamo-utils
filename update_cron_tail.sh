#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Update cron-tail.txt with the last 100 lines of cron.log

CRON_LOG="~/nvidia/dynamo_ci/logs/cron.log"
OUTPUT_FILE="~/nvidia/dynamo_ci/logs/cron-tail.txt"

if [ -f "$CRON_LOG" ]; then
    tail -100 "$CRON_LOG" > "$OUTPUT_FILE"
else
    echo "cron.log not found at $CRON_LOG" > "$OUTPUT_FILE"
fi
