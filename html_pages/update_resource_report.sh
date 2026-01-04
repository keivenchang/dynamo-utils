#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# update_resource_report.sh
#
# Thin wrapper around:
#   update_html_pages.sh --show-local-resources
#
# Cron Example:
#   * * * * * NVIDIA_HOME=$HOME/nvidia /path/to/update_resource_report.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/update_html_pages.sh" --show-local-resources


