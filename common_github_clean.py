#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GitHub API utilities for Dynamo.

This module provides:
- GitHubAPIClient for interacting with GitHub REST API
- Caching utilities for PR checks, check runs, and workflow jobs
- Utility functions for parsing GitHub URLs and check runs
"""

import json
import logging
import os
import re
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests
import yaml

# Import from common modules
from common_types import CIStatus
from common import (
    DEFAULT_STABLE_AFTER_HOURS,
    DEFAULT_UNSTABLE_TTL_S,
    DEFAULT_STABLE_TTL_S,
    DEFAULT_OPEN_PRS_TTL_S,
    DEFAULT_CLOSED_PRS_TTL_S,
    DEFAULT_NO_PR_TTL_S,
    DEFAULT_RAW_LOG_URL_TTL_S,
    DEFAULT_RAW_LOG_TEXT_TTL_S,
    DEFAULT_RAW_LOG_TEXT_MAX_BYTES,
    DEFAULT_RAW_LOG_ERROR_SNIPPET_TAIL_BYTES,
    dynamo_utils_cache_dir,
    resolve_cache_path,
    PRInfo,
)

# Module logger
_logger = logging.getLogger(__name__)

# ==============================================================================
# API inventory (where the dashboard data comes from)
#
# The dashboards in this repo are built from a mix of:
# - Local git metadata (branch/commit subject/SHA/time) from GitPython
# - GitHub REST v3 (https://api.github.com) for PRs, check-runs, Actions job logs
#
# GitHub REST endpoints used (core):
# - PR lookup / branch→PR mapping:
#   - GET /repos/{owner}/{repo}/pulls                       (paged; open PR list)
#   - GET /repos/{owner}/{repo}/pulls/{pr_number}           (head sha, base ref, mergeable_state, etc.)
#   - GET /repos/{owner}/{repo}/commits/{sha}/pulls         (best-effort: find PR number from commit SHA)
#
# - Checks / CI rows (used to render the "Details" tree):
#   - GET /repos/{owner}/{repo}/commits/{sha}/check-runs     (status+conclusion+timestamps+URLs per check)
#   - GET /repos/{owner}/{repo}/commits/{sha}/status         (legacy fallback; coarse success/failure/pending)
#
# - "raw log" links + snippet inputs (GitHub Actions jobs):
#   - GET /repos/{owner}/{repo}/actions/jobs/{job_id}/logs
#       - when called with redirects disabled: capture Location header → direct "[raw log]" link (short-lived)
#       - when downloaded: returns a ZIP of log files → extract text → cache for snippet parsing
#
# - PR comments (very rough "unresolved conversations" approximation):
#   - GET /repos/{owner}/{repo}/pulls/{pr_number}/comments
#
# Optional / best-effort (may require elevated permissions):
# - Required status checks (branch protection):
#   - GET /repos/{owner}/{repo}/branches/{base_ref}/protection/required_status_checks
#     (often 403 unless token has admin permissions; if unavailable we simply don't mark "required")
# ==============================================================================
