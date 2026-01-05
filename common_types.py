#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Common shared enums/types that must be used by both:
- `common.py` (API/data layer)
- `html_pages/*` dashboard renderers

This module MUST NOT import `common.py` or any html_pages modules to avoid cycles.
"""

from __future__ import annotations

from enum import Enum


class CIStatus(str, Enum):
    """Canonical normalized status strings used across dashboards and API normalization."""

    SUCCESS = "success"
    FAILURE = "failure"
    IN_PROGRESS = "in_progress"
    PENDING = "pending"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class MarkerStatus(str, Enum):
    """Marker suffix values for build/log status marker files."""

    RUNNING = "RUNNING"
    PASSED = "PASSED"
    FAILED = "FAILED"
    KILLED = "KILLED"


