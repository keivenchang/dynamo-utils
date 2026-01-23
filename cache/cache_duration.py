#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Job Duration Cache

Caches duration calculations from:
1. Raw GitHub Actions log files (parsed timestamps)
2. GitHub Actions API job responses (start/completion times)

Both are immutable once calculated, so no TTL is needed.

Cache file: ~/.cache/dynamo-utils/duration-cache.json

Cache key format:
- Raw log: "raw:<mtime_ns>:<size_bytes>:<filename>"
- API job: "job:<job_id>"
"""

from __future__ import annotations

import sys
from pathlib import Path as PathLib

# Ensure parent directory is in sys.path for imports
_this_dir = PathLib(__file__).resolve().parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))
_parent_dir = _this_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from cache_base import BaseDiskCache
from common import dynamo_utils_cache_dir


class DurationCache(BaseDiskCache):
    """Disk-backed cache for job duration calculations with automatic persistence.
    
    Stats (hit/miss/write) are tracked automatically by BaseDiskCache.
    """

    def __init__(self, *, cache_file: Path):
        super().__init__(cache_file=cache_file, schema_version=1)
        # self.stats inherited from BaseDiskCache

    def _raw_log_key(self, *, raw_log_path: Path) -> str:
        """Generate cache key for raw log duration."""
        try:
            st = raw_log_path.stat()
            return f"raw:{st.st_mtime_ns}:{st.st_size}:{raw_log_path.name}"
        except Exception:
            return f"raw:0:0:{raw_log_path.name}"

    def _job_url_key(self, *, job_id: int) -> str:
        """Generate cache key for API job duration."""
        return f"job:{job_id}"

    def get_raw_log_duration(self, *, raw_log_path: Path) -> Optional[str]:
        """
        Get cached duration for a raw log file.

        Args:
            raw_log_path: Path to raw log file

        Returns:
            Cached duration string (e.g. "5m 32s") or None if not found
            
        Note: Stats (hit/miss) are tracked automatically by _check_item()
        """
        with self._mu:
            self._load_once()
            key = self._raw_log_key(raw_log_path=raw_log_path)
            return self._check_item(key)  # Automatically tracks hit/miss

    def put_raw_log_duration(self, *, raw_log_path: Path, duration: str) -> None:
        """
        Store duration for a raw log file.

        Args:
            raw_log_path: Path to raw log file
            duration: Duration string (e.g. "5m 32s")
            
        Note: Stats (write) are tracked automatically by _set_item()
        """
        with self._mu:
            self._load_once()
            key = self._raw_log_key(raw_log_path=raw_log_path)
            self._set_item(key, duration)  # Automatically tracks write
            self._persist()

    def get_job_duration(self, *, job_id: int) -> Optional[str]:
        """
        Get cached duration for a job (from API).

        Args:
            job_id: GitHub Actions job ID

        Returns:
            Cached duration string (e.g. "5m 32s") or None if not found
            
        Note: Stats (hit/miss) are tracked automatically by _check_item()
        """
        with self._mu:
            self._load_once()
            key = self._job_url_key(job_id=job_id)
            return self._check_item(key)  # Automatically tracks hit/miss

    def put_job_duration(self, *, job_id: int, duration: str) -> None:
        """
        Store duration for a job (from API).

        Args:
            job_id: GitHub Actions job ID
            duration: Duration string (e.g. "5m 32s")
            
        Note: Stats (write) are tracked automatically by _set_item()
        """
        with self._mu:
            self._load_once()
            key = self._job_url_key(job_id=job_id)
            self._set_item(key, duration)  # Automatically tracks write
            self._persist()


# Singleton cache used by all dashboard generators
DURATION_CACHE = DurationCache(
    cache_file=(dynamo_utils_cache_dir() / "duration-cache.json")
)
