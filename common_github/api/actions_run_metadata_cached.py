# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cached GitHub Actions workflow run metadata (indexed by run_id).

This module manages caching for GitHub Actions workflow run metadata fetched via:
  GET /repos/{owner}/{repo}/actions/runs/{run_id}

Primary use case: Fetch run-level metadata to detect reruns via run_attempt field.

Example API Response:
  {
    "id": 21507141526,
    "run_number": 1467,
    "run_attempt": 2,
    "status": "in_progress",
    "conclusion": null,
    "created_at": "2026-01-30T06:48:50Z",
    "updated_at": "2026-01-30T17:05:56Z",
    "run_started_at": "2026-01-30T06:48:51Z",
    "html_url": "https://github.com/owner/repo/actions/runs/21507141526",
    "previous_attempt_url": "https://github.com/owner/repo/actions/runs/21507141526/attempts/1"
  }

Cached Fields:
  - run_attempt: Integer indicating rerun count (1 = original, 2+ = rerun)
  - run_number: Workflow run number (unique within workflow)
  - status: "queued", "in_progress", "completed"
  - conclusion: "success", "failure", "cancelled", "skipped", null
  - created_at: ISO timestamp when run was created
  - updated_at: ISO timestamp when run was last updated
  - run_started_at: ISO timestamp when run started executing
  - previous_attempt_url: URL to previous attempt (if run_attempt > 1)

Caching strategy:
  - Key format: <owner>/<repo>:run_metadata:<run_id>
  - Value: {ts, metadata: {...}, etag}
  - TTL: ADAPTIVE based on completion status
    * Completed runs: 30 days (immutable)
    * In-progress runs: 1-5 minutes (min 60s, max 300s)
  - ETag support: 100% 304 rate for completed runs

Purpose: Enable smart rerun detection by checking run_attempt field.
  - run_attempt == 1: Original run
  - run_attempt > 1: Rerun (show badge, hide superseded attempts)

Architecture:
  Follows the common_github/api/ pattern with dedicated BaseDiskCache subclass.
"""
from __future__ import annotations

import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    from common_github import GitHubAPIClient

# Add parent to sys.path for imports
_parent = Path(__file__).resolve().parent.parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from cache.cache_base import BaseDiskCache
from common_github.api.base_cached import CachedResourceBase

TTL_POLICY_DESCRIPTION = (
    "adaptive: completed=30d (immutable), in_progress=1-5m (min 60s, max 300s)"
)

CACHE_NAME = "actions_run_metadata"
API_CALL_FORMAT = "REST GET /repos/{owner}/{repo}/actions/runs/{run_id} (etag)"
CACHE_KEY_FORMAT = "{owner}/{repo}:run_metadata:{run_id}"
CACHE_FILE_DEFAULT = "actions_run_metadata.json"


class _ActionsRunMetadataCache(BaseDiskCache):
    """Private cache for GitHub Actions workflow run metadata."""
    
    _SCHEMA_VERSION = 1
    _DEFAULT_TTL = 30 * 24 * 3600  # 30 days (completed runs never change)
    
    def __init__(self, *, cache_file: Path):
        super().__init__(cache_file=cache_file, schema_version=self._SCHEMA_VERSION)
    
    def get_if_fresh(self, key: str, ttl_s: int) -> Optional[Dict[str, Any]]:
        """Get cached run metadata if fresh.
        
        Uses adaptive TTL:
        - Completed runs: 30 days (immutable)
        - In-progress runs: 1-5 minutes (min 60s, max 300s)
        
        Args:
            key: Cache key (e.g., "owner/repo:run_metadata:12345")
            ttl_s: TTL in seconds for in-progress runs
            
        Returns:
            Dict with 'metadata', 'ts', and optionally 'etag' keys, or None if not cached/stale
        """
        with self._mu:
            self._load_once()
            ent = self._check_item(key)  # Automatically tracks hit/miss
            
            if not isinstance(ent, dict):
                return None
            
            metadata = ent.get("metadata")
            if not isinstance(metadata, dict):
                return None
            
            # Check if run is completed
            status = str(metadata.get("status", "")).strip().lower()
            if status == "completed":
                # Completed runs are immutable - cache forever
                return ent
            
            # In-progress/queued runs - use provided TTL (min 60s, max 5 minutes)
            ts = int(ent.get("ts", 0) or 0)
            now = int(time.time())
            
            if ts:
                effective_ttl = max(60, min(int(ttl_s), 300))  # Min 1 minute, max 5 minutes
                if (now - ts) <= effective_ttl:
                    return ent
            
            return None
    
    def get_stale_etag(self, key: str) -> Optional[str]:
        """Get ETag from stale cache entry for conditional requests."""
        return self.get_etag(key)
    
    def put(self, key: str, metadata: Dict[str, Any], etag: Optional[str] = None) -> None:
        """Store workflow run metadata with optional ETag.
        
        Args:
            key: Cache key (e.g., "owner/repo:run_metadata:12345")
            metadata: Run metadata dict
            etag: Optional ETag from response headers
        """
        now = int(time.time())
        with self._mu:
            self._load_once()
            entry = {
                "ts": now,
                "metadata": metadata,
            }
            if etag:
                entry["etag"] = etag.strip()
            self._set_item(key, entry)  # Automatically tracks write
            self._persist()
    
    def refresh_timestamp(self, key: str) -> None:
        """Refresh timestamp on cache entry (for 304 Not Modified responses)."""
        with self._mu:
            self._load_once()
            items = self._get_items()
            entry = items.get(key)
            
            if isinstance(entry, dict):
                entry["ts"] = int(time.time())
                self._set_item(key, entry)
                self._persist()


def _get_cache_file() -> Path:
    """Get cache file path, handling imports from different contexts."""
    try:
        import common
        return common.dynamo_utils_cache_dir() / CACHE_FILE_DEFAULT
    except ImportError:
        return Path.home() / ".cache" / "dynamo-utils" / CACHE_FILE_DEFAULT


# Private module-level cache instance (singleton)
_CACHE = _ActionsRunMetadataCache(cache_file=_get_cache_file())


# =============================================================================
# CachedResourceBase Implementation
# =============================================================================


class ActionsRunMetadataCached(CachedResourceBase[Optional[Dict[str, Any]]]):
    """Cached resource for querying workflow run metadata."""
    
    def __init__(self, api: "GitHubAPIClient", *, ttl_s: int):
        super().__init__(api)
        self._ttl_s = int(ttl_s)

    @property
    def cache_name(self) -> str:
        return CACHE_NAME
    
    def api_call_format(self) -> str:
        return API_CALL_FORMAT
    
    def cache_key(self, **kwargs: Any) -> str:
        owner = str(kwargs["owner"])
        repo = str(kwargs["repo"])
        run_id = str(kwargs["run_id"])
        return f"{owner}/{repo}:run_metadata:{run_id}"
    
    def inflight_lock_key(self, **kwargs: Any) -> Optional[str]:
        # Deduplicate concurrent identical run metadata fetches
        return self.cache_key(**kwargs)
    
    def empty_value(self) -> Optional[Dict[str, Any]]:
        return None
    
    def cache_read(self, *, key: str) -> Optional[Dict[str, Any]]:
        """Read from cache using adaptive TTL."""
        return _CACHE.get_if_fresh(key, ttl_s=int(self._ttl_s))
    
    def value_from_cache_entry(self, *, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract run metadata from cache entry."""
        metadata = entry.get("metadata")
        if isinstance(metadata, dict):
            return metadata
        return None
    
    def is_cache_entry_fresh(self, *, entry: Dict[str, Any], now: int) -> bool:
        """_CACHE.get_if_fresh already applied TTL and adaptive logic."""
        return True
    
    def cache_write(self, *, key: str, value: Optional[Dict[str, Any]], meta: Dict[str, Any]) -> None:
        """Write to cache."""
        if value is None:
            return
        
        etag = meta.get("etag")
        _CACHE.put(key, metadata=value, etag=etag)
    
    def fetch(self, **kwargs: Any) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Fetch workflow run metadata from GitHub API."""
        owner = str(kwargs["owner"])
        repo = str(kwargs["repo"])
        run_id = str(kwargs["run_id"])
        
        url = f"{self.api.base_url}/repos/{owner}/{repo}/actions/runs/{run_id}"
        
        try:
            resp = self.api._rest_get(url, timeout=15, use_etag=True)
            
            if resp.status_code < 200 or resp.status_code >= 300:
                return None, {}
            
            data = resp.json() if hasattr(resp, "json") else {}
            
            # Extract relevant fields
            metadata = {
                "run_attempt": int(data.get("run_attempt", 1)),
                "run_number": int(data.get("run_number", 0)),
                "status": str(data.get("status", "")),
                "conclusion": data.get("conclusion"),  # Can be None
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "run_started_at": data.get("run_started_at"),
                "previous_attempt_url": data.get("previous_attempt_url"),
                "html_url": data.get("html_url"),
            }
            
            # Get ETag from response headers if available
            etag = None
            if hasattr(resp, "headers"):
                etag = resp.headers.get("ETag")
            
            return metadata, {"etag": etag}
        except Exception as e:
            # Log but don't fail - return None to indicate unavailable
            import logging
            logging.getLogger(__name__).warning(
                f"Failed to fetch run metadata for {run_id}: {e}"
            )
            return None, {}


# =============================================================================
# Public API
# =============================================================================


def get_cache_sizes() -> Tuple[int, int]:
    """Get memory and disk cache sizes for cache statistics.
    
    Returns:
        Tuple of (memory_count, disk_count_before_run)
    """
    return _CACHE.get_cache_sizes()


def get_run_metadata_cached(
    api: "GitHubAPIClient",
    owner: str,
    repo: str,
    run_id: str,
    ttl_s: int = 300,
    skip_fetch: bool = False,
) -> Optional[Dict[str, Any]]:
    """Get workflow run metadata with caching.
    
    Args:
        api: GitHubAPIClient instance
        owner: Repository owner
        repo: Repository name
        run_id: Workflow run ID
        ttl_s: Cache TTL in seconds (adaptive based on status)
        skip_fetch: If True, only return cached data (no network fetch)
    
    Returns:
        Dict with run metadata (run_attempt, status, etc.) or None if unavailable
    """
    if not run_id or not str(run_id).strip():
        return None
    
    cache_only = skip_fetch or bool(api.cache_only_mode)
    
    return ActionsRunMetadataCached(api, ttl_s=ttl_s).get(
        cache_only_mode=cache_only,
        owner=owner,
        repo=repo,
        run_id=str(run_id),
    )
