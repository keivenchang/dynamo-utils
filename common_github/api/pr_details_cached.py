# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PR details cached API (REST).

Resource:
  GET /repos/{owner}/{repo}/pulls/{pr_number}

Example API Response:
  {
    "number": 1234,
    "title": "Add new feature",
    "state": "open",
    "user": {
      "login": "contributor123"
    },
    "head": {
      "ref": "feature-branch",
      "sha": "abc123..."
    },
    "base": {
      "ref": "main"
    },
    "created_at": "2026-01-20T10:00:00Z",
    "updated_at": "2026-01-24T10:30:00Z",
    "merged_at": "2026-01-24T10:30:00Z",
    "mergeable": true,
    "mergeable_state": "clean"
  }

Cached Fields:
  - Full PR object (all fields)
  - Immutable after merge (long TTL for merged PRs)

Cache:
  Internal BaseDiskCache instance (disk + memory)

TTL:
  - Merged/closed PRs: 360 days (immutable)
  - Open PRs: 5 minutes

Usage:
  This cache is populated when get_pr_details() is called for individual PRs
  that are not found in pulls_list (typically older PRs beyond the first 500).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from cache.cache_base import BaseDiskCache
from .base_cached import CachedResourceBase

if TYPE_CHECKING:  # pragma: no cover
    from .. import GitHubAPIClient


TTL_POLICY_DESCRIPTION = "merged/closed: 360d; open: 5m"

CACHE_NAME = "pull_request"  # Match API label in logs
API_CALL_FORMAT = "REST GET /repos/{owner}/{repo}/pulls/{pr_number}"
CACHE_KEY_FORMAT = "{owner}/{repo}:pr:{pr_number}"
CACHE_FILE_DEFAULT = "pr_details.json"


# =============================================================================
# Cache Implementation (private to this module)
# =============================================================================

class _PRDetailsCache(BaseDiskCache):
    """Cache for individual PR details."""
    
    _SCHEMA_VERSION = 1
    
    def __init__(self, *, cache_file: Path):
        super().__init__(cache_file=cache_file, schema_version=self._SCHEMA_VERSION)
    
    def get_if_fresh(self, key: str, ttl_s: int) -> Optional[Dict[str, Any]]:
        """Get cached PR details if fresh."""
        with self._mu:
            self._load_once()
            ent = self._check_item(key)
            
            if not isinstance(ent, dict):
                return None
            
            # Check TTL
            ts = int(ent.get("ts", 0) or 0)
            now = int(time.time())
            
            # Merged/closed PRs are immutable - cache forever
            state = str(ent.get("state", "") or "").lower()
            merged_at = ent.get("merged_at")
            if state == "closed" or merged_at:
                return ent
            
            # Open PRs - use provided TTL
            if ts and (now - ts) <= max(0, int(ttl_s)):
                return ent
            
            return None
    
    def put(self, key: str, pr_data: Dict[str, Any]) -> None:
        """Store PR details."""
        now = int(time.time())
        with self._mu:
            self._load_once()
            entry = {
                "ts": now,
                "state": pr_data.get("state"),
                "merged_at": pr_data.get("merged_at"),
                "pr_data": pr_data,
            }
            self._set_item(key, entry)
            self._persist()


def _get_cache_file() -> Path:
    """Get cache file path."""
    try:
        _module_dir = Path(__file__).resolve().parent.parent
        if str(_module_dir) not in sys.path:
            sys.path.insert(0, str(_module_dir))
        import common
        return common.dynamo_utils_cache_dir() / CACHE_FILE_DEFAULT
    except ImportError:
        return Path.home() / ".cache" / "dynamo-utils" / CACHE_FILE_DEFAULT


# Module-level singleton cache instance (private)
_CACHE = _PRDetailsCache(cache_file=_get_cache_file())


# =============================================================================
# Public API
# =============================================================================

class PRDetailsCached(CachedResourceBase[Optional[Dict[str, Any]]]):
    def __init__(self, api: "GitHubAPIClient", *, ttl_s: int = 300):
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
        pr_number = int(kwargs["pr_number"])
        return f"{owner}/{repo}:pr:{pr_number}"

    def empty_value(self) -> Optional[Dict[str, Any]]:
        return None

    def cache_read(self, *, key: str) -> Optional[Dict[str, Any]]:
        return _CACHE.get_if_fresh(key, ttl_s=self._ttl_s)

    def value_from_cache_entry(self, *, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pr_data = entry.get("pr_data")
        return pr_data if isinstance(pr_data, dict) else None

    def is_cache_entry_fresh(self, *, entry: Dict[str, Any], now: int) -> bool:
        # Already checked in get_if_fresh, but implement for completeness
        state = str(entry.get("state", "") or "").lower()
        merged_at = entry.get("merged_at")
        
        # Merged/closed - always fresh
        if state == "closed" or merged_at:
            return True
        
        # Open - check TTL
        ts = int(entry.get("ts", 0) or 0)
        if not ts:
            return False
        return (now - ts) <= self._ttl_s

    def cache_write(self, *, key: str, value: Optional[Dict[str, Any]], meta: Dict[str, Any]) -> None:
        if value is None or not isinstance(value, dict):
            return
        _CACHE.put(key, value)

    def fetch(self, **kwargs: Any) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        owner = str(kwargs["owner"])
        repo = str(kwargs["repo"])
        pr_number = int(kwargs["pr_number"])
        
        endpoint = f"/repos/{owner}/{repo}/pulls/{pr_number}"
        try:
            pr_data = self.api.get(endpoint)
            return pr_data, {}
        except Exception:
            return None, {}


def get_pr_details_cached(api: "GitHubAPIClient", *, owner: str, repo: str, pr_number: int, ttl_s: int = 300) -> Optional[Dict[str, Any]]:
    """Get individual PR details with caching."""
    return PRDetailsCached(api, ttl_s=ttl_s).get(owner=owner, repo=repo, pr_number=pr_number)


def get_cache_sizes() -> Tuple[int, int]:
    """Get cache sizes for stats reporting (memory count, initial disk count)."""
    return _CACHE.get_cache_sizes()
