# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cached enriched PR information resource.

This module manages caching for enriched PRInfo objects.

Resource:
  Multiple API calls combined into enriched PRInfo:
  - GET /repos/{owner}/{repo}/pulls/{pr_number}
  - GET /repos/{owner}/{repo}/pulls/{pr_number}/reviews
  - GET /repos/{owner}/{repo}/commits/{sha}/check-runs
  - GET /repos/{owner}/{repo}/commits/{sha}/status

Example Enriched PRInfo Structure:
  {
    "number": 1234,
    "title": "Add new feature",
    "state": "open",
    "updated_at": "2026-01-24T10:30:00Z",
    "head_sha": "abc123def456...",
    "base_ref": "main",
    "head_ref": "feature-branch",
    "user": "contributor123",
    "reviews": [...],  # From reviews API
    "checks": [...]    # From check-runs/status APIs
  }

Caching strategy:
  - Key format: <owner>/<repo>#pr:<pr_number>
  - Value: Dict with 'updated_at' (timestamp) and 'pr' (serialized PRInfo object)
  - Invalidation: By 'updated_at' timestamp (if PR unchanged, reuse cache)

TTL Policy:
  - Semantic versioning: cache is fresh if cached 'updated_at' matches current PR 'updated_at'
  - No time-based TTL - relies on GitHub's PR update timestamp

Architecture:
  This module follows the common_github/api/ pattern where each API resource
  has its own dedicated cache implementation (private BaseDiskCache subclass)
  and exposes a CachedResourceBase subclass for unified statistics tracking.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

# Ensure imports work from both html_pages/ and parent directory
_module_dir = Path(__file__).resolve().parent
_parent_dir = _module_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))
if str(_parent_dir.parent) not in sys.path:
    sys.path.insert(0, str(_parent_dir.parent))

from cache.cache_base import BaseDiskCache
from .base_cached import CachedResourceBase

if TYPE_CHECKING:  # pragma: no cover
    from .. import GitHubAPIClient


TTL_POLICY_DESCRIPTION = "by updated_at timestamp (invalidated when PR changes)"


# =============================================================================
# Cache Implementation (private to this module)
# =============================================================================

class _PRInfoCache(BaseDiskCache):
    """Private cache for enriched PR information.
    
    Stores full PRInfo objects keyed by (owner, repo, pr_number).
    The updated_at serves as a natural cache invalidation key - if the PR
    hasn't been updated since we cached it, we can reuse the cached enrichment.
    """
    
    _SCHEMA_VERSION = 1
    
    def __init__(self, *, cache_file: Path):
        super().__init__(cache_file=cache_file, schema_version=self._SCHEMA_VERSION)
    
    def get_if_matches_updated_at(self, key: str, updated_at: str) -> Optional[Dict[str, Any]]:
        """Get cached PR info entry if updated_at matches.
        
        Args:
            key: Cache key (e.g., "owner/repo#pr:123")
            updated_at: Expected updated_at timestamp
            
        Returns:
            Cached entry dict (with 'updated_at' and 'pr'), or None if not cached/stale
        """
        with self._mu:
            self._load_once()
            items = self._get_items()
            ent = items.get(key)
            
            if not isinstance(ent, dict):
                return None
            
            cached_updated_at = str(ent.get("updated_at", "") or "").strip()
            if cached_updated_at == str(updated_at or "").strip():
                pr_dict = ent.get("pr")
                if isinstance(pr_dict, dict):
                    # IMPORTANT: callers expect the full entry dict so they can hydrate from ent["pr"].
                    return ent
            
            # Entry exists but updated_at doesn't match - this is a miss
            return None
    
    def put(self, key: str, updated_at: str, pr_dict: Dict[str, Any]) -> None:
        """Store PR info.
        
        Args:
            key: Cache key
            updated_at: PR updated_at timestamp
            pr_dict: Serialized PR object
        """
        with self._mu:
            self._load_once()
            entry = {
                "updated_at": updated_at,
                "pr": pr_dict,
            }
            self._set_item(key, entry)
            self._persist()


def _get_cache_file() -> Path:
    """Get cache file path, handling imports from different contexts."""
    try:
        import common
        cache_dir = common.dynamo_utils_cache_dir() / "pr-info"
        return cache_dir / "pr_info.json"
    except ImportError:
        return Path.home() / ".cache" / "dynamo-utils" / "pr-info" / "pr_info.json"


# Private module-level cache instance (singleton)
_CACHE = _PRInfoCache(cache_file=_get_cache_file())


# =============================================================================
# Public API using CachedResourceBase
# =============================================================================

class PRInfoCached(CachedResourceBase[Optional[Dict[str, Any]]]):
    """Cached resource for enriched PR info with automatic statistics tracking.
    
    This wraps the _PRInfoCache to provide consistent statistics tracking
    through the CachedResourceBase framework.
    """
    
    def __init__(self, api: "GitHubAPIClient", *, updated_at: str):
        super().__init__(api)
        self._updated_at = str(updated_at or "").strip()
    
    @property
    def cache_name(self) -> str:
        return "pr_info"
    
    def api_call_format(self) -> str:
        return "Multiple APIs (PR details + reviews + checks)"
    
    def cache_key(self, **kwargs: Any) -> str:
        owner = str(kwargs["owner"])
        repo = str(kwargs["repo"])
        pr_number = int(kwargs["pr_number"])
        return f"{owner}/{repo}#pr:{pr_number}"
    
    def empty_value(self) -> Optional[Dict[str, Any]]:
        return None
    
    def cache_read(self, *, key: str) -> Optional[Dict[str, Any]]:
        """Read from cache only if updated_at matches."""
        return _CACHE.get_if_matches_updated_at(key, self._updated_at)
    
    def value_from_cache_entry(self, *, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract the full entry (callers expect both 'updated_at' and 'pr' fields)."""
        # Return the entire entry so callers can access entry["pr"] for hydration
        return entry
    
    def is_cache_entry_fresh(self, *, entry: Dict[str, Any], now: int) -> bool:
        """Check if cached updated_at matches expected updated_at."""
        cached_updated_at = str(entry.get("updated_at", "") or "").strip()
        return cached_updated_at == self._updated_at
    
    def cache_write(self, *, key: str, value: Optional[Dict[str, Any]], meta: Dict[str, Any]) -> None:
        """Write to cache with updated_at timestamp."""
        if value is None:
            return
        pr_dict = value.get("pr") if isinstance(value, dict) else value
        if not isinstance(pr_dict, dict):
            return
        _CACHE.put(key, self._updated_at, pr_dict)
    
    def fetch(self, **kwargs: Any) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Fetch is not implemented - pr_info enrichment happens in __init__.py."""
        # This is never called because pr_info fetching/enrichment is complex
        # and happens inline in __init__.py's get_pr_info methods.
        # The cache is only used for lookup/storage.
        return None, {}


def get_cache_sizes() -> Tuple[int, int]:
    """Get memory and disk cache sizes for cache statistics.
    
    Returns:
        Tuple of (memory_count, disk_count_before_run)
    """
    return _CACHE.get_cache_sizes()


# Backward-compatible API for __init__.py (these delegate to CachedResourceBase now)
def get_if_matches_updated_at(api: "GitHubAPIClient", *, key: str, updated_at: str) -> Optional[Dict[str, Any]]:
    """Get cached PR info entry if updated_at matches.
    
    This function wraps CachedResourceBase.get() to maintain backward compatibility
    with the existing __init__.py code while gaining automatic statistics tracking.
    """
    # Extract owner/repo/pr_number from key for CachedResourceBase.get()
    # Key format: "owner/repo#pr:123"
    try:
        parts = key.split("#pr:")
        if len(parts) != 2:
            return None
        owner_repo = parts[0]
        pr_number = int(parts[1])
        if "/" not in owner_repo:
            return None
        owner, repo = owner_repo.split("/", 1)
    except (ValueError, IndexError):
        return None
    
    # Use CachedResourceBase.get() for automatic stats tracking
    cached = PRInfoCached(api, updated_at=updated_at)
    # Direct cache read to avoid fetch (pr_info enrichment happens in __init__.py)
    entry = cached.cache_read(key=key)
    if entry is not None:
        if cached.is_cache_entry_fresh(entry=entry, now=0):  # now unused for updated_at comparison
            api._cache_hit(cached.cache_name)
            return entry
    api._cache_miss(cached.cache_name)
    return None


def put(api: "GitHubAPIClient", *, key: str, updated_at: str, pr_dict: Dict[str, Any]) -> None:
    """Store PR info with automatic statistics tracking."""
    _CACHE.put(key, updated_at, pr_dict)
    api._cache_write("pr_info", entries=1)
