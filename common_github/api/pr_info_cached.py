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

Architecture:
  This module follows the common_github/api/ pattern where each API resource
  has its own dedicated cache implementation (private BaseDiskCache subclass)
  rather than sharing a global cache instance.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Ensure imports work from both html_pages/ and parent directory
_module_dir = Path(__file__).resolve().parent
_parent_dir = _module_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))
if str(_parent_dir.parent) not in sys.path:
    sys.path.insert(0, str(_parent_dir.parent))

from cache.cache_base import BaseDiskCache


class _PRInfoCache(BaseDiskCache):
    """Private cache for enriched PR information.
    
    Stores full PRInfo objects keyed by (owner, repo, pr_number, updated_at).
    The updated_at serves as a natural cache invalidation key - if the PR
    hasn't been updated since we cached it, we can reuse the cached enrichment.
    
    Stats (hit/miss/write) are tracked automatically by BaseDiskCache.
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
            ent = items.get(key)  # Don't use _check_item() - we'll track stats manually
            
            if not isinstance(ent, dict):
                self.stats.miss += 1
                return None
            
            cached_updated_at = str(ent.get("updated_at", "") or "").strip()
            if cached_updated_at == str(updated_at or "").strip():
                pr_dict = ent.get("pr")
                if isinstance(pr_dict, dict):
                    self.stats.hit += 1
                    # IMPORTANT: callers expect the full entry dict so they can hydrate from ent["pr"].
                    return ent
            
            # Entry exists but updated_at doesn't match - this is a miss
            self.stats.miss += 1
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
            self._set_item(key, entry)  # Automatically tracks write
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


def get_cache_sizes() -> Tuple[int, int]:
    """Get memory and disk cache sizes for cache statistics.
    
    Returns:
        Tuple of (memory_size_bytes, disk_size_bytes)
    """
    return _CACHE.get_cache_sizes()


# Public API for __init__.py to access the cache
def get_if_matches_updated_at(key: str, updated_at: str) -> Optional[Dict[str, Any]]:
    """Get cached PR info entry if updated_at matches."""
    return _CACHE.get_if_matches_updated_at(key, updated_at)


def put(key: str, updated_at: str, pr_dict: Dict[str, Any]) -> None:
    """Store PR info."""
    _CACHE.put(key, updated_at, pr_dict)
