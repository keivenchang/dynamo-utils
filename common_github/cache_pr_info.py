"""Cache for enriched PR information objects.

Caching strategy:
  - Key: <owner>/<repo>#pr:<pr_number>
  - Value: Dict with 'updated_at' (timestamp) and 'pr' (serialized PRInfo object)
  - Invalidation: By 'updated_at' timestamp (if PR unchanged, reuse cache)
  
  Also includes a separate cache for PR head SHA lookups:
  - Key: <owner>/<repo>#pr:<pr_number>
  - Value: Dict with 'ts', 'head_sha', 'state'
  - Tiered TTL (closed/merged PRs cached forever)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure imports work from both html_pages/ and parent directory
_module_dir = Path(__file__).resolve().parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

from cache.cache_base import BaseDiskCache


class PRInfoCache(BaseDiskCache):
    """Cache for enriched PR information.
    
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


class PRHeadSHACache(BaseDiskCache):
    """Cache for PR head SHA lookups (lighter weight than full PR info).
    
    Stores just the head SHA and state for quick lookups without fetching
    the full PR object.
    
    Stats (hit/miss/write) are tracked automatically by BaseDiskCache.
    """
    
    _SCHEMA_VERSION = 1
    _DEFAULT_TTL = 60 * 60  # 1 hour for open PRs
    
    def __init__(self, *, cache_file: Path):
        super().__init__(cache_file=cache_file, schema_version=self._SCHEMA_VERSION)
    
    def get_if_fresh(self, key: str, ttl_s: int) -> Optional[Dict[str, Any]]:
        """Get cached head SHA if fresh.
        
        Args:
            key: Cache key (e.g., "owner/repo#pr:123")
            ttl_s: TTL in seconds (ignored for closed/merged PRs)
            
        Returns:
            Dict with 'head_sha' and 'state', or None if not cached/stale
        """
        with self._mu:
            self._load_once()
            ent = self._check_item(key)  # Automatically tracks hit/miss
            
            if not isinstance(ent, dict):
                return None
            
            state = str(ent.get("state", "") or "").strip().lower()
            
            # Closed/merged PRs are cached forever
            if state in ("closed", "merged"):
                return {
                    "head_sha": ent.get("head_sha"),
                    "state": ent.get("state"),
                }
            
            # For open PRs, check TTL
            ts = int(ent.get("ts", 0) or 0)
            now = int(time.time())
            
            if ts and (now - ts) <= max(0, int(ttl_s)):
                return {
                    "head_sha": ent.get("head_sha"),
                    "state": ent.get("state"),
                }
            
            return None
    
    def put(self, key: str, head_sha: str, state: str) -> None:
        """Store head SHA.
        
        Args:
            key: Cache key
            head_sha: Commit SHA
            state: PR state (open/closed/merged)
        """
        now = int(time.time())
        with self._mu:
            self._load_once()
            entry = {
                "ts": now,
                "head_sha": head_sha,
                "state": state,
            }
            self._set_item(key, entry)  # Automatically tracks write
            self._persist()


# Singleton cache instances
def _get_cache_dir() -> Path:
    """Get cache directory, handling imports from different contexts."""
    try:
        parent_dir = _module_dir.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        import common
        return common.dynamo_utils_cache_dir() / "pr-info"
    except ImportError:
        return Path.home() / ".cache" / "dynamo-utils" / "pr-info"


PR_INFO_CACHE = PRInfoCache(cache_file=_get_cache_dir() / "pr_info.json")
PR_HEAD_SHA_CACHE = PRHeadSHACache(cache_file=_get_cache_dir() / "pr_head_sha.json")
