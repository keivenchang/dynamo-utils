"""Cache for GitHub search/issues API results.

Caching strategy:
  - Key: <owner>/<repo>:search_issues:<pr_numbers_csv>
  - Value: Dict with 'ts', 'results' (list of PR objects with updated_at)
  - TTL-based invalidation
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure imports work from both html_pages/ and parent directory
_module_dir = Path(__file__).resolve().parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

from cache.cache_base import BaseDiskCache


class SearchIssuesCache(BaseDiskCache):
    """Cache for GitHub search/issues API results.
    
    Stores bulk PR metadata from search queries to detect which PRs
    have been updated and need re-enrichment.
    
    Stats (hit/miss/write) are tracked automatically by BaseDiskCache.
    """
    
    _SCHEMA_VERSION = 1
    _DEFAULT_TTL = 60 * 60  # 1 hour default
    
    def __init__(self, *, cache_file: Path):
        super().__init__(cache_file=cache_file, schema_version=self._SCHEMA_VERSION)
    
    def get_if_fresh(self, key: str, ttl_s: int) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results if fresh.
        
        Args:
            key: Cache key (e.g., "owner/repo:search_issues:123,456")
            ttl_s: TTL in seconds
            
        Returns:
            List of PR objects, or None if not cached/stale
        """
        with self._mu:
            self._load_once()
            ent = self._check_item(key)  # Automatically tracks hit/miss
            
            if not isinstance(ent, dict):
                return None
            
            ts = int(ent.get("ts", 0) or 0)
            now = int(time.time())
            
            if ts and (now - ts) <= max(0, int(ttl_s)):
                results = ent.get("results")
                if isinstance(results, list):
                    return results
            
            return None
    
    def put(self, key: str, results: List[Dict[str, Any]]) -> None:
        """Store search results.
        
        Args:
            key: Cache key
            results: List of PR objects
        """
        now = int(time.time())
        with self._mu:
            self._load_once()
            entry = {
                "ts": now,
                "results": results,
            }
            self._set_item(key, entry)  # Automatically tracks write
            self._persist()


# Singleton cache instance
def _get_cache_file() -> Path:
    """Get cache file path, handling imports from different contexts."""
    try:
        parent_dir = _module_dir.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        import common
        return common.dynamo_utils_cache_dir() / "search_issues.json"
    except ImportError:
        return Path.home() / ".cache" / "dynamo-utils" / "search_issues.json"


SEARCH_ISSUES_CACHE = SearchIssuesCache(cache_file=_get_cache_file())
