"""Cache for PR reviews.

Caching strategy:
  - Key: <owner>/<repo>#<pr_number>
  - Value: {ts, reviews: [list of review dicts from GitHub API]}
  - TTL: Short (reviews change frequently, but caching helps repeated runs)
  - ETag: EXCELLENT candidate - reviews rarely change between runs, 304 saves data transfer
  
NOTE: This stores raw GitHub API review responses.
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

from cache_base import BaseDiskCache


class PRReviewsCache(BaseDiskCache):
    """Cache for PR reviews (GitHub /pulls/{pr}/reviews endpoint).
    
    ETag Support:
        This cache stores ETags from GitHub API responses to enable conditional requests.
        When re-fetching, use get_etag() to get the cached ETag, send it in If-None-Match header,
        and GitHub will return 304 Not Modified if unchanged (doesn't count against rate limit).
    """
    
    _SCHEMA_VERSION = 1
    
    def __init__(self, *, cache_file: Path):
        super().__init__(cache_file=cache_file, schema_version=self._SCHEMA_VERSION)
    
    def get_if_fresh(self, key: str, ttl_s: int) -> Optional[Dict[str, Any]]:
        """Get cached PR reviews if fresh.
        
        Args:
            key: Cache key (e.g., "owner/repo#123")
            ttl_s: TTL in seconds
            
        Returns:
            Dict with 'reviews' and optionally 'etag' keys, or None if not cached/stale
        """
        with self._mu:
            self._load_once()
            ent = self._check_item(key)  # Automatically tracks hit/miss
            
            if not isinstance(ent, dict):
                return None
            
            ts = int(ent.get("ts", 0) or 0)
            now = int(time.time())
            
            if ts and (now - ts) <= max(0, int(ttl_s)):
                return ent
            
            return None
    
    def get_stale_etag(self, key: str) -> Optional[str]:
        """Get ETag from stale cache entry for conditional requests.
        
        Even if cache is stale (expired TTL), we can still use the ETag
        for a conditional request. If GitHub returns 304, we know the data
        hasn't changed and can refresh the timestamp without re-downloading.
        """
        return self.get_etag(key)
    
    def put(self, key: str, reviews: List[Dict[str, Any]], etag: Optional[str] = None) -> None:
        """Store PR reviews with optional ETag.
        
        Args:
            key: Cache key
            reviews: List of review objects from GitHub API
            etag: Optional ETag from response headers (for 304 Not Modified support)
        """
        now = int(time.time())
        with self._mu:
            self._load_once()
            entry = {
                "ts": now,
                "reviews": reviews,
            }
            if etag:
                entry["etag"] = etag.strip()
            self._set_item(key, entry)  # Automatically tracks write
            self._persist()
    
    def refresh_timestamp(self, key: str) -> None:
        """Refresh timestamp on cache entry (for 304 Not Modified responses).
        
        When GitHub returns 304, the data hasn't changed, so we just update
        the timestamp to extend the TTL without re-storing the data.
        """
        with self._mu:
            self._load_once()
            items = self._get_items()
            entry = items.get(key)
            
            if isinstance(entry, dict):
                entry["ts"] = int(time.time())
                self._set_item(key, entry)
                self._persist()


# Singleton cache instance
def _get_cache_file() -> Path:
    """Get cache file path, handling imports from different contexts."""
    try:
        parent_dir = _module_dir.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        import common
        return common.dynamo_utils_cache_dir() / "pr_reviews.json"
    except ImportError:
        return Path.home() / ".cache" / "dynamo-utils" / "pr_reviews.json"


PR_REVIEWS_CACHE = PRReviewsCache(cache_file=_get_cache_file())
