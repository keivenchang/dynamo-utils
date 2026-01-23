"""Cache for commit merge dates (when a commit was merged to main branch).

Caching strategy:
  - Key: <owner>/<repo>:<commit_sha>
  - Value: Dict with 'ts', 'merge_date' (ISO timestamp or None)
  - Long TTL (merge dates never change once set)
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

from cache_base import BaseDiskCache


class MergeDatesCache(BaseDiskCache):
    """Cache for commit merge dates.
    
    Stores when commits were merged to the main branch. Once a commit
    is merged, the date never changes, so we can cache aggressively.
    
    Stats (hit/miss/write) are tracked automatically by BaseDiskCache.
    """
    
    _SCHEMA_VERSION = 1
    _DEFAULT_TTL = 365 * 24 * 60 * 60  # 1 year (merge dates are immutable)
    
    def __init__(self, *, cache_file: Path):
        super().__init__(cache_file=cache_file, schema_version=self._SCHEMA_VERSION)
    
    def get_if_fresh(self, key: str, ttl_s: int) -> Optional[str]:
        """Get cached merge date if fresh.
        
        Args:
            key: Cache key (e.g., "owner/repo:abc123def...")
            ttl_s: TTL in seconds
            
        Returns:
            Merge date ISO string, or None if not cached/stale
        """
        with self._mu:
            self._load_once()
            ent = self._check_item(key)  # Automatically tracks hit/miss
            
            if not isinstance(ent, dict):
                return None
            
            ts = int(ent.get("ts", 0) or 0)
            now = int(time.time())
            
            if ts and (now - ts) <= max(0, int(ttl_s)):
                merge_date = ent.get("merge_date")
                if merge_date is not None:  # Can be empty string for "not merged"
                    return str(merge_date) if merge_date else ""
            
            return None
    
    def put(self, key: str, merge_date: Optional[str]) -> None:
        """Store merge date.
        
        Args:
            key: Cache key
            merge_date: ISO timestamp string or None/empty if not merged
        """
        now = int(time.time())
        with self._mu:
            self._load_once()
            entry = {
                "ts": now,
                "merge_date": merge_date or "",
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
        return common.dynamo_utils_cache_dir() / "merge_dates.json"
    except ImportError:
        return Path.home() / ".cache" / "dynamo-utils" / "merge_dates.json"


MERGE_DATES_CACHE = MergeDatesCache(cache_file=_get_cache_file())
