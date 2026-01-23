"""Cache for GitHub pulls list API results.

Caching strategy:
  - Key: <owner>/<repo>:pulls:<state>:<base>
  - Value: Dict with 'ts', 'pulls' (list of PR summary objects)
  - Short TTL (PR list changes frequently)
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


class PullsListCache(BaseDiskCache):
    """Cache for GitHub pulls list API results.
    
    Stores the list of PRs matching certain criteria (state, base branch).
    
    Stats (hit/miss/write) are tracked automatically by BaseDiskCache.
    """
    
    _SCHEMA_VERSION = 1
    _DEFAULT_TTL = 5 * 60  # 5 minutes (PR list changes frequently)
    
    def __init__(self, *, cache_file: Path):
        super().__init__(cache_file=cache_file, schema_version=self._SCHEMA_VERSION)
    
    def get_if_fresh(self, key: str, ttl_s: int) -> Optional[List[Dict[str, Any]]]:
        """Get cached pulls list if fresh.
        
        Args:
            key: Cache key (e.g., "owner/repo:pulls:open:main")
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
                pulls = ent.get("pulls")
                if isinstance(pulls, list):
                    return pulls
            
            return None
    
    def put(self, key: str, pulls: List[Dict[str, Any]]) -> None:
        """Store pulls list.
        
        Args:
            key: Cache key
            pulls: List of PR summary objects
        """
        now = int(time.time())
        with self._mu:
            self._load_once()
            entry = {
                "ts": now,
                "pulls": pulls,
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
        return common.dynamo_utils_cache_dir() / "pulls_list.json"
    except ImportError:
        return Path.home() / ".cache" / "dynamo-utils" / "pulls_list.json"


PULLS_LIST_CACHE = PullsListCache(cache_file=_get_cache_file())
