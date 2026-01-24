"""Cache for GitHub pulls list API results.

Caching strategy:
  - Key: <owner>/<repo>:pulls:<state>:<base>
  - Value: Dict with 'ts', 'pulls' (list of PR summary objects), 'updated_at_epoch' (optional)
  - Adaptive TTL based on most recent PR's updated_at:
    - age < 1h -> 1m
    - age < 2h -> 2m
    - age < 4h -> 4m
    - age >= 4h -> 8m
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure imports work from both html_pages/ and parent directory
_module_dir = Path(__file__).resolve().parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

from cache.cache_base import BaseDiskCache
from cache_ttl_utils import pulls_list_adaptive_ttl_s


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
        """Get cached pulls list if fresh (using adaptive TTL if updated_at is stored).
        
        Args:
            key: Cache key (e.g., "owner/repo:pulls:open:main")
            ttl_s: Fallback TTL in seconds (used if updated_at is not available)
            
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
            
            # Adaptive TTL: use stored updated_at_epoch if available
            updated_at_epoch = ent.get("updated_at_epoch")
            if updated_at_epoch is not None:
                effective_ttl = pulls_list_adaptive_ttl_s(updated_at_epoch, default_ttl_s=ttl_s)
            else:
                effective_ttl = ttl_s
            
            if ts and (now - ts) <= max(0, int(effective_ttl)):
                pulls = ent.get("pulls")
                if isinstance(pulls, list):
                    return pulls
            
            return None
    
    def put(self, key: str, pulls: List[Dict[str, Any]], updated_at_epoch: Optional[int] = None) -> None:
        """Store pulls list.
        
        Args:
            key: Cache key
            pulls: List of PR summary objects
            updated_at_epoch: Most recent PR's updated_at timestamp (for adaptive TTL)
        """
        now = int(time.time())
        with self._mu:
            self._load_once()
            entry = {
                "ts": now,
                "pulls": pulls,
            }
            if updated_at_epoch is not None:
                entry["updated_at_epoch"] = updated_at_epoch
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
