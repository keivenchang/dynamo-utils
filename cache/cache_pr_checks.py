"""Cache for PR check runs/statuses (GitHub Checks API).

Caching strategy:
  - Key: <owner>/<repo>#<pr_number>#<head_sha>
  - Value: Dict with 'ts', 'checks' (list of check run objects)
  - Short TTL (status changes frequently)
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


class PRChecksCache(BaseDiskCache):
    """Cache for PR check runs and statuses.
    
    Stores the raw check run data from GitHub Checks API to avoid
    redundant API calls for PR status.
    
    Stats (hit/miss/write) are tracked automatically by BaseDiskCache.
    """
    
    _SCHEMA_VERSION = 1
    _DEFAULT_TTL = 5 * 60  # 5 minutes (status changes frequently)
    
    def __init__(self, *, cache_file: Path):
        super().__init__(cache_file=cache_file, schema_version=self._SCHEMA_VERSION)
    
    def get_if_fresh(self, key: str, ttl_s: int) -> Optional[List[Dict[str, Any]]]:
        """Get cached check runs if fresh.
        
        Args:
            key: Cache key (e.g., "owner/repo#123#abc123")
            ttl_s: TTL in seconds
            
        Returns:
            List of check run objects, or None if not cached/stale
        """
        with self._mu:
            self._load_once()
            ent = self._check_item(key)  # Automatically tracks hit/miss
            
            if not isinstance(ent, dict):
                return None
            
            ts = int(ent.get("ts", 0) or 0)
            now = int(time.time())
            
            if ts and (now - ts) <= max(0, int(ttl_s)):
                checks = ent.get("checks")
                if isinstance(checks, list):
                    return checks
            
            return None
    
    def put(self, key: str, checks: List[Dict[str, Any]]) -> None:
        """Store check runs.
        
        Args:
            key: Cache key
            checks: List of check run objects
        """
        now = int(time.time())
        with self._mu:
            self._load_once()
            entry = {
                "ts": now,
                "checks": checks,
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
        return common.dynamo_utils_cache_dir() / "pr-checks" / "pr_checks_cache.json"
    except ImportError:
        return Path.home() / ".cache" / "dynamo-utils" / "pr-checks" / "pr_checks_cache.json"


PR_CHECKS_CACHE = PRChecksCache(cache_file=_get_cache_file())
