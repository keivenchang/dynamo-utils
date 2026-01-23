"""Cache for PR branch names (head ref for each PR).

Caching strategy:
  - Key: <owner>/<repo>#<pr_number>
  - Value: Dict with 'ts', 'branch' (branch name string)
  - Medium TTL (branch name rarely changes, but PRs can be rebased)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

# Ensure imports work from both html_pages/ and parent directory
_module_dir = Path(__file__).resolve().parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

from cache_base import BaseDiskCache


class PRBranchCache(BaseDiskCache):
    """Cache for PR branch names (head ref).
    
    Stores the branch name for each PR to avoid redundant API calls.
    
    Stats (hit/miss/write) are tracked automatically by BaseDiskCache.
    """
    
    _SCHEMA_VERSION = 1
    _DEFAULT_TTL = 7 * 24 * 60 * 60  # 7 days
    
    def __init__(self, *, cache_file: Path):
        super().__init__(cache_file=cache_file, schema_version=self._SCHEMA_VERSION)
    
    def get_if_fresh(self, key: str, ttl_s: int) -> Optional[str]:
        """Get cached branch name if fresh.
        
        Args:
            key: Cache key (e.g., "owner/repo#123")
            ttl_s: TTL in seconds
            
        Returns:
            Branch name, or None if not cached/stale
        """
        with self._mu:
            self._load_once()
            ent = self._check_item(key)  # Automatically tracks hit/miss
            
            if not isinstance(ent, dict):
                return None
            
            ts = int(ent.get("ts", 0) or 0)
            now = int(time.time())
            
            if ts and (now - ts) <= max(0, int(ttl_s)):
                branch = ent.get("branch")
                if branch:
                    return str(branch)
            
            return None
    
    def put(self, key: str, branch: str) -> None:
        """Store branch name.
        
        Args:
            key: Cache key
            branch: Branch name
        """
        now = int(time.time())
        with self._mu:
            self._load_once()
            entry = {
                "ts": now,
                "branch": branch,
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
        return common.dynamo_utils_cache_dir() / "pr-branches" / "pr_branch.json"
    except ImportError:
        return Path.home() / ".cache" / "dynamo-utils" / "pr-branches" / "pr_branch.json"


PR_BRANCH_CACHE = PRBranchCache(cache_file=_get_cache_file())
