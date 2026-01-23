"""Cache for GitHub Actions workflow run jobs.

Caching strategy:
  - Key: <owner>/<repo>:run_jobs:<run_id>
  - Value: {ts, jobs: {job_id: job_details}, etag}
  - TTL: 30 days (completed workflow runs never change)
  - ETag: PERFECT candidate - 100% 304 rate for completed runs
  
This cache stores all jobs for a workflow run fetched via:
  GET /repos/{owner}/{repo}/actions/runs/{run_id}/jobs

Why ETags are perfect here:
  - Completed workflow runs are IMMUTABLE (never change)
  - Expected 90-100% 304 Not Modified rate on repeated runs
  - Massive rate limit savings for historical data
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


class ActionsJobsCache(BaseDiskCache):
    """Cache for Actions workflow run jobs (GitHub /actions/runs/{run_id}/jobs endpoint).
    
    ETag Support:
        Completed workflow runs NEVER change, making this the BEST ETag candidate.
        Use get_stale_etag() for conditional requests - expect 100% 304 rate for completed runs.
    """
    
    _SCHEMA_VERSION = 1
    _DEFAULT_TTL = 30 * 24 * 3600  # 30 days (completed runs never change)
    
    def __init__(self, *, cache_file: Path):
        super().__init__(cache_file=cache_file, schema_version=self._SCHEMA_VERSION)
    
    def get_if_fresh(self, key: str, ttl_s: int) -> Optional[Dict[str, Any]]:
        """Get cached workflow run jobs if fresh.
        
        Args:
            key: Cache key (e.g., "owner/repo:run_jobs:12345")
            ttl_s: TTL in seconds (default: 30 days)
            
        Returns:
            Dict with 'jobs', 'ts', and optionally 'etag' keys, or None if not cached/stale
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
        for a conditional request. For completed workflow runs, GitHub will
        return 304 Not Modified 100% of the time since runs are immutable.
        """
        return self.get_etag(key)
    
    def put(self, key: str, jobs: Dict[str, Dict[str, Any]], etag: Optional[str] = None) -> None:
        """Store workflow run jobs with optional ETag.
        
        Args:
            key: Cache key (e.g., "owner/repo:run_jobs:12345")
            jobs: Dict mapping job_id -> job_details
            etag: Optional ETag from response headers (for 304 Not Modified support)
        """
        now = int(time.time())
        with self._mu:
            self._load_once()
            entry = {
                "ts": now,
                "jobs": jobs,
            }
            if etag:
                entry["etag"] = etag.strip()
            self._set_item(key, entry)  # Automatically tracks write
            self._persist()
    
    def refresh_timestamp(self, key: str) -> None:
        """Refresh timestamp on cache entry (for 304 Not Modified responses).
        
        When GitHub returns 304, the data hasn't changed (which is expected for
        completed workflow runs), so we just update the timestamp to extend the TTL.
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
        return common.dynamo_utils_cache_dir() / "actions-jobs" / "actions_jobs_cache.json"
    except ImportError:
        return Path.home() / ".cache" / "dynamo-utils" / "actions-jobs" / "actions_jobs_cache.json"


ACTIONS_JOBS_CACHE = ActionsJobsCache(cache_file=_get_cache_file())
