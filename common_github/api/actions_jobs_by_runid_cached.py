# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cached GitHub Actions workflow run jobs (indexed by run_id).

This module manages caching for GitHub Actions workflow run jobs fetched via:
  GET /repos/{owner}/{repo}/actions/runs/{run_id}/jobs

Primary use case: Batch-fetch all jobs for a specific workflow run.

Example API Response:
  {
    "total_count": 2,
    "jobs": [
      {
        "id": 12345678,
        "run_id": 98765432,
        "status": "completed",
        "conclusion": "success",
        "started_at": "2026-01-24T10:00:00Z",
        "completed_at": "2026-01-24T10:05:30Z",
        "name": "build",
        "html_url": "https://github.com/owner/repo/actions/runs/98765432/job/12345678",
        "steps": [
          {
            "name": "Checkout",
            "status": "completed",
            "conclusion": "success",
            "number": 1,
            "started_at": "2026-01-24T10:00:10Z",
            "completed_at": "2026-01-24T10:00:30Z"
          },
          {
            "name": "Build",
            "status": "completed",
            "conclusion": "success",
            "number": 2,
            "started_at": "2026-01-24T10:00:31Z",
            "completed_at": "2026-01-24T10:05:20Z"
          }
        ]
      }
    ]
  }

Cached Fields (per job):
  - id: Job ID
  - run_id: Workflow run ID
  - status: "queued", "in_progress", "completed"
  - conclusion: "success", "failure", "cancelled", "skipped", null
  - started_at: ISO timestamp
  - completed_at: ISO timestamp
  - name: Job name
  - html_url: Job URL
  - steps: List of step objects with name, status, conclusion, number, timestamps

Caching strategy:
  - Key format: <owner>/<repo>:run_jobs:<run_id>
  - Key format (individual jobs): <owner>/<repo>:job:<job_id>
  - Value: {ts, jobs: {job_id: job_details}, etag}
  - TTL: ADAPTIVE based on completion status
    * Completed runs (all jobs done): ∞ (immutable, cached forever)
    * Incomplete runs: Adaptive 2m/4m/30m/60m/80m/120m based on oldest job age
      (SAME policy as actions_jobs_by_jobid for consistency)
  - ETag support: 100% 304 rate for completed runs
  
Important: GitHub re-runs create NEW run_ids and job_ids. The original run_id
remains immutable forever. When caching by run_id, we only store completed jobs
to avoid storing volatile in-progress job data.

ADAPTIVE TTL BEHAVIOR:
- Run with all completed jobs: Cached forever (immutable)
- Run with ANY in-progress jobs: Uses adaptive TTL based on oldest job's started_at:
  * <1h old: 2 minutes (frequent updates expected)
  * <2h old: 4 minutes
  * <4h old: 30 minutes
  * <8h old: 60 minutes
  * <12h old: 80 minutes
  * ≥12h old: 120 minutes (likely stuck/abandoned)
  
This ensures incomplete runs are checked frequently while minimizing API calls.

Architecture:
  This module follows the common_github/api/ pattern where each API resource
  has its own dedicated cache implementation (private BaseDiskCache subclass)
  rather than sharing a global cache instance.
  
Querying by job_id:
  This cache supports both run-level and job-level queries:
  - Run-level: key format "{owner}/{repo}:run_jobs:{run_id}" for batch fetch
  - Job-level: key format "{owner}/{repo}:job:{job_id}" for individual lookup
  
  The job-level query is automatically used by get_actions_job_status() and
  get_actions_job_details_cached() methods in __init__.py.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from common_github import GitHubAPIClient

# Ensure imports work from both html_pages/ and parent directory
_module_dir = Path(__file__).resolve().parent
_parent_dir = _module_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))
if str(_parent_dir.parent) not in sys.path:
    sys.path.insert(0, str(_parent_dir.parent))

from cache.cache_base import BaseDiskCache
from ..cache_ttl_utils import adaptive_ttl_s
from .base_cached import CachedResourceBase

TTL_POLICY_DESCRIPTION = "completed: 30d; incomplete: adaptive (<1h=2m, <2h=4m, <4h=30m, <8h=60m, <12h=80m, >=12h=120m)"

CACHE_NAME = "actions_job_status"
API_CALL_FORMAT = (
    "REST GET /repos/{owner}/{repo}/actions/jobs/{job_id}\n"
    "NOTE: This endpoint is ONLY used as a fallback when job data isn't\n"
    "      already in cache from the batch run-level fetch.\n"
    "      Primary method: GET /repos/{owner}/{repo}/actions/runs/{run_id}/jobs"
)
CACHE_KEY_FORMAT = "{owner}/{repo}:job:{job_id}"
CACHE_FILE_DEFAULT = "actions_jobs.json"


class _ActionsJobsCache(BaseDiskCache):
    """Private cache for Actions workflow run jobs.
    
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
        
        Uses adaptive TTL for incomplete runs (same as actions_jobs_by_jobid):
        - Completed runs: 30 days (immutable)
        - Incomplete runs: Adaptive 2m/4m/30m/60m/80m/120m based on oldest job age
        
        Args:
            key: Cache key (e.g., "owner/repo:run_jobs:12345" or "owner/repo:job:67890")
            ttl_s: TTL in seconds (default: 30 days for completed, adaptive for incomplete)
            
        Returns:
            Dict with 'jobs', 'ts', and optionally 'etag' keys, or None if not cached/stale
        """
        with self._mu:
            self._load_once()
            ent = self._check_item(key)  # Automatically tracks hit/miss
            
            if not isinstance(ent, dict):
                return None
            
            jobs = ent.get("jobs")
            if not isinstance(jobs, dict):
                return None
            
            # Check if ALL jobs are completed
            all_completed = True
            oldest_started_at = 0
            
            for job_details in jobs.values():
                if not isinstance(job_details, dict):
                    continue
                    
                status = str(job_details.get("status") or "").strip().lower()
                if status != "completed":
                    all_completed = False
                
                # Track oldest started_at for adaptive TTL
                started_at_s = str(job_details.get("started_at") or "").strip()
                if started_at_s:
                    try:
                        dt = datetime.fromisoformat(started_at_s.replace("Z", "+00:00"))
                        started_at_epoch = int(dt.timestamp())
                        if oldest_started_at == 0 or started_at_epoch < oldest_started_at:
                            oldest_started_at = started_at_epoch
                    except Exception:
                        pass
            
            # Completed runs: cached forever (immutable)
            if all_completed:
                return ent
            
            # Incomplete runs: use adaptive TTL (same as actions_jobs_by_jobid)
            ts = int(ent.get("ts", 0) or 0)
            now = int(time.time())
            
            if ts:
                effective_ttl = adaptive_ttl_s(oldest_started_at, default_ttl_s=ttl_s)
                if (now - ts) <= int(effective_ttl):
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
            key: Cache key (e.g., "owner/repo:run_jobs:12345" or "owner/repo:job:67890")
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


def _get_cache_file() -> Path:
    """Get cache file path, handling imports from different contexts."""
    try:
        import common
        return common.dynamo_utils_cache_dir() / CACHE_FILE_DEFAULT
    except ImportError:
        return Path.home() / ".cache" / "dynamo-utils" / CACHE_FILE_DEFAULT


# Private module-level cache instance (singleton)
_CACHE = _ActionsJobsCache(cache_file=_get_cache_file())


# =============================================================================
# CachedResourceBase Implementation (for job status queries)
# =============================================================================


class ActionsJobStatusCached(CachedResourceBase[Optional[str]]):
    """Cached resource for querying individual job status."""
    
    def __init__(self, api: "GitHubAPIClient", *, ttl_s: int):
        super().__init__(api)
        self._ttl_s = int(ttl_s)
    
    @property
    def cache_name(self) -> str:
        return CACHE_NAME
    
    def api_call_format(self) -> str:
        return API_CALL_FORMAT
    
    def cache_key(self, **kwargs: Any) -> str:
        owner = str(kwargs["owner"])
        repo = str(kwargs["repo"])
        job_id = str(kwargs["job_id"])
        return f"{owner}/{repo}:job:{job_id}"
    
    def empty_value(self) -> Optional[str]:
        return None
    
    def cache_read(self, *, key: str) -> Optional[Dict[str, Any]]:
        """Read from cache using adaptive TTL."""
        return _CACHE.get_if_fresh(key, ttl_s=int(self._ttl_s))
    
    def value_from_cache_entry(self, *, entry: Dict[str, Any]) -> Optional[str]:
        """Extract job status from cache entry."""
        jobs_dict = entry.get("jobs")
        if not isinstance(jobs_dict, dict):
            return None
        
        # Extract job_id from cache key (format: "{owner}/{repo}:job:{job_id}")
        # But we don't have access to kwargs here, so we need to iterate
        # This is a bit awkward, but works since job-level keys have single job
        for job_id, job_data in jobs_dict.items():
            if isinstance(job_data, dict):
                status = job_data.get("status")
                return str(status) if status is not None else None
        
        return None
    
    def is_cache_entry_fresh(self, *, entry: Dict[str, Any], now: int) -> bool:
        """_CACHE.get_if_fresh already applied TTL and adaptive logic."""
        return True
    
    def cache_write(self, *, key: str, value: Optional[str], meta: Dict[str, Any]) -> None:
        """Write to cache."""
        if value is None:
            return
        
        job_data = meta.get("job_data")
        if not isinstance(job_data, dict):
            return
        
        job_id = str(meta.get("job_id", ""))
        if not job_id:
            return
        
        # Store in the format expected by the cache
        _CACHE.put(key, jobs={job_id: job_data})
    
    def fetch(self, **kwargs: Any) -> Tuple[Optional[str], Dict[str, Any]]:
        """Fetch job status from GitHub API (fallback only).
        
        NOTE: In practice, this endpoint is rarely called because job data is
        typically already cached from the batch run-level fetch via
        get_actions_job_details_cached() in __init__.py, which uses
        GET /repos/{owner}/{repo}/actions/runs/{run_id}/jobs to fetch all jobs
        for a run in one call.
        
        This individual job endpoint is only used when:
        1. Job not in cache from previous batch fetch
        2. Direct call to get_actions_job_status() without prior run-level fetch
        3. Cache expired and no recent batch fetch has refreshed it
        """
        owner = str(kwargs["owner"])
        repo = str(kwargs["repo"])
        job_id = str(kwargs["job_id"])
        
        url = f"{self.api.base_url}/repos/{owner}/{repo}/actions/jobs/{job_id}"
        resp = self.api._rest_get(url, timeout=10)
        
        if resp.status_code < 200 or resp.status_code >= 300:
            return None, {}
        
        try:
            data = resp.json() or {}
            status = data.get("status")
            if status is None:
                return None, {}
            
            status_s = str(status)
            return status_s, {"job_data": data, "job_id": job_id}
        except (AttributeError, ValueError):
            return None, {}


# =============================================================================
# Public API
# =============================================================================


def get_cache_sizes() -> Tuple[int, int]:
    """Get memory and disk cache sizes for cache statistics.
    
    Returns:
        Tuple of (memory_size_bytes, disk_size_bytes)
    """
    return _CACHE.get_cache_sizes()


# Public API for __init__.py to access the cache
def get_if_fresh(key: str, ttl_s: int) -> Optional[Dict[str, Any]]:
    """Get cached workflow run jobs if fresh."""
    return _CACHE.get_if_fresh(key, ttl_s)


def get_stale_etag(key: str) -> Optional[str]:
    """Get ETag from stale cache entry for conditional requests."""
    return _CACHE.get_stale_etag(key)


def put(key: str, jobs: Dict[str, Dict[str, Any]], etag: Optional[str] = None) -> None:
    """Store workflow run jobs with optional ETag."""
    _CACHE.put(key, jobs, etag)


def refresh_timestamp(key: str) -> None:
    """Refresh timestamp on cache entry (for 304 Not Modified responses)."""
    _CACHE.refresh_timestamp(key)


def get_job_status_from_cache(
    api: "GitHubAPIClient",
    *,
    owner: str,
    repo: str,
    job_id: str,
    ttl_s: int = 60
) -> Optional[str]:
    """Return GitHub Actions job status with adaptive TTL caching.
    
    This queries the unified actions_jobs cache using the job-level key format.
    The cache is automatically populated when fetching jobs by run_id.
    
    Args:
        api: GitHubAPIClient instance
        owner: Repository owner
        repo: Repository name
        job_id: Job ID
        ttl_s: TTL for in-progress jobs (completed jobs cached forever)
    
    Returns:
        Job status string ("completed", "in_progress", etc.) or None if not found
    """
    job_id_s = str(job_id or "").strip()
    if not job_id_s:
        return None
    
    return ActionsJobStatusCached(api, ttl_s=int(ttl_s)).get(
        owner=owner,
        repo=repo,
        job_id=job_id_s
    )

