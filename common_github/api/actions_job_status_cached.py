"""Actions job status cached API (REST).

Resource:
  GET /repos/{owner}/{repo}/actions/jobs/{job_id}

Cache:
  ACTIONS_JOBS_CACHE (disk + memory)
  
OPTIMIZATION: Completed jobs are cached to disk permanently since their status never changes.
This reduces API calls from ~300+ per run to near-zero on subsequent runs.

TTL: Adaptive for non-completed jobs based on job age (2m/4m/30m/60m/80m then 120m), ∞ for completed jobs (immutable).
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from .base_cached import CachedResourceBase

if TYPE_CHECKING:  # pragma: no cover
    from .. import GitHubAPIClient


TTL_POLICY_DESCRIPTION = "adaptive (<1h=2m, <2h=4m, <4h=30m, <8h=60m, <12h=80m, >=12h=120m) for in_progress; ∞ for completed (immutable)"


class ActionsJobStatusCached(CachedResourceBase[Optional[str]]):
    def __init__(self, api: "GitHubAPIClient", *, ttl_s: int):
        super().__init__(api)
        self._ttl_s = int(ttl_s)

    @property
    def cache_name(self) -> str:
        return "actions_job_status"

    def api_call_format(self) -> str:
        return (
            "REST GET /repos/{owner}/{repo}/actions/jobs/{job_id} (etag)\n"
            "Example response fields used:\n"
            "  {\n"
            "    \"status\": \"completed\",\n"
            "    \"started_at\": \"2026-01-24T01:02:03Z\"\n"
            "  }"
        )

    def cache_key(self, **kwargs: Any) -> str:
        owner = str(kwargs["owner"])
        repo = str(kwargs["repo"])
        job_id = str(kwargs["job_id"])
        return f"{owner}/{repo}:jobstatus:{job_id}"

    def empty_value(self) -> Optional[str]:
        return None

    def cache_read(self, *, key: str) -> Optional[Dict[str, Any]]:
        """Read from memory cache first, then disk cache."""
        now = int(time.time())
        
        # Memory cache (all statuses)
        try:
            ent = self.api._actions_job_status_mem_cache.get(key)
            if isinstance(ent, dict):
                ts = int(ent.get("ts", 0) or 0)
                st = str(ent.get("status") or "")
                started_at_epoch = int(ent.get("started_at_epoch", 0) or 0)
                if st:
                    if st.lower() == "completed":
                        # Completed jobs are valid forever
                        return ent
                    # For in_progress, check adaptive TTL
                    eff_ttl = self.api._adaptive_ttl_s(timestamp_epoch=started_at_epoch, default_ttl_s=int(self._ttl_s))
                    if ts and (ts + eff_ttl) > now:
                        return ent
        except (ValueError, TypeError):
            pass

        # Disk cache (only for completed jobs - they never change!)
        disk = self.api._load_actions_job_disk_cache()
        ent = disk.get(key) if isinstance(disk, dict) else None
        if isinstance(ent, dict):
            status_cached = ent.get("status")
            # Completed jobs are cached forever; non-completed jobs respect TTL
            if str(status_cached or "").lower() == "completed":
                # Refresh memory cache
                self.api._actions_job_status_mem_cache[key] = {
                    "ts": now,
                    "status": status_cached,
                    "started_at_epoch": int(ent.get("started_at_epoch", 0) or 0)
                }
                return ent
            # For non-completed status, check TTL
            ts = int(ent.get("ts", 0) or 0)
            started_at_epoch = int(ent.get("started_at_epoch", 0) or 0)
            eff_ttl = self.api._adaptive_ttl_s(timestamp_epoch=started_at_epoch, default_ttl_s=int(self._ttl_s))
            if ts and (now - ts) <= int(eff_ttl):
                # Refresh memory cache
                self.api._actions_job_status_mem_cache[key] = {
                    "ts": ts,
                    "status": status_cached,
                    "started_at_epoch": int(started_at_epoch or 0)
                }
                return ent
        
        return None

    def value_from_cache_entry(self, *, entry: Dict[str, Any]) -> Optional[str]:
        status = entry.get("status")
        return str(status) if status is not None else None

    def is_cache_entry_fresh(self, *, entry: Dict[str, Any], now: int) -> bool:
        """Check if cache entry is still fresh."""
        status = str(entry.get("status", "") or "")
        if status.lower() == "completed":
            return True  # Completed jobs are valid forever
        
        ts = int(entry.get("ts", 0) or 0)
        started_at_epoch = int(entry.get("started_at_epoch", 0) or 0)
        eff_ttl = self.api._adaptive_ttl_s(timestamp_epoch=started_at_epoch, default_ttl_s=int(self._ttl_s))
        return (now - ts) <= eff_ttl

    def cache_write(self, *, key: str, value: Optional[str], meta: Dict[str, Any]) -> None:
        """Write to both memory and disk caches."""
        if value is None:
            return
        
        now = int(time.time())
        started_at_epoch = int(meta.get("started_at_epoch", 0) or 0)
        etag = meta.get("etag")
        
        # Memory cache
        self.api._actions_job_status_mem_cache[key] = {
            "ts": now,
            "status": value,
            "started_at_epoch": started_at_epoch
        }
        
        # Disk cache with ETag
        cache_entry = {
            "ts": now,
            "status": value,
            "started_at_epoch": started_at_epoch
        }
        if etag:
            cache_entry["etag"] = etag
        
        self.api._save_actions_job_disk_cache(key, cache_entry)

    def fetch(self, **kwargs: Any) -> Tuple[Optional[str], Dict[str, Any]]:
        """Fetch job status from GitHub API."""
        owner = str(kwargs["owner"])
        repo = str(kwargs["repo"])
        job_id = str(kwargs["job_id"])
        
        key = self.cache_key(owner=owner, repo=repo, job_id=job_id)
        
        # Get ETag from disk cache for conditional request
        etag = None
        disk = self.api._load_actions_job_disk_cache()
        ent = disk.get(key) if isinstance(disk, dict) else None
        if isinstance(ent, dict):
            etag = ent.get("etag")
        
        url = f"{self.api.base_url}/repos/{owner}/{repo}/actions/jobs/{job_id}"
        resp = self.api._rest_get(url, timeout=10, etag=etag)
        
        # Handle 304 Not Modified - data hasn't changed
        if resp.status_code == 304:
            if isinstance(ent, dict) and ent.get("status"):
                status_cached = ent.get("status")
                # Return cached status with refreshed timestamp in meta
                return str(status_cached), {
                    "started_at_epoch": int(ent.get("started_at_epoch", 0) or 0),
                    "etag": etag
                }
        
        # 404 can happen if job id is invalid / perm issue; just treat as unknown.
        if resp.status_code < 200 or resp.status_code >= 300:
            return None, {}
        
        try:
            data = resp.json() or {}
            status = data.get("status")
            if status is None:
                return None, {}
            
            status_s = str(status)
            started_at_s = str((data.get("started_at") if isinstance(data, dict) else "") or "").strip()
            started_at_epoch = 0
            if started_at_s:
                try:
                    dt = datetime.fromisoformat(started_at_s.replace("Z", "+00:00"))
                    started_at_epoch = int(dt.timestamp())
                except Exception:
                    started_at_epoch = 0
            
            # Extract ETag from response headers
            new_etag = resp.headers.get("ETag") if hasattr(resp, "headers") else None
            
            return status_s, {
                "started_at_epoch": started_at_epoch,
                "etag": new_etag
            }
        except (AttributeError, ValueError):
            return None, {}


def get_actions_job_status_cached(
    api: "GitHubAPIClient",
    *,
    owner: str,
    repo: str,
    job_id: str,
    ttl_s: int = 60
) -> Optional[str]:
    """Return GitHub Actions job status ("completed", "in_progress", ...) with disk cache for completed jobs.
    
    Args:
        api: GitHubAPIClient instance
        owner: Repository owner
        repo: Repository name
        job_id: Job ID
        ttl_s: TTL for in-progress jobs (default 60s; completed jobs cached forever)
    
    Returns:
        Job status string or None if not found
    """
    job_id_s = str(job_id or "").strip()
    if not job_id_s:
        return None
    
    return ActionsJobStatusCached(api, ttl_s=int(ttl_s)).get(owner=owner, repo=repo, job_id=job_id_s)
