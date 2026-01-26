# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GitLab pipeline jobs cached API.

Resources:
  GET /api/v4/projects/{project_id}/pipelines/{pipeline_id}/jobs?per_page=100

TTL:
  - completed pipeline (running=0 and pending=0): cache forever
  - active pipeline: refresh (best-effort, currently time-based for counts)
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from common import resolve_cache_path

from .base_cached import (
    CachedResourceBase,
    cache_entry_fetched_at,
    cache_entry_value,
    is_cache_entry,
    make_cache_entry,
)

if TYPE_CHECKING:  # pragma: no cover
    from .. import GitLabAPIClient

TTL_POLICY_DESCRIPTION = "completed: 7d; active: 2m (counts/details)"

CACHE_NAME_COUNTS = "pipeline_jobs"
CACHE_NAME_DETAILS = "pipeline_jobs_details"
API_CALL_FORMAT = "GET /api/v4/projects/{project_id}/pipelines/{pipeline_id}/jobs?per_page=100"
CACHE_KEY_FORMAT = "<pipeline_id:int>"
CACHE_FILE_COUNTS_DEFAULT = "gitlab_pipeline_jobs.json"
CACHE_FILE_DETAILS_DEFAULT = "gitlab_pipeline_jobs_details.json"


def _epoch_from_iso(ts: str) -> int:
    # Accept GitLab's "Z" suffix.
    dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    return int(dt.timestamp())


class PipelineJobCountsCached(CachedResourceBase[Optional[Dict[str, int]]]):
    def __init__(self, api: "GitLabAPIClient", *, project_id: str, cache_file: str):
        super().__init__(api)
        self.project_id = str(project_id or "169905")
        self.cache_file = str(cache_file or CACHE_FILE_COUNTS_DEFAULT)
        self._cache_path = resolve_cache_path(self.cache_file)
        self._cache: Dict[int, Any] = {}
        if self._cache_path.exists():
            try:
                raw = json.loads(self._cache_path.read_text() or "{}")
                self._cache = {int(k): v for k, v in (raw or {}).items()}
            except (OSError, json.JSONDecodeError, ValueError):
                self._cache = {}
        self._dirty = False

    @property
    def cache_name(self) -> str:
        return CACHE_NAME_COUNTS

    def api_call_format(self) -> str:
        return API_CALL_FORMAT

    def cache_key(self, **kwargs: Any) -> str:
        return str(int(kwargs.get("pipeline_id") or 0))

    @staticmethod
    def _is_completed_counts(counts: Optional[Dict[str, int]]) -> bool:
        c = counts or {}
        return int(c.get("running", 0) or 0) == 0 and int(c.get("pending", 0) or 0) == 0

    def cache_read(self, *, key: str) -> Optional[Dict[str, Any]]:
        try:
            pid = int(key)
        except (ValueError, TypeError):
            return None
        ent = self._cache.get(pid)
        return ent if is_cache_entry(ent) else None

    def cache_write(self, *, key: str, value: Optional[Dict[str, int]], meta: Dict[str, Any]) -> None:
        try:
            pid = int(key)
        except (ValueError, TypeError):
            return
        fetched_at = int(meta.get("fetched_at") or 0)
        self._cache[pid] = make_cache_entry(value=value, fetched_at=fetched_at, ttl_s=self._ttl_s_for_counts(value))
        self._dirty = True

    def is_cache_entry_fresh(self, *, entry: Dict[str, Any], now: int) -> bool:
        fetched_i = cache_entry_fetched_at(entry)
        if fetched_i is None:
            return False
        counts = cache_entry_value(entry) if isinstance(cache_entry_value(entry), dict) else None
        ttl_s = self._ttl_s_for_counts(counts)
        return max(0, int(now) - int(fetched_i)) < int(ttl_s)

    def value_from_cache_entry(self, *, entry: Dict[str, Any]) -> Optional[Dict[str, int]]:
        v = cache_entry_value(entry)
        return v if isinstance(v, dict) else None

    def fetch(self, **kwargs: Any) -> Tuple[Optional[Dict[str, int]], Dict[str, Any]]:
        pipeline_id = int(kwargs.get("pipeline_id") or 0)
        endpoint = f"/api/v4/projects/{self.project_id}/pipelines/{pipeline_id}/jobs"
        jobs = self.api.get(endpoint, params={"per_page": 100}, timeout=10, label=self.cache_name)
        if isinstance(jobs, list) and jobs:
            return _counts_from_jobs(jobs), {"fetched_at": int(time.time())}
        return None, {"fetched_at": int(time.time())}

    def empty_value(self) -> Optional[Dict[str, int]]:
        return None

    def flush(self) -> None:
        if not self._dirty:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps({str(k): v for k, v in self._cache.items()}, indent=2))
        except OSError:
            pass
        self._dirty = False

    def get_many(self, *, pipeline_ids: List[int], skip_fetch: bool) -> Dict[int, Optional[Dict[str, int]]]:
        now_i = int(time.time())
        out: Dict[int, Optional[Dict[str, int]]] = {}
        to_fetch: List[int] = []
        for pid in pipeline_ids or []:
            pid_i = int(pid)
            ent = self.cache_read(key=str(pid_i))
            if ent is not None and self.is_cache_entry_fresh(entry=ent, now=now_i):
                out[pid_i] = self.value_from_cache_entry(entry=ent)
            else:
                out[pid_i] = self.value_from_cache_entry(entry=ent) if ent is not None else None
                if bool(skip_fetch):
                    # skip_fetch: don't enqueue network work
                    pass
                else:
                    to_fetch.append(pid_i)

        if skip_fetch or not to_fetch or not self.api.has_token():
            return out

        def fetch_one(pipeline_id: int) -> Tuple[int, Optional[Dict[str, int]], Dict[str, Any]]:
            v, meta = self.fetch(pipeline_id=pipeline_id)
            return pipeline_id, v, meta

        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = [ex.submit(fetch_one, pid) for pid in to_fetch]
            for fut in futures:
                pid, counts, meta = fut.result()
                out[pid] = counts
                self.cache_write(key=str(pid), value=counts, meta=meta)

        self.flush()
        return out

    @classmethod
    def _ttl_s_for_counts(cls, counts: Optional[Dict[str, int]]) -> int:
        return 7 * 24 * 3600 if cls._is_completed_counts(counts) else 2 * 60


class PipelineJobDetailsCached(CachedResourceBase[Optional[Dict[str, Any]]]):
    def __init__(self, api: "GitLabAPIClient", *, project_id: str, cache_file: str):
        super().__init__(api)
        self.project_id = str(project_id or "169905")
        self.cache_file = str(cache_file or CACHE_FILE_DETAILS_DEFAULT)
        self._cache_path = resolve_cache_path(self.cache_file)
        self._cache: Dict[int, Any] = {}
        if self._cache_path.exists():
            try:
                raw = json.loads(self._cache_path.read_text() or "{}")
                self._cache = {int(k): v for k, v in (raw or {}).items()}
            except (OSError, json.JSONDecodeError, ValueError):
                self._cache = {}
        self._dirty = False

    @property
    def cache_name(self) -> str:
        return CACHE_NAME_DETAILS

    def api_call_format(self) -> str:
        return API_CALL_FORMAT

    def cache_key(self, **kwargs: Any) -> str:
        return str(int(kwargs.get("pipeline_id") or 0))

    @staticmethod
    def _normalize_value(value: Any) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        if not isinstance(value, dict):
            return None
        counts0 = value.get("counts") if isinstance(value.get("counts"), dict) else None
        jobs0 = value.get("jobs") if isinstance(value.get("jobs"), list) else []
        base_counts = {"success": 0, "failed": 0, "running": 0, "pending": 0, "canceled": 0}
        counts = dict(counts0) if isinstance(counts0, dict) else {}
        for k, v in base_counts.items():
            counts.setdefault(k, v)
        return {"counts": counts, "jobs": list(jobs0)}

    @staticmethod
    def _is_completed_value(value_norm: Optional[Dict[str, Any]]) -> bool:
        if not value_norm:
            return False
        counts = value_norm.get("counts") or {}
        return int(counts.get("running", 0) or 0) == 0 and int(counts.get("pending", 0) or 0) == 0

    def cache_read(self, *, key: str) -> Optional[Dict[str, Any]]:
        try:
            pid = int(key)
        except (ValueError, TypeError):
            return None
        ent = self._cache.get(pid)
        return ent if is_cache_entry(ent) else None

    def cache_write(self, *, key: str, value: Optional[Dict[str, Any]], meta: Dict[str, Any]) -> None:
        try:
            pid = int(key)
        except (ValueError, TypeError):
            return
        fetched_at = int(meta.get("fetched_at") or 0)
        v_norm = self._normalize_value(value)
        self._cache[pid] = make_cache_entry(value=v_norm, fetched_at=fetched_at, ttl_s=self._ttl_s_for_value(v_norm))
        self._dirty = True

    def is_cache_entry_fresh(self, *, entry: Dict[str, Any], now: int) -> bool:
        fetched_i = cache_entry_fetched_at(entry)
        if fetched_i is None:
            return False
        v = cache_entry_value(entry)
        v_norm = self._normalize_value(v)
        ttl_s = self._ttl_s_for_value(v_norm)
        return max(0, int(now) - int(fetched_i)) < int(ttl_s)

    def value_from_cache_entry(self, *, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        v = cache_entry_value(entry)
        return v if isinstance(v, dict) else None

    def fetch(self, **kwargs: Any) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        pipeline_id = int(kwargs.get("pipeline_id") or 0)
        endpoint = f"/api/v4/projects/{self.project_id}/pipelines/{pipeline_id}/jobs"
        jobs = self.api.get(endpoint, params={"per_page": 100}, timeout=10, label=CACHE_NAME_COUNTS)
        if not isinstance(jobs, list) or not jobs:
            return None, {"fetched_at": int(time.time())}
        counts = _counts_from_jobs(jobs)
        slim_jobs: List[Dict[str, Any]] = []
        for job in jobs:
            if not isinstance(job, dict):
                continue
            nm = str(job.get("name", "") or "")
            if nm:
                slim_jobs.append({"name": nm, "stage": str(job.get("stage", "") or ""), "status": str(job.get("status", "unknown") or "")})
        return {"counts": counts, "jobs": slim_jobs}, {"fetched_at": int(time.time())}

    def empty_value(self) -> Optional[Dict[str, Any]]:
        return None

    def flush(self) -> None:
        if not self._dirty:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps({str(k): v for k, v in self._cache.items()}, indent=2))
        except OSError:
            pass
        self._dirty = False

    def get_many(self, *, pipeline_ids: List[int], skip_fetch: bool) -> Dict[int, Optional[Dict[str, Any]]]:
        now_i = int(time.time())
        out: Dict[int, Optional[Dict[str, Any]]] = {}
        to_fetch: List[int] = []
        for pid in pipeline_ids or []:
            pid_i = int(pid)
            ent = self.cache_read(key=str(pid_i))
            if ent is not None and self.is_cache_entry_fresh(entry=ent, now=now_i):
                out[pid_i] = self.value_from_cache_entry(entry=ent)
            else:
                out[pid_i] = self.value_from_cache_entry(entry=ent) if ent is not None else None
                if not bool(skip_fetch):
                    to_fetch.append(pid_i)

        if skip_fetch or not to_fetch or not self.api.has_token():
            return out

        def fetch_one(pipeline_id: int) -> Tuple[int, Optional[Dict[str, Any]], Dict[str, Any]]:
            v, meta = self.fetch(pipeline_id=pipeline_id)
            return pipeline_id, v, meta

        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = [ex.submit(fetch_one, pid) for pid in to_fetch]
            for fut in futures:
                pid, details, meta = fut.result()
                out[pid] = details
                self.cache_write(key=str(pid), value=details, meta=meta)

        self.flush()
        return out

    @classmethod
    def _ttl_s_for_value(cls, value_norm: Optional[Dict[str, Any]]) -> int:
        return 7 * 24 * 3600 if cls._is_completed_value(value_norm) else 2 * 60


def _counts_from_jobs(jobs: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {"success": 0, "failed": 0, "running": 0, "pending": 0, "canceled": 0}
    for job in jobs or []:
        if not isinstance(job, dict):
            continue
        status = str(job.get("status", "unknown") or "")
        if status in counts:
            counts[status] += 1
        elif status in ("created", "waiting_for_resource"):
            counts["pending"] += 1
        elif status in ("skipped", "manual"):
            pass
    return counts


def get_cached_pipeline_job_counts(
    api: "GitLabAPIClient",
    *,
    pipeline_ids: List[int],
    cache_file: str = CACHE_FILE_COUNTS_DEFAULT,
    skip_fetch: bool = False,
    project_id: str = "169905",
) -> Dict[int, Optional[Dict[str, int]]]:
    return PipelineJobCountsCached(api, project_id=project_id, cache_file=cache_file).get_many(
        pipeline_ids=pipeline_ids, skip_fetch=skip_fetch
    )


def get_cached_pipeline_job_details(
    api: "GitLabAPIClient",
    *,
    pipeline_ids: List[int],
    cache_file: str = CACHE_FILE_DETAILS_DEFAULT,
    skip_fetch: bool = False,
    project_id: str = "169905",
) -> Dict[int, Optional[Dict[str, Any]]]:
    return PipelineJobDetailsCached(api, project_id=project_id, cache_file=cache_file).get_many(
        pipeline_ids=pipeline_ids, skip_fetch=skip_fetch
    )

