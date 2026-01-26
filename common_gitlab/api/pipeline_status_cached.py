# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GitLab pipeline status cached API.

Resource:
  GET /api/v4/projects/{project_id}/pipelines?sha={sha}&per_page=1

TTL:
  - success: cache forever
  - non-success / None: refetch every run (may be re-run / updated)
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
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

TTL_POLICY_DESCRIPTION = "success: 7d; terminal non-success: 30m; running/pending/unknown: 2m"

CACHE_NAME = "pipeline_status"
API_CALL_FORMAT = "GET /api/v4/projects/{project_id}/pipelines?sha={sha}&per_page=1"
CACHE_KEY_FORMAT = "<sha40>"
CACHE_FILE_DEFAULT = "gitlab_pipeline_status.json"


class PipelineStatusCached(CachedResourceBase[Optional[Dict[str, Any]]]):
    """Cached resource wrapper (GitHub-style metadata + centralized logic)."""

    def __init__(self, api: "GitLabAPIClient", *, project_id: str, cache_file: str):
        super().__init__(api)
        self.project_id = str(project_id or "169905")
        self.cache_file = str(cache_file or CACHE_FILE_DEFAULT)
        self._cache_path = resolve_cache_path(self.cache_file)
        self._cache: Dict[str, Any] = {}
        if self._cache_path.exists():
            try:
                self._cache = json.loads(self._cache_path.read_text() or "{}")
            except (OSError, json.JSONDecodeError):
                self._cache = {}
        self._dirty = False

    @property
    def cache_name(self) -> str:
        return CACHE_NAME

    def api_call_format(self) -> str:
        return API_CALL_FORMAT

    def cache_key(self, **kwargs: Any) -> str:
        return str(kwargs.get("sha") or "")

    def cache_read(self, *, key: str) -> Optional[Dict[str, Any]]:
        sha = str(key)
        ent = self._cache.get(sha)
        return ent if is_cache_entry(ent) else None

    def cache_write(self, *, key: str, value: Optional[Dict[str, Any]], meta: Dict[str, Any]) -> None:
        fetched_at = int(meta.get("fetched_at") or 0)
        status = "unknown"
        if isinstance(value, dict):
            status = str(value.get("status") or "unknown")
        ttl_s = self._ttl_s_for_status(status)
        self._cache[str(key)] = make_cache_entry(value=value, fetched_at=fetched_at, ttl_s=ttl_s)
        self._dirty = True

    def is_cache_entry_fresh(self, *, entry: Dict[str, Any], now: int) -> bool:
        fetched_i = cache_entry_fetched_at(entry)
        if fetched_i is None:
            return False
        v = cache_entry_value(entry)
        status = str(v.get("status") or "unknown") if isinstance(v, dict) else "unknown"
        ttl_s = self._ttl_s_for_status(status)
        return max(0, int(now) - int(fetched_i)) < int(ttl_s)

    def value_from_cache_entry(self, *, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        v = cache_entry_value(entry)
        return v if isinstance(v, dict) else None

    def fetch(self, **kwargs: Any) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        sha = str(kwargs.get("sha") or "")
        endpoint = f"/api/v4/projects/{self.project_id}/pipelines"
        pipelines = self.api.get(endpoint, params={"sha": sha, "per_page": 1}, timeout=10, label=self.cache_name)
        fetched_at = int(time.time())
        if isinstance(pipelines, list) and pipelines:
            p0 = pipelines[0]
            if isinstance(p0, dict):
                return (
                    {"status": p0.get("status", "unknown"), "id": p0.get("id"), "web_url": p0.get("web_url", "")},
                    {"fetched_at": fetched_at},
                )
        return None, {"fetched_at": fetched_at}

    def empty_value(self) -> Optional[Dict[str, Any]]:
        return None

    def flush(self) -> None:
        if not self._dirty:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache, indent=2))
        except OSError:
            pass
        self._dirty = False

    def get_many(self, *, sha_list: List[str], skip_fetch: bool) -> Dict[str, Optional[Dict[str, Any]]]:
        sha_list_norm = [str(s) for s in (sha_list or []) if str(s)]
        now_i = int(time.time())
        out: Dict[str, Optional[Dict[str, Any]]] = {}
        to_fetch: List[str] = []
        for sha in sha_list_norm:
            ent = self.cache_read(key=sha)
            if ent is not None and self.is_cache_entry_fresh(entry=ent, now=now_i):
                out[sha] = self.value_from_cache_entry(entry=ent)
            else:
                # stale or missing
                if bool(skip_fetch):
                    out[sha] = self.value_from_cache_entry(entry=ent) if ent is not None else None
                else:
                    out[sha] = self.value_from_cache_entry(entry=ent) if ent is not None else None
                    to_fetch.append(sha)

        if skip_fetch or not to_fetch or not self.api.has_token():
            return out

        def fetch_one(s: str) -> Tuple[str, Optional[Dict[str, Any]], Dict[str, Any]]:
            v, meta = self.fetch(sha=s)
            return s, v, meta

        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = [ex.submit(fetch_one, sha) for sha in to_fetch]
            for fut in futures:
                sha, info, meta = fut.result()
                out[sha] = info
                self.cache_write(key=sha, value=info, meta=meta)

        self.flush()
        return out

    @staticmethod
    def _ttl_s_for_status(status: str) -> int:
        s = str(status or "unknown")
        if s == "success":
            return 7 * 24 * 3600
        if s in ("failed", "canceled", "skipped"):
            return 30 * 60
        return 2 * 60


def get_cached_pipeline_status(
    api: "GitLabAPIClient",
    *,
    sha_list: List[str],
    cache_file: str = CACHE_FILE_DEFAULT,
    skip_fetch: bool = False,
    project_id: str = "169905",
) -> Dict[str, Optional[Dict[str, Any]]]:
    return PipelineStatusCached(api, project_id=project_id, cache_file=cache_file).get_many(
        sha_list=sha_list, skip_fetch=skip_fetch
    )

