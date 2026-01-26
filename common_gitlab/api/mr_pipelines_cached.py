# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GitLab merge-request → latest pipeline cached API.

Resource:
  GET /api/v4/projects/{project_id}/merge_requests/{mr_iid}/pipelines?per_page=1&order_by=id&sort=desc

TTL:
  - no TTL; only fetched when missing (unless skip_fetch=True)
"""

from __future__ import annotations

import json
import logging
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
from ..exceptions import GitLabNotFoundError

if TYPE_CHECKING:  # pragma: no cover
    from .. import GitLabAPIClient

_logger = logging.getLogger(__name__)

TTL_POLICY_DESCRIPTION = "10m (MR pipelines move frequently; long enough for reuse, short enough to catch reruns)"

CACHE_NAME = "mr_pipelines"
API_CALL_FORMAT = "GET /api/v4/projects/{project_id}/merge_requests/{mr_iid}/pipelines?per_page=1&order_by=id&sort=desc"
CACHE_KEY_FORMAT = "<mr_iid:int>"
CACHE_FILE_DEFAULT = "gitlab_mr_pipelines.json"


class MRPipelinesCached(CachedResourceBase[Optional[Dict[str, Any]]]):
    def __init__(self, api: "GitLabAPIClient", *, project_id: str, cache_file: str):
        super().__init__(api)
        self.project_id = str(project_id or "169905")
        self.cache_file = str(cache_file or CACHE_FILE_DEFAULT)
        self._cache_path = resolve_cache_path(self.cache_file)
        self._cache: Dict[int, Optional[Dict[str, Any]]] = {}
        if self._cache_path.exists():
            try:
                raw = json.loads(self._cache_path.read_text() or "{}")
                self._cache = {int(k): v for k, v in (raw or {}).items()}
            except (OSError, json.JSONDecodeError, ValueError):
                self._cache = {}
        self._dirty = False

    @property
    def cache_name(self) -> str:
        return CACHE_NAME

    def api_call_format(self) -> str:
        return API_CALL_FORMAT

    def cache_key(self, **kwargs: Any) -> str:
        return str(int(kwargs.get("mr_iid") or 0))

    def cache_read(self, *, key: str) -> Optional[Dict[str, Any]]:
        try:
            mr = int(key)
        except (ValueError, TypeError):
            return None
        ent = self._cache.get(mr)
        return ent if is_cache_entry(ent) else None

    def cache_write(self, *, key: str, value: Optional[Dict[str, Any]], meta: Dict[str, Any]) -> None:
        try:
            mr = int(key)
        except (ValueError, TypeError):
            return
        fetched_at = int(meta.get("fetched_at") or 0)
        self._cache[mr] = make_cache_entry(value=value, fetched_at=fetched_at, ttl_s=10 * 60)
        self._dirty = True

    def is_cache_entry_fresh(self, *, entry: Dict[str, Any], now: int) -> bool:
        fetched_i = cache_entry_fetched_at(entry)
        if fetched_i is None:
            return False
        return max(0, int(now) - int(fetched_i)) < 10 * 60

    def value_from_cache_entry(self, *, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        v = cache_entry_value(entry)
        return v if isinstance(v, dict) else None

    def fetch(self, **kwargs: Any) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        mr_iid = int(kwargs.get("mr_iid") or 0)
        endpoint = f"/api/v4/projects/{self.project_id}/merge_requests/{mr_iid}/pipelines"
        pipelines = self.api.get(
            endpoint,
            params={"per_page": 1, "order_by": "id", "sort": "desc"},
            timeout=10,
            label=self.cache_name,
        )
        fetched_at = int(time.time())
        if isinstance(pipelines, list) and pipelines:
            p = pipelines[0]
            if isinstance(p, dict):
                return (
                    {
                        "id": p.get("id"),
                        "status": p.get("status", "unknown"),
                        "web_url": p.get("web_url", ""),
                        "sha": p.get("sha", ""),
                        "ref": p.get("ref", ""),
                    },
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
            self._cache_path.write_text(json.dumps({str(k): v for k, v in self._cache.items()}, indent=2))
        except OSError:
            pass
        self._dirty = False

    def get_many(self, *, mr_numbers: List[int], skip_fetch: bool) -> Dict[int, Optional[Dict[str, Any]]]:
        mr_numbers_norm = [int(m) for m in (mr_numbers or []) if int(m) > 0]
        now_i = int(time.time())
        out: Dict[int, Optional[Dict[str, Any]]] = {}
        to_fetch: List[int] = []
        for mr in mr_numbers_norm:
            ent = self.cache_read(key=str(mr))
            if ent is not None and self.is_cache_entry_fresh(entry=ent, now=now_i):
                out[mr] = self.value_from_cache_entry(entry=ent)
            else:
                out[mr] = self.value_from_cache_entry(entry=ent) if ent is not None else None
                if not bool(skip_fetch):
                    to_fetch.append(mr)

        if skip_fetch or not to_fetch or not self.api.has_token():
            return out

        def fetch_one(mr_iid: int) -> Tuple[int, Optional[Dict[str, Any]], Dict[str, Any]]:
            try:
                v, meta = self.fetch(mr_iid=mr_iid)
                return mr_iid, v, meta
            except GitLabNotFoundError:
                # Some commits reference non-existent MRs in GitLab; treat as "no data".
                return mr_iid, None, {"fetched_at": int(time.time())}

        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = [ex.submit(fetch_one, mr) for mr in to_fetch]
            for fut in futures:
                mr_iid, p, meta = fut.result()
                out[mr_iid] = p
                self.cache_write(key=str(mr_iid), value=p, meta=meta)
        self.flush()
        return out


def get_cached_merge_request_pipelines(
    api: "GitLabAPIClient",
    *,
    mr_numbers: List[int],
    project_id: str = "169905",
    cache_file: str = CACHE_FILE_DEFAULT,
    skip_fetch: bool = False,
) -> Dict[int, Optional[Dict[str, Any]]]:
    return MRPipelinesCached(api, project_id=project_id, cache_file=cache_file).get_many(
        mr_numbers=mr_numbers, skip_fetch=skip_fetch
    )

