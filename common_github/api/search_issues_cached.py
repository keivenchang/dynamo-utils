# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""search/issues cached API (REST).

Used as an optimization to batch-probe PR `updated_at` for a set of PR numbers.
This enables skipping per-PR enrichment fetches when a PR hasn't changed.

Endpoint:
  GET /search/issues?q=repo:OWNER/REPO type:pr number:123 number:456 ...

Caching strategy (preserves existing behavior/stats):
  - disabled flag (mem + disk): if GitHub returns 422 for this repo, disable for 6h
  - results cache (mem + disk): {ts, val: {pr_number: updated_at}}
  - inflight lock: dedupe concurrent identical requests across threads

NOTE: This module uses in-memory dicts + custom disk loading instead of BaseDiskCache
      due to complex logic (disabled flags, chunking, retry behavior). Could be refactored
      to use BaseDiskCache in the future for consistency.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base_cached import CachedResourceBase

if TYPE_CHECKING:  # pragma: no cover
    from .. import GitHubAPIClient


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except (ValueError, TypeError):
        return int(default)


# Keep TTL documentation next to the actual TTL implementation (see SearchIssuesCached.get()).
TTL_POLICY_DESCRIPTION = "ttl_s (default 60s); disabled for 6h after HTTP 422"

CACHE_NAME = "search_issues"
API_CALL_FORMAT = (
    "REST GET /search/issues?q=repo:{owner}/{repo} type:pr number:<n>...\n"
    "Example response fields used (truncated):\n"
    "  {\n"
    "    \"items\": [\n"
    "      {\"number\": 5530, \"updated_at\": \"2026-01-24T03:12:34Z\"}\n"
    "    ]\n"
    "  }"
)
CACHE_KEY_FORMAT = "{owner}/{repo}:search_issues:<nums_csv>"
CACHE_FILE_DEFAULT = "search_issues.json"


class SearchIssuesCached(CachedResourceBase[Dict[int, str]]):
    def __init__(self, api: "GitHubAPIClient", *, ttl_s: int, timeout: int):
        super().__init__(api)
        self._ttl_s = int(ttl_s)
        self._timeout = int(timeout)

    @property
    def cache_name(self) -> str:
        return CACHE_NAME

    def api_call_format(self) -> str:
        return API_CALL_FORMAT

    def cache_key(self, **kwargs: Any) -> str:
        owner = str(kwargs["owner"])
        repo = str(kwargs["repo"])
        nums: List[int] = kwargs["nums"]
        return f"{owner}/{repo}:{self.cache_name}:" + ",".join([str(n) for n in nums])

    def cache_read(self, *, key: str) -> Optional[Dict[str, Any]]:
        # Not used (we override get()).
        return None

    def cache_write(self, *, key: str, value: Dict[int, str], meta: Dict[str, Any]) -> None:
        # Not used (we override get()).
        return None

    def is_cache_entry_fresh(self, *, entry: Dict[str, Any], now: int) -> bool:
        # Not used (we override get()).
        return False

    def value_from_cache_entry(self, *, entry: Dict[str, Any]) -> Dict[int, str]:
        # Not used (we override get()).
        return {}

    def fetch(self, **kwargs: Any) -> tuple[Dict[int, str], Dict[str, Any]]:
        # Not used (we override get()).
        return {}, {}

    def empty_value(self) -> Dict[int, str]:
        return {}

    def get(self, **kwargs: Any) -> Dict[int, str]:
        cache_name = self.cache_name
        owner = str(kwargs["owner"])
        repo = str(kwargs["repo"])
        pr_numbers: List[int] = list(kwargs.get("pr_numbers") or [])

        nums = sorted({_safe_int(x, 0) for x in (pr_numbers or []) if _safe_int(x, 0) > 0})
        if not nums:
            return {}

        disabled_key = f"{owner}/{repo}:{cache_name}_disabled"
        disable_ttl_s = 6 * 3600
        now = int(time.time())

        # 0) disabled? (memory)
        ent = self.api._search_issues_disabled_mem_cache.get(disabled_key)
        if isinstance(ent, dict):
            ts = _safe_int(ent.get("ts", 0) or 0, 0)
            if ts and (now - ts) <= int(disable_ttl_s):
                self.api._cache_hit(f"{cache_name}.disabled_mem")
                return {}

        key = self.cache_key(owner=owner, repo=repo, nums=nums)

        def _coerce_map(val: Any) -> Dict[int, str]:
            if not isinstance(val, dict):
                return {}
            out: Dict[int, str] = {}
            for k, v in val.items():
                ks = str(k)
                vs = str(v or "").strip()
                if not ks.isdigit() or not vs:
                    continue
                out[int(ks)] = vs
            return out

        # 1) results cache (memory)
        ent = self.api._search_issues_mem_cache.get(key)
        if isinstance(ent, dict):
            ts = _safe_int(ent.get("ts", 0) or 0, 0)
            val = _coerce_map(ent.get("val"))
            if ts and val and ((now - ts) <= int(self._ttl_s) or self.api.cache_only_mode):
                if (now - ts) <= int(self._ttl_s):
                    self.api._cache_hit(f"{cache_name}.mem")
                else:
                    self.api._cache_hit(f"{cache_name}.mem_stale_cache_only")
                return val

        # 2) disk (disabled + results)
        disk = self.api._load_search_issues_disk_cache()
        if isinstance(disk, dict):
            dis = disk.get(disabled_key)
            if isinstance(dis, dict):
                ts = _safe_int(dis.get("ts", 0) or 0, 0)
                if ts and (now - ts) <= int(disable_ttl_s):
                    self.api._cache_hit(f"{cache_name}.disabled_disk")
                    self.api._search_issues_disabled_mem_cache[disabled_key] = {"ts": ts, "val": True}
                    return {}

            ent2 = disk.get(key)
            if isinstance(ent2, dict):
                ts = _safe_int(ent2.get("ts", 0) or 0, 0)
                val = _coerce_map(ent2.get("val"))
                if ts and val and ((now - ts) <= int(self._ttl_s) or self.api.cache_only_mode):
                    if (now - ts) <= int(self._ttl_s):
                        self.api._cache_hit(f"{cache_name}.disk")
                    else:
                        self.api._cache_hit(f"{cache_name}.disk_stale_cache_only")
                    self.api._search_issues_mem_cache[key] = {"ts": ts, "val": dict(val)}
                    return val

        if self.api.cache_only_mode:
            self.api._cache_miss(f"{cache_name}.cache_only_empty")
            return {}

        # 3) network, with inflight dedupe
        lock = self.api._inflight_lock(f"{cache_name}:{key}")
        with lock:
            # Re-check mem (another thread may have populated it).
            ent3 = self.api._search_issues_mem_cache.get(key)
            if isinstance(ent3, dict):
                ts = _safe_int(ent3.get("ts", 0) or 0, 0)
                val = _coerce_map(ent3.get("val"))
                if ts and val and ((now - ts) <= int(self._ttl_s) or self.api.cache_only_mode):
                    if (now - ts) <= int(self._ttl_s):
                        self.api._cache_hit(f"{cache_name}.mem")
                    else:
                        self.api._cache_hit(f"{cache_name}.mem_stale_cache_only")
                    return val

            # Re-check disk too.
            disk2 = self.api._load_search_issues_disk_cache()
            if isinstance(disk2, dict):
                ent4 = disk2.get(key)
                if isinstance(ent4, dict):
                    ts = _safe_int(ent4.get("ts", 0) or 0, 0)
                    val = _coerce_map(ent4.get("val"))
                    if ts and val and ((now - ts) <= int(self._ttl_s) or self.api.cache_only_mode):
                        if (now - ts) <= int(self._ttl_s):
                            self.api._cache_hit(f"{cache_name}.disk")
                        else:
                            self.api._cache_hit(f"{cache_name}.disk_stale_cache_only")
                        self.api._search_issues_mem_cache[key] = {"ts": ts, "val": dict(val)}
                        return val

            self.api._cache_miss(f"{cache_name}.network")

            # chunk to avoid overly long query strings
            chunks: List[List[int]] = []
            cur: List[int] = []
            for n in nums:
                cur.append(n)
                if len(cur) >= 25:
                    chunks.append(cur)
                    cur = []
            if cur:
                chunks.append(cur)

            out: Dict[int, str] = {}
            base_url = f"{self.api.base_url}/search/issues"

            for ch in chunks:
                q = f"repo:{owner}/{repo} type:pr " + " ".join([f"number:{n}" for n in ch])
                resp = self.api._rest_get(base_url, timeout=int(self._timeout), params={"q": q, "per_page": 100})

                # 422: disable search/issues temporarily for this repo (optimization-only)
                if hasattr(resp, "status_code") and int(resp.status_code) == 422:
                    self.api._search_issues_disabled_mem_cache[disabled_key] = {"ts": now, "val": True}
                    self.api._save_search_issues_disk_cache(disabled_key, {"ts": now, "val": True, "code": 422})
                    return {}

                if (not hasattr(resp, "status_code")) or int(resp.status_code) < 200 or int(resp.status_code) >= 300:
                    continue

                try:
                    data = resp.json() or {}
                except (json.JSONDecodeError, ValueError):
                    data = {}

                items = data.get("items") if isinstance(data, dict) else None
                if not isinstance(items, list):
                    continue

                for it in items:
                    if not isinstance(it, dict):
                        continue
                    num = it.get("number")
                    upd = it.get("updated_at")
                    if num is None or upd is None:
                        continue
                    n2 = _safe_int(num, 0)
                    if n2 <= 0:
                        continue
                    us = str(upd or "").strip()
                    if not us:
                        continue
                    out[n2] = us

            self.api._search_issues_mem_cache[key] = {"ts": now, "val": dict(out)}
            self.api._save_search_issues_disk_cache(key, {"ts": now, "val": dict(out)})
            return out


def get_pr_updated_at_via_search_issues_cached(
    api: "GitHubAPIClient",
    *,
    owner: str,
    repo: str,
    pr_numbers: List[int],
    ttl_s: int = 60,
    timeout: int = 15,
) -> Dict[int, str]:
    return SearchIssuesCached(api, ttl_s=int(ttl_s), timeout=int(timeout)).get(
        owner=owner, repo=repo, pr_numbers=pr_numbers
    )

