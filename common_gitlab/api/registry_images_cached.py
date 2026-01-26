# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GitLab container registry images cached API.

Resource:
  GET /projects/{project_id}/registry/repositories/{registry_id}/tags

Cache:
  - Key: full commit SHA (40 chars)
  - Value: list of tag dicts (framework/arch/pipeline_id/location/etc)
  - Disk file: under dynamo-utils cache dir (via resolve_cache_path)

Notes:
  - This cache is intentionally "sticky": we only refresh SHAs that are missing
    (or cached as empty) unless `skip_fetch=True`.
  - The dashboard UX must not depend on any *local* build state.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

import requests
import time

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

_logger = logging.getLogger(__name__)

TTL_POLICY_DESCRIPTION = "images present: 30d; empty: 1h (retry for late-published tags); missing: fetch"

CACHE_NAME = "registry_images"
API_CALL_FORMAT = "GET /api/v4/projects/{project_id}/registry/repositories/{registry_id}/tags?per_page=100&page={page}"
CACHE_KEY_FORMAT = "<sha40>"
CACHE_FILE_DEFAULT = "gitlab_commit_sha.json"


class RegistryImagesCached(CachedResourceBase[List[Dict[str, Any]]]):
    """Registry tags cache keyed by full commit SHA.

    Note: fetch() is intentionally not implemented as "one-sha fetch" because
    GitLab tags are paginated by updated_at; we scan pages and extract requested SHAs.
    This class still exposes GitHub-style metadata: cache_name/api_call_format/cache_key.
    """

    def __init__(self, api: "GitLabAPIClient", *, project_id: str, registry_id: str, cache_file: str):
        super().__init__(api)
        self.project_id = str(project_id or "")
        self.registry_id = str(registry_id or "")
        self.cache_file = str(cache_file or CACHE_FILE_DEFAULT)
        self._cache_path = resolve_cache_path(self.cache_file)
        self._cache: Dict[str, Any] = {}
        if self._cache_path.exists():
            try:
                self._cache = json.loads(self._cache_path.read_text() or "{}")
            except (OSError, json.JSONDecodeError):
                self._cache = {}
        if isinstance(self._cache, dict) and "_metadata" in self._cache and isinstance(self._cache.get("_metadata"), dict):
            self._cache = {k: v for k, v in self._cache.items() if k != "_metadata"}
        self._dirty = False

    @property
    def cache_name(self) -> str:
        return CACHE_NAME

    def api_call_format(self) -> str:
        return API_CALL_FORMAT

    def cache_key(self, **kwargs: Any) -> str:
        return str(kwargs.get("sha") or "")

    def cache_read(self, *, key: str) -> Optional[Dict[str, Any]]:
        # Not used for this resource (value is list), but satisfy interface.
        return None

    def cache_write(self, *, key: str, value: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
        fetched_at = int(meta.get("fetched_at") or 0)
        vv = list(value or [])
        self._cache[str(key)] = make_cache_entry(value=vv, fetched_at=fetched_at, ttl_s=self._ttl_s_for_images(vv))
        self._dirty = True

    def is_cache_entry_fresh(self, *, entry: Dict[str, Any], now: int) -> bool:
        return True

    def value_from_cache_entry(self, *, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []

    def fetch(self, **kwargs: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:  # type: ignore[override]
        raise NotImplementedError("RegistryImagesCached uses get_many() (page scan), not per-key fetch().")

    def empty_value(self) -> List[Dict[str, Any]]:
        return []

    def flush(self, *, updated_shas: int) -> None:
        if not self._dirty:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_with_metadata: Dict[str, Any] = {
                "_metadata": {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "total_shas": len(self._cache),
                    "updated_shas": int(updated_shas or 0),
                }
            }
            cache_with_metadata.update(self._cache)
            self._cache_path.write_text(json.dumps(cache_with_metadata, indent=2))
        except OSError:
            pass
        self._dirty = False

    def get_many(self, *, sha_list: List[str], skip_fetch: bool) -> Dict[str, List[Dict[str, Any]]]:
        sha_list_norm = [str(s) for s in (sha_list or []) if str(s)]
        result: Dict[str, List[Dict[str, Any]]] = {}

        if skip_fetch:
            for sha in sha_list_norm:
                ent = self._entry_for_sha(sha)
                v = cache_entry_value(ent) if ent is not None else []
                vv = v if isinstance(v, list) else []
                result[sha] = vv
                if vv:
                    self.api._cache_hit(self.cache_name)
                else:
                    self.api._cache_miss(self.cache_name)
            return result

        # TTL policy:
        # - If images list is non-empty: consider fresh for 30d
        # - If images list is empty: retry after 1h (images may appear later)
        # - If missing: fetch
        now_i = int(time.time())
        shas_to_find: Set[str] = set()
        for sha in sha_list_norm:
            ent = self._entry_for_sha(sha)
            if ent is None:
                shas_to_find.add(sha)
                continue
            fetched_i = cache_entry_fetched_at(ent)
            imgs = cache_entry_value(ent)
            imgs_list = imgs if isinstance(imgs, list) else []
            ttl_s = self._ttl_s_for_images(imgs_list)
            if fetched_i is None:
                shas_to_find.add(sha)
                continue
            if max(0, int(now_i) - int(fetched_i)) >= int(ttl_s):
                shas_to_find.add(sha)
        if not shas_to_find:
            for sha in sha_list_norm:
                ent = self._entry_for_sha(sha)
                v = cache_entry_value(ent) if ent is not None else []
                vv = v if isinstance(v, list) else []
                result[sha] = vv
                if vv:
                    self.api._cache_hit(self.cache_name)
                else:
                    self.api._cache_miss(self.cache_name)
            return result

        if not self.api.has_token():
            _logger.warning("No GitLab token found; cannot fetch registry images. Using cache only.")
            for sha in sha_list_norm:
                v = self._cache.get(sha, [])
                vv = v if isinstance(v, list) else []
                result[sha] = vv
                if vv:
                    self.api._cache_hit(self.cache_name)
                else:
                    self.api._cache_miss(self.cache_name)
            return result

        per_page = 100
        endpoint = f"/api/v4/projects/{self.project_id}/registry/repositories/{self.registry_id}/tags"
        params_page1 = {"per_page": per_page, "page": 1, "order_by": "updated_at", "sort": "desc"}

        # Determine total pages via page-1 headers (needs raw requests for headers).
        url = f"{self.api.base_url}{endpoint}"
        t0 = time.monotonic()
        response = requests.get(url, headers=self.api.headers, params=params_page1, timeout=10)
        dt_s = max(0.0, time.monotonic() - t0)
        self.api._rest_record(label=self.cache_name, endpoint=endpoint, status_code=int(response.status_code), dt_s=dt_s)
        response.raise_for_status()
        first_page_tags = response.json() or []
        total_pages = int(response.headers.get("X-Total-Pages", "1") or "1")

        def fetch_page(page_num: int) -> List[Dict[str, Any]]:
            tags = self.api.get(
                endpoint,
                params={"per_page": per_page, "page": page_num, "order_by": "updated_at", "sort": "desc"},
                label=self.cache_name,
            )
            return tags if isinstance(tags, list) else []

        sha_to_images: Dict[str, List[Dict[str, Any]]] = {}

        def consume_tags(tags: List[Dict[str, Any]]) -> None:
            for tag_info in tags or []:
                if not isinstance(tag_info, dict):
                    continue
                tag_name = str(tag_info.get("name", "") or "")
                if len(tag_name) < 42:
                    continue
                sha_prefix = tag_name[:40]
                if sha_prefix not in shas_to_find:
                    continue
                parts = tag_name.split("-")
                if len(parts) < 4:
                    continue
                sha_to_images.setdefault(sha_prefix, []).append(
                    {
                        "tag": tag_name,
                        "framework": parts[2],
                        "arch": parts[3],
                        "pipeline_id": parts[1],
                        "location": tag_info.get("location", ""),
                        "total_size": tag_info.get("total_size", 0),
                        "created_at": tag_info.get("created_at", ""),
                        "registry_id": str(self.registry_id),
                    }
                )

        consume_tags(first_page_tags if isinstance(first_page_tags, list) else [])

        remaining: Set[str] = {sha for sha in shas_to_find if not sha_to_images.get(sha)}
        pages_fetched = 1
        hard_cap_pages = 200

        if remaining and total_pages > 1:
            next_start = 2
            next_end = min(int(total_pages), 20, hard_cap_pages)

            while remaining and next_start <= int(total_pages) and next_start <= hard_cap_pages:
                with ThreadPoolExecutor(max_workers=8) as executor:
                    future_to_page = {executor.submit(fetch_page, page_num): page_num for page_num in range(next_start, next_end + 1)}
                    for fut in as_completed(future_to_page):
                        tags = fut.result()
                        pages_fetched += 1
                        if tags:
                            consume_tags(tags)
                        remaining = {sha for sha in remaining if not sha_to_images.get(sha)}
                        if not remaining:
                            for f in future_to_page:
                                if not f.done():
                                    f.cancel()
                            break
                next_start = next_end + 1
                next_end = min(int(total_pages), max(next_end * 2, next_start), hard_cap_pages)

        _logger.debug(
            "GitLab registry scan: fetched %d page(s), found images for %d/%d SHAs",
            pages_fetched,
            len([s for s in shas_to_find if sha_to_images.get(s)]),
            len(shas_to_find),
        )

        fetched_at_write = int(time.time())
        for sha in shas_to_find:
            imgs = sha_to_images.get(sha, [])
            self._cache[sha] = make_cache_entry(value=imgs, fetched_at=fetched_at_write, ttl_s=self._ttl_s_for_images(imgs))
            self._dirty = True
        self.flush(updated_shas=len(shas_to_find))

        for sha in sha_list_norm:
            ent = self._entry_for_sha(sha)
            v = cache_entry_value(ent) if ent is not None else []
            vv = v if isinstance(v, list) else []
            result[sha] = vv
            if vv:
                self.api._cache_hit(self.cache_name)
            else:
                self.api._cache_miss(self.cache_name)
        return result

    def _entry_for_sha(self, sha: str) -> Optional[Dict[str, Any]]:
        """Return standardized cache entry for sha."""
        k = str(sha or "")
        if not k:
            return None
        ent = self._cache.get(k)
        return ent if is_cache_entry(ent) else None

    @staticmethod
    def _ttl_s_for_images(images: List[Dict[str, Any]]) -> int:
        return 30 * 24 * 3600 if images else 3600


def get_cached_registry_images_for_shas(
    api: "GitLabAPIClient",
    *,
    project_id: str,
    registry_id: str,
    sha_list: List[str],
    sha_to_datetime: Optional[Dict[str, Any]] = None,  # kept for compatibility; not used for gating
    cache_file: str = CACHE_FILE_DEFAULT,
    skip_fetch: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    _ = sha_to_datetime  # compatibility: no longer used for gating
    return RegistryImagesCached(api, project_id=project_id, registry_id=registry_id, cache_file=cache_file).get_many(
        sha_list=sha_list, skip_fetch=skip_fetch
    )

