# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base class for cached GitLab API resources.

Mirrors `common_github/api/base_cached.py` so GitLab cached resources can follow
the same pattern over time (cache_name/api_call_format/TTL policy enforcement).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .. import GitLabAPIClient

T = TypeVar("T")

CACHE_ENTRY_VALUE_KEY = "v"
CACHE_ENTRY_META_KEY = "meta"
CACHE_ENTRY_FETCHED_AT_KEY = "fetched_at"
CACHE_ENTRY_TTL_S_KEY = "ttl_s"


def make_cache_entry(*, value: Any, fetched_at: int, ttl_s: Optional[int] = None) -> Dict[str, Any]:
    """Standard on-disk cache entry schema for GitLab cached resources.

    Format:
      {"v": <value>, "meta": {"fetched_at": <epoch_seconds>, "ttl_s": <seconds?>}}
    """
    meta: Dict[str, Any] = {CACHE_ENTRY_FETCHED_AT_KEY: int(fetched_at)}
    if ttl_s is not None:
        meta[CACHE_ENTRY_TTL_S_KEY] = int(ttl_s)
    return {CACHE_ENTRY_VALUE_KEY: value, CACHE_ENTRY_META_KEY: meta}


def is_cache_entry(obj: Any) -> bool:
    return (
        isinstance(obj, dict)
        and CACHE_ENTRY_VALUE_KEY in obj
        and CACHE_ENTRY_META_KEY in obj
        and isinstance(obj.get(CACHE_ENTRY_META_KEY), dict)
    )


def cache_entry_value(entry: Dict[str, Any]) -> Any:
    return entry.get(CACHE_ENTRY_VALUE_KEY)


def cache_entry_fetched_at(entry: Dict[str, Any]) -> Optional[int]:
    meta = entry.get(CACHE_ENTRY_META_KEY)
    if not isinstance(meta, dict):
        return None
    v = meta.get(CACHE_ENTRY_FETCHED_AT_KEY)
    if isinstance(v, int):
        return int(v)
    return None


def cache_entry_ttl_s(entry: Dict[str, Any]) -> Optional[int]:
    meta = entry.get(CACHE_ENTRY_META_KEY)
    if not isinstance(meta, dict):
        return None
    v = meta.get(CACHE_ENTRY_TTL_S_KEY)
    if isinstance(v, int):
        return int(v)
    return None


@dataclass(frozen=True)
class CacheLookupResult(Generic[T]):
    entry: Optional[Dict[str, Any]]
    is_fresh: bool


class CachedResourceBase(ABC, Generic[T]):
    """Shared get() flow: cache lookup -> TTL check -> optional fetch -> cache write.

This is intentionally minimal. GitLab cached resources can either use this or keep
specialized batch-fetch logic (e.g., registry tag scans).
"""

    def __init__(self, api: "GitLabAPIClient"):
        self.api = api

    @property
    @abstractmethod
    def cache_name(self) -> str:
        ...

    @abstractmethod
    def api_call_format(self) -> str:
        ...

    @abstractmethod
    def cache_key(self, **kwargs: Any) -> str:
        ...

    @abstractmethod
    def cache_read(self, *, key: str) -> Optional[Dict[str, Any]]:
        ...

    @abstractmethod
    def cache_write(self, *, key: str, value: T, meta: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def is_cache_entry_fresh(self, *, entry: Dict[str, Any], now: int) -> bool:
        ...

    @abstractmethod
    def value_from_cache_entry(self, *, entry: Dict[str, Any]) -> T:
        ...

    @abstractmethod
    def fetch(self, **kwargs: Any) -> Tuple[T, Dict[str, Any]]:
        ...

    @abstractmethod
    def empty_value(self) -> T:
        ...

    def inflight_lock_key(self, **kwargs: Any) -> Optional[str]:
        return None

    def get(self, *, cache_only_mode: Optional[bool] = None, **kwargs: Any) -> T:
        key = self.cache_key(**kwargs)
        now_i = int(time.time())
        cache_only = bool(self.api.cache_only_mode) if cache_only_mode is None else bool(cache_only_mode)

        entry = self.cache_read(key=key)
        if entry is not None:
            if self.is_cache_entry_fresh(entry=entry, now=now_i):
                self.api._cache_hit(self.cache_name)
                return self.value_from_cache_entry(entry=entry)
            self.api._cache_miss(f"{self.cache_name}.expired")
        else:
            self.api._cache_miss(f"{self.cache_name}.missing")

        if cache_only:
            # Best-effort: treat "present but stale" as a hit for visibility (it's a cache read);
            # treat missing entry as a miss.
            if entry is not None:
                self.api._cache_hit(self.cache_name)
                return self.value_from_cache_entry(entry=entry)
            return self.empty_value()

        lock_key = self.inflight_lock_key(**kwargs)
        if lock_key:
            lk = self.api._inflight_lock(lock_key)
            with lk:
                # Re-check cache in case another worker populated it.
                entry2 = self.cache_read(key=key)
                if entry2 is not None and self.is_cache_entry_fresh(entry=entry2, now=now_i):
                    self.api._cache_hit(self.cache_name)
                    return self.value_from_cache_entry(entry=entry2)
                val, meta = self.fetch(**kwargs)
                self.cache_write(key=key, value=val, meta=meta)
                self.api._cache_write(self.cache_name, entries=1)
                return val

        val, meta = self.fetch(**kwargs)
        self.cache_write(key=key, value=val, meta=meta)
        self.api._cache_write(self.cache_name, entries=1)
        return val

