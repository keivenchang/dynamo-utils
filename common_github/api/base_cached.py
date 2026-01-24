"""Base class for cached GitHub API resources.

Goal: make each cached resource readable + debuggable by enforcing a small interface:
- TTL policy
- API call “display format”
- shared cache access pattern
- consistent cache + API statistics reporting
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .. import GitHubAPIClient

T = TypeVar("T")


@dataclass(frozen=True)
class CacheLookupResult(Generic[T]):
    """Normalized cache lookup result used by CachedResourceBase.get()."""

    entry: Optional[Dict[str, Any]]
    # if entry is present but stale under the resource TTL policy
    is_fresh: bool


class CachedResourceBase(ABC, Generic[T]):
    """Base class for a cached resource backed by a disk cache object.

    Subclasses define:
    - cache key format
    - how to read/write cache entries
    - TTL policy (freshness check)
    - the actual API fetch implementation
    - how to record per-resource cache hit/miss/write stats
    """

    def __init__(self, api: "GitHubAPIClient"):
        self.api: GitHubAPIClient = api

    @property
    @abstractmethod
    def cache_name(self) -> str:
        """Short name used for stats keys (e.g. 'required_checks')."""

    @abstractmethod
    def api_call_format(self) -> str:
        """Human-readable description of the API call(s) this resource performs."""

    @abstractmethod
    def cache_key(self, **kwargs: Any) -> str:
        """Return a stable cache key for this resource."""

    @abstractmethod
    def cache_read(self, *, key: str) -> Optional[Dict[str, Any]]:
        """Read a raw cache entry dict or None if missing."""

    @abstractmethod
    def cache_write(self, *, key: str, value: T, meta: Dict[str, Any]) -> None:
        """Write to cache (meta includes ok/pr_state/etc as needed)."""

    @abstractmethod
    def is_cache_entry_fresh(self, *, entry: Dict[str, Any], now: int) -> bool:
        """TTL policy for the raw entry dict."""

    @abstractmethod
    def value_from_cache_entry(self, *, entry: Dict[str, Any]) -> T:
        """Convert cache entry dict into the returned value."""

    @abstractmethod
    def fetch(self, **kwargs: Any) -> Tuple[T, Dict[str, Any]]:
        """Fetch from network and return (value, meta_for_cache_write)."""

    @abstractmethod
    def empty_value(self) -> T:
        """Return an empty value for cache-only mode when no entry is usable."""

    def inflight_lock_key(self, **kwargs: Any) -> Optional[str]:
        """Optional inflight lock key to dedupe concurrent identical fetches."""
        return None

    def get(self, **kwargs: Any) -> T:
        """Shared get() flow: cache lookup -> TTL check -> optional fetch -> cache write."""
        key = self.cache_key(**kwargs)
        now_i = int(time.time())

        entry = self.cache_read(key=key)
        if entry is not None:
            if self.api.cache_only_mode or self.is_cache_entry_fresh(entry=entry, now=now_i):
                self.api._cache_hit(self.cache_name)
                return self.value_from_cache_entry(entry=entry)
            self.api._cache_miss(f"{self.cache_name}.expired")
        else:
            self.api._cache_miss(f"{self.cache_name}.missing")

        if self.api.cache_only_mode:
            return self.empty_value()

        lock_key = self.inflight_lock_key(**kwargs)
        if lock_key:
            lock = self.api._inflight_lock(lock_key)
            with lock:
                # Re-check cache (another thread may have populated it).
                entry2 = self.cache_read(key=key)
                if entry2 is not None:
                    if self.is_cache_entry_fresh(entry=entry2, now=now_i):
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

