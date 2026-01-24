#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Base class for disk-backed caches with locking and persistence.

Eliminates boilerplate code duplication across:
- duration_cache.py
- snippet_cache.py  
- pytest_timings_cache.py
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Tuple

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover - best-effort on non-POSIX
    fcntl = None  # type: ignore


@dataclass
class BaseCacheStats:
    """Basic cache statistics tracked automatically by BaseDiskCache."""
    hit: int = 0
    miss: int = 0
    write: int = 0


class BaseDiskCache:
    """Base class for thread-safe disk-backed caches with inter-process locking.
    
    Provides:
    - Thread-safe in-memory cache with Lock
    - Disk persistence with inter-process locking (fcntl)
    - Lazy loading (load on first access)
    - Merge on write (handle concurrent writers)
    - Cache size tracking (initial disk count vs current memory count)
    
    Subclasses must implement:
    - Cache-specific get/put methods
    - Stats tracking (optional)
    """

    def __init__(self, *, cache_file: Path, schema_version: int = 1):
        self._mu = Lock()
        self._cache_file = Path(cache_file)
        self._schema_version = schema_version
        self._data: Dict[str, Any] = {}
        self._loaded = False
        self._dirty = False
        self._initial_disk_count: Optional[int] = None
        self.stats = BaseCacheStats()  # Track hits/misses/writes automatically

    def _lock_file_path(self) -> Path:
        """Path to lock file (next to cache file)."""
        return self._cache_file.with_name(f".{self._cache_file.name}.lock")

    def _acquire_disk_lock(self, *, timeout_s: float = 10.0) -> Optional[object]:
        """Best-effort inter-process lock for the cache file.
        
        Returns file handle on success, None on failure/timeout.
        """
        if fcntl is None:
            return None
        
        lock_path = self._lock_file_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            fh = open(lock_path, "w")
            start = time.monotonic()
            
            while time.monotonic() - start < float(timeout_s):
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return fh
                except (IOError, OSError):
                    time.sleep(0.1)
            
            fh.close()
            return None
        except Exception:
            return None

    def _release_disk_lock(self, lock_fh: Optional[object]) -> None:
        """Release inter-process lock."""
        if lock_fh is None:
            return
        
        try:
            if fcntl is not None:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        finally:
            try:
                lock_fh.close()
            except Exception:
                pass

    def _load_once(self) -> None:
        """Load cache from disk (once per instance).
        
        Subclasses can override to add schema migration logic.
        """
        if self._loaded:
            return
        self._loaded = True

        if not self._cache_file.exists():
            self._data = self._create_empty_cache()
            self._initial_disk_count = 0
            return

        try:
            raw = json.loads(self._cache_file.read_text() or "{}")
        except Exception:
            raw = {}

        # Support both:
        #  - New schema: {"version": <int>, "items": {...}}
        #  - Legacy schema: {"key1": <val>, "key2": <val>, ...}
        data: Dict[str, Any] = raw if isinstance(raw, dict) else {}

        items: Dict[str, Any] = {}
        if isinstance(data.get("items"), dict):
            items = dict(data.get("items") or {})
        else:
            # Legacy: treat all top-level keys (except "version") as cache items.
            try:
                items = {k: v for (k, v) in data.items() if str(k) != "version"}
            except Exception:
                items = {}

        # Normalize to new schema in memory (we don't force-write migration here).
        normalized = {"version": self._schema_version, "items": items}

        # Track initial disk count (how many entries existed on disk at process start).
        self._initial_disk_count = len(items)
        self._data = normalized

    def _create_empty_cache(self) -> Dict[str, Any]:
        """Create empty cache structure. Subclasses can override."""
        return {"version": self._schema_version, "items": {}}

    def _persist(self) -> None:
        """Persist cache to disk with inter-process merge (best-effort)."""
        if not self._dirty:
            return

        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        items = self._data.get("items") if isinstance(self._data, dict) else {}
        if not isinstance(items, dict):
            items = {}
        mem_items: Dict[str, Any] = dict(items)

        lock_fh = self._acquire_disk_lock(timeout_s=10.0)
        try:
            # Merge with disk state (handle concurrent writers)
            disk_data: Dict[str, Any] = {}
            if self._cache_file.exists():
                try:
                    disk_data = json.loads(self._cache_file.read_text() or "{}")
                except Exception:
                    disk_data = {}
            
            if not isinstance(disk_data, dict):
                disk_data = {}
            
            disk_items = disk_data.get("items")
            if not isinstance(disk_items, dict):
                disk_items = {}

            # Merge: disk first, then memory wins for conflicts
            merged_items = {**disk_items, **mem_items}
            merged = {
                "version": self._schema_version,
                "items": merged_items,
            }

            # Atomic write (tmp file + rename)
            tmp = f"{self._cache_file}.tmp.{os.getpid()}"
            Path(tmp).write_text(json.dumps(merged, separators=(",", ":")))
            os.replace(str(tmp), str(self._cache_file))

            # Update in-memory view to match what we wrote
            self._data = merged
            self._dirty = False
        finally:
            self._release_disk_lock(lock_fh)

    def flush(self) -> None:
        """Persist cache to disk."""
        with self._mu:
            self._persist()

    def get_cache_sizes(self) -> Tuple[int, int]:
        """Return (mem_count, disk_count) for cache entries.
        
        disk_count is the initial count before this run's modifications.
        """
        with self._mu:
            self._load_once()
            mem_count = len(self._data.get("items", {})) if isinstance(self._data, dict) else 0
            disk_count = self._initial_disk_count if self._initial_disk_count is not None else 0
            
            return (mem_count, disk_count)

    def _get_items(self) -> Dict[str, Any]:
        """Get items dict (for subclass use)."""
        items = self._data.get("items") if isinstance(self._data, dict) else {}
        if not isinstance(items, dict):
            return {}
        return items

    def _check_item(self, key: str) -> Optional[Any]:
        """
        Check if an item exists in cache and track hit/miss stats.
        
        Args:
            key: Cache key to check
            
        Returns:
            Item value if found, None otherwise
            
        Note: Automatically increments stats.hit or stats.miss
        """
        items = self._get_items()
        value = items.get(key)
        if value is not None:
            self.stats.hit += 1
        else:
            self.stats.miss += 1
        return value

    def _set_item(self, key: str, value: Any) -> None:
        """
        Set an item and mark dirty (for subclass use).
        
        Note: Automatically increments stats.write
        """
        items = self._get_items()
        items[key] = value
        self._data["items"] = items
        self._dirty = True
        self.stats.write += 1

    def get_etag(self, key: str) -> Optional[str]:
        """
        Get ETag from a cached entry (if present).
        
        Useful for conditional requests (If-None-Match header).
        Returns None if entry doesn't exist or doesn't have an ETag.
        
        Example:
            etag = cache.get_etag("owner/repo#123")
            if etag:
                # Send conditional request with If-None-Match: etag
                # GitHub returns 304 Not Modified if unchanged (doesn't count against rate limit)
        """
        with self._mu:
            self._load_once()
            items = self._get_items()
            entry = items.get(key)
            
            if not isinstance(entry, dict):
                return None
            
            etag = entry.get("etag")
            if isinstance(etag, str) and etag.strip():
                return etag.strip()
            
            return None
    
    def put_with_etag(self, key: str, value: Any, etag: Optional[str] = None) -> None:
        """
        Store an item with optional ETag for future conditional requests.
        
        Args:
            key: Cache key
            value: Value to store (should be a dict for most cache types)
            etag: Optional ETag from response headers (for 304 Not Modified support)
        
        Example:
            response = requests.get(url, headers={"If-None-Match": cached_etag})
            if response.status_code == 304:
                # Use cached data
            else:
                new_etag = response.headers.get("ETag")
                cache.put_with_etag(key, response.json(), etag=new_etag)
        """
        with self._mu:
            self._load_once()
            
            # If value is already a dict with 'ts', preserve it; otherwise add timestamp
            if isinstance(value, dict):
                if "ts" not in value:
                    value["ts"] = int(time.time())
                if etag:
                    value["etag"] = etag.strip()
            else:
                # Wrap non-dict values
                value = {
                    "ts": int(time.time()),
                    "data": value,
                }
                if etag:
                    value["etag"] = etag.strip()
            
            self._set_item(key, value)
            self._persist()

    def __enter__(self) -> Dict[str, Any]:
        """
        Context manager entry: acquire lock and load cache.
        
        Usage:
            with cache_instance as items:
                items["key"] = value  # Modify cache dict directly
            # Cache is automatically saved on exit
        
        Returns:
            The items dict for direct manipulation
        """
        self._mu.acquire()
        try:
            self._load_once()
            self._context_lock_fh = self._acquire_disk_lock(timeout_s=10.0)
            
            # Reload from disk (handle concurrent writers)
            if self._cache_file.exists():
                try:
                    disk_data = json.loads(self._cache_file.read_text() or "{}")
                    if isinstance(disk_data, dict) and isinstance(disk_data.get("items"), dict):
                        self._data = disk_data
                except Exception:
                    pass
            
            return self._get_items()
        except Exception:
            self._mu.release()
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit: save cache and release lock.
        
        Only saves if no exception occurred during context.
        """
        try:
            if exc_type is None:
                self._dirty = True
                self._persist()
        finally:
            if hasattr(self, '_context_lock_fh'):
                self._release_disk_lock(self._context_lock_fh)
                delattr(self, '_context_lock_fh')
            self._mu.release()
        
        return False  # Don't suppress exceptions
