from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import common
import ci_log_errors
from ci_log_errors import snippet as ci_snippet

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover - best-effort on non-POSIX
    fcntl = None  # type: ignore


@dataclass
class SnippetCacheStats:
    """Statistics for snippet cache hits/misses/compute time."""
    hit: int = 0
    miss: int = 0
    write: int = 0
    compute_secs: float = 0.0
    total_secs: float = 0.0


class SnippetCache:
    """Cache extracted error snippets and categories from CI job logs.

    Caching strategy:
      - Key: <ci_log_errors_sha>:<log_filename>
      - Invalidation: (mtime_ns, size_bytes, TTL=365 days)
      - Extract using ci_log_errors.snippet module
      - Categories are parsed from snippet text

    Thread-safe with lock for concurrent access.
    """

    _SCHEMA_VERSION = 1
    _TTL_SECONDS = 365 * 24 * 60 * 60  # 365 days

    def __init__(self, *, cache_file: Path):
        self._mu = Lock()
        self._cache_file = Path(cache_file)
        self._data: Dict[str, object] = {}
        self._loaded = False
        self._dirty = False
        self._ci_log_errors_sha = ""
        self.stats = SnippetCacheStats()
        self._initial_disk_count: Optional[int] = None  # Track disk count before modifications

    def _compute_ci_log_errors_sha(self) -> str:
        """Compute a stable fingerprint for ci_log_errors snippet behavior.

        We intentionally hash file *contents* (not mtimes) so different checkouts/installs
        with identical code produce the same fingerprint. This makes caches reusable.
        """
        base = Path(ci_log_errors.__file__).resolve().parent
        files = [
            base / "engine.py",
            base / "snippet.py",
            base / "render.py",
            base / "regexes.py",
        ]
        h = hashlib.sha1()
        for p in files:
            try:
                h.update(p.read_bytes())
            except OSError:
                # Still include filename so missing files change the fingerprint.
                h.update(str(p).encode("utf-8", errors="ignore"))
                h.update(b"\0")
        return h.hexdigest()[:12]

    def _lock_file_path(self) -> Path:
        # Keep lock file next to cache so it respects DYNAMO_UTILS_CACHE_DIR.
        return self._cache_file.with_name(f".{self._cache_file.name}.lock")

    def _acquire_disk_lock(self, *, timeout_s: float = 10.0) -> Optional[object]:
        """Best-effort inter-process lock for the cache file."""
        if fcntl is None:
            return None
        lock_path = self._lock_file_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
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

    def _release_disk_lock(self, lock_fh: Optional[object]) -> None:
        if lock_fh is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        finally:
            try:
                lock_fh.close()
            except Exception:
                pass

    def _load_once(self) -> None:
        """Load cache from disk (once per instance)."""
        if self._loaded:
            return
        self._loaded = True

        if not self._ci_log_errors_sha:
            self._ci_log_errors_sha = self._compute_ci_log_errors_sha()

        if not self._cache_file.exists():
            self._data = {"ci_log_errors_sha": self._ci_log_errors_sha, "items": {}}
            self._initial_disk_count = 0
            return

        data = json.loads(self._cache_file.read_text() or "{}")
        if not isinstance(data, dict):
            data = {}

        got_sha = str(data.get("ci_log_errors_sha", "") or "")
        want_sha = self._ci_log_errors_sha

        # Ensure items dict exists
        if not isinstance(data.get("items"), dict):
            data["items"] = {}

        # Migration: older versions used a different fingerprint scheme (e.g. 32-char md5).
        # If the only difference is fingerprint format, re-key entries to the new prefix so
        # caches remain warm across upgrades.
        if got_sha and got_sha != want_sha and len(got_sha) != len(want_sha):
            items_any = data.get("items")
            if isinstance(items_any, dict):
                migrated: Dict[str, object] = {}
                for k, v in items_any.items():
                    if not isinstance(k, str) or ":" not in k:
                        continue
                    prefix, rest = k.split(":", 1)
                    if prefix != got_sha:
                        continue
                    new_k = f"{want_sha}:{rest}"
                    if new_k not in items_any:
                        migrated[new_k] = v
                if migrated:
                    items_any.update(migrated)
                    data["items"] = items_any
                    self._dirty = True

        # Always record the current fingerprint for observability.
        data["ci_log_errors_sha"] = want_sha

        # Track initial disk count before any modifications
        self._initial_disk_count = len(data.get("items", {}))
        
        self._data = data

    def _persist(self) -> None:
        """Persist cache to disk with inter-process merge (best-effort)."""
        if not self._dirty:
            return

        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        items = self._data.get("items") if isinstance(self._data, dict) else {}
        if not isinstance(items, dict):
            items = {}
        mem_items: Dict[str, object] = dict(items)

        lock_fh = self._acquire_disk_lock(timeout_s=10.0)
        try:
            disk_data: Dict[str, object] = {}
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

            # Merge: disk first, then memory wins for conflicts.
            merged_items = {**disk_items, **mem_items}
            merged = {
                "ci_log_errors_sha": str(self._ci_log_errors_sha or ""),
                "items": merged_items,
            }

            tmp = f"{self._cache_file}.tmp.{os.getpid()}"
            Path(tmp).write_text(json.dumps(merged, separators=(",", ":")))
            os.replace(str(tmp), str(self._cache_file))

            # Update in-memory view to match what we wrote.
            self._data = merged
            self._dirty = False
        finally:
            self._release_disk_lock(lock_fh)

    @staticmethod
    def _cache_key_for_raw_log(filename: str, ci_log_errors_sha: str) -> str:
        """Generate cache key: <sha>:<filename>"""
        return f"{ci_log_errors_sha}:{filename}"

    @staticmethod
    def _split_categories_from_snippet(snippet: str) -> Tuple[str, List[str]]:
        """Split snippet body from Categories: prefix.

        Returns:
            (snippet_body, categories): snippet without "Categories:" prefix, and list of category strings
        """
        lines = (snippet or "").split("\n", 1)
        if len(lines) >= 2 and lines[0].startswith("Categories:"):
            cat_line = lines[0].replace("Categories:", "", 1).strip()
            categories = [c.strip() for c in cat_line.split(",") if c.strip()]
            snippet_body = lines[1]
            return (snippet_body, categories)
        return (snippet, [])

    def get_if_fresh(self, *, raw_log_path: Path) -> Optional[Tuple[str, List[str]]]:
        """Return cached (snippet, categories) if fresh, else None (no compute)."""
        p = Path(raw_log_path)
        if not p.exists() or not p.is_file():
            self.stats.miss += 1
            return None
        st = p.stat()
        mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
        size_bytes = int(st.st_size)

        filename = str(p.name)
        now_ts = int(time.time())

        with self._mu:
            self._load_once()
            key = self._cache_key_for_raw_log(filename, self._ci_log_errors_sha)
            items = self._data.get("items") if isinstance(self._data, dict) else {}
            if not isinstance(items, dict):
                self.stats.miss += 1
                return None
            ent = items.get(key)
            if not isinstance(ent, dict):
                self.stats.miss += 1
                return None
            cached_mtime = int(ent.get("mtime_ns", -1) or -1)
            cached_size = int(ent.get("size", -1) or -1)
            cached_ts = int(ent.get("ts", 0) or 0)
            if cached_ts > 0 and (now_ts - cached_ts) > self._TTL_SECONDS:
                self.stats.miss += 1
                return None
            if cached_mtime != mtime_ns or cached_size != size_bytes:
                self.stats.miss += 1
                return None
            snippet = str(ent.get("snippet", "") or "")
            categories = list(ent.get("categories", []) or [])
            if not snippet:
                self.stats.miss += 1
                return None
            self.stats.hit += 1
            return (snippet, categories)

    def put_raw_snippet(self, *, raw_log_path: Path, snippet_raw: str) -> Tuple[str, List[str]]:
        """Store a freshly-computed snippet (possibly computed by a worker) and return (body, cats)."""
        p = Path(raw_log_path)
        if not p.exists() or not p.is_file():
            return ("", [])
        st = p.stat()
        mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
        size_bytes = int(st.st_size)
        now_ts = int(time.time())
        body, categories = self._split_categories_from_snippet(str(snippet_raw or ""))

        filename = str(p.name)
        with self._mu:
            self._load_once()
            key = self._cache_key_for_raw_log(filename, self._ci_log_errors_sha)
            items = self._data.get("items") if isinstance(self._data, dict) else {}
            if not isinstance(items, dict):
                items = {}
            items[key] = {
                "mtime_ns": int(mtime_ns),
                "size": int(size_bytes),
                "ts": int(now_ts),
                "snippet": str(body or ""),
                "categories": list(categories or []),
            }
            self._data["items"] = items
            self._dirty = True
            self.stats.write += 1
            self._persist()
        return (str(body or ""), list(categories or []))

    def get_or_compute(
        self,
        *,
        raw_log_path: Path,
    ) -> Tuple[str, List[str]]:
        """Get cached snippet or compute it.

        Returns:
            (snippet_text, categories): snippet without "Categories:" prefix, and list of category strings
        """
        t0 = time.monotonic()

        p = Path(raw_log_path)
        if not p.exists() or not p.is_file():
            return ("", [])

        st = p.stat()
        mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
        size_bytes = int(st.st_size)

        filename = str(p.name)
        cached = self.get_if_fresh(raw_log_path=p)
        if cached is not None:
            self.stats.total_secs += max(0.0, time.monotonic() - t0)
            return cached

        # Cache miss: compute outside locks (avoid blocking other readers/writers).
        self.stats.miss += 1
        t1 = time.monotonic()
        snippet_raw = ci_snippet.extract_error_snippet_from_log_file(p)
        dt_compute = max(0.0, time.monotonic() - t1)
        self.stats.compute_secs += dt_compute

        snippet_body, categories = self.put_raw_snippet(raw_log_path=p, snippet_raw=str(snippet_raw or ""))
        self.stats.total_secs += max(0.0, time.monotonic() - t0)
        return (snippet_body, categories)

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


# Singleton cache used by all dashboard generators
SNIPPET_CACHE = SnippetCache(
    cache_file=(common.dynamo_utils_cache_dir() / "snippet-cache.json")
)
