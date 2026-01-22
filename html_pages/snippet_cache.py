from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import common
import ci_log_errors
from ci_log_errors import snippet as ci_snippet


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
        """Compute fingerprint of ci_log_errors module to invalidate cache on code changes."""
        src_dir = Path(ci_log_errors.__file__).parent
        sha_parts = []
        for pyfile in sorted(src_dir.glob("*.py")):
            sha_parts.append(f"{pyfile.name}:{pyfile.stat().st_mtime_ns}")
        # Use hashlib for stable hash across process restarts (Python's hash() uses random seed per process)
        return hashlib.md5("".join(sha_parts).encode()).hexdigest()

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

        # Invalidate entire cache if ci_log_errors code changed
        got_sha = str(data.get("ci_log_errors_sha", "") or "")
        want_sha = self._ci_log_errors_sha
        if got_sha != want_sha:
            self._data = {"ci_log_errors_sha": want_sha, "items": {}}
            self._dirty = True
            self._initial_disk_count = 0
            return

        # Ensure items dict exists
        if not isinstance(data.get("items"), dict):
            data["items"] = {}

        # Track initial disk count before any modifications
        self._initial_disk_count = len(data.get("items", {}))
        
        self._data = data

    def _persist(self) -> None:
        """Write cache to disk atomically."""
        if not self._dirty:
            return

        self._cache_file.parent.mkdir(parents=True, exist_ok=True)

        # Size cap: keep at most 5000 entries (oldest by timestamp)
        items = self._data.get("items") if isinstance(self._data, dict) else {}
        if not isinstance(items, dict):
            items = {}

        if len(items) > 5200:
            pairs = []
            for k, v in items.items():
                if not isinstance(v, dict):
                    continue
                ts = int(v.get("ts", 0) or 0)
                pairs.append((ts, k))
            pairs.sort()
            # Drop oldest to 5000
            for _ts, k in pairs[: max(0, len(pairs) - 5000)]:
                items.pop(k, None)

        self._data["items"] = items

        # Atomic write
        tmp = str(self._cache_file) + ".tmp"
        Path(tmp).write_text(json.dumps(self._data, separators=(",", ":")))
        Path(tmp).replace(self._cache_file)
        self._dirty = False

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

        with self._mu:
            self._load_once()

            key = self._cache_key_for_raw_log(filename, self._ci_log_errors_sha)
            items = self._data.get("items") if isinstance(self._data, dict) else {}
            if not isinstance(items, dict):
                items = {}

            now_ts = int(time.time())

            # Check cache
            ent = items.get(key)
            if isinstance(ent, dict):
                cached_mtime = int(ent.get("mtime_ns", -1) or -1)
                cached_size = int(ent.get("size", -1) or -1)
                cached_ts = int(ent.get("ts", 0) or 0)

                # Check TTL
                if cached_ts > 0 and (now_ts - cached_ts) > self._TTL_SECONDS:
                    pass  # Expired, compute fresh
                elif cached_mtime == mtime_ns and cached_size == size_bytes:
                    snippet = str(ent.get("snippet", "") or "")
                    categories = list(ent.get("categories", []) or [])
                    if snippet:
                        self.stats.hit += 1
                        self.stats.total_secs += max(0.0, time.monotonic() - t0)
                        return (snippet, categories)

            # Cache miss: compute snippet
            self.stats.miss += 1
            t1 = time.monotonic()

            snippet_raw = ci_snippet.extract_error_snippet_from_log_file(p)

            snippet_body, categories = self._split_categories_from_snippet(snippet_raw)

            dt_compute = max(0.0, time.monotonic() - t1)
            self.stats.compute_secs += dt_compute

            # Update cache
            items[key] = {
                "mtime_ns": mtime_ns,
                "size": size_bytes,
                "ts": now_ts,
                "snippet": str(snippet_body or ""),
                "categories": list(categories or []),
            }
            self._data["items"] = items
            self._dirty = True
            self.stats.write += 1

            # Auto-flush to disk immediately (consistent with GitHub API caches)
            self._persist()

            self.stats.total_secs += max(0.0, time.monotonic() - t0)
            return (str(snippet_body or ""), categories)

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
