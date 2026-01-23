from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import common
import ci_log_errors
from ci_log_errors import snippet as ci_snippet
import sys
from pathlib import Path as _Path

# Import from parent cache/ directory
_parent_dir = _Path(__file__).resolve().parent.parent
_cache_dir = _parent_dir / "cache"
if str(_cache_dir) not in sys.path:
    sys.path.insert(0, str(_cache_dir))

from cache_base import BaseDiskCache, BaseCacheStats


@dataclass
class SnippetCacheStats(BaseCacheStats):
    """Extended stats for snippet cache.
    
    Inherits hit/miss/write from BaseCacheStats, adds custom metrics.
    """
    compute_secs: float = 0.0
    total_secs: float = 0.0


class SnippetCache(BaseDiskCache):
    """Cache extracted error snippets and categories from CI job logs.

    Caching strategy:
      - Key: <ci_log_errors_sha>:<log_filename>
      - Invalidation: (mtime_ns, size_bytes, TTL=365 days)
      - Extract using ci_log_errors.snippet module
      - Categories are parsed from snippet text

    Stats (hit/miss/write) are tracked automatically by BaseDiskCache.
    Custom stats (compute_secs, total_secs) are tracked manually.
    """

    _SCHEMA_VERSION = 1
    _TTL_SECONDS = 365 * 24 * 60 * 60  # 365 days

    def __init__(self, *, cache_file: Path):
        super().__init__(cache_file=cache_file, schema_version=self._SCHEMA_VERSION)
        self._ci_log_errors_sha = ""
        # Override with extended stats
        self.stats = SnippetCacheStats()

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

    def _load_once(self) -> None:
        """Override to handle ci_log_errors fingerprint and migration."""
        if self._loaded:
            return

        if not self._ci_log_errors_sha:
            self._ci_log_errors_sha = self._compute_ci_log_errors_sha()

        # Call parent to load base structure
        super()._load_once()

        # Add/update ci_log_errors_sha in data
        got_sha = str(self._data.get("ci_log_errors_sha", "") or "")
        want_sha = self._ci_log_errors_sha

        # Migration: older versions used a different fingerprint scheme
        if got_sha and got_sha != want_sha and len(got_sha) != len(want_sha):
            items = self._get_items()
            if isinstance(items, dict):
                migrated: Dict[str, object] = {}
                for k, v in items.items():
                    if not isinstance(k, str) or ":" not in k:
                        continue
                    prefix, rest = k.split(":", 1)
                    if prefix != got_sha:
                        continue
                    new_k = f"{want_sha}:{rest}"
                    if new_k not in items:
                        migrated[new_k] = v
                if migrated:
                    items.update(migrated)
                    self._data["items"] = items
                    self._dirty = True

        # Always record the current fingerprint
        self._data["ci_log_errors_sha"] = want_sha

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
        """
        Return cached (snippet, categories) if fresh, else None.
        
        Note: Stats (hit/miss) are tracked automatically via _check_item()
        """
        p = Path(raw_log_path)
        if not p.exists() or not p.is_file():
            with self._mu:
                self.stats.miss += 1  # Manual miss (file doesn't exist)
            return None
        
        st = p.stat()
        mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
        size_bytes = int(st.st_size)
        filename = str(p.name)
        now_ts = int(time.time())

        with self._mu:
            self._load_once()
            key = self._cache_key_for_raw_log(filename, self._ci_log_errors_sha)
            ent = self._check_item(key)  # Automatically tracks hit/miss
            
            if not isinstance(ent, dict):
                return None
            
            # Validate freshness
            cached_mtime = int(ent.get("mtime_ns", -1) or -1)
            cached_size = int(ent.get("size", -1) or -1)
            cached_ts = int(ent.get("ts", 0) or 0)
            
            # Check TTL
            if cached_ts > 0 and (now_ts - cached_ts) > self._TTL_SECONDS:
                return None
            
            # Check mtime/size
            if cached_mtime != mtime_ns or cached_size != size_bytes:
                return None
            
            snippet = str(ent.get("snippet", "") or "")
            categories = list(ent.get("categories", []) or [])
            
            if not snippet:
                return None
            
            return (snippet, categories)

    def put_raw_snippet(self, *, raw_log_path: Path, snippet_raw: str) -> Tuple[str, List[str]]:
        """
        Store a freshly-computed snippet and return (body, cats).
        
        Note: Stats (write) are tracked automatically via _set_item()
        """
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
            entry = {
                "mtime_ns": int(mtime_ns),
                "size": int(size_bytes),
                "ts": int(now_ts),
                "snippet": str(body or ""),
                "categories": list(categories or []),
            }
            self._set_item(key, entry)  # Automatically tracks write
            self._persist()
        
        return (str(body or ""), list(categories or []))

    def get_or_compute(
        self,
        *,
        raw_log_path: Path,
    ) -> Tuple[str, List[str]]:
        """
        Get cached snippet or compute it.

        Returns:
            (snippet_text, categories): snippet without "Categories:" prefix, and list of category strings
        """
        t0 = time.monotonic()

        p = Path(raw_log_path)
        if not p.exists() or not p.is_file():
            return ("", [])

        # Check cache first
        cached = self.get_if_fresh(raw_log_path=p)
        if cached is not None:
            self.stats.total_secs += max(0.0, time.monotonic() - t0)
            return cached

        # Cache miss: compute outside locks (avoid blocking other readers/writers)
        t1 = time.monotonic()
        snippet_raw = ci_snippet.extract_error_snippet_from_log_file(p)
        dt_compute = max(0.0, time.monotonic() - t1)
        self.stats.compute_secs += dt_compute

        snippet_body, categories = self.put_raw_snippet(raw_log_path=p, snippet_raw=str(snippet_raw or ""))
        self.stats.total_secs += max(0.0, time.monotonic() - t0)
        return (snippet_body, categories)


# Singleton cache used by all dashboard generators
SNIPPET_CACHE = SnippetCache(
    cache_file=(common.dynamo_utils_cache_dir() / "snippet-cache.json")
)
