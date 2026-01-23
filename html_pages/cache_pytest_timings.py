from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import common
import sys
from pathlib import Path

# Import from parent cache/ directory
_parent_dir = Path(__file__).resolve().parent.parent
_cache_dir = _parent_dir / "cache"
if str(_cache_dir) not in sys.path:
    sys.path.insert(0, str(_cache_dir))

from cache_base import BaseDiskCache, BaseCacheStats


@dataclass(frozen=True)
class PytestTimingRow:
    """One parsed pytest duration row."""

    display_name: str  # e.g. "[call] tests/foo.py::test_bar[param]"
    duration_str: str  # e.g. "55s" / "1m 43s"
    status_norm: str  # "success" | "failure" | "skipped" | ...


@dataclass
class PytestTimingCacheEntry:
    schema_version: int
    key: str
    step_name: str
    ts_cached: int
    raw_log_mtime_ns: int
    raw_log_size_bytes: int
    rows: List[PytestTimingRow]


@dataclass
class PytestTimingCacheStats(BaseCacheStats):
    """Extended stats for pytest timing cache.
    
    Inherits hit/miss/write from BaseCacheStats, adds custom metrics.
    """
    parse_calls: int = 0
    parse_secs: float = 0.0


class PytestTimingCache(BaseDiskCache):
    """Cache parsed pytest per-test timing rows to disk (JSON).

    Cache keying:
      - key: typically "<job_id>_<step_name>" or "<path>_<step_name>"
      - invalidation: (mtime_ns, size_bytes)

    Notes:
      - Cache boundary is JSON/dicts on disk; in-memory we use concrete dataclasses.
      - Stats (hit/miss/write) are tracked automatically by BaseDiskCache.
      - Custom stats (parse_calls, parse_secs) are tracked manually.
    """

    _SCHEMA_VERSION = 2

    def __init__(self, *, cache_file: Path):
        super().__init__(cache_file=cache_file, schema_version=self._SCHEMA_VERSION)
        # Override with extended stats
        self.stats = PytestTimingCacheStats()

    def _load_once(self) -> None:
        """Override to handle legacy migration and custom entry deserialization."""
        if self._loaded:
            return
        
        # MIGRATION: Old pytest cache used "entries" instead of "items"
        # We need to do this BEFORE calling super() because super() creates empty "items"
        if self._cache_file.exists():
            try:
                raw_data = json.loads(self._cache_file.read_text() or "{}")
                if isinstance(raw_data, dict) and "entries" in raw_data and "items" not in raw_data:
                    # Migrate: rename "entries" to "items"
                    raw_data["items"] = raw_data.pop("entries")
                    # Also fix schema version field name
                    if "schema_version" in raw_data and "version" not in raw_data:
                        raw_data["version"] = raw_data["schema_version"]
                    # Write back migrated data immediately
                    self._cache_file.write_text(json.dumps(raw_data, separators=(",", ":")))
            except Exception:
                pass  # Best-effort migration
        
        # Now call parent to load (will use migrated file)
        super()._load_once()
        
        # Deserialize entries from dict to PytestTimingCacheEntry objects
        items = self._get_items()
        entries: Dict[str, PytestTimingCacheEntry] = {}
        
        for k, v in items.items():
            if not isinstance(k, str) or not isinstance(v, dict):
                continue
            
            # Skip entries with wrong schema version
            if int(v.get("schema_version", 0)) != self._SCHEMA_VERSION:
                continue
            
            rows_in = v.get("rows")
            rows: List[PytestTimingRow] = []
            if isinstance(rows_in, list):
                for r in rows_in:
                    if not isinstance(r, dict):
                        continue
                    rows.append(
                        PytestTimingRow(
                            display_name=str(r.get("display_name", "") or ""),
                            duration_str=str(r.get("duration_str", "") or ""),
                            status_norm=str(r.get("status_norm", "") or ""),
                        )
                    )
            
            entries[k] = PytestTimingCacheEntry(
                schema_version=int(v.get("schema_version", self._SCHEMA_VERSION)),
                key=str(v.get("key", k) or k),
                step_name=str(v.get("step_name", "") or ""),
                ts_cached=int(v.get("ts_cached", 0) or 0),
                raw_log_mtime_ns=int(v.get("raw_log_mtime_ns", 0) or 0),
                raw_log_size_bytes=int(v.get("raw_log_size_bytes", 0) or 0),
                rows=rows,
            )
        
        # Replace dict items with deserialized entries
        for k, entry in entries.items():
            self._data["items"][k] = asdict(entry)

    @staticmethod
    def _key_for_raw_log_and_step(p: Path, step_name: str) -> str:
        """Generate cache key from log path and step name."""
        step_suffix = f"_{step_name.lower().replace(' ', '_')}" if step_name else ""
        stem = str(p.stem or "").strip()
        if stem.isdigit():
            return f"{stem}{step_suffix}"
        return f"{str(p)}{step_suffix}"

    def get_if_fresh(self, *, raw_log_path: Path, step_name: str = "") -> Optional[List[Tuple[str, str, str]]]:
        """
        Get cached pytest timings if cache entry is fresh (matches mtime/size).
        
        Returns:
            List of (display_name, duration_str, status_norm) tuples or None if not found/stale
            
        Note: Stats (hit/miss) are tracked AFTER freshness validation
        """
        p = Path(raw_log_path)
        st = p.stat()
        mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
        size_b = int(st.st_size)

        key = self._key_for_raw_log_and_step(p, step_name)
        
        with self._mu:
            self._load_once()
            # Get item directly without tracking stats (we need to validate freshness first)
            items = self._get_items()
            entry_dict = items.get(key)
            
            if entry_dict is None:
                self.stats.miss += 1
                return None
            
            # Validate freshness
            if not isinstance(entry_dict, dict):
                self.stats.miss += 1
                return None
            if int(entry_dict.get("raw_log_mtime_ns", 0)) != mtime_ns:
                self.stats.miss += 1  # Stale entry = miss
                return None
            if int(entry_dict.get("raw_log_size_bytes", 0)) != size_b:
                self.stats.miss += 1  # Stale entry = miss
                return None
            
            # Entry is fresh! Count as hit
            self.stats.hit += 1
            
            # Deserialize rows
            rows_in = entry_dict.get("rows")
            if not isinstance(rows_in, list):
                return None
            
            result = []
            for r in rows_in:
                if isinstance(r, dict):
                    result.append((
                        str(r.get("display_name", "")),
                        str(r.get("duration_str", "")),
                        str(r.get("status_norm", ""))
                    ))
            
            return result

    def put(
        self,
        *,
        raw_log_path: Path,
        step_name: str = "",
        rows: List[Tuple[str, str, str]],
    ) -> None:
        """
        Store parsed pytest timing rows in cache.
        
        Args:
            raw_log_path: Path to raw log file
            step_name: Name of CI step (for key generation)
            rows: List of (display_name, duration_str, status_norm) tuples
            
        Note: Stats (write) are tracked automatically via _set_item()
        """
        p = Path(raw_log_path)
        st = p.stat()
        mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
        size_b = int(st.st_size)

        key = self._key_for_raw_log_and_step(p, step_name)
        row_objs = [PytestTimingRow(display_name=a, duration_str=b, status_norm=c) for (a, b, c) in (rows or [])]

        entry = PytestTimingCacheEntry(
            schema_version=self._SCHEMA_VERSION,
            key=key,
            step_name=step_name,
            ts_cached=int(time.time()),
            raw_log_mtime_ns=int(mtime_ns),
            raw_log_size_bytes=int(size_b),
            rows=row_objs,
        )

        with self._mu:
            self._load_once()
            self._set_item(key, asdict(entry))  # Automatically tracks write
            self._persist()


# Singleton cache used by all dashboard generators.
PYTEST_TIMINGS_CACHE = PytestTimingCache(
    cache_file=(common.dynamo_utils_cache_dir() / "pytest-test-timings.json")
)
