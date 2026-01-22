from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import common


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
class PytestTimingCacheStats:
    hit: int = 0
    miss: int = 0
    write: int = 0
    parse_calls: int = 0
    parse_secs: float = 0.0


class PytestTimingCache:
    """Cache parsed pytest per-test timing rows to disk (JSON).

    Cache keying:
      - key: typically "<job_id>" (from raw-log file stem) else full path
      - invalidation: (mtime_ns, size_bytes)

    Notes:
      - Cache boundary is JSON/dicts on disk; in-memory we only manipulate concrete dataclasses.
      - This is intentionally independent of the snippet cache.
    """

    _SCHEMA_VERSION = 2

    def __init__(self, *, cache_file: Path):
        self._mu = Lock()
        self._cache_file = Path(cache_file)
        self._entries: Dict[str, PytestTimingCacheEntry] = {}
        self._loaded = False
        self.stats = PytestTimingCacheStats()
        self._initial_disk_count: Optional[int] = None  # Track disk count before modifications

    def _load_once(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._cache_file.exists():
            self._initial_disk_count = 0
            return
        data = json.loads(self._cache_file.read_text() or "{}")
        if not isinstance(data, dict):
            self._initial_disk_count = 0
            return
        entries = data.get("entries")
        if not isinstance(entries, dict):
            self._initial_disk_count = 0
            return
        
        # Track initial disk count before modifications
        self._initial_disk_count = len(entries)
        
        for k, v in entries.items():
            if not isinstance(k, str) or not isinstance(v, dict):
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
            ent = PytestTimingCacheEntry(
                schema_version=int(v.get("schema_version", self._SCHEMA_VERSION) or self._SCHEMA_VERSION),
                key=str(v.get("key", k) or k),
                step_name=str(v.get("step_name", "") or ""),
                ts_cached=int(v.get("ts_cached", 0) or 0),
                raw_log_mtime_ns=int(v.get("raw_log_mtime_ns", 0) or 0),
                raw_log_size_bytes=int(v.get("raw_log_size_bytes", 0) or 0),
                rows=rows,
            )
            # Only accept current schema.
            if int(ent.schema_version) != int(self._SCHEMA_VERSION):
                continue
            self._entries[k] = ent

    def _persist(self) -> None:
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self._SCHEMA_VERSION,
            "ts_written": int(time.time()),
            "entries": {k: asdict(v) for k, v in self._entries.items()},
        }
        tmp = str(self._cache_file) + ".tmp"
        Path(tmp).write_text(json.dumps(payload, separators=(",", ":")))
        Path(tmp).replace(self._cache_file)

    @staticmethod
    def _key_for_raw_log_and_step(p: Path, step_name: str) -> str:
        # Prefer numeric job-id cache keys; else use full path (stable within a repo checkout).
        # Include step_name to distinguish between different test phases (e.g., "Run unit tests" vs "Run e2e tests")
        step_suffix = f"_{step_name.lower().replace(' ', '_')}" if step_name else ""
        stem = str(p.stem or "").strip()
        if stem.isdigit():
            return f"{stem}{step_suffix}"
        return f"{str(p)}{step_suffix}"

    def get_if_fresh(self, *, raw_log_path: Path, step_name: str = "") -> Optional[List[Tuple[str, str, str]]]:
        p = Path(raw_log_path)
        st = p.stat()
        mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
        size_b = int(st.st_size)

        key = self._key_for_raw_log_and_step(p, step_name)
        with self._mu:
            self._load_once()
            ent = self._entries.get(key)
            if not isinstance(ent, PytestTimingCacheEntry):
                self.stats.miss += 1
                return None
            if int(ent.raw_log_mtime_ns) != int(mtime_ns) or int(ent.raw_log_size_bytes) != int(size_b):
                self.stats.miss += 1
                return None
            self.stats.hit += 1
            return [(r.display_name, r.duration_str, r.status_norm) for r in (ent.rows or [])]

    def put(
        self,
        *,
        raw_log_path: Path,
        step_name: str = "",
        rows: List[Tuple[str, str, str]],
    ) -> None:
        p = Path(raw_log_path)
        st = p.stat()
        mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
        size_b = int(st.st_size)

        key = self._key_for_raw_log_and_step(p, step_name)
        row_objs = [PytestTimingRow(display_name=a, duration_str=b, status_norm=c) for (a, b, c) in (rows or [])]

        with self._mu:
            self._load_once()
            self._entries[key] = PytestTimingCacheEntry(
                schema_version=self._SCHEMA_VERSION,
                key=key,
                step_name=step_name,
                ts_cached=int(time.time()),
                raw_log_mtime_ns=int(mtime_ns),
                raw_log_size_bytes=int(size_b),
                rows=row_objs,
            )
            self.stats.write += 1
            self._persist()

    def get_cache_sizes(self) -> Tuple[int, int]:
        """Return (mem_count, disk_count) for cache entries.
        
        disk_count is the initial count before this run's modifications.
        """
        with self._mu:
            self._load_once()
            mem_count = len(self._entries)
            disk_count = self._initial_disk_count if self._initial_disk_count is not None else 0
            
            return (mem_count, disk_count)


# Singleton cache used by all dashboard generators.
PYTEST_TIMINGS_CACHE = PytestTimingCache(
    cache_file=(common.dynamo_utils_cache_dir() / "pytest-test-timings.json")
)

