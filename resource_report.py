#!/usr/bin/env python3
"""
Generate a fancy HTML resource report from the SQLite DB produced by `resource_monitor.py`.

Features:
- Interactive charts (zoom/pan/range buttons) via Plotly.js (loaded from CDN)
- Last N days view (default: 7d) with quick zoom buttons (1d/12h/6h/1h)
- GPU + CPU + memory + IO graphs
- "Spike" annotations for top-process CPU spikes (best-effort: uses stored offenders)
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)

_FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <rect width="100" height="100" fill="#76b900" rx="15"/>
  <text x="50" y="75" font-family="Arial, sans-serif" font-size="70" font-weight="bold" fill="white" text-anchor="middle">K</text>
</svg>"""


def _favicon_data_url() -> str:
    # Inline favicon so it works under any base path (/ vs /dynamo_ci/ etc) and when opened as a file.
    return "data:image/svg+xml," + urllib.parse.quote(_FAVICON_SVG)


def _default_db_path() -> Path:
    # Match `resource_monitor.py` default behavior
    cwd_cache = Path.cwd() / ".cache"
    if cwd_cache.exists() or (Path.cwd() / ".git").exists():
        return cwd_cache / "resource_monitor.sqlite"
    return Path.home() / ".cache" / "dynamo-utils" / "resource_monitor.sqlite"


def _ts_to_iso(ts_unix: float) -> str:
    # Plotly parses ISO8601 strings (localtime is OK; keep it simple)
    return datetime.fromtimestamp(ts_unix).isoformat(timespec="seconds")


@dataclass(frozen=True)
class SampleRow:
    sample_id: int
    ts_unix: float
    interval_s: float
    cpu_percent: Optional[float]
    mem_percent: Optional[float]
    load1: Optional[float]
    net_sent_bps: Optional[float]
    net_recv_bps: Optional[float]
    disk_read_bps: Optional[float]
    disk_write_bps: Optional[float]
    mem_used_bytes: Optional[int]
    mem_total_bytes: Optional[int]
    gpu_util_max: Optional[float]
    gpu_mem_used_mb_total: Optional[float]
    gpu_mem_total_mb_total: Optional[float]


def _connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    return con


def _get_time_window(con: sqlite3.Connection, *, days: float) -> Tuple[float, float]:
    cur = con.cursor()
    max_ts = cur.execute("SELECT MAX(ts_unix) AS t FROM samples").fetchone()["t"]
    if not max_ts:
        raise SystemExit("No samples found in DB.")
    end_ts = float(max_ts)
    start_ts = end_ts - float(days) * 86400.0
    return start_ts, end_ts


def _query_samples(con: sqlite3.Connection, *, start_ts: float) -> List[SampleRow]:
    cur = con.cursor()
    rows = cur.execute(
        """
        WITH gagg AS (
          SELECT
            sample_id,
            MAX(util_gpu) AS gpu_util_max,
            SUM(mem_used_mb) AS gpu_mem_used_mb_total,
            SUM(mem_total_mb) AS gpu_mem_total_mb_total
          FROM gpu_samples
          GROUP BY sample_id
        )
        SELECT
          s.id AS sample_id,
          s.ts_unix, s.interval_s,
          s.cpu_percent, s.mem_percent, s.load1,
          s.net_sent_bps, s.net_recv_bps,
          s.disk_read_bps, s.disk_write_bps,
          s.mem_used_bytes, s.mem_total_bytes,
          gagg.gpu_util_max,
          gagg.gpu_mem_used_mb_total,
          gagg.gpu_mem_total_mb_total
        FROM samples s
        LEFT JOIN gagg ON gagg.sample_id = s.id
        WHERE s.ts_unix >= ?
        ORDER BY s.ts_unix ASC
        """,
        (start_ts,),
    ).fetchall()
    out: List[SampleRow] = []
    for r in rows:
        out.append(
            SampleRow(
                sample_id=int(r["sample_id"]),
                ts_unix=float(r["ts_unix"]),
                interval_s=float(r["interval_s"] or 0.0),
                cpu_percent=(float(r["cpu_percent"]) if r["cpu_percent"] is not None else None),
                mem_percent=(float(r["mem_percent"]) if r["mem_percent"] is not None else None),
                load1=(float(r["load1"]) if r["load1"] is not None else None),
                net_sent_bps=(float(r["net_sent_bps"]) if r["net_sent_bps"] is not None else None),
                net_recv_bps=(float(r["net_recv_bps"]) if r["net_recv_bps"] is not None else None),
                disk_read_bps=(float(r["disk_read_bps"]) if r["disk_read_bps"] is not None else None),
                disk_write_bps=(float(r["disk_write_bps"]) if r["disk_write_bps"] is not None else None),
                mem_used_bytes=(int(r["mem_used_bytes"]) if r["mem_used_bytes"] is not None else None),
                mem_total_bytes=(int(r["mem_total_bytes"]) if r["mem_total_bytes"] is not None else None),
                gpu_util_max=(float(r["gpu_util_max"]) if r["gpu_util_max"] is not None else None),
                gpu_mem_used_mb_total=(
                    float(r["gpu_mem_used_mb_total"]) if r["gpu_mem_used_mb_total"] is not None else None
                ),
                gpu_mem_total_mb_total=(
                    float(r["gpu_mem_total_mb_total"]) if r["gpu_mem_total_mb_total"] is not None else None
                ),
            )
        )
    return out


def _query_gpu_timeseries(con: sqlite3.Connection, *, start_ts: float) -> Dict[int, Dict[str, Any]]:
    """
    Returns:
      {gpu_index: {"name": str, "x": [...], "util": [...], "mem_used": [...], "mem_total": [...], "temp": [...], "power": [...]} }
    """
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT
          s.ts_unix AS ts_unix,
          g.gpu_index AS gpu_index,
          g.name AS name,
          g.util_gpu AS util_gpu,
          g.mem_used_mb AS mem_used_mb,
          g.mem_total_mb AS mem_total_mb,
          g.temp_c AS temp_c,
          g.power_w AS power_w
        FROM gpu_samples g
        JOIN samples s ON s.id = g.sample_id
        WHERE s.ts_unix >= ?
        ORDER BY s.ts_unix ASC, g.gpu_index ASC
        """,
        (start_ts,),
    ).fetchall()

    out: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        idx = int(r["gpu_index"])
        d = out.setdefault(
            idx,
            {
                "name": r["name"] or f"GPU {idx}",
                "x": [],
                "util": [],
                "mem_used": [],
                "mem_total": [],
                "temp": [],
                "power": [],
            },
        )
        d["x"].append(_ts_to_iso(float(r["ts_unix"])))
        d["util"].append(float(r["util_gpu"]) if r["util_gpu"] is not None else None)
        d["mem_used"].append(float(r["mem_used_mb"]) if r["mem_used_mb"] is not None else None)
        d["mem_total"].append(float(r["mem_total_mb"]) if r["mem_total_mb"] is not None else None)
        d["temp"].append(float(r["temp_c"]) if r["temp_c"] is not None else None)
        d["power"].append(float(r["power_w"]) if r["power_w"] is not None else None)
    return out


def _table_exists(con: sqlite3.Connection, name: str) -> bool:
    cur = con.cursor()
    r = cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (name,)
    ).fetchone()
    return bool(r)


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    # Nearest-rank percentile
    k = int((p / 100.0) * (len(sorted_vals) - 1))
    k = max(0, min(len(sorted_vals) - 1, k))
    return float(sorted_vals[k])


def _query_ping_timeseries(con: sqlite3.Connection, *, start_ts: float) -> Dict[str, Dict[str, Any]]:
    """
    Returns:
      {target: {"x": [...], "rtt_ms": [...], "success": [...], "error": [...]} }
    """
    if not _table_exists(con, "ping_samples"):
        return {}
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT
          s.ts_unix AS ts_unix,
          p.target AS target,
          p.rtt_ms AS rtt_ms,
          p.success AS success,
          p.error AS error
        FROM ping_samples p
        JOIN samples s ON s.id = p.sample_id
        WHERE s.ts_unix >= ?
        ORDER BY s.ts_unix ASC, p.target ASC
        """,
        (start_ts,),
    ).fetchall()

    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        tgt = (r["target"] or "").strip()
        if not tgt:
            continue
        d = out.setdefault(tgt, {"x": [], "rtt_ms": [], "success": [], "error": []})
        d["x"].append(_ts_to_iso(float(r["ts_unix"])))
        d["rtt_ms"].append(float(r["rtt_ms"]) if r["rtt_ms"] is not None else None)
        d["success"].append(int(r["success"] or 0))
        d["error"].append((r["error"] or "") if r["error"] is not None else "")
    return out


def _query_ping_summary(con: sqlite3.Connection, *, start_ts: float) -> List[Dict[str, Any]]:
    """
    Returns per-target summary rows: loss%, count, avg, p95.
    """
    if not _table_exists(con, "ping_samples"):
        return []
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT
          p.target AS target,
          COUNT(*) AS n,
          SUM(CASE WHEN p.success != 0 THEN 1 ELSE 0 END) AS ok_n,
          SUM(CASE WHEN p.success = 0 THEN 1 ELSE 0 END) AS fail_n
        FROM ping_samples p
        JOIN samples s ON s.id = p.sample_id
        WHERE s.ts_unix >= ?
        GROUP BY p.target
        ORDER BY p.target ASC
        """,
        (start_ts,),
    ).fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        tgt = (r["target"] or "").strip()
        n = int(r["n"] or 0)
        ok_n = int(r["ok_n"] or 0)
        fail_n = int(r["fail_n"] or 0)
        loss_pct = (100.0 * float(fail_n) / float(n)) if n > 0 else 0.0

        rtts = con.execute(
            """
            SELECT p.rtt_ms
            FROM ping_samples p
            JOIN samples s ON s.id = p.sample_id
            WHERE s.ts_unix >= ? AND p.target = ? AND p.success != 0 AND p.rtt_ms IS NOT NULL
            ORDER BY p.rtt_ms ASC
            """,
            (start_ts, tgt),
        ).fetchall()
        vals = [float(x[0]) for x in rtts if x and x[0] is not None]
        vals.sort()
        avg = (sum(vals) / float(len(vals))) if vals else 0.0
        p95 = _percentile(vals, 95.0) if vals else 0.0

        out.append(
            {
                "target": tgt,
                "n": n,
                "ok_n": ok_n,
                "fail_n": fail_n,
                "loss_pct": loss_pct,
                "avg_ms": avg,
                "p95_ms": p95,
            }
        )
    return out


def _query_net_leaderboard(
    con: sqlite3.Connection, *, start_ts: float, limit: int
) -> List[Dict[str, Any]]:
    """
    Best-effort network "top talkers" leaderboard from net_process_samples.

    Note: net_process_samples typically contains only the top-K rows per sampling interval (from the monitor),
    so this is not "all processes", it's "top-K of top-K".
    """
    if not _table_exists(con, "net_process_samples"):
        return []
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT
          n.proc AS proc,
          SUM(COALESCE(n.sent_bps,0) + COALESCE(n.recv_bps,0)) AS sum_bps,
          AVG(COALESCE(n.sent_bps,0) + COALESCE(n.recv_bps,0)) AS avg_bps,
          MAX(COALESCE(n.sent_bps,0) + COALESCE(n.recv_bps,0)) AS max_bps,
          COUNT(*) AS sample_rows
        FROM net_process_samples n
        JOIN samples s ON s.id = n.sample_id
        WHERE s.ts_unix >= ?
        GROUP BY n.proc
        ORDER BY sum_bps DESC
        LIMIT ?
        """,
        (start_ts, int(limit)),
    ).fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "proc": r["proc"] or "",
                "sum_bps": float(r["sum_bps"] or 0.0),
                "avg_bps": float(r["avg_bps"] or 0.0),
                "max_bps": float(r["max_bps"] or 0.0),
                "sample_rows": int(r["sample_rows"] or 0),
            }
        )
    return out


def _query_net_timeseries(
    con: sqlite3.Connection, *, start_ts: float, procs: Sequence[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Returns:
      {proc: {"x": [...], "sent_bps": [...], "recv_bps": [...], "total_bps": [...]} }
    """
    if not procs:
        return {}
    if not _table_exists(con, "net_process_samples"):
        return {}
    cur = con.cursor()

    # Build an IN (...) clause safely.
    placeholders = ",".join("?" for _ in procs)
    q = f"""
        SELECT
          s.ts_unix AS ts_unix,
          n.proc AS proc,
          n.sent_bps AS sent_bps,
          n.recv_bps AS recv_bps
        FROM net_process_samples n
        JOIN samples s ON s.id = n.sample_id
        WHERE s.ts_unix >= ?
          AND n.proc IN ({placeholders})
        ORDER BY s.ts_unix ASC, n.proc ASC
    """
    rows = cur.execute(q, (start_ts, *list(procs))).fetchall()

    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        proc = (r["proc"] or "").strip()
        if not proc:
            continue
        d = out.setdefault(proc, {"x": [], "sent_bps": [], "recv_bps": [], "total_bps": []})
        sent = float(r["sent_bps"]) if r["sent_bps"] is not None else 0.0
        recv = float(r["recv_bps"]) if r["recv_bps"] is not None else 0.0
        d["x"].append(_ts_to_iso(float(r["ts_unix"])))
        d["sent_bps"].append(sent)
        d["recv_bps"].append(recv)
        d["total_bps"].append(sent + recv)
    return out


def _query_top_io_read_process_per_sample(
    con: sqlite3.Connection, *, start_ts: float
) -> Dict[int, Dict[str, Any]]:
    """
    Best-effort: process_samples only stores "offenders", so this finds the max IO-read process among recorded ones.

    Returns:
      {sample_id: {pid,name,username,cmdline,io_read_bps,io_write_bps,rss_bytes}}
    """
    cur = con.cursor()
    rows = cur.execute(
        """
        WITH ranked AS (
          SELECT
            p.sample_id AS sample_id,
            p.pid AS pid,
            p.name AS name,
            p.username AS username,
            p.cmdline AS cmdline,
            p.io_read_bps AS io_read_bps,
            p.io_write_bps AS io_write_bps,
            p.rss_bytes AS rss_bytes,
            ROW_NUMBER() OVER (
              PARTITION BY p.sample_id
              ORDER BY p.io_read_bps DESC NULLS LAST
            ) AS rn
          FROM process_samples p
          JOIN samples s ON s.id = p.sample_id
          WHERE s.ts_unix >= ?
        )
        SELECT * FROM ranked WHERE rn = 1
        """,
        (start_ts,),
    ).fetchall()

    out: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        sid = int(r["sample_id"])
        out[sid] = {
            "pid": int(r["pid"]) if r["pid"] is not None else None,
            "name": r["name"] or "",
            "username": r["username"] or "",
            "cmdline": r["cmdline"] or "",
            "io_read_bps": float(r["io_read_bps"]) if r["io_read_bps"] is not None else None,
            "io_write_bps": float(r["io_write_bps"]) if r["io_write_bps"] is not None else None,
            "rss_mb": (float(r["rss_bytes"]) / 1024.0 / 1024.0) if r["rss_bytes"] is not None else 0.0,
        }
    return out


def _detect_disk_read_spikes(
    samples: List[SampleRow],
    *,
    min_mibps: float,
    sigma: float,
    max_spikes: int,
) -> List[Dict[str, Any]]:
    """
    Detect spikes on system disk_read_bps (MiB/s) using a robust MAD threshold + local-max guard.
    """
    vals: List[float] = []
    for s in samples:
        v = (float(s.disk_read_bps) / 1024.0 / 1024.0) if s.disk_read_bps is not None else 0.0
        vals.append(v)
    med, mad = _median_mad(vals)
    robust_std = 1.4826 * mad
    thresh = max(float(min_mibps), med + float(sigma) * robust_std)

    spikes: List[Dict[str, Any]] = []
    for i, s in enumerate(samples):
        v = (float(s.disk_read_bps) / 1024.0 / 1024.0) if s.disk_read_bps is not None else 0.0
        if v < thresh:
            continue
        prev_v = vals[i - 1] if i > 0 else -1.0
        next_v = vals[i + 1] if i + 1 < len(vals) else -1.0
        if v < prev_v or v < next_v:
            continue
        spikes.append(
            {
                "sample_id": int(s.sample_id),
                "ts_unix": float(s.ts_unix),
                "ts": _ts_to_iso(float(s.ts_unix)),
                "disk_read_mibps": float(v),
            }
        )

    spikes.sort(key=lambda x: x["disk_read_mibps"], reverse=True)
    spikes = spikes[: int(max_spikes)]
    spikes.sort(key=lambda x: x["ts_unix"])
    return spikes


def _query_docker_leaderboard(
    con: sqlite3.Connection, *, start_ts: float, limit: int
) -> List[Dict[str, Any]]:
    """
    Container leaderboard from docker_container_samples.
    """
    if not _table_exists(con, "docker_container_samples"):
        return []
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT
          d.container_id AS container_id,
          MAX(COALESCE(d.image,'')) AS image,
          MAX(d.container_id) AS container_id,
          MAX(d.name) AS name,
          AVG(COALESCE(d.cpu_percent,0)) AS avg_cpu_percent,
          MAX(COALESCE(d.cpu_percent,0)) AS max_cpu_percent,
          AVG(COALESCE(d.mem_usage_bytes,0)) AS avg_mem_usage_bytes,
          MAX(COALESCE(d.mem_usage_bytes,0)) AS max_mem_usage_bytes,
          AVG(COALESCE(d.mem_percent,0)) AS avg_mem_percent,
          MAX(COALESCE(d.mem_percent,0)) AS max_mem_percent,
          COUNT(*) AS sample_rows
        FROM docker_container_samples d
        JOIN samples s ON s.id = d.sample_id
        WHERE s.ts_unix >= ?
        GROUP BY d.container_id
        ORDER BY avg_mem_usage_bytes DESC, avg_cpu_percent DESC
        LIMIT ?
        """,
        (start_ts, int(limit)),
    ).fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "container_id": r["container_id"] or "",
                "image": r["image"] or "",
                "name": r["name"] or "",
                "avg_cpu_percent": float(r["avg_cpu_percent"] or 0.0),
                "max_cpu_percent": float(r["max_cpu_percent"] or 0.0),
                "avg_mem_usage_bytes": float(r["avg_mem_usage_bytes"] or 0.0),
                "max_mem_usage_bytes": float(r["max_mem_usage_bytes"] or 0.0),
                "avg_mem_percent": float(r["avg_mem_percent"] or 0.0),
                "max_mem_percent": float(r["max_mem_percent"] or 0.0),
                "sample_rows": int(r["sample_rows"] or 0),
            }
        )
    return out


def _query_docker_timeseries(
    con: sqlite3.Connection, *, start_ts: float, keys: Sequence[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Returns:
      {key: {"x": [...], "cpu_percent": [...], "mem_usage_bytes": [...], "mem_percent": [...]} }
    """
    if not keys:
        return {}
    if not _table_exists(con, "docker_container_samples"):
        return {}
    cur = con.cursor()

    placeholders = ",".join("?" for _ in keys)
    q = f"""
        SELECT
          s.ts_unix AS ts_unix,
          d.container_id AS key,
          d.cpu_percent AS cpu_percent,
          d.mem_usage_bytes AS mem_usage_bytes,
          d.mem_percent AS mem_percent
        FROM docker_container_samples d
        JOIN samples s ON s.id = d.sample_id
        WHERE s.ts_unix >= ?
          AND d.container_id IN ({placeholders})
        ORDER BY s.ts_unix ASC, d.container_id ASC
    """
    rows = cur.execute(q, (start_ts, *list(keys))).fetchall()

    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        k = (r["key"] or "").strip()
        if not k:
            continue
        d = out.setdefault(
            k, {"x": [], "cpu_percent": [], "mem_usage_bytes": [], "mem_percent": []}
        )
        d["x"].append(_ts_to_iso(float(r["ts_unix"])))
        d["cpu_percent"].append(float(r["cpu_percent"]) if r["cpu_percent"] is not None else None)
        d["mem_usage_bytes"].append(
            float(r["mem_usage_bytes"]) if r["mem_usage_bytes"] is not None else None
        )
        d["mem_percent"].append(float(r["mem_percent"]) if r["mem_percent"] is not None else None)
    return out


def _query_top_process_per_sample(con: sqlite3.Connection, *, start_ts: float) -> List[Dict[str, Any]]:
    """
    Best-effort: process_samples only stores "offenders", so this finds the max cpu process among recorded ones.
    """
    cur = con.cursor()
    rows = cur.execute(
        """
        WITH ranked AS (
          SELECT
            p.sample_id AS sample_id,
            s.ts_unix AS ts_unix,
            s.interval_s AS interval_s,
            p.pid AS pid,
            p.name AS name,
            p.username AS username,
            p.cmdline AS cmdline,
            p.cpu_percent AS cpu_percent,
            p.rss_bytes AS rss_bytes,
            p.gpu_mem_mb AS gpu_mem_mb,
            ROW_NUMBER() OVER (PARTITION BY p.sample_id ORDER BY p.cpu_percent DESC NULLS LAST) AS rn
          FROM process_samples p
          JOIN samples s ON s.id = p.sample_id
          WHERE s.ts_unix >= ?
        )
        SELECT * FROM ranked WHERE rn = 1 ORDER BY ts_unix ASC
        """,
        (start_ts,),
    ).fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        raw_cpu = float(r["cpu_percent"]) if r["cpu_percent"] is not None else 0.0
        cpu_count = os.cpu_count() or 1
        cpu_total = raw_cpu / float(cpu_count) if cpu_count > 0 else raw_cpu
        out.append(
            {
                "sample_id": int(r["sample_id"]) if r["sample_id"] is not None else None,
                "ts_unix": float(r["ts_unix"]),
                "ts": _ts_to_iso(float(r["ts_unix"])),
                "interval_s": float(r["interval_s"] or 0.0),
                "pid": int(r["pid"]) if r["pid"] is not None else None,
                "name": r["name"] or "",
                "username": r["username"] or "",
                "cmdline": r["cmdline"] or "",
                # `psutil.Process.cpu_percent()` can be >100 on multi-core machines.
                # Normalize to "percent of total machine capacity" (0-100) for plotting alongside system CPU%.
                "cpu_percent": raw_cpu,
                "cpu_percent_total": cpu_total,
                "rss_mb": (float(r["rss_bytes"]) / 1024.0 / 1024.0) if r["rss_bytes"] is not None else 0.0,
                "gpu_mem_mb": float(r["gpu_mem_mb"]) if r["gpu_mem_mb"] is not None else 0.0,
            }
        )
    return out


def _median_mad(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    med = statistics.median(values)
    abs_dev = [abs(v - med) for v in values]
    mad = statistics.median(abs_dev) if abs_dev else 0.0
    return float(med), float(mad)


def _detect_cpu_spikes(
    top_proc: List[Dict[str, Any]],
    *,
    min_cpu_percent: float,
    sigma: float,
    max_spikes: int,
) -> List[Dict[str, Any]]:
    # Robust threshold using MAD.
    # Use normalized "percent of total machine capacity" so spikes align with the system CPU% chart scale.
    vals = [float(p.get("cpu_percent_total") or 0.0) for p in top_proc]
    med, mad = _median_mad(vals)
    # Scale MAD -> approx stddev for normal dist.
    robust_std = 1.4826 * mad
    # Backwards-compatible interpretation:
    # `--min-cpu-spike` is specified in "raw per-process CPU%" units, where 100% ~= 1 core.
    # Convert to "percent of total machine capacity" for our normalized series.
    cpu_count = os.cpu_count() or 1
    min_cpu_total = float(min_cpu_percent) / float(cpu_count) if cpu_count > 0 else float(
        min_cpu_percent
    )
    thresh = max(min_cpu_total, med + float(sigma) * robust_std)

    spikes: List[Dict[str, Any]] = []
    for i, p in enumerate(top_proc):
        v = float(p.get("cpu_percent_total") or 0.0)
        if v < thresh:
            continue
        # Local max guard to reduce flat "plateaus"
        prev_v = float(top_proc[i - 1].get("cpu_percent_total") or 0.0) if i > 0 else -1.0
        next_v = float(top_proc[i + 1].get("cpu_percent_total") or 0.0) if i + 1 < len(top_proc) else -1.0
        if v < prev_v or v < next_v:
            continue
        spikes.append(
            {
                "ts": p["ts"],
                "ts_unix": p["ts_unix"],
                "cpu_percent_total": v,
                "cpu_percent": float(p.get("cpu_percent") or 0.0),
                "pid": p.get("pid"),
                "name": p.get("name"),
                "username": p.get("username"),
                "cmdline": p.get("cmdline"),
                "rss_mb": float(p.get("rss_mb") or 0.0),
                "gpu_mem_mb": float(p.get("gpu_mem_mb") or 0.0),
            }
        )

    # Keep the most significant spikes
    spikes.sort(key=lambda x: x["cpu_percent"], reverse=True)
    spikes = spikes[: int(max_spikes)]
    # For nicer plotting/table order, return chronological
    spikes.sort(key=lambda x: x["ts_unix"])
    return spikes


def _query_cpu_leaderboard(con: sqlite3.Connection, *, start_ts: float, limit: int) -> List[Dict[str, Any]]:
    cur = con.cursor()
    rows = cur.execute(
        """
        WITH window AS (
          SELECT id, interval_s
          FROM samples
          WHERE ts_unix >= ?
        ), agg AS (
          SELECT
            COALESCE(NULLIF(p.cmdline,''), p.name, 'unknown') AS key,
            MAX(p.name) AS name,
            MAX(p.username) AS username,
            MIN(p.pid) AS example_pid,
            SUM(COALESCE(p.cpu_percent,0) / 100.0 * w.interval_s) AS cpu_core_seconds,
            AVG(COALESCE(p.cpu_percent,0)) AS avg_cpu_percent,
            MAX(COALESCE(p.cpu_percent,0)) AS max_cpu_percent,
            COUNT(*) AS sample_rows
          FROM process_samples p
          JOIN window w ON w.id = p.sample_id
          GROUP BY key
        )
        SELECT * FROM agg
        ORDER BY cpu_core_seconds DESC
        LIMIT ?
        """,
        (start_ts, int(limit)),
    ).fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "name": r["name"] or "",
                "username": r["username"] or "",
                "example_pid": int(r["example_pid"]) if r["example_pid"] is not None else None,
                "cpu_core_seconds": float(r["cpu_core_seconds"] or 0.0),
                "avg_cpu_percent": float(r["avg_cpu_percent"] or 0.0),
                "max_cpu_percent": float(r["max_cpu_percent"] or 0.0),
                "sample_rows": int(r["sample_rows"] or 0),
                "cmdline": r["key"] or "",
            }
        )
    return out


def _build_html(payload: Dict[str, Any]) -> str:
    # Plotly from CDN keeps this script dependency-light.
    # (If you prefer fully-offline, we can embed plotly.min.js, but it’s large.)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{payload.get("title","Resource Report")}</title>
  <link rel="icon" href="{_favicon_data_url()}" type="image/svg+xml" />
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    :root {{
      --bg0: #0b1220;
      --bg1: #0e1730;
      --card: rgba(255,255,255,0.06);
      --card2: rgba(255,255,255,0.08);
      --text: rgba(255,255,255,0.92);
      --muted: rgba(255,255,255,0.68);
      --grid: rgba(255,255,255,0.09);
      --accent: #6ee7ff;
      --accent2: #7c3aed;
      --good: #34d399;
      --warn: #fbbf24;
      --bad: #fb7185;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
    }}
    /* Prevent horizontal overflow caused by 100%-width containers + padding. */
    *, *::before, *::after {{
      box-sizing: border-box;
    }}
    html, body {{
      height: 100%;
      margin: 0;
      overflow-x: hidden;
      background: radial-gradient(1200px 700px at 10% 10%, rgba(124,58,237,0.22), transparent 60%),
                  radial-gradient(1000px 700px at 90% 20%, rgba(110,231,255,0.18), transparent 60%),
                  linear-gradient(180deg, var(--bg0), var(--bg1));
      color: var(--text);
      font-family: var(--sans);
    }}
    .wrap {{
      /* Let charts use the full browser width (no fixed max). */
      max-width: none;
      width: 100%;
      margin: 0;
      padding: 24px 18px 60px 18px;
    }}
    .hero {{
      display: flex;
      gap: 16px;
      align-items: flex-end;
      justify-content: space-between;
      padding: 18px 18px;
      border-radius: 16px;
      background: linear-gradient(135deg, rgba(124,58,237,0.16), rgba(110,231,255,0.10));
      border: 1px solid rgba(255,255,255,0.10);
      box-shadow: 0 10px 40px rgba(0,0,0,0.35);
    }}
    .hero h1 {{
      margin: 0;
      font-size: 22px;
      letter-spacing: 0.2px;
    }}
    .hero .brand {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
    }}
    .hero .brand img {{
      width: 22px;
      height: 22px;
    }}
    .meta {{
      font-size: 12px;
      color: var(--muted);
      font-family: var(--mono);
      line-height: 1.6;
      text-align: right;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1.4fr 0.6fr;
      gap: 16px;
      margin-top: 16px;
    }}
    @media (max-width: 1100px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .meta {{ text-align: left; }}
    }}
    .card {{
      background: var(--card);
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }}
    /* Plotly hover labels can extend outside the plot area; don't clip them. */
    .card.chart {{
      overflow: visible;
      display: flex;
      flex-direction: column;
    }}
    .card .hdr {{
      display: flex;
      gap: 10px;
      justify-content: space-between;
      align-items: center;
      padding: 12px 14px;
      background: rgba(255,255,255,0.04);
      border-bottom: 1px solid rgba(255,255,255,0.08);
    }}
    .card .hdr .title {{
      font-weight: 650;
      font-size: 13px;
      letter-spacing: 0.2px;
      color: rgba(255,255,255,0.88);
    }}
    .card .hdr .hint {{
      font-size: 12px;
      color: var(--muted);
      font-family: var(--mono);
    }}
    .card .body {{
      padding: 12px 12px 6px 12px;
      flex: 1 1 auto;
      min-height: 0; /* allow children to flex without overflowing */
    }}
    /* Plot containers: scale with the viewport height instead of a fixed px height. */
    .plot {{
      width: 100%;
      height: 72vh;
      min-height: 520px;
    }}
    @media (max-width: 1100px) {{
      .plot {{
        height: 62vh;
        min-height: 420px;
      }}
    }}
    .pill {{
      display: inline-flex;
      gap: 8px;
      align-items: center;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.06);
      font-size: 12px;
      color: rgba(255,255,255,0.85);
      font-family: var(--mono);
      user-select: none;
    }}
    .table {{
      width: 100%;
      border-collapse: collapse;
      font-family: var(--mono);
      font-size: 12px;
    }}
    .table th, .table td {{
      padding: 8px 8px;
      border-bottom: 1px solid rgba(255,255,255,0.08);
      vertical-align: top;
    }}
    .table th {{
      text-align: left;
      color: rgba(255,255,255,0.75);
      font-weight: 650;
      position: sticky;
      top: 0;
      background: rgba(11,18,32,0.75);
      backdrop-filter: blur(8px);
      z-index: 2;
    }}
    .k {{
      color: rgba(255,255,255,0.82);
      font-weight: 650;
    }}
    .small {{
      color: var(--muted);
      font-size: 11px;
      line-height: 1.35;
    }}
    .badge {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-family: var(--mono);
      font-size: 11px;
      border: 1px solid rgba(255,255,255,0.14);
      background: rgba(255,255,255,0.06);
      color: rgba(255,255,255,0.88);
      margin-right: 6px;
    }}
    .badge.bad {{ border-color: rgba(251,113,133,0.40); background: rgba(251,113,133,0.08); }}
    .badge.warn {{ border-color: rgba(251,191,36,0.38); background: rgba(251,191,36,0.08); }}
    .badge.good {{ border-color: rgba(52,211,153,0.38); background: rgba(52,211,153,0.08); }}
    .mono {{ font-family: var(--mono); }}
    .muted {{ color: var(--muted); }}

    /* Hover label delay:
       Plotly doesn't support a built-in hover delay, so we keep the hover layer hidden by default
       and reveal it after a timeout once the cursor is stable on a point. */
    .hover-delay .hoverlayer {{
      opacity: 0;
      transition: opacity 70ms linear;
    }}
    .hover-delay.show-hover .hoverlayer {{
      opacity: 1;
    }}

    /* Spike table row hover/selection */
    .table tr.spike-row {{
      cursor: pointer;
    }}
    .table tr.spike-row:hover td {{
      background: rgba(255,255,255,0.04);
    }}
    .table tr.spike-row.selected td {{
      background: rgba(251,191,36,0.16);
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div>
        <h1 class="brand">
          <img src="{_favicon_data_url()}" alt="" />
          {payload.get("title","Resource Report")}
        </h1>
        <div class="small" style="margin-top:6px;">
          Interactive charts: zoom / pan / range buttons. Use the x-axis range selector to jump to <span class="k">1d</span> or <span class="k">7d</span>.
        </div>
      </div>
      <div class="meta">
        <div>DB: {payload.get("db_path","")}</div>
        <div>Window: {payload.get("window","")}</div>
        <div>Samples: {payload.get("sample_count",0)} (avg interval: {payload.get("avg_interval_s",0):.2f}s)</div>
      </div>
    </div>

    <div class="grid">
      <div class="card chart">
        <div class="hdr">
          <div class="title">System: CPU / Memory / IO</div>
          <div class="hint">Hover to inspect; drag to zoom; double-click to reset</div>
        </div>
        <div class="body">
          <div id="sys_graph" class="plot"></div>
        </div>
      </div>

      <div style="display:flex; flex-direction:column; gap:16px;">
        <div class="card">
          <div class="hdr">
            <div class="title">Spike events (top process CPU)</div>
            <div class="hint"><span class="pill">MAD threshold</span></div>
          </div>
          <div class="body" style="max-height: 270px; overflow:auto;">
            <table class="table" id="spike_table"></table>
          </div>
        </div>

        <div class="card">
          <div class="hdr">
            <div class="title">Disk read spikes (best-effort)</div>
            <div class="hint"><span class="pill">MAD threshold</span></div>
          </div>
          <div class="body" style="max-height: 270px; overflow:auto;">
            <table class="table" id="disk_spike_table"></table>
          </div>
        </div>
      </div>
    </div>

    <!-- Match the System grid column ratio so the GPU chart has the same width. -->
    <div class="grid" style="grid-template-columns: 1.4fr 0.6fr; margin-top: 16px;">
      <div class="card chart">
        <div class="hdr">
          <div class="title">GPU: Util / Memory / Temp / Power</div>
          <div class="hint">If no GPU samples exist, this stays empty</div>
        </div>
        <div class="body">
          <div id="gpu_graph" class="plot"></div>
        </div>
      </div>

      <div class="card">
        <div class="hdr">
          <div class="title">CPU leaderboard (integrated)</div>
          <div class="hint"><span class="pill">∑ cpu% × interval</span></div>
        </div>
        <div class="body" style="max-height: 560px; overflow:auto;">
          <table class="table" id="leader_table"></table>
        </div>
      </div>
    </div>

    <!-- Containers: match the same chart/table width ratio as System/GPU (this is the 3rd chart) -->
    <div class="grid" style="grid-template-columns: 1.4fr 0.6fr; margin-top: 16px;">
      <div class="card chart">
        <div class="hdr">
          <div class="title">Containers: CPU / Memory</div>
          <div class="hint">From `docker stats --no-stream` (best-effort). Solid=mem GiB, dashed=CPU%.</div>
        </div>
        <div class="body">
          <div id="docker_graph" class="plot"></div>
        </div>
      </div>

      <div class="card">
        <div class="hdr">
          <div class="title">Container summary</div>
          <div class="hint"><span class="pill">avg / max</span></div>
        </div>
        <div class="body" style="max-height: 560px; overflow:auto;">
          <table class="table" id="docker_table"></table>
        </div>
      </div>
    </div>

    <!-- Ping: match the same chart/table width ratio as System/GPU -->
    <div class="grid" style="grid-template-columns: 1.4fr 0.6fr; margin-top: 16px;">
      <div class="card chart">
        <div class="hdr">
          <div class="title">Ping: RTT / availability</div>
          <div class="hint">Lines = RTT (ms). X markers at 0ms = ping failure.</div>
        </div>
        <div class="body">
          <div id="ping_graph" class="plot"></div>
        </div>
      </div>

      <div class="card">
        <div class="hdr">
          <div class="title">Ping summary</div>
          <div class="hint"><span class="pill">loss% / avg / p95</span></div>
        </div>
        <div class="body" style="max-height: 560px; overflow:auto;">
          <table class="table" id="ping_table"></table>
        </div>
      </div>
    </div>

    <!-- Network top talkers: match same width ratio -->
    <div class="grid" style="grid-template-columns: 1.4fr 0.6fr; margin-top: 16px;">
      <div class="card chart">
        <div class="hdr">
          <div class="title">Network: top talkers (best-effort)</div>
          <div class="hint">Uses monitor’s `--net-top` sampling (typically top-K only). Lines = (sent+recv) MiB/s.</div>
        </div>
        <div class="body">
          <div id="net_graph" class="plot"></div>
        </div>
      </div>

      <div class="card">
        <div class="hdr">
          <div class="title">Network offenders (aggregated)</div>
          <div class="hint"><span class="pill">avg / max</span></div>
        </div>
        <div class="body" style="max-height: 560px; overflow:auto;">
          <table class="table" id="net_table"></table>
        </div>
      </div>
    </div>
  </div>

  <script>
  const PAYLOAD = {_json(payload)};
  // Plotly line dash styles (SVG stroke-dasharray presets):
  // 'solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot'
  const DASH = {{
    solid: 'solid',
    dot: 'dot',
    dash: 'dash',
    longdash: 'longdash',
    dashdot: 'dashdot',
    longdashdot: 'longdashdot',
  }};

  function bytesToGiB(b) {{
    if (b === null || b === undefined) return null;
    return b / 1024 / 1024 / 1024;
  }}

  function bpsToMiBps(x) {{
    if (x === null || x === undefined) return null;
    return x / 1024 / 1024;
  }}

  function wrapLine(s, width) {{
    if (!s) return '';
    const w = Math.max(10, width || 80);
    // Hard-wrap long unbroken strings so hover labels don't get clipped.
    const out = [];
    for (let i = 0; i < s.length; i += w) {{
      out.push(s.slice(i, i + w));
    }}
    return out.join('<br>');
  }}

  function buildSpikeTable(spikes) {{
    const el = document.getElementById('spike_table');
    if (!spikes || spikes.length === 0) {{
      el.innerHTML = `<tr><th>status</th></tr><tr><td class="muted">No spikes detected (or process data missing in window).</td></tr>`;
      return;
    }}
    // Show latest first (top of the table), but keep data-idx pointing to the original spike index
    // so table↔plot highlighting still works.
    const rev = spikes.slice().reverse();
    el.innerHTML = `
      <tr>
        <th>time</th>
        <th>cpu%</th>
        <th>proc</th>
        <th>rss</th>
        <th>gpu</th>
      </tr>
      ${{rev.map((s, j) => `
        <tr class="spike-row" data-idx="${{(spikes.length - 1 - j)}}">
          <td class="mono">${{s.ts.replace('T',' ')}}</td>
          <td>
            <span class="badge bad">${{(s.cpu_percent_total ?? s.cpu_percent).toFixed(1)}}%</span>
            <div class="small muted mono" title="raw per-process cpu% (can exceed 100 on multi-core)">
              raw: ${{(s.cpu_percent ?? 0).toFixed(1)}}%
            </div>
          </td>
          <td>
            <div class="mono">${{(s.name||'').slice(0,22)}} <span class="muted">#${{s.pid ?? ''}}</span></div>
            <div class="small muted">${{(s.cmdline||'').slice(0,90)}}</div>
          </td>
          <td class="mono">${{(s.rss_mb||0).toFixed(0)}} MB</td>
          <td class="mono">${{(s.gpu_mem_mb||0).toFixed(0)}} MB</td>
        </tr>
      `).join('')}}
    `;
  }}

  function buildDiskSpikeTable(rows) {{
    const el = document.getElementById('disk_spike_table');
    if (!rows || rows.length === 0) {{
      el.innerHTML = `<tr><th>status</th></tr><tr><td class="muted">No disk read spikes detected (or missing disk/process data).</td></tr>`;
      return;
    }}
    const rev = rows.slice().reverse();
    el.innerHTML = `
      <tr>
        <th>time</th>
        <th>read</th>
        <th>proc</th>
      </tr>
      ${{rev.map((r, j) => `
        <tr class="spike-row disk-spike-row" data-idx="${{(rows.length - 1 - j)}}">
          <td class="mono">${{r.ts.replace('T',' ')}}</td>
          <td class="mono"><span class="badge warn">${{(r.disk_read_mibps ?? 0).toFixed(2)}}</span> MiB/s</td>
          <td>
            <div class="mono">${{(r.proc_name||'').slice(0,22)}} <span class="muted">#${{r.pid ?? ''}}</span></div>
            <div class="small muted">${{(r.cmdline||'').slice(0,90)}}</div>
          </td>
        </tr>
      `).join('')}}
    `;
  }}

  function buildLeaderTable(rows) {{
    const el = document.getElementById('leader_table');
    if (!rows || rows.length === 0) {{
      el.innerHTML = `<tr><th>status</th></tr><tr><td class="muted">No process samples in window.</td></tr>`;
      return;
    }}
    el.innerHTML = `
      <tr>
        <th>cpu_core_s</th>
        <th>avg%</th>
        <th>max%</th>
        <th>process</th>
      </tr>
      ${{rows.map(r => `
        <tr>
          <td class="mono"><span class="badge warn">${{r.cpu_core_seconds.toFixed(1)}}</span></td>
          <td class="mono">${{r.avg_cpu_percent.toFixed(2)}}</td>
          <td class="mono">${{r.max_cpu_percent.toFixed(1)}}</td>
          <td>
            <div class="mono">${{(r.name||'').slice(0,22)}} <span class="muted">@${{(r.username||'')}}</span></div>
            <div class="small muted">${{(r.cmdline||'').slice(0,92)}}</div>
          </td>
        </tr>
      `).join('')}}
    `;
  }}

  function buildPingTable(rows) {{
    const el = document.getElementById('ping_table');
    if (!rows || rows.length === 0) {{
      el.innerHTML = `<tr><th>status</th></tr><tr><td class="muted">No ping samples in window.</td></tr>`;
      return;
    }}
    el.innerHTML = `
      <tr>
        <th>target</th>
        <th>loss%</th>
        <th>avg</th>
        <th>p95</th>
        <th>n</th>
      </tr>
      ${{rows.map(r => `
        <tr>
          <td class="mono">${{r.target}}</td>
          <td class="mono"><span class="badge ${{(r.loss_pct ?? 0) > 5 ? 'bad' : ((r.loss_pct ?? 0) > 0 ? 'warn' : 'good')}}">${{(r.loss_pct ?? 0).toFixed(1)}}%</span></td>
          <td class="mono">${{(r.avg_ms ?? 0).toFixed(1)}} ms</td>
          <td class="mono">${{(r.p95_ms ?? 0).toFixed(1)}} ms</td>
          <td class="mono">${{r.ok_n}}/${{r.n}}</td>
        </tr>
      `).join('')}}
    `;
  }}

  function buildNetTable(rows) {{
    const el = document.getElementById('net_table');
    if (!rows || rows.length === 0) {{
      el.innerHTML = `<tr><th>status</th></tr><tr><td class="muted">No net_process_samples in window (enable monitor with --net-top).</td></tr>`;
      return;
    }}
    el.innerHTML = `
      <tr>
        <th>process</th>
        <th>avg</th>
        <th>max</th>
        <th>rows</th>
      </tr>
      ${{rows.map(r => `
        <tr>
          <td>
            <div class="mono">${{(r.proc||'').slice(0,40)}}</div>
            <div class="small muted">${{(r.proc||'').slice(0,120)}}</div>
          </td>
          <td class="mono"><span class="badge warn">${{bpsToMiBps(r.avg_bps).toFixed(2)}}</span> MiB/s</td>
          <td class="mono">${{bpsToMiBps(r.max_bps).toFixed(2)}} MiB/s</td>
          <td class="mono">${{r.sample_rows}}</td>
        </tr>
      `).join('')}}
    `;
  }}

  function buildDockerTable(rows) {{
    const el = document.getElementById('docker_table');
    if (!rows || rows.length === 0) {{
      el.innerHTML = `<tr><th>status</th></tr><tr><td class="muted">No docker_container_samples in window (enable monitor with --docker-stats).</td></tr>`;
      return;
    }}
    el.innerHTML = `
      <tr>
        <th>container id</th>
        <th>image</th>
        <th>cpu avg/max</th>
        <th>mem avg/max</th>
        <th>rows</th>
      </tr>
      ${{rows.map(r => `
        <tr>
          <td class="mono">${{(r.container_id || '').slice(0,12)}}</td>
          <td>
            <div class="mono">${{(r.image || '').slice(0,36)}}</div>
            <div class="small muted">${{(r.image || '').slice(0,90)}}</div>
          </td>
          <td class="mono"><span class="badge warn">${{(r.avg_cpu_percent ?? 0).toFixed(1)}}%</span> / ${{(r.max_cpu_percent ?? 0).toFixed(1)}}%</td>
          <td class="mono"><span class="badge warn">${{bytesToGiB(r.avg_mem_usage_bytes ?? 0).toFixed(2)}}</span> / ${{bytesToGiB(r.max_mem_usage_bytes ?? 0).toFixed(2)}} GiB</td>
          <td class="mono">${{r.sample_rows}}</td>
        </tr>
      `).join('')}}
    `;
  }}

  function commonLayout(title) {{
    const grid = 'rgba(255,255,255,0.09)';
    // Default initial view: current last 1 hour (ending at "now").
    // (Not "last sample timestamp" — that can be stale and land on e.g. the last hour of a day.)
    // Users can zoom out via range selector buttons (15m/30m/1h/6h/...).
    let defaultStart = null;
    let defaultEnd = null;
    try {{
      // Important: sample timestamps are emitted as "localtime ISO without timezone".
      // If we use toISOString() (UTC + Z), Plotly can interpret the range in a different timezone
      // than the data, causing a shifted initial view (e.g. "starts at 3AM").
      function toLocalIsoSeconds(d) {{
        const pad = (n) => String(n).padStart(2, '0');
        return `${{d.getFullYear()}}-${{pad(d.getMonth()+1)}}-${{pad(d.getDate())}}T${{pad(d.getHours())}}:${{pad(d.getMinutes())}}:${{pad(d.getSeconds())}}`;
      }}
      const end = new Date(); // now (viewer local time)
      const start = new Date(end.getTime() - 60 * 60 * 1000);
      defaultEnd = toLocalIsoSeconds(end);
      defaultStart = toLocalIsoSeconds(start);
    }} catch (e) {{
      // ignore
    }}
    // The card header already provides a title; keep the Plotly title empty to avoid overlapping the legend.
    return {{
      title: {{ text: '', font: {{ size: 13, color: 'rgba(255,255,255,0.85)' }} }},
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      // Give extra headroom for the horizontal legend and extra right-side axes.
      // This prevents the GPU plot from looking "shorter" when its legend wraps to multiple rows.
      margin: {{ l: 55, r: 90, t: 92, b: 40 }},
      hovermode: 'x unified',
      // Our page uses a light hover box by default; since layout font is light, it can look "empty".
      // Force a dark hoverlabel so values are always readable.
      hoverlabel: {{
        bgcolor: 'rgba(15, 23, 42, 0.96)',
        bordercolor: 'rgba(255,255,255,0.18)',
        font: {{
          color: 'rgba(255,255,255,0.92)',
          family: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace',
          size: 12
        }}
      }},
      font: {{ color: 'rgba(255,255,255,0.85)', family: 'ui-sans-serif, system-ui' }},
      xaxis: {{
        type: 'date',
        gridcolor: grid,
        zeroline: false,
        range: (defaultStart && defaultEnd) ? [defaultStart, defaultEnd] : undefined,
        rangeselector: {{
          bgcolor: 'rgba(255,255,255,0.06)',
          bordercolor: 'rgba(255,255,255,0.12)',
          borderwidth: 1,
          buttons: [
            {{count: 15, label: '15m', step: 'minute', stepmode: 'backward'}},
            {{count: 30, label: '30m', step: 'minute', stepmode: 'backward'}},
            {{count: 1, label: '1h', step: 'hour', stepmode: 'backward'}},
            {{count: 6, label: '6h', step: 'hour', stepmode: 'backward'}},
            {{count: 12, label: '12h', step: 'hour', stepmode: 'backward'}},
            {{count: 1, label: '1d', step: 'day', stepmode: 'backward'}},
            {{count: 7, label: '7d', step: 'day', stepmode: 'backward'}},
            {{step: 'all', label: 'all'}}
          ],
        }},
        rangeslider: {{ visible: true, thickness: 0.08, bgcolor: 'rgba(255,255,255,0.03)' }},
      }},
      legend: {{
        orientation: 'h',
        y: 1.24,
        x: 0.0,
        yanchor: 'bottom',
        font: {{ size: 11 }},
        itemwidth: 110,
      }},
    }};
  }}

  function renderSystemGraph() {{
    const s = PAYLOAD.samples;
    const x = s.map(r => r.ts);
    const cpu = s.map(r => r.cpu_percent);
    const mem = s.map(r => r.mem_percent);
    const load1 = s.map(r => r.load1);
    const gpuUtil = s.map(r => r.gpu_util_max);
    const gpuMemPct = s.map(r => r.gpu_mem_percent_total);
    const netOut = s.map(r => bpsToMiBps(r.net_sent_bps));
    const netIn = s.map(r => bpsToMiBps(r.net_recv_bps));
    const diskR = s.map(r => bpsToMiBps(r.disk_read_bps));
    const diskW = s.map(r => bpsToMiBps(r.disk_write_bps));

    const spikes = PAYLOAD.spikes || [];
    const spikeX = spikes.map(e => e.ts);
    const spikeY = spikes.map(e => (e.cpu_percent_total ?? e.cpu_percent));
    const spikeText = spikes.map(e =>
      `${{e.name || ''}} #${{e.pid ?? ''}}` +
      `<br>raw cpu%: ${{(e.cpu_percent ?? 0).toFixed(1)}}` +
      `<br>${{wrapLine((e.cmdline||'').slice(0,240), 80)}}`
    );

    const traces = [
      {{ x, y: cpu, name: 'CPU % (system)', mode: 'lines', line: {{ color: '#6ee7ff', width: 2, dash: DASH.solid }} , yaxis: 'y1' }},
      {{ x, y: mem, name: 'MEM % (system)', mode: 'lines', line: {{ color: '#a78bfa', width: 2, dash: DASH.solid }} , yaxis: 'y1' }},
      // GPU overlays on the same 0-100% axis; use distinct dash styles so they don't blend.
      {{ x, y: gpuUtil, name: 'GPU util % (max)', mode: 'lines', line: {{ color: '#f97316', width: 1.8, dash: DASH.longdash }} , yaxis: 'y1' }},
      {{ x, y: gpuMemPct, name: 'GPU mem % (total)', mode: 'lines', line: {{ color: '#fb7185', width: 1.6, dash: DASH.dashdot }} , yaxis: 'y1' }},
      {{ x, y: load1, name: 'load1', mode: 'lines', line: {{ color: 'rgba(255,255,255,0.45)', width: 1.6, dash: DASH.dot }} , yaxis: 'y2' }},
      {{ x, y: netOut, name: 'net out (MiB/s)', mode: 'lines', line: {{ color: '#34d399', width: 1.6, dash: DASH.solid }} , yaxis: 'y3' }},
      {{ x, y: netIn, name: 'net in (MiB/s)', mode: 'lines', line: {{ color: '#22c55e', width: 1.6, dash: DASH.dot }} , yaxis: 'y3' }},
      {{ x, y: diskR, name: 'disk read (MiB/s)', mode: 'lines', line: {{ color: '#fbbf24', width: 1.6, dash: DASH.solid }} , yaxis: 'y4' }},
      // Use a different color than GPU mem% (also pink) to avoid visual confusion.
      {{ x, y: diskW, name: 'disk write (MiB/s)', mode: 'lines', line: {{ color: '#38bdf8', width: 1.6, dash: DASH.longdashdot }} , yaxis: 'y4' }},
    ];

    const diskSpikes = PAYLOAD.disk_spikes || [];
    const diskSpikeX = diskSpikes.map(e => e.ts);
    const diskSpikeY = diskSpikes.map(e => e.disk_read_mibps);
    const diskSpikeText = diskSpikes.map(e =>
      `${{e.proc_name || ''}} #${{e.pid ?? ''}}` +
      `<br>proc read: ${{(e.proc_io_read_mibps ?? 0).toFixed(2)}} MiB/s` +
      `<br>${{wrapLine((e.cmdline||'').slice(0,240), 80)}}`
    );

    if (spikes.length > 0) {{
      // Plotly has no native marker shadow/glow; simulate with a "glow" marker trace behind the main trace.
      const spikeGlowTraceIndex = traces.length;
      traces.push({{
        x: spikeX,
        y: spikeY,
        name: '',
        mode: 'markers',
        marker: {{ size: 18, color: 'rgba(251,113,133,0.22)', line: {{ width: 0 }} }},
        hoverinfo: 'skip',
        showlegend: false,
        yaxis: 'y1'
      }});

      const spikeTraceIndex = traces.length;
      traces.push({{
        x: spikeX,
        y: spikeY,
        name: 'Top-proc CPU spike',
        mode: 'markers',
        marker: {{ size: 10, color: '#fb7185', line: {{ width: 1, color: 'rgba(255,255,255,0.75)' }} }},
        text: spikeText,
        hovertemplate: '<b>Spike</b><br>%{{x}}<br>top-proc cpu (of total)=%{{y:.1f}}%<br>%{{text}}<extra></extra>',
        yaxis: 'y1'
      }});
      // Expose mapping so the spike table can highlight points on hover/click.
      window.__SPIKE_GLOW_TRACE_INDEX = spikeGlowTraceIndex;
      window.__SPIKE_TRACE_INDEX = spikeTraceIndex;
      window.__SPIKE_COUNT = spikeX.length;
      window.__SPIKE_DEFAULT_COLOR = '#fb7185';
      // Use a high-contrast highlight (yellow) so it’s obvious.
      window.__SPIKE_HILITE_COLOR = '#fbbf24';
      window.__SPIKE_DEFAULT_SIZE = 10;
      window.__SPIKE_HILITE_SIZE = 14;
      window.__SPIKE_GLOW_DEFAULT_COLOR = 'rgba(251,113,133,0.22)';
      window.__SPIKE_GLOW_HILITE_COLOR = 'rgba(251,191,36,0.30)';
      window.__SPIKE_GLOW_DEFAULT_SIZE = 18;
      window.__SPIKE_GLOW_HILITE_SIZE = 24;
    }}

    if (diskSpikes.length > 0) {{
      const diskSpikeTraceIndex = traces.length;
      traces.push({{
        x: diskSpikeX,
        y: diskSpikeY,
        name: 'Disk read spike',
        mode: 'markers',
        marker: {{ size: 10, color: '#fbbf24', symbol: 'diamond', line: {{ width: 1, color: 'rgba(255,255,255,0.75)' }} }},
        text: diskSpikeText,
        hovertemplate: '<b>Disk read spike</b><br>%{{x}}<br>disk read=%{{y:.2f}} MiB/s<br>%{{text}}<extra></extra>',
        yaxis: 'y4'
      }});
      window.__DISK_SPIKE_TRACE_INDEX = diskSpikeTraceIndex;
      window.__DISK_SPIKE_COUNT = diskSpikeX.length;
      window.__DISK_SPIKE_DEFAULT_COLOR = '#fbbf24';
      window.__DISK_SPIKE_HILITE_COLOR = '#fde047';
      window.__DISK_SPIKE_DEFAULT_SIZE = 10;
      window.__DISK_SPIKE_HILITE_SIZE = 14;
    }}

    const layout = commonLayout('');
    layout.yaxis = {{ title: 'CPU% / MEM%', range: [0, 100], gridcolor: 'rgba(255,255,255,0.09)' }};
    // Push right-side axes *outside* the plot area to prevent tick-label pileups.
    // Keep tick labels small; hover shows exact values anyway.
    layout.yaxis2 = {{
      title: 'load1',
      overlaying: 'y',
      side: 'right',
      anchor: 'free',
      position: 1.00,
      showgrid: false,
      tickfont: {{ size: 10 }},
      titlefont: {{ size: 10 }},
      automargin: true,
    }};
    layout.yaxis3 = {{
      title: 'net MiB/s',
      overlaying: 'y',
      side: 'right',
      anchor: 'free',
      position: 1.06,
      showgrid: false,
      tickfont: {{ size: 10 }},
      titlefont: {{ size: 10 }},
      automargin: true,
    }};
    layout.yaxis4 = {{
      title: 'disk MiB/s',
      overlaying: 'y',
      side: 'right',
      anchor: 'free',
      position: 1.12,
      showgrid: false,
      tickfont: {{ size: 10 }},
      titlefont: {{ size: 10 }},
      automargin: true,
    }};

    const config = {{ displayModeBar: true, responsive: true }};
    return Plotly.newPlot('sys_graph', traces, layout, config).then(() => {{
      installHoverDelay('sys_graph');
      installSpikeTableLinkage();
      installDiskSpikeTableLinkage();
    }});
  }}

  function installDiskSpikeTableLinkage() {{
    const table = document.getElementById('disk_spike_table');
    const plot = document.getElementById('sys_graph');
    if (!table || !plot) return;
    const traceIndex = window.__DISK_SPIKE_TRACE_INDEX;
    const spikeCount = Number(window.__DISK_SPIKE_COUNT ?? 0);
    if (traceIndex === undefined || traceIndex === null) return;

    let pinnedIdx = null;

    function setSelectedRow(idx) {{
      const rows = table.querySelectorAll('tr.disk-spike-row');
      rows.forEach(r => {{
        const ridx = r.getAttribute('data-idx');
        if (idx !== null && idx !== undefined && String(idx) === String(ridx)) r.classList.add('selected');
        else r.classList.remove('selected');
      }});
    }}

    function highlightByIdx(idx) {{
      if (idx === null || idx === undefined) return;
      const i = Number(idx);
      if (!Number.isFinite(i) || i < 0 || i >= spikeCount) return;
      const colors = Array(spikeCount).fill(window.__DISK_SPIKE_DEFAULT_COLOR || '#fbbf24');
      colors[i] = window.__DISK_SPIKE_HILITE_COLOR || '#fde047';
      const baseSize = Number(window.__DISK_SPIKE_DEFAULT_SIZE ?? 10);
      const hiSize = Number(window.__DISK_SPIKE_HILITE_SIZE ?? 14);
      const sizes = Array(spikeCount).fill(baseSize);
      sizes[i] = hiSize;
      Plotly.restyle(plot, {{ 'marker.color': [colors], 'marker.size': [sizes] }}, [traceIndex]);
      setSelectedRow(i);
    }}

    function clearHighlight() {{
      const colors = Array(spikeCount).fill(window.__DISK_SPIKE_DEFAULT_COLOR || '#fbbf24');
      const baseSize = Number(window.__DISK_SPIKE_DEFAULT_SIZE ?? 10);
      const sizes = Array(spikeCount).fill(baseSize);
      Plotly.restyle(plot, {{ 'marker.color': [colors], 'marker.size': [sizes] }}, [traceIndex]);
      setSelectedRow(null);
    }}

    table.addEventListener('mouseover', (ev) => {{
      const tr = ev.target.closest('tr.disk-spike-row');
      if (!tr) return;
      if (pinnedIdx !== null) return;
      const idx = tr.getAttribute('data-idx');
      if (idx !== null) highlightByIdx(idx);
    }});
    table.addEventListener('mouseout', (ev) => {{
      const tr = ev.target.closest('tr.disk-spike-row');
      if (!tr) return;
      if (pinnedIdx !== null) return;
      clearHighlight();
    }});
    table.addEventListener('click', (ev) => {{
      const tr = ev.target.closest('tr.disk-spike-row');
      if (!tr) return;
      const idx = tr.getAttribute('data-idx');
      if (idx === null) return;
      const i = Number(idx);
      if (!Number.isFinite(i)) return;
      if (pinnedIdx === i) {{
        pinnedIdx = null;
        clearHighlight();
      }} else {{
        pinnedIdx = i;
        highlightByIdx(i);
      }}
    }});

    plot.on('plotly_click', () => {{
      pinnedIdx = null;
      clearHighlight();
    }});
  }}

  function installSpikeTableLinkage() {{
    const table = document.getElementById('spike_table');
    const plot = document.getElementById('sys_graph');
    if (!table || !plot) return;
    const traceIndex = window.__SPIKE_TRACE_INDEX;
    const glowTraceIndex = window.__SPIKE_GLOW_TRACE_INDEX
    const spikeCount = Number(window.__SPIKE_COUNT ?? 0);
    if (traceIndex === undefined || traceIndex === null) return;

    let pinnedIdx = null;

    function setSelectedRow(idx) {{
      // Visual selection in table
      const rows = table.querySelectorAll('tr.spike-row');
      rows.forEach(r => {{
        const ridx = r.getAttribute('data-idx');
        if (idx !== null && idx !== undefined && String(idx) === String(ridx)) r.classList.add('selected');
        else r.classList.remove('selected');
      }});
    }}

    function highlightSpikeByIdx(idx) {{
      if (idx === null || idx === undefined) return;
      const i = Number(idx);
      if (!Number.isFinite(i) || i < 0 || i >= spikeCount) return;
      // Create per-point styles so only one marker pops.
      const colors = Array(spikeCount).fill(window.__SPIKE_DEFAULT_COLOR || '#fb7185');
      colors[i] = window.__SPIKE_HILITE_COLOR || '#fbbf24';
      const baseSize = Number(window.__SPIKE_DEFAULT_SIZE ?? 10);
      const hiSize = Number(window.__SPIKE_HILITE_SIZE ?? 14);
      const sizes = Array(spikeCount).fill(baseSize);
      sizes[i] = hiSize;
      Plotly.restyle(plot, {{ 'marker.color': [colors], 'marker.size': [sizes] }}, [traceIndex]);

      // Glow layer (if present)
      if (glowTraceIndex !== undefined && glowTraceIndex !== null) {{
        const gColors = Array(spikeCount).fill(window.__SPIKE_GLOW_DEFAULT_COLOR || 'rgba(251,113,133,0.22)');
        gColors[i] = window.__SPIKE_GLOW_HILITE_COLOR || 'rgba(251,191,36,0.30)';
        const gBase = Number(window.__SPIKE_GLOW_DEFAULT_SIZE ?? 18);
        const gHi = Number(window.__SPIKE_GLOW_HILITE_SIZE ?? 24);
        const gSizes = Array(spikeCount).fill(gBase);
        gSizes[i] = gHi;
        Plotly.restyle(plot, {{ 'marker.color': [gColors], 'marker.size': [gSizes] }}, [glowTraceIndex]);
      }}
      setSelectedRow(i);
    }}

    function clearHighlight() {{
      const colors = Array(spikeCount).fill(window.__SPIKE_DEFAULT_COLOR || '#fb7185');
      const baseSize = Number(window.__SPIKE_DEFAULT_SIZE ?? 10);
      const sizes = Array(spikeCount).fill(baseSize);
      Plotly.restyle(plot, {{ 'marker.color': [colors], 'marker.size': [sizes] }}, [traceIndex]);

      if (glowTraceIndex !== undefined && glowTraceIndex !== null) {{
        const gColors = Array(spikeCount).fill(window.__SPIKE_GLOW_DEFAULT_COLOR || 'rgba(251,113,133,0.22)');
        const gBase = Number(window.__SPIKE_GLOW_DEFAULT_SIZE ?? 18);
        const gSizes = Array(spikeCount).fill(gBase);
        Plotly.restyle(plot, {{ 'marker.color': [gColors], 'marker.size': [gSizes] }}, [glowTraceIndex]);
      }}
      setSelectedRow(null);
    }}

    // Event delegation so this works even if the table is rebuilt later.
    table.addEventListener('mouseover', (ev) => {{
      const tr = ev.target.closest('tr.spike-row');
      if (!tr) return;
      if (pinnedIdx !== null) return; // when pinned, ignore hover
      const idx = tr.getAttribute('data-idx');
      if (idx !== null) highlightSpikeByIdx(idx);
    }});
    table.addEventListener('mouseout', (ev) => {{
      const tr = ev.target.closest('tr.spike-row');
      if (!tr) return;
      if (pinnedIdx !== null) return;
      clearHighlight();
    }});
    table.addEventListener('click', (ev) => {{
      const tr = ev.target.closest('tr.spike-row');
      if (!tr) return;
      const idx = tr.getAttribute('data-idx');
      if (idx === null) return;
      const i = Number(idx);
      if (!Number.isFinite(i)) return;
      if (pinnedIdx === i) {{
        pinnedIdx = null;
        clearHighlight();
      }} else {{
        pinnedIdx = i;
        highlightSpikeByIdx(i);
      }}
    }});

    // If user clicks elsewhere on the plot, unpin.
    plot.on('plotly_click', () => {{
      pinnedIdx = null;
      clearHighlight();
    }});
  }}

  function installHoverDelay(divId) {{
    const delayMs = Number(PAYLOAD.hover_delay_ms ?? 0);
    const el = document.getElementById(divId);
    if (!el) return;
    // Mark this plot as opt-in for delayed hover behavior
    el.classList.add('hover-delay');
    let timer = null;

    function clearTimer() {{
      if (timer !== null) {{
        clearTimeout(timer);
        timer = null;
      }}
    }}

    function hideNow() {{
      el.classList.remove('show-hover');
    }}

    function showLater() {{
      hideNow();
      clearTimer();
      if (!delayMs || delayMs <= 0) {{
        el.classList.add('show-hover');
        return;
      }}
      timer = setTimeout(() => {{
        el.classList.add('show-hover');
      }}, delayMs);
    }}

    // On hover updates, restart the delay.
    el.on('plotly_hover', () => showLater());
    el.on('plotly_unhover', () => {{
      clearTimer();
      hideNow();
    }});
    // If the plot redraws or changes range, hide hover immediately.
    el.on('plotly_relayout', () => {{
      clearTimer();
      hideNow();
    }});
  }}

  function installTimeSync(plotIds) {{
    // Sync x-axis range across multiple Plotly charts (zoom/pan/rangeselector/rangeslider/dblclick reset).
    // Plotly emits 'plotly_relayout' with keys like:
    //   'xaxis.range[0]', 'xaxis.range[1]' OR 'xaxis.autorange'
    // We forward only xaxis.* changes to other plots.
    const ids = (plotIds || []).filter(Boolean);
    const plots = ids
      .map(id => document.getElementById(id))
      .filter(Boolean);
    if (plots.length <= 1) return;

    let syncing = false;

    function pickXUpdate(update) {{
      if (!update) return null;
      const out = {{}};
      for (const [k, v] of Object.entries(update)) {{
        if (k === 'xaxis.autorange' ||
            k === 'xaxis.range' ||
            k.startsWith('xaxis.range[') ||
            k.startsWith('xaxis.rangeslider') ||
            k.startsWith('xaxis.showspikes') ||
            k.startsWith('xaxis.spikes')) {{
          out[k] = v;
        }}
      }}
      return Object.keys(out).length ? out : null;
    }}

    function applyToOthers(sourceEl, xUpdate) {{
      if (!xUpdate) return;
      syncing = true;
      const ps = [];
      for (const el of plots) {{
        if (el === sourceEl) continue;
        try {{
          ps.push(Plotly.relayout(el, xUpdate));
        }} catch (e) {{
          // ignore
        }}
      }}
      Promise.allSettled(ps).finally(() => {{
        syncing = false;
      }});
    }}

    for (const el of plots) {{
      el.on('plotly_relayout', (update) => {{
        if (syncing) return;
        const xUpdate = pickXUpdate(update);
        applyToOthers(el, xUpdate);
      }});
    }}
  }}

  function renderGpuGraph() {{
    const g = PAYLOAD.gpus || {{}};
    const keys = Object.keys(g);
    const traces = [];
    for (const k of keys) {{
      const gpu = g[k];
      traces.push({{
        x: gpu.x,
        y: gpu.util,
        name: `GPU${{k}} util%`,
        mode: 'lines',
        line: {{ width: 2, dash: DASH.solid }},
        yaxis: 'y1',
      }});
      traces.push({{
        x: gpu.x,
        y: gpu.mem_used,
        name: `GPU${{k}} mem used (MB)`,
        mode: 'lines',
        line: {{ width: 1.7, dash: DASH.longdash }},
        yaxis: 'y2',
      }});
      traces.push({{
        x: gpu.x,
        y: gpu.temp,
        name: `GPU${{k}} temp (C)`,
        mode: 'lines',
        line: {{ width: 1.4, dash: DASH.dashdot }},
        yaxis: 'y3',
      }});
      traces.push({{
        x: gpu.x,
        y: gpu.power,
        name: `GPU${{k}} power (W)`,
        mode: 'lines',
        line: {{ width: 1.4, dash: DASH.longdashdot }},
        yaxis: 'y4',
      }});
    }}

    const layout = commonLayout('');
    layout.yaxis = {{ title: 'util %', rangemode: 'tozero', gridcolor: 'rgba(255,255,255,0.09)' }};
    layout.yaxis2 = {{
      title: 'mem MB',
      overlaying: 'y',
      side: 'right',
      anchor: 'free',
      position: 1.00,
      showgrid: false,
      tickfont: {{ size: 10 }},
      titlefont: {{ size: 10 }},
      automargin: true,
    }};
    layout.yaxis3 = {{
      title: 'temp C',
      overlaying: 'y',
      side: 'right',
      anchor: 'free',
      position: 1.06,
      showgrid: false,
      tickfont: {{ size: 10 }},
      titlefont: {{ size: 10 }},
      automargin: true,
    }};
    layout.yaxis4 = {{
      title: 'power W',
      overlaying: 'y',
      side: 'right',
      anchor: 'free',
      position: 1.12,
      showgrid: false,
      tickfont: {{ size: 10 }},
      titlefont: {{ size: 10 }},
      automargin: true,
    }};

    const config = {{ displayModeBar: true, responsive: true }};
    if (!keys.length) {{
      return Plotly.newPlot('gpu_graph', [{{x: [PAYLOAD.window_end], y: [0], mode:'text', text:['No GPU samples in window'], textfont:{{size:14}}, hoverinfo:'skip'}}], layout, config)
        .then(() => installHoverDelay('gpu_graph'));
    }} else {{
      return Plotly.newPlot('gpu_graph', traces, layout, config)
        .then(() => installHoverDelay('gpu_graph'));
    }}
  }}

  function renderPingGraph() {{
    const p = PAYLOAD.pings || {{}};
    const targets = Object.keys(p);
    const traces = [];
    for (const t of targets) {{
      const d = p[t];
      traces.push({{
        x: d.x,
        y: d.rtt_ms,
        name: t,
        mode: 'lines+markers',
        marker: {{ size: 5 }},
        line: {{ width: 2 }},
        yaxis: 'y1',
      }});

      // Failures: plot at y=0 with 'x' markers (status output)
      const fx = [];
      const fy = [];
      const ft = [];
      for (let i = 0; i < (d.x || []).length; i++) {{
        const ok = (d.success && d.success[i]) ? 1 : 0;
        if (ok) continue;
        fx.push(d.x[i]);
        fy.push(0);
        const err = (d.error && d.error[i]) ? d.error[i] : 'ping_failed';
        ft.push(`${{t}}<br>${{err}}`);
      }}
      if (fx.length) {{
        traces.push({{
          x: fx,
          y: fy,
          name: `${{t}} fail`,
          mode: 'markers',
          marker: {{ size: 9, symbol: 'x', color: '#fb7185', line: {{ width: 2, color: 'rgba(255,255,255,0.65)' }} }},
          text: ft,
          hovertemplate: '<b>Ping fail</b><br>%{{x}}<br>%{{text}}<extra></extra>',
          showlegend: false,
          yaxis: 'y1',
        }});
      }}
    }}

    const layout = commonLayout('');
    layout.yaxis = {{ title: 'RTT (ms)', rangemode: 'tozero', gridcolor: 'rgba(255,255,255,0.09)' }};
    const config = {{ displayModeBar: true, responsive: true }};
    return Plotly.newPlot('ping_graph', traces, layout, config).then(() => installHoverDelay('ping_graph'));
  }}

  function renderNetGraph() {{
    const n = PAYLOAD.net || {{}};
    const procs = Object.keys(n);
    const traces = [];
    for (const p of procs) {{
      const d = n[p];
      const x = d.x || [];
      const total = (d.total_bps || []).map(v => bpsToMiBps(v));
      const sent = (d.sent_bps || []).map(v => bpsToMiBps(v));
      const recv = (d.recv_bps || []).map(v => bpsToMiBps(v));
      const text = total.map((_, i) => `sent=${{sent[i]?.toFixed(2)}} MiB/s<br>recv=${{recv[i]?.toFixed(2)}} MiB/s`);
      traces.push({{
        x,
        y: total,
        name: p,
        mode: 'lines',
        line: {{ width: 2 }},
        text,
        hovertemplate: '<b>%{{fullData.name}}</b><br>%{{x}}<br>total=%{{y:.2f}} MiB/s<br>%{{text}}<extra></extra>',
      }});
    }}
    const layout = commonLayout('');
    layout.yaxis = {{ title: 'MiB/s', rangemode: 'tozero', gridcolor: 'rgba(255,255,255,0.09)' }};
    const config = {{ displayModeBar: true, responsive: true }};
    return Plotly.newPlot('net_graph', traces, layout, config).then(() => installHoverDelay('net_graph'));
  }}

  function renderDockerGraph() {{
    const d = PAYLOAD.docker || {{}};
    const meta = PAYLOAD.docker_meta || {{}};
    const keys = Object.keys(d);
    const traces = [];
    for (const k of keys) {{
      const r = d[k];
      const x = r.x || [];
      const memGiB = (r.mem_usage_bytes || []).map(v => bytesToGiB(v));
      const cpu = (r.cpu_percent || []).map(v => v);
      const img = (meta[k] && meta[k].image) ? meta[k].image : '';
      const label = `${{k.slice(0,12)}} ${{img}}`.trim();

      traces.push({{
        x,
        y: memGiB,
        name: `${{label}} mem`,
        mode: 'lines',
        line: {{ width: 2, dash: 'solid' }},
        yaxis: 'y1',
      }});
      traces.push({{
        x,
        y: cpu,
        name: `${{label}} cpu`,
        mode: 'lines',
        line: {{ width: 1.8, dash: 'dash' }},
        yaxis: 'y2',
      }});
    }}
    const layout = commonLayout('');
    layout.yaxis = {{ title: 'mem GiB', rangemode: 'tozero', gridcolor: 'rgba(255,255,255,0.09)' }};
    layout.yaxis2 = {{
      title: 'cpu %',
      overlaying: 'y',
      side: 'right',
      anchor: 'free',
      position: 1.00,
      showgrid: false,
      tickfont: {{ size: 10 }},
      titlefont: {{ size: 10 }},
      automargin: true,
    }};
    const config = {{ displayModeBar: true, responsive: true }};
    if (!keys.length) {{
      return Plotly.newPlot('docker_graph', [{{x: [PAYLOAD.window_end], y: [0], mode:'text', text:['No docker stats in window'], textfont:{{size:14}}, hoverinfo:'skip'}}], layout, config)
        .then(() => installHoverDelay('docker_graph'));
    }}
    return Plotly.newPlot('docker_graph', traces, layout, config).then(() => installHoverDelay('docker_graph'));
  }}

  buildSpikeTable(PAYLOAD.spikes || []);
  buildDiskSpikeTable(PAYLOAD.disk_spikes || []);
  buildLeaderTable(PAYLOAD.cpu_leaderboard || []);
  buildDockerTable(PAYLOAD.docker_leaderboard || []);
  buildPingTable(PAYLOAD.ping_summary || []);
  buildNetTable(PAYLOAD.net_leaderboard || []);
  const pSys = renderSystemGraph();
  const pGpu = renderGpuGraph();
  const pDocker = renderDockerGraph();
  const pPing = renderPingGraph();
  const pNet = renderNetGraph();
  Promise.allSettled([pSys, pGpu, pDocker, pPing, pNet]).then(() => {{
    installTimeSync(['sys_graph', 'gpu_graph', 'docker_graph', 'ping_graph', 'net_graph']);
  }});
  </script>
</body>
</html>
"""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate an HTML resource report from resource_monitor.sqlite")
    p.add_argument("--db-path", type=Path, default=_default_db_path(), help="SQLite DB path")
    p.add_argument("--output", type=Path, required=True, help="Output HTML file path")
    p.add_argument("--days", type=float, default=7.0, help="How many days to include (default: 7)")
    p.add_argument("--title", default="keivenc-linux Resource Report", help="HTML title")

    p.add_argument("--min-cpu-spike", type=float, default=50.0, help="Minimum top-process cpu%% for a spike")
    p.add_argument("--spike-sigma", type=float, default=4.0, help="MAD-based sigma threshold multiplier")
    p.add_argument("--max-spikes", type=int, default=50, help="Max spikes to show/annotate")

    p.add_argument(
        "--hover-delay-ms",
        type=int,
        default=700,
        help="Delay (ms) before hover labels appear (default: 700)",
    )

    p.add_argument("--leaderboard-limit", type=int, default=20, help="How many leaderboard rows to include")
    p.add_argument("--docker-limit", type=int, default=5, help="How many containers to include (default: 5)")
    p.add_argument("--net-limit", type=int, default=5, help="How many network offenders to include (default: 5)")
    p.add_argument("--min-disk-read-spike-mibps", type=float, default=50.0, help="Minimum disk read MiB/s for a spike")
    p.add_argument("--disk-spike-sigma", type=float, default=4.0, help="MAD-based sigma threshold multiplier for disk spikes")
    p.add_argument("--max-disk-spikes", type=int, default=50, help="Max disk spikes to show/annotate")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    db_path: Path = args.db_path
    out_path: Path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = _connect(db_path)
    try:
        start_ts, end_ts = _get_time_window(con, days=float(args.days))
        samples = _query_samples(con, start_ts=start_ts)
        gpus = _query_gpu_timeseries(con, start_ts=start_ts)
        top_io = _query_top_io_read_process_per_sample(con, start_ts=start_ts)
        disk_spikes = _detect_disk_read_spikes(
            samples,
            min_mibps=float(args.min_disk_read_spike_mibps),
            sigma=float(args.disk_spike_sigma),
            max_spikes=int(args.max_disk_spikes),
        )
        # Attach best-effort top-IO process info to disk spikes
        for e in disk_spikes:
            sid = int(e.get("sample_id") or 0)
            p = top_io.get(sid) or {}
            e["pid"] = p.get("pid")
            e["proc_name"] = p.get("name", "")
            e["cmdline"] = p.get("cmdline", "")
            e["proc_io_read_mibps"] = (
                float(p.get("io_read_bps") or 0.0) / 1024.0 / 1024.0
                if p.get("io_read_bps") is not None
                else 0.0
            )

        docker_leaderboard = _query_docker_leaderboard(
            con, start_ts=start_ts, limit=int(args.docker_limit)
        )
        docker_keys = [str(r.get("container_id") or "") for r in docker_leaderboard if (r.get("container_id") or "")]
        docker = _query_docker_timeseries(con, start_ts=start_ts, keys=docker_keys)
        docker_meta = {str(r.get("container_id") or ""): {"image": r.get("image") or ""} for r in docker_leaderboard}
        pings = _query_ping_timeseries(con, start_ts=start_ts)
        ping_summary = _query_ping_summary(con, start_ts=start_ts)
        net_leaderboard = _query_net_leaderboard(con, start_ts=start_ts, limit=int(args.net_limit))
        net_procs = [str(r.get("proc") or "") for r in net_leaderboard if (r.get("proc") or "")]
        net = _query_net_timeseries(con, start_ts=start_ts, procs=net_procs)
        top_proc = _query_top_process_per_sample(con, start_ts=start_ts)
        spikes = _detect_cpu_spikes(
            top_proc,
            min_cpu_percent=float(args.min_cpu_spike),
            sigma=float(args.spike_sigma),
            max_spikes=int(args.max_spikes),
        )
        leaderboard = _query_cpu_leaderboard(
            con, start_ts=start_ts, limit=int(args.leaderboard_limit)
        )

        avg_interval = 0.0
        if samples:
            avg_interval = sum(s.interval_s for s in samples) / max(1, len(samples))

        payload: Dict[str, Any] = {
            "title": str(args.title),
            "db_path": str(db_path),
            "window": f"{datetime.fromtimestamp(start_ts)} → {datetime.fromtimestamp(end_ts)} (localtime) | last {args.days:g} days",
            "window_end": _ts_to_iso(end_ts),
            "sample_count": len(samples),
            "avg_interval_s": avg_interval,
            "hover_delay_ms": int(args.hover_delay_ms),
            "samples": [
                {
                    "sample_id": s.sample_id,
                    "ts": _ts_to_iso(s.ts_unix),
                    "cpu_percent": s.cpu_percent,
                    "mem_percent": s.mem_percent,
                    "load1": s.load1,
                    "net_sent_bps": s.net_sent_bps,
                    "net_recv_bps": s.net_recv_bps,
                    "disk_read_bps": s.disk_read_bps,
                    "disk_write_bps": s.disk_write_bps,
                    "mem_used_gib": (s.mem_used_bytes / 1024 / 1024 / 1024) if s.mem_used_bytes else None,
                    "mem_total_gib": (s.mem_total_bytes / 1024 / 1024 / 1024) if s.mem_total_bytes else None,
                    "gpu_util_max": s.gpu_util_max,
                    "gpu_mem_percent_total": (
                        (s.gpu_mem_used_mb_total / s.gpu_mem_total_mb_total * 100.0)
                        if (s.gpu_mem_used_mb_total is not None and s.gpu_mem_total_mb_total)
                        else None
                    ),
                }
                for s in samples
            ],
            "gpus": gpus,
            "docker": docker,
            "docker_leaderboard": docker_leaderboard,
            "docker_meta": docker_meta,
            "pings": pings,
            "ping_summary": ping_summary,
            "net": net,
            "net_leaderboard": net_leaderboard,
            "spikes": spikes,
            "disk_spikes": disk_spikes,
            "cpu_leaderboard": leaderboard,
        }

        out_path.write_text(_build_html(payload), encoding="utf-8")
        print(f"Wrote: {out_path}")
        return 0
    finally:
        try:
            con.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())


