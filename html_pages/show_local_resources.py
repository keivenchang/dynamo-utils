#!/usr/bin/env python3
"""
Generate a fancy HTML resource report from the SQLite DB produced by `resource_monitor.py`.

Features:
- Interactive charts (zoom/pan/range buttons) via Plotly.js (loaded from CDN)
- Last N days view (max: 2d; default: 2d) with quick zoom buttons (1d/12h/6h/1h)
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

# Shared dashboard runtime helper (atomic output writes)
from common_dashboard_runtime import atomic_write_text


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
    # Policy: do not read from repo-local `.cache/`; always use the global dynamo-utils cache dir.
    return Path.home() / ".cache" / "dynamo-utils" / "resource_monitor.sqlite"


def _ts_to_iso(ts_unix: float) -> str:
    # Plotly parses ISO8601 strings (localtime is OK; keep it simple)
    return datetime.fromtimestamp(ts_unix).isoformat(timespec="seconds")

def _query_gh_rate_limit_timeseries(
    con: sqlite3.Connection,
    *,
    start_ts: float,
    resources: Sequence[str],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Query gh_rate_limit_samples joined with samples into Plotly-friendly series.

    Returns:
      (series, latest)
    where series is:
      {resource: {"x":[...], "remaining":[...], "limit":[...], "reset_epoch":[...]}}
    """
    if not _table_exists(con, "gh_rate_limit_samples"):
        return {}, {}
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT
          s.ts_unix AS ts_unix,
          g.resource AS resource,
          g."limit" AS lim,
          g.remaining AS rem,
          g.reset_epoch AS reset_epoch
        FROM gh_rate_limit_samples g
        JOIN samples s ON s.id = g.sample_id
        WHERE s.ts_unix >= ?
        ORDER BY s.ts_unix ASC
        """,
        (float(start_ts),),
    ).fetchall()

    want = {str(r) for r in resources}
    series: Dict[str, Dict[str, Any]] = {}
    latest: Dict[str, Dict[str, Any]] = {}
    for r in resources:
        series[str(r)] = {"x": [], "remaining": [], "limit": [], "reset_epoch": []}

    for row in rows:
        try:
            res = str(row["resource"] or "")
        except Exception:
            continue
        if not res or (want and res not in want):
            continue
        try:
            tsu = float(row["ts_unix"])
        except Exception:
            continue
        x = _ts_to_iso(tsu)
        rem = row["rem"]
        lim = row["lim"]
        reset_epoch = row["reset_epoch"]
        try:
            rem_i = int(rem) if rem is not None else None
            lim_i = int(lim) if lim is not None else None
            reset_i = int(reset_epoch) if reset_epoch is not None else None
        except Exception:
            continue

        d = series.setdefault(res, {"x": [], "remaining": [], "limit": [], "reset_epoch": []})
        d["x"].append(x)
        d["remaining"].append(rem_i)
        d["limit"].append(lim_i)
        d["reset_epoch"].append(reset_i)
        latest[res] = {"ts_unix": int(tsu), "remaining": rem_i, "limit": lim_i, "reset_epoch": reset_i}

    # Drop empties.
    series = {k: v for k, v in series.items() if v.get("x")}
    latest = {k: v for k, v in latest.items() if k in series}
    return series, latest


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
    # Keep timeout low: this script is often run from cron and should fail fast if the DB is locked.
    con = sqlite3.connect(str(db_path), timeout=2.0)
    con.row_factory = sqlite3.Row
    return con


def _wal_checkpoint_truncate(con: sqlite3.Connection) -> List[Tuple[int, int, int]]:
    """
    Attempt to checkpoint and truncate the WAL file.

    Returns SQLite's result rows for PRAGMA wal_checkpoint(TRUNCATE):
      [(busy, log, checkpointed)]
    """
    try:
        rows = con.execute("PRAGMA wal_checkpoint(TRUNCATE);").fetchall()
        out: List[Tuple[int, int, int]] = []
        for r in rows:
            try:
                out.append((int(r[0]), int(r[1]), int(r[2])))
            except Exception:
                continue
        return out
    except Exception:
        return []


def _vacuum_best_effort(db_path: Path) -> bool:
    """
    Best-effort VACUUM to reclaim disk space (shrinks the main .sqlite file).

    VACUUM requires an exclusive lock; if the monitor is running, this will likely fail quickly.
    """
    try:
        con = sqlite3.connect(str(db_path), timeout=2.0)
        try:
            con.execute("PRAGMA foreign_keys=ON;")
            con.execute("VACUUM;")
            return True
        finally:
            con.close()
    except Exception:
        return False


def _prune_db_samples_older_than(
    con: sqlite3.Connection,
    *,
    cutoff_ts_unix: float,
) -> int:
    """
    Delete samples older than cutoff_ts_unix.

    `resource_monitor.py` schema uses ON DELETE CASCADE for child tables, but SQLite only enforces
    that when PRAGMA foreign_keys=ON for the current connection.
    """
    try:
        con.execute("PRAGMA foreign_keys=ON;")
    except Exception:
        pass
    cur = con.cursor()
    try:
        n = cur.execute(
            "SELECT COUNT(*) AS n FROM samples WHERE ts_unix < ?", (float(cutoff_ts_unix),)
        ).fetchone()["n"]
        to_delete = int(n or 0)
    except Exception:
        to_delete = 0
    cur.execute("DELETE FROM samples WHERE ts_unix < ?", (float(cutoff_ts_unix),))
    con.commit()
    return to_delete


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


def _query_top_io_write_process_per_sample(
    con: sqlite3.Connection, *, start_ts: float
) -> Dict[int, Dict[str, Any]]:
    """
    Best-effort: process_samples only stores "offenders", so this finds the max IO-write process among recorded ones.

    Returns:
      {sample_id: {pid,name,username,cmdline,io_write_bps,io_read_bps,rss_bytes}}
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
            p.io_write_bps AS io_write_bps,
            p.io_read_bps AS io_read_bps,
            p.rss_bytes AS rss_bytes,
            ROW_NUMBER() OVER (
              PARTITION BY p.sample_id
              ORDER BY p.io_write_bps DESC NULLS LAST
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
            "io_write_bps": float(r["io_write_bps"]) if r["io_write_bps"] is not None else None,
            "io_read_bps": float(r["io_read_bps"]) if r["io_read_bps"] is not None else None,
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


def _detect_disk_write_spikes(
    samples: List[SampleRow],
    *,
    min_mibps: float,
    sigma: float,
    max_spikes: int,
) -> List[Dict[str, Any]]:
    """
    Detect spikes on system disk_write_bps (MiB/s) using a robust MAD threshold + local-max guard.
    """
    vals: List[float] = []
    for s in samples:
        v = (float(s.disk_write_bps) / 1024.0 / 1024.0) if s.disk_write_bps is not None else 0.0
        vals.append(v)
    med, mad = _median_mad(vals)
    robust_std = 1.4826 * mad
    thresh = max(float(min_mibps), med + float(sigma) * robust_std)

    spikes: List[Dict[str, Any]] = []
    for i, s in enumerate(samples):
        v = (float(s.disk_write_bps) / 1024.0 / 1024.0) if s.disk_write_bps is not None else 0.0
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
                "disk_write_mibps": float(v),
            }
        )

    spikes.sort(key=lambda x: x["disk_write_mibps"], reverse=True)
    spikes = spikes[: int(max_spikes)]
    spikes.sort(key=lambda x: x["ts_unix"])
    return spikes


def _query_top_net_by_sample(
    con: sqlite3.Connection, *, start_ts: float, direction: str
) -> Dict[int, Dict[str, Any]]:
    """
    direction: 'recv' or 'sent'
    Best-effort from net_process_samples (top-K only).
    Returns {sample_id: {pid, proc, recv_bps, sent_bps}}
    """
    if direction not in ("recv", "sent"):
        raise ValueError("direction must be 'recv' or 'sent'")
    if not _table_exists(con, "net_process_samples"):
        return {}
    col = "recv_bps" if direction == "recv" else "sent_bps"
    cur = con.cursor()
    rows = cur.execute(
        f"""
        WITH ranked AS (
          SELECT
            n.sample_id AS sample_id,
            n.pid AS pid,
            n.proc AS proc,
            n.sent_bps AS sent_bps,
            n.recv_bps AS recv_bps,
            ROW_NUMBER() OVER (
              PARTITION BY n.sample_id
              ORDER BY COALESCE(n.{col},0) DESC
            ) AS rn
          FROM net_process_samples n
          JOIN samples s ON s.id = n.sample_id
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
            "proc": r["proc"] or "",
            "sent_bps": float(r["sent_bps"]) if r["sent_bps"] is not None else None,
            "recv_bps": float(r["recv_bps"]) if r["recv_bps"] is not None else None,
        }
    return out


def _query_net_top_points(con: sqlite3.Connection, *, start_ts: float) -> List[Dict[str, Any]]:
    """
    Best-effort: net_process_samples is only collected periodically (may not exist for every system sample_id).

    Returns a list of points sorted by ts_unix:
      {
        sample_id, ts_unix,
        top_recv_pid, top_recv_proc, top_recv_bps,
        top_sent_pid, top_sent_proc, top_sent_bps,
      }
    """
    if not _table_exists(con, "net_process_samples"):
        return []
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT
          s.id AS sample_id,
          s.ts_unix AS ts_unix,
          n.pid AS pid,
          n.proc AS proc,
          n.sent_bps AS sent_bps,
          n.recv_bps AS recv_bps
        FROM net_process_samples n
        JOIN samples s ON s.id = n.sample_id
        WHERE s.ts_unix >= ?
        ORDER BY s.ts_unix ASC
        """,
        (start_ts,),
    ).fetchall()

    by_sid: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        sid = int(r["sample_id"])
        tsu = float(r["ts_unix"])
        pid = int(r["pid"]) if r["pid"] is not None else None
        proc = r["proc"] or ""
        sent_bps = float(r["sent_bps"]) if r["sent_bps"] is not None else None
        recv_bps = float(r["recv_bps"]) if r["recv_bps"] is not None else None

        e = by_sid.get(sid)
        if e is None:
            e = {
                "sample_id": sid,
                "ts_unix": tsu,
                "top_recv_pid": None,
                "top_recv_proc": "",
                "top_recv_bps": None,
                "top_sent_pid": None,
                "top_sent_proc": "",
                "top_sent_bps": None,
            }
            by_sid[sid] = e

        if recv_bps is not None and (e["top_recv_bps"] is None or recv_bps > float(e["top_recv_bps"])):
            e["top_recv_bps"] = recv_bps
            e["top_recv_pid"] = pid
            e["top_recv_proc"] = proc
        if sent_bps is not None and (e["top_sent_bps"] is None or sent_bps > float(e["top_sent_bps"])):
            e["top_sent_bps"] = sent_bps
            e["top_sent_pid"] = pid
            e["top_sent_proc"] = proc

    pts = list(by_sid.values())
    pts.sort(key=lambda x: float(x["ts_unix"]))
    return pts


def _detect_net_spikes(
    samples: List[SampleRow],
    *,
    which: str,
    min_mibps: float,
    sigma: float,
    max_spikes: int,
) -> List[Dict[str, Any]]:
    """
    which: 'recv' or 'sent' for system net_{recv,sent}_bps spikes.
    """
    if which not in ("recv", "sent"):
        raise ValueError("which must be 'recv' or 'sent'")
    vals: List[float] = []
    for s in samples:
        bps = s.net_recv_bps if which == "recv" else s.net_sent_bps
        v = (float(bps) / 1024.0 / 1024.0) if bps is not None else 0.0
        vals.append(v)
    med, mad = _median_mad(vals)
    robust_std = 1.4826 * mad
    thresh = max(float(min_mibps), med + float(sigma) * robust_std)

    spikes: List[Dict[str, Any]] = []
    for i, s in enumerate(samples):
        bps = s.net_recv_bps if which == "recv" else s.net_sent_bps
        v = (float(bps) / 1024.0 / 1024.0) if bps is not None else 0.0
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
                f"net_{which}_mibps": float(v),
            }
        )

    spikes.sort(key=lambda x: float(x.get(f"net_{which}_mibps") or 0.0), reverse=True)
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
      /* Default layout: 60/40 (3/5 + 2/5) */
      grid-template-columns: 3fr 2fr;
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

    .grid.onecol {{
      grid-template-columns: 1fr;
    }}

    /* Keep the spike tables constrained to the same height as the System chart (single scroll area). */
    .spike-scroll {{
      height: 72vh;
      min-height: 520px;
      overflow: auto;
    }}
    @media (max-width: 1100px) {{
      .spike-scroll {{
        height: 62vh;
        min-height: 420px;
      }}
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
          Interactive charts: zoom / pan / range buttons. Use the x-axis range selector to jump to <span class="k">1d</span> or <span class="k">2d</span>.
        </div>
      </div>
      <div class="meta">
        <div>DB: {payload.get("db_path","")}</div>
        <div>Window: {payload.get("window","")}</div>
        <div>Samples: {payload.get("sample_count",0)} (avg interval: {payload.get("avg_interval_s",0):.2f}s)</div>
      </div>
    </div>

    <!-- System: keep the same 3/5 vs 2/5 chart/table ratio as the other dashboard rows. -->
    <div class="grid" style="grid-template-columns: 3fr 2fr;">
      <div class="card chart">
        <div class="hdr">
          <div class="title">System: CPU / Memory / IO</div>
          <div class="hint">Hover to inspect; drag to zoom; double-click to reset</div>
        </div>
        <div class="body">
          <div id="sys_graph" class="plot"></div>
        </div>
      </div>

      <div class="card">
        <div class="hdr">
          <div class="title">Spike events</div>
          <div class="hint"><span class="pill">MAD threshold</span></div>
        </div>
        <!-- One scrollable section containing all spike events (CPU + Disk + Network), newest first. -->
        <div class="body spike-scroll">
          <table class="table" id="spike_table"></table>
        </div>
      </div>
    </div>

    <!-- Match the System grid column ratio so the GPU chart has the same width. -->
    <div class="grid" style="grid-template-columns: 3fr 2fr; margin-top: 16px;">
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
    <div class="grid" style="grid-template-columns: 3fr 2fr; margin-top: 16px;">
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
    <div class="grid" style="grid-template-columns: 3fr 2fr; margin-top: 16px;">
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
    <div class="grid" style="grid-template-columns: 3fr 2fr; margin-top: 16px;">
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

    <!-- GitHub rate limit (recorded by resource_monitor.py into SQLite) -->
    <div class="grid onecol" style="margin-top: 16px;">
      <div class="card chart">
        <div class="hdr">
          <div class="title">GitHub API rate limit</div>
          <div class="hint">Read from SQLite table <span class="mono">gh_rate_limit_samples</span> (written by <span class="mono">resource_monitor.py</span>). Lines = remaining requests.</div>
        </div>
        <div class="body">
          <div id="gh_rate_limit_graph" class="plot"></div>
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
    const cpu = spikes || [];
    const diskR = PAYLOAD.disk_spikes || [];
    const diskW = PAYLOAD.disk_write_spikes || [];
    const netIn = PAYLOAD.net_recv_spikes || [];
    const netOut = PAYLOAD.net_sent_spikes || [];

    const rows = [];
    for (let i = 0; i < cpu.length; i++) {{
      const s = cpu[i] || {{}};
      rows.push({{
        kind: 'cpu',
        idx: i,
        ts: s.ts,
        ts_unix: Number(s.ts_unix || 0),
        badge: 'bad',
        valueFmt: `${{Number((s.cpu_percent_total ?? s.cpu_percent) || 0).toFixed(1)}}%`,
        detail: `raw: ${{Number(s.cpu_percent || 0).toFixed(1)}}%`,
        proc: (s.name && String(s.name).length) ? String(s.name) : '(unattributed)',
        pid: s.pid,
        cmdline: s.cmdline || '',
      }});
    }}
    for (let i = 0; i < diskR.length; i++) {{
      const s = diskR[i] || {{}};
      rows.push({{
        kind: 'disk_read',
        idx: i,
        ts: s.ts,
        ts_unix: Number(s.ts_unix || 0),
        badge: 'warn',
        valueFmt: `${{Number(s.disk_read_mibps || 0).toFixed(2)}} MiB/s`,
        detail: `proc read: ${{Number(s.proc_io_read_mibps || 0).toFixed(2)}} MiB/s`,
        proc: (s.proc_name && String(s.proc_name).length) ? String(s.proc_name) : '(unattributed)',
        pid: s.pid,
        cmdline: s.cmdline || '',
      }});
    }}
    for (let i = 0; i < diskW.length; i++) {{
      const s = diskW[i] || {{}};
      rows.push({{
        kind: 'disk_write',
        idx: i,
        ts: s.ts,
        ts_unix: Number(s.ts_unix || 0),
        badge: 'good',
        valueFmt: `${{Number(s.disk_write_mibps || 0).toFixed(2)}} MiB/s`,
        detail: `proc write: ${{Number(s.proc_io_write_mibps || 0).toFixed(2)}} MiB/s`,
        proc: (s.proc_name && String(s.proc_name).length) ? String(s.proc_name) : '(unattributed)',
        pid: s.pid,
        cmdline: s.cmdline || '',
      }});
    }}
    for (let i = 0; i < netIn.length; i++) {{
      const s = netIn[i] || {{}};
      rows.push({{
        kind: 'net_in',
        idx: i,
        ts: s.ts,
        ts_unix: Number(s.ts_unix || 0),
        badge: 'good',
        valueFmt: `${{Number(s.net_recv_mibps || 0).toFixed(2)}} MiB/s`,
        detail: `proc recv: ${{Number(s.proc_recv_mibps || 0).toFixed(2)}} MiB/s` + (s.attrib_skew_s !== undefined ? ` (skew ${{Number(s.attrib_skew_s).toFixed(1)}}s)` : ''),
        proc: (s.proc && String(s.proc).length) ? String(s.proc) : '(unattributed)',
        pid: s.pid,
        cmdline: '',
      }});
    }}
    for (let i = 0; i < netOut.length; i++) {{
      const s = netOut[i] || {{}};
      rows.push({{
        kind: 'net_out',
        idx: i,
        ts: s.ts,
        ts_unix: Number(s.ts_unix || 0),
        badge: 'warn',
        valueFmt: `${{Number(s.net_sent_mibps || 0).toFixed(2)}} MiB/s`,
        detail: `proc sent: ${{Number(s.proc_sent_mibps || 0).toFixed(2)}} MiB/s` + (s.attrib_skew_s !== undefined ? ` (skew ${{Number(s.attrib_skew_s).toFixed(1)}}s)` : ''),
        proc: (s.proc && String(s.proc).length) ? String(s.proc) : '(unattributed)',
        pid: s.pid,
        cmdline: '',
      }});
    }}

    if (!rows.length) {{
      el.innerHTML = `<tr><th>status</th></tr><tr><td class="muted">No spike events in window.</td></tr>`;
      return;
    }}

    // Most recent first.
    rows.sort((a, b) => Number(b.ts_unix || 0) - Number(a.ts_unix || 0));

    function kindLabel(k) {{
      if (k === 'cpu') return 'CPU';
      if (k === 'disk_read') return 'disk read';
      if (k === 'disk_write') return 'disk write';
      if (k === 'net_in') return 'net in';
      if (k === 'net_out') return 'net out';
      return k;
    }}

    el.innerHTML = `
      <tr>
        <th>time</th>
        <th>type</th>
        <th>value</th>
        <th>proc</th>
      </tr>
      ${{rows.map(r => `
        <tr class="spike-row" data-kind="${{r.kind}}" data-idx="${{r.idx}}">
          <td class="mono">${{String(r.ts||'').replace('T',' ')}}</td>
          <td class="mono"><span class="badge">${{kindLabel(r.kind)}}</span></td>
          <td class="mono"><span class="badge ${{r.badge}}">${{r.valueFmt}}</span><div class="small muted mono">${{r.detail}}</div></td>
          <td>
            <div class="mono" title="${{r.proc}} #${{r.pid ?? ''}}">${{String(r.proc||'').slice(0,34)}} <span class="muted">#${{r.pid ?? ''}}</span></div>
            ${{r.cmdline ? `<div class=\"small muted\" title=\"${{r.cmdline}}\">${{String(r.cmdline).slice(0,90)}}</div>` : ''}}
          </td>
        </tr>
      `).join('')}}
    `;
  }}

  function buildDiskSpikeTable(readRows, writeRows) {{
    const el = document.getElementById('disk_spike_table');
    const rr = readRows || [];
    const wr = writeRows || [];
    if (rr.length === 0 && wr.length === 0) {{
      el.innerHTML = `<tr><th>status</th></tr><tr><td class="muted">No disk spikes detected (or missing disk/process data).</td></tr>`;
      return;
    }}
    // Latest first; keep separate indices per series for linkage.
    const revR = rr.slice().reverse();
    const revW = wr.slice().reverse();
    el.innerHTML = `
      <tr>
        <th>time</th>
        <th>op</th>
        <th>rate</th>
        <th>proc</th>
      </tr>
      ${{revR.map((r, j) => `
        <tr class="spike-row disk-read-spike-row" data-idx="${{(rr.length - 1 - j)}}">
          <td class="mono">${{r.ts.replace('T',' ')}}</td>
          <td class="mono"><span class="badge warn">read</span></td>
          <td class="mono">${{(r.disk_read_mibps ?? 0).toFixed(2)}} MiB/s</td>
          <td>
            <div class="mono" title="${{r.proc_name || '(unattributed)'}} #${{r.pid ?? ''}}">${{(r.proc_name||'(unattributed)').slice(0,22)}} <span class="muted">#${{r.pid ?? ''}}</span></div>
            <div class="small muted" title="${{r.cmdline || ''}}">${{(r.cmdline||'').slice(0,90)}}</div>
          </td>
        </tr>
      `).join('')}}
      ${{revW.map((r, j) => `
        <tr class="spike-row disk-write-spike-row" data-idx="${{(wr.length - 1 - j)}}">
          <td class="mono">${{r.ts.replace('T',' ')}}</td>
          <td class="mono"><span class="badge good">write</span></td>
          <td class="mono">${{(r.disk_write_mibps ?? 0).toFixed(2)}} MiB/s</td>
          <td>
            <div class="mono" title="${{r.proc_name || '(unattributed)'}} #${{r.pid ?? ''}}">${{(r.proc_name||'(unattributed)').slice(0,22)}} <span class="muted">#${{r.pid ?? ''}}</span></div>
            <div class="small muted" title="${{r.cmdline || ''}}">${{(r.cmdline||'').slice(0,90)}}</div>
          </td>
        </tr>
      `).join('')}}
    `;
  }}

  function buildNetSpikeTable(recvRows, sentRows) {{
    const el = document.getElementById('net_spike_table');
    const rr = recvRows || [];
    const sr = sentRows || [];
    if (rr.length === 0 && sr.length === 0) {{
      el.innerHTML = `<tr><th>status</th></tr><tr><td class="muted">No network spikes detected (or missing net/top-talker data).</td></tr>`;
      return;
    }}
    const revR = rr.slice().reverse();
    const revS = sr.slice().reverse();
    el.innerHTML = `
      <tr>
        <th>time</th>
        <th>dir</th>
        <th>rate</th>
        <th>proc</th>
      </tr>
      ${{revR.map((r, j) => `
        <tr class="spike-row net-recv-spike-row" data-idx="${{(rr.length - 1 - j)}}">
          <td class="mono">${{r.ts.replace('T',' ')}}</td>
          <td class="mono"><span class="badge good">in</span></td>
          <td class="mono">${{(r.net_recv_mibps ?? 0).toFixed(2)}} MiB/s</td>
          <td>
            <div class="mono" title="${{r.proc || '(unattributed)'}} #${{r.pid ?? ''}}">${{(r.proc||'(unattributed)').slice(0,34)}} <span class="muted">#${{r.pid ?? ''}}</span></div>
          </td>
        </tr>
      `).join('')}}
      ${{revS.map((r, j) => `
        <tr class="spike-row net-sent-spike-row" data-idx="${{(sr.length - 1 - j)}}">
          <td class="mono">${{r.ts.replace('T',' ')}}</td>
          <td class="mono"><span class="badge warn">out</span></td>
          <td class="mono">${{(r.net_sent_mibps ?? 0).toFixed(2)}} MiB/s</td>
          <td>
            <div class="mono" title="${{r.proc || '(unattributed)'}} #${{r.pid ?? ''}}">${{(r.proc||'(unattributed)').slice(0,34)}} <span class="muted">#${{r.pid ?? ''}}</span></div>
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
            {{count: 2, label: '2d', step: 'day', stepmode: 'backward'}},
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
      `${{(e.proc_name && String(e.proc_name).length) ? e.proc_name : '(unattributed)'}} #${{e.pid ?? ''}}` +
      `<br>proc read: ${{(e.proc_io_read_mibps ?? 0).toFixed(2)}} MiB/s` +
      `<br>${{wrapLine((e.cmdline||'').slice(0,240), 80)}}`
    );

    const diskWriteSpikes = PAYLOAD.disk_write_spikes || [];
    const diskWriteSpikeX = diskWriteSpikes.map(e => e.ts);
    const diskWriteSpikeY = diskWriteSpikes.map(e => e.disk_write_mibps);
    const diskWriteSpikeText = diskWriteSpikes.map(e =>
      `${{(e.proc_name && String(e.proc_name).length) ? e.proc_name : '(unattributed)'}} #${{e.pid ?? ''}}` +
      `<br>proc write: ${{(e.proc_io_write_mibps ?? 0).toFixed(2)}} MiB/s` +
      `<br>${{wrapLine((e.cmdline||'').slice(0,240), 80)}}`
    );

    const netRecvSpikes = PAYLOAD.net_recv_spikes || [];
    const netRecvSpikeX = netRecvSpikes.map(e => e.ts);
    const netRecvSpikeY = netRecvSpikes.map(e => e.net_recv_mibps);
    const netRecvSpikeText = netRecvSpikes.map(e =>
      `${{(e.proc && String(e.proc).length) ? e.proc : '(unattributed)'}} #${{e.pid ?? ''}}` +
      `<br>proc recv: ${{(e.proc_recv_mibps ?? 0).toFixed(2)}} MiB/s` +
      (e.attrib_skew_s !== undefined ? `<br><span class="muted">attribution skew: ${{Number(e.attrib_skew_s).toFixed(1)}}s</span>` : '')
    );

    const netSentSpikes = PAYLOAD.net_sent_spikes || [];
    const netSentSpikeX = netSentSpikes.map(e => e.ts);
    const netSentSpikeY = netSentSpikes.map(e => e.net_sent_mibps);
    const netSentSpikeText = netSentSpikes.map(e =>
      `${{(e.proc && String(e.proc).length) ? e.proc : '(unattributed)'}} #${{e.pid ?? ''}}` +
      `<br>proc sent: ${{(e.proc_sent_mibps ?? 0).toFixed(2)}} MiB/s` +
      (e.attrib_skew_s !== undefined ? `<br><span class="muted">attribution skew: ${{Number(e.attrib_skew_s).toFixed(1)}}s</span>` : '')
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
      window.__DISK_READ_SPIKE_TRACE_INDEX = diskSpikeTraceIndex;
      window.__DISK_READ_SPIKE_COUNT = diskSpikeX.length;
      window.__DISK_READ_SPIKE_DEFAULT_COLOR = '#fbbf24';
      window.__DISK_READ_SPIKE_HILITE_COLOR = '#fde047';
      window.__DISK_READ_SPIKE_DEFAULT_SIZE = 10;
      window.__DISK_READ_SPIKE_HILITE_SIZE = 14;
    }}

    if (diskWriteSpikes.length > 0) {{
      const diskWriteSpikeTraceIndex = traces.length;
      traces.push({{
        x: diskWriteSpikeX,
        y: diskWriteSpikeY,
        name: 'Disk write spike',
        mode: 'markers',
        marker: {{ size: 10, color: '#38bdf8', symbol: 'diamond-open', line: {{ width: 1, color: 'rgba(255,255,255,0.75)' }} }},
        text: diskWriteSpikeText,
        hovertemplate: '<b>Disk write spike</b><br>%{{x}}<br>disk write=%{{y:.2f}} MiB/s<br>%{{text}}<extra></extra>',
        yaxis: 'y4'
      }});
      window.__DISK_WRITE_SPIKE_TRACE_INDEX = diskWriteSpikeTraceIndex;
      window.__DISK_WRITE_SPIKE_COUNT = diskWriteSpikeX.length;
      window.__DISK_WRITE_SPIKE_DEFAULT_COLOR = '#38bdf8';
      window.__DISK_WRITE_SPIKE_HILITE_COLOR = '#a5f3fc';
      window.__DISK_WRITE_SPIKE_DEFAULT_SIZE = 10;
      window.__DISK_WRITE_SPIKE_HILITE_SIZE = 14;
    }}

    if (netRecvSpikes.length > 0) {{
      const netRecvSpikeTraceIndex = traces.length;
      traces.push({{
        x: netRecvSpikeX,
        y: netRecvSpikeY,
        name: 'Net in spike',
        mode: 'markers',
        marker: {{ size: 9, color: '#22c55e', symbol: 'triangle-down', line: {{ width: 1, color: 'rgba(255,255,255,0.75)' }} }},
        text: netRecvSpikeText,
        hovertemplate: '<b>Net in spike</b><br>%{{x}}<br>net in=%{{y:.2f}} MiB/s<br>%{{text}}<extra></extra>',
        yaxis: 'y3'
      }});
      window.__NET_RECV_SPIKE_TRACE_INDEX = netRecvSpikeTraceIndex;
      window.__NET_RECV_SPIKE_COUNT = netRecvSpikeX.length;
      window.__NET_RECV_SPIKE_DEFAULT_COLOR = '#22c55e';
      window.__NET_RECV_SPIKE_HILITE_COLOR = '#fbbf24';
      window.__NET_RECV_SPIKE_DEFAULT_SIZE = 9;
      window.__NET_RECV_SPIKE_HILITE_SIZE = 13;
    }}

    if (netSentSpikes.length > 0) {{
      const netSentSpikeTraceIndex = traces.length;
      traces.push({{
        x: netSentSpikeX,
        y: netSentSpikeY,
        name: 'Net out spike',
        mode: 'markers',
        marker: {{ size: 9, color: '#34d399', symbol: 'triangle-up', line: {{ width: 1, color: 'rgba(255,255,255,0.75)' }} }},
        text: netSentSpikeText,
        hovertemplate: '<b>Net out spike</b><br>%{{x}}<br>net out=%{{y:.2f}} MiB/s<br>%{{text}}<extra></extra>',
        yaxis: 'y3'
      }});
      window.__NET_SENT_SPIKE_TRACE_INDEX = netSentSpikeTraceIndex;
      window.__NET_SENT_SPIKE_COUNT = netSentSpikeX.length;
      window.__NET_SENT_SPIKE_DEFAULT_COLOR = '#34d399';
      window.__NET_SENT_SPIKE_HILITE_COLOR = '#fbbf24';
      window.__NET_SENT_SPIKE_DEFAULT_SIZE = 9;
      window.__NET_SENT_SPIKE_HILITE_SIZE = 13;
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
    }});
  }}

  function _installMarkerTableLinkage(opts) {{
    const table = document.getElementById(opts.tableId);
    const plot = document.getElementById(opts.plotId);
    if (!table || !plot) return;
    const traceIndex = window[opts.traceIndexVar];
    const spikeCount = Number(window[opts.countVar] ?? 0);
    if (traceIndex === undefined || traceIndex === null) return;

    let pinnedIdx = null;

    function setSelectedRow(idx) {{
      const rows = table.querySelectorAll(opts.rowSelector);
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
      const defaultColor = window[opts.defaultColorVar] || opts.fallbackDefaultColor;
      const hiColor = window[opts.hiliteColorVar] || opts.fallbackHiliteColor;
      const baseSize = Number(window[opts.defaultSizeVar] ?? opts.fallbackDefaultSize);
      const hiSize = Number(window[opts.hiliteSizeVar] ?? opts.fallbackHiliteSize);
      const colors = Array(spikeCount).fill(defaultColor);
      colors[i] = hiColor;
      const sizes = Array(spikeCount).fill(baseSize);
      sizes[i] = hiSize;
      Plotly.restyle(plot, {{ 'marker.color': [colors], 'marker.size': [sizes] }}, [traceIndex]);
      setSelectedRow(i);
    }}

    function clearHighlight() {{
      const defaultColor = window[opts.defaultColorVar] || opts.fallbackDefaultColor;
      const baseSize = Number(window[opts.defaultSizeVar] ?? opts.fallbackDefaultSize);
      const colors = Array(spikeCount).fill(defaultColor);
      const sizes = Array(spikeCount).fill(baseSize);
      Plotly.restyle(plot, {{ 'marker.color': [colors], 'marker.size': [sizes] }}, [traceIndex]);
      setSelectedRow(null);
    }}

    table.addEventListener('mouseover', (ev) => {{
      const tr = ev.target.closest(opts.rowSelector);
      if (!tr) return;
      if (pinnedIdx !== null) return;
      const idx = tr.getAttribute('data-idx');
      if (idx !== null) highlightByIdx(idx);
    }});
    table.addEventListener('mouseout', (ev) => {{
      const tr = ev.target.closest(opts.rowSelector);
      if (!tr) return;
      if (pinnedIdx !== null) return;
      clearHighlight();
    }});
    table.addEventListener('click', (ev) => {{
      const tr = ev.target.closest(opts.rowSelector);
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

  function installDiskSpikeTableLinkage() {{
    _installMarkerTableLinkage({{
      tableId: 'disk_spike_table',
      plotId: 'sys_graph',
      rowSelector: 'tr.disk-read-spike-row',
      traceIndexVar: '__DISK_READ_SPIKE_TRACE_INDEX',
      countVar: '__DISK_READ_SPIKE_COUNT',
      defaultColorVar: '__DISK_READ_SPIKE_DEFAULT_COLOR',
      hiliteColorVar: '__DISK_READ_SPIKE_HILITE_COLOR',
      defaultSizeVar: '__DISK_READ_SPIKE_DEFAULT_SIZE',
      hiliteSizeVar: '__DISK_READ_SPIKE_HILITE_SIZE',
      fallbackDefaultColor: '#fbbf24',
      fallbackHiliteColor: '#fde047',
      fallbackDefaultSize: 10,
      fallbackHiliteSize: 14,
    }});
    _installMarkerTableLinkage({{
      tableId: 'disk_spike_table',
      plotId: 'sys_graph',
      rowSelector: 'tr.disk-write-spike-row',
      traceIndexVar: '__DISK_WRITE_SPIKE_TRACE_INDEX',
      countVar: '__DISK_WRITE_SPIKE_COUNT',
      defaultColorVar: '__DISK_WRITE_SPIKE_DEFAULT_COLOR',
      hiliteColorVar: '__DISK_WRITE_SPIKE_HILITE_COLOR',
      defaultSizeVar: '__DISK_WRITE_SPIKE_DEFAULT_SIZE',
      hiliteSizeVar: '__DISK_WRITE_SPIKE_HILITE_SIZE',
      fallbackDefaultColor: '#38bdf8',
      fallbackHiliteColor: '#a5f3fc',
      fallbackDefaultSize: 10,
      fallbackHiliteSize: 14,
    }});
  }}

  function installNetSpikeTableLinkage() {{
    _installMarkerTableLinkage({{
      tableId: 'net_spike_table',
      plotId: 'sys_graph',
      rowSelector: 'tr.net-recv-spike-row',
      traceIndexVar: '__NET_RECV_SPIKE_TRACE_INDEX',
      countVar: '__NET_RECV_SPIKE_COUNT',
      defaultColorVar: '__NET_RECV_SPIKE_DEFAULT_COLOR',
      hiliteColorVar: '__NET_RECV_SPIKE_HILITE_COLOR',
      defaultSizeVar: '__NET_RECV_SPIKE_DEFAULT_SIZE',
      hiliteSizeVar: '__NET_RECV_SPIKE_HILITE_SIZE',
      fallbackDefaultColor: '#22c55e',
      fallbackHiliteColor: '#fbbf24',
      fallbackDefaultSize: 9,
      fallbackHiliteSize: 13,
    }});
    _installMarkerTableLinkage({{
      tableId: 'net_spike_table',
      plotId: 'sys_graph',
      rowSelector: 'tr.net-sent-spike-row',
      traceIndexVar: '__NET_SENT_SPIKE_TRACE_INDEX',
      countVar: '__NET_SENT_SPIKE_COUNT',
      defaultColorVar: '__NET_SENT_SPIKE_DEFAULT_COLOR',
      hiliteColorVar: '__NET_SENT_SPIKE_HILITE_COLOR',
      defaultSizeVar: '__NET_SENT_SPIKE_DEFAULT_SIZE',
      hiliteSizeVar: '__NET_SENT_SPIKE_HILITE_SIZE',
      fallbackDefaultColor: '#34d399',
      fallbackHiliteColor: '#fbbf24',
      fallbackDefaultSize: 9,
      fallbackHiliteSize: 13,
    }});
  }}

  function installSpikeTableLinkage() {{
    // Unified spike table: rows have data-kind + data-idx which map into the corresponding marker traces.
    const table = document.getElementById('spike_table');
    const plot = document.getElementById('sys_graph');
    if (!table || !plot) return;

    let pinned = null; // pinned.kind (string), pinned.idx (number)

    function setSelected(kind, idx) {{
      const rows = table.querySelectorAll('tr.spike-row');
      rows.forEach(r => {{
        const rk = r.getAttribute('data-kind');
        const ri = r.getAttribute('data-idx');
        if (kind && idx !== null && idx !== undefined && rk === kind && String(ri) === String(idx)) r.classList.add('selected');
        else r.classList.remove('selected');
      }});
    }}

    function resetCpu() {{
      const trace = window.__SPIKE_TRACE_INDEX;
      const glow = window.__SPIKE_GLOW_TRACE_INDEX;
      const count = Number(window.__SPIKE_COUNT ?? 0);
      if (trace === undefined || trace === null || count <= 0) return;
      Plotly.restyle(plot, {{
        'marker.color': [Array(count).fill(window.__SPIKE_DEFAULT_COLOR || '#fb7185')],
        'marker.size': [Array(count).fill(Number(window.__SPIKE_DEFAULT_SIZE ?? 10))],
      }}, [trace]);
      if (glow !== undefined && glow !== null) {{
        Plotly.restyle(plot, {{
          'marker.color': [Array(count).fill(window.__SPIKE_GLOW_DEFAULT_COLOR || 'rgba(251,113,133,0.22)')],
          'marker.size': [Array(count).fill(Number(window.__SPIKE_GLOW_DEFAULT_SIZE ?? 18))],
        }}, [glow]);
      }}
    }}

    function resetSimple(prefix, fallbackColor, fallbackSize) {{
      const trace = window[`__${{prefix}}_TRACE_INDEX`];
      const count = Number(window[`__${{prefix}}_COUNT`] ?? 0);
      if (trace === undefined || trace === null || count <= 0) return;
      const c0 = window[`__${{prefix}}_DEFAULT_COLOR`] || fallbackColor;
      const s0 = Number(window[`__${{prefix}}_DEFAULT_SIZE`] ?? fallbackSize);
      Plotly.restyle(plot, {{ 'marker.color': [Array(count).fill(c0)], 'marker.size': [Array(count).fill(s0)] }}, [trace]);
    }}

    function resetAll() {{
      resetCpu();
      resetSimple('DISK_READ_SPIKE', '#fbbf24', 10);
      resetSimple('DISK_WRITE_SPIKE', '#38bdf8', 10);
      resetSimple('NET_RECV_SPIKE', '#22c55e', 9);
      resetSimple('NET_SENT_SPIKE', '#34d399', 9);
      setSelected(null, null);
    }}

    function highlightCpu(idx) {{
      const trace = window.__SPIKE_TRACE_INDEX;
      const glow = window.__SPIKE_GLOW_TRACE_INDEX;
      const count = Number(window.__SPIKE_COUNT ?? 0);
      const i = Number(idx);
      if (trace === undefined || trace === null || count <= 0 || !Number.isFinite(i) || i < 0 || i >= count) return;

      const colors = Array(count).fill(window.__SPIKE_DEFAULT_COLOR || '#fb7185');
      colors[i] = window.__SPIKE_HILITE_COLOR || '#fbbf24';
      const baseSize = Number(window.__SPIKE_DEFAULT_SIZE ?? 10);
      const hiSize = Number(window.__SPIKE_HILITE_SIZE ?? 14);
      const sizes = Array(count).fill(baseSize);
      sizes[i] = hiSize;
      Plotly.restyle(plot, {{ 'marker.color': [colors], 'marker.size': [sizes] }}, [trace]);

      if (glow !== undefined && glow !== null) {{
        const gColors = Array(count).fill(window.__SPIKE_GLOW_DEFAULT_COLOR || 'rgba(251,113,133,0.22)');
        gColors[i] = window.__SPIKE_GLOW_HILITE_COLOR || 'rgba(251,191,36,0.30)';
        const gBase = Number(window.__SPIKE_GLOW_DEFAULT_SIZE ?? 18);
        const gHi = Number(window.__SPIKE_GLOW_HILITE_SIZE ?? 24);
        const gSizes = Array(count).fill(gBase);
        gSizes[i] = gHi;
        Plotly.restyle(plot, {{ 'marker.color': [gColors], 'marker.size': [gSizes] }}, [glow]);
      }}
      setSelected('cpu', i);
    }}

    function highlightSimple(prefix, idx, kind, fallbackColor, fallbackHi, fallbackSize, fallbackHiSize) {{
      const trace = window[`__${{prefix}}_TRACE_INDEX`];
      const count = Number(window[`__${{prefix}}_COUNT`] ?? 0);
      const i = Number(idx);
      if (trace === undefined || trace === null || count <= 0 || !Number.isFinite(i) || i < 0 || i >= count) return;
      const c0 = window[`__${{prefix}}_DEFAULT_COLOR`] || fallbackColor;
      const c1 = window[`__${{prefix}}_HILITE_COLOR`] || fallbackHi;
      const s0 = Number(window[`__${{prefix}}_DEFAULT_SIZE`] ?? fallbackSize);
      const s1 = Number(window[`__${{prefix}}_HILITE_SIZE`] ?? fallbackHiSize);
      const colors = Array(count).fill(c0); colors[i] = c1;
      const sizes = Array(count).fill(s0); sizes[i] = s1;
      Plotly.restyle(plot, {{ 'marker.color': [colors], 'marker.size': [sizes] }}, [trace]);
      setSelected(kind, i);
    }}

    function highlight(kind, idx) {{
      resetAll();
      if (kind === 'cpu') return highlightCpu(idx);
      if (kind === 'disk_read') return highlightSimple('DISK_READ_SPIKE', idx, kind, '#fbbf24', '#fde047', 10, 14);
      if (kind === 'disk_write') return highlightSimple('DISK_WRITE_SPIKE', idx, kind, '#38bdf8', '#a5f3fc', 10, 14);
      if (kind === 'net_in') return highlightSimple('NET_RECV_SPIKE', idx, kind, '#22c55e', '#fbbf24', 9, 13);
      if (kind === 'net_out') return highlightSimple('NET_SENT_SPIKE', idx, kind, '#34d399', '#fbbf24', 9, 13);
    }}

    function clearHighlight() {{
      resetAll();
    }}

    // Event delegation so this works even if the table is rebuilt later.
    table.addEventListener('mouseover', (ev) => {{
      const tr = ev.target.closest('tr.spike-row');
      if (!tr) return;
      if (pinned) return;
      const kind = tr.getAttribute('data-kind') || '';
      const idx = tr.getAttribute('data-idx');
      if (kind && idx !== null) highlight(kind, idx);
    }});
    table.addEventListener('mouseout', (ev) => {{
      const tr = ev.target.closest('tr.spike-row');
      if (!tr) return;
      if (pinned) return;
      clearHighlight();
    }});
    table.addEventListener('click', (ev) => {{
      const tr = ev.target.closest('tr.spike-row');
      if (!tr) return;
      const kind = tr.getAttribute('data-kind') || '';
      const idx = tr.getAttribute('data-idx');
      if (!kind || idx === null) return;
      const i = Number(idx);
      if (!Number.isFinite(i)) return;
      if (pinned && pinned.kind === kind && pinned.idx === i) {{
        pinned = null;
        clearHighlight();
      }} else {{
        pinned = {{ kind, idx: i }};
        highlight(kind, i);
      }}
    }});

    // If user clicks elsewhere on the plot, unpin.
    plot.on('plotly_click', () => {{
      pinned = null;
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

  function renderGhRateLimitGraph() {{
    const d = PAYLOAD.gh_rate_limit || {{}};
    const keys = Object.keys(d || {{}});
    const traces = [];
    for (let i = 0; i < keys.length; i++) {{
      const k = keys[i];
      const row = d[k] || {{}};
      const x = row.x || [];
      const rem = row.remaining || [];
      const lim = row.limit || [];
      // Remaining
      traces.push({{
        x,
        y: rem,
        name: `${{k}} remaining`,
        mode: 'lines',
        line: {{ width: 2.0 }},
        hovertemplate: `resource=${{k}}<br>remaining=%{{y}}<extra></extra>`,
      }});
      // Limit (dashed)
      const limConst = lim && lim.length ? lim[lim.length - 1] : null;
      if (limConst !== null && limConst !== undefined) {{
        traces.push({{
          x,
          y: x.map(() => limConst),
          name: `${{k}} limit`,
          mode: 'lines',
          line: {{ width: 1.2, dash: DASH.dot }},
          hovertemplate: `resource=${{k}}<br>limit=%{{y}}<extra></extra>`,
        }});
      }}
    }}
    const layout = commonLayout('');
    layout.yaxis = {{ title: 'requests remaining', rangemode: 'tozero', gridcolor: 'rgba(255,255,255,0.09)' }};
    const config = {{ displayModeBar: true, responsive: true }};
    if (!keys.length) {{
      return Plotly.newPlot('gh_rate_limit_graph', [{{x: [PAYLOAD.window_end], y: [0], mode:'text', text:['No GitHub rate-limit samples in window'], textfont:{{size:14}}, hoverinfo:'skip'}}], layout, config)
        .then(() => installHoverDelay('gh_rate_limit_graph'));
    }}
    return Plotly.newPlot('gh_rate_limit_graph', traces, layout, config).then(() => installHoverDelay('gh_rate_limit_graph'));
  }}

  buildSpikeTable(PAYLOAD.spikes || []);
  buildLeaderTable(PAYLOAD.cpu_leaderboard || []);
  buildDockerTable(PAYLOAD.docker_leaderboard || []);
  buildPingTable(PAYLOAD.ping_summary || []);
  buildNetTable(PAYLOAD.net_leaderboard || []);
  const pSys = renderSystemGraph();
  const pGpu = renderGpuGraph();
  const pDocker = renderDockerGraph();
  const pPing = renderPingGraph();
  const pNet = renderNetGraph();
  const pGh = renderGhRateLimitGraph();
  Promise.allSettled([pSys, pGpu, pDocker, pPing, pNet, pGh]).then(() => {{
    installTimeSync(['sys_graph', 'gpu_graph', 'docker_graph', 'ping_graph', 'net_graph', 'gh_rate_limit_graph']);
  }});
  </script>
</body>
</html>
"""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate an HTML resource report from resource_monitor.sqlite")
    p.add_argument("--db-path", type=Path, default=_default_db_path(), help="SQLite DB path")
    p.add_argument("--output", type=Path, required=True, help="Output HTML file path")
    p.add_argument("--days", type=float, default=2.0, help="How many days to include (max: 2, default: 2)")
    p.add_argument("--title", default="keivenc-linux Resource Report", help="HTML title")
    p.add_argument(
        "--gh-rate-limit-resources",
        default="core,search,graphql",
        help="Comma-separated GitHub rate-limit resources to chart (default: core,search,graphql)",
    )

    p.add_argument(
        "--prune-db-days",
        type=float,
        default=None,
        help="If set, delete DB samples older than this many days (based on latest sample timestamp). "
        "Example: --prune-db-days 2",
    )
    p.add_argument(
        "--db-checkpoint-truncate",
        action="store_true",
        help="Run PRAGMA wal_checkpoint(TRUNCATE) to keep the -wal file small (safe for cron).",
    )
    p.add_argument(
        "--db-vacuum",
        action="store_true",
        help="Best-effort VACUUM to reclaim disk space (may fail if DB is in use).",
    )

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
    p.add_argument("--min-disk-write-spike-mibps", type=float, default=50.0, help="Minimum disk write MiB/s for a spike")
    p.add_argument("--disk-spike-sigma", type=float, default=4.0, help="MAD-based sigma threshold multiplier for disk spikes")
    p.add_argument("--max-disk-spikes", type=int, default=50, help="Max disk spikes to show/annotate")
    p.add_argument("--min-net-recv-spike-mibps", type=float, default=20.0, help="Minimum net in MiB/s for a spike")
    p.add_argument("--min-net-sent-spike-mibps", type=float, default=20.0, help="Minimum net out MiB/s for a spike")
    p.add_argument("--net-spike-sigma", type=float, default=4.0, help="MAD-based sigma threshold multiplier for net spikes")
    p.add_argument("--max-net-spikes", type=int, default=50, help="Max net spikes to show/annotate")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    db_path: Path = args.db_path
    out_path: Path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = _connect(db_path)
    try:
        days = float(args.days)
        if days > 2.0:
            raise SystemExit("--days is capped at 2 (use 2 or less).")
        if days <= 0.0:
            raise SystemExit("--days must be > 0.")

        # Optional DB prune (keeps resource_monitor.sqlite tidy).
        prune_days = getattr(args, "prune_db_days", None)
        if prune_days is not None:
            pd = float(prune_days)
            if pd <= 0.0:
                raise SystemExit("--prune-db-days must be > 0 (or omit the flag).")
            cur = con.cursor()
            max_ts = cur.execute("SELECT MAX(ts_unix) AS t FROM samples").fetchone()["t"]
            if max_ts:
                end_ts_tmp = float(max_ts)
                cutoff = end_ts_tmp - pd * 86400.0
                deleted = _prune_db_samples_older_than(con, cutoff_ts_unix=cutoff)
                if deleted:
                    print(f"Pruned {deleted} samples older than {pd:g} days.")

        if bool(getattr(args, "db_checkpoint_truncate", False)):
            rows = _wal_checkpoint_truncate(con)
            if rows:
                # (busy, log, checkpointed)
                b, l, c = rows[0]
                print(f"wal_checkpoint(TRUNCATE): busy={b} log={l} checkpointed={c}")

        start_ts, end_ts = _get_time_window(con, days=days)

        gh_resources = [
            s.strip()
            for s in str(getattr(args, "gh_rate_limit_resources", "") or "").split(",")
            if s.strip()
        ]
        if not gh_resources:
            gh_resources = ["core", "search", "graphql"]

        # GitHub rate limit series MUST come from the DB (recorded by resource_monitor.py).
        gh_rate_limit, gh_rate_limit_latest = _query_gh_rate_limit_timeseries(
            con, start_ts=start_ts, resources=gh_resources
        )

        samples = _query_samples(con, start_ts=start_ts)
        avg_interval = 0.0
        if samples:
            avg_interval = sum(s.interval_s for s in samples) / max(1, len(samples))
        gpus = _query_gpu_timeseries(con, start_ts=start_ts)
        top_io = _query_top_io_read_process_per_sample(con, start_ts=start_ts)
        top_io_w = _query_top_io_write_process_per_sample(con, start_ts=start_ts)
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

        disk_write_spikes = _detect_disk_write_spikes(
            samples,
            min_mibps=float(args.min_disk_write_spike_mibps),
            sigma=float(args.disk_spike_sigma),
            max_spikes=int(args.max_disk_spikes),
        )
        for e in disk_write_spikes:
            sid = int(e.get("sample_id") or 0)
            p = top_io_w.get(sid) or {}
            e["pid"] = p.get("pid")
            e["proc_name"] = p.get("name", "")
            e["cmdline"] = p.get("cmdline", "")
            e["proc_io_write_mibps"] = (
                float(p.get("io_write_bps") or 0.0) / 1024.0 / 1024.0
                if p.get("io_write_bps") is not None
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

        net_recv_spikes = _detect_net_spikes(
            samples,
            which="recv",
            min_mibps=float(args.min_net_recv_spike_mibps),
            sigma=float(args.net_spike_sigma),
            max_spikes=int(args.max_net_spikes),
        )
        net_sent_spikes = _detect_net_spikes(
            samples,
            which="sent",
            min_mibps=float(args.min_net_sent_spike_mibps),
            sigma=float(args.net_spike_sigma),
            max_spikes=int(args.max_net_spikes),
        )
        # Network attribution: net_process_samples may only exist on some sample_ids (collected every ~N seconds).
        # Correlate each net spike to the nearest net_process_samples sample in time (within a tolerance).
        net_pts = _query_net_top_points(con, start_ts=start_ts)
        if net_pts:
            import bisect

            ts_list = [float(p["ts_unix"]) for p in net_pts]
            # Allow a little slack vs the system sampling interval, since net-top is periodic and can drift.
            # In practice, net_process_samples can be sparse (tool failures, permissions, interval drift).
            # Use a larger tolerance so we can still show "best-effort" offender attribution; the hover/table
            # includes attrib_skew_s so users can judge how close the match was.
            max_skew_s = max(300.0, 3.0 * float(avg_interval or 0.0))

            def nearest_point(tsu: float) -> Optional[Dict[str, Any]]:
                i = bisect.bisect_left(ts_list, tsu)
                cand: List[Dict[str, Any]] = []
                if 0 <= i < len(net_pts):
                    cand.append(net_pts[i])
                if 0 <= i - 1 < len(net_pts):
                    cand.append(net_pts[i - 1])
                if not cand:
                    return None
                best = min(cand, key=lambda p: abs(float(p["ts_unix"]) - tsu))
                if abs(float(best["ts_unix"]) - tsu) <= max_skew_s:
                    return best
                return None

            for e in net_recv_spikes:
                tsu = float(e.get("ts_unix") or 0.0)
                p = nearest_point(tsu) or {}
                e["pid"] = p.get("top_recv_pid")
                e["proc"] = p.get("top_recv_proc", "")
                e["proc_recv_mibps"] = (
                    float(p.get("top_recv_bps") or 0.0) / 1024.0 / 1024.0
                    if p.get("top_recv_bps") is not None
                    else 0.0
                )
                if p:
                    e["attrib_skew_s"] = abs(float(p.get("ts_unix") or 0.0) - tsu)

            for e in net_sent_spikes:
                tsu = float(e.get("ts_unix") or 0.0)
                p = nearest_point(tsu) or {}
                e["pid"] = p.get("top_sent_pid")
                e["proc"] = p.get("top_sent_proc", "")
                e["proc_sent_mibps"] = (
                    float(p.get("top_sent_bps") or 0.0) / 1024.0 / 1024.0
                    if p.get("top_sent_bps") is not None
                    else 0.0
                )
                if p:
                    e["attrib_skew_s"] = abs(float(p.get("ts_unix") or 0.0) - tsu)
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

        payload: Dict[str, Any] = {
            "title": str(args.title),
            "db_path": str(db_path),
            "gh_rate_limit": gh_rate_limit,
            "gh_rate_limit_latest": gh_rate_limit_latest,
            "window": f"{datetime.fromtimestamp(start_ts)} → {datetime.fromtimestamp(end_ts)} (localtime) | last {days:g} days",
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
            "disk_write_spikes": disk_write_spikes,
            "net_recv_spikes": net_recv_spikes,
            "net_sent_spikes": net_sent_spikes,
            "cpu_leaderboard": leaderboard,
        }

        atomic_write_text(out_path, _build_html(payload), encoding="utf-8")
        print(f"Wrote: {out_path}")

        if bool(getattr(args, "db_vacuum", False)):
            # Run VACUUM after writing output so we don't lock ourselves out while generating.
            # Best-effort only: skip if another process holds the DB.
            ok = _vacuum_best_effort(db_path)
            print(f"VACUUM: {'ok' if ok else 'skipped (db busy)'}")
        return 0
    finally:
        try:
            con.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())


