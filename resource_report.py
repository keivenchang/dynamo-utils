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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


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
        SELECT
          ts_unix, interval_s,
          cpu_percent, mem_percent, load1,
          net_sent_bps, net_recv_bps,
          disk_read_bps, disk_write_bps,
          mem_used_bytes, mem_total_bytes
        FROM samples
        WHERE ts_unix >= ?
        ORDER BY ts_unix ASC
        """,
        (start_ts,),
    ).fetchall()
    out: List[SampleRow] = []
    for r in rows:
        out.append(
            SampleRow(
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


def _query_top_process_per_sample(con: sqlite3.Connection, *, start_ts: float) -> List[Dict[str, Any]]:
    """
    Best-effort: process_samples only stores "offenders", so this finds the max cpu process among recorded ones.
    """
    cur = con.cursor()
    rows = cur.execute(
        """
        WITH ranked AS (
          SELECT
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
        out.append(
            {
                "ts_unix": float(r["ts_unix"]),
                "ts": _ts_to_iso(float(r["ts_unix"])),
                "interval_s": float(r["interval_s"] or 0.0),
                "pid": int(r["pid"]) if r["pid"] is not None else None,
                "name": r["name"] or "",
                "username": r["username"] or "",
                "cmdline": r["cmdline"] or "",
                "cpu_percent": float(r["cpu_percent"]) if r["cpu_percent"] is not None else 0.0,
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
    vals = [float(p.get("cpu_percent") or 0.0) for p in top_proc]
    med, mad = _median_mad(vals)
    # Scale MAD -> approx stddev for normal dist.
    robust_std = 1.4826 * mad
    thresh = max(float(min_cpu_percent), med + float(sigma) * robust_std)

    spikes: List[Dict[str, Any]] = []
    for i, p in enumerate(top_proc):
        v = float(p.get("cpu_percent") or 0.0)
        if v < thresh:
            continue
        # Local max guard to reduce flat "plateaus"
        prev_v = float(top_proc[i - 1].get("cpu_percent") or 0.0) if i > 0 else -1.0
        next_v = float(top_proc[i + 1].get("cpu_percent") or 0.0) if i + 1 < len(top_proc) else -1.0
        if v < prev_v or v < next_v:
            continue
        spikes.append(
            {
                "ts": p["ts"],
                "ts_unix": p["ts_unix"],
                "cpu_percent": v,
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
    html, body {{
      height: 100%;
      margin: 0;
      background: radial-gradient(1200px 700px at 10% 10%, rgba(124,58,237,0.22), transparent 60%),
                  radial-gradient(1000px 700px at 90% 20%, rgba(110,231,255,0.18), transparent 60%),
                  linear-gradient(180deg, var(--bg0), var(--bg1));
      color: var(--text);
      font-family: var(--sans);
    }}
    .wrap {{
      max-width: 1280px;
      margin: 0 auto;
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
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div>
        <h1>⚡ {payload.get("title","Resource Report")}</h1>
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
      <div class="card">
        <div class="hdr">
          <div class="title">System: CPU / Memory / IO</div>
          <div class="hint">Hover to inspect; drag to zoom; double-click to reset</div>
        </div>
        <div class="body">
          <div id="sys_graph" style="height:520px;"></div>
        </div>
      </div>

      <div class="card">
        <div class="hdr">
          <div class="title">Spike events (top process CPU)</div>
          <div class="hint"><span class="pill">MAD threshold</span></div>
        </div>
        <div class="body" style="max-height: 560px; overflow:auto;">
          <table class="table" id="spike_table"></table>
        </div>
      </div>
    </div>

    <div class="grid" style="grid-template-columns: 1.2fr 0.8fr; margin-top: 16px;">
      <div class="card">
        <div class="hdr">
          <div class="title">GPU: Util / Memory / Temp / Power</div>
          <div class="hint">If no GPU samples exist, this stays empty</div>
        </div>
        <div class="body">
          <div id="gpu_graph" style="height:520px;"></div>
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
  </div>

  <script>
  const PAYLOAD = {_json(payload)};

  function bytesToGiB(b) {{
    if (b === null || b === undefined) return null;
    return b / 1024 / 1024 / 1024;
  }}

  function bpsToMiBps(x) {{
    if (x === null || x === undefined) return null;
    return x / 1024 / 1024;
  }}

  function buildSpikeTable(spikes) {{
    const el = document.getElementById('spike_table');
    if (!spikes || spikes.length === 0) {{
      el.innerHTML = `<tr><th>status</th></tr><tr><td class="muted">No spikes detected (or process data missing in window).</td></tr>`;
      return;
    }}
    el.innerHTML = `
      <tr>
        <th>time</th>
        <th>cpu%</th>
        <th>proc</th>
        <th>rss</th>
        <th>gpu</th>
      </tr>
      ${{spikes.map(s => `
        <tr>
          <td class="mono">${{s.ts.replace('T',' ')}}</td>
          <td><span class="badge bad">${{s.cpu_percent.toFixed(1)}}%</span></td>
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

  function commonLayout(title) {{
    const grid = 'rgba(255,255,255,0.09)';
    return {{
      title: {{ text: title, font: {{ size: 13, color: 'rgba(255,255,255,0.85)' }} }},
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      margin: {{ l: 55, r: 20, t: 42, b: 40 }},
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
        rangeselector: {{
          bgcolor: 'rgba(255,255,255,0.06)',
          bordercolor: 'rgba(255,255,255,0.12)',
          borderwidth: 1,
          buttons: [
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
      legend: {{ orientation: 'h', y: 1.08, x: 0.0 }},
    }};
  }}

  function renderSystemGraph() {{
    const s = PAYLOAD.samples;
    const x = s.map(r => r.ts);
    const cpu = s.map(r => r.cpu_percent);
    const mem = s.map(r => r.mem_percent);
    const load1 = s.map(r => r.load1);
    const netOut = s.map(r => bpsToMiBps(r.net_sent_bps));
    const netIn = s.map(r => bpsToMiBps(r.net_recv_bps));
    const diskR = s.map(r => bpsToMiBps(r.disk_read_bps));
    const diskW = s.map(r => bpsToMiBps(r.disk_write_bps));

    const spikes = PAYLOAD.spikes || [];
    const spikeX = spikes.map(e => e.ts);
    const spikeY = spikes.map(e => e.cpu_percent);
    const spikeText = spikes.map(e => `${{e.name || ''}} #${{e.pid ?? ''}}<br>${{(e.cmdline||'').slice(0,120)}}`);

    const traces = [
      {{ x, y: cpu, name: 'CPU % (system)', mode: 'lines', line: {{ color: '#6ee7ff', width: 2 }} , yaxis: 'y1' }},
      {{ x, y: mem, name: 'MEM % (system)', mode: 'lines', line: {{ color: '#a78bfa', width: 2 }} , yaxis: 'y1' }},
      {{ x, y: load1, name: 'load1', mode: 'lines', line: {{ color: 'rgba(255,255,255,0.45)', width: 1.6, dash: 'dot' }} , yaxis: 'y2' }},
      {{ x, y: netOut, name: 'net out (MiB/s)', mode: 'lines', line: {{ color: '#34d399', width: 1.6 }} , yaxis: 'y3' }},
      {{ x, y: netIn, name: 'net in (MiB/s)', mode: 'lines', line: {{ color: '#22c55e', width: 1.6, dash: 'dot' }} , yaxis: 'y3' }},
      {{ x, y: diskR, name: 'disk read (MiB/s)', mode: 'lines', line: {{ color: '#fbbf24', width: 1.6 }} , yaxis: 'y4' }},
      {{ x, y: diskW, name: 'disk write (MiB/s)', mode: 'lines', line: {{ color: '#fb7185', width: 1.6 }} , yaxis: 'y4' }},
    ];

    if (spikes.length > 0) {{
      traces.push({{
        x: spikeX,
        y: spikeY,
        name: 'Top-proc CPU spike',
        mode: 'markers',
        marker: {{ size: 10, color: '#fb7185', line: {{ width: 1, color: 'rgba(255,255,255,0.7)' }} }},
        text: spikeText,
        hovertemplate: '<b>Spike</b><br>%{{x}}<br>top-proc cpu=%{{y:.1f}}%<br>%{{text}}<extra></extra>',
        yaxis: 'y1'
      }});
    }}

    const layout = commonLayout('System timeline');
    layout.yaxis = {{ title: 'CPU% / MEM%', rangemode: 'tozero', gridcolor: 'rgba(255,255,255,0.09)' }};
    layout.yaxis2 = {{ title: 'load1', overlaying: 'y', side: 'right', showgrid: false }};
    layout.yaxis3 = {{ title: 'net MiB/s', anchor: 'x', overlaying: 'y', side: 'right', position: 0.98, showgrid: false }};
    layout.yaxis4 = {{ title: 'disk MiB/s', anchor: 'x', overlaying: 'y', side: 'right', position: 0.92, showgrid: false }};

    const config = {{ displayModeBar: true, responsive: true }};
    Plotly.newPlot('sys_graph', traces, layout, config);
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
        line: {{ width: 2 }},
        yaxis: 'y1',
      }});
      traces.push({{
        x: gpu.x,
        y: gpu.mem_used,
        name: `GPU${{k}} mem used (MB)`,
        mode: 'lines',
        line: {{ width: 1.7, dash: 'dot' }},
        yaxis: 'y2',
      }});
      traces.push({{
        x: gpu.x,
        y: gpu.temp,
        name: `GPU${{k}} temp (C)`,
        mode: 'lines',
        line: {{ width: 1.4, dash: 'dash' }},
        yaxis: 'y3',
      }});
      traces.push({{
        x: gpu.x,
        y: gpu.power,
        name: `GPU${{k}} power (W)`,
        mode: 'lines',
        line: {{ width: 1.4, dash: 'dashdot' }},
        yaxis: 'y4',
      }});
    }}

    const layout = commonLayout('GPU timeline');
    layout.yaxis = {{ title: 'util %', rangemode: 'tozero', gridcolor: 'rgba(255,255,255,0.09)' }};
    layout.yaxis2 = {{ title: 'mem MB', overlaying: 'y', side: 'right', showgrid: false }};
    layout.yaxis3 = {{ title: 'temp C', overlaying: 'y', side: 'right', position: 0.96, showgrid: false }};
    layout.yaxis4 = {{ title: 'power W', overlaying: 'y', side: 'right', position: 0.92, showgrid: false }};

    const config = {{ displayModeBar: true, responsive: true }};
    if (!keys.length) {{
      Plotly.newPlot('gpu_graph', [{{x: [PAYLOAD.window_end], y: [0], mode:'text', text:['No GPU samples in window'], textfont:{{size:14}}, hoverinfo:'skip'}}], layout, config);
    }} else {{
      Plotly.newPlot('gpu_graph', traces, layout, config);
    }}
  }}

  buildSpikeTable(PAYLOAD.spikes || []);
  buildLeaderTable(PAYLOAD.cpu_leaderboard || []);
  renderSystemGraph();
  renderGpuGraph();
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

    p.add_argument("--leaderboard-limit", type=int, default=20, help="How many leaderboard rows to include")
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
            "samples": [
                {
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
                }
                for s in samples
            ],
            "gpus": gpus,
            "spikes": spikes,
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


