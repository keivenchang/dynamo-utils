#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Real-time system monitor: per-process GPU memory, GPU+CPU utilization,
GPU PCIe throughput, and per-process disk I/O.

Polls every 300ms, displays interactive graphs in the browser via Plotly Dash.
Rolling 10-minute window. Built-in snapshot to PNG.

Layout (with GPU):
  1. GPU Memory by Process (GiB) — stacked area (all processes shown)
  2. GPU Utilization + CPU Usage (%) — overlaid lines
  3. GPU PCIe Throughput (MB/s) — TX/RX lines (if supported)
  4. Disk I/O by Process (MB/s) — stacked area (top 8 + Other)

Usage:
    python3 gpu_monitor.py [--port 8050] [--host 127.0.0.1] [--interval 300] [--window 600]
"""

import argparse
import base64
import os
import threading
import time
import traceback
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import numpy  # must import before plotly to avoid numpy 2.x circular-init bug
import dash
import psutil
import plotly.graph_objects as go
import pynvml
from dash import Input, Output, State, dcc, html
from plotly.subplots import make_subplots

HAS_NVML = False
try:
    pynvml.nvmlInit()
    HAS_NVML = True
except pynvml.NVMLError as e:
    print(f"NVML init failed ({e}) -- GPU monitoring disabled")

SNAPSHOT_DIR = Path.home() / "dynamo" / "dynamo-utils.dev" / "html_pages" / "snapshots"

PROCESS_COLORS = [
    "#00e5ff", "#ff1744", "#76ff03", "#ffea00",
    "#d500f9", "#ff9100", "#00e676", "#ff6d00",
    "#448aff", "#e040fb", "#18ffff", "#f50057",
]
OTHER_COLOR = "#555555"

DISK_SCAN_EVERY = 10


def parse_args():
    p = argparse.ArgumentParser(description="Real-time GPU/CPU/Disk monitor")
    p.add_argument("--port", type=int, default=8051)
    p.add_argument("--host", default="127.0.0.1", help="Bind address (0.0.0.0 for remote)")
    p.add_argument("--interval", type=int, default=100, help="Data collection interval ms (default 100)")
    p.add_argument("--display-interval", type=int, default=750, help="Browser refresh interval ms (default 750)")
    p.add_argument("--window", type=int, default=600, help="Rolling window seconds (default 600)")
    p.add_argument("--snapshot-dir", default=str(SNAPSHOT_DIR), help="Directory for PNG snapshots")
    return p.parse_args()


def _resolve_process_name(pid: int) -> str:
    """Produce a short, human-readable label for a PID."""
    try:
        proc = psutil.Process(pid)
        cmdline = proc.cmdline()
        full_cmd = " ".join(cmdline) if cmdline else ""

        if cmdline and os.path.basename(cmdline[0]) == "node":
            label = _classify_node_process(full_cmd)
            if label:
                return f"{label}:{pid}"

        if cmdline and cmdline[0] == "docker" and "exec" in cmdline:
            if "node" in full_cmd:
                return f"Docker/Cursor:{pid}"

        if cmdline:
            base = os.path.basename(cmdline[0])
            if base.startswith("python") and len(cmdline) > 1:
                script = os.path.basename(cmdline[1])
                if len(script) > 25:
                    script = script[:22] + "..."
                return f"{script}:{pid}"
            if len(base) > 25:
                base = base[:22] + "..."
            return f"{base}:{pid}"

        return f"{proc.name()}:{pid}"
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return f"pid:{pid}"


def _classify_node_process(full_cmd: str) -> str:
    """Map Cursor/VS Code node process cmdlines to friendly names."""
    if "--type=extensionHost" in full_cmd:
        return "ExtHost"
    if "cursorpyright" in full_cmd or "pyright" in full_cmd:
        return "Pylance"
    if "pylance" in full_cmd.lower():
        return "Pylance"
    if "--type=fileWatcher" in full_cmd:
        return "FileWatcher"
    if "--type=ptyHost" in full_cmd:
        return "PtyHost"
    if "server-main.js" in full_cmd:
        return "CursorServer"
    if "multiplex-server" in full_cmd:
        return "Multiplex"
    if "forwarder.js" in full_cmd:
        return "PortFwd"
    if "markdown-language-features" in full_cmd:
        return "Markdown"
    if "rust-analyzer" in full_cmd:
        return "RustAnalyzer"
    if "gitlens" in full_cmd.lower():
        return "GitLens"
    if "typescript" in full_cmd.lower():
        return "TSServer"
    if "eslint" in full_cmd.lower():
        return "ESLint"
    if "bootstrap-fork" in full_cmd:
        return "CursorWorker"
    if ".cursor-server" in full_cmd or ".vscode-server" in full_cmd:
        return "CursorNode"
    return ""


class ProcessTracker:
    """Maintains per-process time-series with automatic backfill."""

    def __init__(self, maxlen: int, prune: bool = True):
        self.maxlen = maxlen
        self._prune_enabled = prune
        self._len = 0
        self.series: dict[int, deque[float]] = {}
        self.names: dict[int, str] = {}
        self._colors: dict[int, str] = {}
        self._color_idx = 0

    def record(self, data: dict[int, float], name_resolver):
        """Append one sample. data = {pid: value}. Missing pids get 0."""
        for pid in data:
            if pid not in self.series:
                backfill = min(self._len, self.maxlen)
                self.series[pid] = deque([0.0] * backfill, maxlen=self.maxlen)
                self.names[pid] = name_resolver(pid)

        for pid, dq in self.series.items():
            dq.append(data.get(pid, 0.0))

        self._len += 1
        if self._prune_enabled:
            self._prune_dead()

    def _prune_dead(self):
        """Drop processes that have been zero for 200+ consecutive samples (~60s)."""
        window = min(200, self._len)
        dead = [
            pid
            for pid, dq in self.series.items()
            if all(v == 0.0 for v in list(dq)[-window:])
        ]
        for pid in dead:
            del self.series[pid]
            del self.names[pid]
            self._colors.pop(pid, None)

    def get_color(self, pid: int) -> str:
        if pid not in self._colors:
            self._colors[pid] = PROCESS_COLORS[self._color_idx % len(PROCESS_COLORS)]
            self._color_idx += 1
        return self._colors[pid]

    def get_all(self) -> list[tuple[int, str, list[float]]]:
        """Return all processes (no bucketing). Sorted by peak value descending."""
        if not self.series:
            return []
        sorted_pids = sorted(
            self.series.keys(),
            key=lambda p: max(self.series[p]) if self.series[p] else 0,
            reverse=True,
        )
        result = []
        for pid in sorted_pids:
            vals = list(self.series[pid])
            if max(vals) > 0:
                result.append((pid, self.names[pid], vals))
        return result

    def get_top(self, n: int = 8, bucket_label: str = "Other") -> list[tuple[int, str, list[float]]]:
        """Top N processes by peak value. Rest bucketed under bucket_label."""
        if not self.series:
            return []

        sorted_pids = sorted(
            self.series.keys(),
            key=lambda p: max(self.series[p]) if self.series[p] else 0,
            reverse=True,
        )

        top_pids = sorted_pids[:n]
        rest_pids = sorted_pids[n:]

        result = []
        for pid in top_pids:
            vals = list(self.series[pid])
            if max(vals) > 0:
                result.append((pid, self.names[pid], vals))

        if rest_pids:
            length = len(next(iter(self.series.values())))
            rest_vals = [0.0] * length
            has_data = False
            for pid in rest_pids:
                for i, v in enumerate(self.series[pid]):
                    rest_vals[i] += v
                    if v > 0:
                        has_data = True
            if has_data:
                result.append((-1, bucket_label, rest_vals))

        return result


class MetricsCollector:
    """Samples GPU, CPU, PCIe, and disk metrics into rolling buffers."""

    def __init__(self, window_sec: int, interval_ms: int):
        self.lock = threading.Lock()
        maxlen = int(window_sec * 1000 / interval_ms)
        self.interval_sec = interval_ms / 1000.0
        self.timestamps: deque[float] = deque(maxlen=maxlen)
        self.cpu_pct: deque[float] = deque(maxlen=maxlen)

        self._prev_disk = psutil.disk_io_counters()
        self._prev_mono = time.monotonic()

        self.proc_gpu_mem = ProcessTracker(maxlen, prune=False)
        self.proc_disk_io = ProcessTracker(maxlen, prune=True)
        self._prev_proc_io: dict[int, int] = {}
        self._disk_scan_ctr = DISK_SCAN_EVERY
        self._last_disk_rates: dict[int, float] = {}
        self._last_disk_scan_mono = time.monotonic()

        self.gpu_count = 0
        self.gpu_names: list[str] = []
        self.gpu_mem_total_gib: list[float] = []
        self.gpu_util: list[deque[float]] = []
        self.gpu_pcie_tx: list[deque[float]] = []
        self.gpu_pcie_rx: list[deque[float]] = []
        self.has_pcie = False

        if HAS_NVML:
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                self.gpu_names.append(name)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.gpu_mem_total_gib.append(mem.total / (1024**3))
                self.gpu_util.append(deque(maxlen=maxlen))
                self.gpu_pcie_tx.append(deque(maxlen=maxlen))
                self.gpu_pcie_rx.append(deque(maxlen=maxlen))

            if self.gpu_count > 0:
                try:
                    h = pynvml.nvmlDeviceGetHandleByIndex(0)
                    pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_TX_BYTES)
                    self.has_pcie = True
                except pynvml.NVMLError:
                    self.has_pcie = False

    def sample(self):
        with self.lock:
            self._sample_locked()

    def _sample_locked(self):
        now = time.time()
        now_mono = time.monotonic()
        dt = now_mono - self._prev_mono
        if dt <= 0:
            dt = self.interval_sec

        self.timestamps.append(now)
        self.cpu_pct.append(psutil.cpu_percent(interval=None))
        self._prev_mono = now_mono

        if HAS_NVML:
            gpu_proc_mem: dict[int, float] = {}
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.gpu_util[i].append(util.gpu)
                except pynvml.NVMLError:
                    self.gpu_util[i].append(0.0)

                if self.has_pcie:
                    try:
                        tx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
                        rx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
                        self.gpu_pcie_tx[i].append(tx / 1024.0)
                        self.gpu_pcie_rx[i].append(rx / 1024.0)
                    except pynvml.NVMLError:
                        self.gpu_pcie_tx[i].append(0.0)
                        self.gpu_pcie_rx[i].append(0.0)

                for getter in (
                    pynvml.nvmlDeviceGetComputeRunningProcesses,
                    pynvml.nvmlDeviceGetGraphicsRunningProcesses,
                ):
                    try:
                        for p in getter(handle):
                            mem_gib = (p.usedGpuMemory or 0) / (1024**3)
                            gpu_proc_mem[p.pid] = gpu_proc_mem.get(p.pid, 0.0) + mem_gib
                    except pynvml.NVMLError:
                        pass

            self.proc_gpu_mem.record(gpu_proc_mem, _resolve_process_name)

        self._disk_scan_ctr += 1
        if self._disk_scan_ctr >= DISK_SCAN_EVERY:
            self._disk_scan_ctr = 0
            scan_mono = time.monotonic()
            scan_dt = scan_mono - self._last_disk_scan_mono
            if scan_dt <= 0:
                scan_dt = self.interval_sec * DISK_SCAN_EVERY
            self._last_disk_rates = self._scan_process_disk(scan_dt)
            self._last_disk_scan_mono = scan_mono

        self.proc_disk_io.record(self._last_disk_rates, _resolve_process_name)

    def _scan_process_disk(self, dt: float) -> dict[int, float]:
        rates: dict[int, float] = {}
        current_io: dict[int, int] = {}

        for proc in psutil.process_iter(["pid"]):
            pid = proc.info["pid"]
            try:
                io = proc.io_counters()
                total = io.read_bytes + io.write_bytes
                current_io[pid] = total
                if pid in self._prev_proc_io:
                    delta = total - self._prev_proc_io[pid]
                    if delta > 0:
                        rates[pid] = delta / (1024 * 1024) / dt
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        self._prev_proc_io = current_io
        return rates


def _make_figure(collector, has_gpu, has_pcie, view_minutes=None):
    """Build the Plotly figure. Shared by live callback and snapshot."""
    with collector.lock:
        return _make_figure_locked(collector, has_gpu, has_pcie, view_minutes)


def _make_figure_locked(collector, has_gpu, has_pcie, view_minutes=None):
    if len(collector.timestamps) == 0:
        return go.Figure()

    times = [datetime.fromtimestamp(t) for t in collector.timestamps]

    rows = []
    specs = []
    if has_gpu:
        rows.append("GPU Memory by Process (GiB)")
        specs.append([{}])
        if has_pcie:
            rows.append("GPU Util (%) + PCIe (MB/s)")
            specs.append([{"secondary_y": True}])
        else:
            rows.append("GPU Util (%)")
            specs.append([{}])
    rows.append("CPU Usage (%)")
    specs.append([{}])
    rows.append("Disk I/O by Process (MB/s)")
    specs.append([{}])

    n_rows = len(rows)
    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        specs=specs,
    )

    cur_row = 0

    # --- GPU Memory: all processes, newest on top (reversed so old = bottom) ---
    if has_gpu:
        cur_row += 1
        all_gpu = collector.proc_gpu_mem.get_all()
        if not all_gpu:
            fig.add_trace(
                go.Scatter(
                    x=times, y=[0.0] * len(times), name="GPU Mem: 0",
                    mode="lines", line={"width": 1, "color": "#555555"},
                    legendgroup="gpu_mem", legendgrouptitle_text="GPU Mem",
                ),
                row=cur_row, col=1,
            )
        else:
            for pid, name, vals in reversed(all_gpu):
                color = collector.proc_gpu_mem.get_color(pid)
                fig.add_trace(
                    go.Scatter(
                        x=times, y=vals, name=name,
                        mode="lines",
                        line={"width": 0.5, "color": color},
                        stackgroup="gpu_mem",
                        fillcolor=_alpha(color, 0.6),
                        hovertemplate="%{y:.2f} GiB<extra>" + name + "</extra>",
                        legendgroup="gpu_mem",
                        legendgrouptitle_text="GPU Mem",
                    ),
                    row=cur_row, col=1,
                )

        max_gib = max(collector.gpu_mem_total_gib) if collector.gpu_mem_total_gib else 1
        fig.update_yaxes(range=[0, max_gib * 1.02], title_text="GiB", title_font={"size": 10}, row=cur_row, col=1)
        fig.add_hline(
            y=max_gib, line_dash="dash", line_color="#ff5252", line_width=1,
            annotation_text=f"{max_gib:.1f} GiB total",
            annotation_font_color="#ff5252", annotation_font_size=11,
            row=cur_row, col=1,
        )

        # --- GPU Util (%) + PCIe (MB/s) on secondary y-axis ---
        cur_row += 1
        gpu_colors_line = ["#00e5ff", "#ff1744", "#76ff03", "#d500f9"]
        for i in range(collector.gpu_count):
            color = gpu_colors_line[i % len(gpu_colors_line)]
            label = f"GPU {i} Util" if collector.gpu_count > 1 else "GPU Util"
            fig.add_trace(
                go.Scattergl(
                    x=times, y=list(collector.gpu_util[i]), name=label,
                    line={"color": color, "width": 2},
                    fill="tozeroy", fillcolor=_alpha(color, 0.12),
                    hovertemplate="%{y:.0f}%<extra>" + label + "</extra>",
                    legendgroup="util", legendgrouptitle_text="GPU Util",
                ),
                row=cur_row, col=1,
            )
        fig.update_yaxes(range=[0, 105], title_text="%", title_font={"size": 10}, row=cur_row, col=1)

        if has_pcie:
            pcie_tx_color = "#40c4ff"
            pcie_rx_color = "#ff6e40"
            for i in range(collector.gpu_count):
                suffix = f" (GPU {i})" if collector.gpu_count > 1 else ""
                tx_data = list(collector.gpu_pcie_tx[i])
                rx_data = list(collector.gpu_pcie_rx[i])
                fig.add_trace(
                    go.Scattergl(
                        x=times, y=tx_data,
                        name=f"PCIe TX{suffix}",
                        line={"color": pcie_tx_color, "width": 1.5, "dash": "dash"},
                        hovertemplate="%{y:.1f} MB/s<extra>TX (Host->GPU)</extra>",
                        legendgroup="pcie", legendgrouptitle_text="PCIe",
                    ),
                    row=cur_row, col=1, secondary_y=True,
                )
                fig.add_trace(
                    go.Scattergl(
                        x=times, y=rx_data,
                        name=f"PCIe RX{suffix}",
                        line={"color": pcie_rx_color, "width": 1.5, "dash": "dash"},
                        hovertemplate="%{y:.1f} MB/s<extra>RX (GPU->Host)</extra>",
                        legendgroup="pcie",
                    ),
                    row=cur_row, col=1, secondary_y=True,
                )
            all_pcie = []
            for i in range(collector.gpu_count):
                all_pcie.extend(collector.gpu_pcie_tx[i])
                all_pcie.extend(collector.gpu_pcie_rx[i])
            pcie_max = max(all_pcie) if all_pcie else 100
            pcie_ceil = max(100, pcie_max * 1.2)
            fig.update_yaxes(
                title_text="PCIe MB/s", title_font={"color": "#40c4ff", "size": 10},
                showgrid=False,
                range=[0, pcie_ceil],
                row=cur_row, col=1, secondary_y=True,
            )

    # --- CPU Usage (separate) ---
    cur_row += 1
    fig.add_trace(
        go.Scattergl(
            x=times, y=list(collector.cpu_pct), name="CPU",
            line={"color": "#ffcc00", "width": 1},
            fill="tozeroy", fillcolor="rgba(255,204,0,0.10)",
            hovertemplate="%{y:.1f}%<extra>CPU</extra>",
            legendgroup="cpu", legendgrouptitle_text="CPU",
        ),
        row=cur_row, col=1,
    )
    fig.update_yaxes(range=[0, 105], title_text="%", title_font={"size": 10}, row=cur_row, col=1)

    # --- Disk I/O by Process (top 20 + Other) ---
    cur_row += 1
    top_disk = collector.proc_disk_io.get_top(20)
    for pid, name, vals in top_disk:
        color = OTHER_COLOR if pid == -1 else collector.proc_disk_io.get_color(pid)
        fig.add_trace(
            go.Scatter(
                x=times, y=vals, name=name,
                mode="lines",
                line={"width": 0.5, "color": color},
                stackgroup="disk_io",
                fillcolor=_alpha(color, 0.55),
                hovertemplate="%{y:.1f} MB/s<extra>" + name + "</extra>",
                legendgroup="disk_io", legendgrouptitle_text="Disk I/O",
            ),
            row=cur_row, col=1,
        )

    fig.update_yaxes(title_text="MB/s", title_font={"size": 10}, row=cur_row, col=1)

    # --- Layout ---
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d0d14",
        plot_bgcolor="#141420",
        font={"family": "monospace", "color": "#e0e0e0", "size": 11},
        margin={"l": 70, "r": 20, "t": 40, "b": 35},
        legend={
            "orientation": "v", "yanchor": "top", "y": 1,
            "xanchor": "left", "x": 1.01,
            "font": {"size": 10},
            "bgcolor": "rgba(13,13,20,0.8)",
        },
        hovermode=False,
        uirevision="constant",
    )

    bg_colors = ["#101020", "#181830"]
    y_idx = 0
    for r in range(1, n_rows + 1):
        y_idx += 1
        bg = bg_colors[r % 2]
        x_ref = f"x{r} domain" if r > 1 else "x domain"
        y_ref = f"y{y_idx} domain" if y_idx > 1 else "y domain"
        fig.add_shape(
            type="rect", x0=0, x1=1, y0=0, y1=1,
            xref=x_ref, yref=y_ref,
            fillcolor=bg, line_width=0, layer="below",
        )
        fig.add_annotation(
            text=f"  {rows[r - 1]}  ",
            x=0.0, y=0.97,
            xref=x_ref, yref=y_ref,
            showarrow=False,
            font={"size": 11, "color": "#c0c0d0", "family": "monospace"},
            xanchor="left", yanchor="top",
            bgcolor="rgba(30,30,50,0.92)",
            bordercolor="#444466",
            borderwidth=1,
            borderpad=3,
        )
        fig.update_xaxes(gridcolor="#262640", row=r, col=1)
        fig.update_yaxes(gridcolor="#262640", row=r, col=1)
        if specs[r - 1][0].get("secondary_y"):
            y_idx += 1

    # Apply time window if set (from the 1m/2m/5m/10m buttons)
    if view_minutes and times:
        x_end = times[-1]
        x_start = x_end - timedelta(minutes=view_minutes)
        for r in range(1, n_rows + 1):
            fig.update_xaxes(range=[x_start, x_end], row=r, col=1)


    return fig


def build_app(collector: MetricsCollector, args) -> dash.Dash:
    app = dash.Dash(__name__, title="GPU Monitor")

    has_gpu = collector.gpu_count > 0
    has_pcie = collector.has_pcie

    info_msgs = []
    if not has_gpu:
        info_msgs.append(
            html.Div("No NVIDIA GPUs detected -- showing CPU + Disk only.",
                      style={"color": "#ff6b6b", "marginBottom": "8px", "fontSize": "14px"})
        )
    elif not has_pcie:
        info_msgs.append(
            html.Div("GPU PCIe throughput not supported by driver -- skipping PCIe subplot.",
                      style={"color": "#ffcc00", "marginBottom": "8px", "fontSize": "13px"})
        )

    range_btn_style = {
        "backgroundColor": "#2a2a3c",
        "color": "#e0e0e0",
        "border": "1px solid #444",
        "padding": "6px 14px",
        "cursor": "pointer",
        "fontFamily": "monospace",
        "fontWeight": "bold",
        "marginRight": "4px",
        "borderRadius": "4px",
        "fontSize": "13px",
    }

    app.layout = html.Div(
        style={"backgroundColor": "#0d0d14", "minHeight": "100vh", "padding": "20px", "fontFamily": "monospace"},
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "12px"},
                children=[
                    html.H1("System Monitor", style={"color": "#e0e0e0", "margin": "0", "fontSize": "22px"}),
                    html.Div(children=[
                        # Time-range buttons
                        html.Button("1m", id="range-1m", n_clicks=0, style=range_btn_style),
                        html.Button("2m", id="range-2m", n_clicks=0, style=range_btn_style),
                        html.Button("5m", id="range-5m", n_clicks=0, style=range_btn_style),
                        html.Button("10m", id="range-10m", n_clicks=0, style=range_btn_style),
                        html.Button("All", id="range-all", n_clicks=0, style=range_btn_style),
                        html.Span("  ", style={"marginRight": "12px"}),
                        html.Button("Pause", id="pause-btn", n_clicks=0, style=_btn_style("#ffcc00")),
                        html.Button("Snapshot", id="snapshot-btn", n_clicks=0, style=_btn_style("#00e676")),
                    ]),
                ],
            ),
            *info_msgs,
            html.Div(id="snapshot-status", style={"color": "#00e676", "marginBottom": "6px", "fontSize": "12px"}),
            dcc.Download(id="snapshot-download"),
            dcc.Graph(
                id="live-graph",
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "gpu_monitor",
                        "height": 1080,
                        "width": 1920,
                        "scale": 2,
                    },
                },
                style={"height": "88vh"},
            ),
            dcc.Interval(id="interval", interval=args.display_interval, n_intervals=0),
        ],
    )

    # Shared state between callbacks and background thread
    ui_state = {"minutes": 0, "paused": False, "last_snap": 0}

    # --- Pause toggle ---
    @app.callback(
        Output("pause-btn", "children"),
        Output("pause-btn", "style"),
        Input("pause-btn", "n_clicks"),
    )
    def toggle_pause(n_clicks):
        if n_clicks == 0:
            raise dash.exceptions.PreventUpdate
        ui_state["paused"] = not ui_state["paused"]
        paused = ui_state["paused"]
        label = "Resume" if paused else "Pause"
        color = "#ff5252" if paused else "#ffcc00"
        return label, _btn_style(color)

    # --- Main graph ---
    @app.callback(
        Output("live-graph", "figure"),
        Input("interval", "n_intervals"),
        Input("range-1m", "n_clicks"),
        Input("range-2m", "n_clicks"),
        Input("range-5m", "n_clicks"),
        Input("range-10m", "n_clicks"),
        Input("range-all", "n_clicks"),
    )
    def update_graph(n, r1, r2, r5, r10, rall):
        triggered = dash.ctx.triggered_id
        range_map = {"range-1m": 1, "range-2m": 2, "range-5m": 5, "range-10m": 10, "range-all": 0}
        if triggered in range_map:
            ui_state["minutes"] = range_map[triggered]
        return _make_figure(collector, has_gpu, has_pcie, ui_state["minutes"] or None)

    # --- Snapshot: render server-side, download to client via dcc.Download ---
    @app.callback(
        Output("snapshot-download", "data"),
        Output("snapshot-status", "children"),
        Input("snapshot-btn", "n_clicks"),
    )
    def save_snapshot(n_clicks):
        if not n_clicks or n_clicks <= ui_state["last_snap"]:
            raise dash.exceptions.PreventUpdate
        ui_state["last_snap"] = n_clicks

        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        filename = f"{ts}_GPU.png"

        fig = _make_figure(collector, has_gpu, has_pcie)
        img_bytes = fig.to_image(format="png", width=1920, height=1080, scale=2)
        print(f"Snapshot: {filename} ({len(img_bytes)} bytes)")

        b64 = base64.b64encode(img_bytes).decode("ascii")
        return {"content": b64, "filename": filename, "base64": True}, f"Downloaded: {filename}"

    return app, ui_state


def _btn_style(bg_color: str) -> dict:
    return {
        "backgroundColor": bg_color,
        "color": "#0d0d14",
        "border": "none",
        "padding": "8px 18px",
        "cursor": "pointer",
        "fontFamily": "monospace",
        "fontWeight": "bold",
        "marginRight": "10px",
        "borderRadius": "4px",
        "fontSize": "13px",
    }


def _alpha(hex_color: str, alpha: float) -> str:
    """Convert '#RRGGBB' to 'rgba(r,g,b,alpha)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def main():
    args = parse_args()

    print(f"Collect interval: {args.interval}ms")
    print(f"Display interval: {args.display_interval}ms")
    print(f"Rolling window  : {args.window}s ({args.window // 60}m {args.window % 60}s)")
    print(f"Snapshot dir    : {args.snapshot_dir}")

    if HAS_NVML:
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"GPU {i}: {name} ({mem.total / (1024**3):.1f} GiB)")
    else:
        print("No NVIDIA GPUs detected -- CPU + Disk only")

    print()

    collector = MetricsCollector(window_sec=args.window, interval_ms=args.interval)
    psutil.cpu_percent(interval=None)

    app, ui_state = build_app(collector, args)

    def _collection_loop():
        interval_sec = args.interval / 1000.0
        while True:
            if not ui_state["paused"]:
                collector.sample()
            time.sleep(interval_sec)

    t = threading.Thread(target=_collection_loop, daemon=True)
    t.start()

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
