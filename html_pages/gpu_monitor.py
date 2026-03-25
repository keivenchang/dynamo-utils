#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Real-time GPU/CPU/Disk monitor with smooth 60fps scrolling.

Backend: Flask + flask-socketio pushes incremental data via WebSocket.
Frontend: Raw Plotly.js with extendTraces() + requestAnimationFrame scrolling.

Layout:
  1. GPU Memory by Process (GiB) -- stacked area
  2. GPU Util (%) + PCIe (MB/s)  -- lines, secondary y-axis
  3. CPU Usage (%)               -- line
  4. Disk I/O by Process (MB/s)  -- stacked area

Usage:
    python3 gpu_monitor.py [--port 8051] [--host 127.0.0.1]
"""

import argparse
import json
import os
import threading
import time
from collections import deque
from pathlib import Path

import psutil
import pynvml
from flask import Flask, Response
from flask_socketio import SocketIO

HAS_NVML = False
try:
    pynvml.nvmlInit()
    HAS_NVML = True
except pynvml.NVMLError as e:
    print(f"NVML init failed ({e}) -- GPU monitoring disabled")

PROCESS_COLORS = [
    "#00e5ff", "#ff1744", "#76ff03", "#ffea00",
    "#d500f9", "#ff9100", "#00e676", "#ff6d00",
    "#448aff", "#e040fb", "#18ffff", "#f50057",
    "#64ffda", "#ff80ab", "#b2ff59", "#ffd740",
    "#ea80fc", "#ff9e80", "#a7ffeb", "#ff8a80",
]
OTHER_COLOR = "#555555"
DISK_SCAN_EVERY = 10


def parse_args():
    p = argparse.ArgumentParser(description="Real-time GPU/CPU/Disk monitor")
    p.add_argument("--port", type=int, default=8051)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--interval", type=int, default=100, help="Collection interval ms")
    p.add_argument("--push-interval", type=int, default=100, help="WebSocket push interval ms")
    p.add_argument("--window", type=int, default=600, help="Rolling window seconds")
    return p.parse_args()


def _resolve_process_name(pid: int) -> str:
    try:
        proc = psutil.Process(pid)
        cmdline = proc.cmdline()
        full_cmd = " ".join(cmdline) if cmdline else ""
        proc_name = proc.name()

        # VLLM EngineCore: walk parent chain for model name + launch script
        if proc_name == "VLLM::EngineCore" or "EngineCore" in full_cmd:
            context = _vllm_context(proc)
            if context:
                return f"VLLM::EngineCore {context}:{pid}"
            return f"VLLM::EngineCore:{pid}"

        if cmdline and os.path.basename(cmdline[0]) == "node":
            label = _classify_node_process(full_cmd)
            if label:
                return f"{label}:{pid}"
        if cmdline and cmdline[0] == "docker" and "exec" in cmdline:
            if "node" in full_cmd:
                return f"Docker/Cursor:{pid}"

        # Python with dynamo module: extract module + model context
        if cmdline and os.path.basename(cmdline[0]).startswith("python"):
            if len(cmdline) > 1 and "-m" in cmdline:
                m_idx = cmdline.index("-m")
                if m_idx + 1 < len(cmdline):
                    module = cmdline[m_idx + 1]
                    model = _extract_arg(cmdline, "--model")
                    short_model = os.path.basename(model) if model else ""
                    if short_model:
                        return f"{module} {short_model}:{pid}"
                    return f"{module}:{pid}"
            if len(cmdline) > 1:
                script = os.path.basename(cmdline[1])
                if len(script) > 25:
                    script = script[:22] + "..."
                return f"{script}:{pid}"

        if cmdline:
            base = os.path.basename(cmdline[0])
            if len(base) > 25:
                base = base[:22] + "..."
            return f"{base}:{pid}"
        return f"{proc_name}:{pid}"
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return f"pid:{pid}"


def _vllm_context(proc) -> str:
    """Walk parent chain of a VLLM::EngineCore to find model name and launch script."""
    parts = []
    try:
        parent = proc.parent()
        if parent:
            pcmd = parent.cmdline()
            model = _extract_arg(pcmd, "--model")
            if model:
                parts.append(os.path.basename(model))
            grandparent = parent.parent()
            if grandparent:
                gcmd = grandparent.cmdline()
                if gcmd:
                    script = os.path.basename(gcmd[-1]) if gcmd[-1].endswith(".sh") else ""
                    if not script:
                        for arg in gcmd:
                            if arg.endswith(".sh"):
                                script = os.path.basename(arg)
                                break
                    if script:
                        parts.append(f"({script})")
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return " ".join(parts)


def _extract_arg(cmdline: list[str], flag: str) -> str:
    """Extract the value after a CLI flag like --model."""
    try:
        idx = cmdline.index(flag)
        if idx + 1 < len(cmdline):
            return cmdline[idx + 1]
    except ValueError:
        pass
    return ""


def _classify_node_process(full_cmd: str) -> str:
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
    def __init__(self, maxlen: int, prune: bool = True):
        self.maxlen = maxlen
        self._prune_enabled = prune
        self._len = 0
        self.series: dict[int, deque[float]] = {}
        self.names: dict[int, str] = {}
        self._colors: dict[int, str] = {}
        self._color_idx = 0
        self.first_seen: dict[int, float] = {}

    def record(self, data: dict[int, float], name_resolver, timestamp: float = 0):
        for pid in data:
            if pid not in self.series:
                backfill = min(self._len, self.maxlen)
                self.series[pid] = deque([0.0] * backfill, maxlen=self.maxlen)
                self.names[pid] = name_resolver(pid)
                self.first_seen[pid] = timestamp
        for pid, dq in self.series.items():
            dq.append(data.get(pid, 0.0))
        self._len += 1
        if self._prune_enabled:
            self._prune_dead()

    def _prune_dead(self):
        window = min(200, self._len)
        dead = [pid for pid, dq in self.series.items()
                if all(v == 0.0 for v in list(dq)[-window:])]
        for pid in dead:
            del self.series[pid]
            del self.names[pid]
            self._colors.pop(pid, None)

    def get_color(self, pid: int) -> str:
        if pid not in self._colors:
            self._colors[pid] = PROCESS_COLORS[self._color_idx % len(PROCESS_COLORS)]
            self._color_idx += 1
        return self._colors[pid]

    def get_all_sorted(self) -> list[tuple[int, str, str, list[float]]]:
        """Return (pid, name, color, values) sorted by peak desc."""
        if not self.series:
            return []
        sorted_pids = sorted(self.series.keys(),
                             key=lambda p: max(self.series[p]) if self.series[p] else 0,
                             reverse=True)
        return [(pid, self.names[pid], self.get_color(pid), list(self.series[pid]))
                for pid in sorted_pids if max(self.series[pid]) > 0]

    def get_top_sorted(self, n=20) -> list[tuple[int, str, str, list[float]]]:
        if not self.series:
            return []
        sorted_pids = sorted(self.series.keys(),
                             key=lambda p: max(self.series[p]) if self.series[p] else 0,
                             reverse=True)
        top = sorted_pids[:n]
        rest = sorted_pids[n:]
        result = [(pid, self.names[pid], self.get_color(pid), list(self.series[pid]))
                  for pid in top if max(self.series[pid]) > 0]
        if rest:
            length = len(next(iter(self.series.values())))
            rest_vals = [0.0] * length
            has_data = False
            for pid in rest:
                for i, v in enumerate(self.series[pid]):
                    rest_vals[i] += v
                    if v > 0:
                        has_data = True
            if has_data:
                result.append((-1, "Other", OTHER_COLOR, rest_vals))
        return result


class MetricsCollector:
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
                for getter in (pynvml.nvmlDeviceGetComputeRunningProcesses,
                               pynvml.nvmlDeviceGetGraphicsRunningProcesses):
                    try:
                        for p in getter(handle):
                            mem_gib = (p.usedGpuMemory or 0) / (1024**3)
                            gpu_proc_mem[p.pid] = gpu_proc_mem.get(p.pid, 0.0) + mem_gib
                    except pynvml.NVMLError:
                        pass
            self.proc_gpu_mem.record(gpu_proc_mem, _resolve_process_name, now)
        self._disk_scan_ctr += 1
        if self._disk_scan_ctr >= DISK_SCAN_EVERY:
            self._disk_scan_ctr = 0
            scan_mono = time.monotonic()
            scan_dt = scan_mono - self._last_disk_scan_mono
            if scan_dt <= 0:
                scan_dt = self.interval_sec * DISK_SCAN_EVERY
            self._last_disk_rates = self._scan_process_disk(scan_dt)
            self._last_disk_scan_mono = scan_mono
        self.proc_disk_io.record(self._last_disk_rates, _resolve_process_name, now)

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

    def snapshot_full(self) -> dict:
        """Return full state for initial browser load."""
        with self.lock:
            times = list(self.timestamps)
            gpu_mem = self.proc_gpu_mem.get_all_sorted()
            disk_io = self.proc_disk_io.get_top_sorted(20)
            gpu_util = [list(self.gpu_util[i]) for i in range(self.gpu_count)]
            pcie_tx = [list(self.gpu_pcie_tx[i]) for i in range(self.gpu_count)] if self.has_pcie else []
            pcie_rx = [list(self.gpu_pcie_rx[i]) for i in range(self.gpu_count)] if self.has_pcie else []
            cpu = list(self.cpu_pct)
            return {
                "type": "init",
                "timestamps": times,
                "gpu_count": self.gpu_count,
                "gpu_names": self.gpu_names,
                "gpu_mem_total_gib": self.gpu_mem_total_gib,
                "has_pcie": self.has_pcie,
                "gpu_mem": [{"pid": p, "name": n, "color": c, "vals": v,
                             "first_seen": self.proc_gpu_mem.first_seen.get(p, 0)} for p, n, c, v in gpu_mem],
                "gpu_util": gpu_util,
                "pcie_tx": pcie_tx,
                "pcie_rx": pcie_rx,
                "cpu": cpu,
                "disk_io": [{"pid": p, "name": n, "color": c, "vals": v} for p, n, c, v in disk_io],
            }

    def snapshot_delta(self, since_idx: int) -> dict:
        """Return only new data since since_idx."""
        with self.lock:
            cur_len = len(self.timestamps)
            if since_idx >= cur_len:
                return {"type": "delta", "timestamps": [], "gpu_mem_keys": [], "disk_keys": []}
            new_times = list(self.timestamps)[since_idx:]
            gpu_mem = self.proc_gpu_mem.get_all_sorted()
            disk_io = self.proc_disk_io.get_top_sorted(20)
            gpu_util = [list(self.gpu_util[i])[since_idx:] for i in range(self.gpu_count)]
            pcie_tx = [list(self.gpu_pcie_tx[i])[since_idx:] for i in range(self.gpu_count)] if self.has_pcie else []
            pcie_rx = [list(self.gpu_pcie_rx[i])[since_idx:] for i in range(self.gpu_count)] if self.has_pcie else []
            cpu = list(self.cpu_pct)[since_idx:]
            return {
                "type": "delta",
                "timestamps": new_times,
                "gpu_mem": [{"pid": p, "name": n, "color": c, "vals": v[since_idx:] if since_idx < len(v) else []} for p, n, c, v in gpu_mem],
                "gpu_mem_keys": [p for p, _, _, _ in gpu_mem],
                "gpu_util": gpu_util,
                "pcie_tx": pcie_tx,
                "pcie_rx": pcie_rx,
                "cpu": cpu,
                "disk_io": [{"pid": p, "name": n, "color": c, "vals": v[since_idx:] if since_idx < len(v) else []} for p, n, c, v in disk_io],
                "disk_keys": [p for p, _, _, _ in disk_io],
            }


# ---------------------------------------------------------------------------
# HTML / JS Frontend (embedded)
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GPU Monitor</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0d0d14; color: #e0e0e0; font-family: monospace; overflow: hidden; }
  #toolbar { display: flex; justify-content: space-between; align-items: center; padding: 12px 20px; }
  #toolbar h1 { font-size: 22px; }
  .btn { border: none; padding: 6px 14px; cursor: pointer; font-family: monospace;
         font-weight: bold; margin-left: 4px; border-radius: 4px; font-size: 13px; }
  .range-btn { background: #2a2a3c; color: #e0e0e0; border: 1px solid #444; }
  .range-btn.active { background: #444466; }
  #pause-btn { background: #ffcc00; color: #0d0d14; margin-left: 12px; }
  #snapshot-btn { background: #00e676; color: #0d0d14; }
  #chart { width: 100%; height: calc(100vh - 50px); }
  #status { position: fixed; bottom: 8px; left: 20px; font-size: 11px; color: #666; }
</style>
</head>
<body>
<div id="toolbar">
  <h1>System Monitor</h1>
  <div>
    <button class="btn range-btn" data-min="1">1m</button>
    <button class="btn range-btn active" data-min="2">2m</button>
    <button class="btn range-btn" data-min="5">5m</button>
    <button class="btn range-btn" data-min="10">10m</button>
    <button class="btn range-btn" data-min="0">All</button>
    <button class="btn" id="pause-btn">Pause</button>
    <button class="btn" id="snapshot-btn">Snapshot</button>
  </div>
</div>
<div id="chart"></div>
<div id="status">Connecting...</div>

<script>
(function() {
  "use strict";

  const MAX_POINTS = 60000;
  let viewMinutes = 2;
  let paused = false;
  let gd = null; // plotly graph div
  let config = null; // server config (gpu_count, has_pcie, etc.)
  let traceMap = {}; // maps logical trace id -> plotly trace index
  let traceCount = 0;
  let lastServerLen = 0; // tracks how many points the server has sent total
  let scrollRAF = null;

  const socket = io({transports: ['websocket']});

  socket.on('connect', function() {
    document.getElementById('status').textContent = 'Connected. Waiting for data...';
  });

  socket.on('init', function(msg) {
    config = msg;
    lastServerLen = msg.timestamps.length;
    buildChart(msg);
    document.getElementById('status').textContent = 'Live';
    startScrollLoop();
  });

  socket.on('delta', function(msg) {
    if (paused || !gd || !config) return;
    if (msg.timestamps.length === 0) return;

    lastServerLen += msg.timestamps.length;
    const times = msg.timestamps.map(t => new Date(t * 1000));

    // Check if trace structure changed (new processes)
    const gpuKeysChanged = JSON.stringify(msg.gpu_mem_keys) !== JSON.stringify(config._last_gpu_keys);
    const diskKeysChanged = JSON.stringify(msg.disk_keys) !== JSON.stringify(config._last_disk_keys);

    if (gpuKeysChanged || diskKeysChanged) {
      // Rebuild chart with full data from server
      socket.emit('request_init');
      return;
    }

    // Extend existing traces with new data
    const updateX = [];
    const updateY = [];
    const indices = [];

    // GPU mem traces
    for (let i = 0; i < msg.gpu_mem.length; i++) {
      const key = 'gpu_mem_' + msg.gpu_mem[i].pid;
      if (traceMap[key] !== undefined) {
        indices.push(traceMap[key]);
        updateX.push(times);
        updateY.push(msg.gpu_mem[i].vals);
      }
    }
    // GPU util
    for (let i = 0; i < msg.gpu_util.length; i++) {
      const key = 'gpu_util_' + i;
      indices.push(traceMap[key]);
      updateX.push(times);
      updateY.push(msg.gpu_util[i]);
    }
    // PCIe
    if (config.has_pcie) {
      for (let i = 0; i < msg.pcie_tx.length; i++) {
        indices.push(traceMap['pcie_tx_' + i]);
        updateX.push(times);
        updateY.push(msg.pcie_tx[i]);
        indices.push(traceMap['pcie_rx_' + i]);
        updateX.push(times);
        updateY.push(msg.pcie_rx[i]);
      }
    }
    // CPU
    indices.push(traceMap['cpu']);
    updateX.push(times);
    updateY.push(msg.cpu);
    // Disk
    for (let i = 0; i < msg.disk_io.length; i++) {
      const key = 'disk_' + msg.disk_io[i].pid;
      if (traceMap[key] !== undefined) {
        indices.push(traceMap[key]);
        updateX.push(times);
        updateY.push(msg.disk_io[i].vals);
      }
    }

    if (indices.length > 0) {
      Plotly.extendTraces(gd, {x: updateX, y: updateY}, indices, MAX_POINTS);
    }

  });

  function buildChart(msg) {
    gd = document.getElementById('chart');
    traceMap = {};
    traceCount = 0;
    const traces = [];
    const times = msg.timestamps.map(t => new Date(t * 1000));

    const hasGpu = msg.gpu_count > 0;
    const hasPcie = msg.has_pcie;
    const nRows = (hasGpu ? 2 : 0) + 2; // gpu_mem + gpu_util + cpu + disk

    let row = 0;

    // Row 1: GPU Memory (stacked area)
    if (hasGpu) {
      row++;
      const gpuMem = msg.gpu_mem;
      // Reversed so largest at bottom of stack
      for (let i = gpuMem.length - 1; i >= 0; i--) {
        const m = gpuMem[i];
        traceMap['gpu_mem_' + m.pid] = traceCount++;
        traces.push({
          x: times, y: m.vals, name: m.name, type: 'scatter', mode: 'lines',
          line: {width: 0.5, color: m.color}, stackgroup: 'gpu_mem',
          fillcolor: hexToRgba(m.color, 0.6),
          xaxis: 'x', yaxis: 'y',
          legendgroup: 'gpu_mem', legendgrouptitle: {text: 'GPU Mem'},
        });
      }
      if (gpuMem.length === 0) {
        traceMap['gpu_mem_empty'] = traceCount++;
        traces.push({
          x: times, y: times.map(() => 0), name: 'GPU Mem: 0', type: 'scatter',
          mode: 'lines', line: {width: 1, color: '#555'},
          xaxis: 'x', yaxis: 'y', legendgroup: 'gpu_mem',
        });
      }

      // Row 2: GPU Util + PCIe
      row++;
      const gpuColors = ['#76b900', '#ff1744', '#76ff03', '#d500f9'];
      for (let i = 0; i < msg.gpu_count; i++) {
        const color = gpuColors[i % gpuColors.length];
        const label = msg.gpu_count > 1 ? 'GPU ' + i + ' Util' : 'GPU Util';
        traceMap['gpu_util_' + i] = traceCount++;
        traces.push({
          x: times, y: msg.gpu_util[i], name: label, type: 'scatter',
          line: {color: color, width: 2}, fill: 'tozeroy',
          fillcolor: hexToRgba(color, 0.12),
          xaxis: 'x2', yaxis: 'y2',
          legendgroup: 'util', legendgrouptitle: {text: 'GPU Util'},
        });
      }
      if (hasPcie) {
        for (let i = 0; i < msg.gpu_count; i++) {
          const suffix = msg.gpu_count > 1 ? ' (GPU ' + i + ')' : '';
          traceMap['pcie_tx_' + i] = traceCount++;
          traces.push({
            x: times, y: msg.pcie_tx[i], name: 'PCIe TX' + suffix, type: 'scatter',
            line: {color: '#40c4ff', width: 0.5},
            xaxis: 'x2', yaxis: 'y3',
            legendgroup: 'pcie', legendgrouptitle: {text: 'PCIe'},
          });
          traceMap['pcie_rx_' + i] = traceCount++;
          traces.push({
            x: times, y: msg.pcie_rx[i], name: 'PCIe RX' + suffix, type: 'scatter',
            line: {color: '#ff6e40', width: 0.5},
            xaxis: 'x2', yaxis: 'y3',
            legendgroup: 'pcie',
          });
        }
      }
    }

    // Row 3: CPU
    row++;
    traceMap['cpu'] = traceCount++;
    traces.push({
      x: times, y: msg.cpu, name: 'CPU', type: 'scatter',
      line: {color: '#ffcc00', width: 1}, fill: 'tozeroy',
      fillcolor: 'rgba(255,204,0,0.10)',
      xaxis: hasGpu ? 'x3' : 'x', yaxis: hasGpu ? 'y4' : 'y',
      legendgroup: 'cpu', legendgrouptitle: {text: 'CPU'},
    });

    // Row 4: Disk I/O
    row++;
    const diskData = msg.disk_io;
    for (let i = 0; i < diskData.length; i++) {
      const d = diskData[i];
      traceMap['disk_' + d.pid] = traceCount++;
      traces.push({
        x: times, y: d.vals, name: d.name, type: 'scatter', mode: 'lines',
        line: {width: 0.5, color: d.color}, stackgroup: 'disk_io',
        fillcolor: hexToRgba(d.color, 0.55),
        xaxis: hasGpu ? 'x4' : 'x2', yaxis: hasGpu ? 'y5' : 'y2',
        legendgroup: 'disk_io', legendgrouptitle: {text: 'Disk I/O'},
      });
    }

    // Store keys for change detection
    config._last_gpu_keys = msg.gpu_mem.map(m => m.pid);
    config._last_disk_keys = msg.disk_io.map(d => d.pid);

    const maxGib = msg.gpu_mem_total_gib.length > 0 ? Math.max(...msg.gpu_mem_total_gib) : 1;
    const bgColors = ['#101020', '#181830'];
    const rowLabels = hasGpu
      ? ['GPU Memory by Process (GiB)', hasPcie ? 'GPU Util (%) + PCIe (MB/s)' : 'GPU Util (%)', 'CPU Usage (%)', 'Disk I/O by Process (MB/s)']
      : ['CPU Usage (%)', 'Disk I/O by Process (MB/s)'];

    const domain = function(i, total) {
      const gap = 0.03;
      const h = (1 - gap * (total - 1)) / total;
      const y0 = (total - 1 - i) * (h + gap);
      return [y0, y0 + h];
    };

    const layout = {
      paper_bgcolor: '#0d0d14', plot_bgcolor: '#141420',
      font: {family: 'monospace', color: '#e0e0e0', size: 11},
      margin: {l: 70, r: 20, t: 10, b: 35},
      showlegend: true,
      legend: {orientation: 'v', yanchor: 'top', y: 1, xanchor: 'left', x: 1.01, font: {size: 10}, bgcolor: 'rgba(13,13,20,0.8)'},
      hovermode: false,
      shapes: [], annotations: [],
    };

    const totalRows = hasGpu ? 4 : 2;
    // Build axis layout dynamically
    if (hasGpu) {
      layout.xaxis = {domain: [0, 0.92], anchor: 'y', showticklabels: false, gridcolor: '#262640'};
      layout.yaxis = {domain: domain(0, totalRows), title: {text: 'GiB', font: {size: 10}}, range: [0, maxGib * 1.02], gridcolor: '#262640'};
      layout.xaxis2 = {domain: [0, 0.92], anchor: 'y2', showticklabels: false, gridcolor: '#262640'};
      layout.yaxis2 = {domain: domain(1, totalRows), title: {text: '%', font: {size: 10}}, range: [0, 105], gridcolor: '#262640'};
      if (hasPcie) {
        layout.yaxis3 = {domain: domain(1, totalRows), title: {text: 'PCIe MB/s', font: {color: '#40c4ff', size: 10}}, overlaying: 'y2', side: 'right', showgrid: false, rangemode: 'tozero'};
      }
      layout.xaxis3 = {domain: [0, 0.92], anchor: 'y4', showticklabels: false, gridcolor: '#262640'};
      layout.yaxis4 = {domain: domain(2, totalRows), title: {text: '%', font: {size: 10}}, range: [0, 105], gridcolor: '#262640'};
      layout.xaxis4 = {domain: [0, 0.92], anchor: 'y5', gridcolor: '#262640'};
      layout.yaxis5 = {domain: domain(3, totalRows), title: {text: 'MB/s', font: {size: 10}}, gridcolor: '#262640'};
    } else {
      layout.xaxis = {domain: [0, 0.92], anchor: 'y', showticklabels: false, gridcolor: '#262640'};
      layout.yaxis = {domain: domain(0, totalRows), title: {text: '%', font: {size: 10}}, range: [0, 105], gridcolor: '#262640'};
      layout.xaxis2 = {domain: [0, 0.92], anchor: 'y2', gridcolor: '#262640'};
      layout.yaxis2 = {domain: domain(1, totalRows), title: {text: 'MB/s', font: {size: 10}}, gridcolor: '#262640'};
    }

    // Add background shapes and labels
    const axPairs = hasGpu
      ? [['x','y'], ['x2','y2'], ['x3','y4'], ['x4','y5']]
      : [['x','y'], ['x2','y2']];
    for (let i = 0; i < axPairs.length; i++) {
      layout.shapes.push({
        type: 'rect', x0: 0, x1: 1, y0: 0, y1: 1,
        xref: axPairs[i][0] + ' domain', yref: axPairs[i][1] + ' domain',
        fillcolor: bgColors[i % 2], line: {width: 0}, layer: 'below',
      });
      layout.annotations.push({
        text: '  ' + rowLabels[i] + '  ', x: 0.0, y: 0.97,
        xref: axPairs[i][0] + ' domain', yref: axPairs[i][1] + ' domain',
        showarrow: false, font: {size: 11, color: '#c0c0d0', family: 'monospace'},
        xanchor: 'left', yanchor: 'top',
        bgcolor: 'rgba(30,30,50,0.92)', bordercolor: '#444466', borderwidth: 1, borderpad: 3,
      });
    }

    // GPU total memory dashed line
    if (hasGpu && maxGib > 0) {
      layout.shapes.push({
        type: 'line', x0: 0, x1: 1, y0: maxGib, y1: maxGib,
        xref: 'x domain', yref: 'y',
        line: {color: '#ff5252', width: 1, dash: 'dash'},
      });
      layout.annotations.push({
        text: maxGib.toFixed(1) + ' GiB total', x: 1, y: maxGib,
        xref: 'x domain', yref: 'y', showarrow: false,
        font: {size: 11, color: '#ff5252'}, xanchor: 'right',
      });

    }

    Plotly.newPlot(gd, traces, layout, {
      displayModeBar: true, scrollZoom: true,
      toImageButtonOptions: {format: 'png', filename: 'gpu_monitor', height: 1080, width: 1920, scale: 2},
    });
  }

  function startScrollLoop() {
    if (scrollRAF) cancelAnimationFrame(scrollRAF);
    function tick() {
      if (!paused && gd && viewMinutes > 0) {
        const now = new Date();
        const start = new Date(now.getTime() - viewMinutes * 60000);
        const upd = {};
        const axes = config.gpu_count > 0
          ? ['xaxis', 'xaxis2', 'xaxis3', 'xaxis4']
          : ['xaxis', 'xaxis2'];
        for (const ax of axes) {
          upd[ax + '.range'] = [start, now];
        }
        Plotly.relayout(gd, upd);
      }
      scrollRAF = requestAnimationFrame(tick);
    }
    scrollRAF = requestAnimationFrame(tick);
  }

  // --- Controls ---
  document.querySelectorAll('.range-btn').forEach(btn => {
    btn.addEventListener('click', function() {
      document.querySelectorAll('.range-btn').forEach(b => b.classList.remove('active'));
      this.classList.add('active');
      viewMinutes = parseInt(this.dataset.min);
      if (viewMinutes === 0 && gd) {
        // Reset to autorange
        const axes = config.gpu_count > 0
          ? ['xaxis', 'xaxis2', 'xaxis3', 'xaxis4']
          : ['xaxis', 'xaxis2'];
        const upd = {};
        for (const ax of axes) { upd[ax + '.autorange'] = true; }
        Plotly.relayout(gd, upd);
      }
    });
  });

  document.getElementById('pause-btn').addEventListener('click', function() {
    paused = !paused;
    this.textContent = paused ? 'Resume' : 'Pause';
    this.style.backgroundColor = paused ? '#ff5252' : '#ffcc00';
    socket.emit('pause', paused);
  });

  document.getElementById('snapshot-btn').addEventListener('click', function() {
    if (!gd) return;
    const d = new Date();
    const pad = n => String(n).padStart(2, '0');
    const fname = d.getFullYear() + '-' + pad(d.getMonth()+1) + '-' + pad(d.getDate()) +
      '_' + pad(d.getHours()) + pad(d.getMinutes()) + pad(d.getSeconds()) + '_GPU';
    Plotly.downloadImage(gd, {format: 'png', width: 1920, height: 1080, scale: 2, filename: fname});
  });

  function hexToRgba(hex, alpha) {
    const h = hex.replace('#', '');
    return 'rgba(' + parseInt(h.substring(0,2),16) + ',' +
      parseInt(h.substring(2,4),16) + ',' + parseInt(h.substring(4,6),16) + ',' + alpha + ')';
  }
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Flask + SocketIO Server
# ---------------------------------------------------------------------------

def build_server(collector: MetricsCollector, args):
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "gpu_monitor"
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

    ui_state = {"paused": False}

    @app.route("/")
    def index():
        return Response(HTML_PAGE, mimetype="text/html")

    @socketio.on("connect")
    def handle_connect():
        data = collector.snapshot_full()
        socketio.emit("init", data)

    @socketio.on("request_init")
    def handle_request_init():
        data = collector.snapshot_full()
        socketio.emit("init", data)

    @socketio.on("pause")
    def handle_pause(is_paused):
        ui_state["paused"] = bool(is_paused)

    def push_loop():
        """Background thread that pushes deltas to all clients."""
        push_sec = args.push_interval / 1000.0
        last_len = 0
        while True:
            time.sleep(push_sec)
            if ui_state["paused"]:
                continue
            with collector.lock:
                cur_len = len(collector.timestamps)
            if cur_len <= last_len:
                continue
            delta = collector.snapshot_delta(last_len)
            last_len = cur_len
            socketio.emit("delta", delta)

    return app, socketio, ui_state, push_loop


def main():
    args = parse_args()
    print(f"Collect interval: {args.interval}ms")
    print(f"Push interval   : {args.push_interval}ms")
    print(f"Rolling window  : {args.window}s ({args.window // 60}m {args.window % 60}s)")

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

    app, socketio, ui_state, push_loop = build_server(collector, args)

    # Collection thread
    def _collection_loop():
        interval_sec = args.interval / 1000.0
        while True:
            if not ui_state["paused"]:
                collector.sample()
            time.sleep(interval_sec)

    threading.Thread(target=_collection_loop, daemon=True).start()
    threading.Thread(target=push_loop, daemon=True).start()

    print(f"Open http://{args.host}:{args.port} in your browser")
    socketio.run(app, host=args.host, port=args.port, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
