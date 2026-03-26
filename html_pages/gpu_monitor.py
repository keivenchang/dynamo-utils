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
import socket
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
    p.add_argument("--interval", type=int, default=75, help="Fast collection interval ms (CPU, disk)")
    p.add_argument("--gpu-interval", type=int, default=500, help="Slow collection interval ms (GPU mem, util, PCIe)")
    p.add_argument("--push-interval", type=int, default=150, help="WebSocket push interval ms")
    p.add_argument("--window", type=int, default=900, help="Rolling window seconds (default 900 = 15min)")
    return p.parse_args()


def _resolve_process_name(pid: int) -> str:
    try:
        proc = psutil.Process(pid)
        cmdline = proc.cmdline()
        full_cmd = " ".join(cmdline) if cmdline else ""
        proc_name = proc.name()

        # VLLM EngineCore: walk parent chain for model name + launch script
        if proc_name == "VLLM::EngineCore" or "EngineCore" in full_cmd:
            model, script = _vllm_context(proc)
            parts = ["VLLM::EngineCore"]
            if model:
                parts.append(model)
            if script:
                parts.append(script)
            parts.append(f"PID={pid}")
            return ", ".join(parts)

        # SGLang, TRT-LLM, or other GPU processes with custom names
        if "sglang" in proc_name.lower() or "sglang" in full_cmd.lower() or \
           "trtllm" in proc_name.lower() or "trtllm" in full_cmd.lower():
            model, script = _gpu_process_context(proc)
            parts = [proc_name]
            if model:
                parts.append(model)
            if script:
                parts.append(script)
            parts.append(f"PID={pid}")
            return ", ".join(parts)

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
                        return f"{module}, {short_model}, PID={pid}"
                    return f"{module}, PID={pid}"
            if len(cmdline) > 1:
                script = os.path.basename(cmdline[1])
                if len(script) > 25:
                    script = script[:22] + "..."
                return f"{script}, PID={pid}"

        if cmdline:
            base = os.path.basename(cmdline[0])
            if len(base) > 25:
                base = base[:22] + "..."
            return f"{base}:{pid}"
        return f"{proc_name}:{pid}"
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return f"pid:{pid}"


def _gpu_process_context(proc) -> tuple[str, str]:
    """Walk parent chain of any GPU process to find model name and launch script."""
    model = ""
    script = ""
    try:
        for ancestor in proc.parents():
            acmd = ancestor.cmdline()
            if not acmd:
                continue
            if not model:
                m = _extract_arg(acmd, "--model")
                if m:
                    model = os.path.basename(m)
            if not script:
                for arg in acmd:
                    if arg.endswith(".sh"):
                        script = os.path.basename(arg)
                        break
            if model and script:
                break
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return model, script


def _vllm_context(proc) -> tuple[str, str]:
    """Walk parent chain of a VLLM::EngineCore to find model name and launch script."""
    model = ""
    script = ""
    try:
        parent = proc.parent()
        if parent:
            pcmd = parent.cmdline()
            m = _extract_arg(pcmd, "--model")
            if m:
                model = os.path.basename(m)
            grandparent = parent.parent()
            if grandparent:
                gcmd = grandparent.cmdline()
                if gcmd:
                    for arg in gcmd:
                        if arg.endswith(".sh"):
                            script = os.path.basename(arg)
                            break
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return model, script


def _extract_arg(cmdline: list[str], flag: str) -> str:
    """Extract the value after a CLI flag like --model. Also checks --model-path."""
    for f in (flag, flag + "-path"):
        try:
            idx = cmdline.index(f)
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
        self.sample_counter: int = 0
        self.cpu_pct: deque[float] = deque(maxlen=maxlen)
        self._prev_disk = psutil.disk_io_counters()
        self._prev_net = psutil.net_io_counters()
        self._prev_mono = time.monotonic()
        self.net_sent_mbps: deque[float] = deque(maxlen=maxlen)
        self.net_recv_mbps: deque[float] = deque(maxlen=maxlen)
        self.proc_gpu_mem: list[ProcessTracker] = []
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
                self.proc_gpu_mem.append(ProcessTracker(maxlen, prune=False))
                self.gpu_util.append(deque(maxlen=maxlen))
                self.gpu_pcie_tx.append(deque(maxlen=maxlen))
                self.gpu_pcie_rx.append(deque(maxlen=maxlen))
        self.gpu_temp: list[deque[float]] = [deque(maxlen=maxlen) for _ in range(self.gpu_count)]
        if self.gpu_count > 0:
                try:
                    h = pynvml.nvmlDeviceGetHandleByIndex(0)
                    pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_TX_BYTES)
                    self.has_pcie = True
                except pynvml.NVMLError:
                    self.has_pcie = False

    def sample_fast(self):
        """Fast-tier: CPU + disk I/O (called at --interval rate)."""
        with self.lock:
            now = time.time()
            now_mono = time.monotonic()
            dt = now_mono - self._prev_mono
            if dt <= 0:
                dt = self.interval_sec
            self.timestamps.append(now)
            self.sample_counter += 1
            self.cpu_pct.append(psutil.cpu_percent(interval=None))
            # Network I/O: repeat last (sampled in slow loop)
            self.net_sent_mbps.append(self.net_sent_mbps[-1] if self.net_sent_mbps else 0.0)
            self.net_recv_mbps.append(self.net_recv_mbps[-1] if self.net_recv_mbps else 0.0)
            self._prev_mono = now_mono
            # GPU: PCIe sampled live, util repeats last, memory queried from NVML
            if HAS_NVML:
                for i in range(self.gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    self.gpu_util[i].append(self.gpu_util[i][-1] if self.gpu_util[i] else 0.0)
                    self.gpu_temp[i].append(self.gpu_temp[i][-1] if self.gpu_temp[i] else 0.0)
                    if self.has_pcie:
                        try:
                            tx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
                            rx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
                            self.gpu_pcie_tx[i].append(tx / 1024.0)
                            self.gpu_pcie_rx[i].append(rx / 1024.0)
                        except pynvml.NVMLError:
                            self.gpu_pcie_tx[i].append(self.gpu_pcie_tx[i][-1] if self.gpu_pcie_tx[i] else 0.0)
                            self.gpu_pcie_rx[i].append(self.gpu_pcie_rx[i][-1] if self.gpu_pcie_rx[i] else 0.0)
                    # GPU memory per-process: query NVML every fast tick
                    gpu_proc_mem: dict[int, float] = {}
                    for getter in (pynvml.nvmlDeviceGetComputeRunningProcesses,
                                   pynvml.nvmlDeviceGetGraphicsRunningProcesses):
                        try:
                            for p in getter(handle):
                                mem_gib = (p.usedGpuMemory or 0) / (1024**3)
                                gpu_proc_mem[p.pid] = gpu_proc_mem.get(p.pid, 0.0) + mem_gib
                        except pynvml.NVMLError:
                            pass
                    self.proc_gpu_mem[i].record(gpu_proc_mem, _resolve_process_name, now)
            self.proc_disk_io.record(self._last_disk_rates, _resolve_process_name, now)

    def sample_slow(self):
        """Slow-tier: GPU memory, GPU utilization, disk I/O (called at --gpu-interval rate)."""
        with self.lock:
            now = time.time()
            # Disk I/O per-process scan
            scan_mono = time.monotonic()
            scan_dt = scan_mono - self._last_disk_scan_mono
            if scan_dt <= 0:
                scan_dt = 0.5
            raw_rates = self._scan_process_disk(scan_dt)
            # EMA smooth disk I/O per process (alpha=0.2)
            alpha = 0.2
            for pid, rate in raw_rates.items():
                prev = self._last_disk_rates.get(pid, rate)
                raw_rates[pid] = alpha * rate + (1 - alpha) * prev
            self._last_disk_rates = raw_rates
            self._last_disk_scan_mono = scan_mono

            # Network I/O (smoothed: overwrite last value with averaged rate)
            net = psutil.net_io_counters()
            net_dt = scan_dt if scan_dt > 0 else 0.5
            sent_rate = (net.bytes_sent - self._prev_net.bytes_sent) / (1024 * 1024) / net_dt
            recv_rate = (net.bytes_recv - self._prev_net.bytes_recv) / (1024 * 1024) / net_dt
            self._prev_net = net
            # Heavy EMA smoothing (alpha=0.15) + client-side spline
            alpha = 0.15
            if self.net_sent_mbps:
                self.net_sent_mbps[-1] = alpha * sent_rate + (1 - alpha) * self.net_sent_mbps[-1]
            if self.net_recv_mbps:
                self.net_recv_mbps[-1] = alpha * recv_rate + (1 - alpha) * self.net_recv_mbps[-1]

            # GPU utilization (overwrite last value with fresh reading)
            if HAS_NVML:
                for i in range(self.gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        self.gpu_util[i][-1] = util.gpu
                    except pynvml.NVMLError:
                        pass
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        self.gpu_temp[i][-1] = float(temp)
                    except pynvml.NVMLError:
                        pass

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

    @staticmethod
    def _downsample_init(times, arrays, max_recent=1200, max_total=2400):
        """Downsample for init: keep last max_recent at full res, older data subsampled."""
        n = len(times)
        if n <= max_total:
            return times, arrays
        older_count = n - max_recent
        older_target = max_total - max_recent
        step = max(1, older_count // older_target)
        # Build indices: subsampled older + full recent
        older_indices = list(range(0, older_count, step))
        recent_indices = list(range(older_count, n))
        indices = older_indices + recent_indices
        ds_times = [times[i] for i in indices]
        ds_arrays = [[arr[i] for i in indices] if len(arr) == n else arr for arr in arrays]
        return ds_times, ds_arrays

    def snapshot_full(self) -> dict:
        """Return full state for initial browser load (downsampled for browser perf)."""
        with self.lock:
            times_full = list(self.timestamps)
            gpu_mem_per_gpu_full = []
            all_vals = []
            for gi in range(self.gpu_count):
                tracker = self.proc_gpu_mem[gi]
                procs = tracker.get_all_sorted()
                gpu_mem_per_gpu_full.append(procs)
                for _, _, _, v in procs:
                    all_vals.append(v)
            disk_io_full = self.proc_disk_io.get_top_sorted(20)
            for _, _, _, v in disk_io_full:
                all_vals.append(v)
            gpu_util = [list(self.gpu_util[i]) for i in range(self.gpu_count)]
            all_vals.extend(gpu_util)
            gpu_temp = [list(self.gpu_temp[i]) for i in range(self.gpu_count)]
            all_vals.extend(gpu_temp)
            pcie_tx = [list(self.gpu_pcie_tx[i]) for i in range(self.gpu_count)] if self.has_pcie else []
            pcie_rx = [list(self.gpu_pcie_rx[i]) for i in range(self.gpu_count)] if self.has_pcie else []
            all_vals.extend(pcie_tx)
            all_vals.extend(pcie_rx)
            cpu = list(self.cpu_pct)
            all_vals.append(cpu)
            net_sent = list(self.net_sent_mbps)
            net_recv = list(self.net_recv_mbps)
            all_vals.append(net_sent)
            all_vals.append(net_recv)

            times, ds_arrays = self._downsample_init(times_full, all_vals)

            # Rebuild structures from downsampled arrays
            idx = 0
            gpu_mem_per_gpu = []
            for gi in range(self.gpu_count):
                procs = gpu_mem_per_gpu_full[gi]
                ds_procs = []
                for p, n, c, v in procs:
                    ds_procs.append({"pid": p, "name": n, "color": c, "vals": ds_arrays[idx],
                                     "first_seen": self.proc_gpu_mem[gi].first_seen.get(p, 0)})
                    idx += 1
                gpu_mem_per_gpu.append(ds_procs)
            disk_io = []
            for p, n, c, v in disk_io_full:
                disk_io.append({"pid": p, "name": n, "color": c, "vals": ds_arrays[idx]})
                idx += 1
            ds_gpu_util = [ds_arrays[idx + i] for i in range(self.gpu_count)]
            idx += self.gpu_count
            ds_gpu_temp = [ds_arrays[idx + i] for i in range(self.gpu_count)]
            idx += self.gpu_count
            ds_pcie_tx = [ds_arrays[idx + i] for i in range(len(pcie_tx))]
            idx += len(pcie_tx)
            ds_pcie_rx = [ds_arrays[idx + i] for i in range(len(pcie_rx))]
            idx += len(pcie_rx)
            ds_cpu = ds_arrays[idx]
            idx += 1
            ds_net_sent = ds_arrays[idx]
            idx += 1
            ds_net_recv = ds_arrays[idx]

            return {
                "type": "init",
                "sample_counter": self.sample_counter,
                "timestamps": times,
                "gpu_count": self.gpu_count,
                "gpu_names": self.gpu_names,
                "gpu_mem_total_gib": self.gpu_mem_total_gib,
                "has_pcie": self.has_pcie,
                "gpu_mem": gpu_mem_per_gpu,
                "gpu_util": ds_gpu_util,
                "gpu_temp": ds_gpu_temp,
                "pcie_tx": ds_pcie_tx,
                "pcie_rx": ds_pcie_rx,
                "cpu": ds_cpu,
                "net_sent": ds_net_sent,
                "net_recv": ds_net_recv,
                "disk_io": disk_io,
            }

    def snapshot_delta(self, since_counter: int) -> dict:
        """Return only new data since the given sample_counter value."""
        with self.lock:
            cur_counter = self.sample_counter
            if since_counter >= cur_counter:
                return {"type": "delta", "timestamps": [], "gpu_mem_keys": [], "disk_keys": []}
            new_count = cur_counter - since_counter
            cur_len = len(self.timestamps)
            since_idx = max(0, cur_len - new_count)
            new_times = list(self.timestamps)[since_idx:]
            gpu_mem_per_gpu = []
            gpu_mem_keys_per_gpu = []
            for gi in range(self.gpu_count):
                gm = self.proc_gpu_mem[gi].get_all_sorted()
                gpu_mem_per_gpu.append([
                    {"pid": p, "name": n, "color": c, "vals": v[since_idx:] if since_idx < len(v) else []}
                    for p, n, c, v in gm
                ])
                gpu_mem_keys_per_gpu.append([p for p, _, _, _ in gm])
            disk_io = self.proc_disk_io.get_top_sorted(20)
            gpu_util = [list(self.gpu_util[i])[since_idx:] for i in range(self.gpu_count)]
            gpu_temp = [list(self.gpu_temp[i])[since_idx:] for i in range(self.gpu_count)]
            pcie_tx = [list(self.gpu_pcie_tx[i])[since_idx:] for i in range(self.gpu_count)] if self.has_pcie else []
            pcie_rx = [list(self.gpu_pcie_rx[i])[since_idx:] for i in range(self.gpu_count)] if self.has_pcie else []
            cpu = list(self.cpu_pct)[since_idx:]
            net_sent = list(self.net_sent_mbps)[since_idx:]
            net_recv = list(self.net_recv_mbps)[since_idx:]
            return {
                "type": "delta",
                "timestamps": new_times,
                "gpu_mem": gpu_mem_per_gpu,
                "gpu_mem_keys": gpu_mem_keys_per_gpu,
                "gpu_util": gpu_util,
                "gpu_temp": gpu_temp,
                "pcie_tx": pcie_tx,
                "pcie_rx": pcie_rx,
                "cpu": cpu,
                "net_sent": net_sent,
                "net_recv": net_recv,
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
    <button class="btn range-btn" data-min="15">15m</button>
    <button class="btn" id="pause-btn">Pause</button>
    <button class="btn" id="snapshot-btn">Snapshot</button>
  </div>
</div>
<div id="chart"></div>
<div id="status">Connecting...</div>

<script>
(function() {
  "use strict";

  const MAX_POINTS = 2400;
  let viewMinutes = 2;
  let paused = false;
  let gd = null; // plotly graph div
  let config = null; // server config (gpu_count, has_pcie, etc.)
  let traceMap = {}; // maps logical trace id -> plotly trace index
  let traceCount = 0;
  let lastServerLen = 0; // tracks how many points the server has sent total
  let scrollRAF = null;

  const socket = io({transports: ['polling'], reconnection: true, reconnectionDelay: 500, reconnectionAttempts: Infinity, timeout: 60000});

  socket.on('connect', function() {
    document.getElementById('status').textContent = 'Connected. Waiting for data...';
    if (config) {
      socket.emit('request_init');
    } else {
      setTimeout(function() {
        if (!config) socket.emit('request_init');
      }, 3000);
    }
  });

  socket.on('disconnect', function() {
    document.getElementById('status').textContent = 'Disconnected -- reconnecting...';
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

    // GPU mem traces (per-GPU)
    for (let gi = 0; gi < msg.gpu_mem.length; gi++) {
      const gpuMem = msg.gpu_mem[gi];
      for (let i = 0; i < gpuMem.length; i++) {
        const key = 'gpu_mem_' + gi + '_' + gpuMem[i].pid;
        if (traceMap[key] !== undefined) {
          indices.push(traceMap[key]);
          updateX.push(times);
          updateY.push(gpuMem[i].vals);
        }
      }
    }
    // GPU util
    for (let i = 0; i < msg.gpu_util.length; i++) {
      const key = 'gpu_util_' + i;
      indices.push(traceMap[key]);
      updateX.push(times);
      updateY.push(msg.gpu_util[i]);
    }
    // GPU temp
    if (msg.gpu_temp) {
      for (let i = 0; i < msg.gpu_temp.length; i++) {
        const key = 'gpu_temp_' + i;
        if (traceMap[key] !== undefined) {
          indices.push(traceMap[key]);
          updateX.push(times);
          updateY.push(msg.gpu_temp[i]);
        }
      }
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
    // Network
    if (traceMap['net_sent'] !== undefined) {
      indices.push(traceMap['net_sent']);
      updateX.push(times);
      updateY.push(msg.net_sent);
      indices.push(traceMap['net_recv']);
      updateX.push(times);
      updateY.push(msg.net_recv);
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
    const gpuCount = msg.gpu_count;
    // Rows: 2*N (mem+util per GPU) + 1 cpu + 1 disk+net
    const nRows = (hasGpu ? gpuCount * 2 : 0) + 2;
    // Row weights: GPU mem=5, GPU util=5, CPU=3, Disk+Net=3
    const rowWeights = [];
    if (hasGpu) {
      for (let gi = 0; gi < gpuCount; gi++) {
        rowWeights.push(7); // mem
        rowWeights.push(3); // util+temp
      }
    }
    rowWeights.push(3); // CPU
    rowWeights.push(3); // Disk+Net

    const xax = r => r === 1 ? 'x' : 'x' + r;
    const yax = r => r === 1 ? 'y' : 'y' + r;
    const pcieYIdxPerGpu = []; // secondary y-axis index per GPU

    let row = 0;

    const gpuUtilColors = ['#76b900', '#ff1744', '#448aff', '#ffea00', '#d500f9', '#ff9100', '#00e676', '#ff6d00'];
    const gpuPcieColors = ['#40c4ff', '#ff6e40', '#b388ff', '#ffd54f', '#ea80fc', '#ffab40', '#64ffda', '#ff8a65'];

    // Per-GPU blocks: Memory+PCIe row + Util+Temp row
    const pcieYIdxPerGpu_mem = [];
    const tempYIdxPerGpu = [];
    if (hasGpu) {
      for (let gi = 0; gi < gpuCount; gi++) {
        // Memory + PCIe row
        row++;
        const memRow = row;
        const gpuMem = msg.gpu_mem[gi] || [];
        const stackName = 'gpu_mem_' + gi;
        const memPatterns = ['/', '\\', 'x', '+', '-', '|', '.', ''];
        for (let i = gpuMem.length - 1; i >= 0; i--) {
          const m = gpuMem[i];
          const patIdx = Math.abs(m.pid) % memPatterns.length;
          traceMap['gpu_mem_' + gi + '_' + m.pid] = traceCount++;
          traces.push({
            x: times, y: m.vals, name: m.name, type: 'scatter', mode: 'none',
            line: {width: 0, color: m.color}, stackgroup: stackName,
            fillcolor: hexToRgba(m.color, 0.6),
            fillpattern: {shape: memPatterns[patIdx], solidity: 0.15, size: 8, bgcolor: hexToRgba(m.color, 0.6), fgcolor: 'rgba(13,13,20,0.4)'},
            xaxis: xax(memRow), yaxis: yax(memRow),
            legendgroup: 'gpu_mem_' + gi, legendgrouptitle: {text: 'GPU ' + gi + ' Mem'},
          });
        }
        if (gpuMem.length === 0) {
          traceMap['gpu_mem_' + gi + '_empty'] = traceCount++;
          traces.push({
            x: times, y: times.map(() => 0), name: 'GPU ' + gi + ' Mem: 0', type: 'scatter',
            mode: 'lines', line: {width: 1, color: '#555'},
            xaxis: xax(memRow), yaxis: yax(memRow), legendgroup: 'gpu_mem_' + gi,
          });
        }
        if (hasPcie) {
          const pyIdx = nRows + 1 + gi;
          pcieYIdxPerGpu_mem.push({idx: pyIdx, memRow: memRow});
          const pcieY = 'y' + pyIdx;
          const pColor = gpuPcieColors[gi % gpuPcieColors.length];
          traceMap['pcie_tx_' + gi] = traceCount++;
          traces.push({
            x: times, y: msg.pcie_tx[gi], name: 'PCIe W (GPU ' + gi + ')', type: 'scatter',
            line: {color: pColor, width: 0.5},
            xaxis: xax(memRow), yaxis: pcieY,
            legendgroup: 'pcie_' + gi, legendgrouptitle: {text: 'GPU ' + gi + ' PCIe'},
          });
          traceMap['pcie_rx_' + gi] = traceCount++;
          traces.push({
            x: times, y: msg.pcie_rx[gi], name: 'PCIe R (GPU ' + gi + ')', type: 'scatter',
            line: {color: pColor, width: 0.5, dash: 'dot'},
            xaxis: xax(memRow), yaxis: pcieY,
            legendgroup: 'pcie_' + gi,
          });
        }

        // Util + Temperature row for this GPU
        row++;
        const utilRow = row;
        const uColor = '#76b900';
        traceMap['gpu_util_' + gi] = traceCount++;
        traces.push({
          x: times, y: msg.gpu_util[gi], name: 'GPU ' + gi + ' Util', type: 'scatter',
          line: {color: uColor, width: 2}, fill: 'tozeroy',
          fillcolor: hexToRgba(uColor, 0.25),
          xaxis: xax(utilRow), yaxis: yax(utilRow),
          legendgroup: 'util_' + gi, legendgrouptitle: {text: 'GPU ' + gi + ' Util'},
        });
        // GPU Temperature on secondary y-axis
        const tIdx = nRows + gpuCount + 1 + gi;
        tempYIdxPerGpu.push({idx: tIdx, utilRow: utilRow, gi: gi});
        const tempY = 'y' + tIdx;
        const tColor = '#ff6e40';
        traceMap['gpu_temp_' + gi] = traceCount++;
        traces.push({
          x: times, y: msg.gpu_temp[gi], name: 'GPU ' + gi + ' Temp', type: 'scatter',
          line: {color: tColor, width: 1.5, dash: 'dot'},
          xaxis: xax(utilRow), yaxis: tempY,
          legendgroup: 'util_' + gi,
        });
      }
    }

    // CPU row
    row++;
    const cpuRow = row;
    traceMap['cpu'] = traceCount++;
    traces.push({
      x: times, y: msg.cpu, name: 'CPU', type: 'scatter',
      line: {color: '#ffcc00', width: 1}, fill: 'tozeroy',
      fillcolor: hexToRgba('#ffcc00', 0.25),
      xaxis: xax(cpuRow), yaxis: yax(cpuRow),
      legendgroup: 'cpu', legendgrouptitle: {text: 'CPU'},
    });

    // Disk I/O + Network I/O row (interleaved, network on secondary y-axis)
    row++;
    const diskRow = row;
    const netYIdx = nRows + pcieYIdxPerGpu_mem.length + tempYIdxPerGpu.length + 1;
    const netY = 'y' + netYIdx;
    const diskData = msg.disk_io;
    for (let i = 0; i < diskData.length; i++) {
      const d = diskData[i];
      const diskPatterns = ['/', '\\', 'x', '+', '-', '|', '.', ''];
      const diskPatIdx = Math.abs(d.pid) % diskPatterns.length;
      traceMap['disk_' + d.pid] = traceCount++;
      traces.push({
        x: times, y: d.vals, name: d.name, type: 'scatter', mode: 'lines',
        line: {width: 0.5, color: d.color}, stackgroup: 'disk_io',
        fillcolor: hexToRgba(d.color, 0.55),
        fillpattern: {shape: diskPatterns[diskPatIdx], solidity: 0.15, size: 8, bgcolor: hexToRgba(d.color, 0.55), fgcolor: 'rgba(13,13,20,0.4)'},
        xaxis: xax(diskRow), yaxis: yax(diskRow),
        legendgroup: 'disk_io', legendgrouptitle: {text: 'Disk I/O'},
      });
    }
    // Network I/O overlaid
    traceMap['net_sent'] = traceCount++;
    traces.push({
      x: times, y: msg.net_sent, name: 'Net TX', type: 'scatter',
      line: {color: '#18ffff', width: 0.5, shape: 'spline', smoothing: 1.3},
      xaxis: xax(diskRow), yaxis: netY,
      legendgroup: 'net', legendgrouptitle: {text: 'Network'},
    });
    traceMap['net_recv'] = traceCount++;
    traces.push({
      x: times, y: msg.net_recv, name: 'Net RX', type: 'scatter',
      line: {color: '#ff80ab', width: 0.5, shape: 'spline', smoothing: 1.3},
      xaxis: xax(diskRow), yaxis: netY,
      legendgroup: 'net',
    });

    // Store keys for change detection (per-GPU arrays)
    config._last_gpu_keys = msg.gpu_mem.map(gm => gm.map(m => m.pid));
    config._last_disk_keys = msg.disk_io.map(d => d.pid);

    const bgColors = ['#101020', '#181830'];
    const rowLabels = [];
    if (hasGpu) {
      for (let gi = 0; gi < gpuCount; gi++) {
        rowLabels.push(hasPcie ? 'GPU ' + gi + ' Memory (GiB) + PCIe' : 'GPU ' + gi + ' Memory (GiB)');
        rowLabels.push('GPU ' + gi + ' Util (%) + Temp (\u00b0C)');
      }
    }
    rowLabels.push('CPU Usage (%)');
    rowLabels.push('Disk I/O (MB/s) + Network (MB/s)');

    const domain = function(i, total) {
      const gap = 0.025;
      const totalWeight = rowWeights.reduce((a, b) => a + b, 0);
      const totalGap = gap * (total - 1);
      const usable = 1 - totalGap;
      // Compute y0 from bottom up (row 0 = top)
      let y1 = 1;
      for (let r = 0; r < i; r++) {
        y1 -= (rowWeights[r] / totalWeight) * usable + gap;
      }
      const h = (rowWeights[i] / totalWeight) * usable;
      return [y1 - h, y1];
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

    // Build axis layout dynamically for all rows
    for (let r = 1; r <= nRows; r++) {
      const xName = r === 1 ? 'xaxis' : 'xaxis' + r;
      const yName = r === 1 ? 'yaxis' : 'yaxis' + r;
      const yRef = r === 1 ? 'y' : 'y' + r;
      const showTick = (r === nRows);
      const xRange = viewMinutes > 0 ? [new Date(Date.now() - viewMinutes * 60000), new Date()] : undefined;
      layout[xName] = {domain: [0, 0.92], anchor: yRef, showticklabels: showTick, gridcolor: '#262640', range: xRange};
      layout[yName] = {domain: domain(r - 1, nRows), gridcolor: '#262640'};
    }

    // Set y-axis types per row
    if (hasGpu) {
      for (let gi = 0; gi < gpuCount; gi++) {
        const memRowIdx = 2 * gi + 1;
        const utilRowIdx = 2 * gi + 2;
        const memYName = memRowIdx === 1 ? 'yaxis' : 'yaxis' + memRowIdx;
        const utilYName = 'yaxis' + utilRowIdx;
        const maxGib = msg.gpu_mem_total_gib[gi] || 1;
        layout[memYName].title = {text: '<span style="color:#76b900">GPU ' + gi + ' GiB</span>', font: {size: 10}};
        layout[memYName].range = [0, maxGib * 1.02];
        layout[utilYName].title = {text: '<span style="color:#76b900">GPU ' + gi + ' %</span>', font: {size: 10}};
        layout[utilYName].range = [0, 105];
      }
      // PCIe secondary axes (one per GPU mem row)
      for (const pci of pcieYIdxPerGpu_mem) {
        const pcieYName = 'yaxis' + pci.idx;
        const gi = (pci.memRow - 1) / 2;
        const pColor = gpuPcieColors[gi % gpuPcieColors.length];
        layout[pcieYName] = {
          domain: domain(pci.memRow - 1, nRows),
          title: {text: 'PCIe (log)', font: {color: pColor, size: 10}},
          overlaying: yax(pci.memRow), side: 'right', showgrid: false,
          type: 'log', autorange: true,
        };
      }
      // GPU Temp secondary axes (one per GPU util row)
      for (const tmp of tempYIdxPerGpu) {
        const tempYName = 'yaxis' + tmp.idx;
        layout[tempYName] = {
          domain: domain(tmp.utilRow - 1, nRows),
          title: {text: '\u00b0C', font: {color: '#ff6e40', size: 10}},
          overlaying: yax(tmp.utilRow), side: 'right', showgrid: false,
          range: [20, 95],
        };
      }
    }
    const cpuYName = cpuRow === 1 ? 'yaxis' : 'yaxis' + cpuRow;
    layout[cpuYName].title = {text: '<span style="color:#ffcc00">CPU %</span>', font: {size: 10}};
    layout[cpuYName].range = [0, 105];
    const diskYName = diskRow === 1 ? 'yaxis' : 'yaxis' + diskRow;
    layout[diskYName].title = {text: '<span style="color:#ff9800">Disk I/O MB/s</span>', font: {size: 10}};
    layout[diskYName].rangemode = 'tozero';
    // Network secondary y-axis (log) overlaying disk row
    const netYName = 'yaxis' + netYIdx;
    layout[netYName] = {
      domain: domain(diskRow - 1, nRows),
      title: {text: 'Net MB/s (log)', font: {color: '#18ffff', size: 10}},
      overlaying: yax(diskRow), side: 'right', showgrid: false,
      type: 'log', autorange: true,
    };

    // Add background shapes and labels
    const axPairs = [];
    for (let r = 1; r <= nRows; r++) {
      axPairs.push([xax(r), yax(r)]);
    }
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

    // GPU total memory dashed line per GPU
    if (hasGpu) {
      for (let gi = 0; gi < gpuCount; gi++) {
        const maxGib = msg.gpu_mem_total_gib[gi] || 1;
        const yr = yax(2 * gi + 1);
        const xr = xax(2 * gi + 1);
        layout.shapes.push({
          type: 'line', x0: 0, x1: 1, y0: maxGib, y1: maxGib,
          xref: xr + ' domain', yref: yr,
          line: {color: '#ff5252', width: 1, dash: 'dash'},
        });
        layout.annotations.push({
          text: maxGib.toFixed(1) + ' GiB', x: 1, y: maxGib,
          xref: xr + ' domain', yref: yr, showarrow: false,
          font: {size: 10, color: '#ff5252'}, xanchor: 'right',
        });
      }
    }

    Plotly.newPlot(gd, traces, layout, {
      displayModeBar: true, scrollZoom: true,
      toImageButtonOptions: {format: 'png', filename: 'gpu_monitor', height: 1080, width: 1920, scale: 2},
    });
  }

  function startScrollLoop() {
    if (scrollRAF) clearInterval(scrollRAF);
    scrollRAF = setInterval(function() {
      if (!paused && gd && viewMinutes > 0) {
        const now = new Date();
        const start = new Date(now.getTime() - viewMinutes * 60000);
        const upd = {};
        const gc = config.gpu_count || 0;
        const nr = (gc > 0 ? gc * 2 : 0) + 2;
        for (let r = 1; r <= nr; r++) {
          const ax = r === 1 ? 'xaxis' : 'xaxis' + r;
          upd[ax + '.range'] = [start, now];
        }
        Plotly.relayout(gd, upd);
      }
    }, 200);
  }

  // --- Controls ---
  document.querySelectorAll('.range-btn').forEach(btn => {
    btn.addEventListener('click', function() {
      document.querySelectorAll('.range-btn').forEach(b => b.classList.remove('active'));
      this.classList.add('active');
      viewMinutes = parseInt(this.dataset.min);
      if (viewMinutes === 0 && gd) {
        // Reset to autorange
        const gc2 = config.gpu_count || 0;
        const nr2 = (gc2 > 0 ? gc2 * 2 : 0) + 2;
        const upd = {};
        for (let r = 1; r <= nr2; r++) {
          const ax = r === 1 ? 'xaxis' : 'xaxis' + r;
          upd[ax + '.autorange'] = true;
        }
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
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                        ping_timeout=60, ping_interval=25, max_http_buffer_size=50 * 1024 * 1024)

    ui_state = {"paused": False}

    @app.route("/")
    def index():
        return Response(HTML_PAGE, mimetype="text/html")

    @socketio.on("connect")
    def handle_connect():
        from flask import request as flask_request
        sid = flask_request.sid
        # Wait for data to be available before sending init
        def send_init():
            for _ in range(50):
                with collector.lock:
                    if len(collector.timestamps) > 5:
                        break
                time.sleep(0.1)
            data = collector.snapshot_full()
            socketio.emit("init", data, to=sid)
        threading.Thread(target=send_init, daemon=True).start()

    @socketio.on("request_init")
    def handle_request_init():
        from flask import request as flask_request
        data = collector.snapshot_full()
        socketio.emit("init", data, to=flask_request.sid)

    @socketio.on("pause")
    def handle_pause(is_paused):
        ui_state["paused"] = bool(is_paused)

    def push_loop():
        """Background thread that pushes deltas to all clients."""
        push_sec = args.push_interval / 1000.0
        last_counter = 0
        while True:
            time.sleep(push_sec)
            if ui_state["paused"]:
                continue
            with collector.lock:
                cur_counter = collector.sample_counter
            if cur_counter <= last_counter:
                continue
            try:
                delta = collector.snapshot_delta(last_counter)
                last_counter = cur_counter
                socketio.emit("delta", delta)
            except Exception as e:
                print(f"[push_loop] error: {e}", flush=True)

    return app, socketio, ui_state, push_loop


def main():
    args = parse_args()
    print(f"Fast collect : {args.interval}ms (CPU, PCIe, GPU mem)")
    print(f"Slow collect : {args.gpu_interval}ms (GPU util, disk I/O, network)")
    print(f"Push interval: {args.push_interval}ms")
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
    def _fast_loop():
        interval_sec = args.interval / 1000.0
        while True:
            if not ui_state["paused"]:
                collector.sample_fast()
            time.sleep(interval_sec)

    def _slow_loop():
        interval_sec = args.gpu_interval / 1000.0
        while True:
            if not ui_state["paused"]:
                collector.sample_slow()
            time.sleep(interval_sec)

    threading.Thread(target=_fast_loop, daemon=True).start()
    threading.Thread(target=_slow_loop, daemon=True).start()
    threading.Thread(target=push_loop, daemon=True).start()

    print(f"Open http://{args.host}:{args.port} in your browser")
    import socketserver
    socketserver.TCPServer.allow_reuse_address = True
    socketio.run(app, host=args.host, port=args.port, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
