#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Real-time GPU/CPU/Disk monitor with smooth 60fps scrolling.

Backend: Flask + flask-socketio pushes incremental data via WebSocket.
Frontend: Raw Plotly.js with extendTraces() + requestAnimationFrame scrolling.

Layout:
  1. GPU Memory by Process (GiB) -- stacked area
  2. GPU Util (%) + PCIe (GB/s)  -- lines, secondary y-axis
  3. CPU Usage (%)               -- line
  4. Disk I/O by Process (MB/s)  -- stacked area

Usage:
    python3 gpu_monitor.py [--port 8051] [--host 127.0.0.1]
"""

import argparse
import os
import threading
import time
from collections import deque

import socketserver

import psutil
import pynvml
from flask import Flask, Response, request as flask_request
from flask_socketio import SocketIO

HAS_NVML = False
try:
    pynvml.nvmlInit()
    HAS_NVML = True
except pynvml.NVMLError as e:
    print(f"NVML init failed ({e}) -- GPU monitoring disabled")

PROCESS_COLORS = [
    "#00e5ff",
    "#ff1744",
    "#76ff03",
    "#ffea00",
    "#d500f9",
    "#ff9100",
    "#00e676",
    "#ff6d00",
    "#448aff",
    "#e040fb",
    "#18ffff",
    "#f50057",
    "#64ffda",
    "#ff80ab",
    "#b2ff59",
    "#ffd740",
    "#ea80fc",
    "#ff9e80",
    "#a7ffeb",
    "#ff8a80",
]
OTHER_COLOR = "#555555"
DISK_SCAN_EVERY = 10


def parse_args():
    p = argparse.ArgumentParser(description="Real-time GPU/CPU/Disk monitor")
    p.add_argument("--port", type=int, default=8051)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument(
        "--interval",
        type=int,
        default=75,
        help="Fast collection interval ms (CPU only)",
    )
    p.add_argument(
        "--pcie-interval",
        type=int,
        default=35,
        help="PCIe collection interval ms (default 35)",
    )
    p.add_argument(
        "--gpu-interval",
        type=int,
        default=500,
        help="Slow collection interval ms (GPU mem, util, temp, disk, network)",
    )
    p.add_argument(
        "--push-interval", type=int, default=150, help="WebSocket push interval ms"
    )
    p.add_argument(
        "--window",
        type=int,
        default=900,
        help="Rolling window seconds (default 900 = 15min)",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Show top N processes per chart, rest grouped as Other",
    )
    p.add_argument(
        "--fill-patterns",
        action="store_true",
        default=False,
        help="Enable fill patterns on stacked areas",
    )
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
        if (
            "sglang" in proc_name.lower()
            or "sglang" in full_cmd.lower()
            or "trtllm" in proc_name.lower()
            or "trtllm" in full_cmd.lower()
        ):
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

    def get_all_sorted(self) -> list[tuple[int, str, str, list[float]]]:
        """Return (pid, name, color, values) sorted by peak desc."""
        if not self.series:
            return []
        sorted_pids = sorted(
            self.series.keys(),
            key=lambda p: max(self.series[p]) if self.series[p] else 0,
            reverse=True,
        )
        return [
            (pid, self.names[pid], self.get_color(pid), list(self.series[pid]))
            for pid in sorted_pids
            if max(self.series[pid]) > 0
        ]

    def get_top_sorted(
        self, n=20, sort_by: str = "recency"
    ) -> list[tuple[int, str, str, list[float]]]:
        if not self.series:
            return []

        if sort_by == "recency":

            def _key(p):
                s = self.series[p]
                last_active = -1
                for i in range(len(s) - 1, -1, -1):
                    if s[i] > 0:
                        last_active = i
                        break
                return (last_active, max(s) if s else 0)
        else:

            def _key(p):
                return sum(self.series[p])

        sorted_pids = sorted(self.series.keys(), key=_key, reverse=True)
        top = sorted_pids[:n]
        rest = sorted_pids[n:]
        result = [
            (pid, self.names[pid], self.get_color(pid), list(self.series[pid]))
            for pid in top
            if max(self.series[pid]) > 0
        ]
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
    def __init__(
        self,
        window_sec: int,
        interval_ms: int,
        pcie_interval_ms: int = 35,
        slow_interval_ms: int = 500,
        top_n: int = 10,
        fill_patterns: bool = False,
    ):
        self.lock = threading.Lock()
        maxlen_fast = int(window_sec * 1000 / interval_ms)
        maxlen_pcie = int(window_sec * 1000 / pcie_interval_ms)
        maxlen_slow = int(window_sec * 1000 / slow_interval_ms)
        self.top_n = top_n
        self.fill_patterns = fill_patterns
        # 3 independent timestamp groups
        self.ts_fast: deque[float] = deque(maxlen=maxlen_fast)
        self.ts_pcie: deque[float] = deque(maxlen=maxlen_pcie)
        self.ts_slow: deque[float] = deque(maxlen=maxlen_slow)
        self.counter_fast: int = 0
        self.counter_pcie: int = 0
        self.counter_slow: int = 0
        # Fast group: CPU only
        self.cpu_pct: deque[float] = deque(maxlen=maxlen_fast)
        # Slow group: GPU mem, GPU util, GPU temp, network, disk I/O
        self.proc_gpu_mem: list[ProcessTracker] = []
        self._prev_net = psutil.net_io_counters()
        self._prev_mono = time.monotonic()
        self.net_sent_mbps: deque[float] = deque(maxlen=maxlen_slow)
        self.net_recv_mbps: deque[float] = deque(maxlen=maxlen_slow)
        self.gpu_util: list[deque[float]] = []
        self.proc_disk_io = ProcessTracker(maxlen_slow, prune=True)
        self._prev_proc_io: dict[int, int] = {}
        self._last_disk_rates: dict[int, float] = {}
        self._last_disk_scan_mono = time.monotonic()
        # PCIe group
        self.gpu_pcie_tx: list[deque[float]] = []
        self.gpu_pcie_rx: list[deque[float]] = []
        self.has_pcie = False
        self.gpu_count = 0
        self.gpu_names: list[str] = []
        self.gpu_mem_total_gib: list[float] = []
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
                self.proc_gpu_mem.append(ProcessTracker(maxlen_slow, prune=False))
                self.gpu_util.append(deque(maxlen=maxlen_slow))
                self.gpu_pcie_tx.append(deque(maxlen=maxlen_pcie))
                self.gpu_pcie_rx.append(deque(maxlen=maxlen_pcie))
        self.gpu_temp: list[deque[float]] = [
            deque(maxlen=maxlen_slow) for _ in range(self.gpu_count)
        ]
        if self.gpu_count > 0:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_TX_BYTES)
                self.has_pcie = True
            except pynvml.NVMLError:
                self.has_pcie = False

    def sample_fast(self):
        """Fast-tier: CPU only (called at --interval rate)."""
        with self.lock:
            now = time.time()
            self.ts_fast.append(now)
            self.counter_fast += 1
            self.cpu_pct.append(psutil.cpu_percent(interval=None))

    def sample_pcie(self):
        """PCIe-tier: PCIe throughput per GPU (called at --pcie-interval rate)."""
        if not self.has_pcie:
            return
        with self.lock:
            now = time.time()
            self.ts_pcie.append(now)
            self.counter_pcie += 1
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                try:
                    tx = pynvml.nvmlDeviceGetPcieThroughput(
                        handle, pynvml.NVML_PCIE_UTIL_TX_BYTES
                    )
                    rx = pynvml.nvmlDeviceGetPcieThroughput(
                        handle, pynvml.NVML_PCIE_UTIL_RX_BYTES
                    )
                    self.gpu_pcie_tx[i].append(tx / (1024.0 * 1024.0))
                    self.gpu_pcie_rx[i].append(rx / (1024.0 * 1024.0))
                except pynvml.NVMLError:
                    self.gpu_pcie_tx[i].append(
                        self.gpu_pcie_tx[i][-1] if self.gpu_pcie_tx[i] else 0.0
                    )
                    self.gpu_pcie_rx[i].append(
                        self.gpu_pcie_rx[i][-1] if self.gpu_pcie_rx[i] else 0.0
                    )

    def sample_slow(self):
        """Slow-tier: GPU util, GPU temp, network, disk I/O (called at --gpu-interval rate)."""
        with self.lock:
            now = time.time()
            self.ts_slow.append(now)
            self.counter_slow += 1
            # Disk I/O per-process scan
            scan_mono = time.monotonic()
            scan_dt = scan_mono - self._last_disk_scan_mono
            if scan_dt <= 0:
                scan_dt = 0.5
            raw_rates = self._scan_process_disk(scan_dt)
            alpha = 0.2
            for pid, rate in raw_rates.items():
                prev = self._last_disk_rates.get(pid, rate)
                raw_rates[pid] = alpha * rate + (1 - alpha) * prev
            self._last_disk_rates = raw_rates
            self._last_disk_scan_mono = scan_mono
            self.proc_disk_io.record(self._last_disk_rates, _resolve_process_name, now)
            # Network I/O (append with EMA smoothing)
            net = psutil.net_io_counters()
            net_dt = scan_dt if scan_dt > 0 else 0.5
            sent_rate = (
                (net.bytes_sent - self._prev_net.bytes_sent) / (1024 * 1024) / net_dt
            )
            recv_rate = (
                (net.bytes_recv - self._prev_net.bytes_recv) / (1024 * 1024) / net_dt
            )
            self._prev_net = net
            alpha_net = 0.15
            prev_sent = self.net_sent_mbps[-1] if self.net_sent_mbps else 0.0
            prev_recv = self.net_recv_mbps[-1] if self.net_recv_mbps else 0.0
            self.net_sent_mbps.append(
                alpha_net * sent_rate + (1 - alpha_net) * prev_sent
            )
            self.net_recv_mbps.append(
                alpha_net * recv_rate + (1 - alpha_net) * prev_recv
            )
            # GPU: memory per-process, utilization, temperature
            if HAS_NVML:
                for i in range(self.gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    # GPU memory per-process
                    gpu_proc_mem: dict[int, float] = {}
                    for getter in (
                        pynvml.nvmlDeviceGetComputeRunningProcesses,
                        pynvml.nvmlDeviceGetGraphicsRunningProcesses,
                    ):
                        try:
                            for p in getter(handle):
                                mem_gib = (p.usedGpuMemory or 0) / (1024**3)
                                gpu_proc_mem[p.pid] = (
                                    gpu_proc_mem.get(p.pid, 0.0) + mem_gib
                                )
                        except pynvml.NVMLError:
                            pass
                    self.proc_gpu_mem[i].record(
                        gpu_proc_mem, _resolve_process_name, now
                    )
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        self.gpu_util[i].append(util.gpu)
                    except pynvml.NVMLError:
                        self.gpu_util[i].append(
                            self.gpu_util[i][-1] if self.gpu_util[i] else 0.0
                        )
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                        self.gpu_temp[i].append(float(temp))
                    except pynvml.NVMLError:
                        self.gpu_temp[i].append(
                            self.gpu_temp[i][-1] if self.gpu_temp[i] else 0.0
                        )

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
    def _downsample_init(times, arrays, max_recent=1000, max_total=1500):
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
        ds_arrays = [
            [arr[i] for i in indices] if len(arr) == n else arr for arr in arrays
        ]
        return ds_times, ds_arrays

    def snapshot_full(self) -> dict:
        """Return full state for initial browser load (downsampled for browser perf)."""
        with self.lock:
            # --- Fast group: CPU only ---
            cpu = list(self.cpu_pct)
            ts_fast_ds, fast_ds = self._downsample_init(list(self.ts_fast), [cpu])
            ds_cpu = fast_ds[0]

            # --- PCIe group ---
            pcie_vals = []
            pcie_tx_raw = (
                [list(self.gpu_pcie_tx[i]) for i in range(self.gpu_count)]
                if self.has_pcie
                else []
            )
            pcie_rx_raw = (
                [list(self.gpu_pcie_rx[i]) for i in range(self.gpu_count)]
                if self.has_pcie
                else []
            )
            pcie_vals.extend(pcie_tx_raw)
            pcie_vals.extend(pcie_rx_raw)
            ts_pcie_ds, pcie_ds = self._downsample_init(list(self.ts_pcie), pcie_vals)
            pidx = 0
            ds_pcie_tx = [pcie_ds[pidx + i] for i in range(len(pcie_tx_raw))]
            pidx += len(pcie_tx_raw)
            ds_pcie_rx = [pcie_ds[pidx + i] for i in range(len(pcie_rx_raw))]

            # --- Slow group: GPU mem, GPU util, temp, network, disk ---
            slow_vals = []
            gpu_mem_per_gpu_full = []
            for gi in range(self.gpu_count):
                procs = self.proc_gpu_mem[gi].get_top_sorted(self.top_n)
                gpu_mem_per_gpu_full.append(procs)
                for _, _, _, v in procs:
                    slow_vals.append(v)
            gpu_util = [list(self.gpu_util[i]) for i in range(self.gpu_count)]
            gpu_temp = [list(self.gpu_temp[i]) for i in range(self.gpu_count)]
            slow_vals.extend(gpu_util)
            slow_vals.extend(gpu_temp)
            net_sent = list(self.net_sent_mbps)
            net_recv = list(self.net_recv_mbps)
            slow_vals.append(net_sent)
            slow_vals.append(net_recv)
            disk_io_full = self.proc_disk_io.get_top_sorted(5, sort_by="total")
            for _, _, _, v in disk_io_full:
                slow_vals.append(v)
            ts_slow_ds, slow_ds = self._downsample_init(list(self.ts_slow), slow_vals)
            sidx = 0
            gpu_mem_per_gpu = []
            for gi in range(self.gpu_count):
                ds_procs = []
                for p, n, c, v in gpu_mem_per_gpu_full[gi]:
                    ds_procs.append(
                        {
                            "pid": p,
                            "name": n,
                            "color": c,
                            "vals": slow_ds[sidx],
                            "first_seen": self.proc_gpu_mem[gi].first_seen.get(p, 0),
                        }
                    )
                    sidx += 1
                gpu_mem_per_gpu.append(ds_procs)
            ds_gpu_util = [slow_ds[sidx + i] for i in range(self.gpu_count)]
            sidx += self.gpu_count
            ds_gpu_temp = [slow_ds[sidx + i] for i in range(self.gpu_count)]
            sidx += self.gpu_count
            ds_net_sent = slow_ds[sidx]
            sidx += 1
            ds_net_recv = slow_ds[sidx]
            sidx += 1
            disk_io = []
            for p, n, c, v in disk_io_full:
                disk_io.append({"pid": p, "name": n, "color": c, "vals": slow_ds[sidx]})
                sidx += 1

            return {
                "type": "init",
                "counter_fast": self.counter_fast,
                "counter_pcie": self.counter_pcie,
                "counter_slow": self.counter_slow,
                "ts_fast": ts_fast_ds,
                "ts_pcie": ts_pcie_ds,
                "ts_slow": ts_slow_ds,
                "hostname": os.uname().nodename,
                "gpu_count": self.gpu_count,
                "gpu_names": self.gpu_names,
                "gpu_mem_total_gib": self.gpu_mem_total_gib,
                "has_pcie": self.has_pcie,
                "fill_patterns": self.fill_patterns,
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

    def snapshot_delta(
        self, since_fast: int, since_pcie: int, since_slow: int, step: int = 1
    ) -> dict:
        """Return new data since the given counters, subsampled by step."""
        with self.lock:
            # Fast group (CPU only)
            new_fast = self.counter_fast - since_fast
            fast_len = len(self.ts_fast)
            fast_idx = max(0, fast_len - new_fast) if new_fast > 0 else fast_len
            sl_f = slice(fast_idx, None, step)
            new_ts_fast = list(self.ts_fast)[sl_f]
            cpu = list(self.cpu_pct)[sl_f]

            # PCIe group
            new_pcie = self.counter_pcie - since_pcie
            pcie_len = len(self.ts_pcie)
            pcie_idx = max(0, pcie_len - new_pcie) if new_pcie > 0 else pcie_len
            sl_p = slice(pcie_idx, None, step)
            new_ts_pcie = list(self.ts_pcie)[sl_p]
            pcie_tx = (
                [list(self.gpu_pcie_tx[i])[sl_p] for i in range(self.gpu_count)]
                if self.has_pcie
                else []
            )
            pcie_rx = (
                [list(self.gpu_pcie_rx[i])[sl_p] for i in range(self.gpu_count)]
                if self.has_pcie
                else []
            )

            # Slow group (GPU mem, util, temp, network, disk)
            new_slow = self.counter_slow - since_slow
            slow_len = len(self.ts_slow)
            slow_idx = max(0, slow_len - new_slow) if new_slow > 0 else slow_len
            sl_s = slice(slow_idx, None, step)
            new_ts_slow = list(self.ts_slow)[sl_s]
            gpu_mem_per_gpu = []
            gpu_mem_keys_per_gpu = []
            for gi in range(self.gpu_count):
                gm = self.proc_gpu_mem[gi].get_top_sorted(self.top_n)
                gpu_mem_per_gpu.append(
                    [
                        {
                            "pid": p,
                            "name": n,
                            "color": c,
                            "vals": v[sl_s] if slow_idx < len(v) else [],
                        }
                        for p, n, c, v in gm
                    ]
                )
                gpu_mem_keys_per_gpu.append([p for p, _, _, _ in gm])
            gpu_util = [list(self.gpu_util[i])[sl_s] for i in range(self.gpu_count)]
            gpu_temp = [list(self.gpu_temp[i])[sl_s] for i in range(self.gpu_count)]
            net_sent = list(self.net_sent_mbps)[sl_s]
            net_recv = list(self.net_recv_mbps)[sl_s]
            disk_io = self.proc_disk_io.get_top_sorted(5, sort_by="total")

            return {
                "type": "delta",
                "counter_fast": self.counter_fast,
                "counter_pcie": self.counter_pcie,
                "counter_slow": self.counter_slow,
                "ts_fast": new_ts_fast,
                "ts_pcie": new_ts_pcie,
                "ts_slow": new_ts_slow,
                "gpu_mem": gpu_mem_per_gpu,
                "gpu_mem_keys": gpu_mem_keys_per_gpu,
                "gpu_util": gpu_util,
                "gpu_temp": gpu_temp,
                "pcie_tx": pcie_tx,
                "pcie_rx": pcie_rx,
                "cpu": cpu,
                "net_sent": net_sent,
                "net_recv": net_recv,
                "disk_io": [
                    {
                        "pid": p,
                        "name": n,
                        "color": c,
                        "vals": v[sl_s] if slow_idx < len(v) else [],
                    }
                    for p, n, c, v in disk_io
                ],
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

  const MAX_POINTS = 3000;
  let viewMinutes = 2;
  let paused = false;
  let syncing = false;
  let gd = null;
  let config = null;
  let traceMap = {};
  let traceCount = 0;
  let prev_srv_fast = 0, prev_srv_pcie = 0, prev_srv_slow = 0;
  let cur_srv_fast = 0, cur_srv_pcie = 0, cur_srv_slow = 0;
  let lastRateCheck = Date.now();
  let rate_fast = 0, rate_pcie = 0, rate_slow = 0;
  let rowAnnotations = []; // [{index, baseText, group}]
  

  const socket = io({transports: ['polling'], reconnection: true, reconnectionDelay: 500, reconnectionAttempts: Infinity, timeout: 60000});

  socket.on('connect', function() {
    document.getElementById('status').textContent = 'Connected. Waiting for data...';
    socket.emit('view_minutes', viewMinutes);
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
    var savedRange = null;
    var savedVM = viewMinutes;
    if (gd && viewMinutes === 0) {
      var ax1 = gd.layout.xaxis;
      if (ax1 && ax1.range) savedRange = [ax1.range[0], ax1.range[1]];
    }
    config = msg;
    buildChart(msg);
    if (savedRange && savedVM === 0) {
      viewMinutes = 0;
      var gc = config.gpu_count || 0;
      var nr = (gc > 0 ? gc * 2 : 0) + 2;
      var upd = {};
      for (var r = 1; r <= nr; r++) {
        var ax = r === 1 ? 'xaxis' : 'xaxis' + r;
        upd[ax + '.range'] = savedRange;
        upd[ax + '.autorange'] = false;
      }
      syncing = true;
      Plotly.relayout(gd, upd).then(function() { syncing = false; });
    }
    if (msg.hostname) {
      document.querySelector('#toolbar h1').textContent = 'System Monitor - ' + msg.hostname;
      document.title = 'System Monitor - ' + msg.hostname;
    }
    if (viewMinutes > 0) {
      document.getElementById('status').textContent = 'Live';
    }
    startScrollLoop();
  });

  socket.on('delta', function(msg) {
    if (!gd || !config) return;
    const tf = msg.ts_fast || [], tp = msg.ts_pcie || [], ts = msg.ts_slow || [];
    if (tf.length === 0 && tp.length === 0 && ts.length === 0) return;

    cur_srv_fast = msg.counter_fast || cur_srv_fast;
    cur_srv_pcie = msg.counter_pcie || cur_srv_pcie;
    cur_srv_slow = msg.counter_slow || cur_srv_slow;

    const gpuKeysChanged = JSON.stringify(msg.gpu_mem_keys) !== JSON.stringify(config._last_gpu_keys);
    const diskKeysChanged = JSON.stringify(msg.disk_keys) !== JSON.stringify(config._last_disk_keys);
    if (gpuKeysChanged || diskKeysChanged) {
      socket.emit('request_init');
      return;
    }

    const gc = config.gpu_count || 0;
    const nr = (gc > 0 ? gc * 2 : 0) + 2;

    // Fast group: CPU only
    if (tf.length > 0) {
      const times_f = tf.map(t => new Date(t * 1000));
      const fx = [], fy = [], fi = [];
      fi.push(traceMap['cpu']); fx.push(times_f); fy.push(msg.cpu);
      if (fi.length > 0) Plotly.extendTraces(gd, {x: fx, y: fy}, fi, MAX_POINTS);
    }

    // PCIe group
    if (tp.length > 0 && config.has_pcie) {
      const times_p = tp.map(t => new Date(t * 1000));
      const px = [], py = [], pi = [];
      for (let i = 0; i < msg.pcie_tx.length; i++) {
        pi.push(traceMap['pcie_tx_' + i]); px.push(times_p); py.push(msg.pcie_tx[i]);
        pi.push(traceMap['pcie_rx_' + i]); px.push(times_p); py.push(msg.pcie_rx[i]);
      }
      if (pi.length > 0) Plotly.extendTraces(gd, {x: px, y: py}, pi, MAX_POINTS);
    }

    // Slow group: GPU mem, GPU util, temp, network, disk
    if (ts.length > 0) {
      const times_s = ts.map(t => new Date(t * 1000));
      const sx = [], sy = [], si = [];
      for (let gi = 0; gi < msg.gpu_mem.length; gi++) {
        for (let i = 0; i < msg.gpu_mem[gi].length; i++) {
          const key = 'gpu_mem_' + gi + '_' + msg.gpu_mem[gi][i].pid;
          if (traceMap[key] !== undefined) {
            si.push(traceMap[key]); sx.push(times_s); sy.push(msg.gpu_mem[gi][i].vals);
          }
        }
      }
      for (let i = 0; i < msg.gpu_util.length; i++) {
        si.push(traceMap['gpu_util_' + i]); sx.push(times_s); sy.push(msg.gpu_util[i]);
      }
      if (msg.gpu_temp) {
        for (let i = 0; i < msg.gpu_temp.length; i++) {
          if (traceMap['gpu_temp_' + i] !== undefined) {
            si.push(traceMap['gpu_temp_' + i]); sx.push(times_s); sy.push(msg.gpu_temp[i]);
          }
        }
      }
      for (let i = 0; i < msg.disk_io.length; i++) {
        const key = 'disk_' + msg.disk_io[i].pid;
        if (traceMap[key] !== undefined) {
          si.push(traceMap[key]); sx.push(times_s); sy.push(msg.disk_io[i].vals);
        }
      }
      if (traceMap['net_sent'] !== undefined) {
        si.push(traceMap['net_sent']); sx.push(times_s); sy.push(msg.net_sent);
        si.push(traceMap['net_recv']); sx.push(times_s); sy.push(msg.net_recv);
      }
      if (si.length > 0) Plotly.extendTraces(gd, {x: sx, y: sy}, si, MAX_POINTS);
    }

    // Scroll
    if (!paused && viewMinutes > 0) {
      const now = new Date();
      const start = new Date(now.getTime() - viewMinutes * 60000);
      for (let r = 1; r <= nr; r++) {
        const ax = r === 1 ? 'xaxis' : 'xaxis' + r;
        gd.layout[ax].range = [start, now];
        gd.layout[ax].autorange = false;
      }
    }

    // Update row annotation rates
    for (const ra of rowAnnotations) {
      let rateStr = '';
      if (ra.group === 'slow_pcie') rateStr = rate_slow + ' samples/s, ' + rate_pcie + ' samples/s';
      else if (ra.group === 'fast') rateStr = rate_fast + ' samples/s';
      else if (ra.group === 'slow') rateStr = rate_slow + ' samples/s';
      else if (ra.group === 'pcie') rateStr = rate_pcie + ' samples/s';
      gd.layout.annotations[ra.index].text = '  ' + ra.baseText + '  (' + rateStr + ')  ';
    }
    syncing = true;
    Plotly.relayout(gd, {});
    syncing = false;
  });

  function buildChart(msg) {
    gd = document.getElementById('chart');
    traceMap = {};
    traceCount = 0;
    rowAnnotations = [];
    const traces = [];
    const times_fast = (msg.ts_fast || []).map(t => new Date(t * 1000));
    const times_pcie = (msg.ts_pcie || []).map(t => new Date(t * 1000));
    const times_slow = (msg.ts_slow || []).map(t => new Date(t * 1000));

    const hasGpu = msg.gpu_count > 0;
    const hasPcie = msg.has_pcie;
    const gpuCount = msg.gpu_count;
    const sysLegName = hasGpu ? (gpuCount === 1 ? 'legend2' : 'legend' + (gpuCount + 1)) : 'legend';
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
        const gpuLeg = gi === 0 ? 'legend' : 'legend' + (gi + 1);
        const memPatterns = ['/', '\\', 'x', '+', '-', '|', '.'];
        const memColors = gpuMem.map(m => m.color);
        const similarSet = findSimilarColors(memColors, 30);
        for (let i = gpuMem.length - 1; i >= 0; i--) {
          const m = gpuMem[i];
          traceMap['gpu_mem_' + gi + '_' + m.pid] = traceCount++;
          const trace = {
            x: times_slow, y: m.vals, name: m.name, type: 'scatter', mode: 'none',
            line: {width: 0, color: m.color}, stackgroup: stackName,
            fillcolor: hexToRgba(m.color, 0.6),
            xaxis: xax(memRow), yaxis: yax(memRow),
            legendgroup: 'gpu_mem_' + gi, legendgrouptitle: {text: 'GPU ' + gi + ' Mem'}, legend: gpuLeg,
          };
          if (similarSet.has(i)) {
            const pat = memPatterns[Math.abs(m.pid) % memPatterns.length];
            trace.fillpattern = {shape: pat, solidity: 0.15, size: 8, bgcolor: hexToRgba(m.color, 0.6), fgcolor: 'rgba(13,13,20,0.4)'};
          }
          traces.push(trace);
        }
        if (gpuMem.length === 0) {
          traceMap['gpu_mem_' + gi + '_empty'] = traceCount++;
          traces.push({
            x: times_slow, y: times_slow.map(() => 0), name: 'GPU ' + gi + ' Mem: 0', type: 'scatter',
            mode: 'lines', line: {width: 1, color: '#555'},
            xaxis: xax(memRow), yaxis: yax(memRow), legendgroup: 'gpu_mem_' + gi, legend: gpuLeg,
          });
        }
        if (hasPcie) {
          const pyIdx = nRows + 1 + gi;
          pcieYIdxPerGpu_mem.push({idx: pyIdx, memRow: memRow});
          const pcieY = 'y' + pyIdx;
          const pColor = gpuPcieColors[gi % gpuPcieColors.length];
          traceMap['pcie_tx_' + gi] = traceCount++;
          traces.push({
            x: times_pcie, y: msg.pcie_tx[gi], name: 'PCIe W (GPU ' + gi + ')', type: 'scatter',
            line: {color: pColor, width: 0.5},
            xaxis: xax(memRow), yaxis: pcieY,
            legendgroup: 'pcie_' + gi, legendgrouptitle: {text: 'GPU ' + gi + ' PCIe'}, legend: gpuLeg,
          });
          traceMap['pcie_rx_' + gi] = traceCount++;
          traces.push({
            x: times_pcie, y: msg.pcie_rx[gi], name: 'PCIe R (GPU ' + gi + ')', type: 'scatter',
            line: {color: pColor, width: 0.5, dash: 'dot'},
            xaxis: xax(memRow), yaxis: pcieY,
            legendgroup: 'pcie_' + gi, legend: gpuLeg,
          });
        }

        // Util + Temperature row for this GPU
        row++;
        const utilRow = row;
        const uColor = '#76b900';
        traceMap['gpu_util_' + gi] = traceCount++;
        traces.push({
          x: times_slow, y: msg.gpu_util[gi], name: 'GPU ' + gi + ' Util', type: 'scatter',
          line: {color: uColor, width: 2}, fill: 'tozeroy',
          fillcolor: hexToRgba(uColor, 0.25),
          xaxis: xax(utilRow), yaxis: yax(utilRow),
          legendgroup: 'util_' + gi, legendgrouptitle: {text: 'GPU ' + gi + ' Util'}, legend: gpuLeg,
        });
        // GPU Temperature on secondary y-axis
        const tIdx = nRows + gpuCount + 1 + gi;
        tempYIdxPerGpu.push({idx: tIdx, utilRow: utilRow, gi: gi});
        const tempY = 'y' + tIdx;
        const tColor = '#ff6e40';
        traceMap['gpu_temp_' + gi] = traceCount++;
        traces.push({
          x: times_slow, y: msg.gpu_temp[gi], name: 'GPU ' + gi + ' Temp', type: 'scatter',
          line: {color: tColor, width: 1.5, dash: 'dot'},
          xaxis: xax(utilRow), yaxis: tempY,
          legendgroup: 'util_' + gi, legend: gpuLeg,
        });
      }
    }

    // CPU row
    row++;
    const cpuRow = row;
    traceMap['cpu'] = traceCount++;
    traces.push({
      x: times_fast, y: msg.cpu, name: 'CPU', type: 'scatter',
      line: {color: '#ffcc00', width: 1}, fill: 'tozeroy',
      fillcolor: hexToRgba('#ffcc00', 0.25),
      xaxis: xax(cpuRow), yaxis: yax(cpuRow),
      legendgroup: 'cpu', legendgrouptitle: {text: 'CPU'}, legend: sysLegName,
    });

    // Disk I/O + Network I/O row (interleaved, network on secondary y-axis)
    row++;
    const diskRow = row;
    const netYIdx = nRows + pcieYIdxPerGpu_mem.length + tempYIdxPerGpu.length + 1;
    const netY = 'y' + netYIdx;
    const diskData = msg.disk_io;
    const diskPatterns = ['/', '\\', 'x', '+', '-', '|', '.', ''];
    for (let i = 0; i < diskData.length; i++) {
      const d = diskData[i];
      const diskPatIdx = Math.abs(d.pid) % diskPatterns.length;
      traceMap['disk_' + d.pid] = traceCount++;
      const dTrace = {
        x: times_slow, y: d.vals, name: d.name, type: 'scatter', mode: 'lines',
        line: {width: 0.5, color: d.color, shape: 'spline', smoothing: 0.8}, stackgroup: 'disk_io',
        fillcolor: hexToRgba(d.color, 0.55),
        xaxis: xax(diskRow), yaxis: yax(diskRow),
        legendgroup: 'disk_io', legendgrouptitle: {text: 'Disk I/O'}, legend: sysLegName,
      };
      if (msg.fill_patterns) {
        dTrace.fillpattern = {shape: diskPatterns[diskPatIdx], solidity: 0.15, size: 8, bgcolor: hexToRgba(d.color, 0.55), fgcolor: 'rgba(13,13,20,0.4)'};
      }
      traces.push(dTrace);
    }
    // Network I/O overlaid
    traceMap['net_sent'] = traceCount++;
    traces.push({
      x: times_slow, y: msg.net_sent, name: 'Net TX', type: 'scatter',
      line: {color: '#18ffff', width: 0.5, shape: 'spline', smoothing: 1.3},
      xaxis: xax(diskRow), yaxis: netY,
      legendgroup: 'net', legendgrouptitle: {text: 'Network'}, legend: sysLegName,
    });
    traceMap['net_recv'] = traceCount++;
    traces.push({
      x: times_slow, y: msg.net_recv, name: 'Net RX', type: 'scatter',
      line: {color: '#ff80ab', width: 0.5, shape: 'spline', smoothing: 1.3},
      xaxis: xax(diskRow), yaxis: netY,
      legendgroup: 'net', legend: sysLegName,
    });

    // Store keys for change detection (per-GPU arrays)
    config._last_gpu_keys = msg.gpu_mem.map(gm => gm.map(m => m.pid));
    config._last_disk_keys = msg.disk_io.map(d => d.pid);

    const bgColors = ['#101020', '#181830'];
    const rowLabels = [];
    const rowGroups = [];
    if (hasGpu) {
      for (let gi = 0; gi < gpuCount; gi++) {
        rowLabels.push(hasPcie ? 'GPU ' + gi + ' Memory (GiB) + PCIe (GB/s)' : 'GPU ' + gi + ' Memory (GiB)');
        rowGroups.push(hasPcie ? 'slow_pcie' : 'slow');
        rowLabels.push('GPU ' + gi + ' %-util + GPU \u00b0C');
        rowGroups.push('slow');
      }
    }
    rowLabels.push('CPU Usage (%)');
    rowGroups.push('fast');
    rowLabels.push('Disk I/O (MB/s) + Network (MB/s)');
    rowGroups.push('slow');

    // Per-row gaps: small between mem+util (same GPU), large between GPU groups and before CPU
    const rowGaps = [];
    if (hasGpu) {
      for (let gi = 0; gi < gpuCount; gi++) {
        if (gi > 0) rowGaps.push(0.04);  // gap before this GPU group's mem row
        else rowGaps.push(0);             // no gap before first row
        rowGaps.push(0.002);              // tiny gap between mem and util (same GPU)
      }
    }
    rowGaps.push(hasGpu ? 0.04 : 0);     // gap before CPU
    rowGaps.push(0.02);                   // gap before Disk+Net

    const domain = function(i, total) {
      const totalWeight = rowWeights.reduce((a, b) => a + b, 0);
      const totalGap = rowGaps.reduce((a, b) => a + b, 0);
      const usable = 1 - totalGap;
      let y1 = 1;
      for (let r = 0; r < i; r++) {
        y1 -= (rowWeights[r] / totalWeight) * usable + rowGaps[r + 1];
      }
      const h = (rowWeights[i] / totalWeight) * usable;
      return [y1 - h, y1];
    };

    const layout = {
      paper_bgcolor: '#0d0d14', plot_bgcolor: '#141420',
      font: {family: 'monospace', color: '#e0e0e0', size: 11},
      margin: {l: 70, r: 20, t: 10, b: 35},
      showlegend: true,
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
        layout[memYName].rangemode = 'tozero';
        layout[memYName].autorange = true;
        layout[utilYName].title = {text: '<span style="color:#76b900">GPU ' + gi + ' %-util</span>', font: {size: 10}};
        layout[utilYName].range = [0, 105];
      }
      // PCIe secondary axes (one per GPU mem row)
      for (const pci of pcieYIdxPerGpu_mem) {
        const pcieYName = 'yaxis' + pci.idx;
        const gi = (pci.memRow - 1) / 2;
        const pColor = gpuPcieColors[gi % gpuPcieColors.length];
        layout[pcieYName] = {
          domain: domain(pci.memRow - 1, nRows),
          title: {text: 'PCIe GB/s (log)', font: {color: pColor, size: 10}},
          overlaying: yax(pci.memRow), side: 'right', showgrid: false,
          type: 'log', autorange: true,
        };
      }
      // GPU Temp secondary axes (one per GPU util row)
      for (const tmp of tempYIdxPerGpu) {
        const tempYName = 'yaxis' + tmp.idx;
        layout[tempYName] = {
          domain: domain(tmp.utilRow - 1, nRows),
          title: {text: 'GPU \u00b0C', font: {color: '#ff6e40', size: 10}},
          overlaying: yax(tmp.utilRow), side: 'right', showgrid: false,
          autorange: true, minallowed: 30,
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

    // Create per-group legends aligned to their chart rows (grow downward)
    const legendStyle = {orientation: 'v', xanchor: 'left', x: 1.01, font: {size: 10}, bgcolor: 'rgba(13,13,20,0.8)', tracegroupgap: 1, itemsizing: 'constant', valign: 'top'};
    if (hasGpu) {
      for (let gi = 0; gi < gpuCount; gi++) {
        const memDom = domain(2 * gi, nRows);
        const legName = gi === 0 ? 'legend' : 'legend' + (gi + 1);
        layout[legName] = Object.assign({}, legendStyle, {yanchor: 'top', y: memDom[1]});
      }
    }
    const cpuDom = domain(nRows - 2, nRows);
    layout[sysLegName] = Object.assign({}, legendStyle, {yanchor: 'top', y: cpuDom[1]});

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
      rowAnnotations.push({index: layout.annotations.length, baseText: rowLabels[i], group: rowGroups[i]});
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

    // Sync zoom/pan across all x-axes and freeze auto-scroll on manual interaction
    gd.on('plotly_relayout', function(ev) {
      if (syncing) return;
      // Detect user-initiated x-range change (zoom/pan)
      let newRange = null;
      for (const key in ev) {
        const m = key.match(/^xaxis(\d*)\.range\[(\d)\]$/);
        if (m) {
          if (!newRange) newRange = [null, null];
          newRange[parseInt(m[2])] = ev[key];
        }
      }
      if (!newRange || newRange[0] === null || newRange[1] === null) return;
      // Freeze auto-scroll
      paused = true;
      viewMinutes = 0;
      document.querySelectorAll('.range-btn').forEach(b => b.classList.remove('active'));
      document.getElementById('pause-btn').textContent = 'Resume';
      document.getElementById('pause-btn').style.backgroundColor = '#ff5252';
      document.getElementById('status').textContent = 'Zoomed (click a time button or Resume)';
      // Sync all x-axes to the same range
      syncing = true;
      const gc = config.gpu_count || 0;
      const nr = (gc > 0 ? gc * 2 : 0) + 2;
      const upd = {};
      for (let r = 1; r <= nr; r++) {
        const ax = r === 1 ? 'xaxis' : 'xaxis' + r;
        upd[ax + '.range'] = [newRange[0], newRange[1]];
      }
      Plotly.relayout(gd, upd).then(function() { syncing = false; });
    });

    gd.on('plotly_doubleclick', function() {
      viewMinutes = 2;
      paused = false;
      document.querySelectorAll('.range-btn').forEach(b => b.classList.remove('active'));
      document.querySelector('.range-btn[data-min="2"]').classList.add('active');
      document.getElementById('pause-btn').textContent = 'Pause';
      document.getElementById('pause-btn').style.backgroundColor = '#ffcc00';
      socket.emit('view_minutes', viewMinutes);
      document.getElementById('status').textContent = 'Live';
      doScroll();
      return false;
    });
  }

  function doScroll() {
    if (paused || !gd || viewMinutes <= 0) return;
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

  setInterval(function() {
    const now = Date.now();
    const elapsed = (now - lastRateCheck) / 1000;
    if (elapsed > 0) {
      rate_fast = ((cur_srv_fast - prev_srv_fast) / elapsed).toFixed(1);
      rate_pcie = ((cur_srv_pcie - prev_srv_pcie) / elapsed).toFixed(1);
      rate_slow = ((cur_srv_slow - prev_srv_slow) / elapsed).toFixed(1);
    }
    prev_srv_fast = cur_srv_fast;
    prev_srv_pcie = cur_srv_pcie;
    prev_srv_slow = cur_srv_slow;
    lastRateCheck = now;
    const el = document.getElementById('status');
    const base = paused ? 'Paused' : 'Live';
    el.textContent = base + '  |  fast ' + rate_fast + ' samples/s  pcie ' + rate_pcie + ' samples/s  slow ' + rate_slow + ' samples/s';
    if (gd && rowAnnotations.length > 0) {
      for (const ra of rowAnnotations) {
        let rateStr = '';
        if (ra.group === 'slow_pcie') rateStr = rate_slow + ' samples/s, ' + rate_pcie + ' samples/s';
        else if (ra.group === 'fast') rateStr = rate_fast + ' samples/s';
        else if (ra.group === 'slow') rateStr = rate_slow + ' samples/s';
        else if (ra.group === 'pcie') rateStr = rate_pcie + ' samples/s';
        gd.layout.annotations[ra.index].text = '  ' + ra.baseText + '  (' + rateStr + ')  ';
      }
    }
  }, 5000);

  function startScrollLoop() {
    // Scrolling now happens in delta handler after extendTraces
  }

  // --- Controls ---
  document.querySelectorAll('.range-btn').forEach(btn => {
    btn.addEventListener('click', function() {
      document.querySelectorAll('.range-btn').forEach(b => b.classList.remove('active'));
      this.classList.add('active');
      viewMinutes = parseInt(this.dataset.min);
      paused = false;
      document.getElementById('pause-btn').textContent = 'Pause';
      document.getElementById('pause-btn').style.backgroundColor = '#ffcc00';
      document.getElementById('status').textContent = 'Live';
      socket.emit('view_minutes', viewMinutes);
      doScroll();
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
    if (!paused) {
      // Resuming: restore viewMinutes from active button, default 2m
      const activeBtn = document.querySelector('.range-btn.active');
      viewMinutes = activeBtn ? parseInt(activeBtn.dataset.min) : 2;
      if (!activeBtn) {
        document.querySelector('.range-btn[data-min="2"]').classList.add('active');
      }
      document.getElementById('status').textContent = 'Live';
      socket.emit('view_minutes', viewMinutes);
    }
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
  function hexToHue(hex) {
    const h = hex.replace('#', '');
    const r = parseInt(h.substring(0,2),16)/255;
    const g = parseInt(h.substring(2,4),16)/255;
    const b = parseInt(h.substring(4,6),16)/255;
    const mx = Math.max(r,g,b), mn = Math.min(r,g,b), d = mx - mn;
    if (d === 0) return {hue: 0, sat: 0};
    let hue;
    if (mx === r) hue = ((g - b) / d + 6) % 6;
    else if (mx === g) hue = (b - r) / d + 2;
    else hue = (r - g) / d + 4;
    return {hue: hue * 60, sat: d / mx};
  }
  function findSimilarColors(colors, threshold) {
    const needs = new Set();
    const hsv = colors.map(c => hexToHue(c));
    for (let i = 0; i < hsv.length; i++) {
      if (hsv[i].sat < 0.15) continue;
      for (let j = i + 1; j < hsv.length; j++) {
        if (hsv[j].sat < 0.15) continue;
        let dh = Math.abs(hsv[i].hue - hsv[j].hue);
        if (dh > 180) dh = 360 - dh;
        if (dh < threshold) { needs.add(i); needs.add(j); }
      }
    }
    return needs;
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
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode="threading",
        ping_timeout=60,
        ping_interval=25,
        max_http_buffer_size=50 * 1024 * 1024,
    )

    ui_state = {"paused": False}

    @app.route("/")
    def index():
        return Response(HTML_PAGE, mimetype="text/html")

    @socketio.on("connect")
    def handle_connect():
        sid = flask_request.sid

        def send_init():
            for _ in range(50):
                with collector.lock:
                    if len(collector.ts_fast) > 5:
                        break
                time.sleep(0.1)
            data = collector.snapshot_full()
            socketio.emit("init", data, to=sid)

        threading.Thread(target=send_init, daemon=True).start()

    @socketio.on("request_init")
    def handle_request_init():
        data = collector.snapshot_full()
        socketio.emit("init", data, to=flask_request.sid)

    @socketio.on("pause")
    def handle_pause(is_paused):
        ui_state["paused"] = bool(is_paused)

    @socketio.on("view_minutes")
    def handle_view_minutes(vm):
        ui_state["view_minutes"] = int(vm)

    def push_loop():
        """Background thread that pushes deltas to all clients."""
        last_fast = 0
        last_pcie = 0
        last_slow = 0
        while True:
            vm = ui_state.get("view_minutes", 2)
            if vm <= 2:
                push_sec = args.push_interval / 1000.0
            elif vm <= 5:
                push_sec = 0.5
            elif vm <= 10:
                push_sec = 1.0
            else:
                push_sec = 2.0
            time.sleep(push_sec)
            with collector.lock:
                cf = collector.counter_fast
                cp = collector.counter_pcie
                cs = collector.counter_slow
            if cf <= last_fast and cp <= last_pcie and cs <= last_slow:
                continue
            try:
                if vm <= 2:
                    step = 1
                elif vm <= 5:
                    step = 4
                elif vm <= 10:
                    step = 8
                else:
                    step = 12
                delta = collector.snapshot_delta(
                    last_fast, last_pcie, last_slow, step=step
                )
                last_fast = cf
                last_pcie = cp
                last_slow = cs
                socketio.emit("delta", delta)
            except Exception as e:
                print(f"[push_loop] error: {e}", flush=True)
                raise

    return app, socketio, ui_state, push_loop


def main():
    args = parse_args()
    print(f"Fast collect : {args.interval}ms (CPU only)")
    print(f"PCIe collect : {args.pcie_interval}ms")
    print(f"Slow collect : {args.gpu_interval}ms (GPU mem, util, temp, disk, network)")
    print(f"Push interval: {args.push_interval}ms")
    print(
        f"Rolling window  : {args.window}s ({args.window // 60}m {args.window % 60}s)"
    )

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

    collector = MetricsCollector(
        window_sec=args.window,
        interval_ms=args.interval,
        pcie_interval_ms=args.pcie_interval,
        slow_interval_ms=args.gpu_interval,
        top_n=args.top_n,
        fill_patterns=args.fill_patterns,
    )
    psutil.cpu_percent(interval=None)

    app, socketio, ui_state, push_loop = build_server(collector, args)

    def _fast_loop():
        interval_sec = args.interval / 1000.0
        while True:
            collector.sample_fast()
            time.sleep(interval_sec)

    def _pcie_loop():
        interval_sec = args.pcie_interval / 1000.0
        while True:
            collector.sample_pcie()
            time.sleep(interval_sec)

    def _slow_loop():
        interval_sec = args.gpu_interval / 1000.0
        while True:
            collector.sample_slow()
            time.sleep(interval_sec)

    threading.Thread(target=_fast_loop, daemon=True).start()
    threading.Thread(target=_pcie_loop, daemon=True).start()
    threading.Thread(target=_slow_loop, daemon=True).start()
    threading.Thread(target=push_loop, daemon=True).start()

    print(f"Open http://{args.host}:{args.port} in your browser")
    socketserver.TCPServer.allow_reuse_address = True
    socketio.run(app, host=args.host, port=args.port, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
