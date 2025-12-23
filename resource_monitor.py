#!/usr/bin/env python3
"""
Resource monitor (single-instance) that periodically samples:
- System CPU / memory / load
- Network IO and disk IO (bytes + per-second rates)
- NVIDIA GPU utilization + memory (if nvidia-smi is available)
- "Top offender" processes (CPU%, RSS, IO rate, GPU memory)

All samples are appended to a lightweight SQLite database.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import signal
import socket
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import psutil

LOGGER = logging.getLogger("resource_monitor")


def _now_unix() -> float:
    return time.time()


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def _run_cmd(cmd: Sequence[str], timeout_s: float = 2.0) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return p.returncode, p.stdout, p.stderr
    except Exception as e:
        return 1, "", str(e)


@dataclass(frozen=True)
class Thresholds:
    cpu_percent: float
    rss_mb: float
    io_mbps: float
    gpu_mem_mb: float


class SingleInstanceLock:
    """Simple single-instance lock using fcntl.flock (Linux)."""

    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self._fd: Optional[int] = None

    def acquire_or_exit(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        self._fd = fd
        try:
            import fcntl  # Linux only

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.ftruncate(fd, 0)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            os.fsync(fd)
        except BlockingIOError:
            raise SystemExit(
                f"Another instance is already running (lock held): {self.lock_path}"
            )

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            import fcntl

            fcntl.flock(self._fd, fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            os.close(self._fd)
        except Exception:
            pass
        self._fd = None


class ResourceDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA foreign_keys=ON;")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS samples (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts_unix REAL NOT NULL,
              hostname TEXT NOT NULL,
              interval_s REAL NOT NULL,
              cpu_percent REAL,
              load1 REAL,
              load5 REAL,
              load15 REAL,
              mem_total_bytes INTEGER,
              mem_used_bytes INTEGER,
              mem_available_bytes INTEGER,
              mem_percent REAL,
              swap_used_bytes INTEGER,
              net_bytes_sent INTEGER,
              net_bytes_recv INTEGER,
              net_sent_bps REAL,
              net_recv_bps REAL,
              disk_read_bytes INTEGER,
              disk_write_bytes INTEGER,
              disk_read_bps REAL,
              disk_write_bps REAL,
              extra_json TEXT
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS gpu_samples (
              sample_id INTEGER NOT NULL,
              gpu_index INTEGER NOT NULL,
              name TEXT,
              util_gpu REAL,
              mem_used_mb REAL,
              mem_total_mb REAL,
              temp_c REAL,
              power_w REAL,
              PRIMARY KEY (sample_id, gpu_index),
              FOREIGN KEY(sample_id) REFERENCES samples(id) ON DELETE CASCADE
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS process_samples (
              sample_id INTEGER NOT NULL,
              pid INTEGER NOT NULL,
              name TEXT,
              username TEXT,
              cmdline TEXT,
              cpu_percent REAL,
              rss_bytes INTEGER,
              vms_bytes INTEGER,
              io_read_bytes INTEGER,
              io_write_bytes INTEGER,
              io_read_bps REAL,
              io_write_bps REAL,
              gpu_mem_mb REAL,
              PRIMARY KEY (sample_id, pid),
              FOREIGN KEY(sample_id) REFERENCES samples(id) ON DELETE CASCADE
            );
            """
        )

        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_samples_ts ON samples(ts_unix);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_proc_sample ON process_samples(sample_id);"
        )
        self.conn.commit()

    def insert_sample(
        self,
        *,
        ts_unix: float,
        hostname: str,
        interval_s: float,
        cpu_percent: Optional[float],
        load: Tuple[Optional[float], Optional[float], Optional[float]],
        mem: psutil._common.svmem,
        swap: psutil._common.sswap,
        net_totals: psutil._common.snetio,
        net_rates: Tuple[Optional[float], Optional[float]],
        disk_totals: Optional[psutil._common.sdiskio],
        disk_rates: Tuple[Optional[float], Optional[float]],
        extra: Dict[str, Any],
    ) -> int:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO samples(
              ts_unix, hostname, interval_s,
              cpu_percent, load1, load5, load15,
              mem_total_bytes, mem_used_bytes, mem_available_bytes, mem_percent, swap_used_bytes,
              net_bytes_sent, net_bytes_recv, net_sent_bps, net_recv_bps,
              disk_read_bytes, disk_write_bytes, disk_read_bps, disk_write_bps,
              extra_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts_unix,
                hostname,
                interval_s,
                cpu_percent,
                load[0],
                load[1],
                load[2],
                int(mem.total),
                int(mem.used),
                int(mem.available),
                float(mem.percent),
                int(swap.used),
                int(net_totals.bytes_sent),
                int(net_totals.bytes_recv),
                net_rates[0],
                net_rates[1],
                int(disk_totals.read_bytes) if disk_totals else None,
                int(disk_totals.write_bytes) if disk_totals else None,
                disk_rates[0],
                disk_rates[1],
                _safe_json(extra),
            ),
        )
        sample_id = int(cur.lastrowid)
        return sample_id

    def insert_gpu_samples(self, sample_id: int, gpus: List[Dict[str, Any]]) -> None:
        if not gpus:
            return
        cur = self.conn.cursor()
        for g in gpus:
            cur.execute(
                """
                INSERT OR REPLACE INTO gpu_samples(
                  sample_id, gpu_index, name, util_gpu, mem_used_mb, mem_total_mb, temp_c, power_w
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sample_id,
                    int(g.get("index", -1)),
                    g.get("name"),
                    g.get("util_gpu"),
                    g.get("mem_used_mb"),
                    g.get("mem_total_mb"),
                    g.get("temp_c"),
                    g.get("power_w"),
                ),
            )

    def insert_process_samples(self, sample_id: int, procs: List[Dict[str, Any]]) -> None:
        if not procs:
            return
        cur = self.conn.cursor()
        for p in procs:
            cur.execute(
                """
                INSERT OR REPLACE INTO process_samples(
                  sample_id, pid, name, username, cmdline,
                  cpu_percent, rss_bytes, vms_bytes,
                  io_read_bytes, io_write_bytes, io_read_bps, io_write_bps,
                  gpu_mem_mb
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sample_id,
                    int(p["pid"]),
                    p.get("name"),
                    p.get("username"),
                    p.get("cmdline"),
                    p.get("cpu_percent"),
                    p.get("rss_bytes"),
                    p.get("vms_bytes"),
                    p.get("io_read_bytes"),
                    p.get("io_write_bytes"),
                    p.get("io_read_bps"),
                    p.get("io_write_bps"),
                    p.get("gpu_mem_mb"),
                ),
            )


def _get_loadavg() -> Tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        return os.getloadavg()
    except Exception:
        return None, None, None


def _get_disk_io() -> Optional[psutil._common.sdiskio]:
    try:
        return psutil.disk_io_counters()
    except Exception:
        return None


def _parse_nvidia_smi_csv(stdout: str) -> List[List[str]]:
    rows: List[List[str]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([c.strip() for c in line.split(",")])
    return rows


def _get_gpu_metrics() -> Tuple[List[Dict[str, Any]], Dict[int, float]]:
    """
    Returns:
      - list of per-gpu dicts
      - map pid -> gpu_mem_mb (aggregate across GPUs if multiple entries)
    """
    gpus: List[Dict[str, Any]] = []
    pid_to_gpu_mem: Dict[int, float] = {}

    # Per-GPU stats
    rc, out, _err = _run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=2.5,
    )
    if rc == 0 and out.strip():
        for row in _parse_nvidia_smi_csv(out):
            try:
                gpus.append(
                    {
                        "index": int(row[0]),
                        "name": row[1],
                        "util_gpu": float(row[2]) if row[2] != "" else None,
                        "mem_used_mb": float(row[3]) if row[3] != "" else None,
                        "mem_total_mb": float(row[4]) if row[4] != "" else None,
                        "temp_c": float(row[5]) if row[5] != "" else None,
                        "power_w": float(row[6]) if row[6] != "" else None,
                    }
                )
            except Exception:
                continue

    # Per-process GPU memory (best-effort; utilization per process is not available via nvidia-smi)
    rc2, out2, _err2 = _run_cmd(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=2.5,
    )
    if rc2 == 0 and out2.strip():
        for row in _parse_nvidia_smi_csv(out2):
            try:
                pid = int(row[0])
                mem_mb = float(row[1]) if row[1] != "" else 0.0
                pid_to_gpu_mem[pid] = pid_to_gpu_mem.get(pid, 0.0) + mem_mb
            except Exception:
                continue

    return gpus, pid_to_gpu_mem


def _format_cmdline(cmdline: Sequence[str]) -> str:
    if not cmdline:
        return ""
    try:
        return " ".join(shlex.quote(p) for p in cmdline)
    except Exception:
        return " ".join(cmdline)


def _collect_processes(
    *,
    dt_s: float,
    pid_to_gpu_mem: Dict[int, float],
    prev_io: Dict[int, Tuple[int, int]],
    thresholds: Thresholds,
    top_k: int,
) -> Tuple[List[Dict[str, Any]], Dict[int, Tuple[int, int]]]:
    """
    Collect candidate "offender" processes. Returns:
      - list of process dicts to store
      - updated prev_io map
    """
    procs: List[Dict[str, Any]] = []
    next_prev_io: Dict[int, Tuple[int, int]] = {}

    # Iterate once and gather basics
    for p in psutil.process_iter(attrs=["pid", "name", "username", "cmdline"]):
        try:
            pid = int(p.info["pid"])
            name = p.info.get("name")
            username = p.info.get("username")
            cmdline = _format_cmdline(p.info.get("cmdline") or [])

            cpu_pct = float(p.cpu_percent(None))  # since last call for this process
            mem = p.memory_info()
            rss = int(mem.rss)
            vms = int(mem.vms)

            io_read_bytes = None
            io_write_bytes = None
            io_read_bps = None
            io_write_bps = None
            try:
                io = p.io_counters()
                io_read_bytes = int(io.read_bytes)
                io_write_bytes = int(io.write_bytes)
                prev = prev_io.get(pid)
                if prev and dt_s > 0:
                    io_read_bps = (io_read_bytes - prev[0]) / dt_s
                    io_write_bps = (io_write_bytes - prev[1]) / dt_s
                next_prev_io[pid] = (io_read_bytes, io_write_bytes)
            except Exception:
                pass

            gpu_mem_mb = pid_to_gpu_mem.get(pid)

            procs.append(
                {
                    "pid": pid,
                    "name": name,
                    "username": username,
                    "cmdline": cmdline,
                    "cpu_percent": cpu_pct,
                    "rss_bytes": rss,
                    "vms_bytes": vms,
                    "io_read_bytes": io_read_bytes,
                    "io_write_bytes": io_write_bytes,
                    "io_read_bps": io_read_bps,
                    "io_write_bps": io_write_bps,
                    "gpu_mem_mb": gpu_mem_mb,
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        except Exception:
            continue

    # Pick top offenders across metrics
    def top_by(key: str, k: int) -> List[Dict[str, Any]]:
        return sorted(
            procs,
            key=lambda x: (x.get(key) is not None, x.get(key) or 0),
            reverse=True,
        )[:k]

    top_cpu = top_by("cpu_percent", top_k)
    top_mem = top_by("rss_bytes", top_k)
    top_gpu = top_by("gpu_mem_mb", top_k)
    top_io = sorted(
        procs,
        key=lambda x: (
            (x.get("io_read_bps") is not None or x.get("io_write_bps") is not None),
            (x.get("io_read_bps") or 0) + (x.get("io_write_bps") or 0),
        ),
        reverse=True,
    )[:top_k]

    chosen: Dict[int, Dict[str, Any]] = {}
    for p in top_cpu + top_mem + top_gpu + top_io:
        chosen[int(p["pid"])] = p

    # Also include anything above thresholds
    rss_thresh_bytes = thresholds.rss_mb * 1024 * 1024
    io_thresh_bps = thresholds.io_mbps * 1024 * 1024
    for p in procs:
        cpu_ok = (p.get("cpu_percent") or 0.0) >= thresholds.cpu_percent
        mem_ok = (p.get("rss_bytes") or 0) >= rss_thresh_bytes
        io_ok = ((p.get("io_read_bps") or 0.0) + (p.get("io_write_bps") or 0.0)) >= io_thresh_bps
        gpu_ok = (p.get("gpu_mem_mb") or 0.0) >= thresholds.gpu_mem_mb
        if cpu_ok or mem_ok or io_ok or gpu_ok:
            chosen[int(p["pid"])] = p

    return list(chosen.values()), next_prev_io


def _ensure_cpu_percent_primed() -> None:
    # Prime system-wide cpu_percent and per-process cpu_percent, so next call has meaning.
    try:
        psutil.cpu_percent(None)
    except Exception:
        pass
    for p in psutil.process_iter():
        try:
            p.cpu_percent(None)
        except Exception:
            continue


class Monitor:
    def __init__(
        self,
        *,
        db: ResourceDB,
        hostname: str,
        interval_s: float,
        thresholds: Thresholds,
        top_k: int,
        once: bool,
        duration_s: Optional[float],
    ):
        self.db = db
        self.hostname = hostname
        self.interval_s = interval_s
        self.thresholds = thresholds
        self.top_k = top_k
        self.once = once
        self.duration_s = duration_s
        self._stop = False

        self._prev_net: Optional[psutil._common.snetio] = None
        self._prev_disk: Optional[psutil._common.sdiskio] = None
        self._prev_ts: Optional[float] = None
        self._prev_proc_io: Dict[int, Tuple[int, int]] = {}

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> int:
        _ensure_cpu_percent_primed()
        start_ts = _now_unix()

        while True:
            if self._stop:
                return 0
            if self.duration_s is not None and (_now_unix() - start_ts) >= self.duration_s:
                return 0

            ts = _now_unix()
            dt_s = self.interval_s
            if self._prev_ts is not None:
                dt_s = max(0.001, ts - self._prev_ts)

            # System metrics
            cpu_percent = None
            try:
                cpu_percent = float(psutil.cpu_percent(None))
            except Exception:
                pass

            load = _get_loadavg()
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()

            net = psutil.net_io_counters()
            net_sent_bps = None
            net_recv_bps = None
            if self._prev_net is not None:
                net_sent_bps = (net.bytes_sent - self._prev_net.bytes_sent) / dt_s
                net_recv_bps = (net.bytes_recv - self._prev_net.bytes_recv) / dt_s

            disk = _get_disk_io()
            disk_read_bps = None
            disk_write_bps = None
            if disk is not None and self._prev_disk is not None:
                disk_read_bps = (disk.read_bytes - self._prev_disk.read_bytes) / dt_s
                disk_write_bps = (disk.write_bytes - self._prev_disk.write_bytes) / dt_s

            gpus, pid_to_gpu_mem = _get_gpu_metrics()

            offenders, self._prev_proc_io = _collect_processes(
                dt_s=dt_s,
                pid_to_gpu_mem=pid_to_gpu_mem,
                prev_io=self._prev_proc_io,
                thresholds=self.thresholds,
                top_k=self.top_k,
            )

            # Persist
            extra = {
                "pid": os.getpid(),
                "python": os.environ.get("VIRTUAL_ENV") or "",
                "uname": " ".join(os.uname()) if hasattr(os, "uname") else "",
                "gpu_present": bool(gpus),
            }

            cur = self.db.conn.cursor()
            try:
                cur.execute("BEGIN;")
                sample_id = self.db.insert_sample(
                    ts_unix=ts,
                    hostname=self.hostname,
                    interval_s=float(self.interval_s),
                    cpu_percent=cpu_percent,
                    load=load,
                    mem=mem,
                    swap=swap,
                    net_totals=net,
                    net_rates=(net_sent_bps, net_recv_bps),
                    disk_totals=disk,
                    disk_rates=(disk_read_bps, disk_write_bps),
                    extra=extra,
                )
                self.db.insert_gpu_samples(sample_id, gpus)
                self.db.insert_process_samples(sample_id, offenders)
                self.db.conn.commit()
            except Exception as e:
                self.db.conn.rollback()
                LOGGER.exception("Failed to write sample: %s", e)

            # Update prev counters
            self._prev_ts = ts
            self._prev_net = net
            self._prev_disk = disk

            LOGGER.info(
                "sample: cpu=%.1f%% mem=%.1f%% net=%.1f/%.1f MB/s disk=%.1f/%.1f MB/s offenders=%d gpus=%d",
                (cpu_percent or 0.0),
                float(mem.percent),
                ((net_sent_bps or 0.0) / (1024 * 1024)),
                ((net_recv_bps or 0.0) / (1024 * 1024)),
                ((disk_read_bps or 0.0) / (1024 * 1024)),
                ((disk_write_bps or 0.0) / (1024 * 1024)),
                len(offenders),
                len(gpus),
            )

            if self.once:
                return 0

            # Sleep (interruptible)
            end_sleep = time.monotonic() + self.interval_s
            while time.monotonic() < end_sleep:
                if self._stop:
                    return 0
                time.sleep(0.2)


def _default_db_path() -> Path:
    # Always use a stable per-user location by default.
    return Path.home() / ".cache" / "dynamo-utils" / "resource_monitor.sqlite"


def _default_lock_path(db_path: Path) -> Path:
    return db_path.with_suffix(".lock")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-instance resource monitor -> SQLite")
    p.add_argument("--db-path", type=Path, default=_default_db_path(), help="SQLite DB path")
    p.add_argument("--lock-path", type=Path, default=None, help="Lock file path (default: <db>.lock)")
    p.add_argument("--interval-seconds", type=float, default=15.0, help="Sampling interval (seconds)")
    p.add_argument("--once", action="store_true", help="Take one sample and exit")
    p.add_argument("--duration-seconds", type=float, default=None, help="Run for N seconds then exit")

    p.add_argument("--top-k", type=int, default=12, help="Top-k processes per metric to store")
    p.add_argument("--cpu-threshold", type=float, default=50.0, help="Record any process >= this CPU%%")
    p.add_argument("--rss-threshold-mb", type=float, default=2048.0, help="Record any process >= this RSS (MB)")
    p.add_argument("--io-threshold-mbps", type=float, default=50.0, help="Record any process >= this IO rate (MB/s)")
    p.add_argument("--gpu-mem-threshold-mb", type=float, default=1024.0, help="Record any process >= this GPU mem (MB)")

    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    db_path: Path = args.db_path
    lock_path: Path = args.lock_path or _default_lock_path(db_path)

    lock = SingleInstanceLock(lock_path)
    lock.acquire_or_exit()

    db = ResourceDB(db_path)
    hostname = socket.gethostname()

    thresholds = Thresholds(
        cpu_percent=float(args.cpu_threshold),
        rss_mb=float(args.rss_threshold_mb),
        io_mbps=float(args.io_threshold_mbps),
        gpu_mem_mb=float(args.gpu_mem_threshold_mb),
    )

    monitor = Monitor(
        db=db,
        hostname=hostname,
        interval_s=float(args.interval_seconds),
        thresholds=thresholds,
        top_k=int(args.top_k),
        once=bool(args.once),
        duration_s=args.duration_seconds,
    )

    def _handle_sig(_signum, _frame) -> None:
        LOGGER.warning("signal received; stopping...")
        monitor.request_stop()

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    try:
        LOGGER.info("DB: %s", db_path)
        LOGGER.info("Lock: %s", lock_path)
        return monitor.run()
    finally:
        try:
            db.close()
        finally:
            lock.release()


if __name__ == "__main__":
    raise SystemExit(main())


