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
import re
import shlex
import shutil
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

# Optional per-process network bandwidth attribution (host)
# --------------------------------------------------------
# This monitor already records system-wide network totals, but those cannot be attributed to a PID.
# If you want "who used the bandwidth", you need an external collector. Two practical options:
#
# 1) nethogs (pcap-based, works without eBPF)
#    sudo apt-get update -y
#    sudo apt-get install -y nethogs
#    NOTE: requires root (or CAP_NET_RAW/CAP_NET_ADMIN). The script will skip collection if it can't run.
#
# 2) BCC/eBPF tools (more accurate/cheaper, but can fail if headers/toolchain/kernel don't match)
#    sudo apt-get update -y
#    sudo apt-get install -y bpfcc-tools linux-headers-$(uname -r)
#    Example tools: /usr/sbin/tcptop-bpfcc, /usr/sbin/tcpconnect-bpfcc
#    NOTE: on some hosts these may fail to compile BPF programs. The script remains tolerant.
#
# Enable via CLI: --net-top (defaults off).

DEFAULT_PING_TARGETS: Tuple[str, ...] = (
    "www.yahoo.com",
    "www.google.com",
    "10.110.41.1",
    "10.110.40.1",
)

_PING_TIME_RE = re.compile(r"time[=<]\s*([0-9.]+)\s*ms")


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


def _run_cmd_maybe_sudo(
    cmd: Sequence[str],
    *,
    timeout_s: float,
    allow_sudo: bool = True,
) -> Tuple[int, str, str]:
    """
    Run a command; if it fails due to permissions and sudo is available, retry with `sudo -n`.

    This keeps the monitor usable when run as a normal user but the underlying tooling
    (docker stats / nethogs) requires elevated privileges.
    """
    rc, out, err = _run_cmd(cmd, timeout_s=timeout_s)
    if rc == 0 or not allow_sudo:
        return rc, out, err
    if not shutil.which("sudo"):
        return rc, out, err
    # Special-case: nethogs often fails as non-root with pcap handler errors (not always "permission denied").
    # Since we already require `sudo -n` (non-interactive), it's safe to retry on ANY failure for nethogs.
    try:
        exe = str(cmd[0]) if cmd else ""
        base = os.path.basename(exe)
        if base == "nethogs":
            sudo_cmd = ["sudo", "-n", *list(cmd)]
            return _run_cmd(sudo_cmd, timeout_s=timeout_s)
    except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
        pass
    # Common permission-denied patterns
    msg = f"{out}\n{err}".lower()
    if "permission denied" not in msg and "got permission denied" not in msg and "operation not permitted" not in msg:
        return rc, out, err
    sudo_cmd = ["sudo", "-n", *list(cmd)]
    return _run_cmd(sudo_cmd, timeout_s=timeout_s)

def _ping_rtt_ms(target: str, *, timeout_s: float) -> Tuple[Optional[float], Optional[str]]:
    """
    Best-effort ICMP ping RTT measurement.

    Returns:
      (rtt_ms, error) where rtt_ms is float on success; error is a short string on failure.
    """
    ping = shutil.which("ping")
    if not ping:
        return None, "ping_not_found"

    # Linux iputils ping:
    # -c 1: one probe
    # -n: numeric output (avoid reverse DNS delays)
    # -W <sec>: per-probe timeout (seconds)
    # -w <sec>: overall deadline (seconds)
    w = max(1, int(timeout_s))
    cmd = [ping, "-n", "-c", "1", "-W", str(w), "-w", str(w), str(target)]
    rc, out, err = _run_cmd(cmd, timeout_s=float(timeout_s) + 0.5)
    if rc == 0:
        m = _PING_TIME_RE.search(out)
        if m:
            try:
                return float(m.group(1)), None
            except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
                return None, "parse_error"
        return None, "parse_missing_time"

    msg = (err or out).strip()
    if msg:
        msg = msg.replace("\n", " ")[:200]
    else:
        msg = f"rc={rc}"
    return None, msg


def _collect_ping_samples(targets: Sequence[str], *, timeout_s: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for t in targets:
        rtt_ms, err = _ping_rtt_ms(t, timeout_s=timeout_s)
        rows.append(
            {
                "target": str(t),
                "rtt_ms": rtt_ms,
                "success": 1 if rtt_ms is not None else 0,
                "error": err,
            }
        )
    return rows


def _collect_net_top_nethogs(*, top_k: int, timeout_s: float) -> List[Dict[str, Any]]:
    """
    Best-effort per-process network throughput using nethogs trace mode.

    Returns a list of rows with:
      pid (int), proc (str), sent_bps (float), recv_bps (float), tool (str)
    """
    exe = shutil.which("nethogs")
    if not exe:
        return []

    # Trace mode with one refresh. Default view mode is KB/s.
    # -t: trace mode (stdout)
    # -c 1: one update then exit
    # -d 1: 1s refresh window
    # -C: capture TCP+UDP
    # -b: short program name (less noise)
    cmd = [exe, "-t", "-c", "1", "-d", "1", "-C", "-b"]
    rc, out, err = _run_cmd_maybe_sudo(cmd, timeout_s=timeout_s, allow_sudo=True)
    if rc != 0 or not out.strip():
        # Don't spam logs; callers may choose to log at DEBUG.
        return []

    lines = out.splitlines()
    # Only parse the last "Refreshing:" block.
    start = 0
    for i, ln in enumerate(lines):
        if ln.strip().startswith("Refreshing"):
            start = i + 1

    rows: List[Dict[str, Any]] = []
    for ln in lines[start:]:
        if "\t" not in ln:
            continue
        parts = [p.strip() for p in ln.split("\t") if p.strip() != ""]
        if len(parts) < 3:
            continue
        proc = parts[0]
        try:
            sent_kbs = float(parts[1])
            recv_kbs = float(parts[2])
        except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
            continue

        # nethogs often reports proc as "name/PID/UID" (or "unknown TCP/0/0").
        pid = 0
        try:
            segs = proc.split("/")
            if len(segs) >= 2 and segs[1].isdigit():
                pid = int(segs[1])
        except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
            pid = 0

        rows.append(
            {
                "pid": pid,
                "proc": proc,
                "sent_bps": float(sent_kbs) * 1024.0,
                "recv_bps": float(recv_kbs) * 1024.0,
                "tool": "nethogs",
            }
        )

    rows.sort(key=lambda r: float(r.get("sent_bps", 0.0)) + float(r.get("recv_bps", 0.0)), reverse=True)
    return rows[: max(0, int(top_k))]


def _collect_github_rate_limit(
    *,
    resources: Sequence[str],
    timeout_s: float,
) -> List[Dict[str, Any]]:
    """
    Best-effort GitHub rate limit sampling via `gh api rate_limit`.

    Returns rows like:
      {resource, limit, remaining, used, reset_epoch}
    """
    exe = shutil.which("gh")
    if not exe:
        return []
    rc, out, _err = _run_cmd([exe, "api", "rate_limit"], timeout_s=timeout_s)
    if rc != 0 or not out.strip():
        return []
    try:
        payload = json.loads(out)
    except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
        return []
    res = payload.get("resources")
    if not isinstance(res, dict):
        return []

    out_rows: List[Dict[str, Any]] = []
    for name in resources:
        info = res.get(name)
        if not isinstance(info, dict):
            continue
        try:
            limit = info.get("limit")
            remaining = info.get("remaining")
            used = info.get("used")
            reset_epoch = info.get("reset")
            out_rows.append(
                {
                    "resource": str(name),
                    "limit": int(limit) if limit is not None else None,
                    "remaining": int(remaining) if remaining is not None else None,
                    "used": int(used) if used is not None else None,
                    "reset_epoch": int(reset_epoch) if reset_epoch is not None else None,
                }
            )
        except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
            continue
    return out_rows


def _parse_bytes(s: str) -> Optional[int]:
    """
    Parse Docker-ish byte strings like: '0B', '12.3kB', '4.1MB', '1.2GiB', '512KiB'.
    Returns integer bytes or None if parsing fails.
    """
    if not s:
        return None
    t = str(s).strip()
    if not t:
        return None
    if t.lower() == "0b":
        return 0
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]+)\s*$", t)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)

    # IEC (base-1024)
    iec = {
        "B": 1,
        "KiB": 1024,
        "MiB": 1024**2,
        "GiB": 1024**3,
        "TiB": 1024**4,
        "PiB": 1024**5,
    }
    if unit in iec:
        return int(val * iec[unit])

    # SI (base-1000) - docker often uses kB/MB/GB for IO.
    si = {
        "b": 1,
        "B": 1,
        "kB": 1000,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "TB": 1000**4,
        "PB": 1000**5,
    }
    if unit in si:
        return int(val * si[unit])
    return None


def _parse_pair_bytes(s: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse 'X / Y' style strings into bytes.
    """
    if not s:
        return None, None
    parts = [p.strip() for p in str(s).split("/") if p.strip()]
    if len(parts) != 2:
        return None, None
    return _parse_bytes(parts[0]), _parse_bytes(parts[1])


def _collect_docker_stats(*, timeout_s: float) -> List[Dict[str, Any]]:
    """
    Best-effort per-container CPU/memory stats using `docker stats --no-stream`.

    Returns rows:
      {container_id, name, cpu_percent, mem_usage_bytes, mem_limit_bytes, mem_percent, pids,
       net_rx_bytes, net_tx_bytes, block_read_bytes, block_write_bytes}

    Tolerant of missing docker, permissions, or parse errors (returns []).
    """
    exe = shutil.which("docker")
    if not exe:
        return []
    # Map container ID -> image name (docker stats doesn't expose Image)
    id_to_image: Dict[str, str] = {}
    try:
        rc_ps, out_ps, _err_ps = _run_cmd_maybe_sudo(
            [exe, "ps", "--format", "{{.ID}}\t{{.Image}}"],
            timeout_s=min(2.0, timeout_s),
            allow_sudo=True,
        )
        if rc_ps == 0 and out_ps.strip():
            for ln in out_ps.splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split("\t", 1)
                if len(parts) != 2:
                    continue
                cid, img = parts[0].strip(), parts[1].strip()
                if cid:
                    id_to_image[cid] = img
    except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
        pass

    cmd = [exe, "stats", "--no-stream", "--format", "{{json .}}"]
    rc, out, _err = _run_cmd_maybe_sudo(cmd, timeout_s=timeout_s, allow_sudo=True)
    if rc != 0 or not out.strip():
        return []

    rows: List[Dict[str, Any]] = []
    for ln in out.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            d = json.loads(ln)
        except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
            continue

        cid = str(d.get("ID") or "").strip()
        name = str(d.get("Name") or "").strip()
        if not cid and not name:
            continue
        image = id_to_image.get(cid, "")

        cpu_percent = None
        try:
            cpu_percent = float(str(d.get("CPUPerc") or "").strip().rstrip("%"))
        except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
            cpu_percent = None

        mem_usage_bytes, mem_limit_bytes = _parse_pair_bytes(str(d.get("MemUsage") or ""))
        mem_percent = None
        try:
            mem_percent = float(str(d.get("MemPerc") or "").strip().rstrip("%"))
        except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
            mem_percent = None

        net_rx_bytes, net_tx_bytes = _parse_pair_bytes(str(d.get("NetIO") or ""))
        blk_read_bytes, blk_write_bytes = _parse_pair_bytes(str(d.get("BlockIO") or ""))

        pids = None
        try:
            pids = int(str(d.get("PIDs") or "").strip())
        except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
            pids = None

        rows.append(
            {
                "container_id": cid,
                "name": name,
                "image": image,
                "cpu_percent": cpu_percent,
                "mem_usage_bytes": mem_usage_bytes,
                "mem_limit_bytes": mem_limit_bytes,
                "mem_percent": mem_percent,
                "pids": pids,
                "net_rx_bytes": net_rx_bytes,
                "net_tx_bytes": net_tx_bytes,
                "block_read_bytes": blk_read_bytes,
                "block_write_bytes": blk_write_bytes,
            }
        )
    return rows

@dataclass(frozen=True)
class Thresholds:
    cpu_percent: float
    rss_mb: float
    io_mbps: float
    gpu_mem_mb: float


class SingleInstanceLock:
    """
    Single-instance lock using fcntl.flock (Linux) plus a PID guard written into the lock file.

    Behavior:
    - Normal start: takes an exclusive non-blocking flock on the lock file and writes our PID into it.
    - While running: if the PID stored in the lock file changes (someone "force started" another instance),
      the current process notices and exits gracefully.
    - Force start: overwrites the PID in the lock file first (to signal any existing instance to exit),
      then waits to acquire the flock.
    """

    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self._fd: Optional[int] = None
        self._pid: int = int(os.getpid())

    def _read_pid(self) -> Optional[int]:
        try:
            raw = self.lock_path.read_text(encoding="utf-8").strip()
            if not raw:
                return None
            return int(raw.splitlines()[0].strip())
        except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
            return None

    def _write_pid(self, pid: int) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        # Write via low-level fd to avoid partial writes.
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            os.ftruncate(fd, 0)
            os.write(fd, str(int(pid)).encode("utf-8"))
            os.fsync(fd)
        finally:
            try:
                os.close(fd)
            except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
                pass

    def pid_changed(self) -> bool:
        """Return True if the PID stored in the lock file differs from our PID."""
        pid = self._read_pid()
        return pid is not None and int(pid) != int(self._pid)

    def acquire_or_exit(
        self,
        *,
        force_start: bool = False,
        force_timeout_s: float = 10.0,
        force_poll_s: float = 0.2,
    ) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        self._fd = fd
        try:
            import fcntl  # Linux only

            if force_start:
                # Signal any existing instance to exit by clobbering the PID in the lock file.
                # Note: this does NOT break an existing flock; we still wait until we can acquire it.
                try:
                    self._write_pid(self._pid)
                except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
                    pass

                deadline = time.monotonic() + max(0.1, float(force_timeout_s))
                while True:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise SystemExit(
                                f"Force-start timed out waiting for lock: {self.lock_path}"
                            )
                        time.sleep(max(0.05, float(force_poll_s)))
            else:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

            # Record our PID in the lock file (also acts as the "PID guard" checked by the running loop).
            os.ftruncate(fd, 0)
            os.write(fd, str(self._pid).encode("utf-8"))
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
        except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
            pass
        try:
            os.close(self._fd)
        except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
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
        except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
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
            """
            CREATE TABLE IF NOT EXISTS ping_samples (
              sample_id INTEGER NOT NULL,
              target TEXT NOT NULL,
              rtt_ms REAL,
              success INTEGER NOT NULL,
              error TEXT,
              PRIMARY KEY (sample_id, target),
              FOREIGN KEY(sample_id) REFERENCES samples(id) ON DELETE CASCADE
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS net_process_samples (
              sample_id INTEGER NOT NULL,
              pid INTEGER NOT NULL,
              proc TEXT NOT NULL,
              sent_bps REAL,
              recv_bps REAL,
              tool TEXT,
              PRIMARY KEY (sample_id, pid, proc),
              FOREIGN KEY(sample_id) REFERENCES samples(id) ON DELETE CASCADE
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS docker_container_samples (
              sample_id INTEGER NOT NULL,
              container_id TEXT NOT NULL,
              name TEXT,
              image TEXT,
              cpu_percent REAL,
              mem_usage_bytes INTEGER,
              mem_limit_bytes INTEGER,
              mem_percent REAL,
              pids INTEGER,
              net_rx_bytes INTEGER,
              net_tx_bytes INTEGER,
              block_read_bytes INTEGER,
              block_write_bytes INTEGER,
              PRIMARY KEY (sample_id, container_id),
              FOREIGN KEY(sample_id) REFERENCES samples(id) ON DELETE CASCADE
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS gh_rate_limit_samples (
              sample_id INTEGER NOT NULL,
              resource TEXT NOT NULL,
              "limit" INTEGER,
              remaining INTEGER,
              used INTEGER,
              reset_epoch INTEGER,
              PRIMARY KEY (sample_id, resource),
              FOREIGN KEY(sample_id) REFERENCES samples(id) ON DELETE CASCADE
            );
            """
        )

        # Backwards-compatible migration for older DBs created before `image` existed.
        try:
            cols = [r[1] for r in cur.execute("PRAGMA table_info(docker_container_samples);").fetchall()]
            if "image" not in cols:
                cur.execute("ALTER TABLE docker_container_samples ADD COLUMN image TEXT;")
        except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
            pass

        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_samples_ts ON samples(ts_unix);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_proc_sample ON process_samples(sample_id);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_ping_target ON ping_samples(target);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_net_proc ON net_process_samples(proc);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_docker_name ON docker_container_samples(name);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_gh_rate_limit_resource ON gh_rate_limit_samples(resource);"
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
        # psutil's public API returns namedtuples; avoid depending on psutil._common types.
        # mem/swap are optional so callers can downsample memory writes (store NULLs between samples).
        mem: Optional[Any],
        swap: Optional[Any],
        net_totals: Any,
        net_rates: Tuple[Optional[float], Optional[float]],
        disk_totals: Optional[Any],
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
                (int(mem.total) if mem is not None else None),
                (int(mem.used) if mem is not None else None),
                (int(mem.available) if mem is not None else None),
                (float(mem.percent) if mem is not None else None),
                (int(swap.used) if swap is not None else None),
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
        lastrowid = cur.lastrowid
        if lastrowid is None:
            raise RuntimeError("Failed to insert sample row (no lastrowid).")
        sample_id = int(lastrowid)
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

    def insert_ping_samples(self, sample_id: int, pings: List[Dict[str, Any]]) -> None:
        if not pings:
            return
        cur = self.conn.cursor()
        for p in pings:
            cur.execute(
                """
                INSERT OR REPLACE INTO ping_samples(
                  sample_id, target, rtt_ms, success, error
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    sample_id,
                    str(p.get("target") or ""),
                    p.get("rtt_ms"),
                    int(p.get("success") or 0),
                    p.get("error"),
                ),
            )

    def insert_net_process_samples(self, sample_id: int, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        cur = self.conn.cursor()
        for r in rows:
            cur.execute(
                """
                INSERT OR REPLACE INTO net_process_samples(
                  sample_id, pid, proc, sent_bps, recv_bps, tool
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    sample_id,
                    int(r.get("pid") or 0),
                    str(r.get("proc") or ""),
                    r.get("sent_bps"),
                    r.get("recv_bps"),
                    r.get("tool"),
                ),
            )

    def insert_docker_container_samples(self, sample_id: int, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        cur = self.conn.cursor()
        for r in rows:
            cid = str(r.get("container_id") or "")
            if not cid:
                continue
            cur.execute(
                """
                INSERT OR REPLACE INTO docker_container_samples(
                  sample_id, container_id, name, image,
                  cpu_percent, mem_usage_bytes, mem_limit_bytes, mem_percent, pids,
                  net_rx_bytes, net_tx_bytes, block_read_bytes, block_write_bytes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sample_id,
                    cid,
                    r.get("name"),
                    r.get("image"),
                    r.get("cpu_percent"),
                    r.get("mem_usage_bytes"),
                    r.get("mem_limit_bytes"),
                    r.get("mem_percent"),
                    r.get("pids"),
                    r.get("net_rx_bytes"),
                    r.get("net_tx_bytes"),
                    r.get("block_read_bytes"),
                    r.get("block_write_bytes"),
                ),
            )

    def insert_gh_rate_limit_samples(self, sample_id: int, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        cur = self.conn.cursor()
        for r in rows:
            cur.execute(
                """
                INSERT OR REPLACE INTO gh_rate_limit_samples(
                  sample_id, resource, "limit", remaining, used, reset_epoch
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    sample_id,
                    str(r.get("resource") or ""),
                    int(r.get("limit")) if r.get("limit") is not None else None,
                    int(r.get("remaining")) if r.get("remaining") is not None else None,
                    int(r.get("used")) if r.get("used") is not None else None,
                    int(r.get("reset_epoch")) if r.get("reset_epoch") is not None else None,
                ),
            )


def _get_loadavg() -> Tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        return os.getloadavg()
    except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
        return None, None, None


def _get_disk_io() -> Optional[psutil._common.sdiskio]:
    try:
        return psutil.disk_io_counters()
    except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
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
            except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
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
            except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
                continue

    return gpus, pid_to_gpu_mem


def _format_cmdline(cmdline: Sequence[str]) -> str:
    if not cmdline:
        return ""
    try:
        return " ".join(shlex.quote(p) for p in cmdline)
    except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
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
            except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
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
        except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
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
    except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
        pass
    for p in psutil.process_iter():
        try:
            p.cpu_percent(None)
        except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
            continue


class Monitor:
    def __init__(
        self,
        *,
        db: ResourceDB,
        lock: SingleInstanceLock,
        hostname: str,
        interval_s: float,
        mem_interval_s: float,
        thresholds: Thresholds,
        top_k: int,
        net_top: bool,
        net_top_interval_s: float,
        net_top_k: int,
        docker_stats: bool,
        docker_stats_interval_s: float,
        gh_rate_limit: bool,
        gh_rate_limit_interval_s: float,
        gh_rate_limit_resources: Sequence[str],
        once: bool,
        duration_s: Optional[float],
    ):
        self.db = db
        self.lock = lock
        self.hostname = hostname
        self.interval_s = interval_s
        self.mem_interval_s = float(mem_interval_s)
        self.thresholds = thresholds
        self.top_k = top_k
        self.net_top = net_top
        self.net_top_interval_s = float(net_top_interval_s)
        self.net_top_k = int(net_top_k)
        self.docker_stats = bool(docker_stats)
        self.docker_stats_interval_s = float(docker_stats_interval_s)
        self.gh_rate_limit = bool(gh_rate_limit)
        self.gh_rate_limit_interval_s = float(gh_rate_limit_interval_s)
        self.gh_rate_limit_resources = [str(s) for s in gh_rate_limit_resources]
        self.once = once
        self.duration_s = duration_s
        self._stop = False

        self._prev_net: Optional[psutil._common.snetio] = None
        self._prev_disk: Optional[psutil._common.sdiskio] = None
        self._prev_ts: Optional[float] = None
        self._prev_proc_io: Dict[int, Tuple[int, int]] = {}
        self._prev_net_top_ts: float = 0.0
        self._prev_docker_stats_ts: float = 0.0
        self._prev_gh_rate_limit_ts: float = 0.0
        self._prev_mem_ts: float = 0.0

    def request_stop(self) -> None:
        self._stop = True

    def _should_exit_due_to_pid_guard(self) -> bool:
        """
        If the lock file PID has changed, another instance has been force-started.
        Exit so the new instance can take over.
        """
        try:
            if self.lock.pid_changed():
                LOGGER.warning("PID guard changed in lock file; exiting (new instance takeover).")
                return True
        except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
            # If we can't read the file, don't kill the monitor; keep running.
            pass
        return False

    def run(self) -> int:
        _ensure_cpu_percent_primed()
        start_ts = _now_unix()

        while True:
            if self._should_exit_due_to_pid_guard():
                return 0
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
            except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
                pass

            load = _get_loadavg()
            # Memory can be downsampled to reduce DB churn/size; store NULLs between samples.
            mem = None
            swap = None
            if (ts - float(self._prev_mem_ts)) >= max(1.0, float(self.mem_interval_s)) or self._prev_mem_ts == 0.0:
                self._prev_mem_ts = ts
                try:
                    mem = psutil.virtual_memory()
                except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
                    mem = None
                try:
                    swap = psutil.swap_memory()
                except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
                    swap = None

            net = psutil.net_io_counters()
            net_sent_bps = None
            net_recv_bps = None
            if self._prev_net is not None:
                d_sent = net.bytes_sent - self._prev_net.bytes_sent
                d_recv = net.bytes_recv - self._prev_net.bytes_recv
                # Counters can reset (iface bounce, namespace changes, wrap); never emit negative rates.
                if d_sent >= 0:
                    net_sent_bps = d_sent / dt_s
                if d_recv >= 0:
                    net_recv_bps = d_recv / dt_s

            disk = _get_disk_io()
            disk_read_bps = None
            disk_write_bps = None
            if disk is not None and self._prev_disk is not None:
                d_read = disk.read_bytes - self._prev_disk.read_bytes
                d_write = disk.write_bytes - self._prev_disk.write_bytes
                # Same counter-reset guard as net.
                if d_read >= 0:
                    disk_read_bps = d_read / dt_s
                if d_write >= 0:
                    disk_write_bps = d_write / dt_s

            gpus, pid_to_gpu_mem = _get_gpu_metrics()

            offenders, self._prev_proc_io = _collect_processes(
                dt_s=dt_s,
                pid_to_gpu_mem=pid_to_gpu_mem,
                prev_io=self._prev_proc_io,
                thresholds=self.thresholds,
                top_k=self.top_k,
            )

            # Network latency (best-effort; store for querying/averaging)
            ping_rows = _collect_ping_samples(
                DEFAULT_PING_TARGETS,
                timeout_s=min(1.0, max(0.2, float(self.interval_s) / 5.0)),
            )

            # Per-process network attribution (optional; best-effort)
            net_top_rows: List[Dict[str, Any]] = []
            if self.net_top:
                if (ts - float(self._prev_net_top_ts)) >= max(1.0, float(self.net_top_interval_s)):
                    self._prev_net_top_ts = ts
                    try:
                        net_top_rows = _collect_net_top_nethogs(
                            top_k=int(self.net_top_k),
                            timeout_s=min(5.0, max(1.5, float(self.net_top_interval_s))),
                        )
                    except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
                        # Never fail the monitor due to optional tooling.
                        net_top_rows = []

            # Per-container resource usage (optional; best-effort)
            docker_rows: List[Dict[str, Any]] = []
            if self.docker_stats:
                if (ts - float(self._prev_docker_stats_ts)) >= max(1.0, float(self.docker_stats_interval_s)):
                    self._prev_docker_stats_ts = ts
                    try:
                        docker_rows = _collect_docker_stats(timeout_s=min(4.0, max(1.0, float(self.docker_stats_interval_s))))
                    except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
                        docker_rows = []

            # GitHub API rate limit (optional; best-effort)
            gh_rl_rows: List[Dict[str, Any]] = []
            if self.gh_rate_limit:
                if (ts - float(self._prev_gh_rate_limit_ts)) >= max(10.0, float(self.gh_rate_limit_interval_s)):
                    self._prev_gh_rate_limit_ts = ts
                    try:
                        gh_rl_rows = _collect_github_rate_limit(
                            resources=self.gh_rate_limit_resources,
                            timeout_s=min(4.0, max(1.0, float(self.gh_rate_limit_interval_s) / 4.0)),
                        )
                    except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
                        gh_rl_rows = []

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
                self.db.insert_ping_samples(sample_id, ping_rows)
                self.db.insert_net_process_samples(sample_id, net_top_rows)
                self.db.insert_docker_container_samples(sample_id, docker_rows)
                self.db.insert_gh_rate_limit_samples(sample_id, gh_rl_rows)
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
                float(mem.percent) if mem is not None else -1.0,
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
                if self._should_exit_due_to_pid_guard():
                    return 0
                if self._stop:
                    return 0
                time.sleep(0.2)


def _default_db_path() -> Path:
    # Policy: do not write to repo-local `.cache/`; always use the global dynamo-utils cache dir.
    try:
        override = os.environ.get("DYNAMO_UTILS_CACHE_DIR", "").strip()
        if override:
            return Path(override).expanduser() / "resource_monitor.sqlite"
    except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
        pass
    return Path.home() / ".cache" / "dynamo-utils" / "resource_monitor.sqlite"


def _default_lock_path(db_path: Path) -> Path:
    # Default to a GLOBAL lock so starting from different CWDs (and thus different default DB paths)
    # doesn't accidentally allow multiple monitors to run.
    try:
        override = os.environ.get("DYNAMO_UTILS_CACHE_DIR", "").strip()
        if override:
            return Path(override).expanduser() / "resource_monitor.lock"
    except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
        pass
    return Path.home() / ".cache" / "dynamo-utils" / "resource_monitor.lock"


def _per_db_lock_path(db_path: Path) -> Path:
    return db_path.with_suffix(".lock")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-instance resource monitor -> SQLite")
    p.add_argument("--db-path", type=Path, default=_default_db_path(), help="SQLite DB path")
    p.add_argument(
        "--lock-path",
        type=Path,
        default=None,
        help="Lock file path (default: ~/.cache/dynamo-utils/resource_monitor.lock)",
    )
    p.add_argument(
        "--per-db-lock",
        action="store_true",
        help="Use a per-db lock file (<db>.lock) instead of the global default lock",
    )
    p.add_argument(
        "--run-ignore-lock",
        action="store_true",
        help="Run-ignore-lock: request takeover by clobbering PID in lock file to signal an existing instance to exit, then wait for the flock",
    )
    p.add_argument(
        "--run-ignore-lock-timeout-seconds",
        type=float,
        default=10.0,
        help="How long to wait for the existing instance to release the lock when using --run-ignore-lock (default: 10s)",
    )
    # Defaults are tuned for a "minimal args" invocation on keivenc-linux.
    p.add_argument("--interval-seconds", type=float, default=17.0, help="Sampling interval (seconds)")
    p.add_argument(
        "--mem-interval-seconds",
        type=float,
        default=57.0,
        help="How often to store system memory/swap fields in samples (seconds)",
    )
    p.add_argument("--once", action="store_true", help="Take one sample and exit")
    p.add_argument("--duration-seconds", type=float, default=None, help="Run for N seconds then exit")

    p.add_argument("--top-k", type=int, default=5, help="Top-k processes per metric to store")
    p.add_argument("--cpu-threshold", type=float, default=50.0, help="Record any process >= this CPU%%")
    p.add_argument("--rss-threshold-mb", type=float, default=2048.0, help="Record any process >= this RSS (MB)")
    p.add_argument("--io-threshold-mbps", type=float, default=50.0, help="Record any process >= this IO rate (MB/s)")
    p.add_argument("--gpu-mem-threshold-mb", type=float, default=1024.0, help="Record any process >= this GPU mem (MB)")

    p.add_argument(
        "--net-top",
        action="store_true",
        default=True,
        help="Best-effort per-process network throughput attribution (requires nethogs and root); stored in net_process_samples (default: enabled)",
    )
    p.add_argument(
        "--net-top-interval-seconds",
        type=float,
        default=17.0,
        help="How often to sample per-process network usage when --net-top is enabled (seconds)",
    )
    p.add_argument(
        "--net-top-k",
        type=int,
        default=3,
        help="How many top talkers to store per net-top sample",
    )

    p.add_argument(
        "--docker-stats",
        action="store_true",
        default=True,
        help="Best-effort per-container CPU/memory sampling via `docker stats --no-stream`; stored in docker_container_samples (default: enabled)",
    )
    p.add_argument(
        "--docker-stats-interval-seconds",
        type=float,
        default=19.0,
        help="How often to sample docker container stats when enabled (seconds)",
    )

    p.add_argument(
        "--gh-rate-limit",
        action="store_true",
        default=True,
        help="Best-effort GitHub API rate limit sampling via `gh api rate_limit`; stored in gh_rate_limit_samples (default: enabled)",
    )
    p.add_argument(
        "--gh-rate-limit-interval-seconds",
        type=float,
        default=59.0,
        help="How often to sample GitHub rate limits when enabled (seconds)",
    )
    p.add_argument(
        "--gh-rate-limit-resources",
        default="core,search,graphql",
        help="Comma-separated GitHub rate-limit resources to record",
    )

    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    level = logging._nameToLevel[str(args.log_level).upper()]
    logging.basicConfig(level=level)

    db_path: Path = args.db_path
    if args.lock_path is not None:
        lock_path = args.lock_path
    else:
        lock_path = _per_db_lock_path(db_path) if bool(args.per_db_lock) else _default_lock_path(db_path)

    lock = SingleInstanceLock(lock_path)
    lock.acquire_or_exit(
        force_start=bool(args.run_ignore_lock),
        force_timeout_s=float(args.run_ignore_lock_timeout_seconds),
    )

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
        lock=lock,
        hostname=hostname,
        interval_s=float(args.interval_seconds),
        mem_interval_s=float(args.mem_interval_seconds),
        thresholds=thresholds,
        top_k=int(args.top_k),
        net_top=bool(args.net_top),
        # Keep net-top attribution at least as frequent as the main sampling interval
        # (avoids many "unattributed" net spike rows in the report).
        net_top_interval_s=min(float(args.net_top_interval_seconds), float(args.interval_seconds)),
        net_top_k=int(args.net_top_k),
        docker_stats=bool(args.docker_stats),
        docker_stats_interval_s=float(args.docker_stats_interval_seconds),
        gh_rate_limit=bool(args.gh_rate_limit),
        gh_rate_limit_interval_s=float(args.gh_rate_limit_interval_seconds),
        gh_rate_limit_resources=[
            s.strip()
            for s in str(args.gh_rate_limit_resources or "").split(",")
            if s.strip()
        ],
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


