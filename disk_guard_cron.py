#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the shared cleanup coordinator with bounded, dated temporary logs."""

import argparse
import datetime
import fcntl
import os
import signal
import stat
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo


def run_cleanup(command: list[str], log, timeout: float = 1800) -> int:
    process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Terminate the exact group we launched, including cleaners, not just its shell.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        # The coordinator can exit before a child does. Kill any remaining members.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return 124


def trim_log(log) -> None:
    if log.seek(0, os.SEEK_END) > 10 * 1024**2:
        log.seek(-(1024**2), os.SEEK_END)
        retained = log.read()
        log.seek(0)
        log.write(retained)
        log.truncate()
    log.seek(0, os.SEEK_END)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maintenance", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(f"/tmp/dynamo-disk-guard-{os.getuid()}")
    root.mkdir(mode=0o700, exist_ok=True)
    metadata = root.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o077
    ):
        raise PermissionError(f"unsafe log directory {root}")
    lock_fd = os.open(
        root / "scheduler.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                f"cleanup: another scheduler owns {root / 'scheduler.lock'}",
                file=sys.stderr,
            )
            return 0
        now = datetime.datetime.now(ZoneInfo("America/Los_Angeles"))
        for path in root.glob("*.log"):
            if path.is_symlink():
                continue
            if now.timestamp() - path.stat().st_mtime > 15 * 86400:
                path.unlink()
        mode = "maintenance" if args.maintenance else "pressure"
        log_path = root / f"{now:%Y-%m-%d}-{mode}.log"
        fd = os.open(log_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "r+b") as log:
            trim_log(log)
            log.write(f"\n{now.isoformat()} START {mode}\n".encode())
            log.flush()
            command = [str(Path(__file__).parent / "clean_system.sh")]
            command += (
                ["--maintenance", "--keep-days", "15", "--retain-dynamo-images", "2"]
                if args.maintenance
                else ["--pressure-only"]
            )
            if args.maintenance and now.weekday() != 6:
                command.append("--skip-images")
            if args.dry_run:
                command.append("--dry-run")
            result = run_cleanup(command, log)
            log.write(f"END rc={result}\n".encode())
            log.flush()
            trim_log(log)
        if result:
            print(f"cleanup failed rc={result}; see {log_path}", file=sys.stderr)
        return result
    finally:
        os.close(lock_fd)


if __name__ == "__main__":
    raise SystemExit(main())
