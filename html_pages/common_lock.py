# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PID-file based component lock with cooperative cancellation.

Each component (show_commit_history, show_local_branches, etc.) gets its own
PID file at {repo_path}/.{component_name}.pid.  A background daemon thread
polls the PID file every 5 seconds; if another process overwrites it, the
current holder exits immediately via os._exit(0).
"""

import atexit
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

POLL_INTERVAL_S = 5


def _is_pid_alive(pid: int) -> bool:
    """Check whether a process with the given PID exists (without sending a signal)."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we don't own it -- still alive.
        return True


class ComponentLock:
    """PID-file lock with background polling for cooperative cancellation.

    Usage::

        lock = ComponentLock(repo_path, "show_commit_history", force=args.run_ignore_lock)
        if not lock.acquire():
            return 0  # another instance is running, skip
        try:
            ... do work ...
        finally:
            lock.release()
    """

    def __init__(self, repo_path: Path, component_name: str, *, force: bool = False):
        self.pid_file = Path(repo_path) / f".{component_name}.pid"
        self.my_pid = os.getpid()
        self.force = force
        self._owned = False
        self._poll_thread: threading.Thread | None = None

    def acquire(self) -> bool:
        """Try to become the owner.  Returns True if acquired, False if should skip."""
        existing_pid = self._read_pid()

        if existing_pid is not None:
            if existing_pid == self.my_pid:
                return True

            alive = _is_pid_alive(existing_pid)

            if alive and not self.force:
                logger.warning(
                    "Component locked by pid=%d (alive), skipping (pid_file=%s)",
                    existing_pid, self.pid_file,
                )
                return False

            if alive and self.force:
                logger.warning(
                    "Superseding pid=%d (alive) via --run-ignore-lock (pid_file=%s)",
                    existing_pid, self.pid_file,
                )
            else:
                logger.warning(
                    "Reclaiming stale lock (pid=%d dead) (pid_file=%s)",
                    existing_pid, self.pid_file,
                )

        self._write_pid()
        self._owned = True
        atexit.register(self.release)
        self._start_poll_thread()
        return True

    def check_still_owner(self) -> bool:
        """Returns True if we still own the PID file."""
        current_pid = self._read_pid()
        if current_pid is None or current_pid != self.my_pid:
            return False
        return True

    def release(self):
        """Remove PID file if we still own it."""
        if not self._owned:
            return
        try:
            current_pid = self._read_pid()
            if current_pid == self.my_pid:
                self.pid_file.unlink(missing_ok=True)
                logger.info("Released lock (pid_file=%s)", self.pid_file)
        except OSError:
            pass
        self._owned = False

    def _read_pid(self) -> int | None:
        try:
            text = self.pid_file.read_text().strip()
            return int(text) if text else None
        except (FileNotFoundError, ValueError, OSError):
            return None

    def _write_pid(self):
        tmp = self.pid_file.with_suffix(".pid.tmp")
        tmp.write_text(str(self.my_pid))
        os.replace(tmp, self.pid_file)
        logger.info("Wrote PID %d to %s", self.my_pid, self.pid_file)

    def _start_poll_thread(self):
        def poll():
            while True:
                time.sleep(POLL_INTERVAL_S)
                if not self.check_still_owner():
                    new_pid = self._read_pid()
                    logger.warning(
                        "Superseded by pid=%s (was pid=%d), exiting (pid_file=%s)",
                        new_pid, self.my_pid, self.pid_file,
                    )
                    os._exit(0)

        t = threading.Thread(target=poll, daemon=True, name=f"pid-poll-{self.pid_file.stem}")
        t.start()
        self._poll_thread = t
