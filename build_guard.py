#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coordinate builds and cleanup without serializing independent builds."""

import argparse
import fcntl
import math
import os
import shutil
import stat
import subprocess
from pathlib import Path

GIB = 1024**3
DEFAULT_MIN_FREE_GIB = 200
DEFAULT_TARGET_FREE_GIB = 350
LEASE_FILENAME = ".build-cleanup.lock"


def open_lease(repo: Path) -> int:
    repo = repo.resolve()
    fd = os.open(repo / LEASE_FILENAME, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise PermissionError("build lease must be a regular file")
        # Root containers must leave the host's cleaner able to reopen this inode.
        if os.geteuid() == 0:
            owner = repo.stat()
            os.fchown(fd, owner.st_uid, owner.st_gid)
        return fd
    except OSError:
        os.close(fd)
        raise


class BuildLease:
    def __init__(self, repo: Path, exclusive: bool = False):
        self.path = repo.resolve() / LEASE_FILENAME
        self.exclusive = exclusive
        self.fd = -1

    def __enter__(self):
        self.fd = open_lease(self.path.parent)
        try:
            mode = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH
            fcntl.flock(self.fd, mode | fcntl.LOCK_NB)
        except OSError:
            os.close(self.fd)
            self.fd = -1
            raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        os.close(self.fd)
        self.fd = -1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=os.environ.get("DYNAMO_BUILD_MIN_FREE_GIB", DEFAULT_MIN_FREE_GIB),
    )
    parser.add_argument(
        "--check-space",
        action="store_true",
        help="Check space without launching a build",
    )
    parser.add_argument(
        "--lock-path",
        action="store_true",
        help="Print the canonical lease path without creating it",
    )
    parser.add_argument(
        "--prepare-lock",
        action="store_true",
        help="Create a repository-owned lease and print its path for shell callers",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if (
        not (command or args.check_space or args.lock_path or args.prepare_lock)
        or not math.isfinite(args.min_free_gib)
        or args.min_free_gib < 0
    ):
        parser.error("a command and non-negative free-space limit are required")
    if args.prepare_lock:
        os.close(open_lease(args.repo))
    if args.lock_path or args.prepare_lock:
        print(args.repo.resolve() / LEASE_FILENAME)
        return 0
    with BuildLease(args.repo) as lease:
        available = shutil.disk_usage(args.repo).free
        if available < args.min_free_gib * GIB:
            print(
                f"build: insufficient free space ({available / GIB:.1f} GiB); "
                "run cleanup first"
            )
            return 3
        if args.check_space:
            return 0
        return subprocess.run(command, check=False, pass_fds=(lease.fd,)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
