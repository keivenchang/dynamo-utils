#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reclaim safe user-owned disk space before the root filesystem fills."""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import math
import os
import pwd
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from build_guard import DEFAULT_MIN_FREE_GIB, DEFAULT_TARGET_FREE_GIB, GIB, BuildLease

DOC_LOCK_PATH = (
    Path(pwd.getpwuid(Path(__file__).stat().st_uid).pw_dir)
    / "dev/ai-config/agents/skills/doc-edit-lock/scripts/doc_lock.py"
)
if DOC_LOCK_PATH.is_file():
    sys.path.insert(0, str(DOC_LOCK_PATH.parent))
    from doc_lock import DocumentLocks
else:
    DocumentLocks = None


DEFAULT_HIGH_WATERMARK = 90.0
DEFAULT_LOW_WATERMARK = 80.0
DEFAULT_CRITICAL_WATERMARK = 98.0
DEFAULT_TRANSCRIPT_KEEP_DAYS = 14.0
DEFAULT_PRESSURE_TRANSCRIPT_KEEP_HOURS = 24.0
DEFAULT_TARGET_MIN_AGE_HOURS = 24.0
DEFAULT_MAX_TRANSCRIPT_FILES = 1_000
DEFAULT_MAX_TARGETS = 20


@dataclass(frozen=True)
class DiskState:
    total: int
    used: int
    available: int
    percent: float


@dataclass(frozen=True)
class TargetCandidate:
    repo: Path
    target: Path
    size: int
    age_hours: float
    device: int
    inode: int
    mtime: float
    target_root: Path
    profile: Path
    cutoff: float


@dataclass(frozen=True)
class TranscriptCandidate:
    path: Path
    device: int
    inode: int
    mtime: float


@dataclass(frozen=True)
class ActivitySnapshot:
    open_files: set[Path]
    open_inodes: set[tuple[int, int]]
    process_references: set[Path]
    complete: bool


def _format_bytes(value: int) -> str:
    amount = float(value)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or suffix == "TiB":
            return f"{amount:.1f}{suffix}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def disk_state(path: Path) -> DiskState:
    usage = shutil.disk_usage(path)
    denominator = usage.used + usage.free
    percent = 100.0 * usage.used / denominator if denominator else 0.0
    return DiskState(usage.total, usage.used, usage.free, percent)


def _normalized_proc_target(raw_target: str) -> Path | None:
    deleted_suffix = " (deleted)"
    if raw_target.endswith(deleted_suffix):
        raw_target = raw_target[: -len(deleted_suffix)]
    if not raw_target.startswith("/"):
        return None
    return Path(
        raw_target
    )  # procfs already resolves symlinks; avoid restatting every mapped library.


def _process_disappeared(error: OSError) -> bool:
    return error.errno in (errno.ENOENT, errno.ESRCH, errno.EBADF)


def _process_path(raw: str, mounts: list[tuple[Path, Path]]) -> Path | None:
    path = _normalized_proc_target(raw)
    if path is None:
        return None
    for destination, source in mounts:
        if _is_within(path, destination):
            return source / path.relative_to(destination)
    return path


def _process_bind_mounts(process_dir: Path) -> list[tuple[Path, Path]]:
    # mountinfo's root records host bind sources even when a process sees /workspace.
    mappings = []
    for line in (process_dir / "mountinfo").read_text().splitlines():
        fields = line.split()
        if len(fields) < 5:
            raise ValueError("malformed process mount record")
        root, destination = fields[3:5]
        for escape, value in (("\\040", " "), ("\\011", "\t"), ("\\134", "\\")):
            root = root.replace(escape, value)
            destination = destination.replace(escape, value)
        if root.startswith("/") and root != "/":
            mappings.append((Path(destination), Path(root)))
    return sorted(mappings, key=lambda pair: len(pair[0].parts), reverse=True)


def process_references(proc_root: Path = Path("/proc")) -> ActivitySnapshot:
    """Return open files and all process path references visible through procfs."""
    open_files: set[Path] = set()
    open_inodes: set[tuple[int, int]] = set()
    references: set[Path] = set()
    complete = True
    bind_cache = {}
    try:
        process_dirs = list(proc_root.iterdir())
    except OSError as error:
        print(f"warning: cannot inspect {proc_root}: {error}", file=sys.stderr)
        return ActivitySnapshot(open_files, open_inodes, references, False)

    for process_dir in process_dirs:
        if not process_dir.name.isdigit():
            continue
        try:
            status = (process_dir / "stat").read_text().rpartition(") ")[2].split()[0]
            if status == "Z":
                # Zombies have released their files and mount namespace.
                continue
            namespace = (process_dir / "ns" / "mnt").stat()
            namespace_key = (namespace.st_dev, namespace.st_ino)
            if namespace_key not in bind_cache:
                bind_cache[namespace_key] = _process_bind_mounts(process_dir)
            bind_mounts = bind_cache[namespace_key]
        except (OSError, ValueError) as error:
            if isinstance(error, ValueError) or not _process_disappeared(error):
                complete = False
            continue
        for link_name in ("cwd", "exe"):
            try:
                target = _process_path(
                    os.readlink(process_dir / link_name), bind_mounts
                )
            except OSError as error:
                if not _process_disappeared(error):
                    complete = False
                continue
            if target is not None and target != Path("/"):
                references.add(target)
        fd_dir = process_dir / "fd"
        try:
            fd_links = list(fd_dir.iterdir())
        except OSError as error:
            if not _process_disappeared(error):
                complete = False
            continue
        for fd_link in fd_links:
            try:
                target = _process_path(os.readlink(fd_link), bind_mounts)
                metadata = fd_link.stat()
            except OSError as error:
                if not _process_disappeared(error):
                    complete = False
                continue
            if target is None:
                continue
            open_files.add(target)
            open_inodes.add((metadata.st_dev, metadata.st_ino))
            if target != Path("/"):
                references.add(target)
        try:
            for line in (process_dir / "maps").read_text().splitlines():
                fields = line.split(None, 5)
                if len(fields) == 6:
                    target = _process_path(fields[5], bind_mounts)
                    if target is not None:
                        references.add(target)
        except OSError as error:
            if not _process_disappeared(error):
                complete = False
    return ActivitySnapshot(open_files, open_inodes, references, complete)


def docker_mount_sources() -> tuple[set[Path], bool]:
    if shutil.which("docker") is None:
        return set(), True
    try:
        containers = subprocess.run(
            ["docker", "ps", "-q"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(
            f"warning: could not list active Docker containers: {error}",
            file=sys.stderr,
        )
        return set(), False
    container_ids = containers.stdout.split()
    if containers.returncode != 0:
        print("warning: could not list active Docker containers", file=sys.stderr)
        return set(), False
    if not container_ids:
        return set(), True
    try:
        inspected = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{range .Mounts}}{{println .Source}}{{end}}",
                *container_ids,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(
            f"warning: could not inspect active Docker mounts: {error}", file=sys.stderr
        )
        return set(), False
    if inspected.returncode != 0:
        print("warning: could not inspect active Docker mounts", file=sys.stderr)
        return set(), False
    return (
        {
            Path(os.path.realpath(line))
            for line in inspected.stdout.splitlines()
            if line.startswith("/")
        },
        True,
    )


def privileged_process_references() -> ActivitySnapshot:
    if os.geteuid() == 0:
        return process_references()
    try:
        result = subprocess.run(
            [
                "sudo",
                "-n",
                "/usr/bin/python3",
                str(Path(__file__).resolve()),
                "--activity-snapshot",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(
            f"warning: privileged process inspection failed: {error}", file=sys.stderr
        )
        return ActivitySnapshot(set(), set(), set(), False)
    if result.returncode != 0:
        print(
            "warning: privileged process inspection did not complete", file=sys.stderr
        )
        return ActivitySnapshot(set(), set(), set(), False)
    try:
        payload = json.loads(result.stdout)
        return ActivitySnapshot(
            {Path(path) for path in payload["open_files"]},
            {(int(device), int(inode)) for device, inode in payload["open_inodes"]},
            {Path(path) for path in payload["process_references"]},
            bool(payload["complete"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print(
            f"warning: invalid privileged process inspection result: {error}",
            file=sys.stderr,
        )
        return ActivitySnapshot(set(), set(), set(), False)


def inode_open_state(
    device: int, inode: int, proc_root: Path = Path("/proc")
) -> tuple[bool, bool]:
    complete = True
    try:
        process_dirs = list(proc_root.iterdir())
    except OSError:
        return False, False
    for process_dir in process_dirs:
        if not process_dir.name.isdigit():
            continue
        try:
            fd_links = list((process_dir / "fd").iterdir())
        except OSError as error:
            if not _process_disappeared(error):
                complete = False
            continue
        for fd_link in fd_links:
            try:
                metadata = fd_link.stat()
            except OSError as error:
                if not _process_disappeared(error):
                    complete = False
                continue
            if (metadata.st_dev, metadata.st_ino) == (device, inode):
                return True, complete
    return False, complete


def privileged_inode_open_state(device: int, inode: int) -> tuple[bool, bool]:
    if os.geteuid() == 0:
        return inode_open_state(device, inode)
    try:
        result = subprocess.run(
            [
                "sudo",
                "-n",
                "/usr/bin/python3",
                str(Path(__file__).resolve()),
                "--inode-open-state",
                str(device),
                str(inode),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"warning: privileged inode inspection failed: {error}", file=sys.stderr)
        return False, False
    if result.returncode == 0:
        return True, True
    if result.returncode == 1:
        return False, True
    print("warning: privileged inode inspection was incomplete", file=sys.stderr)
    return False, False


def complete_target_activity() -> tuple[set[Path], set[Path], bool]:
    snapshot = privileged_process_references()
    mounts, mounts_complete = docker_mount_sources()
    return snapshot.process_references, mounts, snapshot.complete and mounts_complete


def _is_within(path: Path, parent: Path) -> bool:
    # All callers use absolute, normalized paths. Avoid millions of exception-based
    # relative_to checks when scanning container memory maps.
    path_text, parent_text = str(path), str(parent)
    return path_text == parent_text or path_text.startswith(
        parent_text.rstrip("/") + "/"
    )


def _paths_overlap(first: Path, second: Path) -> bool:
    return _is_within(first, second) or _is_within(second, first)


def filesystem_mount_points(
    mountinfo: Path = Path("/proc/self/mountinfo"),
) -> tuple[set[Path], bool]:
    escape_map = {"\\040": " ", "\\011": "\t", "\\012": "\n", "\\134": "\\"}
    try:
        lines = mountinfo.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        print(f"warning: cannot inspect filesystem mounts: {error}", file=sys.stderr)
        return set(), False
    mount_points: set[Path] = set()
    for line in lines:
        fields = line.split()
        if len(fields) < 5:
            print("warning: malformed filesystem mount record", file=sys.stderr)
            return set(), False
        value = fields[4]
        for escaped, replacement in escape_map.items():
            value = value.replace(escaped, replacement)
        mount_points.add(Path(os.path.realpath(value)))
    return mount_points, True


def _repo_is_active(
    repo: Path,
    process_paths: set[Path],
    mount_sources: set[Path],
) -> bool:
    canonical_repo = Path(os.path.realpath(repo))
    return any(_is_within(path, canonical_repo) for path in process_paths) or any(
        _paths_overlap(source, canonical_repo) for source in mount_sources
    )


def _allocated_size(path: Path) -> int:
    result = subprocess.run(
        ["du", "-sx", "--block-size=1", "--", str(path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise OSError(
            result.stderr.strip() or f"du failed with exit {result.returncode}"
        )
    return int(result.stdout.split()[0])


def _verified_git_repo(repo: Path) -> bool:
    if repo.is_symlink():
        return False
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo}",
            "-C",
            str(repo),
            "rev-parse",
            "--show-toplevel",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        return False
    try:
        reported_root = Path(result.stdout.strip()).resolve(strict=True)
        actual_root = repo.resolve(strict=True)
    except OSError:
        return False
    return reported_root == actual_root


def _verified_target_owner(repo: Path) -> bool:
    if repo.is_symlink():
        return False
    if _verified_git_repo(repo):
        return True
    manifest = repo / "Cargo.toml"
    if manifest.is_file() and not manifest.is_symlink():
        return (
            re.search(
                r"(?m)^\s*\[(package|workspace)\]\s*(?:#.*)?$", manifest.read_text()
            )
            is not None
        )
    # Standalone CARGO_TARGET_DIRs contain compiler metadata, not project sources.
    marker = repo / ".rustc_info.json"
    if marker.is_file() and not marker.is_symlink():
        metadata = json.loads(marker.read_text())
        return "rustc_fingerprint" in metadata and any(
            (repo / profile / ".cargo-lock").is_file()
            for profile in ("debug", "release")
        )
    return False


def _tree_has_recent_file(root: Path, cutoff: float) -> bool:
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            print(
                f"warning: cannot inspect target activity under {directory}: {error}",
                file=sys.stderr,
            )
            return True
        for entry in entries:
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError:
                return True
            if stat.S_ISREG(metadata.st_mode) and metadata.st_mtime >= cutoff:
                return True
            if stat.S_ISDIR(metadata.st_mode) and not entry.is_symlink():
                pending.append(Path(entry.path))
    return False


def transcript_candidates(
    home: Path,
    cutoff: float,
    open_files: set[Path],
    open_inodes: set[tuple[int, int]],
) -> list[TranscriptCandidate]:
    candidates: list[TranscriptCandidate] = []
    roots = (home / ".codex" / "sessions", home / ".claude" / "projects")
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.jsonl"):
            try:
                metadata = path.lstat()
            except OSError:
                continue
            if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
                continue
            normalized = Path(os.path.realpath(path))
            inode = (metadata.st_dev, metadata.st_ino)
            if (
                metadata.st_mtime < cutoff
                and normalized not in open_files
                and inode not in open_inodes
            ):
                candidates.append(
                    TranscriptCandidate(
                        path, metadata.st_dev, metadata.st_ino, metadata.st_mtime
                    )
                )
    return sorted(candidates, key=lambda candidate: candidate.mtime)


def _remove_transcript_candidate(
    candidate: TranscriptCandidate,
    cutoff: float,
    open_files: set[Path],
    open_inodes: set[tuple[int, int]],
    dry_run: bool,
) -> tuple[bool, int]:
    path = candidate.path
    try:
        metadata = path.lstat()
    except OSError:
        return False, 0

    def metadata_matches(current: os.stat_result) -> bool:
        inode = (current.st_dev, current.st_ino)
        return (
            stat.S_ISREG(current.st_mode)
            and not path.is_symlink()
            and inode == (candidate.device, candidate.inode)
            and current.st_mtime == candidate.mtime
            and current.st_mtime < cutoff
            and Path(os.path.realpath(path)) not in open_files
            and inode not in open_inodes
        )

    if not metadata_matches(metadata):
        return False, 0
    allocated = metadata.st_blocks * 512
    if not dry_run:
        is_open, inspection_complete = privileged_inode_open_state(
            candidate.device, candidate.inode
        )
        if is_open or not inspection_complete:
            return False, 0
        try:
            current = path.lstat()
            if not metadata_matches(current):
                return False, 0
            path.unlink()
        except OSError:
            return False, 0
    return True, allocated


def cleanup_transcripts(
    home: Path,
    cutoff: float,
    open_files: set[Path],
    open_inodes: set[tuple[int, int]],
    dry_run: bool,
    max_files: int,
) -> tuple[int, int, int]:
    candidates = transcript_candidates(home, cutoff, open_files, open_inodes)
    selected = candidates[:max_files]
    removed = 0
    reclaimed = 0
    skipped = 0
    for candidate in selected:
        removed_candidate, allocated = _remove_transcript_candidate(
            candidate,
            cutoff,
            open_files,
            open_inodes,
            dry_run,
        )
        if removed_candidate:
            removed += 1
            reclaimed += allocated
        else:
            skipped += 1
    remaining = max(0, len(candidates) - len(selected))
    action = "would remove" if dry_run else "removed"
    print(
        f"transcripts: {action} {removed} files ({_format_bytes(reclaimed)}), "
        f"revalidation_skips={skipped}, eligible_remaining={remaining}"
    )
    return removed, reclaimed, skipped


def discover_repositories(dev_root: Path, tmp_root: Path | None = None) -> list[Path]:
    """Bound discovery to repo families and shallow temporary Rust projects."""
    roots = []
    if dev_root.is_dir():
        for child in dev_root.iterdir():
            if child.is_symlink() or not child.is_dir():
                continue
            if child.name.startswith(("dynamo", "frontend-crates")):
                roots.append(child)
                if child.name in ("dynamo", "frontend-crates"):
                    roots.extend(
                        p for p in child.iterdir() if p.is_dir() and not p.is_symlink()
                    )
    if tmp_root is not None and tmp_root.is_dir():
        pending = [(tmp_root, 0)]
        while pending:
            directory, depth = pending.pop()
            if directory != tmp_root and (
                (directory / ".git").exists()
                or (directory / "Cargo.toml").is_file()
                or (directory / ".rustc_info.json").is_file()
            ):
                roots.append(directory)
                continue
            if depth >= 3:
                continue
            try:
                children = list(directory.iterdir())
            except PermissionError:
                continue
            for child in children:
                if child.is_symlink() or not child.is_dir():
                    continue
                if child.stat().st_uid != os.getuid() or child.name.startswith(
                    ("yolo", "yomux", "yo7", "Chrome", ".")
                ):
                    continue
                if child.name in ("target", "node_modules", "venv", "__pycache__"):
                    continue
                pending.append((child, depth + 1))
    return sorted({p.resolve() for p in roots if _verified_target_owner(p)})


def _output_directories(repo: Path) -> list[tuple[Path, Path, Path]]:
    result = []
    if (repo / ".rustc_info.json").is_file():
        target_roots = [repo]
    else:
        target_roots = list(repo.iterdir())
    for target in target_roots:
        if (
            target.is_symlink()
            or not target.is_dir()
            or not (
                target == repo
                or target.name == "target"
                or target.name.startswith("target-")
            )
        ):
            continue
        cargo_roots = [target]
        cargo_roots.extend(
            p
            for p in target.iterdir()
            if p.is_dir()
            and not p.is_symlink()
            and p.name in (".vllm", ".sglang", ".trtllm")
        )
        for cargo_root in cargo_roots:
            if not (cargo_root / ".rustc_info.json").is_file():
                continue
            for profile_name in ("debug", "release"):
                profile = cargo_root / profile_name
                if profile.is_symlink() or not profile.is_dir():
                    continue
                for name in (
                    "deps",
                    "incremental",
                    "build",
                    ".fingerprint",
                    "examples",
                ):
                    output = profile / name
                    if output.is_dir() and not output.is_symlink():
                        result.append((output, cargo_root, profile))
    return result


def _has_tracked_files(repo: Path, output: Path) -> bool:
    if not _verified_git_repo(repo):
        return False
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo}",
            "-C",
            str(repo),
            "ls-files",
            "--",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    return bool(result.stdout)


def _installed_library(name: str) -> bool:
    return name.endswith((".so", ".dylib", ".dll")) or ".so." in name


def _disposable_bytes(root: Path) -> int:
    total = 0
    seen = set()
    pending = [root]
    while pending:
        for entry in os.scandir(pending.pop()):
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(Path(entry.path))
            elif (
                stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode)
            ) and not _installed_library(entry.name):
                key = (metadata.st_dev, metadata.st_ino)
                if key not in seen:
                    total += metadata.st_blocks * 512
                    seen.add(key)
    return total


def target_candidates(
    dev_root: Path,
    process_paths: set[Path],
    mount_sources: set[Path],
    activity_complete: bool,
    pressured_device: int,
    mount_points: set[Path],
    mounts_complete: bool,
    now: float,
    minimum_age_hours: float,
    include_recent: bool,
    tmp_root: Path | None = None,
) -> list[TargetCandidate]:
    candidates = []
    if not math.isfinite(minimum_age_hours) or minimum_age_hours < 0:
        raise ValueError("target minimum age must be finite and non-negative")
    if not activity_complete or not mounts_complete:
        print(
            "target: skip all targets because safety discovery was incomplete",
            file=sys.stderr,
        )
        return candidates
    for repo in discover_repositories(dev_root, tmp_root):
        for output, cargo_root, profile in _output_directories(repo):
            metadata = output.lstat()
            if metadata.st_dev != pressured_device:
                continue
            if any(_is_within(point, output) for point in mount_points | mount_sources):
                continue
            if any(_is_within(path, output) for path in process_paths):
                print(f"target: skip active output {output}")
                continue
            cutoff = now - minimum_age_hours * 3600
            age_hours = max(0.0, (now - metadata.st_mtime) / 3600)
            # Even critical pressure never relaxes the minimum-age requirement.
            if age_hours < minimum_age_hours or _tree_has_recent_file(output, cutoff):
                continue
            if _has_tracked_files(repo, output):
                print(f"target: skip tracked output {output}")
                continue
            disposable = _disposable_bytes(output)
            if not disposable:
                continue
            candidates.append(
                TargetCandidate(
                    repo,
                    output,
                    disposable,
                    age_hours,
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_mtime,
                    cargo_root,
                    profile,
                    cutoff,
                )
            )
    return sorted(
        candidates, key=lambda candidate: (-candidate.age_hours, -candidate.size)
    )


def _target_identity_matches(candidate: TargetCandidate) -> bool:
    try:
        current = candidate.target.lstat()
    except OSError:
        return False
    return (
        not candidate.target.is_symlink()
        and candidate.target.resolve() == candidate.target
        and (current.st_dev, current.st_ino, current.st_mtime)
        == (candidate.device, candidate.inode, candidate.mtime)
        and (candidate.target_root / ".rustc_info.json").is_file()
        and candidate.target.parent == candidate.profile
        and candidate.profile.name in ("debug", "release")
        and candidate.profile.parent == candidate.target_root
        and _is_within(candidate.target_root, candidate.repo)
        and candidate.target.name
        in ("deps", "incremental", "build", ".fingerprint", "examples")
    )


def _delete_output_contents(
    fd: int, device: int, cutoff: float, check_ownership: Callable[[], None]
) -> None:
    for entry in os.scandir(fd):
        check_ownership()
        metadata = entry.stat(follow_symlinks=False)
        if metadata.st_dev != device:
            raise OSError("refusing to cross filesystem during cleanup")
        if stat.S_ISDIR(metadata.st_mode):
            child_fd = os.open(
                entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            try:
                child_meta = os.fstat(child_fd)
                if (child_meta.st_dev, child_meta.st_ino) != (
                    metadata.st_dev,
                    metadata.st_ino,
                ):
                    raise OSError("output directory changed during cleanup")
                _delete_output_contents(child_fd, device, cutoff, check_ownership)
                empty = not any(os.scandir(child_fd))
            finally:
                os.close(child_fd)
            if empty:
                check_ownership()
                os.rmdir(entry.name, dir_fd=fd)
        elif stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            if _installed_library(entry.name):
                continue
            if metadata.st_mtime >= cutoff:
                raise OSError("output became active during cleanup")
            os.unlink(entry.name, dir_fd=fd)
        else:
            raise OSError("refusing to delete non-file runtime artifact")


def _process_identity(pid: int) -> str:
    fields = Path(f"/proc/{pid}/stat").read_text().rpartition(") ")[2].split()
    if fields[0] == "Z":
        raise ProcessLookupError("cleanup supervisor exited")
    return fields[19]


def _require_supervisor(supervisor: tuple[int, str] | None) -> None:
    if supervisor is not None and _process_identity(supervisor[0]) != supervisor[1]:
        raise ProcessLookupError("cleanup supervisor changed")


def _writable_output(root: Path) -> bool:
    pending = [root]
    while pending:
        directory = pending.pop()
        if not os.access(directory, os.W_OK | os.X_OK):
            return False
        for entry in os.scandir(directory):
            if entry.is_dir(follow_symlinks=False):
                pending.append(Path(entry.path))
    return True


def _elevated_remove_target(candidate: TargetCandidate) -> bool:
    payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(candidate).items()
    }
    payload["supervisor"] = [os.getpid(), _process_identity(os.getpid())]
    result = subprocess.run(
        [
            "sudo",
            "-n",
            "/usr/bin/timeout",
            "--foreground",
            "--signal=TERM",
            "--kill-after=10",
            "540",
            "/usr/bin/python3",
            str(Path(__file__).resolve()),
            "--remove-target",
            json.dumps(payload),
        ],
        check=False,
    )
    if result.returncode:
        print(
            f"target: elevated cleanup skipped or failed rc={result.returncode}: "
            f"{candidate.target}",
            file=sys.stderr,
        )
    return result.returncode == 0


def _remove_target(
    candidate: TargetCandidate, dry_run: bool, supervisor: tuple[int, str] | None = None
) -> bool:
    if not _target_identity_matches(candidate) or not _verified_target_owner(
        candidate.repo
    ):
        print(
            f"warning: target identity changed; skipping {candidate.target}",
            file=sys.stderr,
        )
        return False
    if dry_run:
        return True
    if os.geteuid() != 0 and not _writable_output(candidate.target):
        return _elevated_remove_target(candidate)
    _require_supervisor(supervisor)
    try:
        with BuildLease(candidate.repo, exclusive=True):
            # Cargo itself locks this retained inode, so direct cargo callers also
            # coordinate with cleanup. Never unlink the profile or its lock file.
            fd = os.open(
                candidate.profile / ".cargo-lock",
                os.O_CREAT | os.O_RDONLY | os.O_NOFOLLOW,
                0o600,
            )
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                paths, sources, complete = complete_target_activity()
                points, mounts_complete = filesystem_mount_points()
                if (
                    not complete
                    or not mounts_complete
                    or any(_is_within(path, candidate.target) for path in paths)
                    or any(
                        _is_within(point, candidate.target)
                        for point in sources | points
                    )
                    or not _target_identity_matches(candidate)
                    or _has_tracked_files(candidate.repo, candidate.target)
                    or _tree_has_recent_file(candidate.target, candidate.cutoff)
                ):
                    print(
                        f"warning: target safety changed; skipping {candidate.target}",
                        file=sys.stderr,
                    )
                    return False
                if DocumentLocks is None:
                    print(
                        "warning: document lease helper unavailable; skipping target",
                        file=sys.stderr,
                    )
                    return False
                with DocumentLocks(
                    [str(candidate.target)], agent="dynamo-disk-cleaner"
                ) as documents:
                    # Reread after acquiring the document lease. This same process
                    # owns all leases and mutates through directory descriptors.
                    if (
                        not _target_identity_matches(candidate)
                        or _has_tracked_files(candidate.repo, candidate.target)
                        or _tree_has_recent_file(candidate.target, candidate.cutoff)
                    ):
                        return False
                    output_fd = os.open(
                        candidate.target, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                    )
                    last_heartbeat = time.monotonic()

                    def check_ownership() -> None:
                        nonlocal last_heartbeat
                        _require_supervisor(supervisor)
                        if time.monotonic() - last_heartbeat >= 30:
                            documents.heartbeat()
                            last_heartbeat = time.monotonic()
                        documents.assert_owned()

                    try:
                        metadata = os.fstat(output_fd)
                        if (metadata.st_dev, metadata.st_ino) != (
                            candidate.device,
                            candidate.inode,
                        ):
                            return False
                        check_ownership()
                        _delete_output_contents(
                            output_fd,
                            candidate.device,
                            candidate.cutoff,
                            check_ownership,
                        )
                    finally:
                        os.close(output_fd)
                    return True
            finally:
                os.close(fd)
    except (BlockingIOError, PermissionError) as error:
        print(
            f"target: busy or inaccessible {candidate.target}: {error}", file=sys.stderr
        )
        return False


def cleanup_targets(
    root_path: Path,
    dev_root: Path,
    process_paths: set[Path],
    mount_sources: set[Path],
    activity_complete: bool,
    low_watermark: float,
    critical_watermark: float,
    minimum_age_hours: float,
    max_targets: int,
    dry_run: bool,
    tmp_root: Path | None = None,
    maintenance: bool = False,
    target_free_gib: float = DEFAULT_TARGET_FREE_GIB,
) -> tuple[int, int]:
    initial = disk_state(root_path)
    mount_points, mounts_complete = filesystem_mount_points()
    try:
        pressured_device = root_path.stat().st_dev
    except OSError as error:
        print(f"warning: cannot inspect pressured filesystem: {error}", file=sys.stderr)
        pressured_device = -1
        mounts_complete = False
    candidates = target_candidates(
        dev_root,
        process_paths,
        mount_sources,
        activity_complete,
        pressured_device,
        mount_points,
        mounts_complete,
        time.time(),
        minimum_age_hours,
        include_recent=False,
        tmp_root=tmp_root,
    )
    removed = 0
    estimated_reclaimed = 0
    simulated_used = initial.used
    simulated_available = initial.available
    for candidate in candidates:
        if removed >= max_targets:
            break
        current_percent = (
            100.0 * simulated_used / (simulated_used + simulated_available)
            if dry_run
            else disk_state(root_path).percent
        )
        current_free = (
            simulated_available if dry_run else disk_state(root_path).available
        )
        if (
            not maintenance
            and current_percent <= low_watermark
            and current_free >= target_free_gib * GIB
        ):
            break
        action = "would remove" if dry_run else "removing"
        print(
            f"target: {action} {candidate.target} "
            f"({_format_bytes(candidate.size)}, age={candidate.age_hours:.1f}h)"
        )
        before_size = candidate.size if dry_run else _allocated_size(candidate.target)
        if not _remove_target(candidate, dry_run):
            continue
        removed += 1
        reclaimed = (
            candidate.size
            if dry_run
            else max(
                0,
                before_size
                - (
                    _allocated_size(candidate.target)
                    if candidate.target.exists()
                    else 0
                ),
            )
        )
        estimated_reclaimed += reclaimed
        simulated_used = max(0, simulated_used - candidate.size)
        simulated_available += candidate.size
    print(
        f"targets: {'would remove' if dry_run else 'removed'} {removed}, "
        f"estimated_reclaimed={_format_bytes(estimated_reclaimed)}"
    )
    return removed, estimated_reclaimed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean old agent transcripts and inactive build targets "
            "under root-disk pressure."
        )
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report without deleting files"
    )
    parser.add_argument(
        "--pressure-only",
        action="store_true",
        help="Exit immediately when usage is below the high watermark",
    )
    parser.add_argument("--root-path", type=Path, default=Path("/"))
    parser.add_argument("--tmp-root", type=Path, default=Path("/tmp"))
    parser.add_argument(
        "--maintenance",
        action="store_true",
        help="Remove seven-day-old build output even without disk pressure",
    )
    parser.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    parser.add_argument(
        "--target-free-gib", type=float, default=DEFAULT_TARGET_FREE_GIB
    )
    parser.add_argument("--home", type=Path, default=Path.home())
    parser.add_argument(
        "--dev-root",
        type=Path,
        default=Path(os.environ.get("NVIDIA_HOME", str(Path.home() / "dev"))),
    )
    parser.add_argument("--high-watermark", type=float, default=DEFAULT_HIGH_WATERMARK)
    parser.add_argument("--low-watermark", type=float, default=DEFAULT_LOW_WATERMARK)
    parser.add_argument(
        "--critical-watermark", type=float, default=DEFAULT_CRITICAL_WATERMARK
    )
    parser.add_argument(
        "--transcript-keep-days",
        type=float,
        default=DEFAULT_TRANSCRIPT_KEEP_DAYS,
        help="Keep transcripts this many days (default: 14)",
    )
    parser.add_argument(
        "--skip-transcripts",
        action="store_true",
        help="Do not inspect or remove Claude/Codex transcripts",
    )
    parser.add_argument(
        "--pressure-transcript-keep-hours",
        type=float,
        default=DEFAULT_PRESSURE_TRANSCRIPT_KEEP_HOURS,
    )
    parser.add_argument(
        "--target-min-age-hours", type=float, default=DEFAULT_TARGET_MIN_AGE_HOURS
    )
    parser.add_argument(
        "--max-transcript-files", type=int, default=DEFAULT_MAX_TRANSCRIPT_FILES
    )
    parser.add_argument("--max-targets", type=int, default=DEFAULT_MAX_TARGETS)
    parser.add_argument("--remove-target", help=argparse.SUPPRESS)
    parser.add_argument(
        "--activity-snapshot", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--inode-open-state",
        nargs=2,
        type=int,
        metavar=("DEVICE", "INODE"),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _terminate(signum: int, frame) -> None:
    raise SystemExit(128 + signum)


def main() -> int:
    signal.signal(signal.SIGTERM, _terminate)
    args = parse_args()
    if args.remove_target is not None:
        if os.geteuid() != 0:
            raise PermissionError("exact-candidate mutator requires elevation")
        payload = json.loads(args.remove_target)
        supervisor = tuple(payload.pop("supervisor"))
        for key in ("repo", "target", "target_root", "profile"):
            payload[key] = Path(payload[key])
        candidate = TargetCandidate(**payload)
        owner = Path(__file__).stat().st_uid
        home = Path(pwd.getpwuid(owner).pw_dir)
        dev = home / "dev"
        approved = (
            candidate.repo.parent in (dev, dev / "dynamo", dev / "frontend-crates")
            and candidate.repo.name.startswith(("dynamo", "frontend-crates"))
        ) or (
            _is_within(candidate.repo, Path("/tmp"))
            and len(candidate.repo.relative_to("/tmp").parts) <= 3
        )
        if (
            not approved
            or candidate.repo.stat().st_uid != owner
            or not math.isfinite(candidate.cutoff)
            or candidate.cutoff > time.time() - 24 * 3600
        ):
            raise PermissionError("candidate is outside the approved cleanup scope")
        _require_supervisor(supervisor)
        return 0 if _remove_target(candidate, False, supervisor) else 5
    if args.activity_snapshot:
        snapshot = process_references()
        print(
            json.dumps(
                {
                    "open_files": sorted(str(path) for path in snapshot.open_files),
                    "open_inodes": sorted(snapshot.open_inodes),
                    "process_references": sorted(
                        str(path) for path in snapshot.process_references
                    ),
                    "complete": snapshot.complete,
                }
            )
        )
        return 0 if snapshot.complete else 5
    if args.inode_open_state is not None:
        is_open, complete = inode_open_state(*args.inode_open_state)
        if not complete:
            return 5
        return 0 if is_open else 1
    numeric = (
        args.low_watermark,
        args.high_watermark,
        args.critical_watermark,
        args.target_min_age_hours,
        args.transcript_keep_days,
        args.pressure_transcript_keep_hours,
        args.min_free_gib,
        args.target_free_gib,
    )
    if not all(math.isfinite(value) for value in numeric):
        print("error: numeric cleanup limits must be finite", file=sys.stderr)
        return 2
    if not 0.0 <= args.low_watermark < args.high_watermark <= 100.0:
        print(
            "error: require 0 <= low-watermark < high-watermark <= 100", file=sys.stderr
        )
        return 2
    if not args.high_watermark <= args.critical_watermark <= 100.0:
        print(
            "error: critical-watermark must be at least high-watermark and at most 100",
            file=sys.stderr,
        )
        return 2
    if not args.transcript_keep_days.is_integer() or args.transcript_keep_days < 1:
        print("error: transcript-keep-days must be a positive integer", file=sys.stderr)
        return 2
    if (
        min(
            args.transcript_keep_days,
            args.pressure_transcript_keep_hours,
            args.target_min_age_hours,
            args.max_transcript_files,
            args.max_targets,
            args.min_free_gib,
            args.target_free_gib,
        )
        < 0
    ):
        print(
            "error: retention, age, and limit values must be non-negative",
            file=sys.stderr,
        )
        return 2

    if args.target_free_gib < args.min_free_gib:
        print("error: target-free-gib must be at least min-free-gib", file=sys.stderr)
        return 2

    before = disk_state(args.root_path)
    print(
        f"disk: path={args.root_path} used={before.percent:.1f}% "
        f"available={_format_bytes(before.available)}"
    )
    under_pressure = (
        before.percent >= args.high_watermark
        or before.available < args.min_free_gib * GIB
    )
    if args.pressure_only and not under_pressure:
        print(
            f"disk: below {args.high_watermark:.1f}% high watermark; no cleanup needed"
        )
        return 0

    if args.skip_transcripts:
        print("transcripts: skipped by request")
    else:
        snapshot = privileged_process_references()
        keep_seconds = (
            args.pressure_transcript_keep_hours * 3600.0
            if under_pressure
            else args.transcript_keep_days * 86400.0
        )
        if snapshot.complete:
            cleanup_transcripts(
                args.home,
                time.time() - keep_seconds,
                snapshot.open_files,
                snapshot.open_inodes,
                args.dry_run,
                args.max_transcript_files,
            )
        else:
            print(
                "transcripts: skipped because process inspection was incomplete",
                file=sys.stderr,
            )

    after_transcripts = before if args.dry_run else disk_state(args.root_path)
    if args.maintenance or (
        under_pressure
        and (
            after_transcripts.percent > args.low_watermark
            or after_transcripts.available < args.target_free_gib * GIB
        )
    ):
        process_paths, mount_sources, activity_complete = complete_target_activity()
        cleanup_targets(
            args.root_path,
            args.dev_root,
            process_paths,
            mount_sources,
            activity_complete,
            args.low_watermark,
            args.critical_watermark,
            7 * 24
            if args.maintenance and not under_pressure
            else args.target_min_age_hours,
            args.max_targets,
            args.dry_run,
            tmp_root=args.tmp_root,
            maintenance=args.maintenance and not under_pressure,
            target_free_gib=args.target_free_gib,
        )

    after = disk_state(args.root_path)
    print(
        f"disk: {'unchanged by dry run' if args.dry_run else 'after cleanup'} "
        f"used={after.percent:.1f}% available={_format_bytes(after.available)}"
    )
    if under_pressure and not args.dry_run and after.percent >= args.critical_watermark:
        print(
            f"error: critical disk pressure remains at {after.percent:.1f}%",
            file=sys.stderr,
        )
        return 4
    if (
        under_pressure
        and not args.dry_run
        and (
            after.percent >= args.high_watermark
            or after.available < args.min_free_gib * GIB
        )
    ):
        print(
            f"error: disk pressure remains above {args.high_watermark:.1f}% "
            f"at {after.percent:.1f}%",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
