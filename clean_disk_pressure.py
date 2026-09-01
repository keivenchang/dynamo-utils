#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reclaim safe user-owned disk space before the root filesystem fills."""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_HIGH_WATERMARK = 90.0
DEFAULT_LOW_WATERMARK = 85.0
DEFAULT_CRITICAL_WATERMARK = 98.0
DEFAULT_TRANSCRIPT_KEEP_DAYS = 14.0
DEFAULT_PRESSURE_TRANSCRIPT_KEEP_HOURS = 24.0
DEFAULT_TARGET_MIN_AGE_HOURS = 24.0
DEFAULT_MAX_TRANSCRIPT_FILES = 1_000
DEFAULT_MAX_TARGETS = 3


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
    return Path(os.path.realpath(raw_target))


def _process_disappeared(error: OSError) -> bool:
    return error.errno in (errno.ENOENT, errno.ESRCH, errno.EBADF)


def process_references(proc_root: Path = Path("/proc")) -> ActivitySnapshot:
    """Return open files and all process path references visible through procfs."""
    open_files: set[Path] = set()
    open_inodes: set[tuple[int, int]] = set()
    references: set[Path] = set()
    complete = True
    try:
        process_dirs = list(proc_root.iterdir())
    except OSError as error:
        print(f"warning: cannot inspect {proc_root}: {error}", file=sys.stderr)
        return ActivitySnapshot(open_files, open_inodes, references, False)

    for process_dir in process_dirs:
        if not process_dir.name.isdigit():
            continue
        for link_name in ("cwd", "exe"):
            try:
                target = _normalized_proc_target(os.readlink(process_dir / link_name))
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
                target = _normalized_proc_target(os.readlink(fd_link))
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
        print(f"warning: could not list active Docker containers: {error}", file=sys.stderr)
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
        print(f"warning: could not inspect active Docker mounts: {error}", file=sys.stderr)
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
            ["sudo", "-n", "/usr/bin/python3", str(Path(__file__).resolve()), "--activity-snapshot"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"warning: privileged process inspection failed: {error}", file=sys.stderr)
        return ActivitySnapshot(set(), set(), set(), False)
    if result.returncode != 0:
        print("warning: privileged process inspection did not complete", file=sys.stderr)
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
        print(f"warning: invalid privileged process inspection result: {error}", file=sys.stderr)
        return ActivitySnapshot(set(), set(), set(), False)


def inode_open_state(device: int, inode: int, proc_root: Path = Path("/proc")) -> tuple[bool, bool]:
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
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _paths_overlap(first: Path, second: Path) -> bool:
    return _is_within(first, second) or _is_within(second, first)


def filesystem_mount_points(mountinfo: Path = Path("/proc/self/mountinfo")) -> tuple[set[Path], bool]:
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
        raise OSError(result.stderr.strip() or f"du failed with exit {result.returncode}")
    return int(result.stdout.split()[0])


def _verified_git_repo(repo: Path) -> bool:
    if repo.is_symlink():
        return False
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
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


def _tree_has_recent_file(root: Path, cutoff: float) -> bool:
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            print(f"warning: cannot inspect target activity under {directory}: {error}", file=sys.stderr)
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
                    TranscriptCandidate(path, metadata.st_dev, metadata.st_ino, metadata.st_mtime)
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
) -> list[TargetCandidate]:
    candidates: list[TargetCandidate] = []
    if not activity_complete or not mounts_complete:
        print("target: skip all targets because safety discovery was incomplete", file=sys.stderr)
        return candidates
    if not dev_root.is_dir():
        return candidates
    for repo in dev_root.iterdir():
        if not repo.is_dir() or not (
            repo.name.startswith("dynamo") or repo.name.startswith("frontend-crates")
        ):
            continue
        target = repo / "target"
        try:
            metadata = target.lstat()
        except OSError:
            continue
        if not stat.S_ISDIR(metadata.st_mode) or target.is_symlink():
            continue
        if metadata.st_dev != pressured_device:
            print(f"target: skip different filesystem {target}")
            continue
        canonical_target = Path(os.path.realpath(target))
        if any(_is_within(mount_point, canonical_target) for mount_point in mount_points):
            print(f"target: skip target containing a mountpoint {target}")
            continue
        if not _verified_git_repo(repo):
            print(f"target: skip unverified repository {repo}")
            continue
        if _repo_is_active(repo, process_paths, mount_sources):
            print(f"target: skip active repository {repo}")
            continue
        age_hours = max(0.0, (now - metadata.st_mtime) / 3600.0)
        cutoff = now - minimum_age_hours * 3600.0
        if not include_recent and (
            age_hours < minimum_age_hours or _tree_has_recent_file(target, cutoff)
        ):
            print(f"target: skip recent repository {repo} (age={age_hours:.1f}h)")
            continue
        try:
            size = _allocated_size(target)
        except (OSError, subprocess.TimeoutExpired, ValueError) as error:
            print(f"warning: cannot size {target}: {error}", file=sys.stderr)
            continue
        candidates.append(
            TargetCandidate(
                repo,
                target,
                size,
                age_hours,
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mtime,
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.size, reverse=True)


def _target_identity_matches(candidate: TargetCandidate) -> bool:
    target = candidate.target
    repo = candidate.repo
    try:
        current_target = target.lstat()
    except OSError:
        return False
    return not (
        target.parent != repo
        or target.name != "target"
        or target.is_symlink()
        or current_target.st_dev != candidate.device
        or current_target.st_ino != candidate.inode
        or current_target.st_mtime != candidate.mtime
    )


def _remove_target(candidate: TargetCandidate, dry_run: bool) -> bool:
    target = candidate.target
    repo = candidate.repo
    if not _target_identity_matches(candidate):
        print(f"warning: target safety check changed for {target}; skipping", file=sys.stderr)
        return False
    if not _verified_git_repo(repo):
        print(f"warning: repository safety check changed for {repo}; skipping", file=sys.stderr)
        return False
    if dry_run:
        return True
    current_process_paths, current_mount_sources, activity_complete = complete_target_activity()
    if not activity_complete:
        print("warning: activity recheck was incomplete; skipping all target deletion", file=sys.stderr)
        return False
    if _repo_is_active(repo, current_process_paths, current_mount_sources):
        print(f"warning: repository became active before cleanup; skipping {repo}", file=sys.stderr)
        return False
    current_mount_points, mounts_complete = filesystem_mount_points()
    canonical_target = Path(os.path.realpath(target))
    if not mounts_complete or any(
        _is_within(mount_point, canonical_target) for mount_point in current_mount_points
    ):
        print(f"warning: target mount safety changed before cleanup; skipping {target}", file=sys.stderr)
        return False
    if not _target_identity_matches(candidate):
        print(f"warning: target identity changed before cleanup; skipping {target}", file=sys.stderr)
        return False
    result = subprocess.run(
        ["/usr/bin/find", str(target), "-xdev", "-depth", "-delete"],
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode != 0 and target.exists():
        result = subprocess.run(
            ["sudo", "-n", "/usr/bin/find", str(target), "-xdev", "-depth", "-delete"],
            check=False,
            capture_output=True,
            text=True,
            timeout=900,
        )
    if result.returncode == 0 and not target.exists():
        return True
    print(f"warning: could not remove {target}: {result.stderr.strip()}", file=sys.stderr)
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
        include_recent=initial.percent >= critical_watermark,
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
        if current_percent <= low_watermark:
            break
        action = "would remove" if dry_run else "removing"
        print(
            f"target: {action} {candidate.target} "
            f"({_format_bytes(candidate.size)}, age={candidate.age_hours:.1f}h)"
        )
        if not _remove_target(candidate, dry_run):
            continue
        removed += 1
        estimated_reclaimed += candidate.size
        simulated_used = max(0, simulated_used - candidate.size)
        simulated_available += candidate.size
    print(
        f"targets: {'would remove' if dry_run else 'removed'} {removed}, "
        f"estimated_reclaimed={_format_bytes(estimated_reclaimed)}"
    )
    return removed, estimated_reclaimed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean old agent transcripts and inactive build targets under root-disk pressure."
    )
    parser.add_argument("--dry-run", action="store_true", help="Report without deleting files")
    parser.add_argument(
        "--pressure-only",
        action="store_true",
        help="Exit immediately when usage is below the high watermark",
    )
    parser.add_argument("--root-path", type=Path, default=Path("/"))
    parser.add_argument("--home", type=Path, default=Path.home())
    parser.add_argument(
        "--dev-root",
        type=Path,
        default=Path(os.environ.get("NVIDIA_HOME", str(Path.home() / "dev"))),
    )
    parser.add_argument("--high-watermark", type=float, default=DEFAULT_HIGH_WATERMARK)
    parser.add_argument("--low-watermark", type=float, default=DEFAULT_LOW_WATERMARK)
    parser.add_argument("--critical-watermark", type=float, default=DEFAULT_CRITICAL_WATERMARK)
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
    parser.add_argument("--target-min-age-hours", type=float, default=DEFAULT_TARGET_MIN_AGE_HOURS)
    parser.add_argument("--max-transcript-files", type=int, default=DEFAULT_MAX_TRANSCRIPT_FILES)
    parser.add_argument("--max-targets", type=int, default=DEFAULT_MAX_TARGETS)
    parser.add_argument("--activity-snapshot", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--inode-open-state",
        nargs=2,
        type=int,
        metavar=("DEVICE", "INODE"),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
    if not 0.0 <= args.low_watermark < args.high_watermark <= 100.0:
        print("error: require 0 <= low-watermark < high-watermark <= 100", file=sys.stderr)
        return 2
    if not args.high_watermark <= args.critical_watermark <= 100.0:
        print("error: critical-watermark must be at least high-watermark and at most 100", file=sys.stderr)
        return 2
    if not args.transcript_keep_days.is_integer() or args.transcript_keep_days < 1:
        print("error: transcript-keep-days must be a positive integer", file=sys.stderr)
        return 2
    if min(
        args.transcript_keep_days,
        args.pressure_transcript_keep_hours,
        args.target_min_age_hours,
        args.max_transcript_files,
        args.max_targets,
    ) < 0:
        print("error: retention, age, and limit values must be non-negative", file=sys.stderr)
        return 2

    before = disk_state(args.root_path)
    print(
        f"disk: path={args.root_path} used={before.percent:.1f}% "
        f"available={_format_bytes(before.available)}"
    )
    under_pressure = before.percent >= args.high_watermark
    if args.pressure_only and not under_pressure:
        print(f"disk: below {args.high_watermark:.1f}% high watermark; no cleanup needed")
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
            print("transcripts: skipped because process inspection was incomplete", file=sys.stderr)

    after_transcripts = before if args.dry_run else disk_state(args.root_path)
    if under_pressure and after_transcripts.percent > args.low_watermark:
        process_paths, mount_sources, activity_complete = complete_target_activity()
        cleanup_targets(
            args.root_path,
            args.dev_root,
            process_paths,
            mount_sources,
            activity_complete,
            args.low_watermark,
            args.critical_watermark,
            args.target_min_age_hours,
            args.max_targets,
            args.dry_run,
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
    if under_pressure and not args.dry_run and after.percent >= args.high_watermark:
        print(
            f"error: disk pressure remains above {args.high_watermark:.1f}% "
            f"at {after.percent:.1f}%",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
