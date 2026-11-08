#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import errno
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

import clean_disk_pressure


def _old_file(path: Path, size: int = 4096) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    old = time.time() - 3 * 86400
    os.utime(path, (old, old))
    for parent in path.parents:
        if parent.name == "target" or parent.name.startswith("target-"):
            (parent / ".rustc_info.json").write_text("{}")
            output = parent / "debug" / "deps"
            if output.exists():
                os.utime(output, (old, old))
            break


def test_transcript_cleanup_preserves_open_and_recent_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_codex = tmp_path / ".codex" / "sessions" / "old.jsonl"
    open_claude = tmp_path / ".claude" / "projects" / "open.jsonl"
    recent_codex = tmp_path / ".codex" / "sessions" / "recent.jsonl"
    _old_file(old_codex)
    _old_file(open_claude)
    _old_file(recent_codex)
    os.utime(recent_codex, None)
    monkeypatch.setattr(
        clean_disk_pressure,
        "privileged_inode_open_state",
        lambda _device, _inode: (False, True),
    )

    removed, _, failures = clean_disk_pressure.cleanup_transcripts(
        tmp_path,
        time.time() - 86400,
        {open_claude},
        set(),
        dry_run=False,
        max_files=100,
    )

    assert removed == 1
    assert failures == 0
    assert not old_codex.exists()
    assert open_claude.exists()
    assert recent_codex.exists()


def test_transcript_cleanup_rechecks_open_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_codex = tmp_path / ".codex" / "sessions" / "opened-after-discovery.jsonl"
    _old_file(old_codex)
    monkeypatch.setattr(
        clean_disk_pressure,
        "privileged_inode_open_state",
        lambda _device, _inode: (True, True),
    )

    removed, _, skipped = clean_disk_pressure.cleanup_transcripts(
        tmp_path,
        time.time() - 86400,
        set(),
        set(),
        dry_run=False,
        max_files=100,
    )

    assert removed == 0
    assert skipped == 1
    assert old_codex.exists()


def test_transcript_revalidation_rejects_replaced_or_open_inode(tmp_path: Path) -> None:
    replaced = tmp_path / ".codex" / "sessions" / "replaced.jsonl"
    opened = tmp_path / ".claude" / "projects" / "opened.jsonl"
    _old_file(replaced)
    _old_file(opened)
    cutoff = time.time() - 86400
    candidates = clean_disk_pressure.transcript_candidates(
        tmp_path, cutoff, set(), set()
    )
    by_name = {candidate.path.name: candidate for candidate in candidates}

    replaced.unlink()
    _old_file(replaced)
    replaced_result = clean_disk_pressure._remove_file_candidate(
        by_name["replaced.jsonl"], cutoff, set(), set(), dry_run=False
    )
    open_inode = {(by_name["opened.jsonl"].device, by_name["opened.jsonl"].inode)}
    opened_result = clean_disk_pressure._remove_file_candidate(
        by_name["opened.jsonl"], cutoff, set(), open_inode, dry_run=False
    )

    assert replaced_result == (False, 0)
    assert opened_result == (False, 0)
    assert replaced.exists()
    assert opened.exists()


def test_target_candidates_require_git_repo_and_skip_active_repo(
    tmp_path: Path,
) -> None:
    inactive_repo = tmp_path / "dynamo__inactive"
    active_repo = tmp_path / "frontend-crates__active"
    recently_built_repo = tmp_path / "dynamo__recent-build"
    not_repo = tmp_path / "dynamo__not-repo"
    for repo in (inactive_repo, active_repo, recently_built_repo):
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for repo in (inactive_repo, active_repo, recently_built_repo, not_repo):
        target = repo / "target"
        (target / "debug" / "deps").mkdir(parents=True)
        _old_file(target / "debug" / "deps" / "artifact")
        old = time.time() - 3 * 86400
        os.utime(target, (old, old))
    os.utime(recently_built_repo / "target" / "debug" / "deps" / "artifact", None)

    candidates = clean_disk_pressure.target_candidates(
        tmp_path,
        {active_repo / "target" / "debug" / "deps" / "artifact"},
        set(),
        True,
        tmp_path.stat().st_dev,
        set(),
        True,
        time.time(),
        minimum_age_hours=24,
        include_recent=False,
    )

    assert [candidate.repo for candidate in candidates] == [inactive_repo]


def test_target_candidates_fail_closed_and_honor_parent_mount(tmp_path: Path) -> None:
    repo = tmp_path / "dynamo__candidate"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _old_file(repo / "target" / "debug" / "deps" / "artifact")
    old = time.time() - 3 * 86400
    os.utime(repo / "target", (old, old))

    incomplete = clean_disk_pressure.target_candidates(
        tmp_path,
        set(),
        set(),
        False,
        tmp_path.stat().st_dev,
        set(),
        True,
        time.time(),
        24,
        False,
    )
    parent_mounted = clean_disk_pressure.target_candidates(
        tmp_path,
        set(),
        {tmp_path.resolve()},
        True,
        tmp_path.stat().st_dev,
        set(),
        True,
        time.time(),
        24,
        False,
    )
    wrong_filesystem = clean_disk_pressure.target_candidates(
        tmp_path,
        set(),
        set(),
        True,
        tmp_path.stat().st_dev + 1,
        set(),
        True,
        time.time(),
        24,
        False,
    )
    nested_mount = clean_disk_pressure.target_candidates(
        tmp_path,
        set(),
        set(),
        True,
        tmp_path.stat().st_dev,
        {(repo / "target" / "debug" / "deps" / "mounted").resolve()},
        True,
        time.time(),
        24,
        False,
    )

    assert incomplete == []
    assert (
        len(parent_mounted) == 1
    )  # A bind-mounted source alone is not an active build.
    assert wrong_filesystem == []
    assert nested_mount == []


def test_remove_target_rejects_replaced_directory(tmp_path: Path) -> None:
    repo = tmp_path / "dynamo__candidate"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _old_file(repo / "target" / "debug" / "deps" / "old-artifact")
    old = time.time() - 3 * 86400
    os.utime(repo / "target", (old, old))
    candidates = clean_disk_pressure.target_candidates(
        tmp_path,
        set(),
        set(),
        True,
        tmp_path.stat().st_dev,
        set(),
        True,
        time.time(),
        24,
        False,
    )
    assert len(candidates) == 1

    shutil.rmtree(repo / "target")
    _old_file(repo / "target" / "debug" / "deps" / "replacement-artifact")
    os.utime(repo / "target" / "debug" / "deps", (old + 10, old + 10))

    assert not clean_disk_pressure._remove_target(candidates[0], dry_run=False)
    assert (repo / "target" / "debug" / "deps" / "replacement-artifact").exists()


def test_remove_target_rechecks_identity_after_activity_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "dynamo__candidate"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _old_file(repo / "target" / "debug" / "deps" / "old-artifact")
    old = time.time() - 3 * 86400
    os.utime(repo / "target", (old, old))
    candidate = clean_disk_pressure.target_candidates(
        tmp_path,
        set(),
        set(),
        True,
        tmp_path.stat().st_dev,
        set(),
        True,
        time.time(),
        24,
        False,
    )[0]

    def replace_during_activity_scan() -> tuple[set[Path], set[Path], bool]:
        shutil.rmtree(repo / "target")
        _old_file(repo / "target" / "debug" / "deps" / "replacement-artifact")
        os.utime(repo / "target" / "debug" / "deps", (old + 10, old + 10))
        return set(), set(), True

    monkeypatch.setattr(
        clean_disk_pressure, "complete_target_activity", replace_during_activity_scan
    )

    assert not clean_disk_pressure._remove_target(candidate, dry_run=False)
    assert (repo / "target" / "debug" / "deps" / "replacement-artifact").exists()


def test_path_boundary_does_not_treat_similar_prefix_as_child() -> None:
    assert clean_disk_pressure._is_within(Path("/tmp/repo/target/a"), Path("/tmp/repo"))
    assert not clean_disk_pressure._is_within(
        Path("/tmp/repo-old/target"), Path("/tmp/repo")
    )


def test_clean_system_retains_pressure_failure_after_other_cleanup(
    tmp_path: Path,
) -> None:
    source_root = Path(__file__).parent
    script = tmp_path / "clean_system.sh"
    shutil.copy2(source_root / "clean_system.sh", script)
    (tmp_path / "container").mkdir()
    marker = tmp_path / "steps"
    stubs = {
        tmp_path / "clean_disk_pressure.py": (
            "#!/bin/sh\nprintf 'disk\\n' >> \"$MARKER\"\nexit 3\n"
        ),
        tmp_path / "container" / "clean_old_local_dynamo_images.sh": (
            "#!/bin/sh\nprintf 'docker\\n' >> \"$MARKER\"\n"
        ),
        tmp_path / "clean_log.sh": "#!/bin/sh\nprintf 'logs\\n' >> \"$MARKER\"\n",
    }
    for path, contents in stubs.items():
        path.write_text(contents, encoding="utf-8")
        path.chmod(0o755)
    environment = os.environ.copy()
    environment["MARKER"] = str(marker)
    environment["USER"] = f"clean-system-test-{os.getpid()}"

    result = subprocess.run([str(script)], check=False, env=environment)

    assert result.returncode == 3
    assert marker.read_text(encoding="utf-8").splitlines() == ["disk", "docker", "logs"]


@pytest.mark.parametrize(("percent", "expected"), [(95.0, 3), (99.0, 4)])
def test_unresolved_pressure_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    percent: float,
    expected: int,
) -> None:
    state = clean_disk_pressure.DiskState(
        1000, int(percent * 10), int((100 - percent) * 10), percent
    )
    snapshot = clean_disk_pressure.ActivitySnapshot(set(), set(), set(), True)
    monkeypatch.setattr(clean_disk_pressure, "disk_state", lambda _path: state)
    monkeypatch.setattr(
        clean_disk_pressure, "privileged_process_references", lambda: snapshot
    )
    monkeypatch.setattr(
        clean_disk_pressure, "cleanup_transcripts", lambda *args, **kwargs: (0, 0, 0)
    )
    monkeypatch.setattr(
        clean_disk_pressure, "complete_target_activity", lambda: (set(), set(), False)
    )
    monkeypatch.setattr(
        clean_disk_pressure, "cleanup_targets", lambda *args, **kwargs: (0, 0)
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "clean_disk_pressure.py",
            "--home",
            str(tmp_path),
            "--dev-root",
            str(tmp_path),
            "--tmp-root",
            str(tmp_path),
        ],
    )

    assert clean_disk_pressure.main() == expected


def _pin(path: Path) -> clean_disk_pressure.PinnedOutputFile:
    metadata = path.lstat()
    return clean_disk_pressure.PinnedOutputFile(
        path,
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_nlink,
        metadata.st_blocks * 512,
        clean_disk_pressure._installed_library(path.name),
    )


def test_pinned_cleanup_preserves_libraries_and_reconciles_hardlinks(
    tmp_path: Path,
) -> None:
    first, second, retained = (
        tmp_path / name for name in ("first", "second", "keep.so")
    )
    _old_file(first)
    os.link(first, second)
    _old_file(retained)
    pins = [_pin(path) for path in (first, second, retained)]
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    removed = []
    try:
        clean_disk_pressure._delete_pinned_output_files(
            fd, pins, time.time() - 86400, lambda: None, removed
        )
    finally:
        os.close(fd)
    assert removed == [str(first), str(second)]
    assert retained.read_bytes() == b"x" * 4096
    assert set(tmp_path.iterdir()) == {retained}


@pytest.mark.parametrize(
    "drift", ["added", "replaced", "modified", "hardlink", "symlink"]
)
def test_pinned_cleanup_rejects_drift_before_any_deletion(
    tmp_path: Path, drift: str
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _old_file(first)
    _old_file(second)
    pins = [_pin(first), _pin(second)]
    if drift == "added":
        _old_file(tmp_path / "unapproved")
    elif drift == "replaced":
        second.rename(tmp_path / "old-inode")
        _old_file(second)
        (tmp_path / "old-inode").unlink()
    elif drift == "modified":
        second.write_bytes(b"changed")
    elif drift == "hardlink":
        os.link(second, tmp_path.parent / f"other-link-{tmp_path.name}")
    elif drift == "symlink":
        second.unlink()
        second.symlink_to(first)
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    removed = []
    try:
        with pytest.raises(ValueError):
            clean_disk_pressure._delete_pinned_output_files(
                fd, pins, time.time() - 86400, lambda: None, removed
            )
    finally:
        os.close(fd)
    assert removed == []
    assert first.exists()


def test_pinned_cleanup_rejects_documents_without_touching_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "artifact"
    _old_file(source)
    original = _pin(source)
    document = clean_disk_pressure.PinnedOutputFile(
        tmp_path / "evidence.txt",
        original.device,
        original.inode,
        original.mode,
        original.uid,
        original.size,
        original.mtime_ns,
        original.nlink,
        original.allocated_bytes,
        False,
    )
    monkeypatch.setattr(clean_disk_pressure.os, "listdir", lambda _fd: ["evidence.txt"])
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(ValueError, match="not a supported pinned compiler file"):
            clean_disk_pressure._validate_pinned_output_files(
                fd, [document], time.time()
            )
    finally:
        os.close(fd)
    assert source.exists()


def test_pinned_cleanup_rechecks_each_file_and_never_deletes_new_names(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _old_file(first)
    _old_file(second)
    pins = [_pin(first), _pin(second)]
    calls = []

    def change_before_second_unlink() -> None:
        calls.append(True)
        if len(calls) == 2:
            _old_file(tmp_path / "unapproved")
            second.write_bytes(b"changed")

    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    removed = []
    try:
        with pytest.raises(ValueError, match="before unlink"):
            clean_disk_pressure._delete_pinned_output_files(
                fd, pins, time.time() - 86400, change_before_second_unlink, removed
            )
    finally:
        os.close(fd)
    assert removed == [str(first)]
    assert second.read_bytes() == b"changed"
    assert (tmp_path / "unapproved").exists()


def test_process_paths_join_device_roots_and_emit_bind_aliases() -> None:
    process_mounts = [("8:1", Path("/tmp/job"), Path("/workspace"))]
    host_mounts = [
        ("8:1", Path("/"), Path("/mnt/sda")),
        ("8:1", Path("/"), Path("/mnt/nvme-stage")),
        ("8:2", Path("/"), Path("/wrong-device")),
    ]
    assert clean_disk_pressure._process_paths(
        "/workspace/deps", process_mounts, host_mounts
    ) == {Path("/mnt/sda/tmp/job/deps"), Path("/mnt/nvme-stage/tmp/job/deps")}


def test_process_references_include_arguments_and_environment(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self/mountinfo").write_text("1 0 8:1 / /mnt/sda rw - ext4 disk rw\n")
    process = proc / "123"
    (process / "ns").mkdir(parents=True)
    (process / "fd").mkdir()
    (process / "root/workspace").mkdir(parents=True)
    (process / "root/workspace/env-link").symlink_to("/workspace/env-only")
    (process / "ns/mnt").touch()
    (process / "stat").write_text("123 (agent) S 1 2\n")
    (process / "mountinfo").write_text(
        "2 0 8:1 /tmp/job /workspace rw - ext4 disk rw\n"
    )
    (process / "cmdline").write_bytes(b"runner\0--input=/workspace/arguments-only\0")
    (process / "environ").write_bytes(b"OUTPUT=/workspace/env-link:/workspace/second\0")
    (process / "maps").write_bytes(b"")
    snapshot = clean_disk_pressure.process_references(proc)
    assert snapshot.complete
    assert snapshot.process_references >= {
        Path("/mnt/sda/tmp/job/arguments-only"),
        Path("/mnt/sda/tmp/job/env-only"),
        Path("/mnt/sda/tmp/job/second"),
    }


def test_mount_records_recognize_namespace_handles(tmp_path: Path) -> None:
    record = tmp_path / "mountinfo"
    record.write_text(
        "1 0 8:1 / /mnt/sda rw - ext4 disk rw\n"
        "2 1 0:4 mnt:[123] /run/ns rw - nsfs nsfs rw\n"
    )
    assert clean_disk_pressure._mount_records(record) == [
        ("8:1", Path("/"), Path("/mnt/sda"))
    ]


def test_process_symlinks_resolve_inside_process_root(tmp_path: Path) -> None:
    process = tmp_path / "process"
    (process / "root/workspace/deps").mkdir(parents=True)
    (process / "root/workspace/absolute").symlink_to("/workspace/deps")
    (process / "root/workspace/relative").symlink_to("deps")
    for name in ("absolute", "relative"):
        assert clean_disk_pressure._resolve_process_path(
            f"/workspace/{name}", process
        ) == Path("/workspace/deps")


def test_double_slash_paths_stay_in_linux_root() -> None:
    mounts = [("8:1", Path("/"), Path("/"))]
    assert clean_disk_pressure._process_paths("//tmp/job", mounts, mounts) == {
        Path("/tmp/job")
    }


def test_oversized_process_title_fragment_is_not_an_inspection_failure(
    tmp_path: Path,
) -> None:
    process = tmp_path / "process"
    (process / "root/workspace").mkdir(parents=True)
    raw = "/workspace/" + "x" * 512
    assert clean_disk_pressure._resolve_process_path(raw, process) == Path(raw)


def test_process_lookup_path_limit_still_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = tmp_path / "process"
    (process / "root/workspace").mkdir(parents=True)
    original = Path.lstat

    def lookup(path: Path):
        if path.name == "short":
            raise OSError(errno.ENAMETOOLONG, "accumulated lookup path too long")
        return original(path)

    monkeypatch.setattr(Path, "lstat", lookup)
    with pytest.raises(OSError) as error:
        clean_disk_pressure._resolve_process_path("/workspace/short", process)
    assert error.value.errno == errno.ENAMETOOLONG
