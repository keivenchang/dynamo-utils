#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
import time
from pathlib import Path

import clean_disk_pressure
import pytest


def _old_file(path: Path, size: int = 4096) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    old = time.time() - 3 * 86400
    os.utime(path, (old, old))


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
        clean_disk_pressure, "privileged_inode_open_state", lambda _device, _inode: (False, True)
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
        clean_disk_pressure, "privileged_inode_open_state", lambda _device, _inode: (True, True)
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
    candidates = clean_disk_pressure.transcript_candidates(tmp_path, cutoff, set(), set())
    by_name = {candidate.path.name: candidate for candidate in candidates}

    replaced.unlink()
    _old_file(replaced)
    replaced_result = clean_disk_pressure._remove_transcript_candidate(
        by_name["replaced.jsonl"], cutoff, set(), set(), dry_run=False
    )
    open_inode = {(by_name["opened.jsonl"].device, by_name["opened.jsonl"].inode)}
    opened_result = clean_disk_pressure._remove_transcript_candidate(
        by_name["opened.jsonl"], cutoff, set(), open_inode, dry_run=False
    )

    assert replaced_result == (False, 0)
    assert opened_result == (False, 0)
    assert replaced.exists()
    assert opened.exists()


def test_target_candidates_require_git_repo_and_skip_active_repo(tmp_path: Path) -> None:
    inactive_repo = tmp_path / "dynamo__inactive"
    active_repo = tmp_path / "frontend-crates__active"
    recently_built_repo = tmp_path / "dynamo__recent-build"
    not_repo = tmp_path / "dynamo__not-repo"
    for repo in (inactive_repo, active_repo, recently_built_repo):
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for repo in (inactive_repo, active_repo, recently_built_repo, not_repo):
        target = repo / "target"
        target.mkdir(parents=True)
        _old_file(target / "artifact")
        old = time.time() - 3 * 86400
        os.utime(target, (old, old))
    os.utime(recently_built_repo / "target" / "artifact", None)

    candidates = clean_disk_pressure.target_candidates(
        tmp_path,
        {active_repo / "src"},
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
    _old_file(repo / "target" / "artifact")
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
        {(repo / "target" / "mounted").resolve()},
        True,
        time.time(),
        24,
        False,
    )

    assert incomplete == []
    assert parent_mounted == []
    assert wrong_filesystem == []
    assert nested_mount == []


def test_remove_target_rejects_replaced_directory(tmp_path: Path) -> None:
    repo = tmp_path / "dynamo__candidate"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _old_file(repo / "target" / "old-artifact")
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
    _old_file(repo / "target" / "replacement-artifact")
    os.utime(repo / "target", (old + 10, old + 10))

    assert not clean_disk_pressure._remove_target(candidates[0], dry_run=False)
    assert (repo / "target" / "replacement-artifact").exists()


def test_remove_target_rechecks_identity_after_activity_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "dynamo__candidate"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _old_file(repo / "target" / "old-artifact")
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
        _old_file(repo / "target" / "replacement-artifact")
        os.utime(repo / "target", (old + 10, old + 10))
        return set(), set(), True

    monkeypatch.setattr(
        clean_disk_pressure, "complete_target_activity", replace_during_activity_scan
    )

    assert not clean_disk_pressure._remove_target(candidate, dry_run=False)
    assert (repo / "target" / "replacement-artifact").exists()


def test_path_boundary_does_not_treat_similar_prefix_as_child() -> None:
    assert clean_disk_pressure._is_within(Path("/tmp/repo/target/a"), Path("/tmp/repo"))
    assert not clean_disk_pressure._is_within(Path("/tmp/repo-old/target"), Path("/tmp/repo"))


def test_clean_system_retains_pressure_failure_after_other_cleanup(tmp_path: Path) -> None:
    source_root = Path(__file__).parent
    script = tmp_path / "clean_system.sh"
    shutil.copy2(source_root / "clean_system.sh", script)
    (tmp_path / "container").mkdir()
    marker = tmp_path / "steps"
    stubs = {
        tmp_path / "clean_disk_pressure.py": "#!/bin/sh\nprintf 'disk\\n' >> \"$MARKER\"\nexit 3\n",
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
    state = clean_disk_pressure.DiskState(1000, int(percent * 10), int((100 - percent) * 10), percent)
    snapshot = clean_disk_pressure.ActivitySnapshot(set(), set(), set(), True)
    monkeypatch.setattr(clean_disk_pressure, "disk_state", lambda _path: state)
    monkeypatch.setattr(clean_disk_pressure, "privileged_process_references", lambda: snapshot)
    monkeypatch.setattr(clean_disk_pressure, "cleanup_transcripts", lambda *args, **kwargs: (0, 0, 0))
    monkeypatch.setattr(clean_disk_pressure, "complete_target_activity", lambda: (set(), set(), False))
    monkeypatch.setattr(clean_disk_pressure, "cleanup_targets", lambda *args, **kwargs: (0, 0))
    monkeypatch.setattr(
        "sys.argv",
        [
            "clean_disk_pressure.py",
            "--home",
            str(tmp_path),
            "--dev-root",
            str(tmp_path),
        ],
    )

    assert clean_disk_pressure.main() == expected
