# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import clean_disk_pressure as cleaner
import disk_guard_cron
from build_guard import BuildLease
from container import clean_local_dynamo_images as images


@pytest.fixture(autouse=True)
def clean_test_environment(monkeypatch):
    allowed = {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "PYTEST_CURRENT_TEST",
    }
    for key in list(os.environ):
        if key not in allowed:
            monkeypatch.delenv(key)


def rust_project(root: Path, git: bool = True) -> Path:
    root.mkdir(parents=True)
    if git:
        subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "Cargo.toml").write_text('[package]\nname = "probe"\nversion = "0.1.0"\n')
    target = root / "target"
    output = target / "debug" / "deps"
    output.mkdir(parents=True)
    (target / ".rustc_info.json").write_text('{"rustc_fingerprint": 1}')
    (output / "libprobe.rlib").write_bytes(b"x" * 8192)
    old = time.time() - 10 * 86400
    for path in (output, output / "libprobe.rlib"):
        os.utime(path, (old, old))
    return output


def discover(dev: Path, tmp: Path | None = None, paths: set[Path] | None = None):
    return cleaner.target_candidates(
        dev,
        paths or set(),
        set(),
        True,
        dev.stat().st_dev,
        set(),
        True,
        time.time(),
        24,
        False,
        tmp_root=tmp,
    )


def test_nested_and_temporary_rust_projects_without_source_deletion(tmp_path: Path):
    dev = tmp_path / "dev"
    nested = dev / "frontend-crates" / "frontend-crates__old"
    temporary = tmp_path / "tmp" / "probe"
    nested_output = rust_project(nested)
    temp_output = rust_project(temporary, git=False)
    actual = {candidate.target for candidate in discover(dev, temporary.parent)}
    assert actual == {nested_output, temp_output}
    assert (nested / "Cargo.toml").exists()


def test_standalone_cargo_target_discovery(tmp_path: Path):
    dev = tmp_path / "dev"
    dev.mkdir()
    target = tmp_path / "tmp" / "probe-target"
    output = target / "debug" / "deps"
    output.mkdir(parents=True)
    (output / "libprobe.rlib").write_bytes(b"x")
    (target / ".rustc_info.json").write_text('{"rustc_fingerprint": 1}')
    (target / "debug" / ".cargo-lock").touch()
    old = time.time() - 10 * 86400
    os.utime(output / "libprobe.rlib", (old, old))
    os.utime(output, (old, old))
    assert [row.target for row in discover(dev, target.parent)] == [output]


def test_plain_target_inside_temporary_evidence_is_discovered(tmp_path: Path):
    dev = tmp_path / "dev"
    dev.mkdir()
    temporary = tmp_path / "tmp" / "evidence"
    output = rust_project(temporary, git=False)
    (temporary / "Cargo.toml").unlink()
    (output.parent / ".cargo-lock").touch()
    assert [row.target for row in discover(dev, temporary.parent)] == [output]
    assert (temporary / "target" / ".rustc_info.json").exists()


def capacity_snapshot(tmp: Path, name: str, age_days: float) -> Path:
    path = tmp / "job" / "capacity" / name
    path.mkdir(parents=True)
    old = time.time() - age_days * 86400
    for name in ("nodes.json", "nodes.exit", "pods.json", "verdict.json"):
        file = path / name
        file.write_bytes(b"snapshot")
        os.utime(file, (old, old))
    os.utime(path, (old, old))
    return path


@pytest.fixture
def capacity_activity(monkeypatch):
    monkeypatch.setattr(cleaner, "filesystem_mount_points", lambda: (set(), True))
    monkeypatch.setattr(cleaner, "docker_mount_sources", lambda: (set(), True))
    monkeypatch.setattr(
        cleaner, "privileged_process_references",
        lambda: cleaner.ActivitySnapshot(set(), set(), set(), True),
    )
    monkeypatch.setattr(cleaner, "privileged_inode_open_state", lambda *_: (False, True))


def test_capacity_cleanup_skips_missing_tmp_root(tmp_path: Path, capacity_activity):
    root = tmp_path / "root"
    root.mkdir()
    missing = tmp_path / "missing"
    assert cleaner.cleanup_capacity_snapshots(missing, root, 1, 0, 100, False) == (0, 0)


def test_capacity_retains_newest_and_recent_snapshots(tmp_path: Path, capacity_activity):
    root = tmp_path / "tmp"
    old = capacity_snapshot(root, "20260101T010000000000", 3)
    retained = capacity_snapshot(root, "20260102T010000000000", 2)
    recent = capacity_snapshot(root, "20260103T010000000000", 0.1)
    result = cleaner.cleanup_capacity_snapshots(root, root, 1, 2, 100, True)
    assert result[0] == 1 and old.exists()
    result = cleaner.cleanup_capacity_snapshots(root, root, 1, 2, 100, False)
    assert result[0] == 1 and result[1] > 0
    assert not old.exists() and retained.exists() and recent.exists()


def test_capacity_age_and_budget(tmp_path: Path, capacity_activity):
    root = tmp_path / "tmp"
    old = capacity_snapshot(root, "20260101T010000000000", 3)
    other = capacity_snapshot(root, "20260102T010000000000", 2)
    recent = capacity_snapshot(root, "20260103T010000000000", 0.1)
    assert cleaner.cleanup_capacity_snapshots(root, root, 1, 0, 0, False) == (0, 0)
    assert cleaner.cleanup_capacity_snapshots(root, root, 1, 0, 1, False)[0] == 1
    assert old.exists() and not other.exists() and recent.exists()
    assert cleaner.cleanup_capacity_snapshots(root, root, 1, 0, 100, False)[0] == 1
    assert recent.exists()


@pytest.mark.parametrize("protection", ["active", "incomplete", "unknown", "symlink", "mount"])
def test_capacity_safety_preserves_snapshot(tmp_path: Path, capacity_activity, monkeypatch, protection):
    root = tmp_path / "tmp"
    old = capacity_snapshot(root, "20260101T010000000000", 3)
    if protection == "active":
        monkeypatch.setattr(
            cleaner, "privileged_process_references",
            lambda: cleaner.ActivitySnapshot(set(), set(), {old.parent.parent / "running"}, True),
        )
    elif protection == "incomplete":
        monkeypatch.setattr(
            cleaner, "privileged_process_references",
            lambda: cleaner.ActivitySnapshot(set(), set(), set(), False),
        )
    elif protection == "unknown":
        (old / "lease.pid").touch()
    elif protection == "mount":
        monkeypatch.setattr(cleaner, "filesystem_mount_points", lambda: ({old}, True))
    else:
        (old / "pods.json").unlink()
        (old / "pods.json").symlink_to(old / "nodes.json")
    assert cleaner.cleanup_capacity_snapshots(root, root, 1, 0, 100, False)[0] == 0
    assert (old / "nodes.json").exists()


def test_capacity_late_open_file_is_preserved(tmp_path: Path, capacity_activity, monkeypatch):
    root = tmp_path / "tmp"
    old = capacity_snapshot(root, "20260101T010000000000", 3)
    monkeypatch.setattr(cleaner, "privileged_inode_open_state", lambda *_: (True, True))
    assert cleaner.cleanup_capacity_snapshots(root, root, 1, 0, 100, False) == (0, 0)
    assert (old / "pods.json").exists()


@pytest.mark.parametrize("replacement", [False, True])
def test_capacity_preflights_all_files_and_parent(
    tmp_path: Path, capacity_activity, monkeypatch, replacement
):
    root = tmp_path / "tmp"
    old = capacity_snapshot(root, "20260101T010000000000", 3)
    names = {p.name for p in old.iterdir()}
    pods_inode = (old / "pods.json").stat().st_ino
    protected = root / "protected"
    changed = False

    def inspect(device, inode):
        nonlocal changed
        if replacement and not changed:
            old.rename(protected)
            old.symlink_to(protected)
            changed = True
        return (not replacement and inode == pods_inode), True

    monkeypatch.setattr(cleaner, "privileged_inode_open_state", inspect)
    assert cleaner.cleanup_capacity_snapshots(root, root, 1, 0, 100, False) == (0, 0)
    assert {p.name for p in (protected if replacement else old).iterdir()} == names


def test_capacity_disappearing_parent_is_skipped(
    tmp_path: Path, capacity_activity, monkeypatch
):
    root = tmp_path / "tmp"
    old = capacity_snapshot(root, "20260101T010000000000", 3)
    missing = root / "disappeared"
    original = Path.iterdir

    def entries(path):
        if path == root:
            return iter([missing, old.parent.parent])
        return original(path)

    monkeypatch.setattr(Path, "iterdir", entries)
    assert cleaner.cleanup_capacity_snapshots(root, root, 1, 0, 100, True)[0] == 1
    assert old.exists()


def test_symlinks_and_tracked_files_are_not_candidates(tmp_path: Path):
    dev = tmp_path / "dev"
    repo = dev / "dynamo" / "dynamo__old"
    rust_project(repo)
    subprocess.run(
        ["git", "-C", str(repo), "add", "-f", "target/debug/deps/libprobe.rlib"],
        check=True,
    )
    (dev / "dynamo__alias").symlink_to(repo)
    assert discover(dev) == []


def test_shared_builds_coexist_and_exclude_cleanup(tmp_path: Path):
    with BuildLease(tmp_path), BuildLease(tmp_path):
        with pytest.raises(BlockingIOError):
            with BuildLease(tmp_path, exclusive=True):
                pass
    with BuildLease(tmp_path, exclusive=True):
        with pytest.raises(BlockingIOError):
            with BuildLease(tmp_path):
                pass


def test_cleanup_keeps_shared_libraries_and_cargo_lock(tmp_path: Path, monkeypatch):
    repo = tmp_path / "dev" / "dynamo__old"
    output = rust_project(repo)
    library = output / "libinstalled.so"
    library.write_bytes(b"installed")
    old = time.time() - 10 * 86400
    os.utime(library, (old, old))
    os.utime(output, (old, old))
    candidate = discover(repo.parent)[0]
    monkeypatch.setattr(
        cleaner, "complete_target_activity", lambda: (set(), set(), True)
    )
    monkeypatch.setattr(cleaner, "filesystem_mount_points", lambda: (set(), True))
    assert cleaner._remove_target(candidate, False)
    assert not (output / "libprobe.rlib").exists()
    assert library.read_bytes() == b"installed"
    assert (repo / "target" / "debug" / ".cargo-lock").exists()
    assert (repo / "Cargo.toml").exists()
    assert not Path(str(output) + ".lock").exists()


def test_cargo_lock_and_late_activity_prevent_deletion(tmp_path: Path, monkeypatch):
    repo = tmp_path / "dev" / "dynamo__old"
    output = rust_project(repo)
    candidate = discover(repo.parent)[0]
    lock = output.parent / ".cargo-lock"
    with lock.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        assert not cleaner._remove_target(candidate, False)
    monkeypatch.setattr(
        cleaner,
        "complete_target_activity",
        lambda: ({output / "libprobe.rlib"}, set(), True),
    )
    monkeypatch.setattr(cleaner, "filesystem_mount_points", lambda: (set(), True))
    assert not cleaner._remove_target(candidate, False)
    assert (output / "libprobe.rlib").exists()


def test_recent_writes_without_directory_mtime_change_prevent_deletion(
    tmp_path: Path, monkeypatch
):
    repo = tmp_path / "dev" / "dynamo__old"
    output = rust_project(repo)
    candidate = discover(repo.parent)[0]
    (output / "libprobe.rlib").write_bytes(b"new build")
    monkeypatch.setattr(
        cleaner, "complete_target_activity", lambda: (set(), set(), True)
    )
    monkeypatch.setattr(cleaner, "filesystem_mount_points", lambda: (set(), True))
    assert not cleaner._remove_target(candidate, False)


def test_container_paths_map_to_host_resources():
    mappings = [
        (
            Path("/workspace/target/.vllm"),
            Path("/home/user/dev/dynamo/tree/target/.vllm"),
        ),
        (Path("/workspace"), Path("/home/user/dev/dynamo/tree")),
    ]
    assert cleaner._process_path(
        "/workspace/target/.vllm/debug/deps/x", mappings
    ) == Path("/home/user/dev/dynamo/tree/target/.vllm/debug/deps/x")


def test_free_space_floor_triggers_even_below_percentage_threshold(
    tmp_path: Path, monkeypatch
):
    state = cleaner.DiskState(1000 * 1024**3, 850 * 1024**3, 150 * 1024**3, 85)
    monkeypatch.setattr(cleaner, "disk_state", lambda path: state)
    monkeypatch.setattr(
        cleaner, "complete_target_activity", lambda: (set(), set(), True)
    )
    called = []
    monkeypatch.setattr(
        cleaner, "cleanup_targets", lambda *a, **kw: called.append(kw) or (0, 0)
    )
    monkeypatch.setattr(
        sys, "argv", ["clean_disk_pressure.py", "--pressure-only", "--skip-transcripts",
                    "--tmp-root", str(tmp_path)]
    )
    assert cleaner.main() == 3
    assert called[0]["target_free_gib"] == 350


def test_old_images_only_and_all_container_references_protected():
    now = datetime.datetime.now(datetime.timezone.utc)

    def image(identity, days, tag):
        return {
            "Id": identity,
            "Created": (now - datetime.timedelta(days=days)).isoformat(),
            "RepoTags": [tag],
        }

    rows = [
        image("used", 60, "dynamo:1234567-vllm-dev"),
        image("old", 45, "dynamo:2345678-vllm-dev"),
        image("young", 10, "dynamo:3456789-vllm-dev"),
        image("foreign", 70, "vllm:1234567-vllm-dev"),
        image("mixed", 60, "dynamo:4567890-vllm-dev"),
    ]
    rows[-1]["RepoTags"].append("other:important")
    assert [row["Id"] for row in images.candidates(rows, {"used"}, now, 30, 1)] == [
        "old"
    ]


def test_scheduler_timeout_stops_descendants(tmp_path: Path):
    marker = tmp_path / "child.json"
    command = [
        sys.executable,
        "-c",
        (
            "import subprocess,sys,json; "
            "p=subprocess.Popen([sys.executable,'-c',"
            "'import signal; signal.pause()']); "
            "open(sys.argv[1],'w').write(json.dumps(p.pid)); p.wait()"
        ),
        str(marker),
    ]
    with (tmp_path / "run.log").open("wb") as log:
        assert disk_guard_cron.run_cleanup(command, log, timeout=1) == 124
    pid = json.loads(marker.read_text())
    status = Path(f"/proc/{pid}/stat")
    assert not status.exists() or status.read_text().split()[2] == "Z"


def test_scheduler_log_is_bounded(tmp_path: Path):
    with (tmp_path / "run.log").open("w+b") as log:
        log.write(b"x" * 11 * 1024**2)
        disk_guard_cron.trim_log(log)
        assert log.seek(0, os.SEEK_END) == 1024**2


def test_idle_source_context_does_not_hide_unused_output(tmp_path: Path):
    repo = tmp_path / "dev" / "dynamo__idle"
    output = rust_project(repo)
    assert [row.target for row in discover(repo.parent, paths={repo})] == [output]
    assert discover(repo.parent, paths={output / "libprobe.rlib"}) == []


def test_shared_libraries_alone_do_not_consume_cleanup_limit(tmp_path: Path):
    repo = tmp_path / "dev" / "dynamo__old"
    output = rust_project(repo)
    (output / "libprobe.rlib").rename(output / "libinstalled.so")
    old = time.time() - 10 * 86400
    os.utime(output, (old, old))
    assert discover(repo.parent) == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_invalid_age_never_selects_output(tmp_path: Path, value: float):
    repo = tmp_path / "dev" / "dynamo__old"
    rust_project(repo)
    with pytest.raises(ValueError):
        cleaner.target_candidates(
            repo.parent,
            set(),
            set(),
            True,
            repo.stat().st_dev,
            set(),
            True,
            time.time(),
            value,
            False,
        )


def test_main_rejects_nonfinite_age(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["clean_disk_pressure.py", "--target-min-age-hours", "nan"]
    )
    assert cleaner.main() == 2


def test_measured_zombies_do_not_block_process_inspection(tmp_path: Path):
    proc = tmp_path / "123"
    proc.mkdir()
    (proc / "stat").write_text("123 (exited worker) Z 0")
    snapshot = cleaner.process_references(tmp_path)
    assert snapshot.complete
    assert not snapshot.process_references


def test_new_container_reference_blocks_image_deletion(monkeypatch):
    row = {
        "Id": "old-id",
        "RepoTags": ["dynamo:1234567-vllm-dev"],
        "Created": "2020-01-01T00:00:00+00:00",
    }
    calls = []

    def docker(*args):
        calls.append(args)
        if args[:2] == ("image", "ls"):
            return "old-id"
        if args[:2] == ("image", "inspect"):
            return json.dumps([row])
        raise AssertionError(args)

    used = iter([set(), {"old-id"}])
    monkeypatch.setattr(images, "docker", docker)
    monkeypatch.setattr(images, "container_images", lambda: next(used))
    monkeypatch.setattr(sys, "argv", ["clean_local_dynamo_images.py", "--retain", "0"])
    assert images.main() == 0
    assert not any(call[:2] == ("image", "rm") for call in calls)


def test_build_guard_lock_path_does_not_create_lease(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "build_guard.py"),
            "--repo",
            str(tmp_path),
            "--lock-path",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert Path(result.stdout.strip()) == tmp_path / ".build-cleanup.lock"
    assert not (tmp_path / ".build-cleanup.lock").exists()


def test_build_guard_space_check_blocks_command_without_running_it(tmp_path: Path):
    marker = tmp_path / "command-ran"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "build_guard.py"),
            "--repo",
            str(tmp_path),
            "--min-free-gib",
            "1000000000",
            "--",
            sys.executable,
            "-c",
            "import sys; open(sys.argv[1], 'w').close()",
            str(marker),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 3
    assert not marker.exists()
    with BuildLease(tmp_path, exclusive=True):
        pass


def test_readonly_cargo_lock_allows_cleanup(tmp_path: Path, monkeypatch):
    repo = tmp_path / "dev" / "dynamo__old"
    output = rust_project(repo)
    lock = output.parent / ".cargo-lock"
    lock.touch(mode=0o444)
    candidate = discover(repo.parent)[0]
    monkeypatch.setattr(
        cleaner, "complete_target_activity", lambda: (set(), set(), True)
    )
    monkeypatch.setattr(cleaner, "filesystem_mount_points", lambda: (set(), True))
    assert cleaner._remove_target(candidate, False)
    assert lock.exists()
    assert not (output / "libprobe.rlib").exists()


def test_root_lease_creation_preserves_repo_owner(tmp_path: Path, monkeypatch):
    import_guard = sys.modules[BuildLease.__module__]
    monkeypatch.setattr(import_guard.os, "geteuid", lambda: 0)
    changed = []
    monkeypatch.setattr(
        import_guard.os, "fchown", lambda fd, uid, gid: changed.append((uid, gid))
    )
    with BuildLease(tmp_path):
        pass
    assert changed == [(tmp_path.stat().st_uid, tmp_path.stat().st_gid)]


def test_elevation_starts_without_parent_mutation_lease(tmp_path: Path, monkeypatch):
    repo = tmp_path / "dev" / "dynamo__old"
    rust_project(repo)
    candidate = discover(repo.parent)[0]
    monkeypatch.setattr(cleaner.os, "geteuid", lambda: 1)
    monkeypatch.setattr(cleaner, "_writable_output", lambda output: False)
    called = []

    def elevated(actual):
        assert actual == candidate
        with BuildLease(repo, exclusive=True):
            called.append(actual)
        return True

    monkeypatch.setattr(cleaner, "_elevated_remove_target", elevated)
    assert cleaner._remove_target(candidate, False)
    assert called == [candidate]


def test_supervisor_identity_change_blocks_mutation(monkeypatch):
    monkeypatch.setattr(cleaner, "_process_identity", lambda pid: "new-start")
    with pytest.raises(ProcessLookupError):
        cleaner._require_supervisor((123, "old-start"))


def test_elevated_secondary_scope_is_owned_and_on_secondary_device(monkeypatch):
    secondary = Path('/mnt/sda/tmp')
    repo = secondary / 'job' / 'cargo-target'
    real_stat = Path.stat
    real_lstat = Path.lstat
    owner = Path(cleaner.__file__).stat().st_uid
    secondary_device = Path('/').stat().st_dev + 1

    def fake_metadata(path, *args, **kwargs):
        if path in (secondary, repo):
            return os.stat_result([0o40755, 123, secondary_device, 1, owner, 0, 0, 0, 0, 0])
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'stat', fake_metadata)
    monkeypatch.setattr(Path, 'lstat', lambda p: fake_metadata(p) if p == secondary else real_lstat(p))
    monkeypatch.setattr(Path, 'resolve', lambda p: p)
    monkeypatch.setattr(os, 'geteuid', lambda: 0)
    monkeypatch.setattr(cleaner, '_require_supervisor', lambda _: None)
    removed = []
    monkeypatch.setattr(cleaner, '_remove_target', lambda candidate, *a: removed.append(candidate) or True)
    payload = dict(repo=str(repo), target=str(repo / 'debug/deps'), size=1, age_hours=48,
                   device=secondary_device, inode=123, mtime=0, target_root=str(repo),
                   profile=str(repo / 'debug'), cutoff=time.time() - 48 * 3600,
                   supervisor=[1, 'test'])
    monkeypatch.setattr(sys, 'argv', ['clean_disk_pressure.py', '--remove-target', json.dumps(payload)])
    assert cleaner.main() == 0
    assert len(removed) == 1
    secondary_device = Path('/').stat().st_dev
    with pytest.raises(PermissionError):
        cleaner.main()
    assert len(removed) == 1


def test_bad_capacity_snapshot_does_not_block_older_valid_snapshot(tmp_path, capacity_activity):
    root = tmp_path / 'tmp'
    old = capacity_snapshot(root, '20260101T010000000000', 3)
    bad = capacity_snapshot(root, '20260102T010000000000', 2)
    (bad / 'nodes.exit').unlink()
    assert cleaner.cleanup_capacity_snapshots(root, root, 1, 0, 100, False)[0] == 1
    assert not old.exists() and bad.exists()


def test_secondary_scheduler_routes_only_validated_scratch(monkeypatch):
    monkeypatch.setattr(disk_guard_cron.socket, 'gethostname', lambda: 'keivenc-linux1')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, '/mnt/sda/tmp\n', ''))
    command = disk_guard_cron.secondary_cleanup_command(True, True)
    assert command[command.index('--root-path') + 1] == '/mnt/sda/tmp'
    assert command[command.index('--tmp-root') + 1] == '/mnt/sda/tmp'
    assert '--skip-transcripts' in command and '--maintenance' in command and '--dry-run' in command
