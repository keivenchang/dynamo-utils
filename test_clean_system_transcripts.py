from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _make_script_tree(tmp_path: Path) -> tuple[Path, Path]:
    source_root = Path(__file__).parent
    script = tmp_path / "clean_system.sh"
    shutil.copy2(source_root / "clean_system.sh", script)
    shutil.copy2(source_root / "build_guard.py", tmp_path / "build_guard.py")
    (tmp_path / "container").mkdir()
    stubs = {
        tmp_path / "clean_disk_pressure.py": "#!/bin/sh\nexit 0\n",
        tmp_path / "container" / "clean_old_local_dynamo_images.sh": "#!/bin/sh\nexit 0\n",
        tmp_path / "clean_log.sh": "#!/bin/sh\nexit 0\n",
    }
    for path, contents in stubs.items():
        path.write_text(contents, encoding="utf-8")
        path.chmod(0o755)
    home = tmp_path / "home"
    (home / ".claude" / "projects").mkdir(parents=True)
    (home / ".codex" / "sessions").mkdir(parents=True)
    return script, home


def test_clean_system_skips_transcripts_without_option(tmp_path: Path) -> None:
    script, home = _make_script_tree(tmp_path)
    marker = tmp_path / "args"
    pressure = tmp_path / "clean_disk_pressure.py"
    pressure.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$MARKER\"\n", encoding="utf-8")
    pressure.chmod(0o755)
    environment = os.environ.copy()
    environment["HOME"] = str(home)
    environment["MARKER"] = str(marker)
    environment["USER"] = f"clean-system-transcript-default-{os.getpid()}"

    result = subprocess.run([str(script)], check=False, env=environment, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert marker.read_text(encoding="utf-8").splitlines() == ["--skip-transcripts"]


def test_clean_system_forwards_transcript_retention_to_guarded_cleaner(tmp_path: Path) -> None:
    script, home = _make_script_tree(tmp_path)
    marker = tmp_path / "args"
    pressure = tmp_path / "clean_disk_pressure.py"
    pressure.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$MARKER\"\n", encoding="utf-8")
    pressure.chmod(0o755)
    environment = os.environ.copy()
    environment["HOME"] = str(home)
    environment["MARKER"] = str(marker)
    environment["USER"] = f"clean-system-transcript-test-{os.getpid()}"

    result = subprocess.run(
        [str(script), "--transcript-keep-days", "7", "--dry-run"],
        check=False,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert marker.read_text(encoding="utf-8").splitlines() == ["--transcript-keep-days", "7", "--dry-run"]


def test_transcript_keep_days_requires_positive_integer(tmp_path: Path) -> None:
    script, home = _make_script_tree(tmp_path)
    pressure = tmp_path / "clean_disk_pressure.py"
    shutil.copy2(Path(__file__).parent / "clean_disk_pressure.py", pressure)
    pressure.chmod(0o755)
    environment = os.environ.copy()
    environment["HOME"] = str(home)
    environment["USER"] = f"clean-system-transcript-invalid-{os.getpid()}"

    result = subprocess.run(
        [str(script), "--transcript-keep-days", "0", "--pressure-only"],
        check=False,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "transcript-keep-days must be a positive integer" in result.stderr
