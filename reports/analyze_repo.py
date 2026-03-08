#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Repository Analyzer

Analyzes a git repository over time using read-only plumbing commands (no checkout).

Supported analyses (use one or more flags):
  --python    Pytest test growth: files, functions, markers
  --loc       Lines of code: Python, Rust, total
  --rust      Rust test growth: #[test] / #[tokio::test] counts
  --commits   Commit volume per period
"""

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ===== Pytest defaults =====

TRACKED_MARKERS_DEFAULT = [
    "pre_merge",
    "post_merge",
    "parallel",
    "integration",
    "gpu_0",
    "gpu_1",
    "gpu_2",
    "skip",
    "unmarked",
]

SKIP_MARKERS = {"skip", "skipif"}

_RE_TEST_FUNC = re.compile(r"^\s*(?:async\s+)?def\s+(test_\w+)\s*\(")
_RE_PYTEST_MARK = re.compile(r"^\s*@pytest\.mark\.(\w+)")

# ===== LoC extensions =====

PYTHON_EXTS = {".py"}
RUST_EXTS = {".rs"}
DOC_EXTS = {".md", ".rst", ".txt", ".html", ".htm", ".xml", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".csv"}

# Cached per-repo empty tree hash (varies between SHA-1 and SHA-256 repos)
_empty_tree_cache: Dict[str, str] = {}

# Commits that must never be used as snapshots (e.g. contain large binaries
# that cause hangs, or represent broken intermediate states).
# Use short prefixes; matching is prefix-based.
BLACKLISTED_COMMITS: Set[str] = {
    "222c2e85c",  # adds 1.5 MB binary .gz file, causes checkout hangs (see .cursorrules)
}


# ---------------------------------------------------------------------------
# Disk cache — results keyed by commit SHA are immutable and safe to cache
# ---------------------------------------------------------------------------

CACHE_DIR = Path.home() / ".cache" / "dynamo-utils" / "analyze_repo"


class AnalysisCache:
    """Simple JSON-file cache keyed by (category, commit_sha, params_hash).

    Git commit SHAs are immutable, so cached analysis results never go stale.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _params_hash(*parts: str) -> str:
        if not parts:
            return ""
        combined = "|".join(parts)
        return hashlib.sha256(combined.encode()).hexdigest()[:8]

    def _path(self, category: str, commit: str, *params: str) -> Path:
        suffix = ""
        if params:
            suffix = "_" + self._params_hash(*params)
        return CACHE_DIR / category / f"{commit[:12]}{suffix}.json"

    def get(self, category: str, commit: str, *params: str) -> Optional[Dict]:
        if not self.enabled:
            return None
        path = self._path(category, commit, *params)
        if not path.exists():
            self.misses += 1
            return None
        with open(path) as f:
            self.hits += 1
            return json.load(f)

    def put(self, category: str, commit: str, data: Dict, *params: str) -> None:
        if not self.enabled:
            return
        path = self._path(category, commit, *params)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, separators=(",", ":"))

    def clear(self) -> int:
        count = 0
        if CACHE_DIR.exists():
            for f in CACHE_DIR.rglob("*.json"):
                f.unlink()
                count += 1
        return count

    def summary(self) -> str:
        total = self.hits + self.misses
        if total == 0:
            return "Cache: no lookups"
        return f"Cache: {self.hits}/{total} hits, {self.misses} computed"


_cache = AnalysisCache(enabled=True)


# ---------------------------------------------------------------------------
# Git helpers (read-only plumbing, no checkout)
# ---------------------------------------------------------------------------


def run_git(
    args: List[str], repo_path: str, timeout: int = 60
) -> Tuple[int, str, str]:
    """Run a git command and return (exit_code, stdout, stderr)."""
    cmd = ["git", "-C", repo_path] + args
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(
            f"  Command timed out after {timeout}s: {' '.join(cmd)}", file=sys.stderr
        )
        return -1, "", "Timeout"


def _is_blacklisted(sha: str) -> bool:
    """Check if a commit SHA matches any blacklisted prefix."""
    for prefix in BLACKLISTED_COMMITS:
        if sha.startswith(prefix):
            return True
    return False


def get_commit_at_date(repo_path: str, target_date: str, branch: str = "main") -> Optional[str]:
    """Get the latest commit hash at or before a specific date (plumbing).

    Restricts to *branch* (default: main) to avoid picking up commits
    from side branches (e.g. fern docs branch) that have incomplete trees.
    Skips commits in BLACKLISTED_COMMITS by fetching a few candidates.
    """
    rc, stdout, _ = run_git(
        ["rev-list", "-n", "10", f"--before={target_date} 23:59:59", branch],
        repo_path,
    )
    if rc != 0 or not stdout.strip():
        return None
    for sha in stdout.strip().splitlines():
        sha = sha.strip()
        if sha and not _is_blacklisted(sha):
            return sha
    return None


def list_test_files_at_commit(
    repo_path: str, commit: str, test_dir: str
) -> List[str]:
    """List test files (test_*.py, *_test.py) at a commit.

    If test_dir is "." or empty, scans the entire tree.
    """
    cmd = ["ls-tree", "-r", "--name-only", commit]
    if test_dir and test_dir != ".":
        cmd += ["--", test_dir + "/"]
    rc, stdout, _ = run_git(cmd, repo_path)
    if rc != 0 or not stdout.strip():
        return []
    paths = []
    for path in stdout.strip().split("\n"):
        if not path.endswith(".py"):
            continue
        basename = os.path.basename(path)
        if basename.startswith("test_") or basename.endswith("_test.py"):
            paths.append(path)
    return paths


def read_file_at_commit(repo_path: str, commit: str, path: str) -> Optional[str]:
    """Read file contents at a specific commit via git-show (plumbing)."""
    rc, stdout, _ = run_git(["show", f"{commit}:{path}"], repo_path)
    if rc != 0:
        return None
    return stdout


# ---------------------------------------------------------------------------
# Pytest collection via pytest --collect-only (preferred, accurate)
# ---------------------------------------------------------------------------


def _get_repo_head(repo_path: str) -> Optional[str]:
    """Get the current HEAD commit SHA."""
    rc, stdout, _ = run_git(["rev-parse", "HEAD"], repo_path)
    if rc != 0:
        return None
    return stdout.strip()


def _run_pytest_collect(
    repo_path: str, marker_expr: Optional[str] = None, timeout: int = 180
) -> Tuple[bool, int, int]:
    """Run pytest --collect-only and return (success, test_count, file_count).

    Parses the summary line like "526/1871 tests collected (1345 deselected)"
    and counts unique file paths from the test IDs listed above it.
    """
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q", "--no-header"]
    if marker_expr:
        cmd += ["-m", marker_expr]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False,
            timeout=timeout, cwd=repo_path,
        )
    except subprocess.TimeoutExpired:
        return (False, 0, 0)

    stdout = result.stdout
    test_count = 0
    file_paths: Set[str] = set()

    for line in stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if "::" in line and "collected" not in line and "deselected" not in line:
            # Skip pytest-mypy plugin items (e.g. "path/to/file.py::mypy")
            parts = line.split("::")
            if len(parts) >= 2 and parts[-1].strip() in ("mypy", "mypy-status"):
                continue
            file_path = parts[0]
            if file_path.endswith(".py"):
                file_paths.add(file_path)
                test_count += 1

    # exit code 5 = no tests collected (valid for marker filtering)
    if test_count == 0 and result.returncode == 5:
        return (True, 0, 0)

    if test_count > 0 or file_paths:
        return (True, test_count, len(file_paths))

    return (False, 0, 0)


def analyze_pytests_via_collect(
    repo_path: str,
    tracked_markers: List[str],
) -> Optional[Dict]:
    """Count pytest tests using pytest --collect-only.

    Runs once for total count, then once per marker.  Returns dict with same
    shape as analyze_pytests_snapshot, or None if collection fails.
    """
    head_sha = _get_repo_head(repo_path)
    if head_sha:
        params = (",".join(sorted(tracked_markers)),)
        cached = _cache.get("pytests_collect", head_sha, *params)
        if cached is not None:
            print("    (cached)", file=sys.stderr, flush=True)
            return cached

    print("    total ...", end="", file=sys.stderr, flush=True)
    ok, total_tests, total_files = _run_pytest_collect(repo_path)
    if not ok:
        print(" FAILED", file=sys.stderr)
        return None
    print(f" {total_tests} tests, {total_files} files", file=sys.stderr, flush=True)

    real_tracked = [m for m in tracked_markers if m != "unmarked"]
    marker_counts: Dict[str, int] = {}

    for marker in real_tracked:
        print(f"    -m {marker} ...", end="", file=sys.stderr, flush=True)
        ok_m, count, _ = _run_pytest_collect(repo_path, marker_expr=marker)
        if not ok_m:
            print(" FAILED", file=sys.stderr)
            return None
        marker_counts[marker] = count
        print(f" {count}", file=sys.stderr, flush=True)

    if "unmarked" in tracked_markers and real_tracked:
        not_expr = " or ".join(real_tracked)
        print("    -m unmarked ...", end="", file=sys.stderr, flush=True)
        ok_u, unmarked, _ = _run_pytest_collect(
            repo_path, marker_expr=f"not ({not_expr})"
        )
        if ok_u:
            marker_counts["unmarked"] = unmarked
            print(f" {unmarked}", file=sys.stderr, flush=True)
        else:
            marker_counts["unmarked"] = 0
            print(" FAILED (set to 0)", file=sys.stderr, flush=True)

    result = {
        "test_files": total_files,
        "total_tests": total_tests,
        "markers": marker_counts,
    }
    # Cache collect results keyed by HEAD commit
    head_sha = _get_repo_head(repo_path)
    if head_sha:
        params = (",".join(sorted(tracked_markers)),)
        _cache.put("pytests_collect", head_sha, result, *params)
    return result


def collect_at_commit(
    repo_path: str,
    commit: str,
    tracked_markers: List[str],
) -> Optional[Dict]:
    """Try pytest --collect-only at a specific commit via a temporary git worktree.

    Creates a detached worktree, runs collection there, cleans up.
    Returns dict compatible with analyze_pytests_snapshot, or None on failure.
    """
    # Fast path: check cache before creating a worktree
    params = (",".join(sorted(tracked_markers)),)
    cached = _cache.get("pytests_collect", commit, *params)
    if cached is not None:
        return cached

    worktree_dir = tempfile.mkdtemp(prefix="analyze_wt_")
    try:
        rc, _, stderr = run_git(
            ["worktree", "add", "--detach", "--force", worktree_dir, commit],
            repo_path,
            timeout=60,
        )
        if rc != 0:
            print(f"worktree failed: {stderr.strip()}", file=sys.stderr)
            return None

        result = analyze_pytests_via_collect(worktree_dir, tracked_markers)
        return result
    finally:
        run_git(["worktree", "remove", "--force", worktree_dir], repo_path, timeout=30)
        if os.path.exists(worktree_dir):
            shutil.rmtree(worktree_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Pytest marker parsing (AST-based with regex fallback)
# ---------------------------------------------------------------------------


def _extract_marker_name(decorator: ast.expr) -> Optional[str]:
    """Extract marker name from @pytest.mark.foo or @pytest.mark.foo(...)."""
    node = decorator
    if isinstance(node, ast.Call):
        node = node.func
    if not isinstance(node, ast.Attribute):
        return None
    marker_name = node.attr
    mid = node.value
    if not isinstance(mid, ast.Attribute) or mid.attr != "mark":
        return None
    top = mid.value
    if not isinstance(top, ast.Name) or top.id != "pytest":
        return None
    return marker_name


def _extract_pytestmark_markers(node: ast.Assign) -> Set[str]:
    """Extract markers from module-level pytestmark = ... assignments."""
    markers: Set[str] = set()
    if not any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
        return markers
    value = node.value
    items = value.elts if isinstance(value, ast.List) else [value]
    for item in items:
        name = _extract_marker_name(item)
        if name:
            markers.add(name)
    return markers


def _decorators_to_markers(decorator_list: List[ast.expr]) -> Set[str]:
    """Collect pytest.mark.* names from a decorator list."""
    markers: Set[str] = set()
    for dec in decorator_list:
        name = _extract_marker_name(dec)
        if name:
            markers.add(name)
    return markers


def parse_tests_and_markers(content: str) -> List[Tuple[str, Set[str]]]:
    """
    Parse a Python test file and return (test_name, {markers}) tuples.

    Handles module-level pytestmark, class-level, and function-level markers.
    Falls back to regex if the file has syntax errors.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _parse_tests_and_markers_regex(content)

    module_markers: Set[str] = set()
    results: List[Tuple[str, Set[str]]] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            module_markers |= _extract_pytestmark_markers(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                func_markers = _decorators_to_markers(node.decorator_list)
                results.append((node.name, module_markers | func_markers))
        if isinstance(node, ast.ClassDef):
            class_markers = _decorators_to_markers(node.decorator_list)
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if child.name.startswith("test_"):
                        func_markers = _decorators_to_markers(child.decorator_list)
                        results.append(
                            (child.name, module_markers | class_markers | func_markers)
                        )
    return results


def _parse_tests_and_markers_regex(content: str) -> List[Tuple[str, Set[str]]]:
    """Regex fallback for files that fail ast.parse (syntax errors, etc.)."""
    lines = content.split("\n")
    results: List[Tuple[str, Set[str]]] = []

    for i, line in enumerate(lines):
        m = _RE_TEST_FUNC.match(line)
        if not m:
            continue
        test_name = m.group(1)
        markers: Set[str] = set()
        j = i - 1
        while j >= 0:
            prev = lines[j].strip()
            mark_match = _RE_PYTEST_MARK.match(prev)
            if mark_match:
                markers.add(mark_match.group(1))
                j -= 1
                continue
            if not prev or prev.startswith("#") or prev.startswith("@"):
                j -= 1
                continue
            break
        results.append((test_name, markers))
    return results


# ---------------------------------------------------------------------------
# Analysis: --pytests
# ---------------------------------------------------------------------------


def analyze_pytests_snapshot(
    repo_path: str,
    commit: str,
    test_dir: str,
    tracked_markers: List[str],
) -> Dict:
    """Count pytest test functions and markers at a commit."""
    params = (test_dir, ",".join(sorted(tracked_markers)))
    cached = _cache.get("pytests_ast", commit, *params)
    if cached is not None:
        return cached

    test_files = list_test_files_at_commit(repo_path, commit, test_dir)

    real_tracked = set(m for m in tracked_markers if m != "unmarked")
    marker_counts: Dict[str, int] = defaultdict(int)
    total_tests = 0
    marked_count = 0

    for path in test_files:
        content = read_file_at_commit(repo_path, commit, path)
        if content is None:
            continue
        for _test_name, markers in parse_tests_and_markers(content):
            total_tests += 1
            normalized = {("skip" if m in SKIP_MARKERS else m) for m in markers}
            has_tracked = False
            for nm in normalized:
                if nm in real_tracked:
                    marker_counts[nm] += 1
                    has_tracked = True
            if has_tracked:
                marked_count += 1

    if "unmarked" in tracked_markers:
        marker_counts["unmarked"] = total_tests - marked_count

    result = {
        "test_files": len(test_files),
        "total_tests": total_tests,
        "markers": dict(marker_counts),
    }
    _cache.put("pytests_ast", commit, result, *params)
    return result


# ---------------------------------------------------------------------------
# Analysis: --loc
# ---------------------------------------------------------------------------


def _get_empty_tree(repo_path: str) -> str:
    """Get the empty tree hash for this repo (handles SHA-1 and SHA-256)."""
    if repo_path not in _empty_tree_cache:
        rc, stdout, _ = run_git(["hash-object", "-t", "tree", "/dev/null"], repo_path)
        _empty_tree_cache[repo_path] = stdout.strip()
    return _empty_tree_cache[repo_path]


def analyze_loc_snapshot(repo_path: str, commit: str) -> Dict:
    """Count lines of code by language at a commit.

    Uses `git diff --numstat EMPTY_TREE COMMIT` for a single-command line count
    of every file in the tree. Much faster than reading files individually.
    """
    cached = _cache.get("loc", commit)
    if cached is not None:
        return cached

    empty_tree = _get_empty_tree(repo_path)
    rc, stdout, _ = run_git(
        ["diff", "--numstat", empty_tree, commit],
        repo_path,
        timeout=120,
    )
    if rc != 0:
        return {"python_loc": 0, "rust_loc": 0, "doc_loc": 0, "total_loc": 0}

    python_loc = 0
    rust_loc = 0
    doc_loc = 0
    total_loc = 0

    for line in stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t", 2)
        if len(parts) < 3:
            continue
        added_str, _deleted_str, filepath = parts
        if added_str == "-":
            continue
        lines = int(added_str)
        total_loc += lines

        ext = os.path.splitext(filepath)[1].lower()
        if ext in PYTHON_EXTS:
            python_loc += lines
        elif ext in RUST_EXTS:
            rust_loc += lines
        elif ext in DOC_EXTS:
            doc_loc += lines

    result = {
        "python_loc": python_loc,
        "rust_loc": rust_loc,
        "doc_loc": doc_loc,
        "total_loc": total_loc,
    }
    _cache.put("loc", commit, result)
    return result


def analyze_loc_churn(
    repo_path: str, prev_commit: Optional[str], curr_commit: str
) -> Dict:
    """Compute gross lines added and deleted between two commits.

    If prev_commit is None, diffs against the empty tree (all lines are 'added').
    """
    cache_base = prev_commit or "EMPTY"
    cached = _cache.get("loc_churn", curr_commit, cache_base)
    if cached is not None:
        return cached

    base = prev_commit if prev_commit else _get_empty_tree(repo_path)
    rc, stdout, _ = run_git(
        ["diff", "--numstat", base, curr_commit],
        repo_path,
        timeout=120,
    )
    if rc != 0:
        return {"lines_added": 0, "lines_deleted": 0}

    added = 0
    deleted = 0
    for line in stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t", 2)
        if len(parts) < 3:
            continue
        a_str, d_str, _ = parts
        if a_str == "-":
            continue
        added += int(a_str)
        deleted += int(d_str)

    result = {"lines_added": added, "lines_deleted": deleted}
    _cache.put("loc_churn", curr_commit, result, cache_base)
    return result


# ---------------------------------------------------------------------------
# Analysis: --rust
# ---------------------------------------------------------------------------


def analyze_rust_tests_snapshot(repo_path: str, commit: str) -> Dict:
    """Count Rust test functions at a commit using git-grep."""
    cached = _cache.get("rust", commit)
    if cached is not None:
        return cached

    test_files = 0
    total_tests = 0

    # git grep -c matches-per-file for #[test] and #[tokio::test]
    for pattern in [r"#\[test\]", r"#\[tokio::test\]", r"#\[rstest\]"]:
        rc, stdout, _ = run_git(
            ["grep", "-c", "--no-color", pattern, commit, "--", "*.rs"],
            repo_path,
        )
        if rc != 0:
            continue
        for line in stdout.strip().split("\n"):
            if not line:
                continue
            # Format: COMMIT:path/file.rs:COUNT
            parts = line.rsplit(":", 1)
            if len(parts) == 2:
                count = int(parts[1])
                total_tests += count

    # Count unique .rs files containing any test attribute
    rc, stdout, _ = run_git(
        ["grep", "-l", "--no-color", r"#\[test\]", commit, "--", "*.rs"],
        repo_path,
    )
    files_with_test: Set[str] = set()
    if rc == 0 and stdout.strip():
        for line in stdout.strip().split("\n"):
            # strip COMMIT: prefix
            path = line.split(":", 1)[-1] if ":" in line else line
            files_with_test.add(path)

    for pattern in [r"#\[tokio::test\]", r"#\[rstest\]"]:
        rc, stdout, _ = run_git(
            ["grep", "-l", "--no-color", pattern, commit, "--", "*.rs"],
            repo_path,
        )
        if rc == 0 and stdout.strip():
            for line in stdout.strip().split("\n"):
                path = line.split(":", 1)[-1] if ":" in line else line
                files_with_test.add(path)

    test_files = len(files_with_test)

    result = {
        "rust_test_files": test_files,
        "rust_total_tests": total_tests,
    }
    _cache.put("rust", commit, result)
    return result


# ---------------------------------------------------------------------------
# Analysis: --commits
# ---------------------------------------------------------------------------


def count_commits_in_range(
    repo_path: str, after_date: str, before_date: str, branch: str = "main"
) -> int:
    """Count commits between two dates on *branch*."""
    rc, stdout, _ = run_git(
        [
            "rev-list",
            "--count",
            branch,
            f"--after={after_date} 00:00:00",
            f"--before={before_date} 23:59:59",
        ],
        repo_path,
    )
    if rc != 0 or not stdout.strip():
        return 0
    return int(stdout.strip())


def count_commits_before(repo_path: str, before_date: str, branch: str = "main") -> int:
    """Count total commits up to a date on *branch*."""
    rc, stdout, _ = run_git(
        ["rev-list", "--count", branch, f"--before={before_date} 23:59:59"],
        repo_path,
    )
    if rc != 0 or not stdout.strip():
        return 0
    return int(stdout.strip())


def count_unique_authors_in_range(
    repo_path: str, after_date: str, before_date: str, branch: str = "main"
) -> int:
    """Count unique commit authors between two dates on *branch*."""
    rc, stdout, _ = run_git(
        [
            "log",
            branch,
            "--format=%ae",
            f"--after={after_date} 00:00:00",
            f"--before={before_date} 23:59:59",
        ],
        repo_path,
    )
    if rc != 0 or not stdout.strip():
        return 0
    return len(set(stdout.strip().split("\n")))


# ---------------------------------------------------------------------------
# Date generation
# ---------------------------------------------------------------------------


def generate_monthly_dates(start_date: str, end_date: str) -> List[str]:
    """Generate 1st-of-month dates from start to end, plus end date if different."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    dates: List[str] = []
    current = datetime(start.year, start.month, 1)
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)

    if dates and dates[-1] != end_date:
        dates.append(end_date)
    return dates


def generate_daily_dates(start_date: str, end_date: str) -> List[str]:
    """Generate daily dates from start to end (inclusive)."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates: List[str] = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


def generate_weekly_dates(start_date: str, end_date: str) -> List[str]:
    """Generate weekly dates (every Monday) from start to end, plus end date."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    days_since_monday = start.weekday()
    current = start - timedelta(days=days_since_monday)

    dates: List[str] = []
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(weeks=1)

    if dates and dates[-1] != end_date:
        dates.append(end_date)
    return dates


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_delta(current: int, previous: Optional[int]) -> str:
    """Format a value with its delta: "12 (+12)", "2 (-10)", "5"."""
    if previous is None:
        return str(current)
    diff = current - previous
    if diff == 0:
        return str(current)
    sign = "+" if diff > 0 else ""
    return f"{current} ({sign}{diff:>d})"


HEADER_DISPLAY: Dict[str, Tuple[str, str]] = {
    "Date": ("", "Date"),
    "Py Test": ("Py Test", "Files"),
    "Py Total": ("Py Total", "Tests"),
    "pre_merge": ("pre_", "merge"),
    "post_merge": ("post_", "merge"),
    "parallel": ("para-", "llel"),
    "integration": ("integr-", "ation"),
    "gpu_0": ("", "gpu_0"),
    "gpu_1": ("", "gpu_1"),
    "gpu_2": ("", "gpu_2"),
    "skip": ("", "skip"),
    "unmarked": ("un-", "marked"),
    "Python LoC": ("Python", "LoC"),
    "Rust LoC": ("Rust", "LoC"),
    "Docs LoC": ("Docs", "LoC"),
    "Total LoC": ("Total", "LoC"),
    "Added": ("Lines", "Added"),
    "Deleted": ("Lines", "Deleted"),
    "Rust Files": ("Rust", "Files"),
    "Rust Tests": ("Rust", "Tests"),
    "Commits": ("", "Commits"),
    "Cumulative": ("Cumul-", "ative"),
    "Authors": ("", "Authors"),
}


def _header_lines(name: str) -> Tuple[str, str]:
    if name in HEADER_DISPLAY:
        return HEADER_DISPLAY[name]
    return ("", name)


def print_generic_table(
    col_keys: List[str],
    rows: List[List[str]],
    group_boundaries: Optional[Set[int]] = None,
) -> None:
    """Print a table with two-line headers.

    group_boundaries: column indices where a group starts (use || before that column).
    """
    boundaries = group_boundaries or set()
    h_line1 = [_header_lines(k)[0] for k in col_keys]
    h_line2 = [_header_lines(k)[1] for k in col_keys]

    col_widths = [max(len(h_line1[i]), len(h_line2[i])) for i in range(len(col_keys))]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(cells: List[str]) -> str:
        parts: List[str] = []
        for i, (cell, w) in enumerate(zip(cells, col_widths)):
            sep = "||" if i in boundaries else "|"
            parts.append(f"{sep} {cell:<{w}} ")
        return "".join(parts) + "|"

    def fmt_sep() -> str:
        parts: List[str] = []
        for i, w in enumerate(col_widths):
            sep = "||" if i in boundaries else "|"
            parts.append(f"{sep}{'-' * (w + 2)}")
        return "".join(parts) + "|"

    print(fmt_row(h_line1))
    print(fmt_row(h_line2))
    print(fmt_sep())
    for row in rows:
        print(fmt_row(row))


# ---------------------------------------------------------------------------
# Combined table builder
# ---------------------------------------------------------------------------


def build_combined_table(
    all_results: Dict[str, List[Dict]],
    tracked_markers: List[str],
    num_rows: int,
) -> None:
    """Build a single table with all analyses side by side, keyed by Date."""
    col_keys: List[str] = ["Date"]
    group_boundaries: Set[int] = set()

    has_pytests = "pytests" in all_results
    has_loc = "loc" in all_results
    has_rust = "rust" in all_results
    has_commits = "commits" in all_results

    if has_pytests:
        group_boundaries.add(len(col_keys))
        col_keys += ["Py Test", "Py Total"] + tracked_markers
    if has_loc:
        group_boundaries.add(len(col_keys))
        col_keys += ["Python LoC", "Rust LoC", "Docs LoC", "Total LoC", "Added", "Deleted"]
    if has_rust:
        group_boundaries.add(len(col_keys))
        col_keys += ["Rust Files", "Rust Tests"]
    if has_commits:
        group_boundaries.add(len(col_keys))
        col_keys += ["Commits", "Cumulative", "Authors"]

    # Track previous values for deltas (one prev per int column)
    prev: Dict[str, Optional[int]] = {}

    rows: List[List[str]] = []
    for i in range(num_rows):
        row: List[str] = []

        # Date (from whichever analysis is present)
        date = ""
        for history in all_results.values():
            date = history[i].get("date", "")
            break
        row.append(date)

        if has_pytests:
            s = all_results["pytests"][i]
            method = s.get("method", "")
            if method == "AST" and row:
                row[0] = row[0] + "*"
            tf = s.get("test_files", 0)
            tt = s.get("total_tests", 0)
            markers = s.get("markers", {})
            row.append(format_delta(tf, prev.get("tf")))
            row.append(format_delta(tt, prev.get("tt")))
            for m in tracked_markers:
                row.append(str(markers.get(m, 0)))
            prev["tf"], prev["tt"] = tf, tt

        if has_loc:
            s = all_results["loc"][i]
            py = s.get("python_loc", 0)
            rs = s.get("rust_loc", 0)
            doc = s.get("doc_loc", 0)
            total = s.get("total_loc", 0)
            row.append(str(py))
            row.append(str(rs))
            row.append(str(doc))
            row.append(str(total))
            row.append(str(s.get("lines_added", 0)))
            row.append(str(s.get("lines_deleted", 0)))

        if has_rust:
            s = all_results["rust"][i]
            rf = s.get("rust_test_files", 0)
            rt = s.get("rust_total_tests", 0)
            row.append(format_delta(rf, prev.get("rf")))
            row.append(format_delta(rt, prev.get("rt")))
            prev["rf"], prev["rt"] = rf, rt

        if has_commits:
            s = all_results["commits"][i]
            c = s.get("period_commits", 0)
            cum = s.get("cumulative_commits", 0)
            authors = s.get("authors", 0)
            row.append(str(c))
            row.append(str(cum))
            row.append(str(authors))

        rows.append(row)

    print_generic_table(col_keys, rows, group_boundaries)

    # Footnote for AST fallback rows
    if has_pytests:
        methods = [r.get("method", "") for r in all_results["pytests"]]
        has_ast = "AST" in methods
        has_collect = "collect" in methods
        if has_ast and has_collect:
            print("\n* = AST-based count (pytest --collect-only used for unmarked rows)")
        elif has_ast and not has_collect:
            print("\n* = all rows used AST (pytest --collect-only unavailable)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a git repository over time (read-only plumbing, no checkout)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Analysis flags (at least one required):
  --python    Pytest test growth (files, functions, markers)
  --loc       Lines of code (Python, Rust, total)
  --rust      Rust test growth (#[test] / #[tokio::test])
  --commits   Commit volume per period

Examples:
  %(prog)s --repo ~/dynamo/dynamo3 --months 6 --python
  %(prog)s --repo ~/dynamo/dynamo3 --weeks 12 --loc
  %(prog)s --repo ~/dynamo/dynamo3 --months 14 --python --loc --rust --commits
  %(prog)s --repo ~/dynamo/dynamo3 --start 2025-01-01 --weekly --loc --commits
        """,
    )

    # Analysis flags
    parser.add_argument("--python", action="store_true", help="Show pytest growth stats")
    parser.add_argument("--loc", action="store_true", help="Show lines-of-code stats")
    parser.add_argument("--rust", action="store_true", help="Show Rust test growth stats")
    parser.add_argument("--commits", action="store_true", help="Show commit volume stats")

    # Time range
    parser.add_argument("--repo", type=str, required=True, help="Path to git repository")
    parser.add_argument("--months", type=int, help="Months to go back from today")
    parser.add_argument("--weeks", type=int, help="Weeks to go back (implies --weekly)")
    parser.add_argument("--weekly", action="store_true", help="Weekly snapshots (every Monday)")
    parser.add_argument("--daily", action="store_true", help="Daily snapshots")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument(
        "--end", type=str, default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD, default: today)",
    )

    # Pytest-specific
    parser.add_argument(
        "--test-dir", type=str, default=".",
        help="Test directory relative to repo root (default: . = entire tree)",
    )
    parser.add_argument(
        "--markers", type=str, nargs="+", default=TRACKED_MARKERS_DEFAULT,
        help="Markers to track (default: pre_merge post_merge parallel integration skip unmarked)",
    )
    parser.add_argument("--json", type=str, help="Write results to a JSON file")

    # Collection strategy
    parser.add_argument(
        "--collect", action="store_true",
        help="Try pytest --collect-only for every commit (via git worktrees), fall back to AST",
    )

    # Cache control
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Bypass disk cache (recompute everything)",
    )
    parser.add_argument(
        "--clear-cache", action="store_true",
        help="Delete all cached results and exit",
    )

    args = parser.parse_args()

    # Cache control
    global _cache
    if args.clear_cache:
        n = _cache.clear()
        print(f"Cleared {n} cached entries from {CACHE_DIR}", file=sys.stderr)
        sys.exit(0)
    if args.no_cache:
        _cache = AnalysisCache(enabled=False)

    # Validate analysis selection
    if not any([args.python, args.loc, args.rust, args.commits]):
        print("Error: Specify at least one of --python, --loc, --rust, --commits", file=sys.stderr)
        sys.exit(1)

    if args.weeks:
        args.weekly = True

    time_specs = sum(1 for x in [args.months, args.weeks, args.start] if x)
    if time_specs > 1:
        print("Error: Specify only one of --months, --weeks, or --start", file=sys.stderr)
        sys.exit(1)
    if time_specs == 0:
        print("Error: Must specify --months, --weeks, or --start", file=sys.stderr)
        sys.exit(1)

    if args.months:
        today = datetime.now()
        start = datetime(today.year, today.month, 1)
        for _ in range(args.months):
            if start.month == 1:
                start = datetime(start.year - 1, 12, 1)
            else:
                start = datetime(start.year, start.month - 1, 1)
        args.start = start.strftime("%Y-%m-%d")

    if args.weeks:
        today = datetime.now()
        start = today - timedelta(weeks=args.weeks)
        args.start = start.strftime("%Y-%m-%d")

    repo_path = os.path.abspath(args.repo)

    if not os.path.isdir(repo_path):
        print(f"Error: Repository path does not exist: {repo_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(os.path.join(repo_path, ".git")):
        print(f"Error: Not a git repository: {repo_path}", file=sys.stderr)
        sys.exit(1)

    if args.daily:
        dates = generate_daily_dates(args.start, args.end)
        interval_label = "daily"
    elif args.weekly:
        dates = generate_weekly_dates(args.start, args.end)
        interval_label = "weekly"
    else:
        dates = generate_monthly_dates(args.start, args.end)
        interval_label = "monthly"

    analyses = []
    if args.python:
        analyses.append("pytests")
    if args.loc:
        analyses.append("loc")
    if args.rust:
        analyses.append("rust")
    if args.commits:
        analyses.append("commits")

    print(
        f"Analyzing from {args.start} to {args.end} ({len(dates)} {interval_label} snapshots)",
        file=sys.stderr,
    )
    print(f"Repository: {repo_path}", file=sys.stderr)
    print(f"Analyses:   {', '.join(analyses)}", file=sys.stderr)
    print("", file=sys.stderr)

    # Resolve commits for each date
    date_commits: List[Tuple[str, Optional[str]]] = []
    for date in dates:
        print(f"  {date}: ", end="", file=sys.stderr, flush=True)
        commit = get_commit_at_date(repo_path, date)
        if not commit:
            print("no commits yet", file=sys.stderr)
        else:
            print(f"{commit[:9]}", file=sys.stderr)
        date_commits.append((date, commit))

    print("", file=sys.stderr)

    # Run each analysis
    all_results: Dict[str, List[Dict]] = {}

    if args.python:
        print("  Running pytest analysis...", file=sys.stderr, flush=True)
        history: List[Dict] = []

        if args.collect:
            # --collect: try pytest --collect-only for EVERY commit via worktrees
            print("  Mode: pytest --collect-only per commit (worktree), AST fallback", file=sys.stderr)
            for idx, (date, commit) in enumerate(date_commits):
                if not commit:
                    history.append({"date": date, "test_files": 0, "total_tests": 0, "markers": {}, "method": ""})
                    continue
                print(f"  [{idx+1}/{len(date_commits)}] {date} ({commit[:9]}): ", end="", file=sys.stderr, flush=True)
                snap = collect_at_commit(repo_path, commit, args.markers)
                if snap is not None:
                    snap["date"] = date
                    snap["method"] = "collect"
                    print("collect OK", file=sys.stderr, flush=True)
                else:
                    snap = analyze_pytests_snapshot(repo_path, commit, args.test_dir, args.markers)
                    snap["date"] = date
                    snap["method"] = "AST"
                    print("AST fallback", file=sys.stderr, flush=True)
                history.append(snap)
        else:
            # Default: try collect at HEAD only, AST for the rest
            head_sha = _get_repo_head(repo_path)
            collect_cache: Optional[Dict] = None
            if head_sha:
                print(f"  Trying pytest --collect-only at HEAD ({head_sha[:9]})...", file=sys.stderr, flush=True)
                collect_cache = analyze_pytests_via_collect(repo_path, args.markers)
                if collect_cache:
                    print("  Pytest collect succeeded (will use for HEAD snapshot).", file=sys.stderr, flush=True)
                else:
                    print("  Pytest collect failed; using AST for all snapshots.", file=sys.stderr, flush=True)

            # Determine which row gets the collect result:
            # exact SHA match first, otherwise the last row (closest to HEAD).
            collect_row_idx: Optional[int] = None
            if collect_cache and head_sha:
                for idx, (_d, c) in enumerate(date_commits):
                    if c == head_sha:
                        collect_row_idx = idx
                        break
                if collect_row_idx is None:
                    collect_row_idx = len(date_commits) - 1

            for idx, (date, commit) in enumerate(date_commits):
                if not commit:
                    history.append({"date": date, "test_files": 0, "total_tests": 0, "markers": {}, "method": ""})
                    continue
                if collect_cache and idx == collect_row_idx:
                    snap = dict(collect_cache)
                    snap["date"] = date
                    snap["method"] = "collect"
                else:
                    snap = analyze_pytests_snapshot(repo_path, commit, args.test_dir, args.markers)
                    snap["date"] = date
                    snap["method"] = "AST"
                history.append(snap)

        all_results["pytests"] = history

    if args.loc:
        print("  Running LoC analysis...", file=sys.stderr, flush=True)
        history = []
        prev_loc_commit: Optional[str] = None
        for date, commit in date_commits:
            if not commit:
                history.append({
                    "date": date, "python_loc": 0, "rust_loc": 0, "doc_loc": 0,
                    "total_loc": 0, "lines_added": 0, "lines_deleted": 0,
                })
                continue
            snap = analyze_loc_snapshot(repo_path, commit)
            churn = analyze_loc_churn(repo_path, prev_loc_commit, commit)
            snap.update(churn)
            snap["date"] = date
            history.append(snap)
            prev_loc_commit = commit
        all_results["loc"] = history

    if args.rust:
        print("  Running Rust test analysis...", file=sys.stderr, flush=True)
        history = []
        for date, commit in date_commits:
            if not commit:
                history.append({"date": date, "rust_test_files": 0, "rust_total_tests": 0})
                continue
            snap = analyze_rust_tests_snapshot(repo_path, commit)
            snap["date"] = date
            history.append(snap)
        all_results["rust"] = history

    if args.commits:
        print("  Running commit analysis...", file=sys.stderr, flush=True)
        history = []
        # Use the day before the first date as the implicit "previous" so the
        # first row shows actual daily commits instead of all-time cumulative.
        first_date_dt = datetime.strptime(dates[0], "%Y-%m-%d")
        prev_date: Optional[str] = (first_date_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        for date, commit in date_commits:
            period = count_commits_in_range(repo_path, prev_date, date)
            cumulative = count_commits_before(repo_path, date)
            authors = count_unique_authors_in_range(repo_path, prev_date, date)
            history.append({
                "date": date,
                "period_commits": period,
                "cumulative_commits": cumulative,
                "authors": authors,
            })
            prev_date = date
        all_results["commits"] = history

    print("", file=sys.stderr)

    build_combined_table(all_results, args.markers, len(dates))

    # JSON output
    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"JSON written to: {args.json}", file=sys.stderr)

    print(f"  {_cache.summary()}", file=sys.stderr)


if __name__ == "__main__":
    main()
