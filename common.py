"""
Dynamo utilities package.

Shared constants and utilities for dynamo Docker management scripts.
"""

import hashlib
import io
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys

import threading
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from zoneinfo import ZoneInfo

from common_types import CIStatus, MarkerStatus

# Log/snippet detection lives in the shared library: `dynamo-utils.PRODUCTION/ci_log_errors/`.
# ======================================================================================
# API inventory (where the dashboard data comes from)
#
# GitHub and GitLab API documentation has been moved to dedicated modules:
# - GitHub API: see common_github.py (GitHubAPIClient class)
# - GitLab API: see common_gitlab.py (GitLabAPIClient class)
#
# The dashboards in this repo are built from a mix of:
# - Local git metadata (branch/commit subject/SHA/time) from GitPython
# - GitHub REST v3 (https://api.github.com) for PRs, check-runs, Actions job logs
# - GitLab REST v4 (https://gitlab-master.nvidia.com) for pipeline status lines
# ======================================================================================

# ======================================================================================

# Global logger for the module
_logger = logging.getLogger(__name__)

class PhaseTimer:
    """Tiny timing helper for coarse "where is time going?" instrumentation.

    Intended usage:
      t = PhaseTimer()
      with t.phase("scan"):
          ...
      with t.phase("render"):
          ...
      print(t.format_one_line())
    """

    def __init__(self) -> None:
        self._t0 = time.monotonic()
        self._t_last = self._t0
        self._dur_s: Dict[str, float] = {}

    @contextmanager
    def phase(self, name: str):
        t0 = time.monotonic()
        try:
            yield
        finally:
            self._dur_s[str(name)] = float(self._dur_s.get(str(name), 0.0)) + max(0.0, time.monotonic() - t0)

    def mark(self, name: str) -> None:
        """Record time since the last mark into `name`, and advance the 'last' cursor."""
        now = time.monotonic()
        self._dur_s[str(name)] = float(self._dur_s.get(str(name), 0.0)) + max(0.0, now - self._t_last)
        self._t_last = now

    def start(self) -> float:
        """Return a monotonic timestamp suitable for passing to `stop()`."""
        return float(time.monotonic())

    def stop(self, name: str, started_at: float) -> None:
        """Accumulate elapsed seconds since `started_at` into `name`."""
        dt = max(0.0, float(time.monotonic()) - float(started_at))
        self._dur_s[str(name)] = float(self._dur_s.get(str(name), 0.0)) + dt

    def time_call(self, name: str, fn, *args, **kwargs):
        """Time a callable and return its result (best-effort)."""
        t0 = self.start()
        try:
            return fn(*args, **kwargs)
        finally:
            self.stop(str(name), t0)

    def total_s(self) -> float:
        return max(0.0, time.monotonic() - self._t0)

    def as_dict(self, *, include_total: bool = True) -> Dict[str, float]:
        d = dict(self._dur_s)
        if include_total:
            d["total"] = self.total_s()
        return d

    @staticmethod
    def format_seconds(s: float) -> str:
        return f"{float(s):.2f}s"

    def format_one_line(self, *, keys: Optional[List[str]] = None) -> str:
        d = self.as_dict(include_total=True)
        if keys:
            parts = [f"{k}={self.format_seconds(d.get(k, 0.0))}" for k in keys]
        else:
            # Deterministic-ish order: total last; others alphabetically.
            ks = sorted([k for k in d.keys() if k != "total"])
            parts = [f"{k}={self.format_seconds(d[k])}" for k in ks]
            parts.append(f"total={self.format_seconds(d.get('total', 0.0))}")
        return ", ".join(parts)

#
# Cache policy constants (single source of truth)
#
# These are intentionally defined at module level so call sites don't duplicate
# literals (8h / 5min / 30d / etc) across scripts.
#
DEFAULT_STABLE_AFTER_HOURS: int = 8
# ^ Age threshold (hours) that flips certain caches from "fast refresh" to "stable refresh".
#   Used primarily for per-commit CI status caches (e.g., GitHub Actions check-runs):
#   - If a commit is newer than this threshold (e.g., 2 hours old), we use `DEFAULT_UNSTABLE_TTL_S`
#     so the dashboard updates quickly while CI is still settling.
#   - If a commit is older than this threshold (e.g., 3 days old), we use `DEFAULT_STABLE_TTL_S`
#     so we don’t keep re-checking historical commits over and over.
DEFAULT_UNSTABLE_TTL_S: int = 180
# ^ "Fast refresh" TTL (seconds) for things that change quickly.
#   Example: for a commit from ~3 minutes ago, we may re-check GitHub Actions status every 3 minutes.
DEFAULT_STABLE_TTL_S: int = 2 * 3600
# ^ "Stable" TTL (seconds) for older/less-changing data (commits > 8 hours old).
#   CI can still be re-run even for old commits, so we don't cache forever.
#   Example: for a commit from yesterday, we cache its GitHub Actions status for 3 days.
DEFAULT_OPEN_PRS_TTL_S: int = 60
# ^ TTL (seconds) for caches keyed by *open* PRs (expected to change often).
#   Example: required-checks / PR metadata for an open PR can be refreshed every 5 minutes.
DEFAULT_CLOSED_PRS_TTL_S: int = 365 * 24 * 3600
# ^ TTL (seconds) for caches keyed by *closed/merged* PRs.
#   Merged PRs are IMMUTABLE and never change, so we cache them for 1 year (effectively forever).
#   Example: merge-date for a merged PR is cached for 365 days.
DEFAULT_NO_PR_TTL_S: int = 24 * 3600
# ^ TTL (seconds) for "negative cache" entries (when we *didn't* find something).
#   Example: if SHA→PR mapping wasn't found, we remember that for 24 hours so we don't re-query constantly,
#   but we retry later in case the mapping appears.
DEFAULT_RAW_LOG_URL_TTL_S: int = 3600
# ^ TTL (seconds) for cached raw log *redirect URL* (GitHub Actions provides time-limited signed URLs).
#   The redirect URL expires quickly, so we keep TTL short (1 hour).
#   Example: the redirect URL for job `59030780729` is valid for ~1 hour before needing refresh.
DEFAULT_RAW_LOG_TEXT_TTL_S: int = 365 * 24 * 3600
# ^ TTL (seconds) for cached raw log *content* (the downloaded text we parse/snippet locally).
#   Completed job logs are IMMUTABLE and never change, so we cache them for 1 year (effectively forever).
#   Example: once we download job `59030780729`, keep its `.log` for 365 days to avoid re-downloading.
DEFAULT_RAW_LOG_TEXT_MAX_BYTES: int = 0
# ^ Max bytes to download when caching raw log content.
#   Example: set to `10*1024*1024` to cap downloads at 10MB; `0` means "no cap".
DEFAULT_RAW_LOG_ERROR_SNIPPET_TAIL_BYTES: int = 512 * 1024
# ^ When extracting error snippets, only read the tail of very large log files (bytes).
#   Example: for a 50MB log, only read the last 512KB to find failures quickly.


def format_duration_compact(seconds: int) -> str:
    """Convert a duration in seconds to a compact human-readable string (e.g., '5m', '2h', '30d')."""
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h"
    return f"{seconds // 86400}d"


# ======================================================================================
# IMPORTANT: Cache location policy (dynamo-utils)
#
# All *persistent* caches for dynamo-utils MUST live under:
#   - $DYNAMO_UTILS_CACHE_DIR       (explicit override), else
#   - ~/.cache/dynamo-utils         (default)
#
# Do NOT write caches into a repo checkout (e.g. `commits/.cache/...`), because:
#   - it dirties the checkout (untracked files) and can break automation like `git checkout`
#   - it scatters caches across multiple clones/paths
#
# Legacy behavior:
#   - Some older call sites passed paths like ".cache/foo.json".
#     `resolve_cache_path()` strips the leading ".cache/" and places the file in the global
#     cache dir so older paths still work without polluting the repo.
# ======================================================================================

def dynamo_utils_cache_dir() -> Path:
    """Return the cache directory for dynamo-utils.

    Resolution order:
    - DYNAMO_UTILS_CACHE_DIR (explicit override)
    - ~/.cache/dynamo-utils
    """
    override = os.environ.get("DYNAMO_UTILS_CACHE_DIR")
    if override:
        return Path(override).expanduser()

    return Path.home() / ".cache" / "dynamo-utils"


def resolve_cache_path(cache_file: str) -> Path:
    """Resolve a cache file path into the global dynamo-utils cache directory.

    - Absolute paths are used as-is.
    - Relative paths are rooted under `dynamo_utils_cache_dir()`.
    - If the relative path starts with ".cache/", that prefix is stripped so older
      call sites can pass ".cache/foo.json" but still land in ~/.cache/dynamo-utils/foo.json.
    """
    p = Path(cache_file).expanduser()
    if p.is_absolute():
        return p

    # Normalize any leading "./"
    rel = Path(*p.parts[1:]) if p.parts[:1] == (".",) else p

    # Strip leading ".cache/" if present
    if rel.parts[:1] == (".cache",):
        rel = Path(*rel.parts[1:])

    return dynamo_utils_cache_dir() / rel

# Third-party imports
try:
    import docker  # type: ignore[import-untyped]
except ImportError:
    docker = None  # type: ignore[assignment]

import git  # type: ignore[import-not-found]
import requests
import yaml

# Supported frameworks
# Used by V2 for default framework list and validation
FRAMEWORKS_UPPER = ["VLLM", "SGLANG", "TRTLLM"]

# DEPRECATED: V1 and retag script reference only - kept for backward compatibility
# V2 uses FRAMEWORKS_UPPER directly
FRAMEWORKS = [f.lower() for f in FRAMEWORKS_UPPER]
FRAMEWORK_NAMES = {"vllm": "VLLM", "sglang": "SGLang", "trtllm": "TensorRT-LLM"}

# Marker file suffixes (shared by build_images.py and show_commit_history.py)
MARKER_RUNNING = MarkerStatus.RUNNING.value
MARKER_PASSED = MarkerStatus.PASSED.value
MARKER_FAILED = MarkerStatus.FAILED.value
MARKER_KILLED = MarkerStatus.KILLED.value

def normalize_framework(framework: str) -> str:
    """Normalize framework name (e.g. engine name like "vllm") to canonical lowercase form. DEPRECATED: V1/retag only."""
    return framework.lower()

def get_framework_display_name(framework: str) -> str:
    """Get display name for framework. DEPRECATED: V1/retag only."""
    normalized = normalize_framework(framework)
    return FRAMEWORK_NAMES.get(normalized, normalized.upper())


# Used by retag script only. V1 and V2 do not use these dataclasses.
@dataclass
class DynamoImageInfo:
    """Dynamo-specific Docker image information.

    DEPRECATION: retag script only. V1 and V2 do not use this.
    """
    version: Optional[str] = None      # Parsed version (e.g., "0.1.0.dev.ea07d51fc")
    framework: Optional[str] = None    # Framework name (vllm, sglang, trtllm)
    target: Optional[str] = None       # Target type (local-dev, dev, etc.)
    latest_tag: Optional[str] = None   # Corresponding latest tag

    def matches_sha(self, sha: str) -> bool:
        """Check if this image matches the specified SHA."""
        return bool(self.version and sha in self.version)

    def is_framework_image(self) -> bool:
        """Check if this has framework information."""
        return self.framework is not None

    def get_latest_tag(self, repository: str = "dynamo") -> str:
        """Get the latest tag for this dynamo image."""
        if self.latest_tag:
            return self.latest_tag
        if self.framework:
            if self.target == "dev":
                # dev target maps to just latest-framework (no -dev suffix)
                return f"{repository}:latest-{self.framework}"
            elif self.target:
                # other targets like local-dev keep the suffix
                return f"{repository}:latest-{self.framework}-{self.target}"
            else:
                return f"{repository}:latest-{self.framework}"
        return f"{repository}:latest"


@dataclass
class DockerImageInfo:
    """Comprehensive Docker image information.

    DEPRECATION: retag script only. V1 and V2 do not use this.
    """
    name: str                    # Full image name (repo:tag)
    repository: str              # Repository name
    tag: str                     # Tag name
    image_id: str               # Docker image ID
    created_at: str             # Creation timestamp
    size_bytes: int             # Size in bytes
    size_human: str             # Human readable size
    labels: Dict[str, str]      # Image labels

    # Dynamo-specific information (optional)
    dynamo_info: Optional[DynamoImageInfo] = None

    def matches_sha(self, sha: str) -> bool:
        """Check if this image matches the specified SHA."""
        return bool(self.dynamo_info and self.dynamo_info.matches_sha(sha))

    def is_dynamo_image(self) -> bool:
        """Check if this is a dynamo image."""
        return self.repository in ["dynamo", "dynamo-base"]

    def is_dynamo_framework_image(self) -> bool:
        """Check if this is a dynamo framework image."""
        return bool(self.is_dynamo_image() and self.dynamo_info and self.dynamo_info.is_framework_image())

    def get_latest_tag(self) -> str:
        """Get the latest tag for this image."""
        if self.dynamo_info:
            return self.dynamo_info.get_latest_tag(self.repository)
        return f"{self.repository}:latest"


def get_terminal_width(padding: int = 2, default: int = 118) -> int:
    """
    Detect terminal width using multiple methods in order of preference.

    This function tries several approaches to detect the terminal width,
    which is useful in various environments including PTY/TTY contexts:

    1. Check $COLUMNS environment variable (set by interactive shells)
    2. Try 'tput cols' command (works in more environments than ioctl)
    3. Use shutil.get_terminal_size() (ioctl-based, may return default 80)
    4. Fall back to provided default

    Args:
        padding: Number of characters to subtract from detected width (default: 2)
        default: Default width to use if detection fails (default: 118, i.e., 120 - 2)

    Returns:
        Terminal width in columns, minus the specified padding

    Examples:
        >>> # Get terminal width with default 2-char padding
        >>> width = get_terminal_width()
        >>>
        >>> # Get terminal width with custom padding
        >>> width = get_terminal_width(padding=4)
        >>>
        >>> # Get terminal width with custom default fallback
        >>> width = get_terminal_width(default=78)  # 80 - 2
    """
    term_width = None

    try:
        # Method 1: Check $COLUMNS environment variable (set by shell)
        columns_env = os.environ.get('COLUMNS')
        if columns_env and columns_env.isdigit():
            term_width = int(columns_env) - padding

        # Method 2: Try tput cols (works in more environments than ioctl)
        if term_width is None:
            try:
                result = subprocess.run(
                    ['tput', 'cols'],
                    capture_output=True,
                    text=True,
                    timeout=1
                )
                if result.returncode == 0 and result.stdout.strip().isdigit():
                    term_width = int(result.stdout.strip()) - padding
            except (OSError, subprocess.SubprocessError, ValueError):
                pass

        # Method 3: Use shutil.get_terminal_size() (ioctl-based)
        if term_width is None:
            term_width = shutil.get_terminal_size().columns - padding

    except OSError:
        term_width = default

    # Final fallback to default
    if term_width is None:
        term_width = default

    return term_width


class BaseUtils:
    """Base class for all utility classes with common logger and cmd functionality"""

    def __init__(self, dry_run: bool = False, verbose: bool = False):
        self.dry_run = dry_run
        self.verbose = verbose

        # Set up logger with class name
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)

        # Remove any existing handlers
        self.logger.handlers.clear()

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)

        # Create custom formatter that handles DRYRUN prefix and shows class/method
        class DryRunFormatter(logging.Formatter):
            def __init__(self, dry_run_instance) -> None:
                super().__init__()
                self.dry_run_instance = dry_run_instance

            def format(self, record: logging.LogRecord) -> str:
                if self.dry_run_instance.verbose:
                    # Verbose mode: show location info
                    location = f"{record.name}.{record.funcName}" if record.funcName != '<module>' else record.name
                    prefix = "DRYRUN" if self.dry_run_instance.dry_run else ""
                    if prefix:
                        return f"{prefix} {record.levelname} - [{location}] {record.getMessage()}"
                    else:
                        return f"{record.levelname} - [{location}] {record.getMessage()}"
                else:
                    # Simple mode: just the message
                    return record.getMessage()

        formatter = DryRunFormatter(self)
        console_handler.setFormatter(formatter)

        # Add handler to logger
        self.logger.addHandler(console_handler)

        # Prevent propagation to root logger
        self.logger.propagate = False

    def cmd(self, command: List[str], return_tuple: bool = False, **kwargs: Any) -> Any:
        """Execute command with dry-run support.

        Args:
            command: Command to execute as list of strings
            return_tuple: If True, return (success, stdout, stderr). If False, return CompletedProcess
            **kwargs: Additional arguments passed to subprocess.run()

        Returns:
            CompletedProcess object (default) or (success, stdout, stderr) tuple if return_tuple=True
        """
        cmd_str = " ".join(shlex.quote(str(arg)) for arg in command)
        self.logger.debug(f"+ {cmd_str}")

        if self.dry_run:
            if return_tuple:
                self.logger.info(f"DRY RUN: Would execute: {' '.join(command)}")
                return True, "", ""
            else:
                # Return a mock completed process in dry-run mode
                mock_result: subprocess.CompletedProcess[str] = subprocess.CompletedProcess(command, 0)
                mock_result.stdout = ""
                mock_result.stderr = ""
                return mock_result

        # Set default kwargs for tuple interface
        if return_tuple:
            kwargs.setdefault('capture_output', True)
            kwargs.setdefault('text', True)
            kwargs.setdefault('check', False)

        try:
            result = subprocess.run(command, **kwargs)

            if return_tuple:
                success = result.returncode == 0
                return success, result.stdout or "", result.stderr or ""
            else:
                return result

        except Exception as e:
            if return_tuple:
                return False, "", str(e)
            else:
                # Re-raise for CompletedProcess interface
                raise


# Git utilities using GitPython API (NO subprocess calls)
class GitUtils(BaseUtils):
    """Git utilities using GitPython API only - NO subprocess calls to git.

    Provides clean API for git operations without any subprocess calls.
    All operations use GitPython's native API.

    Example:
        git_utils = GitUtils(repo_path="/path/to/repo")
        commits = git_utils.get_recent_commits(max_count=50)
        git_utils.checkout(commit_sha)
    """

    def __init__(self, repo_path: Any, dry_run: bool = False, verbose: bool = False):
        """Initialize GitUtils.

        Args:
            repo_path: Path to git repository (Path object or str)
            dry_run: Dry-run mode
            verbose: Verbose logging
        """
        super().__init__(dry_run, verbose)

        self.repo_path = Path(repo_path) if not isinstance(repo_path, Path) else repo_path

        try:
            self.repo = git.Repo(self.repo_path)
            self.logger.debug(f"Initialized git repo at {self.repo_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize git repository at {self.repo_path}: {e}")
            raise

    def get_current_branch(self) -> Optional[str]:
        """Get current branch name.

        Returns:
            Branch name or None if detached HEAD
        """
        try:
            if self.repo.head.is_detached:
                return None
            return self.repo.active_branch.name
        except Exception as e:
            self.logger.error(f"Failed to get current branch: {e}")
            return None

    def get_current_commit(self) -> str:
        """Get current commit SHA.

        Returns:
            Full commit SHA (40 characters)
        """
        return self.repo.head.commit.hexsha

    def checkout(self, ref: str) -> bool:
        """Checkout a specific commit, branch, or tag using GitPython API.

        Args:
            ref: Commit SHA, branch name, or tag name

        Returns:
            True if successful, False otherwise
        """
        if self.dry_run:
            self.logger.info(f"DRY RUN: Would checkout {ref}")
            return True

        try:
            self.repo.git.checkout(ref)
            self.logger.debug(f"Checked out {ref}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to checkout {ref}: {e}")
            return False

    def get_recent_commits(self, max_count: int = 50, branch: str = 'main') -> List[Any]:
        """Get recent commits from a branch using GitPython API.

        Args:
            max_count: Maximum number of commits to retrieve
            branch: Branch name (default: 'main')

        Returns:
            List of GitPython commit objects

            Example return value (commit objects have these attributes):
            [
                <git.Commit "21a03b316dc1e5031183965e5798b0d9fe2e64b3">,  # commit.hexsha
                <git.Commit "5fe0476e605d2564234f00e8123461e1594a9ce7">,  # commit.message
                <git.Commit "826eea05c9b3c7a68f04fb70dd44d7783d224df5">   # commit.author.name, commit.committed_datetime
            ]
        """
        try:
            commits = list(self.repo.iter_commits(branch, max_count=max_count))
            self.logger.debug(f"Retrieved {len(commits)} commits from {branch}")
            return commits
        except Exception as e:
            self.logger.error(f"Failed to get commits from {branch}: {e}")
            return []

class DynamoRepositoryUtils(BaseUtils):
    """Utilities for Dynamo repository operations including Docker image SHA calculation."""

    def __init__(self, repo_path: Any, dry_run: bool = False, verbose: bool = False):
        """
        Initialize DynamoRepositoryUtils.

        Args:
            repo_path: Path to Dynamo repository (Path object or str)
            dry_run: Dry-run mode
            verbose: Verbose logging
        """
        super().__init__(dry_run, verbose)
        self.repo_path = Path(repo_path) if not isinstance(repo_path, Path) else repo_path

    def get_release_branch_fork_points(
        self,
        limit: int = 5,
        base_ref: str = "origin/main",
        release_ref_glob: str = "refs/remotes/origin/release/*",
        github_owner: str = "ai-dynamo",
        github_repo: str = "dynamo",
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Dynamo-specific helper: for the latest N origin/release/* branches, compute each branch's
        fork-point against origin/main and return a mapping from fork-point SHA to release metadata.

        Returns:
          { "<fork_sha>": [ {"label": "release/v0.8.0", "url": "...", "branch": "release/0.8.0"}, ... ] }
        """
        # Prefer GitUtils (GitPython) API for git operations (no manual subprocess calls here).
        git_utils = GitUtils(repo_path=self.repo_path, dry_run=self.dry_run, verbose=self.verbose)

        if not self.dry_run:
            # Keep remote refs reasonably fresh; avoid tags to reduce work.
            try:
                git_utils.repo.git.fetch("origin", "--prune", "--no-tags", "--quiet")
            except git.exc.GitCommandError:
                # Non-fatal; proceed with whatever refs we have
                pass

        try:
            out = git_utils.repo.git.for_each_ref("--format=%(refname:short)", release_ref_glob)
            branches = [b.strip() for b in out.splitlines() if b.strip()]
        except git.exc.GitCommandError:
            branches = []
        if not branches:
            return {}

        def parse_semver(branch_ref: str) -> Optional[tuple]:
            tail = branch_ref.split("/")[-1].lstrip(".")
            if tail.startswith("v"):
                tail = tail[1:]
            parts = tail.split(".")
            nums: List[int] = []
            for p in parts:
                try:
                    nums.append(int(p))
                except ValueError:
                    return None
            while len(nums) < 3:
                nums.append(0)
            return (nums[0], nums[1], nums[2])

        # Pick the latest N semver branches PLUS all non-semver-named release branches
        # (e.g. release/deepseekv4, release/kimi-k2 — these don't fit the X.Y.Z pattern
        # but are still important release lines). The `limit` only caps semver branches.
        semver_branches: List[tuple] = []
        nonsemver_branches: List[str] = []
        for br in branches:
            ver = parse_semver(br)
            if ver is None:
                nonsemver_branches.append(br)
            else:
                semver_branches.append((ver, br))
        semver_branches.sort(key=lambda x: x[0], reverse=True)
        nonsemver_branches.sort()
        selected = [br for _, br in semver_branches[:limit]] + nonsemver_branches

        fork_map: Dict[str, List[Dict[str, str]]] = {}
        for branch_ref in selected:
            mb = ""

            # Preferred: fork-point
            try:
                mb = (git_utils.repo.git.merge_base("--fork-point", base_ref, branch_ref) or "").strip()
            except git.exc.GitCommandError:
                mb = ""

            # Fallback: normal merge-base
            if not mb:
                try:
                    mb = (git_utils.repo.git.merge_base(base_ref, branch_ref) or "").strip()
                except git.exc.GitCommandError:
                    mb = ""

            if not mb:
                continue

            # Count commits on main since the fork point (how far main has moved past).
            commits_past = 0
            try:
                count_out = (git_utils.repo.git.rev_list("--count", f"{mb}..{base_ref}") or "").strip()
                commits_past = int(count_out) if count_out.isdigit() else 0
            except git.exc.GitCommandError:
                commits_past = 0

            branch_name = "/".join(branch_ref.split("/")[1:])  # drop "origin/"
            label = branch_name  # e.g. "release/v0.8.0" or "release/deepseekv4"
            url = f"https://github.com/{github_owner}/{github_repo}/tree/{urllib.parse.quote(branch_name, safe='')}"

            fork_map.setdefault(mb, []).append({
                "label": label,
                "url": url,
                "branch": branch_name,
                "commits_past": str(commits_past),
            })

        return fork_map

    def get_release_branch_cherry_picks(
        self,
        main_commit_shas: List[str],
        base_ref: str = "origin/main",
        release_ref_glob: str = "refs/remotes/origin/release/*",
        github_owner: str = "ai-dynamo",
        github_repo: str = "dynamo",
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        For each commit in main_commit_shas, find which release/* branches cherry-picked it.
        Uses git patch-id (stable across cherry-picks) to detect equivalent commits.

        Returns:
          { "<main_sha>": [ {"branch": "release/deepseekv4", "branch_short": "deepseekv4",
                              "pr_number": "8705", "branch_sha": "01002df76ae...",
                              "url": "https://github.com/.../pull/8705"}, ... ] }
        """
        if not main_commit_shas:
            return {}

        git_utils = GitUtils(repo_path=self.repo_path, dry_run=self.dry_run, verbose=self.verbose)

        try:
            out = git_utils.repo.git.for_each_ref("--format=%(refname:short)", release_ref_glob)
            branches = [b.strip() for b in out.splitlines() if b.strip()]
        except git.exc.GitCommandError:
            return {}
        if not branches:
            return {}

        def _patch_id(sha: str) -> Optional[str]:
            """Compute patch-id for a commit (stable across cherry-picks). Returns None on failure."""
            try:
                # `git show <sha> | git patch-id` -> "<patch-id> <commit-sha>\n"
                show_out = git_utils.repo.git.show(sha)
                proc = subprocess.run(
                    ["git", "patch-id"],
                    input=show_out,
                    capture_output=True,
                    text=True,
                    cwd=str(self.repo_path),
                    timeout=10,
                )
                line = (proc.stdout or "").strip()
                if not line:
                    return None
                return line.split()[0]
            except (subprocess.SubprocessError, OSError, git.exc.GitCommandError):
                return None

        # Pre-compute patch-ids for the visible main commit window (one per commit).
        main_patch_ids: Dict[str, str] = {}  # patch_id -> main_sha
        for sha in main_commit_shas:
            pid = _patch_id(sha)
            if pid:
                main_patch_ids[pid] = sha

        if not main_patch_ids:
            return {}

        # Regex to extract PR number from commit subject like "fix: foo (#8705)"
        pr_re = re.compile(r"\(#(\d+)\)")

        cherry_map: Dict[str, List[Dict[str, str]]] = {}

        for branch_ref in branches:
            branch_name = "/".join(branch_ref.split("/")[1:])  # drop "origin/"
            branch_short = branch_name.split("/", 1)[1] if "/" in branch_name else branch_name

            # Get commits on this branch but NOT on main (branch-only commits).
            try:
                branch_only_log = git_utils.repo.git.log(
                    "--format=%H%x09%s",
                    f"{base_ref}..{branch_ref}",
                )
            except git.exc.GitCommandError:
                continue

            for line in branch_only_log.splitlines():
                if "\t" not in line:
                    continue
                branch_sha, subject = line.split("\t", 1)
                pid = _patch_id(branch_sha)
                if not pid:
                    continue
                main_sha = main_patch_ids.get(pid)
                if not main_sha:
                    continue
                m = pr_re.search(subject)
                pr_number = m.group(1) if m else ""
                if pr_number:
                    url = f"https://github.com/{github_owner}/{github_repo}/pull/{pr_number}"
                else:
                    url = f"https://github.com/{github_owner}/{github_repo}/commit/{branch_sha}"
                cherry_map.setdefault(main_sha, []).append({
                    "branch": branch_name,
                    "branch_short": branch_short,
                    "pr_number": pr_number,
                    "branch_sha": branch_sha,
                    "url": url,
                })

        return cherry_map

    def generate_docker_image_sha(self, full_hash: bool = False) -> str:
        """
        Generate Docker image SHA for the current HEAD commit.

        Delegates to generate_docker_image_sha_for_commit("HEAD"). The result is
        an UPPERCASE SHA256 of the full `git ls-tree -r HEAD -- container/` output,
        so any change to any tracked file under container/ changes the hash.

        Args:
            full_hash: If True, return full 64-char SHA. If False, return first 6 chars.

        Returns:
            UPPERCASE SHA256 hash string, or error code:
            - "NO_CONTAINER_DIR": container directory doesn't exist
            - "ERROR": error during calculation
        """
        return self.generate_docker_image_sha_for_commit("HEAD", full_hash=full_hash)

    def generate_docker_image_sha_for_commit(self, commit_sha: str, full_hash: bool = False) -> str:
        """
        Generate Docker image SHA for a specific commit using git plumbing (no checkout required).

        Hashes the full recursive tree listing of container/ (mode, type, blob hash, path
        for every tracked file). Any change to any file under container/ changes the hash.

        Returns UPPERCASE hex. Convention: image SHA is always uppercase to
        distinguish from git commit SHAs (lowercase).

        Args:
            commit_sha: Git commit SHA (short or full), or "HEAD".
            full_hash: If True, return full 64-char SHA. If False, return first 6 chars.

        Returns:
            UPPERCASE SHA256 hash string, or error code:
            - "NO_CONTAINER_DIR": no container/ tree at that commit
            - "ERROR": git command failed
        """
        try:
            result = subprocess.run(
                ["git", "-C", str(self.repo_path), "ls-tree", "-r", commit_sha, "--", "container/"],
                capture_output=True, text=True, check=True,
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"git ls-tree failed for {commit_sha}: {e.stderr.strip()}")
            return "ERROR"

        if not result.stdout.strip():
            return "NO_CONTAINER_DIR"

        sha_full = hashlib.sha256(result.stdout.encode()).hexdigest().upper()
        return sha_full if full_hash else sha_full[:6]

    MAX_DOCKER_IMAGE_SHA_LOOKBACK = 100

    def get_last_n_docker_image_shas(self, n: int) -> List[Tuple[str, str]]:
        """
        Get the last N unique Docker image SHAs from git history.

        Only examines commits that actually changed container/ (via git log),
        and stops as soon as N unique image SHAs are found.
        Uses git plumbing -- never modifies the working tree.

        Args:
            n: Number of unique Docker image SHAs to find.

        Returns:
            List of tuples [(commit_sha_9char, image_sha_6char_UPPER), ...] in
            chronological order (oldest first, newest last).
        """
        result = subprocess.run(
            ["git", "-C", str(self.repo_path), "log",
             "--format=%H", f"-{self.MAX_DOCKER_IMAGE_SHA_LOOKBACK}", "--", "container/"],
            capture_output=True, text=True, check=True,
        )
        container_commits = [h.strip() for h in result.stdout.strip().split('\n') if h.strip()]

        if not container_commits:
            return []

        results: List[Tuple[str, str]] = []
        prev_image_sha: Optional[str] = None
        prev_commit_sha: Optional[str] = None

        for full_sha in container_commits:
            commit_sha = full_sha[:9]
            image_sha = self.generate_docker_image_sha_for_commit(full_sha)

            if image_sha in ("ERROR", "NO_CONTAINER_DIR"):
                self.logger.debug(f"Skipping commit {commit_sha}: {image_sha}")
                continue

            if image_sha != prev_image_sha:
                if prev_image_sha is not None and prev_image_sha not in {s for _, s in results}:
                    results.append((prev_commit_sha, prev_image_sha))
                    if len(results) >= n:
                        break
                prev_image_sha = image_sha

            prev_commit_sha = commit_sha

        if len(results) < n and prev_image_sha is not None and prev_image_sha not in {s for _, s in results}:
            results.append((prev_commit_sha, prev_image_sha))

        return list(reversed(results))

    def find_docker_image_sha_origin(self, commit_sha: str) -> Tuple[str, str]:
        """
        Find the commit that first introduced the Docker image SHA at the given commit.

        Walks backward from commit_sha through container/ history until the
        image SHA changes. The oldest commit still matching is the origin.

        Args:
            commit_sha: Git commit SHA (short or full), or "HEAD".

        Returns:
            Tuple of (introducing_commit_sha_9char, image_sha_6char_UPPER).

        Raises:
            ValueError: If the Docker image SHA cannot be computed for commit_sha.
        """
        target_sha = self.generate_docker_image_sha_for_commit(commit_sha, full_hash=True)
        if target_sha in ("ERROR", "NO_CONTAINER_DIR"):
            raise ValueError(f"Cannot compute Docker image SHA for {commit_sha}: {target_sha}")

        result = subprocess.run(
            ["git", "-C", str(self.repo_path), "log",
             commit_sha, "--format=%H", f"-{self.MAX_DOCKER_IMAGE_SHA_LOOKBACK}", "--", "container/"],
            capture_output=True, text=True, check=True,
        )
        container_commits = [h.strip() for h in result.stdout.strip().split('\n') if h.strip()]

        origin = container_commits[0] if container_commits else commit_sha
        for full_sha in container_commits:
            img_sha = self.generate_docker_image_sha_for_commit(full_sha, full_hash=True)
            if img_sha == target_sha:
                origin = full_sha
            else:
                break

        return (origin[:9], target_sha[:6])

    DOCKER_IMAGE_SHA_FILE = ".last_docker_image_sha"
    COMPILATION_SHA_FILE = ".last_compilation_sha"

    def get_stored_docker_image_sha(self) -> str:
        """
        Get stored Docker image SHA from file.

        Returns:
            Stored SHA string, or empty string if not found
        """
        sha_file = self.repo_path / self.DOCKER_IMAGE_SHA_FILE
        if sha_file.exists():
            stored = sha_file.read_text().strip()
            self.logger.debug(f"Found stored Docker image SHA: {stored[:12]}")
            return stored
        self.logger.debug("No stored Docker image SHA found")
        return ""

    def store_docker_image_sha(self, sha: str) -> None:
        """
        Store current Docker image SHA to file.

        Args:
            sha: Docker image SHA to store
        """
        sha_file = self.repo_path / self.DOCKER_IMAGE_SHA_FILE
        sha_file.write_text(sha)
        self.logger.info(f"Stored Docker image SHA: {sha[:12]}")

    def get_stored_compilation_sha(self) -> str:
        """Get stored compilation SHA from file (HEAD commit at last successful compilation)."""
        sha_file = self.repo_path / self.COMPILATION_SHA_FILE
        if sha_file.exists():
            stored = sha_file.read_text().strip()
            self.logger.debug(f"Found stored compilation SHA: {stored}")
            return stored
        self.logger.debug("No stored compilation SHA found")
        return ""

    def store_compilation_sha(self, sha: str) -> None:
        """Store HEAD commit SHA after successful compilation."""
        sha_file = self.repo_path / self.COMPILATION_SHA_FILE
        sha_file.write_text(sha)
        self.logger.info(f"Stored compilation SHA: {sha}")

    def check_if_rebuild_needed(self, force_run: bool = False) -> bool:
        """
        Check if rebuild is needed based on Docker image SHA comparison.

        Compares current Docker image SHA with stored SHA to determine if
        container files have changed since last build.

        Args:
            force_run: If True, proceed with rebuild even if SHA unchanged

        Returns:
            True if rebuild is needed, False otherwise
        """
        self.logger.info("\nChecking if rebuild is needed based on file changes...")
        self.logger.info(f"Docker image SHA file: {self.repo_path}/{self.DOCKER_IMAGE_SHA_FILE}")

        current_sha = self.generate_docker_image_sha(full_hash=True)
        if current_sha in ("NO_CONTAINER_DIR", "ERROR"):
            self.logger.warning(f"Failed to generate Docker image SHA: {current_sha}")
            return True  # Assume rebuild needed

        stored_sha = self.get_stored_docker_image_sha()

        if stored_sha:
            if current_sha == stored_sha:
                if force_run:
                    self.logger.info(f"Docker image SHA unchanged ({current_sha[:12]}) but --run-no-matter-what specified - proceeding")
                    return True
                else:
                    self.logger.info(f"Docker image SHA unchanged ({current_sha[:12]}) - skipping rebuild")
                    self.logger.info("Use --run-no-matter-what to force rebuild")
                    return False  # No rebuild needed
            else:
                self.logger.info("Docker image SHA changed:")
                self.logger.info(f"  Previous: {stored_sha[:12]}")
                self.logger.info(f"  Current:  {current_sha[:12]}")
                self.logger.info("Rebuild needed")
                self.store_docker_image_sha(current_sha)
                return True
        else:
            self.logger.info("No previous Docker image SHA found - rebuild needed")
            self.store_docker_image_sha(current_sha)
            return True


class DockerUtils(BaseUtils):
    """Unified Docker utility class with comprehensive image management."""

    def __init__(self, dry_run: bool = False, verbose: bool = False):
        super().__init__(dry_run, verbose)

        if docker is None:
            self.logger.error("Docker package not found. Install with: pip install docker")
            raise ImportError("docker package required")

        # Initialize Docker client
        try:
            self.logger.debug("Equivalent: docker version")
            self.client = docker.from_env()
            self.client.ping()
            self.logger.debug("Docker client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Docker client: {e}")
            raise

    def _format_size(self, size_bytes: int) -> str:
        """Format size in human readable format."""
        if size_bytes == 0:
            return "0 B"

        size_float = float(size_bytes)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_float < 1024.0:
                return f"{size_float:.1f} {unit}"
            size_float = size_float / 1024.0
        return f"{size_float:.1f} PB"

    def _parse_dynamo_image(self, image_name: str) -> Optional[DynamoImageInfo]:
        """Parse dynamo image name to extract framework, version, and target.

        Walks the hyphen-separated segments of the tag to find a known framework
        and a known target. Everything before the framework is the version/SHA.

        New format (primary):
          dynamo:A1B2C3.d56439ec2-sglang-dev-cuda12.9-amd64
          dynamo:A1B2C3.d56439ec2-trtllm-dev-orig-cuda13.1-amd64
          dynamo:A1B2C3.d56439ec2-vllm-local-dev-cuda12.9-amd64

        Legacy format (still supported; TODO: remove after 2026-04-20):
          dynamo:A1B2C3.d56439ec2-sglang-cuda12.9-dev
          dynamo:A1B2C3.d56439ec2-trtllm-cuda13.1-dev-orig
        """
        all_frameworks = set(FRAMEWORKS) | {"none"}
        known_targets = {"local-dev", "dev-orig", "runtime", "dev"}

        # Strip repo prefix: "dynamo:tag" or "dynamo-base:tag" -> "tag"
        # TODO: remove dynamo-base support after Q3 2026
        if image_name.startswith("dynamo-base:"):
            tag = image_name[len("dynamo-base:"):]
        elif image_name.startswith("dynamo:"):
            tag = image_name[len("dynamo:"):]
        else:
            return None

        # Split tag into segments: e.g. "EBC003.56b448a60-sglang-dev-cuda12.9-amd64"
        parts = tag.split("-")
        if len(parts) < 2:
            return None

        # Find framework: scan all segments for known names
        fw_matches = [(i, parts[i].lower()) for i, part in enumerate(parts) if part.lower() in all_frameworks]
        if not fw_matches:
            return None
        if len(fw_matches) > 1:
            self.logger.error(
                f"Multiple frameworks in tag '{image_name}': "
                f"{', '.join(f'{name!r} at segment {idx}' for idx, name in fw_matches)}"
            )
            return None
        fw_idx, framework = fw_matches[0]
        if fw_idx == 0:
            return None

        version_part = "-".join(parts[:fw_idx])

        # After framework, try to find target by scanning forward from fw_idx+1.
        # New format: target immediately follows framework (before cuda/arch).
        # Legacy format: target is at the end (after cuda).
        remaining = parts[fw_idx + 1:]
        target = ""

        # Forward scan: try longest multi-segment targets first from the start
        # e.g. remaining = ["local", "dev", "cuda12.9", "amd64"] -> "local-dev"
        # e.g. remaining = ["dev", "orig", "cuda12.9", "amd64"] -> "dev-orig"
        for length in range(min(len(remaining), 2), 0, -1):
            candidate = "-".join(remaining[:length])
            if candidate in known_targets:
                target = candidate
                break

        # Fallback: legacy format where target is at the end
        # TODO: remove after 2026-04-20
        if not target:
            for length in range(min(len(remaining), 2), 0, -1):
                candidate = "-".join(remaining[-length:])
                if candidate in known_targets:
                    target = candidate
                    break

        return DynamoImageInfo(
            version=version_part,
            framework=framework,
            target=target,
            latest_tag=None,
        )

    def get_image_info(self, image_name: str) -> Optional[DockerImageInfo]:
        """Get comprehensive information about a Docker image.

        DEPRECATION: V1 + retag script only. V2 uses docker.from_env() directly.
        """
        self.logger.debug(f"Equivalent: docker inspect {image_name}")

        try:
            image = self.client.images.get(image_name)

            # Parse repository and tag
            if ':' in image_name:
                repository, tag = image_name.split(':', 1)
            else:
                repository = image_name
                tag = 'latest'

            # Get basic image info
            size_bytes = image.attrs.get('Size', 0)
            created_at = image.attrs.get('Created', '')
            labels = image.attrs.get('Config', {}).get('Labels') or {}

            # Parse dynamo-specific info
            dynamo_info = self._parse_dynamo_image(image_name)

            return DockerImageInfo(
                name=image_name,
                repository=repository,
                tag=tag,
                image_id=str(image.id),  # Full ID
                created_at=created_at,
                size_bytes=size_bytes,
                size_human=self._format_size(size_bytes),
                labels=labels,
                dynamo_info=dynamo_info
            )

        except Exception as e:
            self.logger.error(f"Failed to get image info for {image_name}: {e}")
            return None

    def list_images(self, name_filter: Optional[str] = None) -> List[DockerImageInfo]:
        """List all Docker images with optional name filtering.

        DEPRECATION: retag script only. V1 and V2 do not use this.

        Returns:
            List of DockerImageInfo objects sorted by creation date (newest first)

            Example return value:
            [
                DockerImageInfo(
                    repository="gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo",
                    tag="21a03b316-38895507-vllm-amd64",
                    image_id="sha256:1234abcd",
                    created_at="2025-11-20T22:15:32",
                    size_mb=13411.0,
                    digest="sha256:c048ae310fcf16471200d512056fdc835b28bdfad7d3df97a46f5ad870541e13"
                ),
                DockerImageInfo(
                    repository="gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo",
                    tag="5fe0476e6-38888909-sglang-arm64",
                    image_id="sha256:5678efgh",
                    created_at="2025-11-19T10:00:00",
                    size_mb=7997.4,
                    digest="sha256:abcd1234..."
                )
            ]
        """
        self.logger.debug("Equivalent: docker images --format table")

        try:
            images = []
            for image in self.client.images.list():
                for tag in image.tags:
                    if tag and (not name_filter or name_filter in tag):
                        image_info = self.get_image_info(tag)
                        if image_info:
                            images.append(image_info)

            # Sort by creation date (newest first)
            images.sort(key=lambda x: x.created_at, reverse=True)
            return images

        except Exception as e:
            self.logger.error(f"Failed to list images: {e}")
            return []

    def list_dynamo_images(self, framework: Optional[str] = None, target: Optional[str] = None, sha: Optional[str] = None) -> List[DockerImageInfo]:
        """List dynamo framework images with optional filtering.

        DEPRECATION: retag script only. V1 and V2 do not use this.
        """
        # Search for both dynamo and dynamo-base images
        dynamo_images = []
        for prefix in ["dynamo:", "dynamo-base:"]:
            images = self.list_images(name_filter=prefix)
            dynamo_images.extend([img for img in images if img.is_dynamo_image()])

        # Apply filters
        if framework:
            framework = normalize_framework(framework)
            dynamo_images = [img for img in dynamo_images
                           if img.dynamo_info and img.dynamo_info.framework == framework]

        if target:
            dynamo_images = [img for img in dynamo_images
                           if img.dynamo_info and img.dynamo_info.target == target]

        if sha:
            dynamo_images = [img for img in dynamo_images if img.matches_sha(sha)]

        return dynamo_images

    def tag_image(self, source_tag: str, target_tag: str) -> bool:
        """Tag a Docker image.

        DEPRECATION: retag script only. V1 and V2 do not use this.
        """
        self.logger.debug(f"Equivalent: docker tag {source_tag} {target_tag}")

        if self.dry_run:
            self.logger.info(f"DRY RUN: Would execute: docker tag {source_tag} {target_tag}")
            return True

        try:
            source_image = self.client.images.get(source_tag)

            # Parse target tag
            if ':' in target_tag:
                repository, tag = target_tag.split(':', 1)
            else:
                repository = target_tag
                tag = 'latest'

            source_image.tag(repository, tag)
            self.logger.info(f"✓ Tagged: {source_tag} -> {target_tag}")
            return True

        except Exception as e:
            self.logger.error(f"✗ Failed to tag {source_tag} -> {target_tag}: {e}")
            return False

    def retag_to_latest(self, images: List[DockerImageInfo]) -> Dict[str, int]:
        """Retag multiple images to their latest tags.

        DEPRECATION: retag script only. V1 and V2 do not use this.
        """
        results = {'success': 0, 'failed': 0}

        for image in images:
            if image.is_dynamo_framework_image():
                latest_tag = image.get_latest_tag()
                if self.tag_image(image.name, latest_tag):
                    results['success'] += 1
                else:
                    results['failed'] += 1

        return results


