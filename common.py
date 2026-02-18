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

# Log/snippet detection lives in the shared library: `dynamo-utils/ci_log_errors/`.
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

    @staticmethod
    def format_one_line_dict(d: Dict[str, Any], *, keys: Optional[List[str]] = None) -> str:
        """Format a dict of {phase: seconds} similarly to `format_one_line()`."""
        try:
            dd: Dict[str, float] = {str(k): float(v) for k, v in (d or {}).items() if v is not None}
        except (ValueError, TypeError):
            dd = {}
        if keys:
            parts = []
            for k in keys:
                parts.append(f"{k}={PhaseTimer.format_seconds(dd.get(k, 0.0))}")
            return ", ".join(parts)
        ks = sorted([k for k in dd.keys() if k != "total"])
        parts = [f"{k}={PhaseTimer.format_seconds(dd.get(k, 0.0))}" for k in ks]
        if "total" in dd:
            parts.append(f"total={PhaseTimer.format_seconds(dd.get('total', 0.0))}")
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

    def get_commit(self, sha: str) -> Optional[Any]:
        """Get commit object by SHA using GitPython API.

        Args:
            sha: Commit SHA (full or short)

        Returns:
            GitPython commit object or None if not found
        """
        try:
            return self.repo.commit(sha)
        except Exception as e:
            self.logger.error(f"Failed to get commit {sha}: {e}")
            return None

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

    def get_commit_info(self, commit: Any) -> Dict[str, Any]:
        """Extract information from a commit object.

        Args:
            commit: GitPython commit object

        Returns:
            Dictionary with commit information

            Example return value:
            {
                "sha_full": "21a03b316dc1e5031183965e5798b0d9fe2e64b3",
                "sha_short": "21a03b316",
                "author_name": "John Doe",
                "author_email": "john@nvidia.com",
                "committer_name": "John Doe",
                "committer_email": "john@nvidia.com",
                "date": datetime.datetime(2025, 11, 20, 17, 5, 58),
                "message": "Fix Docker image fetching for recent commits\\n\\nDetailed description...",
                "message_first_line": "Fix Docker image fetching for recent commits",
                "parents": ["5fe0476e605d2564234f00e8123461e1594a9ce7"]
            }
        """

        return {
            'sha_full': commit.hexsha,
            'sha_short': commit.hexsha[:9],
            'author_name': commit.author.name,
            'author_email': commit.author.email,
            'committer_name': commit.committer.name,
            'committer_email': commit.committer.email,
            'date': datetime.fromtimestamp(commit.committed_date),
            'message': commit.message.strip(),
            'message_first_line': commit.message.split('\n')[0] if commit.message else '',
            'parents': [p.hexsha for p in commit.parents]
        }

    def is_dirty(self) -> bool:
        """Check if repository has uncommitted changes.

        Returns:
            True if there are uncommitted changes, False otherwise
        """
        return self.repo.is_dirty()

    def get_untracked_files(self) -> List[str]:
        """Get list of untracked files.

        Returns:
            List of untracked file paths

            Example return value:
            [
                "test_output.txt",
                ".env.local",
                "debug.log",
                "temp/cache_data.json"
            ]
        """
        return self.repo.untracked_files

    def get_tags(self) -> List[str]:
        """Get all repository tags.

        Returns:
            List of tag names

            Example return value:
            [
                "v1.0.0",
                "v1.0.1",
                "v1.1.0",
                "release-2025-11-20"
            ]
        """
        return [tag.name for tag in self.repo.tags]

    def get_branches(self, remote: bool = False) -> List[str]:
        """Get list of branches.

        Args:
            remote: If True, return remote branches. If False, return local branches.

        Returns:
            List of branch names

            Example return value (local):
            ["main", "feature/docker-caching", "bugfix/timezone-issue"]

            Example return value (remote):
            ["origin/main", "origin/develop", "origin/feature/docker-caching"]
        """
        if remote:
            return [ref.name for ref in self.repo.remote().refs]
        else:
            return [head.name for head in self.repo.heads]



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

        semver_branches = []
        for br in branches:
            ver = parse_semver(br)
            if ver is None:
                continue
            semver_branches.append((ver, br))
        semver_branches.sort(key=lambda x: x[0], reverse=True)
        selected = [br for _, br in semver_branches[:limit]]

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

            ver_tail = branch_ref.split("/")[-1].lstrip(".")
            label = f"release/{ver_tail}" if ver_tail.startswith("v") else f"release/v{ver_tail}"
            branch_name = "/".join(branch_ref.split("/")[1:])  # drop "origin/"
            url = f"https://github.com/{github_owner}/{github_repo}/tree/{urllib.parse.quote(branch_name, safe='')}"

            fork_map.setdefault(mb, []).append({"label": label, "url": url, "branch": branch_name})

        return fork_map

    def generate_docker_image_sha(self, full_hash: bool = False) -> str:
        """
        Generate Docker image SHA for the current HEAD commit.

        Delegates to generate_docker_image_sha_for_commit("HEAD") so the result
        is deterministic (only committed files are hashed, untracked/gitignored
        artifacts like rendered Dockerfiles are excluded).

        Args:
            full_hash: If True, return full 64-char SHA. If False, return first 12 chars.

        Returns:
            SHA256 hash string, or error code:
            - "NO_CONTAINER_DIR": container directory doesn't exist
            - "NO_FILES": no relevant files found
            - "ERROR": error during calculation
        """
        return self.generate_docker_image_sha_for_commit("HEAD", full_hash=full_hash)

    def generate_docker_image_sha_for_commit(self, commit_sha: str, full_hash: bool = False) -> str:
        """
        Generate Docker image SHA for a specific commit using git plumbing (no checkout required).

        Hashes only committed (tracked) files under container/, so the result is
        deterministic regardless of untracked/gitignored artifacts on disk.

        Args:
            commit_sha: Git commit SHA (short or full), or "HEAD".
            full_hash: If True, return full 64-char SHA. If False, return first 12 chars.

        Returns:
            SHA256 hash string, or error code:
            - "NO_CONTAINER_DIR": no container/ tree at that commit
            - "NO_FILES": no relevant files after filtering
            - "ERROR": git command failed
        """
        excluded_extensions = {'.md', '.rst', '.log', '.bak', '.tmp', '.swp', '.swo', '.orig', '.rej'}
        excluded_filenames = {'README', 'CHANGELOG', 'LICENSE', 'NOTICE', 'AUTHORS', 'CONTRIBUTORS'}
        excluded_specific = {'launch_message.txt'}

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

        # Parse ls-tree output: "<mode> <type> <blob_hash>\t<path>"
        entries = []
        for line in result.stdout.strip().split('\n'):
            meta, path = line.split('\t', 1)
            _mode, _type, blob_hash = meta.split()
            if _type != "blob":
                continue

            rel_path = Path(path)
            parts_under_container = rel_path.relative_to("container").parts
            if any(part.startswith('.') for part in parts_under_container):
                continue
            if rel_path.suffix.lower() in excluded_extensions:
                continue
            if rel_path.stem.upper() in excluded_filenames:
                continue
            if rel_path.name.lower() in excluded_specific:
                continue

            entries.append((str(rel_path), blob_hash))

        if not entries:
            return "NO_FILES"

        # git ls-tree output is already sorted lexicographically
        hasher = hashlib.sha256()
        for file_rel_path, blob_hash in entries:
            content = subprocess.run(
                ["git", "-C", str(self.repo_path), "cat-file", "-p", blob_hash],
                capture_output=True, check=True,
            ).stdout
            hasher.update(file_rel_path.encode('utf-8'))
            hasher.update(b'\n')
            hasher.update(content)
            hasher.update(b'\n')

        sha_full = hasher.hexdigest()
        return sha_full if full_hash else sha_full[:12]

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
            List of tuples [(commit_sha_9char, image_sha_7char), ...] in
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

            if image_sha in ("ERROR", "NO_CONTAINER_DIR", "NO_FILES"):
                self.logger.debug(f"Skipping commit {commit_sha}: {image_sha}")
                continue

            if image_sha != prev_image_sha:
                if prev_image_sha is not None and prev_image_sha not in {s for _, s in results}:
                    results.append((prev_commit_sha, prev_image_sha[:7]))
                    if len(results) >= n:
                        break
                prev_image_sha = image_sha

            prev_commit_sha = commit_sha

        if len(results) < n and prev_image_sha is not None and prev_image_sha not in {s for _, s in results}:
            results.append((prev_commit_sha, prev_image_sha[:7]))

        return list(reversed(results))

    DOCKER_IMAGE_SHA_FILE = ".last_docker_image_sha"
    DOCKER_IMAGE_SHA_FILE_COMPAT = ".last_build_composite_sha"  # TODO: remove after 2026-02-25

    def get_stored_docker_image_sha(self) -> str:
        """
        Get stored Docker image SHA from file.

        Returns:
            Stored SHA string, or empty string if not found
        """
        sha_file = self.repo_path / self.DOCKER_IMAGE_SHA_FILE
        compat_file = self.repo_path / self.DOCKER_IMAGE_SHA_FILE_COMPAT
        if sha_file.exists():
            stored = sha_file.read_text().strip()
            self.logger.debug(f"Found stored Docker image SHA: {stored[:12]}")
            return stored
        if compat_file.exists() and not compat_file.is_symlink():
            stored = compat_file.read_text().strip()
            self.logger.debug(f"Found stored Docker image SHA (compat): {stored[:12]}")
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
        compat_file = self.repo_path / self.DOCKER_IMAGE_SHA_FILE_COMPAT
        sha_file.write_text(sha)
        if compat_file.exists() or compat_file.is_symlink():
            compat_file.unlink()
        compat_file.symlink_to(self.DOCKER_IMAGE_SHA_FILE)
        self.logger.info(f"Stored Docker image SHA: {sha[:12]}")

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
        if current_sha in ("NO_CONTAINER_DIR", "NO_FILES", "ERROR"):
            self.logger.warning(f"Failed to generate Docker image SHA: {current_sha}")
            return True  # Assume rebuild needed

        stored_sha = self.get_stored_docker_image_sha()

        if stored_sha:
            if current_sha == stored_sha:
                if force_run:
                    self.logger.info(f"Docker image SHA unchanged ({current_sha[:12]}) but --run-ignore-lock specified - proceeding")
                    return True
                else:
                    self.logger.info(f"Docker image SHA unchanged ({current_sha[:12]}) - skipping rebuild")
                    self.logger.info("Use --run-ignore-lock to force rebuild")
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
        and a known target. Everything before the framework is the version/SHA,
        everything after is optional attributes + target. There is only one
        framework and one target per SHA.

        Supported formats (non-exhaustive):
          dynamo:v0.1.0.dev.ea07d51fc-sglang-local-dev
          dynamo:d56439ec2-sglang-local-dev
          dynamo:d56439ec2-sglang-cuda12.9-local-dev
          dynamo:d56439ec2-trtllm-cuda13.1-dev-orig
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

        # Split tag into segments: e.g. "ea02149e4-sglang-cuda12.9-local-dev"
        # -> ["ea02149e4", "sglang", "cuda12.9", "local", "dev"]
        parts = tag.split("-")
        if len(parts) < 2:
            return None

        # Find frameworks: scan all segments for known names
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

        # Find target: try joining trailing segments (longest match first)
        # e.g. parts = [..., "local", "dev"] -> try "local-dev" then "dev"
        remaining = parts[fw_idx + 1:]
        target_matches = []
        for length in range(len(remaining), 0, -1):
            candidate = "-".join(remaining[-length:])
            if candidate in known_targets:
                target_matches.append(candidate)
        if len(target_matches) > 1:
            # Multiple lengths matched (e.g. "dev" and "local-dev"); longest wins,
            # but if two non-overlapping targets matched that's a real collision.
            unique = set(target_matches)
            # "local-dev" contains "dev", so if both matched it's not a collision
            # (longest-first already picked the right one). But if genuinely
            # different targets appear, error out.
            longest = max(unique, key=len)
            non_substrings = {t for t in unique if t not in longest}
            if non_substrings:
                self.logger.error(
                    f"Multiple targets in tag '{image_name}': {sorted(unique)}"
                )
                return None
        target = target_matches[0] if target_matches else ""

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

    def filter_unused_build_args(self, docker_command: str) -> str:
        """Remove unused --build-arg flags from Docker build commands for base images.

        DEPRECATION: V1 only. V2 does not use this.

        Base images (dynamo-base) don't use most build arguments. Removing unused
        args helps Docker recognize when builds are truly identical.
        """

        if not re.search(r'--tag\s+dynamo-base:', docker_command):
            # Only filter base image builds
            return docker_command

        # List of build args that are typically unused by base images
        unused_args = {
            'PYTORCH_VERSION', 'CUDA_VERSION', 'PYTHON_VERSION',
            'FRAMEWORK_VERSION', 'TARGET_ARCH', 'BUILD_TYPE'
        }

        # Split command into parts
        parts = docker_command.split()
        filtered_parts = []
        filtered_args = []

        i = 0
        while i < len(parts):
            if parts[i] == '--build-arg' and i + 1 < len(parts):
                arg_name = parts[i + 1].split('=')[0]
                if arg_name in unused_args:
                    filtered_args.append(arg_name)
                    i += 2  # Skip both --build-arg and its value
                else:
                    filtered_parts.extend([parts[i], parts[i + 1]])
                    i += 2
            else:
                filtered_parts.append(parts[i])
                i += 1

        if filtered_args and self.verbose:
            self.logger.info(f"Filtered {len(filtered_args)} unused base image build args: {', '.join(sorted(filtered_args))}")

        return ' '.join(filtered_parts)

    def normalize_command(self, docker_command: str) -> str:
        """Normalize Docker command by removing whitespace and sorting build args.

        DEPRECATION: V1 only. V2 does not use this.

        Helps identify functionally identical commands with different formatting.
        """

        # Remove extra whitespace and normalize
        normalized = ' '.join(docker_command.split())

        # Sort build args to make commands with same args but different order equivalent
        # Find all --build-arg KEY=VALUE pairs
        build_args = []
        other_parts = []

        parts = normalized.split()
        i = 0
        while i < len(parts):
            if parts[i] == '--build-arg' and i + 1 < len(parts):
                build_args.append(f"--build-arg {parts[i + 1]}")
                i += 2
            else:
                other_parts.append(parts[i])
                i += 1

        # Sort build args for consistent ordering
        build_args.sort()

        # Reconstruct command with sorted build args
        if build_args:
            # Insert sorted build args after 'docker build' but before other args
            docker_build_idx = -1
            for idx, part in enumerate(other_parts):
                if part == 'build':
                    docker_build_idx = idx
                    break

            if docker_build_idx >= 0:
                result_parts = other_parts[:docker_build_idx + 1] + build_args + other_parts[docker_build_idx + 1:]
            else:
                result_parts = other_parts + build_args
        else:
            result_parts = other_parts

        return ' '.join(result_parts)

    def extract_base_image_from_command(self, docker_cmd: str) -> str:
        """Extract the base/FROM image from docker build command arguments"""

        # Look for --build-arg DYNAMO_BASE_IMAGE=... (framework-specific builds)
        match = re.search(r'--build-arg\s+DYNAMO_BASE_IMAGE=([^\s]+)', docker_cmd)
        if match:
            return match.group(1)

        # Look for --build-arg BASE_IMAGE=... and BASE_IMAGE_TAG=... (base builds)
        base_image_match = re.search(r'--build-arg\s+BASE_IMAGE=([^\s]+)', docker_cmd)
        base_tag_match = re.search(r'--build-arg\s+BASE_IMAGE_TAG=([^\s]+)', docker_cmd)

        if base_image_match and base_tag_match:
            return f"{base_image_match.group(1)}:{base_tag_match.group(1)}"
        elif base_image_match:
            return base_image_match.group(1)

        # Look for --build-arg DEV_BASE=... (local-dev builds)
        dev_base_match = re.search(r'--build-arg\s+DEV_BASE=([^\s]+)', docker_cmd)
        if dev_base_match:
            return dev_base_match.group(1)

        # Return empty string if no base image found
        return ""

    def extract_image_tag_from_command(self, docker_cmd: str) -> str:
        """
        Extract the output tag from docker build command --tag argument.
        Returns the tag string, or empty string if no tag found.
        Raises error if multiple tags are found (should not happen after get_build_commands validation).
        """

        # Find all --tag arguments in the command
        tags = re.findall(r'--tag\s+([^\s]+)', docker_cmd)

        if len(tags) == 0:
            return ""
        elif len(tags) == 1:
            return tags[0]
        else:
            # This should not happen if get_build_commands validation is working
            self.logger.error(f"Multiple --tag arguments found in command: {tags}")
            return tags[0]  # Return first tag as fallback


# GitHub API utilities


def select_shas_for_network_fetch(
    sha_list: List[str],
    sha_to_datetime: Optional[Dict[str, datetime]],
    *,
    max_fetch: int = 5,
    recent_hours: int = DEFAULT_STABLE_AFTER_HOURS,
) -> set[str]:
    """Select a small set of SHAs where we're allowed to do network fetches.

    Policy:
    - Only consider the newest `max_fetch` SHAs (order as given in sha_list).
    - Only allow fetch if the SHA's datetime is within `recent_hours`.
    """
    allow: set[str] = set()
    if not sha_list or not sha_to_datetime:
        return allow

    now_utc = datetime.now(timezone.utc)
    cutoff_s = float(recent_hours) * 3600.0
    for sha in sha_list[: int(max_fetch or 0)]:
        dt = sha_to_datetime.get(sha)
        if dt is None:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_s = (now_utc - dt.astimezone(timezone.utc)).total_seconds()
        if age_s < cutoff_s:
            allow.add(sha)
    return allow


