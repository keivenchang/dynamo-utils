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
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

# Log/snippet detection lives in `common_log_errors.py`.

# Global logger for the module
_logger = logging.getLogger(__name__)

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
DEFAULT_UNSTABLE_TTL_S: int = 300
# ^ "Fast refresh" TTL (seconds) for things that change quickly.
#   Example: for a commit from ~5 minutes ago, we may re-check GitHub Actions status every 5 minutes.
DEFAULT_STABLE_TTL_S: int = 30 * 24 * 3600
# ^ "Stable" TTL (seconds) for older/less-changing data.
#   Example: for a commit from last week, we only refresh its GitHub Actions status about once per 30 days
#   (because it is past `DEFAULT_STABLE_AFTER_HOURS`).
DEFAULT_OPEN_PRS_TTL_S: int = 300
# ^ TTL (seconds) for caches keyed by *open* PRs (expected to change often).
#   Example: required-checks / PR metadata for an open PR can be refreshed every 5 minutes.
DEFAULT_CLOSED_PRS_TTL_S: int = 14 * 24 * 3600
# ^ TTL (seconds) for caches keyed by *closed/merged* PRs (mostly stable).
#   Example: merge-date for a merged PR is basically immutable, but we still allow a refresh every ~14 days.
DEFAULT_NO_PR_TTL_S: int = 24 * 3600
# ^ TTL (seconds) for "negative cache" entries (when we *didn't* find something).
#   Example: if SHA→PR mapping wasn't found, we remember that for 24 hours so we don't re-query constantly,
#   but we retry later in case the mapping appears.
DEFAULT_RAW_LOG_URL_TTL_S: int = 3600
# ^ TTL (seconds) for GitHub job raw-log *redirect URLs* (signed + short-lived).
#   Example: a `Location:` URL returned by `/actions/jobs/{id}/logs` can expire in ~1 hour.
DEFAULT_RAW_LOG_TEXT_TTL_S: int = 30 * 24 * 3600
# ^ TTL (seconds) for cached raw log *content* (the downloaded text we parse/snippet locally).
#   Example: once we download job `59030780729`, keep its `.log` for ~30 days to avoid re-downloading.
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
# Do NOT write caches into a repo checkout (e.g. `dynamo_latest/.cache/...`), because:
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

try:
    import docker  # type: ignore[import-untyped]
except ImportError:
    docker = None  # type: ignore[assignment]

try:
    import yaml  # type: ignore[import-not-found]
    HAS_YAML = True
except ImportError:
    yaml = None  # type: ignore[assignment]
    HAS_YAML = False

try:
    import requests  # type: ignore[import-not-found]
    HAS_REQUESTS = True
except ImportError:
    requests = None  # type: ignore[assignment]
    HAS_REQUESTS = False

# GitPython is required - hard error if not installed
try:
    import git  # type: ignore[import-not-found]
except ImportError as e:
    raise ImportError("GitPython is required. Install with: pip install gitpython") from e

# Supported frameworks
# Used by V2 for default framework list and validation
FRAMEWORKS_UPPER = ["VLLM", "SGLANG", "TRTLLM"]

# DEPRECATED: V1 and retag script reference only - kept for backward compatibility
# V2 uses FRAMEWORKS_UPPER directly
FRAMEWORKS = [f.lower() for f in FRAMEWORKS_UPPER]
FRAMEWORK_NAMES = {"vllm": "VLLM", "sglang": "SGLang", "trtllm": "TensorRT-LLM"}

# Marker file suffixes (shared by build_images.py and show_commit_history.py)
MARKER_RUNNING = 'RUNNING'
MARKER_PASSED = 'PASSED'
MARKER_FAILED = 'FAILED'
MARKER_KILLED = 'KILLED'

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
            except Exception:
                pass

        # Method 3: Use shutil.get_terminal_size() (ioctl-based)
        if term_width is None:
            term_width = shutil.get_terminal_size().columns - padding

    except Exception:
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
    """Utilities for Dynamo repository operations including Composite Docker SHA (CDS) calculation."""

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
            except Exception:
                # Non-fatal; proceed with whatever refs we have
                pass

        try:
            out = git_utils.repo.git.for_each_ref("--format=%(refname:short)", release_ref_glob)
            branches = [b.strip() for b in out.splitlines() if b.strip()]
        except Exception:
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
            except Exception:
                mb = ""

            # Fallback: normal merge-base
            if not mb:
                try:
                    mb = (git_utils.repo.git.merge_base(base_ref, branch_ref) or "").strip()
                except Exception:
                    mb = ""

            if not mb:
                continue

            ver_tail = branch_ref.split("/")[-1].lstrip(".")
            label = f"release/{ver_tail}" if ver_tail.startswith("v") else f"release/v{ver_tail}"
            branch_name = "/".join(branch_ref.split("/")[1:])  # drop "origin/"
            url = f"https://github.com/{github_owner}/{github_repo}/tree/{urllib.parse.quote(branch_name, safe='')}"

            fork_map.setdefault(mb, []).append({"label": label, "url": url, "branch": branch_name})

        return fork_map

    def generate_composite_sha(self, full_hash: bool = False) -> str:
        """
        Generate Composite Docker SHA (CDS) from container directory files.

        This creates a SHA256 hash of all relevant files in the container directory,
        excluding documentation, temporary files, etc. This hash can be used to
        determine if a rebuild is needed.

        Args:
            full_hash: If True, return full 64-char SHA. If False, return first 12 chars.

        Returns:
            SHA256 hash string, or error code:
            - "NO_CONTAINER_DIR": container directory doesn't exist
            - "NO_FILES": no relevant files found
            - "ERROR": error during calculation
        """

        container_dir = self.repo_path / "container"
        if not container_dir.exists():
            self.logger.warning(f"Container directory not found: {container_dir}")
            return "NO_CONTAINER_DIR"

        # Excluded patterns (matching V1)
        excluded_extensions = {'.md', '.rst', '.log', '.bak', '.tmp', '.swp', '.swo', '.orig', '.rej'}
        excluded_filenames = {'README', 'CHANGELOG', 'LICENSE', 'NOTICE', 'AUTHORS', 'CONTRIBUTORS'}
        excluded_specific = {'launch_message.txt'}

        # Collect files to hash
        files_to_hash = []
        for file_path in sorted(container_dir.rglob('*')):
            if not file_path.is_file():
                continue
            # Skip hidden files
            if any(part.startswith('.') for part in file_path.relative_to(container_dir).parts):
                continue
            # Skip excluded extensions
            if file_path.suffix.lower() in excluded_extensions:
                continue
            # Skip excluded names
            if file_path.stem.upper() in excluded_filenames:
                continue
            # Skip specific files
            if file_path.name.lower() in excluded_specific:
                continue
            files_to_hash.append(file_path.relative_to(self.repo_path))

        if not files_to_hash:
            self.logger.warning("No files found to hash in container directory")
            return "NO_FILES"

        self.logger.debug(f"Hashing {len(files_to_hash)} files from container directory")

        # Calculate Composite Docker SHA (CDS)
        try:
            with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                try:
                    for file_rel_path in files_to_hash:
                        full_path = self.repo_path / file_rel_path
                        if full_path.exists():
                            temp_file.write(str(file_rel_path).encode('utf-8'))
                            temp_file.write(b'\n')
                            with open(full_path, 'rb') as f:
                                temp_file.write(f.read())
                            temp_file.write(b'\n')

                    temp_file.flush()
                    with open(temp_path, 'rb') as f:
                        sha_full = hashlib.sha256(f.read()).hexdigest()
                        result = sha_full if full_hash else sha_full[:12]
                        self.logger.debug(f"Composite Docker SHA: {result}")
                        return result
                finally:
                    temp_path.unlink(missing_ok=True)
        except Exception as e:
            self.logger.error(f"Error calculating Composite Docker SHA (CDS): {e}")
            return "ERROR"

    def get_stored_composite_sha(self) -> str:
        """
        Get stored Composite Docker SHA (CDS) from file.

        Returns:
            Stored SHA string, or empty string if not found
        """
        sha_file = self.repo_path / ".last_build_composite_sha"
        if sha_file.exists():
            stored = sha_file.read_text().strip()
            self.logger.debug(f"Found stored Composite Docker SHA (CDS): {stored[:12]}")
            return stored
        self.logger.debug("No stored Composite Docker SHA (CDS) found")
        return ""

    def store_composite_sha(self, sha: str) -> None:
        """
        Store current Composite Docker SHA (CDS) to file.

        Args:
            sha: Composite Docker SHA (CDS) to store
        """
        sha_file = self.repo_path / ".last_build_composite_sha"
        sha_file.write_text(sha)
        self.logger.info(f"Stored Composite Docker SHA (CDS): {sha[:12]}")

    def check_if_rebuild_needed(self, force_run: bool = False) -> bool:
        """
        Check if rebuild is needed based on Composite Docker SHA (CDS) comparison.

        Compares current Composite Docker SHA (CDS) with stored SHA to determine if
        container files have changed since last build.

        Args:
            force_run: If True, proceed with rebuild even if SHA unchanged

        Returns:
            True if rebuild is needed, False otherwise
        """
        self.logger.info("\nChecking if rebuild is needed based on file changes...")
        self.logger.info(f"Composite Docker SHA file: {self.repo_path}/.last_build_composite_sha")

        # Generate current Composite Docker SHA (CDS) (full hash, not truncated)
        current_sha = self.generate_composite_sha(full_hash=True)
        if current_sha in ("NO_CONTAINER_DIR", "NO_FILES", "ERROR"):
            self.logger.warning(f"Failed to generate Composite Docker SHA (CDS): {current_sha}")
            return True  # Assume rebuild needed

        # Get stored Composite Docker SHA (CDS)
        stored_sha = self.get_stored_composite_sha()

        if stored_sha:
            if current_sha == stored_sha:
                if force_run:
                    self.logger.info(f"Composite Docker SHA (CDS) unchanged ({current_sha[:12]}) but --force-run specified - proceeding")
                    return True
                else:
                    self.logger.info(f"Composite Docker SHA (CDS) unchanged ({current_sha[:12]}) - skipping rebuild")
                    self.logger.info("Use --force-run to force rebuild")
                    return False  # No rebuild needed
            else:
                self.logger.info("Composite Docker SHA (CDS) changed:")
                self.logger.info(f"  Previous: {stored_sha[:12]}")
                self.logger.info(f"  Current:  {current_sha[:12]}")
                self.logger.info("Rebuild needed")
                self.store_composite_sha(current_sha)
                return True
        else:
            self.logger.info("No previous Composite Docker SHA (CDS) found - rebuild needed")
            self.store_composite_sha(current_sha)
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
        """Parse dynamo image name to extract framework and version info."""
        # Pattern for dynamo images: (dynamo|dynamo-base):v{version}-{framework}-{target}
        # Examples:
        #   dynamo:v0.1.0.dev.ea07d51fc-sglang-local-dev
        #   dynamo-base:v0.1.0.dev.ea07d51fc-vllm-dev
        pattern = r'^(?:dynamo|dynamo-base):v(.+?)-([^-]+)(?:-(.+))?$'
        match = re.match(pattern, image_name)

        if not match:
            return None

        version_part, framework, target = match.groups()

        # Validate framework
        normalized_framework = normalize_framework(framework)
        if normalized_framework not in FRAMEWORKS:
            return None

        return DynamoImageInfo(
            version=version_part,
            framework=normalized_framework,
            target=target or "",
            latest_tag=None  # Will be computed later if needed
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
                image_id=str(image.id) if getattr(image, "id", None) else "",  # Full ID
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

@dataclass
class FailedCheck:
    """Information about a failed CI check"""
    name: str
    job_url: str
    run_id: str
    duration: str
    # Raw job log download URL (usually a time-limited blob URL). Optional because it may
    # require auth, may be unavailable for older jobs, or may fail due to rate limits.
    raw_log_url: Optional[str] = None
    is_required: bool = False
    error_summary: Optional[str] = None


@dataclass
class RunningCheck:
    """Information about a running CI check"""
    name: str
    check_url: str
    is_required: bool = False
    elapsed_time: Optional[str] = None


@dataclass
class PRInfo:
    """Pull request information"""
    number: int
    title: str
    url: str
    state: str
    is_merged: bool
    review_decision: Optional[str]
    mergeable_state: str
    unresolved_conversations: int
    ci_status: Optional[str]
    # Head commit SHA for the PR (used to link to per-commit checks page).
    head_sha: Optional[str] = None
    base_ref: Optional[str] = None
    created_at: Optional[str] = None
    has_conflicts: bool = False
    conflict_message: Optional[str] = None
    blocking_message: Optional[str] = None
    # Names of required status checks for this PR's base branch (from GitHub branch protection).
    # Used to label checks as REQUIRED vs optional in the UI.
    required_checks: List[str] = field(default_factory=list)
    failed_checks: List['FailedCheck'] = field(default_factory=list)
    running_checks: List['RunningCheck'] = field(default_factory=list)
    rerun_url: Optional[str] = None


@dataclass(frozen=True)
class GHPRCheckRow:
    """A single row from `gh pr checks` (tab-separated output)."""

    name: str
    status_raw: str
    duration: str = ""
    url: str = ""
    description: str = ""
    is_required: bool = False

    @property
    def status_norm(self) -> str:
        s = (self.status_raw or "").strip().lower()
        # gh pr checks typically uses: pass, fail, skipped, cancelled, pending, in_progress, queued, neutral
        if s in {"pass", "success"}:
            return "success"
        # Treat "skipped"/"neutral" as success-like for UI aggregation.
        if s in {"skipped", "skip", "neutral"}:
            return "success"
        if s in {"fail", "failure", "timed_out", "action_required"}:
            return "failure"
        if s in {"in_progress", "running"}:
            return "in_progress"
        if s in {"pending", "queued"}:
            return "pending"
        if s in {"cancelled", "canceled"}:
            return "cancelled"
        return "unknown"


@dataclass(frozen=True)
class GitHubChecksCounts:
    """Typed bucket counts for GitHub checks.

    Keep this as a dataclass (not a Dict[str, int]) so:
    - we don’t have a pile of literal-string keys spread across the codebase
    - type-checkers can catch mistakes
    """

    success_required: int = 0
    success_optional: int = 0
    failure_required: int = 0
    failure_optional: int = 0
    in_progress_required: int = 0
    in_progress_optional: int = 0
    pending: int = 0
    cancelled: int = 0
    other: int = 0
    total: int = 0


@dataclass(frozen=True)
class GitHubChecksNames:
    """Typed bucket name lists for GitHub checks (for tooltips)."""

    success_required: Tuple[str, ...] = field(default_factory=tuple)
    success_optional: Tuple[str, ...] = field(default_factory=tuple)
    failure_required: Tuple[str, ...] = field(default_factory=tuple)
    failure_optional: Tuple[str, ...] = field(default_factory=tuple)
    in_progress_required: Tuple[str, ...] = field(default_factory=tuple)
    in_progress_optional: Tuple[str, ...] = field(default_factory=tuple)
    pending: Tuple[str, ...] = field(default_factory=tuple)
    cancelled: Tuple[str, ...] = field(default_factory=tuple)
    other: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GitHubChecksSummary:
    """Bucketed GitHub checks summary (shared by multiple HTML generators)."""

    counts: GitHubChecksCounts
    names: GitHubChecksNames


def summarize_pr_check_rows(rows: Iterable[GHPRCheckRow]) -> GitHubChecksSummary:
    """Summarize `gh pr checks` rows into bucketed counts + name lists.

    Args:
        rows: Iterable of `GHPRCheckRow`

    Returns:
        GitHubChecksSummary where `names[...]` holds raw check names for tooltips.
    """
    # Use local scalars/lists for clarity and performance, then freeze into dataclasses at the end.
    success_required = 0
    success_optional = 0
    failure_required = 0
    failure_optional = 0
    in_progress_required = 0
    in_progress_optional = 0
    pending = 0
    cancelled = 0
    other = 0

    names_success_required: List[str] = []
    names_success_optional: List[str] = []
    names_failure_required: List[str] = []
    names_failure_optional: List[str] = []
    names_in_progress_required: List[str] = []
    names_in_progress_optional: List[str] = []
    names_pending: List[str] = []
    names_cancelled: List[str] = []
    names_other: List[str] = []

    for r in rows:
        try:
            name = str(getattr(r, "name", "") or "")
            status = str(getattr(r, "status_norm", "") or "unknown")
            is_req = bool(getattr(r, "is_required", False))
        except Exception:
            continue

        if status == "success":
            if is_req:
                success_required += 1
                if name:
                    names_success_required.append(name)
            else:
                success_optional += 1
                if name:
                    names_success_optional.append(name)
        elif status == "failure":
            if is_req:
                failure_required += 1
                if name:
                    names_failure_required.append(name)
            else:
                failure_optional += 1
                if name:
                    names_failure_optional.append(name)
        elif status == "in_progress":
            if is_req:
                in_progress_required += 1
                if name:
                    names_in_progress_required.append(name)
            else:
                in_progress_optional += 1
                if name:
                    names_in_progress_optional.append(name)
        elif status == "pending":
            pending += 1
            if name:
                names_pending.append(name)
        elif status == "cancelled":
            cancelled += 1
            if name:
                names_cancelled.append(name)
        else:
            other += 1
            if name:
                names_other.append(name)

    total = (
        success_required
        + success_optional
        + failure_required
        + failure_optional
        + in_progress_required
        + in_progress_optional
        + pending
        + cancelled
        + other
    )
    return GitHubChecksSummary(
        counts=GitHubChecksCounts(
            success_required=success_required,
            success_optional=success_optional,
            failure_required=failure_required,
            failure_optional=failure_optional,
            in_progress_required=in_progress_required,
            in_progress_optional=in_progress_optional,
            pending=pending,
            cancelled=cancelled,
            other=other,
            total=total,
        ),
        names=GitHubChecksNames(
            success_required=tuple(names_success_required),
            success_optional=tuple(names_success_optional),
            failure_required=tuple(names_failure_required),
            failure_optional=tuple(names_failure_optional),
            in_progress_required=tuple(names_in_progress_required),
            in_progress_optional=tuple(names_in_progress_optional),
            pending=tuple(names_pending),
            cancelled=tuple(names_cancelled),
            other=tuple(names_other),
        ),
    )


def summarize_check_runs(check_runs: Iterable[Dict[str, Any]]) -> GitHubChecksSummary:
    """Summarize commit check-runs (GitHub REST `.../check-runs`) into buckets.

    Input shape (examples):
      - check_runs entry:
        {
          "name": "build-test-amd64",
          "status": "completed"|"in_progress"|"queued",
          "conclusion": "success"|"failure"|"cancelled"|"skipped"|None,
          "html_url": "https://..."
          "is_required": true|false,   # optional; callers can pre-annotate
        }

    Returns:
        GitHubChecksSummary with the same bucket keys as `summarize_pr_check_rows`.
    """
    success_required = 0
    success_optional = 0
    failure_required = 0
    failure_optional = 0
    in_progress_required = 0
    in_progress_optional = 0
    pending = 0
    cancelled = 0
    other = 0

    names_success_required: List[str] = []
    names_success_optional: List[str] = []
    names_failure_required: List[str] = []
    names_failure_optional: List[str] = []
    names_in_progress_required: List[str] = []
    names_in_progress_optional: List[str] = []
    names_pending: List[str] = []
    names_cancelled: List[str] = []
    names_other: List[str] = []

    for cr in check_runs or []:
        try:
            name = str(cr.get("name", "") or "")
            status_raw = str(cr.get("status", "") or "").strip().lower()
            concl_raw = cr.get("conclusion", None)
            conclusion = str(concl_raw or "").strip().lower()
            is_req = bool(cr.get("is_required", False))
        except Exception:
            continue

        # Normalize check-run states into the same buckets we use for `gh pr checks`.
        if conclusion in {"success", "neutral", "skipped"}:
            if is_req:
                success_required += 1
                if name:
                    names_success_required.append(name)
            else:
                success_optional += 1
                if name:
                    names_success_optional.append(name)
        elif conclusion in {"failure", "timed_out", "action_required"}:
            if is_req:
                failure_required += 1
                if name:
                    names_failure_required.append(name)
            else:
                failure_optional += 1
                if name:
                    names_failure_optional.append(name)
        elif conclusion in {"cancelled", "canceled"}:
            cancelled += 1
            if name:
                names_cancelled.append(name)
        elif status_raw in {"in_progress"}:
            if is_req:
                in_progress_required += 1
                if name:
                    names_in_progress_required.append(name)
            else:
                in_progress_optional += 1
                if name:
                    names_in_progress_optional.append(name)
        elif status_raw in {"queued", "pending"}:
            pending += 1
            if name:
                names_pending.append(name)
        else:
            other += 1
            if name:
                names_other.append(name)

    total = (
        success_required
        + success_optional
        + failure_required
        + failure_optional
        + in_progress_required
        + in_progress_optional
        + pending
        + cancelled
        + other
    )
    return GitHubChecksSummary(
        counts=GitHubChecksCounts(
            success_required=success_required,
            success_optional=success_optional,
            failure_required=failure_required,
            failure_optional=failure_optional,
            in_progress_required=in_progress_required,
            in_progress_optional=in_progress_optional,
            pending=pending,
            cancelled=cancelled,
            other=other,
            total=total,
        ),
        names=GitHubChecksNames(
            success_required=tuple(names_success_required),
            success_optional=tuple(names_success_optional),
            failure_required=tuple(names_failure_required),
            failure_optional=tuple(names_failure_optional),
            in_progress_required=tuple(names_in_progress_required),
            in_progress_optional=tuple(names_in_progress_optional),
            pending=tuple(names_pending),
            cancelled=tuple(names_cancelled),
            other=tuple(names_other),
        ),
    )


class GitHubAPIClient:
    """GitHub API client with automatic token detection and rate limit handling.

    Features:
    - Automatic token detection (--token arg > GH_TOKEN env > GITHUB_TOKEN env > GitHub CLI config)
    - Request/response handling with proper error messages
    - Support for parallel API calls with ThreadPoolExecutor

    Example:
        client = GitHubAPIClient()
        pr_data = client.get("/repos/owner/repo/pulls/123")
    """

    @staticmethod
    def get_github_token_from_cli() -> Optional[str]:
        """Get GitHub token from GitHub CLI configuration.

        Reads the token from ~/.config/gh/hosts.yml if available.

        Returns:
            GitHub token string, or None if not found
        """
        if not HAS_YAML:
            return None
        assert yaml is not None

        try:
            gh_config_path = Path.home() / '.config' / 'gh' / 'hosts.yml'
            if gh_config_path.exists():
                with open(gh_config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    if config and 'github.com' in config:
                        github_config = config['github.com']
                        if 'oauth_token' in github_config:
                            return github_config['oauth_token']
                        if 'users' in github_config:
                            for user, user_config in github_config['users'].items():
                                if 'oauth_token' in user_config:
                                    return user_config['oauth_token']
        except Exception:
            pass
        return None

    def __init__(self, token: Optional[str] = None, *, debug_rest: bool = False):
        """Initialize GitHub API client.

        Args:
            token: GitHub personal access token. If not provided, will try:
                   1. GH_TOKEN environment variable
                   2. GITHUB_TOKEN environment variable
                   3. GitHub CLI config (~/.config/gh/hosts.yml)
        """
        if not HAS_REQUESTS:
            raise ImportError("requests package required for GitHub API client. Install with: pip install requests")
        assert requests is not None

        # Token priority:
        # 1) explicit arg
        # 2) env var (prefer GH_TOKEN for compatibility with `gh`, then GITHUB_TOKEN)
        # 3) GitHub CLI config (~/.config/gh/hosts.yml)
        self.token = token or os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN') or self.get_github_token_from_cli()
        self.base_url = "https://api.github.com"
        self.headers = {'Accept': 'application/vnd.github.v3+json'}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._debug_rest = bool(debug_rest)

        if self.token:
            self.headers['Authorization'] = f'token {self.token}'

        # Per-run REST call accounting (helps debug "why so many API calls?").
        # Counts only requests made via this client (not `gh` CLI subprocess calls).
        self._rest_calls_total: int = 0
        self._rest_calls_by_label: Dict[str, int] = {}
        # Per-run REST error accounting (helps debug auth/policy issues like 403).
        self._rest_errors_total: int = 0
        self._rest_errors_by_status: Dict[int, int] = {}
        self._rest_last_error: Dict[str, Any] = {}

        # ----------------------------
        # Persistent caches (disk + memory)
        # ----------------------------
        # Cache for job logs (two-tier: memory + disk)
        # - memory: { "53317461976": "Failed unit tests: ..." }
        # - disk:   ~/.cache/dynamo-utils/job-logs/job_logs_cache.json:
        #   {
        #     "53317461976": "Failed unit tests:\n ...",
        #     "53317461234": null
        #   }
        self._job_log_cache: Dict[str, Optional[str]] = {}
        self._cache_dir = dynamo_utils_cache_dir() / "job-logs"

        # Cache for gh pr checks rows (two-tier: memory + disk, short TTL because status changes).
        self._pr_checks_mem_cache: Dict[str, Dict[str, Any]] = {}
        self._pr_checks_cache_dir = dynamo_utils_cache_dir() / "pr-checks"

        # Cache for repo-wide pull request listing (short TTL; used to avoid per-branch API calls).
        self._pulls_list_mem_cache: Dict[str, Dict[str, Any]] = {}
        self._pulls_list_cache_dir = dynamo_utils_cache_dir() / "pulls"

        # Cache for "closed/merged PRs per branch" lookups (long TTL; these don't change often).
        self._pr_branch_mem_cache: Dict[str, Dict[str, Any]] = {}
        self._pr_branch_cache_dir = dynamo_utils_cache_dir() / "pr-branches"

        # Cache for resolved Actions job raw-log redirect URLs (short TTL; URLs are time-limited).
        self._raw_log_url_mem_cache: Dict[str, Dict[str, Any]] = {}
        self._raw_log_url_cache_dir = dynamo_utils_cache_dir() / "raw-log-urls"

        # Cache for downloaded raw log text (two-tier: memory + disk).
        self._raw_log_text_mem_cache: Dict[str, Dict[str, Any]] = {}
        self._raw_log_text_cache_dir = dynamo_utils_cache_dir() / "raw-log-text"
        self._raw_log_text_index_file = self._raw_log_text_cache_dir / "index.json"

        # Cache for Actions job status lookups (short TTL; prevents caching partial logs).
        self._actions_job_status_mem_cache: Dict[str, Dict[str, Any]] = {}

    def _rest_label_for_url(self, url: str) -> str:
        """Coarse label for a REST request URL (keeps job ids / SHAs from exploding cardinality)."""
        try:
            u = str(url or "")
            if "/rate_limit" in u:
                return "rate_limit"
            # Actions jobs
            if re.search(r"/repos/[^/]+/[^/]+/actions/jobs/\\d+/logs\\b", u):
                return "actions_job_logs_zip"
            if re.search(r"/repos/[^/]+/[^/]+/actions/jobs/\\d+\\b", u):
                return "actions_job_status"
            # Check-runs
            if "/check-runs" in u:
                return "check_runs"
            # PR / pulls
            if re.search(r"/repos/[^/]+/[^/]+/pulls\\b", u):
                return "pulls_list"
            if re.search(r"/repos/[^/]+/[^/]+/pulls/\\d+\\b", u):
                return "pull_request"
            # Default: bucket by first few path segments
            m = re.search(r"https?://api\\.github\\.com(/[^?]*)", u)
            path = m.group(1) if m else u
            parts = [p for p in path.split("/") if p]
            return "/".join(parts[:4]) if parts else "unknown"
        except Exception:
            return "unknown"

    def _rest_get(self, url: str, *, timeout: int = 10, allow_redirects: bool = True, stream: bool = False, params: Optional[Dict[str, Any]] = None):
        """requests.get wrapper that increments per-run counters.

        Note:
            We sometimes run on machines where a stale/non-compliant GitHub token is present in the
            environment (classic PAT > 366 day lifetime, fine-grained token not granted repo access, etc).
            In those cases, GitHub returns 403 for requests that would otherwise succeed anonymously
            (especially for public repos). To make dashboards resilient, we do a best-effort retry of
            *GET* requests without Authorization on a narrow class of known token-related 403s.
        """
        assert requests is not None
        label = self._rest_label_for_url(url)
        try:
            self._rest_calls_total += 1
            self._rest_calls_by_label[label] = int(self._rest_calls_by_label.get(label, 0) or 0) + 1
        except Exception:
            pass
        if self._debug_rest:
            try:
                self.logger.debug("GH REST GET [%s] %s", label, url)
            except Exception:
                pass
        headers = dict(self.headers or {})
        resp = requests.get(url, headers=headers, params=params, timeout=timeout, allow_redirects=allow_redirects, stream=stream)

        # Best-effort retry without Authorization for a narrow class of token-related 403s.
        try:
            if int(getattr(resp, "status_code", 0) or 0) == 403 and ("Authorization" in headers):
                body_txt = ""
                try:
                    body_txt = (resp.text or "")
                except Exception:
                    body_txt = ""
                body_lc = body_txt.lower()
                token_related = (
                    ("enterprise forbids access via" in body_lc)
                    or ("forbids access via a personal access tokens" in body_lc)
                    or ("forbids access via a fine-grained personal access tokens" in body_lc)
                    or ("resource not accessible by personal access token" in body_lc)
                )
                if token_related:
                    headers_no_auth = dict(headers)
                    headers_no_auth.pop("Authorization", None)
                    # Count this retry as a REST call too (it is).
                    try:
                        self._rest_calls_total += 1
                        self._rest_calls_by_label[label] = int(self._rest_calls_by_label.get(label, 0) or 0) + 1
                    except Exception:
                        pass
                    if self._debug_rest:
                        try:
                            self.logger.debug("GH REST GET [%s] retrying without Authorization (token-related 403)", label)
                        except Exception:
                            pass
                    resp2 = requests.get(url, headers=headers_no_auth, params=params, timeout=timeout, allow_redirects=allow_redirects, stream=stream)
                    # Prefer the retry if it succeeded (or at least changed the failure mode).
                    if int(getattr(resp2, "status_code", 0) or 0) < 400:
                        resp = resp2
                    else:
                        # If the retry still fails, keep the original response (it likely contains the most useful policy message).
                        pass
        except Exception:
            pass
        try:
            code = int(getattr(resp, "status_code", 0) or 0)
            if code >= 400:
                self._rest_errors_total += 1
                self._rest_errors_by_status[code] = int(self._rest_errors_by_status.get(code, 0) or 0) + 1
                # Keep last error small (HTML stats should not explode).
                body = ""
                try:
                    body = (resp.text or "")[:300]
                except Exception:
                    body = ""
                self._rest_last_error = {
                    "status": code,
                    "url": str(url or ""),
                    "body": body,
                }
        except Exception:
            pass
        if self._debug_rest:
            try:
                rem = resp.headers.get("X-RateLimit-Remaining")
                code = getattr(resp, "status_code", None)
                self.logger.debug("GH REST RESP [%s] status=%s remaining=%s", label, str(code), str(rem))
            except Exception:
                pass
        return resp

    def get_rest_error_stats(self) -> Dict[str, Any]:
        """Return best-effort per-run REST error stats for displaying in dashboards."""
        try:
            return {
                "total": int(self._rest_errors_total),
                "by_status": dict(sorted(self._rest_errors_by_status.items(), key=lambda kv: (-kv[1], kv[0]))),
                "last": dict(self._rest_last_error or {}),
            }
        except Exception:
            return {"total": 0, "by_status": {}, "last": {}}

    def get_rest_call_stats(self) -> Dict[str, Any]:
        """Return per-run REST call stats for debugging."""
        try:
            return {
                "total": int(self._rest_calls_total),
                "by_label": dict(sorted(self._rest_calls_by_label.items(), key=lambda kv: (-kv[1], kv[0]))),
            }
        except Exception:
            return {"total": 0, "by_label": {}}

    def get_actions_job_status(self, *, owner: str, repo: str, job_id: str, ttl_s: int = 120) -> Optional[str]:
        """Return GitHub Actions job status ("completed", "in_progress", ...) with a short memory cache."""
        if not HAS_REQUESTS:
            return None
        assert requests is not None

        try:
            job_id_s = str(job_id or "").strip()
            if not job_id_s:
                return None
        except Exception:
            return None

        now = int(time.time())
        try:
            ent = self._actions_job_status_mem_cache.get(job_id_s)
            if ent and int(ent.get("ts", 0) or 0) + int(ttl_s) > now:
                st = ent.get("status")
                return str(st) if st is not None else None
        except Exception:
            pass

        url = f"{self.base_url}/repos/{owner}/{repo}/actions/jobs/{job_id_s}"
        try:
            resp = self._rest_get(url, timeout=10)
        except Exception:
            return None

        # 404 can happen if job id is invalid / perm issue; just treat as unknown.
        if resp.status_code < 200 or resp.status_code >= 300:
            return None

        try:
            data = resp.json() or {}
            status = data.get("status")
            if status is None:
                return None
            status_s = str(status)
        except Exception:
            return None

        try:
            self._actions_job_status_mem_cache[job_id_s] = {"ts": now, "status": status_s}
        except Exception:
            pass
        return status_s

    @staticmethod
    def _format_seconds_delta(seconds: int) -> str:
        """Format a positive/negative seconds delta as a short human string."""
        s = int(seconds)
        if s <= 0:
            return "0s"
        m, sec = divmod(s, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        if d:
            return f"{d}d {h}h {m}m"
        if h:
            return f"{h}h {m}m"
        if m:
            return f"{m}m {sec}s"
        return f"{sec}s"

    @staticmethod
    def compute_checks_cache_ttl_s(
        last_change_dt: Optional[datetime],
        *,
        refresh: bool = False,
        stable_after_hours: int = DEFAULT_STABLE_AFTER_HOURS,
        short_ttl_s: int = DEFAULT_UNSTABLE_TTL_S,
        stable_ttl_s: int = DEFAULT_STABLE_TTL_S,
    ) -> int:
        """Compute an appropriate cache TTL for CI/checks based on "how recently things changed".

        Heuristic:
        - If the underlying ref hasn't changed in >stable_after_hours, treat checks as stable and
          keep cached results for a long time.
        - If it changed recently, keep a short TTL so in-progress checks update.
        - If refresh=True, force fetch (TTL=0).
        """
        if refresh:
            return 0
        ttl = int(short_ttl_s)
        try:
            if last_change_dt is None:
                return ttl
            dt = last_change_dt
            if getattr(dt, "tzinfo", None) is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age_s = (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds()
            if age_s >= float(stable_after_hours) * 3600.0:
                ttl = max(ttl, int(stable_ttl_s))
        except Exception:
            return int(short_ttl_s)
        return ttl

    def check_core_rate_limit_or_raise(self) -> None:
        """Fail fast if GitHub REST (core) quota is exhausted.

        This makes scripts exit with a clear message instead of spamming API calls.
        """
        assert requests is not None
        url = f"{self.base_url}/rate_limit"
        try:
            resp = self._rest_get(url, timeout=10)
        except Exception as e:
            raise RuntimeError(f"Failed to query GitHub rate limit: {e}") from e

        remaining_hdr = resp.headers.get("X-RateLimit-Remaining")
        reset_hdr = resp.headers.get("X-RateLimit-Reset")
        limit_hdr = resp.headers.get("X-RateLimit-Limit")

        # Prefer headers (works even on 403), fall back to JSON if needed.
        try:
            remaining = int(remaining_hdr) if remaining_hdr is not None else None
        except Exception:
            remaining = None
        try:
            reset_epoch = int(reset_hdr) if reset_hdr is not None else None
        except Exception:
            reset_epoch = None
        try:
            limit = int(limit_hdr) if limit_hdr is not None else None
        except Exception:
            limit = None

        if remaining is None or reset_epoch is None:
            try:
                data = resp.json()
                core = (data or {}).get("resources", {}).get("core", {})
                remaining = int(core.get("remaining")) if remaining is None and core.get("remaining") is not None else remaining
                reset_epoch = int(core.get("reset")) if reset_epoch is None and core.get("reset") is not None else reset_epoch
                limit = int(core.get("limit")) if limit is None and core.get("limit") is not None else limit
            except Exception:
                pass

        if remaining is None or reset_epoch is None:
            # If we can't parse, don't block execution.
            return

        if remaining > 0:
            return

        now = int(time.time())
        seconds_until = reset_epoch - now
        reset_local = datetime.fromtimestamp(reset_epoch).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        raise RuntimeError(
            "GitHub API core (REST) quota exhausted "
            f"(remaining={remaining}"
            + (f", limit={limit}" if limit is not None else "")
            + "). "
            f"Resets at {reset_local} (in {self._format_seconds_delta(seconds_until)})."
        )

    def get_core_rate_limit_info(self) -> Optional[Dict[str, Any]]:
        """Return GitHub REST (core) rate limit info.

        Returns:
            Dict like:
              {
                "remaining": 1234,
                "limit": 5000,
                "reset_epoch": 1766947200,
                "reset_local": "2025-12-28 14:00:00 PST",
                "reset_pt": "2025-12-28 14:00:00 PST",
                "seconds_until_reset": 1234,
              }
            or None if not available.
        """
        if not HAS_REQUESTS:
            return None
        assert requests is not None
        url = f"{self.base_url}/rate_limit"
        try:
            resp = self._rest_get(url, timeout=10)
        except Exception:
            return None

        remaining_hdr = resp.headers.get("X-RateLimit-Remaining")
        reset_hdr = resp.headers.get("X-RateLimit-Reset")
        limit_hdr = resp.headers.get("X-RateLimit-Limit")

        remaining = None
        reset_epoch = None
        limit = None
        try:
            remaining = int(remaining_hdr) if remaining_hdr is not None else None
        except Exception:
            remaining = None
        try:
            reset_epoch = int(reset_hdr) if reset_hdr is not None else None
        except Exception:
            reset_epoch = None
        try:
            limit = int(limit_hdr) if limit_hdr is not None else None
        except Exception:
            limit = None

        if remaining is None or reset_epoch is None:
            try:
                data = resp.json()
                core = (data or {}).get("resources", {}).get("core", {})
                if remaining is None and core.get("remaining") is not None:
                    remaining = int(core.get("remaining"))
                if reset_epoch is None and core.get("reset") is not None:
                    reset_epoch = int(core.get("reset"))
                if limit is None and core.get("limit") is not None:
                    limit = int(core.get("limit"))
            except Exception:
                pass

        if remaining is None or reset_epoch is None:
            return None

        now = int(time.time())
        seconds_until = int(reset_epoch) - now
        reset_local = datetime.fromtimestamp(int(reset_epoch)).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        try:
            reset_pt = (
                datetime.fromtimestamp(int(reset_epoch), tz=timezone.utc)
                .astimezone(ZoneInfo("America/Los_Angeles"))
                .strftime("%Y-%m-%d %H:%M:%S %Z")
            )
        except Exception:
            reset_pt = reset_local

        return {
            "remaining": int(remaining),
            "limit": int(limit) if limit is not None else None,
            "reset_epoch": int(reset_epoch),
            "reset_local": reset_local,
            "reset_pt": reset_pt,
            "seconds_until_reset": seconds_until,
        }

    def _pr_checks_cache_key(self, owner: str, repo: str, pr_number: int) -> str:
        return f"{owner}/{repo}#{int(pr_number)}"

    def _pulls_list_cache_key(self, owner: str, repo: str, state: str) -> str:
        return f"{owner}/{repo}:{state}"

    def _pr_branch_cache_key(self, owner: str, repo: str, branch: str) -> str:
        return f"{owner}/{repo}:{branch}"

    def _load_pr_branch_disk_cache(self) -> Dict[str, Any]:
        cache_file = self._pr_branch_cache_dir / "pr_branch_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    return data if isinstance(data, dict) else {}
            except Exception:
                return {}
        return {}

    def _save_pr_branch_disk_cache(self, data: Dict[str, Any]) -> None:
        try:
            self._pr_branch_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self._pr_branch_cache_dir / "pr_branch_cache.json"
            tmp = str(cache_file) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, cache_file)
        except Exception:
            pass

    @staticmethod
    def _pr_info_min_to_dict(pr: "PRInfo") -> Dict[str, Any]:
        return {
            "number": int(pr.number),
            "title": pr.title,
            "url": pr.url,
            "state": pr.state,
            "is_merged": bool(pr.is_merged),
            "mergeable_state": pr.mergeable_state,
            "head_sha": pr.head_sha,
            "base_ref": pr.base_ref,
            "created_at": pr.created_at,
        }

    @staticmethod
    def _pr_info_min_from_dict(d: Dict[str, Any]) -> Optional["PRInfo"]:
        try:
            return PRInfo(
                number=int(d.get("number") or 0),
                title=str(d.get("title") or ""),
                url=str(d.get("url") or ""),
                state=str(d.get("state") or ""),
                is_merged=bool(d.get("is_merged") or False),
                review_decision=None,
                mergeable_state=str(d.get("mergeable_state") or "unknown"),
                unresolved_conversations=0,
                ci_status=None,
                head_sha=d.get("head_sha"),
                base_ref=d.get("base_ref"),
                created_at=d.get("created_at"),
                has_conflicts=False,
                conflict_message=None,
                blocking_message=None,
                required_checks=[],
                failed_checks=[],
                running_checks=[],
                rerun_url=None,
            )
        except Exception:
            return None

    def _load_pulls_list_disk_cache(self) -> Dict[str, Any]:
        cache_file = self._pulls_list_cache_dir / "pulls_open_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    return data if isinstance(data, dict) else {}
            except Exception:
                return {}
        return {}

    def _save_pulls_list_disk_cache(self, data: Dict[str, Any]) -> None:
        try:
            self._pulls_list_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self._pulls_list_cache_dir / "pulls_open_cache.json"
            tmp = str(cache_file) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, cache_file)
        except Exception:
            # Best-effort only.
            pass

    def list_pull_requests(
        self, owner: str, repo: str, *, state: str = "open", ttl_s: int = DEFAULT_OPEN_PRS_TTL_S
    ) -> List[Dict[str, Any]]:
        """List pull requests for a repo with a short-lived cache.

        This is used to avoid calling /pulls?head=... once per local branch.

        Args:
            owner: Repository owner
            repo: Repository name
            state: "open" or "all" (default: "open")
            ttl_s: Cache TTL in seconds

        Returns:
            List of PR dicts (GitHub REST API /pulls response objects).
        """
        cache_key = self._pulls_list_cache_key(owner, repo, state)
        now = int(time.time())

        # Memory cache
        mem = self._pulls_list_mem_cache.get(cache_key)
        if mem and isinstance(mem, dict):
            ts = int(mem.get("ts") or 0)
            if (now - ts) <= int(ttl_s):
                items = mem.get("items")
                if isinstance(items, list):
                    return items  # type: ignore[return-value]

        # Disk cache
        disk = self._load_pulls_list_disk_cache()
        entry = disk.get(cache_key) if isinstance(disk, dict) else None
        if isinstance(entry, dict):
            ts = int(entry.get("ts") or 0)
            if (now - ts) <= int(ttl_s):
                items = entry.get("items")
                if isinstance(items, list):
                    self._pulls_list_mem_cache[cache_key] = {"ts": ts, "items": items}
                    return items  # type: ignore[return-value]

        # Fetch from API (paginate)
        endpoint = f"/repos/{owner}/{repo}/pulls"
        items: List[Dict[str, Any]] = []
        try:
            page = 1
            while True:
                params = {"state": state, "per_page": 100, "page": page}
                chunk = self.get(endpoint, params=params)
                if not chunk:
                    break
                if isinstance(chunk, list):
                    items.extend(chunk)
                else:
                    break
                if len(chunk) < 100:
                    break
                page += 1
        except Exception as e:
            # Best-effort: if we can't list PRs (rate limit, network, auth), just return empty.
            # Also negative-cache the failure for a short window to avoid spamming retries when
            # scanning multiple clones of the same repo in one run.
            print(f"Error listing PRs for {owner}/{repo}: {e}", file=sys.stderr)
            self._pulls_list_mem_cache[cache_key] = {"ts": now, "items": []}
            if isinstance(disk, dict):
                disk[cache_key] = {"ts": now, "items": []}
                self._save_pulls_list_disk_cache(disk)
            return []

        # Save caches
        self._pulls_list_mem_cache[cache_key] = {"ts": now, "items": items}
        if isinstance(disk, dict):
            disk[cache_key] = {"ts": now, "items": items}
            self._save_pulls_list_disk_cache(disk)
        return items

    def get_pr_info_for_branches(
        self,
        owner: str,
        repo: str,
        branches: Iterable[str],
        *,
        include_closed: bool = True,
        refresh_closed: bool = False,
        closed_ttl_s: int = DEFAULT_CLOSED_PRS_TTL_S,
        no_pr_ttl_s: int = DEFAULT_NO_PR_TTL_S,
    ) -> Dict[str, List["PRInfo"]]:
        """Get PR information for many branches efficiently.

        Strategy:
        - list open PRs once for the repo (cached)
        - match PRs to local branches by head.ref
        - only for matched OPEN PRs, fetch required checks / unresolved conv / checks summary
        - for branches without an open PR, optionally resolve closed/merged PRs with a long-lived cache
        """
        branch_set = {b for b in branches if b}
        if not branch_set:
            return {}

        # Build head.ref/head.label -> list[pr_data]
        #
        # IMPORTANT: local branch names may be stored in different formats:
        # - Typical: "<branch_ref>" (matches GitHub PR head.ref directly)
        # - Fork-tracking convention: "<fork_owner>/<branch_ref>"
        #   In that case, GitHub PR head.label is "<fork_owner>:<branch_ref>" and head.ref is "<branch_ref>".
        #   We match both so fork branches still show their PRs.
        pr_datas = self.list_pull_requests(owner, repo, state="open")
        label_to_branch: Dict[str, str] = {}
        for b in branch_set:
            try:
                if "/" in b:
                    pre, rest = b.split("/", 1)
                    pre = (pre or "").strip()
                    rest = (rest or "").strip()
                    if pre and rest:
                        label_to_branch[f"{pre}:{rest}"] = b
            except Exception:
                continue
        head_to_prs: Dict[str, List[Dict[str, Any]]] = {}
        for pr_data in pr_datas:
            try:
                head = (pr_data.get("head") or {})
                head_ref = head.get("ref")
                head_label = head.get("label")
            except Exception:
                head_ref = None
                head_label = None

            branch_name: Optional[str] = None
            if head_ref and head_ref in branch_set:
                branch_name = str(head_ref)
            elif head_label and str(head_label) in label_to_branch:
                branch_name = label_to_branch[str(head_label)]

            if not branch_name:
                continue
            head_to_prs.setdefault(branch_name, []).append(pr_data)

        result: Dict[str, List[PRInfo]] = {b: [] for b in branch_set}

        # Per-PR enrichment can be parallelized (bounded).
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures: List[Tuple[str, Dict[str, Any], Any]] = []
            for branch_name, prs in head_to_prs.items():
                for pr_data in prs:
                    futures.append((branch_name, pr_data, executor.submit(self._pr_info_from_pr_data, owner, repo, pr_data)))

            for branch_name, _pr_data, fut in futures:
                try:
                    pr_info = fut.result()
                    if pr_info is not None:
                        result.setdefault(branch_name, []).append(pr_info)
                except Exception:
                    # Best-effort: skip broken PRs.
                    continue

        if not include_closed:
            return result

        # Resolve closed/merged PRs for branches with no OPEN PR match.
        now = int(time.time())
        disk = self._load_pr_branch_disk_cache()

        missing: List[str] = []
        for b in branch_set:
            if result.get(b):
                # Branch already has at least one open PR; we don't additionally fetch historical PRs by default.
                continue

            key = self._pr_branch_cache_key(owner, repo, b)

            if not refresh_closed:
                # Memory cache
                mem = self._pr_branch_mem_cache.get(key)
                if isinstance(mem, dict):
                    ts = int(mem.get("ts") or 0)
                    prs_d = mem.get("prs")
                    if isinstance(prs_d, list):
                        ttl = int(no_pr_ttl_s) if len(prs_d) == 0 else int(closed_ttl_s)
                        if (now - ts) <= ttl:
                            prs: List[PRInfo] = []
                            for d in prs_d:
                                if isinstance(d, dict):
                                    pr = self._pr_info_min_from_dict(d)
                                    if pr is not None:
                                        prs.append(pr)
                            result[b] = prs
                            continue

                # Disk cache
                entry = disk.get(key) if isinstance(disk, dict) else None
                if isinstance(entry, dict):
                    ts = int(entry.get("ts") or 0)
                    prs_d = entry.get("prs")
                    if isinstance(prs_d, list):
                        ttl = int(no_pr_ttl_s) if len(prs_d) == 0 else int(closed_ttl_s)
                        if (now - ts) <= ttl:
                            prs = []
                            for d in prs_d:
                                if isinstance(d, dict):
                                    pr = self._pr_info_min_from_dict(d)
                                    if pr is not None:
                                        prs.append(pr)
                            self._pr_branch_mem_cache[key] = {"ts": ts, "prs": prs_d}
                            result[b] = prs
                            continue

            missing.append(b)

        if not missing:
            return result

        def fetch_branch(branch_name: str) -> Tuple[str, List[Dict[str, Any]]]:
            endpoint = f"/repos/{owner}/{repo}/pulls"
            # Try the normal "same-repo" head first, then fall back to fork-style "owner/branch" naming.
            prs_data = self.get(endpoint, params={"head": f"{owner}:{branch_name}", "state": "all", "per_page": 30})
            if (not prs_data) and "/" in (branch_name or ""):
                try:
                    pre, rest = branch_name.split("/", 1)
                    pre = (pre or "").strip()
                    rest = (rest or "").strip()
                    if pre and rest:
                        prs_data = self.get(endpoint, params={"head": f"{pre}:{rest}", "state": "all", "per_page": 30})
                except Exception:
                    pass
            out_prs: List[Dict[str, Any]] = []
            if isinstance(prs_data, list):
                for pr_data in prs_data:
                    if not isinstance(pr_data, dict):
                        continue
                    st = str(pr_data.get("state") or "").lower()
                    is_merged = pr_data.get("merged_at") is not None
                    if st != "open" or is_merged:
                        out_prs.append(
                            {
                                "number": pr_data.get("number"),
                                "title": pr_data.get("title") or "",
                                "url": pr_data.get("html_url") or "",
                                "state": pr_data.get("state") or "",
                                "is_merged": bool(is_merged),
                                "mergeable_state": pr_data.get("mergeable_state") or "unknown",
                                "head_sha": (pr_data.get("head") or {}).get("sha"),
                                "base_ref": (pr_data.get("base") or {}).get("ref", "main"),
                                "created_at": pr_data.get("created_at"),
                            }
                        )
            out_prs.sort(key=lambda d: int(d.get("number") or 0), reverse=True)
            return branch_name, out_prs

        with ThreadPoolExecutor(max_workers=6) as executor:
            futs = [executor.submit(fetch_branch, b) for b in missing]
            for fut in as_completed(futs):
                try:
                    branch_name, prs_d = fut.result()
                except Exception:
                    continue
                key = self._pr_branch_cache_key(owner, repo, branch_name)
                self._pr_branch_mem_cache[key] = {"ts": now, "prs": prs_d}
                if isinstance(disk, dict):
                    disk[key] = {"ts": now, "prs": prs_d}
                prs: List[PRInfo] = []
                for d in prs_d:
                    if isinstance(d, dict):
                        pr = self._pr_info_min_from_dict(d)
                        if pr is not None:
                            prs.append(pr)
                result[branch_name] = prs

        if isinstance(disk, dict):
            self._save_pr_branch_disk_cache(disk)

        return result

    def _pr_info_from_pr_data(self, owner: str, repo: str, pr_data: Dict[str, Any]) -> Optional["PRInfo"]:
        """Convert a /pulls list response object into a PRInfo (with extra lookups)."""
        try:
            pr_number = int(pr_data["number"])
        except Exception:
            return None

        # Fetch PR checks data once (consolidates 2 of the 3 'gh pr checks' calls)
        checks_data = self._fetch_pr_checks_data(owner, repo, pr_number)

        # Required checks + unresolved conversations are API/gh lookups; do them sequentially here
        # (we parallelize at a higher level across PRs).
        required_checks = self.get_required_checks(owner, repo, pr_number)
        unresolved_count = self.count_unresolved_conversations(owner, repo, pr_number)

        # Now process checks data synchronously (no need to parallelize since data is already fetched)
        head_sha = (pr_data.get("head") or {}).get("sha")
        ci_status = self.get_ci_status(owner, repo, head_sha, pr_number, checks_data=checks_data)
        failed_checks, rerun_url = self.get_failed_checks(owner, repo, head_sha, required_checks, pr_number, checks_data=checks_data)
        running_checks = self.get_running_checks(pr_number, owner, repo, required_checks, checks_data=checks_data)

        mergeable = None  # list endpoint doesn't include this, but we can use mergeable_state when present
        mergeable_state = pr_data.get("mergeable_state", "unknown")
        has_conflicts = (mergeable is False) or (mergeable_state == "dirty")

        base_branch = (pr_data.get("base") or {}).get("ref", "main")

        conflict_message = None
        if has_conflicts:
            conflict_message = f"This branch has conflicts that must be resolved (merge {base_branch} into this branch)"

        blocking_message = None
        if mergeable_state in ["blocked", "unstable", "behind"]:
            if mergeable_state == "unstable":
                blocking_message = "Merging is blocked - Waiting on code owner review or required status checks"
            elif mergeable_state == "blocked":
                blocking_message = "Merging is blocked - Required reviews or checks not satisfied"
            elif mergeable_state == "behind":
                blocking_message = "This branch is out of date with the base branch"

        return PRInfo(
            number=pr_number,
            title=pr_data.get("title", ""),
            url=pr_data.get("html_url", ""),
            state=pr_data.get("state", ""),
            is_merged=pr_data.get("merged_at") is not None,
            review_decision=pr_data.get("reviewDecision"),
            mergeable_state=mergeable_state,
            unresolved_conversations=unresolved_count,
            ci_status=ci_status,
            head_sha=head_sha,
            base_ref=base_branch,
            created_at=pr_data.get("created_at"),
            has_conflicts=has_conflicts,
            conflict_message=conflict_message,
            blocking_message=blocking_message,
            required_checks=sorted(list(required_checks or set())),
            failed_checks=failed_checks,
            running_checks=running_checks,
            rerun_url=rerun_url,
        )

    def _load_pr_checks_disk_cache(self) -> Dict[str, Any]:
        cache_file = self._pr_checks_cache_dir / "pr_checks_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_pr_checks_disk_cache(self, cache: Dict[str, Any]) -> None:
        try:
            self._pr_checks_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self._pr_checks_cache_dir / "pr_checks_cache.json"
            with open(cache_file, "w") as f:
                json.dump(cache, f, indent=2)
        except Exception:
            pass

    def get_pr_checks_rows(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        *,
        required_checks: Optional[set] = None,
        ttl_s: int = 300,
        skip_fetch: bool = False,
    ) -> List[GHPRCheckRow]:
        """Get structured rows from `gh pr checks`, with a short-lived persistent cache.

        Note: the raw-log links are time-limited, but the checks rows themselves are cheap to cache
        for a short TTL to speed repeated HTML generation.
        """
        required_checks = required_checks or set()
        key = self._pr_checks_cache_key(owner, repo, pr_number)
        now = int(datetime.now(timezone.utc).timestamp())

        # 1) memory cache
        try:
            ent = self._pr_checks_mem_cache.get(key)
            if ent and isinstance(ent, dict):
                ts = int(ent.get("ts", 0) or 0)
                if ts and (now - ts) <= max(0, int(ttl_s)):
                    rows = ent.get("rows") or []
                    out: List[GHPRCheckRow] = []
                    for r in rows:
                        try:
                            name = str(r.get("name", "") or "")
                            out.append(
                                GHPRCheckRow(
                                    name=name,
                                    status_raw=str(r.get("status_raw", "") or ""),
                                    duration=str(r.get("duration", "") or ""),
                                    url=str(r.get("url", "") or ""),
                                    description=str(r.get("description", "") or ""),
                                    is_required=(name in required_checks) or bool(r.get("is_required", False)),
                                )
                            )
                        except Exception:
                            continue
                    return out
        except Exception:
            pass

        # 2) disk cache
        try:
            disk = self._load_pr_checks_disk_cache()
            ent = disk.get(key) if isinstance(disk, dict) else None
            if ent and isinstance(ent, dict):
                ts = int(ent.get("ts", 0) or 0)
                if ts and (now - ts) <= max(0, int(ttl_s)):
                    rows = ent.get("rows") or []
                    out: List[GHPRCheckRow] = []
                    for r in rows:
                        try:
                            name = str(r.get("name", "") or "")
                            out.append(
                                GHPRCheckRow(
                                    name=name,
                                    status_raw=str(r.get("status_raw", "") or ""),
                                    duration=str(r.get("duration", "") or ""),
                                    url=str(r.get("url", "") or ""),
                                    description=str(r.get("description", "") or ""),
                                    is_required=(name in required_checks) or bool(r.get("is_required", False)),
                                )
                            )
                        except Exception:
                            continue
                    # promote to memory
                    self._pr_checks_mem_cache[key] = {"ts": ts, "rows": rows}
                    return out
        except Exception:
            pass

        if skip_fetch:
            return []

        # 3) fetch via gh
        #
        # NOTE:
        # `gh pr checks` is convenient but can fail in environments where GitHub auth is poisoned
        # (enterprise token lifetime policy, fine-grained token not granted repo access, etc).
        # When it fails/returns empty, fall back to GitHub REST check-runs for the PR head SHA.
        try:
            result = subprocess.run(
                ["gh", "pr", "checks", str(int(pr_number)), "--repo", f"{owner}/{repo}"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            stdout = (result.stdout or "").strip()
            if stdout:
                rows_dicts: List[Dict[str, Any]] = []
                out: List[GHPRCheckRow] = []
                for line in stdout.split("\n"):
                    if not line.strip():
                        continue
                    parts = line.split("\t")
                    if len(parts) < 2:
                        continue
                    name = (parts[0] or "").strip()
                    status_raw = (parts[1] or "").strip()
                    duration = (parts[2] or "").strip() if len(parts) > 2 else ""
                    url = (parts[3] or "").strip() if len(parts) > 3 else ""
                    description = (parts[4] or "").strip() if len(parts) > 4 else ""
                    is_req = name in required_checks
                    out.append(
                        GHPRCheckRow(
                            name=name,
                            status_raw=status_raw,
                            duration=duration,
                            url=url,
                            description=description,
                            is_required=is_req,
                        )
                    )
                    rows_dicts.append(
                        {
                            "name": name,
                            "status_raw": status_raw,
                            "duration": duration,
                            "url": url,
                            "description": description,
                            "is_required": is_req,
                        }
                    )

                # persist caches
                try:
                    self._pr_checks_mem_cache[key] = {"ts": now, "rows": rows_dicts}
                    disk = self._load_pr_checks_disk_cache()
                    if not isinstance(disk, dict):
                        disk = {}
                    disk[key] = {"ts": now, "rows": rows_dicts}
                    self._save_pr_checks_disk_cache(disk)
                except Exception:
                    pass

                return out

        except Exception:
            # Fall through to REST fallback below.
            pass

        # 4) fallback: REST check-runs for PR head SHA (best-effort; works for public repos without auth)
        try:
            pr = self.get(f"/repos/{owner}/{repo}/pulls/{int(pr_number)}", timeout=10) or {}
            head_sha = (((pr.get("head") or {}) if isinstance(pr, dict) else {}) or {}).get("sha")
            head_sha = str(head_sha or "").strip()
            if not head_sha:
                return []

            data = self.get(f"/repos/{owner}/{repo}/commits/{head_sha}/check-runs", params={"per_page": 100}, timeout=10) or {}
            check_runs = data.get("check_runs") if isinstance(data, dict) else None
            if not isinstance(check_runs, list) or not check_runs:
                return []

            def _parse_iso(s: str) -> Optional[datetime]:
                try:
                    ss = str(s or "").strip()
                    if not ss:
                        return None
                    # GitHub timestamps look like "2025-12-29T10:52:07Z"
                    if ss.endswith("Z"):
                        ss = ss[:-1] + "+00:00"
                    return datetime.fromisoformat(ss)
                except Exception:
                    return None

            def _format_dur(start_iso: Optional[str], end_iso: Optional[str]) -> str:
                try:
                    st = _parse_iso(start_iso or "")
                    en = _parse_iso(end_iso or "")
                    if not st or not en:
                        return ""
                    sec = int((en - st).total_seconds())
                    if sec < 0:
                        return ""
                    m, s2 = divmod(sec, 60)
                    h, m = divmod(m, 60)
                    if h:
                        return f"{h}h {m}m"
                    if m:
                        return f"{m}m {s2}s"
                    return f"{s2}s"
                except Exception:
                    return ""

            rows_dicts = []
            out = []
            for cr in check_runs:
                if not isinstance(cr, dict):
                    continue
                name = str(cr.get("name", "") or "").strip()
                status = str(cr.get("status", "") or "").strip().lower()
                conclusion = str(cr.get("conclusion", "") or "").strip().lower()

                # Map REST check-run (status+conclusion) into a gh-like status_raw.
                status_raw = ""
                if status and status != "completed":
                    # in_progress / queued
                    status_raw = status
                else:
                    # completed; use conclusion
                    if conclusion in {"success"}:
                        status_raw = "pass"
                    elif conclusion in {"failure"}:
                        status_raw = "fail"
                    elif conclusion in {"cancelled", "canceled"}:
                        status_raw = "cancelled"
                    elif conclusion in {"skipped"}:
                        status_raw = "skipped"
                    elif conclusion in {"neutral"}:
                        status_raw = "neutral"
                    elif conclusion in {"timed_out"}:
                        status_raw = "timed_out"
                    elif conclusion in {"action_required"}:
                        status_raw = "action_required"
                    else:
                        status_raw = "unknown"

                duration = _format_dur(cr.get("started_at"), cr.get("completed_at"))
                url = str(cr.get("html_url", "") or "").strip()
                description = ""
                try:
                    out_obj = cr.get("output") or {}
                    if isinstance(out_obj, dict):
                        description = str(out_obj.get("title", "") or "").strip()
                except Exception:
                    description = ""

                is_req = bool(name and (name in required_checks))
                row = GHPRCheckRow(
                    name=name,
                    status_raw=status_raw,
                    duration=duration,
                    url=url,
                    description=description,
                    is_required=is_req,
                )
                out.append(row)
                rows_dicts.append(
                    {
                        "name": name,
                        "status_raw": status_raw,
                        "duration": duration,
                        "url": url,
                        "description": description,
                        "is_required": is_req,
                    }
                )

            if not out:
                return []

            # persist caches (same shape as gh output cache)
            try:
                self._pr_checks_mem_cache[key] = {"ts": now, "rows": rows_dicts}
                disk = self._load_pr_checks_disk_cache()
                if not isinstance(disk, dict):
                    disk = {}
                disk[key] = {"ts": now, "rows": rows_dicts}
                self._save_pr_checks_disk_cache(disk)
            except Exception:
                pass

            return out
        except Exception:
            return []

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, timeout: int = 10) -> Optional[Dict]:
        """Make GET request to GitHub API.

        Args:
            endpoint: API endpoint (e.g., "/repos/owner/repo/pulls/123")
            params: Query parameters
            timeout: Request timeout in seconds

        Returns:
            JSON response as dict, or None if request failed

            Example return value for pull request endpoint:
            {
                "number": 1234,
                "title": "Add Docker image caching improvements",
                "state": "open",
                "head": {
                    "sha": "21a03b316dc1e5031183965e5798b0d9fe2e64b3",
                    "ref": "feature/docker-caching"
                },
                "base": {"ref": "main"},
                "mergeable": true,
                "mergeable_state": "clean",
                "user": {"login": "johndoe"},
                "created_at": "2025-11-20T10:00:00Z"
            }

            Example return value for check runs endpoint:
            {
                "total_count": 5,
                "check_runs": [
                    {
                        "id": 12345678,
                        "name": "build-test-amd64",
                        "status": "completed",
                        "conclusion": "success",
                        "html_url": "https://github.com/owner/repo/actions/runs/12345678"
                    }
                ]
            }
        """
        url = f"{self.base_url}{endpoint}" if endpoint.startswith('/') else f"{self.base_url}/{endpoint}"

        try:
            assert requests is not None
            response = self._rest_get(url, timeout=timeout, params=params)

            if response.status_code == 403:
                # Check if it's a rate limit error
                if 'X-RateLimit-Remaining' in response.headers and response.headers['X-RateLimit-Remaining'] == '0':
                    raise Exception(
                        "GitHub API rate limit exceeded. Use --token argument or set GH_TOKEN/GITHUB_TOKEN environment variable."
                    )
                else:
                    raise Exception(f"GitHub API returned 403 Forbidden: {response.text}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:  # type: ignore[union-attr]
            raise Exception(f"GitHub API request failed for {endpoint}: {e}")

    def has_token(self) -> bool:
        """Check if a GitHub token is configured."""
        return self.token is not None

    def _fetch_pr_checks_data(self, owner: str, repo: str, pr_number: int) -> Optional[dict]:
        """Fetch PR checks data once and parse for reuse by multiple methods.

        This method consolidates the subprocess call to 'gh pr checks' to avoid
        calling it multiple times for the same PR.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number

        Returns:
            Parsed dict (example shape):
                {
                  "stdout": "build-test\tpass\t35m\thttps://...\n...",
                  "checks": [
                    {"name": "build-test", "status": "pass", "duration": "35m", "url": "https://..."},
                    {"name": "backend-status-check", "status": "fail", "duration": "2s", "url": "https://..."}
                  ]
                }
            Returns None if the subprocess call fails.
        """
        try:
            result = subprocess.run(
                ['gh', 'pr', 'checks', str(pr_number), '--repo', f'{owner}/{repo}'],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Parse output even if return code is non-zero (failed checks)
            if not result.stdout:
                return None

            # Parse tab-separated output: check_name\tstatus\tduration\turl
            checks = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue

                parts = line.split('\t')
                if len(parts) >= 2:
                    checks.append({
                        'name': parts[0].strip(),
                        'status': parts[1].strip(),
                        'duration': parts[2].strip() if len(parts) > 2 else '',
                        'url': parts[3].strip() if len(parts) > 3 else ''
                    })

            return {
                'stdout': result.stdout,
                'checks': checks
            }

        except Exception:
            return None

    def get_ci_status(self, owner: str, repo: str, sha: str, pr_number: Optional[int] = None,
                     checks_data: Optional[dict] = None) -> Optional[str]:
        """Get CI status for a commit by checking PR checks.

        Args:
            owner: Repository owner
            repo: Repository name
            sha: Commit SHA
            pr_number: Optional PR number for faster lookup
            checks_data: Optional pre-fetched checks data from _fetch_pr_checks_data()

        Returns:
            CI status string ('passed', 'failed', 'running'), or None if unavailable
        """
        # If checks_data is provided, use it instead of fetching
        if checks_data:
            checks = checks_data.get('checks', [])
            if not checks:
                return None

            # Count check statuses
            has_fail = any(c['status'].lower() == 'fail' for c in checks)
            has_pending = any(c['status'].lower() in ('pending', 'queued', 'in_progress') for c in checks)

            if has_fail:
                return 'failed'
            elif has_pending:
                return 'running'
            else:
                return 'passed'

        # If we don't have PR number, try to find it
        if not pr_number:
            # Try to find PR from commit
            endpoint = f"/repos/{owner}/{repo}/commits/{sha}/pulls"
            try:
                pulls = self.get(endpoint)
                if pulls and len(pulls) > 0:
                    pr_number = pulls[0]['number']
                else:
                    # Fallback to legacy status API
                    endpoint = f"/repos/{owner}/{repo}/commits/{sha}/status"
                    data = self.get(endpoint)
                    if data:
                        state = data.get('state')
                        if state == 'success':
                            return 'passed'
                        elif state == 'failure':
                            return 'failed'
                        elif state == 'pending':
                            return 'running'
                    return None
            except Exception:
                return None

        # Use gh CLI to get check status (most reliable)
        try:
            result = subprocess.run(
                ['gh', 'pr', 'checks', str(pr_number), '--repo', f'{owner}/{repo}'],
                capture_output=True,
                text=True,
                timeout=10
            )

            # gh pr checks returns non-zero if there are failures or pending checks
            # We still want to parse the output even if return code is non-zero
            if not result.stdout:
                return None

            # Parse output - count pass/fail/pending
            lines = result.stdout.lower().split('\n')
            has_fail = any('fail' in line for line in lines)
            has_pending = any('pending' in line for line in lines)

            if has_fail:
                return 'failed'
            elif has_pending:
                return 'running'
            else:
                return 'passed'

        except Exception:
            return None

    def get_pr_details(self, owner: str, repo: str, pr_number: int) -> Optional[dict]:
        """Get full PR details including mergeable status.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number

        Returns:
            PR details as dict, or None if request failed

            Example return value:
            {
                "number": 1234,
                "title": "Add Docker image caching improvements",
                "state": "open",
                "head": {"sha": "21a03b316dc1e5031183965e5798b0d9fe2e64b3"},
                "base": {"ref": "main"},
                "mergeable": true,
                "mergeable_state": "clean",
                "user": {"login": "johndoe"},
                "created_at": "2025-11-20T10:00:00Z",
                "updated_at": "2025-11-20T17:05:58Z"
            }
        """
        endpoint = f"/repos/{owner}/{repo}/pulls/{pr_number}"
        try:
            return self.get(endpoint)
        except Exception:
            return None

    def get_commit_check_status(self, owner: str, repo: str, sha: str) -> dict:
        """Get aggregated CI status from GitHub Actions check-runs for a commit.

        Prioritizes test checks over build checks:
        - If test checks (deploy-test-*) are in_progress/queued → "building" (yellow)
        - If only builds are in_progress but tests failed → "failed" (red)
        - If any failures and nothing in progress → "failed" (red)
        - If all completed successfully → "success" (green)
        - If no checks → "unknown" (gray)

        Args:
            owner: Repository owner
            repo: Repository name
            sha: Commit SHA (can be short or full)

        Returns:
            Dict with status info:
            {
                "status": "building"|"failed"|"success"|"unknown",
                "in_progress": int,
                "queued": int,
                "failed": int,
                "success": int,
                "total": int
            }
        """
        endpoint = f"/repos/{owner}/{repo}/commits/{sha}/check-runs"
        try:
            result = self.get(endpoint)
            if not result:
                return {"status": "unknown", "in_progress": 0, "queued": 0, "failed": 0, "success": 0, "total": 0}

            # Count by type
            test_in_progress = 0
            test_queued = 0
            build_in_progress = 0
            build_queued = 0
            failed = 0
            success = 0

            for check in result.get('check_runs', []):
                check_name = check['name']
                status = check['status']
                conclusion = check.get('conclusion')

                # Classify as test or build
                is_test = 'deploy-test-' in check_name or 'test' in check_name.lower()

                if status == 'in_progress':
                    if is_test:
                        test_in_progress += 1
                    else:
                        build_in_progress += 1
                elif status == 'queued':
                    if is_test:
                        test_queued += 1
                    else:
                        build_queued += 1
                elif status == 'completed':
                    if conclusion == 'failure':
                        failed += 1
                    elif conclusion == 'success':
                        success += 1

            total = result.get('total_count', 0)
            total_in_progress = test_in_progress + build_in_progress
            total_queued = test_queued + build_queued

            # Determine overall status with priority logic
            if test_in_progress > 0 or test_queued > 0:
                # Test checks still running → yellow
                overall_status = "building"
            elif failed > 0:
                # Tests done with failures → red (even if builds still running)
                overall_status = "failed"
            elif total_in_progress > 0 or total_queued > 0:
                # Only builds running, no test failures → yellow
                overall_status = "building"
            elif total == 0:
                overall_status = "unknown"
            else:
                # All done, no failures → green
                overall_status = "success"

            return {
                "status": overall_status,
                "in_progress": total_in_progress,
                "queued": total_queued,
                "failed": failed,
                "success": success,
                "total": total
            }
        except Exception as e:
            self.logger.warning(f"Failed to get check status for {sha}: {e}")
            return {"status": "unknown", "in_progress": 0, "queued": 0, "failed": 0, "success": 0, "total": 0}

    def count_unresolved_conversations(self, owner: str, repo: str, pr_number: int) -> int:
        """Count unresolved conversation threads in a PR.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number

        Returns:
            Count of unresolved conversations (approximated as top-level comments)
        """
        endpoint = f"/repos/{owner}/{repo}/pulls/{pr_number}/comments"
        try:
            comments = self.get(endpoint)
            if not comments:
                return 0
            # Count top-level comments (those without in_reply_to_id are conversation starters)
            unresolved = sum(1 for comment in comments if not comment.get('in_reply_to_id'))
            return unresolved
        except Exception:
            return 0

    def _load_disk_cache(self) -> Dict[str, Optional[str]]:
        """Load job logs cache from disk."""
        cache_file = self._cache_dir / "job_logs_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_disk_cache(self, cache: Dict[str, Optional[str]]) -> None:
        """Save job logs cache to disk."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self._cache_dir / "job_logs_cache.json"
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
        except Exception:
            pass  # Fail silently if we can't save cache

    def _save_to_disk_cache(self, job_id: str, error_summary: str) -> None:
        """Save a single job log to disk cache."""
        try:
            disk_cache = self._load_disk_cache()
            disk_cache[job_id] = error_summary
            self._save_disk_cache(disk_cache)
        except Exception:
            pass  # Fail silently if we can't save cache

    def get_required_checks(self, owner: str, repo: str, pr_number: int) -> set:
        """Get the list of required check names for a PR using gh CLI.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: PR number

        Returns:
            Set of required check names
        """
        try:
            # Use gh CLI to get required checks (more reliable than API)
            result = subprocess.run(
                ['gh', 'pr', 'checks', str(pr_number), '--repo', f'{owner}/{repo}', '--required'],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Note: gh returns exit code 1 if there are failed checks, but still outputs the data
            # So we check stderr for actual errors, not return code
            if result.stderr and 'error' in result.stderr.lower():
                return set()

            # Parse output to extract check names
            # Format: "check_name\tstatus\tduration\turl"
            required_checks = set()
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    # Split on tabs, get first column (check name)
                    parts = line.split('\t')
                    if parts:
                        check_name = parts[0].strip()
                        if check_name:
                            required_checks.add(check_name)

            return required_checks

        except Exception:
            # If we can't get required checks, return empty set
            return set()

    def get_cached_required_checks(
        self,
        pr_numbers: List[int],
        owner: str = "ai-dynamo",
        repo: str = "dynamo",
        cache_file: str = "github_required_checks.json",
        skip_fetch: bool = False,
        stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[int, List[str]]:
        """Get required check names for a list of PRs, with a persistent cache.

        Uses `gh pr checks --required` which reflects branch protection "required status checks".

        Cache file format (in dynamo-utils cache dir):
            {
              "1234": ["Build and Test - dynamo", "lychee", ...],
              "5678": [],
              ...
            }

        Args:
            pr_numbers: PR numbers to query
            owner: GitHub org/user
            repo: GitHub repo
            cache_file: Cache file name (stored via resolve_cache_path)
            skip_fetch: If True, do not call gh; return cached values only

        Returns:
            Mapping PR number -> list of required check names (may be empty)
        """
        from concurrent.futures import ThreadPoolExecutor

        cache_path = resolve_cache_path(str(cache_file))
        cache: Dict[str, List[str]] = {}
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text()) or {}
            except Exception:
                cache = {}

        # Normalize input & preserve order-ish
        pr_numbers_unique: List[int] = []
        seen = set()
        for n in pr_numbers:
            if n is None:
                continue
            try:
                n_int = int(n)
            except Exception:
                continue
            if n_int not in seen:
                seen.add(n_int)
                pr_numbers_unique.append(n_int)

        result: Dict[int, List[str]] = {}
        prs_to_fetch: List[int] = []

        for pr in pr_numbers_unique:
            key = str(pr)
            if key in cache:
                result[pr] = cache.get(key, []) or []
            else:
                if skip_fetch:
                    result[pr] = []
                else:
                    prs_to_fetch.append(pr)

        cache_updated = False
        if prs_to_fetch:
            def fetch_one(pr: int) -> tuple[int, List[str]]:
                required = self.get_required_checks(owner, repo, pr) or set()
                # Stable, deterministic ordering
                return pr, sorted(required)

            # Small number of workers: each call shells out to `gh`.
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(fetch_one, pr) for pr in prs_to_fetch]
                for future in futures:
                    try:
                        pr, required_list = future.result()
                    except Exception:
                        continue
                    cache[str(pr)] = required_list
                    result[pr] = required_list
                    cache_updated = True

        if cache_updated:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(cache, indent=2))
            except Exception:
                pass

        # Stats (best-effort): why did we call `gh` vs use cache?
        try:
            if isinstance(stats, dict):
                stats["required_checks_cache"] = {
                    "total_prs": len(pr_numbers_unique),
                    "hits": len(pr_numbers_unique) - len(prs_to_fetch),
                    "misses": len(prs_to_fetch),
                    "skip_fetch": bool(skip_fetch),
                    "cache_file": str(cache_path),
                    "miss_reason": "not_in_cache" if prs_to_fetch else "",
                    "miss_prs_sample": [int(x) for x in prs_to_fetch[:10]],
                }
        except Exception:
            pass

        return result

    def _extract_pytest_summary(self, all_lines: list) -> Optional[str]:
        """Extract pytest short test summary from log lines.

        Args:
            all_lines: List of log lines

        Returns:
            Formatted summary with failed test names, or None if not a pytest failure
        """
        try:
            # Find the "short test summary info" section
            summary_start_idx = None
            summary_end_idx = None

            for i, line in enumerate(all_lines):
                if '=== short test summary info ===' in line or '==== short test summary info ====' in line:
                    summary_start_idx = i
                elif summary_start_idx is not None and '===' in line and 'failed' in line.lower() and 'passed' in line.lower():
                    # Found end line like "===== 4 failed, 329 passed, 3 skipped, 284 deselected in 422.68s (0:07:02) ====="
                    summary_end_idx = i
                    break

            if summary_start_idx is None:
                return None

            # Extract failed test lines
            failed_tests = []
            for i in range(summary_start_idx + 1, summary_end_idx if summary_end_idx else len(all_lines)):
                line = all_lines[i]
                # Look for lines that start with "FAILED" after timestamp/prefix
                if 'FAILED' in line:
                    # Extract the test name
                    # Format: "FAILED lib/bindings/python/tests/test_metrics_registry.py::test_counter_introspection"
                    parts = line.split('FAILED')
                    if len(parts) >= 2:
                        test_name = parts[1].strip()
                        # Remove any trailing info like " - AssertionError: ..."
                        if ' - ' in test_name:
                            test_name = test_name.split(' - ')[0].strip()
                        failed_tests.append(test_name)

            if not failed_tests:
                return None

            # Format the output
            result = "Failed unit tests:\n\n"
            for test in failed_tests:
                result += f"  • {test}\n"

            # Add summary line if available
            if summary_end_idx:
                summary_line = all_lines[summary_end_idx]
                # Extract just the summary part (after the timestamp and ====)
                # Format: "2025-11-04T20:25:55.7767823Z ==== 5 failed, 4 passed, 320 deselected, 111 warnings in 1637.48s (0:27:17) ===="
                if '====' in summary_line:
                    # Split by ==== and get the middle part
                    parts = summary_line.split('====')
                    if len(parts) >= 2:
                        summary_text = parts[1].strip()
                        if summary_text:
                            result += f"\nSummary: {summary_text}"

            return result

        except Exception:
            return None

    def get_job_error_summary(self, run_id: str, job_url: str, owner: str, repo: str) -> Optional[str]:
        """Get error summary by fetching job logs via gh CLI (with disk + memory caching).

        Args:
            run_id: Workflow run ID
            job_url: Job URL (contains job ID)
            owner: Repository owner
            repo: Repository name

        Returns:
            Error summary string or None
        """
        job_id = "unknown"
        try:
            # Extract job ID from URL
            # Example: https://github.com/ai-dynamo/dynamo/actions/runs/18697156351/job/53317461976
            if '/job/' not in job_url:
                return None

            job_id = job_url.split('/job/')[1].split('?')[0]

            # Check in-memory cache first (fastest)
            if job_id in self._job_log_cache:
                return self._job_log_cache[job_id]

            # Check disk cache second
            disk_cache = self._load_disk_cache()
            if job_id in disk_cache:
                # Load into memory cache for faster subsequent access
                self._job_log_cache[job_id] = disk_cache[job_id]
                return disk_cache[job_id]

            # Fetch job logs via gh api (more reliable than gh run view)
            result = subprocess.run(
                ['gh', 'api', f'/repos/{owner}/{repo}/actions/jobs/{job_id}/logs'],
                capture_output=True,
                text=True,
                timeout=15
            )

            if result.returncode != 0 or not result.stdout:
                # If logs aren't available, return placeholder
                error_msg = result.stderr if result.stderr else "Unknown error"
                error_summary = f"Could not fetch job logs: {error_msg}\n\nView full logs at:\n{job_url}"
                self._job_log_cache[job_id] = error_summary
                self._save_to_disk_cache(job_id, error_summary)
                return error_summary

            # Get all log lines
            all_lines = result.stdout.strip().split('\n')

            # First, try to extract pytest short test summary (most useful for test failures)
            pytest_summary = self._extract_pytest_summary(all_lines)
            if pytest_summary:
                self._job_log_cache[job_id] = pytest_summary
                self._save_to_disk_cache(job_id, pytest_summary)
                return pytest_summary

            # Filter for error-related lines with surrounding context
            error_keywords = ['error', 'fail', 'Error', 'ERROR', 'FAIL', 'fatal', 'FATAL', 'broken']
            error_indices = []

            # Find all lines with error keywords
            for i, line in enumerate(all_lines):
                if line.strip() and not line.startswith('#'):
                    if any(keyword in line for keyword in error_keywords):
                        error_indices.append(i)

            # If we found error lines, extract them with surrounding context
            if error_indices:
                # For each error, get 10 lines before and 5 lines after
                context_lines = set()
                for error_idx in error_indices:
                    # Add lines before (up to 10)
                    for i in range(max(0, error_idx - 10), error_idx):
                        context_lines.add(i)
                    # Add error line itself
                    context_lines.add(error_idx)
                    # Add lines after (up to 5)
                    for i in range(error_idx + 1, min(len(all_lines), error_idx + 6)):
                        context_lines.add(i)

                # Sort indices and extract lines
                sorted_indices = sorted(context_lines)
                error_lines = []
                for idx in sorted_indices:
                    line = all_lines[idx]
                    if line.strip() and not line.startswith('#'):
                        error_lines.append(line)

                # Use up to last 80 lines with context (increased from 50)
                relevant_errors = error_lines[-80:]
                summary = '\n'.join(relevant_errors)

                # Limit length to 5000 chars (increased from 3000 for more context)
                if len(summary) > 5000:
                    summary = summary[:5000] + '\n\n...(truncated, view full logs at job URL above)'

                self._job_log_cache[job_id] = summary
                self._save_to_disk_cache(job_id, summary)
                return summary

            # If no error keywords found, get last 40 lines as fallback
            last_lines = [line for line in all_lines if line.strip() and not line.startswith('#')][-40:]
            if last_lines:
                summary = '\n'.join(last_lines)
                if len(summary) > 5000:
                    summary = summary[:5000] + '\n\n...(truncated)'
                self._job_log_cache[job_id] = summary
                self._save_to_disk_cache(job_id, summary)
                return summary

            error_summary = f"No error details found in logs.\n\nView full logs at:\n{job_url}"
            self._job_log_cache[job_id] = error_summary
            self._save_to_disk_cache(job_id, error_summary)
            return error_summary

        except subprocess.TimeoutExpired:
            error_summary = f"Log fetch timed out.\n\nView full logs at:\n{job_url}"
            self._job_log_cache[job_id] = error_summary
            self._save_to_disk_cache(job_id, error_summary)
            return error_summary

        except Exception as e:
            return f"Error fetching logs: {str(e)}\n\nView full logs at:\n{job_url}"

    def get_job_raw_log_url(self, job_url: str, owner: str, repo: str, timeout: int = 10) -> Optional[str]:
        """Return the raw job log download URL for a GitHub Actions job.

        GitHub exposes a job log download endpoint:
          GET /repos/{owner}/{repo}/actions/jobs/{job_id}/logs
        which typically returns a 302 redirect to a time-limited blob URL.

        We intentionally do NOT follow redirects so we can capture the final URL
        and render it as a direct "raw log" link in HTML.
        """
        try:
            if "/job/" not in job_url:
                return None
            job_id = job_url.split("/job/")[1].split("?")[0]
            if not job_id:
                return None

            if not HAS_REQUESTS:
                return None
            assert requests is not None

            url = f"{self.base_url}/repos/{owner}/{repo}/actions/jobs/{job_id}/logs"
            resp = self._rest_get(url, timeout=timeout, allow_redirects=False)
            if resp.status_code in (301, 302, 303, 307, 308):
                return resp.headers.get("Location")
            return None
        except Exception:
            return None

    def get_job_raw_log_url_cached(
        self,
        *,
        job_url: str,
        owner: str,
        repo: str,
        ttl_s: int = DEFAULT_RAW_LOG_URL_TTL_S,
        timeout: int = 10,
    ) -> Optional[str]:
        """Cached wrapper around get_job_raw_log_url (memory + disk, short TTL).

        Note: The redirect URL is time-limited, so we keep TTL short.
        """
        try:
            key = str(job_url or "")
            if not key:
                return None
            now = int(datetime.now(timezone.utc).timestamp())

            # 1) memory cache
            ent = self._raw_log_url_mem_cache.get(key)
            if ent and isinstance(ent, dict):
                ts = int(ent.get("ts", 0) or 0)
                if ts and (now - ts) <= max(0, int(ttl_s)):
                    return ent.get("url")  # may be None
        except Exception:
            pass

        # 2) disk cache
        try:
            self._raw_log_url_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self._raw_log_url_cache_dir / "raw_log_urls.json"
            if cache_file.exists():
                disk = json.loads(cache_file.read_text() or "{}")
            else:
                disk = {}
            ent = disk.get(key) if isinstance(disk, dict) else None
            if isinstance(ent, dict):
                ts = int(ent.get("ts", 0) or 0)
                if ts and (now - ts) <= max(0, int(ttl_s)):
                    # promote to memory
                    self._raw_log_url_mem_cache[key] = {"ts": ts, "url": ent.get("url")}
                    return ent.get("url")
        except Exception:
            pass

        raw = self.get_job_raw_log_url(job_url=job_url, owner=owner, repo=repo, timeout=timeout)
        try:
            now = int(datetime.now(timezone.utc).timestamp())
            self._raw_log_url_mem_cache[str(job_url or "")] = {"ts": now, "url": raw}
            # persist to disk (best-effort)
            try:
                self._raw_log_url_cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file = self._raw_log_url_cache_dir / "raw_log_urls.json"
                disk = {}
                if cache_file.exists():
                    try:
                        disk = json.loads(cache_file.read_text() or "{}")
                    except Exception:
                        disk = {}
                if isinstance(disk, dict):
                    disk[str(job_url or "")] = {"ts": now, "url": raw}
                    tmp = str(cache_file) + ".tmp"
                    Path(tmp).write_text(json.dumps(disk))
                    os.replace(tmp, cache_file)
            except Exception:
                pass
        except Exception:
            pass
        return raw

    def get_job_raw_log_text_cached(
        self,
        *,
        job_url: str,
        owner: str,
        repo: str,
        ttl_s: int = DEFAULT_RAW_LOG_TEXT_TTL_S,
        timeout: int = 30,
        max_bytes: int = DEFAULT_RAW_LOG_TEXT_MAX_BYTES,
        assume_completed: bool = False,
    ) -> Optional[str]:
        """Download and cache the *text* content of a GitHub Actions job log.

        - Uses the stable API endpoint `/actions/jobs/{job_id}/logs` (returns a ZIP).
        - Extracts and concatenates text files from the ZIP.
        - Caches on disk per job_id to support later parsing without repeated downloads.
        """
        try:
            if "/job/" not in (job_url or ""):
                return None
            job_id = str(job_url.split("/job/")[1].split("?")[0] or "").strip()
            if not job_id:
                return None
        except Exception:
            return None

        # IMPORTANT: never cache logs for jobs that are not done.
        # The /actions/jobs/{id}/logs endpoint can return partial logs while a job is running.
        # If the caller already proved completion (e.g., via check-runs), it can set assume_completed=True
        # to avoid an extra REST call to /actions/jobs/{id}.
        if not bool(assume_completed):
            try:
                st = str(self.get_actions_job_status(owner=owner, repo=repo, job_id=job_id) or "").lower()
                if not st or st != "completed":
                    return None
            except Exception:
                # Conservative: if we can't confirm the job is completed, don't cache.
                return None

        now = int(datetime.now(timezone.utc).timestamp())

        # 1) memory cache
        try:
            ent = self._raw_log_text_mem_cache.get(job_id)
            if isinstance(ent, dict):
                ts = int(ent.get("ts", 0) or 0)
                if ts and (now - ts) <= max(0, int(ttl_s)):
                    return ent.get("text")
        except Exception:
            pass

        # 2) disk cache (per job_id)
        try:
            self._raw_log_text_cache_dir.mkdir(parents=True, exist_ok=True)
            txt_path = self._raw_log_text_cache_dir / f"{job_id}.log"
            legacy_txt_path = self._raw_log_text_cache_dir / f"{job_id}.txt"
            meta = {}
            if self._raw_log_text_index_file.exists():
                try:
                    meta = json.loads(self._raw_log_text_index_file.read_text() or "{}")
                except Exception:
                    meta = {}
            ent = meta.get(job_id) if isinstance(meta, dict) else None
            # Prefer .log; fall back to legacy .txt.
            chosen_path = txt_path if txt_path.exists() else legacy_txt_path
            if chosen_path.exists() and isinstance(ent, dict):
                ts = int(ent.get("ts", 0) or 0)
                # Only trust cache entries that were recorded as completed.
                # (Older caches may have been populated while the job was in-progress, yielding partial logs.)
                if bool(ent.get("completed", False)) and ts and (now - ts) <= max(0, int(ttl_s)):
                    text = chosen_path.read_text(encoding="utf-8", errors="replace")
                    self._raw_log_text_mem_cache[job_id] = {"ts": ts, "text": text}
                    # Best-effort migrate legacy .txt -> .log for future runs.
                    if chosen_path == legacy_txt_path and (not txt_path.exists()):
                        try:
                            tmp = str(txt_path) + ".tmp"
                            Path(tmp).write_text(text, encoding="utf-8", errors="replace")
                            os.replace(tmp, txt_path)
                        except Exception:
                            pass
                    return text
                # If entry exists but isn't trusted, remove the stale local file so callers can refetch.
                if chosen_path.exists() and not bool(ent.get("completed", False)):
                    try:
                        if chosen_path == legacy_txt_path and txt_path.exists():
                            # keep preferred .log if it exists
                            pass
                        else:
                            chosen_path.unlink()
                    except Exception:
                        pass
        except Exception:
            pass

        # 3) fetch + extract
        if not HAS_REQUESTS:
            return None
        assert requests is not None

        tmp_zip_path: Optional[Path] = None
        try:
            api_url = f"{self.base_url}/repos/{owner}/{repo}/actions/jobs/{job_id}/logs"
            resp = self._rest_get(api_url, timeout=timeout, allow_redirects=True, stream=True)
            resp.raise_for_status()

            # Download the ZIP to disk to avoid buffering large logs in memory.
            self._raw_log_text_cache_dir.mkdir(parents=True, exist_ok=True)
            tmp_zip_path = self._raw_log_text_cache_dir / f"{job_id}.zip.tmp"
            limit = int(max_bytes or 0)
            got = 0
            with open(tmp_zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    got += len(chunk)
                    if limit and got >= limit:
                        break

            # Extract to a temp .log, then atomically move into place.
            txt_path = self._raw_log_text_cache_dir / f"{job_id}.log"
            tmp_txt = str(txt_path) + ".tmp"
            with open(tmp_txt, "w", encoding="utf-8", errors="replace") as out:
                try:
                    with zipfile.ZipFile(str(tmp_zip_path)) as zf:  # type: ignore[name-defined]
                        names = list(zf.namelist())
                        for name in names:
                            try:
                                data = zf.read(name)
                            except Exception:
                                continue
                            try:
                                t = data.decode("utf-8", errors="replace")
                            except Exception:
                                continue
                            if not t:
                                continue
                            if len(names) > 1:
                                out.write(f"===== {name} =====\n")
                            out.write(t)
                            if not t.endswith("\n"):
                                out.write("\n")
                except Exception:
                    # If it wasn't a zip for some reason, fall back to a best-effort decode.
                    try:
                        raw_bytes = Path(tmp_zip_path).read_bytes()
                        out.write(raw_bytes.decode("utf-8", errors="replace"))
                    except Exception:
                        pass

            os.replace(tmp_txt, txt_path)

            # Best-effort persist index + mem cache
            try:
                meta = {}
                if self._raw_log_text_index_file.exists():
                    try:
                        meta = json.loads(self._raw_log_text_index_file.read_text() or "{}")
                    except Exception:
                        meta = {}
                if not isinstance(meta, dict):
                    meta = {}
                try:
                    size_b = int(txt_path.stat().st_size)
                except Exception:
                    size_b = 0
                meta[job_id] = {"ts": now, "bytes": size_b, "completed": True}
                tmp_meta = str(self._raw_log_text_index_file) + ".tmp"
                Path(tmp_meta).write_text(json.dumps(meta))
                os.replace(tmp_meta, self._raw_log_text_index_file)
            except Exception:
                pass

            try:
                text = txt_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                text = ""
            self._raw_log_text_mem_cache[job_id] = {"ts": now, "text": text}
            return text
        except Exception:
            return None
        finally:
            try:
                if tmp_zip_path and tmp_zip_path.exists():
                    tmp_zip_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass

    def get_failed_checks(self, owner: str, repo: str, sha: str, required_checks: set, pr_number: Optional[int] = None,
                         checks_data: Optional[dict] = None) -> Tuple[List[FailedCheck], Optional[str]]:
        """Get failed CI checks for a commit using gh CLI.

        Args:
            owner: Repository owner
            repo: Repository name
            sha: Commit SHA (unused, kept for compatibility)
            required_checks: Set of required check names
            pr_number: PR number (optional, if available use gh pr checks)
            checks_data: Optional pre-fetched checks data from _fetch_pr_checks_data()

        Returns:
            Tuple of (List of FailedCheck objects, rerun_url)

            Example return value:
            (
                [
                    FailedCheck(
                        name="build-test-arm64",
                        conclusion="failure",
                        job_id="12345678",
                        job_url="https://github.com/owner/repo/actions/runs/12345678",
                        rerun_url="https://github.com/owner/repo/actions/runs/12345678/rerun"
                    ),
                    FailedCheck(
                        name="lint",
                        conclusion="failure",
                        job_id="87654321",
                        job_url="https://github.com/owner/repo/actions/runs/87654321",
                        rerun_url=None
                    )
                ],
                "https://github.com/owner/repo/actions/runs/99999999/rerun-all-jobs"
            )
        """
        try:

            # If checks_data is provided, use it instead of fetching
            if checks_data:
                checks = checks_data.get('checks', [])
                failed_checks = []
                rerun_run_id = None

                for check in checks:
                    # Only process failed checks
                    if check['status'].lower() != 'fail':
                        continue

                    check_name = check['name']
                    html_url = check['url']
                    duration = check['duration']

                    # Extract run ID from html_url
                    check_run_id = ''
                    if '/runs/' in html_url:
                        check_run_id = html_url.split('/runs/')[1].split('/')[0]
                        if not rerun_run_id and check_run_id:
                            rerun_run_id = check_run_id

                    # Check if this is a required check
                    is_required = check_name in required_checks

                    # Get error summary from job logs
                    error_summary = None
                    if check_run_id and html_url:
                        error_summary = self.get_job_error_summary(check_run_id, html_url, owner, repo)

                    failed_check = FailedCheck(
                        name=check_name,
                        job_url=html_url,
                        raw_log_url=None,
                        run_id=check_run_id,
                        duration=duration,
                        is_required=is_required,
                        error_summary=error_summary
                    )
                    failed_checks.append(failed_check)

                # Sort: required checks first, then by name
                failed_checks.sort(key=lambda x: (not x.is_required, x.name))

                # Generate rerun URL if we have a run_id
                rerun_url = None
                if rerun_run_id:
                    rerun_url = f"https://github.com/{owner}/{repo}/actions/runs/{rerun_run_id}"

                return failed_checks, rerun_url

            # Use gh pr checks if PR number is available (more reliable)
            if pr_number:
                result = subprocess.run(
                    ['gh', 'pr', 'checks', str(pr_number), '--repo', f'{owner}/{repo}'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                # gh pr checks returns non-zero when there are failed checks
                # We only care if the command itself failed (no output)
                # So just check if we got output

                if not result.stdout.strip():
                    return [], None

                failed_checks = []
                rerun_run_id = None  # Track run_id for rerun command

                # Parse tab-separated output
                for line in result.stdout.strip().split('\n'):
                    if not line.strip():
                        continue

                    parts = line.split('\t')
                    if len(parts) < 2:
                        continue

                    check_name = parts[0]
                    status = parts[1]

                    # Only process failed checks
                    if status != 'fail':
                        continue

                    duration = parts[2] if len(parts) > 2 else ''
                    html_url = parts[3] if len(parts) > 3 else ''

                    # Extract run ID from html_url
                    check_run_id = ''
                    if '/runs/' in html_url:
                        check_run_id = html_url.split('/runs/')[1].split('/')[0]
                        # Save the first valid run_id we find for the rerun command
                        if not rerun_run_id and check_run_id:
                            rerun_run_id = check_run_id

                    # Check if this is a required check
                    is_required = check_name in required_checks

                    # Get error summary from job logs
                    error_summary = None
                    if check_run_id and html_url:
                        error_summary = self.get_job_error_summary(check_run_id, html_url, owner, repo)

                    failed_check = FailedCheck(
                        name=check_name,
                        job_url=html_url,
                        raw_log_url=None,
                        run_id=check_run_id,
                        duration=duration,
                        is_required=is_required,
                        error_summary=error_summary
                    )
                    failed_checks.append(failed_check)

                # Sort: required checks first, then by name
                failed_checks.sort(key=lambda x: (not x.is_required, x.name))

                # Generate rerun URL if we have a run_id
                rerun_url = None
                if rerun_run_id:
                    rerun_url = f"https://github.com/{owner}/{repo}/actions/runs/{rerun_run_id}"

                return failed_checks, rerun_url

            # Fallback to GitHub API if no PR number
            endpoint = f"/repos/{owner}/{repo}/commits/{sha}/check-runs"
            data = self.get(endpoint)
            if not data or 'check_runs' not in data:
                return [], None

            failed_checks = []
            run_id = None

            for check in data['check_runs']:
                if check.get('conclusion') == 'failure':
                    check_name = check['name']

                    # Extract run ID from html_url
                    html_url = check.get('html_url', '')
                    if '/runs/' in html_url:
                        run_id = html_url.split('/runs/')[1].split('/')[0]

                    # Format duration
                    started = check.get('started_at', '')
                    completed = check.get('completed_at', '')
                    duration = ''
                    if started and completed:
                        try:
                            start_time = datetime.fromisoformat(started.replace('Z', '+00:00'))
                            end_time = datetime.fromisoformat(completed.replace('Z', '+00:00'))
                            duration_sec = int((end_time - start_time).total_seconds())
                            if duration_sec >= 60:
                                duration = f"{duration_sec // 60}m{duration_sec % 60}s"
                            else:
                                duration = f"{duration_sec}s"
                        except Exception:
                            duration = "unknown"

                    # Check if this is a required check
                    is_required = check_name in required_checks

                    # Get error summary from job logs
                    error_summary = None
                    if run_id and html_url:
                        error_summary = self.get_job_error_summary(run_id, html_url, owner, repo)

                    failed_check = FailedCheck(
                        name=check_name,
                        job_url=html_url,
                        raw_log_url=None,
                        run_id=run_id or '',
                        duration=duration,
                        is_required=is_required,
                        error_summary=error_summary
                    )
                    failed_checks.append(failed_check)

            # Sort: required checks first, then by name
            failed_checks.sort(key=lambda x: (not x.is_required, x.name))

            # Generate rerun URL if we have a run_id
            rerun_url = None
            if run_id:
                rerun_url = f"https://github.com/{owner}/{repo}/actions/runs/{run_id}"

            return failed_checks, rerun_url

        except Exception as e:
            print(f"Error fetching failed checks: {e}", file=sys.stderr)
            return [], None

    def get_running_checks(self, pr_number: int, owner: str, repo: str, required_checks: set,
                          checks_data: Optional[dict] = None) -> List[RunningCheck]:
        """Get running CI checks for a PR.

        Args:
            pr_number: PR number
            owner: Repository owner
            repo: Repository name
            required_checks: Set of required check names
            checks_data: Optional pre-fetched checks data from _fetch_pr_checks_data()

        Returns:
            List of RunningCheck objects
        """
        try:
            # If checks_data is provided, use it instead of fetching
            if checks_data:
                checks = checks_data.get('checks', [])
                running_checks = []

                for check in checks:
                    status = check['status'].lower()
                    # Check if it's running (gh pr checks returns: pending, queued, in_progress)
                    if status in ('pending', 'queued', 'in_progress'):
                        name = check['name']
                        check_url = check['url']
                        is_required = name in required_checks

                        # Use duration from gh pr checks (already formatted like "1m30s")
                        elapsed_time = check['duration'] if check['duration'] else 'queued'

                        running_check = RunningCheck(
                            name=name,
                            check_url=check_url,
                            is_required=is_required,
                            elapsed_time=elapsed_time
                        )
                        running_checks.append(running_check)

                # Sort: required checks first, then by name
                running_checks.sort(key=lambda x: (not x.is_required, x.name))
                return running_checks

            # Fallback: use gh pr view if checks_data not provided
            result = subprocess.run(
                ['gh', 'pr', 'view', str(pr_number), '--repo', f'{owner}/{repo}', '--json', 'statusCheckRollup'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0 or not result.stdout:
                return []

            data = json.loads(result.stdout)
            checks = data.get('statusCheckRollup', [])

            running_checks = []
            for check in checks:
                status = check.get('status', '').upper()
                # Check if it's running
                if status in ('IN_PROGRESS', 'QUEUED', 'PENDING'):
                    name = check.get('name', '')
                    check_url = check.get('detailsUrl', '')
                    is_required = check.get('isRequired', False) or (name in required_checks)

                    # Calculate elapsed time
                    elapsed_time = None
                    started_at = check.get('startedAt')
                    if started_at and started_at != '0001-01-01T00:00:00Z':
                        try:
                            start_time = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                            now = datetime.now(timezone.utc)
                            elapsed = now - start_time
                            total_seconds = int(elapsed.total_seconds())

                            if total_seconds < 60:
                                elapsed_time = f'{total_seconds}s'
                            else:
                                minutes = total_seconds // 60
                                seconds = total_seconds % 60
                                elapsed_time = f'{minutes}m{seconds}s'
                        except Exception:
                            elapsed_time = None
                    elif status == 'QUEUED':
                        elapsed_time = 'queued'

                    running_check = RunningCheck(
                        name=name,
                        check_url=check_url,
                        is_required=is_required,
                        elapsed_time=elapsed_time
                    )
                    running_checks.append(running_check)

            # Sort: required checks first, then by name
            running_checks.sort(key=lambda x: (not x.is_required, x.name))

            return running_checks

        except Exception as e:
            print(f"Error fetching running checks for PR {pr_number}: {e}", file=sys.stderr)
            return []

    def get_pr_info(self, owner: str, repo: str, branch: str) -> List[PRInfo]:
        """Get PR information for a branch.

        Args:
            owner: Repository owner
            repo: Repository name
            branch: Branch name

        Returns:
            List of PRInfo objects

            Example return value:
            [
                PRInfo(
                    number=1234,
                    title="Add Docker image caching improvements",
                    url="https://github.com/owner/repo/pull/1234",
                    state="open",
                    mergeable_state="clean",
                    sha="21a03b316dc1e5031183965e5798b0d9fe2e64b3",
                    checks_status="success"
                ),
                PRInfo(
                    number=1233,
                    title="Fix timezone handling in cache",
                    url="https://github.com/owner/repo/pull/1233",
                    state="closed",
                    mergeable_state=None,
                    sha="5fe0476e605d2564234f00e8123461e1594a9ce7",
                    checks_status="failure"
                )
            ]
        """
        try:
            pr_by_branch = self.get_pr_info_for_branches(owner, repo, [branch])
            return pr_by_branch.get(branch, []) or []

        except Exception as e:
            print(f"Error fetching PR info for {branch}: {e}", file=sys.stderr)
            return []

    def get_cached_pr_merge_dates(self, pr_numbers: List[int],
                                  owner: str = "ai-dynamo",
                                  repo: str = "dynamo",
                                  cache_file: str = '.github_pr_merge_dates_cache.json',
                                  stats: Optional[Dict[str, Any]] = None) -> Dict[int, Optional[str]]:
        """Get merge dates for pull requests with caching.

        Merge dates are cached permanently since they don't change once a PR is merged.

        Args:
            pr_numbers: List of PR numbers
            owner: Repository owner (default: ai-dynamo)
            repo: Repository name (default: dynamo)
            cache_file: Path to cache file

        Returns:
            Dictionary mapping PR number to merge date string (YYYY-MM-DD HH:MM:SS)
            Returns None for PRs that are not merged or not found

        Example:
            >>> client = GitHubAPIClient()
            >>> merge_dates = client.get_cached_pr_merge_dates([4965, 5009])
            >>> merge_dates
            {4965: "2025-12-18 12:34:56", 5009: None}

        Cache file format (.github_pr_merge_dates_cache.json):
        {
            "4965": "2025-12-18 12:34:56",
            "5009": null
        }
        """

        # Load cache
        cache = {}
        pr_cache_path = resolve_cache_path(cache_file)
        if pr_cache_path.exists():
            try:
                cache_raw = json.loads(pr_cache_path.read_text())
                cache = {int(k): v for k, v in cache_raw.items()}
            except Exception:
                pass

        # Prepare result and track if cache was updated
        result = {}
        cache_updated = False
        logger = logging.getLogger('common')

        # First pass: collect cached results and PRs to fetch
        prs_to_fetch = []
        pr_numbers_unique = []
        seen = set()
        for pr_num in pr_numbers:
            try:
                pr_i = int(pr_num)
            except Exception:
                continue
            if pr_i in seen:
                continue
            seen.add(pr_i)
            pr_numbers_unique.append(pr_i)

        for pr_num in pr_numbers_unique:
            if pr_num in cache:
                result[pr_num] = cache[pr_num]
            else:
                prs_to_fetch.append(pr_num)

        # Fetch uncached PRs in parallel
        if prs_to_fetch:
            def fetch_pr_merge_date(pr_num):
                """Helper function to fetch a single PR's merge date"""
                try:
                    pr_details = self.get_pr_details(owner, repo, pr_num)

                    if pr_details and pr_details.get('merged_at'):
                        # Parse ISO timestamp: "2025-12-18T12:34:56Z" (UTC)
                        merged_at = pr_details['merged_at']
                        dt_utc = datetime.fromisoformat(merged_at.replace('Z', '+00:00'))

                        # Convert to Pacific time (PST/PDT)
                        dt_pacific = dt_utc.astimezone(ZoneInfo('America/Los_Angeles'))
                        merge_date = dt_pacific.strftime('%Y-%m-%d %H:%M:%S')
                        return (pr_num, merge_date)
                    else:
                        # PR not merged or not found
                        return (pr_num, None)
                except Exception as e:
                    # Log error but continue with other PRs
                    logger.debug(f"Failed to fetch PR {pr_num} merge date: {e}")
                    return (pr_num, None)

            # Fetch in parallel with 10 workers
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(fetch_pr_merge_date, pr_num) for pr_num in prs_to_fetch]

                # Collect results as they complete
                for future in futures:
                    try:
                        pr_num, merge_date = future.result()
                        result[pr_num] = merge_date
                        cache[pr_num] = merge_date
                        cache_updated = True
                    except Exception as e:
                        logger.debug(f"Failed to get future result: {e}")

        # Save updated cache
        if cache_updated:
            try:
                pr_cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_str_keys = {str(k): v for k, v in cache.items()}
                pr_cache_path.write_text(json.dumps(cache_str_keys, indent=2))
            except Exception:
                pass

        # Stats (best-effort): merge dates are immutable, so misses mean "not cached yet".
        try:
            if isinstance(stats, dict):
                stats["merge_dates_cache"] = {
                    "total_prs": len(pr_numbers_unique),
                    "hits": len(pr_numbers_unique) - len(prs_to_fetch),
                    "misses": len(prs_to_fetch),
                    "cache_file": str(pr_cache_path),
                    "miss_reason": "not_in_cache" if prs_to_fetch else "",
                    "miss_prs_sample": [int(x) for x in prs_to_fetch[:10]],
                }
        except Exception:
            pass

        return result

    def get_github_actions_status(
        self,
        owner,
        repo,
        sha_list,
        cache_file=None,
        skip_fetch=False,
        *,
        ttl_s: int = DEFAULT_UNSTABLE_TTL_S,
        fetch_allowlist: Optional[set] = None,
        sha_to_datetime: Optional[Dict[str, datetime]] = None,
        stable_after_hours: int = DEFAULT_STABLE_AFTER_HOURS,
        stable_ttl_s: int = DEFAULT_STABLE_TTL_S,
        flush_every: int = 10,
        stats: Optional[Dict[str, Any]] = None,
    ):
        """Get GitHub Actions check status for commits.

        Args:
            owner: Repository owner
            repo: Repository name
            sha_list: List of commit SHAs (full 40-char)
            cache_file: Path to cache file (default: .github_actions_status_cache.json)
            skip_fetch: If True, only return cached data

        Returns:
            Dict mapping SHA -> status info:
            {
                "sha": {
                    "status": "success|failure|pending|in_progress|null",
                    "conclusion": "success|failure|cancelled|skipped|timed_out|action_required|null",
                    "total_count": int,
                    "check_runs": [...]
                }
            }
        """
        from pathlib import Path
        from concurrent.futures import ThreadPoolExecutor

        if cache_file is None:
            cache_file = resolve_cache_path('.github_actions_status_cache.json')
        else:
            cache_file = resolve_cache_path(str(cache_file))

        # Load cache
        cache = {}
        if cache_file.exists():
            try:
                cache = json.loads(cache_file.read_text())
            except Exception:
                cache = {}

        now = int(time.time())

        def _unpack(ent: Any) -> tuple[int, Any]:
            # New format: {"ts": 123, "data": {...}}
            if isinstance(ent, dict) and "ts" in ent and "data" in ent:
                try:
                    return int(ent.get("ts") or 0), ent.get("data")
                except Exception:
                    return 0, ent.get("data")
            # Legacy: cached value directly (no ts)
            return 0, ent

        def _pack(ts: int, data: Any) -> Dict[str, Any]:
            return {"ts": int(ts), "data": data}

        def _allow_fetch(sha: str) -> bool:
            if skip_fetch:
                return False
            if fetch_allowlist is None:
                return True
            try:
                # Normally, respect the allowlist (used to cap network calls).
                #
                # However, if the cache indicates this SHA is still "in progress"/"pending",
                # we must allow a refresh even if the commit is older than the time window.
                # Otherwise, an in-progress cache entry can get "stuck" forever.
                if sha in fetch_allowlist:
                    return True
                ent = cache.get(sha) if isinstance(cache, dict) else None
                _ts, _data = _unpack(ent)
                if isinstance(_data, dict):
                    st = str(_data.get("status", "") or "").strip().lower()
                    if st in ("in_progress", "pending"):
                        return True
                    crs = _data.get("check_runs")
                    if isinstance(crs, list):
                        for cr in crs:
                            if not isinstance(cr, dict):
                                continue
                            cr_status = str(cr.get("status", "") or "").strip().lower()
                            cr_concl = str(cr.get("conclusion", "") or "").strip().lower()
                            if cr_status in ("queued", "in_progress", "pending"):
                                return True
                            if cr_status and cr_status != "completed" and cr_concl in ("", "null", "none"):
                                return True
                return False
            except Exception:
                return False

        shas_to_fetch: List[str] = []
        result: Dict[str, Any] = {}
        # Stats (best-effort)
        _stats = {
            "total_shas": 0,
            "cache_hit_fresh": 0,
            "cache_hit_stale_no_fetch": 0,
            "cache_stale_refresh": 0,
            "cache_miss_fetch": 0,
            "cache_miss_no_fetch": 0,
            "skip_fetch": bool(skip_fetch),
            "allowlist_size": int(len(fetch_allowlist)) if isinstance(fetch_allowlist, set) else None,
            "cache_file": str(cache_file),
            "miss_reason_samples": {},  # reason -> [sha_short,...]
        }

        for sha in sha_list:
            _stats["total_shas"] += 1
            ent = cache.get(sha) if isinstance(cache, dict) else None
            ts, data = _unpack(ent)

            if ent is not None:
                # If cached, return it, but optionally refresh if stale AND allowed.
                result[sha] = data
                stale = True
                try:
                    per_sha_ttl = int(ttl_s)
                    if sha_to_datetime is not None:
                        dt = sha_to_datetime.get(sha)
                        per_sha_ttl = self.compute_checks_cache_ttl_s(
                            dt,
                            refresh=False,
                            stable_after_hours=int(stable_after_hours),
                            short_ttl_s=int(ttl_s),
                            stable_ttl_s=int(stable_ttl_s),
                        )
                    if ts and (now - ts) <= max(0, int(per_sha_ttl)):
                        stale = False
                except Exception:
                    stale = True
                if stale and _allow_fetch(sha):
                    shas_to_fetch.append(sha)
                    _stats["cache_stale_refresh"] += 1
                elif stale and not _allow_fetch(sha):
                    _stats["cache_hit_stale_no_fetch"] += 1
                else:
                    _stats["cache_hit_fresh"] += 1
            else:
                if _allow_fetch(sha):
                    shas_to_fetch.append(sha)
                    _stats["cache_miss_fetch"] += 1
                else:
                    result[sha] = None
                    _stats["cache_miss_no_fetch"] += 1
                    try:
                        _stats["miss_reason_samples"].setdefault("blocked_by_allowlist_or_skip_fetch", []).append(str(sha)[:9])
                    except Exception:
                        pass

        def _flush_cache_to_disk() -> None:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                tmp = str(cache_file) + ".tmp"
                Path(tmp).write_text(json.dumps(cache, indent=2))
                os.replace(tmp, cache_file)
            except Exception:
                pass

        # Fetch uncached SHAs in parallel
        cache_updated = False
        if shas_to_fetch:
            def fetch_check_status(sha):
                """Helper to fetch check status for a single commit"""
                try:
                    # Use GitHub API to get check runs
                    result = subprocess.run(
                        ['gh', 'api', f'/repos/{owner}/{repo}/commits/{sha}/check-runs',
                         '--jq', '.'],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )

                    if result.returncode == 0 and result.stdout.strip():
                        data = json.loads(result.stdout)
                        check_runs = data.get('check_runs', [])
                        total_count = data.get('total_count', 0)

                        # Determine overall status
                        if total_count == 0:
                            status = 'null'
                            conclusion = 'null'
                        else:
                            # Check for any failures
                            has_failure = any(cr.get('conclusion') in ['failure', 'timed_out', 'action_required']
                                            for cr in check_runs)
                            has_pending = any(cr.get('status') in ['queued', 'in_progress']
                                            for cr in check_runs)
                            has_cancelled = any(cr.get('conclusion') == 'cancelled' for cr in check_runs)

                            if has_failure:
                                status = 'completed'
                                conclusion = 'failure'
                            elif has_pending:
                                status = 'in_progress'
                                conclusion = 'null'
                            elif has_cancelled:
                                status = 'completed'
                                conclusion = 'cancelled'
                            else:
                                # All succeeded or skipped
                                status = 'completed'
                                conclusion = 'success'

                        return (sha, {
                            'status': status,
                            'conclusion': conclusion,
                            'total_count': total_count,
                            'check_runs': check_runs
                        })
                    else:
                        return (sha, None)
                except Exception as e:
                    _logger.debug(f"Failed to fetch check status for {sha[:8]}: {e}")
                    return (sha, None)

            # Fetch in parallel with 10 workers
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(fetch_check_status, sha) for sha in shas_to_fetch]
                flushed = 0
                for future in futures:
                    sha, status_info = future.result()
                    cache[sha] = _pack(int(time.time()), status_info)
                    result[sha] = status_info
                    cache_updated = True
                    flushed += 1
                    # Periodically flush cache so a crash/kill doesn't lose fetched data.
                    if int(flush_every or 0) > 0 and (flushed % int(flush_every) == 0):
                        _flush_cache_to_disk()

        # Save cache if updated
        if cache_updated:
            _flush_cache_to_disk()

        # Expose stats to caller
        try:
            if isinstance(stats, dict):
                # Also report how many SHAs we actually fetched this run.
                _stats["fetched_shas"] = int(len(shas_to_fetch))
                stats["github_actions_status_cache"] = _stats
        except Exception:
            pass

        return result

        # unreachable


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

    try:
        now_utc = datetime.now(timezone.utc)
        cutoff_s = float(recent_hours) * 3600.0
        for sha in sha_list[: int(max_fetch or 0)]:
            dt = sha_to_datetime.get(sha)
            if dt is None:
                continue
            if getattr(dt, "tzinfo", None) is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age_s = (now_utc - dt.astimezone(timezone.utc)).total_seconds()
            if age_s < cutoff_s:
                allow.add(sha)
    except Exception:
        return set()
    return allow


def normalize_check_name(name: str) -> str:
    """Normalize GitHub/GitLab check names for robust comparison."""
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def is_required_check_name(check_name: str, required_names_normalized: set[str]) -> bool:
    """Return True iff the check name is in the required set (branch protection)."""
    name_norm = normalize_check_name(check_name)
    if not name_norm:
        return False
    return name_norm in (required_names_normalized or set())


def format_gh_check_run_duration(check_run: Dict[str, Any]) -> str:
    """Compute a short duration string for a GitHub check_run dict.

    Args:
        check_run: A dict like the items returned by GitHub's check-runs API.

    Returns:
        A short duration string like "3s", "2m 10s", "1h 4m", or "" if unknown.

    Example check_run dict (partial):
        {
          "name": "backend-status-check",
          "status": "completed",
          "conclusion": "success",
          "started_at": "2025-12-24T09:06:10Z",
          "completed_at": "2025-12-24T09:06:13Z",
          "html_url": "https://github.com/.../actions/runs/.../job/..."
        }
    """
    try:
        started = str(check_run.get("started_at", "") or "")
        completed = str(check_run.get("completed_at", "") or "")
        if not started or not completed:
            return ""
        st = datetime.fromisoformat(started.replace("Z", "+00:00"))
        ct = datetime.fromisoformat(completed.replace("Z", "+00:00"))
        delta_s = int((ct - st).total_seconds())
        return GitHubAPIClient._format_seconds_delta(delta_s)
    except Exception:
        return ""


class GitLabAPIClient:
    """GitLab API client with automatic token detection and error handling.

    Features:
    - Automatic token detection (--token arg > GITLAB_TOKEN env > ~/.config/gitlab-token)
    - Request/response handling with proper error messages
    - Container registry queries

    Example:
        client = GitLabAPIClient()
        # Use get_cached_registry_images_for_shas for fetching Docker images
    """


    @staticmethod
    def get_gitlab_token_from_file() -> Optional[str]:
        """Get GitLab token from ~/.config/gitlab-token file.

        Returns:
            GitLab token string, or None if not found
        """
        try:
            token_file = Path.home() / '.config' / 'gitlab-token'
            if token_file.exists():
                return token_file.read_text().strip()
        except Exception:
            pass
        return None

    def __init__(self, token: Optional[str] = None, base_url: str = "https://gitlab-master.nvidia.com"):
        """Initialize GitLab API client.

        Args:
            token: GitLab personal access token. If not provided, will try:
                   1. GITLAB_TOKEN environment variable
                   2. ~/.config/gitlab-token file
            base_url: GitLab instance URL (default: https://gitlab-master.nvidia.com)
        """
        # Token priority: 1) provided token, 2) environment variable, 3) config file
        self.token = token or os.environ.get('GITLAB_TOKEN') or self.get_gitlab_token_from_file()
        self.base_url = base_url.rstrip('/')
        self.headers = {}

        if self.token:
            self.headers['PRIVATE-TOKEN'] = self.token

    def has_token(self) -> bool:
        """Check if a GitLab token is configured."""
        return self.token is not None

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, timeout: int = 10) -> Optional[Any]:
        """Make GET request to GitLab API.

        Args:
            endpoint: API endpoint (e.g., "/api/v4/projects/169905")
            params: Query parameters
            timeout: Request timeout in seconds

        Returns:
            JSON response (dict or list), or None if request failed

            Example return value for registry tags endpoint:
            [
                {
                    "name": "21a03b316dc1e5031183965e5798b0d9fe2e64b3-38895507-vllm-amd64",
                    "path": "dl/ai-dynamo/dynamo:21a03b316dc1e5031183965e5798b0d9fe2e64b3-38895507-vllm-amd64",
                    "location": "gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo:21a03b316...",
                    "created_at": "2025-11-20T22:15:32.829+00:00"
                },
                {
                    "name": "5fe0476e605d2564234f00e8123461e1594a9ce7-38888909-sglang-arm64",
                    "path": "dl/ai-dynamo/dynamo:5fe0476e605d2564234f00e8123461e1594a9ce7-38888909-sglang-arm64",
                    "location": "gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo:5fe0476e6...",
                    "created_at": "2025-11-19T10:00:00.000+00:00"
                }
            ]

            Example return value for pipelines endpoint:
            [
                {
                    "id": 38895507,
                    "status": "success",
                    "web_url": "https://gitlab-master.nvidia.com/dl/ai-dynamo/dynamo/-/pipelines/38895507",
                    "ref": "main",
                    "sha": "21a03b316dc1e5031183965e5798b0d9fe2e64b3"
                }
            ]
        """
        if not HAS_REQUESTS:
            # Fallback to urllib for basic GET requests

            url = f"{self.base_url}{endpoint}" if endpoint.startswith('/') else f"{self.base_url}/{endpoint}"

            if params:
                url += '?' + urllib.parse.urlencode(params)

            try:
                req = urllib.request.Request(url, headers=self.headers)
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    return json.loads(response.read().decode())
            except Exception:
                return None

        # Use requests if available
        url = f"{self.base_url}{endpoint}" if endpoint.startswith('/') else f"{self.base_url}/{endpoint}"

        try:
            assert requests is not None
            response = requests.get(url, headers=self.headers, params=params, timeout=timeout)

            if response.status_code == 401:
                raise Exception("GitLab API returned 401 Unauthorized. Check your token.")
            elif response.status_code == 403:
                raise Exception("GitLab API returned 403 Forbidden. Token may lack permissions.")
            elif response.status_code == 404:
                raise Exception(f"GitLab API returned 404 Not Found for {endpoint}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:  # type: ignore[union-attr]
            raise Exception(f"GitLab API request failed for {endpoint}: {e}")

    def get_cached_registry_images_for_shas(self, project_id: str, registry_id: str,
                                           sha_list: List[str],
                                           sha_to_datetime: Optional[Dict[str, datetime]] = None,
                                           cache_file: str = '.gitlab_commit_sha_cache.json',
                                           skip_fetch: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """Get container registry images for commit SHAs with caching.

        Optimized caching logic:
        - If skip_fetch=True: Only return cached data, no API calls
        - If skip_fetch=False: Use binary search to find tags for recent commits (within 8 hours)
          - Only fetches pages needed for recent SHAs
          - Tracks visited pages to avoid redundant API calls
          - Only updates cache for recent SHAs found

        Args:
            project_id: GitLab project ID
            registry_id: Container registry ID
            sha_list: List of full commit SHAs (40 characters)
            sha_to_datetime: Optional dict mapping SHA to committed_datetime for time-based filtering
            cache_file: Path to cache file (default: .gitlab_commit_sha_cache.json)
            skip_fetch: If True, only return cached data without fetching from GitLab

        Returns:
            Dictionary mapping SHA to list of image info dicts

        Cache file format (.gitlab_commit_sha_cache.json):
            {
                "21a03b316dc1e5031183965e5798b0d9fe2e64b3": [
                    {
                        "tag": "21a03b316dc1e5031183965e5798b0d9fe2e64b3-38895507-vllm-amd64",
                        "framework": "vllm",
                        "arch": "amd64",
                        "pipeline_id": "38895507",
                        "location": "gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo:21a03b316...",
                        "total_size": 15000000000,
                        "created_at": "2024-11-20T13:00:00Z"
                    }
                ],
                "5fe0476e605d2564234f00e8123461e1594a9ce7": []
            }
        """

        # Load cache
        cache = {}
        cache_path = resolve_cache_path(cache_file)
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text())
            except Exception:
                pass

        # Initialize result for requested SHAs
        result = {}

        if skip_fetch:
            # Only return cached data - NO API calls
            for sha in sha_list:
                result[sha] = cache.get(sha, [])

            # Warn if no images found in cache
            if not any(result.values()):
                _logger.warning("⚠️  No Docker images found in cache. Consider running without --skip-gitlab-fetch to fetch fresh data.")

            return result
        else:
            # Identify recent SHAs (within stable window)
            now_utc = datetime.now(timezone.utc)
            eight_hours_ago_utc = now_utc - timedelta(hours=DEFAULT_STABLE_AFTER_HOURS)

            recent_shas = set()
            if sha_to_datetime:
                for sha in sha_list:
                    commit_time = sha_to_datetime.get(sha)
                    if commit_time:
                        # Normalize to UTC for comparison
                        if commit_time.tzinfo is None:
                            # Naive datetime, assume UTC
                            commit_time_utc = commit_time.replace(tzinfo=timezone.utc)
                        else:
                            commit_time_utc = commit_time.astimezone(timezone.utc)

                        if commit_time_utc >= eight_hours_ago_utc:
                            recent_shas.add(sha)

            _logger.debug(f"Found {len(recent_shas)} SHAs within 8 hours (out of {len(sha_list)} total)")

            if not recent_shas:
                # No recent SHAs, just return cached data
                for sha in sha_list:
                    result[sha] = cache.get(sha, [])
                return result

            # Fetch ALL pages first, then filter by SHA
            per_page = 100

            if not self.has_token():
                # No token, show big warning
                print("\n" + "="*80)
                print("⚠️  WARNING: No GitLab token found!")
                print("="*80)
                print("Cannot fetch Docker registry images without a GitLab token.")
                print("\nTo set up a token:")
                print("1. Create a personal access token at:")
                print("   https://gitlab-master.nvidia.com/-/profile/personal_access_tokens")
                print("2. Set it using one of these methods:")
                print("   - Export GITLAB_TOKEN environment variable")
                print("   - Save to ~/.config/gitlab-token file")
                print("="*80 + "\n")
                return {sha: [] for sha in sha_list}

            # Fetch page 1 first to get total pages from headers
            endpoint = f"/api/v4/projects/{project_id}/registry/repositories/{registry_id}/tags"
            params = {
                'per_page': per_page,
                'page': 1,
                'order_by': 'updated_at',
                'sort': 'desc'
            }

            try:
                # Make direct request to get headers
                if HAS_REQUESTS and requests is not None:
                    url = f"{self.base_url}{endpoint}"
                    response = requests.get(url, headers=self.headers, params=params, timeout=10)
                    response.raise_for_status()
                    first_page_tags = response.json()
                    total_pages = int(response.headers.get('X-Total-Pages', '1'))
                else:
                    # Fallback: use get method and assume 1 page
                    first_page_tags = self.get(endpoint, params=params)
                    total_pages = 1

                if first_page_tags is None:
                    first_page_tags = []

                _logger.debug(f"Total pages available: {total_pages}")

            except Exception as e:
                _logger.warning(f"Failed to fetch page 1 to determine total pages: {e}")
                print("\n" + "="*80)
                print("⚠️  ERROR: Failed to fetch Docker registry tags from GitLab!")
                print("="*80)
                print(f"Error: {e}")
                print("="*80 + "\n")
                return {sha: [] for sha in sha_list}

            # Collect all tags from all pages
            all_tags = list(first_page_tags)  # Start with page 1 tags
            lock = threading.Lock()

            def fetch_page(page_num: int) -> List[Dict[str, Any]]:
                """Fetch a single page of tags."""
                endpoint = f"/api/v4/projects/{project_id}/registry/repositories/{registry_id}/tags"
                params = {
                    'per_page': per_page,
                    'page': page_num,
                    'order_by': 'updated_at',
                    'sort': 'desc'
                }

                try:
                    tags = self.get(endpoint, params=params)
                    if tags is None:
                        return []
                    return tags
                except Exception as e:
                    _logger.debug(f"Failed to fetch page {page_num}: {e}")
                    return []

            # Helper to check if tag is older than 8 hours
            def is_old_tag(tag: dict) -> bool:
                tag_created = tag.get('created_at', '')
                if not tag_created:
                    return False
                try:
                    tag_time = datetime.fromisoformat(tag_created.replace('Z', '+00:00'))
                    return tag_time < eight_hours_ago_utc
                except Exception:
                    return False

            # Check if first page has old tags
            found_old_tags = any(is_old_tag(tag) for tag in first_page_tags)
            pages_fetched = 1

            # Fetch remaining pages until we hit old tags
            if total_pages > 1 and not found_old_tags:
                _logger.debug(f"Fetching up to {total_pages} pages with early termination...")

                with ThreadPoolExecutor(max_workers=8) as executor:
                    # Submit all remaining pages
                    future_to_page = {executor.submit(fetch_page, page_num): page_num
                                     for page_num in range(2, total_pages + 1)}

                    # Process results as they complete
                    for future in as_completed(future_to_page):
                        page_num = future_to_page[future]
                        tags = future.result()
                        pages_fetched += 1

                        if tags:
                            all_tags.extend(tags)

                            # Stop if we found old tags
                            if any(is_old_tag(tag) for tag in tags):
                                found_old_tags = True
                                _logger.debug(f"Found tags older than 8 hours at page {page_num}, stopping early")
                                # Cancel remaining futures
                                for f in future_to_page:
                                    if not f.done():
                                        f.cancel()
                                break

                        if pages_fetched % 10 == 0:
                            _logger.debug(f"Fetched {pages_fetched}/{total_pages} pages...")

            _logger.debug(f"Fetched {pages_fetched} pages (stopped early: {found_old_tags}), total tags: {len(all_tags)}")

            # Now filter tags by SHA
            sha_to_images = {}
            recent_shas_set = set(recent_shas)

            for tag_info in all_tags:
                tag_name = tag_info.get('name', '')
                # Check if this tag matches any of our recent SHAs
                for sha in recent_shas_set:
                    if tag_name.startswith(sha + '-'):
                        if sha not in sha_to_images:
                            sha_to_images[sha] = []

                        parts = tag_name.split('-')
                        if len(parts) >= 4:
                            sha_to_images[sha].append({
                                'tag': tag_name,
                                'framework': parts[2],
                                'arch': parts[3],
                                'pipeline_id': parts[1],
                                'location': tag_info.get('location', ''),
                                'total_size': tag_info.get('total_size', 0),
                                'created_at': tag_info.get('created_at', '')
                            })

            found_count = len([sha for sha in recent_shas if sha in sha_to_images and sha_to_images[sha]])
            _logger.debug(f"Found tags for {found_count}/{len(recent_shas)} recent SHAs")

            # Update cache only for recent SHAs we found
            for sha, images in sha_to_images.items():
                cache[sha] = images

            # Build result for all requested SHAs (use cache for non-recent ones)
            for sha in sha_list:
                result[sha] = cache.get(sha, [])

            # Warn if no images found for recent SHAs
            if recent_shas and not any(result[sha] for sha in recent_shas):
                _logger.warning(f"⚠️  No Docker images found for any of the {len(recent_shas)} recent SHAs (within 8 hours)")
                _logger.warning("This might mean the commits haven't been built yet or the builds failed.")

            # Save updated cache with timestamp
            try:
                cache_with_metadata = {
                    '_metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'total_shas': len(cache),
                        'recent_shas_updated': len(sha_to_images)
                    }
                }
                cache_with_metadata.update(cache)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(cache_with_metadata, indent=2))
                _logger.debug(f"Updated cache with {len(sha_to_images)} recent SHAs")
            except Exception as e:
                _logger.warning(f"Failed to save cache: {e}")

        return result

    def get_cached_pipeline_status(self, sha_list: List[str],
                                  cache_file: str = '.gitlab_pipeline_status_cache.json',
                                  skip_fetch: bool = False) -> Dict[str, Optional[Dict[str, Any]]]:
        """Get GitLab CI pipeline status for commits with intelligent caching.

        Caching strategy:
        - If skip_fetch=True: Only return cached data, no API calls
        - If skip_fetch=False:
          - "success" status: Cached permanently (won't change)
          - "failed", "running", "pending", etc.: Always refetched (might be re-run)
          - None/missing: Always fetched

        Args:
            sha_list: List of full commit SHAs (40 characters)
            cache_file: Path to cache file (default: .gitlab_pipeline_status_cache.json)
            skip_fetch: If True, only return cached data without fetching from GitLab

        Returns:
            Dictionary mapping SHA to pipeline status dict (or None if no pipeline found)

            Example return value:
            {
                "21a03b316dc1e5031183965e5798b0d9fe2e64b3": {
                    "status": "success",
                    "id": 38895507,
                    "web_url": "https://gitlab-master.nvidia.com/dl/ai-dynamo/dynamo/-/pipelines/38895507"
                },
                "5fe0476e605d2564234f00e8123461e1594a9ce7": None
            }

        Cache file format (.gitlab_pipeline_status_cache.json) - internally used:
        {
            "21a03b316dc1e5031183965e5798b0d9fe2e64b3": {
                "status": "success",
                "id": 38895507,
                "web_url": "https://gitlab-master.nvidia.com/dl/ai-dynamo/dynamo/-/pipelines/38895507"
            },
            "5fe0476e605d2564234f00e8123461e1594a9ce7": {
                "status": "failed",
                "id": 38888909,
                "web_url": "https://gitlab-master.nvidia.com/dl/ai-dynamo/dynamo/-/pipelines/38888909"
            }
        }
        """

        # Load cache
        cache = {}
        pipeline_cache_path = resolve_cache_path(cache_file)
        if pipeline_cache_path.exists():
            try:
                cache = json.loads(pipeline_cache_path.read_text())
            except Exception:
                pass

        # If skip_fetch=True, only return cached data - NO API calls
        if skip_fetch:
            result = {}
            for sha in sha_list:
                result[sha] = cache.get(sha)
            return result

        # Check which SHAs need to be fetched
        # Only cache "success" status permanently; refetch others as they might change
        shas_to_fetch = []
        result = {}

        for sha in sha_list:
            if sha in cache:
                cached_info = cache[sha]
                # If pipeline succeeded, use cached value
                # If pipeline failed/running/pending, refetch as it might have been re-run
                if cached_info and cached_info.get('status') == 'success':
                    result[sha] = cached_info
                else:
                    # Non-success status or None - refetch to check for updates
                    shas_to_fetch.append(sha)
                    result[sha] = cached_info  # Use cached value temporarily
            else:
                shas_to_fetch.append(sha)
                result[sha] = None

        # Fetch missing SHAs and non-success statuses from GitLab in parallel
        if shas_to_fetch and self.has_token():

            def fetch_pipeline_status(sha):
                """Helper function to fetch pipeline status for a single SHA"""
                try:
                    # Get pipelines for this commit
                    endpoint = f"/api/v4/projects/169905/pipelines"
                    params = {'sha': sha, 'per_page': 1}
                    pipelines = self.get(endpoint, params=params)

                    if pipelines and len(pipelines) > 0:
                        pipeline = pipelines[0]  # Most recent pipeline
                        status_info = {
                            'status': pipeline.get('status', 'unknown'),
                            'id': pipeline.get('id'),
                            'web_url': pipeline.get('web_url', ''),
                        }
                        return (sha, status_info)
                    else:
                        return (sha, None)
                except Exception:
                    return (sha, None)

            # Fetch in parallel with 10 workers
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(fetch_pipeline_status, sha) for sha in shas_to_fetch]

                # Collect results as they complete
                for future in futures:
                    try:
                        sha, status_info = future.result()
                        result[sha] = status_info
                        cache[sha] = status_info
                    except Exception:
                        pass

            # Save updated cache
            try:
                pipeline_cache_path.parent.mkdir(parents=True, exist_ok=True)
                pipeline_cache_path.write_text(json.dumps(cache, indent=2))
            except Exception:
                pass

        return result

    def get_cached_pipeline_job_counts(self, pipeline_ids: List[int],
                                      cache_file: str = '.gitlab_pipeline_jobs_cache.json',
                                      skip_fetch: bool = False) -> Dict[int, Optional[Dict[str, int]]]:
        """Get GitLab CI pipeline job counts with intelligent caching.

        Caching strategy:
        - If skip_fetch=True: Only return cached data, no API calls
        - If skip_fetch=False:
          - Completed pipelines (running=0, pending=0): Cached forever, never refetched
          - Active pipelines (running>0 or pending>0): Refetch if older than 30 minutes

        Args:
            pipeline_ids: List of pipeline IDs
            cache_file: Path to cache file (default: .gitlab_pipeline_jobs_cache.json)
            skip_fetch: If True, only return cached data without fetching from GitLab

        Returns:
            Dictionary mapping pipeline ID to job counts dict (or None if fetch failed)

            Example return value:
            {
                40355198: {
                    "success": 16,
                    "failed": 0,
                    "running": 6,
                    "pending": 0
                },
                40341238: {
                    "success": 11,
                    "failed": 13,
                    "running": 0,
                    "pending": 0
                }
            }

        Cache file format (with timestamps):
            {
                "40355198": {
                    "counts": {"success": 16, "failed": 0, "running": 6, "pending": 0},
                    "fetched_at": "2025-12-17T18:15:00Z"
                }
            }
        """
        # Load cache
        cache = {}
        jobs_cache_path = resolve_cache_path(cache_file)
        if jobs_cache_path.exists():
            try:
                cache = json.loads(jobs_cache_path.read_text())
                # Convert string keys back to int
                cache = {int(k): v for k, v in cache.items()}
            except Exception:
                pass

        # Helper function to extract counts from cache entry (handles old and new format)
        def extract_counts(entry):
            if not entry:
                return None
            return entry.get('counts', entry) if isinstance(entry, dict) else entry

        # Helper function to check if pipeline is completed (no running/pending jobs)
        def is_completed(entry):
            counts = extract_counts(entry)
            if not counts:
                return False
            return counts.get('running', 0) == 0 and counts.get('pending', 0) == 0

        # Helper function to check if cache entry is fresh
        # Completed pipelines (no running/pending) are cached forever
        # Active pipelines (running/pending) must be < 30 minutes old
        def is_fresh(entry, now, age_limit):
            if not isinstance(entry, dict) or 'fetched_at' not in entry:
                return False  # Old format or missing timestamp = stale

            # If pipeline is completed, cache forever
            if is_completed(entry):
                return True

            # Otherwise, check if < 30 minutes old
            try:
                fetched_at = datetime.fromisoformat(entry['fetched_at'].replace('Z', '+00:00'))
                return (now - fetched_at) < age_limit
            except Exception:
                return False  # Invalid timestamp = stale

        now = datetime.now(timezone.utc)
        cache_age_limit = timedelta(minutes=30)

        # If skip_fetch=True, only return cached data - NO API calls
        if skip_fetch:
            return {pid: extract_counts(cache.get(pid)) for pid in pipeline_ids}

        # Determine which pipelines need fetching
        pipeline_ids_to_fetch = []
        result = {}

        for pipeline_id in pipeline_ids:
            cached_entry = cache.get(pipeline_id)

            if cached_entry and is_fresh(cached_entry, now, cache_age_limit):
                # Cache is fresh, use it
                result[pipeline_id] = extract_counts(cached_entry)
            else:
                # Not in cache, or cache is stale - refetch
                pipeline_ids_to_fetch.append(pipeline_id)
                result[pipeline_id] = extract_counts(cached_entry)  # Use existing data temporarily if available

        # Fetch missing pipeline job counts from GitLab in parallel
        if pipeline_ids_to_fetch and self.has_token():
            fetch_timestamp = now.isoformat().replace('+00:00', 'Z')

            def fetch_pipeline_jobs(pipeline_id):
                """Helper function to fetch job counts for a single pipeline"""
                try:
                    # Get jobs for this pipeline
                    endpoint = f"/api/v4/projects/169905/pipelines/{pipeline_id}/jobs"
                    params = {'per_page': 100}  # Get up to 100 jobs
                    jobs = self.get(endpoint, params=params)

                    if jobs:
                        # Count jobs by status
                        counts = {
                            'success': 0,
                            'failed': 0,
                            'running': 0,
                            'pending': 0
                        }

                        for job in jobs:
                            status = job.get('status', 'unknown')
                            if status in counts:
                                counts[status] += 1
                            # Map other statuses to main categories
                            elif status in ('skipped', 'manual', 'canceled'):
                                pass  # Don't count these
                            elif status in ('created', 'waiting_for_resource'):
                                counts['pending'] += 1

                        # Return with timestamp
                        return (pipeline_id, counts, fetch_timestamp)
                    else:
                        return (pipeline_id, None, None)
                except Exception:
                    return (pipeline_id, None, None)

            # Fetch in parallel with 10 workers
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(fetch_pipeline_jobs, pid) for pid in pipeline_ids_to_fetch]

                # Collect results as they complete
                for future in futures:
                    try:
                        pipeline_id, counts, timestamp = future.result()
                        result[pipeline_id] = counts
                        if counts is not None and timestamp is not None:
                            cache[pipeline_id] = {
                                'counts': counts,
                                'fetched_at': timestamp
                            }
                        else:
                            cache[pipeline_id] = None
                    except Exception:
                        pass

            # Save updated cache
            try:
                jobs_cache_path.parent.mkdir(parents=True, exist_ok=True)
                # Convert int keys to string for JSON
                cache_str_keys = {str(k): v for k, v in cache.items()}
                jobs_cache_path.write_text(json.dumps(cache_str_keys, indent=2))
            except Exception:
                pass

        return result

    def get_cached_pipeline_job_details(
        self,
        pipeline_ids: List[int],
        cache_file: str = ".gitlab_pipeline_jobs_details_cache.json",
        skip_fetch: bool = False,
    ) -> Dict[int, Optional[Dict[str, Any]]]:
        """Get GitLab CI pipeline job details (counts + job list) with intelligent caching.

        This is a richer variant of `get_cached_pipeline_job_counts` that also returns a
        slim per-job list for UI tooltips.

        Caching strategy:
        - If skip_fetch=True: Only return cached data, no API calls
        - If skip_fetch=False:
          - Completed pipelines (running=0, pending=0): Cached forever, never refetched
          - Active pipelines (running>0 or pending>0): Refetch if older than 30 minutes

        Args:
            pipeline_ids: List of pipeline IDs
            cache_file: Path to cache file
            skip_fetch: If True, only return cached data without fetching from GitLab

        Returns:
            Dictionary mapping pipeline ID -> details dict (or None if unavailable)

            Example return value:
            {
                40118215: {
                    "counts": {"success": 15, "failed": 8, "running": 0, "pending": 0},
                    "jobs": [
                        {"stage": "build", "name": "build-dynamo-image-amd64", "status": "success"},
                        {"stage": "test", "name": "pre-merge-vllm", "status": "failed"},
                    ],
                    "fetched_at": "2025-12-18T02:44:20.118368Z"
                },
                40118216: None
            }
        """
        # Load cache
        cache: Dict[int, Any] = {}
        jobs_cache_path = resolve_cache_path(cache_file)
        if jobs_cache_path.exists():
            try:
                raw = json.loads(jobs_cache_path.read_text())
                # Convert string keys back to int
                cache = {int(k): v for k, v in raw.items()}
            except Exception:
                cache = {}

        def normalize_entry(entry: Any) -> Optional[Dict[str, Any]]:
            """Normalize cache entry to {counts, jobs, fetched_at} (best-effort)."""
            if not entry:
                return None
            if not isinstance(entry, dict):
                return None
            # New format
            if "counts" in entry or "jobs" in entry or "fetched_at" in entry:
                counts = entry.get("counts") if isinstance(entry.get("counts"), dict) else None
                jobs = entry.get("jobs") if isinstance(entry.get("jobs"), list) else []
                fetched_at = entry.get("fetched_at")
                # Ensure expected keys exist even if older cache entries are missing fields.
                base_counts = {"success": 0, "failed": 0, "running": 0, "pending": 0, "canceled": 0}
                if counts:
                    for k, v in base_counts.items():
                        counts.setdefault(k, v)
                return {
                    "counts": counts or base_counts,
                    "jobs": jobs,
                    "fetched_at": fetched_at,
                }
            # Old format fallback: counts-only dict
            if all(k in entry for k in ("success", "failed", "running", "pending")):
                entry.setdefault("canceled", 0)
                return {"counts": entry, "jobs": [], "fetched_at": None}
            return None

        def is_completed(entry_norm: Optional[Dict[str, Any]]) -> bool:
            if not entry_norm:
                return False
            counts = entry_norm.get("counts") or {}
            return counts.get("running", 0) == 0 and counts.get("pending", 0) == 0

        def is_fresh(entry_norm: Optional[Dict[str, Any]]) -> bool:
            if not entry_norm:
                return False
            # Completed pipelines are cached forever.
            #
            # Active pipelines (running/pending > 0) should be refetched aggressively to avoid
            # showing stale status. We intentionally treat them as *not fresh* regardless of age.
            if is_completed(entry_norm):
                return True
            return False

        # NOTE: We intentionally refetch active pipelines on every run, so we no longer
        # use a time-based "freshness" threshold here.

        # skip_fetch => return cached values only
        if skip_fetch:
            return {pid: normalize_entry(cache.get(pid)) for pid in pipeline_ids}

        pipeline_ids_to_fetch: List[int] = []
        result: Dict[int, Optional[Dict[str, Any]]] = {}

        for pipeline_id in pipeline_ids:
            cached_entry_norm = normalize_entry(cache.get(pipeline_id))
            if cached_entry_norm and is_fresh(cached_entry_norm):
                result[pipeline_id] = cached_entry_norm
            else:
                pipeline_ids_to_fetch.append(pipeline_id)
                result[pipeline_id] = cached_entry_norm

        if pipeline_ids_to_fetch and self.has_token():
            fetch_timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

            def fetch_pipeline_jobs_details(pipeline_id: int) -> Tuple[int, Optional[Dict[str, Any]]]:
                try:
                    endpoint = f"/api/v4/projects/169905/pipelines/{pipeline_id}/jobs"
                    params = {"per_page": 100}
                    jobs = self.get(endpoint, params=params)
                    if not jobs:
                        return pipeline_id, None

                    counts = {"success": 0, "failed": 0, "running": 0, "pending": 0, "canceled": 0}
                    slim_jobs: List[Dict[str, Any]] = []

                    for job in jobs:
                        status = job.get("status", "unknown")
                        name = job.get("name", "")
                        stage = job.get("stage", "")

                        # Counts (map GitLab statuses to our buckets)
                        if status in counts:
                            counts[status] += 1
                        elif status in ("created", "waiting_for_resource"):
                            counts["pending"] += 1
                        elif status in ("skipped", "manual"):
                            pass
                        else:
                            # Keep unknown statuses out of counts
                            pass

                        # Tooltip list (keep it light)
                        if name:
                            slim_jobs.append({"name": name, "stage": stage, "status": status})

                    details = {"counts": counts, "jobs": slim_jobs, "fetched_at": fetch_timestamp}
                    return pipeline_id, details
                except Exception:
                    return pipeline_id, None

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(fetch_pipeline_jobs_details, pid) for pid in pipeline_ids_to_fetch]
                for future in futures:
                    try:
                        pid, details = future.result()
                        result[pid] = details
                        cache[pid] = details
                    except Exception:
                        pass

            # Save updated cache
            try:
                jobs_cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_str_keys = {str(k): v for k, v in cache.items()}
                jobs_cache_path.write_text(json.dumps(cache_str_keys, indent=2))
            except Exception:
                pass

        return result

    def get_cached_merge_request_pipelines(
        self,
        mr_numbers: List[int],
        project_id: str = "169905",
        cache_file: str = ".gitlab_mr_pipelines_cache.json",
        skip_fetch: bool = False,
    ) -> Dict[int, Optional[Dict[str, Any]]]:
        """Get most recent pipeline for each Merge Request (MR IID) with caching.

        This helps link a PR/MR to a pipeline even when the final merge commit SHA
        doesn't have a pipeline (e.g. pipeline is created for merge_request_event only).

        Args:
            mr_numbers: List of MR IIDs (internal IDs)
            project_id: GitLab project ID (default: 169905 for dl/ai-dynamo/dynamo)
            cache_file: Cache file name under the dynamo-utils cache dir
            skip_fetch: If True, only return cached data (no API calls)

        Returns:
            Mapping MR IID -> pipeline dict (id, status, web_url, sha, ref), or None.

        Cache format:
            {
              "5063": {"id": 40743226, "status": "success", "web_url": "...", "sha": "...", "ref": "..."},
              "5064": null
            }
        """
        # Load cache
        cache: Dict[int, Optional[Dict[str, Any]]] = {}
        cache_path = resolve_cache_path(cache_file)
        if cache_path.exists():
            try:
                raw = json.loads(cache_path.read_text())
                cache = {int(k): v for k, v in raw.items()}
            except Exception:
                cache = {}

        if skip_fetch:
            return {mr: cache.get(mr) for mr in mr_numbers}

        result: Dict[int, Optional[Dict[str, Any]]] = {}
        cache_updated = False

        # Determine which MRs to fetch (only if missing from cache or cached None).
        to_fetch = [mr for mr in mr_numbers if mr not in cache]
        for mr in mr_numbers:
            if mr in cache:
                result[mr] = cache[mr]
            else:
                result[mr] = None

        if to_fetch and self.has_token():
            logger = logging.getLogger("common")

            def fetch_one(mr_iid: int) -> Tuple[int, Optional[Dict[str, Any]]]:
                try:
                    endpoint = f"/api/v4/projects/{project_id}/merge_requests/{mr_iid}/pipelines"
                    pipelines = self.get(endpoint, params={"per_page": 1, "order_by": "id", "sort": "desc"}, timeout=10)
                    if isinstance(pipelines, list) and pipelines:
                        p = pipelines[0]
                        if isinstance(p, dict):
                            return mr_iid, {
                                "id": p.get("id"),
                                "status": p.get("status", "unknown"),
                                "web_url": p.get("web_url", ""),
                                "sha": p.get("sha", ""),
                                "ref": p.get("ref", ""),
                            }
                    return mr_iid, None
                except Exception as e:
                    logger.debug(f"Failed to fetch MR {mr_iid} pipelines: {e}")
                    return mr_iid, None

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(fetch_one, mr) for mr in to_fetch]
                for fut in futures:
                    try:
                        mr_iid, p = fut.result()
                        result[mr_iid] = p
                        cache[mr_iid] = p
                        cache_updated = True
                    except Exception:
                        pass

        # Save updated cache
        if cache_updated:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_str = {str(k): v for k, v in cache.items()}
                cache_path.write_text(json.dumps(cache_str, indent=2))
            except Exception:
                pass

        return result

    @staticmethod
    def parse_mr_number_from_message(message: str) -> Optional[int]:
        """Parse MR/PR number from commit message (e.g., '... (#1234)')

        Args:
            message: Commit message to parse

        Returns:
            MR number if found, None otherwise

        Example:
            >>> GitLabAPIClient.parse_mr_number_from_message("feat: Add feature (#1234)")
            1234
            >>> GitLabAPIClient.parse_mr_number_from_message("fix: Bug fix")
            None
        """
        match = re.search(r'#(\d+)', message)
        if match:
            return int(match.group(1))
        return None

    def get_cached_mr_merge_dates(self, mr_numbers: List[int],
                                  project_id: str = "169905",
                                  cache_file: str = '.gitlab_mr_merge_dates_cache.json',
                                  skip_fetch: bool = False) -> Dict[int, Optional[str]]:
        """Get merge dates for merge requests with caching.

        Merge dates are cached permanently since they don't change once a MR is merged.

        Args:
            mr_numbers: List of MR IIDs (internal IDs)
            project_id: GitLab project ID (default: 169905 for dynamo)
            cache_file: Path to cache file
            skip_fetch: If True, only return cached data without fetching from GitLab

        Returns:
            Dictionary mapping MR number to merge date string (YYYY-MM-DD HH:MM:SS)
            Returns None for MRs that are not merged or not found

        Example:
            >>> client = GitLabAPIClient()
            >>> merge_dates = client.get_cached_mr_merge_dates([4965, 5009])
            >>> merge_dates
            {4965: "2025-12-18 12:34:56", 5009: None}

        Cache file format (.gitlab_mr_merge_dates_cache.json):
        {
            "4965": "2025-12-18 12:34:56",
            "5009": null
        }
        """

        # Load cache
        cache = {}
        mr_cache_path = resolve_cache_path(cache_file)
        if mr_cache_path.exists():
            try:
                # Keys are stored as strings in JSON, convert back to int
                cache_raw = json.loads(mr_cache_path.read_text())
                cache = {int(k): v for k, v in cache_raw.items()}
            except Exception:
                pass

        # If skip_fetch=True, only return cached data
        if skip_fetch:
            return {mr_num: cache.get(mr_num) for mr_num in mr_numbers}

        # Prepare result and track if cache was updated
        result = {}
        cache_updated = False

        for mr_num in mr_numbers:
            # Check cache first
            if mr_num in cache:
                result[mr_num] = cache[mr_num]
                continue

            # Fetch from GitLab API
            try:
                endpoint = f"/api/v4/projects/{project_id}/merge_requests/{mr_num}"
                response = self.get(endpoint, timeout=5)

                if response and response.get('merged_at'):
                    # Parse ISO timestamp: "2025-12-18T12:34:56.000Z"
                    merged_at = response['merged_at']
                    dt = datetime.fromisoformat(merged_at.replace('Z', '+00:00'))
                    merge_date = dt.strftime('%Y-%m-%d %H:%M:%S')
                    result[mr_num] = merge_date
                    cache[mr_num] = merge_date
                    cache_updated = True
                else:
                    # MR not merged or not found
                    result[mr_num] = None
                    cache[mr_num] = None
                    cache_updated = True
            except Exception as e:
                # Log error but continue with other MRs
                logger = logging.getLogger('common')
                logger.debug(f"Failed to fetch MR {mr_num} merge date: {e}")
                result[mr_num] = None
                cache[mr_num] = None
                cache_updated = True

        # Save updated cache
        if cache_updated:
            try:
                mr_cache_path.parent.mkdir(parents=True, exist_ok=True)
                # Convert int keys to string for JSON
                cache_str_keys = {str(k): v for k, v in cache.items()}
                mr_cache_path.write_text(json.dumps(cache_str_keys, indent=2))
            except Exception:
                pass

        return result


