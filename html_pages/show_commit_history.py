#!/usr/bin/env python3
"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Dynamo Commit History Generator - Standalone Tool

Generates commit history with Image SHA (hash of container/ contents; previously shown as CDS) and Docker image detection.
HTML-only dashboard generator.
"""

import argparse
try:
    import git  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    git = None  # type: ignore[assignment]
import glob
import html
import json
import logging
import os
import re
import subprocess
import sys
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple, Any
from zoneinfo import ZoneInfo

# Ensure we can import sibling utilities (common.py) from the parent dynamo-utils directory
_THIS_DIR = Path(__file__).resolve().parent
_UTILS_DIR = _THIS_DIR.parent
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Shared dashboard helpers (UI + workflow graph)
from common_dashboard_lib import (
    CIStatus,
    PASS_PLUS_STYLE,
    TreeNodeVM,
    build_and_test_dynamo_phases_from_actions_job,
    check_line_html,
    ci_should_expand_by_default,
    ci_status_icon_context,
    compact_ci_summary_html,
    disambiguate_check_run_name,
    extract_actions_job_id_from_url,
    render_tree_divs,
    required_badge_html,
    mandatory_badge_html,
    status_icon_html,
)

# Dashboard runtime (HTML-only) helpers
from common_dashboard_runtime import (
    atomic_write_text,
    materialize_job_raw_log_text_local_link,
    prune_dashboard_raw_logs,
    prune_partial_raw_log_caches,
)

# Log/snippet helpers (shared library: `dynamo-utils/ci_log_errors/`)
from ci_log_errors import extract_error_snippet_from_log_file

# Import utilities from common module
import common
from common import (
    DynamoRepositoryUtils,
    GitLabAPIClient,
    GitHubAPIClient,
    PhaseTimer,
    classify_ci_kind,
    format_gh_check_run_duration,
    dynamo_utils_cache_dir,
    MARKER_RUNNING,
    MARKER_PASSED,
    MARKER_FAILED,
    MARKER_KILLED,
    summarize_check_runs,
    select_shas_for_network_fetch,
    normalize_check_name,
    is_required_check_name,
)

# Import Jinja2 for HTML template rendering
try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False
    Environment = None  # type: ignore[assignment]
    FileSystemLoader = None  # type: ignore[assignment]
    select_autoescape = None  # type: ignore[assignment]

# Constants for build status values
STATUS_UNKNOWN = 'unknown'
STATUS_SUCCESS = 'success'
STATUS_FAILED = 'failed'
STATUS_BUILDING = 'building'

_normalize_check_name = normalize_check_name
_is_required_check_name = is_required_check_name

class CommitHistoryGenerator:
    """Generate commit history with Image SHA (hash of container/ contents) and Docker images"""

    def __init__(
        self,
        repo_path: Path,
        verbose: bool = False,
        debug: bool = False,
        skip_gitlab_fetch: bool = False,
        github_token: Optional[str] = None,
        allow_anonymous_github: bool = False,
        max_github_api_calls: int = 100,
    ):
        """
        Initialize the commit history generator

        Args:
            repo_path: Path to the Dynamo repository
            verbose: Enable verbose output (INFO level)
            debug: Enable debug output (DEBUG level)
            skip_gitlab_fetch: Skip fetching GitLab registry data, use cached data only
        """
        self.repo_path = Path(repo_path)
        self.verbose = verbose
        self.debug = debug
        self.skip_gitlab_fetch = skip_gitlab_fetch
        self.logger = self._setup_logger()
        # Cache files live in ~/.cache/dynamo-utils to avoid polluting the repo checkout
        self.cache_file = dynamo_utils_cache_dir() / "commit_history.json"
        # Persistent snippet cache (speeds up repeated runs by avoiding re-parsing the same raw logs).
        self.snippet_cache_file = dynamo_utils_cache_dir() / "commit_history_snippets.json"
        self._snippet_cache_dirty = False
        self._snippet_cache_data: Dict[str, object] = {}
        self._ci_log_errors_fingerprint: str = ""
        # Per-run performance counters (shown in Statistics).
        self._perf: Dict[str, float] = {}
        self._perf_i: Dict[str, int] = {}
        self.gitlab_client = GitLabAPIClient()  # Single instance for all GitLab operations
        require_auth = not bool(allow_anonymous_github)
        self.github_client = GitHubAPIClient(
            token=github_token,
            debug_rest=bool(debug),
            require_auth=require_auth,
            allow_anonymous_fallback=bool(allow_anonymous_github),
            max_rest_calls=int(max_github_api_calls),
        )  # Single instance for all GitHub operations
        # Timing breakdown for the most recent `show_commit_history()` run (best-effort).
        self._last_timings: Dict[str, float] = {}
        # GitHub network fetching is governed by cache TTLs in `common.py`.
        # We allow fetches when entries are missing or stale so the dashboard self-heals and
        # raw logs can be materialized whenever they are needed.

    def _ci_log_errors_fp(self) -> str:
        """Fingerprint the ci_log_errors implementation (used for snippet cache invalidation)."""
        fp = str(getattr(self, "_ci_log_errors_fingerprint", "") or "")
        if fp:
            return fp
        try:
            # Hash the core implementation files; this avoids false hits when snippet logic changes.
            base = _UTILS_DIR / "ci_log_errors"
            files = [
                base / "engine.py",
                base / "snippet.py",
                base / "render.py",
                base / "regexes.py",
            ]
            h = hashlib.sha1()
            for p in files:
                try:
                    h.update(p.read_bytes())
                except Exception:
                    # Still include filename so missing files change the fingerprint.
                    h.update(str(p).encode("utf-8", errors="ignore"))
                    h.update(b"\0")
            fp = h.hexdigest()[:12]
        except Exception:
            fp = "unknown"
        self._ci_log_errors_fingerprint = fp
        return fp

    def _acquire_cache_lock(self, timeout: float = 10.0) -> Optional[object]:
        """Acquire file lock for snippet cache. Returns lock file handle or None on failure."""
        import fcntl
        lock_file = self.snippet_cache_file.parent / ".snippet_cache.lock"
        try:
            lock_file.parent.mkdir(parents=True, exist_ok=True)
            fh = open(lock_file, "w")
            # Try to acquire exclusive lock with timeout
            start = time.monotonic()
            while time.monotonic() - start < timeout:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return fh
                except (IOError, OSError):
                    time.sleep(0.1)
            # Timeout
            fh.close()
            return None
        except Exception:
            return None

    def _release_cache_lock(self, lock_fh: Optional[object]) -> None:
        """Release file lock for snippet cache."""
        if lock_fh is None:
            return
        try:
            import fcntl
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            lock_fh.close()
        except Exception:
            pass

    def _load_snippet_cache(self) -> None:
        """Load persistent snippet cache with file locking (best-effort)."""
        if self._snippet_cache_data:
            return
        
        lock_fh = self._acquire_cache_lock(timeout=5.0)
        try:
            data: Dict[str, object] = {}
            try:
                if self.snippet_cache_file.exists():
                    data = json.loads(self.snippet_cache_file.read_text() or "{}")
            except Exception:
                data = {}
            if not isinstance(data, dict):
                data = {}
            # Invalidate on ci_log_errors changes.
            want_fp = self._ci_log_errors_fp()
            got_fp = str(data.get("ci_log_errors_fp", "") or "")
            if got_fp != want_fp:
                data = {"ci_log_errors_fp": want_fp, "items": {}}
                self._snippet_cache_dirty = True
            # Normalize structure.
            if not isinstance(data.get("items"), dict):
                data["items"] = {}
            self._snippet_cache_data = data
        finally:
            self._release_cache_lock(lock_fh)

    def _save_snippet_cache(self) -> None:
        """Persist snippet cache with file locking: lock, read-merge, write, unlock (best-effort)."""
        if not bool(self._snippet_cache_dirty):
            return
        
        lock_fh = self._acquire_cache_lock(timeout=10.0)
        if lock_fh is None:
            # Could not acquire lock, skip save (best-effort)
            return
        
        try:
            self.snippet_cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Read existing cache from disk
            disk_data: Dict[str, object] = {}
            try:
                if self.snippet_cache_file.exists():
                    disk_data = json.loads(self.snippet_cache_file.read_text() or "{}")
            except Exception:
                disk_data = {}
            if not isinstance(disk_data, dict):
                disk_data = {}
            if not isinstance(disk_data.get("items"), dict):
                disk_data["items"] = {}
            
            # Step 2: Merge in-memory data with disk data (in-memory wins for conflicts)
            mem_data = self._snippet_cache_data if isinstance(self._snippet_cache_data, dict) else {}
            mem_items = mem_data.get("items") if isinstance(mem_data, dict) else {}
            if not isinstance(mem_items, dict):
                mem_items = {}
            
            disk_items = disk_data.get("items")
            if not isinstance(disk_items, dict):
                disk_items = {}
            
            # Merge: disk items + memory items (memory overwrites disk)
            merged_items = {**disk_items, **mem_items}
            
            # Preserve ci_log_errors fingerprint from memory
            merged_data = {
                "ci_log_errors_fp": mem_data.get("ci_log_errors_fp", disk_data.get("ci_log_errors_fp", "")),
                "items": merged_items
            }
            
            # Step 3: Size cap - keep at most ~5000 entries (oldest by ts)
            if len(merged_items) > 5200:
                try:
                    pairs = []
                    for k, v in merged_items.items():
                        if not isinstance(v, dict):
                            continue
                        ts = int(v.get("ts", 0) or 0)
                        pairs.append((ts, k))
                    pairs.sort()
                    # Drop oldest to ~5000
                    for _ts, k in pairs[: max(0, len(pairs) - 5000)]:
                        try:
                            merged_items.pop(k, None)
                        except Exception:
                            pass
                except Exception:
                    pass
            
            # Step 4: Write merged data atomically
            atomic_write_text(self.snippet_cache_file, json.dumps(merged_data, indent=2), encoding="utf-8")
            self._snippet_cache_dirty = False
        except Exception:
            # Don't fail page generation on cache persistence issues.
            pass
        finally:
            self._release_cache_lock(lock_fh)

    def _snippet_cache_key_for_raw_log(self, raw_log_path: Path) -> str:
        # Use filename (job_id.log) to keep keys stable even if the dashboard path changes.
        return str(raw_log_path.name or raw_log_path)

    def _snippet_from_cached_raw_log(self, raw_log_path: Path) -> str:
        """Return an error snippet for a raw log file, using the persistent cache when possible."""
        t0 = time.monotonic()
        try:
            p = Path(raw_log_path)
            if not p.exists() or not p.is_file():
                return ""
        except Exception:
            return ""

        # Note: Cache is loaded once at the start of show_commit_history(), not per-call
        data = self._snippet_cache_data if isinstance(self._snippet_cache_data, dict) else {}
        items = data.get("items") if isinstance(data.get("items"), dict) else {}
        key = self._snippet_cache_key_for_raw_log(Path(raw_log_path))
        try:
            st = Path(raw_log_path).stat()
            mtime_ns = int(getattr(st, "st_mtime_ns", 0) or 0)
            size = int(getattr(st, "st_size", 0) or 0)
        except Exception:
            mtime_ns = 0
            size = 0

        # Debug: track cache lookups
        if not hasattr(self, '_snippet_cache_debug'):
            self._snippet_cache_debug = {"hits": [], "misses": [], "reasons": {}}
        
        ent = items.get(key) if isinstance(items, dict) else None
        if isinstance(ent, dict):
            try:
                cached_mtime = int(ent.get("mtime_ns", -1) or -1)
                cached_size = int(ent.get("size", -1) or -1)
                if cached_mtime == mtime_ns and cached_size == size:
                    sn = str(ent.get("snippet", "") or "")
                    if sn:
                        self._perf_i["snippet.cache.hit"] = int(self._perf_i.get("snippet.cache.hit", 0) or 0) + 1
                        self._perf["snippet.total_secs"] = float(self._perf.get("snippet.total_secs", 0.0) or 0.0) + max(0.0, time.monotonic() - t0)
                        self._snippet_cache_debug["hits"].append(key)
                        return sn
                else:
                    # Cache entry exists but is stale
                    reason = f"stale (cached_mtime={cached_mtime} != {mtime_ns} or cached_size={cached_size} != {size})"
                    self._snippet_cache_debug["misses"].append(key)
                    self._snippet_cache_debug["reasons"][key] = reason
            except Exception as e:
                self._snippet_cache_debug["misses"].append(key)
                self._snippet_cache_debug["reasons"][key] = f"exception: {e}"
        else:
            # No cache entry
            self._snippet_cache_debug["misses"].append(key)
            self._snippet_cache_debug["reasons"][key] = "no entry" if key not in items else "entry not dict"

        # Cache miss/stale: compute and store.
        self._perf_i["snippet.cache.miss"] = int(self._perf_i.get("snippet.cache.miss", 0) or 0) + 1
        t1 = time.monotonic()
        try:
            snippet = extract_error_snippet_from_log_file(Path(raw_log_path))
        except Exception:
            snippet = ""
        dt_compute = max(0.0, time.monotonic() - t1)
        self._perf["snippet.compute_secs"] = float(self._perf.get("snippet.compute_secs", 0.0) or 0.0) + float(dt_compute)
        try:
            if isinstance(items, dict):
                items[key] = {
                    "mtime_ns": mtime_ns,
                    "size": size,
                    "ts": int(time.time()),
                    "snippet": str(snippet or ""),
                }
                data["items"] = items
                self._snippet_cache_data = data
                self._snippet_cache_dirty = True
        except Exception:
            pass
        self._perf["snippet.total_s"] = float(self._perf.get("snippet.total_s", 0.0) or 0.0) + max(0.0, time.monotonic() - t0)
        return str(snippet or "")

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('CommitHistoryGenerator')
        if self.debug:
            logger.setLevel(logging.DEBUG)
        elif self.verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

        handler = logging.StreamHandler()
        if self.debug:
            handler.setLevel(logging.DEBUG)
        elif self.verbose:
            handler.setLevel(logging.INFO)
        else:
            handler.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(handler)
        
        # Also configure the common module logger
        import common
        common_logger = logging.getLogger('common')
        if self.debug:
            common_logger.setLevel(logging.DEBUG)
        elif self.verbose:
            common_logger.setLevel(logging.INFO)
        else:
            common_logger.setLevel(logging.WARNING)
        if not common_logger.handlers:
            common_logger.addHandler(handler)

        # GitHub REST debug logging (GitHubAPIClient uses logger name == class name).
        gh_logger = logging.getLogger('GitHubAPIClient')
        if self.debug:
            gh_logger.setLevel(logging.DEBUG)
        elif self.verbose:
            gh_logger.setLevel(logging.INFO)
        else:
            gh_logger.setLevel(logging.WARNING)
        if not gh_logger.handlers:
            gh_logger.addHandler(handler)
        
        return logger

    def show_commit_history(
        self,
        max_commits: int = 50,
        output_path: Optional[Path] = None,
        logs_dir: Optional[Path] = None,
        export_pipeline_pr_csv: Optional[Path] = None,
    ) -> int:
        """Show recent commit history with Image SHA (hash of container/ contents; previously shown as CDS)

        Args:
            max_commits: Maximum number of commits to show
            output_path: Path for HTML output file (optional, auto-detected if not provided)
            logs_dir: Path to logs directory for build reports (optional, defaults to repo_path/logs)

        Returns:
            Exit code (0 for success, 1 for failure)
        """
        generation_t0 = time.monotonic()
        phase_t = PhaseTimer()

        # Initialize repo utils for this operation
        repo_utils = DynamoRepositoryUtils(self.repo_path, dry_run=False, verbose=self.verbose)

        # Load snippet cache once at start (speeds up repeated snippet extraction)
        self._load_snippet_cache()

        # Cache stats (printed at end in verbose/debug mode; avoids needing to pipe output).
        cache_stats: Dict[str, dict] = {}
        meta_stats: Dict[str, int] = {"full_hit": 0, "miss": 0}

        # Load cache (commit_history.json in dynamo-utils cache dir)
        # Format: {
        #   "<full_commit_sha>": {
        #     "composite_docker_sha": "746bc31d05b3",  # image SHA (hash of container/ contents; formerly shown as CDS)
        #     "author": "John Doe",
        #     "date": "2025-12-17 20:03:39",
        #     "merge_date": "2025-12-17 21:15:30",
        #     "message": "feat: Add new feature (#1234)",
        #     "full_message": "feat: Add new feature (#1234)\n\nDetailed description...",
        #     "stats": {"files": 5, "insertions": 100, "deletions": 50},
        #     "changed_files": ["path/to/file1.py", "path/to/file2.py"]
        #   },
        #   ...
        # }
        # Fields:
        #   - composite_docker_sha: 12-character image SHA (hash of container/ directory contents)
        #   - author: Commit author name
        #   - date: Commit timestamp (YYYY-MM-DD HH:MM:SS)
        #   - merge_date: When MR was merged (YYYY-MM-DD HH:MM:SS), null if not found/merged
        #   - message: First line of commit message
        #   - full_message: Full commit message
        #   - stats: Dict with files, insertions, deletions counts
        #   - changed_files: List of file paths changed in this commit
        cache = {}
        t0 = phase_t.start()
        if self.cache_file.exists():
            try:
                cache = json.loads(self.cache_file.read_text())
                self.logger.debug(f"Loaded cache with {len(cache)} entries")
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
        phase_t.stop("cache_load", t0)

        repo = None
        # IMPORTANT: restore to the original ref (branch name when possible).
        # Restoring to a SHA (repo.head.commit.hexsha) leaves the repo in detached HEAD.
        original_ref = None
        try:
            if git is None:
                raise ImportError("GitPython is required. Install with: pip install gitpython")
            repo = git.Repo(self.repo_path)
            t0 = phase_t.start()
            commits = list(repo.iter_commits('HEAD', max_count=max_commits))
            phase_t.stop("git_iter_commits", t0)
            # Record the original ref so we can restore it at the end without detaching.
            # - If we started on a branch, restore that branch name (e.g. "main")
            # - If we started detached, best-effort restore the original SHA
            try:
                if repo.head.is_detached:
                    original_ref = repo.head.commit.hexsha
                else:
                    original_ref = repo.active_branch.name
            except Exception:
                # Fallback: SHA (may restore detached)
                original_ref = repo.head.commit.hexsha

            _INLINE_TRAILER_RE = re.compile(
                r"\\s+(signed-off-by|co-authored-by|reviewed-by|tested-by|acked-by|suggested-by|fixes)\\s*:",
                flags=re.IGNORECASE,
            )

            def _clean_subject_line(s: str) -> str:
                """Clean commit subject lines for display.

                Some commits embed trailers like `Signed-off-by:` / `Co-authored-by:` on the *same*
                line as the subject. For the dashboard we want the human subject only.
                """
                # First, take ONLY the first line (Git subject). Some repos/users put trailers on
                # subsequent lines, and we never want those in the subject display.
                line = str(s or "").splitlines()[0].strip() if str(s or "") else ""
                if not line:
                    return ""
                m = _INLINE_TRAILER_RE.search(line)
                if m:
                    return line[: m.start()].rstrip()
                return line

            # Collect PR numbers for commits (cheap; used for HTML and for pipeline->PR exports).
            pr_to_merge_date: Dict[int, Optional[str]] = {}
            sha_to_pr_number: Dict[str, int] = {}
            pr_to_required_checks: Dict[int, List[str]] = {}
            pr_numbers: List[int] = []
            for commit in commits:
                message = _clean_subject_line(commit.message)
                pr_num = GitLabAPIClient.parse_mr_number_from_message(message)
                if pr_num:
                    pr_numbers.append(pr_num)
                    sha_to_pr_number[commit.hexsha] = pr_num

            cache_only_github = bool(getattr(self.github_client, "cache_only_mode", False))
            if pr_numbers and (not self.skip_gitlab_fetch) and (not cache_only_github):
                # Batch fetch merge dates for all PRs (GitHub)
                self.logger.info(f"Fetching merge dates for {len(pr_numbers)} PRs...")
                t0 = phase_t.start()
                pr_to_merge_date = self.github_client.get_cached_pr_merge_dates(
                    pr_numbers,
                    cache_file="github_pr_merge_dates.json",
                    stats=cache_stats,
                )
                phase_t.stop("github_merge_dates", t0)
                self.logger.info(
                    f"Got merge dates for {sum(1 for v in pr_to_merge_date.values() if v)} PRs"
                )

                if pr_numbers:
                    # Batch fetch required checks for all PRs (branch protection required checks).
                    # If skip_gitlab_fetch=True, we still read from cache (skip_fetch=True).
                    self.logger.info(f"Fetching required checks for {len(pr_numbers)} PRs...")
                t0 = phase_t.start()
                pr_to_required_checks = self.github_client.get_cached_required_checks(
                    pr_numbers,
                    owner="ai-dynamo",
                    repo="dynamo",
                    cache_file="github_required_checks.json",
                    skip_fetch=bool(self.skip_gitlab_fetch) or bool(cache_only_github),
                )
                phase_t.stop("github_required_checks", t0)
                self.logger.info(
                    "Got required-checks metadata for "
                    f"{sum(1 for v in pr_to_required_checks.values() if v)}/{len(set(pr_numbers))} PRs"
                )

            # Collect commit data
            commit_data = []
            cache_updated = False
            sha_to_message_first_line: Dict[str, str] = {}

            # HTML-only: no terminal formatting / printing
            try:
                t0 = phase_t.start()
                for i, commit in enumerate(commits):
                    sha_short = commit.hexsha[:9]
                    sha_full = commit.hexsha
                    t_sha_total = time.monotonic()

                    # Defaults
                    full_message = ""
                    files_changed = 0
                    insertions = 0
                    deletions = 0
                    changed_files: List[str] = []

                    cached_entry = cache.get(sha_full)
                    merge_date = None
                    commit_dt_pt = commit.committed_datetime.astimezone(ZoneInfo("America/Los_Angeles"))
                    date_epoch = int(commit_dt_pt.timestamp())
                    author_email = ""

                    if cached_entry and isinstance(cached_entry, dict):
                        meta_stats["full_hit"] += 1
                        self._perf_i["composite_sha.cache.hit"] = int(self._perf_i.get("composite_sha.cache.hit", 0) or 0) + 1
                        composite_sha = cached_entry['composite_docker_sha']
                        date_str = cached_entry['date']
                        author_name = cached_entry['author']
                        author_email = str(cached_entry.get("author_email") or "")
                        message_first_line = _clean_subject_line(cached_entry['message'])
                        merge_date = cached_entry.get('merge_date')
                        full_message = cached_entry['full_message']
                        files_changed = cached_entry['stats']['files']
                        insertions = cached_entry['stats']['insertions']
                        deletions = cached_entry['stats']['deletions']
                        changed_files = cached_entry['changed_files']

                        # Normalize cached subject line if needed.
                        try:
                            if str(cached_entry.get("message") or "") != str(message_first_line or "") and message_first_line:
                                cached_entry["message"] = message_first_line
                                cache_updated = True
                        except Exception:
                            pass

                        # Backfill author_email if missing.
                        if not author_email:
                            try:
                                author_email = str(getattr(commit.author, "email", "") or "")
                            except Exception:
                                author_email = ""
                            if author_email:
                                cached_entry["author_email"] = author_email
                                cache_updated = True

                        # Backfill merge_date if missing and available.
                        if merge_date is None:
                            pr_number = GitLabAPIClient.parse_mr_number_from_message(message_first_line)
                            if pr_number and pr_number in pr_to_merge_date:
                                merge_date = pr_to_merge_date[pr_number]
                                if merge_date:
                                    cached_entry['merge_date'] = merge_date
                                    cache_updated = True
                    else:
                        # Cache miss: compute from git
                        date_str = commit_dt_pt.strftime('%Y-%m-%d %H:%M:%S')
                        author_name = commit.author.name
                        try:
                            author_email = str(getattr(commit.author, "email", "") or "")
                        except Exception:
                            author_email = ""
                        message_first_line = _clean_subject_line(commit.message)
                        pr_number = GitLabAPIClient.parse_mr_number_from_message(message_first_line)
                        if pr_number and pr_number in pr_to_merge_date:
                            merge_date = pr_to_merge_date[pr_number]

                        meta_stats["miss"] += 1
                        self._perf_i["composite_sha.cache.miss"] = int(self._perf_i.get("composite_sha.cache.miss", 0) or 0) + 1
                        t_sha = time.monotonic()
                        try:
                            repo.git.checkout(commit.hexsha)
                            composite_sha = repo_utils.generate_composite_sha()
                        except Exception as e:
                            composite_sha = "ERROR"
                            self.logger.error(f"Failed to calculate composite SHA for {sha_short}: {e}")
                            self._perf_i["composite_sha.errors"] = int(self._perf_i.get("composite_sha.errors", 0) or 0) + 1
                        self._perf["composite_sha.compute_secs"] = float(self._perf.get("composite_sha.compute_secs", 0.0) or 0.0) + max(0.0, time.monotonic() - t_sha)

                        stats = commit.stats.total
                        files_changed = stats['files']
                        insertions = stats['insertions']
                        deletions = stats['deletions']
                        full_message = commit.message.strip()
                        changed_files = list(commit.stats.files.keys())

                        cache_entry = {
                            'composite_docker_sha': composite_sha,
                            'author': author_name,
                            'author_email': author_email,
                            'date': date_str,
                            'merge_date': merge_date,
                            'message': message_first_line,
                            'full_message': full_message,
                            'stats': {
                                'files': files_changed,
                                'insertions': insertions,
                                'deletions': deletions,
                            },
                            'changed_files': changed_files,
                        }
                        cache[sha_full] = cache_entry
                        cache_updated = True

                    # Total time spent obtaining composite SHA (hit + miss path).
                    self._perf["composite_sha.total_secs"] = float(self._perf.get("composite_sha.total_secs", 0.0) or 0.0) + max(
                        0.0, time.monotonic() - t_sha_total
                    )

                    # Always include this commit in the HTML, regardless of cache hit/miss.
                    commit_data.append(
                        {
                            "sha_short": sha_short,
                            "sha_full": sha_full,
                            "composite_sha": composite_sha,
                            "date": date_str,
                            "date_epoch": date_epoch,
                            "merge_date": merge_date,
                            "committed_datetime": commit.committed_datetime,
                            "author": author_name,
                            "author_email": author_email,
                            "message": message_first_line,
                            "full_message": full_message,
                            "files_changed": files_changed,
                            "insertions": insertions,
                            "deletions": deletions,
                            "changed_files": changed_files,
                        }
                    )
                    self.logger.debug(f"Processed commit {i+1}/{len(commits)}: {sha_short}")

                    sha_to_message_first_line[sha_full] = message_first_line

            finally:
                # Restore original ref (best-effort)
                if repo is not None and original_ref is not None:
                    try:
                        repo.git.checkout(original_ref)
                    except Exception:
                        pass

            # Optional: export pipeline -> PR mapping as CSV for this commit window.
            if export_pipeline_pr_csv:
                try:
                    sha_full_list = list(sha_to_message_first_line.keys())
                    gitlab_pipelines = self._get_gitlab_pipeline_statuses(sha_full_list)
                    export_pipeline_pr_csv.parent.mkdir(parents=True, exist_ok=True)
                    import csv

                    with export_pipeline_pr_csv.open("w", newline="") as f:
                        w = csv.DictWriter(
                            f,
                            fieldnames=[
                                "pipeline_id",
                                "pipeline_web_url",
                                "pipeline_status",
                                "sha",
                                "commit_web_url",
                                "pr_number",
                                "pr_web_url",
                                "commit_message",
                            ],
                        )
                        w.writeheader()
                        for sha in sha_full_list:
                            p = gitlab_pipelines.get(sha)
                            if not p:
                                continue
                            pipeline_id = p.get("id", "")
                            pipeline_web_url = p.get("web_url", "")
                            pipeline_status = p.get("status", "")
                            pr_number = sha_to_pr_number.get(sha)
                            pr_web_url = f"https://github.com/ai-dynamo/dynamo/pull/{pr_number}" if pr_number else ""
                            commit_web_url = f"https://github.com/ai-dynamo/dynamo/commit/{sha}"
                            w.writerow(
                                {
                                    "pipeline_id": pipeline_id,
                                    "pipeline_web_url": pipeline_web_url,
                                    "pipeline_status": pipeline_status,
                                    "sha": sha,
                                    "commit_web_url": commit_web_url,
                                    "pr_number": pr_number or "",
                                    "pr_web_url": pr_web_url,
                                    "commit_message": sha_to_message_first_line.get(sha, ""),
                                }
                            )
                except Exception as e:
                    self.logger.warning(f"Failed to export pipeline→PR CSV: {e}")

            phase_t.stop("process_commits", t0)

            # Generate HTML (HTML-only)
            if True:
                # Set default logs_dir if not provided
                if logs_dir is None:
                    # For dynamo_latest, default to ../dynamo_ci/logs for build logs
                    repo_abs_path = self.repo_path.resolve()
                    if repo_abs_path.name == "dynamo_latest":
                        logs_dir = repo_abs_path.parent / "dynamo_ci" / "logs"
                    else:
                        logs_dir = self.repo_path / "logs"
                # Determine output path first
                if output_path is None:
                    # Auto-detect output path.
                    #
                    # Policy: prefer writing to the web-served dashboard location:
                    #   <repo_path>/../dynamo_latest/index.html
                    # This keeps manual runs consistent with update_html_pages.sh.
                    #
                    # If that directory doesn't exist, fall back to <repo_path>/logs/commit-history.html
                    # (or ./commit-history.html as a last resort).
                    repo_abs_path = self.repo_path.resolve()
                    nvidia_home = repo_abs_path.parent
                    dynamo_latest_dir = nvidia_home / "dynamo_latest"
                    logs_dir_temp = self.repo_path / "logs"

                    if dynamo_latest_dir.exists():
                        output_path = dynamo_latest_dir / "index.html"
                    elif logs_dir_temp.exists():
                        output_path = logs_dir_temp / "commit-history.html"
                    else:
                        output_path = Path("commit-history.html")

                # Make the timing breakdown available to the HTML generator.
                # Note: render_html/write_output timings are measured outside, but the HTML generator
                # can still show the earlier phases (cache/git/process) via `_last_timings`.
                try:
                    self._last_timings = dict(phase_t.as_dict(include_total=True))
                except Exception:
                    self._last_timings = {}

                t0 = phase_t.start()
                html_content = self._generate_commit_history_html(
                    commit_data,
                    logs_dir,
                    output_path,
                    sha_to_pr_number=sha_to_pr_number,
                    pr_to_required_checks=pr_to_required_checks,
                    cache_stats=cache_stats,
                    generation_t0=generation_t0,
                )
                phase_t.stop("render_html", t0)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                t0 = phase_t.start()
                atomic_write_text(output_path, html_content, encoding="utf-8")
                phase_t.stop("write_output", t0)
                # Cache miss summary
                if self.verbose or self.debug:
                    try:
                        self.logger.info(
                            "Cache stats: commit_metadata full_hit=%s miss=%s",
                            meta_stats.get("full_hit", 0),
                            meta_stats.get("miss", 0),
                        )
                        md = cache_stats.get("merge_dates_cache") or {}
                        if md:
                            self.logger.info(
                                "Cache stats: merge_dates hits=%s misses=%s reason=%s sample_misses=%s",
                                md.get("hits"),
                                md.get("misses"),
                                md.get("miss_reason"),
                                md.get("miss_prs_sample"),
                            )
                        rc = cache_stats.get("required_checks_cache") or {}
                        if rc:
                            self.logger.info(
                                "Cache stats: required_checks hits=%s misses=%s reason=%s skip_fetch=%s sample_misses=%s",
                                rc.get("hits"),
                                rc.get("misses"),
                                rc.get("miss_reason"),
                                rc.get("skip_fetch"),
                                rc.get("miss_prs_sample"),
                            )
                        gha = cache_stats.get("github_actions_status_cache") or {}
                        if gha:
                            self.logger.info(
                                "Cache stats: github_actions_status total=%s fetched=%s hit_fresh=%s stale_refresh=%s miss_fetch=%s miss_no_fetch=%s",
                                gha.get("total_shas"),
                                gha.get("fetched_shas"),
                                gha.get("cache_hit_fresh"),
                                gha.get("cache_stale_refresh"),
                                gha.get("cache_miss_fetch"),
                                gha.get("cache_miss_no_fetch"),
                            )
                            if self.debug:
                                self.logger.debug("Cache stats: github_actions_status details=%s", gha)
                    except Exception:
                        pass

            # Save cache if updated
            if cache_updated:
                t0 = phase_t.start()
                try:
                    self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                    # Atomic write: avoid partial/corrupted JSON which can cause "cache misses" on next run.
                    atomic_write_text(self.cache_file, json.dumps(cache, indent=2), encoding="utf-8")
                    self.logger.debug(f"Cache saved with {len(cache)} entries")
                except Exception as e:
                    self.logger.warning(f"Failed to save cache: {e}")
                phase_t.stop("cache_save", t0)

            # Persist snippet cache (best-effort).
            try:
                self._save_snippet_cache()
            except Exception:
                pass

            # Persist timings for HTML page stats / debugging (best-effort).
            try:
                self._last_timings = dict(phase_t.as_dict(include_total=True))
            except Exception:
                self._last_timings = {}

            return 0
        except KeyboardInterrupt:
            self.logger.warning("Operation interrupted by user")
            # Try to restore HEAD
            try:
                if repo is not None and original_ref is not None:
                    repo.git.checkout(original_ref)
            except:
                pass
            return 1
        except Exception as e:
            self.logger.error(f"Failed to get commit history: {e}")
            return 1

    def _generate_commit_history_html(
        self,
        commit_data: List[dict],
        logs_dir: Path,
        output_path: Path,
        sha_to_pr_number: Optional[Dict[str, int]] = None,
        pr_to_required_checks: Optional[Dict[int, List[str]]] = None,
        cache_stats: Optional[Dict[str, dict]] = None,
        generation_t0: Optional[float] = None,
    ) -> str:
        """Generate HTML report for commit history with Docker image detection

        Args:
            commit_data: List of commit dictionaries with sha_short, sha_full, composite_sha, date, author, message
            logs_dir: Path to logs directory for build reports
            output_path: Path where the HTML file will be written (used for relative path calculation)
            sha_to_pr_number: Mapping full commit SHA -> PR number (if known)
            pr_to_required_checks: Mapping PR number -> list of required check names (if known)

        Returns:
            HTML content as string
        """
        if not HAS_JINJA2:
            raise ImportError("jinja2 is required for HTML generation. Install with: pip install jinja2")
        # Make type checkers happy: these are only None when HAS_JINJA2 is False.
        assert Environment is not None and FileSystemLoader is not None and select_autoescape is not None
        
        # Get local Docker images containing SHAs
        docker_images = self._get_local_docker_images_by_sha([c['sha_short'] for c in commit_data])

        # Get GitLab container registry images for commits (with caching)
        gitlab_images_raw = self._get_cached_gitlab_images_from_sha(commit_data)

        # Get GitLab CI pipeline statuses
        gitlab_pipelines = self._get_gitlab_pipeline_statuses([c['sha_full'] for c in commit_data])

        # Get GitLab MR pipelines for PR-linked commits (fallback when SHA has no pipeline)
        sha_to_pr_number = sha_to_pr_number or {}
        pr_numbers = sorted({v for v in sha_to_pr_number.values() if v})
        mr_pipelines: Dict[int, Optional[dict]] = {}
        if pr_numbers:
            try:
                mr_pipelines = self.gitlab_client.get_cached_merge_request_pipelines(
                    pr_numbers,
                    project_id="169905",
                    cache_file="gitlab_mr_pipelines.json",
                    skip_fetch=self.skip_gitlab_fetch,
                )
            except Exception:
                mr_pipelines = {}

        # Get GitLab CI pipeline job counts
        pipeline_ids = [p['id'] for p in gitlab_pipelines.values() if p and 'id' in p]
        pipeline_job_counts = {}
        if pipeline_ids:
            pipeline_job_counts = self._get_gitlab_pipeline_job_counts(pipeline_ids)

        # Prime required-checks metadata (GitHub) in a capped way:
        # - always read cache for all PRs (no API calls)
        # - only allow network fetch for PRs associated with allow_fetch_shas
        try:
            pr_to_required_checks = self.github_client.get_cached_required_checks(
                pr_numbers,
                cache_file="github_required_checks.json",
                skip_fetch=True,
                stats=cache_stats,
            )
        except Exception:
            pr_to_required_checks = {}

        try:
            pr_numbers_allow = sorted({p for p in (sha_to_pr_number.get(sha) for sha in sha_full_list) if p})
            if pr_numbers_allow:
                fetched_required = self.github_client.get_cached_required_checks(
                    pr_numbers_allow,
                    cache_file="github_required_checks.json",
                    skip_fetch=False,
                    stats=cache_stats,
                )
                pr_to_required_checks.update(fetched_required or {})
        except Exception:
            pass

        # Get GitHub Actions check status for commits:
        # - allow network fetch for any SHA that is cache-missing/stale (TTL policy lives in `common.py`).
        #
        # NOTE: We do *not* age-gate fetch here anymore. The cache TTL policy already ensures
        # old commits refresh very rarely (DEFAULT_STABLE_TTL_S), but we still need to fetch
        # at least once to populate the cache; otherwise some SHAs show no GitHub dropdown.
        sha_full_list = [c['sha_full'] for c in commit_data]
        sha_to_dt = {c['sha_full']: c.get('committed_datetime') for c in commit_data}

        # Raw-log policy:
        # - We cache raw log *content* on disk for later parsing.
        # - The dashboard links `[raw log]` to a *repo-local* stable file under:
        # Raw logs are stored under the global cache dir (~/.cache/dynamo-utils/raw-log-text),
        # and dashboards link to them under the page root via `raw-log-text/<job_id>.log`.
        #   (never to ephemeral GitHub signed URLs).
        # - We allow network fetch of missing raw logs for any SHA; `common.py` enforces:
        #   - only cache when job status is completed
        #   - size caps and other safeguards

        cache_only_github = bool(getattr(self.github_client, "cache_only_mode", False))
        github_actions_status = self.github_client.get_github_actions_status(
            owner='ai-dynamo',
            repo='dynamo',
            sha_list=sha_full_list,
            cache_file="github_actions_status.json",
            # Do NOT tie GitHub fetch behavior to --skip-gitlab-fetch.
            # GitHub fetching is already capped via fetch_allowlist + TTL policy.
            skip_fetch=bool(cache_only_github),
            fetch_allowlist=set(sha_full_list),
            sha_to_datetime=sha_to_dt,          # age-based cache policy (>8h => stable TTL)
            stats=cache_stats,
        )

        # Annotate GitHub check runs with "is_required" using PR required-checks + fallback patterns.
        pr_to_required_checks = pr_to_required_checks or {}
        
        # Build a mapping of job names to short names from YAML (shared across all commits)
        from common_dashboard_lib import parse_workflow_yaml_and_build_mapping_pass
        try:
            yaml_mappings = parse_workflow_yaml_and_build_mapping_pass(
                repo_root=Path(self.repo_path),
                commit_sha="HEAD",  # Use HEAD as a representative commit for YAML parsing
            )
            job_name_to_short = yaml_mappings.get('job_name_to_id', {})
        except Exception:
            job_name_to_short = {}
        
        for sha_full, gha in (github_actions_status or {}).items():
            if not gha or not gha.get('check_runs'):
                continue
            pr_number = sha_to_pr_number.get(sha_full)
            required_list = pr_to_required_checks.get(pr_number, []) if pr_number else []
            required_norm = {_normalize_check_name(n) for n in (required_list or []) if n}
            # Note: Sorting is now handled by PASS 4 (sort_by_name_pass) in the centralized pipeline.
            for check in (gha.get('check_runs', []) or []):
                try:
                    check_name = check.get('name', '')
                    check['is_required'] = _is_required_check_name(check_name, required_norm)
                    # Add short name for tooltip display
                    check['short_name'] = job_name_to_short.get(check_name, check_name)
                except Exception:
                    # Best-effort; never break page generation.
                    check['is_required'] = False
                    check['short_name'] = check.get('name', '')

        # Process GitLab images: deduplicate and format
        gitlab_images = {}
        for sha_full, registry_imgs in gitlab_images_raw.items():
            if not registry_imgs:
                continue
                
            # Deduplicate: keep only the most recent image per framework/arch combination
            deduped_imgs = {}
            for img in registry_imgs:
                key = (img['framework'], img['arch'])
                if key not in deduped_imgs or img['pipeline_id'] > deduped_imgs[key]['pipeline_id']:
                    deduped_imgs[key] = img
            
            # Format the images for display
            # Sort by arch first (all x86 together, then all ARM), then by framework
            formatted_imgs = []
            def sort_key(item):
                framework, arch = item[0]
                # amd64 comes before arm64: 0 for amd64, 1 for arm64
                arch_order = 0 if arch == 'amd64' else 1
                return (arch_order, framework)

            for (framework, arch), img in sorted(deduped_imgs.items(), key=sort_key):
                total_size = img.get('total_size', 0)
                created_at = img.get('created_at', '')
                
                # Format size (bytes to human-readable)
                if total_size > 0:
                    if total_size >= 1_000_000_000:  # GB
                        size_display = f"{total_size / 1_000_000_000:.2f}GB"
                    elif total_size >= 1_000_000:  # MB
                        size_display = f"{total_size / 1_000_000:.1f}MB"
                    else:
                        size_display = f"{total_size / 1_000:.1f}KB"
                else:
                    size_display = "N/A"
                
                # Format created timestamp (ISO 8601 to readable, convert to PT)
                created_display = "N/A"
                if created_at:
                    try:
                        # Parse UTC timestamp
                        dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        dt = dt.astimezone(ZoneInfo('America/Los_Angeles'))
                        created_display = dt.strftime('%Y-%m-%d %H:%M')
                    except:
                        created_display = created_at[:16] if len(created_at) >= 16 else created_at
                
                formatted_imgs.append({
                    **img,
                    'size_display': size_display,
                    'created_display': created_display
                })
            
            gitlab_images[sha_full] = formatted_imgs
        
        # Process commit messages for PR links and prepare Image-SHA alternation state.
        #
        # UX: in the dashboard, the Image SHA badge background alternates per *unique image SHA* so it's easy
        # to visually scan for groups. The actual hue (green/red/grey) is assigned later, after we
        # compute local build status (PASS/FAIL/BUILD/UNKNOWN).
        unique_cds: List[str] = []
        seen_cds: set[str] = set()
        for commit in (commit_data or []):
            cds = str(commit.get("composite_sha", "") or "")
            if cds and cds not in seen_cds:
                unique_cds.append(cds)
                seen_cds.add(cds)

        cds_to_parity: Dict[str, int] = {cds: (i % 2) for i, cds in enumerate(unique_cds)}

        for commit in commit_data:
            # Handle PR links
            message = commit['message']
            pr_match = re.search(r'\(#(\d+)\)', message)
            if pr_match:
                pr_number = pr_match.group(1)
                pr_link = f"https://github.com/ai-dynamo/dynamo/pull/{pr_number}"
                message = re.sub(
                    r'\(#(\d+)\)',
                    f'(<a href="{pr_link}" class="pr-link" target="_blank">#{pr_number}</a>)',
                    message
                )
                commit['message'] = message
                try:
                    commit['pr_number'] = int(pr_number)
                except Exception:
                    commit['pr_number'] = None
            else:
                commit['pr_number'] = sha_to_pr_number.get(commit.get("sha_full", ""))

            composite_sha = str(commit.get("composite_sha", "") or "")
            commit["cds_parity"] = int(cds_to_parity.get(composite_sha, 0))
            # Filled in later after local build status is computed.
            commit["composite_bg_color"] = "#e5e7eb"
            commit["composite_text_color"] = "#111827"

        # Compute fork-points for the last 5 release branches and annotate matching commits.
        # This helps identify "cut points" for release lines (e.g., release/v0.8.0).
        try:
            repo_utils = DynamoRepositoryUtils(self.repo_path, dry_run=False, verbose=self.verbose)
            fork_map = repo_utils.get_release_branch_fork_points(limit=5)
            for commit in commit_data:
                sha_full = commit['sha_full']
                releases = fork_map.get(sha_full, [])
                if releases:
                    commit['release_forkpoints'] = releases
                else:
                    commit['release_forkpoints'] = []
        except Exception as e:
            self.logger.warning(f"Failed to compute release fork points: {e}")
            for commit in commit_data:
                commit['release_forkpoints'] = []
        
        # Build log paths dictionary and status indicators
        t_markers = time.monotonic()
        log_paths = {}  # Maps sha_short to list of (date, path) tuples
        composite_to_status = {}  # Maps composite_sha to status (with priority: building > failed > success)
        commit_to_status = {}  # Maps commit sha_short to its own build status
        composite_to_commits = {}  # Maps composite_sha to list of commit SHAs

        # Status priority for conflict resolution (higher number = higher priority)
        # Building > Failed > Success (if any build is still running, show as building)
        status_priority = {STATUS_UNKNOWN: 0, STATUS_SUCCESS: 1, STATUS_FAILED: 2, STATUS_BUILDING: 3}

        # First pass: Group commits by composite_sha to minimize glob searches
        for commit in commit_data:
            sha_short = commit['sha_short']
            composite_sha = commit['composite_sha']

            # Track which commits have this composite SHA
            if composite_sha not in composite_to_commits:
                composite_to_commits[composite_sha] = []
            composite_to_commits[composite_sha].append(sha_short)

        # Second pass: Search for build logs (once per composite_sha to minimize filesystem scans)
        # Many commits share the same composite_sha (Docker image), so we only need to search once per unique image
        composite_searched = set()

        for commit in commit_data:
            sha_short = commit['sha_short']
            composite_sha = commit['composite_sha']

            # Skip if we already searched for this composite_sha
            if composite_sha in composite_searched:
                continue
            composite_searched.add(composite_sha)

            # Search for build logs for ANY commit with this composite_sha
            # Build logs are keyed by individual commit SHA, not composite SHA
            matching_logs_for_composite = []
            for commit_sha in composite_to_commits[composite_sha]:
                log_filename = f"*.{commit_sha}.report.html"
                search_pattern = str(logs_dir / "*" / log_filename)
                logs_for_this_commit = glob.glob(search_pattern)
                if logs_for_this_commit:
                    # Store logs for this specific commit
                    log_paths[commit_sha] = []
                    for log_file in sorted(logs_for_this_commit):
                        log_path = Path(log_file).resolve()
                        # Extract date from filename (format: YYYY-MM-DD.sha.report.html)
                        date_str = log_path.name.split('.')[0]
                        try:
                            # Calculate relative path from output HTML file to log file
                            output_dir = output_path.resolve().parent
                            relative_path = os.path.relpath(log_path, output_dir)
                            log_paths[commit_sha].append((date_str, relative_path))
                        except ValueError:
                            log_paths[commit_sha].append((date_str, str(log_path)))
                    matching_logs_for_composite.extend(logs_for_this_commit)

            # If any commit in this composite_sha has logs, determine status
            if matching_logs_for_composite:
                self._perf_i["marker.composite.with.reports"] = int(self._perf_i.get("marker.composite.with.reports", 0) or 0) + 1

                # Use the most recent log for status determination
                log_path = Path(sorted(matching_logs_for_composite)[-1])

                # Determine build status using only the LATEST build date
                log_dir = log_path.parent
                all_status_files = []

                # Collect all status files for any commit in this composite_sha
                for commit_sha in composite_to_commits[composite_sha]:
                    for status_suffix in [MARKER_RUNNING, MARKER_FAILED, MARKER_PASSED]:
                        pattern = str(log_dir / f"*.{commit_sha}.*.{status_suffix}")
                        all_status_files.extend(glob.glob(pattern))

                status = STATUS_UNKNOWN  # Default status
                if all_status_files:
                    self._perf_i["marker.composite.with.status"] = int(self._perf_i.get("marker.composite.with.status", 0) or 0) + 1
                    # Extract dates from filenames (format: YYYY-MM-DD.sha.task.STATUS)
                    # Group files by date
                    from collections import defaultdict
                    files_by_date = defaultdict(list)
                    for f in all_status_files:
                        filename = Path(f).name
                        date_part = filename.split('.')[0]  # Extract YYYY-MM-DD
                        files_by_date[date_part].append(f)

                    # Use only the LATEST date
                    latest_date = max(files_by_date.keys())
                    date_files = files_by_date[latest_date]

                    # Check status for the latest build run
                    running_files = [f for f in date_files if f.endswith(f'.{MARKER_RUNNING}')]
                    # Exclude none-compilation and none-sanity from failure count (they always fail by design)
                    fail_files = [f for f in date_files
                                 if f.endswith(f'.{MARKER_FAILED}')
                                 and not (('none-' in f) and ('compilation' in f or 'sanity' in f))]
                    pass_files = [f for f in date_files if f.endswith(f'.{MARKER_PASSED}')]

                    # Determine status based on latest build run
                    # Priority: RUNNING > FAIL > PASS (if still running, show building even if some failed)
                    if running_files:
                        status = STATUS_BUILDING
                    elif fail_files:
                        status = STATUS_FAILED
                    elif pass_files:
                        status = STATUS_SUCCESS

                # Store status for all commits with logs in this composite_sha
                for commit_sha in composite_to_commits[composite_sha]:
                    if commit_sha in log_paths:
                        commit_to_status[commit_sha] = status

                # Store composite SHA status
                composite_to_status[composite_sha] = status
            else:
                self._perf_i["marker.composite.without.reports"] = int(self._perf_i.get("marker.composite.without.reports", 0) or 0) + 1
                # No report yet, status unknown
                if composite_sha not in composite_to_status:
                    composite_to_status[composite_sha] = STATUS_UNKNOWN

        # Pass 2: Assign status to all commits
        # Commits with logs get their own status, commits without logs inherit from Image SHA
        build_status = {}
        for commit in commit_data:
            sha_short = commit['sha_short']
            sha_full = commit['sha_full']
            composite_sha = commit['composite_sha']

            # Use per-commit status if available, otherwise inherit from Image SHA
            if sha_short in commit_to_status:
                # This commit has its own build logs
                build_status[sha_short] = {
                    'status': commit_to_status[sha_short],
                    'inherited': False
                }
            elif composite_sha in composite_to_status:
                # Inherit status from Image SHA
                build_status[sha_short] = {
                    'status': composite_to_status[composite_sha],
                    'inherited': True
                }
            else:
                # No status available
                build_status[sha_short] = {
                    'status': STATUS_UNKNOWN,
                    'inherited': False
                }

            # Note: We do NOT override local build status based on GitLab/GitHub status
            # The status indicator reflects LOCAL builds only (.PASS/.FAIL/.RUNNING markers)
            # GitHub Actions and GitLab pipeline status are shown separately in their own columns

        # Marker scan time (glob+grouping) regardless of cache.
        self._perf["marker.total_secs"] = float(self._perf.get("marker.total_secs", 0.0) or 0.0) + max(0.0, time.monotonic() - t_markers)
        
        # Now that local build status is known, color the Image SHA badge accordingly.
        #
        # UX request:
        # - Image SHA badge background is ALWAYS alternating greys (for scan-ability / grouping)
        # - Image SHA text color indicates status:
        #   - PASS: alternating greens
        #   - FAIL: alternating reds
        #   - no local build data: black (to contrast with the grey)
        #
        # Background greys (higher contrast, but still light enough so black text remains readable).
        grey_bg_a = "#a7b3c7"  # medium grey-blue (more contrasty)
        grey_bg_b = "#e5e7eb"  # light grey

        # Text colors for PASS/FAIL (single shade; no alternation).
        green_fg = "#2da44e"
        red_fg = "#c83a3a"
        neutral_fg = "#111827"

        # Image status chip colors (alternating shades for scan-ability).
        # Use a darker + lighter pair for alternating IMAGE:... label backgrounds.
        green_a = "#238636"  # darker green
        green_b = "#63d887"  # lighter green
        red_a = "#c83a3a"    # darker red
        red_b = "#e06060"    # lighter red
        yellow_a = "#ffc107" # darker yellow/amber (same as report.html header)
        yellow_b = "#ffd54f" # lighter yellow
        grey_a = "#8c959f"   # darker grey
        grey_b = "#b1bac4"   # lighter grey

        def _fg_for_bg(hex_color: str) -> str:
            """Pick a readable foreground color for a hex background."""
            try:
                s = str(hex_color or "").lstrip("#")
                if len(s) != 6:
                    return "#111827"
                r = int(s[0:2], 16)
                g = int(s[2:4], 16)
                b = int(s[4:6], 16)
                # Relative luminance (rough sRGB).
                lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
                return "#ffffff" if lum < 0.55 else "#111827"
            except Exception:
                return "#111827"

        for commit in commit_data:
            try:
                sha_short = str(commit.get("sha_short", "") or "")
                st = str((build_status.get(sha_short) or {}).get("status", STATUS_UNKNOWN) or STATUS_UNKNOWN)
                parity = int(commit.get("cds_parity", 0) or 0)
                # Background always alternates grey.
                commit["composite_bg_color"] = grey_bg_a if (parity % 2 == 0) else grey_bg_b

                # IMAGE:... label colors (alternating shades; foreground picked for contrast).
                if st == STATUS_SUCCESS:
                    bg = green_a if (parity % 2 == 0) else green_b
                elif st == STATUS_FAILED:
                    bg = red_a if (parity % 2 == 0) else red_b
                elif st == STATUS_BUILDING:
                    bg = yellow_a if (parity % 2 == 0) else yellow_b
                else:
                    bg = grey_a if (parity % 2 == 0) else grey_b
                commit["image_label_bg_color"] = bg
                commit["image_label_fg_color"] = _fg_for_bg(bg)

                # Text color encodes status.
                if st == STATUS_SUCCESS:
                    commit["composite_text_color"] = green_fg
                elif st == STATUS_FAILED:
                    commit["composite_text_color"] = red_fg
                else:
                    commit["composite_text_color"] = neutral_fg
            except Exception:
                continue
        
        # Generate timestamp (PT)
        generated_time = datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M:%S %Z')
        
        # Calculate GitHub Actions status counts (overall summary)
        gha_success_count = 0
        gha_failed_count = 0
        gha_in_progress_count = 0
        gha_other_count = 0

        for sha_full in [c['sha_full'] for c in commit_data]:
            gha_status = github_actions_status.get(sha_full)
            if gha_status and gha_status.get('conclusion') == 'success':
                gha_success_count += 1
            elif gha_status and gha_status.get('conclusion') == 'failure':
                gha_failed_count += 1
            elif gha_status and gha_status.get('status') == 'in_progress':
                gha_in_progress_count += 1
            else:
                gha_other_count += 1

        # Calculate per-commit check statistics (includes required vs optional failure split)
        gha_per_commit_stats = {}
        for sha_full in [c['sha_full'] for c in commit_data]:
            gha_status = github_actions_status.get(sha_full)
            if not gha_status or not gha_status.get('check_runs'):
                gha_per_commit_stats[sha_full] = {
                    'success_required': 0,
                    'success_optional': 0,
                    'failure_required': 0,
                    'failure_optional': 0,
                    'in_progress_required': 0,
                    'in_progress_optional': 0,
                    'pending': 0,
                    'cancelled': 0,
                    'other': 0,
                    'total': 0
                }
                continue

            summary = summarize_check_runs(gha_status.get('check_runs', []) or [])
            stats = {
                'success_required': int(summary.counts.success_required),
                'success_optional': int(summary.counts.success_optional),
                'failure_required': int(summary.counts.failure_required),
                'failure_optional': int(summary.counts.failure_optional),
                'in_progress_required': int(summary.counts.in_progress_required),
                'in_progress_optional': int(summary.counts.in_progress_optional),
                'pending': int(summary.counts.pending),
                'cancelled': int(summary.counts.cancelled),
                'other': int(summary.counts.other),
                'total': int(summary.counts.total),
            }
            gha_per_commit_stats[sha_full] = stats

        # (Check line HTML is rendered via the shared dashboard UI helper `check_line_html`)

        def _status_norm_for_check_run(*, status: str, conclusion: str) -> str:
            s = (status or "").strip().lower()
            c = (conclusion or "").strip().lower()
            if c in (CIStatus.SUCCESS.value, CIStatus.NEUTRAL.value, CIStatus.SKIPPED.value):
                return CIStatus.SUCCESS.value
            if c in ("failure", "timed_out", "action_required"):
                return CIStatus.FAILURE.value
            if c in (CIStatus.CANCELLED.value, "canceled"):
                return CIStatus.CANCELLED.value
            if s in (CIStatus.IN_PROGRESS.value, "in progress"):
                return CIStatus.IN_PROGRESS.value
            if s in ("queued", CIStatus.PENDING.value):
                return CIStatus.PENDING.value
            return CIStatus.UNKNOWN.value

        def _duration_str_to_seconds(s: str) -> float:
            """Best-effort parse of durations like '43m 33s', '30m33s', '2s', '1h 4m'."""
            try:
                total = 0.0
                for m in re.finditer(r"([0-9]+)\s*([hms])", str(s or "").lower()):
                    v = float(m.group(1))
                    u = m.group(2)
                    if u == "h":
                        total += v * 3600.0
                    elif u == "m":
                        total += v * 60.0
                    elif u == "s":
                        total += v
                return total
            except Exception:
                return 0.0

        def _subtree_needs_attention(node: TreeNodeVM, rollup_status: str, has_required_failure: bool) -> bool:
            # Shared policy with branches page.
            return ci_should_expand_by_default(
                rollup_status=str(rollup_status or ""),
                has_required_failure=bool(has_required_failure),
            )

        def _build_github_checks_tree_html(*, repo_path: Path, sha_full: str) -> str:
            gha = github_actions_status.get(sha_full) if github_actions_status else None
            check_runs = (gha.get("check_runs") if isinstance(gha, dict) else None) or []
            # Snapshot of API-returned check names (before we inject placeholders).
            # We use this to scope workflow-YAML inference to "relevant" workflows, and to avoid
            # placeholder-driven matching (which can otherwise make us infer too much when API data is missing).
            api_present_names = [
                str((cr or {}).get("name", "") or "").strip()
                for cr in (check_runs or [])
                if isinstance(cr, dict) and str((cr or {}).get("name", "") or "").strip()
            ]
            # Inject "expected but missing" placeholder checks so the commit-history tree matches
            # the local-branches tree UX (and makes missing required checks visible).
            try:
                from common_dashboard_lib import EXPECTED_CHECK_PLACEHOLDER_SYMBOL  # local import
                from common import normalize_check_name  # local import

                present_norm = {
                    normalize_check_name(str((cr or {}).get("name", "") or ""))
                    for cr in (check_runs or [])
                    if isinstance(cr, dict)
                }
                # Track what we've already seen/added so we never append duplicate placeholders.
                seen_norm = set(present_norm)
                required_names: List[str] = []
                try:
                    # Branch protection required checks for main (best-effort; may be empty on 403).
                    required_names = list(
                        self.github_client.get_required_checks_for_base_ref(owner="ai-dynamo", repo="dynamo", base_ref="main") or []
                    )
                except Exception:
                    required_names = []
                required_norm = {normalize_check_name(x) for x in (required_names or []) if str(x).strip()}
                expected_all = sorted({*set(required_names or [])}, key=lambda s: str(s).lower())
                for nm0 in expected_all:
                    n0 = normalize_check_name(nm0)
                    if n0 and n0 not in seen_norm:
                        check_runs.append(
                            {
                                "name": str(nm0),
                                "status": "queued",
                                "conclusion": "",
                                "html_url": "",
                                "details_url": "",
                                "is_required": (n0 in required_norm),
                                # Marker so rendering can show "— ◇" like branches page.
                                "_expected_placeholder": True,
                                "_expected_placeholder_symbol": EXPECTED_CHECK_PLACEHOLDER_SYMBOL,
                            }
                        )
                        seen_norm.add(n0)

                # Note: Expected checks (◇) inference from workflow YAML has been removed.
                # Only actual check runs from the API are displayed.
            except Exception:
                pass
            # Note: Sorting is now handled by PASS 4 (sort_by_name_pass) in the centralized pipeline.
            if not check_runs:
                # Always render a stable placeholder so every commit row can show the dropdown.
                root = TreeNodeVM(
                    node_key=f"gha-root:{sha_full}",
                    label_html=(
                        f'<span style="font-weight: 600;">GitHub checks</span> '
                        f'<a href="https://github.com/ai-dynamo/dynamo/commit/{html.escape(sha_full)}/checks" '
                        f'target="_blank" style="color: #0969da; font-size: 11px; text-decoration: none;">[checks]</a>'
                    ),
                    children=[
                        TreeNodeVM(
                            node_key=f"gha-empty:{sha_full}",
                            label_html='<span style="color: #57606a; font-size: 12px;">(no check data cached/fetched for this SHA)</span>',
                            children=[],
                            collapsible=False,
                        )
                    ],
                    collapsible=True,
                    default_expanded=True,
                    triangle_tooltip="GitHub checks",
                )
                # Return the div-based tree HTML
                return render_tree_divs([root])
            # Always allow raw-log fetch when missing (subject to `common.py` rules: only cache when
            # job status is completed; size caps; etc).
            allow_raw_logs = True
            raw_log_prefetch_budget = {"n": 10**12}

            # Flat view: show the exact check-run list (like GitHub Checks UI / our Details list),
            # without workflow YAML parsing or run-class bucketing.
            snippet_cache: Dict[str, str] = {}

            def snippet_for_raw_href(raw_href: str) -> str:
                if not raw_href:
                    return ""
                if raw_href in snippet_cache:
                    return snippet_cache[raw_href]
                try:
                    snippet = self._snippet_from_cached_raw_log(Path(self.repo_path) / raw_href)
                except Exception:
                    snippet = ""
                snippet_cache[raw_href] = snippet
                return snippet

            name_counts: Dict[str, int] = {}
            for cr0 in check_runs:
                try:
                    nm0 = str((cr0 or {}).get("name", "") or "")
                    name_counts[nm0] = int(name_counts.get(nm0, 0) or 0) + 1
                except Exception:
                    pass

            # Build CIJobNode objects (not TreeNodeVM) so run_all_passes can process them
            from common_branch_nodes import CIJobNode  # local import
            
            ci_job_nodes: List[CIJobNode] = []
            for cr in check_runs:
                if not isinstance(cr, dict):
                    continue
                name = str(cr.get("name", "") or "").strip()
                if not name:
                    continue
                is_expected_placeholder = bool(cr.get("_expected_placeholder", False))
                url = "" if is_expected_placeholder else str(cr.get("html_url", "") or cr.get("details_url", "") or "").strip()
                st = _status_norm_for_check_run(status=str(cr.get("status", "") or ""), conclusion=str(cr.get("conclusion", "") or ""))
                is_req = bool(cr.get("is_required", False))
                dur = "" if is_expected_placeholder else format_gh_check_run_duration(cr)

                raw_href = ""
                raw_size = 0
                snippet = ""
                if st == "failure" and str(cr.get("status", "") or "").lower() == "completed":
                    allow_fetch = bool(allow_raw_logs) and int(raw_log_prefetch_budget.get("n", 0) or 0) > 0
                    try:
                        raw_href = (
                            materialize_job_raw_log_text_local_link(
                                self.github_client,
                                job_url=url,
                                job_name=name,
                                owner="ai-dynamo",
                                repo="dynamo",
                                page_root_dir=Path(self.repo_path),
                                allow_fetch=bool(allow_fetch),
                                assume_completed=True,
                            )
                            or ""
                        )
                    except Exception:
                        raw_href = ""
                    if allow_fetch:
                        raw_log_prefetch_budget["n"] = int(raw_log_prefetch_budget.get("n", 0) or 0) - 1

                if raw_href:
                    try:
                        raw_size = int((Path(self.repo_path) / raw_href).stat().st_size)
                    except Exception:
                        raw_size = 0
                    snippet = snippet_for_raw_href(raw_href) if raw_href else ""

                # Disambiguate name if there are duplicates (adds [job ID] suffix)
                disambiguated_name = disambiguate_check_run_name(name, url, name_counts=name_counts)

                # Create CIJobNode with minimal fields - let run_all_passes handle the rest
                node = CIJobNode(
                    job_id=disambiguated_name,  # Use disambiguated name with [job ID] if duplicate
                    display_name=name,  # Keep original name for YAML matching
                    status=st,
                    duration=dur,
                    log_url=url,
                    is_required=is_req,
                    children=[],
                    page_root_dir=Path(self.repo_path),
                    context_key=f"{sha_full}:{name}",
                    github_api=self.github_client,
                    raw_log_href=raw_href,
                    raw_log_size_bytes=raw_size,
                    error_snippet_text=snippet,
                )
                # Set core_job_name for YAML matching (same pattern as build_ci_nodes_from_pr)
                node.core_job_name = name  # Original name without disambiguation
                ci_job_nodes.append(node)

            # Use centralized CI tree processing pipeline - it handles YAML augmentation, grouping, sorting
            children: List[TreeNodeVM]
            try:
                from common_dashboard_lib import run_all_passes
                
                children = run_all_passes(
                    ci_nodes=ci_job_nodes,
                    repo_root=Path(repo_path),
                    commit_sha=sha_full,
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"run_all_passes failed for {sha_full[:7]}: {e}")
                # Fallback: convert CI nodes to TreeVM directly
                children = [node.to_tree_vm() for node in ci_job_nodes]

            root = TreeNodeVM(
                node_key=f"gha-root:{sha_full}",
                label_html=(
                    f'<span style="font-weight: 600;">GitHub checks</span> '
                    f'<a href="https://github.com/ai-dynamo/dynamo/commit/{html.escape(sha_full)}/checks" '
                    f'target="_blank" style="color: #0969da; font-size: 11px; text-decoration: none;">[checks]</a>'
                ),
                children=children,
                collapsible=True,
                default_expanded=True,
                triangle_tooltip="GitHub checks",
            )
            # Return the div-based tree HTML
            return render_tree_divs([root])

        def _build_gitlab_checks_tree_html(*, sha_full: str, sha_short: str) -> str:
            nodes: List[TreeNodeVM] = []

            pr_num = sha_to_pr_number.get(sha_full) if sha_to_pr_number else None
            pipeline = gitlab_pipelines.get(sha_full) if gitlab_pipelines else None
            if (not pipeline) and pr_num and mr_pipelines:
                pipeline = mr_pipelines.get(int(pr_num))

            if pipeline and isinstance(pipeline, dict):
                pid = pipeline.get("id")
                web_url = str(pipeline.get("web_url", "") or "")
                status = str(pipeline.get("status", "") or "")

                job_data = pipeline_job_counts.get(pid) if (pipeline_job_counts and pid is not None) else None
                jobs = []
                if isinstance(job_data, dict) and "counts" in job_data:
                    jobs = job_data.get("jobs") or []

                children: List[TreeNodeVM] = []
                for j in sorted(jobs, key=lambda x: (str(x.get("stage", "") or ""), str(x.get("name", "") or ""))):
                    j_stage = str(j.get("stage", "") or "unknown")
                    j_name = str(j.get("name", "") or "")
                    j_status = str(j.get("status", "") or "")
                    label_stage = ".pre" if j_stage == "pre" else j_stage
                    job_label = f"{label_stage}.{j_name}"
                    is_mandatory = (
                        j_name.startswith(".pre")
                        or (".pre" in job_label)
                        or (label_stage in ("pre", ".pre"))
                        or j_name.startswith("build")
                        or j_name.startswith("test")
                    )
                    icon = status_icon_html(
                        status_norm=(
                            "in_progress"
                            if j_status == "running"
                            else (
                                "pending"
                                if j_status in ("pending", "created", "waiting_for_resource")
                                else ("cancelled" if j_status in ("canceled", "cancelled") else ("success" if j_status == "success" else ("failure" if j_status == "failed" else "unknown")))
                            )
                        ),
                        is_required=is_mandatory,
                    )
                    badge = mandatory_badge_html(
                        is_mandatory=bool(is_mandatory),
                        status_norm=(
                            "in_progress"
                            if j_status == "running"
                            else (
                                "pending"
                                if j_status in ("pending", "created", "waiting_for_resource")
                                else (
                                    "cancelled"
                                    if j_status in ("canceled", "cancelled")
                                    else ("success" if j_status == "success" else ("failure" if j_status == "failed" else "unknown"))
                                )
                            )
                        ),
                    )
                    children.append(
                        TreeNodeVM(
                            node_key=f"gl:{sha_short}:{job_label}",
                            label_html=(
                                f'{icon} <span style="font-family: SFMono-Regular, Consolas, Liberation Mono, Menlo, monospace; font-size: 12px;">'
                                f"{html.escape(job_label)}</span>{badge}"
                            ),
                            children=[],
                            collapsible=True,
                            default_expanded=False,
                        )
                    )

                root = TreeNodeVM(
                    node_key=f"gl-root:{sha_full}",
                    label_html=(
                        f'<span style="font-weight: 600;">GitLab pipeline</span> '
                        f'<a href="{html.escape(web_url, quote=True)}" target="_blank" style="color: #0969da; font-size: 11px; text-decoration: none;">[pipeline]</a> '
                        f'<span style="color: #57606a; font-size: 12px;">({html.escape(status)})</span>'
                    ),
                    children=children,
                    collapsible=True,
                    default_expanded=True,
                    triangle_tooltip="GitLab pipeline jobs",
                )
                return render_tree_divs([root])

            # Always return a placeholder tree so the GitLab dropdown doesn't disappear.
            root = TreeNodeVM(
                node_key=f"gl-root:{sha_full}",
                label_html=(
                    f'<span style="font-weight: 600;">GitLab pipeline</span> '
                    f'<a href="https://gitlab-master.nvidia.com/dl/ai-dynamo/dynamo/-/commit/{html.escape(sha_full)}" '
                    f'target="_blank" style="color: #0969da; font-size: 11px; text-decoration: none;">[commit]</a>'
                ),
                children=[
                    TreeNodeVM(
                        node_key=f"gl-empty:{sha_full}",
                        label_html='<span style="color: #57606a; font-size: 12px;">(no pipeline/job data found or cached for this SHA)</span>',
                        children=[],
                        collapsible=False,
                    )
                ],
                collapsible=True,
                default_expanded=True,
                triangle_tooltip="GitLab pipeline (if available)",
            )
            return render_tree_divs([root])

        # Attach per-commit trees to commit dictionaries for the template to embed (split GH vs GL).
        #
        # IMPORTANT: do not assign to a local variable named `html` in this function.
        # This module imports `html` (stdlib) and the nested helpers call `html.escape(...)`.
        # If we shadow `html`, Python will treat it as a local/free variable and crash at runtime.
        render_t = PhaseTimer()
        build_github_s = 0.0
        build_gitlab_s = 0.0
        slow_github: List[tuple[float, str]] = []
        slow_gitlab: List[tuple[float, str]] = []
        with render_t.phase("build_trees"):
            for c in commit_data:
                sha_full = str(c.get("sha_full", "") or "")
                sha_short = str(c.get("sha_short", "") or "")
                try:
                    t0 = time.monotonic()
                    c["github_checks_tree_html"] = _build_github_checks_tree_html(repo_path=self.repo_path, sha_full=sha_full)
                    dt = max(0.0, time.monotonic() - t0)
                    build_github_s += dt
                    slow_github.append((dt, sha_short or sha_full[:9]))
                except Exception:
                    # Never drop the toggles; keep stable placeholders even on errors.
                    try:
                        c["github_checks_tree_html"] = _build_github_checks_tree_html(repo_path=self.repo_path, sha_full=sha_full)
                    except Exception:
                        c["github_checks_tree_html"] = ""

                try:
                    t0 = time.monotonic()
                    c["gitlab_checks_tree_html"] = _build_gitlab_checks_tree_html(sha_full=sha_full, sha_short=sha_short)
                    dt = max(0.0, time.monotonic() - t0)
                    build_gitlab_s += dt
                    slow_gitlab.append((dt, sha_short or sha_full[:9]))
                except Exception:
                    try:
                        c["gitlab_checks_tree_html"] = _build_gitlab_checks_tree_html(sha_full=sha_full, sha_short=sha_short)
                    except Exception:
                        c["gitlab_checks_tree_html"] = ""

        # Render template
        template_dir = Path(__file__).parent
        env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )
        # Expose shared formatters to the template so the GitHub column summary matches other dashboards.
        env.globals["compact_ci_summary_html"] = compact_ci_summary_html
        template = env.get_template('show_commit_history.j2')

        # Page statistics (shown in an expandable block at the bottom of the HTML).
        elapsed_s = None
        if generation_t0 is not None:
            try:
                elapsed_s = max(0.0, time.monotonic() - float(generation_t0))
            except Exception:
                elapsed_s = None

        page_stats: List[tuple[str, Optional[str]]] = []
        if elapsed_s is not None:
            page_stats.append(("Generation time", f"{elapsed_s:.2f}s"))
            # Keep a consistent "generation.total_secs" row for dashboards.
            page_stats.append(("generation.total_secs", f"{elapsed_s:.2f}s"))
        # GitHub API stats (structured; rendered with <pre> blocks in Statistics).
        try:
            from common_dashboard_lib import github_api_stats_rows  # local import

            mode = "cache-only" if bool(getattr(self.github_client, "cache_only_mode", False)) else "normal"
            api_rows = github_api_stats_rows(
                github_api=self.github_client,
                max_github_api_calls=None,
                mode=mode,
                mode_reason="",
                extra_cache_stats=cache_stats if isinstance(cache_stats, dict) else None,
                top_n=15,
            )
            page_stats.extend(list(api_rows or []))
        except Exception:
            pass

        page_stats.append(("Commits shown", str(len(commit_data))))
        page_stats.append(("skip_gitlab_fetch", "true" if self.skip_gitlab_fetch else "false"))

        # Performance breakdown (best-effort; collected across the run).
        try:
            perf = dict(getattr(self, "_perf", {}) or {})
            perfi = dict(getattr(self, "_perf_i", {}) or {})
            if perf or perfi:
                # Composite SHA
                page_stats.append(("composite_sha.cache.hit", str(int(perfi.get("composite_sha.cache.hit", 0) or 0))))
                page_stats.append(("composite_sha.cache.miss", str(int(perfi.get("composite_sha.cache.miss", 0) or 0))))
                page_stats.append(("composite_sha.errors", str(int(perfi.get("composite_sha.errors", 0) or 0))))
                page_stats.append(("composite_sha.total_secs", f"{float(perf.get('composite_sha.total_secs') or 0.0):.2f}s"))
                page_stats.append(("composite_sha.compute_secs", f"{float(perf.get('composite_sha.compute_secs') or 0.0):.2f}s"))
                # Snippets
                page_stats.append(("snippet.cache.hit", str(int(perfi.get("snippet.cache.hit", 0) or 0))))
                page_stats.append(("snippet.cache.miss", str(int(perfi.get("snippet.cache.miss", 0) or 0))))
                page_stats.append(("snippet.total_secs", f"{float(perf.get('snippet.total_secs') or 0.0):.2f}s"))
                page_stats.append(("snippet.compute_secs", f"{float(perf.get('snippet.compute_secs') or 0.0):.2f}s"))
                # Markers / local build reports (grouped by composite_sha / Docker image)
                page_stats.append(("marker.composite.unique", str(int(perfi.get("marker.composite.with.reports", 0) or 0) + int(perfi.get("marker.composite.without.reports", 0) or 0))))
                page_stats.append(("marker.composite.with.reports", str(int(perfi.get("marker.composite.with.reports", 0) or 0))))
                page_stats.append(("marker.composite.with.status", str(int(perfi.get("marker.composite.with.status", 0) or 0))))
                page_stats.append(("marker.composite.without.reports", str(int(perfi.get("marker.composite.without.reports", 0) or 0))))
                page_stats.append(("marker.total_secs", f"{float(perf.get('marker.total_secs') or 0.0):.2f}s"))

                # GitLab cache hit/miss (cache files under ~/.cache/dynamo-utils)
                for k in [
                    "gitlab.cache.registry_images.hit",
                    "gitlab.cache.registry_images.miss",
                    "gitlab.cache.pipeline_status.hit",
                    "gitlab.cache.pipeline_status.miss",
                    "gitlab.cache.pipeline_jobs.hit",
                    "gitlab.cache.pipeline_jobs.miss",
                ]:
                    if k in perfi:
                        page_stats.append((k, str(int(perfi.get(k, 0) or 0))))
                page_stats.append(("gitlab.registry_images.total_secs", f"{float(perf.get('gitlab.registry_images.total_secs') or 0.0):.2f}s"))
                page_stats.append(("gitlab.pipeline_status.total_secs", f"{float(perf.get('gitlab.pipeline_status.total_secs') or 0.0):.2f}s"))
                page_stats.append(("gitlab.pipeline_jobs.total_secs", f"{float(perf.get('gitlab.pipeline_jobs.total_secs') or 0.0):.2f}s"))
        except Exception:
            pass

        # GitLab API totals (best-effort).
        try:
            gl = getattr(self, "gitlab_client", None)
            if gl is not None and hasattr(gl, "get_rest_call_stats"):
                st = gl.get_rest_call_stats() or {}
                page_stats.append(("gitlab.rest.calls", str(int(st.get("total") or 0))))
                page_stats.append(("gitlab.rest.total_secs", f"{float(st.get('time_total_s') or 0.0):.2f}s"))
                es = st.get("errors_by_status") if isinstance(st, dict) else None
                if isinstance(es, dict) and es:
                    page_stats.append(
                        ("gitlab.rest.errors_by_status", ", ".join([f"{k}={v}" for k, v in list(es.items())[:8]]))
                    )
                # Top endpoints (time + count) to spot hot paths quickly.
                try:
                    by_ep = st.get("by_endpoint") if isinstance(st, dict) else {}
                    t_by_ep = st.get("time_by_endpoint_s") if isinstance(st, dict) else {}
                    if isinstance(by_ep, dict) and isinstance(t_by_ep, dict) and (by_ep or t_by_ep):
                        labels = sorted(set(list(by_ep.keys()) + list(t_by_ep.keys())))
                        labels.sort(key=lambda k: (-int(by_ep.get(k, 0) or 0), -float(t_by_ep.get(k, 0.0) or 0.0), str(k)))
                        labels = labels[:10]
                        w = max(10, max(len(str(x)) for x in labels))
                        lines = [f"{'endpoint':<{w}}  calls   time"]
                        for k in labels:
                            c = int(by_ep.get(k, 0) or 0)
                            t = float(t_by_ep.get(k, 0.0) or 0.0)
                            lines.append(f"{str(k):<{w}}  {c:>5d}  {t:>6.2f}s")
                        page_stats.append(("gitlab.rest.by_endpoint_top10", "\n".join(lines)))
                except Exception:
                    pass
        except Exception:
            pass

        # GitHub cache hit/miss (derived from `cache_stats` we already collect).
        try:
            cs = cache_stats if isinstance(cache_stats, dict) else {}
            if cs:
                md = cs.get("merge_dates_cache") if isinstance(cs.get("merge_dates_cache"), dict) else {}
                rc = cs.get("required_checks_cache") if isinstance(cs.get("required_checks_cache"), dict) else {}
                gha = cs.get("github_actions_status_cache") if isinstance(cs.get("github_actions_status_cache"), dict) else {}
                if isinstance(md, dict) and md:
                    page_stats.append(("github.cache.merge_dates.hits", str(md.get("hits"))))
                    page_stats.append(("github.cache.merge_dates.misses", str(md.get("misses"))))
                if isinstance(rc, dict) and rc:
                    page_stats.append(("github.cache.required_checks.hits", str(rc.get("hits"))))
                    page_stats.append(("github.cache.required_checks.misses", str(rc.get("misses"))))
                if isinstance(gha, dict) and gha:
                    page_stats.append(("github.cache.actions_status.hit_fresh", str(gha.get("cache_hit_fresh"))))
                    page_stats.append(("github.cache.actions_status.stale_refresh", str(gha.get("cache_stale_refresh"))))
                    page_stats.append(("github.cache.actions_status.miss_fetch", str(gha.get("cache_miss_fetch"))))
                    page_stats.append(("github.cache.actions_status.miss_no_fetch", str(gha.get("cache_miss_no_fetch"))))
        except Exception:
            pass

        # Include timing breakdown if available (best-effort).
        try:
            t = getattr(self, "_last_timings", None) or {}
            if isinstance(t, dict) and t:
                # Note: total/render/write are measured outside this HTML generator; we show those separately
                # (or via elapsed_s above) to avoid stale/partial totals.
                for k in ["cache_load", "git_iter_commits", "github_merge_dates", "github_required_checks", "process_commits", "cache_save"]:
                    if k in t:
                        page_stats.append((f"{k}.total_secs", f"{float(t[k]):.2f}s"))
        except Exception:
            pass

        # Timing rows we want to show, but can't know until after rendering. Use placeholders and
        # patch them into the final HTML string to avoid a second expensive template render.
        PH_BUILD = "__TIMING_HTML_BUILD_TREES__"
        PH_TPL = "__TIMING_HTML_TEMPLATE_RENDER__"
        PH_RENDER = "__TIMING_RENDER_HTML__"
        page_stats.append(("render_html.total_secs", PH_RENDER))
        page_stats.append(("html_build_trees.total_secs", PH_BUILD))
        page_stats.append(("html_build_trees_github.total_secs", "__TIMING_HTML_BUILD_TREES_GH__"))
        page_stats.append(("html_build_trees_gitlab.total_secs", "__TIMING_HTML_BUILD_TREES_GL__"))
        page_stats.append(("html_template_render.total_secs", PH_TPL))
        # Top slow commits (helps pinpoint which commit(s) dominate tree-building).
        try:
            slow_github_sorted = sorted(slow_github, key=lambda x: -float(x[0]))[:5]
            slow_gitlab_sorted = sorted(slow_gitlab, key=lambda x: -float(x[0]))[:5]
            if slow_github_sorted:
                page_stats.append(("html_slowest_commits_github.total_secs", ", ".join([f"{sha}={dt:.2f}s" for dt, sha in slow_github_sorted])))
            if slow_gitlab_sorted:
                page_stats.append(("html_slowest_commits_gitlab.total_secs", ", ".join([f"{sha}={dt:.2f}s" for dt, sha in slow_gitlab_sorted])))
        except Exception:
            pass

        # Sort stats for readability (Generation time first, then other keys grouped by prefix).
        try:
            gen = [(k, v) for (k, v) in page_stats if k == "Generation time"]
            
            # Sort function: group by prefix (before first _ or .) then by full key
            def prefix_sort_key(kv):
                k = str(kv[0])
                # Extract prefix: everything before first underscore or dot
                if "_" in k or "." in k:
                    # Find the position of the first _ or .
                    underscore_pos = k.find("_") if "_" in k else len(k)
                    dot_pos = k.find(".") if "." in k else len(k)
                    split_pos = min(underscore_pos, dot_pos)
                    prefix = k[:split_pos]
                else:
                    prefix = k
                return (prefix, k)
            
            # Apply prefix-based grouping to all stats except Generation time
            other = sorted([(k, v) for (k, v) in page_stats if k != "Generation time"], key=prefix_sort_key)
            page_stats[:] = gen + other
        except Exception:
            pass
        except Exception:
            pass

        # Build tree time is known now.
        build_trees_s = 0.0
        try:
            build_trees_s = float(render_t.as_dict(include_total=False).get("build_trees") or 0.0)
        except Exception:
            build_trees_s = 0.0

        t0_tpl = time.monotonic()
        rendered_html = template.render(
            commits=commit_data,
            docker_images=docker_images,
            gitlab_images=gitlab_images,
            gitlab_pipelines=gitlab_pipelines,
            mr_pipelines=mr_pipelines,
            pipeline_job_counts=pipeline_job_counts,
            log_paths=log_paths,
            build_status=build_status,
            github_actions_status=github_actions_status,
            generated_time=generated_time,
            commit_count=len(commit_data),
            gha_success_count=gha_success_count,
            gha_failed_count=gha_failed_count,
            gha_in_progress_count=gha_in_progress_count,
            gha_other_count=gha_other_count,
            gha_per_commit_stats=gha_per_commit_stats,
            page_stats=page_stats,
            # Icons (shared look; legend/tooltips/status bar must match across dashboards)
            **ci_status_icon_context(),
            pass_plus_style=PASS_PLUS_STYLE,
        )
        tpl_render_s = max(0.0, time.monotonic() - t0_tpl)

        # Patch placeholders into the final HTML.
        try:
            rendered_html = rendered_html.replace(PH_BUILD, f"{build_trees_s:.2f}s")
            rendered_html = rendered_html.replace(PH_TPL, f"{tpl_render_s:.2f}s")
            rendered_html = rendered_html.replace(PH_RENDER, f"{(build_trees_s + tpl_render_s):.2f}s")
            rendered_html = rendered_html.replace("__TIMING_HTML_BUILD_TREES_GH__", f"{build_github_s:.2f}s")
            rendered_html = rendered_html.replace("__TIMING_HTML_BUILD_TREES_GL__", f"{build_gitlab_s:.2f}s")
        except Exception:
            pass
        return rendered_html

    def _get_cached_gitlab_images_from_sha(self, commit_data: List[dict]) -> dict:
        """Get Docker images mapped by commit SHA using cache.
        
        Simplified logic:
        - If skip_gitlab_fetch=True: Only use cache
        - If skip_gitlab_fetch=False: Fetch tags for recent commits (within 8 hours) using binary search
        
        Args:
            commit_data: List of commit dictionaries with sha_full and committed_datetime
            
        Returns:
            Dictionary mapping SHA to list of registry image info
        """
        cache_file = "gitlab_commit_sha.json"

        sha_full_list = [c['sha_full'] for c in commit_data]
        sha_to_datetime = {c['sha_full']: c['committed_datetime'] for c in commit_data}
        
        self.logger.debug(f"Getting Docker images for {len(sha_full_list)} SHAs")

        # Cache hit/miss accounting (best-effort).
        try:
            cache_path = common.resolve_cache_path(cache_file)
            cache0 = json.loads(cache_path.read_text() or "{}") if cache_path.exists() else {}
            if isinstance(cache0, dict):
                hit = sum(1 for sha in sha_full_list if sha in cache0)
                self._perf_i["gitlab.cache.registry_images.hit"] = int(self._perf_i.get("gitlab.cache.registry_images.hit", 0) or 0) + int(hit)
                self._perf_i["gitlab.cache.registry_images.miss"] = int(self._perf_i.get("gitlab.cache.registry_images.miss", 0) or 0) + int(
                    max(0, len(sha_full_list) - hit)
                )
        except Exception:
            pass

        t0 = time.monotonic()
        
        result = self.gitlab_client.get_cached_registry_images_for_shas(
            project_id="169905",  # dl/ai-dynamo/dynamo
            registry_id="85325",  # Main dynamo registry
            sha_list=sha_full_list,
            sha_to_datetime=sha_to_datetime,
            cache_file=cache_file,
            skip_fetch=self.skip_gitlab_fetch
        )
        self._perf["gitlab.registry_images.total_secs"] = float(self._perf.get("gitlab.registry_images.total_secs", 0.0) or 0.0) + max(
            0.0, time.monotonic() - t0
        )
        
        cached_count = sum(1 for v in result.values() if v)
        self.logger.debug(f"Found Docker images for {cached_count}/{len(sha_full_list)} SHAs")
        
        return result

    def _get_gitlab_pipeline_statuses(self, sha_full_list: List[str]) -> dict:
        """Get GitLab CI pipeline status for commits using the centralized cache.

        Cache file format (gitlab_pipeline_status.json; stored under ~/.cache/dynamo-utils via resolve_cache_path()):
            {
                "<full_commit_sha>": {
                    "status": "failed",
                    "id": 38895507,
                    "web_url": "https://gitlab-master.nvidia.com/.../pipelines/38895507"
                },
                "<commit_sha_with_no_pipeline>": null,
                ...
            }

        Fields:
            - status: Pipeline status (success, failed, running, etc.)
            - id: GitLab pipeline ID (integer)
            - web_url: Full URL to view the pipeline in GitLab UI
            - null: Commit has no associated pipeline

        Args:
            sha_full_list: List of full commit SHAs (40 characters)

        Returns:
            Dictionary mapping SHA to pipeline status dict with 'status', 'id', 'web_url'
        """
        cache_file = "gitlab_pipeline_status.json"

        self.logger.debug(f"Getting pipeline status for {len(sha_full_list)} commits")

        # Cache hit/miss accounting (best-effort).
        try:
            cache_path = common.resolve_cache_path(cache_file)
            cache0 = json.loads(cache_path.read_text() or "{}") if cache_path.exists() else {}
            if isinstance(cache0, dict):
                hit = sum(1 for sha in sha_full_list if sha in cache0)
                self._perf_i["gitlab.cache.pipeline_status.hit"] = int(self._perf_i.get("gitlab.cache.pipeline_status.hit", 0) or 0) + int(hit)
                self._perf_i["gitlab.cache.pipeline_status.miss"] = int(self._perf_i.get("gitlab.cache.pipeline_status.miss", 0) or 0) + int(
                    max(0, len(sha_full_list) - hit)
                )
        except Exception:
            pass

        t0 = time.monotonic()
        
        result = self.gitlab_client.get_cached_pipeline_status(sha_full_list, cache_file=cache_file, skip_fetch=self.skip_gitlab_fetch)
        self._perf["gitlab.pipeline_status.total_secs"] = float(self._perf.get("gitlab.pipeline_status.total_secs", 0.0) or 0.0) + max(
            0.0, time.monotonic() - t0
        )
        
        cached_count = sum(1 for v in result.values() if v is not None)
        self.logger.debug(f"Found pipeline status for {cached_count}/{len(sha_full_list)} commits")

        return result

    def _get_gitlab_pipeline_job_counts(self, pipeline_ids: List[int]) -> dict:
        """Get GitLab CI pipeline job details (counts + individual job info) using the centralized cache.

        Cache file format (in dynamo-utils cache dir; see `cache_file` below):
            {
                "40118215": {
                    "counts": {
                        "success": 15,
                        "failed": 8,
                        "running": 0,
                        "pending": 0
                    },
                    "jobs": [
                        {"stage": "build", "name": "build-dynamo-image-amd64", "status": "success"},
                        {"stage": "test", "name": "pre-merge-vllm", "status": "failed"},
                        ...
                    ],
                    "fetched_at": "2025-12-18T02:44:20.118368Z"
                },
                ...
            }

        Fields:
            - counts: Dict with job counts for this pipeline
                - success: Number of successful jobs (integer)
                - failed: Number of failed jobs (integer)
                - running: Number of currently running jobs (integer)
                - pending: Number of pending jobs (integer)
            - jobs: List of individual job details (for tooltip display)
                - stage: Job stage (string), e.g. "build", "test", "pre"
                - name: Job name (string)
                - status: Job status (success/failed/running/pending/etc.)
            - fetched_at: ISO 8601 timestamp when this data was fetched

        Note: Completed pipelines (running=0, pending=0) are cached forever.
              Active pipelines are refetched if older than 30 minutes.

        Args:
            pipeline_ids: List of pipeline IDs

        Returns:
            Dictionary mapping pipeline ID to job details dict with 'counts' and 'jobs'
        """
        # v3: include per-job "stage" in cached job list and split pending vs canceled counts.
        # Bumping cache key forces a refetch even for completed pipelines (cached forever).
        cache_file = "gitlab_pipeline_jobs_details_v3.json"

        self.logger.debug(f"Getting job details for {len(pipeline_ids)} pipelines")

        # Cache hit/miss accounting (best-effort).
        try:
            keys = [str(int(x)) for x in (pipeline_ids or [])]
            cache_path = common.resolve_cache_path(cache_file)
            cache0 = json.loads(cache_path.read_text() or "{}") if cache_path.exists() else {}
            if isinstance(cache0, dict):
                hit = sum(1 for k in keys if k in cache0)
                self._perf_i["gitlab.cache.pipeline_jobs.hit"] = int(self._perf_i.get("gitlab.cache.pipeline_jobs.hit", 0) or 0) + int(hit)
                self._perf_i["gitlab.cache.pipeline_jobs.miss"] = int(self._perf_i.get("gitlab.cache.pipeline_jobs.miss", 0) or 0) + int(
                    max(0, len(keys) - hit)
                )
        except Exception:
            pass

        t0 = time.monotonic()

        result = self.gitlab_client.get_cached_pipeline_job_details(pipeline_ids, cache_file=cache_file, skip_fetch=self.skip_gitlab_fetch)
        self._perf["gitlab.pipeline_jobs.total_secs"] = float(self._perf.get("gitlab.pipeline_jobs.total_secs", 0.0) or 0.0) + max(
            0.0, time.monotonic() - t0
        )

        cached_count = sum(1 for v in result.values() if v is not None)
        self.logger.debug(f"Found job details for {cached_count}/{len(pipeline_ids)} pipelines")

        return result

    def _get_local_docker_images_by_sha(self, sha_list: List[str]) -> dict:
        """Get Docker images containing each SHA in their tag

        Args:
            sha_list: List of short SHAs (9 characters)

        Returns:
            Dictionary mapping SHA to list of image details (dicts with tag, id, size, created)
            
            Example return value:
            {
                "21a03b316": [
                    {
                        "tag": "gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo:21a03b316-38895507-vllm-amd64",
                        "id": "1234abcd5678",
                        "size": "13.4GB",
                        "created": "2025-11-20 22:15:32"
                    },
                    {
                        "tag": "gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo:21a03b316-38895507-vllm-arm64",
                        "id": "9876fedc5432",
                        "size": "8.0GB",
                        "created": "2025-11-20 22:16:35"
                    }
                ],
                "5fe0476e6": []
            }
        """
        sha_to_images = {sha: [] for sha in sha_list}

        try:
            # Get all Docker images with detailed information
            result = subprocess.run(
                ['docker', 'images', '--format', '{{.Repository}}:{{.Tag}}|{{.ID}}|{{.Size}}|{{.CreatedAt}}'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                images = result.stdout.strip().split('\n')

                # Filter images containing each SHA and parse details
                for image_line in images:
                    if not image_line or '<none>:<none>' in image_line:
                        continue

                    parts = image_line.split('|')
                    if len(parts) < 4:
                        continue

                    tag = parts[0]
                    image_id = parts[1]
                    size = parts[2]
                    # Parse created timestamp: "2025-10-18 09:27:29 -0700 PDT"
                    # Extract date and time (first two parts)
                    created_parts = parts[3].split() if parts[3] else []
                    if len(created_parts) >= 2:
                        created = f"{created_parts[0]} {created_parts[1]}"  # Date and time
                    else:
                        created = parts[3].strip() if parts[3] else ''

                    for sha in sha_list:
                        if sha in tag:
                            sha_to_images[sha].append({
                                'tag': tag,
                                'id': image_id,
                                'size': size,
                                'created': created
                            })

        except Exception as e:
            self.logger.warning(f"Failed to get Docker images: {e}")

        return sha_to_images


def main():
    """Main entry point for the commit history generator"""
    parser = argparse.ArgumentParser(
        description='Generate Dynamo commit history with composite SHAs and Docker images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show last 50 commits (HTML-only)
  %(prog)s

  # Show last 20 commits with verbose output (HTML-only)
  %(prog)s --max-commits 20 --verbose

  # Use custom repository path
  %(prog)s --repo-path /path/to/dynamo_ci
        """
    )

    parser.add_argument(
        '--repo-path',
        type=Path,
        default=Path.cwd(),
        help='Path to the Dynamo repository (default: current directory)'
    )

    parser.add_argument(
        '--max-commits',
        type=int,
        default=50,
        help='Maximum number of commits to show (default: 50)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output (INFO level logging)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output (DEBUG level logging, shows all API calls)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        help='Output path for HTML file (default: auto-detect from repo)'
    )

    parser.add_argument(
        '--skip-gitlab-fetch',
        action='store_true',
        help='Skip fetching GitLab registry data, use cached data only (much faster)'
    )
    # NOTE: Removed --max-github-fetch-commits.
    # GitHub fetch behavior is now governed by cache TTLs in `common.py`.
    parser.add_argument(
        '--token',
        help='GitHub personal access token (or set GH_TOKEN/GITHUB_TOKEN env var)'
    )
    parser.add_argument(
        '--allow-anonymous-github',
        action='store_true',
        help='Allow anonymous GitHub REST calls (60/hr core rate limit). By default we require auth to avoid rate limiting.'
    )
    parser.add_argument(
        '--max-github-api-calls',
        type=int,
        default=100,
        help='Hard cap on GitHub REST API network calls per invocation (cached reads do not count). Default: 100.'
    )

    parser.add_argument(
        '--logs-dir',
        type=Path,
        help='Path to logs directory for build reports (default: repo-path/logs)'
    )

    parser.add_argument(
        '--export-pipeline-pr-csv',
        type=Path,
        help='Write a CSV mapping GitLab pipeline URL/ID -> commit SHA -> PR number for the commit window'
    )

    args = parser.parse_args()

    # Validate repository path
    if not args.repo_path.exists():
        sys.stderr.write(f"Error: Repository path does not exist: {args.repo_path}\n")
        return 1

    if not (args.repo_path / '.git').exists():
        sys.stderr.write(f"Error: Not a git repository: {args.repo_path}\n")
        return 1

    # Prune locally-served raw logs to avoid unbounded growth and delete any partial/unverified artifacts.
    # We only render `[raw log]` links when the local file exists (or was materialized),
    # so pruning won't produce dead links on a freshly generated page.
    try:
        _ = prune_dashboard_raw_logs(page_root_dir=args.repo_path, max_age_days=30)
        # Also remove any partial/unverified raw logs (legacy cache artifacts, missing completed=true, etc).
        _ = prune_partial_raw_log_caches(page_root_dirs=[args.repo_path])
    except Exception:
        pass

    # Create generator and run
    generator = CommitHistoryGenerator(
        repo_path=args.repo_path,
        verbose=args.verbose,
        debug=args.debug,
        skip_gitlab_fetch=args.skip_gitlab_fetch,
        github_token=args.token,
        allow_anonymous_github=bool(args.allow_anonymous_github),
        max_github_api_calls=int(args.max_github_api_calls),
    )
    # Fail fast if exhausted; detailed stats are rendered into the HTML Statistics section.
    cache_only_reason = ""
    try:
        generator.github_client.check_core_rate_limit_or_raise()
    except Exception as e:
        # Switch to cache-only mode (no new GitHub network calls).
        cache_only_reason = str(e)
        try:
            generator.github_client.set_cache_only_mode(True)
        except Exception:
            pass

    rc = 1
    try:
        rc = generator.show_commit_history(
        max_commits=args.max_commits,
        output_path=args.output,
        logs_dir=args.logs_dir,
        export_pipeline_pr_csv=args.export_pipeline_pr_csv,
    )
        # Debug: print snippet cache analysis
        if hasattr(generator, '_snippet_cache_debug'):
            debug = generator._snippet_cache_debug
            print(f"\n{'='*80}", file=sys.stderr)
            print(f"SNIPPET CACHE DEBUG", file=sys.stderr)
            print(f"{'='*80}", file=sys.stderr)
            print(f"Total hits: {len(debug['hits'])}", file=sys.stderr)
            print(f"Total misses: {len(debug['misses'])}", file=sys.stderr)
            print(f"\nMiss reasons breakdown:", file=sys.stderr)
            from collections import Counter
            reason_counts = Counter(debug['reasons'].values())
            for reason, count in reason_counts.most_common():
                print(f"  {reason}: {count}", file=sys.stderr)
            
            if debug['misses']:
                print(f"\nFirst 10 missed keys:", file=sys.stderr)
                for key in debug['misses'][:10]:
                    reason = debug['reasons'].get(key, 'unknown')
                    print(f"  {key}: {reason}", file=sys.stderr)
        
        return rc
    finally:
        # No stdout/stderr run-stats; the HTML Statistics section contains the breakdowns.
        pass


if __name__ == '__main__':
    sys.exit(main())
