#!/usr/bin/env python3
"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Dynamo Commit History Generator - Standalone Tool

Generates commit history with Image SHA (hash of container/ contents; previously shown as CDS) and Docker image detection.
HTML-only dashboard generator.
"""

import argparse
import concurrent.futures
import csv
import fcntl
import glob
import hashlib
import html
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict

import git  # type: ignore[import-not-found]
from datetime import datetime, timezone
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
    EXPECTED_CHECK_PLACEHOLDER_SYMBOL,
    PASS_PLUS_STYLE,
    TreeNodeVM,
    build_and_test_dynamo_phases_from_actions_job,
    build_page_stats,
    check_line_html,
    ci_should_expand_by_default,
    ci_status_icon_context,
    ci_subsection_tuples_for_job,
    compact_ci_summary_html,
    disambiguate_check_run_name,
    extract_actions_job_id_from_url,
    github_api_stats_rows,
    job_name_wants_pytest_details,
    parse_workflow_yaml_and_build_mapping_pass,
    render_tree_divs,
    required_badge_html,
    mandatory_badge_html,
    run_all_passes,
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
from ci_log_errors import snippet as ci_snippet
from cache_snippet import SNIPPET_CACHE
from cache_commit_history import COMMIT_HISTORY_CACHE

# Import utilities from common and API modules
from common import (
    DynamoRepositoryUtils,
    PhaseTimer,
    dynamo_utils_cache_dir,
    resolve_cache_path,
    MARKER_RUNNING,
    MARKER_PASSED,
    MARKER_FAILED,
    MARKER_KILLED,
    select_shas_for_network_fetch,
)
from common_branch_nodes import CIJobNode
from common_github import (
    GitHubAPIClient,
    COMMIT_HISTORY_PERF_STATS,
    classify_ci_kind,
    format_gh_check_run_duration,
    summarize_check_runs,
    normalize_check_name,
    is_required_check_name,
)
from common_gitlab import (
    GitLabAPIClient,
)

# Jinja2 for HTML template rendering
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

# Global logger
logger = logging.getLogger(__name__)

# Constants for build status values
STATUS_UNKNOWN = 'unknown'
STATUS_SUCCESS = 'success'
STATUS_FAILED = 'failed'
STATUS_BUILDING = 'building'

# Grafana Individual Job Details dashboard URL template
# Example: https://grafana.nvidia.com/d/beyv28rcnhs74b/individual-job-details?orgId=283&var-branch=pull-request%2F5516&var-job=All&var-job_status=All&${__url_time_range}&var-repo=All&var-commit=All&var-workflow=All
GRAFANA_PR_URL_TEMPLATE = "https://grafana.nvidia.com/d/beyv28rcnhs74b/individual-job-details?orgId=283&var-branch=pull-request%2F{pr_number}&var-job=All&var-job_status=All&${{__url_time_range}}&var-repo=All&var-commit=All&var-workflow=All"


_normalize_check_name = normalize_check_name
_is_required_check_name = is_required_check_name


def _extract_error_snippet_worker(raw_log_path: str) -> str:
    """Worker process entrypoint: extract snippet text for a single raw log file.

    Keep this function top-level so it is picklable for multiprocessing.
    Do not write caches in the worker; the parent merges results and persists.
    """
    return str(ci_snippet.extract_error_snippet_from_log_file(Path(raw_log_path)) or "")


class CommitHistoryGenerator:
    """Generate commit history with Image SHA (hash of container/ contents) and Docker images"""

    def __init__(
        self,
        repo_path: Path,
        verbose: bool = False,
        debug: bool = False,
        gitlab_fetch_skip: bool = False,
        github_token: Optional[str] = None,
        allow_anonymous_github: bool = False,
        max_github_api_calls: int = 100,
        parallel_workers: int = 0,
        disable_snippet_cache_read: bool = False,
        enable_success_build_test_logs: bool = False,
        run_verifier_pass: bool = False,
    ):
        """
        Initialize the commit history generator

        Args:
            repo_path: Path to the Dynamo repository
            verbose: Enable verbose output (INFO level)
            debug: Enable debug output (DEBUG level)
            gitlab_fetch_skip: Skip fetching GitLab registry data, use cached data only
        """
        self.repo_path = Path(repo_path)
        self.verbose = verbose
        self.debug = debug
        self.gitlab_fetch_skip = bool(gitlab_fetch_skip)
        self.parallel_workers = int(parallel_workers or 0)
        self.disable_snippet_cache_read = bool(disable_snippet_cache_read)
        # Opt-in because downloading successful build-test raw logs can be expensive.
        # When enabled, we cache successful build-test raw logs so we can parse pytest slowest tests.
        self.enable_success_build_test_logs = bool(enable_success_build_test_logs)
        self.run_verifier_pass = bool(run_verifier_pass)
        self.logger = self._setup_logger()
        # NOTE: Commit history cache is now handled by COMMIT_HISTORY_CACHE singleton
        # NOTE: Snippet caching is handled by the unified SNIPPET_CACHE (snippet-cache.json)
        # imported from cache_snippet.py and shared across all dashboard generators.
        # Per-run performance counters are now tracked globally in COMMIT_HISTORY_PERF_STATS
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

    def _snippets_for_raw_hrefs(self, raw_hrefs: List[str]) -> Dict[str, tuple[str, list[str]]]:
        """Batch snippet extraction for raw log hrefs (optionally in multiple processes).

        Args:
            raw_hrefs: list[str] of repo-relative paths (hrefs) to local raw log files, e.g.:
              [
                "dynamo_ci/logs/60392310930.log",
                "dynamo_ci/logs/60392310931.log",
              ]

        Returns:
            Dict[str, Tuple[str, List[str]]] mapping raw_href -> (snippet_body, categories):
              {
                "dynamo_ci/logs/60392310930.log": (
                  "<snippet text without Categories header>",
                  ["pytest-error", "python-error"],
                ),
                "dynamo_ci/logs/60392310931.log": ("", []),
              }

        Notes:
        - Cache reads + writes happen in the parent process only.
        - Worker processes only compute snippet text for cache misses; parent merges into cache.
        """
        out: Dict[str, tuple[str, list[str]]] = {}
        if not raw_hrefs:
            return out

        # De-dupe while preserving order.
        seen: set[str] = set()
        hrefs: List[str] = []
        for h in raw_hrefs:
            hh = str(h or "")
            if not hh or hh in seen:
                continue
            seen.add(hh)
            hrefs.append(hh)
        misses: List[Tuple[str, Path]] = []  # (href, local_path)
        for href in hrefs:
            p = Path(self.repo_path) / href
            if not p.exists() or not p.is_file():
                out[href] = ("", [])
                continue

            # For testing parallel parsing: treat everything as a miss, regardless of cache contents.
            if bool(self.disable_snippet_cache_read):
                SNIPPET_CACHE.stats.miss += 1
                misses.append((href, p))
                continue

            cached = SNIPPET_CACHE.get_if_fresh(raw_log_path=p)
            if cached is not None:
                out[href] = cached
                continue

            misses.append((href, p))

        if not misses:
            return out

        # Compute misses and store into the unified shared snippet cache.
        if int(self.parallel_workers or 0) > 1:
            max_workers = int(self.parallel_workers)
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
                futs: Dict[concurrent.futures.Future, Tuple[str, Path]] = {}
                for href, p in misses:
                    fut = ex.submit(_extract_error_snippet_worker, str(p))
                    futs[fut] = (href, p)

                for fut in concurrent.futures.as_completed(futs):
                    href, p = futs[fut]
                    sn = str(fut.result() or "")
                    body, cats = SNIPPET_CACHE.put_raw_snippet(raw_log_path=p, snippet_raw=sn)
                    out[href] = (body, cats)
        else:
            for href, p in misses:
                sn = str(ci_snippet.extract_error_snippet_from_log_file(p) or "")
                body, cats = SNIPPET_CACHE.put_raw_snippet(raw_log_path=p, snippet_raw=sn)
                out[href] = (body, cats)

        return out

    def _snippet_from_cached_raw_log(self, raw_log_path: Path) -> tuple[str, list[str]]:
        """Return an error snippet and categories for a raw log file, using the persistent cache when possible.
        
        Returns:
            (snippet_text, categories): snippet without "Categories:" prefix, and list of category strings
        """
        p = Path(raw_log_path)
        if not p.exists() or not p.is_file():
            return ("", [])
        # Unified snippet cache (disk-backed with lock+merge).
        return SNIPPET_CACHE.get_or_compute(raw_log_path=p)

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

        # common_dashboard_lib logging
        dashboard_lib_logger = logging.getLogger('common_dashboard_lib')
        if self.debug:
            dashboard_lib_logger.setLevel(logging.DEBUG)
        elif self.verbose:
            dashboard_lib_logger.setLevel(logging.INFO)
        else:
            dashboard_lib_logger.setLevel(logging.WARNING)
        if not dashboard_lib_logger.handlers:
            dashboard_lib_logger.addHandler(handler)

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

        # NOTE: Snippet cache is now automatically managed by SNIPPET_CACHE (unified cache)
        # and doesn't require explicit loading at startup.

        meta_stats: Dict[str, int] = {"full_hit": 0, "miss": 0}

        # Load cache from COMMIT_HISTORY_CACHE singleton
        # Cache entries modified during this run will be auto-saved
        t0 = phase_t.start()
        self.logger.debug(f"Loaded commit history cache with entries")
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
            if repo.head.is_detached:
                original_ref = repo.head.commit.hexsha
            else:
                original_ref = repo.active_branch.name

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

            cache_only_github = bool(self.github_client.cache_only_mode)

            # Batch fetch merge dates for all PRs (GitHub)
            # NOTE: gitlab_fetch_skip only affects GitLab calls, not GitHub calls
            if pr_numbers and (not cache_only_github):
                self.logger.info(f"Fetching merge dates for {len(pr_numbers)} PRs...")
                t0 = phase_t.start()
                pr_to_merge_date = self.github_client.get_cached_pr_merge_dates(
                    pr_numbers,
                    cache_file="github_pr_merge_dates.json",
                )
                phase_t.stop("github_merge_dates", t0)
                self.logger.info(
                    f"Got merge dates for {sum(1 for v in pr_to_merge_date.values() if v)} PRs"
                )

            # Fetch required checks for all PRs (GraphQL per-PR required checks).
            # NOTE: gitlab_fetch_skip only affects GitLab calls, not GitHub calls
            # TODO: Unhardcode "ai-dynamo"/"dynamo" - should be class attributes or config
            pr_to_required_checks: Dict[int, List[str]] = {}
            if pr_numbers and not cache_only_github:
                self.logger.info(f"Fetching required checks for {len(pr_numbers)} PRs...")
                t0 = phase_t.start()
                for pr_num in set(pr_numbers):
                    try:
                        required_set = self.github_client.get_required_checks(
                            owner="ai-dynamo",
                            repo="dynamo",
                            pr_number=pr_num,
                        )
                        pr_to_required_checks[pr_num] = sorted(required_set) if required_set else []
                    except (OSError, subprocess.SubprocessError):
                        pr_to_required_checks[pr_num] = []
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

                    cached_entry = COMMIT_HISTORY_CACHE.get(sha_full)
                    merge_date = None
                    commit_dt_pt = commit.committed_datetime.astimezone(ZoneInfo("America/Los_Angeles"))
                    date_epoch = int(commit_dt_pt.timestamp())
                    author_email = ""

                    if cached_entry and isinstance(cached_entry, dict):
                        meta_stats["full_hit"] += 1
                        COMMIT_HISTORY_PERF_STATS.composite_sha_cache_hit += 1
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
                        if str(cached_entry.get("message") or "") != str(message_first_line or "") and message_first_line:
                            cached_entry["message"] = message_first_line
                            COMMIT_HISTORY_CACHE.put(sha_full, cached_entry)
                            cache_updated = True

                        # Backfill author_email if missing.
                        if not author_email:
                            author_email = str(commit.author.email or "")
                            if author_email:
                                cached_entry["author_email"] = author_email
                                COMMIT_HISTORY_CACHE.put(sha_full, cached_entry)
                                cache_updated = True

                        # Backfill merge_date if missing and available.
                        if merge_date is None:
                            pr_number = GitLabAPIClient.parse_mr_number_from_message(message_first_line)
                            if pr_number and pr_number in pr_to_merge_date:
                                merge_date = pr_to_merge_date[pr_number]
                                if merge_date:
                                    cached_entry['merge_date'] = merge_date
                                    COMMIT_HISTORY_CACHE.put(sha_full, cached_entry)
                                    cache_updated = True
                    else:
                        # Cache miss: compute from git
                        date_str = commit_dt_pt.strftime('%Y-%m-%d %H:%M:%S')
                        author_name = commit.author.name
                        author_email = str(commit.author.email or "")
                        message_first_line = _clean_subject_line(commit.message)
                        pr_number = GitLabAPIClient.parse_mr_number_from_message(message_first_line)
                        if pr_number and pr_number in pr_to_merge_date:
                            merge_date = pr_to_merge_date[pr_number]

                        meta_stats["miss"] += 1
                        COMMIT_HISTORY_PERF_STATS.composite_sha_cache_miss += 1
                        t_sha = time.monotonic()
                        try:
                            repo.git.checkout(commit.hexsha)
                            composite_sha = repo_utils.generate_composite_sha()
                        except Exception as e:
                            composite_sha = "ERROR"
                            self.logger.error(f"Failed to calculate composite SHA for {sha_short}: {e}")
                            COMMIT_HISTORY_PERF_STATS.composite_sha_errors += 1
                        COMMIT_HISTORY_PERF_STATS.composite_sha_compute_secs += max(0.0, time.monotonic() - t_sha)

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
                        COMMIT_HISTORY_CACHE.put(sha_full, cache_entry)
                        cache_updated = True

                    # Total time spent obtaining composite SHA (hit + miss path).
                    COMMIT_HISTORY_PERF_STATS.composite_sha_total_secs += max(0.0, time.monotonic() - t_sha_total)

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

                    # Periodically save cache to avoid losing progress if killed (every 10 commits)
                    if cache_updated and (i + 1) % 10 == 0:
                        COMMIT_HISTORY_CACHE.flush()
                        self.logger.debug(f"Periodic cache save: {i+1}/{len(commits)} commits processed")
                        cache_updated = False  # Reset flag after save

            finally:
                # Restore original ref (best-effort)
                if repo is not None and original_ref is not None:
                    repo.git.checkout(original_ref)

            # Optional: export pipeline -> PR mapping as CSV for this commit window.
            if export_pipeline_pr_csv:
                try:
                    sha_full_list = list(sha_to_message_first_line.keys())
                    gitlab_pipelines = self._get_gitlab_pipeline_statuses(sha_full_list)
                    export_pipeline_pr_csv.parent.mkdir(parents=True, exist_ok=True)

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
                            # TODO: Unhardcode github.com/ai-dynamo/dynamo URLs
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
                self._last_timings = dict(phase_t.as_dict(include_total=True))

                t0 = phase_t.start()
                html_content = self._generate_commit_history_html(
                    commit_data,
                    logs_dir,
                    output_path,
                    sha_to_pr_number=sha_to_pr_number,
                    pr_to_merge_date=pr_to_merge_date,
                    pr_to_required_checks=pr_to_required_checks,
                    generation_t0=generation_t0,
                    branch_name=original_ref if original_ref and not repo.head.is_detached else "main",
                )
                phase_t.stop("render_html", t0)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                t0 = phase_t.start()
                atomic_write_text(output_path, html_content, encoding="utf-8")
                phase_t.stop("write_output", t0)
                # Cache miss summary
                if self.verbose or self.debug:
                    self.logger.info(
                        "Cache stats: commit_metadata full_hit=%s miss=%s",
                        meta_stats.get("full_hit", 0),
                        meta_stats.get("miss", 0),
                    )


            # Save cache if updated
            if cache_updated:
                t0 = phase_t.start()
                COMMIT_HISTORY_CACHE.flush()
                self.logger.debug(f"Cache saved (final)")
                phase_t.stop("cache_save", t0)

            # Persist snippet cache to disk (best-effort).
            # The unified SNIPPET_CACHE automatically flushes on program exit,
            # but we do it explicitly here to ensure data is saved.
            try:
                SNIPPET_CACHE.flush()
            except Exception:
                pass

            # Persist timings for HTML page stats / debugging (best-effort).
            self._last_timings = dict(phase_t.as_dict(include_total=True))

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
            self.logger.error(f"Failed to get commit history: {e}", exc_info=True)
            return 1

    def _generate_commit_history_html(
        self,
        commit_data: List[dict],
        logs_dir: Path,
        output_path: Path,
        sha_to_pr_number: Optional[Dict[str, int]] = None,
        pr_to_merge_date: Optional[Dict[int, Optional[str]]] = None,
        pr_to_required_checks: Optional[Dict[int, List[str]]] = None,
        generation_t0: Optional[float] = None,
        branch_name: Optional[str] = None,
    ) -> str:
        """Generate HTML report for commit history with Docker image detection

        Args:
            commit_data: List of commit dictionaries with sha_short, sha_full, composite_sha, date, author, message
            logs_dir: Path to logs directory for build reports
            output_path: Path where the HTML file will be written (used for relative path calculation)
            sha_to_pr_number: Mapping full commit SHA -> PR number (if known)
            pr_to_merge_date: Mapping PR number -> merge date string (if known/merged)
            pr_to_required_checks: Mapping PR number -> list of required check names (if known)

        Returns:
            HTML content as string
        """
        # Initialize optional parameters
        pr_to_merge_date = pr_to_merge_date or {}

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
            # TODO: Unhardcode project_id="169905" (dl/ai-dynamo/dynamo) - should be config
            mr_pipelines = self.gitlab_client.get_cached_merge_request_pipelines(
                pr_numbers,
                project_id="169905",
                cache_file="gitlab_mr_pipelines.json",
                skip_fetch=self.gitlab_fetch_skip,
            )

        # Get GitLab CI pipeline job counts
        pipeline_ids = [p['id'] for p in gitlab_pipelines.values() if p and 'id' in p]
        pipeline_job_counts = {}
        if pipeline_ids:
            pipeline_job_counts = self._get_gitlab_pipeline_job_counts(pipeline_ids)

        # Build SHA list for later use
        sha_full_list = [c['sha_full'] for c in commit_data]

        # Prime required-checks metadata (GitHub) in a capped way:
        # - only allow network fetch for PRs associated with allow_fetch_shas
        # - cache will automatically be checked first (get_required_checks uses 7-day TTL cache)
        # TODO: Unhardcode "ai-dynamo"/"dynamo" - should be class attributes or config
        pr_to_required_checks: Dict[int, List[str]] = {}
        pr_numbers_allow = sorted({p for p in (sha_to_pr_number.get(sha) for sha in sha_full_list) if p})
        if pr_numbers_allow:
            for pr_num in pr_numbers_allow:
                try:
                    required_set = self.github_client.get_required_checks(
                        owner="ai-dynamo",
                        repo="dynamo",
                        pr_number=pr_num,
                    )
                    pr_to_required_checks[pr_num] = sorted(required_set) if required_set else []
                except (OSError, subprocess.SubprocessError):
                    pr_to_required_checks[pr_num] = []

        # Get GitHub Actions check status for commits:
        # - allow network fetch for any SHA that is cache-missing/stale (TTL policy lives in `common.py`).
        #
        # NOTE: We do *not* age-gate fetch here anymore. The cache TTL policy already ensures
        # old commits refresh very rarely (DEFAULT_STABLE_TTL_S), but we still need to fetch
        # at least once to populate the cache; otherwise some SHAs show no GitHub dropdown.
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

        cache_only_github = bool(self.github_client.cache_only_mode)

        # Batch fetch check runs using get_pr_checks_rows() for all unique PRs
        # Group SHAs by PR number
        pr_to_shas: Dict[int, List[str]] = {}
        for sha in sha_full_list:
            pr_num = sha_to_pr_number.get(sha)
            if pr_num:
                pr_to_shas.setdefault(pr_num, []).append(sha)

        # Fetch check runs for each PR and convert to SHA-keyed format
        # Use ThreadPoolExecutor to parallelize I/O-bound GitHub API calls
        github_actions_status: Dict[str, Dict[str, Any]] = {}

        def fetch_pr_checks(pr_num: int, shas: List[str]) -> Tuple[int, List[str], List[Any]]:
            """Fetch check runs for a single PR (parallelizable worker)."""
            # For commit history, ALL commits are from merged PRs (they're on main branch).
            # Even if pr_to_merge_date doesn't have the date, we should use long TTL.
            # 
            # Calculate TTL:
            # 1. If PR has merge_date OR we're in commit history context: 360 days (immutable)
            # 2. Fallback (shouldn't happen for commit history): adaptive based on commit age
            merge_date = pr_to_merge_date.get(pr_num)
            
            # Commit history assumption: all PRs are merged, use long TTL
            # (We're only showing commits that made it to main branch)
            ttl_s = 360 * 24 * 3600  # 360 days - PRs in commit history are immutable
            
            # Note: We don't use adaptive TTL here because commit history only shows
            # merged commits, so PR checks are immutable

            rows = self.github_client.get_pr_checks_rows(
                owner='ai-dynamo',
                repo='dynamo',
                pr_number=pr_num,
                skip_fetch=bool(cache_only_github),
                ttl_s=ttl_s,
            )
            return (pr_num, shas, rows)

        # Parallelize API calls using ThreadPoolExecutor (I/O-bound work)
        max_workers = min(32, len(pr_to_shas) or 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(fetch_pr_checks, pr_num, shas): (pr_num, shas)
                for pr_num, shas in pr_to_shas.items()
            }

            for future in concurrent.futures.as_completed(futures):
                pr_num, shas, rows = future.result()

                # Convert GHPRCheckRow objects to dict format for all SHAs in this PR
                check_runs_dicts = []
                for row in rows:
                    check_runs_dicts.append({
                        'name': row.name,
                        'short_name': row.name,  # Will be overwritten by enrichment step if YAML mapping exists
                        'status': row.status_raw if row.status_raw not in {'pass', 'fail'} else ('completed' if row.status_raw in {'pass', 'fail'} else row.status_raw),
                        'conclusion': 'success' if row.status_raw == 'pass' else ('failure' if row.status_raw == 'fail' else row.status_raw),
                        'html_url': row.url,
                        'details_url': row.url,
                        'run_id': row.run_id,  # Pass through for prefetching
                        'job_id': row.job_id,  # Pass through for stable identity
                    })

                # Assign same check runs to all SHAs in this PR
                for sha in shas:
                    github_actions_status[sha] = {
                        'status': 'completed',
                        'conclusion': 'success',
                        'total_count': len(check_runs_dicts),
                        'check_runs': check_runs_dicts,
                    }

        # OPTIMIZATION (2026-01-18): Batch prefetch job details for all runs
        # Extract all unique run_ids from check_runs and batch fetch them in one call per run
        # instead of 1 call per job. This reduces API calls by 90-95% (500-1000 → 10-20 calls).
        run_ids_to_prefetch: Set[str] = set()
        for sha_full, gha in github_actions_status.items():
            if not gha or not isinstance(gha.get('check_runs'), list):
                continue
            for check in gha['check_runs']:
                if not isinstance(check, dict):
                    continue
                url = str(check.get('html_url', '') or check.get('details_url', '')).strip()
                # Extract run_id from URLs like: https://github.com/owner/repo/actions/runs/123/job/456
                match = re.search(r'/actions/runs/(\d+)/', url)
                if match:
                    run_ids_to_prefetch.add(match.group(1))

        if run_ids_to_prefetch:
            self.logger.info(f"Batch prefetching job details for {len(run_ids_to_prefetch)} unique workflow runs")
            try:
                job_map = self.github_client.get_actions_runs_jobs_batched(
                    owner='ai-dynamo',
                    repo='dynamo',
                    run_ids=list(run_ids_to_prefetch),
                    ttl_s=30 * 24 * 3600,  # 30 days cache
                )
                self.logger.info(f"Prefetched {len(job_map)} job details into cache (will skip individual API calls)")
            except Exception as e:
                self.logger.warning(f"Batch prefetch failed (will fall back to individual fetches): {e}")

        # Annotate GitHub check runs with "is_required" using PR required-checks + fallback patterns.
        pr_to_required_checks = pr_to_required_checks or {}
        
        # Build a mapping of job names to short names from YAML (shared across all commits)
        _, yaml_mappings = parse_workflow_yaml_and_build_mapping_pass(
            flat_nodes=[],  # Empty list since we only need YAML parsing, not node modification
            repo_root=Path(self.repo_path),
            commit_sha="HEAD",  # Use HEAD as a representative commit for YAML parsing
        )
        job_name_to_short = yaml_mappings.get('job_name_to_id', {})

        for sha_full, gha in (github_actions_status or {}).items():
            if not gha or not gha.get('check_runs'):
                continue
            pr_number = sha_to_pr_number.get(sha_full)
            required_list = pr_to_required_checks.get(pr_number, []) if pr_number else []
            required_norm = {_normalize_check_name(n) for n in (required_list or []) if n}
            # Note: Sorting is now handled by PASS 4 (sort_by_name_pass) in the centralized pipeline.
            for check in (gha.get('check_runs', []) or []):
                check_name = check.get('name', '')
                check['is_required'] = _is_required_check_name(check_name, required_norm)
                # Add short name for tooltip display
                check['short_name'] = job_name_to_short.get(check_name, check_name)

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
        
        # Process commit messages for PR links and prepare Image-SHA grouping state.
        #
        # UX:
        # - Commit SHAs are rendered with alternating background colors *by Image SHA* (see template CSS),
        #   so commits that share an Image SHA share the same commit-SHA highlight.
        # - IMAGE:xxxxxx badges no longer use background colors; they use a compact status-dot icon instead.
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
                # TODO: Unhardcode github.com/ai-dynamo/dynamo URL
                pr_link = f"https://github.com/ai-dynamo/dynamo/pull/{pr_number}"
                message = re.sub(
                    r'\(#(\d+)\)',
                    f'(<a href="{pr_link}" class="pr-link" target="_blank">#{pr_number}</a>)',
                    message
                )
                commit['message'] = message

                # Apply same PR link transformation to full_message (for commit dropdown)
                if 'full_message' in commit:
                    full_message = commit['full_message']
                    full_message = re.sub(
                        r'\(#(\d+)\)',
                        f'(<a href=\"{pr_link}\" class=\"pr-link\" target=\"_blank\">#{pr_number}</a>)',
                        full_message
                    )
                    commit['full_message'] = full_message
                commit['pr_number'] = int(pr_number)
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
                COMMIT_HISTORY_PERF_STATS.marker_composite_with_reports += 1

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
                    COMMIT_HISTORY_PERF_STATS.marker_composite_with_status += 1
                    # Extract dates from filenames (format: YYYY-MM-DD.sha.task.STATUS)
                    # Group files by date
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
                COMMIT_HISTORY_PERF_STATS.marker_composite_without_reports += 1
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
        COMMIT_HISTORY_PERF_STATS.marker_total_secs += max(0.0, time.monotonic() - t_markers)
        
        # Now that local build status is known, add status icons for IMAGE SHA badges.
        #
        # UX request:
        # - Remove background colors from IMAGE badges
        # - Add a fixed-size filled circle icon in front of IMAGE:...
        #   - SUCCESS: green circle with check
        #   - FAILED: red circle with X
        #   - BUILDING: yellow circle with hourglass
        #   - UNKNOWN: gray circle (no glyph)

        def _image_status_icon_html(*, status: str) -> str:
            """Return a fixed-size (12x12) status dot for IMAGE:... that matches tree-node style.

            Important: route through shared `status_icon_html` so all dashboards/icons are consistent.
            """
            st = str(status or STATUS_UNKNOWN).strip().lower()
            if st == str(STATUS_SUCCESS).strip().lower():
                return status_icon_html(status_norm="success", is_required=True)
            if st == str(STATUS_FAILED).strip().lower():
                return status_icon_html(status_norm="failure", is_required=True)
            if st == str(STATUS_BUILDING).strip().lower():
                return status_icon_html(status_norm="in_progress", is_required=False)
            return status_icon_html(status_norm="unknown", is_required=False)

        for commit in commit_data:
            sha_short = str(commit.get("sha_short", "") or "")
            st = str((build_status.get(sha_short) or {}).get("status", STATUS_UNKNOWN) or STATUS_UNKNOWN)
            commit["image_status_icon"] = _image_status_icon_html(status=st)
        
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
            if c == CIStatus.SKIPPED.value:
                return CIStatus.SKIPPED.value
            if c in (CIStatus.SUCCESS.value, CIStatus.NEUTRAL.value):
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

        def _build_github_checks_tree_html(*, repo_path: Path, sha_full: str, required_names: List[str]) -> Tuple[str, dict]:
            # INSTRUMENTATION: Track detailed timing breakdown
            timing_breakdown = {}
            t_total = time.monotonic()
            
            # Initialize raw log timing stats tracking
            if not hasattr(self.github_client, '_raw_log_timing_stats'):
                self.github_client._raw_log_timing_stats = {
                    'extract': 0.0,
                    'api': 0.0,
                    'path_setup': 0.0,
                    'exists_check': 0.0,
                    'symlink_check': 0.0,
                    'count': 0,
                }
            
            t0 = time.monotonic()
            gha = github_actions_status.get(sha_full) if github_actions_status else None
            check_runs = (gha.get("check_runs") if isinstance(gha, dict) else None) or []
            timing_breakdown['fetch_check_runs'] = time.monotonic() - t0
            
            # Debug: log when check_runs is empty but gha exists
            if self.debug and gha and not check_runs:
                logger.debug(f"GitHub tree for {sha_full[:9]}: gha exists but check_runs is empty. gha keys: {list(gha.keys()) if isinstance(gha, dict) else 'not dict'}")
            elif self.debug and not gha:
                logger.debug(f"GitHub tree for {sha_full[:9]}: No gha data in github_actions_status")
            elif self.debug and check_runs:
                logger.debug(f"GitHub tree for {sha_full[:9]}: Found {len(check_runs)} check runs")
            # Inject "expected but missing" placeholder checks so the commit-history tree matches
            # the local-branches tree UX (and makes missing required checks visible).
            present_norm = {
                normalize_check_name(str((cr or {}).get("name", "") or ""))
                for cr in (check_runs or [])
                if isinstance(cr, dict)
            }
            # Track what we've already seen/added so we never append duplicate placeholders.
            seen_norm = set(present_norm)
            # required_names is now passed as parameter (fetched once before parallel loop)
            required_norm = {normalize_check_name(x) for x in (required_names or []) if str(x).strip()}
            expected_all = sorted({*set(required_names or [])}, key=lambda s: str(s).lower())
            for nm0 in expected_all:
                n0 = normalize_check_name(nm0)
                if n0 and n0 not in seen_norm:
                    check_runs.append(
                        {
                            "name": str(nm0),
                            "short_name": str(nm0),  # Will be overwritten by enrichment if YAML mapping exists
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
            # Note: Sorting is now handled by PASS 4 (sort_by_name_pass) in the centralized pipeline.
            if not check_runs:
                # Always render a stable placeholder so every commit row can show the dropdown.
                # TODO: Unhardcode github.com/ai-dynamo/dynamo URL
                root = TreeNodeVM(
                    node_key=f"gha-root:{sha_full}",
                    label_html=(
                        f'<span style="font-weight: 600;">GitHub Actions</span> '
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
                    collapsible=False,
                    default_expanded=True,
                    triangle_tooltip="GitHub Actions",
                )
                # Return the div-based tree HTML
                return render_tree_divs([root])
            # Always allow raw-log fetch when missing (subject to `common.py` rules: only cache when
            # job status is completed; size caps; etc).
            allow_raw_logs = True
            raw_log_prefetch_budget = {"n": 10**12}

            # Flat view: show the exact check-run list (like GitHub Checks UI / our Details list),
            # without workflow YAML parsing or run-class bucketing.
            # Collect all raw hrefs for this commit so we can batch snippet extraction (optionally parallel).
            raw_hrefs_for_commit: List[str] = []

            name_counts: Dict[str, int] = {}
            for cr0 in check_runs:
                nm0 = str((cr0 or {}).get("name", "") or "")
                name_counts[nm0] = int(name_counts.get(nm0, 0) or 0) + 1

            # First pass: materialize raw logs (if needed) and collect per-check info.
            # Second pass: batch-extract snippets for any raw logs we found.
            check_infos: List[Dict[str, object]] = []
            # Build CIJobNode objects (not TreeNodeVM) so run_all_passes can process them
            
            t0 = time.monotonic()
            t_duration_calc = 0.0
            t_raw_log_materialize = 0.0
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
                run_id = str(cr.get("run_id", "") or "").strip()
                job_id = str(cr.get("job_id", "") or "").strip()

                raw_href = ""
                raw_size = 0
                # Fetch raw logs for:
                # 1. Failed jobs (for error snippets)
                # 2. Build-test jobs (for pytest timing extraction, regardless of success/failure)
                is_build_test = job_name_wants_pytest_details(name)
                cr_status_lc = str(cr.get("status", "") or "").lower()
                should_fetch_raw_log = (
                    # Always fetch raw logs for failed jobs (for error snippets).
                    (st == "failure" and cr_status_lc == "completed")
                    # Optional: fetch raw logs for successful build-test jobs so we can parse pytest timings.
                    or (self.enable_success_build_test_logs and is_build_test and st == "success" and cr_status_lc == "completed")
                )

                if should_fetch_raw_log:
                    allow_fetch = bool(allow_raw_logs) and int(raw_log_prefetch_budget.get("n", 0) or 0) > 0
                    try:
                        # Raw logs are stored relative to the output HTML directory (page root),
                        # not relative to the git repo root.
                        t_raw = time.monotonic()
                        page_root_dir = (output_path.parent if output_path else Path(self.repo_path)).resolve()
                        raw_href = (
                            materialize_job_raw_log_text_local_link(
                                self.github_client,
                                job_url=url,
                                job_name=name,
                                owner="ai-dynamo",
                                repo="dynamo",
                                page_root_dir=page_root_dir,
                                allow_fetch=bool(allow_fetch),
                                assume_completed=True,
                            )
                            or ""
                        )
                        t_raw_log_materialize += time.monotonic() - t_raw
                        if allow_fetch:
                            raw_log_prefetch_budget["n"] = int(raw_log_prefetch_budget.get("n", 0) or 0) - 1
                    except (OSError, ValueError, KeyError) as e:
                        # Best-effort: raw log materialization is optional
                        logger.debug(f"Failed to materialize raw log: {e}")
                        pass

                if raw_href:
                    page_root_dir = (output_path.parent if output_path else Path(self.repo_path)).resolve()
                    raw_size = int((page_root_dir / raw_href).stat().st_size)
                    # Only extract error snippets for failed jobs (not for successful build-test jobs)
                    # We fetch raw logs for successful build-test jobs to get pytest timings,
                    # but we don't need to extract error snippets from them
                    if st == "failure":
                        raw_hrefs_for_commit.append(str(raw_href))

                # Calculate duration (priority: raw log > API job details)
                dur = ""
                if not is_expected_placeholder:
                    t_dur = time.monotonic()
                    if raw_href:
                        # If we have/downloaded raw log, extract duration from it (cached)
                        from common_github import calculate_duration_from_raw_log
                        page_root_dir = (output_path.parent if output_path else Path(self.repo_path)).resolve()
                        raw_log_path = page_root_dir / raw_href
                        dur = calculate_duration_from_raw_log(raw_log_path)

                    if not dur and url:
                        # Fallback: get duration from API job details (cached)
                        from common_github import calculate_duration_from_job_url
                        dur = calculate_duration_from_job_url(
                            github_api=self.github_client,
                            job_url=url,
                            owner="ai-dynamo",
                            repo="dynamo"
                        )
                    t_duration_calc += time.monotonic() - t_dur

                # Disambiguate name if there are duplicates (adds [job ID] suffix)
                disambiguated_name = disambiguate_check_run_name(name, url, name_counts=name_counts)

                check_infos.append(
                    {
                        "name": name,
                        "disambiguated_name": disambiguated_name,
                        "url": url,
                        "status_norm": st,
                        "duration": dur,
                        "is_required": is_req,
                        "raw_href": raw_href,
                        "raw_size": raw_size,
                        "run_id": run_id,
                        "job_id": job_id,
                    }
                )

            timing_breakdown['process_check_runs'] = time.monotonic() - t0
            timing_breakdown['duration_calculation'] = t_duration_calc
            timing_breakdown['raw_log_materialize'] = t_raw_log_materialize

            t0 = time.monotonic()
            try:
                logger.debug(f"[_build_github_checks_tree_html] {sha_full[:9]}: About to call self._snippets_for_raw_hrefs with {len(raw_hrefs_for_commit)} hrefs")
                logger.debug(f"[_build_github_checks_tree_html] {sha_full[:9]}: type(self._snippets_for_raw_hrefs) = {type(self._snippets_for_raw_hrefs)}")
                logger.debug(f"[_build_github_checks_tree_html] {sha_full[:9]}: self._snippets_for_raw_hrefs = {self._snippets_for_raw_hrefs}")
                snippets_by_href = self._snippets_for_raw_hrefs(raw_hrefs_for_commit) if raw_hrefs_for_commit else {}
            except TypeError as e:
                logger.error(f"[_build_github_checks_tree_html] {sha_full[:9]}: TypeError calling _snippets_for_raw_hrefs: {e}. self={self}, raw_hrefs_for_commit={len(raw_hrefs_for_commit) if raw_hrefs_for_commit else 0}")
                import traceback
                traceback.print_exc()
                raise
            timing_breakdown['snippet_extraction'] = time.monotonic() - t0
            
            t0 = time.monotonic()
            for info in check_infos:
                name = str(info.get("name", "") or "")
                disambiguated_name = str(info.get("disambiguated_name", "") or name)
                url = str(info.get("url", "") or "")
                st = str(info.get("status_norm", "") or "unknown")
                dur = str(info.get("duration", "") or "")
                is_req = bool(info.get("is_required", False))
                raw_href = str(info.get("raw_href", "") or "")
                raw_size = int(info.get("raw_size", 0) or 0)
                run_id = str(info.get("run_id", "") or "")
                job_id = str(info.get("job_id", "") or "")
                
                snippet, snippet_categories = snippets_by_href.get(raw_href, ("", [])) if raw_href else ("", [])

                # Duration already calculated earlier (from raw log or API), stored in check_infos

                # Create CIJobNode with minimal fields - let run_all_passes handle the rest
                node = CIJobNode(
                    job_id=disambiguated_name,  # Use disambiguated name with [job ID] if duplicate
                    display_name=name,  # Keep original name for YAML matching
                    status=st,
                    duration=dur,
                    log_url=url,
                    actions_job_id=job_id,  # Pass job_id for stable identity
                    run_id=run_id,  # Pass run_id for batch prefetching
                    is_required=is_req,
                    children=[],
                    page_root_dir=(output_path.parent if output_path else Path(self.repo_path)).resolve(),
                    context_key=f"{sha_full}:{name}",
                    github_api=self.github_client,
                    raw_log_href=raw_href,
                    raw_log_size_bytes=raw_size,
                    error_snippet_text=snippet,
                    error_snippet_categories=snippet_categories,
                )
                # Set core_job_name for YAML matching (same pattern as build_ci_nodes_from_pr)
                node.core_job_name = name  # Original name without disambiguation
                
                # NOTE: Steps/tests are populated via the shared pipeline pass:
                # common_dashboard_lib.add_job_steps_and_tests_pass (used by local/remote branches too).
                
                ci_job_nodes.append(node)
                
                # (intentionally removed) duplicate CIJobNode creation
                # The CIJobNode was already created above (disambiguated_name) and appended once.

            timing_breakdown['build_ci_job_nodes'] = time.monotonic() - t0

            # Use centralized CI tree processing pipeline - it handles YAML augmentation, grouping, sorting
            t0 = time.monotonic()
            children: List[TreeNodeVM] = run_all_passes(
                ci_nodes=ci_job_nodes,
                repo_root=Path(repo_path),
                commit_sha=sha_full,
                github_api=self.github_client,
                run_verifier_pass=self.run_verifier_pass,
            )
            timing_breakdown['run_all_passes'] = time.monotonic() - t0

            # TODO: Unhardcode github.com/ai-dynamo/dynamo URL
            t0 = time.monotonic()
            root = TreeNodeVM(
                node_key=f"gha-root:{sha_full}",
                label_html=(
                    f'<span style="font-weight: 600;">GitHub Actions</span> '
                    f'<a href="https://github.com/ai-dynamo/dynamo/commit/{html.escape(sha_full)}/checks" '
                    f'target="_blank" style="color: #0969da; font-size: 11px; text-decoration: none;">[checks]</a>'
                ),
                children=children,
                collapsible=False,
                default_expanded=True,
                triangle_tooltip="GitHub Actions",
            )
            # Return the div-based tree HTML
            logger.debug(f"[_build_github_checks_tree_html] {sha_full[:9]}: Rendering tree with {len(children)} children nodes")
            tree_html = render_tree_divs([root])
            timing_breakdown['render_tree_divs'] = time.monotonic() - t0
            timing_breakdown['total'] = time.monotonic() - t_total
            
            # Log timing breakdown for this commit
            if self.debug:
                breakdown_str = ", ".join([f"{k}={v:.3f}s" for k, v in timing_breakdown.items()])
                print(f"[TIMING] {sha_full[:9]}: {breakdown_str}", flush=True)
            
            logger.debug(f"[_build_github_checks_tree_html] {sha_full[:9]}: render_tree_divs returned {len(tree_html)} chars")
            return tree_html, timing_breakdown

        def _build_gitlab_checks_tree_html(*, sha_full: str, sha_short: str) -> str:
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
                        f'<span style="font-weight: 600;">GitLab</span> '
                        f'<a href="{html.escape(web_url, quote=True)}" target="_blank" style="color: #0969da; font-size: 11px; text-decoration: none;">[pipeline]</a> '
                        f'<span style="color: #57606a; font-size: 12px;">({html.escape(status)})</span>'
                    ),
                    children=children,
                    collapsible=False,
                    default_expanded=True,
                    triangle_tooltip="GitLab",
                )
                return render_tree_divs([root])

            # Always return a placeholder tree so the GitLab dropdown doesn't disappear.
            root = TreeNodeVM(
                node_key=f"gl-root:{sha_full}",
                label_html=(
                    f'<span style="font-weight: 600;">GitLab</span> '
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
                collapsible=False,
                default_expanded=True,
                triangle_tooltip="GitLab",
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
        
        # Accumulate detailed timing breakdown across all commits
        timing_totals = {
            'fetch_check_runs': 0.0,
            'inject_placeholders': 0.0,
            'process_check_runs': 0.0,
            'duration_calculation': 0.0,
            'raw_log_materialize': 0.0,
            'snippet_extraction': 0.0,
            'build_ci_job_nodes': 0.0,
            'run_all_passes': 0.0,
            'render_tree_divs': 0.0,
        }

        def build_github_tree(commit_dict: dict) -> Tuple[str, str, str, float, dict]:
            """Build GitHub checks tree for a single commit (parallelizable worker)."""
            sha_full = str(commit_dict.get("sha_full", "") or "")
            sha_short = str(commit_dict.get("sha_short", "") or "")
            t0 = time.monotonic()
            timing_breakdown = {}
            try:
                logger.debug(f"[build_github_tree] Starting for {sha_short} ({sha_full[:12]})")
                tree_html, timing_breakdown = _build_github_checks_tree_html(
                    repo_path=self.repo_path,
                    sha_full=sha_full,
                    required_names=required_names  # Fetched once before parallel loop
                )
                if tree_html:
                    logger.debug(f"[build_github_tree] SUCCESS for {sha_short}: Generated {len(tree_html)} chars of HTML")
                else:
                    logger.warning(f"[build_github_tree] EMPTY tree_html for {sha_short} ({sha_full[:12]}) - check_runs may be empty or render failed")
            except (KeyError, ValueError, TypeError) as e:
                # Best-effort: tree building is optional
                logger.warning(f"[build_github_tree] EXCEPTION for {sha_full[:8]}: {type(e).__name__}: {e}")
                tree_html = ""
            except Exception as e:
                # Catch any other exception
                logger.error(f"[build_github_tree] UNEXPECTED EXCEPTION for {sha_full[:8]}: {type(e).__name__}: {e}", exc_info=True)
                tree_html = ""
            dt = max(0.0, time.monotonic() - t0)
            return (sha_full, sha_short, tree_html, dt, timing_breakdown)

        def build_gitlab_tree(commit_dict: dict) -> Tuple[str, str, str, float]:
            """Build GitLab checks tree for a single commit (parallelizable worker)."""
            sha_full = str(commit_dict.get("sha_full", "") or "")
            sha_short = str(commit_dict.get("sha_short", "") or "")
            t0 = time.monotonic()
            tree_html = _build_gitlab_checks_tree_html(sha_full=sha_full, sha_short=sha_short)
            dt = max(0.0, time.monotonic() - t0)
            return (sha_full, sha_short, tree_html, dt)

        with render_t.phase("build_trees"):
            # OPTIMIZATION (2026-01-22): Ensure raw log directory exists ONCE before parallel loop
            # Previously: mkdir() called inside materialize_job_raw_log_text_local_link() for each job
            # Problem: mkdir(exist_ok=True) still does filesystem stat checks → expensive when called 100+ times
            # Now: Create directory once upfront, workers just use it
            from common_dashboard_runtime import dashboard_served_raw_log_repo_cache_dir
            page_root_dir = (output_path.parent if output_path else Path(self.repo_path)).resolve()
            dest_dir = dashboard_served_raw_log_repo_cache_dir(page_root_dir=page_root_dir)
            try:
                dest_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                # If dest_dir is a symlink, mkdir would fail; ensure the resolved target exists instead
                Path(dest_dir).resolve().mkdir(parents=True, exist_ok=True)
            
            # OPTIMIZATION (2026-01-18): Fix parallelization bug
            # Fetch required checks ONCE before parallel loop (avoid 100x redundant API calls)
            # Previously: called inside each worker → 100 API calls (32 workers racing on cache)
            # Now: called once here and passed to workers → 1 API call
            # Benefit: 99 redundant API calls eliminated per run
            #
            # NOTE (2026-01-19): Branch protection API requires admin perms and returns 403.
            # DO NOT use get_required_checks_for_base_ref() anymore - it doesn't work.
            # Instead: fetch an open PR targeting main and use its required checks as a proxy.
            # TODO: Unhardcode "ai-dynamo"/"dynamo" - should be class attributes or config
            required_names: List[str] = []
            try:
                # Get open PRs targeting main and try multiple until we find one with required checks
                # (Some PRs may be from forks or have different required check configurations)
                open_prs = self.github_client.get_open_prs(
                    owner="ai-dynamo",
                    repo="dynamo",
                    max_prs=20,
                )
                # Filter for PRs targeting main
                main_prs = [pr for pr in open_prs if pr.base_ref == "main"]

                # Try up to 5 PRs to find one with required checks
                for pr in main_prs[:5]:
                    required_set = self.github_client.get_required_checks(
                        owner="ai-dynamo",
                        repo="dynamo",
                        pr_number=pr.number
                    )
                    if required_set:
                        required_names = sorted(required_set)
                        break  # Found required checks, stop searching
            except (OSError, subprocess.SubprocessError):  # Network/subprocess errors only
                required_names = []  # Best-effort: continue even if this fails

            # Build trees in parallel using ThreadPoolExecutor
            max_workers = min(32, len(commit_data) or 1)
            github_results: Dict[str, Tuple[str, float]] = {}
            gitlab_results: Dict[str, Tuple[str, float]] = {}

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all GitHub tree builds
                github_futures = {executor.submit(build_github_tree, c): c for c in commit_data}
                # Submit all GitLab tree builds
                gitlab_futures = {executor.submit(build_gitlab_tree, c): c for c in commit_data}

                # Collect GitHub results
                for future in concurrent.futures.as_completed(github_futures):
                    sha_full, sha_short, tree_html, dt, timing_breakdown = future.result()
                    github_results[sha_full] = (tree_html, dt)
                    build_github_s += dt
                    slow_github.append((dt, sha_short or sha_full[:9]))
                    
                    # Accumulate timing breakdown
                    for key, value in timing_breakdown.items():
                        if key in timing_totals:
                            timing_totals[key] += value
                    
                    if not tree_html:
                        logger.warning(f"[collect_github_results] {sha_short} ({sha_full[:12]}): tree_html is EMPTY")

                # Collect GitLab results
                for future in concurrent.futures.as_completed(gitlab_futures):
                    sha_full, sha_short, tree_html, dt = future.result()
                    gitlab_results[sha_full] = (tree_html, dt)
                    build_gitlab_s += dt
                    slow_gitlab.append((dt, sha_short or sha_full[:9]))

            # Attach results to commit dictionaries
            logger.debug(f"[attach_results] Attaching tree HTML to {len(commit_data)} commits. github_results has {len(github_results)} entries")
            for c in commit_data:
                sha_full = str(c.get("sha_full", "") or "")
                sha_short = str(c.get("sha_short", "") or "")
                if sha_full in github_results:
                    tree_html = github_results[sha_full][0]
                    c["github_checks_tree_html"] = tree_html
                    if not tree_html:
                        logger.warning(f"[attach_results] {sha_short} ({sha_full[:12]}): Attaching EMPTY github_checks_tree_html")
                else:
                    logger.debug(f"[attach_results] {sha_short} ({sha_full[:12]}): NOT in github_results dict")
                if sha_full in gitlab_results:
                    c["gitlab_checks_tree_html"] = gitlab_results[sha_full][0]

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
            elapsed_s = max(0.0, time.monotonic() - float(generation_t0))

        # Use shared build_page_stats() for consistent statistics across all dashboards
        page_stats = build_page_stats(
            generation_time_secs=elapsed_s,
            github_api=self.github_client,
            max_github_api_calls=None,
            cache_only_mode=bool(self.github_client.cache_only_mode),
            cache_only_reason="",
            commits_shown=len(commit_data),
            gitlab_fetch_skip=self.gitlab_fetch_skip,
            gitlab_client=self.gitlab_client,
        )

        # NOTE: GitHub cache statistics (merge_dates, required_checks, actions_status) are now
        # handled by build_page_stats via the extra_cache_stats parameter (no longer duplicated here).

        # Include timing breakdown if available (best-effort).
        # Add phase timings with descriptions
        t = self._last_timings or {}
        timing_descriptions = {
            "cache_load": "Load cached data (merge dates, required checks, etc.)",
            "git_iter_commits": "Iterate git commits and extract metadata",
            "github_merge_dates": "Fetch GitHub PR merge dates",
            "github_required_checks": "Fetch required checks for PRs",
            "process_commits": "Process commit data and build structures",
            "cache_save": "Save data to cache files",
        }
        if isinstance(t, dict) and t:
            for k in ["cache_load", "git_iter_commits", "github_merge_dates", "github_required_checks", "process_commits", "cache_save"]:
                if k in t:
                    desc = timing_descriptions.get(k, f"{k.replace('_', ' ').title()} time")
                    page_stats.append((f"phase.{k}.secs", f"{float(t[k]):.2f}s", desc))

        # HTML rendering timing (use placeholders, patched after render completes)
        PH_BUILD = "__TIMING_HTML_BUILD_TREES__"
        PH_TPL = "__TIMING_HTML_TEMPLATE_RENDER__"
        PH_RENDER = "__TIMING_RENDER_HTML__"
        page_stats.append(("html.render.total_secs", PH_RENDER, "[WALL-CLOCK] Total HTML generation time (all phases)"))
        page_stats.append(("html.build_trees.total_secs", PH_BUILD, "[WALL-CLOCK] Build all CI trees (GitHub + GitLab, parallel with max 32 workers)"))
        page_stats.append(("html.build_trees.github_secs", "__TIMING_HTML_BUILD_TREES_GH__", "[AGGREGATE] GitHub tree building (sum of all parallel thread times - divide by ~32 for avg wall-clock)"))
        page_stats.append(("html.build_trees.gitlab_secs", "__TIMING_HTML_BUILD_TREES_GL__", "[AGGREGATE] GitLab tree building (sum of all parallel thread times - divide by ~32 for avg wall-clock)"))
        page_stats.append(("html.template.render_secs", PH_TPL, "[WALL-CLOCK] Jinja2 template rendering (single-threaded)"))

        # Top slow commits (helps pinpoint which commit(s) dominate tree-building)
        slow_github_sorted = sorted(slow_github, key=lambda x: -float(x[0]))[:5]
        slow_gitlab_sorted = sorted(slow_gitlab, key=lambda x: -float(x[0]))[:5]
        if slow_github_sorted:
            page_stats.append(("html.slowest_commits.github", ", ".join([f"{sha}={dt:.2f}s" for dt, sha in slow_github_sorted]), "Top 5 slowest commits (GitHub tree building)"))
        if slow_gitlab_sorted:
            page_stats.append(("html.slowest_commits.gitlab", ", ".join([f"{sha}={dt:.2f}s" for dt, sha in slow_gitlab_sorted]), "Top 5 slowest commits (GitLab tree building)"))

        # Detailed timing breakdown (per-operation breakdown within GitHub tree building)
        # NOTE: All timings are ACCUMULATED across all parallel threads (not wall-clock)
        timing_descriptions = {
            'duration_calculation': '[AGGREGATE] Calculate job durations from raw logs or API (cached via duration-cache.json)',
            'run_all_passes': '[AGGREGATE] YAML enrichment + tree processing (8 passes: steps, pytest, grouping, sorting)',
            'snippet_extraction': '[AGGREGATE] Extract error snippets from failed job logs (cached via snippet-cache.json)',
            'process_check_runs': '[AGGREGATE] Process all check runs (wall-clock per-commit, includes raw_log_materialize + duration_calculation + snippet_extraction + misc overhead)',
            'raw_log_materialize': '[AGGREGATE] Download/materialize raw log files from GitHub (or copy from global cache to page cache)',
            'build_ci_job_nodes': '[AGGREGATE] Build CIJobNode objects from check runs',
            'render_tree_divs': '[AGGREGATE] Render final HTML tree structure using template',
            'fetch_check_runs': '[AGGREGATE] Fetch check run data from GitHub API (cached via github-checks-cache.json)',
            'inject_placeholders': '[AGGREGATE] Inject missing required checks as placeholder nodes',
        }
        
        # Sort by time (descending) and add to page_stats
        sorted_timing = sorted(timing_totals.items(), key=lambda x: -x[1])
        for key, total_time in sorted_timing:
            if total_time > 0:  # Only show non-zero timings
                pct = (total_time / build_github_s * 100) if build_github_s > 0 else 0
                desc = timing_descriptions.get(key, f"{key.replace('_', ' ').title()}")
                page_stats.append((f"html.build_trees.github.{key}.secs", f"{total_time:.2f}s ({pct:.1f}%)", desc))
        
        # Add raw_log_materialize breakdown (if available)
        if hasattr(self.github_client, '_raw_log_timing_stats'):
            stats = self.github_client._raw_log_timing_stats
            if stats.get('count', 0) > 0:
                page_stats.append(("html.raw_log.breakdown.extract", f"{stats['extract']:.2f}s", "[raw_log_materialize] URL parsing and job ID extraction"))
                page_stats.append(("html.raw_log.breakdown.api", f"{stats['api']:.2f}s", "[raw_log_materialize] API calls (job details, status checks)"))
                page_stats.append(("html.raw_log.breakdown.path_setup", f"{stats['path_setup']:.2f}s", "[raw_log_materialize] Path setup and directory creation"))
                page_stats.append(("html.raw_log.breakdown.exists_check", f"{stats['exists_check']:.2f}s", "[raw_log_materialize] File existence checks"))
                page_stats.append(("html.raw_log.breakdown.symlink_check", f"{stats['symlink_check']:.2f}s", "[raw_log_materialize] Symlink resolution and validation"))
                page_stats.append(("html.raw_log.breakdown.calls", str(stats['count']), "[raw_log_materialize] Total number of materialize calls"))
        # Sort stats for readability (generation.total_secs first, then other keys grouped by prefix).
        gen = [stat for stat in page_stats if stat[0] == "generation.total_secs"]

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

        # Apply prefix-based grouping to all stats except generation.total_secs
        other = sorted([stat for stat in page_stats if stat[0] != "generation.total_secs"], key=prefix_sort_key)
        page_stats[:] = gen + other

        # Build tree time is known now.
        build_trees_s = float(render_t.as_dict(include_total=False).get("build_trees") or 0.0)

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
            branch_name=branch_name or "main",
            # Icons (shared look; legend/tooltips/status bar must match across dashboards)
            **ci_status_icon_context(),
            pass_plus_style=PASS_PLUS_STYLE,
            grafana_pr_url_template=GRAFANA_PR_URL_TEMPLATE,
        )
        tpl_render_s = max(0.0, time.monotonic() - t0_tpl)

        # Patch placeholders into the final HTML.
        rendered_html = rendered_html.replace(PH_BUILD, f"{build_trees_s:.2f}s")
        rendered_html = rendered_html.replace(PH_TPL, f"{tpl_render_s:.2f}s")
        rendered_html = rendered_html.replace(PH_RENDER, f"{(build_trees_s + tpl_render_s):.2f}s")
        rendered_html = rendered_html.replace("__TIMING_HTML_BUILD_TREES_GH__", f"{build_github_s:.2f}s")
        rendered_html = rendered_html.replace("__TIMING_HTML_BUILD_TREES_GL__", f"{build_gitlab_s:.2f}s")
        return rendered_html

    def _get_cached_gitlab_images_from_sha(self, commit_data: List[dict]) -> dict:
        """Get Docker images mapped by commit SHA using cache.
        
        Simplified logic:
        - If gitlab_fetch_skip=True: Only use cache
        - If gitlab_fetch_skip=False: Fetch tags for recent commits (within 8 hours) using binary search
        
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
        cache_path = resolve_cache_path(cache_file)
        cache0 = json.loads(cache_path.read_text() or "{}") if cache_path.exists() else {}
        if isinstance(cache0, dict):
            hit = sum(1 for sha in sha_full_list if sha in cache0)
            COMMIT_HISTORY_PERF_STATS.gitlab_cache_registry_images_hit += int(hit)
            COMMIT_HISTORY_PERF_STATS.gitlab_cache_registry_images_miss += int(max(0, len(sha_full_list) - hit))

        t0 = time.monotonic()
        
        # TODO: Unhardcode project_id="169905" (dl/ai-dynamo/dynamo) - should be config
        result = self.gitlab_client.get_cached_registry_images_for_shas(
            project_id="169905",  # dl/ai-dynamo/dynamo
            registry_id="85325",  # Main dynamo registry
            sha_list=sha_full_list,
            sha_to_datetime=sha_to_datetime,
            cache_file=cache_file,
            skip_fetch=self.gitlab_fetch_skip
        )
        COMMIT_HISTORY_PERF_STATS.gitlab_registry_images_total_secs += max(0.0, time.monotonic() - t0)
        
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
        cache_path = resolve_cache_path(cache_file)
        cache0 = json.loads(cache_path.read_text() or "{}") if cache_path.exists() else {}
        if isinstance(cache0, dict):
            hit = sum(1 for sha in sha_full_list if sha in cache0)
            COMMIT_HISTORY_PERF_STATS.gitlab_cache_pipeline_status_hit += int(hit)
            COMMIT_HISTORY_PERF_STATS.gitlab_cache_pipeline_status_miss += int(max(0, len(sha_full_list) - hit))

        t0 = time.monotonic()
        
        result = self.gitlab_client.get_cached_pipeline_status(sha_full_list, cache_file=cache_file, skip_fetch=self.gitlab_fetch_skip)
        COMMIT_HISTORY_PERF_STATS.gitlab_pipeline_status_total_secs += max(0.0, time.monotonic() - t0)
        
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
            cache_path = resolve_cache_path(cache_file)
            cache0 = json.loads(cache_path.read_text() or "{}") if cache_path.exists() else {}
            if isinstance(cache0, dict):
                hit = sum(1 for k in keys if k in cache0)
                COMMIT_HISTORY_PERF_STATS.gitlab_cache_pipeline_jobs_hit += int(hit)
                COMMIT_HISTORY_PERF_STATS.gitlab_cache_pipeline_jobs_miss += int(max(0, len(keys) - hit))
        except (ValueError, TypeError, KeyError) as e:
            # Best-effort: cache accounting is optional
            logger.debug(f"Failed to update cache stats: {e}")
            pass

        t0 = time.monotonic()

        result = self.gitlab_client.get_cached_pipeline_job_details(pipeline_ids, cache_file=cache_file, skip_fetch=self.gitlab_fetch_skip)
        COMMIT_HISTORY_PERF_STATS.gitlab_pipeline_jobs_total_secs += max(0.0, time.monotonic() - t0)

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

Environment Variables:
  GH_TOKEN / GITHUB_TOKEN
      GitHub personal access token (alternative to --github-token).
      Priority: --github-token > GH_TOKEN > GITHUB_TOKEN > ~/.config/gh/hosts.yml

  DYNAMO_UTILS_CACHE_DIR
      Override default cache directory (~/.cache/dynamo-utils)

  MAX_GITHUB_API_CALLS
      Can be set when using update_html_pages.sh to override --max-github-api-calls default

  MAX_COMMITS
      Can be set when using update_html_pages.sh to override --max-commits default
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
        '--parallel-workers',
        type=int,
        default=0,
        help='Number of worker processes to parallelize raw log snippet parsing (default: 0 = single process)'
    )
    parser.add_argument(
        '--enable-success-build-test-logs',
        action='store_true',
        help='Opt-in: also cache raw logs for successful *-build-test jobs so we can parse pytest slowest tests under "Run tests" (slower).'
    )
    parser.add_argument(
        '--run-verifier-pass',
        action='store_true',
        help='Enable verification passes (verify_job_details_pass, verify_tree_structure_pass) to validate tree structure and job details'
    )
    parser.add_argument(
        '--disable-snippet-cache-read',
        action='store_true',
        help='TESTING ONLY: ignore existing snippet cache contents (forces cache misses; useful for benchmarking --parallel-workers)'
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
        '--skip-gitlab-api',
        dest='gitlab_fetch_skip',
        action='store_true',
        help='Skip fetching GitLab registry data, use cached data only (much faster)'
    )
    # NOTE: Removed --max-github-fetch-commits.
    # GitHub fetch behavior is now governed by cache TTLs in `common.py`.
    parser.add_argument(
        '--github-token',
        help='GitHub personal access token (preferred). If omitted, we try ~/.config/github-token or ~/.config/gh/hosts.yml.'
    )
    parser.add_argument(
        '--allow-anonymous-github',
        action='store_true',
        help='Allow anonymous GitHub REST calls (60/hr core rate limit). By default we require auth to avoid rate limiting.'
    )
    parser.add_argument(
        '--max-github-api-calls',
        type=int,
        default=500,
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
    _ = prune_dashboard_raw_logs(page_root_dir=args.repo_path, max_age_days=90)
    # Also remove any partial/unverified raw logs (legacy cache artifacts, missing completed=true, etc).
    _ = prune_partial_raw_log_caches(page_root_dirs=[args.repo_path])

    # Create generator and run
    generator = CommitHistoryGenerator(
        repo_path=args.repo_path,
        verbose=args.verbose,
        debug=args.debug,
        gitlab_fetch_skip=bool(args.gitlab_fetch_skip),
        github_token=args.github_token,
        allow_anonymous_github=bool(args.allow_anonymous_github),
        max_github_api_calls=int(args.max_github_api_calls),
        parallel_workers=int(args.parallel_workers or 0),
        disable_snippet_cache_read=bool(args.disable_snippet_cache_read),
        enable_success_build_test_logs=bool(args.enable_success_build_test_logs),
        run_verifier_pass=bool(args.run_verifier_pass),
    )
    # Fail fast if exhausted; detailed stats are rendered into the HTML Statistics section.
    try:
        generator.github_client.check_core_rate_limit_or_raise()
    except RuntimeError as e:
        # Switch to cache-only mode (no new GitHub network calls).
        logger.warning(f"GitHub rate limit exceeded, switching to cache-only mode: {e}")
        generator.github_client.set_cache_only_mode(True)

    rc = 1
    try:
        rc = generator.show_commit_history(
        max_commits=args.max_commits,
        output_path=args.output,
        logs_dir=args.logs_dir,
        export_pipeline_pr_csv=args.export_pipeline_pr_csv,
    )
        # Debug: snippet cache analysis (avoid printing from library code; use logger).
        if bool(args.debug) and hasattr(generator, "_snippet_cache_debug"):

            log = logging.getLogger(__name__)
            debug = generator._snippet_cache_debug
            log.debug("SNIPPET CACHE DEBUG: hits=%d misses=%d", len(debug.get("hits", []) or []), len(debug.get("misses", []) or []))
            reason_counts = Counter((debug.get("reasons", {}) or {}).values())
            for reason, count in reason_counts.most_common():
                log.debug("  %s: %d", reason, count)
            misses = debug.get("misses", []) or []
            if misses:
                log.debug("First 10 missed keys:")
                for key in misses[:10]:
                    reason = (debug.get("reasons", {}) or {}).get(key, "unknown")
                    log.debug("  %s: %s", key, reason)
        
        return rc
    finally:
        # No stdout/stderr run-stats; the HTML Statistics section contains the breakdowns.
        pass


if __name__ == '__main__':
    sys.exit(main())
