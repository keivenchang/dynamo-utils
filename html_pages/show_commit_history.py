#!/usr/bin/env python3
"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Dynamo Commit History Generator - Standalone Tool

Generates commit history with Composite Docker SHAs (CDS) and Docker image detection.
Can output to terminal or HTML format.
"""

import argparse
try:
    import git  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    git = None  # type: ignore[assignment]
import glob
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set

# Ensure we can import sibling utilities (common.py) from the parent dynamo-utils directory
_THIS_DIR = Path(__file__).resolve().parent
_UTILS_DIR = _THIS_DIR.parent
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Shared UI snippets (keep styling consistent with show_dynamo_branches.py)
from html_ui import GH_STATUS_TOOLTIP_CSS, GH_STATUS_TOOLTIP_JS, PASS_PLUS_STYLE

# Import utilities from common module
from common import (
    DynamoRepositoryUtils,
    GitLabAPIClient,
    GitHubAPIClient,
    get_terminal_width,
    dynamo_utils_cache_dir,
    MARKER_RUNNING,
    MARKER_PASSED,
    MARKER_FAILED,
    MARKER_KILLED,
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

# Try to import pytz, but make it optional
try:
    import pytz
    HAS_PYTZ = True
except ImportError:
    HAS_PYTZ = False
    pytz = None  # type: ignore[assignment]

# Constants for build status values
STATUS_UNKNOWN = 'unknown'
STATUS_SUCCESS = 'success'
STATUS_FAILED = 'failed'
STATUS_BUILDING = 'building'

def _normalize_check_name(name: str) -> str:
    """Normalize check names for robust comparison."""
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def _is_required_check_name(check_name: str, required_names_normalized: Set[str]) -> bool:
    """Classify whether a check run is required (blocking) vs optional (informational).

    Primary source of truth: branch-protection required checks from `gh pr checks --required`,
    passed in via `required_names_normalized`.

    Fallback/overrides: hard-coded patterns for known blocking checks (user-provided).
    """
    name_norm = _normalize_check_name(check_name)
    if not name_norm:
        return False
    if name_norm in required_names_normalized:
        return True

    # Fallback patterns (case-insensitive via normalized string).
    # These cover common "must pass" checks even when we can't resolve PR required checks.
    mandatory_patterns = [
        r"^build and test - dynamo$",
        r"check for broken markdown links",
        r"docs link",  # e.g. "Docs Link check"
        r"\blychee\b",
        r"\b(vllm|sglang|trtllm)\s*\(amd64\)\b",
        r"rust[- ]pre[- ]merge",
    ]
    return any(re.search(pat, name_norm) for pat in mandatory_patterns)


class CommitHistoryGenerator:
    """Generate commit history with Composite Docker SHAs (CDS) and Docker images"""

    def __init__(self, repo_path: Path, verbose: bool = False, debug: bool = False, skip_gitlab_fetch: bool = False):
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
        self.gitlab_client = GitLabAPIClient()  # Single instance for all GitLab operations
        self.github_client = GitHubAPIClient()  # Single instance for all GitHub operations

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
        
        return logger

    def show_commit_history(
        self,
        max_commits: int = 50,
        html_output: bool = False,
        output_path: Optional[Path] = None,
        logs_dir: Optional[Path] = None,
        export_pipeline_pr_csv: Optional[Path] = None,
    ) -> int:
        """Show recent commit history with Composite Docker SHAs (CDS)

        Args:
            max_commits: Maximum number of commits to show
            html_output: Generate HTML output instead of terminal output
            output_path: Path for HTML output file (optional, auto-detected if not provided)
            logs_dir: Path to logs directory for build reports (optional, defaults to repo_path/logs)

        Returns:
            Exit code (0 for success, 1 for failure)
        """
        # Initialize repo utils for this operation
        repo_utils = DynamoRepositoryUtils(self.repo_path, dry_run=False, verbose=self.verbose)

        # Load cache (.cache/commit_history.json in repo_path)
        # Format (new): {
        #   "<full_commit_sha>": {
        #     "composite_docker_sha": "746bc31d05b3",
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
        # Format (old, backward compatible): {
        #   "<full_commit_sha>": "<composite_docker_sha>",
        #   ...
        # }
        # Fields:
        #   - composite_docker_sha: 12-character hash of container/ directory contents
        #   - author: Commit author name
        #   - date: Commit timestamp (YYYY-MM-DD HH:MM:SS)
        #   - merge_date: When MR was merged (YYYY-MM-DD HH:MM:SS), null if not found/merged
        #   - message: First line of commit message
        #   - full_message: Full commit message
        #   - stats: Dict with files, insertions, deletions counts
        #   - changed_files: List of file paths changed in this commit
        cache = {}
        if self.cache_file.exists():
            try:
                cache = json.loads(self.cache_file.read_text())
                self.logger.debug(f"Loaded cache with {len(cache)} entries")
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")

        repo = None
        # IMPORTANT: restore to the original ref (branch name when possible).
        # Restoring to a SHA (repo.head.commit.hexsha) leaves the repo in detached HEAD.
        original_ref = None
        try:
            if git is None:
                raise ImportError("GitPython is required. Install with: pip install gitpython")
            repo = git.Repo(self.repo_path)
            commits = list(repo.iter_commits('HEAD', max_count=max_commits))
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

            # Collect PR numbers for commits (cheap; used for HTML and for pipeline->PR exports).
            pr_to_merge_date: Dict[int, Optional[str]] = {}
            sha_to_pr_number: Dict[str, int] = {}
            pr_to_required_checks: Dict[int, List[str]] = {}
            pr_numbers: List[int] = []
            for commit in commits:
                message = commit.message.strip().split('\n')[0]
                pr_num = GitLabAPIClient.parse_mr_number_from_message(message)
                if pr_num:
                    pr_numbers.append(pr_num)
                    sha_to_pr_number[commit.hexsha] = pr_num

            if html_output:
                if pr_numbers and not self.skip_gitlab_fetch:
                    # Batch fetch merge dates for all PRs (GitHub)
                    self.logger.info(f"Fetching merge dates for {len(pr_numbers)} PRs...")
                    pr_to_merge_date = self.github_client.get_cached_pr_merge_dates(
                        pr_numbers,
                        cache_file="github_pr_merge_dates.json"
                    )
                    self.logger.info(f"Got merge dates for {sum(1 for v in pr_to_merge_date.values() if v)} PRs")

                if pr_numbers:
                    # Batch fetch required checks for all PRs (branch protection required checks).
                    # If skip_gitlab_fetch=True, we still read from cache (skip_fetch=True).
                    self.logger.info(f"Fetching required checks for {len(pr_numbers)} PRs...")
                    pr_to_required_checks = self.github_client.get_cached_required_checks(
                        pr_numbers,
                        owner="ai-dynamo",
                        repo="dynamo",
                        cache_file="github_required_checks.json",
                        skip_fetch=self.skip_gitlab_fetch,
                    )
                    self.logger.info("Got required-checks metadata for "
                                     f"{sum(1 for v in pr_to_required_checks.values() if v)}/{len(set(pr_numbers))} PRs")

            # Collect commit data
            commit_data = []
            cache_updated = False
            sha_to_message_first_line: Dict[str, str] = {}

            # Terminal width settings (initialize unconditionally so static analyzers don't complain)
            term_width = get_terminal_width(padding=2, default=118)
            sha_width = 10
            composite_width = 13
            date_width = 20
            author_width = 20
            separator_width = 3
            fixed_width = sha_width + composite_width + date_width + author_width + separator_width
            message_width = max(30, term_width - fixed_width)

            if not html_output:
                # Terminal output mode

                print(f"\nCommit History with Composite Docker SHAs")
                print(f"Repository: {self.repo_path}")
                print(f"Showing {len(commits)} most recent commits:\n")
                print(f"{'Commit SHA':<{sha_width}} {'Composite Docker SHA':<{composite_width}} {'Date':<{date_width}} {'Author':<{author_width}} Message")
                print("-" * term_width)

            try:
                for i, commit in enumerate(commits):
                    sha_short = commit.hexsha[:9]
                    sha_full = commit.hexsha

                    # Defaults to keep static analyzers happy (only used in HTML mode)
                    full_message = ""
                    files_changed = 0
                    insertions = 0
                    deletions = 0
                    changed_files: List[str] = []

                    # Check cache first - handle both old format (string) and new format (dict)
                    cached_entry = cache.get(sha_full)
                    merge_date = None  # Initialize merge_date

                    if cached_entry and isinstance(cached_entry, dict):
                        # New format: Full metadata cached
                        composite_sha = cached_entry['composite_docker_sha']
                        date_str = cached_entry['date']
                        author_name = cached_entry['author']
                        message_first_line = cached_entry['message']
                        merge_date = cached_entry.get('merge_date')  # May be None if not in cache

                        if html_output:
                            full_message = cached_entry['full_message']
                            files_changed = cached_entry['stats']['files']
                            insertions = cached_entry['stats']['insertions']
                            deletions = cached_entry['stats']['deletions']
                            changed_files = cached_entry['changed_files']

                        self.logger.debug(f"Cache hit (full metadata) for {sha_short}: {composite_sha}")

                        # Update merge_date from batch fetch if not in cache
                        if merge_date is None and html_output:
                            pr_number = GitLabAPIClient.parse_mr_number_from_message(message_first_line)
                            if pr_number and pr_number in pr_to_merge_date:
                                merge_date = pr_to_merge_date[pr_number]
                                if merge_date:
                                    self.logger.debug(f"Got merge date for PR {pr_number}: {merge_date}")
                                    # Update cache with merge_date
                                    cached_entry['merge_date'] = merge_date
                                    cache_updated = True
                    else:
                        # Old format or cache miss: Need to fetch from git
                        # Convert commit time from UTC to Pacific time
                        from zoneinfo import ZoneInfo
                        commit_dt_pacific = commit.committed_datetime.astimezone(ZoneInfo('America/Los_Angeles'))
                        date_str = commit_dt_pacific.strftime('%Y-%m-%d %H:%M:%S')
                        author_name = commit.author.name
                        message_first_line = commit.message.strip().split('\n')[0]

                        # Get merge_date from batch fetch if HTML mode
                        if html_output:
                            pr_number = GitLabAPIClient.parse_mr_number_from_message(message_first_line)
                            if pr_number and pr_number in pr_to_merge_date:
                                merge_date = pr_to_merge_date[pr_number]
                                if merge_date:
                                    self.logger.debug(f"Got merge date for PR {pr_number}: {merge_date}")

                        # Check if we have old-format composite SHA cached
                        if cached_entry and isinstance(cached_entry, str):
                            composite_sha = cached_entry
                            self.logger.debug(f"Cache hit (old format) for {sha_short}: {composite_sha}")
                            need_checkout = False
                        else:
                            # Need to calculate composite SHA
                            need_checkout = True
                            try:
                                repo.git.checkout(commit.hexsha)
                                composite_sha = repo_utils.generate_composite_sha()
                                self.logger.debug(f"Calculated {sha_short}: {composite_sha}")
                            except Exception as e:
                                composite_sha = "ERROR"
                                self.logger.error(f"Failed to calculate composite SHA for {sha_short}: {e}")

                        if html_output:
                            # Get commit stats (expensive operation)
                            stats = commit.stats.total
                            files_changed = stats['files']
                            insertions = stats['insertions']
                            deletions = stats['deletions']

                            # Get full commit message
                            full_message = commit.message.strip()

                            # Get list of changed files
                            changed_files = list(commit.stats.files.keys())

                            # Update cache with full metadata (new format)
                            cache[sha_full] = {
                                'composite_docker_sha': composite_sha,
                                'author': author_name,
                                'date': date_str,
                                'merge_date': merge_date,  # May be None if not found
                                'message': message_first_line,
                                'full_message': full_message,
                                'stats': {
                                    'files': files_changed,
                                    'insertions': insertions,
                                    'deletions': deletions
                                },
                                'changed_files': changed_files
                            }
                            cache_updated = True
                        else:
                            # Terminal mode: Only cache composite SHA if calculated
                            if need_checkout:
                                cache[sha_full] = composite_sha
                                cache_updated = True

                    if not html_output:
                        # Terminal: show progress or final result
                        author_str = author_name[:author_width-1]
                        message_str = message_first_line[:message_width]
                        if i == 0 or cached_entry:
                            # First commit or cached: show immediately
                            print(f"{sha_short:<{sha_width}} {composite_sha:<{composite_width}} {date_str:<{date_width}} {author_str:<{author_width}} {message_str}")
                        else:
                            # Calculating: show placeholder then overwrite
                            print(f"\r{sha_short:<{sha_width}} {composite_sha:<{composite_width}} {date_str:<{date_width}} {author_str:<{author_width}} {message_str}")

                    if html_output:
                        # Collect data for HTML generation
                        commit_data.append({
                            'sha_short': sha_short,
                            'sha_full': sha_full,
                            'composite_sha': composite_sha,
                            'date': date_str,
                            'merge_date': merge_date,  # May be None if not found
                            'committed_datetime': commit.committed_datetime,  # Store datetime object for time-based filtering
                            'author': author_name,
                            'message': message_first_line,
                            'full_message': full_message,
                            'files_changed': files_changed,
                            'insertions': insertions,
                            'deletions': deletions,
                            'changed_files': changed_files
                        })
                        self.logger.debug(f"Processed commit {i+1}/{len(commits)}: {sha_short}")

                    # Always keep a minimal per-SHA message map (used for exports)
                    sha_to_message_first_line[sha_full] = message_first_line

            finally:
                # Restore original ref (best-effort)
                if repo is not None and original_ref is not None:
                    try:
                        repo.git.checkout(original_ref)
                        if not html_output:
                            print(f"\nRestored HEAD to {original_ref}")
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
                    print(f"\nExported pipeline→PR mapping: {export_pipeline_pr_csv}")
                except Exception as e:
                    self.logger.warning(f"Failed to export pipeline→PR CSV: {e}")

            # Generate HTML if requested
            if html_output:
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
                    # Auto-detect: Write to logs directory within the repo (or current directory if logs doesn't exist)
                    logs_dir_temp = self.repo_path / "logs"
                    if logs_dir_temp.exists():
                        output_path = logs_dir_temp / "commit-history.html"
                    else:
                        output_path = Path("commit-history.html")

                html_content = self._generate_commit_history_html(
                    commit_data,
                    logs_dir,
                    output_path,
                    sha_to_pr_number=sha_to_pr_number,
                    pr_to_required_checks=pr_to_required_checks,
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(html_content)
                print(f"\nHTML report generated: {output_path}")
                if original_ref is not None:
                    print(f"Restored HEAD to {original_ref}")

            # Save cache if updated
            if cache_updated:
                try:
                    self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                    self.cache_file.write_text(json.dumps(cache, indent=2))
                    self.logger.debug(f"Cache saved with {len(cache)} entries")
                except Exception as e:
                    self.logger.warning(f"Failed to save cache: {e}")

            return 0
        except KeyboardInterrupt:
            print("\n\nOperation interrupted by user")
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

        # Get GitHub Actions check status for commits
        github_actions_status = self.github_client.get_github_actions_status(
            owner='ai-dynamo',
            repo='dynamo',
            sha_list=[c['sha_full'] for c in commit_data],
            cache_file="github_actions_status.json",
            skip_fetch=self.skip_gitlab_fetch  # Reuse the skip_fetch flag
        )

        # Annotate GitHub check runs with "is_required" using PR required-checks + fallback patterns.
        pr_to_required_checks = pr_to_required_checks or {}
        for sha_full, gha in (github_actions_status or {}).items():
            if not gha or not gha.get('check_runs'):
                continue
            pr_number = sha_to_pr_number.get(sha_full)
            required_list = pr_to_required_checks.get(pr_number, []) if pr_number else []
            required_norm = {_normalize_check_name(n) for n in (required_list or []) if n}
            for check in gha.get('check_runs', []):
                try:
                    check_name = check.get('name', '')
                    check['is_required'] = _is_required_check_name(check_name, required_norm)
                except Exception:
                    # Best-effort; never break page generation.
                    check['is_required'] = False

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
                
                # Format created timestamp (ISO 8601 to readable, convert to PDT)
                created_display = "N/A"
                if created_at:
                    try:
                        # Parse UTC timestamp
                        dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        # Convert to PDT if pytz is available
                        if HAS_PYTZ and pytz is not None:
                            pdt = pytz.timezone('America/Los_Angeles')
                            dt = dt.astimezone(pdt)
                        created_display = dt.strftime('%Y-%m-%d %H:%M')
                    except:
                        created_display = created_at[:16] if len(created_at) >= 16 else created_at
                
                formatted_imgs.append({
                    **img,
                    'size_display': size_display,
                    'created_display': created_display
                })
            
            gitlab_images[sha_full] = formatted_imgs
        
        # Process commit messages for PR links and assign CDS colors
        #
        # CDS badge colors: alternate dark/light for high contrast. We do this per *unique CDS*
        # encountered in the commit list, so the page alternates visually while keeping a stable
        # color for each CDS within the page.
        # Softer greys (requested): alternating light grey / dark grey.
        # Keep text readable (white on dark, dark on light).
        dark_cds_bg = "#4b5563"   # gray-600
        light_cds_bg = "#e5e7eb"  # gray-200
        dark_cds_fg = "#ffffff"
        light_cds_fg = "#111827"
        unique_cds = []
        seen_cds = set()
        for commit in commit_data:
            cds = commit['composite_sha']
            if cds not in seen_cds:
                unique_cds.append(cds)
                seen_cds.add(cds)

        cds_to_color: Dict[str, str] = {}
        cds_to_text_color: Dict[str, str] = {}
        for i, cds in enumerate(unique_cds):
            if i % 2 == 0:
                cds_to_color[cds] = dark_cds_bg
                cds_to_text_color[cds] = dark_cds_fg
            else:
                cds_to_color[cds] = light_cds_bg
                cds_to_text_color[cds] = light_cds_fg

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

            # Assign color deterministically based on Composite Docker SHA (CDS)
            composite_sha = commit['composite_sha']
            commit['composite_bg_color'] = cds_to_color[composite_sha]
            commit['composite_text_color'] = cds_to_text_color[composite_sha]

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
        log_paths = {}  # Maps sha_short to list of (date, path) tuples
        composite_to_status = {}  # Maps composite_sha to status (with priority: building > failed > success)
        commit_to_status = {}  # Maps commit sha_short to its own build status
        composite_to_commits = {}  # Maps composite_sha to list of commit SHAs

        # Status priority for conflict resolution (higher number = higher priority)
        # Building > Failed > Success (if any build is still running, show as building)
        status_priority = {STATUS_UNKNOWN: 0, STATUS_SUCCESS: 1, STATUS_FAILED: 2, STATUS_BUILDING: 3}

        # Pass 1: Collect all statuses and map composite SHA to commits
        for commit in commit_data:
            sha_short = commit['sha_short']
            composite_sha = commit['composite_sha']

            # Track which commits have this composite SHA
            if composite_sha not in composite_to_commits:
                composite_to_commits[composite_sha] = []
            composite_to_commits[composite_sha].append(sha_short)

            # Search for ALL build logs (independent of Docker image existence)
            log_filename = f"*.{sha_short}.report.html"
            search_pattern = str(logs_dir / "*" / log_filename)
            matching_logs = glob.glob(search_pattern)

            if matching_logs:
                # Store all build attempts with dates
                log_paths[sha_short] = []
                for log_file in sorted(matching_logs):
                    log_path = Path(log_file).resolve()
                    # Extract date from filename (format: YYYY-MM-DD.sha.report.html)
                    date_str = log_path.name.split('.')[0]
                    try:
                        # Calculate relative path from output HTML file to log file
                        output_dir = output_path.resolve().parent
                        relative_path = os.path.relpath(log_path, output_dir)
                        log_paths[sha_short].append((date_str, relative_path))
                    except ValueError:
                        log_paths[sha_short].append((date_str, str(log_path)))

                # Use the most recent log for status determination
                log_path = Path(sorted(matching_logs)[-1])

                # Determine build status using only the LATEST build date
                log_dir = log_path.parent
                all_status_files = []

                # Collect all status files for this SHA (from all dates)
                for status_suffix in [MARKER_RUNNING, MARKER_FAILED, MARKER_PASSED]:
                    pattern = str(log_dir / f"*.{sha_short}.*.{status_suffix}")
                    all_status_files.extend(glob.glob(pattern))

                status = STATUS_UNKNOWN  # Default status
                if all_status_files:
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

                # Store per-commit status
                commit_to_status[sha_short] = status

                # Also update composite SHA status with priority (building > failed > success)
                # This is used for commits without their own logs (inherited status)
                if composite_sha not in composite_to_status:
                    composite_to_status[composite_sha] = status
                else:
                    # If new status has higher priority, replace it
                    if status_priority[status] > status_priority[composite_to_status[composite_sha]]:
                        composite_to_status[composite_sha] = status
            else:
                # No report yet, status unknown
                if composite_sha not in composite_to_status:
                    composite_to_status[composite_sha] = STATUS_UNKNOWN
                # Don't override existing status if we have no information

        # Pass 2: Assign status to all commits
        # Commits with logs get their own status, commits without logs inherit from CDS
        build_status = {}
        for commit in commit_data:
            sha_short = commit['sha_short']
            sha_full = commit['sha_full']
            composite_sha = commit['composite_sha']

            # Use per-commit status if available, otherwise inherit from CDS
            if sha_short in commit_to_status:
                # This commit has its own build logs
                build_status[sha_short] = {
                    'status': commit_to_status[sha_short],
                    'inherited': False
                }
            elif composite_sha in composite_to_status:
                # Inherit status from CDS
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
        
        # Generate timestamp
        if HAS_PYTZ:
            if HAS_PYTZ and pytz is not None:
                pdt = pytz.timezone('America/Los_Angeles')
                generated_time = datetime.now(pdt).strftime('%Y-%m-%d %H:%M:%S %Z')
            else:
                generated_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        else:
            generated_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
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

            stats = {
                'success_required': 0,
                'success_optional': 0,
                'failure_required': 0,
                'failure_optional': 0,
                'in_progress_required': 0,
                'in_progress_optional': 0,
                'pending': 0,
                'cancelled': 0,
                'other': 0,
            }
            for check in gha_status.get('check_runs', []):
                conclusion = check.get('conclusion')
                status = check.get('status')
                is_required = bool(check.get('is_required', False))

                if conclusion == 'success':
                    if is_required:
                        stats['success_required'] += 1
                    else:
                        stats['success_optional'] += 1
                elif conclusion in ('failure', 'timed_out', 'action_required'):
                    if is_required:
                        stats['failure_required'] += 1
                    else:
                        stats['failure_optional'] += 1
                elif status == 'in_progress':
                    if is_required:
                        stats['in_progress_required'] += 1
                    else:
                        stats['in_progress_optional'] += 1
                elif status in ('queued', 'pending'):
                    stats['pending'] += 1
                elif conclusion == 'cancelled':
                    stats['cancelled'] += 1
                else:
                    stats['other'] += 1

            stats['total'] = sum(stats.values())
            gha_per_commit_stats[sha_full] = stats

        # Render template
        template_dir = Path(__file__).parent
        env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )
        template = env.get_template('show_commit_history.j2')

        return template.render(
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
            # Shared UI snippets
            gh_status_tooltip_css=GH_STATUS_TOOLTIP_CSS,
            gh_status_tooltip_js=GH_STATUS_TOOLTIP_JS,
            pass_plus_style=PASS_PLUS_STYLE,
        )

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
        
        result = self.gitlab_client.get_cached_registry_images_for_shas(
            project_id="169905",  # dl/ai-dynamo/dynamo
            registry_id="85325",  # Main dynamo registry
            sha_list=sha_full_list,
            sha_to_datetime=sha_to_datetime,
            cache_file=cache_file,
            skip_fetch=self.skip_gitlab_fetch
        )
        
        cached_count = sum(1 for v in result.values() if v)
        self.logger.debug(f"Found Docker images for {cached_count}/{len(sha_full_list)} SHAs")
        
        return result

    def _get_gitlab_pipeline_statuses(self, sha_full_list: List[str]) -> dict:
        """Get GitLab CI pipeline status for commits using the centralized cache.

        Cache file format (.cache/gitlab_pipeline_status.json):
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
        
        result = self.gitlab_client.get_cached_pipeline_status(sha_full_list, cache_file=cache_file, skip_fetch=self.skip_gitlab_fetch)
        
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

        result = self.gitlab_client.get_cached_pipeline_job_details(pipeline_ids, cache_file=cache_file, skip_fetch=self.skip_gitlab_fetch)

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
  # Show last 50 commits in terminal
  %(prog)s

  # Show last 20 commits with verbose output
  %(prog)s --max-commits 20 --verbose

  # Generate HTML report
  %(prog)s --html

  # Use custom repository path
  %(prog)s --repo-path /path/to/dynamo_ci --html
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
        '--html',
        action='store_true',
        help='Generate HTML output instead of terminal output'
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
        print(f"Error: Repository path does not exist: {args.repo_path}")
        return 1

    if not (args.repo_path / '.git').exists():
        print(f"Error: Not a git repository: {args.repo_path}")
        return 1

    # Create generator and run
    generator = CommitHistoryGenerator(
        repo_path=args.repo_path,
        verbose=args.verbose,
        debug=args.debug,
        skip_gitlab_fetch=args.skip_gitlab_fetch
    )

    return generator.show_commit_history(
        max_commits=args.max_commits,
        html_output=args.html,
        output_path=args.output,
        logs_dir=args.logs_dir,
        export_pipeline_pr_csv=args.export_pipeline_pr_csv,
    )


if __name__ == '__main__':
    sys.exit(main())
