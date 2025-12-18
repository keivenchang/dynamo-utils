#!/usr/bin/env python3
"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Dynamo Commit History Generator - Standalone Tool

Generates commit history with Composite Docker SHAs (CDS) and Docker image detection.
Can output to terminal or HTML format.
"""

import argparse
import git
import glob
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict

# Import utilities from common module
from common import DynamoRepositoryUtils, GitLabAPIClient, get_terminal_width

# Import Jinja2 for HTML template rendering
try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False

# Try to import pytz, but make it optional
try:
    import pytz
    HAS_PYTZ = True
except ImportError:
    HAS_PYTZ = False


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
        self.cache_file = self.repo_path / ".commit_history_cache.json"
        self.gitlab_client = GitLabAPIClient()  # Single instance for all GitLab operations

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('CommitHistoryGenerator')
        if self.debug:
            logger.setLevel(logging.DEBUG)
        elif self.verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

        if not logger.handlers:
            handler = logging.StreamHandler()
            if self.debug:
                handler.setLevel(logging.DEBUG)
            elif self.verbose:
                handler.setLevel(logging.INFO)
            else:
                handler.setLevel(logging.WARNING)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
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

    def show_commit_history(self, max_commits: int = 50, html_output: bool = False, output_path: Path = None, logs_dir: Path = None) -> int:
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

        # Load cache (.commit_history_cache.json in repo_path)
        # Format (new): {
        #   "<full_commit_sha>": {
        #     "composite_docker_sha": "746bc31d05b3",
        #     "author": "John Doe",
        #     "date": "2025-12-17 20:03:39",
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

        try:
            repo = git.Repo(self.repo_path)
            commits = list(repo.iter_commits('HEAD', max_count=max_commits))
            original_head = repo.head.commit.hexsha

            # Collect commit data
            commit_data = []
            cache_updated = False

            if not html_output:
                # Terminal output mode
                term_width = get_terminal_width(padding=2, default=118)
                sha_width = 10
                composite_width = 13
                date_width = 20
                author_width = 20
                separator_width = 3
                fixed_width = sha_width + composite_width + date_width + author_width + separator_width
                message_width = max(30, term_width - fixed_width)

                print(f"\nCommit History with Composite Docker SHAs")
                print(f"Repository: {self.repo_path}")
                print(f"Showing {len(commits)} most recent commits:\n")
                print(f"{'Commit SHA':<{sha_width}} {'Composite Docker SHA':<{composite_width}} {'Date':<{date_width}} {'Author':<{author_width}} Message")
                print("-" * term_width)

            try:
                for i, commit in enumerate(commits):
                    sha_short = commit.hexsha[:9]
                    sha_full = commit.hexsha

                    # Check cache first - handle both old format (string) and new format (dict)
                    cached_entry = cache.get(sha_full)
                    if cached_entry and isinstance(cached_entry, dict):
                        # New format: Full metadata cached
                        composite_sha = cached_entry['composite_docker_sha']
                        date_str = cached_entry['date']
                        author_name = cached_entry['author']
                        message_first_line = cached_entry['message']

                        if html_output:
                            full_message = cached_entry['full_message']
                            files_changed = cached_entry['stats']['files']
                            insertions = cached_entry['stats']['insertions']
                            deletions = cached_entry['stats']['deletions']
                            changed_files = cached_entry['changed_files']

                        self.logger.debug(f"Cache hit (full metadata) for {sha_short}: {composite_sha}")
                    else:
                        # Old format or cache miss: Need to fetch from git
                        date_str = commit.committed_datetime.strftime('%Y-%m-%d %H:%M:%S')
                        author_name = commit.author.name
                        message_first_line = commit.message.strip().split('\n')[0]

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

            finally:
                # Restore original HEAD
                repo.git.checkout(original_head)
                if not html_output:
                    print(f"\nRestored HEAD to {original_head[:9]}")

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

                html_content = self._generate_commit_history_html(commit_data, logs_dir, output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(html_content)
                print(f"\nHTML report generated: {output_path}")
                print(f"Restored HEAD to {original_head[:9]}")

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
                repo.git.checkout(original_head)
            except:
                pass
            return 1
        except Exception as e:
            self.logger.error(f"Failed to get commit history: {e}")
            return 1

    def _generate_commit_history_html(self, commit_data: List[dict], logs_dir: Path, output_path: Path) -> str:
        """Generate HTML report for commit history with Docker image detection

        Args:
            commit_data: List of commit dictionaries with sha_short, sha_full, composite_sha, date, author, message
            logs_dir: Path to logs directory for build reports
            output_path: Path where the HTML file will be written (used for relative path calculation)

        Returns:
            HTML content as string
        """
        if not HAS_JINJA2:
            raise ImportError("jinja2 is required for HTML generation. Install with: pip install jinja2")
        
        # Get local Docker images containing SHAs
        docker_images = self._get_local_docker_images_by_sha([c['sha_short'] for c in commit_data])

        # Get GitLab container registry images for commits (with caching)
        gitlab_images_raw = self._get_cached_gitlab_images_from_sha(commit_data)

        # Get GitLab CI pipeline statuses
        gitlab_pipelines = self._get_gitlab_pipeline_statuses([c['sha_full'] for c in commit_data])

        # Get GitLab CI pipeline job counts
        pipeline_ids = [p['id'] for p in gitlab_pipelines.values() if p and 'id' in p]
        pipeline_job_counts = {}
        if pipeline_ids:
            pipeline_job_counts = self._get_gitlab_pipeline_job_counts(pipeline_ids)

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
                        if HAS_PYTZ:
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
        
        # Process commit messages for PR links and assign colors
        bg_colors = [
            '#bbccee',  # Light blue
            '#cceeff',  # Pale cyan
            '#ccddaa',  # Light green-yellow
            '#eeeebb',  # Light yellow
            '#ffcccc',  # Light pink
            '#dddddd',  # Light gray
            '#ffe5b4',  # Light peach
            '#e7d4f7',  # Light lavender
        ]

        # Build deterministic CDS-to-color mapping
        # Same CDS always gets same color across all HTML generations
        unique_cds = []
        seen_cds = set()
        for commit in commit_data:
            cds = commit['composite_sha']
            if cds not in seen_cds:
                unique_cds.append(cds)
                seen_cds.add(cds)

        cds_to_color = {}
        for i, cds in enumerate(unique_cds):
            cds_to_color[cds] = bg_colors[i % len(bg_colors)]

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

            # Assign color deterministically based on Composite Docker SHA (CDS)
            composite_sha = commit['composite_sha']
            commit['composite_bg_color'] = cds_to_color[composite_sha]
        
        # Build log paths dictionary and status indicators
        log_paths = {}  # Maps sha_short to list of (date, path) tuples
        composite_to_status = {}  # Maps composite_sha to status (with priority: failed > building > success)
        composite_to_commits = {}  # Maps composite_sha to list of commit SHAs

        # Status priority for conflict resolution (higher number = higher priority)
        status_priority = {'unknown': 0, 'success': 1, 'building': 2, 'failed': 3}

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

                # Determine build status by traversing build history chronologically
                # Traverse from oldest to newest, updating status as we go
                log_dir = log_path.parent
                all_status_files = []

                # Collect all status files for this SHA (from all dates)
                for status_suffix in ['RUNNING', 'FAIL', 'PASS']:
                    pattern = str(log_dir / f"*.{sha_short}.*.{status_suffix}")
                    all_status_files.extend(glob.glob(pattern))

                status = 'unknown'  # Default status
                if all_status_files:
                    # Extract dates from filenames (format: YYYY-MM-DD.sha.task.STATUS)
                    # Group files by date for chronological traversal
                    from collections import defaultdict
                    files_by_date = defaultdict(list)
                    for f in all_status_files:
                        filename = Path(f).name
                        date_part = filename.split('.')[0]  # Extract YYYY-MM-DD
                        files_by_date[date_part].append(f)

                    # Traverse dates chronologically (oldest to newest)
                    for date in sorted(files_by_date.keys()):
                        date_files = files_by_date[date]

                        # Check status for this date's build run
                        running_files = [f for f in date_files if f.endswith('.RUNNING')]
                        fail_files = [f for f in date_files if f.endswith('.FAIL')]
                        pass_files = [f for f in date_files if f.endswith('.PASS')]

                        # Update status based on this build run (priority: FAIL > RUNNING > PASS)
                        if fail_files:
                            status = 'failed'
                        elif running_files:
                            status = 'building'
                        elif pass_files:
                            status = 'success'
                        # If this date has no files, keep previous status

                # Update composite SHA status with priority (failed > building > success)
                if composite_sha not in composite_to_status:
                    composite_to_status[composite_sha] = status
                else:
                    # If new status has higher priority, replace it
                    if status_priority[status] > status_priority[composite_to_status[composite_sha]]:
                        composite_to_status[composite_sha] = status
            else:
                # No report yet, status unknown
                if composite_sha not in composite_to_status:
                    composite_to_status[composite_sha] = 'unknown'
                # Don't override existing status if we have no information

        # Pass 2: Assign status to all commits based on composite SHA
        # Commits with logs get regular status, commits without logs get inherited status
        build_status = {}
        for commit in commit_data:
            sha_short = commit['sha_short']
            composite_sha = commit['composite_sha']

            if composite_sha in composite_to_status:
                # Check if this commit has logs (not inherited)
                has_logs = sha_short in log_paths
                build_status[sha_short] = {
                    'status': composite_to_status[composite_sha],
                    'inherited': not has_logs
                }
        
        # Generate timestamp
        if HAS_PYTZ:
            pdt = pytz.timezone('America/Los_Angeles')
            generated_time = datetime.now(pdt).strftime('%Y-%m-%d %H:%M:%S %Z')
        else:
            generated_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
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
            pipeline_job_counts=pipeline_job_counts,
            log_paths=log_paths,
            build_status=build_status,
            generated_time=generated_time
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
        cache_file = str(self.repo_path / ".gitlab_commit_sha_cache.json")
        
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

        Cache file format (.gitlab_pipeline_status_cache.json):
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
        cache_file = str(self.repo_path / ".gitlab_pipeline_status_cache.json")

        self.logger.debug(f"Getting pipeline status for {len(sha_full_list)} commits")
        
        result = self.gitlab_client.get_cached_pipeline_status(sha_full_list, cache_file=cache_file, skip_fetch=self.skip_gitlab_fetch)
        
        cached_count = sum(1 for v in result.values() if v is not None)
        self.logger.debug(f"Found pipeline status for {cached_count}/{len(sha_full_list)} commits")

        return result

    def _get_gitlab_pipeline_job_counts(self, pipeline_ids: List[int]) -> dict:
        """Get GitLab CI pipeline job counts using the centralized cache.

        Cache file format (.gitlab_pipeline_jobs_cache.json):
            {
                "40118215": {
                    "counts": {
                        "success": 15,
                        "failed": 8,
                        "running": 0,
                        "pending": 0
                    },
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
            - fetched_at: ISO 8601 timestamp when this data was fetched

        Note: Completed pipelines (running=0, pending=0) are cached forever.
              Active pipelines are refetched if older than 30 minutes.

        Args:
            pipeline_ids: List of pipeline IDs

        Returns:
            Dictionary mapping pipeline ID to job counts dict with 'success', 'failed', 'running', 'pending'
        """
        cache_file = str(self.repo_path / ".gitlab_pipeline_jobs_cache.json")

        self.logger.debug(f"Getting job counts for {len(pipeline_ids)} pipelines")

        result = self.gitlab_client.get_cached_pipeline_job_counts(pipeline_ids, cache_file=cache_file, skip_fetch=self.skip_gitlab_fetch)

        cached_count = sum(1 for v in result.values() if v is not None)
        self.logger.debug(f"Found job counts for {cached_count}/{len(pipeline_ids)} pipelines")

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
        logs_dir=args.logs_dir
    )


if __name__ == '__main__':
    sys.exit(main())
