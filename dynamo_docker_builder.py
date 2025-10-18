#!/usr/bin/env python3
"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

DynamoDockerBuilder - Automated Docker Build and Test System
Tests Dockerfile.vllm, Dockerfile.sglang, and Dockerfile.trtllm with comprehensive reporting
- Builds and tests both dev and local-dev Docker targets
- HTML email notifications with failure details and GitHub PR links
- SHA-based rebuild detection to avoid unnecessary builds
- Ultra-compact email formatting for quick scanning
- Build timeout: 1 hour, Container test timeout: 2 minutes
"""

import argparse
import hashlib
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set, Union

from jinja2 import Template

from common import FRAMEWORKS_UPPER, get_framework_display_name, normalize_framework, BaseUtils, DockerUtils, FrameworkInfo
import psutil
import docker
import git
import zoneinfo


class GitUtils(BaseUtils):
    """Utility class for Git operations"""
    
    def __init__(self, dry_run: bool = False, verbose: bool = False):
        super().__init__(dry_run, verbose)
        # Cache for git.Repo objects to avoid repeated instantiation
        self._repo_cache: Dict[str, git.Repo] = {}
    
    def _get_repo(self, repo_dir: Path) -> git.Repo:
        """Get cached git.Repo object or create new one"""
        repo_key = str(repo_dir.absolute())
        if repo_key not in self._repo_cache:
            self.logger.debug(f"Opening git repository: {repo_dir}")
            self._repo_cache[repo_key] = git.Repo(repo_dir)
        return self._repo_cache[repo_key]
    
    def _get_diff_statistics(self, repo_dir: Path, parent_sha: str, commit_sha: str) -> Dict[str, Any]:
        """
        Get git diff statistics between two commits.
        
        Returns:
            Dict containing:
                - changed_files: List of changed file paths
                - file_stats: List of dicts with file, additions, deletions, total_changes
                - diff_stats: List of formatted strings for display
                - total_additions: Total lines added
                - total_deletions: Total lines deleted
        """
        changed_files = []
        file_stats = []
        diff_stats = []
        total_additions = 0
        total_deletions = 0
        
        self.logger.debug(f"Equivalent: git -C {repo_dir} diff --stat {parent_sha}..{commit_sha}")
        try:
            import subprocess
            result = subprocess.run(
                ['git', '-C', str(repo_dir), 'diff', '--stat', f'{parent_sha}..{commit_sha}'],
                capture_output=True, text=True, check=True
            )
            
            # Parse the diff --stat output
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if '|' in line and ('+' in line or '-' in line):
                    # Parse lines like: " path/to/file.py | 15 ++++++++-------"
                    parts = line.split('|')
                    if len(parts) >= 2:
                        filename = parts[0].strip()
                        stats_part = parts[1].strip()
                        
                        # Extract numbers and +/- symbols
                        import re
                        numbers = re.findall(r'\d+', stats_part)
                        plus_count = stats_part.count('+')
                        minus_count = stats_part.count('-')
                        
                        if numbers:
                            total_changes = int(numbers[0])
                            additions = plus_count
                            deletions = minus_count
                            
                            changed_files.append(filename)
                            file_stats.append({
                                'file': filename,
                                'additions': additions,
                                'deletions': deletions,
                                'total_changes': total_changes
                            })
                            
                            # Format for display: "path/to/file.py +5/-3"
                            if additions > 0 and deletions > 0:
                                diff_stats.append(f"{filename} +{additions}/-{deletions}")
                            elif additions > 0:
                                diff_stats.append(f"{filename} +{additions}")
                            elif deletions > 0:
                                diff_stats.append(f"{filename} -{deletions}")
                            else:
                                diff_stats.append(f"{filename} (no changes)")
                            
                            total_additions += additions
                            total_deletions += deletions
            
            # Parse summary line like: " 3 files changed, 25 insertions(+), 8 deletions(-)"
            summary_line = lines[-1] if lines else ""
            if 'changed' in summary_line:
                # Extract totals from summary if our parsing missed anything
                import re
                insertions_match = re.search(r'(\d+) insertions?\(\+\)', summary_line)
                deletions_match = re.search(r'(\d+) deletions?\(-\)', summary_line)
                
                if insertions_match:
                    total_additions = int(insertions_match.group(1))
                if deletions_match:
                    total_deletions = int(deletions_match.group(1))
                    
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"Failed to get diff stats: {e}")
            # Return empty stats on failure
            
        return {
            'changed_files': changed_files,
            'file_stats': file_stats,
            'diff_stats': diff_stats,
            'total_additions': total_additions,
            'total_deletions': total_deletions
        }
    
    def get_commit_info(self, repo_dir: Path) -> Dict[str, Any]:
        """Get comprehensive git information using GitPython library"""
        repo = self._get_repo(repo_dir)
        
        self.logger.debug(f"Equivalent: git -C {repo_dir} log -1 --format='%h %cd %s %B %an %ae' --date=iso")
        commit = repo.head.commit
        
        sha = commit.hexsha[:7]
        
        commit_datetime = commit.committed_datetime
        
        # Convert to UTC and Pacific timezones
        utc_datetime = commit_datetime.astimezone(zoneinfo.ZoneInfo("UTC"))
        utc_timestamp = utc_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')
        
        pacific_tz = zoneinfo.ZoneInfo("America/Los_Angeles")
        pacific_datetime = commit_datetime.astimezone(pacific_tz)
        pacific_timestamp = pacific_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')
        
        commit_message = commit.summary
        full_commit_message = commit.message
        
        author_name = commit.author.name
        author_email = commit.author.email
        
        # GitPython doesn't have direct access to %al (author login), so we'll extract from email or set as unknown
        # This is a limitation, but %al is often not reliable anyway
        author_login = author_email.split('@')[0] if '@' in author_email else "unknown"
        
        author = f"{author_name} <{author_email}> ({author_login})"
        
        # Get diff statistics
        self.logger.debug(f"Equivalent: git -C {repo_dir} diff-tree --no-commit-id --name-only -r HEAD")
        if commit.parents:  # If this isn't the initial commit
            parent = commit.parents[0]
            diff_info = self._get_diff_statistics(repo_dir, parent.hexsha, commit.hexsha)
            changed_files = diff_info['changed_files']
            file_stats = diff_info['file_stats']
            diff_stats = diff_info['diff_stats']
            total_additions = diff_info['total_additions']
            total_deletions = diff_info['total_deletions']
        else:
            # Initial commit - no parent to compare against
            changed_files = []
            file_stats = []
            diff_stats = []
            total_additions = 0
            total_deletions = 0
        
        return {
            'sha': sha,
            'utc_timestamp': utc_timestamp,
            'pacific_timestamp': pacific_timestamp,
            'commit_message': commit_message,
            'full_commit_message': full_commit_message,
            'author': author,
            'author_name': author_name,
            'author_email': author_email,
            'author_login': author_login,
            'changed_files': changed_files,
            'file_stats': file_stats,
            'diff_stats': diff_stats,
            'total_files_changed': len(changed_files),
            'total_additions': total_additions,
            'total_deletions': total_deletions
        }

    def get_commit_history(self, repo_dir: Path, max_commits: int = 50) -> List[Dict[str, Any]]:
        """Get commit history with basic information for each commit"""
        repo = self._get_repo(repo_dir)
        
        self.logger.debug(f"Equivalent: git -C {repo_dir} log --oneline -n {max_commits}")
        
        commits = []
        try:
            # Get commits from the current branch, limited by max_commits
            for i, commit in enumerate(repo.iter_commits(max_count=max_commits)):
                if i >= max_commits:
                    break
                    
                sha = commit.hexsha[:7]
                commit_datetime = commit.committed_datetime
                
                # Convert to UTC and Pacific timezones
                utc_datetime = commit_datetime.astimezone(zoneinfo.ZoneInfo("UTC"))
                utc_timestamp = utc_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')
                
                pacific_tz = zoneinfo.ZoneInfo("America/Los_Angeles")
                pacific_datetime = commit_datetime.astimezone(pacific_tz)
                pacific_timestamp = pacific_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')
                
                commit_message = commit.summary
                author_name = commit.author.name
                author_email = commit.author.email
                
                commits.append({
                    'sha': sha,
                    'full_sha': commit.hexsha,
                    'utc_timestamp': utc_timestamp,
                    'pacific_timestamp': pacific_timestamp,
                    'commit_message': commit_message,
                    'author_name': author_name,
                    'author_email': author_email,
                    'commit_obj': commit  # Keep reference for checkout operations
                })
                
        except Exception as e:
            self.logger.error(f"Failed to get commit history: {e}")
            return []
            
        return commits

    def setup_repository(self, repo_dir: Path, repo_url: str, repo_sha: str = None, no_checkout: bool = False) -> None:
        """Setup git repository - clone if needed, fetch, and checkout using GitPython"""
        if repo_dir.exists():
            if repo_sha:
                self.logger.info(f"Repository directory exists, will checkout specific SHA: {repo_sha}")
            else:
                self.logger.info("Repository directory exists, updating from main branch")

            # Check if it's a git repository (only in non-dry-run mode)
            if not self.dry_run and not (repo_dir / ".git").exists():
                self.logger.error("Repository exists but is not a git repository")
                sys.exit(1)
        else:
            self.logger.info(f"Cloning repository to {repo_dir}")
            if not self.dry_run:
                self.logger.debug(f"Equivalent: git clone {repo_url} {repo_dir}")
                git.Repo.clone_from(repo_url, repo_dir)

        if not self.dry_run:
            repo = self._get_repo(repo_dir)
        
        if not no_checkout and not self.dry_run:
            self.logger.debug(f"Equivalent: git -C {repo_dir} fetch origin")
            repo.remotes.origin.fetch()

        if repo_sha and not self.dry_run:
            self.logger.info(f"Checking out specific SHA: {repo_sha}")
            self.logger.debug(f"Equivalent: git -C {repo_dir} checkout {repo_sha}")
            repo.git.checkout(repo_sha)

            self.logger.debug(f"Equivalent: git -C {repo_dir} rev-parse HEAD")
            current_sha = repo.head.commit.hexsha
            if current_sha.startswith(repo_sha) or repo_sha.startswith(current_sha):
                self.logger.info(f"Successfully checked out SHA: {current_sha}")
            else:
                self.logger.error(f"SHA mismatch: requested {repo_sha}, got {current_sha}")
                sys.exit(1)
        elif not no_checkout and not self.dry_run:
            self.logger.debug(f"Equivalent: git -C {repo_dir} checkout main")
            repo.git.checkout('main')
            self.logger.debug(f"Equivalent: git -C {repo_dir} pull origin main")
            repo.remotes.origin.pull('main')

        self.logger.info("Repository setup complete")

    def setup_dynamo_ci(self, dynamo_ci_dir: Path, repo_sha: str = None, no_checkout: bool = False) -> None:
        """Setup or update dynamo_ci repository"""
        self.logger.info("Setting up dynamo_ci repository...")

        # Validate conflicting options
        if no_checkout and repo_sha:
            self.logger.error("--no-checkout and --repo-sha are mutually exclusive")
            self.logger.error("--no-checkout skips all git operations, but --repo-sha requires checking out a specific commit")
            self.logger.error("Use either --no-checkout (to use existing repo as-is) or --repo-sha (to checkout specific commit)")
            sys.exit(1)

        if no_checkout:
            self.logger.info("NO-CHECKOUT MODE: Skipping git operations, using existing repository")

            if not dynamo_ci_dir.exists():
                self.logger.error(f"dynamo_ci directory does not exist at {dynamo_ci_dir}")
                self.logger.error("Cannot use --no-checkout without existing repository")
                sys.exit(1)

            self.logger.info(f"SUCCESS: Using existing repository at {dynamo_ci_dir}")
            return

        # Use setup_repository to setup the dynamo_ci repository
        self.setup_repository(
            dynamo_ci_dir,
            "git@github.com:ai-dynamo/dynamo.git",
            repo_sha=repo_sha,
            no_checkout=no_checkout
        )

    def get_current_sha(self, repo_dir: Path) -> str:
        """Get the current git commit SHA (short version) - fails if unable to get SHA using GitPython"""
        self.logger.debug(f"Equivalent: git -C {repo_dir} rev-parse --short HEAD")
        repo = self._get_repo(repo_dir)
        return repo.head.commit.hexsha[:7]

    def get_stored_composite_sha(self, repo_dir: Path) -> str:
        """Get stored composite SHA from file"""
        sha_file = repo_dir / ".last_build_composite_sha"
        if sha_file.exists():
            return sha_file.read_text().strip()
        return ""

    def store_composite_sha(self, repo_dir: Path, sha: str) -> None:
        """Store current composite SHA to file"""
        sha_file = repo_dir / ".last_build_composite_sha"
        sha_file.write_text(sha)
        self.logger.info(f"Stored composite SHA in repository: {sha}")
    
    def generate_composite_sha_from_container_dir(self, repo_dir: Path) -> Tuple[Optional[str], List[Path]]:
        """Generate composite SHA from all container files recursively"""
        container_dir = repo_dir / "container"

        if container_dir.exists():
            self.logger.debug(f"Generating composite SHA from container directory: {container_dir}")
        else:
            self.logger.error(f"Container directory not found: {container_dir}")
            return None, []

        # Define file extensions and patterns to exclude
        excluded_extensions = {'.md', '.rst', '.log', '.bak', '.tmp', '.swp', '.swo', '.orig', '.rej'}
        excluded_filenames = {'README', 'CHANGELOG', 'LICENSE', 'NOTICE', 'AUTHORS', 'CONTRIBUTORS'}
        excluded_specific_files = {'launch_message.txt'}

        # Get all files in container directory recursively, sorted for consistent hashing
        files_to_hash = []
        excluded_count = 0
        for file_path in sorted(container_dir.rglob('*')):
            if file_path.is_file():
                # Skip hidden files/directories
                if any(part.startswith('.') for part in file_path.relative_to(container_dir).parts):
                    excluded_count += 1
                    continue
                
                # Skip files with excluded extensions
                if file_path.suffix.lower() in excluded_extensions:
                    excluded_count += 1
                    continue
                
                # Skip files with excluded names
                if file_path.stem.upper() in excluded_filenames:
                    excluded_count += 1
                    continue
                
                # Skip specific excluded files
                if file_path.name.lower() in excluded_specific_files:
                    excluded_count += 1
                    continue
                
                # Store relative path from repo_dir for consistent hashing
                rel_path = file_path.relative_to(repo_dir)
                files_to_hash.append(rel_path)

        if not files_to_hash:
            self.logger.error("No files found in container directory for composite SHA calculation")
            return None, []

        with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as temp_file:
            temp_path = Path(temp_file.name)

            try:
                # Concatenate all files that exist
                found_files = 0
                for file_rel_path in files_to_hash:
                    full_path = repo_dir / file_rel_path
                    if full_path.exists():
                        # Write file path first (for uniqueness), then file content
                        temp_file.write(str(file_rel_path).encode('utf-8'))
                        temp_file.write(b'\n')
                        with open(full_path, 'rb') as f:
                            temp_file.write(f.read())
                        temp_file.write(b'\n')
                        found_files += 1
                    else:
                        self.logger.warning(f"File not found for composite SHA calculation: {file_rel_path}")

                if found_files == 0:
                    self.logger.error("No files found for composite SHA calculation")
                    return None, []

                # Generate SHA256 of concatenated files
                temp_file.flush()
                with open(temp_path, 'rb') as f:
                    sha = hashlib.sha256(f.read()).hexdigest()

                self.logger.info(f"Generated composite SHA from {found_files} container files (excluded {excluded_count}): {sha[:12]}...")
                
                # Return the list of files that were actually used
                used_files = []
                for file_rel_path in files_to_hash:
                    full_path = repo_dir / file_rel_path
                    if full_path.exists():
                        used_files.append(file_rel_path)
                
                return sha, used_files

            finally:
                temp_path.unlink(missing_ok=True)




class BuildShUtils(BaseUtils):
    """Utility class for build.sh-related operations"""
    
    def __init__(self, dynamo_ci_dir: Path, docker_utils: 'DockerUtils', dry_run: bool = False):
        super().__init__(dry_run)
        self.dynamo_ci_dir = dynamo_ci_dir
        self.docker_utils = docker_utils
        # Cache for build commands to avoid repeated expensive calls
        self._build_commands_cache: Dict[str, Tuple[bool, List[str]]] = {}
    
    def extract_base_image_tag(self, framework: str) -> str:
        """Extract the dynamo-base image tag from build commands for a framework"""
        try:
            # Get build commands for the framework
            success, docker_commands = self.get_build_commands(framework, None)
            if not success:
                return ""
            for cmd in docker_commands:
                if "--tag" in cmd and "dynamo-base:" in cmd:
                    # Extract the tag using regex
                    match = re.search(r'--tag\s+(dynamo-base:[^\s]+)', cmd)
                    if match:
                        return match.group(1)
            
            return ""
        except Exception as e:
            self.logger.debug(f"Failed to extract base image tag for {framework}: {e}")
            return ""
    
    def get_build_commands(self, framework: str, docker_target_type: Optional[str]) -> Tuple[bool, List[str]]:
        """Get docker build commands from build.sh --dry-run and filter out latest tags"""
        # Create cache key from parameters
        cache_key = f"{framework}:{docker_target_type or 'dev'}"
        
        # Check cache first
        if cache_key in self._build_commands_cache:
            self.logger.debug(f"Using cached build commands for {cache_key}")
            return self._build_commands_cache[cache_key]
        
        self.logger.info("Getting docker build commands from build.sh --dry-run...")

        # Build command - use --target flag only when docker_target_type is specified
        cmd = ["./container/build.sh", "--dry-run", "--framework", framework]
        if docker_target_type is not None:
            cmd.extend(["--target", docker_target_type])

        # Execute build.sh --dry-run (safe to run even in dry-run mode)
        self.logger.debug(f"Executing: cd {self.dynamo_ci_dir} && {' '.join(cmd)}")
        build_result = self.cmd(cmd, capture_output=True, text=True, cwd=self.dynamo_ci_dir)

        if build_result.returncode != 0:
            self.logger.error(f"Failed to get build commands for {framework}")
            if build_result.stderr:
                self.logger.error(f"Error: {build_result.stderr}")
            return False, []

        # Extract and filter docker commands to remove latest tags
        docker_commands = []
        output_lines = build_result.stdout.split('\n') + build_result.stderr.split('\n')

        # parse the --dry-run output to get the docker build commands
        for line in output_lines:
            line = line.strip()
            if line.startswith("docker build"):
                # Remove --tag arguments containing ":latest" using regex
                line = re.sub(r'--tag\s+\S*:latest\S*(?:\s|$)', '', line)
                
                # Validate: only one --tag should remain after filtering out :latest tags
                tag_matches = re.findall(r'--tag\s+(\S+)', line)
                
                if len(tag_matches) > 1:
                    self.logger.error(f"Multiple --tag arguments found in docker build command for {framework}")
                    self.logger.error(f"Command: {line}")
                    self.logger.error(f"Tags found: {tag_matches}")
                    return False, []

                if line.strip():
                    # Apply unused argument filtering
                    final_command = self.docker_utils.filter_unused_build_args(line)
                    if final_command:
                        docker_commands.append(final_command)

        if not docker_commands:
            self.logger.error(f"No docker build commands found for {framework}")
            return False, []

        result = True, docker_commands
        
        # Cache the result for future calls
        self._build_commands_cache[cache_key] = result
        return result


class EmailTemplateRenderer:
    """Handles email template rendering and HTML formatting for DynamoDockerBuilder reports"""
    
    def __init__(self):
        """Initialize the EmailTemplateRenderer"""
        pass
    
    def generate_v2_report(
        self,
        task_tree: Dict[str, 'BaseTask'],
        git_info: Dict[str, Any],
        date: str,
        docker_utils: 'DockerUtils',
        repo_path: str = None,
        log_dir: str = None
    ) -> str:
        """
        Generate HTML report from task tree using the existing email template
        
        Args:
            task_tree: Dictionary of task_id -> BaseTask
            git_info: Git commit information
            date: Build date string
            docker_utils: DockerUtils instance for fetching image info (required)
            repo_path: Path to the repository (optional)
            log_dir: Path to the log directory (optional)
        
        Returns:
            HTML report as string
        """
        from jinja2 import Template
        
        # Count task results
        total_tasks = len(task_tree)
        build_tasks = sum(1 for t in task_tree.values() if t.task_type == 'build')
        test_tasks = sum(1 for t in task_tree.values() if t.task_type in ('sanity_check', 'compilation'))
        succeeded = sum(1 for t in task_tree.values() if t.status == 'success')
        failed = sum(1 for t in task_tree.values() if t.status == 'failed')
        skipped = sum(1 for t in task_tree.values() if t.status == 'skipped')
        
        # Organize tasks by framework
        # Template expects framework objects with 'name' and 'targets' list
        frameworks_list = []
        frameworks_dict = {}
        
        # Collect compilation information
        compilation_tasks = []
        for task_id, task in task_tree.items():
            if task.task_type == 'compilation':
                compilation_tasks.append(task)
        
        for task_id, task in task_tree.items():
            if not task.framework:
                continue
            
            # Skip compilation tasks for framework-based structure (handled separately)
            if task.task_type == 'compilation':
                continue
            
            if task.framework not in frameworks_dict:
                frameworks_dict[task.framework] = {
                    'name': task.framework,
                    'targets_dict': {}
                }
            
            # Skip if no target
            if not task.target:
                continue
            
            target = task.target
            if target not in frameworks_dict[task.framework]['targets_dict']:
                frameworks_dict[task.framework]['targets_dict'][target] = {
                    'name': target,
                    'success': None,
                    'build_success': None,
                    'build_time': '-',
                    'test_success': None,
                    'test_time': '-',
                    'container_size': None,
                    'image_tag': None,
                    'image_id': None,
                    'error_output': None
                }
            
            target_data = frameworks_dict[task.framework]['targets_dict'][target]
            
            # Populate based on task type
            if task.task_type == 'build':
                # Handle skipped, success, and failure
                if task.status == 'skipped':
                    target_data['build_time'] = 'skipped'
                    target_data['build_success'] = None  # Neither success nor failure
                elif task.status == 'success':
                    target_data['build_success'] = True
                    target_data['build_time'] = f"{task.duration:.1f}s" if task.duration else '-'
                elif task.status == 'failed':
                    target_data['build_success'] = False
                    target_data['build_time'] = f"{task.duration:.1f}s" if task.duration else 'failed'
                    # Capture error output
                    if task.error_message:
                        target_data['error_output'] = task.error_message
                
                if hasattr(task, 'image_id') and task.image_id:
                    target_data['image_id'] = task.image_id
                if hasattr(task, 'image_tag') and task.image_tag:
                    target_data['image_tag'] = task.image_tag
                if hasattr(task, 'image_size') and task.image_size:
                    target_data['container_size'] = task.image_size
            elif task.task_type == 'sanity_check':
                # Handle skipped, success, and failure
                if task.status == 'skipped':
                    target_data['test_time'] = 'skipped'
                    target_data['test_success'] = None  # Neither success nor failure
                elif task.status == 'success':
                    target_data['test_success'] = True
                    target_data['test_time'] = f"{task.duration:.1f}s" if task.duration else '-'
                elif task.status == 'failed':
                    target_data['test_success'] = False
                    target_data['test_time'] = f"{task.duration:.1f}s" if task.duration else 'failed'
                    # Capture error output
                    if task.error_message:
                        if target_data['error_output']:
                            target_data['error_output'] += '\n\n' + task.error_message
                        else:
                            target_data['error_output'] = task.error_message
        
        # Calculate overall success for each target (after all tasks are processed)
        for framework in frameworks_dict.values():
            for target_data in framework['targets_dict'].values():
                # Skipped tasks (None) don't count as failure
                # Only actual failures (False) count as failure
                has_failure = (target_data['build_success'] is False or target_data['test_success'] is False)
                has_success = (target_data['build_success'] is True or target_data['test_success'] is True)
                
                if has_failure:
                    target_data['success'] = False
                elif has_success:
                    target_data['success'] = True
                else:
                    # Everything skipped - show as None (skipped/neutral state)
                    target_data['success'] = None
        
        # Convert dict to list for template iteration
        for framework in frameworks_dict.values():
            framework['targets'] = list(framework['targets_dict'].values())
            del framework['targets_dict']
        frameworks_list = list(frameworks_dict.values())
        
        # Fetch container size and image ID from Docker for existing images
        for framework in frameworks_list:
            for target in framework['targets']:
                if target['image_tag'] and not target['container_size']:
                    try:
                        image_info = docker_utils.get_image_info(target['image_tag'])
                        if image_info:
                            target['container_size'] = image_info.size_human
                            # Get image ID in docker images format (12 chars)
                            if not target['image_id'] and image_info.image_id:
                                # Convert full SHA to docker images format
                                full_id = image_info.image_id
                                if full_id.startswith('sha256:'):
                                    target['image_id'] = full_id[7:19]  # Extract 12 chars after 'sha256:'
                                else:
                                    target['image_id'] = full_id[:12]  # First 12 chars if no prefix
                    except Exception:
                        # Image not found or error - leave as '-'
                        pass
        
        # Process compilation information
        workspace_compilation = None
        if compilation_tasks:
            # Combine all compilation tasks into a single compilation report
            success_tasks = [task for task in compilation_tasks if task.status == 'success']
            failed_tasks = [task for task in compilation_tasks if task.status == 'failed']
            skipped_tasks = [task for task in compilation_tasks if task.status == 'skipped']
            
            # Determine overall compilation status
            if skipped_tasks and not success_tasks and not failed_tasks:
                # All compilation tasks were skipped
                compilation_success = None  # Neither success nor failure - skipped
            elif failed_tasks:
                # At least one compilation task failed
                compilation_success = False
            else:
                # All non-skipped tasks succeeded
                compilation_success = True
            
            total_time = sum(task.duration for task in compilation_tasks if task.duration)
            
            # Get output snippets from the last few lines of compilation logs
            output_snippets = {}
            for task in compilation_tasks:
                if task.status == 'failed' and task.error_message:
                    # For failed compilation, show error output
                    output_snippets['error'] = task.error_message.split('\n')[-10:]  # Last 10 lines
                elif task.status == 'success' and log_dir:
                    # For successful compilation, try to read last few lines from log file
                    try:
                        log_file = Path(log_dir) / f"{date.split()[0]}.{git_info.get('sha', 'unknown')}.{task.task_id}.log"
                        if log_file.exists():
                            with open(log_file, 'r') as f:
                                lines = f.readlines()
                                output_snippets['success'] = [line.rstrip() for line in lines[-10:]]  # Last 10 lines
                    except Exception:
                        pass
            
            workspace_compilation = {
                'success': compilation_success,
                'time': f"{total_time:.1f}s" if total_time else ('skipped' if compilation_success is None else '-'),
                'output_snippets': output_snippets if output_snippets else None,
                'frameworks': [task.framework for task in compilation_tasks],
                'task_count': len(compilation_tasks),
                'skipped_count': len(skipped_tasks),
                'success_count': len(success_tasks),
                'failed_count': len(failed_tasks)
            }
        
        # Overall status
        overall_status = "✅ ALL TESTS PASSED" if failed == 0 and succeeded > 0 else ("❌ SOME TESTS FAILED" if failed > 0 else "⚠️ NO TESTS RUN")
        status_color = "#28a745" if failed == 0 and succeeded > 0 else ("#dc3545" if failed > 0 else "#ffc107")
        
        # Prepare template data
        template_data = {
            'overall_status': overall_status,
            'status_color': status_color,
            'build_date': date,
            'total_builds': build_tasks,
            'total_tests': test_tasks,
            'passed_tests': succeeded,
            'failed_tests': failed,
            'git_info': git_info,
            'frameworks': frameworks_list,
            'workspace_compilation': workspace_compilation,
            # Add summary boxes data
            'total_tasks': total_tasks,
            'succeeded': succeeded,
            'failed': failed,
            'skipped': skipped,
            # Add repository and log directory
            'repo_path': repo_path or 'N/A',
            'log_dir': log_dir or 'N/A',
        }
        
        # Render template
        template = Template(self.get_email_template())
        return template.render(**template_data)

    def get_email_template(self) -> str:
        """Get the Jinja2 email template"""
        return """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>DynamoDockerBuilder - {{ git_info.sha[:8] }} - {{ overall_status }}</title>
<style>
body { font-family: Arial, sans-serif; margin: 10px; line-height: 1.3; }
.header { background-color: {{ status_color }}; color: white; padding: 15px 20px; border-radius: 4px; margin-bottom: 10px; text-align: center; }
.summary { background-color: #f8f9fa; padding: 4px 6px; border-radius: 2px; margin: 3px 0; }
.summary-boxes { width: 100%; margin: 15px 0; }
.summary-boxes table { width: 100%; border-collapse: separate; border-spacing: 10px; }
.summary-box { padding: 12px; border-radius: 6px; text-align: center; width: 25%; }
.summary-box.total { background: #e8f4f8; border-left: 4px solid #3498db; }
.summary-box.success { background: #e8f8f5; border-left: 4px solid #27ae60; }
.summary-box.failed { background: #fde8e8; border-left: 4px solid #e74c3c; }
.summary-box.skipped { background: #f9f9f9; border-left: 4px solid #95a5a6; }
.summary-box .number { font-size: 28px; font-weight: bold; margin: 5px 0; display: block; }
.summary-box .label { font-size: 13px; color: #7f8c8d; text-transform: uppercase; font-weight: 600; display: block; }
.results { margin: 10px 0; }
.framework { margin: 10px 0; padding: 8px; border: 1px solid #dee2e6; border-radius: 4px; background-color: #ffffff; }
.framework-header { background-color: #007bff; color: white; padding: 8px 12px; margin: -8px -8px 8px -8px; border-radius: 4px 4px 0 0; font-weight: bold; }
.results-chart { display: table; width: 100%; border-collapse: collapse; margin: 8px 0; }
.chart-row { display: table-row; }
.chart-cell { display: table-cell; padding: 6px 12px; border: 1px solid #dee2e6; vertical-align: middle; }
.chart-header { background-color: #f8f9fa; font-weight: bold; text-align: center; }
.chart-target { font-weight: bold; background-color: #f1f3f4; }
.chart-status { text-align: center; }
.chart-timing { text-align: right; font-family: monospace; font-size: 0.9em; }
.success { color: #28a745; font-weight: bold; }
.failure { color: #dc3545; font-weight: bold; }
.git-info { background-color: #e9ecef; padding: 4px 6px; border-radius: 2px; font-family: monospace; font-size: 0.9em; }
.error-output { background-color: #2d3748; color: #e2e8f0; padding: 8px; border-radius: 3px; font-family: 'Courier New', monospace; font-size: 0.85em; margin: 8px 0; overflow-x: auto; white-space: pre-wrap; width: 100%; box-sizing: border-box; display: block; }
p { margin: 1px 0; }
h3 { margin: 4px 0 2px 0; font-size: 1.0em; }
h4 { margin: 3px 0 1px 0; font-size: 0.95em; }
h2 { margin: 0; font-size: 1.2em; font-weight: bold; }
</style>
</head>
<body>
<div class="header">
<h2>DynamoDockerBuilder - {{ git_info.sha[:8] }} - {{ overall_status }}</h2>
</div>

<div class="summary-boxes">
<table>
<tr>
<td class="summary-box total">
<div class="number">{{ total_tasks }}</div>
<div class="label">Total Tasks</div>
</td>
<td class="summary-box success">
<div class="number">{{ succeeded }}</div>
<div class="label">Succeeded</div>
</td>
<td class="summary-box failed">
<div class="number">{{ failed }}</div>
<div class="label">Failed</div>
</td>
<td class="summary-box skipped">
<div class="number">{{ skipped }}</div>
<div class="label">Skipped</div>
</td>
</tr>
</table>
</div>

<div class="summary">
<p><strong>Build & Test Date:</strong> {{ build_date }}</p>
</div>

<div class="git-info">
<p><strong>Commit SHA:</strong> {{ git_info.sha }}</p>
<p><strong>Commit Date:</strong> {{ git_info.date }}</p>
<p><strong>Author:</strong> {{ git_info.author | safe }}</p>
<div style="background-color: #f8f9fa; padding: 6px; border-radius: 3px; margin: 3px 0;">
<strong>Commit Message:</strong>
<pre style="margin: 3px 0; white-space: pre-wrap; font-family: monospace; font-size: 0.9em;">{{ git_info.full_message | e }}</pre>
</div>
{% if git_info.total_additions is defined or git_info.total_deletions is defined %}
<p><strong>Changes Summary:</strong> +{{ git_info.total_additions | default(0) }}/-{{ git_info.total_deletions | default(0) }} lines</p>
{% endif %}
{% if git_info.diff_stats %}
<p><strong>Files Changed with Line Counts:</strong></p>
<div style="background-color: #f8f9fa; padding: 6px; border-radius: 3px; font-family: monospace; font-size: 0.9em; margin: 3px 0;">
{% for stat in git_info.diff_stats %}
• {{ stat | e }}<br>
{% endfor %}
</div>
{% endif %}
</div>

{% if workspace_compilation %}
<div class="framework">
<div class="framework-header">Compilation</div>
<div class="results-chart">
<div class="chart-row">
<div class="chart-cell chart-header">Status</div>
<div class="chart-cell chart-header">Time</div>
<div class="chart-cell chart-header">Details</div>
</div>
<div class="chart-row">
<div class="chart-cell chart-status">
{% if workspace_compilation.success is none %}
<span style="color: #95a5a6; font-weight: bold;">⊘ SKIPPED</span>
{% elif workspace_compilation.success %}
<span class="success">✅ PASSED</span>
{% else %}
<span class="failure">❌ FAILED</span>
{% endif %}
</div>
<div class="chart-cell chart-timing">{{ workspace_compilation.time }}</div>
<div class="chart-cell">One-time setup for local-dev containers (cargo build, maturin develop, uv pip install)</div>
</div>
</div>

{% if workspace_compilation.output_snippets %}
<div style="margin-top: 12px;">
<h4 style="margin-bottom: 6px;">Compilation Output (last few lines):</h4>

{% if workspace_compilation.output_snippets.cargo_build %}
<div style="margin: 8px 0;">
<strong>📦 Cargo Build:</strong>
<div class="error-output">{% for line in workspace_compilation.output_snippets.cargo_build %}{{ line }}
{% endfor %}</div>
</div>
{% endif %}

{% if workspace_compilation.output_snippets.maturin_develop %}
<div style="margin: 8px 0;">
<strong>🐍 Maturin Python Bindings:</strong>
<div class="error-output">{% for line in workspace_compilation.output_snippets.maturin_develop %}{{ line }}
{% endfor %}</div>
</div>
{% endif %}

{% if workspace_compilation.output_snippets.uv_install %}
<div style="margin: 8px 0;">
<strong>📚 UV Package Install:</strong>
<div class="error-output">{% for line in workspace_compilation.output_snippets.uv_install %}{{ line }}
{% endfor %}</div>
</div>
{% endif %}
</div>
{% endif %}
</div>
{% endif %}

{% if base_builds %}
<div class="framework">
<div class="framework-header">dynamo-base Build Times</div>
<div class="results-chart">
<div class="chart-row">
<div class="chart-cell chart-header">Target</div>
<div class="chart-cell chart-header">Status</div>
<div class="chart-cell chart-header">Docker Image Build Time</div>
<div class="chart-cell chart-header">Container Size</div>
<div class="chart-cell chart-header">Image Tag</div>
<div class="chart-cell chart-header">IMAGE ID</div>
</div>
{% for base_build in base_builds %}
<div class="chart-row">
<div class="chart-cell chart-target">{{ base_build.name }}</div>
<div class="chart-cell chart-status">
<span class="success">✅ PASSED</span>
</div>
<div class="chart-cell chart-timing">{{ base_build.build_time }}</div>
<div class="chart-cell chart-timing">{{ base_build.container_size if base_build.container_size else '-' }}</div>
<div class="chart-cell chart-timing" style="font-family: monospace; font-size: 0.8em; white-space: nowrap;">{{ base_build.image_tag }}</div>
<div class="chart-cell chart-timing" style="font-family: monospace; font-size: 0.8em; white-space: nowrap;">{{ base_build.image_id if base_build.image_id else '-' }}</div>
</div>
{% endfor %}
</div>
</div>
{% endif %}

<div class="results">
{% for framework in frameworks %}
<div class="framework">
<div class="framework-header">Dockerfile.{{ framework.name.lower() }}</div>
<div class="results-chart">
<div class="chart-row">
<div class="chart-cell chart-header">Target</div>
<div class="chart-cell chart-header">Status</div>
<div class="chart-cell chart-header">Docker Image Build Time</div>
<div class="chart-cell chart-header">sanity_check.py</div>
<div class="chart-cell chart-header">Container Size</div>
<div class="chart-cell chart-header">Image Tag</div>
<div class="chart-cell chart-header">IMAGE ID</div>
</div>
{% for target in framework.targets %}
<div class="chart-row">
<div class="chart-cell chart-target">{{ target.name }}</div>
<div class="chart-cell chart-status">
{% if target.success is none %}
<span style="color: #95a5a6; font-weight: bold;">⊘ SKIPPED</span>
{% elif target.success %}
<span class="success">✅ PASSED</span>
{% else %}
<span class="failure">❌ FAILED</span>
{% endif %}
</div>
<div class="chart-cell chart-timing">{{ target.build_time }}</div>
<div class="chart-cell chart-timing">{{ target.test_time }}</div>
<div class="chart-cell chart-timing">{{ target.container_size if target.container_size else '-' }}</div>
<div class="chart-cell chart-timing" style="font-family: monospace; font-size: 0.8em; white-space: nowrap;">{{ target.image_tag if target.image_tag else '-' }}</div>
<div class="chart-cell chart-timing" style="font-family: monospace; font-size: 0.8em; white-space: nowrap;">{{ target.image_id if target.image_id else '-' }}</div>
</div>
{% endfor %}
</div>
{% for target in framework.targets %}
{% if not target.success and target.error_output %}
<div class="error-output">{{ target.error_output }}</div>
{% endif %}
{% endfor %}
</div>
{% endfor %}
</div>

<div class="summary">
<p><strong>Repository:</strong> {{ repo_path }}</p>
<p><strong>Log Directory:</strong> {{ log_dir }}</p>
</div>

<p><em>This email was generated automatically by DynamoDockerBuilder.</em></p>
</body>
</html>"""

    def convert_pr_links(self, message: str) -> str:
        """Convert PR references like (#3107) to GitHub links"""
        # Pattern to match (#number)
        pr_pattern = r'\(#(\d+)\)'

        def replace_pr(match: re.Match[str]) -> str:
            pr_number = match.group(1)
            return f'(<a href="https://github.com/ai-dynamo/dynamo/pull/{pr_number}" style="color: #0066cc;">#{pr_number}</a>)'

        return re.sub(pr_pattern, replace_pr, message)

    def html_escape(self, text: str) -> str:
        """Escape HTML special characters"""
        return (text.replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;")
                   .replace('"', "&quot;")
                   .replace("'", "&#x27;"))

    def format_author_html(self, author: str) -> str:
        """Format author information as proper HTML with mailto link"""

        # Pattern to match "Name <email> (login)" or "Name <email>"
        pattern = r'^(.+?)\s*<([^>]+)>(?:\s*\(([^)]+)\))?$'
        match = re.match(pattern, author.strip())

        if match:
            name = match.group(1).strip()
            email = match.group(2).strip()
            login = match.group(3).strip() if match.group(3) else None

            # Create HTML with clickable email
            if login:
                return f'{self.html_escape(name)} &lt;<a href="mailto:{email}" style="color: #0066cc; text-decoration: none;">{email}</a>&gt; ({self.html_escape(login)})'
            else:
                return f'{self.html_escape(name)} &lt;<a href="mailto:{email}" style="color: #0066cc; text-decoration: none;">{email}</a>&gt;'
        else:
            # Fallback: just escape the whole string
            return self.html_escape(author)

    def _send_html_email_via_smtp(
        self,
        email: str,
        html_content: str,
        subject_prefix: str,
        git_sha: str,
        failed_tasks: List[str],
        logger
    ) -> None:
        """Shared SMTP email sending logic for both v1 and v2"""
        try:
            # Determine overall status
            overall_status = "SUCCESS" if not failed_tasks else "FAILURE"
            
            # Create subject line
            status_prefix = "SUCC" if overall_status == "SUCCESS" else "FAIL"
            
            # Include failed task names in subject if any
            if failed_tasks:
                failure_summary = ", ".join(failed_tasks)
                subject = f"{status_prefix}: {subject_prefix} - {git_sha} ({failure_summary})"
            else:
                subject = f"{status_prefix}: {subject_prefix} - {git_sha}"

            # Create email file with proper CRLF formatting
            email_file = Path(f"/tmp/dynamo_email_{os.getpid()}.txt")

            # Write email content directly
            email_content = f'Subject: {subject}\r\nFrom: {subject_prefix} <dynamo-docker-builder@nvidia.com>\r\nTo: {email}\r\nMIME-Version: 1.0\r\nContent-Type: text/html; charset=UTF-8\r\n\r\n{html_content}\r\n'

            with open(email_file, 'w', encoding='utf-8') as f:
                f.write(email_content)

            # Send email using curl
            result = subprocess.run([
                'curl', '--url', 'smtp://smtp.nvidia.com:25',
                '--mail-from', 'dynamo-docker-builder@nvidia.com',
                '--mail-rcpt', email,
                '--upload-file', str(email_file)
            ], capture_output=True, text=True)

            # Clean up
            email_file.unlink(missing_ok=True)

            if result.returncode == 0:
                logger.info(f"SUCCESS: Email notification sent to {email}")
                logger.info(f"  Subject: {subject}")
                if failed_tasks:
                    logger.info(f"  Failed tasks: {', '.join(failed_tasks)}")
            else:
                logger.error(f"Failed to send email: {result.stderr}")

        except Exception as e:
            logger.error(f"Error sending email notification: {e}")


@dataclass
class BaseTask:
    """
    Base class for all tasks in the dependency tree.
    Contains common fields shared by build and execute tasks.
    """
    # Task identification
    task_id: str                # e.g., "dynamo-base-VLLM", "VLLM-dev", "VLLM-compilation"
    task_type: str              # e.g., "build", "compilation", "sanity_check"
    framework: Optional[str] = None     # e.g., "VLLM", "TRTLLM", "SGLANG", or None for dynamo-base
    target: Optional[str] = None        # e.g., "dev", "local-dev", "runtime"
    description: str = ""       # e.g., "Build dynamo-base image", "Run workspace compilation"
    
    # Dependency tree
    depends_on: List[str] = field(default_factory=list)  # e.g., ["dynamo-base-VLLM", "VLLM-dev"]
    children: List[str] = field(default_factory=list)    # e.g., ["VLLM-dev", "TRTLLM-dev"]
    
    # Execution state
    status: str = 'pending'     # e.g., "pending", "ready", "in_progress", "success", "failed", "skipped"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration: float = 0.0
    
    # Execution results
    success: Optional[bool] = None
    error_message: Optional[str] = None
    log_file: Optional[str] = None
    
    def is_ready(self, task_tree: Dict[str, 'BaseTask']) -> bool:
        """Check if all dependencies are completed successfully or skipped"""
        if self.status != 'pending':
            return False

        for dep_id in self.depends_on:
            dep_task = task_tree.get(dep_id)
            if not dep_task:
                return False
            # A task is ready if all its dependencies are either successful or skipped
            if dep_task.status not in ('success', 'skipped'):
                return False
        
        return True
    
    def add_dependency(self, parent_task_id: str) -> None:
        """Add a parent dependency"""
        if parent_task_id not in self.depends_on:
            self.depends_on.append(parent_task_id)
    
    def add_child(self, child_task_id: str) -> None:
        """Add a child dependency"""
        if child_task_id not in self.children:
            self.children.append(child_task_id)


@dataclass
class BuildTask(BaseTask):
    """
    Build task - represents a Docker image build operation.
    Inherits common fields from BaseTask and adds build-specific fields.
    """
    # Build-specific fields
    docker_command: str = ""    # e.g., "docker build -f Dockerfile --target dev --tag dynamo:dev ..."
    base_image: str = ""        # e.g., "nvidia/cuda:12.4.0-devel-ubuntu22.04" (FROM image)
    image_tag: str = ""         # e.g., "dynamo-base:v0.1.0.dev.8388e1628" (single output tag)
    
    # Build results
    git_sha: Optional[str] = None               # e.g., "6a1391eb4"
    container_id: Optional[str] = None          # e.g., "8f3c9d2e1a4b"
    image_id: Optional[str] = None              # e.g., "sha256:ab155e5b17bd..."
    image_size: Optional[str] = None            # e.g., "15.2GB"
    build_log: Optional[str] = None             # e.g., "Step 1/25: FROM nvidia/cuda..."
    failure_details: Optional[str] = None       # e.g., "ERROR: Package 'vllm' not found"
    
    def __post_init__(self):
        """Ensure task_type is set to 'build'"""
        self.task_type = "build"


@dataclass
class ExecuteTask(BaseTask):
    """
    Execute task - represents running commands in containers (compilation, sanity checks, etc.).
    Inherits common fields from BaseTask and adds execute-specific fields.
    """
    # Execute-specific fields
    command: str = ""           # e.g., "./container/run.sh --image dynamo:dev --mount-workspace ..."
    image_name: str = ""        # e.g., "dynamo:v0.1.0.dev.8388e1628-vllm-local-dev"
    timeout: Optional[float] = None     # e.g., 1800.0 (seconds)
    
    # Execution results
    return_code: Optional[int] = None   # e.g., 0, 1, 137 (killed)
    stdout: str = ""            # e.g., "All tests passed!"
    stderr: str = ""            # e.g., "ERROR: Test failed"
    
    # Compilation-specific results (for compilation-type execute tasks)
    compilation_success: Optional[bool] = None  # e.g., True, False
    compilation_time: float = 0.0               # e.g., 123.4 (seconds)
    compilation_output: Dict[str, List[str]] = field(default_factory=dict)  # e.g., {"cargo_build": ["line1", "line2"]}


class DynamoDockerBuilder(BaseUtils):
    """DynamoDockerBuilder - Next generation build system with dependency tree"""
    
    # Framework constants (using shared constants)
    FRAMEWORKS: List[str] = FrameworkInfo.get_frameworks_upper()
    
    def __init__(self, verbose: bool = False) -> None:
        # Configuration flags - set before calling super().__init__
        self.dry_run = False
        
        # Call parent constructor (sets up logger automatically)
        super().__init__(dry_run=self.dry_run, verbose=verbose)
        
        self.script_dir = Path(__file__).parent.absolute()
        self.dynamo_ci_dir = self.script_dir.parent / "dynamo_ci"
        self.date = datetime.now().strftime("%Y-%m-%d")
        self.log_dir = self.dynamo_ci_dir / "logs" / self.date
        
        # Lock file for preventing concurrent runs
        self.lock_file = self.script_dir / f".{Path(__file__).name}.lock"
        
        # Configuration flags (will be set from args)
        self.sanity_check_only = False
        self.no_checkout = False
        self.force_run = False
        self.parallel = False
        self.email = None
        self.targets = ["dev", "local-dev"]  # Default targets
        self.repo_sha = None
        
        # Existing run detection
        self.executed_commands: Set[str] = set()  # Track executed docker commands
        
        # Initialize utility classes
        self.git_utils = GitUtils(dry_run=self.dry_run)
        self.docker_utils = DockerUtils(dry_run=self.dry_run)
        self.buildsh_utils = BuildShUtils(self.dynamo_ci_dir, self.docker_utils, dry_run=self.dry_run)
        
        # Task tree - dictionary mapping task_id to BaseTask instances (BuildTask or ExecuteTask)
        self.task_tree: Dict[str, BaseTask] = {}
        
        # Track seen commands to avoid duplicates
        self.seen_commands: Dict[str, str] = {}  # command -> task_id
    
    def _setup_logging_dir(self) -> None:
        """Create the log directory if it doesn't exist"""
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Log directory ready: {self.log_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create log directory {self.log_dir}: {e}")
            raise
    
    def _check_if_rebuild_needed(self) -> bool:
        """Check if rebuild is needed based on composite SHA"""
        self.logger.info("Checking if rebuild is needed based on file changes...")
        self.logger.info(f"Composite SHA file location: {self.dynamo_ci_dir}/.last_build_composite_sha")

        # Generate current composite SHA
        current_sha, _ = self.git_utils.generate_composite_sha_from_container_dir(self.dynamo_ci_dir)
        if current_sha is None:
            self.logger.error("Failed to generate current composite SHA")
            return True  # Assume rebuild needed if we can't determine

        # Get stored composite SHA
        stored_sha = self.git_utils.get_stored_composite_sha(self.dynamo_ci_dir)

        if stored_sha:
            if current_sha == stored_sha:
                if self.force_run:
                    self.logger.info(f"Composite SHA unchanged ({current_sha}) but --force-run specified - proceeding")
                    return True  # Rebuild needed (forced)
                else:
                    self.logger.info(f"Composite SHA unchanged ({current_sha}) - skipping rebuild")
                    self.logger.info("Use --force-run to force rebuild")
                    return False  # Rebuild not needed
            else:
                self.logger.info("Composite SHA changed:")
                self.logger.info(f"  Previous: {stored_sha}")
                self.logger.info(f"  Current:  {current_sha}")
                self.logger.info("Rebuild needed")
                self.git_utils.store_composite_sha(self.dynamo_ci_dir, current_sha)
                return True  # Rebuild needed
        else:
            self.logger.info("No previous composite SHA found - rebuild needed")
            self.git_utils.store_composite_sha(self.dynamo_ci_dir, current_sha)
            return True  # Rebuild needed

    def is_command_executed(self, docker_command: str) -> bool:
        """Check if a docker command has already been executed
        
        Args:
            docker_command: Complete docker build command as a string
            
        Returns:
            True if command was already executed, False otherwise
        """
        # Normalize the command by removing extra whitespace and sorting build args
        # This helps catch functionally identical commands with different formatting
        normalized_cmd = self.docker_utils.normalize_command(docker_command)
        return normalized_cmd in self.executed_commands

    def mark_command_executed(self, docker_command: str) -> None:
        """Mark a docker command as executed
        
        Args:
            docker_command: Complete docker build command as a string
        """
        normalized_cmd = self.docker_utils.normalize_command(docker_command)
        self.executed_commands.add(normalized_cmd)
        self.logger.debug(f"Marked command as executed: {normalized_cmd[:100]}...")

    def check_if_running(self) -> None:
        """Check if another instance of this script is already running"""
        script_name = Path(__file__).name
        current_pid = os.getpid()

        # Skip lock check if --force-run is specified
        if self.force_run:
            self.logger.warning("FORCE-RUN MODE: Bypassing process lock check")
            # Still create our own lock file
            self.lock_file.write_text(str(current_pid))
            self.logger.info(f"Created process lock file: {self.lock_file} (PID: {current_pid})")
            import atexit
            atexit.register(lambda: self.lock_file.unlink(missing_ok=True))
            return

        # Check if lock file exists
        if self.lock_file.exists():
            try:
                existing_pid = int(self.lock_file.read_text().strip())

                # Check if the process is still running and is our script
                if psutil.pid_exists(existing_pid):
                    try:
                        proc = psutil.Process(existing_pid)
                        if script_name in " ".join(proc.cmdline()):
                            self.logger.error(f"Another instance of {script_name} is already running (PID: {existing_pid})")
                            self.logger.error(f"If you're sure no other instance is running, remove the lock file:")
                            self.logger.error(f"  rm '{self.lock_file}'")
                            self.logger.error(f"Or kill the existing process:")
                            self.logger.error(f"  kill {existing_pid}")
                            sys.exit(1)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        # Process exists but it's not our script, remove stale lock file
                        self.logger.warning(f"Removing stale lock file (PID {existing_pid} is not our script)")
                        self.lock_file.unlink()
                else:
                    # Process doesn't exist, remove stale lock file
                    self.logger.warning(f"Removing stale lock file (PID {existing_pid} no longer exists)")
                    self.lock_file.unlink()
            except (ValueError, FileNotFoundError):
                # Invalid lock file content, remove it
                self.logger.warning("Removing invalid lock file")
                self.lock_file.unlink(missing_ok=True)

        # Create lock file with current PID
        self.lock_file.write_text(str(current_pid))
        self.logger.info(f"Created process lock file: {self.lock_file} (PID: {current_pid})")

        # Set up cleanup on exit
        import atexit
        atexit.register(lambda: self.lock_file.unlink(missing_ok=True))
    
    def _build_dependency_tree(self, frameworks: List[str], targets: List[str]) -> None:
        """
        Build a dependency tree for all build and execute tasks.
        
        Structure:
        1. Root tasks (can run immediately, in parallel):
           - dynamo-base builds for each framework (TRTLLM, VLLM, SGLANG)
        
        2. Framework dev builds (depend on their dynamo-base):
           - TRTLLM-dev depends on dynamo-base-TRTLLM
           - VLLM-dev depends on dynamo-base-VLLM
           - SGLANG-dev depends on dynamo-base-SGLANG
           
        3. Framework local-dev builds (depend on their dev):
           - TRTLLM-local-dev depends on TRTLLM-dev
           - VLLM-local-dev depends on VLLM-dev
           - SGLANG-local-dev depends on SGLANG-dev
           
        4. Compilation tasks (depend on their local-dev, run in parallel):
           - TRTLLM-compilation depends on TRTLLM-local-dev
           - VLLM-compilation depends on VLLM-local-dev
           - SGLANG-compilation depends on SGLANG-local-dev
           
        5. Sanity checks (depend on ALL compilations):
           - TRTLLM-sanity depends on all compilations
           - VLLM-sanity depends on all compilations
           - SGLANG-sanity depends on all compilations
        """
        self.logger.info("Building dependency tree...")
        self.logger.info("")
        
        # Step 1: Create all build tasks for each framework/target
        for framework in frameworks:
            self.logger.info(f"Creating tasks for {framework}...")
            
            for target in targets:
                # Get docker build commands from build.sh --dry-run
                target_arg = None if target == "dev" else target
                success, docker_commands = self.buildsh_utils.get_build_commands(framework, target_arg)
                
                if not success:
                    self.logger.error(f"  Failed to get build commands for {framework}/{target}")
                    continue
                
                # Process each docker command
                for docker_cmd in docker_commands:
                    # Check if we've seen this command before (duplicate detection)
                    if docker_cmd in self.seen_commands:
                        self.logger.debug(f"  Skipping duplicate command for {framework}/{target}")
                        continue
                    
                    # Extract metadata from command
                    base_image = self.docker_utils.extract_base_image_from_command(docker_cmd)
                    image_tag = self.docker_utils.extract_image_tag_from_command(docker_cmd)
                    
                    if not image_tag:
                        self.logger.warning(f"  No output tag found for {framework}/{target}")
                        continue
                    
                    # Determine task_id based on output tag
                    # Use framework as prefix for consistency
                    # Examples: "VLLM-dynamo-base", "VLLM-dev", "VLLM-local-dev"
                    if "dynamo-base" in image_tag:
                        task_id = f"{framework}-dynamo-base"
                        task_target = "base"
                    elif "local-dev" in image_tag:
                        task_id = f"{framework}-local-dev"
                        task_target = "local-dev"
                    else:
                        task_id = f"{framework}-dev"
                        task_target = "dev"
                    
                    # Create the build task
                    task = BuildTask(
                        task_id=task_id,
                        task_type="build",
                        framework=framework,
                        target=task_target,
                        description=f"Build {framework} {task_target} image",
                        docker_command=docker_cmd,
                        base_image=base_image,
                        image_tag=image_tag
                    )
                    
                    self.task_tree[task_id] = task
                    self.seen_commands[docker_cmd] = task_id
                    self.logger.info(f"  Created task: {task_id}")
        
        self.logger.info("")
        self.logger.info("Establishing dependencies...")
        
        # Step 2: Establish dependencies between build tasks
        for task_id, task in self.task_tree.items():
            # Only process build tasks
            if not isinstance(task, BuildTask):
                continue
            
            # Determine parent dependencies based on base_image
            if task.base_image:
                # Find which task produces this base_image
                # Prefer tasks with matching framework to handle cases where multiple
                # tasks produce the same tag (before conflict resolution)
                parent_task_id = None
                fallback_parent_id = None
                
                for other_id, other_task in self.task_tree.items():
                    if other_id == task_id:
                        continue
                    # Check if other_task is a BuildTask and produces the base_image we need
                    if isinstance(other_task, BuildTask) and task.base_image == other_task.image_tag:
                        # Prefer parent with matching framework
                        if task.framework and other_task.framework == task.framework:
                            parent_task_id = other_id
                            break
                        elif not fallback_parent_id:
                            fallback_parent_id = other_id
                
                # Use framework-matched parent if found, otherwise use fallback
                parent_task_id = parent_task_id or fallback_parent_id
                
                if parent_task_id:
                    task.add_dependency(parent_task_id)
                    self.task_tree[parent_task_id].add_child(task_id)
                    self.logger.info(f"  {task_id} depends on {parent_task_id}")
        
        # Step 3: Create compilation tasks (one per framework with local-dev)
        self.logger.info("")
        self.logger.info("Creating compilation tasks...")
        compilation_task_ids = []
        
        for framework in frameworks:
            local_dev_task_id = f"{framework}-local-dev"
            if local_dev_task_id not in self.task_tree:
                continue
            
            local_dev_task = self.task_tree[local_dev_task_id]
            
            # Ensure it's a build task (should always be true for local-dev)
            if not isinstance(local_dev_task, BuildTask):
                self.logger.warning(f"  {local_dev_task_id} is not a BuildTask")
                continue
            
            # Get the local-dev image tag
            local_dev_image = local_dev_task.image_tag
            if not local_dev_image or 'local-dev' not in local_dev_image:
                self.logger.warning(f"  No local-dev image tag found for {framework}")
                continue
            
            # Create compilation execute task
            compilation_task_id = f"{framework}-compilation"
            compilation_task = ExecuteTask(
                task_id=compilation_task_id,
                task_type="compilation",
                framework=framework,
                target="local-dev",
                description=f"Run workspace compilation in {framework} local-dev container",
                command=f"./container/run.sh --image {local_dev_image} --mount-workspace -- bash -c 'cargo build --locked --profile dev --features dynamo-llm/block-manager && cd /workspace/lib/bindings/python && maturin develop && uv pip install -e .'",
                image_name=local_dev_image,
                timeout=30 * 60.0
            )
            
            # Compilation depends on its local-dev build
            compilation_task.add_dependency(local_dev_task_id)
            local_dev_task.add_child(compilation_task_id)
            
            self.task_tree[compilation_task_id] = compilation_task
            compilation_task_ids.append(compilation_task_id)
            self.logger.info(f"  Created task: {compilation_task_id} (depends on {local_dev_task_id})")
        
        # Step 4: Create sanity check tasks (depend on ALL compilations)
        self.logger.info("")
        self.logger.info("Creating sanity check tasks...")
        
        for framework in frameworks:
            # Create sanity checks for both dev and local-dev targets
            for target in ['dev', 'local-dev']:
                task_id = f"{framework}-{target}"
                if task_id not in self.task_tree:
                    continue
                
                task = self.task_tree[task_id]
                
                # Ensure it's a build task
                if not isinstance(task, BuildTask):
                    self.logger.warning(f"  {task_id} is not a BuildTask")
                    continue
                
                # Get the image tag
                image_tag = task.image_tag
                if not image_tag:
                    continue
                
                # Create sanity check execute task
                sanity_task_id = f"{framework}-{target}-sanity"
                
                # Both dev and local-dev sanity checks should run after compilation
                compilation_task_id = f"{framework}-compilation"
                if compilation_task_id not in self.task_tree:
                    self.logger.warning(f"  No compilation task found for {framework}, skipping {target} sanity check")
                    continue
                
                depends_on_id = compilation_task_id
                description = f"Run sanity_check.py in {framework} {target} container (after compilation)"
                
                sanity_task = ExecuteTask(
                    task_id=sanity_task_id,
                    task_type="sanity_check",
                    framework=framework,
                    target=target,
                    description=description,
                    command=f"./container/run.sh --image {image_tag} --mount-workspace --entrypoint deploy/sanity_check.py",
                    image_name=image_tag,
                    timeout=120.0
                )
                
                # Add dependency
                sanity_task.add_dependency(depends_on_id)
                self.task_tree[depends_on_id].add_child(sanity_task_id)
                
                self.task_tree[sanity_task_id] = sanity_task
                self.logger.info(f"  Created task: {sanity_task_id} (depends on {depends_on_id})")
        
        self.logger.info("")
        self.logger.info(f"Dependency tree built: {len(self.task_tree)} tasks total")
    
    def _cleanup_existing_logs(self, frameworks: List[str]) -> None:
        """Clean up existing log files for the current SHA and specified frameworks"""
        commit_sha = self.git_utils.get_current_sha(self.dynamo_ci_dir)
        
        self.logger.info("")
        self.logger.info(f"Cleaning up existing log files for SHA: {commit_sha}")
        
        # Build patterns for all tasks related to specified frameworks
        # Task IDs follow pattern: {framework}-* (e.g., VLLM-dynamo-base, VLLM-dev, VLLM-compilation, VLLM-dev-sanity)
        patterns = []
        for framework in frameworks:
            framework_lower = framework.lower()
            # Matches all tasks with framework prefix
            patterns.extend([
                f"{self.date}.{commit_sha}.{framework_lower}-*.log",
                f"{self.date}.{commit_sha}.{framework_lower}-*.SUCC",
                f"{self.date}.{commit_sha}.{framework_lower}-*.FAIL"
            ])
        
        removed_files = []
        for pattern in patterns:
            for file_path in self.log_dir.glob(pattern):
                if file_path.is_file():
                    file_path.unlink()
                    removed_files.append(file_path.name)
        
        if removed_files:
            self.logger.info(f"Removed {len(removed_files)} existing log file(s) for {', '.join(frameworks)}")
            for file_name in removed_files:
                self.logger.debug(f"  Removed: {file_name}")
        else:
            self.logger.info(f"No existing log files found for {', '.join(frameworks)}")
    
    def _apply_sanity_check_only_mode(self) -> None:
        """
        Mark build and compilation tasks as skipped when in sanity-check-only mode.
        Also verify that required images exist for sanity checks.
        """
        if not self.sanity_check_only:
            return
        
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("SANITY-CHECK-ONLY MODE")
        self.logger.info("=" * 80)
        self.logger.info("Skipping build and compilation tasks...")
        self.logger.info("")
        
        # Track which images are required for sanity checks
        required_images: Set[str] = set()
        
        # Find all sanity check tasks and collect their required images
        sanity_tasks = []
        for task_id, task in self.task_tree.items():
            if isinstance(task, ExecuteTask):
                if task.task_type == "sanity_check":
                    sanity_tasks.append(task)
                    if task.image_name:
                        required_images.add(task.image_name)
                elif task.task_type == "compilation":
                    # Skip compilation tasks
                    task.status = 'skipped'
                    self.logger.info(f"  Skipping: {task_id} (compilation)")
            elif isinstance(task, BuildTask):
                # Skip all build tasks
                task.status = 'skipped'
                self.logger.info(f"  Skipping: {task_id} (build)")
        
        # Verify required images exist
        self.logger.info("")
        self.logger.info("Verifying required images exist...")
        missing_images = []
        
        for image_name in sorted(required_images):
            if self.dry_run:
                self.logger.info(f"  Would check: {image_name}")
            else:
                try:
                    # Try to get image info to verify it exists
                    docker_info = self.docker_utils.get_image_info(image_name)
                    if docker_info is None:
                        missing_images.append(image_name)
                        self.logger.error(f"  ❌ Missing: {image_name}")
                    else:
                        self.logger.info(f"  ✅ Found: {image_name} ({docker_info.size_human})")
                except Exception as e:
                    missing_images.append(image_name)
                    self.logger.error(f"  ❌ Missing: {image_name} ({e})")
        
        if missing_images and not self.dry_run:
            self.logger.error("")
            self.logger.error("=" * 80)
            self.logger.error("ERROR: Missing required Docker images")
            self.logger.error("=" * 80)
            self.logger.error("The following images are required but not found:")
            for img in missing_images:
                self.logger.error(f"  - {img}")
            self.logger.error("")
            self.logger.error("Please build the images first by running without --sanity-check-only")
            sys.exit(1)
        
        self.logger.info("")
        self.logger.info(f"Sanity check mode: {len(sanity_tasks)} sanity checks will run")
        self.logger.info("=" * 80)
    
    def _resolve_duplicate_images(self) -> None:
        """
        Resolve conflicts where multiple build tasks produce the same image tag.
        Rename conflicting tags by appending the framework name.
        Update the image_tag, docker_command, and any dependent tasks' base_image.
        """
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("Resolving image tag conflicts...")
        self.logger.info("=" * 80)
        
        # Track which image tags have been seen
        image_tag_to_tasks: Dict[str, List[str]] = {}  # image_tag -> [task_ids]
        
        # Step 1: Identify all tasks that produce each image tag
        for task_id, task in self.task_tree.items():
            if not isinstance(task, BuildTask):
                continue
            
            if not task.image_tag:
                continue
            
            # Skip external registry images (they're not built by us)
            if '/' in task.image_tag and not task.image_tag.startswith('dynamo'):
                continue
            
            image_tag = task.image_tag
            if image_tag not in image_tag_to_tasks:
                image_tag_to_tasks[image_tag] = []
            image_tag_to_tasks[image_tag].append(task_id)
        
        # Step 2: Find conflicts (tags produced by multiple tasks)
        conflicts: Set[str] = set()
        for image_tag, task_ids in image_tag_to_tasks.items():
            if len(task_ids) > 1:
                conflicts.add(image_tag)
                self.logger.info(f"  Conflict detected: {image_tag}")
                for task_id in task_ids:
                    self.logger.info(f"    - {task_id}")
        
        if not conflicts:
            self.logger.info("  No image tag conflicts found")
            return
        
        # Step 3: Rename conflicting tags by appending framework suffix
        # Track mapping from old producer task to new tag
        task_to_new_tag: Dict[str, str] = {}  # task_id -> new_tag
        
        for task_id, task in self.task_tree.items():
            if not isinstance(task, BuildTask):
                continue
            
            if not task.image_tag or task.image_tag not in conflicts:
                continue
            
            # Check if tag already has framework suffix
            if task.framework and task.image_tag.endswith(f"-{task.framework.lower()}"):
                continue
            
            # Create new tag with framework suffix
            old_tag = task.image_tag
            if task.framework:
                new_tag = f"{old_tag}-{task.framework.lower()}"
            else:
                # Fallback: use part of task_id
                new_tag = f"{old_tag}-{task_id.split('-')[0].lower()}"
            
            task_to_new_tag[task_id] = new_tag
            
            # Update the task's image_tag
            task.image_tag = new_tag
            
            # Update the docker command to use the new tag
            if task.docker_command:
                # Replace the old tag in --tag arguments
                task.docker_command = re.sub(
                    rf'--tag\s+{re.escape(old_tag)}(?=\s|$)',
                    f'--tag {new_tag}',
                    task.docker_command
                )
            
            self.logger.info(f"  Renamed: {task_id}")
            self.logger.info(f"    Old: {old_tag}")
            self.logger.info(f"    New: {new_tag}")
        
        # Step 4: Update base_image references in dependent tasks
        # Each dependent task should use the renamed tag from its parent
        for task_id, task in self.task_tree.items():
            if not isinstance(task, BuildTask):
                continue
            
            if not task.base_image or task.base_image not in conflicts:
                continue
            
            # Find which parent task this depends on
            for parent_task_id in task.depends_on:
                parent_task = self.task_tree.get(parent_task_id)
                if not isinstance(parent_task, BuildTask):
                    continue
                
                # Check if parent's image_tag matches our base_image (before renaming)
                # The parent should have been renamed already
                if parent_task_id in task_to_new_tag:
                    old_base = task.base_image
                    new_base = task_to_new_tag[parent_task_id]
                    task.base_image = new_base
                    self.logger.info(f"  Updated base_image: {task_id}")
                    self.logger.info(f"    {old_base} -> {new_base}")
                    
                    # Also update docker command if it references the old tag
                    if task.docker_command and old_base in task.docker_command:
                        task.docker_command = task.docker_command.replace(old_base, new_base)
                    break
        
        # Step 5: Update image_name in execute tasks
        # Build a reverse mapping: for each execute task, find the build task it depends on
        for task_id, task in self.task_tree.items():
            if not isinstance(task, ExecuteTask):
                continue
            
            if not task.image_name:
                continue
            
            # Check if image_name references a conflicting tag
            if task.image_name not in conflicts:
                continue
            
            # Find the build task this execute task depends on (directly or indirectly)
            for dep_task_id in task.depends_on:
                dep_task = self.task_tree.get(dep_task_id)
                if not dep_task:
                    continue
                
                # If the dependency is a build task and it was renamed, use its new tag
                if isinstance(dep_task, BuildTask) and dep_task_id in task_to_new_tag:
                    old_image = task.image_name
                    new_image = task_to_new_tag[dep_task_id]
                    task.image_name = new_image
                    self.logger.info(f"  Updated image_name: {task_id}")
                    self.logger.info(f"    {old_image} -> {new_image}")
                    
                    # Also update command if it references the old tag
                    if task.command and old_image in task.command:
                        task.command = task.command.replace(old_image, new_image)
                    break
                
                # If the dependency is an execute task, we need to trace back further
                # For now, we'll handle direct dependencies on build tasks
        
        self.logger.info(f"  Resolved {len(conflicts)} tag conflict(s) by renaming {len(task_to_new_tag)} image(s)")
    
    def _print_dependency_tree(self) -> None:
        """
        Visualize the dependency tree, showing parallel execution opportunities.
        """
        self.logger.info("=" * 80)
        self.logger.info("DEPENDENCY TREE")
        self.logger.info("=" * 80)
        self.logger.info("")
        
        # Find root tasks (tasks with no dependencies)
        root_tasks = [task_id for task_id, task in self.task_tree.items() if not task.depends_on]
        
        def print_task_tree(task_id: str, indent: int = 0, visited: set = None) -> None:
            """Recursively print task tree"""
            if visited is None:
                visited = set()
            
            if task_id in visited:
                return
            visited.add(task_id)
            
            task = self.task_tree[task_id]
            prefix = "  " * indent
            
            # Status icons
            status_icon = {
                'pending': '⏳',
                'ready': '🔓',
                'in_progress': '🔄',
                'success': '✅',
                'failed': '❌',
                'skipped': '⏭️'
            }.get(task.status, '❓')
            
            # Task type icons
            type_icon = {
                'build': '🔨',
                'compilation': '⚙️',
                'sanity_check': '🔍'
            }.get(task.task_type, '📋')
            
            self.logger.info(f"{prefix}{status_icon} {type_icon} {task_id}")
            self.logger.info(f"{prefix}    Type: {task.task_type}")
            if task.framework:
                self.logger.info(f"{prefix}    Framework: {task.framework}")
            if task.target:
                self.logger.info(f"{prefix}    Target: {task.target}")
            
            # Show build-specific info
            if isinstance(task, BuildTask):
                if task.base_image:
                    self.logger.info(f"{prefix}    From: {task.base_image}")
                if task.image_tag:
                    self.logger.info(f"{prefix}    Produces: {task.image_tag}")
            
            # Show execute-specific info
            if isinstance(task, ExecuteTask) and task.image_name:
                self.logger.info(f"{prefix}    Uses image: {task.image_name}")
                if task.timeout:
                    self.logger.info(f"{prefix}    Timeout: {task.timeout}s")
            
            if task.depends_on:
                self.logger.info(f"{prefix}    Depends on: {', '.join(task.depends_on)}")
            
            # Print children
            if task.children:
                self.logger.info(f"{prefix}    ↓")
                for child_id in task.children:
                    print_task_tree(child_id, indent + 1, visited)
        
        # Print tree starting from each root
        self.logger.info("Root tasks (can run in parallel):")
        self.logger.info("")
        for root_id in root_tasks:
            print_task_tree(root_id)
            self.logger.info("")
        
        # Print summary statistics
        self.logger.info("=" * 80)
        self.logger.info("SUMMARY")
        self.logger.info("=" * 80)
        
        # Count tasks by type
        type_counts: Dict[str, int] = {}
        status_counts: Dict[str, int] = {}
        
        for task in self.task_tree.values():
            type_counts[task.task_type] = type_counts.get(task.task_type, 0) + 1
            status_counts[task.status] = status_counts.get(task.status, 0) + 1
        
        self.logger.info(f"Total tasks: {len(self.task_tree)}")
        self.logger.info("")
        self.logger.info("By type:")
        for task_type, count in sorted(type_counts.items()):
            self.logger.info(f"  {task_type}: {count}")
        
        self.logger.info("")
        self.logger.info("By status:")
        for status, count in sorted(status_counts.items()):
            self.logger.info(f"  {status}: {count}")
        
        self.logger.info("=" * 80)
    
    def _execute_tasks(self) -> bool:
        """
        Execute all tasks in dependency order.
        Returns True if all tasks succeed, False otherwise.
        """
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("EXECUTING TASKS")
        self.logger.info("=" * 80)
        self.logger.info("")
        
        # Track execution state
        executed_count = 0
        failed_count = 0
        skipped_count = 0
        
        # Find all tasks that are ready to execute (no dependencies or all dependencies complete)
        while True:
            # Find ready tasks
            ready_tasks = []
            for task_id, task in self.task_tree.items():
                if task.status == 'pending' and task.is_ready(self.task_tree):
                    ready_tasks.append(task_id)
                elif task.status == 'skipped':
                    # Count skipped tasks
                    if task_id not in [t for t in self.task_tree.values() if t.status == 'skipped']:
                        skipped_count += 1
            
            if not ready_tasks:
                # Check if all tasks are complete
                pending_tasks = [t for t in self.task_tree.values() if t.status == 'pending']
                if not pending_tasks:
                    break
                else:
                    # We have pending tasks but none are ready - deadlock or missing dependencies
                    self.logger.error("Execution stuck - tasks remain pending but none are ready")
                    for task in pending_tasks:
                        self.logger.error(f"  Pending: {task.task_id} (depends on: {', '.join(task.depends_on)})")
                    return False

            # Execute ready tasks
            # TODO: Implement parallel execution when self.parallel is True
            # For now, execute sequentially even if --parallel is specified
            for task_id in ready_tasks:
                task = self.task_tree[task_id]
                
                # Skip already-skipped tasks (marked by --sanity-check-only mode)
                if task.status == 'skipped':
                    # Don't log this - already logged in _apply_sanity_check_only_mode
                    continue
                
                self.logger.info("")
                self.logger.info(f"Executing: {task_id}")
                self.logger.info(f"  Type: {task.task_type}")
                self.logger.info(f"  Description: {task.description}")
                
                task.status = 'in_progress'
                task.started_at = datetime.now()
                
                success = False
                
                if isinstance(task, BuildTask):
                    # Execute build task
                    success = self._execute_build_task(task)
                elif isinstance(task, ExecuteTask):
                    # Execute execute task (compilation, sanity check)
                    # Generate docker command from run.sh and execute it
                    command = self._generate_docker_command(task.command)
                    success = self._execute_command(
                        task=task,
                        command=command,
                        original_command=task.command,
                        timeout=task.timeout if task.timeout else 1800.0
                    )
                
                task.completed_at = datetime.now()
                task.duration = (task.completed_at - task.started_at).total_seconds()
                task.success = success
                
                if success:
                    task.status = 'success'
                    executed_count += 1
                    self.logger.info(f"  ✅ {task_id}: Success ({task.duration:.1f}s)")
                else:
                    task.status = 'failed'
                    failed_count += 1
                    self.logger.error(f"  ❌ {task_id}: Failed ({task.duration:.1f}s)")
                    # Don't continue if a task fails
                    self.logger.error("Stopping execution due to task failure")
                    return False
        
        # Count skipped tasks
        skipped_count = sum(1 for t in self.task_tree.values() if t.status == 'skipped')
        
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("EXECUTION SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Total tasks: {len(self.task_tree)}")
        self.logger.info(f"  Executed: {executed_count}")
        self.logger.info(f"  Failed: {failed_count}")
        self.logger.info(f"  Skipped: {skipped_count}")
        self.logger.info("=" * 80)
        
        return failed_count == 0
    
    def _execute_build_task(self, task: BuildTask) -> bool:
        """Execute a build task (docker build command)"""
        # Check if this command was already executed
        if self.is_command_executed(task.docker_command):
            self.logger.info(f"  Command already executed, skipping: {task.task_id}")
            task.status = 'skipped'
            return True
        
        # Execute the docker build command (no timeout for builds)
        success = self._execute_command(
            task=task,
            command=task.docker_command,
            timeout=float('inf')  # No timeout for builds
        )
        
        # Mark command as executed if successful
        if success:
            self.mark_command_executed(task.docker_command)
        
        # If successful, extract image ID from log file
        if success and not self.dry_run:
            commit_sha = self.git_utils.get_current_sha(self.dynamo_ci_dir)
            log_suffix = f"{commit_sha}.{task.task_id.lower()}"
            log_file = self.log_dir / f"{self.date}.{log_suffix}.log"
            
            log_content = log_file.read_text()
            image_id_match = re.search(r'writing image (sha256:[a-f0-9]+)', log_content)
            if image_id_match:
                full_id = image_id_match.group(1)
                # Extract 12 chars in docker images format
                task.image_id = full_id[7:19] if full_id.startswith('sha256:') else full_id[:12]
        
        return success
    
    def _execute_command(self, task: BaseTask, command: str, original_command: str = None, timeout: float = 1800.0) -> bool:
        """
        Execute a command with real-time output streaming to log file.
        
        Args:
            task: The task being executed
            command: The command to execute
            original_command: The original command (before transformation), for logging
            timeout: Timeout in seconds
        
        Returns:
            True if command succeeded, False otherwise
        """
        # Get git commit SHA for log file naming
        commit_sha = self.git_utils.get_current_sha(self.dynamo_ci_dir)
        log_suffix = f"{commit_sha}.{task.task_id.lower()}"
        
        log_file = self.log_dir / f"{self.date}.{log_suffix}.log"
        success_file = self.log_dir / f"{self.date}.{log_suffix}.SUCC"
        fail_file = self.log_dir / f"{self.date}.{log_suffix}.FAIL"
        
        # In dry-run mode, just show what would be executed and return
        if self.dry_run:
            self.logger.info(f"  Would execute: {command[:200]}...")
            return True
        
        # Execute command with real-time output to log file
        try:
            with open(log_file, 'a') as f:
                f.write("=" * 42 + "\n")
                f.write(f"Task: {task.task_id}\n")
                f.write(f"Type: {task.task_type}\n")
                f.write(f"Date: {datetime.now().strftime('%c')}\n")
                if original_command:
                    f.write(f"Original Command: {original_command}\n")
                    f.write(f"Actual Command: {command}\n")
                else:
                    f.write(f"Command: {command}\n")
                f.write(f"Timeout: {timeout}s\n")
                f.write("=" * 42 + "\n")
                f.flush()
                
                start_time = time.time()
                with subprocess.Popen(
                    command,
                    shell=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    text=True, 
                    bufsize=1,
                    cwd=self.dynamo_ci_dir
                ) as process:
                    for line in iter(process.stdout.readline, ''):
                        if time.time() - start_time > timeout:
                            self.logger.error(f"  Command timed out after {timeout}s")
                            process.terminate()
                            process.wait(timeout=5)
                            f.write(f"\nExecution Status: TIMEOUT (after {timeout}s)\n")
                            with open(fail_file, 'w') as ff:
                                ff.write(f"{task.task_id} timed out at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            task.error_message = f"Timeout after {timeout}s"
                            return False
                        
                        line = line.rstrip()
                        if line:
                            print(line)
                            f.write(line + '\n')
                            f.flush()
                    
                    return_code = process.wait()

                # Store return code if task supports it
                if hasattr(task, 'return_code'):
                    task.return_code = return_code

                if return_code == 0:
                    f.write("Execution Status: SUCCESS\n")
                    with open(success_file, 'w') as sf:
                        sf.write(f"{task.task_id} completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    return True
                else:
                    self.logger.error(f"  Command failed with exit code {return_code}")
                    f.write(f"Execution Status: FAILED (exit code {return_code})\n")
                    with open(fail_file, 'w') as ff:
                        ff.write(f"{task.task_id} failed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    
                    # Capture last 30 lines of log for error reporting
                    try:
                        with open(log_file, 'r') as log_f:
                            log_lines = log_f.readlines()
                            last_lines = log_lines[-30:] if len(log_lines) > 30 else log_lines
                            error_output = ''.join(last_lines)
                            if len(log_lines) > 30:
                                task.error_message = f"... (showing last 30 lines)\n{error_output}"
                            else:
                                task.error_message = error_output
                    except Exception:
                        task.error_message = f"Command failed with exit code {return_code}"
                    
                    return False

        except Exception as e:
            self.logger.error(f"  Exception during execution: {e}")
            task.error_message = str(e)
            with open(log_file, 'a') as f:
                f.write(f"Execution Status: EXCEPTION ({e})\n")
            with open(fail_file, 'w') as ff:
                ff.write(f"{task.task_id} exception at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            return False

    
    def _generate_docker_command(self, run_sh_command: str) -> str:
        """
        Generate docker command from run.sh command.
        
        This method:
        1. Calls run.sh --dry-run to get the base docker command
        2. Removes -it flags for non-interactive execution
        3. Appends any container command (after ' -- ') if present
        
        Args:
            run_sh_command: The run.sh command (e.g., './container/run.sh --image foo -- bash -c ...')
        
        Returns:
            The actual docker command to execute
        """
        if not run_sh_command.startswith('./container/run.sh'):
            raise ValueError(f"Expected run.sh command, got: {run_sh_command}")
        
        # Split command at ' -- ' to separate run.sh args from container command
        container_cmd = None
        run_sh_part = run_sh_command
        
        if ' -- ' in run_sh_command:
            run_sh_part, container_cmd = run_sh_command.split(' -- ', 1)
        
        # Run run.sh --dry-run to get the base docker command
        dry_run_cmd = run_sh_part.replace('./container/run.sh', './container/run.sh --dry-run')
        try:
            result = subprocess.run(
                dry_run_cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.dynamo_ci_dir
            )
            
            if result.returncode == 0:
                # Extract docker command from dry-run output
                output_text = result.stdout + result.stderr
                for line in output_text.split('\n'):
                    if line.strip().startswith('docker run'):
                        # Remove -it flags for non-interactive execution
                        docker_cmd = re.sub(r'-it\b', '', line.strip())
                        docker_cmd = re.sub(r'\s+', ' ', docker_cmd).strip()
                        
                        # Append container command if present
                        if container_cmd:
                            return f'{docker_cmd} {container_cmd}'
                        return docker_cmd
        except Exception as e:
            self.logger.warning(f"  Failed to extract docker command: {e}, using original")
        
        return run_sh_command
    
    def show_composite_sha(self, repo_sha: str = None) -> int:
        """Show which files are used for composite SHA calculation and the resulting SHA"""
        # Setup repository if repo_sha is specified
        if repo_sha:
            self.logger.info(f"Setting up repository for SHA: {repo_sha}")
            self.git_utils.setup_dynamo_ci(self.dynamo_ci_dir, repo_sha=repo_sha, no_checkout=False)
        
        # Generate composite SHA with file list
        sha, files_used = self.git_utils.generate_composite_sha_from_container_dir(self.dynamo_ci_dir)
        
        if sha is None:
            self.logger.error("Failed to generate composite SHA")
            return 1
        
        print(f"\nComposite SHA Calculation for {'SHA ' + repo_sha if repo_sha else 'current state'}:")
        print(f"Repository: {self.dynamo_ci_dir}")
        print(f"Container directory: {self.dynamo_ci_dir / 'container'}")
        print(f"\nResulting SHA: {sha}")
        print(f"\nFiles used in calculation ({len(files_used)} files):")
        
        if not files_used:
            print("  No files found for SHA calculation")
            return 1
        
        # Group files by directory for better readability
        files_by_dir = {}
        for file_path in sorted(files_used):
            dir_path = file_path.parent
            if dir_path not in files_by_dir:
                files_by_dir[dir_path] = []
            files_by_dir[dir_path].append(file_path.name)
        
        for dir_path in sorted(files_by_dir.keys()):
            print(f"\n  {dir_path}/")
            for filename in sorted(files_by_dir[dir_path]):
                print(f"    {filename}")
        
        # Show exclusion rules
        print(f"\nExclusion rules applied:")
        print(f"  - Hidden files/directories (starting with '.')")
        print(f"  - Extensions: .md, .rst, .log, .bak, .tmp, .swp, .swo, .orig, .rej")
        print(f"  - Filenames: README, CHANGELOG, LICENSE, NOTICE, AUTHORS, CONTRIBUTORS")
        print(f"  - Specific files: launch_message.txt")
        
        return 0
    
    def show_commit_history(self, max_commits: int = 50) -> int:
        """Show all past commits with their composite SHAs"""
        self.logger.info("Analyzing commit history and generating composite SHAs...")
        
        # Setup repository (use existing or clone if needed)
        self.git_utils.setup_dynamo_ci(self.dynamo_ci_dir, no_checkout=self.no_checkout)
        
        # Get commit history
        commits = self.git_utils.get_commit_history(self.dynamo_ci_dir, max_commits=max_commits)
        
        if not commits:
            self.logger.error("No commits found in repository")
            return 1
        
        print(f"\nCommit History with Composite SHAs")
        print(f"Repository: {self.dynamo_ci_dir}")
        print(f"Showing {len(commits)} most recent commits:\n")
        print(f"{'Commit SHA':<10} {'Composite SHA':<16} {'Date':<20} {'Author':<25} {'Message'}")
        print("-" * 120)
        
        # Store original HEAD position to restore later
        repo = self.git_utils._get_repo(self.dynamo_ci_dir)
        original_head = repo.head.commit.hexsha
        
        try:
            for i, commit_info in enumerate(commits):
                commit_sha = commit_info['sha']
                full_sha = commit_info['full_sha']
                
                # Format the output first (without composite SHA)
                date_str = commit_info['pacific_timestamp'][:19]  # Remove timezone for brevity
                author_str = commit_info['author_name'][:24]  # Truncate long author names
                message_str = commit_info['commit_message'][:60]  # Truncate long messages
                
                # Print the line with "CALCULATING..." placeholder first
                print(f"{commit_sha:<10} {'CALCULATING...':<16} {date_str:<20} {author_str:<25} {message_str}", end='', flush=True)
                
                # Now calculate the composite SHA for this commit
                if not self.dry_run:
                    try:
                        self.logger.debug(f"Checking out commit {commit_sha} for composite SHA calculation")
                        repo.git.checkout(full_sha)
                        
                        # Temporarily suppress the composite SHA generation logging
                        original_level = self.git_utils.logger.level
                        self.git_utils.logger.setLevel(logging.WARNING)
                        
                        # Generate composite SHA for this commit
                        composite_sha, _ = self.git_utils.generate_composite_sha_from_container_dir(self.dynamo_ci_dir)
                        
                        # Restore original logging level
                        self.git_utils.logger.setLevel(original_level)
                        
                        if composite_sha is None:
                            composite_sha = "ERROR"
                        else:
                            composite_sha = composite_sha[:12]  # Show first 12 characters
                            
                    except Exception as e:
                        self.logger.warning(f"Failed to checkout commit {commit_sha}: {e}")
                        composite_sha = "UNAVAILABLE"
                else:
                    # In dry-run mode, we can't checkout commits
                    composite_sha = "DRY-RUN"
                
                # Go back to beginning of line and overwrite with the actual composite SHA
                print(f"\r{commit_sha:<10} {composite_sha:<16} {date_str:<20} {author_str:<25} {message_str}")
                
                # Add a small delay to avoid overwhelming the system
                if not self.dry_run and i < len(commits) - 1:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            self.logger.info("\nOperation interrupted by user")
            return 1
        except Exception as e:
            self.logger.error(f"Error during commit history analysis: {e}")
            return 1
        finally:
            # Restore original HEAD position
            if not self.dry_run:
                try:
                    self.logger.debug(f"Restoring original HEAD position: {original_head[:7]}")
                    repo.git.checkout(original_head)
                except Exception as e:
                    self.logger.warning(f"Failed to restore original HEAD position: {e}")
        
        print(f"\nAnalysis complete. Processed {len(commits)} commits.")
        if self.dry_run:
            print("Note: Composite SHAs shown as 'DRY-RUN' because --dry-run mode was used.")
        
        return 0
    
    def main(self) -> int:
        """Main function for DynamoDockerBuilder"""
        parser = create_argument_parser()
        parser.description = "DynamoDockerBuilder - Automated Docker Build and Test System with Dependency Tree"
        args = parser.parse_args()

        # Update repo path if specified
        if args.repo_path:
            self.dynamo_ci_dir = Path(args.repo_path).absolute()
            self.log_dir = self.dynamo_ci_dir / "logs" / self.date

        # Set configuration flags
        self.dry_run = args.dry_run
        self.sanity_check_only = args.sanity_check_only
        self.no_checkout = args.no_checkout
        self.force_run = args.force_run
        self.parallel = args.parallel
        self.email = args.email
        self.repo_sha = args.repo_sha

        # Handle show-composite-sha flag
        if hasattr(args, 'show_composite_sha') and args.show_composite_sha:
            return self.show_composite_sha(self.repo_sha)
        
        # Handle show-commit-history flag
        if hasattr(args, 'show_commit_history') and args.show_commit_history:
            max_commits = getattr(args, 'max_commits', 50)
            return self.show_commit_history(max_commits)

        # Parse targets
        self.targets = [target.strip() for target in args.target.split(',') if target.strip()]
        if not self.targets:
            self.targets = ["dev", "local-dev"]  # Fallback to default

        # Validate targets
        valid_targets = ["dev", "local-dev", "runtime", "dynamo_base", "framework"]
        invalid_targets = [t for t in self.targets if t not in valid_targets]
        if invalid_targets:
            valid_targets_str = ", ".join(valid_targets)
            invalid_targets_str = ", ".join(invalid_targets)
            self.logger.error(f"Invalid target(s) '{invalid_targets_str}'. Valid Docker build targets are: {valid_targets_str}")
            return 1

        # Determine which frameworks to test
        if args.framework:
            frameworks_to_test = []
            for framework in args.framework:
                framework_upper = framework.upper()
                if framework_upper not in self.FRAMEWORKS:
                    valid_frameworks = ", ".join(self.FRAMEWORKS)
                    self.logger.error(f"Invalid framework '{framework}'. Valid options are: {valid_frameworks}")
                    return 1
                frameworks_to_test.append(framework_upper)
        else:
            frameworks_to_test = list(self.FRAMEWORKS)

        # Check if another instance is already running (unless in sanity-check-only mode)
        if not self.sanity_check_only:
            self.check_if_running()

        # Print header
        print("=" * 80)
        self.logger.info(f"Starting DynamoDockerBuilder - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("V2 Implementation - Dependency Tree Architecture")
        print("=" * 80)

        if self.dry_run:
            self.logger.info("DRY-RUN MODE: Commands will be shown but not executed")
        if self.sanity_check_only:
            self.logger.info("TEST-ONLY MODE: Skipping build step and only running tests")
        if self.no_checkout:
            self.logger.info("NO-CHECKOUT MODE: Skipping git operations and using existing repository")
        if self.force_run:
            self.logger.info("FORCE-RUN MODE: Will run regardless of file changes")
        if self.repo_sha:
            self.logger.info(f"REPO-SHA MODE: Will checkout specific SHA: {self.repo_sha}")
        if self.sanity_check_only:
            self.logger.info("SANITY-CHECK-ONLY MODE: Will skip builds and compilation, only run sanity checks")
        if self.parallel:
            self.logger.info("PARALLEL MODE: Tasks will be executed in parallel when dependencies allow")
        else:
            self.logger.info("SERIAL MODE: Tasks will be executed sequentially (use --parallel for parallel execution)")
        
        self.logger.info(f"Dynamo CI Directory: {self.dynamo_ci_dir}")
        self.logger.info(f"Log Directory: {self.log_dir}")

        # Setup repository
        if not self.no_checkout:
            self.git_utils.setup_dynamo_ci(self.dynamo_ci_dir, self.repo_sha, self.no_checkout)
        
        # Check if rebuild is needed based on file changes (unless in sanity-check-only mode)
        if not self.sanity_check_only:
            rebuild_needed = self._check_if_rebuild_needed()
            if not rebuild_needed:
                self.logger.info("SUCCESS: No rebuild needed - all files unchanged")
                self.logger.info("Exiting early (use --force-run to force rebuild)")
                self.logger.info("Preserving existing log files")
                return 0
        
        # Build dependency tree
        self._build_dependency_tree(frameworks_to_test, self.targets)
        
        # Resolve duplicate images
        self._resolve_duplicate_images()
        
        # Apply sanity-check-only mode (marks tasks as skipped and verifies images exist)
        self._apply_sanity_check_only_mode()
        
        # Visualize dependency tree
        self._print_dependency_tree()
        
        # Set up log directory
        self._setup_logging_dir()
        
        # Clean up existing log files for the frameworks we're testing
        self._cleanup_existing_logs(frameworks_to_test)
        
        # Execute tasks
        success = self._execute_tasks()
        
        self.logger.info("")
        self.logger.info("=" * 80)
        if success:
            self.logger.info("All tasks completed successfully!")
        else:
            self.logger.error("Some tasks failed")
        self.logger.info("=" * 80)
        
        # Generate HTML report and prepare git info
        commit_sha = self.git_utils.get_current_sha(self.dynamo_ci_dir)
        html_file = self.log_dir / f"{self.date}.{commit_sha}.report.html"
        
        # Get git information for both report and email
        git_info = self.git_utils.get_commit_info(self.dynamo_ci_dir)
        git_info['sha'] = commit_sha
        
        # Map git_info keys to template expectations
        pacific = git_info.get('pacific_timestamp', '')
        utc = git_info.get('utc_timestamp', '')
        git_info['date'] = f"{pacific} ({utc})" if pacific and utc else (pacific or utc or '')
        git_info['message'] = git_info.get('commit_message', '')
        git_info['full_message'] = git_info.get('full_commit_message', '')
        
        # Build & Test Date with both PDT and UTC
        import zoneinfo
        now_utc = datetime.now(zoneinfo.ZoneInfo("UTC"))
        now_pdt = datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles"))
        build_date = f"{now_pdt.strftime('%Y-%m-%d %H:%M:%S %Z')} ({now_utc.strftime('%Y-%m-%d %H:%M:%S %z')})"
        
        # Generate HTML content (for both report and email)
        email_renderer = EmailTemplateRenderer()
        html_content = email_renderer.generate_v2_report(
            task_tree=self.task_tree,
            git_info=git_info,
            date=build_date,
            docker_utils=self.docker_utils,
            repo_path=str(self.dynamo_ci_dir),
            log_dir=str(self.log_dir)
        )
        
        # Save HTML report to file (skip in dry-run)
        if not self.dry_run:
            try:
                html_file.write_text(html_content)
                self.logger.info(f"HTML report saved to: {html_file}")
            except Exception as e:
                self.logger.error(f"Failed to generate HTML report: {e}")
        
        # Send email notification if requested
        if self.email:
            self.logger.info(f"Email notifications requested for: {self.email}")
            if not self.dry_run:
                # Use EmailTemplateRenderer's shared SMTP sending logic
                email_renderer._send_html_email_via_smtp(
                    email=self.email,
                    html_content=html_content,
                    subject_prefix="DynamoDockerBuilder2",
                    git_sha=git_info.get('sha', 'unknown'),
                    failed_tasks=[t.task_id for t in self.task_tree.values() if t.status == 'failed'],
                    logger=self.logger
                )
            else:
                self.logger.info(f"DRY-RUN: Would send email notification to {self.email}")
            self.logger.info(f"  (HTML report available at: {html_file})")
        
        return 0 if success else 1

def create_argument_parser() -> argparse.ArgumentParser:
    """Create shared argument parser for DynamoDockerBuilder and DynamoDockerBuilder"""
    parser = argparse.ArgumentParser(description="DynamoDockerBuilder - Automated Docker Build and Test System")
    parser.add_argument('-f', '--framework', '--frameworks', action='append', dest='framework',
                      help=f"Test specific framework ({', '.join(FRAMEWORKS_UPPER)}) - case insensitive. Can be specified multiple times.")
    parser.add_argument('--target', default='dev,local-dev',
                      help="Comma-separated Docker build targets to test: dev, local-dev (default: dev,local-dev)")
    parser.add_argument('-a', '--all', action='store_true', dest='all_frameworks',
                      help="Test all frameworks (default)")
    parser.add_argument('--sanity-check-only', '--test-only', action='store_true', dest='sanity_check_only',
                      help="Skip build and compilation steps, only run sanity checks (assumes images already exist)")
    parser.add_argument('--no-checkout', action='store_true',
                      help="Skip git operations and use existing repository")
    parser.add_argument('--force-run', action='store_true',
                      help="Force run even if files haven't changed or another process is running")
    parser.add_argument('--dry-run', '--dryrun', action='store_true', dest='dry_run',
                      help="Show commands that would be executed without running them")
    parser.add_argument('--repo-path', type=str,
                      help="Path to the dynamo repository (default: ../dynamo_ci)")
    parser.add_argument('--email', type=str,
                      help="Email address for notifications (sends email if specified)")
    parser.add_argument('--repo-sha', type=str,
                      help="Git SHA to checkout instead of latest main branch")
    parser.add_argument('--parallel', action='store_true',
                      help="Execute tasks in parallel when possible (default: serial execution)")
    parser.add_argument('--show-composite-sha', action='store_true',
                      help="Show which files are used for composite SHA calculation and the resulting SHA for the specified --repo-sha")
    parser.add_argument('--show-commit-history', action='store_true',
                      help="Show all past commits with their composite SHAs in chronological order")
    parser.add_argument('--max-commits', type=int, default=50,
                      help="Maximum number of commits to show in commit history (default: 50)")
    return parser


if __name__ == "__main__":
    builder = DynamoDockerBuilder()
    sys.exit(builder.main())
