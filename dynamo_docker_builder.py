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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from jinja2 import Template
import psutil
import docker
import git
import zoneinfo


class BaseUtils:
    """Base class for all utility classes with common logger and cmd functionality"""
    
    def __init__(self, dry_run: bool = False, logger=None):
        self.dry_run = dry_run
        self.logger = logger or logging.getLogger(self.__class__.__name__)
    
    def cmd(self, command: List[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Execute command with dry-run support - base implementation"""
        cmd_str = " ".join(shlex.quote(str(arg)) for arg in command)
        self.logger.debug(f"+ {cmd_str}")
        
        if self.dry_run:
            # Return a mock completed process in dry-run mode
            mock_result = subprocess.CompletedProcess(command, 0)
            mock_result.stdout = ""
            mock_result.stderr = ""
            return mock_result
        else:
            return subprocess.run(command, **kwargs)


class GitUtils(BaseUtils):
    """Utility class for Git operations"""
    
    def __init__(self, dry_run: bool = False, logger=None):
        super().__init__(dry_run, logger)
        # Cache for git.Repo objects to avoid repeated instantiation
        self._repo_cache: Dict[str, git.Repo] = {}
    
    def _get_repo(self, repo_dir: Path) -> git.Repo:
        """Get cached git.Repo object or create new one"""
        repo_key = str(repo_dir.absolute())
        if repo_key not in self._repo_cache:
            self.logger.debug(f"Opening git repository: {repo_dir}")
            self._repo_cache[repo_key] = git.Repo(repo_dir)
        return self._repo_cache[repo_key]
    
    def get_commit_info(self, repo_dir: Path) -> Dict[str, Any]:
        """Get comprehensive git information using GitPython library"""
        repo = self._get_repo(repo_dir)
        
        self.logger.debug(f"Equivalent: git -C {repo_dir} log -1 --format='%h %cd %s %B %an %ae' --date=iso")
        commit = repo.head.commit
        
        sha = commit.hexsha[:7]
        
        commit_datetime = commit.committed_datetime
        utc_timestamp = commit_datetime.strftime('%Y-%m-%d %H:%M:%S %z')
        
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
        
        changed_files = []
        file_stats = []
        
        self.logger.debug(f"Equivalent: git -C {repo_dir} diff-tree --no-commit-id --name-only -r HEAD")
        if commit.parents:  # If this isn't the initial commit
            parent = commit.parents[0]
            diff = parent.diff(commit)
            
            for diff_item in diff:
                if diff_item.a_path:
                    filename = diff_item.a_path
                    changed_files.append(filename)
                    
                    file_stats.append({
                        'file': filename,
                        'additions': 'unknown',
                        'deletions': 'unknown'
                    })
        
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
            'total_files_changed': len(changed_files)
        }

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
    
    def generate_composite_sha_from_container_dir(self, repo_dir: Path) -> Optional[str]:
        """Generate composite SHA from all container files recursively"""
        container_dir = repo_dir / "container"

        if container_dir.exists():
            self.logger.debug(f"Generating composite SHA from container directory: {container_dir}")
        else:
            self.logger.error(f"Container directory not found: {container_dir}")
            return None

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
            return None

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
                    return None

                # Generate SHA256 of concatenated files
                temp_file.flush()
                with open(temp_path, 'rb') as f:
                    sha = hashlib.sha256(f.read()).hexdigest()

                self.logger.info(f"Generated composite SHA from {found_files} container files (excluded {excluded_count}): {sha[:12]}...")
                return sha

            finally:
                temp_path.unlink(missing_ok=True)



class DockerUtils(BaseUtils):
    """Utility class for Docker operations"""
    
    def __init__(self, dry_run: bool = False, logger=None):
        super().__init__(dry_run, logger)
        
        # Initialize Docker client - required for operation
        try:
            self.logger.info("Equivalent: docker version")
            self.client = docker.from_env()
            # Test connection
            self.client.ping()
            self.logger.debug("Docker client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Docker client: {e}")
            self.logger.error("Docker client is required for this script to function.")
            self.logger.error("Please ensure Docker daemon is running and accessible.")
            sys.exit(1)
    
    def get_image_info(self, image_tag: str) -> Dict[str, str]:
        """Get Docker image information including size and full repo:tag"""
        try:
            # Primary approach: Use Docker API directly
            self.logger.info(f"Equivalent: docker inspect {image_tag}")
            try:
                image = self.client.images.get(image_tag)
                
                # Get image size
                size_bytes = image.attrs.get('Size', 0)
                
                # Format size in human readable format
                if size_bytes == 0:
                    size = "0 B"
                else:
                    # Convert bytes to human readable format
                    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                        if size_bytes < 1024.0:
                            size = f"{size_bytes:.1f} {unit}"
                            break
                        size_bytes /= 1024.0
                    else:
                        size = f"{size_bytes:.1f} PB"
                
                # Get repo:tag from image tags
                repo_tag = image_tag
                if image.tags:
                    # Use the first tag that matches our query or just the first tag
                    matching_tags = [tag for tag in image.tags if image_tag in tag]
                    if matching_tags:
                        repo_tag = matching_tags[0]
                    else:
                        repo_tag = image.tags[0]
                
                return {
                    'repo_tag': repo_tag,
                    'size': size,
                    'size_bytes': str(image.attrs.get('Size', 0))
                }
            except docker.errors.ImageNotFound:
                self.logger.debug(f"Docker image not found via API: {image_tag}")

                # Try alternative API approach: list images with filter
                self.logger.info(f"Equivalent: docker images --filter reference={image_tag}")
                images = self.client.images.list(filters={'reference': image_tag})
                if images:
                    image = images[0]
                    
                    # Get image size
                    size_bytes = image.attrs.get('Size', 0)
                    
                    # Format size in human readable format
                    if size_bytes == 0:
                        size = "0 B"
                    else:
                        # Convert bytes to human readable format
                        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                            if size_bytes < 1024.0:
                                size = f"{size_bytes:.1f} {unit}"
                                break
                            size_bytes /= 1024.0
                        else:
                            size = f"{size_bytes:.1f} PB"
                    
                    # Get repo:tag from image tags
                    repo_tag = image_tag
                    if image.tags:
                        repo_tag = image.tags[0]
                    
                    return {
                        'repo_tag': repo_tag,
                        'size': size,
                        'size_bytes': str(image.attrs.get('Size', 0))
                    }
                else:
                    # Image truly not found
                    raise docker.errors.ImageNotFound(f"Image {image_tag} not found")

        except docker.errors.ImageNotFound:
            # Image doesn't exist - return default values
            return {
                'repo_tag': image_tag,
                'size': 'Not Found',
                'size_bytes': '0'
            }
        except Exception as e:
            self.logger.warning(f"Docker API failed for {image_tag}: {e}")
            # Only fall back to CLI if Docker API completely fails
            try:
                # Final fallback to subprocess only if API is broken
                # Equivalent: docker images --format 'table {{.Repository}}:{{.Tag}}\t{{.Size}}' --no-trunc <image_tag>
                result = subprocess.run([
                    'docker', 'images', '--format', 'table {{.Repository}}:{{.Tag}}\t{{.Size}}',
                    '--no-trunc', image_tag
                ], capture_output=True, text=True)

                if result.returncode == 0 and result.stdout.strip():
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:  # Skip header line
                        data_line = lines[1]
                        parts = data_line.split('\t')
                        if len(parts) >= 2:
                            repo_tag = parts[0].strip()
                            size = parts[1].strip()
                        else:
                            repo_tag = image_tag
                            size = "Unknown"
                    else:
                        repo_tag = image_tag
                        size = "Unknown"
                    
                    # Get size in bytes using CLI
                    # Equivalent: docker inspect --format '{{.Size}}' <image_tag>
                    try:
                        inspect_result = subprocess.run([
                            'docker', 'inspect', '--format', '{{.Size}}', image_tag
                        ], capture_output=True, text=True)
                        size_bytes = inspect_result.stdout.strip() if inspect_result.returncode == 0 else "0"
                    except Exception:
                        size_bytes = "0"

                    return {
                        'repo_tag': repo_tag,
                        'size': size,
                        'size_bytes': size_bytes
                    }

                # Fallback if all methods fail
                return {
                    'repo_tag': image_tag,
                    'size': 'Unknown',
                    'size_bytes': '0'
                }
            except Exception as cli_e:
                self.logger.warning(f"Both Docker API and CLI failed for {image_tag}: API={e}, CLI={cli_e}")
                return {
                    'repo_tag': image_tag,
                    'size': 'Error',
                    'size_bytes': '0'
                }
    
    def get_image_id(self, image_tag: str) -> str:
        """Get Docker image ID using Docker API - fails hard if image not found"""
        self.logger.info(f"Equivalent: docker inspect --format '{{{{.Id}}}}' {image_tag}")
        image = self.client.images.get(image_tag)
        full_image_id = image.id
        # Get short image ID (first 12 chars after sha256:)
        if full_image_id.startswith('sha256:'):
            return full_image_id[7:19]
        else:
            return full_image_id[:12]
    
    def normalize_command(self, docker_command: str) -> str:
        """
        Normalize docker command for consistent comparison.
        
        This removes variations in whitespace, argument ordering, etc.
        to enable accurate duplicate detection.
        
        Args:
            docker_command: Raw docker build command string
            
        Returns:
            Normalized command string for comparison
        """
        if not docker_command.strip():
            return ""
            
        # Remove extra whitespace and normalize
        normalized = ' '.join(docker_command.split())
        
        # Sort build arguments for consistent comparison
        # This handles cases where --build-arg arguments appear in different orders
        parts = normalized.split()
        build_args = []
        other_parts = []
        
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
            # Find where to insert build args (after docker build but before context)
            if 'docker' in other_parts and 'build' in other_parts:
                try:
                    build_idx = other_parts.index('build')
                    # Insert sorted build args after 'build'
                    result_parts = other_parts[:build_idx + 1] + build_args + other_parts[build_idx + 1:]
                except ValueError:
                    result_parts = other_parts + build_args
            else:
                result_parts = other_parts + build_args
        else:
            result_parts = other_parts
            
        return ' '.join(result_parts)

    def filter_unused_build_args(self, docker_command: str) -> str:
        """
        Remove unused --build-arg flags from Docker build commands for base images.
        
        Base images (dynamo-base) don't use most of the build arguments that are
        passed to framework-specific builds. Removing unused args helps Docker
        recognize when builds are truly identical.
        
        Args:
            docker_command: Docker build command string
            
        Returns:
            Filtered command string with unused build args removed
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
        
        if filtered_args:
            self.logger.info(f"Filtered {len(filtered_args)} unused base image build args: {', '.join(sorted(filtered_args))}")
        
        return ' '.join(filtered_parts)


class BuildShUtils(BaseUtils):
    """Utility class for build.sh-related operations"""
    
    def __init__(self, dynamo_ci_dir: Path, docker_utils: 'DockerUtils', dry_run: bool = False, logger=None):
        super().__init__(dry_run, logger)
        self.dynamo_ci_dir = dynamo_ci_dir
        self.docker_utils = docker_utils
        # Cache for build commands to avoid repeated expensive calls
        self._build_commands_cache: Dict[str, Tuple[bool, List[str], str]] = {}
    
    def extract_base_image_tag(self, framework: str) -> str:
        """Extract the dynamo-base image tag from build commands for a framework"""
        try:
            # Get build commands for the framework
            success, docker_commands, _ = self.get_build_commands(framework, None)
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
    
    def get_build_commands(self, framework: str, docker_target_type: Optional[str]) -> Tuple[bool, List[str], str]:
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
            return False, [], ""

        # Extract and filter docker commands to remove latest tags
        docker_commands = []
        versioned_tags = []
        framework_lower = framework.lower()
        output_lines = build_result.stdout.split('\n') + build_result.stderr.split('\n')

        for line in output_lines:
            line = line.strip()
            if line.startswith("docker build"):
                # Remove --tag arguments containing ":latest" using regex
                line = re.sub(r'--tag\s+\S*:latest\S*(?:\s|$)', '', line)
                
                # Extract versioned tags for image discovery
                tag_matches = re.findall(r'--tag\s+(\S+)', line)
                for tag in tag_matches:
                    if framework_lower in tag:
                        versioned_tags.append(tag)

                if line.strip():
                    # Apply unused argument filtering
                    final_command = self.docker_utils.filter_unused_build_args(line)
                    if final_command:
                        docker_commands.append(final_command)

        if not docker_commands:
            self.logger.error(f"No docker build commands found for {framework}")
            return False, [], ""

        # Discover the appropriate image tag for this target type
        image_tag = ""
        if docker_target_type == "local-dev":
            # Look for local-dev tag
            for tag in versioned_tags:
                if "local-dev" in tag and framework_lower in tag:
                    image_tag = tag
                    break
        else:
            # Look for dev tag (without local-dev)
            for tag in versioned_tags:
                if "local-dev" not in tag and framework_lower in tag:
                    image_tag = tag
                    break

        if image_tag:
            result = True, docker_commands, image_tag
        else:
            target_desc = "dev" if docker_target_type is None else docker_target_type
            self.logger.error(f"Could not find versioned tag for {framework} {target_desc} image")
            result: Tuple[bool, List[str], str] = False, [], ""
        
        # Cache the result for future calls
        self._build_commands_cache[cache_key] = result
        return result


class EmailTemplateRenderer:
    """Handles email template rendering and HTML formatting for DynamoDockerBuilder reports"""
    
    def __init__(self):
        """Initialize the EmailTemplateRenderer"""
        pass
    
    def get_email_template(self) -> str:
        """Get the Jinja2 email template"""
        return """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
body { font-family: Arial, sans-serif; margin: 10px; line-height: 1.3; }
.header { background-color: {{ status_color }}; color: white; padding: 15px 20px; border-radius: 4px; margin-bottom: 10px; text-align: center; }
.summary { background-color: #f8f9fa; padding: 4px 6px; border-radius: 2px; margin: 3px 0; }
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
<h2>DynamoDockerBuilder - {{ overall_status }}</h2>
</div>

<div class="summary">
<p><strong>Build & Test Date:</strong> {{ build_date }}</p>
<p><strong>Total Builds:</strong> {{ total_builds }} | <strong>Total Tests:</strong> {{ total_tests }}</p>
<p><strong>Passed:</strong> <span class="success">{{ passed_tests }}</span> | <strong>Failed:</strong> <span class="failure">{{ failed_tests }}</span></p>
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
<div class="framework-header">Workspace Compilation</div>
<div class="results-chart">
<div class="chart-row">
<div class="chart-cell chart-header">Status</div>
<div class="chart-cell chart-header">Time</div>
<div class="chart-cell chart-header">Details</div>
</div>
<div class="chart-row">
<div class="chart-cell chart-status">
{% if workspace_compilation.success %}
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
<h4 style="margin-bottom: 6px;">Compilation Output Snippets (last 7 lines):</h4>

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
{% if target.success %}
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
    
    def send_email_notification(
        self,
        email: str,
        results: Dict[str, bool],
        git_info: Dict[str, Any],
        dynamo_ci_dir: Path,
        log_dir: Path,
        base_build_times: Dict[str, float],
        base_image_ids: Dict[str, str],
        workspace_compile_success: Optional[bool],
        workspace_compile_time: float,
        workspace_compile_output: Dict[str, List[str]],
        failure_details: Optional[Dict[str, str]],
        build_times: Optional[Dict[str, float]],
        dry_run: bool,
        logger,
        docker_utils,
        buildsh_utils
    ) -> None:
        """Send email notification with test results using Jinja2 template
        
        Args:
            email: Recipient email address
            results: Dict[str, bool] mapping test keys (e.g. 'VLLM_dev') to success boolean
            git_info: Git commit information dictionary
            dynamo_ci_dir: Path to dynamo CI directory
            log_dir: Path to log directory
            base_build_times: Dictionary of base image build times
            base_image_ids: Dictionary of base image IDs
            workspace_compile_success: Workspace compilation success status (None if not attempted)
            workspace_compile_time: Workspace compilation time in seconds
            workspace_compile_output: Dictionary of output snippets from compilation
            failure_details: Optional[Dict[str, str]] mapping failed test keys to error output strings
            build_times: Optional[Dict[str, float]] mapping timing keys to duration in seconds
            dry_run: Whether this is a dry run
            logger: Logger instance
            docker_utils: DockerUtils instance
            buildsh_utils: BuildShUtils instance
        """
        if not email:
            return

        if failure_details is None:
            failure_details = {}

        if dry_run:
            logger.info(f"DRY-RUN: Would send email notification to {email}")
            return

        try:
            # Count results (only count tests that were actually run)
            total_tests = len(results)
            passed_tests = sum(1 for success in results.values() if success)
            failed_tests = sum(1 for success in results.values() if not success)

            # Calculate total builds (same as total tests since each target involves both build and test)
            # In test-only mode, builds are skipped but we still show the count for clarity
            total_builds = total_tests

            # Collect failed tests for summary
            failed_tests_list = [key for key, success in results.items() if not success]

            # Determine overall status
            overall_status = "SUCCESS" if failed_tests == 0 else "FAILURE"
            status_color = "#28a745" if failed_tests == 0 else "#dc3545"

            # Prepare framework data for template
            frameworks = []
            tested_frameworks = set()
            for key in results.keys():
                framework = key.split('_')[0]  # Extract framework from key like "VLLM_dev"
                tested_frameworks.add(framework)

            for framework in sorted(tested_frameworks):
                # Get all targets tested for this framework
                framework_targets = set()
                for key in results.keys():
                    if key.startswith(f"{framework}_"):
                        target = key[len(f"{framework}_"):]
                        framework_targets.add(target)

                targets = []
                for target in sorted(framework_targets):
                    framework_target = f"{framework}_{target}"
                    if framework_target in results:  # Only show targets that were actually tested
                        success = results[framework_target]

                        # Get timing information if available
                        timing_info = ""
                        build_time_str = ""
                        test_time_str = ""

                        if build_times:
                            build_time_key = f"{framework_target}_build"
                            test_time_key = f"{framework_target}_test"

                            # Get build time (may not exist in test-only mode)
                            if build_time_key in build_times:
                                build_time = build_times[build_time_key]
                                build_time_str = f"{build_time:.1f}s"
                            else:
                                build_time_str = "-"  # No build in test-only mode

                            # Get test time (should always exist)
                            if test_time_key in build_times:
                                test_time = build_times[test_time_key]
                                test_time_str = f"{test_time:.1f}s"
                            else:
                                test_time_str = "-"

                            # Create timing info for backward compatibility (used in some places)
                            if build_time_str != "-" and test_time_str != "-":
                                timing_info = f" (build: {build_time_str}, sanity_check.py: {test_time_str})"
                            elif test_time_str != "-":
                                timing_info = f" (sanity_check.py: {test_time_str})"

                        # Get error output if available
                        error_output = ""
                        if not success and framework_target in failure_details and failure_details[framework_target]:
                            error_lines = failure_details[framework_target].split('\n')
                            if len(error_lines) > 25:
                                error_output = '\n'.join(error_lines[-25:])  # Show last 25 lines
                                error_output = "... (showing last 25 lines)\n" + error_output
                            else:
                                error_output = failure_details[framework_target]

                        # Get Docker image information if build was successful
                        container_size = ""
                        image_tag = ""
                        image_id = ""
                        if success and not dry_run:
                            # Try to get the built image tag for this framework/target combination
                            try:
                                # Convert target type for build command discovery
                                docker_target_type = target if target != "dev" else None
                                _, _, discovered_tag = buildsh_utils.get_build_commands(framework, docker_target_type)
                                if discovered_tag:
                                    docker_info = docker_utils.get_image_info(discovered_tag)
                                    container_size = docker_info['size']
                                    image_tag = docker_info['repo_tag']
                                    
                                    # Get image ID using Docker API
                                    image_id = docker_utils.get_image_id(discovered_tag)
                            except Exception as e:
                                logger.debug(f"Failed to get Docker info for {framework_target}: {e}")
                                image_id = "Error"

                        targets.append({
                            'name': target,
                            'success': success,
                            'timing_info': timing_info,
                            'build_time': build_time_str,
                            'test_time': test_time_str,
                            'error_output': error_output,
                            'container_size': container_size,
                            'image_tag': image_tag,
                            'image_id': image_id
                        })

                frameworks.append({
                    'name': framework,
                    'targets': targets
                })

            # Prepare base builds data for template
            base_builds = []
            for base_key, base_time in base_build_times.items():
                # Extract framework name from base_key (e.g., "dynamo-base-TRTLLM" -> "TRTLLM")
                framework_name = base_key.replace("dynamo-base-", "").upper()
                
                # Get the actual dynamo-base tag dynamically from build commands for this specific framework
                image_tag = buildsh_utils.extract_base_image_tag(framework_name)
                if not image_tag:
                    # Fallback to constructed tag if extraction fails
                    image_tag = f"dynamo-base:v0.1.0.dev.{git_info['sha']}"
                
                # Get stored image ID if available
                image_id = base_image_ids.get(base_key, "-")
                container_size = "Not Found"  # Will be populated if we can get docker image info
                
                # Try to get actual Docker image information in non-dry-run mode
                # Use the framework-specific image tag to get the correct container size
                if not dry_run and image_tag:
                    try:
                        docker_info = docker_utils.get_image_info(image_tag)
                        container_size = docker_info['size']
                        # Update image_id if we get it from Docker API and don't have it stored
                        if image_id == "-" and docker_info['image_id'] != 'Unknown':
                            image_id = docker_info['image_id']
                    except Exception as e:
                        logger.debug(f"Failed to get Docker info for {framework_name} base image {image_tag}: {e}")
                        container_size = "Not Found"
                
                base_builds.append({
                    'name': framework_name,
                    'build_time': f"{base_time:.1f}s",
                    'container_size': container_size,
                    'image_tag': image_tag,
                    'image_id': image_id
                })

            # Prepare template context
            context = {
                'overall_status': overall_status,
                'status_color': status_color,
                'total_builds': total_builds,
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'build_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' PDT',
                'workspace_compilation': {
                    'attempted': workspace_compile_success is not None,
                    'success': workspace_compile_success,
                    'time': f"{workspace_compile_time:.1f}s" if workspace_compile_time > 0 else "0.0s",
                    'output_snippets': workspace_compile_output
                } if workspace_compile_success is not None else None,
                'git_info': {
                    'sha': git_info['sha'],
                    'date': git_info['pacific_timestamp'],
                    'message': self.convert_pr_links(self.html_escape(git_info['commit_message'])),
                    'full_message': git_info['full_commit_message'],
                    'author': self.format_author_html(git_info['author']),
                    'total_additions': git_info.get('total_additions', 0),
                    'total_deletions': git_info.get('total_deletions', 0),
                    'diff_stats': git_info.get('diff_stats', [])
                },
                'base_builds': base_builds,
                'frameworks': frameworks,
                'repo_path': str(dynamo_ci_dir),
                'log_dir': str(log_dir)
            }

            # Render template
            template = Template(self.get_email_template())
            html_content = template.render(context)

            # Create subject line
            status_prefix = "SUCC" if overall_status == "SUCCESS" else "FAIL"
            if failed_tests_list:
                failure_summary = ", ".join(failed_tests_list)
                subject = f"{status_prefix}: DynamoDockerBuilder - {git_info['sha']} ({failure_summary})"
            else:
                subject = f"{status_prefix}: DynamoDockerBuilder - {git_info['sha']}"

            # Create email file with proper CRLF formatting
            email_file = Path(f"/tmp/dynamo_email_{os.getpid()}.txt")

            # Write email content directly to avoid printf format specifier issues
            email_content = f'Subject: {subject}\r\nFrom: DynamoDockerBuilder <dynamo-docker-builder@nvidia.com>\r\nTo: {email}\r\nMIME-Version: 1.0\r\nContent-Type: text/html; charset=UTF-8\r\n\r\n{html_content}\r\n'

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
                logger.info(f"SUCCESS: Email notification sent to {email} (using Jinja2 template)")
            else:
                logger.error(f"Failed to send email: {result.stderr}")

        except Exception as e:
            logger.error(f"Error sending email notification: {e}")


class DynamoDockerBuilder(BaseUtils):
    """DynamoDockerBuilder - Main class for automated Docker build testing and reporting"""

    # Framework constants
    FRAMEWORKS: List[str] = ["TRTLLM", "VLLM", "SGLANG"]

    def __init__(self) -> None:
        # Configuration flags - set before calling super().__init__
        self.dry_run = False
        
        # Set up logger first
        self._setup_logger()
        
        # Call parent constructor with logger
        super().__init__(dry_run=self.dry_run, logger=self.logger)
        
        self.script_dir = Path(__file__).parent.absolute()
        self.dynamo_ci_dir = self.script_dir.parent / "dynamo_ci"
        self.date = datetime.now().strftime("%Y-%m-%d")
        self.log_dir = self.dynamo_ci_dir / "logs" / self.date

        # Additional configuration flags
        self.test_only = False
        self.no_checkout = False
        self.force_run = False
        self.email = None
        self.targets = ["dev", "local-dev"]  # Default targets
        self.repo_sha = None  # SHA to checkout

        # Lock file for preventing concurrent runs
        self.lock_file = self.script_dir / f".{Path(__file__).name}.lock"

        # Initialize utility classes
        self.git_utils = GitUtils(dry_run=self.dry_run, logger=self.logger)
        self.docker_utils = DockerUtils(dry_run=self.dry_run, logger=self.logger)
        self.buildsh_utils = BuildShUtils(self.dynamo_ci_dir, self.docker_utils, dry_run=self.dry_run, logger=self.logger)
        self.email_renderer = EmailTemplateRenderer()

        # Track build times for email reporting
        self.build_times: Dict[str, float] = {}

        # Track base image build times separately
        self.base_build_times: Dict[str, float] = {}
        
        # Track base image IDs separately  
        # Example: {"dynamo-base-VLLM": "ab155e5b17bd", "dynamo-base-TRTLLM": "5a4b7288c7f6"}
        self.base_image_ids: Dict[str, str] = {}

        # Track executed docker commands to prevent duplicates
        self.executed_commands: set = set()

        # Track workspace compilation
        self.workspace_compiled = False
        self.workspace_compile_time = 0.0
        self.workspace_compile_success = None  # None = not attempted, True = success, False = failed
        self.workspace_compile_output = {}  # Dict of output snippets from compilation steps

    def _setup_logger(self) -> None:
        """Set up the logger with appropriate formatting"""
        self.logger = logging.getLogger('DynamoDockerBuilder')
        self.logger.setLevel(logging.DEBUG)

        # Remove any existing handlers
        self.logger.handlers.clear()

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Create custom formatter that handles DRYRUN prefix
        class DryRunFormatter(logging.Formatter):
            def __init__(self, dry_run_instance: 'DynamoDockerBuilder') -> None:
                super().__init__()
                self.dry_run_instance = dry_run_instance

            def format(self, record: logging.LogRecord) -> str:
                if self.dry_run_instance.dry_run:
                    return f"DRYRUN {record.levelname} - {record.getMessage()}"
                else:
                    return f"{record.levelname} - {record.getMessage()}"

        formatter = DryRunFormatter(self)
        console_handler.setFormatter(formatter)

        # Add handler to logger
        self.logger.addHandler(console_handler)

        # Prevent propagation to root logger
        self.logger.propagate = False

    def rename_target_for_build(self, target: str) -> Optional[str]:
        """Convert target name for build commands. 'dev' becomes None, others stay as-is"""
        return None if target == "dev" else target

    def get_failure_details(self, framework: str, docker_target_type: str) -> str:
        """Get failure details from log files for a failed test"""
        try:
            commit_sha = self.git_utils.get_current_sha(self.dynamo_ci_dir)
            framework_lower = framework.lower()

            # Determine log file suffix
            if docker_target_type == "local-dev":
                log_suffix = f"{commit_sha}.{framework_lower}.local-dev"
            else:
                log_suffix = f"{commit_sha}.{framework_lower}.dev"

            log_file = self.log_dir / f"{self.date}.{log_suffix}.log"

            if log_file.exists():
                # Read last 20 lines of the log file
                result = self.cmd(['tail', '-20', str(log_file)],
                                      capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()

            return f"No log file found at {log_file}"
        except Exception as e:
            return f"Error reading log file: {e}"

    def cmd(self, command: List[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Execute command with dry-run support"""
        # Show command in shell tracing format
        cmd_str = " ".join(shlex.quote(str(arg)) for arg in command)
        self.logger.debug(f"+ {cmd_str}")

        # Commands that are safe to execute in dry-run mode (no side effects)
        is_safe_command = False
        
        if len(command) >= 1:
            cmd_base = command[0]
            
            # build.sh and run.sh with --dry-run flag (no side effects)
            if cmd_base in ['./container/build.sh', './container/run.sh']:
                is_safe_command = '--dry-run' in command
            
            # Other read-only commands
            elif cmd_base in ['ls', 'cat', 'head', 'tail', 'grep', 'find', 'which', 'echo']:
                is_safe_command = True
        
        if self.dry_run and not is_safe_command:
            # Return a mock completed process for commands with side effects in dry-run
            mock_result = subprocess.CompletedProcess(command, 0)
            mock_result.stdout = ""
            mock_result.stderr = ""
            return mock_result
        else:
            # Execute safe commands and all commands in non-dry-run mode
            return subprocess.run(command, **kwargs)

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

    def setup_logging_dir(self) -> None:
        """Create date-based log directory"""
        self.logger.info("Setting up date-based logging directory...")
        if self.dry_run:
            self.logger.debug(f"+ mkdir -p {self.log_dir}")
        else:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"SUCCESS: Date-based log directory created at {self.log_dir}")

    def cleanup_existing_logs(self, framework: Optional[str] = None) -> None:
        """Clean up existing log files for current date and current SHA only (preserve other SHAs)"""
        # Get current commit SHA for precise cleanup
        current_sha = self.git_utils.get_current_sha(self.dynamo_ci_dir)
        
        if framework:
            self.logger.info(f"Cleaning up existing log files for date: {self.date}, SHA: {current_sha}, framework: {framework}")
            framework_lower = framework.lower()
        else:
            self.logger.info(f"Cleaning up existing log files for date: {self.date}, SHA: {current_sha}")

        if self.dry_run:
            # In dry-run mode, just show what would be removed
            if framework:
                patterns = [f"{self.date}.{current_sha}.{framework_lower}.*.log",
                           f"{self.date}.{current_sha}.{framework_lower}.*.SUCC",
                           f"{self.date}.{current_sha}.{framework_lower}.*.FAIL"]
            else:
                patterns = [f"{self.date}.{current_sha}.*.log", 
                           f"{self.date}.{current_sha}.*.SUCC", 
                           f"{self.date}.{current_sha}.*.FAIL"]

            for pattern in patterns:
                self.logger.debug(f"+ rm -f {self.log_dir}/{pattern}")

            if framework:
                self.logger.info(f"Would remove existing log files for {self.date}.{current_sha} and {framework} (dry-run)")
            else:
                self.logger.info(f"Would remove existing log files for {self.date}.{current_sha} (dry-run)")
        else:
            # Remove existing log files only for current SHA (preserve other SHAs)
            if framework:
                # Framework-specific patterns for current SHA only
                patterns = [f"{self.date}.{current_sha}.{framework_lower}.*.log",
                           f"{self.date}.{current_sha}.{framework_lower}.*.SUCC",
                           f"{self.date}.{current_sha}.{framework_lower}.*.FAIL"]
            else:
                # All files for the current SHA only
                patterns = [f"{self.date}.{current_sha}.*.log", 
                           f"{self.date}.{current_sha}.*.SUCC", 
                           f"{self.date}.{current_sha}.*.FAIL"]

            removed_files = []

            for pattern in patterns:
                for file_path in self.log_dir.glob(pattern):
                    if file_path.is_file():
                        file_path.unlink()
                        removed_files.append(file_path.name)
                        self.logger.info(f"Removed existing file: {file_path.name}")

            if removed_files:
                if framework:
                    self.logger.info(f"SUCCESS: Removed {len(removed_files)} existing log files for {framework} (SHA: {current_sha})")
                else:
                    self.logger.info(f"SUCCESS: Removed {len(removed_files)} existing log files (SHA: {current_sha})")
            else:
                if framework:
                    self.logger.info(f"No existing log files found for {self.date}.{current_sha} and {framework}")
                else:
                    self.logger.info(f"No existing log files found for {self.date}.{current_sha}")
                    
            # Show preserved files from other SHAs
            if framework:
                other_sha_pattern = f"{self.date}.*.{framework_lower}.*.log"
            else:
                other_sha_pattern = f"{self.date}.*.log"
                
            other_files = [f for f in self.log_dir.glob(other_sha_pattern) 
                          if not f.name.startswith(f"{self.date}.{current_sha}.")]
            
            if other_files:
                self.logger.info(f"Preserved {len(other_files)} log files from other SHAs on {self.date}")
                # Show a few examples
                for f in other_files[:3]:
                    self.logger.debug(f"  Preserved: {f.name}")
                if len(other_files) > 3:
                    self.logger.debug(f"  ... and {len(other_files) - 3} more")

    def check_if_rebuild_needed(self) -> bool:
        """Check if rebuild is needed based on composite SHA"""
        self.logger.info("Checking if rebuild is needed based on file changes...")
        self.logger.info(f"Composite SHA file location: {self.dynamo_ci_dir}/.last_build_composite_sha")

        # Generate current composite SHA
        current_sha = self.git_utils.generate_composite_sha_from_container_dir(self.dynamo_ci_dir)
        if current_sha is None:
            self.logger.error("Failed to generate current composite SHA")
            return False

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
        """Mark a docker command as executed to prevent duplicates
        
        Args:
            docker_command: Complete docker build command as a string
        """
        normalized_cmd = self.docker_utils.normalize_command(docker_command)
        self.executed_commands.add(normalized_cmd)
        self.logger.info(f"Marked command as executed: {normalized_cmd[:100]}...")


    def execute_build_commands(self, framework: str, docker_commands: List[str], log_file: Path, fail_file: Path) -> Tuple[bool, Dict[str, float]]:
        """Execute docker build commands with real-time output, skipping already executed commands
        
        Returns:
            (success: bool, base_build_times: Dict[str, float])
            base_build_times contains timing for any dynamo-base builds that were executed
        """
        base_build_times: Dict[str, float] = {}
        
        try:
            # Filter out already executed commands
            new_commands = []
            skipped_commands = []
            
            for docker_cmd in docker_commands:
                if self.is_command_executed(docker_cmd):
                    skipped_commands.append(docker_cmd)
                    self.logger.info(f"Skipping already executed command: {docker_cmd[:80]}...")
                else:
                    new_commands.append(docker_cmd)
            
            with open(log_file, 'a') as f:
                f.write(f"Extracted {len(docker_commands)} docker build commands\n")
                if skipped_commands:
                    f.write(f"Skipped {len(skipped_commands)} already executed commands\n")
                if new_commands:
                    f.write(f"Executing {len(new_commands)} new commands\n")

                for i, docker_cmd in enumerate(new_commands):
                    # Check if this is a base image build using regex to match --tag dynamo-base:
                    is_base_build = bool(re.search(r'--tag\s+dynamo-base:', docker_cmd))
                    
                    self.logger.info(f"Executing docker build command {i+1}/{len(new_commands)} (of {len(docker_commands)} total)")
                    if is_base_build:
                        self.logger.info("  → Building dynamo-base image")
                    
                    self.logger.debug(f"+ {docker_cmd}")
                    f.write(f"+ {docker_cmd}\n")
                    f.flush()

                    # Track timing for this specific command
                    cmd_start_time = time.time()

                    # Run docker command with real-time output
                    with subprocess.Popen(
                        docker_cmd,
                        shell=True, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.STDOUT,
                        text=True, 
                        bufsize=1
                    ) as process:
                        # Stream output line by line
                        for line in iter(process.stdout.readline, ''):
                            line = line.rstrip()
                            if line:
                                print(line)  # Show on console
                                f.write(line + '\n')  # Write to log file
                                f.flush()  # Ensure immediate write
                        
                        return_code = process.wait()

                    cmd_end_time = time.time()
                    cmd_duration = cmd_end_time - cmd_start_time

                    if return_code != 0:
                        self.logger.error(f"Docker build command {i+1} failed for {framework}")
                        f.write(f"Build Status: FAILED (docker command {i+1})\n")
                        with open(fail_file, 'a') as fail_f:
                            fail_f.write(f"{framework} docker build failed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        return False, base_build_times
                    else:
                        # Mark command as executed on success
                        self.mark_command_executed(docker_cmd)
                        
                        # Track base image build time separately
                        if is_base_build:
                            # Use framework key for reliable base image identification
                            base_key = f"dynamo-base-{framework}"
                            base_build_times[base_key] = cmd_duration
                            
                            # Capture the IMAGE ID from the build output
                            # Look for the "writing image sha256:..." line in the output
                            try:
                                # Read the current log content to find the image ID
                                with open(log_file, 'r') as log_f:
                                    log_content = log_f.read()
                                    
                                # Look for the Docker image ID in the build output
                                image_id_match = re.search(r'writing image (sha256:[a-f0-9]+)', log_content)
                                if image_id_match:
                                    full_image_id = image_id_match.group(1)
                                    # Store the short image ID (first 12 chars after sha256:)
                                    short_image_id = full_image_id[7:19] if full_image_id.startswith('sha256:') else full_image_id[:12]
                                    self.base_image_ids[base_key] = short_image_id
                                    self.logger.debug(f"Captured base image ID for {framework}: {short_image_id}")
                            except Exception as e:
                                self.logger.debug(f"Failed to capture base image ID for {framework}: {e}")
                            
                            self.logger.info(f"  → Base image build for {framework} completed in {cmd_duration:.1f}s")

                if new_commands:
                    self.logger.info(f"SUCCESS: Build completed for {framework} ({len(new_commands)} new commands executed, {len(skipped_commands)} skipped)")
                else:
                    self.logger.info(f"SUCCESS: Build completed for {framework} (all {len(skipped_commands)} commands were already executed)")
                f.write("Build Status: SUCCESS (no latest tags)\n")
                return True, base_build_times

        except Exception as e:
            self.logger.error(f"Build failed for {framework}: {e}")
            with open(log_file, 'a') as log_f:
                log_f.write(f"Build Status: FAILED ({e})\n")
            with open(fail_file, 'a') as fail_f:
                fail_f.write(f"{framework} build failed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            return False, {}

    def build_framework_image(self, framework: str, docker_target_type: str, log_file: Path, fail_file: Path) -> Tuple[bool, Optional[float]]:
        """Build a specific framework image
        
        Args:
            framework: Framework name - one of "VLLM", "SGLANG", or "TRTLLM"
            docker_target_type: Docker build target type - either "dev" or "local-dev"
            log_file: Path to log file for build output
            fail_file: Path to failure file for error tracking
            
        Returns:
            (success: bool, build_time: Optional[float])
            
            Examples:
                Success case: (True, 45.2)
                Failure case: (False, None)
        """
        print("=" * 60)
        self.logger.info(f"Building {framework} framework ({docker_target_type} target)...")
        print("=" * 60)

        build_start_time = time.time()

        # Get build commands
        target_param = self.rename_target_for_build(docker_target_type)
        success, docker_commands, _image_tag = self.buildsh_utils.get_build_commands(framework, target_param)
        if not success:
            if not self.dry_run:
                with open(log_file, 'a') as log_f:
                    log_f.write("Build Status: FAILED (could not get docker commands)\n")
                with open(fail_file, 'a') as fail_f:
                    fail_f.write(f"{framework} build command extraction failed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            return False, None

        if self.dry_run:
            # Show filtered commands that would be executed, with duplicate detection
            new_commands = []
            skipped_commands = []
            
            for cmd in docker_commands:
                if self.is_command_executed(cmd):
                    skipped_commands.append(cmd)
                    self.logger.info(f"Skipping already executed command: {cmd[:80]}...")
                else:
                    new_commands.append(cmd)
                    # Mark as executed in dry-run mode too for consistency
                    self.mark_command_executed(cmd)
            
            self.logger.info("Filtered docker commands (removing latest tags and duplicates):")
            for cmd in new_commands:
                print(f"+ {cmd}")
            
            if skipped_commands:
                self.logger.info(f"Skipped {len(skipped_commands)} already executed commands in dry-run")
                
            self.logger.info(f"SUCCESS: Build commands prepared for {framework} (dry-run: {len(new_commands)} new, {len(skipped_commands)} skipped)")
            
            # Calculate build time for dry-run (time spent preparing commands)
            build_end_time = time.time()
            build_time = build_end_time - build_start_time
            return True, build_time  # Return early in dry-run mode
        else:
            # Execute the commands
            success, base_times = self.execute_build_commands(framework, docker_commands, log_file, fail_file)
            if success:
                # Store base build times in the instance variable for email reporting
                for base_key, base_time in base_times.items():
                    self.base_build_times[base_key] = base_time
            else:
                return False, None

        build_end_time = time.time()
        build_time = build_end_time - build_start_time
        
        return True, build_time

    def run_workspace_compilation(self, framework: str, docker_target_type: str) -> Tuple[bool, float, Dict[str, List[str]]]:
        """Run one-time workspace compilation commands for local-dev containers
        
        This runs cargo build, maturin develop, and uv pip install to compile the workspace.
        Only runs once for the very first local-dev container test.
        
        Args:
            framework: Framework name for logging context
            docker_target_type: Docker target type (should be "local-dev")
            
        Returns:
            Tuple of (success: bool, compilation_time: float, output_snippets: Dict[str, List[str]])
        """
        self.logger.info("Running workspace compilation for local-dev container...")
        
        # Determine image name for the container
        target_param = self.rename_target_for_build(docker_target_type)
        success, _, image_tag = self.buildsh_utils.get_build_commands(framework, target_param)
        if not success or not image_tag:
            self.logger.error("Failed to determine image tag for workspace compilation")
            return False, 0.0, {}
        
        # Commands to run for workspace compilation
        compile_commands = [
            "cargo build --locked --profile dev --features dynamo-llm/block-manager",
            "(cd /workspace/lib/bindings/python && maturin develop)",
            "uv pip install -e ."
        ]
        
        # Track output snippets for each command
        output_snippets = {
            'cargo_build': [],
            'maturin_develop': [],
            'uv_install': []
        }
        snippet_keys = ['cargo_build', 'maturin_develop', 'uv_install']
        
        compile_start_time = time.time()
        
        if self.dry_run:
            self.logger.info("DRY-RUN: Workspace compilation commands:")
            for cmd in compile_commands:
                self.logger.debug(f"+ {cmd}")
            compile_end_time = time.time()
            compile_time = compile_end_time - compile_start_time
            self.logger.info(f"DRY-RUN: Workspace compilation completed in {compile_time:.1f}s")
            return True, compile_time, output_snippets
        
        try:
            # Create the full docker run command similar to sanity check
            docker_cmd_result = self.cmd(
                ["./container/run.sh", "--dry-run", "--image", image_tag, "--mount-workspace"],
                capture_output=True, text=True, cwd=self.dynamo_ci_dir
            )
            
            if docker_cmd_result.returncode == 0:
                # Extract base docker command and remove -it flags
                docker_lines = [line.strip() for line in docker_cmd_result.stdout.split('\n') if line.strip().startswith('docker run')]
                
                if docker_lines:
                    base_docker_cmd = docker_lines[0]
                    # Remove interactive flags
                    base_docker_cmd = re.sub(r'-it\b', '', base_docker_cmd)
                    base_docker_cmd = re.sub(r'\s+', ' ', base_docker_cmd).strip()
                    
                    # Run each compilation command in sequence
                    for i, compile_cmd in enumerate(compile_commands, 1):
                        self.logger.info(f"Running compilation step {i}/{len(compile_commands)}: {compile_cmd}")
                        
                        # Create full command: docker run ... bash -c "compile_cmd"
                        full_cmd = f'{base_docker_cmd} bash -c "{compile_cmd}"'
                        
                        # Execute the command with real-time output
                        step_start_time = time.time()
                        try:
                            with subprocess.Popen(
                                full_cmd,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                bufsize=1
                            ) as process:
                                output_lines = []
                                timeout_seconds = 1800  # 30 minutes per step
                                
                                # Stream output with timeout
                                for line in iter(process.stdout.readline, ''):
                                    if time.time() - step_start_time > timeout_seconds:
                                        self.logger.error(f"Compilation step {i} timed out after {timeout_seconds/60:.1f} minutes")
                                        process.terminate()
                                        process.wait(timeout=5)
                                        return False, 0.0, output_snippets
                                    
                                    line = line.rstrip()
                                    if line:
                                        self.logger.info(f"  {line}")
                                        output_lines.append(line)
                                
                                return_code = process.wait()
                            
                            # Store last 7 lines of output for this step
                            snippet_key = snippet_keys[i-1]
                            output_snippets[snippet_key] = output_lines[-7:] if len(output_lines) >= 7 else output_lines
                            
                            if return_code != 0:
                                self.logger.error(f"Workspace compilation step {i} failed (exit code: {return_code})")
                                self.logger.error(f"Command: {compile_cmd}")
                                if output_lines:
                                    self.logger.error("Last few lines of output:")
                                    for line in output_lines[-10:]:
                                        self.logger.error(line)
                                return False, 0.0, output_snippets
                            
                            step_duration = time.time() - step_start_time
                            self.logger.info(f"Compilation step {i} completed successfully in {step_duration:.1f}s")
                                
                        except subprocess.TimeoutExpired:
                            self.logger.error(f"Compilation step {i} timed out")
                            return False, 0.0, output_snippets
                        except Exception as e:
                            self.logger.error(f"Exception during compilation step {i}: {e}")
                            return False, 0.0, output_snippets
                
                compile_end_time = time.time()
                compile_time = compile_end_time - compile_start_time
                self.logger.info(f"SUCCESS: Workspace compilation completed in {compile_time:.1f}s")
                return True, compile_time, output_snippets
                
            else:
                self.logger.error("Failed to get docker run command for workspace compilation")
                return False, 0.0, output_snippets
                
        except subprocess.TimeoutExpired:
            self.logger.error("Workspace compilation timed out (30 minutes)")
            return False, 0.0, output_snippets
        except Exception as e:
            self.logger.error(f"Error during workspace compilation: {e}")
            return False, 0.0, output_snippets

    def run_sanity_check_on_image(self, framework: str, docker_target_type: str) -> bool:
        """Test a specific framework+target image (optionally building it first)

        This function performs a multi-step process:
        1. Build the framework image (unless --test-only mode is enabled)
        1.5. Run workspace compilation (one-time setup) only for the first local-dev container
        2. Test the image by running sanity_check.py in a container

        Args:
            framework: Framework name - one of "VLLM", "SGLANG", or "TRTLLM"
                      - "VLLM": vLLM framework for LLM inference
                      - "SGLANG": SGLang framework for structured generation
                      - "TRTLLM": TensorRT-LLM framework for optimized inference
            docker_target_type: Docker build target type - either "dev" or "local-dev"
                       - "dev": Regular framework image (e.g., dynamo:v0.1.0.dev.abc123-vllm)
                       - "local-dev": Local development image with user permissions
                                     (e.g., dynamo:v0.1.0.dev.abc123-vllm-local-dev)
                                     
        Returns:
            bool: True if both build (if performed) and test succeed, False otherwise
        """
        framework_lower = framework.lower()

        # Get git commit SHA for log file naming
        commit_sha = self.git_utils.get_current_sha(self.dynamo_ci_dir)

        # Get image tag using our build commands method (reuse the tag discovery logic)
        target_param = self.rename_target_for_build(docker_target_type)
        success, _, image_name = self.buildsh_utils.get_build_commands(framework, target_param)
        if not success:
            self.logger.error(f"Failed to discover image tag for {framework} {docker_target_type}")
            return False

        # Log file suffix
        if docker_target_type == "local-dev":
            log_suffix = f"{commit_sha}.{framework_lower}.local-dev"
        else:
            log_suffix = f"{commit_sha}.{framework_lower}.dev"

        log_file = self.log_dir / f"{self.date}.{log_suffix}.log"
        success_file = self.log_dir / f"{self.date}.{log_suffix}.SUCC"
        fail_file = self.log_dir / f"{self.date}.{log_suffix}.FAIL"

        self.logger.info(f"Testing framework: {framework} ({docker_target_type} image)")

        # Track timing for this framework/target combination
        test_key = f"{framework}_{docker_target_type}"
        start_time = time.time()

        # Change to dynamo_ci directory
        os.chdir(self.dynamo_ci_dir)

        # Log start of framework test (only in non-dry-run mode)
        if self.dry_run:
            self.logger.info(f"Would write framework test start to: {log_file}")
        else:
            with open(log_file, 'a') as f:
                f.write("=" * 42 + "\n")
                f.write(f"Framework: {framework} ({docker_target_type} image)\n")
                f.write(f"Image: {image_name}\n")
                f.write(f"Date: {datetime.now().strftime('%c')}\n")
                f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if self.test_only:
                    f.write("Mode: TEST-ONLY (skipping build)\n")
                f.write("=" * 42 + "\n")

        # Step 1: Build the framework (skip if TEST_ONLY is true)
        build_time = None
        if self.test_only:
            self.logger.info(f"Skipping build step for {framework} (test-only mode)")
            if not self.dry_run:
                with open(log_file, 'a') as f:
                    f.write("Build Status: SKIPPED (test-only mode)\n")
        else:
            self.logger.info(f"Step 1: Building {framework} framework ({docker_target_type} target)...")
            build_success, build_time = self.build_framework_image(framework, docker_target_type, log_file, fail_file)
            if not build_success:
                    return False

        # Step 1.5: Run workspace compilation (one-time setup) only for the first local-dev container
        # This happens AFTER the image is built so the local-dev image exists
        if docker_target_type == "local-dev" and not self.workspace_compiled:
            self.logger.info("Step 1.5: Running workspace compilation (one-time setup for local-dev)...")
            compile_success, compile_time, output_snippets = self.run_workspace_compilation(framework, docker_target_type)
            
            self.workspace_compiled = True
            self.workspace_compile_time = compile_time
            self.workspace_compile_success = compile_success
            self.workspace_compile_output = output_snippets
            self.build_times['workspace_compilation'] = compile_time
            
            if not compile_success:
                self.logger.error("Workspace compilation failed - aborting test")
                return False
            else:
                self.logger.info(f"SUCCESS: Workspace compilation completed in {compile_time:.1f}s")
        elif docker_target_type == "local-dev" and self.workspace_compiled:
            self.logger.info("Workspace compilation already completed - skipping")
        else:
            self.logger.debug(f"Skipping workspace compilation for {docker_target_type} target")

        # Step 2: Run the container with sanity_check.py
        print("=" * 60)
        self.logger.info(f"Step 2: Running sanity_check.py test for {framework} ({docker_target_type} target)...")
        print("=" * 60)

        # Get the docker command from container/run.sh dry-run, then execute without -it flags
        try:
            docker_cmd_result = self.cmd(
                ["./container/run.sh", "--dry-run", "--image", image_name, "--mount-workspace", "--entrypoint", "deploy/sanity_check.py"],
                capture_output=True, text=True, cwd=self.dynamo_ci_dir
            )

            if docker_cmd_result.returncode == 0:
                # Extract docker command and remove -it flags
                docker_cmd = None
                # Check both stdout and stderr for the docker command
                output_text = docker_cmd_result.stdout + docker_cmd_result.stderr
                for line in output_text.split('\n'):
                    if line.startswith("docker run"):
                        docker_cmd = line.replace(" -it ", " ")
                        break

                if docker_cmd and not self.dry_run:
                    prefix = "DRYRUN +" if self.dry_run else "+"
                    self.logger.debug(f"+ timeout 30 {docker_cmd}")
                    try:
                        with open(log_file, 'a') as f:
                            # Run container test with real-time output
                            with subprocess.Popen(
                                f"timeout 30 {docker_cmd}",
                                shell=True, 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.STDOUT,
                                text=True, 
                                bufsize=1
                            ) as process:
                                # Stream output line by line
                                for line in iter(process.stdout.readline, ''):
                                    line = line.rstrip()
                                    if line:
                                        print(line)  # Show on console
                                        f.write(line + '\n')  # Write to log file
                                        f.flush()  # Ensure immediate write
                                
                                return_code = process.wait()

                        if return_code == 0:
                            self.logger.info(f"SUCCESS: Container test completed for {framework}")
                            with open(log_file, 'a') as log_f:
                                log_f.write("Container Test Status: SUCCESS\n")
                            with open(success_file, 'a') as success_f:
                                success_f.write(f"{framework} test completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        else:
                            self.logger.error(f"Container test failed for {framework}")
                            with open(log_file, 'a') as log_f:
                                log_f.write("Container Test Status: FAILED\n")
                            with open(fail_file, 'a') as fail_f:
                                fail_f.write(f"{framework} container test failed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            return False

                    except subprocess.TimeoutExpired:
                        self.logger.error(f"Container test timeout for {framework}")
                        with open(log_file, 'a') as log_f:
                            log_f.write("Container Test Status: TIMEOUT\n")
                        with open(fail_file, 'a') as fail_f:
                            fail_f.write(f"{framework} container test timeout at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        return False

                elif docker_cmd:
                    prefix = "DRYRUN +" if self.dry_run else "+"
                    self.logger.debug(f"+ timeout 30 {docker_cmd}")
                    self.logger.info(f"SUCCESS: Container test completed for {framework} (dry-run)")
                    self.logger.info(f"Would write success to: {success_file}")
                else:
                    self.logger.error(f"Could not extract docker command for {framework}")
                    return False
            else:
                self.logger.error(f"Failed to get docker command for {framework}")
                return False

        except Exception as e:
            self.logger.error(f"Exception during container test for {framework}: {e}")
            return False

        # Log end of framework test (only in non-dry-run mode)
        if self.dry_run:
            self.logger.info(f"Would write framework test end to: {log_file}")
        else:
            with open(log_file, 'a') as f:
                f.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("Overall Status: SUCCESS\n\n")

        # Store timing information for email reporting
        end_time = time.time()
        total_time = end_time - start_time

        # Calculate build time if build was performed
        if build_time is not None:
            test_time = total_time - build_time
            self.build_times[f"{test_key}_build"] = build_time
            self.build_times[f"{test_key}_test"] = test_time
            self.logger.debug(f"Build time for {test_key}: {build_time:.1f}s")
            self.logger.debug(f"Test time for {test_key}: {test_time:.1f}s")
        else:
            # Test-only mode: no build time, all time is test time
            self.build_times[f"{test_key}_test"] = total_time
            self.logger.debug(f"Test-only time for {test_key}: {total_time:.1f}s")

        # Store total time (build + test)
        self.build_times[f"{test_key}_total"] = total_time
        self.logger.debug(f"Total time for {test_key}: {total_time:.1f}s")

        return True

    def main(self) -> int:
        """Main function"""
        parser = argparse.ArgumentParser(description="DynamoDockerBuilder - Automated Docker Build and Test System")
        parser.add_argument("-f", "--framework", "--frameworks", type=str, action='append', dest='framework',
                          help="Test specific framework (VLLM, SGLANG, TRTLLM) - case insensitive. Can be specified multiple times.")
        parser.add_argument("--target", type=str, default="dev,local-dev",
                          help="Comma-separated Docker build targets to test: dev, local-dev (default: dev,local-dev)")
        parser.add_argument("-a", "--all", action="store_true", default=True,
                          help="Test all frameworks (default)")
        parser.add_argument("--test-only", action="store_true",
                          help="Skip build step and only run container tests")
        parser.add_argument("--no-checkout", action="store_true",
                          help="Skip git operations and use existing repository")
        parser.add_argument("--force-run", action="store_true",
                          help="Force run even if files haven't changed or another process is running")
        parser.add_argument("--dry-run", "--dryrun", action="store_true",
                          help="Show commands that would be executed without running them")
        parser.add_argument("--repo-path", type=str, default=None,
                          help="Path to the dynamo repository (default: ../dynamo_ci)")
        parser.add_argument("--email", type=str, default=None,
                          help="Email address for notifications (sends email if specified)")
        parser.add_argument("--repo-sha", type=str, default=None,
                          help="Git SHA to checkout instead of latest main branch")

        args = parser.parse_args()

        # Update repo path if specified
        if args.repo_path:
            self.dynamo_ci_dir = Path(args.repo_path).absolute()
            self.log_dir = self.dynamo_ci_dir / "logs" / self.date

        # Set configuration flags
        self.dry_run = args.dry_run
        self.test_only = args.test_only
        self.no_checkout = args.no_checkout
        self.force_run = args.force_run
        self.email = args.email
        self.repo_sha = args.repo_sha

        # Parse targets
        self.targets = [target.strip() for target in args.target.split(',') if target.strip()]
        if not self.targets:
            self.targets = ["dev", "local-dev"]  # Fallback to default

        # Validate targets - only allow known Docker build targets
        valid_targets = ["dev", "local-dev", "runtime", "dynamo_base", "framework"]
        invalid_targets = [t for t in self.targets if t not in valid_targets]
        if invalid_targets:
            valid_targets_str = ", ".join(valid_targets)
            invalid_targets_str = ", ".join(invalid_targets)
            self.logger.error(f"Invalid target(s) '{invalid_targets_str}'. Valid Docker build targets are: {valid_targets_str}")
            self.logger.error("Note: Targets are Docker build targets (dev, local-dev, runtime, dynamo_base, framework), not framework names (VLLM, SGLANG, TRTLLM)")
            return 1

        # Determine which frameworks to test
        if args.framework:
            # Normalize framework names to uppercase for case-insensitive matching
            frameworks_to_test = []
            for framework in args.framework:
                framework_upper = framework.upper()
                if framework_upper not in self.FRAMEWORKS:
                    valid_frameworks = ", ".join(self.FRAMEWORKS)
                    self.logger.error(f"Invalid framework '{framework}'. Valid options are: {valid_frameworks} (case insensitive)")
                    return 1
                frameworks_to_test.append(framework_upper)
        else:
            # Test all frameworks by default
            frameworks_to_test = list(self.FRAMEWORKS)

        print("=" * 60)
        self.logger.info(f"Starting DynamoDockerBuilder - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # Check if another instance is already running
        self.check_if_running()

        if self.dry_run:
            self.logger.info("DRY-RUN MODE: Commands will be shown but not executed")
        if self.test_only:
            self.logger.info("TEST-ONLY MODE: Skipping build step and only running tests")
        if self.no_checkout:
            self.logger.info("NO-CHECKOUT MODE: Skipping git operations and using existing repository")
        if self.force_run:
            self.logger.info("FORCE-RUN MODE: Will run even if files haven't changed or another process is running")
        if self.repo_sha:
            self.logger.info(f"REPO-SHA MODE: Will checkout specific SHA: {self.repo_sha}")

        if not self.dry_run:
            self.logger.info(f"Date: {self.date}")
        self.logger.info(f"Dynamo CI Directory: {self.dynamo_ci_dir}")
        self.logger.info(f"Log Directory: {self.log_dir}")

        # Setup repository and logging
        self.git_utils.setup_dynamo_ci(self.dynamo_ci_dir, self.repo_sha, self.no_checkout)
        self.setup_logging_dir()

        # Check if rebuild is needed based on file changes
        rebuild_needed = self.check_if_rebuild_needed()

        if not rebuild_needed:
            self.logger.info("SUCCESS:No rebuild needed - all files unchanged")
            self.logger.info("Exiting early (use --force-run to force rebuild)")
            self.logger.info("Preserving existing log files")
            return 0

        # Run tests and collect detailed results for email notification
        test_results: Dict[str, bool] = {}
        failure_details: Dict[str, str] = {}
        overall_success = True

        # Log what we're testing
        if len(frameworks_to_test) == 1:
            self.logger.info(f"Testing single framework: {frameworks_to_test[0]}")
        elif len(frameworks_to_test) == len(self.FRAMEWORKS):
            self.logger.info("Testing all frameworks")
        else:
            frameworks_str = ", ".join(frameworks_to_test)
            self.logger.info(f"Testing multiple frameworks: {frameworks_str}")

        # Clean up logs (all frameworks if testing all, specific ones if testing subset)
        if len(frameworks_to_test) == len(self.FRAMEWORKS):
            self.cleanup_existing_logs()  # Clean all
        else:
            for framework in frameworks_to_test:
                self.cleanup_existing_logs(framework=framework)  # Clean specific ones

        # Test each framework and collect detailed results
        overall_success = True
        for framework in frameworks_to_test:
            targets_str = ", ".join(self.targets)
            self.logger.info(f"Starting tests for framework: {framework} (targets: {targets_str})")

            # Test all configured targets
            framework_success = True
            for target in self.targets:
                target_success = self.run_sanity_check_on_image(framework, target)
                test_results[f"{framework}_{target}"] = target_success

                # Collect failure details for failed tests
                if not target_success:
                    failure_details[f"{framework}_{target}"] = self.get_failure_details(framework, target)
                    framework_success = False

            if not framework_success:
                overall_success = False

            if framework_success:
                self.logger.info(f"SUCCESS: All tests completed successfully for framework: {framework}")
            else:
                # Identify which targets failed for this framework
                failed_targets = [target for target in self.targets 
                                if not test_results.get(f"{framework}_{target}", True)]
                failed_targets_str = ", ".join(failed_targets)
                self.logger.error(f"Some tests failed for framework: {framework} (failed targets: {failed_targets_str})")

        # Send email notification if email is specified and tests actually ran
        if self.email:
            # Get git information
            git_info = self.git_utils.get_commit_info(self.dynamo_ci_dir)
            
            # Delegate to EmailTemplateRenderer
            self.email_renderer.send_email_notification(
                email=self.email,
                results=test_results,
                git_info=git_info,
                dynamo_ci_dir=self.dynamo_ci_dir,
                log_dir=self.log_dir,
                base_build_times=self.base_build_times,
                base_image_ids=self.base_image_ids,
                workspace_compile_success=self.workspace_compile_success,
                workspace_compile_time=self.workspace_compile_time,
                workspace_compile_output=self.workspace_compile_output,
                failure_details=failure_details,
                build_times=self.build_times,
                dry_run=self.dry_run,
                logger=self.logger,
                docker_utils=self.docker_utils,
                buildsh_utils=self.buildsh_utils
            )

        # Return appropriate exit code
        if overall_success:
            self.logger.info("SUCCESS:All tests completed successfully")
            return 0
        else:
            self.logger.error("Some tests failed")
            return 1


if __name__ == "__main__":
    builder = DynamoDockerBuilder()
    sys.exit(builder.main())
