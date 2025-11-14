#!/usr/bin/env python3
"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Dynamo Commit History Generator - Standalone Tool

Generates commit history with composite SHAs and Docker image detection.
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
    """Generate commit history with composite SHAs and Docker images"""

    def __init__(self, repo_path: Path, verbose: bool = False):
        """
        Initialize the commit history generator

        Args:
            repo_path: Path to the Dynamo repository
            verbose: Enable verbose output
        """
        self.repo_path = Path(repo_path)
        self.verbose = verbose
        self.logger = self._setup_logger()
        self.cache_file = self.repo_path / ".commit_history_cache.json"

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('CommitHistoryGenerator')
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG if self.verbose else logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def show_commit_history(self, max_commits: int = 50, html_output: bool = False, output_path: Path = None, logs_dir: Path = None) -> int:
        """Show recent commit history with composite SHAs

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

        # Load cache
        cache = {}
        if self.cache_file.exists():
            try:
                cache = json.loads(self.cache_file.read_text())
                if self.verbose:
                    print(f"Loaded cache with {len(cache)} entries")
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
                    date_str = commit.committed_datetime.strftime('%Y-%m-%d %H:%M:%S')
                    author_name = commit.author.name
                    message_first_line = commit.message.strip().split('\n')[0]

                    if not html_output:
                        # Terminal: show progress with placeholder
                        author_str = author_name[:author_width-1]
                        message_str = message_first_line[:message_width]
                        print(f"{sha_short:<{sha_width}} {'CALCULATING...':<{composite_width}} {date_str:<{date_width}} {author_str:<{author_width}} {message_str}", end='', flush=True)

                    # Check cache first
                    if sha_full in cache:
                        composite_sha = cache[sha_full]
                        if self.verbose:
                            print(f"Cache hit for {sha_short}: {composite_sha}")
                    else:
                        # Checkout commit and calculate composite SHA
                        try:
                            repo.git.checkout(commit.hexsha)
                            composite_sha = repo_utils.generate_composite_sha()
                            # Update cache
                            cache[sha_full] = composite_sha
                            cache_updated = True
                            if self.verbose:
                                print(f"Calculated and cached {sha_short}: {composite_sha}")
                        except Exception as e:
                            composite_sha = "ERROR"

                    if html_output:
                        # Get commit stats (files changed, insertions, deletions)
                        stats = commit.stats.total
                        files_changed = stats['files']
                        insertions = stats['insertions']
                        deletions = stats['deletions']

                        # Get full commit message
                        full_message = commit.message.strip()

                        # Get list of changed files
                        changed_files = list(commit.stats.files.keys())

                        # Collect data for HTML generation
                        commit_data.append({
                            'sha_short': sha_short,
                            'sha_full': sha_full,
                            'composite_sha': composite_sha,
                            'date': date_str,
                            'author': author_name,
                            'message': message_first_line,
                            'full_message': full_message,
                            'files_changed': files_changed,
                            'insertions': insertions,
                            'deletions': deletions,
                            'changed_files': changed_files
                        })
                        if self.verbose:
                            print(f"Processed commit {i+1}/{len(commits)}: {sha_short}")
                    else:
                        # Terminal: overwrite line with actual composite SHA
                        author_str = author_name[:author_width-1]
                        message_str = message_first_line[:message_width]
                        print(f"\r{sha_short:<{sha_width}} {composite_sha:<{composite_width}} {date_str:<{date_width}} {author_str:<{author_width}} {message_str}")

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
                html_content = self._generate_commit_history_html(commit_data, logs_dir)
                # Determine output path
                if output_path is None:
                    # Auto-detect: Write to logs directory within the repo (or current directory if logs doesn't exist)
                    logs_dir = self.repo_path / "logs"
                    if logs_dir.exists():
                        output_path = logs_dir / "commit-history.html"
                    else:
                        output_path = Path("commit-history.html")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(html_content)
                print(f"\nHTML report generated: {output_path}")
                print(f"Restored HEAD to {original_head[:9]}")

            # Save cache if updated
            if cache_updated:
                try:
                    self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                    self.cache_file.write_text(json.dumps(cache, indent=2))
                    if self.verbose:
                        print(f"Cache saved with {len(cache)} entries")
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

    def _generate_commit_history_html(self, commit_data: List[dict], logs_dir: Path) -> str:
        """Generate HTML report for commit history with Docker image detection

        Args:
            commit_data: List of commit dictionaries with sha_short, sha_full, composite_sha, date, author, message
            logs_dir: Path to logs directory for build reports

        Returns:
            HTML content as string
        """
        if not HAS_JINJA2:
            raise ImportError("jinja2 is required for HTML generation. Install with: pip install jinja2")
        
        # Get Docker images containing SHAs
        docker_images = self._get_docker_images_by_sha([c['sha_short'] for c in commit_data])
        
        # Get GitLab container registry images for commits (with caching)
        gitlab_images_raw = self._get_cached_gitlab_images([c['sha_full'] for c in commit_data])
        
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
            formatted_imgs = []
            for (framework, arch), img in sorted(deduped_imgs.items()):
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
        
        current_color_index = 0
        previous_composite_sha = None
        
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
            
            # Assign color based on composite SHA changes
            composite_sha = commit['composite_sha']
            if previous_composite_sha is not None and composite_sha != previous_composite_sha:
                # Composite SHA changed, move to next color
                current_color_index = (current_color_index + 1) % len(bg_colors)
            
            commit['composite_bg_color'] = bg_colors[current_color_index]
            previous_composite_sha = composite_sha
        
        # Build log paths dictionary
        log_paths = {}
        for commit in commit_data:
            sha_short = commit['sha_short']
            if sha_short in docker_images and docker_images[sha_short]:
                # Search for build log
                log_filename = f"*.{sha_short}.report.html"
                search_pattern = str(logs_dir / "*" / log_filename)
                matching_logs = glob.glob(search_pattern)
                
                if matching_logs:
                    log_path = Path(sorted(matching_logs)[-1])
                    try:
                        nvidia_dir = Path.home() / 'nvidia'
                        relative_parts = log_path.relative_to(nvidia_dir)
                        log_paths[sha_short] = f"../{relative_parts}"
                    except ValueError:
                        log_paths[sha_short] = str(log_path)
        
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
            log_paths=log_paths,
            generated_time=generated_time
        )

    def _get_cached_gitlab_images(self, sha_full_list: List[str]) -> dict:
        """Get GitLab registry images with caching.
        
        Args:
            sha_full_list: List of full commit SHAs (40 characters)
            
        Returns:
            Dictionary mapping SHA to list of registry image info
        """
        # Load cache
        cache = {}
        gitlab_cache_file = self.repo_path / ".gitlab_registry_cache.json"
        if gitlab_cache_file.exists():
            try:
                cache = json.loads(gitlab_cache_file.read_text())
                if self.verbose:
                    self.logger.info(f"Loaded GitLab cache with {len(cache)} entries")
            except Exception as e:
                self.logger.warning(f"Failed to load GitLab cache: {e}")
        
        # Check which SHAs need to be fetched
        shas_to_fetch = []
        result = {}
        
        for sha in sha_full_list:
            if sha in cache:
                result[sha] = cache[sha]
                if self.verbose:
                    self.logger.info(f"Cache hit for GitLab registry: {sha[:9]}")
            else:
                shas_to_fetch.append(sha)
                result[sha] = []
        
        # Fetch missing SHAs from GitLab
        if shas_to_fetch:
            if self.verbose:
                self.logger.info(f"Fetching {len(shas_to_fetch)} SHAs from GitLab registry")
            
            gitlab_client = GitLabAPIClient()
            fresh_data = gitlab_client.get_registry_images_for_shas(
                project_id="169905",  # dl/ai-dynamo/dynamo
                registry_id="85325",  # Main dynamo registry
                sha_list=shas_to_fetch
            )
            
            # Update result and cache
            for sha, images in fresh_data.items():
                result[sha] = images
                cache[sha] = images
            
            # Save updated cache
            try:
                gitlab_cache_file.parent.mkdir(parents=True, exist_ok=True)
                gitlab_cache_file.write_text(json.dumps(cache, indent=2))
                if self.verbose:
                    self.logger.info(f"Saved GitLab cache with {len(cache)} entries")
            except Exception as e:
                self.logger.warning(f"Failed to save GitLab cache: {e}")
        
        return result

    def _get_docker_images_by_sha(self, sha_list: List[str]) -> dict:
        """Get Docker images containing each SHA in their tag

        Args:
            sha_list: List of short SHAs (9 characters)

        Returns:
            Dictionary mapping SHA to list of image details (dicts with tag, id, size, created)
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
        '--verbose',
        action='store_true',
        help='Enable verbose output'
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
        verbose=args.verbose
    )

    return generator.show_commit_history(
        max_commits=args.max_commits,
        html_output=args.html,
        output_path=args.output,
        logs_dir=args.logs_dir
    )


if __name__ == '__main__':
    sys.exit(main())
