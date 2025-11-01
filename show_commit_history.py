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
import json
import logging
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict

# Import utilities from common module
from common import DynamoRepositoryUtils, get_terminal_width

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

    def show_commit_history(self, max_commits: int = 50, html_output: bool = False, output_path: Path = None) -> int:
        """Show recent commit history with composite SHAs

        Args:
            max_commits: Maximum number of commits to show
            html_output: Generate HTML output instead of terminal output
            output_path: Path for HTML output file (optional, auto-detected if not provided)

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

                print(f"\nCommit History with Composite SHAs")
                print(f"Repository: {self.repo_path}")
                print(f"Showing {len(commits)} most recent commits:\n")
                print(f"{'Commit SHA':<{sha_width}} {'Composite SHA':<{composite_width}} {'Date':<{date_width}} {'Author':<{author_width}} Message")
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
                        # Collect data for HTML generation
                        commit_data.append({
                            'sha_short': sha_short,
                            'sha_full': sha_full,
                            'composite_sha': composite_sha,
                            'date': date_str,
                            'author': author_name,
                            'message': message_first_line
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
                html_content = self._generate_commit_history_html(commit_data)
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

    def _generate_commit_history_html(self, commit_data: List[dict]) -> str:
        """Generate HTML report for commit history with Docker image detection

        Args:
            commit_data: List of commit dictionaries with sha_short, sha_full, composite_sha, date, author, message

        Returns:
            HTML content as string
        """
        # Get Docker images containing SHAs
        docker_images = self._get_docker_images_by_sha([c['sha_short'] for c in commit_data])

        html = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Dynamo Commit History</title>
<style>
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
    margin: 10px;
    line-height: 1.3;
    background-color: #f6f8fa;
    font-size: 13px;
}
.header {
    background-color: #24292f;
    color: white;
    padding: 10px 15px;
    border-radius: 4px;
    margin-bottom: 10px;
}
.header h1 {
    margin: 0;
    font-size: 18px;
}
.container {
    background-color: white;
    border: 1px solid #d0d7de;
    border-radius: 4px;
    overflow: hidden;
}
table {
    width: 100%;
    border-collapse: collapse;
}
th {
    background-color: #f6f8fa;
    padding: 6px 8px;
    text-align: left;
    font-weight: 600;
    border-bottom: 1px solid #d0d7de;
    position: sticky;
    top: 0;
    font-size: 12px;
}
td {
    padding: 6px 8px;
    border-bottom: 1px solid #d0d7de;
    vertical-align: top;
}
tr:hover {
    background-color: #f6f8fa;
}
.sha {
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
    font-size: 12px;
}
.sha a {
    color: #0969da;
    text-decoration: none;
}
.sha a:hover {
    text-decoration: underline;
}
.message {
    color: #24292f;
}
.pr-link {
    color: #0969da;
    text-decoration: none;
    font-weight: 500;
}
.pr-link:hover {
    text-decoration: underline;
}
.docker-images {
    margin-top: 8px;
}
details {
    margin-top: 4px;
}
summary {
    cursor: pointer;
    color: #0969da;
    font-size: 13px;
    font-weight: 500;
}
summary:hover {
    text-decoration: underline;
}
.image-list {
    margin: 8px 0;
    padding-left: 20px;
}
.image-tag {
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
    font-size: 12px;
    background-color: #f6f8fa;
    padding: 2px 6px;
    border-radius: 3px;
    display: block;
    margin: 4px 0;
}
.date {
    color: #57606a;
    font-size: 14px;
}
.author {
    color: #24292f;
    font-size: 14px;
}
.composite-sha-link {
    color: #0969da;
    text-decoration: none;
    cursor: pointer;
    border-bottom: 1px dotted #0969da;
}
.composite-sha-link:hover {
    text-decoration: underline;
}
</style>
<script>
function toggleDockerImages(event, linkElement) {
    event.preventDefault();
    // Find the current row
    var currentRow = linkElement.closest('tr');
    // Find the next row (which should be the docker-images-row)
    var dockerRow = currentRow.nextElementSibling;

    if (dockerRow && dockerRow.classList.contains('docker-images-row')) {
        // Toggle visibility
        if (dockerRow.style.display === 'none') {
            dockerRow.style.display = 'table-row';
        } else {
            dockerRow.style.display = 'none';
        }
    }
}
</script>
</head>
<body>
"""
        # Add generation timestamp in PDT at the top
        if HAS_PYTZ:
            pdt = pytz.timezone('America/Los_Angeles')
            generated_time = datetime.now(pdt).strftime('%Y-%m-%d %H:%M:%S %Z')
        else:
            generated_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        html += f"""
<div class="header">
<h1>Dynamo Commit History</h1>
<p style="margin: 5px 0 0 0; opacity: 0.9;">Recent commits with composite SHAs and Docker images</p>
<p style="margin: 5px 0 0 0; opacity: 0.7; font-size: 12px;">Page generated: {generated_time}</p>
</div>
"""

        html += """
<div class="container">
<table>
<thead>
<tr>
<th style="width: 120px;">Commit SHA</th>
<th style="width: 140px;">Composite SHA</th>
<th style="width: 180px;">Date/Time (PDT)</th>
<th style="width: 150px;">Author</th>
<th>Message</th>
</tr>
</thead>
<tbody>
"""

        for commit in commit_data:
            sha_short = commit['sha_short']
            sha_full = commit['sha_full']
            composite_sha = commit['composite_sha']
            date_str = commit['date']
            author = commit['author']
            message = commit['message']

            # Create GitHub commit link
            commit_link = f"https://github.com/ai-dynamo/dynamo/commit/{sha_full}"

            # Extract PR number and create PR link
            pr_match = re.search(r'\(#(\d+)\)', message)
            if pr_match:
                pr_number = pr_match.group(1)
                pr_link = f"https://github.com/ai-dynamo/dynamo/pull/{pr_number}"
                # Replace (#1234) with clickable link
                message = re.sub(
                    r'\(#(\d+)\)',
                    f'(<a href="{pr_link}" class="pr-link" target="_blank">#{pr_number}</a>)',
                    message
                )

            # Check if Docker images exist for this commit
            has_docker_images = sha_short in docker_images and docker_images[sha_short]

            # Make composite SHA clickable if Docker images exist
            if has_docker_images:
                composite_sha_html = f'<a href="#" class="composite-sha-link" onclick="toggleDockerImages(event, this); return false;">{composite_sha}</a>'
            else:
                composite_sha_html = composite_sha

            html += f"""
<tr>
<td class="sha"><a href="{commit_link}" target="_blank">{sha_short}</a></td>
<td class="sha">{composite_sha_html}</td>
<td class="date">{date_str}</td>
<td class="author">{author}</td>
<td>
<div class="message">{message}</div>
</td>
</tr>
"""

            # Add Docker images row if any exist for this SHA
            if has_docker_images:
                images = docker_images[sha_short]
                html += f"""
<tr class="docker-images-row" style="display: none;">
<td colspan="5" style="padding: 0; background-color: #f6f8fa;">
<div style="padding: 4px 8px; border-top: 1px solid #d0d7de;">
<table style="width: 100%; border: none; background-color: white; border-collapse: collapse;">
<thead>
<tr>
<th style="width: 50%; padding: 3px 6px; border: none;">Image Name:Tag</th>
<th style="width: 15%; padding: 3px 6px; border: none;">Image ID</th>
<th style="width: 10%; padding: 3px 6px; border: none;">Size</th>
<th style="width: 25%; padding: 3px 6px; border: none;">Created (PDT)</th>
</tr>
</thead>
<tbody>
"""
                for image in sorted(images, key=lambda x: x['tag']):
                    html += f"""
<tr>
<td style="font-family: monospace; font-size: 11px; padding: 2px 6px; border: none;">{image['tag']}</td>
<td style="font-family: monospace; font-size: 11px; padding: 2px 6px; border: none;">{image['id']}</td>
<td style="font-size: 11px; padding: 2px 6px; border: none;">{image['size']}</td>
<td style="font-size: 11px; padding: 2px 6px; border: none;">{image['created']}</td>
</tr>
"""
                html += """
</tbody>
</table>
</div>
</td>
</tr>
"""

        html += """
</tbody>
</table>
</div>
</body>
</html>
"""

        return html

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
        output_path=args.output
    )


if __name__ == '__main__':
    sys.exit(main())
