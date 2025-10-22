#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Show dynamo branches with PR information using a node-based tree structure.
Supports parallel data gathering for improved performance.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote
from zoneinfo import ZoneInfo

import git
import requests

# Import GitHub utilities from common module
from common import GitHubAPIClient, FailedCheck, PRInfo




@dataclass
class BranchNode:
    """Base class for tree nodes"""
    label: str
    children: List["BranchNode"] = field(default_factory=list)

    def add_child(self, child: "BranchNode") -> "BranchNode":
        """Add a child node and return it for chaining"""
        self.children.append(child)
        return child

    def render(self, prefix: str = "", is_last: bool = True, is_root: bool = True) -> List[str]:
        """Render the tree node and its children as text lines"""
        lines = []

        # Determine the connector
        if not is_root:
            connector = "└─" if is_last else "├─"
            current_prefix = prefix + connector + " "
        else:
            current_prefix = ""

        # Build the line content
        line_content = self._format_content()
        # Only render if there's actual content (not just whitespace/prefix)
        # Example: "└─ keivenchang/DIS-442 [935d949] ⭐"
        if line_content.strip():
            lines.append(current_prefix + line_content)

        # Render children
        # Example child: "   └─ 📖 PR #3676: feat: add TensorRT-LLM Prometheus metrics support"
        for i, child in enumerate(self.children):
            is_last_child = i == len(self.children) - 1
            if is_root:
                child_prefix = ""
            else:
                child_prefix = prefix + ("   " if is_last else "│  ")
            lines.extend(child.render(child_prefix, is_last_child, False))

        return lines

    def render_html(self, prefix: str = "", is_last: bool = True, is_root: bool = True) -> List[str]:
        """Render the tree node and its children as HTML lines"""
        lines = []

        # Determine the connector
        if not is_root:
            connector = "└─" if is_last else "├─"
            current_prefix = prefix + connector + " "
        else:
            current_prefix = ""

        # Build the line content
        line_content = self._format_html_content()
        # Only render if there's actual content (not just whitespace/prefix)
        # Example: "└─ keivenchang/DIS-442 [935d949] ⭐"
        if line_content.strip():
            lines.append(current_prefix + line_content)

        # Render children
        # Example child: "   └─ 📖 PR #3676: feat: add TensorRT-LLM Prometheus metrics support"
        for i, child in enumerate(self.children):
            is_last_child = i == len(self.children) - 1
            if is_root:
                child_prefix = ""
            else:
                child_prefix = prefix + ("   " if is_last else "│  ")
            lines.extend(child.render_html(child_prefix, is_last_child, False))

        return lines

    def print_tree(self) -> None:
        """Print the tree to console"""
        for line in self.render():
            print(line)

    def _format_content(self) -> str:
        """Format the node content for text output (override in subclasses)"""
        return self.label

    def _format_html_content(self) -> str:
        """Format the node content for HTML output (override in subclasses)"""
        return self.label


@dataclass
class RepoNode(BranchNode):
    """Repository node"""
    path: Optional[Path] = None
    error: Optional[str] = None
    remote_url: Optional[str] = None
    is_correct_repo: bool = True

    def _format_content(self) -> str:
        if self.error:
            return f"\033[1m{self.label}\033[0m\n  \033[91m⚠️  {self.error}\033[0m"
        if not self.is_correct_repo:
            return f"\033[1m{self.label}\033[0m\n  \033[91m⚠️  Not ai-dynamo/dynamo repository\033[0m\n  Remote: {self.remote_url}"
        return f"\033[1m{self.label}\033[0m"

    def _format_html_content(self) -> str:
        # Make repo name clickable (relative URL to the directory)
        # Remove trailing slash from label for URL
        repo_dirname = self.label.rstrip('/')
        repo_link = f'<a href="{repo_dirname}/" class="repo-name">{self.label}</a>'

        if self.error:
            return f'{repo_link}\n<span class="error">⚠️  {self.error}</span>'
        if not self.is_correct_repo:
            return f'{repo_link}\n<span class="error">⚠️  Not ai-dynamo/dynamo repository</span>\n  Remote: {self.remote_url}'
        return repo_link


@dataclass
class SectionNode(BranchNode):
    """Section node (e.g., 'Branches with PRs', 'Local-only branches')"""
    pass


@dataclass
class BranchInfoNode(BranchNode):
    """Branch information node"""
    sha: Optional[str] = None
    is_current: bool = False
    commit_url: Optional[str] = None

    def _format_content(self) -> str:
        marker = " ⭐" if self.is_current else ""
        sha_str = f" [{self.sha}]" if self.sha else ""
        if self.is_current:
            return f"\033[1m{self.label}\033[0m{sha_str}{marker}"
        return f"{self.label}{sha_str}{marker}"

    def _format_html_content(self) -> str:
        marker = " ⭐" if self.is_current else ""
        sha_str = ""
        if self.sha:
            if self.commit_url:
                sha_str = f' [<a href="{self.commit_url}" target="_blank">{self.sha}</a>]'
            else:
                sha_str = f' [{self.sha}]'

        if self.is_current:
            return f'<span class="current">{self.label}</span>{sha_str}{marker}'
        return f'{self.label}{sha_str}{marker}'


@dataclass
class PRNode(BranchNode):
    """Pull request node"""
    pr: Optional[PRInfo] = None

    def _format_content(self) -> str:
        if not self.pr:
            return ""
        if self.pr.is_merged:
            emoji = '🔀'
        elif self.pr.state == 'open':
            emoji = '📖'
        else:
            emoji = '❌'

        # Truncate title at 80 characters
        title = self.pr.title[:80] + '...' if len(self.pr.title) > 80 else self.pr.title
        # Return just the PR info, URL will be shown separately
        return f"{emoji} PR #{self.pr.number}: {title}"

    def _format_html_content(self) -> str:
        if not self.pr:
            return ""
        if self.pr.is_merged:
            emoji = '🔀'
        elif self.pr.state == 'open':
            emoji = '📖'
        else:
            emoji = '❌'

        # Truncate title at 80 characters
        title = self.pr.title[:80] + '...' if len(self.pr.title) > 80 else self.pr.title
        return f'{emoji} <a href="{self.pr.url}" target="_blank">PR #{self.pr.number}</a>: {title}'


@dataclass
class PRURLNode(BranchNode):
    """PR URL node"""
    url: Optional[str] = None

    def _format_content(self) -> str:
        # Don't show URL separately in terminal - it's already visible in the PR title line
        # Matches v1 behavior where URL is shown as a separate line
        if not self.url:
            return ""
        return f"URL: {self.url}"

    def _format_html_content(self) -> str:
        # URL is already in the PR link for HTML, so this can be omitted
        return ""


@dataclass
class PRStatusNode(BranchNode):
    """PR status information node"""
    pr: Optional[PRInfo] = None

    def _format_content(self) -> str:
        if not self.pr:
            return ""
        status_parts = []

        if self.pr.review_decision == 'APPROVED':
            status_parts.append("Review: ✅ Approved")
        elif self.pr.review_decision == 'CHANGES_REQUESTED':
            status_parts.append("Review: 🔴 Changes Requested")

        if self.pr.unresolved_conversations > 0:
            status_parts.append(f"💬 Unresolved: {self.pr.unresolved_conversations}")

        if self.pr.ci_status:
            ci_icon = "✅" if self.pr.ci_status == "passed" else "❌" if self.pr.ci_status == "failed" else "⏳"
            status_parts.append(f"CI: {ci_icon} {self.pr.ci_status}")

        if status_parts:
            return f"Status: {', '.join(status_parts)}"
        return ""

    def _format_html_content(self) -> str:
        return self._format_content()


@dataclass
class BlockedMessageNode(BranchNode):
    """Blocked message node"""

    def _format_content(self) -> str:
        return f"🚫 {self.label}"

    def _format_html_content(self) -> str:
        return self.label


@dataclass
class ConflictWarningNode(BranchNode):
    """Conflict warning node"""

    def _format_content(self) -> str:
        return f"⚠️  {self.label}"

    def _format_html_content(self) -> str:
        return self.label


@dataclass
class FailedTestNode(BranchNode):
    """Failed test node"""
    failed_check: Optional[FailedCheck] = None

    def _format_content(self) -> str:
        if not self.failed_check:
            return ""
        required_marker = " [REQUIRED]" if self.failed_check.is_required else ""
        error_indicator = " [Error details available]" if self.failed_check.error_summary else ""
        return f"❌ {self.failed_check.name}{required_marker} ({self.failed_check.duration}){error_indicator}"

    def _format_html_content(self) -> str:
        if not self.failed_check:
            return ""
        required_marker = ' <span style="color: #cc0000; font-weight: bold;">[REQUIRED]</span>' if self.failed_check.is_required else ""

        # Create unique ID for this error details section
        import hashlib
        detail_id = hashlib.md5(f"{self.failed_check.name}{self.failed_check.job_url}".encode()).hexdigest()[:8]

        base_html = f'❌ <a href="{self.failed_check.job_url}" target="_blank">{self.failed_check.name}</a>{required_marker} ({self.failed_check.duration})'

        # Extract job ID for raw log link
        job_id = None
        if '/job/' in self.failed_check.job_url:
            job_id = self.failed_check.job_url.split('/job/')[1].split('?')[0]

        # Add raw log link
        if job_id:
            # GitHub provides raw logs at the job URL
            base_html += f' <a href="{self.failed_check.job_url}" target="_blank" style="color: #666; font-size: 11px; margin-left: 5px;">[raw log]</a>'

        # Add expandable error details if available
        if self.failed_check.error_summary:
            # Escape HTML in error summary
            import html
            escaped_error = html.escape(self.failed_check.error_summary)
            # Keep "Show error" on the same line, but put the error div on a new line
            base_html += f' <span style="cursor: pointer; color: #0066cc; margin-left: 10px;" onclick="document.getElementById(\'error_{detail_id}\').style.display = document.getElementById(\'error_{detail_id}\').style.display === \'none\' ? \'block\' : \'none\'">▶ Show error</span><div id="error_{detail_id}" style="display: none; margin-left: 20px; margin-top: 5px; padding: 10px; background-color: #fff5f5; border-left: 3px solid #cc0000; font-family: monospace; font-size: 11px; white-space: pre-wrap;">{escaped_error}</div>'

        return base_html


@dataclass
class RerunLinkNode(BranchNode):
    """Rerun link node"""
    url: Optional[str] = None
    run_id: Optional[str] = None

    def _format_content(self) -> str:
        if not self.url or not self.run_id:
            return ""
        return f"🔄 Restart: gh run rerun {self.run_id} --repo ai-dynamo/dynamo --failed"

    def _format_html_content(self) -> str:
        if not self.url or not self.run_id:
            return ""
        return f'🔄 <a href="{self.url}" target="_blank">Restart failed jobs</a> (or: <code>gh run rerun {self.run_id} --repo ai-dynamo/dynamo --failed</code>)'






class LocalRepoScanner:
    """Scanner for local repository branches"""

    def __init__(self, token: Optional[str] = None):
        self.github_api = GitHubAPIClient(token=token)

    def scan_repositories(self, base_dir: Path) -> BranchNode:
        """Scan all dynamo* repositories and build tree structure"""
        root = BranchNode(label="")

        # Find all dynamo* directories
        repo_dirs = sorted([d for d in base_dir.iterdir()
                          if d.is_dir() and d.name.startswith('dynamo')])

        # Scan each repository in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_dir = {
                executor.submit(self._scan_repository, repo_dir): repo_dir
                for repo_dir in repo_dirs
            }

            for future in as_completed(future_to_dir):
                repo_node = future.result()
                if repo_node:
                    root.add_child(repo_node)

        # Sort children by name
        root.children.sort(key=lambda n: n.label)

        return root

    def _scan_repository(self, repo_dir: Path) -> Optional[RepoNode]:
        """Scan a single repository"""
        repo_name = f"{repo_dir.name}/"
        repo_node = RepoNode(label=repo_name, path=repo_dir)

        try:
            repo = git.Repo(repo_dir)
        except Exception as e:
            repo_node.error = f"Not a valid git repository: {e}"
            return repo_node

        # Check if it's the correct repo
        try:
            remote = repo.remote('origin')
            remote_url = next(remote.urls)
            repo_node.remote_url = remote_url

            if 'ai-dynamo/dynamo' not in remote_url:
                repo_node.is_correct_repo = False
                return repo_node
        except Exception:
            repo_node.error = "No origin remote found"
            return repo_node

        # Get current branch
        try:
            current_branch = repo.active_branch.name
        except Exception:
            current_branch = None

        # Collect branch information
        branches_with_prs = {}
        local_only_branches = []

        # Get remote branches
        try:
            remote = repo.remote('origin')
            remote.fetch()
        except Exception:
            pass

        # Scan all local branches
        for branch in repo.branches:
            branch_name = branch.name

            # Skip main branches
            if branch_name in ['main', 'master']:
                continue

            # Check if branch has remote tracking
            try:
                tracking_branch = branch.tracking_branch()
                has_remote = tracking_branch is not None
            except Exception:
                has_remote = False

            # Get commit SHA
            try:
                sha = branch.commit.hexsha[:7]
            except Exception:
                sha = None

            is_current = (branch_name == current_branch)

            if has_remote:
                # Get PR info in parallel (will be gathered later)
                branches_with_prs[branch_name] = {
                    'sha': sha,
                    'is_current': is_current,
                    'branch': branch
                }
            else:
                local_only_branches.append({
                    'name': branch_name,
                    'sha': sha,
                    'is_current': is_current
                })

        # Fetch PR information in parallel
        if branches_with_prs:
            pr_section = SectionNode(label="Branches with PRs")

            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_branch = {
                    executor.submit(
                        self.github_api.get_pr_info,
                        'ai-dynamo',
                        'dynamo',
                        branch_name
                    ): (branch_name, info)
                    for branch_name, info in branches_with_prs.items()
                }

                branch_results = []
                for future in as_completed(future_to_branch):
                    branch_name, info = future_to_branch[future]
                    prs = future.result()
                    if prs:
                        branch_results.append((branch_name, info, prs))

            # Build branch nodes
            for branch_name, info, prs in sorted(branch_results):
                commit_url = f"https://github.com/ai-dynamo/dynamo/commit/{info['sha']}" if info['sha'] else None
                branch_node = BranchInfoNode(
                    label=branch_name,
                    sha=info['sha'],
                    is_current=info['is_current'],
                    commit_url=commit_url
                )

                # Add PR nodes
                for pr in prs:
                    pr_node = PRNode(label="", pr=pr)
                    branch_node.add_child(pr_node)

                    # Add URL node (text only)
                    url_node = PRURLNode(label="", url=pr.url)
                    pr_node.add_child(url_node)

                    # Add status node
                    status_node = PRStatusNode(label="", pr=pr)
                    pr_node.add_child(status_node)

                    # Add conflict warning if applicable
                    if pr.conflict_message:
                        conflict_node = ConflictWarningNode(label=pr.conflict_message)
                        pr_node.add_child(conflict_node)

                    # Add blocking message if applicable
                    if pr.blocking_message:
                        blocked_node = BlockedMessageNode(label=pr.blocking_message)
                        pr_node.add_child(blocked_node)

                    # Add failed checks if any
                    if pr.failed_checks:
                        for failed_check in pr.failed_checks:
                            failed_test_node = FailedTestNode(label="", failed_check=failed_check)
                            pr_node.add_child(failed_test_node)

                        # Add rerun link after all failed checks
                        if pr.rerun_url and pr.failed_checks[0].run_id:
                            rerun_node = RerunLinkNode(label="", url=pr.rerun_url, run_id=pr.failed_checks[0].run_id)
                            pr_node.add_child(rerun_node)

                pr_section.add_child(branch_node)

            if pr_section.children:
                repo_node.add_child(pr_section)

        # Add local-only branches
        if local_only_branches:
            local_section = SectionNode(label="Local-only branches")

            for branch_info in local_only_branches:
                branch_node = BranchInfoNode(
                    label=branch_info['name'],
                    sha=branch_info['sha'],
                    is_current=branch_info['is_current']
                )
                local_section.add_child(branch_node)

            repo_node.add_child(local_section)

        # Add "no branches" message if needed
        if not repo_node.children:
            no_branches = BranchNode(label="No branches with PRs or local-only branches")
            repo_node.add_child(no_branches)

        return repo_node


def generate_html(root: BranchNode) -> str:
    """Generate HTML output from tree"""
    # Get current time in both UTC and PDT
    now_utc = datetime.now(ZoneInfo('UTC'))
    now_pdt = datetime.now(ZoneInfo('America/Los_Angeles'))

    # Format timestamps
    utc_str = now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')
    pdt_str = now_pdt.strftime('%Y-%m-%d %H:%M:%S %Z')

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Dynamo Branch Status</title>
    <style>
        body {{
            font-family: 'Courier New', monospace;
            margin: 20px;
            background-color: #ffffff;
            color: #000000;
            font-size: 13px;
            line-height: 1.4;
        }}
        a {{ color: #0000ee; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .repo-name {{ font-weight: bold; margin-top: 10px; }}
        .current {{ font-weight: bold; }}
        .indent {{ margin-left: 20px; }}
        .error {{ color: #cc0000; }}
        .timestamp {{ color: #666666; font-size: 12px; margin-bottom: 10px; }}
    </style>
</head>
<body>
<pre><span class="timestamp">Generated: {pdt_str} / {utc_str}</span>

"""

    # Render all children (skip root)
    for i, child in enumerate(root.children):
        is_last = i == len(root.children) - 1
        lines = child.render_html(prefix="", is_last=is_last, is_root=True)
        html += "\n".join(lines) + "\n\n"

    html += """</pre>
</body>
</html>
"""

    return html


def compute_state_hash(root: BranchNode) -> str:
    """Compute hash of current state for change detection"""
    # Render the tree and hash it
    lines = []
    for child in root.children:
        lines.extend(child.render())
    content = "\n".join(lines)
    return hashlib.sha256(content.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description='Show dynamo branches with PR information (node-based version)'
    )
    parser.add_argument(
        'base_dir',
        type=Path,
        nargs='?',
        default=Path.cwd(),
        help='Base directory to search for dynamo* repos (default: current directory)'
    )
    parser.add_argument(
        '--token',
        help='GitHub personal access token (or set GITHUB_TOKEN env var)'
    )
    parser.add_argument(
        '--html',
        action='store_true',
        help='Output in HTML format'
    )
    args = parser.parse_args()

    base_dir = args.base_dir

    base_dir = base_dir.resolve()

    # Scan repositories
    scanner = LocalRepoScanner(token=args.token)
    root = scanner.scan_repositories(base_dir)

    # Output
    if args.html:
        print(generate_html(root))
    else:
        for child in root.children:
            child.print_tree()
            print()


if __name__ == '__main__':
    main()
