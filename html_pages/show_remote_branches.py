#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generate a "remote" branches/PR dashboard for a GitHub user.

Goal: IDENTICAL look & feel as `show_local_branches.py`, but organized by GitHub username
instead of local repository directory.

Structure:
- UserNode (GitHub user) → BranchInfoNode (remote branch) → BranchCommitMessageNode, BranchMetadataNode,
  PRNode, PRStatusWithJobsNode (PASSED/FAILED pill) → CIJobNode (CI jobs with hierarchy)

IMPORTANT: Architecture Rule
---------------------------
⚠️ show_remote_branches.py and show_local_branches.py should NEVER import from each other!
   - ALL shared code lives in: common_branch_nodes.py, common_dashboard_lib.py, common.py
   - ✅ REFACTORED: PRStatusWithJobsNode and _build_ci_hierarchy_nodes now in common_branch_nodes.py

This ensures IDENTICAL rendering logic for status pills, CI hierarchy, and all formatting
between local and remote branch dashboards.

Other shared utilities:
- `BranchInfoNode`, `BranchCommitMessageNode`, `BranchMetadataNode`, `generate_html` from `common_branch_nodes.py`
- `GitHubAPIClient` from `common.py`
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

# Ensure we can import sibling utilities (common.py) from the parent dynamo-utils directory
_THIS_DIR = Path(__file__).resolve().parent
_UTILS_DIR = _THIS_DIR.parent
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from common_github import GitHubAPIClient, PRInfo  # noqa: E402
from html_pages.common_dashboard_runtime import prune_dashboard_raw_logs, prune_partial_raw_log_caches  # noqa: E402

# Import shared branch/PR node classes and helpers
# All shared code now lives in common_branch_nodes.py (no more cross-imports!)
from common_branch_nodes import (  # noqa: E402
    DYNAMO_OWNER,
    DYNAMO_REPO,
    DYNAMO_REPO_SLUG,
    BranchNode,
    BranchInfoNode,
    CIJobNode,
    BranchCommitMessageNode,
    BranchMetadataNode,
    PRNode,
    PRStatusWithJobsNode,
    PRURLNode,
    RawLogValidationError,
    RepoNode,
    RerunLinkNode,
    _assume_completed_for_check_row,
    _duration_str_to_seconds,
    _format_age_compact,
    _format_branch_metadata_suffix,
    _format_base_branch_inline,
    _is_known_required_check,
    _pr_needs_attention,
    _strip_repo_prefix_for_clipboard,
    find_local_clone_of_repo,
    generate_html,
    gitdir_from_git_file,
    looks_like_git_repo_dir,
    origin_url_from_git_config,
)
from common_dashboard_lib import TreeNodeVM, build_page_stats  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Optional as Opt  # noqa: E402
import html as html_module  # noqa: E402
from common_dashboard_lib import github_api_stats_rows  # noqa: E402
from common_branch_nodes import mock_get_open_pr_info_for_author  # noqa: E402
from snippet_cache import SNIPPET_CACHE  # noqa: E402


@dataclass
class UserNode(BranchNode):
    """User node (like RepoNode but for GitHub users) - collapsible directory containing branches."""
    user_id: Opt[str] = None
    repo_slug: Opt[str] = None
    
    def __init__(self, label: str = "", user_id: Opt[str] = None, repo_slug: Opt[str] = None, **kwargs):
        super().__init__(label=label, **kwargs)
        self.user_id = user_id
        self.repo_slug = repo_slug
    
    def _format_html_content(self) -> str:
        # Make user node clickable (link to GitHub user page)
        user_link = f'<a href="https://github.com/{html_module.escape(self.user_id or "")}" target="_blank" style="color: #0969da; text-decoration: none; font-weight: 600;">{html_module.escape(self.label or "")}</a>'
        
        # Add repo info if available
        if self.repo_slug:
            repo_info = f' <span style="color: #666; font-size: 12px;">({html_module.escape(self.repo_slug)})</span>'
            return f"{user_link}{repo_info}"
        return user_link
    
    def to_tree_vm(self) -> TreeNodeVM:
        """User nodes are collapsible directories (like RepoNode)."""
        # Use a simple, URL-friendly key based on the user ID
        key = f"user:{self.user_id or self.label}"
        has_children = bool(self.children)
        return TreeNodeVM(
            node_key=key,
            label_html=self._format_html_content(),
            children=[c.to_tree_vm() for c in (self.children or [])],
            collapsible=bool(has_children),
            default_expanded=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show remote PRs for a GitHub user (HTML-only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  GH_TOKEN / GITHUB_TOKEN
      GitHub personal access token (alternative to --github-token)
      Priority: --github-token > GH_TOKEN > GITHUB_TOKEN > ~/.config/gh/hosts.yml

  DYNAMO_UTILS_CACHE_DIR
      Override default cache directory (~/.cache/dynamo-utils)

  MAX_GITHUB_API_CALLS
      Can be set when using update_html_pages.sh to override --max-github-api-calls default
        """
    )
    parser.add_argument("--github-user", required=True, help="GitHub username (author of PRs)")
    parser.add_argument("--owner", default=DYNAMO_OWNER, help=f"GitHub owner/org (default: {DYNAMO_OWNER})")
    parser.add_argument("--github-repo", default=DYNAMO_REPO, help=f"GitHub repo (default: {DYNAMO_REPO})")
    parser.add_argument("--repo-root", type=Path, default=None, help="Path to a local clone of the repo (for workflow YAML parsing)")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd(), help="Directory to search for a local clone (default: cwd)")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path (default: <base-dir>/remote_prs_<user>.html)")
    parser.add_argument(
        "--github-token",
        help="GitHub personal access token (preferred). If omitted, we try ~/.config/github-token or ~/.config/gh/hosts.yml.",
    )
    parser.add_argument("--allow-anonymous-github", action="store_true", help="Allow anonymous GitHub REST calls (60/hr core rate limit)")
    parser.add_argument("--max-github-api-calls", type=int, default=100, help="Hard cap on GitHub REST API network calls per invocation")
    parser.add_argument("--max-prs", type=int, default=50, help="Cap PRs shown (default: 50)")
    parser.add_argument("--refresh-checks", action="store_true", help="Force-refresh checks cache TTLs (more GitHub calls)")
    parser.add_argument(
        "--enable-success-build-test-logs",
        action="store_true",
        help='Opt-in: cache raw logs for successful *-build-test jobs so we can parse pytest slowest tests under "Run tests" (slower).',
    )
    parser.add_argument("--use-text-trees", action="store_true", help="Use old text-based tree rendering instead of interactive <div> (legacy)")
    parser.add_argument("--create-dummy-prs", action="store_true", help="Create 2 dummy PRs to visualize YAML structure (for testing)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Configure logging
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    user = str(args.github_user or "").strip()
    owner = str(args.owner or "").strip()
    repo = str(args.github_repo or "").strip()
    base_dir = Path(args.base_dir).resolve()

    output = args.output
    if output is None:
        safe_user = "".join([c for c in user if c.isalnum() or c in {"-", "_"}]) or "user"
        output = base_dir / f"remote_prs_{safe_user}.html"
    output = Path(output).resolve()

    # Determine local repo root for workflow YAML inference (best-effort).
    repo_root = Path(args.repo_root).resolve() if args.repo_root is not None else None
    if repo_root is None:
        repo_root = find_local_clone_of_repo(base_dir, repo_slug=f"{owner}/{repo}")
    if repo_root is None:
        # Degrade gracefully: YAML inference will be unavailable, but PR/CI rows still render from APIs.
        repo_root = base_dir

    page_root_dir = output.parent

    # Prune locally-served raw logs to avoid unbounded growth and delete any partial/unverified artifacts.
    # Best-effort: pruning should never block dashboard generation.
    try:
        _ = prune_dashboard_raw_logs(page_root_dir=page_root_dir, max_age_days=30)
        _ = prune_partial_raw_log_caches(page_root_dirs=[page_root_dir])
    except (OSError, RuntimeError, ValueError) as e:
        logging.getLogger(__name__).warning("Failed to prune dashboard raw logs: %s", e)

    gh = GitHubAPIClient(
        token=args.github_token,
        require_auth=(not bool(args.allow_anonymous_github)),
        allow_anonymous_fallback=bool(args.allow_anonymous_github),
        max_rest_calls=int(args.max_github_api_calls),
    )

    # Cache-only fallback if exhausted.
    cache_only_reason = ""
    try:
        gh.check_core_rate_limit_or_raise()
    except RuntimeError as e:
        cache_only_reason = str(e)
        gh.set_cache_only_mode(True)

    t0 = time.monotonic()
    
    # Create dummy PRs mode: use mock PRInfo objects for testing YAML structure
    if args.create_dummy_prs:
        prs = mock_get_open_pr_info_for_author(
            owner=owner,
            repo=repo,
            author=user,
            num_prs=2,
        )
    else:
        prs = gh.get_open_pr_info_for_author(owner, repo, author=user, max_prs=int(args.max_prs))

    root = BranchNode(label="")

    allow_fetch_checks = not gh.cache_only_mode

    def _branch_display_for_pr(pr: PRInfo) -> str:
        head_owner = str(pr.head_owner or "").strip()
        head_ref = str(pr.head_ref or "").strip()
        head_label = str(pr.head_label or "").strip()
        if head_owner and head_ref:
            return f"{head_owner}/{head_ref}"
        if head_label:
            return head_label.replace(":", "/", 1)
        if head_ref:
            return head_ref
        return f"pr#{pr.number}"

    def _dt_from_iso(s: str) -> Optional[datetime]:
        try:
            dt = datetime.fromisoformat(str(s or "").replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=ZoneInfo("UTC"))
            return dt
        except ValueError:
            return None

    def _pt_str(dt: Optional[datetime]) -> Optional[str]:
        if dt is None:
            return None
        return dt.astimezone(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M PT")

    def build_root(prs_ordered: List[PRInfo]) -> BranchNode:
        """Build tree: root -> UserNode -> branches (same structure as local: root -> RepoNode -> branches)."""
        root = BranchNode(label="")
        
        # Create a user node (like RepoNode for local branches)
        user_node = UserNode(
            label=f"{user}",
            user_id=user,
            repo_slug=f"{owner}/{repo}",
        )
        root.add_child(user_node)
        
        # Add each PR as a branch under the user node (matching local branch structure)
        for pr in (prs_ordered or []):
            branch_name = _branch_display_for_pr(pr)
            head_sha_full = str(pr.head_sha or "").strip()
            sha7 = head_sha_full[:7]
            created_dt = _dt_from_iso(str(pr.created_at or ""))
            updated_dt = _dt_from_iso(str(pr.updated_at or "")) or created_dt
            commit_time_pt = _pt_str(updated_dt)
            commit_url = f"https://github.com/{owner}/{repo}/commit/{head_sha_full}" if head_sha_full else ""

            # Use PR title as commit message (since we don't have local git access for remote).
            # Strip PR number prefix if present (e.g., "#5335 feat: ..." -> "feat: ...")
            # The CommitMessageNode will append the PR number as a link.
            commit_msg = str(pr.title or "").strip()
            commit_msg = re.sub(r'^#\d+\s+', '', commit_msg)

            branch_node = BranchInfoNode(
                label=branch_name,
                sha=sha7 or None,
                is_current=False,
                commit_url=commit_url or None,
                commit_time_pt=commit_time_pt,
                commit_datetime=updated_dt,
                commit_message=commit_msg,  # Show commit message (PR title) with (#PR) appended
                pr=pr,
            )
            user_node.add_child(branch_node)  # Add to user_node, not root

            status_node = PRStatusWithJobsNode(
                label="",
                pr=pr,
                github_api=gh,
                repo_root=repo_root,
                page_root_dir=output.parent,
                refresh_checks=bool(args.refresh_checks),
                branch_commit_dt=updated_dt,
                allow_fetch_checks=bool(allow_fetch_checks),
                context_key=f"remote:{owner}/{repo}:{branch_name}:{sha7}:pr{pr.number}",
                enable_success_build_test_logs=bool(args.enable_success_build_test_logs),
            )
            branch_node.add_child(status_node)  # Add directly to branch_node

            # CI nodes are built inside PRStatusWithJobsNode via run_all_passes().
        return root

    prs_list = list(prs or [])
    # Sort PRs (server-side, like local branches)
    # Default: by most recent activity (updated_at, then created_at)
    prs_list = sorted(
        prs_list,
        key=lambda p: (
            (_dt_from_iso(str(p.updated_at or "")) or _dt_from_iso(str(p.created_at or "")) or datetime.min.replace(tzinfo=ZoneInfo("UTC"))),
            int(p.number or 0),
        ),
        reverse=True,
    )

    elapsed_s = max(0.0, time.monotonic() - t0)

    # Build the tree FIRST
    root = build_root(prs_list) if prs_list else build_root([])

    # Force tree conversion to TreeNodeVM to trigger snippet extraction and cache stats population.
    # This must happen BEFORE build_page_stats() so statistics are captured correctly.
    _ = root.to_tree_vm()

    # THEN capture statistics (now includes snippet extraction stats)
    page_stats = build_page_stats(
        generation_time_secs=elapsed_s,
        github_api=gh,
        max_github_api_calls=int(args.max_github_api_calls),
        cache_only_mode=bool(gh.cache_only_mode),
        cache_only_reason=cache_only_reason or "",
        repo_info=f"{owner}/{repo}",
        github_user=user,
        prs_shown=len(prs),
    )

    # Generate HTML (same as local branches - no client-side sorting)
    html = generate_html(
        root,
        page_stats=page_stats,
        page_title=f'<span style="color: #fbbf24;">Augmented</span> Pull Requests [{user}]',
        header_title=f'<span style="color: #fbbf24;">Augmented</span> Pull Requests [{user}]',
        tree_sortable=False,  # No sort controls, render triangles normally
        use_div_trees=not args.use_text_trees,  # Default to div trees unless --use-text-trees is specified
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")

    # Flush shared caches to disk
    SNIPPET_CACHE.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


