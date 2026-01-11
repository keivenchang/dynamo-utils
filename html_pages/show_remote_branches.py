#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generate a "remote" branches/PR dashboard for a GitHub user.

Goal: IDENTICAL look & feel as `show_local_branches.py`, but organized by GitHub username
instead of local repository directory.

Structure:
- UserNode (GitHub user) → BranchInfoNode (remote branch) → CommitMessageNode, MetadataNode,
  PRNode, PRStatusNode (PASSED/FAILED pill) → CIJobTreeNode (CI jobs with hierarchy)

This intentionally reuses core components from `show_local_branches.py`:
- `PRNode`, `PRURLNode`, `PRStatusNode`, `_build_ci_hierarchy_nodes`
- These have the complete implementations (versions in common_branch_nodes.py are stubs)
- Ensures IDENTICAL rendering logic for status pills, CI hierarchy, and all formatting

Other shared utilities:
- `BranchInfoNode`, `CommitMessageNode`, `MetadataNode`, `generate_html` from `common_branch_nodes.py`
- `GitHubAPIClient` from `common.py`
"""

from __future__ import annotations

import argparse
import os
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

from common import GitHubAPIClient  # noqa: E402
from html_pages.common_dashboard_runtime import prune_dashboard_raw_logs, prune_partial_raw_log_caches  # noqa: E402

# Import shared branch/PR node classes and helpers
# NOTE: PRStatusNode and _build_ci_hierarchy_nodes are imported from show_local_branches.py
# because the versions in common_branch_nodes.py are incomplete stubs.
from common_branch_nodes import (  # noqa: E402
    DYNAMO_OWNER,
    DYNAMO_REPO,
    DYNAMO_REPO_SLUG,
    BranchNode,
    BranchInfoNode,
    CommitMessageNode,
    MetadataNode,
    PRNode,
    PRURLNode,
    RepoNode,
    SectionNode,
    _format_age_compact,
    _format_branch_metadata_suffix,
    _format_base_branch_inline,
    _pr_needs_attention,
    _strip_repo_prefix_for_clipboard,
    generate_html,
    looks_like_git_repo_dir,
    gitdir_from_git_file,
    origin_url_from_git_config,
    find_local_clone_of_repo,
)
from common_dashboard_lib import TreeNodeVM  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Optional as Opt  # noqa: E402
import html as html_module  # noqa: E402

# Import PRStatusNode and _build_ci_hierarchy_nodes from show_local_branches.py
# These have the correct, complete implementations:
# - PRStatusNode: Generates the PASSED/FAILED/RUNNING pill and applies the centralized CI pipeline
# - _build_ci_hierarchy_nodes: Builds CIJobTreeNode objects from PR check runs
import sys as _sys_for_import
_local_branches_path = Path(__file__).parent / "show_local_branches.py"
if str(_local_branches_path.parent) not in _sys_for_import.path:
    _sys_for_import.path.insert(0, str(_local_branches_path.parent))
from show_local_branches import PRStatusNode, _build_ci_hierarchy_nodes  # noqa: E402


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


class RemoteBranchInfoNode(BranchInfoNode):
    """Branch line for remote PRs (inherits new structure from BranchInfoNode).

    Structure (same as local):
    - Branch line: copy + label + → base + [SHA]
    - Child 1: Commit message (PR#)
    - Child 2: (modified, created, age)
    - Child 3+: PRStatusNode
    """
    # No override needed - BranchInfoNode.to_tree_vm() already implements the correct structure.


class RemotePRStatusNode(PRStatusNode):
    """Remote PR status node (inherits collapse logic from PRStatusNode).
    
    Policy: collapsed for PASSED, expanded for FAILED/RUNNING (same as local).
    """
    # No override needed - PRStatusNode.to_tree_vm() already implements the correct logic.


def main() -> int:
    parser = argparse.ArgumentParser(description="Show remote PRs for a GitHub user (HTML-only)")
    parser.add_argument("--github-user", required=True, help="GitHub username (author of PRs)")
    parser.add_argument("--owner", default=DYNAMO_OWNER, help=f"GitHub owner/org (default: {DYNAMO_OWNER})")
    parser.add_argument("--github-repo", default=DYNAMO_REPO, help=f"GitHub repo (default: {DYNAMO_REPO})")
    parser.add_argument("--repo-root", type=Path, default=None, help="Path to a local clone of the repo (for workflow YAML inference)")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd(), help="Directory to search for a local clone (default: cwd)")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path (default: <base-dir>/remote_prs_<user>.html)")
    parser.add_argument("--token", help="GitHub personal access token (or login with gh so ~/.config/gh/hosts.yml exists)")
    parser.add_argument("--allow-anonymous-github", action="store_true", help="Allow anonymous GitHub REST calls (60/hr core rate limit)")
    parser.add_argument("--max-github-api-calls", type=int, default=100, help="Hard cap on GitHub REST API network calls per invocation")
    parser.add_argument("--max-prs", type=int, default=50, help="Cap PRs shown (default: 50)")
    parser.add_argument("--refresh-checks", action="store_true", help="Force-refresh checks cache TTLs (more GitHub calls)")
    args = parser.parse_args()

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
    try:
        _ = prune_dashboard_raw_logs(page_root_dir=page_root_dir, max_age_days=30)
        _ = prune_partial_raw_log_caches(page_root_dirs=[page_root_dir])
    except Exception:
        pass

    gh = GitHubAPIClient(
        token=args.token,
        require_auth=(not bool(args.allow_anonymous_github)),
        allow_anonymous_fallback=bool(args.allow_anonymous_github),
        max_rest_calls=int(args.max_github_api_calls),
    )

    # Cache-only fallback if exhausted.
    cache_only_reason = ""
    try:
        gh.check_core_rate_limit_or_raise()
    except Exception as e:
        cache_only_reason = str(e)
        try:
            gh.set_cache_only_mode(True)
        except Exception:
            pass

    t0 = time.monotonic()
    prs = gh.get_open_pr_info_for_author(owner, repo, author=user, max_prs=int(args.max_prs))

    root = BranchNode(label="")

    allow_fetch_checks = not bool(getattr(gh, "cache_only_mode", False))

    def _branch_display_for_pr(pr) -> str:
        try:
            head_owner = str(getattr(pr, "head_owner", "") or "").strip()
        except Exception:
            head_owner = ""
        try:
            head_ref = str(getattr(pr, "head_ref", "") or "").strip()
        except Exception:
            head_ref = ""
        try:
            head_label = str(getattr(pr, "head_label", "") or "").strip()
        except Exception:
            head_label = ""
        if head_owner and head_ref:
            return f"{head_owner}/{head_ref}"
        if head_label:
            return head_label.replace(":", "/", 1)
        if head_ref:
            return head_ref
        return f"pr#{getattr(pr, 'number', '')}"

    def _dt_from_iso(s: str) -> Optional[datetime]:
        try:
            dt = datetime.fromisoformat(str(s or "").replace("Z", "+00:00"))
            if getattr(dt, "tzinfo", None) is None:
                dt = dt.replace(tzinfo=ZoneInfo("UTC"))
            return dt
        except Exception:
            return None

    def _pt_str(dt: Optional[datetime]) -> Optional[str]:
        try:
            if dt is None:
                return None
            return dt.astimezone(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M PT")
        except Exception:
            return None

    def build_root(prs_ordered: List[object]) -> BranchNode:
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
            sha7 = ""
            try:
                sha7 = str(getattr(pr, "head_sha", "") or "").strip()[:7]
            except Exception:
                sha7 = ""
            created_dt = _dt_from_iso(str(getattr(pr, "created_at", "") or ""))
            updated_dt = _dt_from_iso(str(getattr(pr, "updated_at", "") or "")) or created_dt
            commit_time_pt = _pt_str(updated_dt)
            commit_url = ""
            try:
                head_sha_full = str(getattr(pr, "head_sha", "") or "").strip()
                if head_sha_full:
                    commit_url = f"https://github.com/{owner}/{repo}/commit/{head_sha_full}"
            except Exception:
                commit_url = ""

            # Use PR title as commit message (since we don't have local git access for remote).
            # Strip PR number prefix if present (e.g., "#5335 feat: ..." -> "feat: ...")
            # The CommitMessageNode will append the PR number as a link.
            commit_msg = str(getattr(pr, "title", "") or "").strip()
            # Remove leading "#1234 " pattern if present
            import re
            commit_msg = re.sub(r'^#\d+\s+', '', commit_msg)

            branch_node = RemoteBranchInfoNode(
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

            status_node = RemotePRStatusNode(
                label="",
                pr=pr,
                github_api=gh,
                refresh_checks=bool(args.refresh_checks),
                branch_commit_dt=updated_dt,
                allow_fetch_checks=bool(allow_fetch_checks),
                context_key=f"remote:{owner}/{repo}:{branch_name}:{sha7}:pr{getattr(pr, 'number', '')}",
            )
            branch_node.add_child(status_node)  # Add directly to branch_node

            # CI hierarchy as children of the PR status line.
            try:
                for ci_node in _build_ci_hierarchy_nodes(
                    repo_root,
                    pr,
                    github_api=gh,
                    page_root_dir=page_root_dir,
                    checks_ttl_s=int(GitHubAPIClient.compute_checks_cache_ttl_s(None, refresh=bool(args.refresh_checks))),
                    skip_fetch=(not bool(allow_fetch_checks)),
                    validate_raw_logs=True,
                ):
                    try:
                        if hasattr(ci_node, "context_key"):
                            setattr(ci_node, "context_key", str(status_node.context_key or ""))
                    except Exception:
                        pass
                    status_node.add_child(ci_node)
            except Exception:
                pass

            # Conflict/blocking messages (add directly to branch_node).
            try:
                msg = getattr(pr, "conflict_message", None)
                if msg:
                    branch_node.add_child(BranchNode(label=str(msg)))
            except Exception:
                pass
            try:
                msg = getattr(pr, "blocking_message", None)
                if msg:
                    branch_node.add_child(BranchNode(label=str(msg)))
            except Exception:
                pass
        return root

    prs_list = list(prs or [])
    # Sort PRs (server-side, like local branches)
    # Default: by most recent activity (updated_at, then created_at)
    try:
        prs_list = sorted(
            prs_list,
            key=lambda p: (
                (_dt_from_iso(getattr(p, "updated_at", None) or "") or _dt_from_iso(getattr(p, "created_at", None) or "") or datetime.min.replace(tzinfo=ZoneInfo("UTC"))),
                int(getattr(p, "number", 0) or 0),
            ),
            reverse=True,
        )
    except Exception:
        pass

    elapsed_s = max(0.0, time.monotonic() - t0)
    page_stats = [
        ("Generation time", f"{elapsed_s:.2f}s"),
        ("Repo", f"{owner}/{repo}"),
        ("GitHub user", user),
        ("PRs shown", str(len(prs))),
    ]
    try:
        from common_dashboard_lib import github_api_stats_rows  # local import

        mode = "cache-only" if bool(getattr(gh, "cache_only_mode", False)) else "normal"
        page_stats.extend(
            list(
                github_api_stats_rows(
                    github_api=gh,
                    max_github_api_calls=int(args.max_github_api_calls),
                    mode=mode,
                    mode_reason=cache_only_reason or "",
                    top_n=15,
                )
                or []
            )
        )
    except Exception:
        pass

    # Generate HTML (same as local branches - no client-side sorting)
    html = generate_html(
        build_root(prs_list) if prs_list else build_root([]),
        page_stats=page_stats,
        page_title=f"Remote PR Info ({user})",
        header_title=f"Remote PR Info ({user})",
        tree_sortable=False,  # No sort controls, render triangles normally
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


