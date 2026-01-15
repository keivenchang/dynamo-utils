#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generate a "local" branches/PR dashboard from a local Git repository.

Goal: IDENTICAL look & feel as `show_remote_branches.py`, but organized by local
Git repository directory instead of GitHub username.

Structure:
- RepoNode (local repo) → BranchInfoNode (local branch) → CommitMessageNode, MetadataNode,
  PRNode, PRStatusNode (PASSED/FAILED pill) → CIJobNode (CI jobs with hierarchy)

IMPORTANT: Architecture Rule
---------------------------
⚠️ show_remote_branches.py and show_local_branches.py should NEVER import from each other!
   - ALL shared code lives in: common_branch_nodes.py, common_dashboard_lib.py, common.py
   - ✅ REFACTORED: PRStatusNode and _build_ci_hierarchy_nodes now in common_branch_nodes.py

This ensures IDENTICAL rendering logic for status pills, CI hierarchy, and all formatting
between local and remote branch dashboards.

Other shared utilities:
- `BranchInfoNode`, `CommitMessageNode`, `MetadataNode`, `generate_html` from `common_branch_nodes.py`
- `GitHubAPIClient` from `common.py`
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

# Ensure we can import sibling utilities (common.py) from the parent dynamo-utils directory
_THIS_DIR = Path(__file__).resolve().parent
_UTILS_DIR = _THIS_DIR.parent
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from common import GitHubAPIClient, PRInfo  # noqa: E402
from html_pages.common_dashboard_runtime import prune_dashboard_raw_logs, prune_partial_raw_log_caches  # noqa: E402

# Import shared branch/PR node classes and helpers
# All shared code now lives in common_branch_nodes.py (no more cross-imports!)
from common_branch_nodes import (  # noqa: E402
    DYNAMO_OWNER,
    DYNAMO_REPO,
    DYNAMO_REPO_SLUG,
    BranchNode,
    BranchInfoNode,
    BlockedMessageNode,
    CIJobNode,
    CommitMessageNode,
    ConflictWarningNode,
    MetadataNode,
    PRNode,
    PRStatusNode,
    PRURLNode,
    RawLogValidationError,
    RepoNode,
    RerunLinkNode,
    SectionNode,
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
from common_dashboard_lib import TreeNodeVM  # noqa: E402


def get_local_branches_info(repo_dir: Path) -> List[Dict]:
    """Get information about local branches using git commands.
    
    Returns a list of dicts with keys:
    - branch_name: str
    - sha: str (7-char)
    - sha_full: str (full 40-char)
    - is_current: bool
    - commit_time_pt: str (Pacific Time formatted)
    - commit_dt: datetime
    - commit_message: str
    """
    try:
        import git  # GitPython
    except ImportError:
        logging.error("GitPython not available - cannot read local branches")
        return []
    
    try:
        repo = git.Repo(str(repo_dir))
    except Exception as e:
        logging.error(f"Failed to open git repo at {repo_dir}: {e}")
        return []
    
    branches_info = []
    try:
        current_branch_name = repo.active_branch.name if not repo.head.is_detached else None
    except Exception:
        current_branch_name = None
    
    for branch in repo.branches:
        try:
            commit = branch.commit
            sha_full = str(commit.hexsha)
            sha7 = sha_full[:7]
            commit_ts = commit.committed_date  # Unix timestamp
            commit_dt = datetime.fromtimestamp(commit_ts, tz=ZoneInfo("UTC"))
            commit_time_pt = commit_dt.astimezone(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M PT")
            commit_msg = str(commit.message or "").strip().split("\n")[0]  # First line only
            
            branches_info.append({
                "branch_name": branch.name,
                "sha": sha7,
                "sha_full": sha_full,
                "is_current": (branch.name == current_branch_name),
                "commit_time_pt": commit_time_pt,
                "commit_dt": commit_dt,
                "commit_message": commit_msg,
            })
        except Exception as e:
            logging.warning(f"Failed to get info for branch {branch.name}: {e}")
            continue
    
    # Sort by commit date (most recent first)
    branches_info.sort(key=lambda x: x["commit_dt"], reverse=True)
    return branches_info


def get_pr_for_branch(
    branch_name: str,
    github_api: GitHubAPIClient,
    owner: str,
    repo: str,
) -> Optional[PRInfo]:
    """Get PR info for a local branch (if it exists on GitHub)."""
    try:
        # Try to find PR by branch name
        prs = github_api.get_open_prs(owner=owner, repo=repo, head=f"{owner}:{branch_name}", max_prs=1)
        if prs:
            return prs[0]
    except Exception as e:
        logging.debug(f"No PR found for branch {branch_name}: {e}")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Show local branches with PR information (HTML-only)")
    parser.add_argument("--repo-path", type=Path, default=Path.cwd(), help="Path to local Git repository (default: cwd)")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path (default: <repo-path>/local_branches.html)")
    parser.add_argument("--token", help="GitHub personal access token (or login with gh so ~/.config/gh/hosts.yml exists)")
    parser.add_argument("--allow-anonymous-github", action="store_true", help="Allow anonymous GitHub REST calls (60/hr core rate limit)")
    parser.add_argument("--max-github-api-calls", type=int, default=100, help="Hard cap on GitHub REST API network calls per invocation")
    parser.add_argument("--max-branches", type=int, default=50, help="Cap branches shown (default: 50)")
    parser.add_argument("--refresh-checks", action="store_true", help="Force-refresh checks cache TTLs (more GitHub calls)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Configure logging
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    
    repo_path = Path(args.repo_path).resolve()
    if not looks_like_git_repo_dir(repo_path):
        logging.error(f"Not a git repository: {repo_path}")
        return 1
    
    output = args.output
    if output is None:
        output = repo_path / "local_branches.html"
    output = Path(output).resolve()
    
    page_root_dir = output.parent
    
    # Prune old logs
    try:
        _ = prune_dashboard_raw_logs(page_root_dir=page_root_dir, max_age_days=30)
        _ = prune_partial_raw_log_caches(page_root_dirs=[page_root_dir])
    except Exception:
        pass
    
    # GitHub API client
    gh = GitHubAPIClient(
        token=args.token,
        require_auth=(not bool(args.allow_anonymous_github)),
        allow_anonymous_fallback=bool(args.allow_anonymous_github),
        max_rest_calls=int(args.max_github_api_calls),
    )
    
    # Cache-only fallback if exhausted
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
    
    # Get local branches
    branches_info = get_local_branches_info(repo_path)
    branches_info = branches_info[:args.max_branches]
    
    # Determine owner/repo from origin URL
    origin_url = origin_url_from_git_config(repo_path)
    owner = DYNAMO_OWNER
    repo = DYNAMO_REPO
    if origin_url:
        match = re.search(r'github\.com[:/]([^/]+)/([^/.]+)', origin_url)
        if match:
            owner = match.group(1)
            repo = match.group(2)
    
    # Build tree
    root = BranchNode(label="")
    repo_node = RepoNode(label=repo_path.name, repo_path=str(repo_path))
    root.add_child(repo_node)
    
    allow_fetch_checks = not bool(gh.cache_only_mode)
    
    for branch_info in branches_info:
        branch_name = branch_info["branch_name"]
        sha7 = branch_info["sha"]
        sha_full = branch_info["sha_full"]
        is_current = branch_info["is_current"]
        commit_time_pt = branch_info["commit_time_pt"]
        commit_dt = branch_info["commit_dt"]
        commit_msg = branch_info["commit_message"]
        
        commit_url = f"https://github.com/{owner}/{repo}/commit/{sha_full}" if sha_full else None
        
        # Try to get PR for this branch
        pr = get_pr_for_branch(branch_name, gh, owner, repo)
        
        # Create branch node
        branch_node = BranchInfoNode(
            label=branch_name,
            sha=sha7,
            is_current=is_current,
            commit_url=commit_url,
            commit_time_pt=commit_time_pt,
            commit_datetime=commit_dt,
            commit_message=commit_msg,
            pr=pr,  # Pass PR to BranchInfoNode
        )
        repo_node.add_child(branch_node)
        
        # If PR exists, add PR status node
        if pr:
            checks_ttl_s = GitHubAPIClient.compute_checks_cache_ttl_s(
                commit_dt,
                refresh=bool(args.refresh_checks),
            )
            
            status_node = PRStatusNode(
                label="",
                pr=pr,
                github_api=gh,
                refresh_checks=bool(args.refresh_checks),
                branch_commit_dt=commit_dt,
                allow_fetch_checks=bool(allow_fetch_checks),
                context_key=f"local:{repo_path.name}:{branch_name}:{sha7}:pr{pr.number}",
            )
            branch_node.add_child(status_node)
            
            # CI nodes will be built later in the pipeline passes
    
    elapsed_s = max(0.0, time.monotonic() - t0)
    page_stats = [
        ("Generation time", f"{elapsed_s:.2f}s"),
        ("Repo", f"{owner}/{repo}"),
        ("Branches shown", str(len(branches_info))),
    ]
    
    try:
        from common_dashboard_lib import github_api_stats_rows  # local import
        
        mode = "cache-only" if bool(gh.cache_only_mode) else "normal"
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
    
    # Generate HTML (same as remote branches)
    html = generate_html(
        root,
        page_stats=page_stats,
        page_title="Local Branch Info",
        header_title="Local Branch Info",
        tree_sortable=True,  # Enable client-side sorting by modified/created/branch name
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    logging.info(f"Generated: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
