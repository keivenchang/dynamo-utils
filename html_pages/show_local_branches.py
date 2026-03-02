#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Show dynamo branches with PR information using a node-based tree structure.
Supports parallel data gathering for improved performance.

Structure:
- RepoNode (local directory) → BranchInfoNode (branch) → BranchCommitMessageNode, BranchMetadataNode,
  PRStatusWithJobsNode (PASSED/FAILED pill) → CIJobNode (CI jobs with hierarchy)

IMPORTANT: Architecture Rule
---------------------------
⚠️ show_local_branches.py and show_remote_branches.py should NEVER import from each other!
   - ALL shared code lives in: common_branch_nodes.py, common_dashboard_lib.py, common.py
   - ✅ REFACTORED: PRStatusWithJobsNode and _build_ci_hierarchy_nodes now in common_branch_nodes.py

This is the reference implementation for branch/PR dashboards. show_remote_branches.py
imports the same shared components from common_branch_nodes.py to ensure IDENTICAL
rendering logic.

Key Features:
- Client-side sorting (by modified date, created date, or branch name)
- Collapsible repository directories with URL state persistence
- Full CI job hierarchy with parent-child relationships
- Workflow status for branches with remotes but no PRs
- Cached GitHub API calls with smart TTL
- Failure snippets with raw log caching

NOTE: The actual implementation of PRStatusWithJobsNode, _build_ci_hierarchy_nodes, and related
classes/functions are in common_branch_nodes.py. This file only imports them.
"""

import argparse
import getpass
import html
import logging
import json
import os
import re
import stat
import subprocess
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Ensure we can import sibling utilities (common.py) from the parent dynamo-utils directory
_THIS_DIR = Path(__file__).resolve().parent
_UTILS_DIR = _THIS_DIR.parent
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import git  # type: ignore[import-not-found]

# Shared dashboard helpers (UI + workflow graph)
from common_dashboard_lib import (
    CIStatus,
    EXPECTED_CHECK_PLACEHOLDER_SYMBOL,
    PASS_PLUS_STYLE,
    TreeNodeVM,
    build_page_stats,
    check_line_html,
    ci_should_expand_by_default,
    compact_ci_summary_html,
    disambiguate_check_run_name,
    extract_actions_job_id_from_url,
    render_tree_divs,
    required_badge_html,
    status_icon_html,
)

# Dashboard runtime (HTML-only) helpers
from common_dashboard_runtime import (
    atomic_write_text,
    materialize_job_raw_log_text_local_link,
    prune_dashboard_raw_logs,
    prune_partial_raw_log_caches,
)

# Log/snippet helpers (shared library: `dynamo-utils.PRODUCTION/ci_log_errors/`)
from ci_log_errors import snippet as ci_snippet

# Snippet cache (shared across all dashboards)
from cache_snippet import SNIPPET_CACHE

# Jinja2 for HTML template rendering
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Import utilities from common and common_github modules
import common
from common import (
    PhaseTimer,
    dynamo_utils_cache_dir,
)
from common_github import (
    FailedCheck,
    GHPRCheckRow,
    GitHubAPIClient,
    PRInfo,
    classify_ci_kind,
    summarize_pr_check_rows,
)

def _detect_branch_merged_locally(*, repo_dir: Path, branch_sha: Optional[str], pr: Optional[PRInfo] = None) -> bool:
    """Best-effort local merge detection: check if branch changes are in main.
    
    GitHub merge commits create new SHAs, so we can't rely on ancestry alone.
    Instead, check if a commit with the same PR number exists in main.
    """
    sha = str(branch_sha or "").strip()
    if not sha:
        return False
    
    # Method 1: Direct ancestry check (works for fast-forward merges)
    if _is_sha_ancestor_of_main(repo_dir=repo_dir, sha=sha):
        return True
    
    # Method 2: Check if commit with same PR number exists in main (handles merge commits)
    pr_num = None
    
    if pr and pr.number:
        pr_num = str(pr.number)
    else:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_dir), "log", "-1", "--format=%s", sha],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                text=True,
            )
            if result.returncode == 0:
                commit_msg = result.stdout.strip()
                pr_match = re.search(r'\(#(\d+)\)', commit_msg)
                if pr_match:
                    pr_num = pr_match.group(1)
        except Exception:
            pass
    
    if pr_num:
        # Determine base_ref for the PR-number grep
        has_origin_main = subprocess.run(
            ["git", "-C", str(repo_dir), "show-ref", "--verify", "--quiet", "refs/remotes/origin/main"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
        base_ref = "origin/main" if has_origin_main else "main"
        try:
            search_result = subprocess.run(
                ["git", "-C", str(repo_dir), "log", "--oneline", base_ref, "--grep", f"(#{pr_num})"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                text=True,
            )
            if search_result.returncode == 0 and search_result.stdout.strip():
                return True
        except Exception:
            pass
    
    return False

# Import shared node classes and refactored functions from common_branch_nodes
from common_branch_nodes import (
    BranchInfoNode,
    BranchNode,
    PRStatusWithJobsNode,
    RepoNode,
    _copy_icon_svg,
    _is_sha_ancestor_of_main,
    looks_like_git_repo_dir,
    gitdir_from_git_file,
    origin_url_from_git_config,
    find_local_clone_of_repo,
)
from dataclasses import dataclass  # noqa: E402
import html as html_module  # noqa: E402


# Local user parent node (similar role to show_remote_branches.py's UserNode, but for local directories).
@dataclass
class LocalUserNode(BranchNode):
    """Top-level local user node containing all scanned local repositories."""

    user_id: Optional[str] = None
    base_dir: Optional[str] = None

    def __init__(
        self,
        label: str = "",
        *,
        user_id: Optional[str] = None,
        base_dir: Optional[Path] = None,
        **kwargs,
    ):
        super().__init__(label=label, **kwargs)
        self.user_id = user_id
        self.base_dir = str(base_dir) if base_dir is not None else None

    def _format_html_content(self) -> str:
        label_html = html_module.escape(self.label or "")
        base_suffix = ""
        if self.base_dir:
            base_suffix = f' <span style="color: #666; font-size: 12px;">({html_module.escape(self.base_dir)})</span>'
        return f'<span style="font-weight: 600;">{label_html}</span>{base_suffix}'

    def to_tree_vm(self) -> TreeNodeVM:
        key = f"local-user:{self.user_id or self.label or 'local'}"
        kids = [c.to_tree_vm() for c in (self.children or [])]
        return TreeNodeVM(
            node_key=key,
            label_html=self._format_html_content(),
            children=kids,
            collapsible=bool(kids),
            default_expanded=True,
        )


# Local-specific RepoNode subclass with path and error tracking
@dataclass
class LocalRepoNode(RepoNode):
    """Repository node for local branches (extends shared RepoNode)."""
    repo_path: Optional[str] = None
    error: Optional[str] = None
    remote_url: Optional[str] = None
    rel_path: Optional[str] = None  # Relative path for folder icon link
    
    def __init__(self, label: str, *, repo_path: Optional[Path] = None, **kwargs):
        super().__init__(label=label, **kwargs)
        self.repo_path = str(repo_path) if repo_path is not None else None
        self.error = None
        self.remote_url = None
        self.rel_path = None
    
    def _format_html_content(self) -> str:
        """Format repo node with symlink target if applicable."""
        # If the repository directory itself is a symlink, show the target for clarity.
        link_suffix = ""
        p = Path(self.repo_path) if self.repo_path is not None else None
        if p is not None and p.is_symlink():
            tgt = str(p.readlink().resolve())
            link_suffix = f' <span style="color: #666; font-size: 11px;">→ {html_module.escape(tgt)}</span>'

        label_html = html_module.escape(self.label)
        
        if self.error:
            error_html = f' <span style="color: #c83a3a; font-size: 11px;">(Error: {html_module.escape(self.error)})</span>'
            return f'<span style="font-weight: 600;">{label_html}</span>{link_suffix}{error_html}'
        return f'<span style="font-weight: 600;">{label_html}</span>{link_suffix}'
    
    def to_tree_vm(self) -> TreeNodeVM:
        """Convert local repo node to TreeNodeVM with optional folder icon link."""
        kids = [c.to_tree_vm() for c in (self.children or [])]
        
        # Create folder icon that links to the directory (appears before the triangle)
        folder_icon = ""
        if self.rel_path:
            # Use folder SVG icon that links to the directory
            folder_icon = (
                f'<a href="{html_module.escape(self.rel_path)}/" '
                f'onclick="event.preventDefault(); window.location.href=this.href;" '
                f'style="text-decoration: none; margin-right: 4px;" '
                f'title="Open directory: {html_module.escape(self.rel_path)}">'
                f'<svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align: middle;">'
                f'<path fill="#54aeff" d="M1.75 2.5a.25.25 0 00-.25.25v10.5c0 .138.112.25.25.25h12.5a.25.25 0 00.25-.25v-8.5a.25.25 0 00-.25-.25h-6.5a.75.75 0 01-.75-.75V2.75a.25.25 0 00-.25-.25z"/>'
                f'</svg>'
                f'</a>'
            )
        
        return TreeNodeVM(
            node_key=f"repo:{self.label}",
            label_html=folder_icon + self._format_html_content(),
            children=kids,
            collapsible=bool(kids),
            default_expanded=True,
        )


def _tree_has_current_branch(node: BranchNode) -> bool:
    """Return True if any BranchInfoNode in this subtree is marked as current.

    This prevents rendering the current branch twice (once inside a section like "Branches"
    and again as a top-level fallback line).
    """
    if isinstance(node, BranchInfoNode) and bool(node.is_current):
        return True
    for child in (node.children or []):
        if _tree_has_current_branch(child):
            return True
    return False


_COPY_BTN_STYLE = (
    "padding: 1px 4px; font-size: 10px; line-height: 1; background-color: transparent; color: #57606a; "
    "border: 1px solid #d0d7de; border-radius: 5px; cursor: pointer; display: inline-flex; "
    "align-items: center; vertical-align: baseline; margin-right: 4px;"
)

#
# Repo constants (avoid scattering hardcoded strings)
#

DYNAMO_OWNER = "ai-dynamo"
DYNAMO_REPO = "dynamo"
DYNAMO_REPO_SLUG = f"{DYNAMO_OWNER}/{DYNAMO_REPO}"



_COPY_ICON_SVG = _copy_icon_svg(size_px=12)


class LocalRepoScanner:
    """Scanner for local repository branches"""

    def __init__(
        self,
        token: Optional[str] = None,
        refresh_closed_prs: bool = False,
        *,
        max_branches: Optional[int] = None,
        max_checks_fetch: Optional[int] = None,
        allow_anonymous_github: bool = False,
        max_github_api_calls: int = 100,
        enable_success_build_test_logs: bool = False,
        repo_root: Optional[Path] = None,
        page_root_dir: Optional[Path] = None,
    ):
        require_auth = not bool(allow_anonymous_github)
        self.github_api = GitHubAPIClient(
            token=token,
            require_auth=require_auth,
            allow_anonymous_fallback=bool(allow_anonymous_github),
            max_rest_calls=int(max_github_api_calls),
        )
        self.refresh_closed_prs = bool(refresh_closed_prs)
        self.max_branches = int(max_branches) if max_branches is not None else None
        self.max_checks_fetch = int(max_checks_fetch) if max_checks_fetch is not None else None
        self.cache_only_github: bool = False
        self.enable_success_build_test_logs = bool(enable_success_build_test_logs)
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
        self.page_root_dir = Path(page_root_dir).resolve() if page_root_dir is not None else None

    @staticmethod
    def _is_world_readable_executable_dir(p: Path) -> bool:
        """
        True iff `p` is a directory with world-readable + world-executable permissions (o+r and o+x).

        For directories, readable enables listing and executable enables traversal.
        """
        st = p.stat()
        mode = st.st_mode
        return bool(mode & stat.S_IROTH) and bool(mode & stat.S_IXOTH)

    def scan_repositories(self, base_dir: Path) -> BranchNode:
        """Scan all git repositories under `base_dir` (direct children only) and build tree structure."""
        root = BranchNode(label="")

        # Match show_remote_branches.py style: root -> UserNode -> (things).
        # For local branches, we group all repo directories under a single local-user parent node.
        user_id = str(getpass.getuser() or "").strip() or "local"
        user_node = LocalUserNode(label=user_id, user_id=user_id, base_dir=base_dir)
        root.add_child(user_node)

        # Discover git repos among direct children (and include base_dir itself if it's a repo).
        # We intentionally do NOT walk the whole tree because this workspace can be huge.
        candidate_dirs: list[Path] = []
        if looks_like_git_repo_dir(base_dir) and self._is_world_readable_executable_dir(base_dir):
            candidate_dirs.append(base_dir)

        for d in base_dir.iterdir():
            if not d.is_dir():
                continue
            if not self._is_world_readable_executable_dir(d):
                continue
            if not looks_like_git_repo_dir(d):
                continue
            candidate_dirs.append(d)

        repo_dirs = sorted(candidate_dirs, key=lambda p: p.name)

        # Scan each repository in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_dir = {
                executor.submit(self._scan_repository, repo_dir, base_dir): repo_dir
                for repo_dir in repo_dirs
            }

            for future in as_completed(future_to_dir):
                repo_node = future.result()
                if repo_node:
                    user_node.add_child(repo_node)

        # Sort repos by name (within the user node)
        user_node.children.sort(key=lambda n: n.label)

        return root

    @staticmethod
    def _format_pt_time(dt) -> Optional[str]:
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        pt = dt.astimezone(ZoneInfo("America/Los_Angeles"))
        return f"{pt.strftime('%Y-%m-%d %H:%M')} PT"

    @staticmethod
    def _scan_and_classify_branches(
        repo,
        current_branch: Optional[str],
        repo_node: LocalRepoNode,
    ) -> Tuple[Dict[str, dict], List[dict]]:
        """Scan local branches, fetch remotes, classify into remote-tracking and local-only.

        Returns:
            (branches_with_prs, local_only_branches) where branches_with_prs maps
            branch_name -> info dict and local_only_branches is a list of info dicts.
        """
        remote = repo.remote('origin')
        try:
            remote.fetch()
        except (git.exc.GitCommandError, OSError) as e:
            repo_node.error = f"WARNING: git fetch failed (using cached refs): {e}"

        branches_with_prs: Dict[str, dict] = {}
        local_only_branches: List[dict] = []

        for branch in repo.branches:  # type: ignore[attr-defined]
            branch_name = branch.name
            if branch_name in ['main', 'master']:
                continue

            tracking_branch = branch.tracking_branch()
            has_remote = tracking_branch is not None

            if not has_remote:
                remote_branches = [ref.name for ref in repo.remote().refs]  # type: ignore[attr-defined]
                if f'origin/{branch_name}' in remote_branches:
                    has_remote = True

            sha = branch.commit.hexsha[:7]
            commit_dt = branch.commit.committed_datetime
            commit_msg = str(branch.commit.message or "").strip()
            is_current = (branch_name == current_branch)

            if has_remote:
                branches_with_prs[branch_name] = {
                    'sha': sha,
                    'is_current': is_current,
                    'branch': branch,
                    'commit_time_pt': LocalRepoScanner._format_pt_time(commit_dt),
                    'commit_dt': commit_dt,
                    'commit_message': commit_msg,
                }
            else:
                local_only_branches.append({
                    'name': branch_name,
                    'sha': sha,
                    'is_current': is_current,
                    'commit_time_pt': LocalRepoScanner._format_pt_time(commit_dt),
                    'commit_dt': commit_dt,
                    'commit_message': commit_msg,
                })

        return branches_with_prs, local_only_branches

    def _attach_dynamo_pr_branches(
        self,
        repo_node: LocalRepoNode,
        repo_dir: Path,
        branches_with_prs: Dict[str, dict],
    ) -> None:
        """Fetch PRs for dynamo branches and attach BranchInfoNode + PRStatusWithJobsNode to repo_node."""
        pr_infos_by_branch = self.github_api.get_pr_info_for_branches(
            DYNAMO_OWNER,
            DYNAMO_REPO,
            branches_with_prs.keys(),
            include_closed=True,
            refresh_closed=self.refresh_closed_prs,
        )

        branch_results = []
        for branch_name, info in branches_with_prs.items():
            prs = pr_infos_by_branch.get(branch_name) or []
            if prs:
                branch_results.append((branch_name, info, prs))

        branch_results_sorted = sorted(
            branch_results,
            key=lambda t: (
                (t[1].get("commit_dt") or datetime.min.replace(tzinfo=ZoneInfo("UTC"))),
                str(t[0] or ""),
            ),
            reverse=True,
        )

        if self.max_branches is not None and self.max_branches > 0:
            branch_results_sorted = branch_results_sorted[: int(self.max_branches)]

        allow_fetch_branch_names: Set[str] = set()
        if bool(self.cache_only_github):
            allow_fetch_branch_names = set()
        elif self.max_checks_fetch is not None and self.max_checks_fetch > 0:
            allow_fetch_branch_names = {bn for (bn, _info, _prs) in branch_results_sorted[: int(self.max_checks_fetch)]}

        for branch_name, info, prs in branch_results_sorted:
            commit_url = f"https://github.com/{DYNAMO_REPO_SLUG}/commit/{info['sha']}" if info['sha'] else None
            pr = prs[0] if prs else None

            commit_msg = info.get('commit_message', '')
            if pr:
                commit_msg = str(pr.title or "").strip()
                commit_msg = re.sub(r'^#\d+\s+', '', commit_msg)

            branch_node = BranchInfoNode(
                label=branch_name,
                sha=info['sha'],
                is_current=info['is_current'],
                merged_local=_detect_branch_merged_locally(repo_dir=repo_dir, branch_sha=info.get("sha"), pr=pr),
                commit_url=commit_url,
                commit_time_pt=info.get('commit_time_pt'),
                commit_datetime=info.get('commit_dt'),
                commit_message=commit_msg,
                pr=pr,
            )

            if pr:
                allow_fetch_checks = (branch_name in allow_fetch_branch_names) if allow_fetch_branch_names else True
                if bool(self.cache_only_github):
                    allow_fetch_checks = False

                branch_dt = info.get("commit_dt")
                checks_ttl_s = GitHubAPIClient.compute_checks_cache_ttl_s(
                    branch_dt,
                    refresh=bool(self.refresh_closed_prs),
                    pr_merged=bool(pr.is_merged),
                )

                status_node = PRStatusWithJobsNode(
                    label="",
                    pr=pr,
                    github_api=self.github_api,
                    repo_root=self.repo_root,
                    page_root_dir=self.page_root_dir,
                    refresh_checks=bool(self.refresh_closed_prs),
                    branch_commit_dt=branch_dt,
                    allow_fetch_checks=bool(allow_fetch_checks),
                    context_key=f"{repo_dir.name}:{branch_name}:{info.get('sha','')}",
                    enable_success_build_test_logs=bool(self.enable_success_build_test_logs),
                )
                branch_node.add_child(status_node)

            repo_node.add_child(branch_node)

        # Remote-tracking branches without PRs.
        for branch_name, info in sorted(branches_with_prs.items(), key=lambda kv: kv[0]):
            prs = pr_infos_by_branch.get(branch_name) or []
            if prs:
                continue
            commit_url = f"https://github.com/{DYNAMO_REPO_SLUG}/commit/{info['sha']}" if info.get("sha") else None
            branch_node = BranchInfoNode(
                label=branch_name,
                sha=info.get("sha"),
                is_current=bool(info.get("is_current", False)),
                merged_local=_detect_branch_merged_locally(repo_dir=repo_dir, branch_sha=info.get("sha")),
                commit_url=commit_url,
                commit_time_pt=info.get("commit_time_pt"),
                commit_datetime=info.get("commit_dt"),
                commit_message=info.get("commit_message"),
            )
            repo_node.add_child(branch_node)

    def _scan_repository(self, repo_dir: Path, page_root_dir: Path) -> Optional[LocalRepoNode]:
        """Scan a single repository."""
        repo_name = f"{repo_dir.name}/"
        repo_node = LocalRepoNode(label=repo_name, repo_path=repo_dir)

        try:
            rel_path = os.path.relpath(repo_dir, page_root_dir)
            repo_node.rel_path = rel_path
        except (ValueError, OSError):
            repo_node.rel_path = None

        if Path(repo_dir).is_symlink():
            return repo_node

        if git is None:
            repo_node.error = "GitPython is required. Install with: pip install gitpython"
            return repo_node

        try:
            repo = git.Repo(repo_dir)
        except (git.exc.InvalidGitRepositoryError, git.exc.NoSuchPathError) as e:
            repo_node.error = f"Not a valid git repository: {e}"
            return repo_node

        is_dynamo_repo = False
        remote = repo.remote('origin')
        remote_url = next(remote.urls)
        repo_node.remote_url = remote_url
        is_dynamo_repo = (DYNAMO_REPO_SLUG in remote_url)

        if repo.head.is_detached:
            current_branch = None
        else:
            current_branch = repo.active_branch.name

        head_commit = repo.head.commit
        head_sha = head_commit.hexsha[:7]
        head_commit_dt = head_commit.committed_datetime

        branches_with_prs, local_only_branches = self._scan_and_classify_branches(
            repo, current_branch, repo_node,
        )

        if branches_with_prs and is_dynamo_repo:
            self._attach_dynamo_pr_branches(repo_node, repo_dir, branches_with_prs)

        # Non-dynamo repos: show branches without PR/CI lookup.
        if not is_dynamo_repo:
            combined: list[dict] = []
            for branch_name, info in branches_with_prs.items():
                combined.append({
                    "name": branch_name,
                    "sha": info.get("sha"),
                    "is_current": bool(info.get("is_current")),
                    "commit_time_pt": info.get("commit_time_pt"),
                    "commit_dt": info.get("commit_dt"),
                })
            combined.extend(local_only_branches)
            for b in sorted(combined, key=lambda x: x.get("name", "")):
                repo_node.add_child(
                    BranchInfoNode(
                        label=b.get("name", ""),
                        sha=b.get("sha"),
                        is_current=bool(b.get("is_current", False)),
                        commit_time_pt=b.get("commit_time_pt"),
                        commit_datetime=b.get("commit_dt"),
                    )
                )

        # Local-only branches for dynamo repos.
        if local_only_branches and is_dynamo_repo:
            for branch_info in local_only_branches:
                branch_node = BranchInfoNode(
                    label=branch_info['name'],
                    sha=branch_info['sha'],
                    is_current=branch_info['is_current'],
                    merged_local=_detect_branch_merged_locally(repo_dir=repo_dir, branch_sha=branch_info.get("sha")),
                    commit_time_pt=branch_info.get('commit_time_pt'),
                    commit_datetime=branch_info.get('commit_dt'),
                    commit_message=branch_info.get('commit_message'),
                )
                repo_node.add_child(branch_node)

        # Fallback: add current branch if not already present (common when on main).
        has_current_line = _tree_has_current_branch(repo_node)
        if not has_current_line:
            current_label = current_branch or "HEAD"
            commit_url = f"https://github.com/{DYNAMO_REPO_SLUG}/commit/{head_sha}" if (head_sha and is_dynamo_repo) else None
            repo_node.add_child(
                BranchInfoNode(
                    label=current_label,
                    sha=head_sha,
                    is_current=True,
                    commit_url=commit_url,
                    commit_time_pt=self._format_pt_time(head_commit_dt),
                    commit_datetime=head_commit_dt,
                )
            )

        if not repo_node.children:
            no_branches = BranchNode(label="No branches with PRs or local-only branches")
            repo_node.add_child(no_branches)

        return repo_node


def generate_html(
    root: BranchNode,
    *,
    page_stats: Optional[List[tuple[str, str]]] = None,
    page_title: Optional[str] = None,
    header_title: Optional[str] = None,
    tree_html_override: Optional[str] = None,
    tree_html_alt: Optional[str] = None,
    tree_sort_default: Optional[str] = None,
    tree_sortable: bool = False,
) -> str:
    """Generate HTML output from tree"""
    # Get current time in both UTC and PDT
    now_utc = datetime.now(ZoneInfo('UTC'))
    now_pdt = datetime.now(ZoneInfo('America/Los_Angeles'))

    # Format timestamps
    utc_str = now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')
    pdt_str = now_pdt.strftime('%Y-%m-%d %H:%M:%S %Z')

    tree_html = str(tree_html_override or "").rstrip()
    if not tree_html:
        # Render all children (skip root) into div-based tree (modern collapsible UI)
        tree_vms = [child.to_tree_vm() for child in root.children]
        tree_html = render_tree_divs(tree_vms)
    else:
        tree_html = tree_html + ("\n" if not tree_html.endswith("\n") else "")

    env = Environment(
        loader=FileSystemLoader(str(_THIS_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("show_local_branches.j2")
    return template.render(
        generated_time=pdt_str,
        page_title=str(page_title) if page_title is not None else None,
        header_title=str(header_title) if header_title is not None else None,
        copy_icon_svg=_COPY_ICON_SVG,
        tree_html=tree_html,
        tree_html_alt=(str(tree_html_alt) if tree_html_alt is not None else ""),
        tree_sort_default=(str(tree_sort_default) if tree_sort_default is not None else ""),
        tree_sortable=bool(tree_sortable),
        page_stats=page_stats,
        success_icon_html=status_icon_html(status_norm="success", is_required=False),
        success_required_icon_html=status_icon_html(status_norm="success", is_required=True),
        failure_required_icon_html=status_icon_html(status_norm="failure", is_required=True),
        failure_optional_icon_html=status_icon_html(status_norm="failure", is_required=False),
        in_progress_icon_html=status_icon_html(status_norm="in_progress", is_required=False),
        pending_icon_html=status_icon_html(status_norm="pending", is_required=False),
        cancelled_icon_html=status_icon_html(status_norm="cancelled", is_required=False),
        skipped_icon_html=status_icon_html(status_norm="skipped", is_required=False),
    )
def main():
    phase_t = PhaseTimer()

    parser = argparse.ArgumentParser(
        description='Show local branches with PR information (HTML-only)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  GH_TOKEN / GITHUB_TOKEN
      GitHub personal access token (alternative to --github-token)
      Priority: --github-token > GH_TOKEN > GITHUB_TOKEN > ~/.config/gh/hosts.yml

  DYNAMO_UTILS_CACHE_DIR
      Override default cache directory (~/.cache/dynamo-utils)

  DYNAMO_UTILS_DISABLE_INTERNAL_RATE_LIMIT_CHECKS
      Set to 1 to disable internal /rate_limit API calls (for quota experiments)

  MAX_GITHUB_API_CALLS
      Can be set when using update_html_pages.sh to override --max-github-api-calls default
        """
    )
    # Keep backward compatibility with the historical positional `base_dir`, but prefer --repo-path.
    parser.add_argument(
        'base_dir',
        type=Path,
        nargs='?',
        default=None,
        help='[deprecated] Base directory to search for git repos (direct children only). Prefer --repo-path.'
    )
    parser.add_argument(
        '--repo-path',
        type=Path,
        default=None,
        help='Path to the base directory to scan (direct children only) (default: current directory)'
    )
    parser.add_argument(
        '--repo-root',
        type=Path,
        default=None,
        help='Path to a local clone of the repo for workflow YAML parsing (optional; auto-detect under --repo-path if omitted)'
    )
    parser.add_argument(
        '--github-token',
        help='GitHub personal access token (preferred). If omitted, we try ~/.config/github-token or ~/.config/gh/hosts.yml.'
    )
    parser.add_argument(
        '--allow-anonymous-github',
        action='store_true',
        help='Allow anonymous GitHub REST calls (60/hr core rate limit). By default we require auth to avoid rate limiting.'
    )
    parser.add_argument(
        '--refresh-closed-prs',
        action='store_true',
        help='Refresh cached closed/merged PR mappings (more GitHub API calls)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output HTML file path (default: speedoflight/dynamo/users/<user>/local.html)'
    )
    parser.add_argument(
        '--max-branches',
        type=int,
        default=None,
        help='If set, cap the number of branches-with-PRs shown per repo (newest first)'
    )
    parser.add_argument(
        '--max-checks-fetch',
        type=int,
        default=None,
        help='If set, only allow network fetch of checks/CI hierarchy for the top N newest branches-with-PRs (others are cache-only)'
    )
    parser.add_argument(
        '--max-github-api-calls',
        type=int,
        default=100,
        help='Hard cap on GitHub REST API network calls per invocation (cached reads do not count). Default: 100.'
    )
    parser.add_argument(
        '--enable-success-build-test-logs',
        action='store_true',
        help='Opt-in: cache raw logs for successful *-build-test jobs so we can parse pytest slowest tests under "Run tests" (slower).'
    )
    args = parser.parse_args()

    base_dir = (args.repo_path or args.base_dir or Path.cwd()).resolve()
    out_path = (Path(args.output).resolve() if args.output is not None else (Path.home() / "dynamo/speedoflight/dynamo/users/keivenchang/local.html"))
    page_root_dir = out_path.parent.resolve()

    repo_root = Path(args.repo_root).resolve() if args.repo_root is not None else None
    if repo_root is None:
        repo_root = find_local_clone_of_repo(base_dir, repo_slug=f"{DYNAMO_OWNER}/{DYNAMO_REPO}")
    if repo_root is None:
        # Best-effort fallback: YAML parsing will degrade gracefully.
        repo_root = base_dir

    # Prune locally-served raw logs to avoid unbounded growth and delete any partial/unverified artifacts.
    # We only render `[raw log]` links when the local file exists (or was materialized),
    # so pruning won't produce dead links on a freshly generated page.
    with phase_t.phase("prune"):
        _ = prune_dashboard_raw_logs(page_root_dir=page_root_dir, max_age_days=90)
        _ = prune_partial_raw_log_caches(page_root_dirs=[page_root_dir])

    # Scan repositories
    scanner = LocalRepoScanner(
        token=args.github_token,
        refresh_closed_prs=bool(args.refresh_closed_prs),
        max_branches=args.max_branches,
        max_checks_fetch=args.max_checks_fetch,
        allow_anonymous_github=bool(args.allow_anonymous_github),
        max_github_api_calls=int(args.max_github_api_calls),
        enable_success_build_test_logs=bool(args.enable_success_build_test_logs),
        repo_root=repo_root,
        page_root_dir=page_root_dir,
    )
    # Cache-only fallback if exhausted.
    try:
        scanner.github_api.check_core_rate_limit_or_raise()
    except RuntimeError as e:
        # Switch to cache-only mode (no new network calls; use existing caches).
        logger.warning(f"GitHub rate limit exceeded, switching to cache-only mode: {e}")
        scanner.cache_only_github = True
        scanner.github_api.set_cache_only_mode(True)
        # Also disable refresh knobs in cache-only mode.
        scanner.refresh_closed_prs = False
        # Record reason (displayed in Statistics section).
        cache_only_reason = str(e)
    else:
        cache_only_reason = ""

    generation_t0 = time.monotonic()
    root = None
    with phase_t.phase("scan"):
        root = scanner.scan_repositories(base_dir)

    # Output (HTML-only)
    assert root is not None
    # Page statistics (shown in an expandable block at the bottom of the HTML).
    #
    # Note: we want the HTML page to show a *breakdown* (prune/scan/render/write/total .total_secs).
    # That means we need to measure render/write first, then re-render once so the stats reflect
    # those timings. This is intentionally low-tech and stable.
    elapsed_s = max(0.0, time.monotonic() - generation_t0)

    pr_count = 0

    # New structure: root -> LocalUserNode -> LocalRepoNode -> ...
    maybe_user = (root.children or [None])[0]
    repos_scanned = len(maybe_user.children or []) if maybe_user is not None else 0

    # Force tree conversion to TreeNodeVM to trigger snippet extraction and cache stats population.
    # This must happen BEFORE build_page_stats() so statistics are captured correctly.
    _ = root.to_tree_vm()

    # Helper to update stats after phase timing measurement
    def _upsert_stat(page_stats: List[tuple[str, Optional[str], str]], k: str, v: str, desc: str = "") -> None:
        for i, stat in enumerate(page_stats):
            kk = stat[0]
            if kk == k:
                # Preserve existing description if not provided
                existing_desc = stat[2] if len(stat) > 2 else ""
                page_stats[i] = (k, v, desc or existing_desc)
                return
        page_stats.append((k, v, desc))

    # Initial stats using shared function (now includes snippet extraction stats)
    page_stats = build_page_stats(
        generation_time_secs=elapsed_s,
        github_api=scanner.github_api,
        max_github_api_calls=int(args.max_github_api_calls),
        cache_only_mode=bool(scanner.cache_only_github),
        cache_only_reason=cache_only_reason or "",
        repos_scanned=repos_scanned,
        prs_shown=pr_count,
        max_branches=args.max_branches,
        max_checks_fetch=args.max_checks_fetch,
        refresh_closed_prs=bool(args.refresh_closed_prs),
    )

    # Render once to measure render time.
    with phase_t.phase("render"):
        _ = generate_html(
            root,
            page_stats=page_stats,
            page_title=f'<span style="color: #fbbf24;">Augmented</span> Local Pull Requests [{base_dir}]',
            header_title=f'<span style="color: #fbbf24;">Augmented</span> Local Pull Requests [{base_dir}]',
        )

    # Update the page stats with the timing breakdown before producing the final HTML.
    tdict = phase_t.as_dict(include_total=True)
    # Make "generation.total_secs" reflect total wall time so it matches the timing breakdown.
    elapsed_total = float(tdict.get("total") or 0.0)
    _upsert_stat(page_stats, "generation.total_secs", f"{elapsed_total:.2f}s")
    for k in ["prune", "scan", "render", "write", "total"]:
        if k in tdict:
            _upsert_stat(page_stats, f"{k}.total_secs", f"{float(tdict.get(k) or 0.0):.2f}s")

    # Final render + atomic write to destination (single visible update).
    out_path = args.output
    if out_path is None:
        out_path = Path.home() / "dynamo/speedoflight/dynamo/users/keivenchang/local.html"
    with phase_t.phase("render_final"):
        html_output2 = generate_html(
            root,
            page_stats=page_stats,
            page_title=f'<span style="color: #fbbf24;">Augmented</span> Local Pull Requests [{base_dir}]',
            header_title=f'<span style="color: #fbbf24;">Augmented</span> Local Pull Requests [{base_dir}]',
        )
    with phase_t.phase("write"):
        atomic_write_text(out_path, html_output2, encoding="utf-8")

    # Flush shared caches to disk (snippet cache, pytest timings cache, etc.)
    SNIPPET_CACHE.flush()

    # No stdout/stderr run-stats; the HTML Statistics section contains the breakdowns.


if __name__ == '__main__':
    main()
