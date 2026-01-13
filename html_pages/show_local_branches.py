#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Show dynamo branches with PR information using a node-based tree structure.
Supports parallel data gathering for improved performance.

Structure:
- RepoNode (local directory) → BranchInfoNode (branch) → CommitMessageNode, MetadataNode,
  PRNode, PRStatusNode (PASSED/FAILED pill) → CIJobNode (CI jobs with hierarchy)

IMPORTANT: Architecture Rule
---------------------------
⚠️ show_local_branches.py and show_remote_branches.py should NEVER import from each other!
   - ALL shared code lives in: common_branch_nodes.py, common_dashboard_lib.py, common.py
   - ✅ REFACTORED: PRStatusNode and _build_ci_hierarchy_nodes now in common_branch_nodes.py

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

NOTE: The actual implementation of PRStatusNode, _build_ci_hierarchy_nodes, and related
classes/functions are in common_branch_nodes.py. This file only imports them.
"""

import argparse
import hashlib
import html
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
import functools
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

# Ensure we can import sibling utilities (common.py) from the parent dynamo-utils directory
_THIS_DIR = Path(__file__).resolve().parent
_UTILS_DIR = _THIS_DIR.parent
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

try:
    import git  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    git = None  # type: ignore[assignment]

# Shared dashboard helpers (UI + workflow graph)
from common_dashboard_lib import (
    CIStatus,
    EXPECTED_CHECK_PLACEHOLDER_SYMBOL,
    PASS_PLUS_STYLE,
    TreeNodeVM,
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

# Log/snippet helpers (shared library: `dynamo-utils/ci_log_errors/`)
from ci_log_errors import extract_error_snippet_from_log_file

# Jinja2 is optional (keep CLI usable in minimal envs).
try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    HAS_JINJA2 = True
except Exception:  # pragma: no cover
    HAS_JINJA2 = False
    Environment = None  # type: ignore[assignment]
    FileSystemLoader = None  # type: ignore[assignment]
    select_autoescape = None  # type: ignore[assignment]

# Import GitHub utilities from common module
import common
from common import (
    FailedCheck,
    GHPRCheckRow,
    GitHubAPIClient,
    PhaseTimer,
    PRInfo,
    classify_ci_kind,
    dynamo_utils_cache_dir,
    summarize_pr_check_rows,
)

# Import shared node classes and refactored functions from common_branch_nodes
from common_branch_nodes import (
    BlockedMessageNode,
    BranchInfoNode,
    BranchNode,
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
    _is_known_required_check,
    generate_html,
)
from dataclasses import dataclass  # noqa: E402
import html as html_module  # noqa: E402


# Local-specific RepoNode subclass with path and error tracking
@dataclass
class LocalRepoNode(RepoNode):
    """Repository node for local branches (extends shared RepoNode)."""
    repo_path: Optional[str] = None
    error: Optional[str] = None
    remote_url: Optional[str] = None
    
    def __init__(self, label: str, *, repo_path: Optional[Path] = None, **kwargs):
        super().__init__(label=label, **kwargs)
        self.repo_path = str(repo_path) if repo_path is not None else None
        self.error = None
        self.remote_url = None
    
    def _format_html_content(self) -> str:
        """Format repo node with symlink target if applicable."""
        # If the repository directory itself is a symlink, show the target for clarity.
        link_suffix = ""
        try:
            p = Path(self.repo_path) if self.repo_path is not None else None
            if p is not None and p.is_symlink():
                tgt = str(p.readlink().resolve())
                link_suffix = f' <span style="color: #666; font-size: 11px;">→ {html_module.escape(tgt)}</span>'
        except Exception:
            pass
        
        label_html = html_module.escape(self.label)
        if self.error:
            error_html = f' <span style="color: #c83a3a; font-size: 11px;">(Error: {html_module.escape(self.error)})</span>'
            return f'<span style="font-weight: 600;">{label_html}</span>{link_suffix}{error_html}'
        return f'<span style="font-weight: 600;">{label_html}</span>{link_suffix}'
    
    def to_tree_vm(self) -> TreeNodeVM:
        """Convert local repo node to TreeNodeVM."""
        kids = [c.to_tree_vm() for c in (self.children or [])]
        return TreeNodeVM(
            node_key=f"repo:{self.label}",
            label_html=self._format_html_content(),
            children=kids,
            collapsible=bool(kids),
            default_expanded=True,
        )


def _tree_has_current_branch(node: BranchNode) -> bool:
    """Return True if any BranchInfoNode in this subtree is marked as current.

    This prevents rendering the current branch twice (once inside a section like "Branches"
    and again as a top-level fallback line).
    """
    if isinstance(node, BranchInfoNode) and bool(getattr(node, "is_current", False)):
        return True
    for child in getattr(node, "children", []) or []:
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



@functools.lru_cache(maxsize=1)
def _copy_icon_svg(*, size_px: int = 12) -> str:
    """Return the shared 'copy' icon SVG (2-squares), sourced from copy_icon_paths.svg."""
    try:
        p = (Path(__file__).resolve().parent / "copy_icon_paths.svg").resolve()
        paths = p.read_text(encoding="utf-8").strip()
    except Exception:
        paths = ""
    return (
        f'<svg width="{int(size_px)}" height="{int(size_px)}" viewBox="0 0 16 16" fill="currentColor" '
        f'style="display: inline-block; vertical-align: middle;">{paths}</svg>'
    )


# Keep a module-level constant for existing call sites / template rendering.
_COPY_ICON_SVG = _copy_icon_svg(size_px=12)


def _format_epoch_pt(epoch_s: Optional[int]) -> Optional[str]:
    """Format an epoch as 'YYYY-mm-dd HH:MM:SS PT'."""
    if epoch_s is None:
        return None
    try:
        dt = datetime.fromtimestamp(int(epoch_s), tz=timezone.utc).astimezone(ZoneInfo("America/Los_Angeles"))
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return None


def _format_seconds_delta_short(seconds: Optional[int]) -> Optional[str]:
    if seconds is None:
        return None
    try:
        return GitHubAPIClient._format_seconds_delta(int(seconds))
    except Exception:
        return None


def _parse_rate_limit_resources(rate_limit_payload: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    """Parse /rate_limit payload into a stable dict: {resource_name: {limit, remaining, used, reset_epoch,...}}."""
    out: Dict[str, Dict[str, object]] = {}
    resources = (rate_limit_payload or {}).get("resources")  # type: ignore[assignment]
    if not isinstance(resources, dict):
        return out
    now = int(time.time())
    for name, info in resources.items():
        if not isinstance(name, str) or not isinstance(info, dict):
            continue
        try:
            limit = info.get("limit")
            remaining = info.get("remaining")
            used = info.get("used")
            reset_epoch = info.get("reset")
            limit_i = int(limit) if limit is not None else None
            remaining_i = int(remaining) if remaining is not None else None
            used_i = int(used) if used is not None else None
            reset_i = int(reset_epoch) if reset_epoch is not None else None
        except Exception:
            continue
        seconds_until = (reset_i - now) if reset_i is not None else None
        out[name] = {
            "limit": limit_i,
            "remaining": remaining_i,
            "used": used_i,
            "reset_epoch": reset_i,
            "reset_pt": _format_epoch_pt(reset_i),
            "seconds_until_reset": seconds_until,
            "until_reset": _format_seconds_delta_short(seconds_until),
        }
    return out


def _rate_limit_history_path() -> Path:
    # Keep this under the shared dynamo-utils cache dir so it persists across runs (host/container).
    return dynamo_utils_cache_dir() / "dashboards" / "show_local_branches" / "gh_rate_limit_history.jsonl"


def _load_rate_limit_history(path: Path, *, max_points: int = 288) -> List[Dict[str, object]]:
    """Load jsonl history (best-effort). Keeps at most max_points newest entries."""
    try:
        if not path.exists():
            return []
        lines = path.read_text().splitlines()
        out: List[Dict[str, object]] = []
        for line in lines[-max_points:]:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                continue
        return out[-max_points:]
    except Exception:
        return []


def _append_rate_limit_history(path: Path, sample: Dict[str, object], *, max_points: int = 288) -> List[Dict[str, object]]:
    """Append a single sample to history and compact to max_points (best-effort). Returns updated history."""
    hist = _load_rate_limit_history(path, max_points=max_points)
    hist.append(sample)
    hist = hist[-max_points:]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(path) + ".tmp"
        with open(tmp, "w") as f:
            for row in hist:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        os.replace(tmp, path)
    except Exception:
        pass
    return hist


def _sparkline_svg_for_percent(points: List[float], *, width: int = 180, height: int = 34) -> str:
    """Render a tiny inline SVG sparkline for a [0..1] percent series."""
    if not points:
        return ""
    # Clamp + keep only finite-ish.
    series = []
    for p in points:
        try:
            v = float(p)
            if v != v:  # NaN
                continue
            series.append(max(0.0, min(1.0, v)))
        except Exception:
            continue
    if len(series) < 2:
        return ""

    pad = 2
    w = max(int(width), 60)
    h = max(int(height), 20)
    x_step = (w - 2 * pad) / float(max(1, len(series) - 1))

    pts = []
    for i, v in enumerate(series):
        x = pad + i * x_step
        # invert y (1.0 at top)
        y = pad + (1.0 - v) * (h - 2 * pad)
        pts.append(f"{x:.2f},{y:.2f}")
    pts_str = " ".join(pts)
    return (
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        f'xmlns="http://www.w3.org/2000/svg" style="vertical-align: middle;">'
        f'<rect x="0" y="0" width="{w}" height="{h}" rx="4" fill="#ffffff" stroke="#d0d7de" />'
        f'<polyline fill="none" stroke="#0969da" stroke-width="1.5" points="{pts_str}" />'
        f"</svg>"
    )


def _html_copy_button(*, clipboard_text: str, title: str) -> str:
    """Return a show_commit_history-style copy button that calls copyFromClipboardAttr(this)."""
    text_escaped = html.escape(clipboard_text, quote=True)
    title_escaped = html.escape(title, quote=True)
    return (
        f'<button data-clipboard-text="{text_escaped}" onclick="copyFromClipboardAttr(this)" '
        f'style="{_COPY_BTN_STYLE}" '
        f'title="{title_escaped}" '
        f'onmouseover="this.style.backgroundColor=\'#f3f4f6\'; this.style.borderColor=\'#8c959f\';" '
        f'onmouseout="this.style.backgroundColor=\'transparent\'; this.style.borderColor=\'#d0d7de\';">'
        f"{_COPY_ICON_SVG}"
        f"</button>"
    )


def _html_small_link(*, url: str, label: str) -> str:
    url_escaped = html.escape(url, quote=True)
    label_escaped = html.escape(label)
    return (
        f' <a href="{url_escaped}" target="_blank" '
        f'style="color: #0969da; font-size: 11px; margin-left: 5px; text-decoration: none;">{label_escaped}</a>'
    )

def _format_utc_datetime_from_iso(iso_utc: str) -> Optional[str]:
    """Format a GitHub ISO timestamp (UTC) as UTC time string (YYYY-MM-DD HH:MM)."""
    try:
        dt = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
        # Ensure tz-aware
        if getattr(dt, "tzinfo", None) is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        dt_utc = dt.astimezone(ZoneInfo("UTC"))
        return dt_utc.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return None


#
# GitHub CI hierarchy helpers (workflow needs graph + PR checks)
#


def _aggregate_status(statuses: Iterable[str]) -> str:
    priority = {"failure": 5, "cancelled": 4, "in_progress": 3, "pending": 2, "success": 1, "unknown": 0}
    best = "unknown"
    best_p = -1
    for s in statuses:
        p = priority.get(s, 0)
        if p > best_p:
            best_p = p
            best = s
    return best










def looks_like_git_repo_dir(p: Path) -> bool:
    """Lightweight git repo detection without invoking GitPython.
    
    Supports normal repos ('.git' directory) and worktrees/submodules ('.git' file).
    """
    try:
        if not p.is_dir():
            return False
        git_marker = p / ".git"
        return git_marker.is_dir() or git_marker.is_file()
    except Exception:
        return False


def gitdir_from_git_file(p: Path) -> Optional[Path]:
    """Handle worktrees where .git is a file containing 'gitdir: <path>'."""
    try:
        txt = (p / ".git").read_text(encoding="utf-8", errors="ignore").strip()
        if not txt.startswith("gitdir:"):
            return None
        rest = txt.split("gitdir:", 1)[1].strip()
        if not rest:
            return None
        gd = Path(rest)
        if not gd.is_absolute():
            gd = (p / gd).resolve()
        return gd if gd.is_dir() else None
    except Exception:
        return None


def origin_url_from_git_config(repo_dir: Path) -> str:
    """Extract origin URL from .git/config without loading GitPython."""
    try:
        git_dir = repo_dir / ".git"
        if git_dir.is_file():
            gd = gitdir_from_git_file(repo_dir)
            if gd:
                git_dir = gd
        config = git_dir / "config"
        if not config.is_file():
            return ""
        txt = config.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r'\[remote\s+"origin"\].*?url\s*=\s*(.+?)(?:\n|\r\n|\r|$)', txt, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
        return ""
    except Exception:
        return ""


def find_local_clone_of_repo(base_dir: Path, *, repo_slug: str) -> Optional[Path]:
    """Find a local clone of a specific repo (e.g. 'ai-dynamo/dynamo') under base_dir.
    
    Returns the first matching repo directory, or None if not found.
    """
    if not base_dir.is_dir():
        return None
    for d in base_dir.iterdir():
        if not looks_like_git_repo_dir(d):
            continue
        url = origin_url_from_git_config(d)
        if repo_slug in url:
            return d
    return None


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

    @staticmethod
    def _is_world_readable_executable_dir(p: Path) -> bool:
        """
        True iff `p` is a directory with world-readable + world-executable permissions (o+r and o+x).

        For directories, readable enables listing and executable enables traversal.
        """
        try:
            st = p.stat()
        except Exception:
            return False
        mode = st.st_mode
        return bool(mode & stat.S_IROTH) and bool(mode & stat.S_IXOTH)

    @staticmethod
    def _looks_like_git_repo_dir(p: Path) -> bool:
        """Lightweight git repo detection (delegates to module-level function)."""
        return looks_like_git_repo_dir(p)

    def scan_repositories(self, base_dir: Path) -> BranchNode:
        """Scan all git repositories under `base_dir` (direct children only) and build tree structure."""
        root = BranchNode(label="")

        # Discover git repos among direct children (and include base_dir itself if it's a repo).
        # We intentionally do NOT walk the whole tree because this workspace can be huge.
        candidate_dirs: list[Path] = []
        if self._looks_like_git_repo_dir(base_dir) and self._is_world_readable_executable_dir(base_dir):
            candidate_dirs.append(base_dir)

        for d in base_dir.iterdir():
            if not d.is_dir():
                continue
            if not self._is_world_readable_executable_dir(d):
                continue
            if not self._looks_like_git_repo_dir(d):
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
                    root.add_child(repo_node)

        # Sort children by name
        root.children.sort(key=lambda n: n.label)

        return root

    def _scan_repository(self, repo_dir: Path, page_root_dir: Path) -> Optional[LocalRepoNode]:
        """Scan a single repository"""
        repo_name = f"{repo_dir.name}/"
        repo_node = LocalRepoNode(label=repo_name, repo_path=repo_dir)

        # <pre>Symlink repos: show repo line (with → target) but don't scan/render nested info
        # (branches/PR/CI). Render as non-expandable in the UI.</pre>
        try:
            if Path(repo_dir).is_symlink():
                return repo_node
        except Exception:
            pass

        if git is None:
            repo_node.error = "GitPython is required. Install with: pip install gitpython"
            return repo_node

        try:
            repo = git.Repo(repo_dir)
        except Exception as e:
            repo_node.error = f"Not a valid git repository: {e}"
            return repo_node

        # <pre>Capture origin URL. We no longer treat "non-dynamo" repos as an error;
        # just skip PR/CI lookups wired to ai-dynamo/dynamo.</pre>
        is_dynamo_repo = False
        try:
            remote = repo.remote('origin')
            remote_url = next(remote.urls)
            repo_node.remote_url = remote_url

            is_dynamo_repo = (DYNAMO_REPO_SLUG in remote_url)
        except Exception:
            repo_node.error = "No origin remote found"
            return repo_node

        # Get current branch
        try:
            current_branch = repo.active_branch.name
        except Exception:
            current_branch = None

        # Always capture current HEAD SHA (for display even when we skip "main" branches, or when detached).
        try:
            head_commit = repo.head.commit
            head_sha = head_commit.hexsha[:7]
            head_commit_dt = getattr(head_commit, "committed_datetime", None)
        except Exception:
            head_sha = None
            head_commit_dt = None

        def _format_pt_time(dt) -> Optional[str]:
            try:
                if dt is None:
                    return None
                # GitPython often returns tz-aware datetimes; if not, assume UTC.
                if getattr(dt, "tzinfo", None) is None:
                    dt = dt.replace(tzinfo=ZoneInfo("UTC"))
                pt = dt.astimezone(ZoneInfo("America/Los_Angeles"))
                return f"{pt.strftime('%Y-%m-%d %H:%M')} PT"
            except Exception:
                return None

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
        for branch in repo.branches:  # type: ignore[attr-defined]
            branch_name = branch.name

            # Skip main branches
            if branch_name in ['main', 'master']:
                continue

            # Check if branch has remote tracking or matching remote branch
            try:
                tracking_branch = branch.tracking_branch()
                has_remote = tracking_branch is not None
            except Exception:
                has_remote = False

            # Also check if a remote branch exists with the same name (even if not tracking)
            if not has_remote:
                try:
                    remote_branches = [ref.name for ref in repo.remote().refs]  # type: ignore[attr-defined]
                    # Check for origin/branch_name
                    if f'origin/{branch_name}' in remote_branches:
                        has_remote = True
                except Exception:
                    pass

            # Get commit SHA
            try:
                sha = branch.commit.hexsha[:7]
                commit_dt = getattr(branch.commit, "committed_datetime", None)
                commit_msg = str(getattr(branch.commit, "message", "") or "").strip()
            except Exception:
                sha = None
                commit_dt = None
                commit_msg = ""

            is_current = (branch_name == current_branch)

            if has_remote:
                # Get PR info in parallel (will be gathered later)
                branches_with_prs[branch_name] = {
                    'sha': sha,
                    'is_current': is_current,
                    'branch': branch,
                    'commit_time_pt': _format_pt_time(commit_dt),
                    'commit_dt': commit_dt,
                    'commit_message': commit_msg,
                }
            else:
                local_only_branches.append({
                    'name': branch_name,
                    'sha': sha,
                    'is_current': is_current,
                    'commit_time_pt': _format_pt_time(commit_dt),
                    'commit_dt': commit_dt,
                    'commit_message': commit_msg,
                })

        # Fetch PR information in parallel
        if branches_with_prs and is_dynamo_repo:
            # Branches with PRs - add directly to repo_node (no section wrapper)
            # NOTE: Avoid per-branch GitHub API calls. We list open PRs once per repo (cached),
            # then match PRs to branch names locally.
            try:
                pr_infos_by_branch = self.github_api.get_pr_info_for_branches(
                    DYNAMO_OWNER,
                    DYNAMO_REPO,
                    branches_with_prs.keys(),
                    include_closed=True,
                    refresh_closed=self.refresh_closed_prs,
                )
            except Exception as e:
                self.logger.warning("Error fetching PR info: %s", e)
                pr_infos_by_branch = {}

            branch_results = []
            for branch_name, info in branches_with_prs.items():
                prs = pr_infos_by_branch.get(branch_name) or []
                if prs:
                    branch_results.append((branch_name, info, prs))

            # Small-mode caps:
            # - Display at most N branches with PRs (choose most-recent by commit_dt).
            # - Only allow network fetch of checks/CI hierarchy for the top K (most recent) branches.
            #
            # This keeps the page fast and keeps GitHub/gh calls bounded.
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
            if bool(getattr(self, "cache_only_github", False)):
                allow_fetch_branch_names = set()
            elif self.max_checks_fetch is not None and self.max_checks_fetch > 0:
                allow_fetch_branch_names = {bn for (bn, _info, _prs) in branch_results_sorted[: int(self.max_checks_fetch)]}

            # Build branch nodes (newest first)
            for branch_name, info, prs in branch_results_sorted:
                commit_url = f"https://github.com/{DYNAMO_REPO_SLUG}/commit/{info['sha']}" if info['sha'] else None
                
                # Get the PR (should be exactly one per branch in this structure)
                pr = prs[0] if prs else None
                
                # Use PR title as commit message if available (like show_remote_branches.py)
                commit_msg = info.get('commit_message', '')
                if pr:
                    commit_msg = str(getattr(pr, "title", "") or "").strip()
                    # Remove leading "#1234 " pattern if present
                    import re
                    commit_msg = re.sub(r'^#\d+\s+', '', commit_msg)
                
                branch_node = BranchInfoNode(
                    label=branch_name,
                    sha=info['sha'],
                    is_current=info['is_current'],
                    commit_url=commit_url,
                    commit_time_pt=info.get('commit_time_pt'),
                    commit_datetime=info.get('commit_dt'),
                    commit_message=commit_msg,
                    pr=pr,  # Pass PR so commit message child shows (#PR) link
                )

                # Add PR status node if PR exists
                if pr:
                    allow_fetch_checks = (branch_name in allow_fetch_branch_names) if allow_fetch_branch_names else True
                    if bool(getattr(self, "cache_only_github", False)):
                        allow_fetch_checks = False
                    
                    pr_state_lc = (getattr(pr, "state", "") or "").lower()
                    # Prefer the branch head commit time for "last push" heuristic.
                    branch_dt = info.get("commit_dt")
                    checks_ttl_s = GitHubAPIClient.compute_checks_cache_ttl_s(
                        branch_dt,
                        refresh=bool(self.refresh_closed_prs),
                    )
                    
                    # Create status node (will contain CI children)
                    status_node = PRStatusNode(
                        label="",
                        pr=pr,
                        github_api=self.github_api,
                        refresh_checks=bool(self.refresh_closed_prs),
                        branch_commit_dt=branch_dt,
                        allow_fetch_checks=bool(allow_fetch_checks),
                        context_key=f"{repo_dir.name}:{branch_name}:{info.get('sha','')}",
                    )

                    branch_node.add_child(status_node)

                    # CI nodes will be built later in the pipeline passes (run_all_passes)
                    # No longer building CI hierarchy here - moved to common_dashboard_lib.py passes

                    # Add conflict warning if applicable
                    if pr.conflict_message:
                        conflict_node = ConflictWarningNode(label=pr.conflict_message)
                        status_node.add_child(conflict_node)

                    # Add blocking message if applicable
                    if pr.blocking_message:
                        blocked_node = BlockedMessageNode(label=pr.blocking_message)
                        status_node.add_child(blocked_node)

                    # With CI hierarchy embedded, we no longer render separate flat running/failed check lines here.

                repo_node.add_child(branch_node)

            # Branches without PRs - add directly to repo_node (no section wrapper)
            #
            # These are remote-tracking branches that don't have active PRs.
            # Just show the branch line; no CI hierarchy (since there's no PR to get checks from).
            for branch_name, info in sorted(branches_with_prs.items(), key=lambda kv: kv[0]):
                prs = pr_infos_by_branch.get(branch_name) or []
                if prs:
                    continue
                commit_url = f"https://github.com/{DYNAMO_REPO_SLUG}/commit/{info['sha']}" if info.get("sha") else None
                branch_node = BranchInfoNode(
                    label=branch_name,
                    sha=info.get("sha"),
                    is_current=bool(info.get("is_current", False)),
                    commit_url=commit_url,
                    commit_time_pt=info.get("commit_time_pt"),
                    commit_datetime=info.get("commit_dt"),
                    commit_message=info.get("commit_message"),
                )
                
                repo_node.add_child(branch_node)

        # For non-dynamo repos: show branches but skip PR/CI lookup (treat as "any other repo").
        # Add directly to repo_node (no section wrapper)
        if not is_dynamo_repo:

            combined = []
            for branch_name, info in branches_with_prs.items():
                combined.append(
                    {
                        "name": branch_name,
                        "sha": info.get("sha"),
                        "is_current": bool(info.get("is_current")),
                        "commit_time_pt": info.get("commit_time_pt"),
                        "commit_dt": info.get("commit_dt"),
                    }
                )
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

        # Add local-only branches - add directly to repo_node (no section wrapper)
        if local_only_branches and is_dynamo_repo:

            for branch_info in local_only_branches:
                branch_node = BranchInfoNode(
                    label=branch_info['name'],
                    sha=branch_info['sha'],
                    is_current=branch_info['is_current'],
                    commit_time_pt=branch_info.get('commit_time_pt'),
                    commit_datetime=branch_info.get('commit_dt'),
                    commit_message=branch_info.get('commit_message'),
                )
                repo_node.add_child(branch_node)

        # If the current checkout didn't show up in the PR/local sections (common when on main),
        # add a single line for it so repos like dynamo_latest/ and dynamo_ci/ are informative.
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
                    commit_time_pt=_format_pt_time(head_commit_dt),
                    commit_datetime=head_commit_dt,
                )
            )

        # Add "no branches" message if needed
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

    if not HAS_JINJA2:
        raise RuntimeError(
            "Jinja2 is required for HTML output. Install with: pip install jinja2"
        )

    # Help type-checkers: these are set only when HAS_JINJA2 is True.
    assert Environment is not None
    assert FileSystemLoader is not None
    assert select_autoescape is not None

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
        description='Show local branches with PR information (HTML-only)'
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
        '--token',
        help='GitHub personal access token (or set GH_TOKEN/GITHUB_TOKEN env var)'
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
        help='Output HTML file path (default: <repo-path>/index.html)'
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
    args = parser.parse_args()

    base_dir = (args.repo_path or args.base_dir or Path.cwd()).resolve()

    # Prune locally-served raw logs to avoid unbounded growth and delete any partial/unverified artifacts.
    # We only render `[raw log]` links when the local file exists (or was materialized),
    # so pruning won't produce dead links on a freshly generated page.
    try:
        with phase_t.phase("prune"):
            _ = prune_dashboard_raw_logs(page_root_dir=base_dir, max_age_days=30)
            _ = prune_partial_raw_log_caches(page_root_dirs=[base_dir])
    except Exception:
        pass

    # Scan repositories
    scanner = LocalRepoScanner(
        token=args.token,
        refresh_closed_prs=bool(args.refresh_closed_prs),
        max_branches=args.max_branches,
        max_checks_fetch=args.max_checks_fetch,
        allow_anonymous_github=bool(args.allow_anonymous_github),
        max_github_api_calls=int(args.max_github_api_calls),
    )
    # Cache-only fallback if exhausted.
    try:
        scanner.github_api.check_core_rate_limit_or_raise()
    except Exception as e:
        # Switch to cache-only mode (no new network calls; use existing caches).
        scanner.cache_only_github = True
        try:
            scanner.github_api.set_cache_only_mode(True)
        except Exception:
            pass
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
    rest_calls = 0
    rest_ok = 0
    rest_err = 0
    rest_err_by_status_s = ""
    try:
        stats = scanner.github_api.get_rest_call_stats() if scanner.github_api else {}
        rest_calls = int(stats.get("total") or 0)
        rest_ok = int(stats.get("success_total") or 0)
        rest_err = int(stats.get("error_total") or 0)
    except Exception:
        rest_calls = 0
        rest_ok = 0
        rest_err = 0
        stats = {}

    try:
        es = scanner.github_api.get_rest_error_stats() if scanner.github_api else {}
        by_status = (es or {}).get("by_status") if isinstance(es, dict) else {}
        if isinstance(by_status, dict) and by_status:
            items = list(by_status.items())[:8]
            rest_err_by_status_s = ", ".join([f"{k}={v}" for k, v in items])
    except Exception:
        rest_err_by_status_s = ""

    pr_count = 0
    try:
        def _count_prs(node: BranchNode) -> int:
            n = 0
            if isinstance(node, PRNode) and getattr(node, "pr", None) is not None:
                n += 1
            for ch in (getattr(node, "children", None) or []):
                n += _count_prs(ch)
            return n

        pr_count = _count_prs(root)
    except Exception:
        pr_count = 0

    page_stats: List[tuple[str, Optional[str]]] = [
        ("Generation time", f"{elapsed_s:.2f}s"),
        ("Repos scanned", str(len(getattr(root, 'children', []) or []))),
        ("PRs shown", str(pr_count)),
    ]

    # Keep all page-stats augmentation in one scope block.
    if True:
        def _upsert_stat(k: str, v: str) -> None:
            try:
                for i, (kk, _vv) in enumerate(page_stats):
                    if kk == k:
                        page_stats[i] = (k, v)
                        return
            except Exception:
                pass
            page_stats.append((k, v))

        def _sort_stats() -> None:
            # Keep logical order for human scanning; sort all stats by prefix grouping.
            try:
                # Extract prefix (before first _ or .) for grouping
                def prefix_sort_key(kv):
                    k = str(kv[0])
                    if "_" in k or "." in k:
                        underscore_pos = k.find("_") if "_" in k else len(k)
                        dot_pos = k.find(".") if "." in k else len(k)
                        split_pos = min(underscore_pos, dot_pos)
                        prefix = k[:split_pos]
                    else:
                        prefix = k
                    return (prefix, k)
                
                # Sort all stats by prefix, then by full key
                page_stats[:] = sorted(page_stats, key=prefix_sort_key)
            except Exception:
                pass

        # GitHub API stats (structured; rendered with <pre> blocks in Statistics).
        try:
            from common_dashboard_lib import github_api_stats_rows  # local import

            mode = "cache-only" if bool(getattr(scanner, "cache_only_github", False)) else "normal"
            api_rows = github_api_stats_rows(
                github_api=scanner.github_api,
                max_github_api_calls=int(args.max_github_api_calls),
                mode=mode,
                mode_reason=cache_only_reason or "",
                top_n=15,
            )
            page_stats.extend(list(api_rows or []))
        except Exception:
            pass

        # Include relevant knobs if set (helps explain “why so many API calls?”).
        if args.max_branches is not None:
            page_stats.append(("max_branches", str(args.max_branches)))
        if args.max_checks_fetch is not None:
            page_stats.append(("max_checks_fetch", str(args.max_checks_fetch)))
        if bool(args.refresh_closed_prs):
            page_stats.append(("refresh_closed_prs", "true"))

        _sort_stats()

        # Render once to measure render time.
        with phase_t.phase("render"):
            _ = generate_html(root, page_stats=page_stats)

        # Update the page stats with the timing breakdown before producing the final HTML.
        try:
            tdict = phase_t.as_dict(include_total=True)
            # Make "Generation time" reflect total wall time so it matches the timing breakdown.
            elapsed_total = float(tdict.get("total") or 0.0)
            _upsert_stat("Generation time", f"{elapsed_total:.2f}s")
            for k in ["prune", "scan", "render", "write", "total"]:
                if k in tdict:
                    _upsert_stat(f"{k}.total_secs", f"{float(tdict.get(k) or 0.0):.2f}s")
        except Exception:
            pass
        _sort_stats()

        # Final render + atomic write to destination (single visible update).
        out_path = args.output
        if out_path is None:
            out_path = base_dir / "index.html"
        with phase_t.phase("render_final"):
            html_output2 = generate_html(root, page_stats=page_stats)
        with phase_t.phase("write"):
            atomic_write_text(out_path, html_output2, encoding="utf-8")

    # No stdout/stderr run-stats; the HTML Statistics section contains the breakdowns.


if __name__ == '__main__':
    main()
