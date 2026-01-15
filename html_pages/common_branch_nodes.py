#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared node classes and helper functions for branch/PR dashboards.

This module contains common tree node classes and utility functions used by both
show_local_branches.py and show_remote_branches.py to avoid code duplication.

IMPORTANT: Architecture Rule
---------------------------
⚠️ show_local_branches.py and show_remote_branches.py should NEVER import from each other!
   - ALL shared code must live in common files (this file, common_dashboard_lib.py, common.py)
   - If you need to share code between show_local/show_remote, PUT IT HERE
   - Never create circular dependencies or cross-imports between show_* scripts

Complete Implementations (all in this file):
- BranchNode: Base class for all tree nodes
- BranchInfoNode: Branch line with copy button, label, SHA, metadata
- CommitMessageNode: Commit message display
- MetadataNode: Branch metadata (modified, created, age)
- PRNode: PR summary line with status pill and links
- PRStatusNode: PR status information with collapsible CI hierarchy
- CIJobNode: CI job node with status icon, duration, logs
- RepoNode: Repository directory node (collapsible)
- SectionNode: Section header for grouping
- BlockedMessageNode, ConflictWarningNode, RerunLinkNode: Special status nodes
- build_ci_nodes_from_pr: Build flat CI job list from real PR check rows (PR first param, requires github_api)
- mock_build_ci_nodes: Build mock CI nodes for dummy PRs (negative PR numbers)
- mock_get_open_pr_info_for_author: Generate dummy PRInfo objects for testing
- Helper functions: _format_age_compact, _html_copy_button, generate_html, etc.
"""

from __future__ import annotations

import html as html_module
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
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
    run_all_passes,
    render_tree_pre_lines,
    required_badge_html,
    status_icon_html,
    _create_snippet_tree_node,
)

# Dashboard runtime (HTML-only) helpers
from common_dashboard_runtime import (
    materialize_job_raw_log_text_local_link,
)

# Import GitHub utilities from common module
from common import (
    FailedCheck,
    GHPRCheckRow,
    GitHubAPIClient,
    PRInfo,
    classify_ci_kind,
    summarize_pr_check_rows,
)


# Additional imports for refactored code
import hashlib
from common_dashboard_lib import (
    create_dummy_nodes_from_yaml_pass,
    ci_subsection_tuples_for_job,
    step_window_snippet_from_cached_raw_log,
)
from common_dashboard_runtime import (
    atomic_write_text,
    prune_dashboard_raw_logs,
    prune_partial_raw_log_caches,
)
import common


def mock_build_ci_nodes(
    pr: PRInfo,
    repo_path: Path,
    *,
    page_root_dir: Optional[Path] = None,
) -> List[BranchNode]:
    """Build mock CI job nodes for dummy PRs (PR number >= 100000000).
    
    Creates fake CIJobNode objects for testing/visualization without hitting the
    GitHub API. Useful for:
    - Testing YAML hierarchy visualization
    - Debugging dashboard layouts
    - CI/CD pipeline testing
    
    Returns List[BranchNode] where each item is a CIJobNode with fake but
    realistic-looking CI job data (success, pending, failure statuses).
    
    Args:
        pr: PRInfo object (must have pr.number >= 100000000 for mock PRs) - FIRST param for consistency
        repo_path: Path to the repository root
        page_root_dir: Root directory (unused, for signature compatibility)
        
    Returns:
        List[BranchNode]: Flat list of mock CIJobNode objects.
        Currently returns 3 fake jobs (success, pending, failure).
        
    Raises:
        ValueError: If pr.number < 100000000 (use build_ci_nodes_from_pr for real PRs)
    """
    if not pr or not getattr(pr, "number", None):
        return []
    
    pr_number = int(pr.number)
    if pr_number < 100000000:
        raise ValueError(f"mock_build_ci_nodes called with non-mock PR number {pr_number}. Mock PRs must have number >= 100000000. Use build_ci_nodes_from_pr for real PRs.")
    
    # Create a few mock CI jobs for visualization
    mock_jobs = []
    
    # Mock job 1: Success (required)
    mock_jobs.append(
        CIJobNode(
            job_id="mock-success",
            display_name="Mock CI Check (success)",
            status="success",
            duration="2m 34s",
            log_url="",
            is_required=True,  # Mark as required
            children=[],
        )
    )
    
    # Mock job 2: Pending (optional)
    mock_jobs.append(
        CIJobNode(
            job_id="mock-pending",
            display_name="Mock CI Check (pending)",
            status="pending",
            duration="",
            log_url="",
            is_required=False,  # Mark as optional
            children=[],
        )
    )
    
    # Mock job 3: Failure (required)
    mock_jobs.append(
        CIJobNode(
            job_id="mock-failure",
            display_name="Mock CI Check (failure)",
            status="failure",
            duration="5m 12s",
            log_url="",
            is_required=True,  # Mark as required
            children=[],
        )
    )
    
    return mock_jobs


def mock_get_open_pr_info_for_author(
    owner: str,
    repo: str,
    *,
    author: str,
    num_prs: int = 2,
) -> List[PRInfo]:
    """
    Generate realistic mock PRInfo objects for testing.
    
    Dummy PRs use NEGATIVE PR numbers (-1, -2, etc.) to distinguish them from real PRs.
    This allows the CI pipeline to route them to mock_build_ci_nodes instead of build_ci_nodes.
    
    This is useful for:
    - Testing YAML hierarchy visualization without real GitHub API calls
    - Debugging dashboard layouts and styling
    - CI/CD pipeline testing
    
    Args:
        owner: GitHub repository owner
        repo: GitHub repository name
        author: PR author username
        num_prs: Number of dummy PRs to create (default: 2)
    
    Returns:
        List of PRInfo objects with realistic-looking data and NEGATIVE PR numbers
    """
    from datetime import datetime, timedelta, timezone
    
    now = datetime.now(timezone.utc)
    
    # Template for realistic PR titles with MOCK branch names
    pr_templates = [
        ("feat: Add matrix expansion support for CI workflows", "mock/feature/ci-matrix-expansion", "pending", "APPROVED", 0),
        ("fix: Update backend status check hierarchy", "mock/bugfix/backend-hierarchy", "success", "REVIEW_REQUIRED", 2),
        ("refactor: Simplify process_ci_tree_passes pipeline", "mock/refactor/tree-passes", "failure", "APPROVED", 0),
        ("docs: Update YAML workflow documentation", "mock/docs/workflow-yaml", "success", "APPROVED", 1),
        ("test: Add integration tests for matrix expansion", "mock/test/matrix-integration", "pending", "CHANGES_REQUESTED", 3),
    ]
    
    prs = []
    for i in range(min(num_prs, len(pr_templates))):
        title, branch, ci_status, review_decision, unresolved = pr_templates[i]
        
        # Realistic timestamps (stagger PRs by days)
        created_dt = now - timedelta(days=7 - i)
        updated_dt = now - timedelta(hours=i + 1)
        
        # Generate realistic-looking SHAs (deterministic for testing)
        import hashlib
        sha_seed = f"{owner}/{repo}/{branch}/{i}"
        sha = hashlib.sha1(sha_seed.encode()).hexdigest()
        
        # Use PR numbers starting at 100000000 for mock PRs (easy to identify as mock)
        pr_number = 100000000 + i
        
        pr = PRInfo(
            number=pr_number,
            title=title,
            url=f"https://github.com/{owner}/{repo}/pull/{pr_number}",
            state="open",
            is_merged=False,
            review_decision=review_decision,
            mergeable_state="clean" if unresolved == 0 else "has_hooks",
            unresolved_conversations=unresolved,
            ci_status=ci_status,
            head_sha=sha,
            head_ref=branch,
            head_owner=owner,
            head_label=f"{owner}:{branch}",
            base_ref="main",
            created_at=created_dt.isoformat(),
            updated_at=updated_dt.isoformat(),
            has_conflicts=False,
            conflict_message=None,
            blocking_message=None,
            required_checks=[],
            failed_checks=[],
            running_checks=[],
            rerun_url=None,
        )
        prs.append(pr)
    
    return prs


# Log/snippet helpers (shared library: `dynamo-utils/ci_log_errors/`)
try:
    from ci_log_errors import extract_error_snippet_from_log_file
except ImportError:
    extract_error_snippet_from_log_file = None  # type: ignore[assignment]

# Jinja2 is optional (keep CLI usable in minimal envs).
try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    HAS_JINJA2 = True
except Exception:  # pragma: no cover
    HAS_JINJA2 = False
    Environment = None  # type: ignore[assignment]
    FileSystemLoader = None  # type: ignore[assignment]
    select_autoescape = None  # type: ignore[assignment]


#
# Repo constants (avoid scattering hardcoded strings)
#

DYNAMO_OWNER = "ai-dynamo"
DYNAMO_REPO = "dynamo"
DYNAMO_REPO_SLUG = f"{DYNAMO_OWNER}/{DYNAMO_REPO}"


#
# Helper functions
#

def _copy_icon_svg(*, size_px: int = 12) -> str:
    """Returns an inline SVG for the 'copy' clipboard icon."""
    try:
        p = (Path(__file__).resolve().parent / "copy_icon_paths.svg").resolve()
        paths = p.read_text(encoding="utf-8").strip()
    except Exception:
        paths = ""
    return (
        f'<svg width="{int(size_px)}" height="{int(size_px)}" viewBox="0 0 16 16" fill="currentColor" '
        f'style="display: inline-block; vertical-align: middle;">{paths}</svg>'
    )


def _html_copy_button(*, clipboard_text: str, title: str) -> str:
    """Returns an HTML copy button that calls the shared copyFromClipboardAttr(this) function.
    
    This matches the approach used in show_local_branches.py for consistency.
    The button stores the text in a data attribute and calls a shared JavaScript function
    from common_dashboard.j2, avoiding the need to embed SVG HTML in JavaScript strings.
    """
    text_esc = html_module.escape(str(clipboard_text or ""), quote=True)
    title_esc = html_module.escape(str(title or ""), quote=True)
    # Style constants matching show_local_branches.py
    btn_style = (
        "padding: 1px 4px; font-size: 10px; line-height: 1; "
        "background-color: transparent; color: #57606a; "
        "border: 1px solid #d0d7de; border-radius: 5px; cursor: pointer; "
        "display: inline-flex; align-items: center; vertical-align: baseline; "
        "margin-right: 4px;"
    )
    return (
        f'<button data-clipboard-text="{text_esc}" onclick="event.preventDefault(); copyFromClipboardAttr(this);" '
        f'style="{btn_style}" '
        f'title="{title_esc}" '
        f'onmouseover="this.style.backgroundColor=\'#f3f4f6\'; this.style.borderColor=\'#8c959f\';" '
        f'onmouseout="this.style.backgroundColor=\'transparent\'; this.style.borderColor=\'#d0d7de\';">'
        f'{_copy_icon_svg(size_px=12)}'
        f'</button>'
    )


def _html_small_link(*, url: str, label: str) -> str:
    """Returns a small styled hyperlink."""
    url_esc = html_module.escape(str(url or ""))
    label_esc = html_module.escape(str(label or ""))
    return f'<a href="{url_esc}" style="font-size: 11px; color: #0969da;">{label_esc}</a>'


def _aggregate_status(statuses: Iterable[str]) -> str:
    """Aggregate multiple statuses into a single rollup status."""
    statuses_list = list(statuses or [])
    if not statuses_list:
        return CIStatus.UNKNOWN
    if any(s == CIStatus.FAILED for s in statuses_list):
        return CIStatus.FAILED
    if any(s == CIStatus.IN_PROGRESS for s in statuses_list):
        return CIStatus.IN_PROGRESS
    if all(s == CIStatus.SUCCESS for s in statuses_list):
        return CIStatus.SUCCESS
    if all(s == CIStatus.SKIPPED for s in statuses_list):
        return CIStatus.SKIPPED
    return CIStatus.UNKNOWN


def _format_age_compact(dt: Optional[datetime]) -> Optional[str]:
    """Format a datetime as a compact relative age string (e.g., '2h', '3d')."""
    if dt is None:
        return None
    try:
        now = datetime.now(tz=ZoneInfo("UTC"))
        if getattr(dt, "tzinfo", None) is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        delta = now - dt
        seconds = int(delta.total_seconds())
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m"
        elif seconds < 86400:
            return f"{seconds // 3600}h"
        else:
            return f"{seconds // 86400}d"
    except Exception:
        return None


def _format_branch_metadata_suffix(
    *,
    commit_time_pt: Optional[str],
    commit_datetime: Optional[datetime],
    created_at: Optional[datetime],
) -> str:
    """Format the metadata suffix for a branch (modified, created, age)."""
    parts = []
    if commit_time_pt:
        parts.append(f'<span style="color: #656d76; font-size: 12px;">modified: {html_module.escape(commit_time_pt)}</span>')
    if created_at:
        created_pt = None
        try:
            if getattr(created_at, "tzinfo", None) is None:
                created_at = created_at.replace(tzinfo=ZoneInfo("UTC"))
            created_pt = created_at.astimezone(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M PT")
        except Exception:
            pass
        if created_pt:
            parts.append(f'<span style="color: #656d76; font-size: 12px;">created: {html_module.escape(created_pt)}</span>')
    age_str = _format_age_compact(commit_datetime)
    if age_str:
        parts.append(f'<span style="color: #656d76; font-size: 12px;">age: {html_module.escape(age_str)}</span>')
    if not parts:
        return ""
    return "(" + ", ".join(parts) + ")"


def _format_commit_tooltip(commit_message: Optional[str]) -> str:
    """Format commit message for use in a title/tooltip attribute.
    
    Multi-line messages are truncated to first 3 lines with "..." if longer.
    """
    msg = str(commit_message or "").strip()
    if not msg:
        return ""
    
    # Split into lines and keep first 3
    lines = msg.split('\n')
    if len(lines) > 3:
        lines = lines[:3] + ['...']
    
    # Rejoin and escape HTML entities
    msg_truncated = '\n'.join(lines)
    return html_module.escape(msg_truncated)


def _format_pr_number_link(pr: Optional[PRInfo]) -> str:
    """Format a PR number as a clickable link with icon.
    
    Handles both real PRs (number > 0, < 100000000) and mock PRs (number >= 100000000).
    """
    if pr is None:
        return ""
    try:
        pr_num = int(getattr(pr, "number", 0) or 0)
        if pr_num <= 0:
            return ""
        pr_url = str(getattr(pr, "url", "") or "").strip()
        if not pr_url:
            return f"#{pr_num}"
        return (
            f'<a href="{html_module.escape(pr_url)}" '
            f'style="color: #0969da; text-decoration: none;" '
            f'title="View PR #{pr_num}">#{pr_num}</a>'
        )
    except Exception:
        return ""


def _format_base_branch_inline(pr: Optional[PRInfo]) -> str:
    """Format the base branch as an inline HTML snippet (→ base_branch)."""
    if pr is None:
        return ""
    try:
        base_ref = str(getattr(pr, "base_ref", "") or "").strip()
        if not base_ref:
            return ""
        # Return the → symbol and base branch name in a muted color
        return f'<span style="color: #656d76;">→ {html_module.escape(base_ref)}</span>'
    except Exception:
        return ""


def _strip_repo_prefix_for_clipboard(branch_name: str) -> str:
    """Strip 'ai-dynamo/' or 'nvidia/' prefix from branch name for clipboard copy."""
    bn = str(branch_name or "").strip()
    for prefix in ["ai-dynamo/", "nvidia/"]:
        if bn.startswith(prefix):
            bn = bn[len(prefix):]
    return bn


def _pr_needs_attention(pr: Optional[PRInfo]) -> bool:
    """Determine if a PR needs attention (failed required checks or conflicts)."""
    if pr is None:
        return False
    try:
        # Check if there are failed required checks
        failed_req = [fc for fc in (getattr(pr, "failed_checks", None) or []) if getattr(fc, "required", False)]
        if failed_req:
            return True
        # Check for conflicts or blocking messages
        if getattr(pr, "conflict_message", None):
            return True
        if getattr(pr, "blocking_message", None):
            return True
    except Exception:
        pass
    return False


def looks_like_git_repo_dir(p: Path) -> bool:
    """Check if a directory looks like a git repository."""
    try:
        dot_git = p / ".git"
        if dot_git.is_dir():
            return True
        if dot_git.is_file():
            # Git submodule or worktree (has .git file pointing to real gitdir)
            return True
    except Exception:
        pass
    return False


def gitdir_from_git_file(p: Path) -> Optional[Path]:
    """Extract the gitdir path from a .git file (for worktrees/submodules)."""
    try:
        dot_git = p / ".git"
        if dot_git.is_file():
            content = dot_git.read_text(encoding="utf-8", errors="ignore").strip()
            if content.startswith("gitdir: "):
                gitdir_rel = content[len("gitdir: "):].strip()
                gitdir_abs = (p / gitdir_rel).resolve()
                if gitdir_abs.exists():
                    return gitdir_abs
    except Exception:
        pass
    return None


def origin_url_from_git_config(repo_dir: Path) -> str:
    """Extract the origin URL from a git repository's config."""
    try:
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def find_local_clone_of_repo(base_dir: Path, *, repo_slug: str) -> Optional[Path]:
    """Find a local clone of a repository by searching for matching origin URLs."""
    try:
        for candidate in base_dir.iterdir():
            if not candidate.is_dir():
                continue
            if not looks_like_git_repo_dir(candidate):
                continue
            origin_url = origin_url_from_git_config(candidate)
            if repo_slug in origin_url:
                return candidate
    except Exception:
        pass
    return None


#
# Tree node classes
#

class BranchNode:
    """Base class for all tree nodes in the branch/PR hierarchy."""
    
    def __init__(
        self,
        label: str,
        *,
        children: Optional[List["BranchNode"]] = None,
        expanded: bool = False,
        status: str = CIStatus.UNKNOWN,
        is_current: bool = False,
    ):
        self.label = str(label or "")
        self.children: List[BranchNode] = list(children or [])
        self.expanded = bool(expanded)
        self.status = str(status or CIStatus.UNKNOWN)
        self.is_current = bool(is_current)
    
    def add_child(self, node: "BranchNode") -> None:
        """Add a child node."""
        self.children.append(node)
    
    def to_tree_vm(self) -> TreeNodeVM:
        """Convert this node and its children to a TreeNodeVM for rendering."""
        kids = [c.to_tree_vm() for c in (self.children or [])]
        return TreeNodeVM(
            node_key=f"branch:{id(self)}",
            label_html=html_module.escape(self.label),
            children=kids,
            collapsible=True,
            default_expanded=self.expanded,
        )


class CIJobNode(BranchNode):
    """A CI job tree node that renders as a check line with status icon, duration, and logs."""
    
    def __init__(
        self,
        *,
        job_id: str,
        display_name: str = "",
        status: str = CIStatus.UNKNOWN,
        duration: str = "",
        log_url: str = "",
        is_required: bool = False,
        children: Optional[List[BranchNode]] = None,
        expanded: bool = False,
        page_root_dir: Optional[Path] = None,
        context_key: str = "",
        github_api=None,  # GitHubAPIClient for raw log materialization
        raw_log_href: str = "",  # Pre-materialized raw log href (relative path)
        raw_log_size_bytes: int = 0,  # Pre-materialized raw log size
        error_snippet_text: str = "",  # Pre-extracted error snippet
        error_snippet_categories: Optional[List[str]] = None,  # Error categories (e.g., pytest-error, python-error)
        short_job_name: str = "",  # Short job name from YAML (e.g., "build-test")
        yaml_dependencies: Optional[List[str]] = None,  # List of job names this depends on (needs:)
    ):
        super().__init__(label="", children=children, expanded=expanded, status=status)
        self.job_id = str(job_id or "")
        self.display_name = str(display_name or "")
        self.duration = str(duration or "")
        self.log_url = str(log_url or "")
        self.is_required = bool(is_required)
        self.page_root_dir = page_root_dir
        self.context_key = str(context_key or "")
        self.github_api = github_api
        self.raw_log_href = str(raw_log_href or "")
        self.raw_log_size_bytes = int(raw_log_size_bytes or 0)
        self.error_snippet_text = str(error_snippet_text or "")
        self.error_snippet_categories = list(error_snippet_categories or [])
        self.short_job_name = str(short_job_name or "")
        self.yaml_dependencies = list(yaml_dependencies or [])
    
    def to_tree_vm(self) -> TreeNodeVM:
        """Convert this CI job node to a TreeNodeVM using check_line_html."""
        # Use pre-materialized raw log data (set during build_ci_nodes_from_pr)
        # No need to re-materialize here - it's already been done
        
        # Use shared check_line_html to generate the HTML (without snippet inline)
        label_html = check_line_html(
            job_id=self.job_id,
            display_name=self.display_name,
            status_norm=self.status,
            is_required=self.is_required,
            duration=self.duration,
            log_url=self.log_url,
            raw_log_href=self.raw_log_href,
            raw_log_size_bytes=self.raw_log_size_bytes,
            error_snippet_text=self.error_snippet_text,
            error_snippet_categories=self.error_snippet_categories,
            short_job_name=self.short_job_name,
            yaml_dependencies=self.yaml_dependencies,
        )
        
        # Build children: existing children + snippet node (if present)
        kids = [c.to_tree_vm() for c in (self.children or [])]
        
        # Add snippet as a child node if we have snippet text
        if self.error_snippet_text and self.error_snippet_text.strip():
            snippet_node = _create_snippet_tree_node(
                dom_id_seed=f"{self.job_id}|{self.display_name}|{self.raw_log_href}|{self.log_url}",
                snippet_text=self.error_snippet_text,
            )
            if snippet_node:
                kids.append(snippet_node)
        
        return TreeNodeVM(
            node_key=f"ci:{self.job_id}:{id(self)}",
            label_html=label_html,
            children=kids,
            collapsible=True,
            default_expanded=self.expanded,
            job_name=self.job_id,  # Set job_name for hierarchy matching and validation
            core_job_name=getattr(self, 'core_job_name', ''),  # Propagate core_job_name for matching in merge
            short_job_name=getattr(self, 'short_job_name', ''),  # Propagate short_job_name from augmentation
        )


class RepoNode(BranchNode):
    """Represents a repository in the tree (for multi-repo views)."""
    
    def __init__(self, label: str, *, children: Optional[List[BranchNode]] = None):
        super().__init__(label=label, children=children, expanded=True)
    
    def to_tree_vm(self) -> TreeNodeVM:
        """Convert repo node to TreeNodeVM with bold styling."""
        kids = [c.to_tree_vm() for c in (self.children or [])]
        label_html = f'<span style="font-weight: 600;">{html_module.escape(self.label)}</span>'
        return TreeNodeVM(
            node_key=f"repo:{id(self)}",
            label_html=label_html,
            children=kids,
            collapsible=False,
            default_expanded=True,
        )


class SectionNode(BranchNode):
    """Represents a section header in the tree (for grouping branches)."""
    
    def __init__(self, label: str, *, children: Optional[List[BranchNode]] = None):
        super().__init__(label=label, children=children, expanded=True)
    
    def to_tree_vm(self) -> TreeNodeVM:
        """Convert section node to TreeNodeVM with header styling."""
        kids = [c.to_tree_vm() for c in (self.children or [])]
        label_html = f'<span style="font-weight: 600; font-size: 14px;">{html_module.escape(self.label)}</span>'
        return TreeNodeVM(
            node_key=f"section:{id(self)}",
            label_html=label_html,
            children=kids,
            collapsible=False,
            default_expanded=True,
        )


class BranchInfoNode(BranchNode):
    """Branch line node with copy button, label, SHA, and metadata."""
    
    def __init__(
        self,
        label: str,
        *,
        sha: Optional[str] = None,
        is_current: bool = False,
        merged_local: bool = False,
        commit_url: Optional[str] = None,
        commit_time_pt: Optional[str] = None,
        commit_datetime: Optional[datetime] = None,
        commit_message: Optional[str] = None,
        created_at: Optional[datetime] = None,
        pr: Optional[PRInfo] = None,
        children: Optional[List[BranchNode]] = None,
    ):
        super().__init__(label=label, children=children, expanded=False, is_current=is_current)
        self.sha = sha
        # Local-only merge detection (e.g., branch tip is already in main) for pages that don't have PR metadata.
        self.merged_local = bool(merged_local)
        self.commit_url = commit_url
        self.commit_time_pt = commit_time_pt
        self.commit_datetime = commit_datetime
        self.commit_message = commit_message
        self.created_at = created_at
        self.pr = pr
    
    def to_tree_vm(self) -> TreeNodeVM:
        """Convert to TreeNodeVM with structured children (commit message + metadata + other)."""
        # Construct the main branch line HTML
        parts = []
        
        # Copy button
        clipboard_text = _strip_repo_prefix_for_clipboard(self.label)
        parts.append(_html_copy_button(clipboard_text=clipboard_text, title=f"Copy branch name: {clipboard_text}"))
        
        try:
            pr_state_lc = (getattr(self.pr, "state", "") or "").lower() if self.pr else ""
            is_merged = bool(getattr(self.pr, "is_merged", False)) if self.pr else False
            is_closed_not_merged = bool(self.pr) and pr_state_lc and pr_state_lc != "open" and not is_merged
        except Exception:
            pr_state_lc = ""
            is_merged = False
            is_closed_not_merged = False

        # Branch state tag (do NOT style/invert the branch name itself).
        if bool(self.merged_local) or bool(is_merged):
            parts.append('<span class="branch-tag merged-tag">Merged</span>')
        elif bool(is_closed_not_merged):
            # Closed tag should be reverse black/white; reuse existing CSS class.
            parts.append('<span class="branch-tag closed-branch">Closed</span>')

        # Branch name (fixed font; keep normal style regardless of merged/closed).
        cls_attr = ' class="current"' if self.is_current else ""
        parts.append(
            f'<span{cls_attr} style="font-family: monospace; font-weight: 700;">{html_module.escape(self.label)}</span>'
        )
        
        # SHA link (if available)
        if self.sha and self.commit_url:
            sha_title = _format_commit_tooltip(self.commit_message)
            parts.append(
                f'<a href="{html_module.escape(self.commit_url)}" '
                f'style="color: #0969da; font-size: 12px; text-decoration: none;" '
                f'title="{sha_title}">[{html_module.escape(self.sha)}]</a>'
            )
        elif self.sha:
            sha_title = _format_commit_tooltip(self.commit_message)
            parts.append(f'<span style="color: #656d76; font-size: 12px;" title="{sha_title}">[{html_module.escape(self.sha)}]</span>')
        
        label_html = " ".join(parts)
        
        # Build children: commit message with metadata inline, then other children
        kids = []
        
        # 1. Commit message with metadata on same line (if available)
        if self.commit_message:
            # Get metadata suffix (already contains HTML tags)
            metadata_suffix = _format_branch_metadata_suffix(
                commit_time_pt=self.commit_time_pt,
                commit_datetime=self.commit_datetime,
                created_at=self.created_at,
            )
            
            # Build label with commit message styled, then PR number (if available), then raw metadata HTML
            commit_parts = []
            
            # Format commit message: first line only
            msg_lines = self.commit_message.split('\n')
            msg_display = msg_lines[0] if msg_lines else self.commit_message
            
            # Style the commit message text
            commit_parts.append(f'<span style="color: #1f2328;">{html_module.escape(msg_display)}</span>')
            
            # Add PR number link (if available) - with parentheses like git log
            if self.pr:
                pr_link = _format_pr_number_link(self.pr)
                if pr_link:
                    commit_parts.append(f'({pr_link})')
            
            if metadata_suffix:
                # Metadata already has HTML tags AND parentheses, append as-is
                commit_parts.append(metadata_suffix)
            
            combined_label = " ".join(commit_parts)
            
            # Create a TreeNodeVM directly instead of using CommitMessageNode
            # to avoid double-escaping the HTML in metadata
            kids.append(TreeNodeVM(
                node_key=f"commit_msg_metadata:{id(self)}",
                label_html=combined_label,
                children=[],
                collapsible=False,
                default_expanded=False,
                skip_dedup=True,
            ))
        
        # 2. Other children (PR status, CI checks, etc.)
        for c in (self.children or []):
            kids.append(c.to_tree_vm())
        
        is_merged_effective = bool(self.merged_local) or bool(is_merged)
        is_closed_effective = bool(is_closed_not_merged)
        default_expanded = not (is_merged_effective or is_closed_effective)

        return TreeNodeVM(
            node_key=f"branch_info:{self.label}:{id(self)}",
            label_html=label_html,
            children=kids,
            collapsible=True,  # Make branches collapsible like local branches
            # If merged or closed, keep collapsed by default (user can still expand manually).
            default_expanded=bool(default_expanded),
        )


class PRNode(BranchNode):
    """PR summary line node (shows PR status pill and links)."""
    
    def __init__(self, label: str, *, pr: Optional[PRInfo] = None, children: Optional[List[BranchNode]] = None):
        super().__init__(label=label, children=children, expanded=False)
        self.pr = pr
    
    def to_tree_vm(self) -> TreeNodeVM:
        """Convert to TreeNodeVM with PR status pill and links."""
        if self.pr is None:
            label_html = html_module.escape(self.label) if self.label else "(no PR)"
            kids = [c.to_tree_vm() for c in (self.children or [])]
            return TreeNodeVM(
                node_key=f"pr:{id(self)}",
                label_html=label_html,
                children=kids,
                collapsible=False,
                default_expanded=True,
            )
        
        # Build PR line: status pill + PR number + title + base branch
        parts = []
        
        # Status pill - get summary from the PR
        try:
            summary = summarize_pr_check_rows(getattr(self.pr, "check_rows", None) or [])
            counts = summary.counts
            status_html = compact_ci_summary_html(
                success_required=int(counts.get("success_required", 0) or 0),
                success_optional=int(counts.get("success_optional", 0) or 0),
                failure_required=int(counts.get("failure_required", 0) or 0),
                failure_optional=int(counts.get("failure_optional", 0) or 0),
                in_progress_required=int(counts.get("in_progress", 0) or 0),
                in_progress_optional=0,
                pending=int(counts.get("pending", 0) or 0),
                cancelled=int(counts.get("cancelled", 0) or 0),
            )
            if status_html:
                parts.append(status_html)
        except Exception:
            pass
        
        # PR number link
        pr_link = _format_pr_number_link(self.pr)
        if pr_link:
            parts.append(pr_link)
        
        # PR title
        title = str(getattr(self.pr, "title", "") or "").strip()
        if title:
            parts.append(html_module.escape(title))
        
        # Base branch (→ base)
        base_branch_html = _format_base_branch_inline(self.pr)
        if base_branch_html:
            parts.append(base_branch_html)
        
        label_html = " ".join(parts)
        kids = [c.to_tree_vm() for c in (self.children or [])]
        
        return TreeNodeVM(
            node_key=f"pr:{getattr(self.pr, 'number', id(self))}",
            label_html=label_html,
            children=kids,
            collapsible=False,
            default_expanded=True,
        )


class PRURLNode(BranchNode):
    """PR URL node (renders as a clickable link)."""
    
    def __init__(self, label: str, *, pr: Optional[PRInfo] = None):
        super().__init__(label=label, children=None, expanded=False)
        self.pr = pr
    
    def to_tree_vm(self) -> TreeNodeVM:
        """Convert to TreeNodeVM with clickable PR URL."""
        url = str(getattr(self.pr, "url", "") or "").strip() if self.pr else ""
        if url:
            label_html = _html_small_link(url=url, label=self.label or "View PR")
        else:
            label_html = html_module.escape(self.label)
        
        return TreeNodeVM(
            node_key=f"pr_url:{id(self)}",
            label_html=label_html,
            children=[],
            collapsible=False,
            default_expanded=False,
        )


class CommitMessageNode(BranchNode):
    """Commit message node with PR number (if available)."""
    
    def __init__(self, label: str, *, pr: Optional[PRInfo] = None):
        super().__init__(label=label, children=None, expanded=False)
        self.pr = pr
    
    def to_tree_vm(self) -> TreeNodeVM:
        """Convert to TreeNodeVM with commit message and optional PR link."""
        parts = []
        
        # Commit message
        msg = str(self.label or "").strip()
        if msg:
            parts.append(f'<span style="color: #1f2328;">{html_module.escape(msg)}</span>')
        
        # PR number link (if available) - with parentheses like git log
        if self.pr:
            pr_link = _format_pr_number_link(self.pr)
            if pr_link:
                parts.append(f'({pr_link})')
        
        label_html = " ".join(parts) if parts else ""
        
        return TreeNodeVM(
            node_key=f"commit_msg:{id(self)}",
            label_html=label_html,
            children=[],
            collapsible=False,
            default_expanded=False,
            skip_dedup=True,  # Always display commit messages, never deduplicate
        )


class MetadataNode(BranchNode):
    """Metadata node (modified, created, age in muted color)."""
    
    def __init__(self, label: str):
        super().__init__(label=label, children=None, expanded=False)
    
    def to_tree_vm(self) -> TreeNodeVM:
        """Convert to TreeNodeVM with muted metadata text."""
        # Label is already formatted with HTML (from _format_branch_metadata_suffix)
        return TreeNodeVM(
            node_key=f"metadata:{id(self)}",
            label_html=self.label,  # Already contains HTML
            children=[],
            collapsible=False,
            default_expanded=False,
            skip_dedup=True,  # Always display metadata, never deduplicate
        )



#
# CI hierarchy builder (STUB - DO NOT USE)
# Import _build_ci_hierarchy_nodes from show_local_branches.py for production use
#


#
# HTML generation (shared by both local and remote)
#

def generate_html(
    root: BranchNode,
    *,
    page_stats: List[Tuple[str, str]],
    page_title: str,
    header_title: str,
    tree_html_override: Optional[str] = None,
    tree_html_alt: Optional[str] = None,
    tree_sort_default: str = "modified",
    tree_sortable: bool = False,
    use_div_trees: bool = False,
) -> str:
    """
    Generate the final HTML page using the Jinja2 template.
    
    Args:
        root: Root BranchNode of the tree
        page_stats: List of (label, value) tuples for the stats footer
        page_title: HTML page title
        header_title: Header title displayed in the page
        tree_html_override: Optional pre-rendered tree HTML (for client-side sorting)
        tree_html_alt: Optional alternate tree HTML (no longer used)
        tree_sort_default: Default sort order ("modified", "created", "branch")
        tree_sortable: Whether to enable client-side sorting
        use_div_trees: Use <div>-based rendering instead of <pre> (experimental)
    
    Returns:
        HTML string
    """
    if not HAS_JINJA2:
        raise RuntimeError("Jinja2 is required for HTML generation")
    
    # Get current time in both UTC and PDT
    now_utc = datetime.now(ZoneInfo('UTC'))
    now_pdt = datetime.now(ZoneInfo('America/Los_Angeles'))
    
    # Format timestamps
    utc_str = now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')
    pdt_str = now_pdt.strftime('%Y-%m-%d %H:%M:%S %Z')
    
    # Render the tree if not provided
    if tree_html_override is None:
        from common_dashboard_lib import render_tree_divs
        
        # If root has no label and has children, render children directly (skip empty root)
        root_vm = root.to_tree_vm()
        if not root_vm.label_html and root_vm.children:
            nodes_to_render = root_vm.children
        else:
            nodes_to_render = [root_vm]
        
        if use_div_trees:
            tree_html_override = render_tree_divs(nodes_to_render)
        else:
            tree_html_override = "\n".join(render_tree_pre_lines(nodes_to_render))
    
    # Load template
    template_dir = Path(__file__).resolve().parent
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html", "j2"]),
    )
    template = env.get_template("show_local_branches.j2")
    
    # Render (matching show_local_branches.py parameters)
    return template.render(
        generated_time=pdt_str,
        page_title=page_title,
        header_title=header_title,
        copy_icon_svg=_copy_icon_svg(size_px=12),
        tree_html=tree_html_override,
        tree_html_alt=tree_html_alt or "",
        tree_sort_default=tree_sort_default,
        tree_sortable=tree_sortable,
        use_div_trees=use_div_trees,
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


# ======================================================================
# RawLogValidationError (moved from show_local_branches.py)
# ======================================================================

class RawLogValidationError(RuntimeError):
    """Raised when we expect a local `[raw log]` for a failed Actions job but cannot produce one."""



# ======================================================================
# _is_known_required_check (moved from show_local_branches.py)
# ======================================================================

def _is_known_required_check(check_name: str) -> bool:
    """Check if a job name matches one of the known required checks.
    
    The 5 known required checks are:
    - backend-status-check
    - DCO
    - copyright-checks
    - dynamo-status-check
    - pre-commit
    """
    check_lower = str(check_name or "").lower()
    
    # Exact matches for short names (to avoid false positives like "dco-comment")
    if check_lower == "dco":
        return True
    if check_lower == "pre-commit":
        return True
    
    # Substring matches for longer, more specific names
    if "backend-status-check" in check_lower:
        return True
    if "copyright-checks" in check_lower:
        return True
    if "dynamo-status-check" in check_lower:
        return True
    
    return False


# ======================================================================
# _duration_str_to_seconds (moved from show_local_branches.py)
# ======================================================================

def _duration_str_to_seconds(s: str) -> float:
    """Best-effort parse of durations like '43m 33s', '30m33s', '2s', '1h 4m'."""
    try:
        total = 0.0
        for m in re.finditer(r"([0-9]+)\s*([hms])", str(s or "").lower()):
            v = float(m.group(1))
            u = m.group(2)
            if u == "h":
                total += v * 3600.0
            elif u == "m":
                total += v * 60.0
            elif u == "s":
                total += v
        return total
    except Exception:
        return 0.0



# ======================================================================
# _assume_completed_for_check_row (moved from show_local_branches.py)
# ======================================================================

def _assume_completed_for_check_row(r: "GHPRCheckRow") -> bool:
    """Infer completion from the PR check row itself (avoid extra per-job API calls).

    `GHPRCheckRow.status_raw` comes from the REST check-run status+conclusion:
    - non-completed: queued/pending/in_progress
    - completed: pass/fail/skipped/cancelled/neutral/timed_out/action_required
    """
    try:
        s = str(getattr(r, "status_raw", "") or "").strip().lower()
        if not s:
            return False
        return s not in {"in_progress", "pending", "queued", "running", "unknown"}
    except Exception:
        return False


# ======================================================================
# build_ci_nodes_from_pr (moved from show_local_branches.py)
# ======================================================================

def build_ci_nodes_from_pr(
    pr: PRInfo,
    github_api: GitHubAPIClient,
    repo_path: Path,
    *,
    page_root_dir: Optional[Path] = None,
    checks_ttl_s: int = 300,
    skip_fetch: bool = False,
    validate_raw_logs: bool = True,
) -> List[BranchNode]:
    """Build a flat list of CI job nodes from a PR's GitHub check runs.
    
    Fetches check-run data from GitHub API and creates CIJobNode objects containing:
    - Full verbatim job names (e.g., "Workflow Name / check-name (event)")
    - Status (success/failure/pending/running/cancelled)
    - Duration, URLs, error snippets, log caching
    - Required vs optional (from branch protection rules)
    
    Returns List[BranchNode] where each item is actually a CIJobNode with
    specific CI job information from the GitHub Actions API.
    
    Args:
        pr: PRInfo object (must have valid pr.number >= 0) - FIRST param for importance
        github_api: GitHub API client (REQUIRED - no Optional, API is essential for data)
        repo_path: Path to the repository root
        page_root_dir: Root directory for caching
        checks_ttl_s: Cache TTL in seconds
        skip_fetch: Skip network fetch if True
        validate_raw_logs: Validate raw logs if True
        
    Returns:
        List[BranchNode]: Flat list of CIJobNode objects (no hierarchy).
        Each node contains specific CI job details from GitHub API.
        
    Raises:
        ValueError: If pr.number is negative (use mock_build_ci_nodes for dummy PRs)
    """
    if not pr or not getattr(pr, "number", None):
        return []
    
    pr_number = int(pr.number)
    if pr_number >= 100000000:
        raise ValueError(f"build_ci_nodes_from_pr called with mock PR number {pr_number}. Use mock_build_ci_nodes instead.")
    
    pr_number = int(pr.number)
    if pr_number >= 100000000:
        raise ValueError(f"build_ci_nodes called with mock PR number {pr_number}. Use mock_build_ci_nodes instead.")
    
    # Ensure required-ness is correct even if PRInfo cache is stale.
    # This uses `gh` (GraphQL statusCheckRollup isRequired), not our REST budget.
    # ALWAYS fetch fresh required checks - don't trust cached pr.required_checks.
    required_set: Set[str] = set()
    try:
        required_set = set(github_api.get_required_checks(DYNAMO_OWNER, DYNAMO_REPO, pr_number) or set())
    except Exception:
        # Fallback to cached pr.required_checks if API call fails
        required_set = set(getattr(pr, "required_checks", []) or [])
    
    rows = github_api.get_pr_checks_rows(
        DYNAMO_OWNER,
        DYNAMO_REPO,
        pr_number,
        commit_sha=str(getattr(pr, "head_sha", None) or ""),
        required_checks=required_set,
        ttl_s=int(checks_ttl_s),
        skip_fetch=bool(skip_fetch),
    )
    
    if not rows:
        return []

    # Inject placeholders for expected checks that are missing from the reported contexts.
    #
    # This makes "missing required checks" visible even when GitHub never posts a check context for them.
    try:
        present_norm = {common.normalize_check_name(str(getattr(r, "name", "") or "")) for r in (rows or [])}
        seen_norm = set(present_norm)
        expected_required = {str(x) for x in (required_set or set()) if str(x).strip()}
        required_norm = {common.normalize_check_name(x) for x in expected_required}

        # Expected checks from branch protection required checks.
        for nm0 in sorted(expected_required, key=lambda s: str(s).lower()):
            n0 = common.normalize_check_name(nm0)
            if n0 and n0 not in seen_norm:
                rows.append(
                    GHPRCheckRow(
                        name=str(nm0),
                        status_raw="pending",
                        duration="",
                        url="",
                        run_id="",
                        job_id="",
                        description="expected",
                        is_required=(common.normalize_check_name(nm0) in required_norm),
                    )
                )
                seen_norm.add(n0)

        # Note: Expected checks (◇) inference from workflow YAML has been removed.
        # Only actual check runs from the API are displayed.
    except Exception:
        pass

    # Note: Sorting is now handled by PASS 4 (sort_by_name_pass) in the centralized pipeline.
    # No need to pre-sort rows here.

    # If the same check name appears multiple times (reruns), append a stable unique id
    # so users can tell them apart.
    name_counts: Dict[str, int] = {}
    for _r in (rows or []):
        try:
            nm0 = str(getattr(_r, "name", "") or "").strip()
            if nm0:
                name_counts[nm0] = int(name_counts.get(nm0, 0) or 0) + 1
        except Exception:
            continue

    # CI view: the expanded tree under the PR status line shows the check list and optional subsections.
    out: List[BranchNode] = []
    missing_failed_raw_logs: List[Tuple[str, str]] = []
    any_failed = False
    rerun_run_id: str = ""
    for r in rows:
        nm = str(getattr(r, "name", "") or "").strip()
        raw = str(getattr(r, "status_raw", "") or "").strip().lower()
        if raw in {"skipped", "skip", "neutral"}:
            st = "skipped"
        elif raw in {"pass", "success"}:
            st = "success"
        elif raw in {"fail", "failure", "timed_out", "action_required"}:
            st = "failure"
            any_failed = True
        elif raw in {"in_progress", "in progress", "running"}:
            st = "in_progress"
        elif raw in {"queued", "pending"}:
            st = "pending"
        elif raw in {"cancelled", "canceled"}:
            st = "cancelled"
        else:
            st = str(getattr(r, "status_norm", "") or "unknown")

        job_url = str(getattr(r, "url", "") or "").strip()
        # Capture a run_id for the "Restart failed jobs" affordance.
        # Prefer the row's parsed run_id (if present), otherwise parse from URL.
        if (not rerun_run_id) and st == "failure":
            try:
                rid = str(getattr(r, "run_id", "") or "").strip()
                if not rid:
                    rid = str(common.parse_actions_run_id_from_url(job_url) or "").strip()
                if rid:
                    rerun_run_id = rid
            except Exception:
                pass
        base_dir = (Path(page_root_dir) if page_root_dir is not None else repo_path)
        raw_href = ""
        raw_size = 0
        snippet = ""

        # For failed GitHub Actions jobs, always materialize a local raw log so the tree shows `[raw log]`.
        # This is intentionally strict to avoid recurring regressions where errors have no raw logs.
        if st == "failure" and (not skip_fetch):
            try:
                from common_dashboard_lib import extract_actions_job_id_from_url  # local import
                from common_dashboard_runtime import materialize_job_raw_log_text_local_link  # local import

                if extract_actions_job_id_from_url(job_url):
                    raw_href = (
                        materialize_job_raw_log_text_local_link(
                            github_api,
                            job_url=job_url,
                            job_name=nm,
                            owner=DYNAMO_OWNER,
                            repo=DYNAMO_REPO,
                            page_root_dir=base_dir,
                            allow_fetch=True,
                            assume_completed=_assume_completed_for_check_row(r),
                        )
                        or ""
                    )
            except Exception:
                raw_href = ""

        if raw_href:
            try:
                raw_size = int((base_dir / raw_href).stat().st_size)
            except Exception:
                raw_size = 0
            try:
                snippet = extract_error_snippet_from_log_file(base_dir / raw_href)
            except Exception:
                snippet = ""
        elif st == "failure":
            # Only validate failures that correspond to GitHub Actions jobs (others like DCO have no raw log).
            try:
                from common_dashboard_lib import extract_actions_job_id_from_url  # local import

                if extract_actions_job_id_from_url(job_url):
                    missing_failed_raw_logs.append((nm, job_url))
            except Exception:
                pass

        # Disambiguate duplicates by job_id (best) or run_id.
        display_name = ""
        try:
            if int(name_counts.get(nm, 0) or 0) > 1:
                jid = str(getattr(r, "job_id", "") or "").strip()
                rid = str(getattr(r, "run_id", "") or "").strip()
                if jid:
                    display_name = f"{nm} [{jid}]"
                elif rid:
                    display_name = f"{nm} [run {rid}]"
        except Exception:
            display_name = ""
        # Placeholder check row (expected but not yet reported).
        try:
            if (not display_name) and (not job_url) and str(getattr(r, "description", "") or "").strip().lower() == "expected":
                display_name = EXPECTED_CHECK_PLACEHOLDER_SYMBOL
        except Exception:
            pass
        
        # Build the full verbatim job name with workflow name and event type
        # Format: "Workflow Name / check-name (event)"
        # Example: "NVIDIA Dynamo Github Validation / dynamo-status-check (pull_request)"
        full_job_name = nm
        try:
            workflow_name = str(getattr(r, "workflow_name", "") or "").strip()
            event = str(getattr(r, "event", "") or "").strip()
            if workflow_name and event:
                full_job_name = f"{workflow_name} / {nm} ({event})"
            elif workflow_name:
                full_job_name = f"{workflow_name} / {nm}"
        except Exception:
            pass

        node = CIJobNode(
            job_id=full_job_name,  # Use full verbatim name as job_id for display
            display_name=nm,  # Use the original check name as display_name (e.g., "Build and Test - dynamo")
            status=str(st or "unknown"),
            duration=str(getattr(r, "duration", "") or ""),
            log_url=job_url,
            is_required=bool(getattr(r, "is_required", False)),
            children=[],
            page_root_dir=page_root_dir,  # Pass page_root_dir for raw log and snippet extraction
            context_key=f"{pr_number}:{full_job_name}",  # Unique context for caching
            github_api=github_api,  # Pass github_api for raw log materialization
            raw_log_href=raw_href,  # Pass pre-materialized raw log data
            raw_log_size_bytes=raw_size,
            error_snippet_text=snippet,
        )
        # Store the core job name (without workflow prefix/event suffix) for matching in merge
        node.core_job_name = nm  # This is the original "name" from the check row
        
        # DEBUG: Verify it was set
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"[build_ci_nodes_from_pr] Set core_job_name='{nm}' on node with job_id='{full_job_name[:50]}'")




        # Shared subsections:
        # - Build and Test - dynamo: phases (a special-case subsection; steps API)
        # - other long-running Actions jobs: long steps (>= 30s)
        try:
            from common_dashboard_lib import ci_subsection_tuples_for_job  # local import
            from common_dashboard_runtime import materialize_job_raw_log_text_local_link  # local import

            dur_s = _duration_str_to_seconds(str(getattr(r, "duration", "") or ""))

            # For Build-and-Test, allow raw-log fetch even on success so fallback parsing can work.
            raw_href_for_sub = raw_href
            if nm == "Build and Test - dynamo" and (not raw_href_for_sub) and (not skip_fetch):
                try:
                    raw_href_for_sub = (
                        materialize_job_raw_log_text_local_link(
                            github_api,
                            job_url=job_url,
                            job_name=nm,
                            owner=DYNAMO_OWNER,
                            repo=DYNAMO_REPO,
                            page_root_dir=base_dir,
                            allow_fetch=True,
                            assume_completed=_assume_completed_for_check_row(r),
                        )
                        or ""
                    )
                except Exception:
                    raw_href_for_sub = raw_href_for_sub or ""

            raw_path_for_sub = (base_dir / raw_href_for_sub) if raw_href_for_sub else None

            sub3 = ci_subsection_tuples_for_job(
                github_api=github_api,
                job_name=nm,
                job_url=job_url,
                raw_log_path=raw_path_for_sub,
                duration_seconds=float(dur_s or 0.0),
                is_required=bool(getattr(r, "is_required", False)),
                long_job_threshold_s=10.0 * 60.0,
                step_min_s=30.0,
            )
            for (sub_name, sub_dur, sub_status) in (sub3 or []):
                if nm == "Build and Test - dynamo":
                    kind2 = classify_ci_kind(str(sub_name))
                    sub_id = f"{kind2}: {sub_name}" if kind2 and kind2 != "check" else str(sub_name)
                else:
                    sub_id = str(sub_name)
                # If this is a failing step, try to scope the snippet to the step window.
                step_snip = ""
                try:
                    if sub_id.startswith("step:") and str(sub_status or "") == "failure" and raw_path_for_sub:
                        from common_dashboard_lib import step_window_snippet_from_cached_raw_log  # local import
                        from common_dashboard_lib import extract_actions_job_id_from_url  # local import

                        jid = extract_actions_job_id_from_url(job_url)
                        job_det = (
                            github_api.get_actions_job_details_cached(owner=DYNAMO_OWNER, repo=DYNAMO_REPO, job_id=jid, ttl_s=7 * 24 * 3600)
                            if (github_api and jid)
                            else None
                        )
                        if isinstance(job_det, dict):
                            step_snip = step_window_snippet_from_cached_raw_log(
                                job=job_det, step_name=str(sub_name), raw_log_path=raw_path_for_sub
                            )
                except Exception:
                    step_snip = ""
                # If we successfully attributed a snippet to a failing step, avoid duplicating it on the parent.
                if (step_snip or "").strip():
                    node.error_snippet_text = ""
                node.children.append(
                    CIJobNode(
                        job_id=sub_id,
                        display_name="",
                        status=str(sub_status or "unknown"),
                        duration=str(sub_dur or ""),
                        log_url=job_url,
                        is_required=False,  # Steps/children are never marked as required - only parent jobs
                        children=[],
                    )
                )
        except Exception:
            pass

        out.append(node)

    # Validation gate: if a failed GitHub Actions job has no local raw log, fail generation
    # (unless the caller explicitly disables it).
    if validate_raw_logs and (not skip_fetch) and missing_failed_raw_logs:
        examples = "; ".join([f"{n} -> {u}" for (n, u) in missing_failed_raw_logs[:8]])
        raise RawLogValidationError(
            f"Missing [cached raw log] for {len(missing_failed_raw_logs)} failed GitHub Actions job(s): {examples}"
        )

    # NOTE: Sorting, failure marking, and expansion are handled by the centralized
    # pipeline in PRStatusNode.to_tree_vm() (via run_all_passes in common_dashboard_lib.py)

    # If CI failed and we can identify a GitHub Actions run_id, include an explicit restart link.
    #
    # This is shown as a sibling of the checks list, so you can click straight into the workflow run page
    # (and/or copy a `gh run rerun ... --failed` command).
    if any_failed and rerun_run_id:
        try:
            out.append(
                RerunLinkNode(
                    label="",
                    url=f"https://github.com/{DYNAMO_REPO_SLUG}/actions/runs/{rerun_run_id}",
                    run_id=rerun_run_id,
                )
            )
        except Exception:
            pass

    return out


# ======================================================================
# PRStatusNode (moved from show_local_branches.py)
# ======================================================================

class PRStatusNode(BranchNode):
    """PR status information node"""
    
    def __init__(
        self,
        *,
        label: str = "",
        pr: Optional[PRInfo] = None,
        github_api: Optional[GitHubAPIClient] = None,
        refresh_checks: bool = False,
        branch_commit_dt: Optional[datetime] = None,
        allow_fetch_checks: bool = True,
        context_key: str = "",
        children: Optional[List[BranchNode]] = None,
    ):
        super().__init__(label=label, children=children, expanded=False)
        self.pr = pr
        self.github_api = github_api
        self.refresh_checks = refresh_checks
        self.branch_commit_dt = branch_commit_dt
        self.allow_fetch_checks = allow_fetch_checks
        self.context_key = context_key

    def _format_content(self) -> str:
        if not self.pr:
            return ""
        status_parts = []

        if self.pr.ci_status:
            ci_icon = "✅" if self.pr.ci_status == "passed" else "❌" if self.pr.ci_status == "failed" else "⏳"

            # If CI failed, show required vs optional failure counts (based on branch protection required checks).
            if self.pr.ci_status == "failed":
                failed = list(getattr(self.pr, "failed_checks", []) or [])
                req_failed = sum(1 for c in failed if getattr(c, "is_required", False))
                opt_failed = sum(1 for c in failed if not getattr(c, "is_required", False))

                if req_failed > 0 and opt_failed > 0:
                    status_parts.append(f"CI: {ci_icon} required failed, {opt_failed} (not-required) failed")
                elif req_failed > 0:
                    status_parts.append(f"CI: {ci_icon} required failed")
                elif opt_failed > 0:
                    status_parts.append(f"CI: {ci_icon} {opt_failed} (not-required) failed")
                else:
                    status_parts.append(f"CI: {ci_icon} failed")
            else:
                status_parts.append(f"CI: {ci_icon} {self.pr.ci_status}")

        if self.pr.review_decision == 'APPROVED':
            status_parts.append("Review: ✅ Approved")
        elif self.pr.review_decision == 'CHANGES_REQUESTED':
            status_parts.append("Review: 🔴 Changes Requested")

        if self.pr.unresolved_conversations > 0:
            status_parts.append(f"💬 Unresolved: {self.pr.unresolved_conversations}")

        if status_parts:
            return f"{', '.join(status_parts)}"
        return ""

    def _format_html_content(self) -> str:
        base_html = self._format_content()

        # Add expandable "Show checks" button for PRs
        if self.pr and self.pr.number:
            import html as html_module
            import uuid

            try:
                pr_state_lc = (getattr(self.pr, "state", "") or "").lower()
                is_closed = bool(getattr(self.pr, "is_merged", False)) or (pr_state_lc and pr_state_lc != "open")
                ttl_s = GitHubAPIClient.compute_checks_cache_ttl_s(
                    self.branch_commit_dt,
                    refresh=bool(self.refresh_checks),
                )
                # For closed/merged PRs, default stable TTL (30d) is already long; no need to override here.

                # NEW APPROACH: Compute summary directly from our CI children nodes (the actual rendered tree).
                # This ensures the tooltip always matches what's displayed.
                tree_sum = self.compute_summary_from_children()
                tree_counts = tree_sum.get("counts", {})
                tree_names = tree_sum.get("names", {})

                counts = {
                    "success_required": int(tree_counts.get("success_required", 0)),
                    "success_optional": int(tree_counts.get("success_optional", 0)),
                    "failure_required": int(tree_counts.get("failure_required", 0)),
                    "failure_optional": int(tree_counts.get("failure_optional", 0)),
                    "in_progress": int(tree_counts.get("in_progress", 0)),
                    "pending": int(tree_counts.get("pending", 0)),
                    "cancelled": int(tree_counts.get("cancelled", 0)),
                    "other": int(tree_counts.get("other", 0)),
                }

                passed_required_jobs = list(tree_names.get("success_required", []))
                passed_optional_jobs = list(tree_names.get("success_optional", []))
                failed_required_jobs = list(tree_names.get("failure_required", []))
                failed_optional_jobs = list(tree_names.get("failure_optional", []))
                progress_required_jobs = list(tree_names.get("in_progress_required", []))
                progress_optional_jobs = list(tree_names.get("in_progress_optional", []))
                pending_jobs = list(tree_names.get("pending", []))
                cancelled_jobs = list(tree_names.get("cancelled", []))
                other_jobs = list(tree_names.get("other", []))

                # Rebuild the "Status:" line for HTML using the shared compact renderer
                # so it matches the GitHub column in commit history.
                ci_summary_html = compact_ci_summary_html(
                    success_required=int(counts["success_required"]),
                    success_optional=int(counts["success_optional"]),
                    failure_required=int(counts["failure_required"]),
                    failure_optional=int(counts["failure_optional"]),
                    in_progress_required=int(counts["in_progress"]),
                    in_progress_optional=0,
                    pending=int(counts["pending"]),
                    cancelled=int(counts["cancelled"]),
                )

                # Tooltip HTML (match show_commit_history look/labels).
                tooltip_parts: list[str] = []
                if passed_required_jobs:
                    tooltip_parts.append(
                        f'<strong style="color: #2da44e;">{status_icon_html(status_norm="success", is_required=True)} Passed (required):</strong> '
                        + ", ".join(sorted(html_module.escape(n) for n in passed_required_jobs))
                    )
                if passed_optional_jobs:
                    tooltip_parts.append(
                        f'<strong style="color: #2da44e;">{status_icon_html(status_norm="success", is_required=False)} Passed (optional):</strong> '
                        + ", ".join(sorted(html_module.escape(n) for n in passed_optional_jobs))
                    )
                if failed_required_jobs:
                    tooltip_parts.append(
                        f'<strong style="color: #c83a3a;">{status_icon_html(status_norm="failure", is_required=True)} Failed (required):</strong> '
                        + ", ".join(sorted(html_module.escape(n) for n in failed_required_jobs))
                    )
                if failed_optional_jobs:
                    tooltip_parts.append(
                        f'<strong style="color: #c83a3a;">{status_icon_html(status_norm="failure", is_required=False)} Failed (optional):</strong> '
                        + ", ".join(sorted(html_module.escape(n) for n in failed_optional_jobs))
                    )
                if progress_required_jobs:
                    tooltip_parts.append(
                        '<strong style="color: #8c959f;">⏳ In Progress (required):</strong> '
                        + ", ".join(sorted(html_module.escape(n) for n in progress_required_jobs))
                    )
                if progress_optional_jobs:
                    tooltip_parts.append(
                        '<strong style="color: #8c959f;">⏳ In Progress (optional):</strong> '
                        + ", ".join(sorted(html_module.escape(n) for n in progress_optional_jobs))
                    )
                if pending_jobs:
                    tooltip_parts.append(
                        '<strong style="color: #8c959f;">⏸ Pending:</strong> '
                        + ", ".join(sorted(html_module.escape(n) for n in pending_jobs))
                    )
                if cancelled_jobs:
                    tooltip_parts.append(
                        '<strong style="color: #8c959f;">✖️ Canceled:</strong> '
                        + ", ".join(sorted(html_module.escape(n) for n in cancelled_jobs))
                    )
                if other_jobs:
                    tooltip_parts.append(
                        '<strong style="color: #8c959f;">Other:</strong> '
                        + ", ".join(sorted(html_module.escape(n) for n in other_jobs))
                    )
                tooltip_html = "<br>".join(tooltip_parts)

                status_parts = []
                if ci_summary_html:
                    ci_summary = str(ci_summary_html)
                    if tooltip_html:
                        ci_summary = (
                            '<span class="gh-status-tooltip" style="margin-left: 2px;">'
                            f'<span style="white-space: nowrap; font-weight: 600; font-size: 12px;">{ci_summary}</span>'
                            f'<span class="tooltiptext">{tooltip_html}</span>'
                            '</span>'
                        )
                    # Roll up the top-level PR status:
                    # - FAILED iff a REQUIRED check failed
                    # - WARN if only non-required checks failed
                    # - RUNNING if anything is in progress/pending
                    # - PASSED otherwise
                    #
                    # IMPORTANT: determining "required" purely from branch protection can be unreliable
                    # (branch protection is often not accessible). Use PRInfo.failed_checks' `is_required`
                    # flags for the FAIL decision.
                    # Top-level pill should reflect REQUIRED checks only:
                    # - FAILED iff a required check failed
                        # - RUNNING if anything is in progress/pending
                        # - PASSED otherwise (even if optional checks failed)
                        if counts["failure_required"] > 0:
                            ci_label = '<span class="status-indicator status-failed">FAILED</span>'
                        elif counts["in_progress"] > 0 or counts["pending"] > 0:
                            ci_label = '<span class="status-indicator status-building">RUNNING</span>'
                        else:
                            ci_label = '<span class="status-indicator status-success">PASSED</span>'

                        # Replace the literal "CI:" with a clickable GitHub icon (links to commit checks page).
                        checks_link = ""
                        try:
                            head_sha = getattr(self.pr, "head_sha", None)
                            if head_sha:
                                checks_url = f"https://github.com/{DYNAMO_REPO_SLUG}/commit/{head_sha}/checks"
                                checks_link = (
                                    f' <a href="{checks_url}" target="_blank" '
                                    f'style="text-decoration: none; color: #24292f; font-weight: 600;" '
                                    f'title="Open GitHub checks for this commit">'
                                    f'<svg height="14" width="14" viewBox="0 0 16 16" fill="currentColor" '
                                    f'style="display: inline-block; vertical-align: middle;">'
                                    f'<path fill-rule="evenodd" clip-rule="evenodd" '
                                    f'd="M8 0C3.58 0 0 3.58 0 8C0 11.54 2.29 14.53 5.47 15.59C5.87 15.66 6.02 15.42 6.02 15.21C6.02 15.02 6.01 14.39 6.01 13.72C4 14.09 3.48 13.23 3.32 12.78C3.23 12.55 2.84 11.84 2.5 11.65C2.22 11.5 1.82 11.13 2.49 11.12C3.12 11.11 3.57 11.7 3.72 11.94C4.44 13.15 5.59 12.81 6.05 12.6C6.12 12.08 6.33 11.73 6.56 11.53C4.78 11.33 2.92 10.64 2.92 7.58C2.92 6.71 3.23 5.99 3.74 5.43C3.66 5.23 3.38 4.41 3.82 3.31C3.82 3.31 4.49 3.1 6.02 4.13C6.66 3.95 7.34 3.86 8.02 3.86C8.7 3.86 9.38 3.95 10.02 4.13C11.55 3.09 12.22 3.31 12.22 3.31C12.66 4.41 12.38 5.23 12.3 5.43C12.81 5.99 13.12 6.7 13.12 7.58C13.12 10.65 11.25 11.33 9.47 11.53C9.76 11.78 10.01 12.26 10.01 13.01C10.01 14.08 10 14.94 10 15.21C10 15.42 10.15 15.67 10.55 15.59C13.71 14.53 16 11.53 16 8C16 3.58 12.42 0 8 0Z"/>'
                                    f'</svg>'
                                    f'</a>'
                                )
                        except Exception:
                            checks_link = ""

                        # Put PASS/FAIL first so the line reads "PASS [icon] <counts>".
                        status_parts.append(f"{ci_label}{checks_link} {ci_summary}")

                    if self.pr.review_decision == 'APPROVED':
                        status_parts.append("Review: ✅ Approved")
                    elif self.pr.review_decision == 'CHANGES_REQUESTED':
                        status_parts.append("Review: 🔴 Changes Requested")

                    if self.pr.unresolved_conversations > 0:
                        status_parts.append(f"💬 Unresolved: {self.pr.unresolved_conversations}")

                    base_html = f"{', '.join(status_parts)}" if status_parts else ""

            except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                # Silently fail if gh command is not available or times out
                pass

        return base_html

    def compute_summary_from_children(self) -> dict:
        """Compute CI summary directly from child CIJobNode objects.
        
        This ensures the tooltip reflects the actual rendered tree, not cached API data.
        Counts only immediate children (top-level jobs), not nested children (steps).
        
        Returns a dict with 'counts' and 'names' matching GitHubChecksSummary structure.
        """
        counts = {
            "success_required": 0,
            "success_optional": 0,
            "failure_required": 0,
            "failure_optional": 0,
            "in_progress": 0,
            "pending": 0,
            "cancelled": 0,
            "other": 0,
        }
        
        names = {
            "success_required": [],
            "success_optional": [],
            "failure_required": [],
            "failure_optional": [],
            "in_progress_required": [],
            "in_progress_optional": [],
            "pending": [],
            "cancelled": [],
            "other": [],
        }
        
        # Only count immediate children (top-level jobs), not recurse into steps
        for child in (self.children or []):
            if not isinstance(child, CIJobNode):
                continue
                
            name = str(getattr(child, "job_id", "") or getattr(child, "display_name", "") or "").strip()
            status = str(getattr(child, "status", "unknown")).strip().lower()
            is_req = bool(getattr(child, "is_required", False))
            
            if status == "success":
                if is_req:
                    counts["success_required"] += 1
                    if name:
                        names["success_required"].append(name)
                else:
                    counts["success_optional"] += 1
                    if name:
                        names["success_optional"].append(name)
            elif status == "failure":
                if is_req:
                    counts["failure_required"] += 1
                    if name:
                        names["failure_required"].append(name)
                else:
                    counts["failure_optional"] += 1
                    if name:
                        names["failure_optional"].append(name)
            elif status in ("in_progress", "running"):
                counts["in_progress"] += 1
                if is_req:
                    if name:
                        names["in_progress_required"].append(name)
                else:
                    if name:
                        names["in_progress_optional"].append(name)
            elif status == "pending":
                counts["pending"] += 1
                if name:
                    names["pending"].append(name)
            elif status in ("cancelled", "canceled"):
                counts["cancelled"] += 1
                if name:
                    names["cancelled"].append(name)
            else:
                counts["other"] += 1
                if name:
                    names["other"].append(name)
        
        return {"counts": counts, "names": names}

    def to_tree_vm(self) -> TreeNodeVM:
        """Show the PASSED/FAILED/RUNNING status line, collapsed for PASSED.
        
        Policy: collapse when PASSED (no required failures), expand when FAILED (has required failures).
        We ignore RUNNING state and optional failures - if it displays "PASSED", it collapses.
        """
        # Sort children alphabetically by display name.
        # Special nodes (RerunLinkNode, ConflictWarningNode, BlockedMessageNode) always stay at the end.
        def _sort_key_for_ci_node(node: BranchNode) -> tuple:
            """Returns (is_special, display_name_lower).
            
            Special nodes (like RerunLinkNode) get priority=1 to sort last.
            Regular CI nodes get priority=0 and sort alphabetically by display name.
            """
            # Check if this is a special node that should always be last
            node_type = type(node).__name__
            is_special = node_type in ('RerunLinkNode', 'ConflictWarningNode', 'BlockedMessageNode')
            
            if is_special:
                return (1, "")  # Special nodes sort last
            
            # Get the display name for sorting
            if isinstance(node, CIJobNode):
                # Use display_name if present, otherwise job_id, otherwise label
                name = getattr(node, 'display_name', '') or getattr(node, 'job_id', '') or getattr(node, 'label', '')
            else:
                # For other BranchNode types, use label
                name = getattr(node, 'label', '')
            
            return (0, str(name).lower())
        
        sorted_children = sorted(
            (self.children or []),
            key=_sort_key_for_ci_node
        )
        
        # Get PR info and GitHub API client for building CI nodes
        pr = getattr(self, "pr", None)
        gh = getattr(self, "github_api", None)
        pr_number = int(getattr(pr, "number", 0)) if pr else None
        
        # Build CI nodes if we don't have any children yet (common case)
        # This happens when PRStatusNode is created without pre-built CI children
        if not sorted_children and pr and pr_number:
            try:
                from pathlib import Path
                
                # Determine repo_root
                repo_root = Path("/home/keivenc/dynamo/dynamo_latest")  # Fallback
                try:
                    if gh and hasattr(gh, "repo_root"):
                        repo_root = Path(gh.repo_root)
                except Exception:
                    pass
                
                # Get cache settings
                page_root_dir = None
                checks_ttl_s = 300
                skip_fetch = False
                try:
                    if hasattr(self, "refresh_checks"):
                        from common import GitHubAPIClient
                        checks_ttl_s = int(GitHubAPIClient.compute_checks_cache_ttl_s(None, refresh=bool(self.refresh_checks)))
                    if hasattr(self, "allow_fetch_checks"):
                        skip_fetch = not bool(self.allow_fetch_checks)
                except Exception:
                    pass
                
                # Build CI nodes based on PR number
                if pr_number >= 100000000:
                    # Mock PR number (>= 100000000) = dummy PR, use mock
                    print(f"[PRStatusNode] Building mock CI nodes for mock PR #{pr_number}")
                    ci_nodes = mock_build_ci_nodes(pr, repo_root, page_root_dir=page_root_dir)
                elif pr_number > 0 and gh:
                    # Real PR, use real API
                    print(f"[PRStatusNode] Building real CI nodes for PR #{pr_number}")
                    ci_nodes = build_ci_nodes_from_pr(
                        pr, gh, repo_root,
                        page_root_dir=page_root_dir,
                        checks_ttl_s=checks_ttl_s,
                        skip_fetch=skip_fetch,
                        validate_raw_logs=True,
                    )
                else:
                    ci_nodes = []
                
                # Add built CI nodes to sorted_children AND update self.children so compute_summary_from_children() can see them
                sorted_children = list(ci_nodes)
                self.children = sorted_children  # Update self.children so the summary computation can access them
                print(f"[PRStatusNode] Built {len(sorted_children)} CI nodes")
            except Exception as e:
                print(f"[PRStatusNode] Error building CI nodes: {e}")
                import traceback
                traceback.print_exc()
        
        # Apply centralized CI tree processing pipeline
        # Pass the BranchNode objects directly (not TreeNodeVM yet)
        kids: List[TreeNodeVM] = []
        try:
            from pathlib import Path
            from common_dashboard_lib import run_all_passes
            
            # Determine repo_root for workflow parsing
            repo_root = Path("/home/keivenc/dynamo/dynamo_latest")  # Fallback
            try:
                if gh and hasattr(gh, "repo_root"):
                    repo_root = Path(gh.repo_root)
            except Exception:
                pass
            
            # Filter out special nodes for pipeline processing
            ci_branch_nodes = [
                child for child in sorted_children
                if isinstance(child, BranchNode) and type(child).__name__ not in ('RerunLinkNode', 'ConflictWarningNode', 'BlockedMessageNode')
            ]
            
            print(f"[PRStatusNode] Passing {len(ci_branch_nodes)} BranchNodes to pipeline")
            
            kids = run_all_passes(
                ci_nodes=ci_branch_nodes,  # Pass List[BranchNode] directly!
                repo_root=repo_root,
            )
        except Exception as e:
            print(f"[PRStatusNode] Error in pipeline: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: convert children manually
            kids = [child.to_tree_vm() for child in sorted_children if isinstance(child, BranchNode)]
        
        # Add back special nodes at the end
        for child in sorted_children:
            node_type = type(child).__name__
            is_special = node_type in ('RerunLinkNode', 'ConflictWarningNode', 'BlockedMessageNode')
            if is_special and isinstance(child, BranchNode):
                kids.append(child.to_tree_vm())
        
        # Determine if we should expand by checking for required failures.
        # We need to check BOTH pr.failed_checks (from GitHub API) AND the actual children
        # (from build_ci_nodes_from_pr) because pr.failed_checks may be empty in some cases.
        pr = getattr(self, "pr", None)
        pr_num = getattr(pr, "number", None)
        
        # Check pr.failed_checks first (from GitHub API)
        required_failed_from_api = any(
            bool(getattr(fc, "is_required", False))
            for fc in (getattr(pr, "failed_checks", None) or [])
        )
        
        # Also check children (CIJobNode objects) for required failures
        required_failed_from_children = any(
            isinstance(child, CIJobNode) 
            and getattr(child, "status", "") == "failure" 
            and getattr(child, "is_required", False)
            for child in sorted_children
        )
        
        # Expand if EITHER source shows required failures
        auto_expand_checks = required_failed_from_api or required_failed_from_children
        
        # If we have required failures but no child nodes (cache-only run),
        # inject a placeholder so the triangle still renders
        if (not kids) and bool(auto_expand_checks):
            kids = [
                TreeNodeVM(
                    node_key=f"PRStatus-empty:{self.context_key}:{getattr(pr, 'number', '')}",
                    label_html='<span style="color: #57606a; font-size: 12px;">(no check details cached)</span>',
                    children=[],
                    collapsible=False,
                    default_expanded=False,
                )
            ]

        return TreeNodeVM(
            node_key=f"PRStatus:{self.context_key}:{getattr(pr, 'number', '')}",
            label_html=self._format_html_content(),
            children=kids,
            collapsible=bool(kids),
            default_expanded=bool(auto_expand_checks),
            triangle_tooltip=None,
        )

    def render_html(self, prefix: str = "", is_last: bool = True, is_root: bool = True) -> List[str]:
        """Render the PASS/FAIL status line with an expandable CI hierarchy subtree."""
        lines: List[str] = []

        if not is_root:
            connector = "└─" if is_last else "├─"
            current_prefix = prefix + connector + " "
        else:
            current_prefix = ""

        has_children = bool(self.children)

        # Expand by default only when something needs attention in the CI subtree:
        # - required failures
        # - any running/pending descendants
        # NOTE: helper nodes (e.g. rerun link) should not force expansion.
        default_expanded = any(
            CIJobNode._subtree_needs_attention(c)
            for c in (self.children or [])
            if isinstance(c, CIJobNode)
        )

        # Unique DOM id per render occurrence.
        pr_num = str(getattr(self.pr, "number", "") or "")
        dom_hash = hashlib.sha1((prefix + "|PRSTATUS|" + pr_num).encode("utf-8")).hexdigest()[:10]
        children_id = f"ci_children_{dom_hash}"

        if has_children:
            triangle_char = "▼" if default_expanded else "▶"
            triangle = (
                f'<span style="display: inline-block; width: 12px; margin-right: 2px; color: #0969da; '
                f'cursor: pointer; user-select: none;" '
                f'title="CI tree (flat; mirrors Details)" '
                f'onclick="toggleCiChildren(\'{children_id}\', this)">{triangle_char}</span>'
            )
        else:
            triangle = '<span style="display: inline-block; width: 12px; margin-right: 2px;"></span>'

        line_content = self._format_html_content()
        if line_content.strip():
            lines.append(current_prefix + triangle + line_content)

        if has_children and lines:
            disp = "inline" if default_expanded else "none"
            # Hide the newline between parent and first child when collapsed.
            lines[-1] = lines[-1] + f'<span id="{children_id}" style="display: {disp};">'

            child_lines: List[str] = []
            for i, child in enumerate(self.children):
                is_last_child = i == len(self.children) - 1
                if is_root:
                    child_prefix = ""
                else:
                    child_prefix = prefix + ("   " if is_last else "│  ")
                child_lines.extend(child.render_html(child_prefix, is_last_child, False))

            if child_lines:
                child_lines[-1] = child_lines[-1] + "</span>"
                lines.extend(child_lines)
            else:
                lines[-1] = lines[-1] + "</span>"

        return lines




# ======================================================================
# BlockedMessageNode (moved from show_local_branches.py)
# ======================================================================

class BlockedMessageNode(BranchNode):
    """Blocked message node"""

    def _format_content(self) -> str:
        return f"🚫 {self.label}"

    def _format_html_content(self) -> str:
        return self.label


# ======================================================================
# ConflictWarningNode (moved from show_local_branches.py)
# ======================================================================

class ConflictWarningNode(BranchNode):
    """Conflict warning node"""

    def _format_content(self) -> str:
        return f"⚠️  {self.label}"

    def _format_html_content(self) -> str:
        return self.label


# ======================================================================
# RerunLinkNode (moved from show_local_branches.py)
# ======================================================================

class RerunLinkNode(BranchNode):
    """Rerun link node"""
    url: Optional[str] = None
    run_id: Optional[str] = None

    def _format_content(self) -> str:
        if not self.url or not self.run_id:
            return ""
        return f"🔄 Restart: gh run rerun {self.run_id} --repo {DYNAMO_REPO_SLUG} --failed"

    def _format_html_content(self) -> str:
        if not self.url or not self.run_id:
            return ""
        cmd = f"gh run rerun {self.run_id} --repo {DYNAMO_REPO_SLUG} --failed"
        copy_btn = _html_copy_button(clipboard_text=cmd, title="Click to copy rerun command")

        return (
            f'🔄 <a href="{self.url}" target="_blank">Restart failed jobs</a> '
            f'(or: {copy_btn}<code>{cmd}</code>)'
        )


#
# Git repository helper functions (shared with show_remote_branches.py)
#

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
