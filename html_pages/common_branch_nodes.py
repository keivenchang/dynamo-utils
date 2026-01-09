#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared node classes and helper functions for branch/PR dashboards.

This module contains the common tree node classes and utility functions
used by both show_local_branches.py and show_remote_branches.py to avoid
tight coupling and code duplication.

IMPORTANT NOTES:
- PRStatusNode and _build_ci_hierarchy_nodes in this file are INCOMPLETE STUBS
- For production use, import these from show_local_branches.py instead
- show_remote_branches.py correctly imports the complete implementations from show_local_branches.py
- This ensures IDENTICAL rendering logic between local and remote branches

Complete implementations:
- BranchNode, BranchInfoNode, CommitMessageNode, MetadataNode, PRNode, RepoNode
- Helper functions: _format_age_compact, _html_copy_button, generate_html, etc.

Stub implementations (DO NOT USE):
- PRStatusNode (missing status pill generation)
- _build_ci_hierarchy_nodes (missing proper caching and data handling)
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
    process_ci_tree_pipeline,
    render_tree_pre_lines,
    required_badge_html,
    status_icon_html,
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
        f'<button data-clipboard-text="{text_esc}" onclick="copyFromClipboardAttr(this)" '
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
    """Format commit message for use in a title/tooltip attribute."""
    msg = str(commit_message or "").strip()
    if not msg:
        return ""
    # Escape HTML entities and truncate for tooltip
    msg_esc = html_module.escape(msg)
    if len(msg_esc) > 200:
        msg_esc = msg_esc[:200] + "..."
    return msg_esc


def _format_pr_number_link(pr: Optional[PRInfo]) -> str:
    """Format a PR number as a clickable link with icon."""
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


class CIJobTreeNode(BranchNode):
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
    ):
        super().__init__(label="", children=children, expanded=expanded, status=status)
        self.job_id = str(job_id or "")
        self.display_name = str(display_name or "")
        self.duration = str(duration or "")
        self.log_url = str(log_url or "")
        self.is_required = bool(is_required)
        self.page_root_dir = page_root_dir
        self.context_key = str(context_key or "")
    
    def to_tree_vm(self) -> TreeNodeVM:
        """Convert this CI job node to a TreeNodeVM using check_line_html."""
        # Materialize raw log if available
        raw_log_href = ""
        raw_log_size_bytes = 0
        error_snippet_text = ""
        
        if self.log_url and self.page_root_dir:
            try:
                raw_log_local = materialize_job_raw_log_text_local_link(
                    page_root_dir=self.page_root_dir,
                    log_url=self.log_url,
                    context_key=self.context_key,
                )
                if raw_log_local and raw_log_local.get("local_path"):
                    local_path_str = str(raw_log_local.get("local_path") or "")
                    if local_path_str:
                        raw_log_href = str(raw_log_local.get("relative_url") or "")
                        log_path_obj = Path(local_path_str)
                        if log_path_obj.exists():
                            raw_log_size_bytes = log_path_obj.stat().st_size
                            # Extract error snippet
                            if extract_error_snippet_from_log_file:
                                snippet = extract_error_snippet_from_log_file(log_path_obj)
                                if snippet and snippet.strip():
                                    error_snippet_text = snippet.strip()
            except Exception:
                pass
        
        # Use shared check_line_html to generate the HTML
        label_html = check_line_html(
            job_id=self.job_id,
            display_name=self.display_name,
            status_norm=self.status,
            is_required=self.is_required,
            duration=self.duration,
            log_url=self.log_url,
            raw_log_href=raw_log_href,
            raw_log_size_bytes=raw_log_size_bytes,
            error_snippet_text=error_snippet_text,
        )
        
        kids = [c.to_tree_vm() for c in (self.children or [])]
        return TreeNodeVM(
            node_key=f"ci:{self.job_id}:{id(self)}",
            label_html=label_html,
            children=kids,
            collapsible=True,
            default_expanded=self.expanded,
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
        commit_url: Optional[str] = None,
        commit_time_pt: Optional[str] = None,
        commit_datetime: Optional[datetime] = None,
        commit_message: Optional[str] = None,
        created_at: Optional[datetime] = None,
        children: Optional[List[BranchNode]] = None,
    ):
        super().__init__(label=label, children=children, expanded=False, is_current=is_current)
        self.sha = sha
        self.commit_url = commit_url
        self.commit_time_pt = commit_time_pt
        self.commit_datetime = commit_datetime
        self.commit_message = commit_message
        self.created_at = created_at
    
    def to_tree_vm(self) -> TreeNodeVM:
        """Convert to TreeNodeVM with structured children (commit message + metadata + other)."""
        # Construct the main branch line HTML
        parts = []
        
        # Copy button
        clipboard_text = _strip_repo_prefix_for_clipboard(self.label)
        parts.append(_html_copy_button(clipboard_text=clipboard_text, title=f"Copy branch name: {clipboard_text}"))
        
        # Branch name (bold if current)
        if self.is_current:
            parts.append(f'<span style="font-weight: 600;">{html_module.escape(self.label)}</span>')
        else:
            parts.append(html_module.escape(self.label))
        
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
        
        # Build children: commit message, metadata, then other children
        kids = []
        
        # 1. Commit message (if available)
        if self.commit_message:
            kids.append(CommitMessageNode(label=self.commit_message, pr=None).to_tree_vm())
        
        # 2. Metadata (modified, created, age)
        metadata_suffix = _format_branch_metadata_suffix(
            commit_time_pt=self.commit_time_pt,
            commit_datetime=self.commit_datetime,
            created_at=self.created_at,
        )
        if metadata_suffix:
            kids.append(MetadataNode(label=metadata_suffix).to_tree_vm())
        
        # 3. Other children
        for c in (self.children or []):
            kids.append(c.to_tree_vm())
        
        return TreeNodeVM(
            node_key=f"branch_info:{self.label}:{id(self)}",
            label_html=label_html,
            children=kids,
            collapsible=True,  # Make branches collapsible like local branches
            default_expanded=True,  # Default to expanded
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
        
        # PR number link (if available)
        if self.pr:
            pr_link = _format_pr_number_link(self.pr)
            if pr_link:
                parts.append(pr_link)
        
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


class PRStatusNode(BranchNode):
    """PR status node that builds CI hierarchy from GitHub checks."""
    
    def __init__(
        self,
        label: str,
        *,
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
    
    def to_tree_vm(self) -> TreeNodeVM:
        """Convert to TreeNodeVM, building CI hierarchy from children."""
        # The children should already be CI job nodes added by the caller
        # We need to convert them to TreeNodeVM and apply the centralized pipeline
        
        # Get repo path for workflow YAML parsing
        # Try to get from self.pr.repo_path or use default
        repo_path = Path("/home/keivenc/dynamo/dynamo_latest")  # Default fallback
        try:
            pr = getattr(self, "pr", None)
            if pr and hasattr(pr, "repo_path"):
                repo_path = Path(pr.repo_path)
        except Exception:
            pass
        
        # Build node_items list for the pipeline
        node_items = []
        for child in (self.children or []):
            if isinstance(child, CIJobTreeNode):
                # Get the job name (without kind prefix for grouping)
                job_name = child.job_id
                # Try to extract the base name (after the prefix)
                match = re.search(r"^(?:test|build|lint|deploy|check):\s*(.+)$", job_name)
                if match:
                    job_name = match.group(1).strip()
                
                # Convert to TreeNodeVM
                vm = child.to_tree_vm()
                node_items.append((job_name, vm))
            else:
                # For non-CI nodes, just convert to TreeNodeVM
                node_items.append((child.label, child.to_tree_vm()))
        
        # Apply the centralized pipeline
        kids = process_ci_tree_pipeline(
            nodes=[],
            repo_root=repo_path,
            node_items=node_items,
        )
        
        # Determine expansion state based on CI status
        should_expand = False
        if self.pr:
            # Check if there are any required failures
            has_required_failure = False
            try:
                failed_checks = getattr(self.pr, "failed_checks", None) or []
                has_required_failure = any(getattr(fc, "required", False) for fc in failed_checks)
            except Exception:
                pass
            
            # Get rollup status
            rollup_status = str(getattr(self.pr, "rollup_status", "") or "").strip()
            if not rollup_status:
                rollup_status = CIStatus.UNKNOWN
            
            should_expand = ci_should_expand_by_default(
                rollup_status=rollup_status,
                has_required_failure=has_required_failure,
            )
        
        return TreeNodeVM(
            node_key=f"pr_status:{id(self)}",
            label_html="",  # Status nodes typically have no label
            children=kids,
            collapsible=True,
            default_expanded=should_expand,
        )


#
# CI hierarchy builder (shared by both local and remote)
#

def _build_ci_hierarchy_nodes(
    repo_path: Path,
    pr: PRInfo,
    *,
    github_api: GitHubAPIClient,
    page_root_dir: Path,
    checks_ttl_s: int,
    skip_fetch: bool = False,
    context_key: str = "",
) -> List[CIJobTreeNode]:
    """
    Build CI hierarchy nodes from PR check runs.
    
    Returns a list of CIJobTreeNode objects representing the CI jobs for this PR.
    The caller is responsible for adding these to a PRStatusNode and applying the
    centralized pipeline.
    """
    nodes = []
    
    if skip_fetch:
        return nodes
    
    # Fetch check runs for this PR
    try:
        check_rows = github_api.get_pr_check_runs(
            owner=DYNAMO_OWNER,
            repo=DYNAMO_REPO,
            pr_number=int(getattr(pr, "number", 0) or 0),
            ttl_seconds=checks_ttl_s,
        )
    except Exception:
        return nodes
    
    # Build CI job nodes from check runs
    for row in (check_rows or []):
        try:
            job_name = str(getattr(row, "name", "") or "").strip()
            if not job_name:
                continue
            
            # Classify the CI kind (test, build, lint, etc.)
            kind = classify_ci_kind(job_name)
            
            # Build display name (with kind prefix if not "check")
            display_name = disambiguate_check_run_name(job_name)
            job_id = display_name
            if kind and kind != "check":
                job_id = f"{kind}: {display_name}"
            
            # Status
            status_str = str(getattr(row, "status", "") or "").lower()
            conclusion = str(getattr(row, "conclusion", "") or "").lower()
            if conclusion == "success":
                status = CIStatus.SUCCESS
            elif conclusion == "failure":
                status = CIStatus.FAILED
            elif conclusion in {"cancelled", "skipped"}:
                status = CIStatus.SKIPPED
            elif status_str == "in_progress":
                status = CIStatus.IN_PROGRESS
            else:
                status = CIStatus.UNKNOWN
            
            # Duration
            duration_str = ""
            try:
                duration_s = int(getattr(row, "duration_seconds", 0) or 0)
                if duration_s > 0:
                    if duration_s < 60:
                        duration_str = f"{duration_s}s"
                    else:
                        duration_str = f"{duration_s // 60}m {duration_s % 60}s"
            except Exception:
                pass
            
            # Log URL
            log_url = str(getattr(row, "html_url", "") or "").strip()
            
            # Required
            is_required = bool(getattr(row, "required", False))
            
            # Create CI job node
            node = CIJobTreeNode(
                job_id=job_id,
                display_name=display_name,
                status=status,
                duration=duration_str,
                log_url=log_url,
                is_required=is_required,
                page_root_dir=page_root_dir,
                context_key=context_key,
            )
            nodes.append(node)
        except Exception:
            continue
    
    return nodes


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
    
    Returns:
        HTML string
    """
    if not HAS_JINJA2:
        raise RuntimeError("Jinja2 is required for HTML generation")
    
    # Get current time in both UTC and PDT
    from datetime import datetime
    from zoneinfo import ZoneInfo
    now_utc = datetime.now(ZoneInfo('UTC'))
    now_pdt = datetime.now(ZoneInfo('America/Los_Angeles'))
    
    # Format timestamps
    utc_str = now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')
    pdt_str = now_pdt.strftime('%Y-%m-%d %H:%M:%S %Z')
    
    # Render the tree if not provided
    if tree_html_override is None:
        tree_html_override = "\n".join(render_tree_pre_lines([root.to_tree_vm()]))
    
    # Load template
    template_dir = Path(__file__).resolve().parent
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html", "j2"]),
    )
    template = env.get_template("show_local_branches.j2")
    
    # Import status icon helper from common_dashboard_lib
    from common_dashboard_lib import status_icon_html
    
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
