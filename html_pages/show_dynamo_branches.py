#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Show dynamo branches with PR information using a node-based tree structure.
Supports parallel data gathering for improved performance.
"""

import argparse
import hashlib
import html
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional
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

# Shared UI snippets (keep styling consistent with show_commit_history)
from html_ui import GH_STATUS_TOOLTIP_CSS, GH_STATUS_TOOLTIP_JS, PASS_PLUS_STYLE

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
from common import FailedCheck, GitHubAPIClient, PRInfo


#
# Small HTML helpers
#

_COPY_ICON_SVG = (
    '<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor" style="display: inline-block;">'
    '<path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path>'
    '<path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>'
    "</svg>"
)

_COPY_BTN_STYLE = (
    "padding: 4px 6px; font-size: 11px; background-color: transparent; color: #57606a; "
    "border: 1px solid #d0d7de; border-radius: 6px; cursor: pointer; display: inline-flex; "
    "align-items: center; vertical-align: middle; margin-right: 6px;"
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


def _html_toggle_span(*, target_id: str, show_text: str, hide_text: str) -> str:
    """Return a small inline toggle span that flips ▶/▼ and toggles display of target div."""
    # Note: keep JS and quoting simple because this is emitted inside an HTML attribute.
    target_id_escaped = html.escape(target_id, quote=True)
    show_escaped = html.escape(show_text, quote=True)
    hide_escaped = html.escape(hide_text, quote=True)
    return (
        f'<span style="cursor: pointer; color: #0969da; margin-left: 10px; font-weight: 500;" '
        f'onclick="var el=document.getElementById(\'{target_id_escaped}\');'
        f'var isHidden=(el.style.display===\'none\'||el.style.display===\'\');'
        f'el.style.display=isHidden?\'block\':\'none\';'
        f'this.textContent=isHidden?\'{hide_escaped}\':\'{show_escaped}\';"'
        f">{show_escaped}</span>"
    )


def _html_small_link(*, url: str, label: str) -> str:
    url_escaped = html.escape(url, quote=True)
    label_escaped = html.escape(label)
    return (
        f' <a href="{url_escaped}" target="_blank" '
        f'style="color: #0969da; font-size: 11px; margin-left: 5px; text-decoration: none;">{label_escaped}</a>'
    )


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
        # Example child: "   └─ 📖 PR#3676: feat: add TensorRT-LLM Prometheus metrics support"
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
        # Example child: "   └─ 📖 PR#3676: feat: add TensorRT-LLM Prometheus metrics support"
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

        # Gray out branch name if its PR is already merged.
        # (BranchInfoNode children include PRNode nodes when present.)
        is_merged_branch = any(
            (getattr(ch, "pr", None) is not None) and bool(getattr(getattr(ch, "pr", None), "is_merged", False))
            for ch in (self.children or [])
        )

        copy_btn = _html_copy_button(clipboard_text=self.label, title="Click to copy branch name")

        cls = "current" if self.is_current else ""
        if is_merged_branch:
            cls = (cls + " merged-branch").strip()

        if cls:
            return f'{copy_btn}<span class="{cls}">{self.label}</span>{sha_str}{marker}'
        return f'{copy_btn}{self.label}{sha_str}{marker}'


@dataclass
class PRNode(BranchNode):
    """Pull request node"""
    pr: Optional[PRInfo] = None

    def _format_content(self) -> str:
        if not self.pr:
            return ""
        state_lc = (self.pr.state or "").lower()
        if self.pr.is_merged:
            emoji = '🔀'
        elif state_lc == 'open':
            emoji = '📖'
        else:
            emoji = '❌'

        # Truncate title at 80 characters
        title = self.pr.title[:80] + '...' if len(self.pr.title) > 80 else self.pr.title

        # Add base branch and created date
        metadata = []
        if self.pr.base_ref:
            metadata.append(f"→ {self.pr.base_ref}")
        if self.pr.created_at:
            # Parse and format date (ISO format: 2025-11-12T17:15:32Z)
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(self.pr.created_at.replace('Z', '+00:00'))
                date_str = dt.strftime('%Y-%m-%d %H:%M')
                metadata.append(f"created {date_str}")
            except:
                pass

        metadata_str = f" ({', '.join(metadata)})" if metadata else ""
        # Prefer standard GitHub-style formatting: "title (#1234)" over "PR#1234: title"
        pr_suffix = f"(#{self.pr.number})"
        if pr_suffix not in title:
            title = f"{title} {pr_suffix}"
        return f"{emoji} {title}{metadata_str}"

    def _format_html_content(self) -> str:
        if not self.pr:
            return ""
        state_lc = (self.pr.state or "").lower()
        if self.pr.is_merged:
            emoji = '🔀'
        elif state_lc == 'open':
            emoji = '📖'
        else:
            emoji = '❌'

        # Truncate title at 80 characters
        title = self.pr.title[:80] + '...' if len(self.pr.title) > 80 else self.pr.title

        # Add base branch and created date
        metadata = []
        if self.pr.base_ref:
            metadata.append(f'<span style="color: #0969da; font-weight: 500;">→ {self.pr.base_ref}</span>')
        if self.pr.created_at:
            # Parse and format date (ISO format: 2025-11-12T17:15:32Z)
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(self.pr.created_at.replace('Z', '+00:00'))
                date_str = dt.strftime('%Y-%m-%d %H:%M')
                metadata.append(f'<span style="color: #666;">created {date_str}</span>')
            except:
                pass

        metadata_str = f" ({', '.join(metadata)})" if metadata else ""

        # Prefer standard GitHub-style formatting: "title (#1234)" over "PR#1234: title"
        pr_suffix = f"(#{self.pr.number})"
        if pr_suffix not in title:
            title = f"{title} {pr_suffix}"

        # Gray out merged PRs
        if self.pr.is_merged:
            return f'<span style="color: #999;">{emoji} <a href="{self.pr.url}" target="_blank" style="color: #999;">{title}</a>{metadata_str}</span>'
        else:
            return f'{emoji} <a href="{self.pr.url}" target="_blank">{title}</a>{metadata_str}'


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
            import subprocess
            import html as html_module
            import uuid

            try:
                # Run gh pr checks to get detailed check information
                result = subprocess.run(
                    ['gh', 'pr', 'checks', str(self.pr.number), '--repo', 'ai-dynamo/dynamo'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                # Display checks output if available (exit code 8 means some checks failed, but output is still useful)
                if result.stdout.strip():
                    # Compact CI summary counts (styled to match show_commit_history.j2):
                    #   - green:  REQ+OPT✓  (passed; required count bold)
                    #   - red:    N✗   (required failures, rendered as a red badge)
                    #   - amber:  N⚠   (optional failures)
                    #   - amber:  N⏳   (in progress)
                    #   - grey:   N⏸   (pending/queued/skipping)
                    #   - grey:   N✖️  (cancelled)
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

                    # Build a hover tooltip (same style as show_commit_history) by collecting
                    # check names into buckets. This lets us distinguish required vs optional passes.
                    passed_required_jobs: list[str] = []
                    passed_optional_jobs: list[str] = []
                    failed_required_jobs: list[str] = []
                    failed_optional_jobs: list[str] = []
                    progress_required_jobs: list[str] = []
                    progress_optional_jobs: list[str] = []
                    pending_jobs: list[str] = []
                    cancelled_jobs: list[str] = []
                    other_jobs: list[str] = []

                    # Parse tab-separated output into a formatted table
                    lines = result.stdout.strip().split('\n')
                    table_html = '<table style="border-collapse: collapse; font-size: 11px; margin-top: 5px;">'
                    table_html += '<tr style="background-color: #e8eaed;"><th style="text-align: left; padding: 4px 8px; border: 1px solid #d0d0d0;">Check Name</th><th style="text-align: left; padding: 4px 8px; border: 1px solid #d0d0d0;">Status</th><th style="text-align: left; padding: 4px 8px; border: 1px solid #d0d0d0;">Duration</th><th style="text-align: left; padding: 4px 8px; border: 1px solid #d0d0d0;">Details</th></tr>'

                    # Determine which checks are required (branch protection).
                    # Prefer the full required_checks list from PRInfo; fall back to "is_required" flags on known checks.
                    required_set = set(getattr(self.pr, "required_checks", []) or [])
                    if not required_set:
                        try:
                            required_set = {
                                c.name
                                for c in (list(getattr(self.pr, "failed_checks", []) or []) + list(getattr(self.pr, "running_checks", []) or []))
                                if getattr(c, "is_required", False)
                            }
                        except Exception:
                            required_set = set()

                    for line in lines:
                        if not line.strip():
                            continue
                        # Split by tabs (gh pr checks uses tabs)
                        parts = line.split('\t')
                        if len(parts) >= 3:
                            name_raw = parts[0]
                            name = html_module.escape(name_raw)
                            status_raw = (parts[1] or "").strip()
                            status = html_module.escape(status_raw)
                            duration = html_module.escape(parts[2]) if len(parts) > 2 else ''
                            url = parts[3] if len(parts) > 3 else ''
                            description = html_module.escape(parts[4]) if len(parts) > 4 else ''

                            # Mark required checks (branch protection) inline.
                            if name_raw in required_set:
                                name += ' <span style="color: #d73a49; font-weight: 700;">[REQUIRED]</span>'

                            # Color code the status
                            status_lc = status_raw.lower()
                            status_color = '#059669' if status_lc in ('pass', 'success') else '#dc2626' if status_lc in ('fail', 'failure') else '#6b7280'
                            status_html = f'<span style="color: {status_color}; font-weight: bold;">{status}</span>'

                            # Update compact summary counts
                            if status_lc in ('pass', 'success'):
                                if name_raw in required_set:
                                    counts["success_required"] += 1
                                    passed_required_jobs.append(name_raw)
                                else:
                                    counts["success_optional"] += 1
                                    passed_optional_jobs.append(name_raw)
                            elif status_lc in ('fail', 'failure'):
                                if name_raw in required_set:
                                    counts["failure_required"] += 1
                                    failed_required_jobs.append(name_raw)
                                else:
                                    counts["failure_optional"] += 1
                                    failed_optional_jobs.append(name_raw)
                            elif status_lc in ('in_progress', 'in progress', 'running'):
                                counts["in_progress"] += 1
                                if name_raw in required_set:
                                    progress_required_jobs.append(name_raw)
                                else:
                                    progress_optional_jobs.append(name_raw)
                            elif status_lc in ('queued', 'pending'):
                                counts["pending"] += 1
                                pending_jobs.append(name_raw)
                            elif status_lc in ('skipping', 'skipped'):
                                # Show as paused (same bucket as pending) for readability.
                                counts["pending"] += 1
                                pending_jobs.append(name_raw)
                            elif status_lc in ('cancelled', 'canceled'):
                                counts["cancelled"] += 1
                                cancelled_jobs.append(name_raw)
                            else:
                                counts["other"] += 1
                                other_jobs.append(name_raw)

                            # Make URL clickable if present
                            if url and url.strip():
                                url_escaped = html_module.escape(url.strip())
                                details = f'<a href="{url_escaped}" target="_blank" style="color: #0969da; text-decoration: none;">View</a>'
                                if description:
                                    details += f' - {description}'
                            elif description:
                                details = description
                            else:
                                details = ''

                            table_html += f'<tr><td style="padding: 4px 8px; border: 1px solid #d0d0d0;">{name}</td><td style="padding: 4px 8px; border: 1px solid #d0d0d0;">{status_html}</td><td style="padding: 4px 8px; border: 1px solid #d0d0d0;">{duration}</td><td style="padding: 4px 8px; border: 1px solid #d0d0d0;">{details}</td></tr>'

                    table_html += '</table>'

                    # Rebuild the "Status:" line for HTML so CI uses the compact colored counts.
                    ci_parts = []
                    success_req = counts["success_required"]
                    success_opt = counts["success_optional"]
                    if success_req > 0 or success_opt > 0:
                        # Convention: 15+5✓ (first number is required, and bold)
                        if success_opt > 0:
                            ci_parts.append(
                                f'<span style="color: #2da44e;">'
                                f'<strong>{success_req}</strong>'
                                f'<span style="{PASS_PLUS_STYLE}">+{success_opt}</span>✓'
                                f'</span>'
                            )
                        else:
                            ci_parts.append(
                                f'<span style="color: #2da44e;">'
                                f'<strong>{success_req}</strong>✓'
                                f'</span>'
                            )
                    if counts["failure_required"] > 0:
                        ci_parts.append(
                            f'<span style="color: #d73a49; font-weight: 800; font-size: 12px;" title="Required failures">'
                            f'{counts["failure_required"]}'
                            f'<span style="display: inline-flex; align-items: center; justify-content: center; width: 12px; height: 12px; margin-left: 2px; border-radius: 999px; background-color: #d73a49; color: #ffffff; font-size: 10px; font-weight: 900; line-height: 1;">✗</span>'
                            f'</span>'
                        )
                    if counts["failure_optional"] > 0:
                        ci_parts.append(
                            f'<span style="color: #f59e0b;" title="Optional failures">'
                            f'{counts["failure_optional"]}<span style="font-size: 13px; font-weight: 900; line-height: 1; margin-left: 2px;">⚠</span>'
                            f'</span>'
                        )
                    if counts["in_progress"] > 0:
                        ci_parts.append(f'<span style="color: #dbab09;">{counts["in_progress"]}⏳</span>')
                    if counts["pending"] > 0:
                        ci_parts.append(f'<span style="color: #8c959f;">{counts["pending"]}⏸</span>')
                    if counts["cancelled"] > 0:
                        ci_parts.append(f'<span style="color: #8c959f;">{counts["cancelled"]}✖️</span>')

                    # Tooltip HTML (match show_commit_history look/labels).
                    tooltip_parts: list[str] = []
                    if passed_required_jobs:
                        tooltip_parts.append(
                            '<strong style="color: #2da44e;">✓ Passed (required):</strong> '
                            + ", ".join(sorted(html_module.escape(n) for n in passed_required_jobs))
                        )
                    if passed_optional_jobs:
                        tooltip_parts.append(
                            '<strong style="color: #2da44e;">✓ Passed (optional):</strong> '
                            + ", ".join(sorted(html_module.escape(n) for n in passed_optional_jobs))
                        )
                    if failed_required_jobs:
                        tooltip_parts.append(
                            '<strong style="color: #d73a49;"><span style="display: inline-flex; align-items: center; justify-content: center; width: 14px; height: 14px; margin-right: 6px; border-radius: 999px; background-color: #d73a49; color: #ffffff; font-size: 11px; font-weight: 900; line-height: 1;">✗</span>Failed (required):</strong> '
                            + ", ".join(sorted(html_module.escape(n) for n in failed_required_jobs))
                        )
                    if failed_optional_jobs:
                        tooltip_parts.append(
                            '<strong style="color: #f59e0b;">⚠ Failed (optional):</strong> '
                            + ", ".join(sorted(html_module.escape(n) for n in failed_optional_jobs))
                        )
                    if progress_required_jobs:
                        tooltip_parts.append(
                            '<strong style="color: #dbab09;">⏳ In Progress (required):</strong> '
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
                    if ci_parts:
                        ci_summary = " ".join(ci_parts)
                        if tooltip_html:
                            ci_summary = (
                                '<span class="gh-status-tooltip" style="margin-left: 2px;">'
                                f'<span style="white-space: nowrap; font-weight: 600; font-size: 11px;">{ci_summary}</span>'
                                f'<span class="tooltiptext">{tooltip_html}</span>'
                                '</span>'
                            )
                        # If there are no required failures (red ✗ badge), call it PASS even if there are optional ⚠.
                        ci_label = (
                            '<span class="status-indicator status-success">PASS</span>'
                            if counts["failure_required"] == 0
                            else '<span class="status-indicator status-failed">FAIL</span>'
                        )
                        # Put PASS/FAIL first so the line reads "Status: PASS, CI: ..."
                        status_parts.append(f"{ci_label} CI: {ci_summary}")

                    if self.pr.review_decision == 'APPROVED':
                        status_parts.append("Review: ✅ Approved")
                    elif self.pr.review_decision == 'CHANGES_REQUESTED':
                        status_parts.append("Review: 🔴 Changes Requested")

                    if self.pr.unresolved_conversations > 0:
                        status_parts.append(f"💬 Unresolved: {self.pr.unresolved_conversations}")

                    base_html = f"{', '.join(status_parts)}" if status_parts else ""

                    # Generate unique ID for this checks div
                    checks_id = f"checks_{uuid.uuid4().hex[:8]}"
                    # Add expandable section with formatted table.
                    # Match show_commit_history behavior: toggle the triangle (▶/▼) based on expanded state.
                    base_html += (
                        " "
                        + _html_toggle_span(
                            target_id=checks_id,
                            show_text="▶ Details",
                            hide_text="▼ Hide details",
                        )
                        + f'<div id="{checks_id}" style="display: none; margin-left: 20px; margin-top: 5px;">{table_html}</div>'
                    )
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                # Silently fail if gh command is not available or times out
                pass

        return base_html


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
        # Keep the per-check failure icon (requested): required gets ❌, optional gets ✗.
        icon = "❌" if self.failed_check.is_required else "✗"
        required_marker = " [REQUIRED]" if self.failed_check.is_required else ""
        error_indicator = " [Error details available]" if self.failed_check.error_summary else ""
        return f"{icon} {self.failed_check.name}{required_marker} ({self.failed_check.duration}){error_indicator}"

    def _format_html_content(self) -> str:
        if not self.failed_check:
            return ""
        # Keep the per-check failure icon (requested): required gets ❌, optional gets ✗.
        icon = "❌" if self.failed_check.is_required else '<span style="color: #000;">✗</span>'
        required_marker = ' <span style="color: #d73a49; font-weight: 700;">[REQUIRED]</span>' if self.failed_check.is_required else ""

        # Create unique ID for this error details section
        detail_id = hashlib.md5(f"{self.failed_check.name}{self.failed_check.job_url}".encode()).hexdigest()[:8]

        base_html = f'{icon} <a href="{self.failed_check.job_url}" target="_blank">{self.failed_check.name}</a>{required_marker} ({self.failed_check.duration})'

        # Add explicit "log" + "raw log" links.
        # - "log" is the normal GitHub job page (HTML)
        # - "raw log" is the direct download URL (typically a time-limited blob URL)
        base_html += _html_small_link(url=str(self.failed_check.job_url), label="[log]")
        if getattr(self.failed_check, "raw_log_url", None):
            base_html += _html_small_link(url=str(self.failed_check.raw_log_url), label="[raw log]")

        # Add expandable error details if available
        if self.failed_check.error_summary:
            # Escape HTML in error summary
            escaped_error = html.escape(self.failed_check.error_summary)
            # Keep "Show error" on the same line, but put the error div on a new line.
            # Also toggle the triangle (▶/▼) like show_commit_history.
            err_id = f"error_{detail_id}"
            base_html += (
                " "
                + _html_toggle_span(
                    target_id=err_id,
                    show_text="▶ Show error",
                    hide_text="▼ Hide error",
                )
                + f'<div id="{err_id}" style="display: none; margin-left: 20px; margin-top: 5px; padding: 10px; background-color: #fff5f5; border-left: 3px solid #cc0000; font-family: monospace; font-size: 11px; white-space: pre-wrap;">{escaped_error}</div>'
            )

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
        cmd = f"gh run rerun {self.run_id} --repo ai-dynamo/dynamo --failed"
        copy_btn = _html_copy_button(clipboard_text=cmd, title="Click to copy rerun command")

        return (
            f'🔄 <a href="{self.url}" target="_blank">Restart failed jobs</a> '
            f'(or: {copy_btn}<code>{cmd}</code>)'
        )


@dataclass
class RunningCheckNode(BranchNode):
    """Running check node"""
    check_name: str = ""
    is_required: bool = False
    check_url: Optional[str] = None
    elapsed_time: Optional[str] = None

    def _format_content(self) -> str:
        if not self.check_name:
            return ""
        required_marker = " [REQUIRED]" if self.is_required else ""
        time_info = f" ({self.elapsed_time})" if self.elapsed_time else ""
        return f"⏳ {self.check_name}{required_marker}{time_info}"

    def _format_html_content(self) -> str:
        if not self.check_name:
            return ""
        required_marker = ' <span style="color: #d73a49; font-weight: 700;">[REQUIRED]</span>' if self.is_required else ""
        time_info = f' <span style="color: #666; font-size: 11px;">({self.elapsed_time})</span>' if self.elapsed_time else ""
        if self.check_url:
            return f'⏳ <a href="{self.check_url}" target="_blank">{self.check_name}</a>{required_marker}{time_info}'
        return f'⏳ {self.check_name}{required_marker}{time_info}'






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

        if git is None:
            repo_node.error = "GitPython is required. Install with: pip install gitpython"
            return repo_node

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

        # Always capture current HEAD SHA (for display even when we skip "main" branches, or when detached).
        try:
            head_sha = repo.head.commit.hexsha[:7]
        except Exception:
            head_sha = None

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

                    # Add running checks if any (show first, before failed checks)
                    if pr.running_checks:
                        for running_check in pr.running_checks:
                            running_check_node = RunningCheckNode(
                                label="",
                                check_name=running_check.name,
                                is_required=running_check.is_required,
                                check_url=running_check.check_url,
                                elapsed_time=running_check.elapsed_time
                            )
                            pr_node.add_child(running_check_node)

                    # Add failed checks if any
                    if pr.failed_checks:
                        for failed_check in pr.failed_checks:
                            failed_test_node = FailedTestNode(label="", failed_check=failed_check)
                            pr_node.add_child(failed_test_node)

                        # Add rerun link after all failed checks
                        # Find the first GitHub Actions run_id for the rerun command
                        github_run_id = next((check.run_id for check in pr.failed_checks if check.run_id), None)
                        if pr.rerun_url and github_run_id:
                            rerun_node = RerunLinkNode(label="", url=pr.rerun_url, run_id=github_run_id)
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

        # If the current checkout didn't show up in the PR/local sections (common when on main),
        # add a single line for it so repos like dynamo_latest/ and dynamo_ci/ are informative.
        has_current_line = any(
            isinstance(ch, BranchInfoNode) and bool(getattr(ch, "is_current", False))
            for ch in (repo_node.children or [])
        )
        if not has_current_line:
            current_label = current_branch or "HEAD"
            commit_url = f"https://github.com/ai-dynamo/dynamo/commit/{head_sha}" if head_sha else None
            repo_node.add_child(
                BranchInfoNode(
                    label=current_label,
                    sha=head_sha,
                    is_current=True,
                    commit_url=commit_url,
                )
            )

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

    # Render all children (skip root) into a single <pre> block payload.
    rendered_lines: list[str] = []
    for i, child in enumerate(root.children):
        is_last = i == len(root.children) - 1
        rendered_lines.extend(child.render_html(prefix="", is_last=is_last, is_root=True))
        rendered_lines.append("")  # spacing between repos
    tree_html = "\n".join(rendered_lines).rstrip() + "\n"

    if not HAS_JINJA2:
        raise RuntimeError(
            "Jinja2 is required for --html output. Install with: pip install jinja2"
        )

    # Help type-checkers: these are set only when HAS_JINJA2 is True.
    assert Environment is not None
    assert FileSystemLoader is not None
    assert select_autoescape is not None

    env = Environment(
        loader=FileSystemLoader(str(_THIS_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("show_dynamo_branches.j2")
    return template.render(
        generated_time=pdt_str,
        gh_status_tooltip_css=GH_STATUS_TOOLTIP_CSS,
        gh_status_tooltip_js=GH_STATUS_TOOLTIP_JS,
        copy_icon_svg=_COPY_ICON_SVG,
        tree_html=tree_html,
    )


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
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file path (default: stdout)'
    )
    args = parser.parse_args()

    base_dir = args.base_dir

    base_dir = base_dir.resolve()

    # Scan repositories
    scanner = LocalRepoScanner(token=args.token)
    root = scanner.scan_repositories(base_dir)

    # Output
    if args.html:
        html_output = generate_html(root)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(html_output)
        else:
            print(html_output)
    else:
        for child in root.children:
            child.print_tree()
            print()


if __name__ == '__main__':
    main()
