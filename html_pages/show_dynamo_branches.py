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
import json
import os
import re
import stat
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
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
    PASS_PLUS_STYLE,
    TreeNodeVM,
    build_check_name_matchers,
    check_line_html,
    load_workflow_specs,
    render_tree_pre_lines,
    required_badge_html,
    status_icon_html,
)

# Dashboard runtime (HTML-only) helpers
from common_dashboard_runtime import (
    materialize_job_raw_log_text_local_link,
    prune_dashboard_raw_logs,
    prune_partial_raw_log_caches,
)

# Log/snippet helpers
from common_log_errors import extract_error_snippet_from_log_file

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
    PRInfo,
    dynamo_utils_cache_dir,
    summarize_pr_check_rows,
)

#
# Repo constants (avoid scattering hardcoded strings)
#

DYNAMO_OWNER = "ai-dynamo"
DYNAMO_REPO = "dynamo"
DYNAMO_REPO_SLUG = f"{DYNAMO_OWNER}/{DYNAMO_REPO}"


#
# Small HTML helpers
#

_COPY_ICON_SVG = (
    '<svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor" '
    'style="display: inline-block; vertical-align: middle;">'
    '<path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path>'
    '<path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>'
    "</svg>"
)

_COPY_BTN_STYLE = (
    "padding: 1px 4px; font-size: 10px; line-height: 1; background-color: transparent; color: #57606a; "
    "border: 1px solid #d0d7de; border-radius: 5px; cursor: pointer; display: inline-flex; "
    "align-items: center; vertical-align: baseline; margin-right: 4px;"
)


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
    return dynamo_utils_cache_dir() / "dashboards" / "show_dynamo_branches" / "gh_rate_limit_history.jsonl"


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
        # Use inline for <span>-based containers (avoids whitespace quirks inside <pre>).
        # For block-like content (tables), the container itself can carry its own margins.
        f'el.style.display=isHidden?\'inline\':\'none\';'
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

    def to_tree_vm(self) -> TreeNodeVM:
        """Convert this node subtree into a TreeNodeVM tree (shared renderer)."""
        # Default behavior: non-collapsible, always-expanded node.
        # Subclasses that previously had custom expand/collapse UI should override this.
        key = f"{self.__class__.__name__}:{self.label}"
        return TreeNodeVM(
            node_key=key,
            label_html=self._format_html_content(),
            children=[c.to_tree_vm() for c in (self.children or [])],
            collapsible=False,
            default_expanded=True,
        )

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
class CIJobTreeNode(BranchNode):
    """A CI node rendered as a tree child under a PR."""

    job_id: str = ""
    display_name: str = ""
    status: str = "unknown"
    duration: str = ""
    url: str = ""
    raw_log_href: str = ""
    raw_log_size_bytes: int = 0
    # Extracted from the local raw log file when available (highlighting handled in the shared dashboard UI helpers).
    error_snippet_text: str = ""
    is_required: bool = False
    failed_check: Optional[FailedCheck] = None

    @dataclass(frozen=True)
    class _Rollup:
        status: str
        has_required_failure: bool
        has_optional_failure: bool

    @staticmethod
    def _is_success_like(node: "CIJobTreeNode") -> bool:
        """Heuristic: treat certain "unknown" jobs as success-like (collapsed/green).

        In practice, `gh pr checks` sometimes yields empty/unknown status for skipped jobs,
        and they often show a duration of 0. We consider these "success-like" so fully
        green/skipped subtrees collapse by default.
        """
        if node.status == "success":
            return True
        if node.status != "unknown":
            return False
        dur = (node.duration or "").strip().lower()
        return dur in {"0", "0s", "0m", "0m0s", "0h", "0h0m", "0h0m0s"}

    @staticmethod
    def _subtree_rollup(node: "CIJobTreeNode") -> "CIJobTreeNode._Rollup":
        """Compute a 'worst descendant' status for icon rendering."""
        # Success-like entire subtree => success rollup.
        if CIJobTreeNode._subtree_all_success(node):
            return CIJobTreeNode._Rollup(status="success", has_required_failure=False, has_optional_failure=False)

        has_required_failure = False
        has_optional_failure = False
        statuses: List[str] = []

        def walk(n: "CIJobTreeNode") -> None:
            nonlocal has_required_failure, has_optional_failure
            st = getattr(n, "status", "unknown") or "unknown"
            is_req = bool(getattr(n, "is_required", False))
            if st == "failure" and is_req:
                has_required_failure = True
                statuses.append("failure")
            elif st == "failure" and not is_req:
                # Optional failures should not turn the parent red; record them separately.
                has_optional_failure = True
                statuses.append("success")
            else:
                statuses.append(st)
            for ch in (getattr(n, "children", None) or []):
                if isinstance(ch, CIJobTreeNode):
                    walk(ch)

        walk(node)

        # Worst-first priority for status rollup.
        priority = ["failure", "in_progress", "pending", "cancelled", "unknown", "success"]
        for s in priority:
            if s in statuses:
                return CIJobTreeNode._Rollup(
                    status=s,
                    has_required_failure=has_required_failure,
                    has_optional_failure=has_optional_failure,
                )
        return CIJobTreeNode._Rollup(status="unknown", has_required_failure=has_required_failure, has_optional_failure=has_optional_failure)

    @staticmethod
    def _subtree_needs_attention(node: "CIJobTreeNode") -> bool:
        """Return True if this subtree should be expanded by default.

        Policy (per UX request):
        - expand for required failures (red ✗) and non-completed states (pending/running/cancelled/unknown)
        - do NOT auto-expand for optional failures (⚠) or all-green subtrees
        """
        # All-success-like subtree: no need to expand.
        if CIJobTreeNode._subtree_all_success(node):
            return False

        # Optional failures (warnings) alone should not force expansion.
        # We therefore expand only if we see a required failure or a non-completed state.
        if getattr(node, "status", "unknown") == "failure" and bool(getattr(node, "is_required", False)):
            return True
        if getattr(node, "status", "unknown") in {"in_progress", "pending", "cancelled"}:
            return True

        # Unknown leaf nodes that aren't treated as success-like should be visible.
        if not getattr(node, "children", None) and getattr(node, "status", "unknown") == "unknown":
            return True

        for ch in (getattr(node, "children", None) or []):
            if isinstance(ch, CIJobTreeNode) and CIJobTreeNode._subtree_needs_attention(ch):
                return True

        return False

    @staticmethod
    def _subtree_all_success(node: "CIJobTreeNode") -> bool:
        """Return True iff this node and all descendants are in a success-like state.

        Note: Some GitHub statuses come back as "unknown" from `gh pr checks` parsing.
        If the node is unknown but all its children are success-like, we treat the subtree
        as success-like for *default expand/collapse* purposes.
        """
        # Any non-success/unknown state should be visible by default.
        if node.status not in {"success", "unknown"}:
            return False
        if node.children:
            # Only collapse if every descendant is success-like.
            for child in node.children:
                if not isinstance(child, CIJobTreeNode):
                    return False
                if not CIJobTreeNode._subtree_all_success(child):
                    return False
            # If children are all success-like, this node is success-like regardless of its own unknown-ness.
            return True
        # Leaf nodes: allow "unknown but 0 duration" to behave like skipped (success-like).
        return CIJobTreeNode._is_success_like(node)

    def _format_content(self) -> str:
        name_part = f" — {self.display_name}" if (self.display_name and self.display_name != self.job_id) else ""
        dur_part = f" ({self.duration})" if self.duration else ""
        return f"{self.job_id}{name_part}{dur_part}"

    def _format_html_content(self) -> str:
        # Prefer FailedCheck URLs when available (they include raw log + error summary)
        job_url = ""
        if self.failed_check is not None:
            job_url = str(getattr(self.failed_check, "job_url", "") or "")
        if not job_url:
            job_url = self.url or ""
        # Icon: show the *worst descendant status* so it’s obvious why a parent expanded.
        roll = self._subtree_rollup(self) if self.children else None
        effective_status = (roll.status if roll is not None else self.status)
        effective_required_failure = (roll.has_required_failure if roll is not None else False)
        effective_optional_failure = (roll.has_optional_failure if roll is not None else False)

        # Error toggle is appended in to_tree_vm() so it can inject a newline safely in <pre>.
        return check_line_html(
            job_id=self.job_id,
            display_name=self.display_name,
            status_norm=effective_status,
            is_required=bool(self.is_required),
            duration=self.duration,
            log_url=job_url,
            raw_log_href=self.raw_log_href,
            raw_log_size_bytes=int(self.raw_log_size_bytes or 0),
            error_snippet_text=(self.error_snippet_text or ""),
            required_failure=bool(effective_required_failure),
            warning_present=bool(effective_optional_failure and effective_status == "success"),
        )

    def render_html(self, prefix: str = "", is_last: bool = True, is_root: bool = True) -> List[str]:
        """Render CI subtree with an expand/collapse triangle.

        Default:
        - success (green): collapsed
        - failure (red): expanded
        - other: expanded (so in-progress/pending stays visible)
        """
        lines: List[str] = []

        if not is_root:
            connector = "└─" if is_last else "├─"
            current_prefix = prefix + connector + " "
        else:
            current_prefix = ""

        # Unique DOM id per *render occurrence* (include prefix so duplicates don't collide).
        dom_hash = hashlib.sha1((prefix + "|" + self.job_id + "|" + (self.url or "")).encode("utf-8")).hexdigest()[:10]
        children_id = f"ci_children_{dom_hash}"

        has_children = bool(self.children)
        # Default expansion rule:
        # - collapse for all-green and warning-only subtrees
        # - expand only when something needs attention (required failure, pending/running/cancelled/unknown)
        default_expanded = self._subtree_needs_attention(self)

        # Use margin (not literal spaces) so we don't create "double spaces" in the rendered text.
        if has_children:
            triangle_char = "▼" if default_expanded else "▶"
            triangle = (
                f'<span style="display: inline-block; width: 12px; margin-right: 2px; color: #0969da; '
                f'cursor: pointer; user-select: none;" '
                f'onclick="toggleCiChildren(\'{children_id}\', this)">{triangle_char}</span>'
            )
        else:
            triangle = '<span style="display: inline-block; width: 12px; margin-right: 2px;"></span>'

        # Error snippet toggle is rendered by shared `check_line_html()` when a local [raw log] exists.
        line_content = self._format_html_content()
        if line_content.strip():
            lines.append(current_prefix + triangle + line_content)

        if has_children and lines:
            # IMPORTANT: wrap children starting at the *end of the parent line* so the newline
            # between parent and first child is hidden when collapsed (prevents blank lines).
            disp = "inline" if default_expanded else "none"
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
                # No children rendered; close immediately.
                lines[-1] = lines[-1] + "</span>"

        return lines

    def to_tree_vm(self) -> TreeNodeVM:
        # Mirror the existing expand/collapse policy and include the error toggle HTML in the label.
        has_children = bool(self.children)
        default_expanded = self._subtree_needs_attention(self) if has_children else False

        return TreeNodeVM(
            node_key=f"CI:{self.job_id}:{self.url}",
            label_html=self._format_html_content(),
            children=[c.to_tree_vm() for c in (self.children or []) if isinstance(c, BranchNode)],
            collapsible=True,  # show triangle placeholder for alignment, even on leaves
            default_expanded=bool(default_expanded),
            triangle_tooltip=None,
        )


def _build_ci_hierarchy_nodes(
    repo_path: Path,
    pr: PRInfo,
    github_api: Optional[GitHubAPIClient] = None,
    *,
    page_root_dir: Optional[Path] = None,
    checks_ttl_s: int = 300,
    skip_fetch: bool = False,
) -> List[CIJobTreeNode]:
    """Build a workflow-needs hierarchy annotated with actual PR check status."""
    if not pr or not getattr(pr, "number", None) or not github_api:
        return []
    required_set: Set[str] = set(getattr(pr, "required_checks", []) or [])
    rows = github_api.get_pr_checks_rows(
        DYNAMO_OWNER,
        DYNAMO_REPO,
        int(pr.number),
        required_checks=required_set,
        ttl_s=int(checks_ttl_s),
        skip_fetch=bool(skip_fetch),
    )
    if not rows:
        return []

    failed_by_name: Dict[str, FailedCheck] = {}
    try:
        for fc in (list(getattr(pr, "failed_checks", []) or [])):
            if getattr(fc, "name", None):
                failed_by_name[str(fc.name)] = fc
    except Exception:
        failed_by_name = {}

    specs = load_workflow_specs(repo_path)
    matchers = build_check_name_matchers(specs)

    grouped: Dict[str, List[GHPRCheckRow]] = {}
    for r in rows:
        mapped: Optional[str] = None
        for job_id, rx in matchers:
            if rx.match(r.name):
                mapped = job_id
                break
        if not mapped:
            mapped = f"check::{r.name}"
        grouped.setdefault(mapped, []).append(r)

    # Hierarchy inclusion policy:
    # Always include *all* check rows (whether green/warn/red/etc). Expand/collapse is a separate concern.
    # This makes the hierarchy list stable/consistent across branches and PR base refs.
    #
    # (We still render the workflow-needs graph: checks are mapped into job_ids when possible.)
    important_ids: Set[str] = set(grouped.keys())

    def add_needs(job_id: str) -> None:
        spec = specs.get(job_id)
        if not spec:
            return
        for dep in spec.needs:
            if dep in grouped and dep not in important_ids:
                important_ids.add(dep)
                add_needs(dep)

    for jid in list(important_ids):
        add_needs(jid)

    needs_map: Dict[str, List[str]] = {}
    for job_id, spec in specs.items():
        if job_id in grouped and job_id in important_ids:
            needs_map[job_id] = [d for d in spec.needs if d in grouped and d in important_ids]

    needed: Set[str] = set()
    for deps in needs_map.values():
        needed.update(deps)
    workflow_roots = sorted([jid for jid in needs_map.keys() if jid not in needed])
    synthetic_roots = sorted([jid for jid in important_ids if jid.startswith("check::")])

    memo: Dict[str, CIJobTreeNode] = {}
    # Cache raw log *content* (not URL) for failed jobs. Keep a small cap so a single PR doesn't
    # trigger an unbounded number of downloads.
    raw_log_prefetch_budget = {"n": 15}
    page_root_dir = Path(page_root_dir) if page_root_dir is not None else repo_path
    snippet_cache: Dict[str, str] = {}

    def snippet_for_raw_href(raw_href: str) -> str:
        if not raw_href:
            return ""
        if raw_href in snippet_cache:
            return snippet_cache[raw_href]
        try:
            snippet = extract_error_snippet_from_log_file(page_root_dir / raw_href)
        except Exception:
            snippet = ""
        snippet_cache[raw_href] = snippet
        return snippet

    def build_node(job_id: str) -> CIJobTreeNode:
        if job_id in memo:
            return memo[job_id]

        if job_id.startswith("check::"):
            check_name = job_id.split("check::", 1)[1]
            bucket = grouped.get(job_id, [])
            node = CIJobTreeNode(label="", job_id=check_name, status=_aggregate_status([b.status_norm for b in bucket]))
            for b in bucket:
                # Cache raw log content for failures (best-effort).
                raw_href = ""
                raw_size = 0
                if (
                    github_api
                    and b.status_norm == "failure"
                ):
                    allow_fetch = int(raw_log_prefetch_budget.get("n", 0) or 0) > 0
                    try:
                        raw_href = (
                            materialize_job_raw_log_text_local_link(
                                github_api,
                                job_url=b.url or "",
                                job_name=b.name,
                                owner=DYNAMO_OWNER,
                                repo=DYNAMO_REPO,
                                page_root_dir=page_root_dir,
                                allow_fetch=bool(allow_fetch),
                            )
                            or ""
                        )
                    except Exception:
                        raw_href = ""
                    if allow_fetch:
                        raw_log_prefetch_budget["n"] = int(raw_log_prefetch_budget.get("n", 0) or 0) - 1
                if raw_href:
                    try:
                        raw_size = int((page_root_dir / raw_href).stat().st_size)
                    except Exception:
                        raw_size = 0
                snippet = snippet_for_raw_href(raw_href) if raw_href else ""
                node.children.append(
                    CIJobTreeNode(
                        label="",
                        job_id=b.name,
                        status=b.status_norm,
                        duration=b.duration,
                        url=b.url,
                        raw_log_href=raw_href,
                        raw_log_size_bytes=int(raw_size or 0),
                        error_snippet_text=snippet,
                        is_required=b.is_required,
                    )
                )
            memo[job_id] = node
            return node

        spec = specs.get(job_id)
        bucket = grouped.get(job_id, [])
        display_name = spec.display_name if spec else ""

        if len(bucket) == 1:
            b = bucket[0]
            raw_href = ""
            raw_size = 0
            if (
                github_api
                and b.status_norm == "failure"
            ):
                allow_fetch = int(raw_log_prefetch_budget.get("n", 0) or 0) > 0
                try:
                    raw_href = (
                        materialize_job_raw_log_text_local_link(
                            github_api,
                            job_url=b.url or "",
                            job_name=b.name,
                            owner=DYNAMO_OWNER,
                            repo=DYNAMO_REPO,
                            page_root_dir=page_root_dir,
                            allow_fetch=bool(allow_fetch),
                        )
                        or ""
                    )
                except Exception:
                    raw_href = ""
                if allow_fetch:
                    raw_log_prefetch_budget["n"] = int(raw_log_prefetch_budget.get("n", 0) or 0) - 1
            if raw_href:
                try:
                    raw_size = int((page_root_dir / raw_href).stat().st_size)
                except Exception:
                    raw_size = 0
            snippet = snippet_for_raw_href(raw_href) if raw_href else ""
            node = CIJobTreeNode(
                label="",
                job_id=job_id,
                display_name=display_name,
                status=b.status_norm,
                duration=b.duration,
                url=b.url,
                raw_log_href=raw_href,
                raw_log_size_bytes=int(raw_size or 0),
                error_snippet_text=snippet,
                is_required=b.is_required,
                failed_check=failed_by_name.get(b.name),
            )
        else:
            node = CIJobTreeNode(
                label="",
                job_id=job_id,
                display_name=display_name,
                status=_aggregate_status([b.status_norm for b in bucket]),
                is_required=any(b.is_required for b in bucket),
            )
            for b in sorted(bucket, key=lambda x: x.name):
                raw_href = ""
                raw_size = 0
                if (
                    github_api
                    and b.status_norm == "failure"
                ):
                    allow_fetch = int(raw_log_prefetch_budget.get("n", 0) or 0) > 0
                    try:
                        raw_href = (
                            materialize_job_raw_log_text_local_link(
                                github_api,
                                job_url=b.url or "",
                                job_name=b.name,
                                owner=DYNAMO_OWNER,
                                repo=DYNAMO_REPO,
                                page_root_dir=page_root_dir,
                                allow_fetch=bool(allow_fetch),
                            )
                            or ""
                        )
                    except Exception:
                        raw_href = ""
                    if allow_fetch:
                        raw_log_prefetch_budget["n"] = int(raw_log_prefetch_budget.get("n", 0) or 0) - 1
                if raw_href:
                    try:
                        raw_size = int((page_root_dir / raw_href).stat().st_size)
                    except Exception:
                        raw_size = 0
                snippet = snippet_for_raw_href(raw_href) if raw_href else ""
                node.children.append(
                    CIJobTreeNode(
                        label="",
                        job_id=b.name,
                        status=b.status_norm,
                        duration=b.duration,
                        url=b.url,
                        raw_log_href=raw_href,
                        raw_log_size_bytes=int(raw_size or 0),
                        error_snippet_text=snippet,
                        is_required=b.is_required,
                        failed_check=failed_by_name.get(b.name),
                    )
                )

        for dep in needs_map.get(job_id, []):
            node.children.append(build_node(dep))

        if not bucket and node.children:
            node.status = _aggregate_status([getattr(c, "status", "unknown") for c in node.children])

        memo[job_id] = node
        return node

    forest: List[CIJobTreeNode] = []
    for r in workflow_roots:
        forest.append(build_node(r))
    for r in synthetic_roots:
        forest.append(build_node(r))
    return forest


@dataclass
class RepoNode(BranchNode):
    """Repository node"""
    path: Optional[Path] = None
    error: Optional[str] = None
    remote_url: Optional[str] = None
    # Historical: we used to warn and stop scanning repos that weren't ai-dynamo/dynamo.
    # Keep the field for backward compatibility, but we no longer treat "non-dynamo" as an error.
    is_correct_repo: bool = True

    def _format_content(self) -> str:
        if self.error:
            return f"\033[1m{self.label}\033[0m\n  \033[91m⚠️  {self.error}\033[0m"
        return f"\033[1m{self.label}\033[0m"

    def _format_html_content(self) -> str:
        # Make repo name clickable (relative URL to the directory)
        # Remove trailing slash from label for URL
        repo_dirname = self.label.rstrip('/')
        repo_link = f'<a href="{repo_dirname}/" class="repo-name">{self.label}</a>'

        if self.error:
            return f'{repo_link}\n<span class="error">⚠️  {self.error}</span>'
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
    commit_time_pt: Optional[str] = None
    commit_datetime: Optional[datetime] = None

    @staticmethod
    def _format_age(dt: Optional[datetime]) -> Optional[str]:
        """Return a compact '(… old)' string for the commit datetime."""
        if dt is None:
            return None
        try:
            now = datetime.now(ZoneInfo("UTC"))
            if getattr(dt, "tzinfo", None) is None:
                dt = dt.replace(tzinfo=ZoneInfo("UTC"))
            delta_s = max(0, int((now - dt.astimezone(ZoneInfo("UTC"))).total_seconds()))
        except Exception:
            return None

        if delta_s < 60:
            return f"({delta_s}s old)"
        if delta_s < 3600:
            return f"({delta_s // 60}m old)"
        if delta_s < 86400:
            h = delta_s // 3600
            m = (delta_s % 3600) // 60
            return f"({h}h {m}m old)" if m else f"({h}h old)"
        d = delta_s // 86400
        h = (delta_s % 86400) // 3600
        return f"({d}d {h}h old)" if h else f"({d}d old)"

    def _format_content(self) -> str:
        sha_str = f" [{self.sha}]" if self.sha else ""
        time_str = f" {self.commit_time_pt}" if self.commit_time_pt else ""
        age = self._format_age(self.commit_datetime)
        # If the branch has a PR child, also show PR created time in the branch line.
        pr_created_pt: Optional[str] = None
        for ch in (self.children or []):
            pr = getattr(ch, "pr", None)
            created_at = getattr(pr, "created_at", None) if pr is not None else None
            if created_at:
                pr_created_pt = _format_utc_datetime_from_iso(created_at)
                break
        if pr_created_pt and age:
            age_plain = age[1:-1] if (age.startswith("(") and age.endswith(")")) else age
            age_str = f" (created {pr_created_pt}, {age_plain})"
        elif age:
            age_str = f" {age}"
        else:
            age_str = ""
        if self.is_current:
            return f"\033[1m{self.label}\033[0m{sha_str}{time_str}{age_str}"
        return f"{self.label}{sha_str}{time_str}{age_str}"

    def _format_html_content(self) -> str:
        sha_str = ""
        if self.sha:
            if self.commit_url:
                sha_str = f' [<a href="{self.commit_url}" target="_blank">{self.sha}</a>]'
            else:
                sha_str = f' [{self.sha}]'
        time_str = (
            f' <span style="color: #666; font-size: 11px;">{html.escape(self.commit_time_pt)}</span>'
            if self.commit_time_pt
            else ""
        )
        age = self._format_age(self.commit_datetime)

        # Prefer the first PR's created time (PT) if present, and show the commit age bold.
        pr_created_pt: Optional[str] = None
        for ch in (self.children or []):
            pr = getattr(ch, "pr", None)
            created_at = getattr(pr, "created_at", None) if pr is not None else None
            if created_at:
                pr_created_pt = _format_utc_datetime_from_iso(created_at)
                break

        if pr_created_pt and age:
            age_plain = age[1:-1] if (age.startswith("(") and age.endswith(")")) else age
            age_str = (
                f' <span style="color: #666; font-size: 11px;">(created {html.escape(pr_created_pt)}, '
                f'<span style="font-weight: 700; color: #24292f;">{html.escape(age_plain)}</span>)</span>'
            )
        elif age:
            age_str = f' <span style="color: #666; font-size: 11px;">{html.escape(age)}</span>'
        else:
            age_str = ""

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
            return f'{copy_btn}<span class="{cls}">{self.label}</span>{sha_str}{time_str}{age_str}'
        return f'{copy_btn}{self.label}{sha_str}{time_str}{age_str}'


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

        # Add base branch (new format)
        base_str = f" → {self.pr.base_ref} branch" if self.pr.base_ref else ""
        # Prefer standard GitHub-style formatting: "title (#1234)" over "PR#1234: title"
        pr_suffix = f"(#{self.pr.number})"
        if pr_suffix not in title:
            title = f"{title} {pr_suffix}"
        return f"{emoji} {title}{base_str}"

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

        base_html = ""
        if self.pr.base_ref:
            base_html = f' <span style="font-weight: 700;">→ {html.escape(self.pr.base_ref)}</span> branch'

        # Prefer standard GitHub-style formatting: "title (#1234)" over "PR#1234: title"
        pr_suffix = f"(#{self.pr.number})"
        if pr_suffix not in title:
            title = f"{title} {pr_suffix}"

        # Gray out merged PRs
        if self.pr.is_merged:
            return f'<span style="color: #999;">{emoji} <a href="{self.pr.url}" target="_blank" style="color: #999;">{title}</a>{base_html}</span>'
        else:
            return f'{emoji} <a href="{self.pr.url}" target="_blank">{title}</a>{base_html}'


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
    github_api: Optional[GitHubAPIClient] = None
    refresh_checks: bool = False
    branch_commit_dt: Optional[datetime] = None
    allow_fetch_checks: bool = True

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

                required_set = set(getattr(self.pr, "required_checks", []) or [])
                rows = (
                    self.github_api.get_pr_checks_rows(
                        DYNAMO_OWNER,
                        DYNAMO_REPO,
                        int(self.pr.number),
                        required_checks=required_set,
                        ttl_s=ttl_s,
                        skip_fetch=(not bool(self.allow_fetch_checks)),
                    )
                    if self.github_api
                    else []
                )

                # Display checks output if available
                if rows:
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

                    table_html = '<table style="border-collapse: collapse; font-size: 11px; margin-top: 5px;">'
                    table_html += '<tr style="background-color: #e8eaed;"><th style="text-align: left; padding: 4px 8px; border: 1px solid #d0d0d0;">Check Name</th><th style="text-align: left; padding: 4px 8px; border: 1px solid #d0d0d0;">Status</th><th style="text-align: left; padding: 4px 8px; border: 1px solid #d0d0d0;">Duration</th><th style="text-align: left; padding: 4px 8px; border: 1px solid #d0d0d0;">Details</th></tr>'

                    # Determine which checks are required (branch protection).
                    # Prefer the full required_checks list from PRInfo; fall back to "is_required" flags on known checks.
                    if not required_set:
                        try:
                            required_set = {
                                c.name
                                for c in (list(getattr(self.pr, "failed_checks", []) or []) + list(getattr(self.pr, "running_checks", []) or []))
                                if getattr(c, "is_required", False)
                            }
                        except Exception:
                            required_set = set()

                    # Shared summary (common.py) so show_commit_history + show_dynamo_branches stay consistent.
                    summary = summarize_pr_check_rows(rows)

                    # Map summary buckets into the local display buckets.
                    counts["success_required"] = int(summary.counts.success_required)
                    counts["success_optional"] = int(summary.counts.success_optional)
                    counts["failure_required"] = int(summary.counts.failure_required)
                    counts["failure_optional"] = int(summary.counts.failure_optional)
                    counts["in_progress"] = int(summary.counts.in_progress_required + summary.counts.in_progress_optional)
                    counts["pending"] = int(summary.counts.pending)
                    counts["cancelled"] = int(summary.counts.cancelled)
                    counts["other"] = int(summary.counts.other)

                    passed_required_jobs = list(summary.names.success_required)
                    passed_optional_jobs = list(summary.names.success_optional)
                    failed_required_jobs = list(summary.names.failure_required)
                    failed_optional_jobs = list(summary.names.failure_optional)
                    progress_required_jobs = list(summary.names.in_progress_required)
                    progress_optional_jobs = list(summary.names.in_progress_optional)
                    pending_jobs = list(summary.names.pending)
                    cancelled_jobs = list(summary.names.cancelled)
                    other_jobs = list(summary.names.other)

                    # Still build the detailed per-check table (this is presentation-specific).
                    for row in rows:
                        name_raw = row.name
                        name = html_module.escape(name_raw)
                        status_raw = (row.status_raw or "").strip()
                        status = html_module.escape(status_raw)
                        duration = html_module.escape(row.duration or "")
                        url = row.url or ""
                        description = html_module.escape(row.description or "")

                        # Mark required checks (branch protection) inline.
                        if row.is_required or (name_raw in required_set):
                            name += ' <span style="color: #d73a49; font-weight: 700;">[REQUIRED]</span>'

                        # Color code the status
                        status_lc = status_raw.lower()
                        status_color = '#059669' if status_lc in ('pass', 'success') else '#dc2626' if status_lc in ('fail', 'failure') else '#6b7280'
                        status_html = f'<span style="color: {status_color}; font-weight: bold;">{status}</span>'

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
                                f'<span style="{PASS_PLUS_STYLE}">+{success_opt}</span>{status_icon_html(status_norm="success", is_required=False)}'
                                f'</span>'
                            )
                        else:
                            ci_parts.append(
                                f'<span style="color: #2da44e;">'
                                f'<strong>{success_req}</strong>{status_icon_html(status_norm="success", is_required=False)}'
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
                            f'<strong style="color: #2da44e;">{status_icon_html(status_norm="success", is_required=False)} Passed (required):</strong> '
                            + ", ".join(sorted(html_module.escape(n) for n in passed_required_jobs))
                        )
                    if passed_optional_jobs:
                        tooltip_parts.append(
                            f'<strong style="color: #2da44e;">{status_icon_html(status_norm="success", is_required=False)} Passed (optional):</strong> '
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
                                f'<span style="white-space: nowrap; font-weight: 600; font-size: 12px;">{ci_summary}</span>'
                                f'<span class="tooltiptext">{tooltip_html}</span>'
                                '</span>'
                            )
                        # If there are no required failures (red ✗ badge), call it PASS even if there are optional ⚠.
                        ci_label = (
                            '<span class="status-indicator status-success">PASS</span>'
                            if counts["failure_required"] == 0
                            else '<span class="status-indicator status-failed">FAIL</span>'
                        )

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
                        # IMPORTANT: avoid <div> inside <pre> because it introduces line breaks / extra whitespace.
                        # Use a <span> container and toggle it between display:none and display:block.
                        + f'<span id="{checks_id}" style="display: none; margin-left: 20px; margin-top: 5px;">{table_html}</span>'
                    )
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                # Silently fail if gh command is not available or times out
                pass

        return base_html

    def render_html(self, prefix: str = "", is_last: bool = True, is_root: bool = True) -> List[str]:
        """Render the PASS/FAIL status line with an expandable CI hierarchy subtree."""
        lines: List[str] = []

        if not is_root:
            connector = "└─" if is_last else "├─"
            current_prefix = prefix + connector + " "
        else:
            current_prefix = ""

        has_children = bool(self.children)

        # Expand by default only when something needs attention (required failure / pending / running / cancelled / unknown).
        default_expanded = any(
            (not isinstance(c, CIJobTreeNode)) or CIJobTreeNode._subtree_needs_attention(c)
            for c in (self.children or [])
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
                f'title="CI hierarchy (derived from .github/workflows/*.yml)" '
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

    def to_tree_vm(self) -> TreeNodeVM:
        # Default expansion: only when something needs attention in the CI subtree.
        default_expanded = any(
            (not isinstance(c, CIJobTreeNode)) or CIJobTreeNode._subtree_needs_attention(c)
            for c in (self.children or [])
        )
        return TreeNodeVM(
            node_key=f"PRStatus:{getattr(self.pr, 'number', '')}",
            label_html=self._format_html_content(),
            children=[c.to_tree_vm() for c in (self.children or []) if isinstance(c, BranchNode)],
            collapsible=True,
            default_expanded=bool(default_expanded),
            triangle_tooltip="CI hierarchy (derived from .github/workflows/*.yml)",
        )


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


class LocalRepoScanner:
    """Scanner for local repository branches"""

    def __init__(
        self,
        token: Optional[str] = None,
        refresh_closed_prs: bool = False,
        *,
        max_branches: Optional[int] = None,
        max_checks_fetch: Optional[int] = None,
    ):
        self.github_api = GitHubAPIClient(token=token)
        self.refresh_closed_prs = bool(refresh_closed_prs)
        self.max_branches = int(max_branches) if max_branches is not None else None
        self.max_checks_fetch = int(max_checks_fetch) if max_checks_fetch is not None else None

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
        """
        Lightweight git repo detection without invoking GitPython.

        Supports normal repos ('.git' directory) and worktrees/submodules ('.git' file).
        """
        try:
            if not p.is_dir():
                return False
            git_marker = p / ".git"
            return git_marker.is_dir() or git_marker.is_file()
        except Exception:
            return False

    def scan_repositories(self, base_dir: Path) -> BranchNode:
        """Scan all git repositories under `base_dir` (direct children only) and build tree structure."""
        root = BranchNode(label="")

        # Discover git repos among direct children (and include base_dir itself if it's a repo).
        #
        # We intentionally do NOT walk the whole tree because this workspace can be huge (targets, caches, etc.).
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

    def _scan_repository(self, repo_dir: Path, page_root_dir: Path) -> Optional[RepoNode]:
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

        # Capture origin URL (if present). We no longer treat "non-dynamo" repos as an error; we
        # just skip PR/CI lookups that are wired to ai-dynamo/dynamo.
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
            except Exception:
                sha = None
                commit_dt = None

            is_current = (branch_name == current_branch)

            if has_remote:
                # Get PR info in parallel (will be gathered later)
                branches_with_prs[branch_name] = {
                    'sha': sha,
                    'is_current': is_current,
                    'branch': branch,
                    'commit_time_pt': _format_pt_time(commit_dt),
                    'commit_dt': commit_dt,
                }
            else:
                local_only_branches.append({
                    'name': branch_name,
                    'sha': sha,
                    'is_current': is_current,
                    'commit_time_pt': _format_pt_time(commit_dt),
                    'commit_dt': commit_dt,
                })

        # Fetch PR information in parallel
        if branches_with_prs and is_dynamo_repo:
            pr_section = SectionNode(label="Branches with PRs")
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
                print(f"Error fetching PR info: {e}", file=sys.stderr)
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
            if self.max_checks_fetch is not None and self.max_checks_fetch > 0:
                allow_fetch_branch_names = {bn for (bn, _info, _prs) in branch_results_sorted[: int(self.max_checks_fetch)]}

            # Build branch nodes (newest first)
            for branch_name, info, prs in branch_results_sorted:
                commit_url = f"https://github.com/{DYNAMO_REPO_SLUG}/commit/{info['sha']}" if info['sha'] else None
                branch_node = BranchInfoNode(
                    label=branch_name,
                    sha=info['sha'],
                    is_current=info['is_current'],
                    commit_url=commit_url,
                    commit_time_pt=info.get('commit_time_pt'),
                    commit_datetime=info.get('commit_dt'),
                )

                # Add PR nodes
                for pr in prs:
                    allow_fetch_checks = (branch_name in allow_fetch_branch_names) if allow_fetch_branch_names else True
                    pr_node = PRNode(label="", pr=pr)
                    branch_node.add_child(pr_node)

                    # Add URL node (text only)
                    url_node = PRURLNode(label="", url=pr.url)
                    pr_node.add_child(url_node)

                    pr_state_lc = (getattr(pr, "state", "") or "").lower()
                    # Prefer the branch head commit time for "last push" heuristic.
                    branch_dt = info.get("commit_dt")
                    checks_ttl_s = GitHubAPIClient.compute_checks_cache_ttl_s(
                        branch_dt,
                        refresh=bool(self.refresh_closed_prs),
                    )

                    status_node = PRStatusNode(
                        label="",
                        pr=pr,
                        github_api=self.github_api,
                        refresh_checks=bool(self.refresh_closed_prs),
                        branch_commit_dt=branch_dt,
                        allow_fetch_checks=bool(allow_fetch_checks),
                    )
                    pr_node.add_child(status_node)

                    # If CI failed, show "Restart failed jobs" immediately after the FAIL line.
                    try:
                        github_run_id = next((check.run_id for check in (pr.failed_checks or []) if check.run_id), None)
                    except Exception:
                        github_run_id = None
                    if pr.failed_checks:
                        rerun_url = pr.rerun_url or (f"https://github.com/{DYNAMO_REPO_SLUG}/actions/runs/{github_run_id}" if github_run_id else None)
                        if rerun_url and github_run_id:
                            pr_node.add_child(RerunLinkNode(label="", url=rerun_url, run_id=github_run_id))

                    # Add CI hierarchy as children of the PR status line (cached; long TTL when no recent pushes).
                    try:
                        for ci_node in _build_ci_hierarchy_nodes(
                            repo_dir,
                            pr,
                            github_api=self.github_api,
                            page_root_dir=page_root_dir,
                            checks_ttl_s=int(checks_ttl_s),
                            skip_fetch=(not bool(allow_fetch_checks)),
                        ):
                            status_node.add_child(ci_node)
                    except Exception as e:
                        # Don't silently drop the tree UI; surface the error so it's actionable.
                        try:
                            status_node.add_child(
                                BranchNode(
                                    label=f'<span style="color: #d73a49;">⚠ CI hierarchy error: {html.escape(str(e))}</span>'
                                )
                            )
                        except Exception:
                            pass
                        try:
                            print(f"[show_dynamo_branches] CI hierarchy error for PR #{getattr(pr, 'number', '?')}: {e}", file=sys.stderr)
                        except Exception:
                            pass

                    # Add conflict warning if applicable
                    if pr.conflict_message:
                        conflict_node = ConflictWarningNode(label=pr.conflict_message)
                        pr_node.add_child(conflict_node)

                    # Add blocking message if applicable
                    if pr.blocking_message:
                        blocked_node = BlockedMessageNode(label=pr.blocking_message)
                        pr_node.add_child(blocked_node)

                    # With CI hierarchy embedded, we no longer render separate flat running/failed check lines here.

                pr_section.add_child(branch_node)

            if pr_section.children:
                repo_node.add_child(pr_section)

            # Also show tracked branches that do NOT have an open PR.
            #
            # This is important for clones like dynamo3/ where you may have many remote-tracking
            # branches locally, but only a subset has active PRs at any given time.
            branches_section = SectionNode(label="Branches")
            for branch_name, info in sorted(branches_with_prs.items(), key=lambda kv: kv[0]):
                prs = pr_infos_by_branch.get(branch_name) or []
                if prs:
                    continue
                commit_url = f"https://github.com/{DYNAMO_REPO_SLUG}/commit/{info['sha']}" if info.get("sha") else None
                branches_section.add_child(
                    BranchInfoNode(
                        label=branch_name,
                        sha=info.get("sha"),
                        is_current=bool(info.get("is_current", False)),
                        commit_url=commit_url,
                        commit_time_pt=info.get("commit_time_pt"),
                        commit_datetime=info.get("commit_dt"),
                    )
                )

            if branches_section.children:
                repo_node.add_child(branches_section)

        # For non-dynamo repos: show branches but skip PR/CI lookup (treat as "any other repo").
        if not is_dynamo_repo:
            all_branches_section = SectionNode(label="Branches")

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
                all_branches_section.add_child(
                    BranchInfoNode(
                        label=b.get("name", ""),
                        sha=b.get("sha"),
                        is_current=bool(b.get("is_current", False)),
                        commit_time_pt=b.get("commit_time_pt"),
                        commit_datetime=b.get("commit_dt"),
                    )
                )

            if all_branches_section.children:
                repo_node.add_child(all_branches_section)

        # Add local-only branches (branches without any matching remote/tracking ref).
        if local_only_branches and is_dynamo_repo:
            local_section = SectionNode(label="Local-only branches")

            for branch_info in local_only_branches:
                branch_node = BranchInfoNode(
                    label=branch_info['name'],
                    sha=branch_info['sha'],
                    is_current=branch_info['is_current'],
                    commit_time_pt=branch_info.get('commit_time_pt'),
                    commit_datetime=branch_info.get('commit_dt'),
                )
                local_section.add_child(branch_node)

            repo_node.add_child(local_section)

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


def generate_html(root: BranchNode, *, page_stats: Optional[List[tuple[str, str]]] = None) -> str:
    """Generate HTML output from tree"""
    # Get current time in both UTC and PDT
    now_utc = datetime.now(ZoneInfo('UTC'))
    now_pdt = datetime.now(ZoneInfo('America/Los_Angeles'))

    # Format timestamps
    utc_str = now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')
    pdt_str = now_pdt.strftime('%Y-%m-%d %H:%M:%S %Z')

    # Render all children (skip root) into a single <pre> block payload, via the shared renderer.
    rendered_lines: list[str] = []
    for i, child in enumerate(root.children):
        is_last = i == len(root.children) - 1
        rendered_lines.extend(render_tree_pre_lines([child.to_tree_vm()]))
        if not is_last:
            rendered_lines.append("")
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
        copy_icon_svg=_COPY_ICON_SVG,
        tree_html=tree_html,
        page_stats=page_stats,
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
        '--refresh-closed-prs',
        action='store_true',
        help='Refresh cached closed/merged PR mappings (more GitHub API calls)'
    )
    parser.add_argument(
        '--html',
        action='store_true',
        help='Output in HTML format'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file path (default: <base_dir>/index.html when --html, else stdout)'
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
    args = parser.parse_args()

    base_dir = (args.repo_path or args.base_dir or Path.cwd()).resolve()

    # Prune locally-served raw logs to avoid unbounded growth and delete any partial/unverified artifacts.
    # We only render `[raw log]` links when the local file exists (or was materialized),
    # so pruning won't produce dead links on a freshly generated page.
    try:
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
    )
    # Report GitHub REST quota before/after the run (and fail fast if exhausted).
    before = scanner.github_api.get_core_rate_limit_info() if scanner.github_api else None
    if before:
        rem_b = before.get("remaining")
        lim_b = before.get("limit")
        reset_pt = before.get("reset_pt")
        secs = int(before.get("seconds_until_reset") or 0)
        print(
            f"GitHub API core quota (before): remaining={rem_b}"
            + (f"/{lim_b}" if lim_b is not None else "")
            + (f", resets at {reset_pt} (in {GitHubAPIClient._format_seconds_delta(secs)})" if reset_pt else "")
        )

    try:
        scanner.github_api.check_core_rate_limit_or_raise()
    except Exception as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(2)

    generation_t0 = time.monotonic()
    root = None
    try:
        root = scanner.scan_repositories(base_dir)
    finally:
        after = scanner.github_api.get_core_rate_limit_info() if scanner.github_api else None
        used = None
        reset_changed = False
        try:
            b_rem = (before or {}).get("remaining")
            a_rem = (after or {}).get("remaining")
            b_reset = (before or {}).get("reset_epoch")
            a_reset = (after or {}).get("reset_epoch")
            if b_reset is not None and a_reset is not None and int(b_reset) != int(a_reset):
                reset_changed = True
            if not reset_changed and b_rem is not None and a_rem is not None:
                used = int(b_rem) - int(a_rem)
                if used < 0:
                    used = None
                    reset_changed = True
        except Exception:
            used = None
            reset_changed = False
        if after:
            rem_a = after.get("remaining")
            lim_a = after.get("limit")
            reset_pt = after.get("reset_pt")
            secs = int(after.get("seconds_until_reset") or 0)
            msg = (
                f"GitHub API core quota (after): remaining={rem_a}"
                + (f"/{lim_a}" if lim_a is not None else "")
                + (f", resets at {reset_pt} (in {GitHubAPIClient._format_seconds_delta(secs)})" if reset_pt else "")
            )
            # Prefer per-run request accounting; quota deltas can be misleading if reset occurs mid-run.
            try:
                stats = scanner.github_api.get_rest_call_stats() if scanner.github_api else {}
                msg += f" | rest_calls={int(stats.get('total') or 0)}"
            except Exception:
                pass
            if reset_changed:
                msg += " | (rate limit window reset during run)"
            elif used is not None:
                msg += f" | used={used}"
            print(msg)

    # Output
    if args.html:
        assert root is not None
        # Page statistics (shown in an expandable block at the bottom of the HTML).
        elapsed_s = max(0.0, time.monotonic() - generation_t0)
        rest_calls = 0
        try:
            stats = scanner.github_api.get_rest_call_stats() if scanner.github_api else {}
            rest_calls = int(stats.get("total") or 0)
        except Exception:
            rest_calls = 0

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

        page_stats: List[tuple[str, str]] = [
            ("Generation time", f"{elapsed_s:.2f}s"),
            ("GitHub REST calls", str(rest_calls)),
            # "APIs left" (remaining quota). Best-effort; do not fail page generation.
            ("GitHub core quota remaining", ""),
            ("Repos scanned", str(len(getattr(root, 'children', []) or []))),
            ("PRs shown", str(pr_count)),
        ]

        # Show last REST error (e.g. 403 token policy) to make missing PR/snippet causes obvious.
        try:
            es = scanner.github_api.get_rest_error_stats() if scanner.github_api else {}
            last = (es or {}).get("last") or {}
            if isinstance(last, dict) and last.get("status"):
                code = last.get("status")
                body = str(last.get("body") or "").strip()
                if len(body) > 160:
                    body = body[:160].rstrip() + "…"
                page_stats.append(("GitHub REST last error", f"{code}: {body}" if body else str(code)))
        except Exception:
            pass

        # Fill in GitHub quota (remaining/limit + reset time) if available.
        try:
            rl = scanner.github_api.get_core_rate_limit_info() if scanner.github_api else None
            if rl:
                rem = rl.get("remaining")
                lim = rl.get("limit")
                reset_pt = rl.get("reset_pt")
                if rem is not None and lim is not None:
                    page_stats[2] = ("GitHub core quota remaining", f"{rem}/{lim}")
                elif rem is not None:
                    page_stats[2] = ("GitHub core quota remaining", str(rem))
                else:
                    page_stats[2] = ("GitHub core quota remaining", "(unknown)")
                if reset_pt:
                    page_stats.append(("GitHub core quota resets", str(reset_pt)))
        except Exception:
            # Keep the row but avoid breaking HTML generation.
            page_stats[2] = ("GitHub core quota remaining", "(unknown)")

        # Include relevant knobs if set (helps explain “why so many API calls?”).
        if args.max_branches is not None:
            page_stats.append(("max_branches", str(args.max_branches)))
        if args.max_checks_fetch is not None:
            page_stats.append(("max_checks_fetch", str(args.max_checks_fetch)))
        if bool(args.refresh_closed_prs):
            page_stats.append(("refresh_closed_prs", "true"))

        html_output = generate_html(root, page_stats=page_stats)
        out_path = args.output
        if out_path is None:
            out_path = base_dir / "index.html"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html_output)
    else:
        assert root is not None
        for child in root.children:
            child.print_tree()
            print()


if __name__ == '__main__':
    main()
