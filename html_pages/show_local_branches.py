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
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
import functools
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
    CIStatus,
    EXPECTED_CHECK_PLACEHOLDER_SYMBOL,
    PASS_PLUS_STYLE,
    TreeNodeVM,
    check_line_html,
    ci_should_expand_by_default,
    compact_ci_summary_html,
    disambiguate_check_run_name,
    extract_actions_job_id_from_url,
    sort_pr_check_rows_by_name,
    render_tree_pre_lines,
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

#
# Repo constants (avoid scattering hardcoded strings)
#

DYNAMO_OWNER = "ai-dynamo"
DYNAMO_REPO = "dynamo"
DYNAMO_REPO_SLUG = f"{DYNAMO_OWNER}/{DYNAMO_REPO}"

class RawLogValidationError(RuntimeError):
    """Raised when we expect a local `[raw log]` for a failed Actions job but cannot produce one."""


#
# Small HTML helpers
#

_COPY_BTN_STYLE = (
    "padding: 1px 4px; font-size: 10px; line-height: 1; background-color: transparent; color: #57606a; "
    "border: 1px solid #d0d7de; border-radius: 5px; cursor: pointer; display: inline-flex; "
    "align-items: center; vertical-align: baseline; margin-right: 4px;"
)


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
        # Default behavior: **collapsible** node when it has children.
        #
        # UX: users expect a triangle (▶/▼) to expand/collapse branch sections and PR details.
        # If we mark these nodes as non-collapsible, `render_tree_pre_lines()` will emit an empty
        # placeholder span instead of the triangle, which looks broken/misaligned.
        key = f"{self.__class__.__name__}:{self.label}"
        has_children = bool(self.children)
        return TreeNodeVM(
            node_key=key,
            label_html=self._format_html_content(),
            children=[c.to_tree_vm() for c in (self.children or [])],
            collapsible=bool(has_children),
            # Preserve historical behavior: expanded by default, but now user-toggleable.
            default_expanded=True,
        )

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
    # Stable context key (repo/branch/SHA/PR) to make DOM ids stable across regenerations.
    context_key: str = ""

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
        if node.status in {CIStatus.SUCCESS, CIStatus.SKIPPED}:
            return True
        if node.status != CIStatus.UNKNOWN:
            return False
        dur = (node.duration or "").strip().lower()
        return dur in {"0", "0s", "0m", "0m0s", "0h", "0h0m", "0h0m0s"}

    @staticmethod
    def _subtree_rollup(node: "CIJobTreeNode") -> "CIJobTreeNode._Rollup":
        """Compute a 'worst descendant' status for icon rendering."""
        # Success-like entire subtree => success rollup.
        if CIJobTreeNode._subtree_all_success(node):
            return CIJobTreeNode._Rollup(status=CIStatus.SUCCESS.value, has_required_failure=False, has_optional_failure=False)

        has_required_failure = False
        has_optional_failure = False
        statuses: List[str] = []

        def walk(n: "CIJobTreeNode") -> None:
            nonlocal has_required_failure, has_optional_failure
            st = getattr(n, "status", CIStatus.UNKNOWN) or CIStatus.UNKNOWN
            is_req = bool(getattr(n, "is_required", False))
            if st == CIStatus.FAILURE and is_req:
                has_required_failure = True
                statuses.append(CIStatus.FAILURE.value)
            elif st == CIStatus.FAILURE and not is_req:
                # Optional failures should not turn the parent red (FAIL), but they SHOULD be visible.
                has_optional_failure = True
                statuses.append(CIStatus.FAILURE.value)
            elif st == CIStatus.SKIPPED:
                # Skipped is success-like for rollup purposes (do not make parents look "worse").
                statuses.append(CIStatus.SUCCESS.value)
            else:
                statuses.append(str(st))
            for ch in (getattr(n, "children", None) or []):
                if isinstance(ch, CIJobTreeNode):
                    walk(ch)

        walk(node)

        # Worst-first priority for status rollup.
        priority = [
            CIStatus.FAILURE.value,
            CIStatus.IN_PROGRESS.value,
            CIStatus.PENDING.value,
            CIStatus.CANCELLED.value,
            CIStatus.UNKNOWN.value,
            CIStatus.SUCCESS.value,
        ]
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
        - expand for required failures (red ✗)
        - expand if ANY descendant is running/pending (so active work is visible)
        - do NOT auto-expand for optional failures only
        """
        # All-success-like subtree: no need to expand.
        if CIJobTreeNode._subtree_all_success(node):
            return False

        # Optional failures (warnings) alone should not force expansion.
        # We therefore expand only if we see a required failure.
        if getattr(node, "status", CIStatus.UNKNOWN) == CIStatus.FAILURE and bool(getattr(node, "is_required", False)):
            return True

        # Expand if ANY descendant is in progress or pending/queued, regardless of optional failures.
        try:
            def walk_running(n: "CIJobTreeNode") -> bool:
                st = str(getattr(n, "status", "") or "").strip().lower()
                if st in {CIStatus.IN_PROGRESS.value, CIStatus.PENDING.value, "running", "building"}:
                    return True
                for ch in (getattr(n, "children", None) or []):
                    if isinstance(ch, CIJobTreeNode) and walk_running(ch):
                        return True
                return False

            if walk_running(node):
                return True
        except Exception:
            pass

        # Shared rule: expand only for required failures or non-completed states.
        roll = CIJobTreeNode._subtree_rollup(node)
        if ci_should_expand_by_default(rollup_status=str(roll.status or ""), has_required_failure=bool(roll.has_required_failure)):
            return True
        # Otherwise, no auto-expand (even if some leaf steps are "unknown").
        return False

    @staticmethod
    def _subtree_all_success(node: "CIJobTreeNode") -> bool:
        """Return True iff this node and all descendants are in a success-like state.

        Note: Some GitHub statuses come back as "unknown" from `gh pr checks` parsing.
        If the node is unknown but all its children are success-like, we treat the subtree
        as success-like for *default expand/collapse* purposes.
        """
        # Any non-success/unknown state should be visible by default.
        if node.status not in {CIStatus.SUCCESS, CIStatus.SKIPPED, CIStatus.UNKNOWN}:
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
        jid = self.job_id
        name_part = f" — {self.display_name}" if (self.display_name and self.display_name != self.job_id) else ""
        dur_part = f" ({self.duration})" if self.duration else ""
        return f"{jid}{name_part}{dur_part}"

    def _format_html_content(self) -> str:
        # Prefer FailedCheck URLs when available (they include raw log + error summary)
        job_url = ""
        if self.failed_check is not None:
            job_url = str(getattr(self.failed_check, "job_url", "") or "")
        if not job_url:
            job_url = self.url or ""
        # Icon policy:
        # - show this node's *own* status
        # - if the node is successful but any descendant failed, append a failure marker (✓/✗)
        roll = self._subtree_rollup(self) if self.children else None
        desc_required_failure = bool(roll.has_required_failure) if roll is not None else False
        desc_optional_failure = bool(roll.has_optional_failure) if roll is not None else False
        effective_status = self.status

        jid = self.job_id
        display_name_eff = self.display_name

        # Error toggle is appended in to_tree_vm() so it can inject a newline safely in <pre>.
        return check_line_html(
            job_id=jid,
            display_name=display_name_eff,
            status_norm=effective_status,
            is_required=bool(self.is_required),
            duration=self.duration,
            log_url=job_url,
            raw_log_href=self.raw_log_href,
            raw_log_size_bytes=int(self.raw_log_size_bytes or 0),
            error_snippet_text=(self.error_snippet_text or ""),
            # For success nodes, this will control the suffix style (required vs optional failure).
            required_failure=bool(desc_required_failure),
        warning_present=bool(effective_status == CIStatus.SUCCESS and (desc_required_failure or desc_optional_failure)),
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
            node_key=f"CI:{self.context_key}:{self.job_id}:{self.url}",
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
    validate_raw_logs: bool = True,
) -> List[BranchNode]:
    """Build a flat CI list that matches the PR "Details" table 1:1."""
    if not pr or not getattr(pr, "number", None) or not github_api:
        return []
    # Ensure required-ness is correct even if PRInfo cache is stale.
    # This uses `gh` (GraphQL statusCheckRollup isRequired), not our REST budget.
    required_set: Set[str] = set(getattr(pr, "required_checks", []) or [])
    if not required_set:
        try:
            required_set = set(github_api.get_required_checks(DYNAMO_OWNER, DYNAMO_REPO, int(pr.number)) or set())
        except Exception:
            required_set = set()
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

        # Expected checks inferred from workflow YAML (PyYAML assumed present).
        try:
            from common_github_workflow import expected_check_names_from_workflows, workflow_paths_for_present_checks  # local import

            present_names = [
                str(getattr(r, "name", "") or "").strip()
                for r in (rows or [])
                if str(getattr(r, "name", "") or "").strip()
            ]
            wf_paths = workflow_paths_for_present_checks(repo_root=Path(repo_path), check_names=list(present_names or []))
            expected_from_yml = expected_check_names_from_workflows(
                repo_root=Path(repo_path),
                workflow_paths=(wf_paths or None) if wf_paths else None,
                cap=200,
            )
            for nm0 in (expected_from_yml or []):
                n0 = common.normalize_check_name(str(nm0 or ""))
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
        except Exception:
            pass
    except Exception:
        pass

    # Sort by job name for stable/scan-friendly output (same order as Details list).
    rows = sort_pr_check_rows_by_name(list(rows))

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

        node = CIJobTreeNode(
            label="",
            job_id=nm,
            display_name=str(display_name or ""),
            status=str(st or "unknown"),
            duration=str(getattr(r, "duration", "") or ""),
            url=job_url,
            raw_log_href=str(raw_href or ""),
            raw_log_size_bytes=int(raw_size or 0),
            # If we can attribute the failure to a specific step, we suppress the parent-level
            # error tags/snippet to avoid duplication (the failing step will carry it).
            error_snippet_text=str(snippet or ""),
            is_required=bool(getattr(r, "is_required", False)),
            children=[],
        )

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
                    CIJobTreeNode(
                        label="",
                        job_id=sub_id,
                        display_name="",
                        status=str(sub_status or "unknown"),
                        duration=str(sub_dur or ""),
                        url=job_url,
                        raw_log_href="",
                        raw_log_size_bytes=0,
                        error_snippet_text=str(step_snip or ""),
                        is_required=bool(getattr(r, "is_required", False)),
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

    # Second pass: best-effort grouping by workflow `jobs.*.needs` (YAML).
    try:
        from common_github_workflow import group_ci_nodes_by_workflow_needs  # local import

        out = list(
            group_ci_nodes_by_workflow_needs(
                repo_root=Path(repo_path),
                items=[
                    (
                        # IMPORTANT: use the actual check/job name for workflow matching.
                        #
                        # `display_name` is for UI disambiguation (and may be a placeholder symbol like "◇"),
                        # and must NOT be used as the workflow grouping key.
                        re.sub(r"^[a-z]+:\\s+", "", str(getattr(n, "job_id", "") or ""), flags=re.IGNORECASE),
                        n,
                    )
                    for n in (out or [])
                ],
            )
            or []
        )
    except Exception:
        pass

    # Post-pass: if a parent has both (amd64) and (arm64) children, group them by arch:
    # - non-arch children first
    # - all (amd64)
    # - all (arm64)
    try:
        _ARCH_RE = re.compile(r"\((amd64|arm64)\)", re.IGNORECASE)

        def _arch_rank(job_id: str) -> int:
            s = str(job_id or "")
            m = _ARCH_RE.search(s)
            if not m:
                return 0
            a = str(m.group(1) or "").strip().lower()
            if a == "amd64":
                return 1
            if a == "arm64":
                return 2
            return 3

        def walk(n: BranchNode) -> None:
            kids = list(getattr(n, "children", None) or [])
            for ch in kids:
                if isinstance(ch, BranchNode):
                    walk(ch)
            # Only reorder CIJobTreeNode children; preserve other tree sections.
            if not isinstance(n, CIJobTreeNode):
                return
            kids2 = [k for k in kids if isinstance(k, CIJobTreeNode)]
            if not kids2:
                return
            ranks = [_arch_rank(getattr(k, "job_id", "")) for k in kids2]
            if not (any(r == 1 for r in ranks) and any(r == 2 for r in ranks)):
                return
            buckets = {0: [], 1: [], 2: [], 3: []}
            for k in kids2:
                buckets[_arch_rank(getattr(k, "job_id", ""))].append(k)
            # Keep any non-CIJobTreeNode children (rare) at the front in original order.
            non_ci = [k for k in kids if not isinstance(k, CIJobTreeNode)]
            n.children = non_ci + buckets[0] + buckets[1] + buckets[2] + buckets[3]

        for top in (out or []):
            if isinstance(top, BranchNode):
                walk(top)
    except Exception:
        pass

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
        # If the repository directory itself is a symlink, show the target for clarity.
        link_suffix = ""
        try:
            p = Path(self.path) if self.path is not None else None
            if p is not None and p.is_symlink():
                tgt = ""
                try:
                    tgt = os.readlink(p)
                except Exception:
                    tgt = ""
                if tgt:
                    link_suffix = f" -> {tgt}"
        except Exception:
            link_suffix = ""

        if self.error:
            return f"\033[1m{self.label}\033[0m{link_suffix}\n  \033[91m⚠️  {self.error}\033[0m"
        return f"\033[1m{self.label}\033[0m{link_suffix}"

    def _format_html_content(self) -> str:
        # Make repo name clickable (relative URL to the directory)
        # Remove trailing slash from label for URL
        repo_dirname = self.label.rstrip('/')
        repo_link = f'<a href="{repo_dirname}/" class="repo-name">{self.label}</a>'

        # If the repository directory itself is a symlink, show the target for clarity.
        #
        # Keep this compact; show the raw link target (often relative) and include the resolved
        # absolute target as a tooltip.
        link_suffix = ""
        try:
            p = Path(self.path) if self.path is not None else None
            if p is not None and p.is_symlink():
                tgt = ""
                try:
                    tgt = os.readlink(p)
                except Exception:
                    tgt = ""
                resolved = ""
                try:
                    resolved = str(p.resolve())
                except Exception:
                    resolved = ""
                if tgt:
                    title = f' title="{html.escape(resolved, quote=True)}"' if resolved else ""
                    link_suffix = (
                        f' <span style="color: #57606a; font-size: 12px; user-select: none;"{title}>'
                        f'→ {html.escape(str(tgt), quote=False)}'
                        f"</span>"
                    )
        except Exception:
            link_suffix = ""

        if self.error:
            return f'{repo_link}{link_suffix}\n<span class="error">⚠️  {self.error}</span>'
        return f"{repo_link}{link_suffix}"

    def to_tree_vm(self) -> TreeNodeVM:
        """Repo nodes are normally expandable, but symlink repos should be inert (no expansion).

        UX: if a repo directory is a symlink, users asked to avoid showing/expanding any nested info
        (branches/PRs/CI) since it is often just a pointer into another location.
        """
        key = f"{self.__class__.__name__}:{self.label}"
        try:
            p = Path(self.path) if self.path is not None else None
            if p is not None and p.is_symlink():
                return TreeNodeVM(
                    node_key=key,
                    label_html=self._format_html_content(),
                    children=[],
                    collapsible=False,
                    default_expanded=False,
                    noncollapsible_icon="square",
                )
        except Exception:
            pass

        # Default behavior for normal (non-symlink) repos: collapsible when it has children.
        has_children = bool(self.children)
        return TreeNodeVM(
            node_key=key,
            label_html=self._format_html_content(),
            children=[c.to_tree_vm() for c in (self.children or [])],
            collapsible=bool(has_children),
            default_expanded=True,
        )


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
            # Closed/unavailable PR: don't prepend a failure icon on the PR title line.
            emoji = ''

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
        pr_url = str(self.pr.url or "").strip()
        gh_icon = ""
        if pr_url:
            # Match the GitHub icon used in show_commit_history.j2
            gh_icon = (
                f'<a href="{html.escape(pr_url, quote=True)}" target="_blank" '
                f'style="text-decoration: none; color: #24292f; margin-right: 4px;" '
                f'title="Open on GitHub">'
                f'<svg height="14" width="14" viewBox="0 0 16 16" fill="currentColor" '
                f'style="display: inline-block; vertical-align: middle;">'
                f'<path fill-rule="evenodd" clip-rule="evenodd" '
                f'd="M8 0C3.58 0 0 3.58 0 8C0 11.54 2.29 14.53 5.47 15.59C5.87 15.66 6.02 15.42 6.02 15.21C6.02 15.02 6.01 14.39 6.01 13.72C4 14.09 3.48 13.23 3.32 12.78C3.23 12.55 2.84 11.84 2.5 11.65C2.22 11.5 1.82 11.13 2.49 11.12C3.12 11.11 3.57 11.7 3.72 11.94C4.44 13.15 5.59 12.81 6.05 12.6C6.12 12.08 6.33 11.73 6.56 11.53C4.78 11.33 2.92 10.64 2.92 7.58C2.92 6.71 3.23 5.99 3.74 5.43C3.66 5.23 3.38 4.41 3.82 3.31C3.82 3.31 4.49 3.1 6.02 4.13C6.66 3.95 7.34 3.86 8.02 3.86C8.7 3.86 9.38 3.95 10.02 4.13C11.55 3.09 12.22 3.31 12.22 3.31C12.66 4.41 12.38 5.23 12.3 5.43C12.81 5.99 13.12 6.7 13.12 7.58C13.12 10.65 11.25 11.33 9.47 11.53C9.76 11.78 10.01 12.26 10.01 13.01C10.01 14.08 10 14.94 10 15.21C10 15.42 10.15 15.67 10.55 15.59C13.71 14.53 16 11.53 16 8C16 3.58 12.42 0 8 0Z"></path>'
                f'</svg></a>'
            )
        state_lc = (self.pr.state or "").lower()
        if self.pr.is_merged:
            emoji = '🔀'
        elif state_lc == 'open':
            emoji = ''
        else:
            # Closed/unavailable PR: don't prepend a failure icon on the PR title line.
            emoji = ''

        # Truncate title at 80 characters
        title = self.pr.title[:80] + '...' if len(self.pr.title) > 80 else self.pr.title

        base_html = ""
        if self.pr.base_ref:
            base_html = f' <span style="font-weight: 700;">→ {html.escape(self.pr.base_ref)}</span> branch'

        # Prefer standard GitHub-style formatting: "title (#1234)" over "PR#1234: title"
        pr_suffix = f"(#{self.pr.number})"
        if pr_suffix not in title:
            title = f"{title} {pr_suffix}"

        # Make the "(#1234)" part a link to the PR (in addition to the GitHub icon).
        title_html: str
        if pr_url and self.pr.number:
            pr_url_esc = html.escape(pr_url, quote=True)
            suffix_html = (
                f'(<a href="{pr_url_esc}" target="_blank" '
                f'style="color: #0969da; text-decoration: none;" '
                f'title="Open PR #{int(self.pr.number)}">#{int(self.pr.number)}</a>)'
            )
            try:
                before, after = str(title).rsplit(pr_suffix, 1)
                title_html = f"{html.escape(before)}{suffix_html}{html.escape(after)}"
            except Exception:
                title_html = html.escape(title).replace(html.escape(pr_suffix), suffix_html)
        else:
            title_html = html.escape(title)

        # Gray out merged PRs
        if self.pr.is_merged:
            # For merged PRs: keep grey style; link is still the GitHub icon.
            return f'<span style="color: #999;">{emoji} {gh_icon}{title_html}{base_html}</span>'
        else:
            # For open PRs: title is plain text; GitHub icon is the link.
            prefix = f"{emoji} " if emoji else ""
            return f'{prefix}{gh_icon}{title_html}{base_html}'


@dataclass
class PRURLNode(BranchNode):
    """PR URL node"""
    url: Optional[str] = None

    def _format_content(self) -> str:
        # (HTML-only) URL is already visible in the PR title line.
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
    # Stable context key (repo/branch/SHA) so the main triangle is stable across refreshes.
    context_key: str = ""

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
                # Refresh required checks directly so FAIL/PASS is correct even when PRInfo cache is stale
                # or we are in "no REST budget" mode. This call uses `gh` GraphQL and is cached on disk.
                if self.github_api and (not required_set):
                    try:
                        required_set = set(
                            self.github_api.get_required_checks(DYNAMO_OWNER, DYNAMO_REPO, int(self.pr.number)) or set()
                        )
                    except Exception:
                        required_set = required_set
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
                    # Details list should mirror the PR's check rows as returned by GitHub (no extra
                    # workflow/run metadata fetches, and no YAML parsing).
                    rows = list(rows)
                    # Sort by job name for stable/scan-friendly output.
                    rows = sort_pr_check_rows_by_name(list(rows))

                    # Compact CI summary counts (styled to match show_commit_history.j2):
                    #   - green:  REQ+OPT✓  (passed; required count bold)
                    #   - red:    N✗   (required failures, rendered as a red badge)
                    #   - red:    N✗   (optional failures)
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

                    # Shared summary (common.py) so show_commit_history + show_local_branches stay consistent.
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
                            '<strong style="color: #c83a3a;"><span style="display: inline-flex; align-items: center; justify-content: center; width: 14px; height: 14px; margin-right: 6px; border-radius: 999px; background-color: #c83a3a; color: #ffffff; font-size: 11px; font-weight: 900; line-height: 1;">✗</span>Failed (required):</strong> '
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
            CIJobTreeNode._subtree_needs_attention(c)
            for c in (self.children or [])
            if isinstance(c, CIJobTreeNode)
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

    def to_tree_vm(self) -> TreeNodeVM:
        # Default expansion: only when something needs attention in the CI subtree.
        default_expanded = any(
            CIJobTreeNode._subtree_needs_attention(c)
            for c in (self.children or [])
            if isinstance(c, CIJobTreeNode)
        )
        return TreeNodeVM(
            node_key=f"PRStatus:{self.context_key}:{getattr(self.pr, 'number', '')}",
            label_html=self._format_html_content(),
            children=[c.to_tree_vm() for c in (self.children or []) if isinstance(c, BranchNode)],
            collapsible=True,
            default_expanded=bool(default_expanded),
            triangle_tooltip="CI tree (flat; mirrors Details)",
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

        # Symlink repos are intentionally treated as "pointers": show the repo line (with → target),
        # but do not scan/render any nested info (branches/PR/CI). Also render as non-expandable in the UI.
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
                    if bool(getattr(self, "cache_only_github", False)):
                        allow_fetch_checks = False
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
                        context_key=f"{repo_dir.name}:{branch_name}:{info.get('sha','')}",
                    )
                    pr_node.add_child(status_node)

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
                            try:
                                if isinstance(ci_node, CIJobTreeNode):
                                    ci_node.context_key = str(status_node.context_key or "")
                            except Exception:
                                pass
                            status_node.add_child(ci_node)
                    except RawLogValidationError:
                        # Hard fail: for failed Actions jobs we require `[raw log]` links.
                        raise
                    except Exception as e:
                        # Don't silently drop the tree UI; surface the error so it's actionable.
                        try:
                            status_node.add_child(
                                BranchNode(
                                    label=f'<span style="color: #c83a3a;">✗ CI hierarchy error: {html.escape(str(e))}</span>'
                                )
                            )
                        except Exception:
                            pass
                        try:
                            self.logger.warning(
                                "[show_local_branches] CI hierarchy error for PR #%s: %s",
                                getattr(pr, "number", "?"),
                                e,
                            )
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
        copy_icon_svg=_COPY_ICON_SVG,
        tree_html=tree_html,
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
    # Note: we want the HTML page to show a *breakdown* (timing.prune/scan/render/write/total).
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
            # Keep logical order for human scanning; only sort timing.* rows.
            try:
                timing = sorted(
                    [(k, v) for (k, v) in page_stats if str(k).startswith("timing.")],
                    key=lambda kv: kv[0],
                )
                other = [(k, v) for (k, v) in page_stats if not str(k).startswith("timing.")]
                page_stats[:] = other + timing
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
                    _upsert_stat(f"timing.{k}", f"{float(tdict.get(k) or 0.0):.2f}s")
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
