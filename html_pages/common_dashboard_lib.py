#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Common dashboard helpers shared by HTML generators under dynamo-utils/html_pages/.

This file intentionally groups previously-split helper modules into one place to:
- avoid UI drift between dashboards
- reduce small-module sprawl
- keep <pre>-safe tree rendering + check-line rendering consistent
- centralize workflow-graph parsing (derived from .github/workflows/*.yml)
"""

from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ======================================================================================
# Shared UI snippets
# ======================================================================================

# Style for the optional pass count in the "REQ+OPT✓" compact CI summary.
PASS_PLUS_STYLE = "font-size: 10px; font-weight: 600; opacity: 0.9;"


# ======================================================================================
# Shared tree UI rendering (<pre>-safe)
# ======================================================================================

@dataclass(frozen=True)
class TreeNodeVM:
    """View-model for a single tree node line.

    - label_html: full HTML for the line content (excluding the tree connectors).
    - children: child nodes.
    - collapsible: if True, render a triangle placeholder (▶/▼) and allow toggling children.
    - default_expanded: initial state for collapsible nodes.
    - triangle_tooltip: optional title for the triangle element.
    - node_key: a stable key for the logical node (used for debugging/caching, not DOM ids).
    """

    node_key: str
    label_html: str
    children: List["TreeNodeVM"] = field(default_factory=list)
    collapsible: bool = False
    default_expanded: bool = False
    triangle_tooltip: Optional[str] = None


def _hash10(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def _triangle_html(*, expanded: bool, children_id: str, tooltip: Optional[str], parent_children_id: Optional[str]) -> str:
    ch = "▼" if expanded else "▶"
    title_attr = f' title="{html.escape(tooltip or "", quote=True)}"' if tooltip else ""
    parent_attr = (
        f' data-parent-children-id="{html.escape(parent_children_id, quote=True)}"'
        if parent_children_id
        else ""
    )
    return (
        f'<span style="display: inline-block; width: 12px; margin-right: 2px; color: #0969da; '
        f'cursor: pointer; user-select: none;"{title_attr} '
        f'data-children-id="{html.escape(children_id, quote=True)}"{parent_attr} '
        f'onclick="toggleTreeChildren(\'{html.escape(children_id, quote=True)}\', this)">{ch}</span>'
    )


def _triangle_placeholder_html() -> str:
    return '<span style="display: inline-block; width: 12px; margin-right: 2px;"></span>'


def render_tree_pre_lines(root_nodes: List[TreeNodeVM]) -> List[str]:
    """Render a forest into lines suitable to be joined with \n and placed inside <pre>."""

    out: List[str] = []
    # NOTE: This function may be called multiple times per HTML page (e.g. per-repo / per-PR).
    # DOM ids must be unique across the whole document, not just per call.
    global _TREE_RENDER_CALL_SEQ
    try:
        _TREE_RENDER_CALL_SEQ += 1
    except NameError:
        _TREE_RENDER_CALL_SEQ = 1
    render_call_id = _TREE_RENDER_CALL_SEQ
    next_dom_id = 0

    def alloc_children_id() -> str:
        # Must be unique within the document. Deterministic IDs (e.g., hashed from labels)
        # are not safe because the same logical node label/template can appear many times.
        nonlocal next_dom_id
        next_dom_id += 1
        return f"tree_children_{render_call_id:x}_{next_dom_id:x}"

    def render_node(
        node: TreeNodeVM,
        prefix: str,
        is_last: bool,
        is_root: bool,
        parent_children_id: Optional[str],
    ) -> None:
        # Tree connector
        if not is_root:
            connector = "└─" if is_last else "├─"
            current_prefix = prefix + connector + " "
        else:
            current_prefix = ""

        # Triangle UI
        children_id: Optional[str] = None
        if node.collapsible:
            if node.children:
                children_id = alloc_children_id()
                tri = _triangle_html(
                    expanded=bool(node.default_expanded),
                    children_id=children_id,
                    tooltip=node.triangle_tooltip,
                    parent_children_id=parent_children_id,
                )
            else:
                tri = _triangle_placeholder_html()
        else:
            tri = ""

        line = (node.label_html or "").strip()
        if line:
            out.append(current_prefix + tri + line)

        # Children rendering
        if not node.children:
            return

        # Compute child prefix continuation
        if is_root:
            child_prefix = ""
        else:
            child_prefix = prefix + ("   " if is_last else "│  ")

        # If collapsible: wrap children in a span appended to the end of the parent line.
        if node.collapsible and children_id and out:
            disp = "inline" if node.default_expanded else "none"
            out[-1] = out[-1] + f'<span id="{children_id}" style="display: {disp};">'
            child_lines_start_idx = len(out)

            for idx, ch in enumerate(node.children):
                render_node(
                    ch,
                    child_prefix,
                    idx == len(node.children) - 1,
                    False,
                    children_id,
                )

            # Close children span on the last rendered child line, or on the parent line if no child line rendered.
            if len(out) > child_lines_start_idx:
                out[-1] = out[-1] + "</span>"
            else:
                out[-1] = out[-1] + "</span>"
            return

        # Non-collapsible: always render children normally.
        for idx, ch in enumerate(node.children):
            render_node(
                ch,
                child_prefix,
                idx == len(node.children) - 1,
                False,
                parent_children_id,
            )

    for i, n in enumerate(root_nodes):
        render_node(n, prefix="", is_last=(i == len(root_nodes) - 1), is_root=True, parent_children_id=None)

    return out


# ======================================================================================
# Shared GitHub/GitLab check/job line rendering (HTML)
# ======================================================================================

def required_badge_html(*, is_required: bool, status_norm: str) -> str:
    """Render a [REQUIRED] badge with shared semantics."""
    if not is_required:
        return ""

    s = (status_norm or "").strip().lower()
    if s == "failure":
        color = "#d73a49"
        weight = "700"
    elif s == "success":
        color = "#2da44e"
        weight = "400"
    else:
        color = "#57606a"
        weight = "400"

    return f' <span style="color: {color}; font-weight: {weight};">[REQUIRED]</span>'


def status_icon_html(
    *,
    status_norm: str,
    is_required: bool,
    required_failure: bool = False,
    warning_present: bool = False,
) -> str:
    """Shared status icon HTML (match show_dynamo_branches)."""
    s = (status_norm or "").strip().lower()

    if s == "success":
        out = (
            '<span style="color: #2da44e; display: inline-flex; vertical-align: text-bottom;">'
            '<svg aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="12" height="12" '
            'data-view-component="true" class="octicon octicon-check-circle-fill" fill="currentColor">'
            '<path fill-rule="evenodd" '
            'd="M8 16A8 8 0 108 0a8 8 0 000 16zm3.78-9.78a.75.75 0 00-1.06-1.06L7 9.94 5.28 8.22a.75.75 0 10-1.06 1.06l2 2a.75.75 0 001.06 0l4-4z">'
            "</path></svg></span>"
        )
        if bool(warning_present):
            out += '<span style="color: #f59e0b; font-size: 13px; font-weight: 900; line-height: 1; margin-left: 2px;">⚠</span>'
        return out
    if s == "failure":
        if is_required or required_failure:
            return (
                '<span style="display: inline-flex; align-items: center; justify-content: center; '
                'width: 12px; height: 12px; border-radius: 999px; background-color: #d73a49; '
                'color: #ffffff; font-size: 10px; font-weight: 900; line-height: 1;">✗</span>'
            )
        return '<span style="color: #f59e0b; font-weight: 900;">⚠</span>'
    if s == "in_progress":
        return '<span style="color: #dbab09;">⏳</span>'
    if s == "pending":
        return (
            '<span style="display: inline-flex; align-items: center; justify-content: center; '
            'width: 12px; height: 12px; border-radius: 999px; background-color: #8c959f; '
            'color: #ffffff; font-size: 10px; font-weight: 900; line-height: 1;">•</span>'
        )
    if s == "cancelled":
        return (
            '<span style="display: inline-flex; align-items: center; justify-content: center; '
            'width: 12px; height: 12px; border-radius: 999px; background-color: #8c959f; '
            'color: #ffffff; font-size: 10px; font-weight: 900; line-height: 1;">×</span>'
        )
    return '<span style="color: #8c959f;">•</span>'


def _small_link_html(*, url: str, label: str) -> str:
    if not url:
        return ""
    return (
        f' <a href="{html.escape(url, quote=True)}" target="_blank" '
        f'style="color: #0969da; font-size: 11px; margin-left: 5px; text-decoration: none;">{html.escape(label)}</a>'
    )


from common_log_errors import format_snippet_html as _format_snippet_html  # shared implementation
from common_log_errors import snippet_categories as _snippet_categories


def _tag_pill_html(*, text: str, monospace: bool = False, kind: str = "category") -> str:
    """Render a small tag/pill for inline display next to links.

    kind:
      - "category": orange-ish pill (high-level error category)
      - "command": gray pill (detected command; not the root cause)
    """
    t = (text or "").strip()
    if not t:
        return ""
    font = "SFMono-Regular, Consolas, Liberation Mono, Menlo, monospace" if monospace else "inherit"
    if kind == "command":
        border = "#d0d7de"
        bg = "#f6f8fa"
        fg = "#57606a"
    else:
        border = "#d4a72c"
        bg = "#fff8c5"
        fg = "#7d4e00"
    return (
        f' <span style="display: inline-block; vertical-align: baseline; '
        f'border: 1px solid {border}; background: {bg}; color: {fg}; '
        f'border-radius: 999px; padding: 1px 6px; font-size: 10px; '
        f'font-weight: 600; line-height: 1; font-family: {font};">'
        f"{html.escape(t)}</span>"
    )


def _snippet_first_command(snippet_text: str) -> str:
    """Extract the first command (1-line) from the snippet header, if present."""
    text = (snippet_text or "").strip()
    if not text:
        return ""
    lines = text.splitlines()
    in_cmd = False
    collected: list[str] = []
    for ln in lines:
        s = ln.rstrip("\n")
        if not in_cmd:
            if s.strip().lower().startswith("commands:"):
                in_cmd = True
            continue
        # Stop at the first blank line (headers are separated from snippet body by a blank line).
        if not s.strip():
            break
        collected.append(s)
        # We only need the first command. If the next line looks like a continuation, we'll
        # still compress it into a single line with " …".
        if len(collected) >= 1:
            break
    if not collected:
        return ""
    first = collected[0].lstrip()
    # Commands are formatted like "  <cmd>".
    if first.startswith("- "):
        first = first[2:].lstrip()
    # Normalize whitespace for inline display.
    first = re.sub(r"\s+", " ", first).strip()
    if len(first) > 90:
        first = first[:90].rstrip() + "…"
    return first


def _error_snippet_toggle_html(*, dom_id_seed: str, snippet_text: str) -> str:
    global _ERROR_SNIP_SEQ
    try:
        _ERROR_SNIP_SEQ += 1
    except NameError:
        _ERROR_SNIP_SEQ = 1
    err_id = f"err_snip_{_ERROR_SNIP_SEQ:x}"
    shown = _format_snippet_html(snippet_text or "")
    if not shown:
        shown = '<span style="color: #57606a;">(no snippet found)</span>'
    return (
        f' <span style="cursor: pointer; color: #0969da; font-size: 11px; margin-left: 5px; '
        f'text-decoration: none; font-weight: 500; user-select: none;" '
        f'onclick="toggleErrorSnippet(\'{html.escape(err_id, quote=True)}\', this)">▶ Snippet</span>'
        f'<span id="{html.escape(err_id, quote=True)}" style="display: none;">'
        f"<br>"
        f'<span style="display: inline-block; border: 1px solid #d0d7de; background: #f6f8fa; '
        f'border-radius: 6px; padding: 6px 8px; white-space: pre-wrap; color: #24292f;">'
        f'<span style="color: #d73a49; font-weight: 700;">Snippet:</span> {shown}'
        f"</span>"
        f"</span>"
    )


def _format_bytes_short(n_bytes: int) -> str:
    try:
        n = int(n_bytes or 0)
    except Exception:
        return ""
    if n <= 0:
        return ""
    gb = 1024**3
    mb = 1024**2
    kb = 1024
    if n >= gb:
        v = n / gb
        s = f"{v:.0f}GB" if v >= 10 else f"{v:.1f}GB"
        return s.replace(".0GB", "GB")
    if n >= mb:
        v = n / mb
        s = f"{v:.0f}MB" if v >= 10 else f"{v:.1f}MB"
        return s.replace(".0MB", "MB")
    if n >= kb:
        v = n / kb
        s = f"{v:.0f}KB" if v >= 10 else f"{v:.1f}KB"
        return s.replace(".0KB", "KB")
    return f"{n}B"


def check_line_html(
    *,
    job_id: str,
    display_name: str = "",
    status_norm: str,
    is_required: bool,
    duration: str = "",
    log_url: str = "",
    raw_log_href: str = "",
    raw_log_size_bytes: int = 0,
    error_snippet_text: str = "",
    required_failure: bool = False,
    warning_present: bool = False,
) -> str:
    icon = status_icon_html(
        status_norm=status_norm,
        is_required=is_required,
        required_failure=required_failure,
        warning_present=warning_present,
    )
    id_html = (
        '<span style="font-family: SFMono-Regular, Consolas, Liberation Mono, Menlo, monospace; font-size: 12px;">'
        + html.escape(job_id or "")
        + "</span>"
    )
    req_html = required_badge_html(is_required=is_required, status_norm=status_norm)

    name_html = ""
    if display_name and display_name != job_id:
        name_html = f'<span style="color: #57606a; font-size: 12px;"> — {html.escape(display_name)}</span>'

    dur_html = f'<span style="color: #57606a; font-size: 12px;"> ({html.escape(duration)})</span>' if duration else ""

    links = ""
    if log_url:
        links += _small_link_html(url=log_url, label="[log]")
    if raw_log_href:
        size_s = _format_bytes_short(raw_log_size_bytes)
        label = f"[raw log {size_s}]" if size_s else "[raw log]"
        links += _small_link_html(url=raw_log_href, label=label)
        # Snippet toggle should come immediately after [raw log] so it's easy to spot.
        links += _error_snippet_toggle_html(dom_id_seed=f"{job_id}|{raw_log_href}", snippet_text=error_snippet_text)
        # Inline tags (same line) after the Snippet toggle: categories + first detected command.
        if (error_snippet_text or "").strip():
            cats = _snippet_categories(error_snippet_text or "")
            for c in cats[:3]:
                links += _tag_pill_html(text=c, monospace=False, kind="category")
            cmd = _snippet_first_command(error_snippet_text or "")
            if cmd:
                links += _tag_pill_html(text=cmd, monospace=True, kind="command")

    return f"{icon} {id_html}{req_html}{name_html}{dur_html}{links}"


def render_gl_job_line_html(*, status_norm: str, name: str, url: str = "", duration: str = "") -> str:
    icon = status_icon_html(status_norm=status_norm, is_required=False)
    name_html = (
        '<span style="font-family: SFMono-Regular, Consolas, Liberation Mono, Menlo, monospace; font-size: 12px;">'
        + html.escape(name or "")
        + "</span>"
    )
    dur_html = f'<span style="color: #57606a; font-size: 12px;"> ({html.escape(duration)})</span>' if duration else ""
    links = _small_link_html(url=url, label="[log]") if url else ""
    return f"{icon} {name_html}{dur_html}{links}"


# ======================================================================================
# Workflow graph helpers (derived from .github/workflows/*.yml)
# ======================================================================================

from common_github_workflow_parser import GitHubWorkflowParser, WorkflowJobSpec


def parse_workflow_jobs(workflow_path: Path) -> Dict[str, WorkflowJobSpec]:
    return GitHubWorkflowParser.parse_workflow_jobs(workflow_path)


def load_workflow_specs(repo_path: Path) -> Dict[str, WorkflowJobSpec]:
    return GitHubWorkflowParser.load_workflow_specs(repo_path)


def build_check_name_matchers(specs: Dict[str, WorkflowJobSpec]) -> List[Tuple[str, re.Pattern]]:
    return GitHubWorkflowParser.build_check_name_matchers(specs)


