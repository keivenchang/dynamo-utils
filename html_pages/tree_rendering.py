# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tree rendering: HTML generators for CI tree views, check lines, and node formatting."""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from common_types import CIStatus
from ci_status_icons import (
    COLOR_GREEN,
    COLOR_RED,
    COLOR_GREY,
    COLOR_YELLOW,
    status_icon_html,
    required_badge_html,
    mandatory_badge_html,
    KNOWN_ERROR_MARKERS,
    PASS_PLUS_STYLE,
    EXPECTED_CHECK_PLACEHOLDER_SYMBOL,
)
from ci_log_errors import render_error_snippet_html as _format_snippet_html
from ci_log_errors import categorize_error_snippet_text as _snippet_categories
from pytest_parsing import (
    GRAFANA_TEST_URL_TEMPLATE,
    pytest_slowest_tests_from_raw_log,
    pytest_results_from_raw_log,
    _parse_iso_utc,
)
from common_github import GitHubAPIClient

if TYPE_CHECKING:
    from tree_passes import TreeNodeVM

logger = logging.getLogger(__name__)


def _hash10(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def _dom_id_from_node_key(node_key: str) -> str:
    """Best-effort stable DOM id for a tree node's children container."""
    k = str(node_key or "")
    if not k:
        return ""
    return f"tree_children_k_{_hash10(k)}"


def _small_link_html(*, url: str, label: str) -> str:
    if not url:
        return ""
    return (
        f' <a href="{html.escape(url, quote=True)}" target="_blank" '
        f'style="color: #0969da; font-size: 11px; margin-left: 5px; text-decoration: none;">{html.escape(label)}</a>'
    )


_ACTIONS_JOB_ID_RE = re.compile(r"/job/([0-9]+)(?:$|[/?#])")


def extract_actions_job_id_from_url(url: str) -> str:
    """Best-effort extraction of the numeric job id from GitHub Actions job URLs."""
    m = _ACTIONS_JOB_ID_RE.search(str(url or ""))
    return str(m.group(1)) if m else ""


def disambiguate_check_run_name(name: str, url: str, *, name_counts: Dict[str, int]) -> str:
    """If multiple runs share the same name, add a stable suffix so the UI doesn't show duplicates."""
    n = str(name or "")
    if int(name_counts.get(n, 0) or 0) <= 1:
        return n
    jid = extract_actions_job_id_from_url(str(url or ""))
    if jid:
        return f"{n} [job {jid}]"
    u = str(url or "")
    if u:
        return f"{n} [{_hash10(u)}]"
    return n


def _triangle_html(
    *,
    expanded: bool,
    children_id: str,
    tooltip: Optional[str],
    parent_children_id: Optional[str],
    url_key: str = "",
) -> str:
    ch = "▼" if expanded else "▶"
    title_attr = f' title="{html.escape(tooltip or "", quote=True)}"' if tooltip else ""
    parent_attr = (
        f' data-parent-children-id="{html.escape(parent_children_id, quote=True)}"'
        if parent_children_id
        else ""
    )
    url_key_attr = f' data-url-key="{html.escape(url_key, quote=True)}"' if url_key else ""
    return (
        f'<span style="display: inline-block; width: 12px; margin-right: 2px; color: #0969da; '
        f'cursor: pointer; user-select: none;"{title_attr} '
        f'data-children-id="{html.escape(children_id, quote=True)}"{parent_attr} '
        f'data-default-expanded="{"1" if expanded else "0"}" '
        f"{url_key_attr} "
        f'onclick="toggleTreeChildren(\'{html.escape(children_id, quote=True)}\', this)">{ch}</span>'
    )


def _triangle_placeholder_html() -> str:
    return '<span style="display: inline-block; width: 12px; margin-right: 2px;"></span>'


def _noncollapsible_icon_html(icon: str) -> str:
    """Render a fixed-width icon placeholder for non-collapsible nodes (keeps alignment)."""
    s = str(icon or "").strip().lower()
    if s == "square":
        return '<span style="display: inline-block; width: 12px; margin-right: 2px; color: #57606a; user-select: none;">■</span>'
    return _triangle_placeholder_html()


def render_tree_divs(root_nodes: List[TreeNodeVM]) -> str:
    """Render a forest as nested <ul>/<li>/<details>/<summary> elements using iamkate.com tree pattern.
    
    Returns HTML string with proper tree structure and CSS classes.
    Based on: https://iamkate.com/code/tree-views/
    """
    global _TREE_RENDER_CALL_SEQ
    try:
        _TREE_RENDER_CALL_SEQ += 1
    except NameError:
        _TREE_RENDER_CALL_SEQ = 1
    render_call_id = _TREE_RENDER_CALL_SEQ
    next_dom_id = 0
    used_ids: Dict[str, int] = {}
    
    # Track displayed nodes to show references for duplicates
    displayed_nodes: Dict[str, str] = {}
    last_reference_text: Optional[str] = None

    def alloc_children_id(node_key: str) -> str:
        nonlocal next_dom_id
        base = _dom_id_from_node_key(node_key)
        if base:
            n = int(used_ids.get(base, 0) or 0) + 1
            used_ids[base] = n
            if n == 1:
                return base
            return f"{base}_{n}"
        next_dom_id += 1
        return f"tree_children_{render_call_id:x}_{next_dom_id:x}"

    def _sha7_from_key(s: str) -> str:
        m = re.findall(r"\b[0-9a-f]{7,40}\b", str(s or ""), re.IGNORECASE)
        if not m:
            return ""
        return str(m[-1])[:7].lower()

    def _repo_token_from_key(s: str) -> str:
        txt = str(s or "")
        m = re.search(r"\b(?:PRStatus|CI|repo):([^:>]+)", txt)
        if not m:
            return ""
        repo = str(m.group(1) or "").strip()
        repo = re.sub(r"[^a-zA-Z0-9._-]+", "-", repo).strip("-").lower()
        return repo[:32]

    def render_node(node: TreeNodeVM, parent_children_id: Optional[str], node_key_path: str, is_last: bool = True) -> str:
        """Render a single node as <li> with optional <details>/<summary> for collapsible nodes.
        
        Based on https://iamkate.com/code/tree-views/ pattern.
        """
        nonlocal last_reference_text
        
        # Check for circular reference (would cause infinite recursion)
        node_key = str(node.node_key or "")
        if node_key and node_key in node_key_path:
            # Circular reference - skip to prevent infinite recursion
            return ""
        
        # Reset last reference text when rendering actual node
        last_reference_text = None

        # Check if node has children or raw content
        has_children = bool(node.children)
        has_raw_content = bool(node.raw_html_content)

        # Determine if this should be collapsible (requires both collapsible=True AND having content)
        is_collapsible = node.collapsible and (has_children or has_raw_content)

        # Skip truly-empty leaf nodes. These can happen when upstream data creates an empty TreeNodeVM
        # (e.g., a trailing separator). Rendering them produces a dangling connector with no label.
        if (not has_children) and (not has_raw_content) and (not (node.label_html or "").strip()):
            return ""

        # Check if this is a repository node (for spacing)
        is_repo_node = node_key.startswith("repo:")

        # Add 'leaf' class for nodes without children, 'repo-node' for repository nodes
        li_classes = []
        if not (has_children or has_raw_content):
            li_classes.append('leaf')
        if is_repo_node:
            li_classes.append('repo-node')

        li_class_attr = f' class="{" ".join(li_classes)}"' if li_classes else ''

        parts = []
        parts.append(f'<li{li_class_attr}>\n')

        if is_collapsible:
            # Collapsible node with children - use <details>/<summary>
            nk = str(node.node_key or "")
            full_key = (str(node_key_path or "") + ">" + nk).strip(">")
            children_id = alloc_children_id(full_key)
            sha7 = _sha7_from_key(full_key) or _sha7_from_key(nk)
            repo = _repo_token_from_key(full_key) or _repo_token_from_key(nk)
            
            # Generate stable URL key using hash of node_key (not full path)
            # This ensures URL params survive page reloads even when tree structure changes
            import hashlib
            node_key_hash = hashlib.sha256(nk.encode()).hexdigest()[:7]
            
            url_key_attr = ""
            if repo and sha7:
                url_key_attr = f' data-url-key="t.{html.escape(repo)}.{html.escape(node_key_hash)}"'
            elif sha7:
                url_key_attr = f' data-url-key="t.{html.escape(node_key_hash)}"'
            else:
                # Fallback: use just the hash
                url_key_attr = f' data-url-key="t.{html.escape(node_key_hash)}"'
            
            parent_attr = f' data-parent-children-id="{html.escape(parent_children_id)}"' if parent_children_id else ''
            expanded = bool(node.default_expanded)
            open_attr = ' open' if expanded else ''
            title_attr = f' title="{html.escape(node.triangle_tooltip)}"' if node.triangle_tooltip else ''
            default_expanded_attr = f' data-default-expanded="{"1" if expanded else "0"}"'
            
            # Add snippet-key attribute for snippet nodes (so category pills can target them)
            snippet_key_attr = ""
            if nk.startswith("snippet:"):
                snippet_key = nk[len("snippet:"):]  # Remove "snippet:" prefix
                snippet_key_attr = f' data-snippet-key="{html.escape(snippet_key)}"'
            
            parts.append(f'<details{open_attr} id="{html.escape(children_id)}"{parent_attr}{url_key_attr}{snippet_key_attr} data-url-state="1">\n')
            parts.append(f'<summary{title_attr}{default_expanded_attr}>\n')
            parts.append(node.label_html or "")
            parts.append('</summary>\n')
            # Wrap all <details> bodies so we can apply a consistent CSS transition for open/close.
            # NOTE: Browsers toggle <details> instantly; the transition is applied to this wrapper.
            parts.append('<div class="details-body">\n')
            # Render raw HTML content if present (e.g., snippet <pre> blocks)
            if node.raw_html_content:
                parts.append(node.raw_html_content)

            # Only render <ul> if there are actual children
            if node.children:
                parts.append('<ul>\n')
                for i, child in enumerate(node.children):
                    is_last_child = (i == len(node.children) - 1)
                    parts.append(render_node(child, children_id, full_key, is_last_child))
                parts.append('</ul>\n')
            parts.append('</div>\n')
            
            parts.append('</details>\n')
        else:
            # Non-collapsible node - render label and children directly (no <details> wrapper)
            parts.append(node.label_html or "")
            # Render raw HTML content if present (e.g., snippet <pre> blocks for non-collapsible nodes)
            if node.raw_html_content:
                parts.append(node.raw_html_content)

            # If node has children but collapsible=False, render them directly without <details>
            if has_children:
                nk = str(node.node_key or "")
                full_key = (str(node_key_path or "") + ">" + nk).strip(">")
                children_id = alloc_children_id(full_key)

                parts.append('<ul>\n')
                for i, child in enumerate(node.children):
                    is_last_child = (i == len(node.children) - 1)
                    parts.append(render_node(child, children_id, full_key, is_last_child))
                parts.append('</ul>\n')

        parts.append('</li>\n')
        return "".join(parts)
    
    out_parts = ['<ul class="tree">\n']
    for i, node in enumerate(root_nodes):
        is_last = (i == len(root_nodes) - 1)
        out_parts.append(render_node(node, None, "", is_last))
    out_parts.append('</ul>')
    
    return "".join(out_parts)


def render_tree_pre_lines(root_nodes: List[TreeNodeVM]) -> List[str]:
    """Render a forest into lines suitable to be joined with \n and placed inside <pre>."""

    out: List[str] = []
    # Prefer stable ids derived from node_key so URL state survives page refreshes.
    # Fall back to a per-render counter when keys collide or are missing.
    global _TREE_RENDER_CALL_SEQ
    try:
        _TREE_RENDER_CALL_SEQ += 1
    except NameError:
        _TREE_RENDER_CALL_SEQ = 1
    render_call_id = _TREE_RENDER_CALL_SEQ
    next_dom_id = 0
    used_ids: Dict[str, int] = {}
    
    # Track displayed nodes to show references for duplicates
    displayed_nodes: Dict[str, str] = {}  # node_key -> label_html (first occurrence)
    last_reference_text: Optional[str] = None  # Track last reference text to avoid consecutive duplicates

    def alloc_children_id(node_key: str) -> str:
        nonlocal next_dom_id
        base = _dom_id_from_node_key(node_key)
        if base:
            n = int(used_ids.get(base, 0) or 0) + 1
            used_ids[base] = n
            if n == 1:
                return base
            # Extremely rare: collisions for identical node_key. Keep deterministic by suffixing.
            return f"{base}_{n}"
        next_dom_id += 1
        return f"tree_children_{render_call_id:x}_{next_dom_id:x}"

    _HEX_SHA_RE = re.compile(r"\b[0-9a-f]{7,40}\b", re.IGNORECASE)

    def _sha7_from_key(s: str) -> str:
        m = _HEX_SHA_RE.findall(str(s or ""))
        if not m:
            return ""
        # Prefer the last SHA-ish token (often the most specific like branch head SHA).
        return str(m[-1])[:7].lower()

    def _repo_token_from_key(s: str) -> str:
        """Extract a repo/dir token for URL readability (not uniqueness)."""
        txt = str(s or "")
        # Match PRStatus:, CI:, or repo: patterns
        m = re.search(r"\b(?:PRStatus|CI|repo):([^:>]+)", txt)
        if not m:
            return ""
        repo = str(m.group(1) or "").strip()
        repo = re.sub(r"[^a-zA-Z0-9._-]+", "-", repo).strip("-").lower()
        return repo[:32]

    def render_node(
        node: TreeNodeVM,
        prefix: str,
        is_last: bool,
        is_root: bool,
        parent_children_id: Optional[str],
        node_key_path: str,
    ) -> None:
        nonlocal last_reference_text
        
        # Check for circular reference (would cause infinite recursion)
        node_key = str(node.node_key or "")
        if node_key and node_key in node_key_path:
            # Circular reference - skip to prevent infinite recursion
            return
            
        # Reset last reference text when rendering actual node
        last_reference_text = None
        
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
                nk = str(node.node_key or "")
                full_key = (str(node_key_path or "") + ">" + nk).strip(">")
                children_id = alloc_children_id(full_key)
                sha7 = _sha7_from_key(full_key) or _sha7_from_key(nk)
                # Compact URL key, SHA-first: t.<sha7>.<h6>
                # Use only the node's own key (not full path) for the hash to ensure
                # URL keys remain stable across different tree orderings (e.g., sort by latest vs branch).
                repo_tok = _repo_token_from_key(full_key)
                if repo_tok and sha7:
                    url_key = f"t.{repo_tok}.{sha7}.{_hash10(nk)[:6]}"
                elif sha7:
                    url_key = f"t.{sha7}.{_hash10(nk)[:6]}"
                elif repo_tok:
                    url_key = f"t.{repo_tok}.{_hash10(nk)[:6]}"
                else:
                    url_key = f"t.{_hash10(nk)[:10]}"
                tri = _triangle_html(
                    expanded=bool(node.default_expanded),
                    children_id=children_id,
                    tooltip=node.triangle_tooltip,
                    parent_children_id=parent_children_id,
                    url_key=url_key,
                )
            else:
                tri = _triangle_placeholder_html()
        else:
            tri = _noncollapsible_icon_html(node.noncollapsible_icon)

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
                    (str(node_key_path or "") + ">" + str(node.node_key or "")).strip(">"),
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
                (str(node_key_path or "") + ">" + str(node.node_key or "")).strip(">"),
            )

    for i, n in enumerate(root_nodes):
        render_node(n, prefix="", is_last=(i == len(root_nodes) - 1), is_root=True, parent_children_id=None, node_key_path="")

    return out


# ======================================================================================
# Shared GitHub/GitLab check/job line rendering (HTML)
# ======================================================================================

def _format_duration_short(seconds: float) -> str:
    """Format seconds as a short duration like '3s', '2m 10s', '1h 4m'."""
    s = int(round(float(seconds)))
    if s <= 0:
        return "0s"
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h}h {m}m" if m else f"{h}h"
    if m > 0:
        return f"{m}m {sec}s" if sec else f"{m}m"
    return f"{sec}s"


def _status_norm_from_actions_step(status: str, conclusion: str) -> str:
    s = (status or "").strip().lower()
    c = (conclusion or "").strip().lower()
    if c == CIStatus.SKIPPED.value:
        return CIStatus.SKIPPED.value
    if c in (CIStatus.SUCCESS.value, CIStatus.NEUTRAL.value):
        return CIStatus.SUCCESS.value
    if c in ("failure", "timed_out", "action_required"):
        return CIStatus.FAILURE.value
    if c in (CIStatus.CANCELLED.value, "canceled"):
        return CIStatus.CANCELLED.value
    # Some API responses omit `conclusion` even for completed successful steps.
    # Treat "completed + empty conclusion" as success so we don't auto-expand clean trees.
    if s == "completed" and c in ("", "null", "none"):
        return CIStatus.SUCCESS.value
    if s in (CIStatus.IN_PROGRESS.value, "in progress"):
        return CIStatus.IN_PROGRESS.value
    if s in ("queued", CIStatus.PENDING.value):
        return CIStatus.PENDING.value
    return CIStatus.UNKNOWN.value


def build_and_test_dynamo_phases_from_actions_job(job: Dict[str, object]) -> List[Tuple[str, str, str, Optional[Dict[str, object]]]]:
    """Extract phase rows from the Actions job `steps` array.

    Returns (phase_name, duration_str, status_norm, step_dict).
    The step_dict contains started_at/completed_at for timestamp-based log filtering.
    """
    steps = job.get("steps") if isinstance(job, dict) else None
    if not isinstance(steps, list) or not steps:
        logger.warning("[build_and_test_dynamo_phases_from_actions_job] No steps found in job")
        return []

    logger.debug(f"[build_and_test_dynamo_phases_from_actions_job] Processing {len(steps)} steps from job")
    def _dur(st: Dict[str, object]) -> str:
        a = _parse_iso_utc(str(st.get("started_at", "") or ""))
        b = _parse_iso_utc(str(st.get("completed_at", "") or ""))
        if not a or not b:
            return ""
        return _format_duration_short((b - a).total_seconds())

    out: List[Tuple[str, str, str, Optional[Dict[str, object]]]] = []

    # Return ALL API step names directly - no filtering, no transformation
    for st in steps:
        if not isinstance(st, dict):
            continue
        nm = str(st.get("name", "") or "")
        status_norm = _status_norm_from_actions_step(
            status=str(st.get("status", "") or ""),
            conclusion=str(st.get("conclusion", "") or ""),
        )
        dur = _dur(st)
        out.append((nm, dur, status_norm, st))
        logger.debug(f"[build_and_test_dynamo_phases_from_actions_job] Added step '{nm}' with step_dict: started_at={st.get('started_at')}, completed_at={st.get('completed_at')}")

    # De-dup while keeping order (some jobs echo repeated step names via composites).
    seen = set()
    uniq: List[Tuple[str, str, str, Optional[Dict[str, object]]]] = []
    for ph in out:
        k = ph[0]
        if k in seen:
            continue
        seen.add(k)
        uniq.append(ph)
    logger.debug(f"[build_and_test_dynamo_phases_from_actions_job] Returning {len(uniq)} unique phases")
    return uniq


def actions_job_steps_over_threshold_from_actions_job(
    job: Dict[str, object], *, min_seconds: float = 30.0
) -> List[Tuple[str, str, str, Optional[Dict[str, object]]]]:
    """Return (step_name, duration_str, status_norm, step_dict) for steps we want to display.

    Policy:
    - show steps with duration >= min_seconds
    - always show failing steps (even if < min_seconds or duration is missing)
    
    Returns 4-tuples with step_dict for timestamp-based filtering.
    """
    steps = job.get("steps") if isinstance(job, dict) else None
    if not isinstance(steps, list) or not steps:
        return []

    out: List[Tuple[str, str, str, Optional[Dict[str, object]]]] = []
    for st in steps:
        if not isinstance(st, dict):
            continue
        nm = str(st.get("name", "") or "").strip()
        if not nm:
            continue
        status_norm = _status_norm_from_actions_step(
            status=str(st.get("status", "") or ""),
            conclusion=str(st.get("conclusion", "") or ""),
        )
        # Duration is best-effort; some steps may not include timestamps.
        dt_s = None
        a = _parse_iso_utc(str(st.get("started_at", "") or ""))
        b = _parse_iso_utc(str(st.get("completed_at", "") or ""))
        if a and b:
            dt_s = float((b - a).total_seconds())

        # Selection rule:
        # - always include failures
        # - if min_seconds <= 0: include all non-failing steps (even if duration is missing)
        # - otherwise include only steps >= threshold
        if status_norm != "failure":
            if float(min_seconds) <= 0.0:
                pass
            elif dt_s is None or float(dt_s) < float(min_seconds):
                continue

        dur_s = _format_duration_short(float(dt_s)) if dt_s is not None else ""
        out.append((nm, dur_s, status_norm, st))  # Include st (step_dict) for timestamps

    # De-dup while keeping order (some composite actions can repeat step names).
    seen = set()
    uniq: List[Tuple[str, str, str, Optional[Dict[str, object]]]] = []
    for (nm, dur, st, step_dict) in out:
        if nm in seen:
            continue
        seen.add(nm)
        uniq.append((nm, dur, st, step_dict))
    return uniq


def actions_job_step_tuples(
    *,
    github_api: Optional[GitHubAPIClient],
    job_url: str,
    min_seconds: float = 10.0,
    ttl_s: int = 30 * 24 * 3600,
) -> List[Tuple[str, str, str, Optional[Dict[str, object]]]]:
    """Fetch job details (cached) and return long-running steps (duration >= min_seconds).
    
    Returns 4-tuples: (step_name, duration_str, status_norm, step_dict) for timestamp-based filtering."""
    t_start = time.monotonic()
    
    if not github_api:
        logger.debug(f"[actions_job_step_tuples] No github_api provided")
        return []
    
    t0 = time.monotonic()
    jid = extract_actions_job_id_from_url(str(job_url or ""))
    t_extract_jid = time.monotonic() - t0
    
    if not jid:
        logger.debug(f"[actions_job_step_tuples] Could not extract job ID from URL: {job_url}")
        return []
    
    t0 = time.monotonic()
    job = github_api.get_actions_job_details_cached(
        owner="ai-dynamo", repo="dynamo", job_id=jid, ttl_s=int(ttl_s)
    ) or {}
    t_get_cached = time.monotonic() - t0
    
    if not isinstance(job, dict):
        logger.debug(f"[actions_job_step_tuples] Job data not a dict for job_id={jid}")
        return []
    steps_in_dict = len(job.get('steps', [])) if isinstance(job.get('steps'), list) else 0
    
    t0 = time.monotonic()
    result = actions_job_steps_over_threshold_from_actions_job(job, min_seconds=float(min_seconds))
    t_filter = time.monotonic() - t0
    
    t_total = time.monotonic() - t_start
    if t_total > 0.05:  # Log if >50ms
        logger.debug(
            f"[actions_job_step_tuples] job_id={jid}, total={t_total:.3f}s "
            f"(extract={t_extract_jid:.3f}s, get_cached={t_get_cached:.3f}s, filter={t_filter:.3f}s), "
            f"steps_in_dict={steps_in_dict}, min_seconds={min_seconds}, filtered_result={len(result)}"
        )
    return result


_PYTEST_DETAIL_JOB_PREFIXES = ("sglang", "vllm", "trtllm", "operator")


def is_build_test_job(job_name: str) -> bool:
    """
    Check if a job name matches the build-test pattern that should have pytest details extracted.

    Matches three patterns:
    1. Jobs with both "build" AND "test" in the name:
       - "Build and Test - dynamo"
       - "sglang-build-test (cuda12.9, amd64)"
       - "vllm-build-test (cuda12.9, arm64)"

    2. Jobs with "pytest" in the name:
       - "pytest (amd64)"
       - "[x86_64] pytest"

    3. Framework jobs with architecture indicator (older pattern):
       - "sglang (amd64)"
       - "[x86_64] vllm (amd64)"
       - "[aarch64] trtllm (arm64)"

    Args:
        job_name: The CI job name to check

    Returns:
        True if the job matches build-test patterns and should have pytest details
    """
    nm = str(job_name or "").strip()
    if not nm:
        return False
    nm_lc = nm.lower()

    # Pattern 1: "build" AND "test" in name (covers both old and new naming)
    if ("build" in nm_lc) and ("test" in nm_lc):
        return True

    # Pattern 2: "pytest" in name
    if "pytest" in nm_lc:
        return True

    # Pattern 3: Framework prefix + architecture indicator (older naming convention)
    # e.g. "sglang (amd64)", "[x86_64] vllm (amd64)"
    for prefix in _PYTEST_DETAIL_JOB_PREFIXES:
        if prefix in nm_lc:
            # Heuristic: these matrix-style jobs almost always contain an arch tuple or keyword
            if ("(" in nm_lc) or ("amd64" in nm_lc) or ("arm64" in nm_lc) or ("aarch64" in nm_lc):
                return True

    return False


def is_python_test_step(step_name: str) -> bool:
    """
    Check if a step name indicates it's a Python test step that should have pytest children.

    Examples that match:
    - "Run tests"
    - "Run e2e tests"
    - "Run unit tests"
    - "test: pytest"
    - "pytest"
    - "test", "tests" (starts with "test")

    Args:
        step_name: The step or phase name to check

    Returns:
        True if this step should have pytest test children parsed from raw logs
    """
    step_lower = str(step_name or "").lower()
    if not step_lower:
        return False
    
    # Be specific: only pytest steps, not "Rust checks (integration tests)" etc.
    return (
        "pytest" in step_lower
        or ("run" in step_lower and "test" in step_lower and "rust" not in step_lower)
        or step_lower.startswith("test")
    )


def job_name_wants_pytest_details(job_name: str) -> bool:
    """
    Shared policy for whether a job should show pytest per-test children under "Run tests".

    This is used by:
    - the step/test injection pass (so UI is consistent across pages)
    - "success raw log" caching policy (so per-test parsing is possible)

    IMPORTANT: keep this conservative; raw-log fetching for successful jobs can be expensive.

    Examples of job names that match (via is_build_test_job()):
    - "Build and Test - dynamo" (special case, always matches)
    - "sglang-build-test (cuda12.9, amd64)" (has "build" AND "test")
    - "[x86_64] vllm-build-test (cuda12.9, amd64)" (has "build" AND "test")
    - "sglang (amd64)" (framework prefix + architecture)
    - "[x86_64] vllm (amd64)" (framework prefix + architecture)
    - "[aarch64] trtllm (arm64)" (framework prefix + architecture)
    """
    nm = str(job_name or "").strip()
    if not nm:
        return False

    # Special case: "Build and Test - dynamo" (canonical job name)
    if nm == "Build and Test - dynamo":
        return True

    # Use common pattern matcher for build-test jobs
    return is_build_test_job(job_name)


def ci_subsection_tuples_for_job(
    *,
    github_api: Optional[GitHubAPIClient],
    job_name: str,
    job_url: str,
    raw_log_path: Optional[Path],
    duration_seconds: float,
    is_required: bool,
    long_job_threshold_s: float = 10.0 * 60.0,
    step_min_s: float = 10.0,
) -> List[Tuple[str, str, str]]:
    """Shared rule: return child tuples for CI subsections.

    Terminology (official):
    - "subsections" is the umbrella term for child rows under a job/check.
    - "phases" are a *special-case* kind of subsection for `Build and Test - dynamo`
      (we keep the name "phases" in code where it's specific to that job).

    Policy:
    - Build and Test - dynamo: return the dedicated phases (status+duration) from Actions job `steps[]`.
    - Other build/test jobs: return all job steps (vllm-build-test, sglang-build-test, trtllm-build-test, etc.).
    - Other long-running Actions jobs: return job steps >= step_min_s.
    """
    t_start = time.monotonic()
    
    nm = str(job_name or "").strip()
    if not nm:
        return []

    # Shared policy: which jobs get full steps + pytest per-test breakdown.
    is_build_test = job_name_wants_pytest_details(nm)

    # Special case: "Build and Test - dynamo" has custom phase extraction
    if nm == "Build and Test - dynamo":
        # Fetch job details once for timestamp-based filtering
        job_dict = None
        jid = extract_actions_job_id_from_url(str(job_url or ""))
        if github_api and jid:
            job_dict = github_api.get_actions_job_details_cached(
                owner="ai-dynamo", repo="dynamo", job_id=jid, ttl_s=30 * 24 * 3600
            ) or {}
        
        t0 = time.monotonic()
        phases = build_and_test_dynamo_phase_tuples(
            github_api=github_api,
            job_url=str(job_url or ""),
            raw_log_path=raw_log_path,
            is_required=bool(is_required),
        )
        t_phases = time.monotonic() - t0
        
        # Also include *non-phase* steps so we can surface useful failures like
        # "Copy test report..." without duplicating the phase rows.
        #
        # Policy for REQUIRED jobs: show all failing steps + steps >= threshold; ignore the rest.
        t0 = time.monotonic()
        steps = actions_job_step_tuples(
            github_api=github_api,
            job_url=str(job_url or ""),
            # Build/test UX: show all steps (not just slow ones).
            min_seconds=0.0,
        )
        t_steps = time.monotonic() - t0

        def _covered_by_phase(step_name: str) -> bool:
            s = str(step_name or "").strip().lower()
            if not s:
                return True
            # Phase-like steps (already represented by `phases`).
            if "build image" in s:
                return True
            if ("rust" in s) and ("check" in s):
                return True
            if "pytest" in s:
                return True
            return False

        extra_steps = [(n, d, st, step_dict) for (n, d, st, step_dict) in (steps or []) if not _covered_by_phase(n)]

        out = [(p[0], p[1], p[2], p[3] if len(p) > 3 else None) for p in (phases or [])]  # Keep step dict
        logger.debug(f"[ci_subsection_tuples_for_job] Extracted {len(out)} phases for job '{nm}'")
        for i, (step_name, _, _, step_dict) in enumerate(out[:5]):  # Log first 5
            has_timestamps = False
            if step_dict and isinstance(step_dict, dict):
                has_timestamps = bool(step_dict.get('started_at') and step_dict.get('completed_at'))
            logger.debug(f"[ci_subsection_tuples_for_job]   Phase {i}: '{step_name}', has_step_dict={bool(step_dict)}, has_timestamps={has_timestamps}")
        out.extend(extra_steps)  # extra_steps now include step_dict (4-tuples)

        # Apply pytest test extraction to "Run tests" steps and "test: pytest" phases
        result = []
        t_pytest_total = 0.0
        for step_name, step_dur, step_status, step_dict in out:
            result.append((step_name, step_dur, step_status, step_dict))  # Include step_dict for timestamps

            # If this is a "Run tests" step or a "test: pytest" phase, parse pytest slowest tests from raw log
            if is_python_test_step(step_name) and raw_log_path:
                t0 = time.monotonic()
                logger.debug(f"[ci_subsection_tuples_for_job] About to call pytest_slowest_tests_from_raw_log for '{step_name}'")
                logger.debug(f"[ci_subsection_tuples_for_job]   step_dict type: {type(step_dict).__name__ if step_dict else 'None'}")
                if step_dict and isinstance(step_dict, dict):
                    logger.debug(f"[ci_subsection_tuples_for_job]   step_dict has started_at: {bool(step_dict.get('started_at'))}, completed_at: {bool(step_dict.get('completed_at'))}")
                    logger.debug(f"[ci_subsection_tuples_for_job]   Timestamps: {step_dict.get('started_at')} -> {step_dict.get('completed_at')}")
                else:
                    logger.warning(f"[ci_subsection_tuples_for_job] WARNING: step_dict is None or not a dict for pytest step '{step_name}'!")
                    
                pytest_tests = pytest_slowest_tests_from_raw_log(
                    raw_log_path=raw_log_path,
                    # Tests: list *all* per-test timings in order (do not filter).
                    min_seconds=0.0,
                    include_all=True,
                    step_name=step_name,
                    step_dict=step_dict,  # Pass step dict for timestamp filtering
                )
                t_pytest_total += time.monotonic() - t0
                # Add pytest tests as indented entries (4-tuples with None step_dict for tests)
                for test_name, test_dur, test_status in pytest_tests:
                    result.append((f"  └─ {test_name}", test_dur, test_status, None))

        t_total = time.monotonic() - t_start
        if t_total > 0.1:  # Log only if >100ms
            logger.debug(
                f"[ci_subsection_tuples_for_job] '{nm}' took {t_total:.3f}s: "
                f"phases={t_phases:.3f}s, steps={t_steps:.3f}s, pytest={t_pytest_total:.3f}s"
            )
        return result

    # REQUIRED jobs: always show failing steps + steps >= threshold (even if job isn't "long-running").
    if bool(is_required):
        return actions_job_step_tuples(github_api=github_api, job_url=str(job_url or ""), min_seconds=float(step_min_s))

    # Build/test jobs (framework builds): show steps + pytest tests.
    # This covers: vllm-build-test, sglang-build-test, trtllm-build-test (with cuda/arch variants), etc.
    if is_build_test:
        # Build/test UX: show all steps (not just slow ones).
        t0 = time.monotonic()
        steps = actions_job_step_tuples(github_api=github_api, job_url=str(job_url or ""), min_seconds=0.0)
        t_steps = time.monotonic() - t0
        
        logger.debug(
            f"[ci_subsection_tuples_for_job] Build-test job '{nm}': "
            f"fetched {len(steps) if steps else 0} steps from API in {t_steps:.3f}s"
        )

        # For each step, check if it's "Run tests" - if so, add pytest tests as additional entries
        result = []
        t_pytest_total = 0.0
        for step_name, step_dur, step_status, step_dict in (steps or []):
            result.append((step_name, step_dur, step_status, step_dict))

            # If this is a "Run tests" step, parse pytest tests from raw log
            if is_python_test_step(step_name) and raw_log_path:
                t0 = time.monotonic()
                
                # Try pytest_results_from_raw_log first (PASSED/FAILED/SKIPPED lines - more comprehensive)
                pytest_tests = pytest_results_from_raw_log(
                    raw_log_path=raw_log_path,
                    min_seconds=0.0,  # Include all tests
                    step_name=step_name,  # Pass step name to verify parallel/serial matches
                    step_dict=step_dict,  # Pass step dict for timestamp filtering
                )
                
                # If no results from PASSED/FAILED lines, fall back to --durations output
                if not pytest_tests:
                    pytest_tests = pytest_slowest_tests_from_raw_log(
                        raw_log_path=raw_log_path,
                        # Tests: list *all* per-test timings in order (do not filter).
                        min_seconds=0.0,
                        include_all=True,
                        step_name=step_name,
                        step_dict=step_dict,  # Pass step dict for timestamp filtering
                    )
                
                t_pytest_total += time.monotonic() - t0
                # Add pytest tests as indented/prefixed entries (4-tuples with None step_dict for tests)
                for test_name, test_dur, test_status in pytest_tests:
                    # Prefix with indentation to show hierarchy
                    result.append((f"  └─ {test_name}", test_dur, test_status, None))

        t_total = time.monotonic() - t_start
        if t_total > 0.1:  # Log only if >100ms
            logger.debug(
                f"[ci_subsection_tuples_for_job] '{nm}' took {t_total:.3f}s: "
                f"steps={t_steps:.3f}s, pytest={t_pytest_total:.3f}s"
            )
        return result

    # Non-required jobs: show steps only for long-running jobs (avoid noise).
    if float(duration_seconds or 0.0) < float(long_job_threshold_s):
        return []

    return actions_job_step_tuples(github_api=github_api, job_url=str(job_url or ""), min_seconds=float(step_min_s))


def _tag_pill_html(*, text: str, monospace: bool = False, kind: str = "category", snippet_key: str = "") -> str:
    """Render a small tag/pill for inline display next to links.

    kind:
      - "category": light red pill (high-level error category)
      - "command": gray pill (detected command; not the root cause)
    
    snippet_key: Optional snippet key to enable click-to-expand functionality
    """
    t = (text or "").strip()
    if not t:
        return ""
    font = "SFMono-Regular, Consolas, Liberation Mono, Menlo, monospace" if monospace else "inherit"
    if kind == "command":
        border = "#d0d7de"
        bg = "#f6f8fa"
        fg = "#57606a"
    elif t == "exceed-action-timeout":
        # Special styling for timeout marker: redder than other error categories
        border = "#ff9999"
        bg = "#ffcccc"
        fg = "#cc0000"
    else:
        # Error categories: lightly red-tinted pill (avoid yellow/orange).
        border = "#ffb8b8"
        bg = "#ffebe9"
        fg = "#8b1a1a"
    
    # Add onclick handler if snippet_key is provided
    onclick_attr = ""
    cursor_style = ""
    if snippet_key:
        # Use preventDefault to stop the summary toggle, then manually handle expansion
        onclick_attr = f' onclick="event.stopPropagation(); event.preventDefault(); try {{ expandSnippetByKey(\'{html.escape(snippet_key, quote=True)}\'); }} catch (e) {{}}"'
        cursor_style = " cursor: pointer;"
    
    return (
        f' <span style="display: inline-block; vertical-align: baseline; '
        f'border: 1px solid {border}; background: {bg}; color: {fg}; '
        f'border-radius: 999px; padding: 1px 6px; font-size: 10px; '
        f'font-weight: 600; line-height: 1; font-family: {font};{cursor_style}"'
        f'{onclick_attr}>'
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


def _create_snippet_tree_node(*, dom_id_seed: str, snippet_text: str) -> Optional[TreeNodeVM]:
    """Create a TreeNodeVM for an error snippet (as a collapsible child node).
    
    Returns a snippet node with a special node_key that starts with "snippet:" so it can be
    identified and targeted by category pill click handlers.
    """
    from tree_passes import TreeNodeVM
    if not snippet_text or not snippet_text.strip():
        return None
    
    # Compact URL key, SHA-first, but prefer the *full numeric Actions job id* when available:
    #   s.<sha7>.j<jobid>
    # Fallback:
    #   s.<sha7>.<h6>  (or s.<h6> if no sha)
    seed_s = str(dom_id_seed or "")
    m = re.findall(r"\b[0-9a-f]{7,40}\b", seed_s, flags=re.IGNORECASE)
    sha7 = (str(m[-1])[:7].lower() if m else "")
    jobid = ""
    m2 = re.search(r"/job/([0-9]{5,})", seed_s)
    jobid = str(m2.group(1)) if m2 else ""
    suffix = _hash10(seed_s)[:6]
    if jobid:
        url_key = f"s.{sha7}.j{jobid}" if sha7 else f"s.j{jobid}"
    else:
        url_key = f"s.{sha7}.{suffix}" if sha7 else f"s.{suffix}"
    
    # Use snippet: prefix in node_key to mark this as a snippet node
    snippet_node_key = f"snippet:{url_key}"
    
    # Format snippet HTML
    shown = _format_snippet_html(snippet_text or "")
    if not shown:
        shown = '<span style="color: #57606a;">(no snippet found)</span>'
    
    # Create the snippet content as a <pre> block (rendered as raw HTML after the summary)
    snippet_content = (
        f'<pre style="display: block; width: 100%; max-width: 100%; box-sizing: border-box; margin: 4px 0 8px 0; '
        f'border: 1px solid #d0d7de; background: #fffbea; '
        f'border-radius: 6px; padding: 6px 8px; overflow-x: auto; color: #24292f; '
        f'font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace; '
        f'font-size: 12px; line-height: 1.45;">'
        f"{shown}"
        f"</pre>"
    )
    
    return TreeNodeVM(
        node_key=snippet_node_key,  # Special prefix to identify snippet nodes
        label_html='<span style="color: #0969da; font-size: 11px;">Snippet</span>',
        raw_html_content=snippet_content,
        children=[],
        collapsible=True,  # Make it collapsible so it gets a triangle
        default_expanded=False,
    )


def _format_bytes_short(n_bytes: int) -> str:
    n = int(n_bytes or 0)
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
    may_be_required: bool = False,
    duration: str = "",
    log_url: str = "",
    raw_log_href: str = "",
    raw_log_size_bytes: int = 0,
    error_snippet_text: str = "",
    error_snippet_categories: Optional[List[str]] = None,
    required_failure: bool = False,
    warning_present: bool = False,
    icon_px: int = 12,
    short_job_name: str = "",  # Short YAML job name (e.g., "build-test")
    yaml_dependencies: Optional[List[str]] = None,  # List of dependencies from YAML
    is_pytest_node: bool = False,  # True if this is a CIPytestNode (for Grafana links)
    run_attempt: int = 0,  # Run attempt number (>1 means rerun)
) -> str:
    # Expected placeholder checks:
    # - Use a normal gray dot (same as queued/pending) instead of the special ◇ symbol
    # - Suppress the redundant trailing marker in the label
    is_expected_placeholder = bool(str(display_name or "").strip() == EXPECTED_CHECK_PLACEHOLDER_SYMBOL)
    if is_expected_placeholder:
        # Use normal gray dot for expected placeholders (status="queued")
        icon = status_icon_html(
            status_norm="pending",  # Gray dot
            is_required=is_required,
            required_failure=required_failure,
            warning_present=warning_present,
            icon_px=int(icon_px or 12),
        )
    else:
        icon = status_icon_html(
            status_norm=status_norm,
            is_required=is_required,
            required_failure=required_failure,
            warning_present=warning_present,
            icon_px=int(icon_px or 12),
        )

    def _format_arch_text(raw_text: str) -> str:
        """Format the job text with arch styling.

        - If the job contains an explicit arch token, prefix with arch alias:
          - `(arm64)` / `(aarch64)` -> `[aarch64] job-name (...)`
          - `(amd64)`              -> `[x86_64] job-name (...)`
        - Otherwise, keep normal styling (no special casing).
        
        Note: Color styling is applied to the entire line, not within this function.
        """
        raw = str(raw_text or "")
        # Only rewrite when we see an explicit arch token (avoid surprising renames).
        # Match architecture even when part of matrix variables like "(cuda12.9, amd64)"
        m = re.search(r"\((?:[^,)]+,\s*)?(arm64|aarch64|amd64)\)", raw, flags=re.IGNORECASE)
        if not m:
            return html.escape(raw)

        arch = str(m.group(1) or "").strip().lower()
        # Prefix with arch alias for grouping/sorting
        if arch in {"arm64", "aarch64"}:
            # Prefix with "[aarch64] " at the beginning
            raw2 = f"[aarch64] {raw}"
            return html.escape(raw2)
        if arch == "amd64":
            # Prefix with "[x86_64] " at the beginning
            raw2 = f"[x86_64] {raw}"
            return html.escape(raw2)
        return html.escape(raw)
    
    # Detect architecture for line-wide color styling
    def _get_arch_color(text: str) -> str:
        """Return the color for the entire line based on architecture."""
        # Match architecture even when part of matrix variables like "(cuda12.9, amd64)"
        m = re.search(r"\((?:[^,)]+,\s*)?(arm64|aarch64|amd64)\)", str(text or ""), flags=re.IGNORECASE)
        if not m:
            return ""  # No special color
        arch = str(m.group(1) or "").strip().lower()
        if arch in {"arm64", "aarch64"}:
            return "#b8860b"  # Dark yellow/gold for arm64
        if arch == "amd64":
            return "#0969da"  # Blue for amd64
        return ""

    line_color = _get_arch_color(short_job_name or job_id or "")
    color_style = f' color: {line_color};' if line_color else ''
    
    # Use short_job_name if available, otherwise fall back to job_id
    display_text = short_job_name if short_job_name else job_id
    
    id_html = (
        f'<span style="font-family: SFMono-Regular, Consolas, Liberation Mono, Menlo, monospace; font-size: 12px;{color_style}">'
        + _format_arch_text(display_text or "")
        + "</span>"
    )
    req_html = required_badge_html(is_required=is_required, status_norm=status_norm)
    
    # Add [may be REQUIRED] badge if this should be required but isn't marked as such
    if may_be_required and not is_required:
        req_html += ' <span style="color: #57606a; font-weight: 400;">[may be REQUIRED]</span>'
    
    # Add rerun badge if run_attempt > 1
    if run_attempt and int(run_attempt) > 1:
        req_html += f' <span style="background: #ddf4ff; color: #0969da; padding: 1px 5px; border-radius: 6px; font-size: 10px; font-weight: 600; display: inline-block; margin-left: 4px;" title="This check was rerun (attempt #{run_attempt})">🔄 attempt #{run_attempt}</span>'

    # Show display_name with double quotes if we have both and they're different
    name_html = ""
    if (not is_expected_placeholder) and short_job_name and display_name and short_job_name != display_name:
        # Show as: short_job_name "display_name" (both in same font)
        name_html = f'<span style="color: #57606a; font-family: SFMono-Regular, Consolas, Liberation Mono, Menlo, monospace; font-size: 12px;"> "{_format_arch_text(display_name)}"</span>'
    
    # Add dependencies tooltip if available
    deps_title = ""
    if yaml_dependencies:
        deps_list = ", ".join(yaml_dependencies)
        deps_title = f' title="needs: {deps_list}"'
    
    # Wrap the job name in a span with dependencies tooltip
    if deps_title:
        id_html = f'<span{deps_title}>{id_html}</span>'

    dur_html = f'<span style="color: #57606a; font-size: 12px;"> ({html.escape(duration)})</span>' if duration else ""

    # Extract job number from log_url for the log link label
    job_num = ""
    if log_url:
        m = re.search(r"/job/([0-9]{5,})", str(log_url))
        job_num = str(m.group(1)) if m else ""

    links = ""
    if log_url:
        log_label = f"[job {job_num}]" if job_num else "[log]"
        links += _small_link_html(url=log_url, label=log_label)
    # Raw-log link (only when present).
    if raw_log_href:
        size_s = _format_bytes_short(raw_log_size_bytes)
        label = f"[cached raw log {size_s}]" if size_s else "[cached raw log]"
        links += _small_link_html(url=raw_log_href, label=label)

    # Show category pills if snippet exists (snippets are now rendered as child nodes)
    snippet_text = str(error_snippet_text or "")
    has_snippet_text = bool(snippet_text.strip())
    if has_snippet_text:
        # Generate snippet key (same logic as _create_snippet_tree_node)
        dom_id_seed = f"{job_id}|{display_name}|{raw_log_href}|{log_url}"
        seed_s = str(dom_id_seed or "")
        m = re.findall(r"\b[0-9a-f]{7,40}\b", seed_s, flags=re.IGNORECASE)
        sha7 = (str(m[-1])[:7].lower() if m else "")
        jobid = ""
        m2 = re.search(r"/job/([0-9]{5,})", seed_s)
        jobid = str(m2.group(1)) if m2 else ""
        suffix = _hash10(seed_s)[:6]
        if jobid:
            snippet_key = f"s.{sha7}.j{jobid}" if sha7 else f"s.j{jobid}"
        else:
            snippet_key = f"s.{sha7}.{suffix}" if sha7 else f"s.{suffix}"

        # Show category pills and command pill with snippet key for click handling
        cats = error_snippet_categories if error_snippet_categories else _snippet_categories(snippet_text)
        for c in (cats or [])[:3]:
            links += _tag_pill_html(text=c, monospace=False, kind="category", snippet_key=snippet_key)
        cmd = _snippet_first_command(snippet_text)
        if cmd:
            links += _tag_pill_html(text=cmd, monospace=True, kind="command", snippet_key=snippet_key)
    
    # Also show category pills even when no snippet (e.g., for pytest-timeout markers)
    elif error_snippet_categories:
        for c in error_snippet_categories[:3]:
            links += _tag_pill_html(text=c, monospace=False, kind="category", snippet_key="")

    # Detect pytest tests and add Grafana button
    # Only add the button if this is explicitly a CIPytestNode
    if is_pytest_node and '::' in str(job_id or ""):
        # Extract the test name after the last "::"
        test_name = str(job_id or "").split('::')[-1].strip()
        if test_name:
            # URL-encode the test name for the Grafana URL
            test_name_encoded = urllib.parse.quote(test_name)
            grafana_url = GRAFANA_TEST_URL_TEMPLATE.format(test_name=test_name_encoded)
            # Add a Grafana button similar to the PR button in show_commit_history.j2
            grafana_button = (
                f' <a href="{html.escape(grafana_url)}" '
                f'target="_blank" '
                f'style="display: inline-block; padding: 1px 6px; background: linear-gradient(180deg, #FF7A28 0%, #F05A28 100%); '
                f'color: white; text-decoration: none; border-radius: 8px; font-weight: 600; font-size: 9px; '
                f'border: 1px solid #D94A1F; box-shadow: 0 1px 2px rgba(240,90,40,0.3); line-height: 1.2;" '
                f'onmouseover="this.style.background=\'linear-gradient(180deg, #FF8A38 0%, #FF7A28 100%)\'; '
                f'this.style.boxShadow=\'0 2px 4px rgba(240,90,40,0.4)\'; this.style.transform=\'translateY(-1px)\';" '
                f'onmouseout="this.style.background=\'linear-gradient(180deg, #FF7A28 0%, #F05A28 100%)\'; '
                f'this.style.boxShadow=\'0 1px 2px rgba(240,90,40,0.3)\'; this.style.transform=\'translateY(0)\';" '
                f'onmousedown="this.style.boxShadow=\'0 1px 2px rgba(0,0,0,0.2) inset\'; this.style.transform=\'translateY(1px)\';" '
                f'onmouseup="this.style.boxShadow=\'0 2px 4px rgba(240,90,40,0.4)\'; this.style.transform=\'translateY(-1px)\';" '
                f'title="View test {html.escape(test_name)} in Grafana Test Details dashboard">'
                f'grafana</a>'
            )
            links += grafana_button

    # Format: [REQUIRED] short-name "long-name" (duration) [log] ...
    return f"{icon} {req_html}{id_html}{name_html}{dur_html}{links}"


def build_and_test_dynamo_phase_tuples(
    *,
    github_api: Optional[GitHubAPIClient],
    job_url: str,
    raw_log_path: Optional[Path] = None,
    is_required: bool = False,
) -> List[Tuple[str, str, str, Optional[Dict[str, object]]]]:
    """Return phase tuples for the Build-and-Test phase breakdown (best-effort).

    Shared helper used by both dashboards to keep logic identical.
    
    Returns:
        List of 4-tuples: (phase_name, duration_str, status_norm, step_dict)
        The step_dict contains started_at/completed_at for timestamp-based log filtering.
    """
    phases3: List[Tuple[str, str, str, Optional[Dict[str, object]]]] = []
    jid = extract_actions_job_id_from_url(str(job_url or ""))
    logger.debug(f"[build_and_test_dynamo_phase_tuples] Called with job_url={job_url}, extracted job_id={jid}")
    if github_api and jid:
        # Use 30-day TTL to match prefetch_actions_job_details_pass (prevents cache misses).
        # Job details are immutable once completed (verified by get_actions_job_details_cached),
        # so there's no correctness benefit to a shorter TTL. The old 600s (10min) TTL caused
        # ~94 cache misses per dashboard render when prefetched data (30d TTL) was >10min old.
        job = github_api.get_actions_job_details_cached(owner="ai-dynamo", repo="dynamo", job_id=jid, ttl_s=30 * 24 * 3600) or {}
        if isinstance(job, dict):
            phases3 = build_and_test_dynamo_phases_from_actions_job(job) or []
            logger.debug(f"[build_and_test_dynamo_phase_tuples] Got {len(phases3)} phases from build_and_test_dynamo_phases_from_actions_job")
        else:
            logger.warning(f"[build_and_test_dynamo_phase_tuples] Job data is not a dict: {type(job)}")
    else:
        logger.warning(f"[build_and_test_dynamo_phase_tuples] No github_api or no job_id (jid={jid})")

    logger.debug(f"[build_and_test_dynamo_phase_tuples] Returning {len(phases3)} phases")
    return phases3  # Return 4-tuples: (name, dur, status, step_dict)
