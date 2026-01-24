#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Common dashboard helpers shared by HTML generators under dynamo-utils/html_pages/.

This file intentionally groups previously-split helper modules into one place to:
- avoid UI drift between dashboards
- reduce small-module sprawl
- keep <pre>-safe tree rendering + check-line rendering consistent

======================================================================================
NODE HIERARCHY REFERENCE
======================================================================================

This section documents the tree node structure used across all dashboards.

Node Hierarchy (with creators):
--------------------------------
```
LocalRepoNode (repository directory)                         ← Created by: show_local_branches.py
└─ BranchInfoNode (individual branch)                        ← Created by: show_local_branches.py, show_remote_branches.py
   ├─ BranchCommitMessageNode (commit message + PR link)     ← Created by: BranchInfoNode.to_tree_vm()
   ├─ BranchMetadataNode (timestamps / age)                  ← Created by: BranchInfoNode.to_tree_vm()
   ├─ ConflictWarningNode                                    ← Created by: BranchInfoNode.to_tree_vm() (when pr.conflict_message exists)
   ├─ BlockedMessageNode                                     ← Created by: BranchInfoNode.to_tree_vm() (when pr.blocking_message exists)
   ├─ PRStatusWithJobsNode (CI status for PRs)               ← Created by: add_pr_status_node_pass
   │  ├─ CIJobNode (CI check/job)                            ← Created by: build_ci_nodes_from_pr()
   │  │  ├─ CIJobNode (nested steps)                         ← Created by: add_job_steps_and_tests_pass
   │  │  │  └─ PytestTestNode (pytest tests)                 ← Created by: pytest_slowest_tests_from_raw_log (within add_job_steps_and_tests_pass)
   │  └─ RerunLinkNode                                       ← Created by: build_ci_nodes_from_pr() (on CI failure)
   └─ (no PR)                                                ← BranchInfoNode only (no workflow-status child node today)

Note: show_commit_history.py uses a different model:
  - Creates CIJobNode directly (no BranchInfoNode wrapper)
  - Renders commit metadata in the Jinja2 template (not via BranchCommitMessageNode)
  - Displays commits as a flat table (not a nested tree)
  - Still uses run_all_passes() for consistent CI job processing
```

Visual Example:
---------------
```
▼ dynamo/                                                    ← LocalRepoNode
│  ├─ [copy] Merged user/branch-1 [SHA]                      ← BranchInfoNode (merged PR)
│  │  ├─ commit message (#1234)                              ← BranchCommitMessageNode
│  │  ├─ (modified ..., created ..., ago)                    ← BranchMetadataNode
│  │  └─ ▶ PASSED  3 ✓26 ✗2                                  ← PRStatusWithJobsNode (collapsed)
│  │     ├─ ✓ check-1 (6m) [log]                             ← CIJobNode (hidden)
│  │     └─ ✗ check-2 (2m) [log] ▶ Snippet                   ← CIJobNode
│  ├─ [copy] user/branch-2 [SHA]                             ← BranchInfoNode (open PR)
│  │  ├─ fix: memory leak (#2345)                            ← BranchCommitMessageNode
│  │  ├─ (modified ..., created ..., ago)                    ← BranchMetadataNode
│  │  ├─ ⚠️  conflicts...                                    ← ConflictWarningNode (optional)
│  │  ├─ 🚫 Blocked by ...                                    ← BlockedMessageNode (optional)
│  │  └─ ▼ FAILED  2 ✓24 ✗1                                  ← PRStatusWithJobsNode (expanded)
│  │     ├─ ✓ check-1 (5m)                                   ← CIJobNode
│  │     └─ ▼ ✗ check-2 (3m) [log] ▶ Snippet                 ← CIJobNode
│  │        ├─ ✓ setup (10s)                                 ← CIStepNode
│  │        ├─ ▼ ✗ Run e2e tests (2m 30s)                    ← CIStepNode (test step)
│  │        │  ├─ ✓ [call] tests/...::test_foo (45s)         ← CIPytestNode
│  │        │  ├─ ✗ [call] tests/...::test_bar (1m 11s)      ← CIPytestNode
│  │        │  └─ ✓ [call] tests/...::test_baz (34s)         ← CIPytestNode
│  │        └─ ✓ cleanup (20s)                               ← CIStepNode
│  │     └─ 🔄 Restart failed jobs                            ← RerunLinkNode (only when CI failed + run_id known)
│  ├─ [copy] Closed user/abandoned [SHA]                     ← BranchInfoNode (closed, not merged)
│  ├─ [copy] user/feature [SHA]                              ← BranchInfoNode (branch with remote, no PR)
│  │  └─ (no CI shown; no PR → no checks fetched today)
│  ├─ [copy] local-branch [SHA]                              ← BranchInfoNode (local-only branch)
│  └─ [copy] Merged old-feature [SHA]                        ← BranchInfoNode (merged local branch)
```

Node Creation Flow:
-------------------
```
BranchInfoNode("feature/DIS-1200")              ← Created by show_local_branches.py
├─ TreeNodeVM(commit message + metadata)        ← Created by BranchInfoNode.to_tree_vm()
└─ PRStatusWithJobsNode(pr=PRInfo(...))         ← Created by add_pr_status_node_pass
   ├─ CIJobNode("[x86_64] vllm (amd64)")        ← Created by build_ci_nodes_from_pr()
   │  ├─ CIJobNode("Build Container")           ← Created by add_job_steps_and_tests_pass()
   │  ├─ CIJobNode("Run tests")                 ← Created by add_job_steps_and_tests_pass()
   │  │  ├─ PytestTestNode("test_serve[agg]")   ← Created by pytest_slowest_tests_from_raw_log()
   │  │  └─ PytestTestNode("test_router[nats]") ← Created by pytest_slowest_tests_from_raw_log()
   │  └─ CIJobNode("Docker Tag and Push")       ← Created by add_job_steps_and_tests_pass()
   ├─ ConflictWarningNode("⚠️ conflicts...")    ← Created by BranchInfoNode.to_tree_vm()
   └─ BlockedMessageNode("🚫 Blocked...")       ← Created by BranchInfoNode.to_tree_vm()
```

======================================================================================
"""

from __future__ import annotations

import hashlib
import html
import logging
import os
import re
import sys
import urllib.parse
import time
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from datetime import datetime, timezone

from common_github import GitHubAPIClient, classify_ci_kind, GITHUB_CACHE_STATS, GITHUB_API_STATS, COMMIT_HISTORY_PERF_STATS
from common_types import CIStatus
from cache_pytest_timings import PYTEST_TIMINGS_CACHE
from cache_snippet import SNIPPET_CACHE
from cache_commit_history import COMMIT_HISTORY_CACHE

# Add parent directory to path for common.py imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import (
    DEFAULT_UNSTABLE_TTL_S,
    DEFAULT_STABLE_TTL_S,
    DEFAULT_OPEN_PRS_TTL_S,
    DEFAULT_CLOSED_PRS_TTL_S,
    DEFAULT_NO_PR_TTL_S,
    DEFAULT_RAW_LOG_TEXT_TTL_S,
)

# Avoid circular import by importing CIJobNode only for type checking
if TYPE_CHECKING:
    from common_branch_nodes import CIJobNode

# Initialize module logger
logger = logging.getLogger(__name__)

# ======================================================================================
# Grafana URL Templates
# ======================================================================================

# Grafana Test Details dashboard URL template (for individual pytest tests)
# Example: https://grafana.nvidia.com/d/bf0set70vqygwb/test-details?orgId=283&var-branch=All&var-test_status=All&var-test=test_serve_deployment%5Baggregated%5D
# Note: Multiple var-test parameters can be present, but only the last one is used for single-select variables
GRAFANA_TEST_URL_TEMPLATE = "https://grafana.nvidia.com/d/bf0set70vqygwb/test-details?orgId=283&var-branch=All&var-test_status=All&var-test={test_name}"


# ======================================================================================
# Helper Functions
# ======================================================================================


def _format_ttl_duration(seconds: int) -> str:
    """Convert TTL seconds to human-readable format (e.g., '5m', '1h', '30d', '365d')."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes}m"
    elif seconds < 86400:
        hours = seconds // 3600
        return f"{hours}h"
    else:
        days = seconds // 86400
        return f"{days}d"

# ======================================================================================
# YAML Parsing Cache (Performance Optimization)
# ======================================================================================
# Cache parsed YAML data to avoid re-parsing on every commit
# Key: (repo_root, workflows_dir_mtime) -> (parent_child_mapping, job_name_to_id, job_to_file)
_yaml_parse_cache: Dict[Tuple[str, float], Tuple[Dict, Dict, Dict]] = {}

# ======================================================================================
# Shared ordering + default-expand policies
# ======================================================================================


# Note: CI job/check sorting is now handled by PASS 4 (sort_by_name_pass)
# in the centralized pipeline (run_passes). No pre-sorting needed.


def ci_should_expand_by_default(*, rollup_status: str, has_required_failure: bool) -> bool:
    """Shared UX rule: expand only when something truly needs attention.

    - expand for required failures (red X icon)
    - do NOT auto-expand long/step-heavy jobs by default (even if they have subsections)
    - expand for in-progress/pending states so "BUILDING" remains visible
    - do NOT auto-expand for optional failures, cancelled, unknown-only leaves, or all-green trees
    """
    if bool(has_required_failure):
        return True
    st = str(rollup_status or "").strip().lower()
    if st in {CIStatus.IN_PROGRESS.value, CIStatus.PENDING.value, "building", "running"}:
        return True
    return False

# ======================================================================================
# Shared UI snippets
# ======================================================================================

# Shared colors (keep consistent across dashboards).
COLOR_GREEN = "#2da44e"
# Slightly deeper than GitHub's default red; still readable and not overly saturated.
COLOR_RED = "#c83a3a"
COLOR_GREY = "#8c959f"
COLOR_YELLOW = "#bf8700"

# Shared SVG icon primitives (fixed-size; avoids emoji/font-size drift between legend/status bar).
def _octicon_svg(*, path_d: str, name: str, width: int = 12, height: int = 12) -> str:
    """Return a fixed-size Octicon-like SVG (16x16 viewBox) using currentColor fill."""
    pd = str(path_d or "").strip()
    if not pd:
        return ""
    nm = html.escape(str(name or "octicon"), quote=True)
    return (
        f'<svg aria-hidden="true" viewBox="0 0 16 16" version="1.1" '
        f'width="{int(width)}" height="{int(height)}" data-view-component="true" '
        f'class="octicon {nm}" fill="currentColor">'
        f'<path fill-rule="evenodd" d="{pd}"></path></svg>'
    )


def _circle_x_fill_svg(*, color: str, width: int = 12, height: int = 12, extra_style: str = "") -> str:
    """Filled circle with a white X (SVG)."""
    st = f"color: {html.escape(str(color or ''), quote=True)}; display: inline-flex; vertical-align: text-bottom;"
    if str(extra_style or "").strip():
        st += " " + str(extra_style).strip()
    return (
        f'<span style="{st}">'
        f'<svg aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="{int(width)}" height="{int(height)}" '
        f'data-view-component="true" class="octicon octicon-x-circle-fill" fill="currentColor">'
        '<circle cx="8" cy="8" r="8" fill="currentColor"></circle>'
        '<path d="M4.5 4.5l7 7m-7 0l7-7" stroke="#fff" stroke-width="2" stroke-linecap="round"></path>'
        "</svg></span>"
    )


def _circle_dot_fill_svg(*, color: str, width: int = 12, height: int = 12, extra_style: str = "") -> str:
    """Filled circle with a white dot (SVG) for 'pending'."""
    st = f"color: {html.escape(str(color or ''), quote=True)}; display: inline-flex; vertical-align: text-bottom;"
    if str(extra_style or "").strip():
        st += " " + str(extra_style).strip()
    return (
        f'<span style="{st}">'
        f'<svg aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="{int(width)}" height="{int(height)}" '
        f'data-view-component="true" class="octicon octicon-dot-circle-fill" fill="currentColor">'
        '<circle cx="8" cy="8" r="8" fill="currentColor"></circle>'
        '<circle cx="8" cy="8" r="2.2" fill="#fff"></circle>'
        "</svg></span>"
    )


def _clock_ring_svg(*, color: str, width: int = 12, height: int = 12, extra_style: str = "") -> str:
    """Clock/ring icon (SVG) for 'in progress'."""
    st = f"color: {html.escape(str(color or ''), quote=True)}; display: inline-flex; vertical-align: text-bottom;"
    if str(extra_style or "").strip():
        st += " " + str(extra_style).strip()
    # Matches the clock icon used in the commit-history legend, but via currentColor.
    return (
        f'<span style="{st}">'
        f'<svg aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="{int(width)}" height="{int(height)}" '
        f'data-view-component="true" class="octicon octicon-clock" fill="currentColor">'
        '<path d="M8 1C4.1 1 1 4.1 1 8s3.1 7 7 7 7-3.1 7-7-3.1-7-7-7zm0 12c-2.8 0-5-2.2-5-5s2.2-5 5-5 5 2.2 5 5-2.2 5-5 5z"></path>'
        '<path d="M8 4v5l3 2"></path>'
        "</svg></span>"
    )


def _dot_svg(*, color: str, width: int = 12, height: int = 12, extra_style: str = "") -> str:
    """Small dot (SVG) for 'unknown/other'."""
    st = f"color: {html.escape(str(color or ''), quote=True)}; display: inline-flex; vertical-align: text-bottom;"
    if str(extra_style or "").strip():
        st += " " + str(extra_style).strip()
    return (
        f'<span style="{st}">'
        f'<svg aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="{int(width)}" height="{int(height)}" '
        f'data-view-component="true" class="octicon octicon-dot" fill="currentColor">'
        '<circle cx="8" cy="8" r="2.6" fill="currentColor"></circle>'
        "</svg></span>"
    )


def ci_status_icon_context() -> Dict[str, str]:
    """Template context: consistent icon HTML used by all dashboards (legend, tooltips, status bar)."""
    return {
        "success_icon_html": status_icon_html(status_norm="success", is_required=False),
        "success_required_icon_html": status_icon_html(status_norm="success", is_required=True),
        "failure_required_icon_html": status_icon_html(status_norm="failure", is_required=True),
        "failure_optional_icon_html": status_icon_html(status_norm="failure", is_required=False),
        "in_progress_icon_html": status_icon_html(status_norm="in_progress", is_required=False),
        "pending_icon_html": status_icon_html(status_norm="pending", is_required=False),
        "cancelled_icon_html": status_icon_html(status_norm="cancelled", is_required=False),
        "skipped_icon_html": status_icon_html(status_norm="skipped", is_required=False),
    }

# CI UX: show "expected but not yet reported" checks.
#
# GitHub required-ness APIs only return contexts that already exist on the commit. For checks that are
# expected (especially required checks) but never start / never post a check-run context (e.g. filtered out),
# the dashboards used to show *nothing*, which is confusing.
#
# Both dashboards should use the exact same symbol so their trees look identical.
EXPECTED_CHECK_PLACEHOLDER_SYMBOL = "◇"

# Back-compat: older callsites import this for the optional pass count styling in the compact CI summary.
# (The current compact rendering no longer uses a "+N" format, but keep the constant to avoid crashes.)
PASS_PLUS_STYLE = "font-size: 10px; font-weight: 600; opacity: 0.9;"


def compact_ci_summary_html(
    *,
    success_required: int = 0,
    success_optional: int = 0,
    failure_required: int = 0,
    failure_optional: int = 0,
    in_progress_required: int = 0,
    in_progress_optional: int = 0,
    pending: int = 0,
    cancelled: int = 0,
) -> str:
    """Render the compact CI summary used in the GitHub column (shared across dashboards).

    This matches the visual style in `show_commit_history.j2`:
    - order:  success(required) N, success(optional) N, failure(required) N, failure(optional) N, then non-terminal states
    - colors: green/red/grey only (no orange)
    """
    sr = int(success_required or 0)
    so = int(success_optional or 0)
    fr = int(failure_required or 0)
    fo = int(failure_optional or 0)
    ip = int(in_progress_required or 0) + int(in_progress_optional or 0)
    pd = int(pending or 0)
    cx = int(cancelled or 0)

    parts: List[str] = []

    # Successes (required first, then optional), icon then count.
    if sr > 0:
        parts.append(
            f'<span style="color: {COLOR_GREEN};" title="Passed (required)">'
            f'{status_icon_html(status_norm=CIStatus.SUCCESS.value, is_required=True)}'
            f"<strong>{sr}</strong></span>"
        )
    if so > 0:
        parts.append(
            f'<span style="color: {COLOR_GREEN};" title="Passed (optional)">'
            f'{status_icon_html(status_norm=CIStatus.SUCCESS.value, is_required=False)}'
            f"<strong>{so}</strong></span>"
        )

    # Failures (required first, then optional), icon then count. All failures are red.
    if fr > 0:
        parts.append(
            f'<span style="color: {COLOR_RED};" title="Failed (required)">'
            f'{status_icon_html(status_norm=CIStatus.FAILURE.value, is_required=True)}'
            f"<strong>{fr}</strong></span>"
        )
    if fo > 0:
        parts.append(
            f'<span style="color: {COLOR_RED};" title="Failed (optional)">'
            f'{status_icon_html(status_norm=CIStatus.FAILURE.value, is_required=False)}'
            f"<strong>{fo}</strong></span>"
        )

    # Non-terminal states: grey (avoid orange).
    if ip > 0:
        parts.append(
            f'<span style="color: {COLOR_GREY};" title="In progress">'
            f'{status_icon_html(status_norm=CIStatus.IN_PROGRESS.value, is_required=False)}'
            f"<strong>{ip}</strong></span>"
        )
    if pd > 0:
        parts.append(
            f'<span style="color: {COLOR_GREY};" title="Pending">'
            f'{status_icon_html(status_norm=CIStatus.PENDING.value, is_required=False)}'
            f"<strong>{pd}</strong></span>"
        )
    if cx > 0:
        parts.append(
            f'<span style="color: {COLOR_GREY};" title="Canceled">'
            f'{status_icon_html(status_norm=CIStatus.CANCELLED.value, is_required=False)}'
            f"<strong>{cx}</strong></span>"
        )

    return " ".join([p for p in parts if str(p or "").strip()])


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
    - noncollapsible_icon: optional icon for non-collapsible nodes (keeps alignment with triangles).
      Supported values: "square" (renders ■). Default: "" (renders a blank placeholder).
    - node_key: a stable key for the logical node (used for debugging/caching, not DOM ids).
    - skip_dedup: if True, always render this node even if it appears multiple times (for CommitMessageNode, MetadataNode).
    - job_name: the CI job name for hierarchy matching (e.g., "backend-status-check", "vllm", "changed-files").
    - workflow_name: the workflow name this job belongs to (e.g., "container-validation-backends", "pre-merge").
    - variant: optional variant/matrix value (e.g., "amd64", "arm64", "cuda-13") for disambiguating similar jobs.
    - pr_number: PR number this node belongs to (for building CI nodes in pipeline). Can be negative for dummy PRs.
    - raw_html_content: optional raw HTML content rendered after the label (e.g., snippet <pre> blocks).
    """

    node_key: str
    label_html: str
    children: List["TreeNodeVM"] = field(default_factory=list)
    collapsible: bool = False
    default_expanded: bool = False
    triangle_tooltip: Optional[str] = None
    noncollapsible_icon: str = ""
    skip_dedup: bool = False
    job_name: str = ""  # CI job name for hierarchy matching (e.g., "backend-status-check")
    core_job_name: str = ""  # Core job name without workflow prefix/event suffix for matching
    short_job_name: str = ""  # Short YAML job name (e.g., "build-test") from augmentation
    workflow_name: str = ""  # Workflow this job belongs to (e.g., "container-validation-backends")
    variant: str = ""  # Optional variant (e.g., "amd64", "arm64") for matrix jobs
    pr_number: Optional[int] = None  # PR number for building CI nodes (negative = dummy PR)
    raw_html_content: str = ""  # Optional raw HTML content rendered after the label


def _dom_id_from_node_key(node_key: str) -> str:
    """Best-effort stable DOM id for a tree node's children container.

    This is used so URL state can target specific nodes across page regenerations.
    The caller should ensure node_key is stable and unique-enough (e.g. include repo/branch/SHA).
    """
    k = str(node_key or "")
    if not k:
        return ""
    # Prefix with a letter for HTML id validity.
    return f"tree_children_k_{_hash10(k)}"


def create_dummy_nodes_from_yaml_pass(nodes: List[TreeNodeVM]) -> List[TreeNodeVM]:
    """Create dummy nodes from YAML when no input nodes are provided.
    
    This pass is only used for debugging/visualization of the YAML structure.
    If nodes are provided, they pass through unchanged.
    If no nodes are provided AND YAML was parsed, creates minimal TreeNodeVM nodes
    for all jobs found in the workflow files.
    
    Args:
        nodes: Input nodes (may be empty)
    
    Returns:
        Original nodes if non-empty, or dummy nodes created from YAML
    """
    if nodes:
        return nodes
    
    if not _workflow_parent_child_mapping:
        return nodes
    
    logger.debug("[create_dummy_nodes_from_yaml_pass] Creating dummy nodes from YAML structure")
    
    # Collect all job names mentioned in the YAML
    all_job_names = set()
    for parent, children in _workflow_parent_child_mapping.items():
        all_job_names.add(parent)
        all_job_names.update(children)
    
    logger.debug(f"[create_dummy_nodes_from_yaml_pass] Job-to-file mapping has {len(_workflow_job_to_file)} entries")
    logger.debug(f"[create_dummy_nodes_from_yaml_pass] Total unique job names from mapping: {len(all_job_names)}")
    
    # Debug: show first 10 entries
    for i, (job_name, workflow_file) in enumerate(sorted(_workflow_job_to_file.items())):
        if i < 10:
            logger.debug(f"[create_dummy_nodes_from_yaml_pass]   {job_name} -> {workflow_file}")
    
    # Helper function to format arch text with colors
    def _format_arch_text_for_placeholder(text: str) -> str:
        """Format job name with architecture colors and annotations."""
        raw = str(text or "")
        # Detect arch token - handle both standalone and matrix format like "(cuda12.9, amd64)"
        m = re.search(r"\((?:[^,)]+,\s*)?(arm64|aarch64|amd64)\)", raw, flags=re.IGNORECASE)
        if not m:
            return html.escape(raw)
        
        arch = str(m.group(1) or "").strip().lower()
        # Determine color and prefix with arch alias
        if arch in {"arm64", "aarch64"}:
            color = "#b8860b"  # Dark yellow/gold for arm64
            raw2 = f"[aarch64] {raw}"
            return f'<span style="color: {color};">{html.escape(raw2)}</span>'
        elif arch == "amd64":
            color = "#0969da"  # Blue for amd64
            raw2 = f"[x86_64] {raw}"
            return f'<span style="color: {color};">{html.escape(raw2)}</span>'
        return html.escape(raw)
    
    # Create minimal TreeNodeVM for each job
    skeleton_nodes = []
    for job_name in sorted(all_job_names):
        # Get job_id if available
        job_id = _workflow_job_name_to_id.get(job_name, "")
        
        # Get workflow filename if available
        workflow_file = _workflow_job_to_file.get(job_name, "")
        
        # Format: job_id (job_name) if they differ, otherwise just job_name
        # job_id is in fixed-width font, job_name (description) is in normal font with lighter gray color
        # Apply arch formatting to get colors and "; aarch64" / "; x86_64" suffixes
        if job_id and job_id != job_name:
            formatted_job_id = _format_arch_text_for_placeholder(job_id)
            display_text = f'<span style="font-family: monospace;">{formatted_job_id}</span> <span style="color: #8c959f;">({job_name})</span>'
        else:
            formatted_job_name = _format_arch_text_for_placeholder(job_name)
            display_text = f'<span style="font-family: monospace;">{formatted_job_name}</span>'
        
        # Add workflow file annotation with full path
        if workflow_file:
            file_path = f'.github/workflows/{workflow_file}'
            file_annotation = f'<span style="color: #888;">[defined in {file_path}]</span>'
        else:
            file_annotation = '<span style="color: #888;">[defined in YAML]</span>'
        
        node = TreeNodeVM(
            node_key=f"skeleton:{job_name}",
            label_html=f'{display_text} {file_annotation}',
            children=[],
            collapsible=False,
            default_expanded=False,
            job_name=job_name,  # IMPORTANT: Set job_name for matching in PASS 1.2
            workflow_name="",
            variant="",
        )        
        skeleton_nodes.append(node)
    
    logger.debug(f"[create_dummy_nodes_from_yaml_pass] Created {len(skeleton_nodes)} dummy nodes from YAML")
    return skeleton_nodes


def run_all_passes(
    ci_nodes: List,  # List[BranchNode] from common_branch_nodes
    repo_root: Path,
    commit_sha: str = "",
    # NEW: Optional parameters for BranchInfoNode processing (PASS -1)
    github_api: Optional = None,  # GitHubAPIClient
    page_root_dir: Optional[Path] = None,
    refresh_checks: bool = False,
    allow_fetch_checks: bool = True,
    enable_success_build_test_logs: bool = False,
    context_prefix: str = "",
    run_verifier_pass: bool = False,  # NEW: Enable verification passes
) -> List[TreeNodeVM]:
    """
    Centralized tree node processing pipeline.
    
    Simple orchestrator that calls passes in sequence.
    
    Can handle two types of input:
    1. List[CIJobNode] - Flat CI nodes (original use case, called from PRStatusWithJobsNode)
    2. List[BranchInfoNode] - Branch hierarchy (new use case, called from show_*.py)
    
    Args:
        ci_nodes: List of BranchNode objects (CIJobNode or BranchInfoNode)
        repo_root: Path to the repository root (for .github/workflows/ parsing)
        commit_sha: Commit SHA for per-commit node uniqueness
        github_api: GitHub API client (required for BranchInfoNode/PR processing)
        page_root_dir: Page root directory (required for BranchInfoNode/PR processing)
        refresh_checks: Force refresh checks cache
        allow_fetch_checks: Allow network fetch for checks
        enable_success_build_test_logs: Cache raw logs for successful jobs
        context_prefix: Prefix for context keys (e.g., "remote:", "local:")
        run_verifier_pass: Enable verification passes (verify_job_details_pass, verify_tree_structure_pass)
    
    Returns:
        Processed list of TreeNodeVM nodes with YAML augmentation applied.
    """
    import time  # For per-pass timing
    pass_timings = {}  # Track timing for each pass
    
    logger.info(f"[run_all_passes] Starting with {len(ci_nodes)} nodes")
    
    # PASS 1: Add PRStatusWithJobsNode as children to BranchInfoNode (if PR exists)
    # This must run BEFORE all other passes (operates on BranchNode layer)
    # Only runs if we have BranchInfoNode instances with github_api and page_root_dir
    from common_branch_nodes import BranchInfoNode
    has_branch_info_nodes = any(isinstance(n, BranchInfoNode) for n in ci_nodes)
    
    if has_branch_info_nodes and github_api and page_root_dir:
        t0 = time.monotonic()
        ci_nodes = add_pr_status_node_pass(
            nodes=ci_nodes,
            github_api=github_api,
            repo_root=repo_root,
            page_root_dir=page_root_dir,
            refresh_checks=refresh_checks,
            allow_fetch_checks=allow_fetch_checks,
            enable_success_build_test_logs=enable_success_build_test_logs,
            context_prefix=context_prefix,
        )
        pass_timings['pass1_add_pr_status'] = time.monotonic() - t0

    # PASS 1.5: Prefetch GitHub Actions job details in batch (batch fetch all jobs by run_id)
    # This populates the job details cache so individual lookups hit cache instead of making API calls
    # OPTIMIZATION: Reduces 500-1000 per-job API calls down to 10-20 per-run batch calls (90-95% reduction)
    t0 = time.monotonic()
    ci_nodes = prefetch_actions_job_details_pass(ci_nodes, github_api=github_api)
    pass_timings['pass1.5_prefetch_job_details'] = time.monotonic() - t0

    # PASS 2: Add job steps and pytest tests to CIJobNode children (before conversion to TreeNodeVM)
    # This must run BEFORE augment_ci_with_yaml_info_pass so children are in place
    t0 = time.monotonic()
    ci_nodes = add_job_steps_and_tests_pass(ci_nodes, repo_root)
    pass_timings['pass2_add_steps_and_tests'] = time.monotonic() - t0

    # PASS 2.5: Verify job details (steps, pytest tests, duration)
    # This runs right after add_job_steps_and_tests_pass to validate that expected data is present
    # Only runs if --run-verifier-pass flag is set
    if run_verifier_pass:
        t0 = time.monotonic()
        verify_job_details_pass(ci_nodes, commit_sha=commit_sha)
        pass_timings['pass2.5_verify_job_details'] = time.monotonic() - t0

    # Parse YAML workflows to build mappings (job names, dependencies, etc.)
    # Note: This is NOT a pass - it doesn't modify nodes, just parses YAML files
    t0 = time.monotonic()
    _, yaml_mappings = parse_workflow_yaml_and_build_mapping_pass([], repo_root, commit_sha=commit_sha)
    pass_timings['yaml_parse'] = time.monotonic() - t0
    
    # PASS 3: Augment CI nodes with YAML information (short names, dependencies)
    # This pass also converts BranchNode to TreeNodeVM
    t0 = time.monotonic()
    augmented_nodes = augment_ci_with_yaml_info_pass(ci_nodes, yaml_mappings)
    pass_timings['pass3_augment_yaml_info'] = time.monotonic() - t0
    
    # PASS 4: Move jobs under parent nodes (BATCHED for performance)
    # Instead of calling move_jobs_by_prefix_pass 25+ times (O(25×n)), we batch all groupings
    # into a single pass that processes all prefixes in one iteration (O(n))
    grouping_rules = [
        # Backend status checks
        ("vllm", "backend-status-check", "", False),
        ("sglang", "backend-status-check", "", False),
        ("trtllm", "backend-status-check", "", False),
        ("operator", "backend-status-check", "", False),
        
        # Other parent nodes
        ("deploy-", "deploy", "", False),
        ("build-test", "dynamo-status-check", "", False),
        ("Post-Merge CI / ", "post-merge-ci", "Post-Merge CI", True),
        ("Nightly CI / ", "nightly-ci", "Nightly CI", True),
        
        # Fast jobs (create parent only if has children)
        ("broken-links-check", "_fast", "Jobs that tend to run fast", True),
        ("build-docs", "_fast", "Jobs that tend to run fast", True),
        ("changed-files", "_fast", "Jobs that tend to run fast", True),
        ("clean", "_fast", "Jobs that tend to run fast", True),
        ("clippy", "_fast", "Jobs that tend to run fast", True),
        ("CodeRabbit", "_fast", "Jobs that tend to run fast", True),
        ("dco-comment", "_fast", "Jobs that tend to run fast", True),
        ("event_file", "_fast", "Jobs that tend to run fast", True),
        ("label", "_fast", "Jobs that tend to run fast", True),
        ("lychee", "_fast", "Jobs that tend to run fast", True),
        ("trigger-ci", "_fast", "Jobs that tend to run fast", True),
        ("Validate PR title", "_fast", "Jobs that tend to run fast", True),
    ]
    
    t0 = time.monotonic()
    grouped_nodes = move_jobs_by_prefix_batch_pass(augmented_nodes, grouping_rules)
    pass_timings['pass4_batch_grouping'] = time.monotonic() - t0
    
    # PASS 5: Sort nodes by name
    t0 = time.monotonic()
    sorted_nodes = sort_nodes_by_name_pass(grouped_nodes)
    pass_timings['pass5_sort_by_name'] = time.monotonic() - t0
    
    # PASS 6: Expand nodes with required failures in descendants
    t0 = time.monotonic()
    final_nodes = expand_required_failure_descendants_pass(sorted_nodes)
    pass_timings['pass6_expand_required_failures'] = time.monotonic() - t0
    
    # PASS 7: Move required jobs to the top (alphabetically sorted)
    t0 = time.monotonic()
    final_nodes = move_required_jobs_to_top_pass(final_nodes)
    pass_timings['pass7_move_required_to_top'] = time.monotonic() - t0
    
    # PASS 8: Verify the final tree structure
    # Only runs if --run-verifier-pass flag is set
    if run_verifier_pass:
        t0 = time.monotonic()
        verify_tree_structure_pass(final_nodes, ci_nodes, commit_sha=commit_sha)
        pass_timings['pass8_verify_structure'] = time.monotonic() - t0
    
    # Log per-pass timing breakdown
    if pass_timings:
        timing_str = ", ".join([f"{k}={v:.3f}s" for k, v in pass_timings.items()])
        logger.debug(f"[run_all_passes] Pass timing breakdown: {timing_str}")
    
    logger.info(f"[run_all_passes] All passes complete (1-8), returning {len(final_nodes)} root nodes")
    return final_nodes


#
# Pass implementations (in run_all_passes order)
# -----------------------------------------------------------------------------

def add_pr_status_node_pass(
    nodes: List,  # List[BranchNode]
    github_api,  # GitHubAPIClient
    repo_root: Path,
    page_root_dir: Path,
    refresh_checks: bool = False,
    allow_fetch_checks: bool = True,
    enable_success_build_test_logs: bool = False,
    context_prefix: str = "",
) -> List:
    """
    Add PRStatusWithJobsNode as child to BranchInfoNode (if PR exists).
    
    This pass runs FIRST and operates on the BranchNode layer.
    It centralizes PR status node creation logic that was previously duplicated
    across show_local_branches.py and show_remote_branches.py.
    
    For each BranchInfoNode that has a PR:
    - Create PRStatusWithJobsNode with the PR
    - Add it as a child to BranchInfoNode
    - PRStatusWithJobsNode.__init__ will automatically:
      - Call build_ci_nodes_from_pr() → creates List[CIJobNode]
      - Call run_all_passes() recursively for CI nodes
      - Store TreeNodeVM children in _ci_children_vm
    
    Args:
        nodes: List of BranchNode (may contain BranchInfoNode instances)
        github_api: GitHub API client
        repo_root: Path to repo root
        page_root_dir: Path to page root directory
        refresh_checks: Force refresh checks cache
        allow_fetch_checks: Allow network fetch for checks
        enable_success_build_test_logs: Cache raw logs for successful jobs
        context_prefix: Prefix for context_key (e.g., "remote:", "local:")
    
    Returns:
        Same list of nodes (modified in-place with new children)
    """
    from common_branch_nodes import BranchInfoNode, PRStatusWithJobsNode
    
    def process_node(node) -> None:
        """Recursively process nodes to add PR status children."""
        # Process this node if it's a BranchInfoNode with a PR
        if isinstance(node, BranchInfoNode) and node.pr is not None:
            # Check if PRStatusWithJobsNode already exists as a child
            has_pr_status = any(
                isinstance(c, PRStatusWithJobsNode) for c in (node.children or [])
            )
            
            if not has_pr_status:
                pr = node.pr
                
                # Build context_key (stable DOM ID seed)
                label = node.label
                sha = node.sha or ""
                context_key = f"{context_prefix}{label}:{sha}:pr{pr.number}"
                
                # Compute branch commit datetime for TTL calculation
                branch_commit_dt = node.commit_datetime
                
                # Create PR status node (CI nodes built inside __init__)
                status_node = PRStatusWithJobsNode(
                    label="",
                    pr=pr,
                    github_api=github_api,
                    repo_root=repo_root,
                    page_root_dir=page_root_dir,
                    refresh_checks=refresh_checks,
                    branch_commit_dt=branch_commit_dt,
                    allow_fetch_checks=allow_fetch_checks,
                    context_key=context_key,
                    enable_success_build_test_logs=enable_success_build_test_logs,
                )
                
                # Add as child to BranchInfoNode
                node.add_child(status_node)
                
                logger.debug(
                    f"[PASS -1] Added PRStatusWithJobsNode for PR #{pr.number} to branch '{label}'"
                )
        
        # Recursively process children
        for child in (node.children or []):
            process_node(child)
    
    # Process all nodes recursively
    for node in nodes:
        process_node(node)
    
    logger.info(f"[PASS -1] add_pr_status_node_pass complete")
    return nodes


def prefetch_actions_job_details_pass(
    ci_nodes: List,
    github_api: Optional[GitHubAPIClient] = None,
) -> List:
    """Prefetch GitHub Actions job details for all jobs in batch (PASS 1.5).

    OPTIMIZATION (2026-01-18): Instead of fetching job details individually (1 API call per job),
    this pass extracts all run_ids and batch-fetches all jobs using /actions/runs/{run_id}/jobs
    (1 API call per run). This populates the job details cache so subsequent individual lookups
    hit cache instead of making API calls.

    Benefits:
        - 90-95% reduction in API calls: 500-1000 per-job calls → 10-20 per-run calls
        - Faster: fewer network round-trips
        - Rate limit friendly: batched fetching

    This pass runs BEFORE add_job_steps_and_tests_pass so the cache is warm when individual
    job details are requested.

    Args:
        ci_nodes: List of BranchNode objects (may contain CIJobNode instances)
        github_api: Optional GitHubAPIClient for batch fetching

    Returns:
        Same list of nodes (unmodified; this pass only populates caches)
    """
    from common_branch_nodes import CIJobNode

    if not github_api:
        return ci_nodes

    logger.info(f"[prefetch_actions_job_details_pass] Extracting run_ids from {len(ci_nodes)} nodes")

    # Extract all unique run_ids from CI jobs
    run_ids: Set[str] = set()

    def extract_run_ids(node):
        """Recursively extract run_ids from CIJobNode instances."""
        if isinstance(node, CIJobNode):
            run_id = str(getattr(node, "run_id", "") or "").strip()
            if run_id and run_id.isdigit():
                run_ids.add(run_id)
        # Recurse into children
        for child in (getattr(node, "children", []) or []):
            extract_run_ids(child)

    for node in ci_nodes:
        extract_run_ids(node)

    if not run_ids:
        logger.info(f"[prefetch_actions_job_details_pass] No run_ids found, skipping batch fetch")
        return ci_nodes

    logger.info(f"[prefetch_actions_job_details_pass] Batch fetching job details for {len(run_ids)} unique runs")

    try:
        # Batch fetch all jobs for these runs (populates cache)
        job_map = github_api.get_actions_runs_jobs_batched(
            owner="ai-dynamo",
            repo="dynamo",
            run_ids=list(run_ids),
            ttl_s=30 * 24 * 3600,  # 30 days cache
        )
        logger.info(f"[prefetch_actions_job_details_pass] Prefetched {len(job_map)} job details into cache")
    except Exception as e:
        logger.warning(f"[prefetch_actions_job_details_pass] Batch fetch failed: {e}")

    logger.info(f"[prefetch_actions_job_details_pass] Complete")
    return ci_nodes


def add_job_steps_and_tests_pass(ci_nodes: List, repo_root: Path) -> List:
    """
    Add job steps and pytest tests as children to CIJobNode objects.
    
    This pass runs BEFORE conversion to TreeNodeVM to ensure step children are in place.
    It handles all the step/test extraction logic that was previously duplicated across
    build_ci_nodes_from_pr and show_commit_history.py.
    
    Args:
        ci_nodes: List of BranchNode objects (may contain CIJobNode instances)
        repo_root: Path to repository root (for resolving raw log paths)
        
    Returns:
        Same list of nodes, with CIJobNode.children populated with step/test nodes
    """
    import time  # For detailed timing
    from common_branch_nodes import CIJobNode, CIStepNode, CIPytestNode, _duration_str_to_seconds
    
    logger.info(f"[add_job_steps_and_tests_pass] Processing {len(ci_nodes)} nodes")
    
    # Track timing breakdown
    timing_stats = {
        'ci_subsection_tuples_for_job': 0.0,
        'create_step_nodes': 0.0,
        'total_jobs_processed': 0,
        'total_steps_created': 0,
    }

    for node in ci_nodes:
        if not isinstance(node, CIJobNode):
            continue

        # Skip if this node already has children (avoid re-processing)
        if node.children:
            logger.debug(f"[add_job_steps_and_tests_pass] Skipping {node.job_id} (already has {len(node.children)} children)")
            continue

        timing_stats['total_jobs_processed'] += 1

        # Extract parameters from the CIJobNode
        job_name = node.display_name or node.job_id or ""
        job_url = node.log_url or ""
        github_api = node.github_api
        page_root_dir = node.page_root_dir
        raw_log_href = node.raw_log_href or ""
        is_required = node.is_required

        # Resolve raw log path
        raw_log_path: Optional[Path] = None
        if raw_log_href and page_root_dir:
            raw_log_path = page_root_dir / raw_log_href

        # Parse duration
        dur_str = node.duration or ""
        dur_seconds = _duration_str_to_seconds(dur_str)

        # Get step tuples using the centralized logic (TIME THIS)
        t0 = time.monotonic()
        step_tuples = ci_subsection_tuples_for_job(
            github_api=github_api,
            job_name=job_name,
            job_url=job_url,
            raw_log_path=raw_log_path,
            duration_seconds=float(dur_seconds or 0.0),
            is_required=bool(is_required),
            long_job_threshold_s=10.0 * 60.0,
            step_min_s=10.0,
        )
        timing_stats['ci_subsection_tuples_for_job'] += time.monotonic() - t0
        timing_stats['ci_subsection_tuples_for_job'] += time.monotonic() - t0

        if not step_tuples:
            # Debug: Log why we skipped this job
            logger.debug(
                f"[add_job_steps_and_tests_pass] No step tuples for job: {job_name} "
                f"(github_api={'present' if github_api else 'MISSING'}, "
                f"job_url={'present' if job_url else 'MISSING'}, "
                f"is_build_test={job_name_wants_pytest_details(job_name)}, "
                f"duration={dur_str})"
            )
            continue

        # Create child nodes for each step (TIME THIS)
        t0 = time.monotonic()
        # Pytest tests (with └─ prefix) should be children of the "Run tests" step
        current_test_parent: Optional[CIJobNode] = None
        for (step_name, step_dur, step_status) in step_tuples:
            step_name_s = str(step_name or "")
            
            # Check if this is a pytest test (has └─ prefix)
            if step_name_s.startswith("  └─ "):
                # This is a pytest test - add as child of the current "Run tests" step
                if current_test_parent:
                    test_name = step_name_s[len("  └─ "):]  # Remove prefix
                    test_node = CIPytestNode(
                        job_id=test_name,
                        display_name="",
                        status=step_status,
                        duration=step_dur,
                        log_url="",
                        children=[],
                    )
                    current_test_parent.children.append(test_node)
            else:
                # This is a regular step
                # For "Build and Test - dynamo", classify the step type
                if job_name == "Build and Test - dynamo":
                    kind = classify_ci_kind(str(step_name))
                    step_id = f"{kind}: {step_name}" if kind and kind != "check" else str(step_name)
                else:
                    step_id = str(step_name)

                step_node = CIStepNode(
                    job_id=step_id,
                    display_name=str(step_name),  # Store original name (without augmentation) for verifier
                    status=step_status,
                    duration=step_dur,
                    log_url="",
                    children=[],
                )
                node.children.append(step_node)

                # If this step name indicates a Python test step, treat it as a test parent
                # Examples: "Run e2e tests", "Run test", "test run", "pytest", "test: pytest"
                if is_python_test_step(step_name_s):
                    current_test_parent = step_node
                else:
                    current_test_parent = None

        timing_stats['create_step_nodes'] += time.monotonic() - t0
        timing_stats['total_steps_created'] += len(node.children)
        logger.debug(f"[add_job_steps_and_tests_pass] Added {len(node.children)} step/test children to {node.job_id}")
    
    # Log detailed timing breakdown
    logger.debug(
        f"[add_job_steps_and_tests_pass] Timing breakdown: "
        f"jobs_processed={timing_stats['total_jobs_processed']}, "
        f"steps_created={timing_stats['total_steps_created']}, "
        f"ci_subsection_tuples={timing_stats['ci_subsection_tuples_for_job']:.3f}s, "
        f"create_nodes={timing_stats['create_step_nodes']:.3f}s"
    )
    
    logger.info(f"[add_job_steps_and_tests_pass] Complete")
    return ci_nodes


def convert_branch_nodes_to_tree_vm_pass(ci_nodes: List) -> List[TreeNodeVM]:
    """Convert BranchNode objects to TreeNodeVM.
    
    Takes a list of BranchNode objects (from build_ci_nodes_from_pr or mock_build_ci_nodes)
    and converts them to TreeNodeVM for rendering.
    
    Args:
        ci_nodes: List of BranchNode objects for a single PR
        
    Returns:
        List of TreeNodeVM objects representing actual CI info from GitHub
    """
    logger.info(f"Converting {len(ci_nodes)} BranchNode objects to TreeNodeVM")
    
    ci_info_nodes: List[TreeNodeVM] = []
    for idx, ci_node in enumerate(ci_nodes):
        # Convert BranchNode to TreeNodeVM (core_job_name is now set in to_tree_vm())
        node_vm = ci_node.to_tree_vm()
        
        # Debug: Check if core_job_name was propagated
        if idx < 5:
            core_name = node_vm.core_job_name or "<none>"
            logger.debug(f"  [{idx}] TreeNodeVM has core_job_name='{core_name}'")
        
        ci_info_nodes.append(node_vm)
    
    logger.info(f"Converted {len(ci_info_nodes)} nodes to TreeNodeVM")
    return ci_info_nodes


def parse_workflow_yaml_and_build_mapping_pass(
    flat_nodes: List[TreeNodeVM],
    repo_root: Path,
    commit_sha: str = "",
) -> List[TreeNodeVM]:
    """
    Parse YAML workflows to build job name mappings and dependencies.
    
    NOTE: This is NOT a pass in the traditional sense - it doesn't modify nodes.
    It only parses .github/workflows/*.yml files and returns mappings for use by PASS 3.
    
    This reads .github/workflows/*.yml files and builds mappings for:
    - job_name_to_id: Maps display names to YAML job IDs
    - parent_child_mapping: Maps job names to their dependencies
    - job_to_file: Maps job IDs to workflow files
    
    The mappings are returned and used by subsequent passes to augment CI nodes.
    
    PERFORMANCE: Results are cached per repo_root + workflows_dir mtime to avoid
    re-parsing YAML files on every commit. Cache is automatically invalidated when
    workflow files change.
    
    Workflow YAML structure:
        job_id:
            name: "display name (possibly with ${{ matrix.var }})"
            needs: [other_job_id1, other_job_id2]
    
    Example:
        backend-status-check:
            needs: [changed-files, vllm, sglang, trtllm, operator]
        vllm:
            name: vllm (${{ matrix.platform.arch }})
            needs: [changed-files]
    
    This creates metadata:
        - "backend-status-check" should have children: [vllm (${{ matrix.platform.arch }}), sglang (${{ ...}}), ...]
        - "vllm (${{ matrix.platform.arch }})" should have parent: backend-status-check
    
    Args:
        flat_nodes: Flat list of TreeNodeVM nodes (actual check runs from GitHub API)
        repo_root: Path to repository root containing .github/workflows/
        commit_sha: Commit SHA for logging
    
    Returns:
        The same flat list of nodes, now annotated with metadata about parents/children
    """
    global _workflow_parent_child_mapping
    global _workflow_job_name_to_id
    global _workflow_job_to_file
    global _yaml_parse_cache
    
    workflows_dir = Path(repo_root) / ".github" / "workflows"
    if not workflows_dir.exists() or not workflows_dir.is_dir():
        logger.debug(f"[parse_workflow_yaml_and_build_mapping_pass] No workflows directory found")
        return flat_nodes, {
            'parent_child_mapping': {},
            'job_name_to_id': {},
            'job_to_file': {},
        }
    
    # Cache key: (repo_root, workflows_dir mtime)
    # Use mtime of the workflows directory to detect changes to any YAML file
    try:
        workflows_dir_mtime = os.path.getmtime(workflows_dir)
    except OSError:
        workflows_dir_mtime = 0.0
    
    cache_key = (str(repo_root), workflows_dir_mtime)
    
    # Check cache
    if cache_key in _yaml_parse_cache:
        # Cache hit! Reuse parsed data
        cached_data = _yaml_parse_cache[cache_key]
        _workflow_parent_child_mapping = cached_data[0].copy()
        _workflow_job_name_to_id = cached_data[1].copy()
        _workflow_job_to_file = cached_data[2].copy()
        logger.debug(f"[parse_workflow_yaml_and_build_mapping_pass] Using cached YAML data (repo_root={repo_root})")
        logger.debug(f"[parse_workflow_yaml_and_build_mapping_pass] Found {len(_workflow_parent_child_mapping)} parent-child relationships")
        return flat_nodes, {
            'parent_child_mapping': dict(_workflow_parent_child_mapping),
            'job_name_to_id': dict(_workflow_job_name_to_id),
            'job_to_file': dict(_workflow_job_to_file),
        }
    
    # Cache miss - parse YAML files
    logger.debug(f"[parse_workflow_yaml_and_build_mapping_pass] Parsing YAML to annotate nodes with parent/child metadata")
    logger.debug(f"[parse_workflow_yaml_and_build_mapping_pass] repo_root={repo_root}")
    
    _workflow_parent_child_mapping = {}  # Reset: job_name -> list of child job_names
    _workflow_job_name_to_id = {}  # Reset: job_name -> job_id
    
    # Parse YAML files to extract needs: relationships
    for workflow_file in workflows_dir.glob("*.yml"):
        with open(workflow_file, 'r') as f:
            workflow_data = yaml.safe_load(f)
        
        if not workflow_data or 'jobs' not in workflow_data:
            continue
        
        jobs = workflow_data.get('jobs', {})
        
        # Build a map of job_id -> job_name for resolving needs
        # Job ID is the key in YAML (e.g., "vllm"), job name is the display name (e.g., "vllm (${{ matrix.platform.arch }})")
        job_id_to_name = {}
        for job_id, job_data in jobs.items():
            if isinstance(job_data, dict):
                job_name = job_data.get('name', job_id)
                job_id_to_name[job_id] = job_name
                # Store reverse mapping for display purposes
                _workflow_job_name_to_id[job_name] = job_id
                # Track which file this job comes from
                _workflow_job_to_file[job_name] = workflow_file.name
        
        # Now process needs relationships using job names
        for job_id, job_data in jobs.items():
            if not isinstance(job_data, dict):
                continue
            
            job_name = job_data.get('name', job_id)
            needs = job_data.get('needs', [])
            
            # Normalize needs to list
            if isinstance(needs, str):
                needs = [needs]
            elif not isinstance(needs, list):
                needs = []
            
            if needs:
                # Resolve job IDs to job names in the needs list
                # e.g., needs: ['vllm'] → ['vllm (${{ matrix.platform.arch }})']
                resolved_needs = []
                for need_id in needs:
                    need_name = job_id_to_name.get(need_id, need_id)
                    resolved_needs.append(need_name)
                
                # Store the mapping: this job_name needs these child job_names
                _workflow_parent_child_mapping[job_name] = resolved_needs
                logger.debug(f"[parse_workflow_yaml_and_build_mapping_pass]   {job_name} needs: {resolved_needs}")
    
    # Store in cache
    _yaml_parse_cache[cache_key] = (
        _workflow_parent_child_mapping.copy(),
        _workflow_job_name_to_id.copy(),
        _workflow_job_to_file.copy(),
    )
    
    logger.debug(f"[parse_workflow_yaml_and_build_mapping_pass] Found {len(_workflow_parent_child_mapping)} parent-child relationships")
    logger.debug(f"[parse_workflow_yaml_and_build_mapping_pass] Returning {len(flat_nodes)} nodes (unchanged, will be connected in PASS 3)")
    
    # Return the populated mappings along with the nodes
    return flat_nodes, {
        'parent_child_mapping': dict(_workflow_parent_child_mapping),
        'job_name_to_id': dict(_workflow_job_name_to_id),
        'job_to_file': dict(_workflow_job_to_file),
    }


# Global to store the parent-child mapping from PASS 1.1
_workflow_parent_child_mapping: Dict[str, List[str]] = {}
# Global to store job_name -> job_id mapping from PASS 1.1
_workflow_job_name_to_id: Dict[str, str] = {}
# Global to store job_name -> workflow filename mapping from PASS 1.1
_workflow_job_to_file: Dict[str, str] = {}


def augment_ci_with_yaml_info_pass(
    original_ci_nodes: List,  # List[BranchNode] - original CIJobNode objects
    yaml_mappings: Dict[str, Dict],  # Mappings from YAML parsing
) -> List[TreeNodeVM]:
    """
    Augment CI nodes with YAML information and convert to TreeNodeVM.
    
    This pass builds a mapping from long check name to short YAML job_id,
    then updates each CIJobNode with short_job_name and yaml_dependencies.
    Finally, it converts the augmented BranchNode objects to TreeNodeVM.
    
    Args:
        original_ci_nodes: Original CIJobNode objects (before conversion to TreeNodeVM)
        yaml_mappings: Dict with keys 'parent_child_mapping', 'job_name_to_id', 'job_to_file'
        
    Returns:
        List of TreeNodeVM nodes with augmented information
    """
    from common_branch_nodes import CIJobNode
    
    logger.info(f"[augment_ci_with_yaml_info_pass] Augmenting {len(original_ci_nodes)} CI nodes with YAML info")
    
    # Extract mappings
    parent_child_mapping = yaml_mappings.get('parent_child_mapping', {})
    job_name_to_id = yaml_mappings.get('job_name_to_id', {})
    
    # Build a hash map: long_name (from check) -> (short_name, dependencies)
    # The job_name_to_id has keys like "Build and Test - dynamo" and values like "build-test"
    long_to_short = {}
    for yaml_job_name, yaml_job_id in job_name_to_id.items():
        # yaml_job_name is the 'name:' field from YAML (e.g., "Build and Test - dynamo")
        # yaml_job_id is the job key from YAML (e.g., "build-test")
        dependencies = parent_child_mapping.get(yaml_job_name, [])
        long_to_short[yaml_job_name] = (yaml_job_id, dependencies)
        logger.debug(f"[augment_ci_with_yaml_info_pass] Mapping: '{yaml_job_name}' -> '{yaml_job_id}'")
    
    logger.info(f"[augment_ci_with_yaml_info_pass] Built mapping with {len(long_to_short)} entries")
    
    # Traverse through each CI node and update
    augmented_count = 0
    
    for node in original_ci_nodes:
        core_name = ""
        if isinstance(node, CIJobNode):
            core_name = str(node.core_job_name or "")

            # Direct lookup in the hash map
            if core_name in long_to_short:
                short_name, dependencies = long_to_short[core_name]
                node.short_job_name = short_name
                node.yaml_dependencies = dependencies
                augmented_count += 1
                logger.debug(
                    f"[augment_ci_with_yaml_info_pass] Augmented '{core_name}' -> short='{short_name}', deps={dependencies}"
                )
                continue

        # Debug only: do not crash on non-CI nodes / missing core names.
        if core_name:
            logger.debug(f"[augment_ci_with_yaml_info_pass] No match for '{core_name}'")
    
    logger.info(f"[augment_ci_with_yaml_info_pass] Augmented {augmented_count}/{len(original_ci_nodes)} CI nodes with YAML info")
    
    # Convert the augmented CIJobNodes to TreeNodeVM
    augmented_tree_nodes = []
    for node in original_ci_nodes:
        augmented_tree_nodes.append(node.to_tree_vm())
    
    return augmented_tree_nodes


def move_jobs_by_prefix_batch_pass(
    nodes: List[TreeNodeVM],
    grouping_rules: List[Tuple[str, str, str, bool]],
) -> List[TreeNodeVM]:
    """
    OPTIMIZED: Batch version of move_jobs_by_prefix_pass that processes all grouping rules
    in a single pass instead of iterating through nodes multiple times.
    
    Performance: O(n) instead of O(n×m) where m=number of rules (~25)
    
    Args:
        nodes: List of TreeNodeVM nodes (root level)
        grouping_rules: List of (prefix, parent_name, parent_label, create_if_has_children) tuples
    
    Returns:
        List of TreeNodeVM nodes with all matching jobs moved under parents
    """
    logger.info(f"[move_jobs_by_prefix_batch_pass] Processing {len(grouping_rules)} grouping rules in single pass")
    
    # Build lookup structures for all rules
    # parent_name -> {prefix_set, label, create_if_has_children, matched_jobs}
    parent_info: Dict[str, dict] = {}
    for prefix, parent_name, parent_label, create_if_has_children in grouping_rules:
        if parent_name not in parent_info:
            parent_info[parent_name] = {
                "prefixes": [],
                "label": parent_label or parent_name,
                "create_if_has_children": create_if_has_children,
                "matched_jobs": [],
                "existing_parent": None,
            }
        parent_info[parent_name]["prefixes"].append(prefix)
    
    # Single pass through all nodes
    remaining_nodes = []
    for node in nodes:
        job_name = node.job_name
        core_name = node.core_job_name
        short_name = node.short_job_name
        
        matched = False
        
        # Check if this node is a parent for any of our grouping rules
        for parent_name, info in parent_info.items():
            if job_name == parent_name or core_name == parent_name or short_name == parent_name:
                info["existing_parent"] = node
                remaining_nodes.append(node)  # Keep parent at root level
                matched = True
                break
        
        if matched:
            continue
        
        # Check if this node matches any prefix
        for parent_name, info in parent_info.items():
            for prefix in info["prefixes"]:
                if job_name.startswith(prefix) or core_name.startswith(prefix) or short_name.startswith(prefix):
                    info["matched_jobs"].append(node)
                    matched = True
                    break
            if matched:
                break
        
        if not matched:
            remaining_nodes.append(node)
    
    # Now process each parent and add matched jobs
    for parent_name, info in parent_info.items():
        matched_jobs = info["matched_jobs"]
        
        if not matched_jobs:
            continue  # No matches for this parent
        
        if info["create_if_has_children"] and not matched_jobs:
            continue  # Skip parent creation if no children
        
        existing_parent = info["existing_parent"]
        
        # Check if parent is synthetic
        is_synthetic = parent_name not in _workflow_job_to_file and parent_name not in _workflow_job_name_to_id.values()
        
        # Determine styling
        label_color = "#57606a" if is_synthetic else "#0969da"
        label_weight = "font-weight: 600;" if is_synthetic else ""
        
        if existing_parent:
            # Add to existing parent
            existing_parent.children.extend(matched_jobs)
            logger.info(f"[move_jobs_by_prefix_batch_pass] Added {len(matched_jobs)} jobs to existing parent '{parent_name}'")
        else:
            # Create new parent
            parent_label = info["label"]
            parent_node = TreeNodeVM(
                node_key=f"parent:{parent_name}",
                label_html=f'<span style="color: {label_color}; {label_weight}">{html.escape(parent_label)}</span>',
                children=matched_jobs,
                job_name=parent_name,
                core_job_name=parent_name,
                short_job_name=parent_name,
                collapsible=True,
                default_expanded=False,
            )
            remaining_nodes.append(parent_node)
            logger.info(f"[move_jobs_by_prefix_batch_pass] Created new parent '{parent_name}' with {len(matched_jobs)} jobs")
    
    logger.info(f"[move_jobs_by_prefix_batch_pass] Returning {len(remaining_nodes)} root nodes")
    return remaining_nodes


def move_jobs_by_prefix_pass(
    nodes: List[TreeNodeVM],
    prefix: str,
    parent_name: str,
    parent_label: str = "",
    create_if_has_children: bool = False,
) -> List[TreeNodeVM]:
    """
    Move jobs matching a prefix under a parent node.
    
    This pass finds all root-level nodes whose job_name starts with the given prefix,
    removes them from the root level, and either creates a new parent node or adds them
    to an existing parent node with the same name.
    
    Args:
        nodes: List of TreeNodeVM nodes (root level)
        prefix: Job name prefix to match (e.g., "deploy-", "build-test")
        parent_name: Name for the parent node (e.g., "deploy", "dynamo-status-check")
        parent_label: Display label for the parent node (default: same as parent_name)
        create_if_has_children: Only create parent if matching children exist (default: False, always group if matches)
    
    Returns:
        List of TreeNodeVM nodes with matching jobs moved under a parent
    """
    # Default parent_label to parent_name if not specified
    if not parent_label:
        parent_label = parent_name
    
    logger.info(f"[move_jobs_by_prefix_pass] Grouping {prefix}* jobs (processing {len(nodes)} root nodes)")
    
    # Separate matching jobs from other jobs, and check if parent already exists
    matching_jobs = []
    other_jobs = []
    existing_parent = None
    
    for node in nodes:
        job_name = node.job_name
        core_name = node.core_job_name
        short_name = node.short_job_name
        
        if job_name == parent_name or core_name == parent_name or short_name == parent_name:
            # Found existing parent node - we'll add to it
            existing_parent = node
            logger.info(f"[move_jobs_by_prefix_pass] Found existing parent node '{parent_name}'")
        elif job_name.startswith(prefix) or core_name.startswith(prefix) or short_name.startswith(prefix):
            # Match by job_name, core_job_name, or short_job_name prefix
            matching_jobs.append(node)
            logger.debug(f"[move_jobs_by_prefix_pass] Matched job '{short_name or core_name or job_name}' with prefix '{prefix}'")
        else:
            other_jobs.append(node)
    
    if not matching_jobs:
        logger.info(f"[move_jobs_by_prefix_pass] No {prefix}* jobs found, returning nodes unchanged")
        return nodes
    
    # If create_if_has_children is True and there are no matching jobs, don't create parent
    if create_if_has_children and not matching_jobs:
        logger.info(f"[move_jobs_by_prefix_pass] create_if_has_children=True but no matching jobs found, skipping parent creation")
        return nodes
    
    logger.info(f"[move_jobs_by_prefix_pass] Found {len(matching_jobs)} {prefix}* jobs to group")
    
    # Check if existing parent is required (check for [REQUIRED] in label_html)
    is_parent_required = False
    if existing_parent:
        is_parent_required = '[REQUIRED]' in str(existing_parent.label_html or '')
    
    # Check if the parent job is synthetic (not defined in YAML)
    # Real jobs are defined in YAML and should stay blue
    # Synthetic parents are created only for grouping and should be gray
    is_synthetic = True  # Default to synthetic
    
    # Check if parent_name exists in YAML either as job_id or job_name
    # _workflow_job_to_file: maps job_name -> file
    # _workflow_job_name_to_id: maps job_name -> job_id
    # So we need to check: parent_name in job_to_file OR parent_name in name_to_id.values()
    if parent_name in _workflow_job_to_file:
        # Found as a job name
        is_synthetic = False
    elif parent_name in _workflow_job_name_to_id.values():
        # Found as a job ID
        is_synthetic = False
    
    if existing_parent:
        if not is_synthetic:
            logger.debug(f"[move_jobs_by_prefix_pass] Parent '{parent_name}' is REAL (found in YAML)")
        else:
            logger.debug(f"[move_jobs_by_prefix_pass] Parent '{parent_name}' is SYNTHETIC (not in YAML)")
    else:
        if not is_synthetic:
            logger.debug(f"[move_jobs_by_prefix_pass] Creating parent '{parent_name}' as REAL (found in YAML)")
        else:
            logger.debug(f"[move_jobs_by_prefix_pass] Creating parent '{parent_name}' as SYNTHETIC (not in YAML)")
    
    # Generate label with short name and long description (if different)
    # Use bold gray for synthetic nodes, blue for real nodes
    label_color = "#57606a" if is_synthetic else "#0969da"
    label_weight = "font-weight: 600;" if is_synthetic else ""
    logger.debug(f"[move_jobs_by_prefix_pass] Parent '{parent_name}' color={label_color}, weight={label_weight}")
    
    if parent_name != parent_label:
        # Format: short_name "long description"
        parent_label_html = (
            f'<span style="color: {label_color}; {label_weight}">{parent_name}</span>'
            f'<span style="color: #57606a; font-family: SFMono-Regular, Consolas, Liberation Mono, Menlo, monospace; font-size: 12px;"> "{parent_label}"</span>'
        )
    else:
        # Just the label
        parent_label_html = f'<span style="color: {label_color}; {label_weight}">{parent_label}</span>'
    
    if existing_parent:
        # Add matching jobs to existing parent's children
        existing_children = list(existing_parent.children or [])
        all_children = existing_children + matching_jobs
        
        # DO NOT recompute parent status - preserve the original status from the API/existing node
        # The parent node has its own intrinsic status that should not change
        
        # Use dataclasses.replace to preserve ALL fields from existing parent (including label_html with status icon)
        from dataclasses import replace
        updated_parent = replace(
            existing_parent,
            children=all_children,
        )
        result = other_jobs + [updated_parent]
        logger.info(f"[move_jobs_by_prefix_pass] Added {len(matching_jobs)} jobs to existing parent '{parent_name}' (preserving original status)")
    else:
        # Create a new synthetic parent node (not from API)
        # Use "unknown" status since synthetic parents don't have their own status
        parent_status = "unknown"
        
        # Hardcode known required parent jobs (these are blocking jobs defined in YAML)
        KNOWN_REQUIRED_PARENTS = {"backend-status-check", "dynamo-status-check"}
        is_parent_required = parent_name in KNOWN_REQUIRED_PARENTS
        
        parent_icon = status_icon_html(status_norm=parent_status, is_required=is_parent_required)
        req_badge = required_badge_html(is_required=is_parent_required, status_norm=parent_status) if is_parent_required else ''
        
        # Create a new parent node
        parent_node = TreeNodeVM(
            node_key=f"stage:{parent_name}",
            label_html=f'{parent_icon}{req_badge}{parent_label_html}',
            children=matching_jobs,
            collapsible=True,
            default_expanded=False,
            triangle_tooltip=f"{parent_name} jobs",
            job_name=parent_name,
            short_job_name=parent_name,
        )
        result = other_jobs + [parent_node]
        logger.info(f"[move_jobs_by_prefix_pass] Created new synthetic parent '{parent_name}' with {len(matching_jobs)} jobs (status=unknown, required={is_parent_required})")
    
    # Return result
    logger.info(f"[move_jobs_by_prefix_pass] Grouping complete, returning {len(result)} root nodes")
    return result


def sort_nodes_by_name_pass(nodes: List[TreeNodeVM]) -> List[TreeNodeVM]:
    """
    Sort nodes alphabetically by job name (recursively).
    
    Sorts nodes at each level by their job_name or label_html for consistent display.
    Special nodes (like status-check jobs) can be preserved at specific positions if needed.
    
    Args:
        nodes: List of TreeNodeVM nodes to sort
        
    Returns:
        Sorted list of TreeNodeVM nodes (children are also recursively sorted)
    """
    def _is_build_test_tree_root(node: TreeNodeVM) -> bool:
        """Return True if this node's children ordering must be preserved.

        Policy:
        - For build-test jobs, preserve the natural order:
          - Actions steps order (as emitted by GitHub)
          - Pytest per-test timing order (as emitted by pytest/log)
        """
        # The YAML "short job name" is the most reliable discriminator when present.
        sj = str(node.short_job_name or "").strip().lower()
        jn = str(node.job_name or "").strip().lower()
        cn = str(node.core_job_name or "").strip().lower()
        # Keep "Build and Test - dynamo" in natural order too.
        if jn == "build and test - dynamo":
            return True
        # Typical cases:
        # - short_job_name: "build-test"
        # - core/job name: "...-build-test"
        if sj == "build-test" or ("build-test" in sj):
            return True
        if ("-build-test" in jn) or ("build-test" in jn):
            return True
        if ("-build-test" in cn) or ("build-test" in cn):
            return True
        return False

    def sort_key(node: TreeNodeVM) -> tuple:
        """Generate sort key for a node.
        
        Returns:
            Tuple of (priority, name) where priority determines order:
            - 0: Regular jobs (sorted alphabetically)
        """
        job_name = str(node.job_name or "")
        label_html = str(node.label_html or "")
        
        # Extract plain text from label_html for sorting (includes arch prefix like "[x86_64]")
        # This ensures jobs are sorted by their display name, not their internal job_name
        if label_html:
            # Strip HTML tags to get the plain text
            import re
            text = re.sub(r'<[^>]+>', '', label_html)
            # Decode HTML entities
            import html as html_module
            text = html_module.unescape(text)
            name = text.strip()
        else:
            # Fallback to job_name if no label_html
            name = job_name if job_name else ""
        
        return (0, name.lower())
    
    # Sort current level
    sorted_nodes = sorted(nodes, key=sort_key)
    
    # Recursively sort children
    result = []
    for node in sorted_nodes:
        children = list(node.children or [])
        if children:
            # Do NOT sort children under build-test jobs; preserve original order (steps/tests).
            if _is_build_test_tree_root(node):
                sorted_children = children
            else:
                sorted_children = sort_nodes_by_name_pass(children)
            # Create new node with sorted children
            result.append(TreeNodeVM(
                node_key=node.node_key,
                label_html=node.label_html,
                children=sorted_children,
                collapsible=node.collapsible,
                default_expanded=node.default_expanded,
                triangle_tooltip=node.triangle_tooltip,
                noncollapsible_icon=node.noncollapsible_icon,
                skip_dedup=node.skip_dedup,
                job_name=node.job_name,
                core_job_name=node.core_job_name,
                short_job_name=node.short_job_name,
                workflow_name=node.workflow_name,
                variant=node.variant,
                pr_number=node.pr_number,
                raw_html_content=node.raw_html_content,
            ))
        else:
            result.append(node)
    
    return result


def expand_required_failure_descendants_pass(nodes: List[TreeNodeVM]) -> List[TreeNodeVM]:
    """
    Expand any node that has a REQUIRED failure anywhere in its descendant subtree.

    This is intentionally a *post-pass* so it can run after any logic that mutates/moves nodes
    (e.g. workflow `jobs.*.needs` grouping).

    Policy:
    - expand for required failures only (not optional failures)
    - only affects nodes that have children (expanding leaves is meaningless)
    """
    # Use the canonical icon HTML (via status_icon_html) to avoid brittle substring heuristics.
    _ICON_FAIL_REQ = status_icon_html(status_norm="failure", is_required=True)

    def walk(n: TreeNodeVM) -> Tuple[TreeNodeVM, bool]:
        # returns: (new_node, has_required_failure_in_subtree)
        new_children: List[TreeNodeVM] = []
        child_req = False
        for ch in (n.children or []):
            ch2, r = walk(ch)
            new_children.append(ch2)
            child_req = child_req or r

        own_req_fail = bool(_ICON_FAIL_REQ in str(n.label_html or ""))
        has_req = bool(own_req_fail or child_req)

        # Expand this node if it has children and any descendant required-failed.
        new_default_expanded = bool(n.default_expanded)
        if bool(new_children) and bool(has_req):
            new_default_expanded = True

        return (
            TreeNodeVM(
                node_key=str(n.node_key or ""),
                label_html=str(n.label_html or ""),
                children=new_children,
                collapsible=bool(n.collapsible),
                default_expanded=bool(new_default_expanded),
                triangle_tooltip=n.triangle_tooltip,
                noncollapsible_icon=n.noncollapsible_icon,
                job_name=n.job_name,
                core_job_name=n.core_job_name,
                short_job_name=n.short_job_name,
                workflow_name=n.workflow_name,
                variant=n.variant,
                pr_number=n.pr_number,
                raw_html_content=n.raw_html_content,
            ),
            has_req,
        )

    out: List[TreeNodeVM] = []
    for n in (nodes or []):
            n2, _ = walk(n)
            out.append(n2)
    return out


def move_required_jobs_to_top_pass(nodes: List[TreeNodeVM]) -> List[TreeNodeVM]:
    """
    Move all REQUIRED jobs to the top, keeping alphabetical order within each group.
    
    This pass separates required and non-required jobs at the ROOT level only,
    preserving all parent-child relationships. A job is considered required if
    its label_html contains the '[REQUIRED]' badge (which comes from the is_required
    attribute of the original CIJobNode).
    
    Args:
        nodes: List of TreeNodeVM nodes (root level)
        
    Returns:
        List with required jobs at top, then non-required jobs, each group alphabetically sorted
    """
    logger.info(f"[move_required_jobs_to_top] Processing {len(nodes)} root nodes")
    
    required_jobs = []
    non_required_jobs = []
    
    for node in nodes:
        # Check if the node's label contains [REQUIRED] badge
        # This badge is added by check_line_html when is_required=True
        if '[REQUIRED]' in str(node.label_html or ''):
            required_jobs.append(node)
            logger.debug(f"[move_required_jobs_to_top] REQUIRED: {node.short_job_name or node.job_name}")
        else:
            non_required_jobs.append(node)
    
    # Sort each group alphabetically by short_job_name or job_name
    def sort_key(node):
        name = node.short_job_name or node.job_name or node.label_html
        return str(name).lower()
    
    required_jobs.sort(key=sort_key)
    non_required_jobs.sort(key=sort_key)
    
    result = required_jobs + non_required_jobs
    
    logger.info(f"[move_required_jobs_to_top] Moved {len(required_jobs)} required jobs to top, {len(non_required_jobs)} non-required after")
    if required_jobs:
        logger.info(f"[move_required_jobs_to_top] Required jobs at top: {[node.short_job_name or node.job_name for node in required_jobs]}")
    
    return result


def verify_tree_structure_pass(tree_nodes: List[TreeNodeVM], original_ci_nodes: List, commit_sha: str = "") -> None:
    """
    Verify the final tree structure for common issues.
    
    This pass checks for:
    - Duplicate short names
    - Missing short names for important jobs
    - Minimum number of required jobs
    
    Args:
        tree_nodes: Final tree structure to verify
        original_ci_nodes: Original CI nodes to check augmentation
        commit_sha: Commit SHA for context in error messages
    """
    from common_branch_nodes import CIJobNode
    
    # Format commit ref for logging
    commit_ref = f" (commit: {commit_sha[:7]})" if commit_sha else ""
    logger.info(f"[verify_tree_structure_pass] Verifying tree structure ({len(tree_nodes)} root nodes){commit_ref}")
    
    # Collect all nodes (including nested)
    all_nodes = []
    def collect_nodes(nodes):
        for node in nodes:
            all_nodes.append(node)
            if node.children:
                collect_nodes(node.children)
    collect_nodes(tree_nodes)
    
    # Check 1: Count required jobs and verify critical required jobs
    required_count = 0
    required_jobs_found = set()
    for node in all_nodes:
        # Check in label_html for REQUIRED badge
        if '[REQUIRED]' in str(node.label_html or ''):
            required_count += 1
            job_name = node.short_job_name or node.job_name or ''
            if job_name:
                required_jobs_found.add(job_name)
    
    # Check for critical required jobs
    CRITICAL_REQUIRED_JOBS = {"backend-status-check", "dynamo-status-check"}
    missing_critical = CRITICAL_REQUIRED_JOBS - required_jobs_found

    if missing_critical:
        logger.error(f"[verify_tree_structure_pass] ❌ CRITICAL: Missing required jobs: {missing_critical}{commit_ref}")
        logger.debug(f"[verify_tree_structure_pass] Required jobs found: {required_jobs_found}{commit_ref}")
        logger.debug(f"[verify_tree_structure_pass] All node names: {[node.job_name for node in all_nodes[:20]]}{commit_ref}")
    # Success case: don't log to avoid verbose output for every commit (200 commits × verbose logging = massive slowdown)
    
    if required_count < 2:
        logger.warning(f"[verify_tree_structure_pass] ⚠️  Only {required_count} required jobs found (expected at least 2: backend-status-check, dynamo-status-check){commit_ref}")
    # Success case: don't log to avoid verbose output
    
    # Check 2: Verify short names were set for original CI nodes
    # Check 2: Count nodes with short names (no warnings about missing ones)
    ci_nodes_with_short_names = 0
    
    for node in original_ci_nodes:
        if isinstance(node, CIJobNode):
            if hasattr(node, 'short_job_name') and node.short_job_name:
                ci_nodes_with_short_names += 1
    
    logger.debug(f"[verify_tree_structure_pass] {ci_nodes_with_short_names} CI nodes have short names")
    
    # Check 3: Look for duplicate short names
    short_name_counts = {}
    for node in original_ci_nodes:
        if isinstance(node, CIJobNode) and hasattr(node, 'short_job_name') and node.short_job_name:
            short_name = node.short_job_name
            # Use a stable underlying ID for duplicate detection.
            #
            # CIJobNode.job_id is a *display* id (often the check name, sometimes prefixed with workflow),
            # and can collide across reruns. Prefer the Actions job id extracted from the log URL.
            log_url = str(node.log_url or "")
            actions_job_id = str(node.actions_job_id or "").strip()
            actions_job_id = actions_job_id or (extract_actions_job_id_from_url(log_url) if log_url else "")
            stable_id = str(actions_job_id or log_url or node.job_id or "")
            core_name = str(node.core_job_name or "")
            if short_name not in short_name_counts:
                short_name_counts[short_name] = []
            short_name_counts[short_name].append((stable_id, core_name))
    
    duplicates = {k: v for k, v in short_name_counts.items() if len(v) > 1}
    if duplicates:
        # Only warn about duplicates where *all* entries share the same job_id.
        # (I.e., not "some repeats" like [a,a,b], but truly the same underlying ID everywhere.)
        same_id_duplicates: List[Tuple[str, str, List[str]]] = []

        for short_name, job_list in duplicates.items():
            job_ids = [str(job_id or "") for job_id, _ in job_list]
            uniq = sorted(set(job_ids))
            if len(uniq) != 1:
                continue
            core_names = [str(core_name or "") for _, core_name in job_list]
            same_id_duplicates.append((short_name, uniq[0], core_names))

        if same_id_duplicates:
            logger.warning(
                f"[verify_tree_structure_pass] ⚠️  Found {len(same_id_duplicates)} duplicate short names with SAME job_ids:{commit_ref}"
            )
            for short_name, job_id, core_names in same_id_duplicates[:10]:
                logger.warning(
                    f"[verify_tree_structure_pass]    - '{short_name}' job_id='{job_id}' used by: {core_names}{commit_ref}"
                )
            if len(same_id_duplicates) > 10:
                logger.warning(
                    f"[verify_tree_structure_pass]    ... and {len(same_id_duplicates) - 10} more duplicates{commit_ref}"
                )
        # Success cases: don't log to avoid verbose output
    # Success case: don't log to avoid verbose output
    
    # Check 4: Verify specific important jobs have short names
    important_jobs = ["Build and Test - dynamo", "dynamo-status-check", "backend-status-check"]
    for important_job in important_jobs:
        found = False
        for node in original_ci_nodes:
            if isinstance(node, CIJobNode):
                core_name = str(node.core_job_name or "")
                job_id = str(node.job_id or "")
                display_name = str(node.display_name or "")
                short_name = str(node.short_job_name or "")
                
                if important_job in core_name or important_job in job_id or important_job in display_name:
                    if short_name:
                        # Success case: don't log to avoid verbose output
                        found = True
                        break
                    # Missing short name: don't log for every commit to avoid verbose output
        if not found:
            # Synthetic nodes like dynamo-status-check won't be in original_ci_nodes
            # Don't log to avoid verbose output for every commit
            pass
    
    # Check 5: Verify all REQUIRED jobs are at the top (root level)
    first_non_required_idx = None
    required_after_non_required = []
    
    for i, node in enumerate(tree_nodes):
        is_required = '[REQUIRED]' in str(node.label_html or '')
        
        if not is_required and first_non_required_idx is None:
            first_non_required_idx = i
        
        if is_required and first_non_required_idx is not None:
            # Found a REQUIRED job after a non-required job
            job_name = node.short_job_name or node.job_name or str(node.label_html)[:50]
            required_after_non_required.append((i, job_name))
    
    if required_after_non_required:
        logger.warning(f"[verify_tree_structure_pass] ⚠️  {len(required_after_non_required)} REQUIRED jobs found AFTER non-required jobs:")
        for idx, job_name in required_after_non_required[:5]:
            logger.warning(f"[verify_tree_structure_pass]    - Position {idx}: '{job_name}'")
        if len(required_after_non_required) > 5:
            logger.warning(f"[verify_tree_structure_pass]    ... and {len(required_after_non_required) - 5} more")
        logger.warning(f"[verify_tree_structure_pass] ⚠️  move_required_jobs_to_top_pass may not be working correctly!")
    else:
        logger.info(f"[verify_tree_structure_pass] ✓ All REQUIRED jobs are at the top (first {first_non_required_idx or len(tree_nodes)} positions)")
    
    # Check 6: Verify build-test is under dynamo-status-check
    dynamo_node = None
    for node in tree_nodes:
        if 'dynamo-status-check' in str(node.short_job_name or node.job_name or '').lower():
            dynamo_node = node
            break
    
    if dynamo_node:
        has_build_test = False
        for child in (dynamo_node.children or []):
            # Check both short_job_name and job_name
            child_short = str(child.short_job_name or '').lower()
            child_full = str(child.job_name or '').lower()
            if 'build-test' in child_short or 'build and test' in child_full:
                has_build_test = True
                break
        
        if has_build_test:
            logger.info(f"[verify_tree_structure_pass] ✓ build-test is under dynamo-status-check")
        else:
            child_names = [(c.short_job_name or '', c.job_name or '') for c in (dynamo_node.children or [])]
            logger.warning(f"[verify_tree_structure_pass] ⚠️  build-test NOT found under dynamo-status-check. Children: {child_names}")
    else:
        logger.warning(f"[verify_tree_structure_pass] ⚠️  dynamo-status-check node not found")
    
    logger.info(f"[verify_tree_structure_pass] Verification complete")


def verify_job_details_pass(ci_nodes: List, commit_sha: str = "") -> None:
    """
    Verify that CI jobs have proper step data, pytest listings, and timing information.

    This pass checks for:
    - Build-test jobs should have step children (job details from GitHub Actions API)
    - Test steps should have pytest test children (parsed from raw logs)
    - All jobs should have duration/timing data

    Args:
        ci_nodes: List of BranchNode objects (CIJobNode instances)
        commit_sha: Commit SHA for context in error messages
    """
    from common_branch_nodes import CIJobNode, CIStepNode, CIPytestNode

    # Temporary exception list (expires ~2026-02-19)
    # These test steps are known to be missing pytest tests and warnings are suppressed
    PYTEST_WARNING_EXCEPTIONS = {
        "trtllm-build-test (cuda13.0, arm64) > Run tests",
        "Build and Test - dynamo > pytest (parallel)",
        "Build and Test - dynamo > pytest (serial)",
    }

    # Format commit ref for logging
    commit_ref = f" (commit: {commit_sha[:7]})" if commit_sha else ""
    logger.info(f"[verify_job_details_pass] Verifying job details for {len(ci_nodes)} nodes{commit_ref}")

    # Counters for reporting
    jobs_checked = 0
    real_github_jobs_checked = 0
    synthetic_aggregators_checked = 0
    jobs_missing_steps = []
    jobs_missing_duration = []
    test_steps_missing_pytest = []

    for node in ci_nodes:
        if not isinstance(node, CIJobNode):
            continue

        # Skip synthetic nodes (steps and pytest tests themselves)
        if node.is_synthetic:
            continue

        jobs_checked += 1
        job_name = node.display_name or node.job_id or ""

        # Check 1: Build-test jobs should have step children
        if job_name_wants_pytest_details(job_name):
            # Skip jobs with template variables (not real jobs yet)
            if "${{" in job_name:
                continue

            job_status = str(node.status or "").lower()
            # Skip cancelled jobs (they don't have step details)
            if job_status in ("cancelled", "canceled"):
                continue

            if not node.children or len(node.children) == 0:
                jobs_missing_steps.append(job_name)
            else:
                # Check 2: Test steps should have pytest test children
                for child in node.children:
                    if isinstance(child, CIStepNode):
                        # Use display_name which has the original step name (without augmentation)
                        step_name = child.display_name or child.job_id or ""
                        # Skip steps with template variables
                        if "${{" in step_name:
                            continue

                        if is_python_test_step(step_name):
                            # This is a test step, it should have pytest children
                            has_pytest_children = any(isinstance(c, CIPytestNode) for c in (child.children or []))
                            if not has_pytest_children:
                                test_steps_missing_pytest.append(f"{job_name} > {step_name}")

        # Check 3: GitHub Actions workflow jobs should have duration/timing data
        # GitHub Actions jobs have log_url that points to /actions/runs/.../job/...
        # GitHub App checks (DCO, CodeRabbit) and external CI (GitLab) don't have duration
        log_url = str(node.log_url or "").strip()
        actions_job_id = str(node.actions_job_id or "").strip()
        is_github_actions_job = "/actions/runs/" in log_url and "/job/" in log_url
        is_external_check = log_url and not is_github_actions_job  # GitHub App or external CI

        if is_github_actions_job:
            real_github_jobs_checked += 1
        elif is_external_check:
            synthetic_aggregators_checked += 1  # Count external checks separately
        elif not log_url and not actions_job_id:
            synthetic_aggregators_checked += 1  # True synthetic aggregators
        else:
            real_github_jobs_checked += 1  # Has actions_job_id but no URL

        duration = str(node.duration or "").strip()
        job_status = str(node.status or "").lower()

        if not duration and is_github_actions_job:
            # Skip jobs with template variables
            if "${{" in job_name:
                pass
            # Skip cancelled jobs (don't care about duration for cancelled jobs)
            elif job_status in ("cancelled", "canceled"):
                pass
            else:
                jobs_missing_duration.append(job_name)

    # Filter out exceptions from test_steps_missing_pytest
    test_steps_missing_pytest_filtered = [
        step for step in test_steps_missing_pytest
        if step not in PYTEST_WARNING_EXCEPTIONS
    ]

    # Summary report (only log if there are issues for this commit)
    has_issues = jobs_missing_steps or test_steps_missing_pytest_filtered or jobs_missing_duration

    if has_issues:
        # Group all issues under one commit header
        logger.warning(f"[verify_job_details_pass] ⚠️  Issues found for commit {commit_sha[:7] if commit_sha else 'unknown'}:")

        if jobs_missing_steps:
            # Show all on one line, comma-separated
            jobs_list = ", ".join(jobs_missing_steps)
            logger.warning(
                f"[verify_job_details_pass]    {len(jobs_missing_steps)} build-test jobs missing steps (out of {jobs_checked} checked): {jobs_list}"
            )

        if test_steps_missing_pytest_filtered:
            # Show all on one line, comma-separated
            steps_list = ", ".join(test_steps_missing_pytest_filtered)
            logger.warning(
                f"[verify_job_details_pass]    {len(test_steps_missing_pytest_filtered)} test steps missing pytest tests: {steps_list}"
            )

        if jobs_missing_duration:
            # Show all on one line, comma-separated
            duration_list = ", ".join(jobs_missing_duration)
            logger.warning(
                f"[verify_job_details_pass]    {len(jobs_missing_duration)} jobs missing duration (out of {real_github_jobs_checked} real jobs checked): {duration_list}"
            )

    # Only log success if there are no issues
    if not jobs_missing_steps and not test_steps_missing_pytest and not jobs_missing_duration:
        logger.info(f"[verify_job_details_pass] ✓ All {real_github_jobs_checked} real GitHub jobs have proper details (plus {synthetic_aggregators_checked} synthetic aggregators){commit_ref}")

    logger.info(f"[verify_job_details_pass] Verification complete: {real_github_jobs_checked} real jobs, {synthetic_aggregators_checked} synthetic aggregators")


def _compute_parent_status_from_children(children: List[TreeNodeVM]) -> str:
    """Compute parent node status based on children statuses.
    
    Rules:
    - If any child is 'failure', parent is 'failure'
    - If all children are 'success', parent is 'success'
    - Otherwise, parent status is 'unknown' or the most common status
    
    Args:
        children: List of child TreeNodeVM nodes
    
    Returns:
        Status string: 'success', 'failure', 'pending', 'in_progress', or 'unknown'
    """
    if not children:
        return "unknown"
    
    # Extract statuses from children's label_html (contains status icons)
    has_failure = False
    has_running = False
    has_pending = False
    success_count = 0
    
    for child in children:
        label = str(child.label_html or "")
        # Check for status indicators in the label HTML. Prefer SVG class names emitted by
        # `status_icon_html`, but keep a few keyword fallbacks for robustness.
        label_l = label.lower()
        if "octicon-x-circle-fill" in label_l or "octicon-x" in label_l or "failure" in label_l or "#c83a3a" in label_l:
            has_failure = True
        elif "octicon-check-circle-fill" in label_l or "octicon-check" in label_l or "success" in label_l:
            success_count += 1
        elif "octicon-clock" in label_l or "running" in label_l or "progress" in label_l or "in_progress" in label_l:
            has_running = True
        elif "octicon-dot-circle-fill" in label_l or "pending" in label_l:
            has_pending = True
    
    # Apply rules
    if has_failure:
        return "failure"
    elif success_count == len(children):
        return "success"
    elif has_running:
        return "in_progress"  # Return 'in_progress' to match CIStatus.IN_PROGRESS
    elif has_pending:
        return "pending"
    else:
        return "unknown"


def _hash10(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


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
            
            # Generate unique URL key using hash of full path
            import hashlib
            full_key_hash = hashlib.sha256(full_key.encode()).hexdigest()[:7]
            
            url_key_attr = ""
            if repo and sha7:
                url_key_attr = f' data-url-key="t.{html.escape(repo)}.{html.escape(full_key_hash)}"'
            elif sha7:
                url_key_attr = f' data-url-key="t.{html.escape(full_key_hash)}"'
            else:
                # Fallback: use just the hash
                url_key_attr = f' data-url-key="t.{html.escape(full_key_hash)}"'
            
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


def _parse_iso_utc(s: str) -> Optional[datetime]:
    x = str(s or "").strip()
    if not x:
        return None
    if x.endswith("Z"):
        return datetime.fromisoformat(x[:-1] + "+00:00")
    return datetime.fromisoformat(x)


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


def build_and_test_dynamo_phases_from_actions_job(job: Dict[str, object]) -> List[Tuple[str, str, str]]:
    """Extract phase rows from the Actions job `steps` array.

    Returns (phase_name, duration_str, status_norm).
    """
    steps = job.get("steps") if isinstance(job, dict) else None
    if not isinstance(steps, list) or not steps:
        return []

    def _dur(st: Dict[str, object]) -> str:
        a = _parse_iso_utc(str(st.get("started_at", "") or ""))
        b = _parse_iso_utc(str(st.get("completed_at", "") or ""))
        if not a or not b:
            return ""
        return _format_duration_short((b - a).total_seconds())

    out: List[Tuple[str, str, str]] = []

    # We match the canonical step names used in the workflow, but keep it fuzzy.
    for st in steps:
        if not isinstance(st, dict):
            continue
        nm = str(st.get("name", "") or "")
        nm_lc = nm.lower()
        status_norm = _status_norm_from_actions_step(
            status=str(st.get("status", "") or ""),
            conclusion=str(st.get("conclusion", "") or ""),
        )
        dur = _dur(st)

        if "build image" in nm_lc:
            out.append(("Build Image", dur, status_norm))
        elif "rust" in nm_lc and "check" in nm_lc:
            out.append(("Rust checks", dur, status_norm))
        elif "pytest" in nm_lc and ("parallel" in nm_lc or "xdist" in nm_lc):
            out.append(("pytest (parallel)", dur, status_norm))
        elif nm_lc.startswith("run pytest") or (("pytest" in nm_lc) and ("parallel" not in nm_lc) and ("xdist" not in nm_lc)):
            # If the workflow has an explicit non-parallel pytest step, call it serial.
            out.append(("pytest (serial)", dur, status_norm))

    # De-dup while keeping order (some jobs echo repeated step names via composites).
    seen = set()
    uniq: List[Tuple[str, str, str]] = []
    for ph in out:
        k = ph[0]
        if k in seen:
            continue
        seen.add(k)
        uniq.append(ph)
    return uniq


def actions_job_steps_over_threshold_from_actions_job(
    job: Dict[str, object], *, min_seconds: float = 30.0
) -> List[Tuple[str, str, str]]:
    """Return (step_name, duration_str, status_norm) for steps we want to display.

    Policy:
    - show steps with duration >= min_seconds
    - always show failing steps (even if < min_seconds or duration is missing)
    """
    steps = job.get("steps") if isinstance(job, dict) else None
    if not isinstance(steps, list) or not steps:
        return []

    out: List[Tuple[str, str, str]] = []
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
        out.append((nm, dur_s, status_norm))

    # De-dup while keeping order (some composite actions can repeat step names).
    seen = set()
    uniq: List[Tuple[str, str, str]] = []
    for (nm, dur, st) in out:
        if nm in seen:
            continue
        seen.add(nm)
        uniq.append((nm, dur, st))
    return uniq


def actions_job_step_tuples(
    *,
    github_api: Optional["GitHubAPIClient"],
    job_url: str,
    min_seconds: float = 10.0,
    ttl_s: int = 30 * 24 * 3600,
) -> List[Tuple[str, str, str]]:
    """Fetch job details (cached) and return long-running steps (duration >= min_seconds)."""
    import time  # For timing analysis
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
    return (
        ("run" in step_lower and "test" in step_lower)
        or "pytest" in step_lower
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
    github_api: Optional["GitHubAPIClient"],
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
    import time  # For detailed timing
    t_start = time.monotonic()
    
    nm = str(job_name or "").strip()
    if not nm:
        return []

    # Shared policy: which jobs get full steps + pytest per-test breakdown.
    is_build_test = job_name_wants_pytest_details(nm)

    # Special case: "Build and Test - dynamo" has custom phase extraction
    if nm == "Build and Test - dynamo":
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

        extra_steps = [(n, d, st) for (n, d, st) in (steps or []) if not _covered_by_phase(n)]

        out = [(p[0], p[1], p[2]) for p in (phases or [])]
        out.extend([(s[0], s[1], s[2]) for s in extra_steps])

        # Apply pytest test extraction to "Run tests" steps and "test: pytest" phases (same as other build-test jobs)
        result = []
        t_pytest_total = 0.0
        for step_name, step_dur, step_status in out:
            result.append((step_name, step_dur, step_status))

            # If this is a "Run tests" step or a "test: pytest" phase, parse pytest slowest tests from raw log
            if is_python_test_step(step_name) and raw_log_path:
                t0 = time.monotonic()
                pytest_tests = pytest_slowest_tests_from_raw_log(
                    raw_log_path=raw_log_path,
                    # Tests: list *all* per-test timings in order (do not filter).
                    min_seconds=0.0,
                    include_all=True,
                    step_name=step_name,
                )
                t_pytest_total += time.monotonic() - t0
                # Add pytest tests as indented entries
                for test_name, test_dur, test_status in pytest_tests:
                    result.append((f"  └─ {test_name}", test_dur, test_status))

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
        for step_name, step_dur, step_status in (steps or []):
            result.append((step_name, step_dur, step_status))

            # If this is a "Run tests" step, parse pytest slowest tests from raw log
            if is_python_test_step(step_name) and raw_log_path:
                t0 = time.monotonic()
                pytest_tests = pytest_slowest_tests_from_raw_log(
                    raw_log_path=raw_log_path,
                    # Tests: list *all* per-test timings in order (do not filter).
                    min_seconds=0.0,
                    include_all=True,
                    step_name=step_name,
                )
                t_pytest_total += time.monotonic() - t0
                # Add pytest tests as indented/prefixed entries
                for test_name, test_dur, test_status in pytest_tests:
                    # Prefix with indentation to show hierarchy
                    result.append((f"  └─ {test_name}", test_dur, test_status))

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


def pytest_slowest_tests_from_raw_log(
    *,
    raw_log_path: Optional[Path],
    min_seconds: float = 10.0,
    include_all: bool = False,
    step_name: str = "",
) -> List[Tuple[str, str, str]]:
    """Parse pytest per-test durations from cached raw log file.
    
    This is based on pytest's "slowest N durations" section (`--durations=N`).
    If CI is configured with `--durations=0`, this section contains all tests.
    
    Args:
        raw_log_path: Path to the raw log file
        min_seconds: Minimum duration to include (default: 10s). Ignored when include_all=True.
        include_all: If True, include all entries in the durations section regardless of duration threshold.
    
    Returns:
        List of (test_name, duration_str, status_norm) tuples, in the same order as the log section.
        
    Example output format:
        [
            ("[call] tests/serve/test_vllm.py::test_serve_deployment[agg]", "1m 43s", "success"),
            ("[setup] tests/kvbm_integration/test_kvbm.py::test_offload_and_onboard[llm_server_kvbm0]", "1m 50s", "failure"),
            ...
        ]
    """
    if not raw_log_path:
        return []

    # Parsed pytest timings cache (disk-backed). Cache boundary is JSON-on-disk; in-memory uses dataclasses.
    from cache_pytest_timings import PYTEST_TIMINGS_CACHE  # local file import

    cached = PYTEST_TIMINGS_CACHE.get_if_fresh(raw_log_path=Path(raw_log_path), step_name=step_name)
    if cached is not None:
        return list(cached)
    
    try:
        t0_parse = time.monotonic()
        p = Path(raw_log_path)
        if not p.exists() or not p.is_file():
            return []
        
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            log_text = f.read()
        
        lines = log_text.split('\n')

        # Build a map from test-id -> status using pytest summary lines.
        # Example lines:
        #   FAILED tests/foo.py::test_bar[param] - AssertionError: ...
        #   ERROR  tests/foo.py::test_baz - ...
        #   SKIPPED tests/foo.py::test_qux - ...
        #   XFAIL tests/foo.py::test_x - ...
        status_by_test: Dict[str, str] = {}

        def _norm_test_id(s: str) -> str:
            return str(s or "").strip()

        # Match status lines with optional timestamp prefix (GitHub Actions format)
        # Example: "2025-11-29T21:55:17.1891443Z FAILED tests/foo.py::test_bar - AssertionError: ..."
        summary_re = re.compile(r'^.*?\s(FAILED|ERROR|XPASS|XFAIL|SKIPPED|PASSED)\s+(.+?)(?:\s+-\s+.*)?\s*$')
        for ln in lines:
            msum = summary_re.match(str(ln or ""))
            if not msum:
                continue
            st_word = str(msum.group(1) or "").strip().upper()
            test_id = _norm_test_id(msum.group(2) or "")
            if not test_id:
                continue
            if st_word in {"FAILED", "ERROR", "XPASS"}:
                status_by_test[test_id] = CIStatus.FAILURE.value
            elif st_word in {"SKIPPED", "XFAIL"}:
                status_by_test[test_id] = CIStatus.SKIPPED.value
            elif st_word == "PASSED":
                status_by_test[test_id] = CIStatus.SUCCESS.value
        
        # Determine which test_type to filter for based on step_name
        # Examples: "Run unit tests" -> "unit", "Run e2e tests" -> "e2e"
        target_test_type = ""
        step_lower = str(step_name or "").lower()
        if "unit" in step_lower:
            target_test_type = "unit"
        elif "e2e" in step_lower:
            target_test_type = "e2e"

        # Look for the "slowest N durations" section
        # Format: "============================= slowest 10 durations ============================="
        # Followed by lines like (with GitHub Actions timestamp prefix):
        # "2026-01-15T22:01:23.5641223Z 110.16s setup    tests/kvbm_integration/test_kvbm.py::test_offload_and_onboard[llm_server_kvbm0]"
        # "2026-01-15T22:01:23.5641223Z 103.05s call     tests/serve/test_vllm.py::test_serve_deployment[agg-request-plane-tcp]"

        test_times: List[Tuple[str, str, str]] = []
        in_slowest_section = False
        current_test_type = ""  # Track which test_type section we're in
        threshold = 0.0 if bool(include_all) else float(min_seconds or 0.0)

        for line in lines:
            # Track which test_type section we're in by looking for pytest action markers
            # Example: "  test_type: unit" or "  test_type: e2e, gpu_1"
            # Match only "test_type:" at the start of a word (not "STR_TEST_TYPE:")
            type_match = re.search(r'\btest_type:\s*(\w+)', line, re.IGNORECASE)
            if type_match:
                current_test_type = type_match.group(1).strip().lower()
            # Start of slowest section
            if 'slowest' in line.lower() and 'duration' in line.lower() and '=====' in line:
                in_slowest_section = True
                continue

            # End of slowest section (next ===== line)
            # Don't break - there may be multiple "slowest durations" sections (multiple pytest runs)
            if in_slowest_section and '=====' in line:
                in_slowest_section = False
                continue

            if in_slowest_section:
                # Skip this section if we're filtering by test_type and it doesn't match
                if target_test_type and current_test_type != target_test_type:
                    continue
                # Parse line format (with GitHub Actions timestamp prefix):
                # "2026-01-15T22:01:23.5641223Z 110.16s setup    tests/..."
                # "2026-01-15T22:01:23.5641223Z 103.05s call     tests/..."
                # Skip timestamp prefix (anything before the duration)
                m = re.match(r'^.*?(\d+\.?\d*)s\s+(setup|call|teardown)\s+(.+)$', str(line or ""))
                if m:
                    duration = float(m.group(1))
                    phase = m.group(2)
                    test_id = str(m.group(3) or "").strip()
                    
                    # Filter by minimum duration unless include_all is set
                    if duration >= threshold:
                        # Format duration as "1m 50s" or "110s"
                        if duration >= 60:
                            mins = int(duration // 60)
                            secs = int(duration % 60)
                            dur_str = f"{mins}m {secs}s"
                        else:
                            dur_str = f"{int(duration)}s"
                        
                        # Include phase in the test name for clarity
                        full_name = f"[{phase}] {test_id}"
                        
                        # Determine status (best-effort) from summary lines; default to success.
                        status_norm = status_by_test.get(test_id, CIStatus.SUCCESS.value)
                        test_times.append((full_name, dur_str, status_norm))
        
        # Persist parsed rows (best-effort).
        from cache_pytest_timings import PYTEST_TIMINGS_CACHE  # local file import

        # Record parse timing on the concrete cache object.
        PYTEST_TIMINGS_CACHE.stats.parse_calls += 1
        PYTEST_TIMINGS_CACHE.stats.parse_secs += max(0.0, float(time.monotonic() - t0_parse))

        PYTEST_TIMINGS_CACHE.put(raw_log_path=p, step_name=step_name, rows=test_times)

        return test_times

    except Exception as e:
        logger.debug(f"Failed to parse pytest slowest durations from {raw_log_path}: {e}")
        return []


def step_window_snippet_from_cached_raw_log(
    *,
    job: Dict[str, object],
    step_name: str,
    raw_log_path: Optional[Path],
) -> str:
    """Extract an error snippet scoped to a specific Actions step time window (best-effort).

    We do not have per-step log URLs. Instead, we:
    - locate the step's started_at/completed_at from the cached job `steps[]`
    - slice the cached raw log by timestamp
    - run the common snippet extractor on the sliced text
    """
    if not raw_log_path:
        return ""
    p = Path(raw_log_path)
    if not p.exists() or not p.is_file():
        return ""
    step = None
    steps = job.get("steps") if isinstance(job, dict) else None
    if isinstance(steps, list):
        for st in steps:
            if isinstance(st, dict) and str(st.get("name", "") or "") == str(step_name or ""):
                step = st
                break
    if not isinstance(step, dict):
        return ""

    a = _parse_iso_utc(str(step.get("started_at", "") or ""))
    b = _parse_iso_utc(str(step.get("completed_at", "") or ""))
    if not a or not b:
        return ""

    # Shared library (dependency-light): `dynamo-utils/ci_log_errors/`
    from ci_log_errors import snippet as ci_snippet  # local import (avoid circulars)

    text = p.read_text(encoding="utf-8", errors="replace")

    # Filter lines by timestamp window (raw logs include ISO-8601 timestamps).
    kept: List[str] = []
    for ln in (text.splitlines() or []):
        # Most lines are prefixed with an ISO timestamp; ignore unparsable lines.
        ts = None
        # Heuristic: take the first token, strip any trailing 'Z'.
        head = (ln.split(" ", 1)[0] if " " in ln else ln).strip()
        ts = _parse_iso_utc(head)
        if not ts:
            continue
        if ts < a or ts > b:
            continue
        kept.append(ln)
    if not kept:
        return ""
    return ci_snippet.extract_error_snippet_from_text("\n".join(kept))

def required_badge_html(*, is_required: bool, status_norm: str) -> str:
    """Render a [REQUIRED] badge with shared semantics."""
    if not is_required:
        return ""

    s = (status_norm or "").strip().lower()
    if s == CIStatus.FAILURE:
        color = COLOR_RED
        weight = "400"
    elif s == CIStatus.SUCCESS:
        color = COLOR_GREEN
        weight = "400"
    else:
        color = "#57606a"
        weight = "400"

    return f' <span style="color: {color}; font-weight: {weight};">[REQUIRED]</span> '


def mandatory_badge_html(*, is_mandatory: bool, status_norm: str) -> str:
    """Render a [MANDATORY] badge (GitLab) following the same color convention as [REQUIRED]."""
    if not is_mandatory:
        return ""

    s = (status_norm or "").strip().lower()
    if s == CIStatus.FAILURE:
        color = COLOR_RED
        weight = "700"
    elif s == CIStatus.SUCCESS:
        color = COLOR_GREEN
        weight = "400"
    else:
        color = "#57606a"
        weight = "400"

    return f' <span style="color: {color}; font-weight: {weight};">[MANDATORY]</span>'


def status_icon_html(
    *,
    status_norm: str,
    is_required: bool,
    required_failure: bool = False,
    warning_present: bool = False,
    icon_px: int = 12,
) -> str:
    """Shared status icon HTML (match all dashboards).

    Conventions:
    - REQUIRED success: green filled circle-check
    - REQUIRED failure: red filled circle X
    - non-required success: small check
    - non-required failure: small X
    - Synthetic items (icon_px=7): colored dots instead of checkmarks/X
    """
    s = (status_norm or "").strip().lower()

    icon_px_i = int(icon_px or 12)
    is_synthetic = (icon_px_i == 7)
    
    if s == CIStatus.SUCCESS:
        if is_synthetic:
            # Synthetic success: green dot (normal size, 12px)
            return _circle_dot_fill_svg(color=COLOR_GREEN, width=12, height=12)
        if bool(is_required):
            # REQUIRED: filled green circle-check.
            out = (
                f'<span style="color: {COLOR_GREEN}; display: inline-flex; vertical-align: text-bottom;">'
                f'<svg aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="{icon_px_i}" height="{icon_px_i}" '
                'data-view-component="true" class="octicon octicon-check-circle-fill" fill="currentColor">'
                '<path fill-rule="evenodd" '
                'd="M8 16A8 8 0 108 0a8 8 0 000 16zm3.78-9.78a.75.75 0 00-1.06-1.06L7 9.94 5.28 8.22a.75.75 0 10-1.06 1.06l2 2a.75.75 0 001.06 0l4-4z">'
                "</path></svg></span>"
            )
        else:
            # Optional success: small check icon (preferred).
            out = (
                f'<span style="color: {COLOR_GREEN}; display: inline-flex; vertical-align: text-bottom;">'
                f'{_octicon_svg(path_d="M13.78 4.22a.75.75 0 00-1.06 0L6.75 10.19 3.28 6.72a.75.75 0 10-1.06 1.06l4 4a.75.75 0 001.06 0l7.5-7.5a.75.75 0 000-1.06z", name="octicon-check", width=icon_px_i, height=icon_px_i)}'
                "</span>"
            )
        if bool(warning_present):
            # Descendant failures: show a red X appended to the success icon.
            out += '<span style="color: #57606a; font-size: 11px; margin: 0 2px;">/</span>'
            if bool(required_failure):
                # REQUIRED descendant failure: filled red circle X (SVG).
                out += _circle_x_fill_svg(color=COLOR_RED, width=icon_px_i, height=icon_px_i, extra_style="margin-left: 2px;")
            else:
                # Optional descendant failure: small X.
                out += (
                    f'<span style="color: {COLOR_RED}; display: inline-flex; vertical-align: text-bottom; margin-left: 2px;">'
                    f'{_octicon_svg(path_d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 11-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z", name="octicon-x", width=icon_px_i, height=icon_px_i)}'
                    "</span>"
                )
        return out
    if s in {CIStatus.SKIPPED, CIStatus.NEUTRAL}:
        if is_synthetic:
            # Synthetic skipped: grey dot (normal size, 12px)
            return _circle_dot_fill_svg(color=COLOR_GREY, width=12, height=12)
        # GitHub-like "skipped": grey circle with a slash.
        return (
            '<span style="color: #8c959f; display: inline-flex; vertical-align: text-bottom;">'
            f'<svg aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="{icon_px_i}" height="{icon_px_i}" '
            'data-view-component="true" class="octicon octicon-circle-slash" fill="currentColor">'
            '<path fill-rule="evenodd" '
            'd="M8 16A8 8 0 108 0a8 8 0 000 16ZM1.5 8a6.5 6.5 0 0110.364-5.083l-8.947 8.947A6.473 6.473 0 011.5 8Zm3.136 5.083 8.947-8.947A6.5 6.5 0 014.636 13.083Z">'
            "</path></svg></span>"
        )
    if s == CIStatus.FAILURE:
        if is_synthetic:
            # Synthetic failure: red dot (normal size, 12px)
            return _circle_dot_fill_svg(color=COLOR_RED, width=12, height=12)
        if bool(is_required or required_failure):
            # REQUIRED: filled red circle X (SVG).
            return _circle_x_fill_svg(color=COLOR_RED, width=icon_px_i, height=icon_px_i)
        # Optional failure: small X.
        return (
            f'<span style="color: {COLOR_RED}; display: inline-flex; vertical-align: text-bottom;">'
            f'{_octicon_svg(path_d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 11-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z", name="octicon-x", width=icon_px_i, height=icon_px_i)}'
            "</span>"
        )
    if s == CIStatus.IN_PROGRESS:
        if is_synthetic:
            # Synthetic in-progress: yellow dot (normal size, 12px)
            return _circle_dot_fill_svg(color=COLOR_YELLOW, width=12, height=12)
        return _clock_ring_svg(color=COLOR_YELLOW, width=icon_px_i, height=icon_px_i)
    if s == CIStatus.PENDING:
        return _circle_dot_fill_svg(color=COLOR_GREY, width=icon_px_i, height=icon_px_i)
    if s == CIStatus.CANCELLED:
        if is_synthetic:
            # Synthetic cancelled: grey dot (normal size, 12px)
            return _circle_dot_fill_svg(color=COLOR_GREY, width=12, height=12)
        return _circle_x_fill_svg(color=COLOR_GREY, width=icon_px_i, height=icon_px_i)
    return _dot_svg(color=COLOR_GREY, width=icon_px_i, height=icon_px_i)


def _small_link_html(*, url: str, label: str) -> str:
    if not url:
        return ""
    return (
        f' <a href="{html.escape(url, quote=True)}" target="_blank" '
        f'style="color: #0969da; font-size: 11px; margin-left: 5px; text-decoration: none;">{html.escape(label)}</a>'
    )


# Shared library (dependency-light): `dynamo-utils/ci_log_errors/`
from ci_log_errors import render_error_snippet_html as _format_snippet_html  # shared implementation
from ci_log_errors import categorize_error_snippet_text as _snippet_categories


def github_api_stats_rows(
    *,
    github_api: Optional["GitHubAPIClient"],
    max_github_api_calls: Optional[int] = None,
    mode: str = "",
    mode_reason: str = "",
    top_n: int = 15,
) -> List[Tuple[str, Optional[str], str]]:
    """Build human-readable GitHub API statistics rows for the footer.

    Returns rows suitable for `page_stats`, including section headers ("## ...") and multiline values.
    Each row is a 3-tuple: (name, value, description).
    """
    rows: List[Tuple[str, Optional[str], str]] = []
    if github_api is None:
        return rows

    def _fmt_kv(d: Dict[str, Any]) -> str:
        parts = []
        for k, v in d.items():
            if v is None or v == "":
                continue
            parts.append(f"{k}: {v}")
        return "\n".join(parts)

    def _fmt_top_counts_and_time(
        *,
        by_label: Dict[str, int],
        time_by_label_s: Dict[str, float],
        n: int,
    ) -> str:
        labels = sorted(set(list(by_label.keys()) + list(time_by_label_s.keys())))
        if not labels:
            return "(none)"
        # Sort by count desc then time desc.
        labels.sort(key=lambda k: (-int(by_label.get(k, 0) or 0), -float(time_by_label_s.get(k, 0.0) or 0.0), k))
        labels = labels[: max(0, int(n))]
        w = max(10, max(len(x) for x in labels))
        out_lines = [f"{'category':<{w}}  calls   time"]
        for k in labels:
            c = int(by_label.get(k, 0) or 0)
            t = float(time_by_label_s.get(k, 0.0) or 0.0)
            out_lines.append(f"{k:<{w}}  {c:>5d}  {t:>6.2f}s")
        return "\n".join(out_lines)

    def _fmt_error_by_label_status(bls: Dict[str, Dict[int, int]], *, n: int) -> str:
        if not bls:
            return "(none)"
        items: List[Tuple[str, str, int]] = []
        for lbl, m in bls.items():
            if not isinstance(m, dict) or not m:
                continue
            total = int(sum(int(v or 0) for v in m.values()))
            inner = ", ".join([f"{int(code)}={int(cnt)}" for code, cnt in m.items()])
            items.append((str(lbl), inner, total))
        items.sort(key=lambda t: (-int(t[2]), t[0]))
        out = []
        for (lbl, inner, _tot) in items[: max(0, int(n))]:
            out.append(f"{lbl}: {inner}")
        more = max(0, len(items) - len(out))
        if more:
            out.append(f"(+{more} more)")
        return "\n".join(out) if out else "(none)"

    # Pull stats (best-effort).
    rest = github_api.get_rest_call_stats() or {}
    errs = github_api.get_rest_error_stats() or {}
    cache = github_api.get_cache_stats() or {}

    by_label = dict(rest.get("by_label") or {}) if isinstance(rest, dict) else {}
    time_by_label_s = dict(rest.get("time_by_label_s") or {}) if isinstance(rest, dict) else {}

    # Budget + mode
    budget: Dict[str, Any] = {}
    if mode:
        budget["mode"] = mode
    if mode_reason:
        budget["reason"] = mode_reason
    if max_github_api_calls is not None:
        budget["max_github_api_calls"] = int(max_github_api_calls)
    if isinstance(rest, dict):
        if rest.get("budget_max") is not None:
            budget["budget_max"] = rest.get("budget_max")
        budget["budget_exhausted"] = "true" if bool(rest.get("budget_exhausted")) else "false"
    rl = github_api.get_core_rate_limit_info() or {}
    rem = rl.get("remaining")
    lim = rl.get("limit")
    reset_pt = rl.get("reset_pt")
    if rem is not None and lim is not None:
        budget["core_remaining"] = f"{rem}/{lim}"
    elif rem is not None:
        budget["core_remaining"] = str(rem)
    if reset_pt:
        budget["core_resets"] = str(reset_pt)

    rows.append(("## GitHub API", None, ""))

    # REST summary (individual flat entries)
    rows.append(("github.rest.calls", str(int(rest.get("total") or 0)), "Total GitHub REST API calls made"))
    rows.append(("github.rest.ok", str(int(rest.get("success_total") or 0)), "Successful API calls"))
    rows.append(("github.rest.errors", str(int(rest.get("error_total") or 0)), "Failed API calls"))
    rows.append(("github.rest.time_total_secs", f"{float(rest.get('time_total_s') or 0.0):.2f}s", "Total time spent in API calls (SUM of all parallel threads - see note below)"))

    # ETag stats (conditional requests - 304s don't count against rate limit!)
    etag_304_total = int(GITHUB_API_STATS.etag_304_total or 0)
    rows.append(("github.rest.etag_304_total", str(etag_304_total), "304 Not Modified responses (FREE - don't count against rate limit!)"))
    if etag_304_total > 0:
        # Show top ETag 304 by endpoint
        etag_304_by_label = dict(GITHUB_API_STATS.etag_304_by_label or {})
        if etag_304_by_label:
            sorted_labels = sorted(etag_304_by_label.items(), key=lambda kv: (-kv[1], kv[0]))
            for lbl, cnt in sorted_labels[:5]:  # Top 5
                rows.append((f"github.rest.etag_304.{lbl}", str(int(cnt)), f"304 responses for {lbl} (cached, free)"))
            if len(sorted_labels) > 5:
                remaining = sum(cnt for lbl, cnt in sorted_labels[5:])
                rows.append(("github.rest.etag_304.other", str(remaining), "304 responses for other endpoints"))

    # Budget & mode (individual flat entries)
    if mode:
        rows.append(("github.mode", mode, "API budget enforcement mode"))
    if mode_reason:
        rows.append(("github.mode_reason", mode_reason, "Reason for current mode"))
    # Note: github.rest.budget_max is populated from GitHubAPIClient state (same as max_github_api_calls parameter)
    if isinstance(rest, dict) and rest.get("budget_max") is not None:
        rows.append(("github.rest.budget_max", str(rest.get("budget_max")), "Maximum API calls allowed"))
    if isinstance(rest, dict):
        rows.append(("github.rest.budget_exhausted", "true" if bool(rest.get("budget_exhausted")) else "false", "Whether API budget was exhausted"))
    if rem is not None and lim is not None:
        rows.append(("github.core_remaining", f"{rem}/{lim}", "GitHub core rate limit remaining"))
    elif rem is not None:
        rows.append(("github.core_remaining", str(rem), "GitHub core rate limit remaining"))
    if reset_pt:
        rows.append(("github.core_resets", str(reset_pt), "When rate limit resets"))

    # Cache summary (individual flat entries)
    rows.append(("github.cache.all.hits", str(int(cache.get("hits_total") or 0)), "Total cache hits across all operations"))
    rows.append(("github.cache.all.misses", str(int(cache.get("misses_total") or 0)), "Total cache misses"))
    rows.append(("github.cache.all.writes_ops", str(int(cache.get("writes_ops_total") or 0)), "Number of cache write operations"))
    rows.append(("github.cache.all.writes_entries", str(int(cache.get("writes_entries_total") or 0)), "Number of entries written to cache"))

    # TTL documentation (always show; independent of whether counters are non-zero in this run).
    rows.append(("github.cache.actions_job_details.ttl.completed", "30d", "Typical TTL used for completed job-details cache (dashboards pass 30d; only cached once completed)"))
    rows.append(("github.cache.actions_job_details.ttl.in_progress", "N/A (not cached)", "While a job is in_progress, job-details are not cached (polling frequency is adaptive via actions_job_status)"))
    rows.append(("github.cache.actions_job_status.ttl.in_progress", "adaptive", "Adaptive TTL for non-completed jobs: <1h=2m, <2h=4m, <4h=30m, <8h=60m, <12h=80m, >=12h=120m (until completed)"))

    # In-progress job-details (uncached)
    #
    # Job details are intentionally NOT cached until a job is completed
    # (steps and completed_at can be missing/incomplete while running).
    inprog = int(getattr(GITHUB_API_STATS, "actions_job_details_in_progress_uncached_total", 0) or 0)
    if inprog:
        rows.append((
            "github.cache.actions_job_details.in_progress_uncached",
            str(inprog),
            "Actions job-details fetches that returned in_progress (not cacheable yet; no TTL; retried on subsequent runs)"
        ))
        try:
            job_ids = sorted(str(x) for x in (getattr(GITHUB_API_STATS, "actions_job_details_in_progress_uncached_job_ids", set()) or set()))
            if job_ids:
                rows.append(("github.cache.actions_job_details.in_progress_uncached.sample_job_ids", ",".join(job_ids[:8]), "Sample job_ids (max 8)"))
        except Exception:
            pass

    # Pytest timings cache (individual flat entries)
    st = PYTEST_TIMINGS_CACHE.stats
    pytest_mem_count, pytest_disk_count = PYTEST_TIMINGS_CACHE.get_cache_sizes()
    rows.append(("pytest.cache.disk", str(pytest_disk_count), "Pytest test duration timings [pytest-test-timings.json] [key: job_id:step_name] [TTL: invalidated by mtime/size]"))
    rows.append(("pytest.cache.mem", str(pytest_mem_count), ""))
    rows.append(("pytest.cache.hits", str(int(st.hit)), ""))
    rows.append(("pytest.cache.misses", str(int(st.miss)), ""))
    rows.append(("pytest.cache.writes", str(int(st.write)), ""))
    rows.append(("pytest.parse_calls", str(int(st.parse_calls)), "Number of pytest timing file parses"))
    rows.append(("pytest.parse_secs", f"{float(st.parse_secs):.2f}s", "Time spent parsing pytest timings"))

    # Duration cache (individual flat entries)
    from cache.cache_duration import DURATION_CACHE
    duration_stats = DURATION_CACHE.stats
    duration_mem_count, duration_disk_count = DURATION_CACHE.get_cache_sizes()
    rows.append(("duration.cache.disk", str(duration_disk_count), "Job duration cache [duration-cache.json] [key: job_id or mtime:size:filename] [TTL: ∞ (immutable)]"))
    rows.append(("duration.cache.mem", str(duration_mem_count), ""))
    rows.append(("duration.cache.hits", str(int(duration_stats.hit)), ""))
    rows.append(("duration.cache.misses", str(int(duration_stats.miss)), ""))
    rows.append(("duration.cache.writes", str(int(duration_stats.write)), ""))


    # Cache entry counts (how many items are stored in each cache)
    # Organized by cache name with all related stats grouped together:
    # cache.github.{name}.disk, cache.github.{name}.mem, cache.github.{name}.hits, cache.github.{name}.misses
    entries = dict(cache.get("entries") or {}) if isinstance(cache, dict) else {}
    
    # Get hits/misses by cache type
    hits_by = dict(cache.get("hits_by") or {}) if isinstance(cache, dict) else {}
    misses_by = dict(cache.get("misses_by") or {}) if isinstance(cache, dict) else {}
    
    if entries:
        rows.append(("## Cache Sizes", None, ""))

        # TTL information for each cache type (using constants from common.py)
        # Format: human-readable duration (e.g., "5m", "1h", "30d", "365d", "∞")
        # Descriptions for each cache type (without _mem/_disk suffix)
        # TTL information and cache key format integrated into descriptions
        cache_descriptions_mem = {
            "pr_checks": "PR check runs [key: owner/repo#PR or owner/repo#PR:sha]",
            "pulls_list": "Pull request list responses [key: owner/repo:state]",
            "pr_branch": "PR branch information [key: owner/repo:branch]",
            "raw_log_text": "Raw CI log text content index [key: job_id]",
            "actions_job_status": "GitHub Actions job status [key: job_id]",
            "actions_job_details": "GitHub Actions job details [key: owner/repo:run_id:job_id]",
            "actions_workflow": "Workflow run metadata [key: owner/repo:run_id]",
            "required_checks": "Required PR check names [key: owner/repo:pr#]",
            "pr_info": "PR metadata (author, labels, reviews) [key: owner/repo#PR]",
            "search_issues": "GitHub issue search results [key: query_hash]",
            "job_log": "Parsed job log content [key: job_id]",
        }

        cache_descriptions_disk = {
            "actions_job_status": "GitHub Actions job status [actions_jobs.json] [key: owner/repo:jobstatus:job_id]",
            "actions_job_details": "GitHub Actions job details [actions_jobs.json] [key: owner/repo:job:job_id]",
            "pr_checks": "PR check runs [pr_checks_cache.json] [key: owner/repo#PR or owner/repo#PR:sha]",
            "pulls_list": "Pull request list responses [pulls_open_cache.json] [key: owner/repo:state]",
            "pr_branch": "PR branch information [pr_branch_cache.json] [key: owner/repo:branch]",
            "pr_info": "Full PR details with required checks and reviews [pr_info.json] [key: owner/repo#PR]",
            "search_issues": "GitHub search/issues API results [search_issues.json] [key: query_hash]",
            "job_log": "Job log error summaries and snippets [job_logs_cache.json] [key: job_id]",
            "raw_log_text": f"Raw CI log text content index [index.json] [key: job_id]",
            "required_checks": "Required PR check names [required_checks.json] [key: owner/repo:pr#]",
            "merge_dates": f"PR merge dates from GitHub API [github_pr_merge_dates.json] [key: owner/repo#PR]",
            "commit_history": "Commit history with metadata [commit_history.json] [key: varies]",
            "commit_history_snippets": "Commit message snippets [commit_history_snippets.json] [key: commit_sha]",
            "pytest_timings": "Pytest test duration timings [pytest-test-timings.json] [key: test_name]",
            "gitlab_pipeline_jobs": "GitLab pipeline job details [gitlab_pipeline_jobs_details_v3.json] [key: project_id:pipeline_id]",
            "gitlab_pipeline_status": "GitLab pipeline status [gitlab_pipeline_status.json] [key: project_id:pipeline_id]",
            "gitlab_mr_pipelines": "GitLab MR pipeline associations [gitlab_mr_pipelines.json] [key: project_id:mr_iid]",
        }

        # TTL documentation for each cache type (shown as separate rows after .disk/.mem/.hits/.misses)
        cache_ttl_descriptions = {
            "actions_job_status": "adaptive (in_progress: <1h=2m, <2h=4m, <4h=30m, <8h=60m, <12h=80m, >=12h=120m; completed: ∞)",
            "actions_job_details": "30d (only cached once completed)",
            "pr_checks": "adaptive (<1h=2m, <2h=4m, <4h=30m, <8h=60m, <12h=80m, >=12h=120m; closed/merged: 360d)",
            "pulls_list": "adaptive (<1h=2m, <2h=4m, <4h=30m, <8h=60m, <12h=80m, >=12h=120m)",
            "pr_branch": "adaptive (<1h=2m, <2h=4m, <4h=30m, <8h=60m, <12h=80m, >=12h=120m; closed/merged: 360d)",
            "pr_info": "by updated_at timestamp (invalidated when PR changes)",
            "search_issues": "varies",
            "job_log": "∞ (immutable, no TTL check)",
            "raw_log_text": f"{_format_ttl_duration(DEFAULT_RAW_LOG_TEXT_TTL_S)} (immutable once completed)",
            "required_checks": "adaptive (<1h=2m, <2h=4m, <4h=30m, <8h=60m, <12h=80m, >=12h=120m; closed/merged: 360d)",
            "merge_dates": f"{_format_ttl_duration(DEFAULT_CLOSED_PRS_TTL_S)} (immutable once merged)",
            "commit_history": "varies",
            "commit_history_snippets": "365d (immutable)",
            "pytest_timings": "varies",
            "gitlab_pipeline_jobs": "varies",
            "gitlab_pipeline_status": "varies",
            "gitlab_mr_pipelines": "varies",
        }

        # Build a unified dict of cache stats: {cache_name: {disk, mem, hits, misses}}
        cache_stats = {}
        
        for cache_name in entries.keys():
            count = int(entries[cache_name])
            if cache_name.endswith("_mem"):
                base_name = cache_name[:-4]  # Remove "_mem" suffix
                if base_name not in cache_stats:
                    cache_stats[base_name] = {}
                cache_stats[base_name]["mem"] = (count, cache_descriptions_mem.get(base_name, f"Cached entries in {base_name}"))
            elif cache_name.endswith("_disk"):
                base_name = cache_name[:-5]  # Remove "_disk" suffix
                if base_name not in cache_stats:
                    cache_stats[base_name] = {}
                cache_stats[base_name]["disk"] = (count, cache_descriptions_disk.get(base_name, f"Cached entries in {base_name}"))
        
        # Add hits/misses from GITHUB_API_STATS for all caches
        # This uses the standard _cache_hit()/_cache_miss() tracking mechanism
        for cache_name in cache_stats.keys():
            # Aggregate all hits/misses for this cache (handles .mem, .disk, .network, etc.)
            # Include both exact matches (e.g., "pr_checks") and prefixed matches (e.g., "pr_checks.mem")
            cache_hits = int(sum(int(v or 0) for k, v in hits_by.items() if str(k) == cache_name or str(k).startswith(f"{cache_name}.")))
            cache_misses = int(sum(int(v or 0) for k, v in misses_by.items() if str(k) == cache_name or str(k).startswith(f"{cache_name}.")))
            
            # Always add hits/misses entries (even if 0) for consistency (no descriptions needed)
            cache_stats[cache_name]["hits"] = (cache_hits, "")
            cache_stats[cache_name]["misses"] = (cache_misses, "")
        
        # Output cache stats grouped by cache name, with disk/mem/hits/misses/ttl together
        for cache_name in sorted(cache_stats.keys()):
            stats = cache_stats[cache_name]
            
            # Show in order: .disk (always), .mem (if >0), .ttl (if available), .hits (if exists), .misses (if exists)
            if "disk" in stats:
                count, desc = stats["disk"]
                rows.append((f"github.cache.{cache_name}.disk", str(count), desc))
            
            if "mem" in stats:
                count, desc = stats["mem"]
                if count > 0:  # Only show mem if count > 0
                    rows.append((f"github.cache.{cache_name}.mem", str(count), ""))
            
            # TTL documentation (separate row after disk/mem)
            if cache_name in cache_ttl_descriptions:
                rows.append((f"github.cache.{cache_name}.ttl", cache_ttl_descriptions[cache_name], "Cache TTL policy"))
            
            if "hits" in stats:
                count, desc = stats["hits"]
                rows.append((f"github.cache.{cache_name}.hits", str(count), desc))
            
            if "misses" in stats:
                count, desc = stats["misses"]
                rows.append((f"github.cache.{cache_name}.misses", str(count), desc))

    # REST by category (top N as individual entries)
    labels = sorted(set(list(by_label.keys()) + list(time_by_label_s.keys())))
    if labels:
        labels.sort(key=lambda k: (-int(by_label.get(k, 0) or 0), -float(time_by_label_s.get(k, 0.0) or 0.0), k))
        for lbl in labels[: max(0, int(top_n))]:
            c = int(by_label.get(lbl, 0) or 0)
            t = float(time_by_label_s.get(lbl, 0.0) or 0.0)
            rows.append((f"github.rest.by_category.{lbl}.calls", str(c), f"API calls for {lbl}"))
            rows.append((f"github.rest.by_category.{lbl}.time_secs", f"{t:.2f}s", f"Time spent in {lbl} calls"))
            
            # Attribution details for pulls_list (if instrumented)
            if lbl == "pulls_list":
                try:
                    total = int(getattr(GITHUB_API_STATS, "pulls_list_network_page_calls_total", 0) or 0)
                    by_bucket = dict(getattr(GITHUB_API_STATS, "pulls_list_network_page_calls_by_cache_age_bucket", {}) or {})
                    by_state = dict(getattr(GITHUB_API_STATS, "pulls_list_network_page_calls_by_state", {}) or {})
                    if total > 0:
                        rows.append((f"github.rest.by_category.{lbl}.total_page_fetches", str(total), "Total /pulls page fetches (should match .calls above)"))
                    if by_state:
                        for st in sorted(by_state.keys()):
                            rows.append((f"github.rest.by_category.{lbl}.by_state.{st}", str(by_state[st]), f"Page fetches for state={st}"))
                    if by_bucket:
                        for bucket in ["no_cache", "<1h", "<2h", "<3h", ">=3h"]:
                            cnt = by_bucket.get(bucket, 0)
                            if cnt > 0:
                                rows.append((f"github.rest.by_category.{lbl}.by_cache_age.{bucket}", str(cnt), f"Page fetches where stale cache was {bucket} old"))
                except Exception:
                    pass
            
            # Attribution details for actions_run (if instrumented)
            if lbl == "actions_run":
                try:
                    prefetch_total = int(getattr(GITHUB_API_STATS, "actions_run_prefetch_total", 0) or 0)
                    if prefetch_total > 0:
                        rows.append((f"github.rest.by_category.{lbl}.prefetch_workflow_metadata", str(prefetch_total), "Workflow metadata prefetch calls in _fetch_pr_checks_data() to batch-fetch (name,event) for all check-runs"))
                except Exception:
                    pass

    # REST errors by status (individual flat entries)
    by_status = (errs or {}).get("by_status") if isinstance(errs, dict) else {}
    if isinstance(by_status, dict) and by_status:
        for status_code, count in list(by_status.items())[:8]:
            rows.append((f"github.rest.errors.by_status.{status_code}", str(count), f"Errors with HTTP status {status_code}"))

    # Last REST error (optional)
    last = (errs or {}).get("last") if isinstance(errs, dict) else None
    last_label = (errs or {}).get("last_label") if isinstance(errs, dict) else ""
    if isinstance(last, dict) and last.get("status"):
        rows.append(("github.rest.last_error.status", str(last.get("status")), "Most recent API error status"))
        if last_label:
            rows.append(("github.rest.last_error.label", str(last_label), "Category of last error"))
        body = str(last.get("body") or "").strip()
        if body:
            if len(body) > 160:
                body = body[:160].rstrip() + "…"
            rows.append(("github.rest.last_error.body", body, "Last error response body"))
        url = str(last.get("url") or "").strip()
        if url:
            rows.append(("github.rest.last_error.url", url, "URL of last failed request"))

    return rows


def build_page_stats(
    *,
    generation_time_secs: Optional[float] = None,
    github_api: Optional[Any] = None,
    max_github_api_calls: Optional[int] = None,
    cache_only_mode: bool = False,
    cache_only_reason: str = "",
    phase_timings: Optional[Dict[str, float]] = None,
    repos_scanned: Optional[int] = None,
    prs_shown: Optional[int] = None,
    commits_shown: Optional[int] = None,
    repo_info: Optional[str] = None,
    github_user: Optional[str] = None,
    max_branches: Optional[int] = None,
    max_checks_fetch: Optional[int] = None,
    refresh_closed_prs: bool = False,
    gitlab_fetch_skip: Optional[bool] = None,
    gitlab_client: Optional[Any] = None,
) -> List[Tuple[str, Optional[str], str]]:
    """Build unified page statistics for all dashboards (local/remote/commit-history).

    This ensures all 3 dashboards show the same statistics structure, even if some values
    are 0 or N/A. Statistics are displayed in a consistent order across all dashboards.

    ALL DASHBOARDS NOW USE UNIFIED get_pr_checks_rows() API for check runs:
    - Local branches: Uses get_pr_checks_rows() → tracks required_checks.* stats
    - Remote branches: Uses get_pr_checks_rows() → tracks required_checks.* stats
    - Commit history: Uses get_pr_checks_rows() → tracks required_checks.* stats

    GitHub cache statistics are read from the global GITHUB_CACHE_STATS object which is
    populated by GitHubAPIClient.get_pr_checks_rows() as it executes.

    Commit-history performance statistics are read from the global COMMIT_HISTORY_PERF_STATS
    object which is populated during commit history processing.

    Args:
        generation_time_secs: Total generation time in seconds
        github_api: GitHubAPIClient instance (for API stats)
        max_github_api_calls: Max API calls limit
        cache_only_mode: Whether GitHub API was in cache-only mode
        cache_only_reason: Reason for cache-only mode
        phase_timings: Dict of phase timings (prune, scan, render, write, total)
        repos_scanned: Number of repositories scanned (local branches)
        prs_shown: Number of PRs shown
        commits_shown: Number of commits shown (commit history)
        repo_info: Repository info string like "owner/repo"
        github_user: GitHub username
        max_branches: max_branches flag value
        max_checks_fetch: max_checks_fetch flag value
        refresh_closed_prs: Whether refresh_closed_prs is enabled
        gitlab_fetch_skip: Whether GitLab fetch was skipped (commit history only)
        gitlab_client: GitLab client for stats (commit history only)

    Returns:
        List of (key, value, description) tuples for the Statistics section
    """
    page_stats: List[Tuple[str, Optional[str], str]] = []

    # 1. Generation time (always first)
    if generation_time_secs is not None:
        page_stats.append(("generation.total_secs", f"{generation_time_secs:.2f}s", "Total dashboard generation time (wall-clock elapsed time)"))

    # 2. Context info (repo, user, counts)
    if repo_info:
        page_stats.append(("dashboard.repo", repo_info, "Repository being displayed"))
    if github_user:
        page_stats.append(("dashboard.github_user", github_user, "GitHub user filter"))
    if repos_scanned is not None:
        page_stats.append(("dashboard.repos_scanned", str(repos_scanned), "Local repositories scanned"))
    if prs_shown is not None:
        page_stats.append(("dashboard.prs_shown", str(prs_shown), "Pull requests displayed"))
    if commits_shown is not None:
        page_stats.append(("dashboard.commits_shown", str(commits_shown), "Commits displayed"))
    
    # 3. GitHub API stats (from github_api_stats_rows)
    if github_api is not None:
        mode = "cache-only" if cache_only_mode else "normal"
        api_rows = github_api_stats_rows(
            github_api=github_api,
            max_github_api_calls=max_github_api_calls,
            mode=mode,
            mode_reason=cache_only_reason,
            top_n=15,
        )
        page_stats.extend(list(api_rows or []))

    # 4. CLI flags (if set)
    if max_branches is not None:
        page_stats.append(("cli.max_branches", str(max_branches), "Max branches to display"))
    if max_checks_fetch is not None:
        page_stats.append(("cli.max_checks_fetch", str(max_checks_fetch), "Max check runs to fetch per commit"))
    if refresh_closed_prs:
        page_stats.append(("cli.refresh_closed_prs", "true", "Whether closed PRs are refreshed"))
    if gitlab_fetch_skip is not None:
        page_stats.append(("cli.gitlab_fetch_skip", "true" if gitlab_fetch_skip else "false", "Whether GitLab API calls were skipped"))

    # 5. Phase timings (prune/scan/render/write/total)
    if phase_timings:
        for phase in ["prune", "scan", "render", "write", "total"]:
            if phase in phase_timings:
                page_stats.append((f"phase.{phase}.total_secs", f"{phase_timings[phase]:.2f}s", f"Time spent in {phase} phase"))
    
    # 6. Performance counters (read from global COMMIT_HISTORY_PERF_STATS)
    # These apply to all dashboards (snippets/markers are used everywhere)
    perf_stats = COMMIT_HISTORY_PERF_STATS

    # Composite SHA (commit-history-only, show real values when available)
    if perf_stats.composite_sha_cache_hit > 0 or perf_stats.composite_sha_cache_miss > 0:
        # Show disk cache size (from COMMIT_HISTORY_CACHE)
        commit_mem_count, commit_disk_count = COMMIT_HISTORY_CACHE.get_cache_sizes()
        page_stats.append(("composite_sha.cache.mem", str(commit_mem_count), "Commit history cache entries (in memory)"))
        page_stats.append(("composite_sha.cache.disk", str(commit_disk_count), "Commit history cache entries (on disk before run)"))
        page_stats.append(("composite_sha.cache.hits", str(perf_stats.composite_sha_cache_hit), ""))
        page_stats.append(("composite_sha.cache.misses", str(perf_stats.composite_sha_cache_miss), ""))
        page_stats.append(("composite_sha.errors", str(perf_stats.composite_sha_errors), "Errors computing composite SHAs (commit history only)"))
        page_stats.append(("composite_sha.total_secs", f"{perf_stats.composite_sha_total_secs:.2f}s", "Total time computing composite SHAs (commit history only)"))
        page_stats.append(("composite_sha.compute_secs", f"{perf_stats.composite_sha_compute_secs:.2f}s", "Time spent in SHA computations (commit history only)"))
    else:
        commit_mem_count, commit_disk_count = COMMIT_HISTORY_CACHE.get_cache_sizes()
        page_stats.append(("composite_sha.cache.mem", str(commit_mem_count), "Commit history cache entries (in memory)"))
        page_stats.append(("composite_sha.cache.disk", str(commit_disk_count), "Commit history cache entries (on disk before run)"))
        page_stats.append(("composite_sha.cache.hits", "(N/A)", ""))
        page_stats.append(("composite_sha.cache.misses", "(N/A)", ""))
        page_stats.append(("composite_sha.errors", "(N/A)", "Errors computing composite SHAs (commit history only)"))
        page_stats.append(("composite_sha.total_secs", "(N/A)", "Total time computing composite SHAs (commit history only)"))
        page_stats.append(("composite_sha.compute_secs", "(N/A)", "Time spent in SHA computations (commit history only)"))

    # Snippet cache (always show, tracked globally in SNIPPET_CACHE)
    snippet_stats = SNIPPET_CACHE.stats
    snippet_mem_count, snippet_disk_count = SNIPPET_CACHE.get_cache_sizes()
    page_stats.append(("snippet.cache.disk", str(snippet_disk_count), "CI log snippet cache [snippet-cache.json] [key: ci_log_errors_sha:log_filename] [TTL: 365d]"))
    page_stats.append(("snippet.cache.mem", str(snippet_mem_count), ""))
    page_stats.append(("snippet.cache.hits", str(int(snippet_stats.hit)), ""))
    page_stats.append(("snippet.cache.misses", str(int(snippet_stats.miss)), ""))
    page_stats.append(("snippet.cache.writes", str(int(snippet_stats.write)), ""))
    page_stats.append(("snippet.compute_secs", f"{float(snippet_stats.compute_secs):.2f}s", "Time extracting snippets from logs"))
    page_stats.append(("snippet.total_secs", f"{float(snippet_stats.total_secs):.2f}s", "Total time in snippet operations"))

    # Markers / local build reports (used by all dashboards)
    if perf_stats.marker_composite_with_reports > 0 or perf_stats.marker_composite_without_reports > 0:
        page_stats.append(("marker.composite.unique", str(perf_stats.marker_composite_with_reports + perf_stats.marker_composite_without_reports), "Unique local build markers found"))
        page_stats.append(("marker.composite.with.reports", str(perf_stats.marker_composite_with_reports), "Markers with test reports"))
        page_stats.append(("marker.composite.with.status", str(perf_stats.marker_composite_with_status), "Markers with status info"))
        page_stats.append(("marker.composite.without.reports", str(perf_stats.marker_composite_without_reports), "Markers without reports"))
        page_stats.append(("marker.total_secs", f"{perf_stats.marker_total_secs:.2f}s", "Time processing build markers"))
    else:
        page_stats.append(("marker.composite.unique", "(N/A)", "Unique local build markers found"))
        page_stats.append(("marker.composite.with.reports", "(N/A)", "Markers with test reports"))
        page_stats.append(("marker.composite.with.status", "(N/A)", "Markers with status info"))
        page_stats.append(("marker.composite.without.reports", "(N/A)", "Markers without reports"))
        page_stats.append(("marker.total_secs", "(N/A)", "Time processing build markers"))

    # GitLab cache stats (commit-history-only, but show consistently)
    has_gitlab_stats = (
        perf_stats.gitlab_cache_registry_images_hit > 0 or
        perf_stats.gitlab_cache_registry_images_miss > 0 or
        perf_stats.gitlab_cache_pipeline_status_hit > 0 or
        perf_stats.gitlab_cache_pipeline_status_miss > 0 or
        perf_stats.gitlab_cache_pipeline_jobs_hit > 0 or
        perf_stats.gitlab_cache_pipeline_jobs_miss > 0
    )
    if has_gitlab_stats:
        page_stats.append(("gitlab.cache.registry_images.hits", str(perf_stats.gitlab_cache_registry_images_hit), ""))
        page_stats.append(("gitlab.cache.registry_images.misses", str(perf_stats.gitlab_cache_registry_images_miss), ""))
        page_stats.append(("gitlab.cache.pipeline_status.hits", str(perf_stats.gitlab_cache_pipeline_status_hit), ""))
        page_stats.append(("gitlab.cache.pipeline_status.misses", str(perf_stats.gitlab_cache_pipeline_status_miss), ""))
        page_stats.append(("gitlab.cache.pipeline_jobs.hits", str(perf_stats.gitlab_cache_pipeline_jobs_hit), ""))
        page_stats.append(("gitlab.cache.pipeline_jobs.misses", str(perf_stats.gitlab_cache_pipeline_jobs_miss), ""))
        page_stats.append(("gitlab.registry_images.total_secs", f"{perf_stats.gitlab_registry_images_total_secs:.2f}s", "Time fetching GitLab registry images (commit history only)"))
        page_stats.append(("gitlab.pipeline_status.total_secs", f"{perf_stats.gitlab_pipeline_status_total_secs:.2f}s", "Time fetching GitLab pipeline status (commit history only)"))
        page_stats.append(("gitlab.pipeline_jobs.total_secs", f"{perf_stats.gitlab_pipeline_jobs_total_secs:.2f}s", "Time fetching GitLab pipeline jobs (commit history only)"))
    else:
        page_stats.append(("gitlab.cache.registry_images.hits", "(N/A)", ""))
        page_stats.append(("gitlab.cache.registry_images.misses", "(N/A)", ""))
        page_stats.append(("gitlab.cache.pipeline_status.hits", "(N/A)", ""))
        page_stats.append(("gitlab.cache.pipeline_status.misses", "(N/A)", ""))
        page_stats.append(("gitlab.cache.pipeline_jobs.hits", "(N/A)", ""))
        page_stats.append(("gitlab.cache.pipeline_jobs.misses", "(N/A)", ""))
        page_stats.append(("gitlab.registry_images.total_secs", "(N/A)", "Time fetching GitLab registry images (commit history only)"))
        page_stats.append(("gitlab.pipeline_status.total_secs", "(N/A)", "Time fetching GitLab pipeline status (commit history only)"))
        page_stats.append(("gitlab.pipeline_jobs.total_secs", "(N/A)", "Time fetching GitLab pipeline jobs (commit history only)"))

    # 6b. GitHub cache statistics (reads from global GITHUB_CACHE_STATS)
    # Note: Individual cache stats (disk/mem/hits/misses) are now grouped together
    # in build_github_cache_stats() and included via page_stats.extend(api_rows) above.
    # No need to add merge_dates or required_checks hits/misses here separately.

    # 7. GitLab API stats (commit history only)
    if gitlab_client is not None:
        if hasattr(gitlab_client, "get_rest_call_stats"):
            st = gitlab_client.get_rest_call_stats() or {}
            gl_total = int(st.get("total", 0) or 0)
            gl_ok = int(st.get("success_total", 0) or 0)
            gl_err = int(st.get("error_total", 0) or 0)
            if gl_total > 0:
                page_stats.append(("gitlab.rest.calls", str(gl_total), "Total GitLab REST API calls"))
                page_stats.append(("gitlab.rest.success", str(gl_ok), "Successful GitLab API calls"))
                page_stats.append(("gitlab.rest.errors", str(gl_err), "Failed GitLab API calls"))

    # Sort all stats alphabetically by key (but preserve section headers with None values)
    section_headers = [s for s in page_stats if s[1] is None]
    regular_stats = [s for s in page_stats if s[1] is not None]
    regular_stats.sort(key=lambda x: x[0].lower())

    return regular_stats


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
    github_api: Optional["GitHubAPIClient"],
    job_url: str,
    raw_log_path: Optional[Path] = None,
    is_required: bool = False,
) -> List[Tuple[str, str, str]]:
    """Return phase tuples for the Build-and-Test phase breakdown (best-effort).

    Shared helper used by both dashboards to keep logic identical.
    """
    phases3: List[Tuple[str, str, str]] = []
    jid = extract_actions_job_id_from_url(str(job_url or ""))
    if github_api and jid:
        # Use 30-day TTL to match prefetch_actions_job_details_pass (prevents cache misses).
        # Job details are immutable once completed (verified by get_actions_job_details_cached),
        # so there's no correctness benefit to a shorter TTL. The old 600s (10min) TTL caused
        # ~94 cache misses per dashboard render when prefetched data (30d TTL) was >10min old.
        job = github_api.get_actions_job_details_cached(owner="ai-dynamo", repo="dynamo", job_id=jid, ttl_s=30 * 24 * 3600) or {}
        if isinstance(job, dict):
            phases3 = build_and_test_dynamo_phases_from_actions_job(job) or []

    return [(str(n), str(d), str(s)) for (n, d, s) in (phases3 or [])]


