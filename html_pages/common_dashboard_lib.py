#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Common dashboard helpers shared by HTML generators under dynamo-utils/html_pages/.

This file intentionally groups previously-split helper modules into one place to:
- avoid UI drift between dashboards
- reduce small-module sprawl
- keep <pre>-safe tree rendering + check-line rendering consistent
"""

from __future__ import annotations

import hashlib
import html
import logging
import os
import re
import sys
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from datetime import datetime, timezone

from common import GitHubAPIClient, classify_ci_kind
from common_types import CIStatus

# Avoid circular import by importing CIJobNode only for type checking
if TYPE_CHECKING:
    from common_branch_nodes import CIJobNode

# Initialize module logger
logger = logging.getLogger(__name__)

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

    - expand for required failures (red ✗)
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
    - order:  ✓(required) N, ✓(optional) N, ✗(required) N, ✗(optional) N, then non-terminal states (grey)
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
    try:
        k = str(node_key or "")
        if not k:
            return ""
        # Prefix with a letter for HTML id validity.
        return f"tree_children_k_{_hash10(k)}"
    except Exception:
        return ""


def parse_workflow_yaml_and_build_mapping_pass(
    flat_nodes: List[TreeNodeVM],
    repo_root: Path,
    commit_sha: str = "",
) -> List[TreeNodeVM]:
    """Parse YAML to annotate nodes with parent/child metadata.
    
    This pass reads .github/workflows/*.yml files and annotates each node with
    information about which jobs it depends on (children) and which jobs depend on it (parents).
    The nodes remain flat and disconnected - actual connections are made in PASS 2.
    
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


def expand_required_failure_descendants_pass(nodes: List[TreeNodeVM]) -> List[TreeNodeVM]:
    """Expand any node that has a REQUIRED failure anywhere in its descendant subtree.

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
                noncollapsible_icon=getattr(n, "noncollapsible_icon", ""),
                job_name=getattr(n, "job_name", ""),
                core_job_name=getattr(n, "core_job_name", ""),
                workflow_name=getattr(n, "workflow_name", ""),
                variant=getattr(n, "variant", ""),
                pr_number=getattr(n, "pr_number", None),
                raw_html_content=getattr(n, "raw_html_content", ""),
            ),
            has_req,
        )

    out: List[TreeNodeVM] = []
    for n in (nodes or []):
            n2, _ = walk(n)
            out.append(n2)
    return out


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
        import html as html_module
        raw = str(text or "")
        # Detect arch token
        m = re.search(r"\((arm64|aarch64|amd64)\)", raw, flags=re.IGNORECASE)
        if not m:
            return html_module.escape(raw)
        
        arch = str(m.group(1) or "").strip().lower()
        # Determine color
        if arch in {"arm64", "aarch64"}:
            color = "#b8860b"  # Dark yellow/gold for arm64
            # Normalize to "(arm64)" and append "; aarch64"
            raw2 = re.sub(r"\(\s*(arm64|aarch64)\s*\)", "(arm64)", raw, flags=re.IGNORECASE)
            raw2 = re.sub(r"\(\s*arm64\s*\)(?!\s*;\s*aarch64\b)", "(arm64); aarch64", raw2, flags=re.IGNORECASE)
            return f'<span style="color: {color};">{html_module.escape(raw2)}</span>'
        elif arch == "amd64":
            color = "#0969da"  # Blue for amd64
            raw2 = re.sub(r"\(\s*amd64\s*\)", "(amd64)", raw, flags=re.IGNORECASE)
            raw2 = re.sub(r"\(\s*amd64\s*\)(?!\s*;\s*x86_64\b)", "(amd64); x86_64\u00a0", raw2, flags=re.IGNORECASE)  # Non-breaking space for alignment
            return f'<span style="color: {color};">{html_module.escape(raw2)}</span>'
        return html_module.escape(raw)
    
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
            core_name = getattr(node_vm, 'core_job_name', '<none>')
            logger.debug(f"  [{idx}] TreeNodeVM has core_job_name='{core_name}'")
        
        ci_info_nodes.append(node_vm)
    
    logger.info(f"Converted {len(ci_info_nodes)} nodes to TreeNodeVM")
    return ci_info_nodes


def run_all_passes(
    ci_nodes: List,  # List[BranchNode] from common_branch_nodes
    repo_root: Path,
    commit_sha: str = "",
) -> List[TreeNodeVM]:
    """
    Centralized CI tree node processing pipeline.
    
    Simple orchestrator that calls passes in sequence.
    NOTE: This function is called ONCE PER PR, not for all PRs at once.
    
    Args:
        ci_nodes: List of BranchNode objects for a SINGLE PR (from build_ci_nodes_from_pr or mock_build_ci_nodes)
        repo_root: Path to the repository root (for .github/workflows/ parsing)
        commit_sha: Commit SHA for per-commit node uniqueness
    
    Returns:
        Processed list of TreeNodeVM nodes with YAML augmentation applied.
    """
    logger.info(f"[run_all_passes] Starting with {len(ci_nodes)} CI nodes (per-PR)")
    
    # PASS 1: Convert BranchNode to TreeNodeVM (actual CI info from GitHub)
    ci_info_nodes = convert_branch_nodes_to_tree_vm_pass(ci_nodes)
    
    # PASS 2: Parse YAML workflows to build mappings (job names, dependencies, etc.)
    # This populates global mappings and returns them for use in subsequent passes
    _, yaml_mappings = parse_workflow_yaml_and_build_mapping_pass([], repo_root, commit_sha=commit_sha)
    
    # PASS 3: Augment CI nodes with YAML information (short names, dependencies)
    augmented_nodes = augment_ci_with_yaml_info_pass(ci_nodes, yaml_mappings)
    
    # PASS 4: Move jobs under parent nodes
    grouped_nodes = augmented_nodes
    grouped_nodes = move_jobs_by_prefix_pass(grouped_nodes, prefix="vllm", parent_name="backend-status-check")
    grouped_nodes = move_jobs_by_prefix_pass(grouped_nodes, prefix="sglang", parent_name="backend-status-check")
    grouped_nodes = move_jobs_by_prefix_pass(grouped_nodes, prefix="trtllm", parent_name="backend-status-check")
    grouped_nodes = move_jobs_by_prefix_pass(grouped_nodes, prefix="operator", parent_name="backend-status-check")

    # Group other jobs under parent nodes
    grouped_nodes = move_jobs_by_prefix_pass(grouped_nodes, prefix="deploy-", parent_name="deploy")
    grouped_nodes = move_jobs_by_prefix_pass(grouped_nodes, prefix="build-test", parent_name="dynamo-status-check")
    grouped_nodes = move_jobs_by_prefix_pass(grouped_nodes, prefix="Post-Merge CI / ", parent_name="post-merge-ci", parent_label="Post-Merge CI", create_if_has_children=True)
    
    # Group fast jobs under _fast parent
    _FAST = "_fast"
    _FAST_LABEL = "Jobs that tend to run fast"
    grouped_nodes = move_jobs_by_prefix_pass(grouped_nodes, prefix="broken-links-check", parent_name=_FAST, parent_label=_FAST_LABEL, create_if_has_children=True)
    grouped_nodes = move_jobs_by_prefix_pass(grouped_nodes, prefix="build-docs", parent_name=_FAST, parent_label=_FAST_LABEL, create_if_has_children=True)
    grouped_nodes = move_jobs_by_prefix_pass(grouped_nodes, prefix="changed-files", parent_name=_FAST, parent_label=_FAST_LABEL, create_if_has_children=True)
    grouped_nodes = move_jobs_by_prefix_pass(grouped_nodes, prefix="clean", parent_name=_FAST, parent_label=_FAST_LABEL, create_if_has_children=True)
    grouped_nodes = move_jobs_by_prefix_pass(grouped_nodes, prefix="clippy", parent_name=_FAST, parent_label=_FAST_LABEL, create_if_has_children=True)
    grouped_nodes = move_jobs_by_prefix_pass(grouped_nodes, prefix="CodeRabbit", parent_name=_FAST, parent_label=_FAST_LABEL, create_if_has_children=True)
    grouped_nodes = move_jobs_by_prefix_pass(grouped_nodes, prefix="dco-comment", parent_name=_FAST, parent_label=_FAST_LABEL, create_if_has_children=True)
    grouped_nodes = move_jobs_by_prefix_pass(grouped_nodes, prefix="event_file", parent_name=_FAST, parent_label=_FAST_LABEL, create_if_has_children=True)
    grouped_nodes = move_jobs_by_prefix_pass(grouped_nodes, prefix="label", parent_name=_FAST, parent_label=_FAST_LABEL, create_if_has_children=True)
    grouped_nodes = move_jobs_by_prefix_pass(grouped_nodes, prefix="lychee", parent_name=_FAST, parent_label=_FAST_LABEL, create_if_has_children=True)
    grouped_nodes = move_jobs_by_prefix_pass(grouped_nodes, prefix="trigger-ci", parent_name=_FAST, parent_label=_FAST_LABEL, create_if_has_children=True)
    
    # PASS 5: Sort nodes by name
    sorted_nodes = sort_nodes_by_name_pass(grouped_nodes)
    
    # PASS 6: Expand nodes with required failures in descendants
    final_nodes = expand_required_failure_descendants_pass(sorted_nodes)
    
    # PASS 7: Move required jobs to the top (alphabetically sorted)
    final_nodes = move_required_jobs_to_top_pass(final_nodes)
    
    # PASS 8: Verify the final tree structure
    verify_tree_structure_pass(final_nodes, ci_nodes, commit_sha=commit_sha)
    
    logger.info(f"[PASS 2-8] YAML parse, augment, group, sort, expand, move required, and verify complete, returning {len(final_nodes)} root nodes")
    return final_nodes


def augment_ci_with_yaml_info_pass(
    original_ci_nodes: List,  # List[BranchNode] - original CIJobNode objects
    yaml_mappings: Dict[str, Dict],  # Mappings from YAML parsing
) -> List[TreeNodeVM]:
    """Augment CI nodes with YAML information (short names, dependencies).
    
    This pass builds a mapping from long check name to short YAML job_id,
    then updates each CIJobNode with short_job_name and yaml_dependencies.
    
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
        if isinstance(node, CIJobNode) and hasattr(node, "core_job_name"):
            core_name = str(getattr(node, "core_job_name", "") or "")

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


def move_required_jobs_to_top_pass(nodes: List[TreeNodeVM]) -> List[TreeNodeVM]:
    """Move all REQUIRED jobs to the top, keeping alphabetical order within each group.
    
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
    """Verify the final tree structure for common issues.
    
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
            job_id = node.job_id
            core_name = getattr(node, 'core_job_name', '')
            if short_name not in short_name_counts:
                short_name_counts[short_name] = []
            short_name_counts[short_name].append((job_id, core_name))
    
    duplicates = {k: v for k, v in short_name_counts.items() if len(v) > 1}
    if duplicates:
        # Separate duplicates into "OK" (different job_ids) and "problematic" (same job_ids)
        ok_duplicates = []
        problem_duplicates = []
        
        for short_name, job_list in duplicates.items():
            job_ids = [job_id for job_id, _ in job_list]
            unique_job_ids = len(set(job_ids))
            if unique_job_ids == len(job_ids):
                # All different job_ids - already disambiguated
                ok_duplicates.append(short_name)
            else:
                # Some job_ids are the same - this is a real duplicate
                core_names = [core_name for _, core_name in job_list]
                problem_duplicates.append((short_name, core_names))
        
        # Only warn about true duplicates (same job_ids)
        if problem_duplicates:
            logger.warning(f"[verify_tree_structure_pass] ⚠️  Found {len(problem_duplicates)} duplicate short names with SAME job_ids:")
            for short_name, core_names in problem_duplicates[:10]:  # Show first 10
                logger.warning(f"[verify_tree_structure_pass]    - '{short_name}' used by: {core_names}")
            if len(problem_duplicates) > 10:
                logger.warning(f"[verify_tree_structure_pass]    ... and {len(problem_duplicates) - 10} more duplicates")
        # Success cases: don't log to avoid verbose output
    # Success case: don't log to avoid verbose output
    
    # Check 4: Verify specific important jobs have short names
    important_jobs = ["Build and Test - dynamo", "dynamo-status-check", "backend-status-check"]
    for important_job in important_jobs:
        found = False
        for node in original_ci_nodes:
            if isinstance(node, CIJobNode):
                core_name = getattr(node, 'core_job_name', '')
                job_id = getattr(node, 'job_id', '')
                display_name = getattr(node, 'display_name', '')
                short_name = getattr(node, 'short_job_name', '')
                
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
        label = str(getattr(child, 'label_html', '') or '')
        # Check for status indicators in the label HTML
        # Also check for red color (#c83a3a) which indicates failure
        if '✗' in label or 'failure' in label.lower() or '#c83a3a' in label:
            has_failure = True
        elif 'success' in label.lower() or 'octicon-check' in label:
            success_count += 1
        elif '⏳' in label or 'running' in label.lower() or 'progress' in label.lower():
            has_running = True
        elif 'pending' in label.lower() or ('border-radius: 999px; background-color: #8c959f' in label and '•' in label):
            # Pending has a gray circle background with white dot
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


def move_jobs_by_prefix_pass(
    nodes: List[TreeNodeVM],
    prefix: str,
    parent_name: str,
    parent_label: str = "",
    create_if_has_children: bool = False,
) -> List[TreeNodeVM]:
    """Generic pass to move jobs matching a prefix under a parent node.
    
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
    """Sort nodes alphabetically by job name (recursively).
    
    Sorts nodes at each level by their job_name or label_html for consistent display.
    Special nodes (like status-check jobs) can be preserved at specific positions if needed.
    
    Args:
        nodes: List of TreeNodeVM nodes to sort
        
    Returns:
        Sorted list of TreeNodeVM nodes (children are also recursively sorted)
    """
    def sort_key(node: TreeNodeVM) -> tuple:
        """Generate sort key for a node.
        
        Returns:
            Tuple of (priority, name) where priority determines order:
            - 0: Regular jobs (sorted alphabetically)
        """
        job_name = str(getattr(node, "job_name", "") or "")
        # Use job_name if available, otherwise extract from label_html
        name = job_name if job_name else str(getattr(node, "label_html", "") or "")
        return (0, name.lower())
    
    # Sort current level
    sorted_nodes = sorted(nodes, key=sort_key)
    
    # Recursively sort children
    result = []
    for node in sorted_nodes:
        children = list(getattr(node, "children", None) or [])
        if children:
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
                workflow_name=node.workflow_name,
                variant=node.variant,
                pr_number=node.pr_number,
            ))
        else:
            result.append(node)
    
    return result


def _hash10(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


_ACTIONS_JOB_ID_RE = re.compile(r"/job/([0-9]+)(?:$|[/?#])")


def extract_actions_job_id_from_url(url: str) -> str:
    """Best-effort extraction of the numeric job id from GitHub Actions job URLs."""
    try:
        m = _ACTIONS_JOB_ID_RE.search(str(url or ""))
        return str(m.group(1)) if m else ""
    except Exception:
        return ""


def disambiguate_check_run_name(name: str, url: str, *, name_counts: Dict[str, int]) -> str:
    """If multiple runs share the same name, add a stable suffix so the UI doesn't show duplicates."""
    try:
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
    except Exception:
        return str(name or "")


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
        import re
        try:
            m = re.findall(r"\b[0-9a-f]{7,40}\b", str(s or ""), re.IGNORECASE)
            if not m:
                return ""
            return str(m[-1])[:7].lower()
        except Exception:
            return ""

    def _repo_token_from_key(s: str) -> str:
        import re
        try:
            txt = str(s or "")
            m = re.search(r"\b(?:PRStatus|CI|repo):([^:>]+)", txt)
            if not m:
                return ""
            repo = str(m.group(1) or "").strip()
            repo = re.sub(r"[^a-zA-Z0-9._-]+", "-", repo).strip("-").lower()
            return repo[:32]
        except Exception:
            return ""

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
        
        has_children = node.collapsible and node.children
        has_raw_content = node.collapsible and node.raw_html_content
        is_collapsible = has_children or has_raw_content
        
        # Check if this is a repository node (for spacing)
        is_repo_node = node_key.startswith("repo:")
        
        # Add 'leaf' class for nodes without children, 'repo-node' for repository nodes
        li_classes = []
        if not is_collapsible:
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
            
            parts.append('</details>\n')
        else:
            # Leaf node - just render the label
            parts.append(node.label_html or "")
            # Render raw HTML content if present (e.g., snippet <pre> blocks for non-collapsible nodes)
            if node.raw_html_content:
                parts.append(node.raw_html_content)
        
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
        try:
            m = _HEX_SHA_RE.findall(str(s or ""))
            if not m:
                return ""
            # Prefer the last SHA-ish token (often the most specific like branch head SHA).
            return str(m[-1])[:7].lower()
        except Exception:
            return ""

    def _repo_token_from_key(s: str) -> str:
        """Extract a repo/dir token for URL readability (not uniqueness)."""
        try:
            txt = str(s or "")
        except Exception:
            return ""
        try:
            # Match PRStatus:, CI:, or repo: patterns
            m = re.search(r"\b(?:PRStatus|CI|repo):([^:>]+)", txt)
            if not m:
                return ""
            repo = str(m.group(1) or "").strip()
            repo = re.sub(r"[^a-zA-Z0-9._-]+", "-", repo).strip("-").lower()
            return repo[:32]
        except Exception:
            return ""

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
            tri = _noncollapsible_icon_html(getattr(node, "noncollapsible_icon", ""))

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
    try:
        s = int(round(float(seconds)))
    except Exception:
        return ""
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
    try:
        x = str(s or "").strip()
        if not x:
            return None
        if x.endswith("Z"):
            return datetime.fromisoformat(x[:-1] + "+00:00")
        return datetime.fromisoformat(x)
    except Exception:
        return None


def _status_norm_from_actions_step(status: str, conclusion: str) -> str:
    s = (status or "").strip().lower()
    c = (conclusion or "").strip().lower()
    if c in (CIStatus.SUCCESS.value, CIStatus.NEUTRAL.value, CIStatus.SKIPPED.value):
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
    try:
        steps = job.get("steps") if isinstance(job, dict) else None
        if not isinstance(steps, list) or not steps:
            return []
    except Exception:
        return []

    def _dur(st: Dict[str, object]) -> str:
        a = _parse_iso_utc(str(st.get("started_at", "") or ""))
        b = _parse_iso_utc(str(st.get("completed_at", "") or ""))
        if not a or not b:
            return ""
        try:
            return _format_duration_short((b - a).total_seconds())
        except Exception:
            return ""

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
    try:
        steps = job.get("steps") if isinstance(job, dict) else None
        if not isinstance(steps, list) or not steps:
            return []
    except Exception:
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
        try:
            a = _parse_iso_utc(str(st.get("started_at", "") or ""))
            b = _parse_iso_utc(str(st.get("completed_at", "") or ""))
            if a and b:
                dt_s = float((b - a).total_seconds())
        except Exception:
            dt_s = None

        # Selection rule:
        # - always include failures
        # - if min_seconds <= 0: include all non-failing steps (even if duration is missing)
        # - otherwise include only steps >= threshold
        if status_norm != "failure":
            try:
                if float(min_seconds) <= 0.0:
                    pass
                elif dt_s is None or float(dt_s) < float(min_seconds):
                    continue
            except Exception:
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
    min_seconds: float = 30.0,
    ttl_s: int = 7 * 24 * 3600,
) -> List[Tuple[str, str, str]]:
    """Fetch job details (cached) and return long-running steps (duration >= min_seconds)."""
    if not github_api:
        return []
    jid = extract_actions_job_id_from_url(str(job_url or ""))
    if not jid:
        return []
    try:
        job = github_api.get_actions_job_details_cached(
            owner="ai-dynamo", repo="dynamo", job_id=jid, ttl_s=int(ttl_s)
        ) or {}
    except Exception:
        job = {}
    if not isinstance(job, dict):
        return []
    return actions_job_steps_over_threshold_from_actions_job(job, min_seconds=float(min_seconds))


def ci_subsection_tuples_for_job(
    *,
    github_api: Optional["GitHubAPIClient"],
    job_name: str,
    job_url: str,
    raw_log_path: Optional[Path],
    duration_seconds: float,
    is_required: bool,
    long_job_threshold_s: float = 10.0 * 60.0,
    step_min_s: float = 30.0,
) -> List[Tuple[str, str, str]]:
    """Shared rule: return child tuples for CI subsections.

    Terminology (official):
    - "subsections" is the umbrella term for child rows under a job/check.
    - "phases" are a *special-case* kind of subsection for `Build and Test - dynamo`
      (we keep the name "phases" in code where it’s specific to that job).

    - Build and Test - dynamo: return the dedicated phases (status+duration) from Actions job `steps[]`.
    - Other long-running Actions jobs: return job steps >= step_min_s.
    """
    nm = str(job_name or "").strip()
    if not nm:
        return []
    if nm == "Build and Test - dynamo":
        try:
            phases = build_and_test_dynamo_phase_tuples(
                github_api=github_api,
                job_url=str(job_url or ""),
                raw_log_path=raw_log_path,
                is_required=bool(is_required),
            )
            # Also include *non-phase* steps so we can surface useful failures like
            # "Copy test report..." without duplicating the phase rows.
            #
            # Policy for REQUIRED jobs: show all failing steps + steps >= threshold; ignore the rest.
            steps = actions_job_step_tuples(
                github_api=github_api,
                job_url=str(job_url or ""),
                min_seconds=float(step_min_s),
            )

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
            return out
        except Exception:
            return []

    # REQUIRED jobs: always show failing steps + steps >= threshold (even if job isn't "long-running").
    if bool(is_required):
        return actions_job_step_tuples(github_api=github_api, job_url=str(job_url or ""), min_seconds=float(step_min_s))

    # Non-required jobs: show steps only for long-running jobs (avoid noise).
    try:
        if float(duration_seconds or 0.0) < float(long_job_threshold_s):
            return []
    except Exception:
        return []

    return actions_job_step_tuples(github_api=github_api, job_url=str(job_url or ""), min_seconds=float(step_min_s))


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
    try:
        steps = job.get("steps") if isinstance(job, dict) else None
        if isinstance(steps, list):
            for st in steps:
                if isinstance(st, dict) and str(st.get("name", "") or "") == str(step_name or ""):
                    step = st
                    break
    except Exception:
        step = None
    if not isinstance(step, dict):
        return ""

    a = _parse_iso_utc(str(step.get("started_at", "") or ""))
    b = _parse_iso_utc(str(step.get("completed_at", "") or ""))
    if not a or not b:
        return ""

    try:
        # Shared library (dependency-light): `dynamo-utils/ci_log_errors/`
        from ci_log_errors import extract_error_snippet_from_text  # local import (avoid circulars)
    except Exception:
        return ""

    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""

    # Filter lines by timestamp window (raw logs include ISO-8601 timestamps).
    kept: List[str] = []
    for ln in (text.splitlines() or []):
        # Most lines are prefixed with an ISO timestamp; ignore unparsable lines.
        ts = None
        try:
            # Heuristic: take the first token, strip any trailing 'Z'.
            head = (ln.split(" ", 1)[0] if " " in ln else ln).strip()
            ts = _parse_iso_utc(head)
        except Exception:
            ts = None
        if not ts:
            continue
        if ts < a or ts > b:
            continue
        kept.append(ln)
    if not kept:
        return ""
    return extract_error_snippet_from_text("\n".join(kept))

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
) -> str:
    """Shared status icon HTML (match all dashboards).

    Conventions:
    - REQUIRED success: green filled circle-check
    - REQUIRED failure: red filled circle X
    - non-required success: small check
    - non-required failure: small X
    """
    s = (status_norm or "").strip().lower()

    if s == CIStatus.SUCCESS:
        if bool(is_required):
            # REQUIRED: filled green circle-check.
            out = (
                f'<span style="color: {COLOR_GREEN}; display: inline-flex; vertical-align: text-bottom;">'
                '<svg aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="12" height="12" '
                'data-view-component="true" class="octicon octicon-check-circle-fill" fill="currentColor">'
                '<path fill-rule="evenodd" '
                'd="M8 16A8 8 0 108 0a8 8 0 000 16zm3.78-9.78a.75.75 0 00-1.06-1.06L7 9.94 5.28 8.22a.75.75 0 10-1.06 1.06l2 2a.75.75 0 001.06 0l4-4z">'
                "</path></svg></span>"
            )
        else:
            # Optional success: small check icon (preferred).
            out = (
                f'<span style="color: {COLOR_GREEN}; display: inline-flex; vertical-align: text-bottom;">'
                f'{_octicon_svg(path_d="M13.78 4.22a.75.75 0 00-1.06 0L6.75 10.19 3.28 6.72a.75.75 0 10-1.06 1.06l4 4a.75.75 0 001.06 0l7.5-7.5a.75.75 0 000-1.06z", name="octicon-check")}'
                "</span>"
            )
        if bool(warning_present):
            # Descendant failures: show a red X appended to the success icon.
            out += '<span style="color: #57606a; font-size: 11px; margin: 0 2px;">/</span>'
            if bool(required_failure):
                # REQUIRED descendant failure: filled red circle X.
                out += (
                    '<span style="display: inline-flex; align-items: center; justify-content: center; '
                    f'width: 12px; height: 12px; margin-left: 2px; border-radius: 999px; background-color: {COLOR_RED}; '
                    'color: #ffffff; font-size: 10px; font-weight: 900; line-height: 1;">✗</span>'
                )
            else:
                # Optional descendant failure: small X.
                out += (
                    f'<span style="color: {COLOR_RED}; display: inline-flex; vertical-align: text-bottom; margin-left: 2px;">'
                    f'{_octicon_svg(path_d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 11-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z", name="octicon-x")}'
                    "</span>"
                )
        return out
    if s in {CIStatus.SKIPPED, CIStatus.NEUTRAL}:
        # GitHub-like "skipped": grey circle with a slash.
        return (
            '<span style="color: #8c959f; display: inline-flex; vertical-align: text-bottom;">'
            '<svg aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="12" height="12" '
            'data-view-component="true" class="octicon octicon-circle-slash" fill="currentColor">'
            '<path fill-rule="evenodd" '
            'd="M8 16A8 8 0 108 0a8 8 0 000 16ZM1.5 8a6.5 6.5 0 0110.364-5.083l-8.947 8.947A6.473 6.473 0 011.5 8Zm3.136 5.083 8.947-8.947A6.5 6.5 0 014.636 13.083Z">'
            "</path></svg></span>"
        )
    if s == CIStatus.FAILURE:
        if bool(is_required or required_failure):
            # REQUIRED: filled red circle X.
            return (
                '<span style="display: inline-flex; align-items: center; justify-content: center; '
                f'width: 12px; height: 12px; border-radius: 999px; background-color: {COLOR_RED}; '
                'color: #ffffff; font-size: 10px; font-weight: 900; line-height: 1;">✗</span>'
            )
        # Optional failure: small X.
        return (
            f'<span style="color: {COLOR_RED}; display: inline-flex; vertical-align: text-bottom;">'
            f'{_octicon_svg(path_d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 11-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z", name="octicon-x")}'
            "</span>"
        )
    if s == CIStatus.IN_PROGRESS:
        return f'<span style="color: {COLOR_GREY};">⏳</span>'
    if s == CIStatus.PENDING:
        return (
            '<span style="display: inline-flex; align-items: center; justify-content: center; '
            'width: 12px; height: 12px; border-radius: 999px; background-color: #8c959f; '
            'color: #ffffff; font-size: 10px; font-weight: 900; line-height: 1;">•</span>'
        )
    if s == CIStatus.CANCELLED:
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


# Shared library (dependency-light): `dynamo-utils/ci_log_errors/`
from ci_log_errors import render_error_snippet_html as _format_snippet_html  # shared implementation
from ci_log_errors import categorize_error_snippet_text as _snippet_categories


def github_api_stats_rows(
    *,
    github_api: Optional["GitHubAPIClient"],
    max_github_api_calls: Optional[int] = None,
    mode: str = "",
    mode_reason: str = "",
    extra_cache_stats: Optional[Dict[str, Any]] = None,
    top_n: int = 15,
) -> List[Tuple[str, Optional[str]]]:
    """Build human-readable GitHub API statistics rows for the footer.

    Returns rows suitable for `page_stats`, including section headers ("## ...") and multiline values.
    """
    rows: List[Tuple[str, Optional[str]]] = []
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
    try:
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
    except Exception:
        pass

    rows.append(("## GitHub API", None))
    rows.append(("Budget & mode", _fmt_kv(budget) or "(none)"))

    # REST summary
    try:
        rest_summary = {
            "calls": int(rest.get("total") or 0),
            "ok": int(rest.get("success_total") or 0),
            "errors": int(rest.get("error_total") or 0),
            "time_total": f"{float(rest.get('time_total_s') or 0.0):.2f}s",
        }
    except Exception:
        rest_summary = {"calls": 0, "ok": 0, "errors": 0, "time_total": "0.00s"}
    rows.append(("REST summary", _fmt_kv(rest_summary)))

    rows.append(
        (
            f"REST by category (top {int(top_n)})",
            _fmt_top_counts_and_time(by_label=by_label, time_by_label_s=time_by_label_s, n=int(top_n)),
        )
    )

    # Errors
    by_status = (errs or {}).get("by_status") if isinstance(errs, dict) else {}
    if isinstance(by_status, dict) and by_status:
        items = list(by_status.items())[:8]
        rows.append(("REST errors by status", ", ".join([f"{k}={v}" for k, v in items])))
    else:
        rows.append(("REST errors by status", "(none)"))

    bls = (errs or {}).get("by_label_status") if isinstance(errs, dict) else {}
    rows.append(("REST errors by category+status", _fmt_error_by_label_status(bls if isinstance(bls, dict) else {}, n=int(top_n))))

    try:
        last = (errs or {}).get("last") if isinstance(errs, dict) else None
        last_label = (errs or {}).get("last_label") if isinstance(errs, dict) else ""
        if isinstance(last, dict) and last.get("status"):
            code = last.get("status")
            body = str(last.get("body") or "").strip()
            if len(body) > 160:
                body = body[:160].rstrip() + "…"
            url = str(last.get("url") or "").strip()
            s = f"{code}: {body}" if body else str(code)
            if last_label:
                s += f"\nlabel: {last_label}"
            if url:
                s += f"\nurl: {url}"
            rows.append(("Last REST error", s))
    except Exception:
        pass

    # Cache summary (keep small; details are already available in README if needed)
    try:
        cache_summary = {
            "hits": int(cache.get("hits_total") or 0),
            "misses": int(cache.get("misses_total") or 0),
            "writes_ops": int(cache.get("writes_ops_total") or 0),
            "writes_entries": int(cache.get("writes_entries_total") or 0),
        }
    except Exception:
        cache_summary = {"hits": 0, "misses": 0, "writes_ops": 0, "writes_entries": 0}
    rows.append(("Cache summary", _fmt_kv(cache_summary)))

    # Include the commit-history-only github_actions_status_cache (if provided).
    if isinstance(extra_cache_stats, dict) and extra_cache_stats:
        try:
            gha = extra_cache_stats.get("github_actions_status_cache") or {}
            if isinstance(gha, dict) and gha:
                d = {
                    "total_shas": int(gha.get("total_shas", 0) or 0),
                    "fetched_shas": int(gha.get("fetched_shas", 0) or 0),
                    "hit_fresh": int(gha.get("cache_hit_fresh", 0) or 0),
                    "stale_refresh": int(gha.get("cache_stale_refresh", 0) or 0),
                    "miss_fetch": int(gha.get("cache_miss_fetch", 0) or 0),
                    "miss_no_fetch": int(gha.get("cache_miss_no_fetch", 0) or 0),
                }
                rows.append(("Actions-status cache (this run)", _fmt_kv(d)))
        except Exception:
            pass

    return rows


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
    try:
        seed_s = str(dom_id_seed or "")
    except Exception:
        seed_s = ""
    try:
        m = re.findall(r"\b[0-9a-f]{7,40}\b", seed_s, flags=re.IGNORECASE)
        sha7 = (str(m[-1])[:7].lower() if m else "")
    except Exception:
        sha7 = ""
    jobid = ""
    try:
        m2 = re.search(r"/job/([0-9]{5,})", seed_s)
        jobid = str(m2.group(1)) if m2 else ""
    except Exception:
        jobid = ""
    try:
        suffix = _hash10(seed_s)[:6]
    except Exception:
        suffix = "x"
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
    may_be_required: bool = False,
    duration: str = "",
    log_url: str = "",
    raw_log_href: str = "",
    raw_log_size_bytes: int = 0,
    error_snippet_text: str = "",
    error_snippet_categories: Optional[List[str]] = None,
    required_failure: bool = False,
    warning_present: bool = False,
    short_job_name: str = "",  # Short YAML job name (e.g., "build-test")
    yaml_dependencies: Optional[List[str]] = None,  # List of dependencies from YAML
) -> str:
    # Expected placeholder checks:
    # - Use the placeholder symbol as the *icon* (instead of the generic pending dot),
    #   since the symbol already communicates "not yet returned by the API".
    # - Also suppress the redundant trailing "— ◇" marker in the label.
    is_expected_placeholder = bool(str(display_name or "").strip() == EXPECTED_CHECK_PLACEHOLDER_SYMBOL)
    if is_expected_placeholder:
        icon = (
            '<span style="display: inline-flex; align-items: center; justify-content: center; '
            f'width: 12px; height: 12px; color: #8c959f; font-size: 12px; font-weight: 900;">{EXPECTED_CHECK_PLACEHOLDER_SYMBOL}</span>'
        )
    else:
        icon = status_icon_html(
            status_norm=status_norm,
            is_required=is_required,
            required_failure=required_failure,
            warning_present=warning_present,
        )

    def _format_arch_text(raw_text: str) -> str:
        """Format the job text with arch styling.

        - If the job contains an explicit arch token:
          - `(arm64)` / `(aarch64)` -> keep the original token `(arm64)` and append `; aarch64`
          - `(amd64)`              -> keep the original token `(amd64)` and append `; x86_64`
        - Otherwise, keep normal styling (no special casing).
        
        Note: Color styling is applied to the entire line, not within this function.
        """
        raw = str(raw_text or "")
        # Only rewrite when we see an explicit arch token (avoid surprising renames).
        m = re.search(r"\((arm64|aarch64|amd64)\)", raw, flags=re.IGNORECASE)
        if not m:
            return html.escape(raw)

        arch = str(m.group(1) or "").strip().lower()
        # Rewrite arch token casing/label.
        if arch in {"arm64", "aarch64"}:
            # Normalize to "(arm64)" and append "; aarch64" immediately after the token.
            raw2 = re.sub(r"\(\s*(arm64|aarch64)\s*\)", "(arm64)", raw, flags=re.IGNORECASE)
            raw2 = re.sub(r"\(\s*arm64\s*\)(?!\s*;\s*aarch64\b)", "(arm64); aarch64", raw2, flags=re.IGNORECASE)
            return html.escape(raw2)
        if arch == "amd64":
            raw2 = re.sub(r"\(\s*amd64\s*\)", "(amd64)", raw, flags=re.IGNORECASE)
            raw2 = re.sub(r"\(\s*amd64\s*\)(?!\s*;\s*x86_64\b)", "(amd64); x86_64\u00a0", raw2, flags=re.IGNORECASE)  # Non-breaking space for alignment
            return html.escape(raw2)
        return html.escape(raw)
    
    # Detect architecture for line-wide color styling
    def _get_arch_color(text: str) -> str:
        """Return the color for the entire line based on architecture."""
        m = re.search(r"\((arm64|aarch64|amd64)\)", str(text or ""), flags=re.IGNORECASE)
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
        try:
            m = re.search(r"/job/([0-9]{5,})", str(log_url))
            job_num = str(m.group(1)) if m else ""
        except Exception:
            job_num = ""

    links = ""
    if log_url:
        log_label = f"[{job_num}.log]" if job_num else "[log]"
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
        try:
            seed_s = str(dom_id_seed or "")
        except Exception:
            seed_s = ""
        try:
            m = re.findall(r"\b[0-9a-f]{7,40}\b", seed_s, flags=re.IGNORECASE)
            sha7 = (str(m[-1])[:7].lower() if m else "")
        except Exception:
            sha7 = ""
        jobid = ""
        try:
            m2 = re.search(r"/job/([0-9]{5,})", seed_s)
            jobid = str(m2.group(1)) if m2 else ""
        except Exception:
            jobid = ""
        try:
            suffix = _hash10(seed_s)[:6]
        except Exception:
            suffix = "x"
        if jobid:
            snippet_key = f"s.{sha7}.j{jobid}" if sha7 else f"s.j{jobid}"
        else:
            snippet_key = f"s.{sha7}.{suffix}" if sha7 else f"s.{suffix}"
        
        # Show category pills and command pill with snippet key for click handling
        cats = error_snippet_categories if error_snippet_categories else _snippet_categories(snippet_text)
        for c in cats[:3]:
            links += _tag_pill_html(text=c, monospace=False, kind="category", snippet_key=snippet_key)
        cmd = _snippet_first_command(snippet_text)
        if cmd:
            links += _tag_pill_html(text=cmd, monospace=True, kind="command", snippet_key=snippet_key)

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
    try:
        jid = extract_actions_job_id_from_url(str(job_url or ""))
        if github_api and jid:
            job = github_api.get_actions_job_details_cached(owner="ai-dynamo", repo="dynamo", job_id=jid, ttl_s=600) or {}
            if isinstance(job, dict):
                phases3 = build_and_test_dynamo_phases_from_actions_job(job) or []
    except Exception:
        phases3 = []

    return [(str(n), str(d), str(s)) for (n, d, s) in (phases3 or [])]


