# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tree pass pipeline: TreeNodeVM dataclass and all *_pass transformation functions."""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
import sys
import time
import yaml
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from datetime import datetime, timezone

from common_types import CIStatus
from common_github import GitHubAPIClient
from ci_status_icons import KNOWN_ERROR_MARKERS

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import (
    DEFAULT_UNSTABLE_TTL_S,
    DEFAULT_STABLE_TTL_S,
    DEFAULT_OPEN_PRS_TTL_S,
    DEFAULT_CLOSED_PRS_TTL_S,
    DEFAULT_NO_PR_TTL_S,
    DEFAULT_RAW_LOG_TEXT_TTL_S,
)

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


def ci_should_expand_by_default(*, rollup_status: str, has_required_failure: bool, has_required_in_progress: bool = False) -> bool:
    """Shared UX rule: expand only when something truly needs attention.

    - expand for required failures (red X icon)
    - do NOT auto-expand long/step-heavy jobs by default (even if they have subsections)
    - expand for REQUIRED in-progress/pending states so "BUILDING" remains visible
    - do NOT auto-expand for optional-only failures/in-progress, cancelled, unknown-only leaves, or all-green trees
    """
    if bool(has_required_failure):
        return True
    # Only expand if REQUIRED jobs are in-progress (not just optional ones)
    if bool(has_required_in_progress):
        return True
    return False

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
    
    logger.debug(f"[run_all_passes] Starting with {len(ci_nodes)} nodes")
    
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
        ("Pre Merge / ", "Pre Merge", "Pre Merge CI", True),
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
    
    logger.debug(f"[run_all_passes] All passes complete (1-8), returning {len(final_nodes)} root nodes")
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
    
    logger.debug(f"[PASS -1] add_pr_status_node_pass complete")
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

    logger.debug(f"[prefetch_actions_job_details_pass] Extracting run_ids from {len(ci_nodes)} nodes")

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
        logger.debug(f"[prefetch_actions_job_details_pass] No run_ids found, skipping batch fetch")
        return ci_nodes

    logger.debug(f"[prefetch_actions_job_details_pass] Batch fetching job details for {len(run_ids)} unique runs")

    try:
        # Batch fetch all jobs for these runs (populates cache)
        job_map = github_api.get_actions_runs_jobs_batched(
            owner="ai-dynamo",
            repo="dynamo",
            run_ids=list(run_ids),
            ttl_s=30 * 24 * 3600,  # 30 days cache
        )
        logger.debug(f"[prefetch_actions_job_details_pass] Prefetched {len(job_map)} job details into cache")
    except Exception as e:
        logger.warning(f"[prefetch_actions_job_details_pass] Batch fetch failed: {e}")

    # Also prefetch run metadata (run_attempt, status, etc.) for rerun detection
    logger.debug(f"[prefetch_actions_job_details_pass] Batch fetching run metadata for {len(run_ids)} unique runs")
    try:
        from common_github.api.actions_run_metadata_cached import get_run_metadata_cached
        metadata_count = 0
        for run_id in run_ids:
            metadata = get_run_metadata_cached(
                api=github_api,
                owner="ai-dynamo",
                repo="dynamo",
                run_id=run_id,
                ttl_s=30 * 24 * 3600,  # 30 days for completed, 5m for in-progress
                skip_fetch=False,
            )
            if metadata:
                metadata_count += 1
        logger.debug(f"[prefetch_actions_job_details_pass] Prefetched {metadata_count} run metadata entries into cache")
    except Exception as e:
        logger.warning(f"[prefetch_actions_job_details_pass] Run metadata fetch failed: {e}")

    logger.debug(f"[prefetch_actions_job_details_pass] Complete")
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
    from tree_rendering import ci_subsection_tuples_for_job, is_python_test_step, job_name_wants_pytest_details
    import time  # For detailed timing
    from common_branch_nodes import CIJobNode, CIStepNode, CIPytestNode, _duration_str_to_seconds
    
    logger.debug(f"[add_job_steps_and_tests_pass] Processing {len(ci_nodes)} nodes")
    
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
        for (step_name, step_dur, step_status, _step_dict) in step_tuples:
            step_name_s = str(step_name or "")
            
            # Check if this is a pytest test (has └─ prefix)
            if step_name_s.startswith("  └─ "):
                # This is a pytest test - add as child of the current "Run tests" step
                if current_test_parent:
                    test_name_full = step_name_s[len("  └─ "):]  # Remove prefix
                    
                    # Extract ONLY known error markers from test name (not pytest parameters)
                    # Pytest parameters like [tcp], [cuda12.9, amd64] should remain in the test name
                    import re
                    
                    error_categories = []
                    test_name_clean = test_name_full
                    
                    # Extract only known error markers (from right to left)
                    while True:
                        match = re.search(r'\s*\[([\w-]+)\]\s*$', test_name_clean)
                        if not match:
                            break
                        marker = match.group(1)
                        if marker in KNOWN_ERROR_MARKERS:
                            error_categories.insert(0, marker)  # Insert at front to maintain order
                            test_name_clean = test_name_clean[:match.start()]
                        else:
                            # Not a known error marker (likely pytest parameter), stop extraction
                            break
                    
                    test_node = CIPytestNode(
                        job_id=test_name_clean.strip(),  # Test name with pytest parameters, without error markers
                        display_name="",
                        status=step_status,
                        duration=step_dur,
                        log_url="",
                        children=[],
                        error_snippet_categories=error_categories if error_categories else None,
                    )
                    current_test_parent.children.append(test_node)
            else:
                # This is a regular step - use the original step name from API
                step_id = str(step_name)

                step_node = CIStepNode(
                    job_id=step_id,
                    display_name=str(step_name),
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
    
    logger.debug(f"[add_job_steps_and_tests_pass] Complete")
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
    logger.debug(f"Converting {len(ci_nodes)} BranchNode objects to TreeNodeVM")
    
    ci_info_nodes: List[TreeNodeVM] = []
    for idx, ci_node in enumerate(ci_nodes):
        # Convert BranchNode to TreeNodeVM (core_job_name is now set in to_tree_vm())
        node_vm = ci_node.to_tree_vm()
        
        # Debug: Check if core_job_name was propagated
        if idx < 5:
            core_name = node_vm.core_job_name or "<none>"
            logger.debug(f"  [{idx}] TreeNodeVM has core_job_name='{core_name}'")
        
        ci_info_nodes.append(node_vm)
    
    logger.debug(f"Converted {len(ci_info_nodes)} nodes to TreeNodeVM")
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
    
    logger.debug(f"[augment_ci_with_yaml_info_pass] Augmenting {len(original_ci_nodes)} CI nodes with YAML info")
    
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
    
    logger.debug(f"[augment_ci_with_yaml_info_pass] Built mapping with {len(long_to_short)} entries")
    
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
    
    logger.debug(f"[augment_ci_with_yaml_info_pass] Augmented {augmented_count}/{len(original_ci_nodes)} CI nodes with YAML info")
    
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
    logger.debug(f"[move_jobs_by_prefix_batch_pass] Processing {len(grouping_rules)} grouping rules in single pass")
    
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
            logger.debug(f"[move_jobs_by_prefix_batch_pass] Added {len(matched_jobs)} jobs to existing parent '{parent_name}'")
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
            logger.debug(f"[move_jobs_by_prefix_batch_pass] Created new parent '{parent_name}' with {len(matched_jobs)} jobs")
    
    logger.debug(f"[move_jobs_by_prefix_batch_pass] Returning {len(remaining_nodes)} root nodes")
    return remaining_nodes


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
    from ci_status_icons import status_icon_html
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
    logger.debug(f"[move_required_jobs_to_top] Processing {len(nodes)} root nodes")
    
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
    
    logger.debug(f"[move_required_jobs_to_top] Moved {len(required_jobs)} required jobs to top, {len(non_required_jobs)} non-required after")
    if required_jobs:
        logger.debug(f"[move_required_jobs_to_top] Required jobs at top: {[node.short_job_name or node.job_name for node in required_jobs]}")
    
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
    from tree_rendering import extract_actions_job_id_from_url
    
    # Format commit ref for logging
    commit_ref = f" (commit: {commit_sha[:7]})" if commit_sha else ""
    logger.debug(f"[verify_tree_structure_pass] Verifying tree structure ({len(tree_nodes)} root nodes){commit_ref}")
    
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
        logger.debug(f"[verify_tree_structure_pass] ✓ All REQUIRED jobs are at the top (first {first_non_required_idx or len(tree_nodes)} positions)")
    
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
            logger.debug(f"[verify_tree_structure_pass] ✓ build-test is under dynamo-status-check")
        else:
            child_names = [(c.short_job_name or '', c.job_name or '') for c in (dynamo_node.children or [])]
            logger.warning(f"[verify_tree_structure_pass] ⚠️  build-test NOT found under dynamo-status-check. Children: {child_names}")
    else:
        logger.warning(f"[verify_tree_structure_pass] ⚠️  dynamo-status-check node not found")
    
    logger.debug(f"[verify_tree_structure_pass] Verification complete")


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
    from tree_rendering import is_python_test_step, job_name_wants_pytest_details
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
    logger.debug(f"[verify_job_details_pass] Verifying job details for {len(ci_nodes)} nodes{commit_ref}")

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
        logger.debug(f"[verify_job_details_pass] ✓ All {real_github_jobs_checked} real GitHub jobs have proper details (plus {synthetic_aggregators_checked} synthetic aggregators){commit_ref}")

    logger.debug(f"[verify_job_details_pass] Verification complete: {real_github_jobs_checked} real jobs, {synthetic_aggregators_checked} synthetic aggregators")
