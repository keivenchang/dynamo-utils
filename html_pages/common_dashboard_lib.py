#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Common dashboard helpers shared by HTML generators under dynamo-utils.PRODUCTION/html_pages/.

FACADE MODULE: This file re-exports symbols from focused sub-modules for backward
compatibility.  All new code should import from the sub-modules directly:

    ci_status_icons   - SVG icons, badges, compact CI summary HTML, color constants
    tree_passes       - TreeNodeVM dataclass, all *_pass pipeline functions
    tree_rendering    - HTML tree rendering, check-line HTML, CI subsection tuples
    pytest_parsing    - Pytest log parsing, test-result extraction, snippet caching
    ci_stats          - GitHub/GitLab API stats rows, page-level stats

======================================================================================
NODE HIERARCHY REFERENCE
======================================================================================

This section documents the tree node structure used across all dashboards.

Node Hierarchy (with creators):
--------------------------------
```
LocalRepoNode (repository directory)                         <- Created by: show_local_branches.py
+-- BranchInfoNode (individual branch)                        <- Created by: show_local_branches.py, show_remote_branches.py
   +-- BranchCommitMessageNode (commit message + PR link)     <- Created by: BranchInfoNode.to_tree_vm()
   +-- BranchMetadataNode (timestamps / age)                  <- Created by: BranchInfoNode.to_tree_vm()
   +-- ConflictWarningNode                                    <- Created by: BranchInfoNode.to_tree_vm() (when pr.conflict_message exists)
   +-- BlockedMessageNode                                     <- Created by: BranchInfoNode.to_tree_vm() (when pr.blocking_message exists)
   +-- PRStatusWithJobsNode (CI status for PRs)               <- Created by: add_pr_status_node_pass
   |  +-- CIJobNode (CI check/job)                            <- Created by: build_ci_nodes_from_pr()
   |  |  +-- CIJobNode (nested steps)                         <- Created by: add_job_steps_and_tests_pass
   |  |  |  +-- PytestTestNode (pytest tests)                 <- Created by: pytest_slowest_tests_from_raw_log (within add_job_steps_and_tests_pass)
   |  +-- RerunLinkNode                                       <- Created by: build_ci_nodes_from_pr() (on CI failure)
   +-- (no PR)                                                <- BranchInfoNode only (no workflow-status child node today)

Note: show_commit_history.py uses a different model:
  - Creates CIJobNode directly (no BranchInfoNode wrapper)
  - Renders commit metadata in the Jinja2 template (not via BranchCommitMessageNode)
  - Displays commits as a flat table (not a nested tree)
  - Still uses run_all_passes() for consistent CI job processing
```
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# ci_status_icons
# ---------------------------------------------------------------------------
from ci_status_icons import (  # noqa: F401
    COLOR_GREEN,
    COLOR_RED,
    COLOR_GREY,
    COLOR_YELLOW,
    EXPECTED_CHECK_PLACEHOLDER_SYMBOL,
    KNOWN_ERROR_MARKERS,
    PASS_PLUS_STYLE,
    compact_ci_summary_html,
    ci_status_icon_context,
    mandatory_badge_html,
    required_badge_html,
    status_icon_html,
)

# ---------------------------------------------------------------------------
# tree_passes
# ---------------------------------------------------------------------------
from tree_passes import (  # noqa: F401
    TreeNodeVM,
    ci_should_expand_by_default,
    create_dummy_nodes_from_yaml_pass,
    run_all_passes,
    add_pr_status_node_pass,
    prefetch_actions_job_details_pass,
    add_job_steps_and_tests_pass,
    convert_branch_nodes_to_tree_vm_pass,
    parse_workflow_yaml_and_build_mapping_pass,
    augment_ci_with_yaml_info_pass,
    move_jobs_by_prefix_batch_pass,
    sort_nodes_by_name_pass,
    expand_required_failure_descendants_pass,
    move_required_jobs_to_top_pass,
    verify_tree_structure_pass,
    verify_job_details_pass,
)

# ---------------------------------------------------------------------------
# tree_rendering
# ---------------------------------------------------------------------------
from tree_rendering import (  # noqa: F401
    _dom_id_from_node_key,
    _hash10,
    _small_link_html,
    _triangle_html,
    _triangle_placeholder_html,
    _noncollapsible_icon_html,
    extract_actions_job_id_from_url,
    disambiguate_check_run_name,
    render_tree_divs,
    render_tree_pre_lines,
    build_and_test_dynamo_phases_from_actions_job,
    actions_job_steps_over_threshold_from_actions_job,
    actions_job_step_tuples,
    is_build_test_job,
    is_python_test_step,
    job_name_wants_pytest_details,
    ci_subsection_tuples_for_job,
    check_line_html,
    build_and_test_dynamo_phase_tuples,
    _tag_pill_html,
    _snippet_first_command,
    _create_snippet_tree_node,
    _format_bytes_short,
)

# ---------------------------------------------------------------------------
# pytest_parsing
# ---------------------------------------------------------------------------
from pytest_parsing import (  # noqa: F401
    GRAFANA_TEST_URL_TEMPLATE,
    PYTEST_SLOWEST_DURATIONS_REGEX,
    PYTEST_SUMMARY_REGEX,
    _parse_iso_utc,
    pytest_slowest_tests_from_raw_log,
    pytest_results_from_raw_log,
    step_window_snippet_from_cached_raw_log,
)

# ---------------------------------------------------------------------------
# ci_stats
# ---------------------------------------------------------------------------
from ci_stats import (  # noqa: F401
    _format_ttl_duration,
    github_api_stats_rows,
    gitlab_api_stats_rows,
    build_page_stats,
)

# ---------------------------------------------------------------------------
# Re-export common_types.CIStatus for backward compatibility
# ---------------------------------------------------------------------------
from common_types import CIStatus  # noqa: F401
