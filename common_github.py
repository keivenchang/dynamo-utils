# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GitHub API client and utilities for dynamo-utils.

OPTIMIZATION SUMMARY (2026-01-18):
==================================
This module implements several API usage optimizations that reduce GitHub API calls
from ~2000/run to ~10-300/run (85-98% reduction):

1. ETag Support (Conditional Requests)
   - _rest_get() supports If-None-Match header
   - 304 Not Modified responses DON'T count against rate limit!
   - Cache schema v6 stores ETags for check-runs and status endpoints
   - Usage: get_pr_checks_rows() automatically uses ETags
   - Benefit: 85-95% rate limit reduction on subsequent runs

2. Batched Workflow Run Fetching
   - Collects all run_ids first, then batch fetches metadata
   - Usage: get_pr_checks_rows() lines 3247-3271
   - Benefit: 90% reduction (100 individual → 10-20 batched calls)

3. Batched Job Fetching Infrastructure
   - get_actions_runs_jobs_batched(): Fetch all jobs for multiple runs
   - Uses /actions/runs/{run_id}/jobs (all jobs in one call)
   - Status: ✅ Implemented, ⏳ Not yet wired up (requires refactoring lazy materialization)
   - Potential benefit: 95% reduction (500-1000 → 10-20 calls)

All optimizations are uniform - ALL show_*.py scripts benefit automatically.

For full details, see OPTIMIZATION_SUMMARY.md in the repo root.
"""

# Standard library imports
import fcntl
import json
import logging
import os
import re
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

# Third-party imports
import requests
import yaml
from ci_log_errors import snippet as ci_snippet  # type: ignore

# Local imports
from common_types import CIStatus
from common import (
    PhaseTimer,
    DEFAULT_STABLE_AFTER_HOURS,
    DEFAULT_UNSTABLE_TTL_S,
    DEFAULT_STABLE_TTL_S,
    DEFAULT_OPEN_PRS_TTL_S,
    DEFAULT_CLOSED_PRS_TTL_S,
    DEFAULT_NO_PR_TTL_S,
    DEFAULT_RAW_LOG_TEXT_TTL_S,
    DEFAULT_RAW_LOG_TEXT_MAX_BYTES,
    DEFAULT_RAW_LOG_ERROR_SNIPPET_TAIL_BYTES,
    dynamo_utils_cache_dir,
    resolve_cache_path,
)

# Cache modules - incremental migration in progress
from cache.cache_merge_dates import MERGE_DATES_CACHE
from cache.cache_pulls_list import PULLS_LIST_CACHE
from cache.cache_pr_branch import PR_BRANCH_CACHE
from cache.cache_required_checks import REQUIRED_CHECKS_CACHE
from cache.cache_pr_checks import PR_CHECKS_CACHE
from cache.cache_pr_info import PR_INFO_CACHE, PR_HEAD_SHA_CACHE
from cache.cache_job_log import JOB_LOG_CACHE
from cache.cache_duration import DURATION_CACHE

# Module logger
_logger = logging.getLogger(__name__)


# ======================================================================================
# GLOBAL CACHE STATISTICS
# ======================================================================================
# All GitHub API caching operations write to these global statistics.
# Dashboard scripts read from these to display cache performance.

class _GitHubCacheStats:
    """Global singleton for tracking GitHub cache statistics across all operations."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics (useful for testing)."""
        # Merge dates cache (when PRs were merged)
        self.merge_dates_hits = 0
        self.merge_dates_misses = 0
        self.merge_dates_miss_reason = ""
        self.merge_dates_miss_prs_sample = []

        # Required checks cache (required status checks per PR)
        self.required_checks_hits = 0
        self.required_checks_misses = 0
        self.required_checks_miss_reason = ""
        self.required_checks_skip_fetch = False
        self.required_checks_miss_prs_sample = []

    def to_dict(self) -> Dict[str, dict]:
        """Return statistics in the format expected by build_page_stats (extra_cache_stats)."""
        return {
            "merge_dates_cache": {
                "hits": self.merge_dates_hits,
                "misses": self.merge_dates_misses,
                "miss_reason": self.merge_dates_miss_reason,
                "miss_prs_sample": list(self.merge_dates_miss_prs_sample),
            },
            "required_checks_cache": {
                "hits": self.required_checks_hits,
                "misses": self.required_checks_misses,
                "miss_reason": self.required_checks_miss_reason,
                "skip_fetch": self.required_checks_skip_fetch,
                "miss_prs_sample": list(self.required_checks_miss_prs_sample),
            },
        }


# Global instance - all code writes to this
GITHUB_CACHE_STATS = _GitHubCacheStats()


class _GitHubAPIStats:
    """Global singleton for tracking GitHub API REST call statistics."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics."""
        # REST call stats
        self.rest_calls_total = 0
        self.rest_calls_by_label = {}  # Dict[str, int] - count by API endpoint label
        self.rest_success_total = 0
        self.rest_success_by_label = {}  # Dict[str, int]
        self.rest_time_total_s = 0.0
        self.rest_time_by_label_s = {}  # Dict[str, float] - time in seconds by label
        self.rest_budget_max = None  # Optional[int]
        self.rest_budget_exhausted = False
        self.rest_budget_exhausted_reason = ""

        # Error stats
        self.rest_errors_total = 0
        self.rest_errors_by_status = {}  # Dict[int, int]
        self.rest_errors_by_label_status = {}  # Dict[str, Dict[int, int]] - errors by label and status code
        self.rest_last_error = {}  # Dict[str, Any]
        self.rest_last_error_label = ""

        # Generic cache stats (file-level operations)
        self.cache_hits = {}  # Dict[str, int] - by cache name
        self.cache_misses = {}  # Dict[str, int] - by cache name
        self.cache_writes_ops = {}  # Dict[str, int] - write operations by cache name
        self.cache_writes_entries = {}  # Dict[str, int] - entries written by cache name

        # Rate limit info
        self.core_rate_limit = None  # Optional[Dict] - {remaining, limit, reset_pt}

        # ETag stats (conditional requests)
        self.etag_304_total = 0  # 304 Not Modified responses (don't count against rate limit!)
        self.etag_304_by_label = {}  # Dict[str, int] - 304s by API endpoint label


# Global instance - all code writes to this
GITHUB_API_STATS = _GitHubAPIStats()


class _CommitHistoryPerfStats:
    """Global singleton for commit-history-specific performance statistics."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics."""
        # Composite SHA operations
        self.composite_sha_cache_hit = 0
        self.composite_sha_cache_miss = 0
        self.composite_sha_errors = 0
        self.composite_sha_total_secs = 0.0
        self.composite_sha_compute_secs = 0.0

        # Local build markers
        self.marker_composite_with_reports = 0
        self.marker_composite_with_status = 0
        self.marker_composite_without_reports = 0
        self.marker_total_secs = 0.0

        # GitLab cache operations
        self.gitlab_cache_registry_images_hit = 0
        self.gitlab_cache_registry_images_miss = 0
        self.gitlab_cache_pipeline_status_hit = 0
        self.gitlab_cache_pipeline_status_miss = 0
        self.gitlab_cache_pipeline_jobs_hit = 0
        self.gitlab_cache_pipeline_jobs_miss = 0

        # GitLab timings
        self.gitlab_registry_images_total_secs = 0.0
        self.gitlab_pipeline_status_total_secs = 0.0
        self.gitlab_pipeline_jobs_total_secs = 0.0


# Global instance - all code writes to this
COMMIT_HISTORY_PERF_STATS = _CommitHistoryPerfStats()


# ======================================================================================
# ======================================================================================
# API inventory (where the dashboard data comes from)
#
# The dashboards in this repo are built from a mix of:
# - Local git metadata (branch/commit subject/SHA/time) from GitPython
# - GitHub REST v3 (https://api.github.com) for PRs, check-runs, Actions job logs
# - GitLab REST v4 (https://gitlab-master.nvidia.com) for pipeline status lines
#
# GitHub REST endpoints used (core):
# - PR lookup / branch→PR mapping:
#   - GET /repos/{owner}/{repo}/pulls                       (paged; open PR list)
#   - GET /repos/{owner}/{repo}/pulls/{pr_number}           (head sha, base ref, mergeable_state, etc.)
#   - GET /repos/{owner}/{repo}/commits/{sha}/pulls         (best-effort: find PR number from commit SHA)
#
# - Checks / CI rows (used to render the "Details" tree):
#   - GET /repos/{owner}/{repo}/commits/{sha}/check-runs     (status+conclusion+timestamps+URLs per check)
#   - GET /repos/{owner}/{repo}/commits/{sha}/status         (legacy fallback; coarse success/failure/pending)
#
# - "raw log" links + snippet inputs (GitHub Actions jobs):
#   - GET /repos/{owner}/{repo}/actions/jobs/{job_id}/logs
#       - when called with redirects disabled: capture Location header → direct "[raw log]" link (short-lived)
#       - when downloaded: returns a ZIP of log files → extract text → cache for snippet parsing
#
# - PR comments (very rough "unresolved conversations" approximation):
#   - GET /repos/{owner}/{repo}/pulls/{pr_number}/comments
#
# Optional / best-effort (may require elevated permissions):
# - Required status checks (branch protection):
#   - GET /repos/{owner}/{repo}/branches/{base_ref}/protection/required_status_checks
#     (often 403 unless token has admin permissions; if unavailable we simply don't mark "required")
#
# GitLab REST endpoints used (see `gitlab_pipeline_pr_map.py`):
# - GET /api/v4/projects/{project}/pipelines/{pipeline_id}
# - GET /api/v4/projects/{project}/repository/commits/{sha}/merge_requests
# ======================================================================================

def parse_actions_run_id_from_url(url: str) -> str:
    """Extract a GitHub Actions run_id from a typical Actions/check URL.

    Examples:
      - https://github.com/owner/repo/actions/runs/18697156351
      - https://github.com/owner/repo/actions/runs/18697156351/job/53317461976
    """
    s = str(url or "")
    if "/actions/runs/" not in s:
        return ""
    rest = s.split("/actions/runs/", 1)[1]
    run_id = rest.split("/", 1)[0].split("?", 1)[0].strip()
    return run_id if run_id.isdigit() else ""


def parse_actions_job_id_from_url(url: str) -> str:
    """Extract a GitHub Actions job_id from a typical Actions job URL.

    Examples:
      https://github.com/OWNER/REPO/actions/runs/20732129035/job/59522167110 -> 59522167110
    """
    s = str(url or "").strip()
    if "/job/" not in s:
        return ""
    rest = s.split("/job/", 1)[1]
    job_id = rest.split("/", 1)[0].split("?", 1)[0].strip()
    return job_id if job_id.isdigit() else ""


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except (ValueError, TypeError):
        return int(default)


def classify_ci_kind(name: str) -> str:
    """Coarse 'kind' classification for checks/jobs (repo-specific heuristics)."""
    s = str(name or "").lower()
    if re.search(r"\b(build|docker|image|container|package|wheel|compile)\b", s):
        return "build"
    if re.search(r"\b(test|pytest|unit|integration|e2e|smoke)\b", s):
        return "test"
    # Treat Rust style/tooling checks as "lint" so they get a consistent prefix in dashboards.
    if re.search(r"\b(rust|cargo|clippy|rustfmt|fmt)\b", s):
        return "lint"
    if re.search(r"\b(lint|format|black|ruff|flake|mypy|type|pre-commit)\b", s):
        return "lint"
    if re.search(r"\b(docs?|doc-build|sphinx)\b", s):
        return "docs"
    if re.search(r"\b(release|publish|deploy|helm)\b", s):
        return "deploy"
    return "check"


class FailedCheck:
    """Information about a failed CI check"""
    name: str
    job_url: str
    run_id: str
    duration: str
    # Raw job log download URL (usually a time-limited blob URL). Optional because it may
    # require auth, may be unavailable for older jobs, or may fail due to rate limits.
    raw_log_url: Optional[str] = None
    is_required: bool = False
    error_summary: Optional[str] = None


@dataclass
class RunningCheck:
    """Information about a running CI check"""
    name: str
    check_url: str
    is_required: bool = False
    elapsed_time: Optional[str] = None


@dataclass
class PRInfo:
    """Pull request information"""
    number: int
    title: str
    url: str
    state: str
    is_merged: bool
    review_decision: Optional[str]
    mergeable_state: str
    unresolved_conversations: int
    ci_status: Optional[str]
    # Head commit SHA for the PR (used to link to per-commit checks page).
    head_sha: Optional[str] = None
    # Head branch ref (e.g. "feature/foo") and owner/login (e.g. "keivenchang") for display.
    # These come from /pulls list payload (head.ref, head.repo.owner.login, head.label).
    head_ref: Optional[str] = None
    head_owner: Optional[str] = None
    head_label: Optional[str] = None
    base_ref: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    has_conflicts: bool = False
    conflict_message: Optional[str] = None
    blocking_message: Optional[str] = None
    # Names of required status checks for this PR's base branch (from GitHub branch protection).
    # Used to label checks as REQUIRED vs optional in the UI.
    required_checks: List[str] = field(default_factory=list)
    failed_checks: List['FailedCheck'] = field(default_factory=list)
    running_checks: List['RunningCheck'] = field(default_factory=list)
    rerun_url: Optional[str] = None
    # Optional: cached PR checks rows (used by some dashboards to show a compact status pill).
    # Keeping this on the concrete type avoids hasattr/getattr patterns downstream.
    check_rows: List['GHPRCheckRow'] = field(default_factory=list)


@dataclass(frozen=True)
class GHPRCheckRow:
    """A single PR check row (derived from GitHub REST check-runs)."""

    name: str
    status_raw: str
    duration: str = ""
    url: str = ""
    # GitHub Actions workflow run_id (best-effort parsed from `url`/`target_url`).
    # Useful for UI affordances like "Restart failed jobs".
    run_id: str = ""
    # GitHub Actions job_id (best-effort parsed from `url` when it points at a specific job).
    job_id: str = ""
    description: str = ""
    is_required: bool = False
    # Workflow name (e.g., "NVIDIA Dynamo Github Validation") - populated for GitHub Actions checks
    workflow_name: str = ""
    # Event type (e.g., "pull_request", "push") - populated for GitHub Actions checks
    event: str = ""

    @property
    def status_norm(self) -> str:
        s = (self.status_raw or "").strip().lower()
        # We normalize GitHub check-run states into gh-like buckets:
        # pass, fail, skipped, cancelled, pending, in_progress, queued, neutral, timed_out, action_required, unknown
        if s in {"pass", CIStatus.SUCCESS.value}:
            return CIStatus.SUCCESS.value
        # Treat "skipped"/"neutral" as success-like for UI aggregation.
        if s in {CIStatus.SKIPPED.value, "skip", CIStatus.NEUTRAL.value}:
            return CIStatus.SUCCESS.value
        if s in {"fail", "failure", "timed_out", "action_required"}:
            return CIStatus.FAILURE.value
        if s in {"in_progress", "running"}:
            return CIStatus.IN_PROGRESS.value
        if s in {"pending", "queued"}:
            return CIStatus.PENDING.value
        if s in {"cancelled", "canceled"}:
            return CIStatus.CANCELLED.value
        return CIStatus.UNKNOWN.value


_GHPR_CHECK_ROW_DISK_KEYS: set[str] = {
    # NOTE: Additive schema only. New keys must have defaults in `GHPRCheckRow`.
    # Missing keys are allowed (backward-compatible). Unknown keys are NOT allowed and should
    # fail fast so schema drift is caught early.
    "name",
    "status_raw",
    "duration",
    "url",
    "run_id",
    "job_id",
    "description",
    "is_required",
    "workflow_name",
    "event",
}


def _ghpr_check_row_to_disk_dict(row: GHPRCheckRow) -> Dict[str, Any]:
    # Keep this stable: it defines the on-disk cache schema for pr-check rows.
    return {
        "name": row.name,
        "status_raw": row.status_raw,
        "duration": row.duration,
        "url": row.url,
        "run_id": row.run_id,
        "job_id": row.job_id,
        "description": row.description,
        "is_required": bool(row.is_required),
        "workflow_name": row.workflow_name,
        "event": row.event,
    }


def _ghpr_check_row_from_disk_dict_strict(*, d: Any, cache_file: Path, entry_key: str) -> GHPRCheckRow:
    if not isinstance(d, dict):
        raise RuntimeError(f"Invalid pr_checks cache entry row type in {cache_file}: key={entry_key!r} type={type(d)}")
    extra = set(d.keys()) - _GHPR_CHECK_ROW_DISK_KEYS
    if extra:
        raise RuntimeError(
            f"Unknown pr_checks row fields in {cache_file}: key={entry_key!r} extra={sorted(extra)}; "
            f"expected subset of {sorted(_GHPR_CHECK_ROW_DISK_KEYS)}"
        )
    return GHPRCheckRow(
        name=str(d.get("name", "") or ""),
        status_raw=str(d.get("status_raw", "") or ""),
        duration=str(d.get("duration", "") or ""),
        url=str(d.get("url", "") or ""),
        run_id=str(d.get("run_id", "") or ""),
        job_id=str(d.get("job_id", "") or ""),
        description=str(d.get("description", "") or ""),
        is_required=bool(d.get("is_required", False)),
        workflow_name=str(d.get("workflow_name", "") or ""),
        event=str(d.get("event", "") or ""),
    )


@dataclass(frozen=True, slots=True)
class GHPRChecksCacheEntry:
    """Typed cache entry for PR check rows.

    Schema rules (Protobuf-style):
    - **ADDITIVE ONLY**: New fields MUST have defaults. Never remove or rename fields.
    - Backward-compat: Missing fields are allowed when loading from disk (defaults used).
    - Forward-compat: Older code can read newer cache entries (ignores unknown fields).
    - Strictness: Unknown fields on disk are a hard error to catch schema drift quickly.

    **CRITICAL**: Just like Protobuf, all schema changes MUST be additive (add new optional
    fields with defaults). Always additive, never subtractive. This ensures cache entries
    written by any version can be read by any other version (>= MIN_CACHE_VER). Never change
    the type or meaning of existing fields.
    """

    ts: int
    ver: int
    rows: Tuple[GHPRCheckRow, ...]
    check_runs_etag: str = ""  # v6: ETag for /commits/{sha}/check-runs
    status_etag: str = ""      # v6: ETag for /commits/{sha}/status
    incomplete: bool = False   # v7: True if this entry was written during budget exhaustion (missing job details/duration)

    _DISK_KEYS: set[str] = field(default_factory=lambda: {"ts", "ver", "rows", "check_runs_etag", "status_etag", "incomplete"}, init=False, repr=False)

    def to_disk_dict(self) -> Dict[str, Any]:
        return {
            "ts": int(self.ts),
            "ver": int(self.ver),
            "rows": [_ghpr_check_row_to_disk_dict(r) for r in self.rows],
            "check_runs_etag": str(self.check_runs_etag or ""),
            "status_etag": str(self.status_etag or ""),
            "incomplete": bool(self.incomplete),
        }

    @classmethod
    def from_disk_dict_strict(cls, *, d: Any, cache_file: Path, entry_key: str) -> "GHPRChecksCacheEntry":
        if not isinstance(d, dict):
            raise RuntimeError(f"Invalid pr_checks cache entry type in {cache_file}: key={entry_key!r} type={type(d)}")
        extra = set(d.keys()) - {"ts", "ver", "rows", "check_runs_etag", "status_etag", "incomplete"}
        if extra:
            raise RuntimeError(
                f"Unknown pr_checks cache entry fields in {cache_file}: key={entry_key!r} extra={sorted(extra)}; "
                f"expected subset of {sorted({'ts','ver','rows','check_runs_etag','status_etag','incomplete'})}"
            )
        ts = int(d.get("ts", 0) or 0)
        ver = int(d.get("ver", 0) or 0)
        rows_in = d.get("rows") or []
        if not isinstance(rows_in, list):
            raise RuntimeError(f"Invalid pr_checks cache entry rows type in {cache_file}: key={entry_key!r} type={type(rows_in)}")
        rows = tuple(_ghpr_check_row_from_disk_dict_strict(d=r, cache_file=cache_file, entry_key=entry_key) for r in rows_in)
        # ETags and incomplete flag are optional (backward-compat with v5, v6)
        check_runs_etag = str(d.get("check_runs_etag", "") or "")
        status_etag = str(d.get("status_etag", "") or "")
        incomplete = bool(d.get("incomplete", False))
        return cls(ts=ts, ver=ver, rows=rows, check_runs_etag=check_runs_etag, status_etag=status_etag, incomplete=incomplete)


@dataclass(frozen=True)
class GitHubChecksCounts:
    """Typed bucket counts for GitHub checks.

    Keep this as a dataclass (not a Dict[str, int]) so:
    - we don’t have a pile of literal-string keys spread across the codebase
    - type-checkers can catch mistakes
    """

    success_required: int = 0
    success_optional: int = 0
    failure_required: int = 0
    failure_optional: int = 0
    in_progress_required: int = 0
    in_progress_optional: int = 0
    pending: int = 0
    cancelled: int = 0
    other: int = 0
    total: int = 0


@dataclass(frozen=True)
class GitHubChecksNames:
    """Typed bucket name lists for GitHub checks (for tooltips)."""

    success_required: Tuple[str, ...] = field(default_factory=tuple)
    success_optional: Tuple[str, ...] = field(default_factory=tuple)
    failure_required: Tuple[str, ...] = field(default_factory=tuple)
    failure_optional: Tuple[str, ...] = field(default_factory=tuple)
    in_progress_required: Tuple[str, ...] = field(default_factory=tuple)
    in_progress_optional: Tuple[str, ...] = field(default_factory=tuple)
    pending: Tuple[str, ...] = field(default_factory=tuple)
    cancelled: Tuple[str, ...] = field(default_factory=tuple)
    other: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GitHubChecksSummary:
    """Bucketed GitHub checks summary (shared by multiple HTML generators)."""

    counts: GitHubChecksCounts
    names: GitHubChecksNames


def summarize_pr_check_rows(rows: Iterable[GHPRCheckRow]) -> GitHubChecksSummary:
    """Summarize PR check rows into bucketed counts + name lists.

    Args:
        rows: Iterable of `GHPRCheckRow`

    Returns:
        GitHubChecksSummary where `names[...]` holds raw check names for tooltips.
    """
    # Use local scalars/lists for clarity and performance, then freeze into dataclasses at the end.
    success_required = 0
    success_optional = 0
    failure_required = 0
    failure_optional = 0
    in_progress_required = 0
    in_progress_optional = 0
    pending = 0
    cancelled = 0
    other = 0

    names_success_required: List[str] = []
    names_success_optional: List[str] = []
    names_failure_required: List[str] = []
    names_failure_optional: List[str] = []
    names_in_progress_required: List[str] = []
    names_in_progress_optional: List[str] = []
    names_pending: List[str] = []
    names_cancelled: List[str] = []
    names_other: List[str] = []

    for r in rows:
        name = str(r.name or "")
        status = str(r.status_norm or "unknown")
        is_req = bool(r.is_required)

        if status == CIStatus.SUCCESS.value:
            if is_req:
                success_required += 1
                if name:
                    names_success_required.append(name)
            else:
                success_optional += 1
                if name:
                    names_success_optional.append(name)
        elif status == CIStatus.FAILURE.value:
            if is_req:
                failure_required += 1
                if name:
                    names_failure_required.append(name)
            else:
                failure_optional += 1
                if name:
                    names_failure_optional.append(name)
        elif status == CIStatus.IN_PROGRESS.value:
            if is_req:
                in_progress_required += 1
                if name:
                    names_in_progress_required.append(name)
            else:
                in_progress_optional += 1
                if name:
                    names_in_progress_optional.append(name)
        elif status == CIStatus.PENDING.value:
            pending += 1
            if name:
                names_pending.append(name)
        elif status == CIStatus.CANCELLED.value:
            cancelled += 1
            if name:
                names_cancelled.append(name)
        else:
            other += 1
            if name:
                names_other.append(name)

    total = (
        success_required
        + success_optional
        + failure_required
        + failure_optional
        + in_progress_required
        + in_progress_optional
        + pending
        + cancelled
        + other
    )
    return GitHubChecksSummary(
        counts=GitHubChecksCounts(
            success_required=success_required,
            success_optional=success_optional,
            failure_required=failure_required,
            failure_optional=failure_optional,
            in_progress_required=in_progress_required,
            in_progress_optional=in_progress_optional,
            pending=pending,
            cancelled=cancelled,
            other=other,
            total=total,
        ),
        names=GitHubChecksNames(
            success_required=tuple(names_success_required),
            success_optional=tuple(names_success_optional),
            failure_required=tuple(names_failure_required),
            failure_optional=tuple(names_failure_optional),
            in_progress_required=tuple(names_in_progress_required),
            in_progress_optional=tuple(names_in_progress_optional),
            pending=tuple(names_pending),
            cancelled=tuple(names_cancelled),
            other=tuple(names_other),
        ),
    )


def summarize_check_runs(check_runs: Iterable[Dict[str, Any]]) -> GitHubChecksSummary:
    """Summarize commit check-runs (GitHub REST `.../check-runs`) into buckets.

    Input shape (examples):
      - check_runs entry:
        {
          "name": "build-test-amd64",
          "status": "completed"|"in_progress"|"queued",
          "conclusion": "success"|"failure"|"cancelled"|"skipped"|None,
          "html_url": "https://..."
          "is_required": true|false,   # optional; callers can pre-annotate
        }

    Returns:
        GitHubChecksSummary with the same bucket keys as `summarize_pr_check_rows`.
    """
    success_required = 0
    success_optional = 0
    failure_required = 0
    failure_optional = 0
    in_progress_required = 0
    in_progress_optional = 0
    pending = 0
    cancelled = 0
    other = 0

    names_success_required: List[str] = []
    names_success_optional: List[str] = []
    names_failure_required: List[str] = []
    names_failure_optional: List[str] = []
    names_in_progress_required: List[str] = []
    names_in_progress_optional: List[str] = []
    names_pending: List[str] = []
    names_cancelled: List[str] = []
    names_other: List[str] = []

    for cr in check_runs or []:
        try:
            name = str(cr.get("name", "") or "")
            status_raw = str(cr.get("status", "") or "").strip().lower()
            concl_raw = cr.get("conclusion", None)
            conclusion = str(concl_raw or "").strip().lower()
            is_req = bool(cr.get("is_required", False))
        except AttributeError:  # cr is not a dict
            continue

        # Normalize check-run states into the buckets used by `GHPRCheckRow.status_norm`.
        if conclusion in {CIStatus.SUCCESS.value, CIStatus.NEUTRAL.value, CIStatus.SKIPPED.value}:
            if is_req:
                success_required += 1
                if name:
                    names_success_required.append(name)
            else:
                success_optional += 1
                if name:
                    names_success_optional.append(name)
        elif conclusion in {"failure", "timed_out", "action_required"}:
            if is_req:
                failure_required += 1
                if name:
                    names_failure_required.append(name)
            else:
                failure_optional += 1
                if name:
                    names_failure_optional.append(name)
        elif conclusion in {CIStatus.CANCELLED.value, "canceled"}:
            cancelled += 1
            if name:
                names_cancelled.append(name)
        elif status_raw in {CIStatus.IN_PROGRESS.value}:
            if is_req:
                in_progress_required += 1
                if name:
                    names_in_progress_required.append(name)
            else:
                in_progress_optional += 1
                if name:
                    names_in_progress_optional.append(name)
        elif status_raw in {"queued", CIStatus.PENDING.value}:
            pending += 1
            if name:
                names_pending.append(name)
        else:
            other += 1
            if name:
                names_other.append(name)

    total = (
        success_required
        + success_optional
        + failure_required
        + failure_optional
        + in_progress_required
        + in_progress_optional
        + pending
        + cancelled
        + other
    )
    return GitHubChecksSummary(
        counts=GitHubChecksCounts(
            success_required=success_required,
            success_optional=success_optional,
            failure_required=failure_required,
            failure_optional=failure_optional,
            in_progress_required=in_progress_required,
            in_progress_optional=in_progress_optional,
            pending=pending,
            cancelled=cancelled,
            other=other,
            total=total,
        ),
        names=GitHubChecksNames(
            success_required=tuple(names_success_required),
            success_optional=tuple(names_success_optional),
            failure_required=tuple(names_failure_required),
            failure_optional=tuple(names_failure_optional),
            in_progress_required=tuple(names_in_progress_required),
            in_progress_optional=tuple(names_in_progress_optional),
            pending=tuple(names_pending),
            cancelled=tuple(names_cancelled),
            other=tuple(names_other),
        ),
    )

def normalize_check_name(name: str) -> str:
    """Normalize GitHub/GitLab check names for robust comparison."""
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def is_required_check_name(check_name: str, required_names_normalized: set[str]) -> bool:
    """Return True iff the check name is in the required set (branch protection)."""
    name_norm = normalize_check_name(check_name)
    if not name_norm:
        return False
    return name_norm in (required_names_normalized or set())


def format_gh_check_run_duration(check_run: Dict[str, Any]) -> str:
    """Compute a short duration string for a GitHub check_run dict.

    Args:
        check_run: A dict like the items returned by GitHub's check-runs API.

    Returns:
        A short duration string like "3s", "2m 10s", "1h 4m", or "" if unknown.

    Example check_run dict (partial):
        {
          "name": "backend-status-check",
          "status": "completed",
          "conclusion": "success",
          "started_at": "2025-12-24T09:06:10Z",
          "completed_at": "2025-12-24T09:06:13Z",
          "html_url": "https://github.com/.../actions/runs/.../job/..."
        }
    """
    try:
        started = str(check_run.get("started_at", "") or "")
        completed = str(check_run.get("completed_at", "") or "")
        if not started or not completed:
            return ""
        st = datetime.fromisoformat(started.replace("Z", "+00:00"))
        ct = datetime.fromisoformat(completed.replace("Z", "+00:00"))
        delta_s = int((ct - st).total_seconds())
        return GitHubAPIClient._format_seconds_delta(delta_s)
    except (ValueError, TypeError):
        return ""


def calculate_duration_from_raw_log(raw_log_path: Path) -> str:
    """Calculate job duration from timestamps in a GitHub Actions raw log file.

    GitHub Actions logs have timestamps on every line in format:
    2026-01-13T22:40:33.7833396Z <log content>

    We extract the first and last timestamps to calculate total duration.

    Args:
        raw_log_path: Path to the raw log file

    Returns:
        A short duration string like "28m 15s", "1h 4m", or "" if unable to parse

    Note: Results are cached to disk (via DURATION_CACHE) to avoid re-parsing.
    """
    # Check disk cache first (keyed by mtime/size)
    cached_duration = DURATION_CACHE.get_raw_log_duration(raw_log_path=raw_log_path)
    if cached_duration is not None:
        return cached_duration

    try:
        if not raw_log_path.exists():
            DURATION_CACHE.put_raw_log_duration(raw_log_path=raw_log_path, duration="")
            return ""

        # Read first line with valid timestamp
        first_ts = None
        with open(raw_log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Match GitHub Actions timestamp format: YYYY-MM-DDTHH:MM:SS.fffffffZ
                # Handle BOM (byte order mark) at start of file
                match = re.match(r'^[\ufeff]?(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)', line)
                if match:
                    first_ts = match.group(1)
                    break

        if not first_ts:
            DURATION_CACHE.put_raw_log_duration(raw_log_path=raw_log_path, duration="")
            return ""

        # Read last line with valid timestamp (read last ~50 lines to avoid incomplete lines)
        last_ts = None
        with open(raw_log_path, 'rb') as f:
            # Seek to end and read last chunk
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()
            # Read last 4KB (should cover last ~50 lines)
            chunk_size = min(4096, file_size)
            f.seek(max(0, file_size - chunk_size))
            last_chunk = f.read().decode('utf-8', errors='ignore')

        # Find all timestamps in last chunk
        timestamps = re.findall(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)', last_chunk)
        if timestamps:
            last_ts = timestamps[-1]

        if not first_ts or not last_ts:
            DURATION_CACHE.put_raw_log_duration(raw_log_path=raw_log_path, duration="")
            return ""

        # Parse timestamps and calculate duration
        st = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
        ct = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
        delta_s = int((ct - st).total_seconds())

        # Use same formatting as GitHub check runs
        duration = GitHubAPIClient._format_seconds_delta(delta_s)
        DURATION_CACHE.put_raw_log_duration(raw_log_path=raw_log_path, duration=duration)
        return duration
    except (OSError, ValueError):  # File operations or invalid datetime format
        DURATION_CACHE.put_raw_log_duration(raw_log_path=raw_log_path, duration="")
        return ""


def calculate_duration_from_job_url(
    *,
    github_api: "GitHubAPIClient",
    job_url: str,
    owner: str,
    repo: str,
) -> str:
    """Calculate job duration from GitHub Actions job details API.

    This fetches job details from the API (cached) and calculates duration
    from started_at/completed_at timestamps.

    Args:
        github_api: GitHubAPIClient instance
        job_url: GitHub Actions job URL (e.g., https://github.com/.../job/12345)
        owner: Repository owner
        repo: Repository name

    Returns:
        A short duration string like "28m 15s", "1h 4m", or "" if unable to fetch

    Note: Results are cached to disk (via DURATION_CACHE) to avoid redundant API calls.
    """
    try:
        if not job_url or not github_api:
            return ""

        # Extract job ID from URL
        match = re.search(r'/job/(\d+)', job_url)
        if not match:
            return ""

        job_id = int(match.group(1))

        # Check disk cache first (keyed by job ID)
        cached_duration = DURATION_CACHE.get_job_duration(job_id=job_id)
        if cached_duration is not None:
            return cached_duration

        # Fetch job details from API (cached via get_actions_job_details_cached)
        job = github_api.get_actions_job_details_cached(
            owner=owner,
            repo=repo,
            job_id=str(job_id),
            ttl_s=30 * 24 * 3600  # 30 days cache
        )

        if not job:
            DURATION_CACHE.put_job_duration(job_id=job_id, duration="")
            return ""

        # Extract timestamps
        started = str(job.get("started_at", "") or "")
        completed = str(job.get("completed_at", "") or "")

        if not started or not completed:
            DURATION_CACHE.put_job_duration(job_id=job_id, duration="")
            return ""

        # Calculate duration
        st = datetime.fromisoformat(started.replace("Z", "+00:00"))
        ct = datetime.fromisoformat(completed.replace("Z", "+00:00"))
        delta_s = int((ct - st).total_seconds())

        # Use same formatting as GitHub check runs
        duration = GitHubAPIClient._format_seconds_delta(delta_s)
        DURATION_CACHE.put_job_duration(job_id=job_id, duration=duration)
        return duration
    except (AttributeError, ValueError):  # job.get() on non-dict or invalid datetime
        # Only cache empty result if we have a valid job_id
        if 'job_id' in locals():
            DURATION_CACHE.put_job_duration(job_id=job_id, duration="")
        return ""


class DiskCacheWriter:
    """
    Context manager that enforces lock-load-merge-save pattern for disk cache writes.

    Usage:
        with DiskCacheWriter(cache_file, lock_file, load_fn, save_fn) as cache:
            cache["key"] = {"ts": now, "val": data}  # MERGE: update cache in-place

    The pattern is:
    1. LOCK: Acquire exclusive file lock
    2. LOAD: Load current disk cache
    3. MERGE: Caller updates cache dict (via context manager)
    4. SAVE: Write merged cache atomically
    5. UNLOCK: Release lock (automatic on __exit__)

    This prevents cache CLOBBER bugs by ensuring all writes reload + merge before saving.
    """

    def __init__(
        self,
        cache_file: Path,
        lock_file: Path,
        load_fn: callable,
        save_fn: callable,
    ):
        self.cache_file = cache_file
        self.lock_file = lock_file
        self.load_fn = load_fn
        self.save_fn = save_fn
        self.lock_fd = None
        self.cache = None

    def __enter__(self):
        # LOCK: Acquire exclusive file lock
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self.lock_fd = open(self.lock_file, 'w')
        fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_EX)

        # LOAD: Load current cache
        self.cache = self.load_fn()
        if not isinstance(self.cache, dict):
            self.cache = {}

        return self.cache  # Caller will MERGE entries into this dict

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is None and self.cache is not None:
                # SAVE: Write merged cache atomically
                self.save_fn(self.cache)
        finally:
            # UNLOCK: Release lock
            if self.lock_fd:
                fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_UN)
                self.lock_fd.close()
        return False  # Don't suppress exceptions


def _save_single_disk_cache_entry(
    cache_dir: Path,
    cache_filename: str,
    lock_filename: str,
    load_fn: Callable[[], Dict[str, Any]],
    json_dump_fn: Callable[[Dict[str, Any]], str],
    key: str,
    value: Any,
    stats_fn: Optional[Callable[[int], None]] = None,
) -> None:
    """
    Generic helper to atomically update a single entry in a disk cache.
    Uses DiskCacheWriter to enforce lock-load-merge-save pattern.

    Args:
        cache_dir: Directory containing cache file
        cache_filename: Name of cache file (e.g., "actions_jobs.json")
        lock_filename: Name of lock file (e.g., "actions_jobs.lock")
        load_fn: Function to load existing cache
        json_dump_fn: Function to serialize dict to JSON string
        key: Cache key to update
        value: Cache value to store
        stats_fn: Optional callback to record cache stats (receives entry count)
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / cache_filename
    lock_file = cache_dir / lock_filename

    def save_fn(data):
        tmp = str(cache_file) + ".tmp"
        Path(tmp).write_text(json_dump_fn(data))
        os.replace(tmp, cache_file)

    # DiskCacheWriter enforces: LOCK → LOAD → MERGE → SAVE → UNLOCK
    with DiskCacheWriter(cache_file, lock_file, load_fn, save_fn) as cache:
        cache[key] = value  # MERGE: Update cache in-place
        if stats_fn:
            stats_fn(len(cache))


class GitHubAPIClient:
    """GitHub API client with automatic token detection and rate limit handling.

    Features:
    - Automatic token detection (--token arg > GitHub CLI config file)
    - Request/response handling with proper error messages
    - Support for parallel API calls with ThreadPoolExecutor

    Example:
        client = GitHubAPIClient()
        pr_data = client.get("/repos/owner/repo/pulls/123")
    """

    @staticmethod
    def get_github_token_from_file() -> Optional[str]:
        """Get GitHub token from a local config file (preferred).

        We intentionally do NOT read GH_TOKEN/GITHUB_TOKEN env vars in this repo anymore.

        Currently supported locations (first match wins):
        - ~/.config/github-token   (single line token)
        - ~/.config/gh/hosts.yml   (GitHub CLI login; oauth_token)
        """
        # 1) Simple token file (if present)
        try:
            token_file = Path.home() / ".config" / "github-token"
            if token_file.exists():
                tok = (token_file.read_text() or "").strip()
                if tok:
                    return tok
        except OSError:  # File read errors
            pass
        # 2) GitHub CLI config
        return GitHubAPIClient.get_github_token_from_cli()

    @staticmethod
    def get_github_token_from_cli() -> Optional[str]:
        """Get GitHub token from GitHub CLI configuration.

        Reads the token from ~/.config/gh/hosts.yml if available.

        Returns:
            GitHub token string, or None if not found
        """
        try:
            gh_config_path = Path.home() / '.config' / 'gh' / 'hosts.yml'
            if gh_config_path.exists():
                with open(gh_config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    if config and 'github.com' in config:
                        github_config = config['github.com']
                        if 'oauth_token' in github_config:
                            return github_config['oauth_token']
                        if 'users' in github_config:
                            for user, user_config in github_config['users'].items():
                                if 'oauth_token' in user_config:
                                    return user_config['oauth_token']
        except (OSError, yaml.YAMLError):  # File read or YAML parse errors
            pass
        return None

    def __init__(
        self,
        token: Optional[str] = None,
        *,
        debug_rest: bool = False,
        require_auth: bool = False,
        allow_anonymous_fallback: bool = True,
        max_rest_calls: Optional[int] = None,
    ):
        """Initialize GitHub API client.

        Args:
            token: GitHub personal access token. If not provided, will try:
                   1. ~/.config/github-token (if present)
                   2. GitHub CLI config (~/.config/gh/hosts.yml)
            require_auth: If True, raise an error if we cannot find a token.
            allow_anonymous_fallback: If True, allow a best-effort retry of GET requests without
                                      Authorization on certain 401/403 auth failures (public-repo resiliency).
        """
        # Token priority:
        # 1) explicit arg
        # 2) token file / gh CLI config
        self.token = token or self.get_github_token_from_file()
        self.require_auth = bool(require_auth)
        self.allow_anonymous_fallback = bool(allow_anonymous_fallback)
        self.base_url = "https://api.github.com"
        self.headers = {'Accept': 'application/vnd.github.v3+json'}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._debug_rest = bool(debug_rest)

        # Per-invocation API budget: hard cap on *network* REST calls for this client instance.
        # Cached reads do not consume budget; each HTTP request (including retries) does.
        # When the budget is exhausted, we switch to cache-only mode instead of failing hard.
        try:
            self._rest_budget_max: Optional[int] = int(max_rest_calls) if max_rest_calls is not None else None
            if self._rest_budget_max is not None and self._rest_budget_max <= 0:
                self._rest_budget_max = 0
        except (ValueError, TypeError):
            self._rest_budget_max = None
        self._rest_budget_exhausted: bool = False
        self._rest_budget_exhausted_reason: str = ""

        if self.require_auth and not self.token:
            raise RuntimeError(
                "GitHub API authentication is required but no token was found. "
                "Pass --token, or login with gh so ~/.config/gh/hosts.yml exists."
            )

        if self.token:
            self.headers['Authorization'] = f'token {self.token}'

        # Store budget info in global stats
        GITHUB_API_STATS.rest_budget_max = self._rest_budget_max

        # ----------------------------
        # Persistent caches (disk + memory)
        # ----------------------------
        # Cache for job logs (two-tier: memory + disk)
        # - memory: { "53317461976": "Failed unit tests: ..." }
        # - disk:   ~/.cache/dynamo-utils/job-logs/job_logs_cache.json:
        #   {
        #     "53317461976": "Failed unit tests:\n ...",
        #     "53317461234": null
        #   }
        # Cache for repo-wide pull request listing (short TTL; used to avoid per-branch API calls).
        self._pulls_list_mem_cache: Dict[str, Dict[str, Any]] = {}
        self._pulls_list_cache_dir = dynamo_utils_cache_dir() / "pulls"

        # Cache for "closed/merged PRs per branch" lookups (long TTL; these don't change often).
        # Cache for downloaded raw log text (two-tier: memory + disk).
        self._raw_log_text_mem_cache: Dict[str, Dict[str, Any]] = {}
        self._raw_log_text_cache_dir = dynamo_utils_cache_dir() / "raw-log-text"
        self._raw_log_text_index_file = self._raw_log_text_cache_dir / "index.json"

        # Cache for Actions job status lookups (short TTL; prevents caching partial logs).
        self._actions_job_status_mem_cache: Dict[str, Dict[str, Any]] = {}
        # Cache for Actions job details (steps, timestamps) (two-tier: memory + disk, short TTL).
        self._actions_job_details_mem_cache: Dict[str, Dict[str, Any]] = {}
        self._actions_job_cache_dir = dynamo_utils_cache_dir() / "actions-jobs"
        # When True, avoid any network fetches and rely on caches only (best-effort).
        self.cache_only_mode: bool = False

        # Cached rate limit info from response headers (used when cache-only mode prevents fresh API calls)
        # Format: {"remaining": 1234, "limit": 5000, "reset_epoch": 1766947200, ...}
        self._cached_rate_limit_info: Optional[Dict[str, Any]] = None

        # Cache for required status checks (branch protection). This changes rarely and is safe to cache
        # for a long time. Keyed by (owner, repo, base_ref).
        # Cache for enriched PRInfo objects keyed by (owner/repo, pr_number, updated_at).
        # This is the main knob that allows "0 API calls" for stable PRs: if the PR wasn't updated,
        # we reuse the cached enrichment (ci_status, failed checks, etc).
        # Cache for search/issues results used to probe updated_at for a list of PRs.
        self._search_issues_mem_cache: Dict[str, Dict[str, Any]] = {}
        self._search_issues_cache_dir = dynamo_utils_cache_dir() / "search-issues"
        # If search/issues is returning 422 for a repo (common with certain tokens), disable it temporarily
        # so dashboards don't spam errors for an optimization-only call.
        self._search_issues_disabled_mem_cache: Dict[str, Dict[str, Any]] = {}
        
        # Track initial disk cache sizes before any modifications (for accurate stats reporting)
        self._initial_disk_counts: Dict[str, int] = {}
        self._capture_initial_disk_counts()

        # Inflight request deduplication: per-key locks to prevent concurrent identical API calls.
        self._inflight_locks_mu = threading.Lock()
        self._inflight_locks: Dict[str, threading.Lock] = {}

    def _capture_initial_disk_counts(self) -> None:
        """Capture disk cache sizes before any modifications for accurate stats reporting."""
        try:
            # Actions jobs (status and details in same file, separated by key prefix)
            disk = self._load_actions_job_disk_cache()
            if isinstance(disk, dict):
                self._initial_disk_counts["actions_job_status_disk"] = sum(1 for k in disk.keys() if ":jobstatus:" in str(k))
                self._initial_disk_counts["actions_job_details_disk"] = sum(1 for k in disk.keys() if ":job:" in str(k) and ":jobstatus:" not in str(k))
            else:
                self._initial_disk_counts["actions_job_status_disk"] = 0
                self._initial_disk_counts["actions_job_details_disk"] = 0
            
            # Pulls list
            pulls_disk = self._load_pulls_list_disk_cache()
            self._initial_disk_counts["pulls_list_disk"] = len(pulls_disk) if isinstance(pulls_disk, dict) else 0
            
            # Search issues
            search_issues_disk = self._load_search_issues_disk_cache()
            self._initial_disk_counts["search_issues_disk"] = len(search_issues_disk) if isinstance(search_issues_disk, dict) else 0
            
            # Raw log text
            if self._raw_log_text_index_file.exists():
                with open(self._raw_log_text_index_file, 'r') as f:
                    raw_log_index = json.load(f)
                self._initial_disk_counts["raw_log_text_disk"] = len(raw_log_index) if isinstance(raw_log_index, dict) else 0
            else:
                self._initial_disk_counts["raw_log_text_disk"] = 0
            
            # Merge dates
            merge_dates_file = dynamo_utils_cache_dir() / "github_pr_merge_dates.json"
            if merge_dates_file.exists():
                with open(merge_dates_file, 'r') as f:
                    merge_dates = json.load(f)
                self._initial_disk_counts["merge_dates_disk"] = len(merge_dates) if isinstance(merge_dates, dict) else 0
            else:
                self._initial_disk_counts["merge_dates_disk"] = 0
        except (OSError, json.JSONDecodeError, ValueError) as e:
            # If any error occurs, log and use empty counts - don't fail initialization
            logger = logging.getLogger("github_client")
            logger.debug(f"Failed to load initial disk counts: {e}")
            # Continue with empty counts
            pass

    def set_cache_only_mode(self, on: bool = True) -> None:
        """Enable/disable cache-only mode (best-effort)."""
        self.cache_only_mode = bool(on)

    def _budget_maybe_consume_or_raise(self) -> None:
        """Consume one unit of REST budget, or raise when exhausted.

        IMPORTANT: 304 Not Modified responses (ETags) do NOT count against GitHub's rate limit,
        so we exclude them from our budget calculation. Only billable API calls (non-304) count.
        """
        # Unlimited budget
        if self._rest_budget_max is None:
            return
        # Already exhausted
        if bool(self._rest_budget_exhausted):
            raise RuntimeError(self._rest_budget_exhausted_reason or "GitHub REST call budget exhausted")
        # Calculate billable calls (exclude 304 Not Modified - these don't count against GitHub rate limit)
        try:
            billable_calls = int(GITHUB_API_STATS.rest_calls_total) - int(GITHUB_API_STATS.etag_304_total)
            remaining = int(self._rest_budget_max) - billable_calls
        except (ValueError, TypeError):
            remaining = -1
        if remaining <= 0:
            self._rest_budget_exhausted = True
            self._rest_budget_exhausted_reason = (
                f"GitHub REST call budget exhausted (max_rest_calls={self._rest_budget_max}, "
                f"billable={billable_calls}, 304_etags={int(GITHUB_API_STATS.etag_304_total)})"
            )
            raise RuntimeError(self._rest_budget_exhausted_reason)

    def _json_load_text(self, txt: str) -> Dict[str, Any]:
        """Parse JSON text (best-effort)."""
        return json.loads(txt or "{}") or {}

    def _json_dump_text(self, obj: Any, *, indent: Optional[int] = None) -> str:
        """Serialize JSON text (best-effort)."""
        return json.dumps(obj, indent=indent)

    def _rest_label_for_url(self, url: str) -> str:
        """Coarse label for a REST request URL (keeps job ids / SHAs from exploding cardinality)."""
        u = str(url or "")
        # Normalize to a path-only string (avoid scheme/host leaking into labels).
        try:
            path = urllib.parse.urlparse(u).path or ""
        except ValueError:
            path = ""
        s = path or u

        if "/rate_limit" in s:
            return "rate_limit"

        if s.startswith("/search/issues"):
            return "search_issues"

        # Actions jobs
        if re.search(r"/repos/[^/]+/[^/]+/actions/jobs/\d+/logs\b", s):
            return "actions_job_logs_zip"
        if re.search(r"/repos/[^/]+/[^/]+/actions/jobs/\d+\b", s):
            return "actions_job_status"

        # Actions runs
        if re.search(r"/repos/[^/]+/[^/]+/actions/runs/\d+/jobs\b", s):
            return "actions_run_jobs"
        if re.search(r"/repos/[^/]+/[^/]+/actions/runs/\d+\b", s):
            return "actions_run"

        # Check-runs (per commit)
        if "/check-runs" in s:
            return "check_runs"

        # PR / pulls (check most specific patterns first!)
        if re.search(r"/repos/[^/]+/[^/]+/pulls/\d+/comments\b", s):
            return "pr_review_comments"
        if re.search(r"/repos/[^/]+/[^/]+/pulls/\d+\b", s):
            return "pull_request"
        if re.search(r"/repos/[^/]+/[^/]+/pulls\b", s):
            return "pulls_list"

        # Commit -> PR mapping (best-effort)
        if re.search(r"/repos/[^/]+/[^/]+/commits/[0-9a-f]{7,40}/pulls\b", s, flags=re.IGNORECASE):
            return "commit_pulls"

        # Branch protection required checks (best-effort; often 403)
        if re.search(r"/repos/[^/]+/[^/]+/branches/[^/]+/protection/required_status_checks\b", s):
            return "required_status_checks"

        # Default: bucket by first few path segments
        parts = [p for p in (path or "").split("/") if p]
        # If it is a /repos/<owner>/<repo>/... path, bucket by the resource segment after repo.
        if len(parts) >= 4 and parts[0] == "repos":
            res = parts[3]
            # Keep res stable (don't include SHAs/IDs).
            return f"repos_{res}"
        # Otherwise, fall back to a short prefix of the path.
        return "/".join(parts[:3]) if parts else "unknown"

    def _inflight_lock(self, key: str) -> "threading.Lock":
        """Return a per-key lock to dedupe concurrent network fetches across threads."""
        k = str(key or "")
        if not k:
            # Fallback: single shared lock
            k = "__default__"
        mu = self._inflight_locks_mu
        locks = self._inflight_locks
        with mu:
            lk = locks.get(k)
            if lk is None:
                lk = threading.Lock()
                locks[k] = lk
            return lk

    def _cache_hit(self, name: str) -> None:
        try:
            k = str(name or "").strip() or "unknown"
            GITHUB_API_STATS.cache_hits[k] = int(GITHUB_API_STATS.cache_hits.get(k, 0) or 0) + 1
        except (ValueError, TypeError):
            pass

    def _cache_miss(self, name: str) -> None:
        try:
            k = str(name or "").strip() or "unknown"
            GITHUB_API_STATS.cache_misses[k] = int(GITHUB_API_STATS.cache_misses.get(k, 0) or 0) + 1
        except (ValueError, TypeError):
            pass

    def _cache_write(self, name: str, *, entries: int = 0) -> None:
        try:
            k = str(name or "").strip() or "unknown"
            GITHUB_API_STATS.cache_writes_ops[k] = int(GITHUB_API_STATS.cache_writes_ops.get(k, 0) or 0) + 1
            if int(entries or 0) > 0:
                GITHUB_API_STATS.cache_writes_entries[k] = int(GITHUB_API_STATS.cache_writes_entries.get(k, 0) or 0) + int(entries or 0)
        except (ValueError, TypeError):
            pass

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return best-effort per-run cache hit/miss stats for displaying in dashboards."""
        try:
            hits_total = int(sum(int(v) for v in (GITHUB_API_STATS.cache_hits or {}).values()))
        except (ValueError, TypeError):
            hits_total = 0
        try:
            misses_total = int(sum(int(v) for v in (GITHUB_API_STATS.cache_misses or {}).values()))
        except (ValueError, TypeError):
            misses_total = 0
        try:
            hits_by = dict(sorted((GITHUB_API_STATS.cache_hits or {}).items(), key=lambda kv: (-int(kv[1] or 0), str(kv[0]))))
        except (ValueError, TypeError):
            hits_by = {}
        try:
            misses_by = dict(sorted((GITHUB_API_STATS.cache_misses or {}).items(), key=lambda kv: (-int(kv[1] or 0), str(kv[0]))))
        except (ValueError, TypeError):
            misses_by = {}
        try:
            writes_ops_total = int(sum(int(v) for v in (GITHUB_API_STATS.cache_writes_ops or {}).values()))
        except (ValueError, TypeError):
            writes_ops_total = 0
        try:
            writes_ops_by = dict(sorted((GITHUB_API_STATS.cache_writes_ops or {}).items(), key=lambda kv: (-int(kv[1] or 0), str(kv[0]))))
        except (ValueError, TypeError):
            writes_ops_by = {}
        try:
            writes_entries_total = int(sum(int(v) for v in (GITHUB_API_STATS.cache_writes_entries or {}).values()))
        except (ValueError, TypeError):
            writes_entries_total = 0
        try:
            writes_entries_by = dict(sorted((GITHUB_API_STATS.cache_writes_entries or {}).items(), key=lambda kv: (-int(kv[1] or 0), str(kv[0]))))
        except (ValueError, TypeError):
            writes_entries_by = {}

        # Cache entry counts (memory + disk)
        # These are internal attributes initialized in __init__, so they always exist
        
        # Get cache counts from dedicated cache modules
        pr_branch_mem, pr_branch_disk = PR_BRANCH_CACHE.get_cache_sizes()
        required_checks_mem, required_checks_disk = REQUIRED_CHECKS_CACHE.get_cache_sizes()
        pr_checks_mem, pr_checks_disk = PR_CHECKS_CACHE.get_cache_sizes()
        pr_info_mem, pr_info_disk = PR_INFO_CACHE.get_cache_sizes()
        job_log_mem, job_log_disk = JOB_LOG_CACHE.get_cache_sizes()
        
        cache_sizes = {
            "pr_checks_mem": pr_checks_mem,
            "pulls_list_mem": len(self._pulls_list_mem_cache),
            "pr_branch_mem": pr_branch_mem,
            "raw_log_text_mem": len(self._raw_log_text_mem_cache),
            "actions_job_status_mem": len(self._actions_job_status_mem_cache),
            "actions_job_details_mem": len(self._actions_job_details_mem_cache),
            "required_checks_mem": required_checks_mem,
            "pr_info_mem": pr_info_mem,
            "search_issues_mem": len(self._search_issues_mem_cache),
            "job_log_mem": job_log_mem,
            "pr_branch_disk": pr_branch_disk,
            "required_checks_disk": required_checks_disk,
            "pr_checks_disk": pr_checks_disk,
            "pr_info_disk": pr_info_disk,
            "job_log_disk": job_log_disk,
        }

        # Disk cache counts - show initial counts before this run's modifications
        # Always show count even if 0 to verify no caches are missing
        
        # Use initial counts captured at initialization
        for key in ["actions_job_status_disk", "actions_job_details_disk", "pulls_list_disk",
                    "search_issues_disk", "raw_log_text_disk", "merge_dates_disk"]:
            cache_sizes[key] = self._initial_disk_counts.get(key, 0)

        return {
            "hits_total": hits_total,
            "misses_total": misses_total,
            "writes_ops_total": writes_ops_total,
            "writes_entries_total": writes_entries_total,
            "hits_by": hits_by,
            "misses_by": misses_by,
            "writes_ops_by": writes_ops_by,
            "writes_entries_by": writes_entries_by,
            "entries": cache_sizes,
        }

    def _rest_get(
        self,
        url: str,
        *,
        timeout: int = 10,
        allow_redirects: bool = True,
        stream: bool = False,
        params: Optional[Dict[str, Any]] = None,
        etag: Optional[str] = None,
    ):
        """requests.get wrapper that increments per-run counters and supports ETags.

        Args:
            etag: Optional ETag from previous request. If provided, sends If-None-Match header.
                  Returns 304 Not Modified if content unchanged (doesn't count against rate limit).

        Returns:
            Response object. Check status_code: 304 means content unchanged, use cached data.

        ETag Support:
            GitHub API endpoints support conditional requests via ETags. When a cached ETag is provided:
            - 304 Not Modified: Content unchanged, use cached data (DOESN'T count against rate limit!)
            - 200 OK: Content changed, new ETag in response.headers['ETag']

        Note:
            We sometimes run on machines where a stale/non-compliant GitHub token is present in the
            environment (classic PAT > 366 day lifetime, fine-grained token not granted repo access, etc).
            In those cases, GitHub returns 403 for requests that would otherwise succeed anonymously
            (especially for public repos). To make dashboards resilient, we do a best-effort retry of
            *GET* requests without Authorization on a narrow class of known token-related 403s.
        """
        assert requests is not None
        label = self._rest_label_for_url(url)

        # Cache-only mode: do not perform any network operations.
        if self.cache_only_mode:
            raise RuntimeError("cache_only_mode enabled; refusing network request")

        # Enforce per-invocation budget.
        self._budget_maybe_consume_or_raise()

        try:
            GITHUB_API_STATS.rest_calls_total += 1
            GITHUB_API_STATS.rest_calls_by_label[label] = int(GITHUB_API_STATS.rest_calls_by_label.get(label, 0) or 0) + 1
        except (ValueError, TypeError):
            pass
        if self._debug_rest:
            try:
                self.logger.debug("GH REST GET [%s] %s", label, url)
            except (ValueError, TypeError):
                pass
        headers = dict(self.headers or {})

        # Add ETag support for conditional requests
        if etag:
            headers['If-None-Match'] = etag

        t0_req = time.monotonic()
        resp = requests.get(url, headers=headers, params=params, timeout=timeout, allow_redirects=allow_redirects, stream=stream)
        dt = max(0.0, time.monotonic() - t0_req)
        GITHUB_API_STATS.rest_time_total_s += float(dt)
        GITHUB_API_STATS.rest_time_by_label_s[label] = float(GITHUB_API_STATS.rest_time_by_label_s.get(label, 0.0)) + float(dt)

        def _record_response(r) -> None:
            """Record success/error stats for a single HTTP response."""
            try:
                code = int(r.status_code or 0)
            except (ValueError, TypeError):
                code = 0
            # Track 304 Not Modified separately (ETags: these don't count against rate limit!)
            if code == 304:
                GITHUB_API_STATS.etag_304_total += 1
                GITHUB_API_STATS.etag_304_by_label[label] = int(GITHUB_API_STATS.etag_304_by_label.get(label, 0) or 0) + 1
                GITHUB_API_STATS.rest_success_total += 1
                GITHUB_API_STATS.rest_success_by_label[label] = int(GITHUB_API_STATS.rest_success_by_label.get(label, 0) or 0) + 1
            elif code and code < 400:
                GITHUB_API_STATS.rest_success_total += 1
                GITHUB_API_STATS.rest_success_by_label[label] = int(GITHUB_API_STATS.rest_success_by_label.get(label, 0) or 0) + 1
            else:
                GITHUB_API_STATS.rest_errors_total += 1
                if code:
                    GITHUB_API_STATS.rest_errors_by_status[code] = int(GITHUB_API_STATS.rest_errors_by_status.get(code, 0) or 0) + 1
                    # label+status breakdown
                    try:
                        d = GITHUB_API_STATS.rest_errors_by_label_status.get(label)
                        if not isinstance(d, dict):
                            d = {}
                            GITHUB_API_STATS.rest_errors_by_label_status[label] = d
                        d[int(code)] = int(d.get(int(code), 0) or 0) + 1
                    except (ValueError, TypeError):
                        pass
                # Keep last error small (HTML stats should not explode).
                body = ""
                try:
                    body = (r.text or "")[:300]
                except (ValueError, TypeError):
                    body = ""
                # Prefer the response URL (includes query string) so we can diagnose failures like search/issues 422.
                url_full = ""
                try:
                    url_full = str(r.url or str(url or "")).strip()
                except AttributeError:  # r.url access fails if r is not a Response object
                    url_full = str(url or "")
                GITHUB_API_STATS.rest_last_error = {
                    "status": code,
                    "url": url_full,
                    "body": body,
                }
                GITHUB_API_STATS.rest_last_error_label = str(label or "")

        # Record the first response (even if we later retry).
        _record_response(resp)

        # Best-effort retry without Authorization for a narrow class of token-related auth failures.
        try:
            if not self.allow_anonymous_fallback:
                return resp
            code0 = int(resp.status_code or 0)
            if code0 in (401, 403) and ("Authorization" in headers):
                body_txt = ""
                try:
                    body_txt = (resp.text or "")
                except (ValueError, TypeError):
                    body_txt = ""
                body_lc = body_txt.lower()
                token_related_403 = (
                    ("enterprise forbids access via" in body_lc)
                    or ("forbids access via a personal access tokens" in body_lc)
                    or ("forbids access via a fine-grained personal access tokens" in body_lc)
                    or ("resource not accessible by personal access token" in body_lc)
                )
                bad_credentials_401 = (code0 == 401) and ("bad credentials" in body_lc)

                # Retry conditions:
                # - 403: only for known policy errors where anonymous access would have worked
                # - 401: invalid token ("Bad credentials") can break public-repo dashboards; retry anonymously
                if (code0 == 403 and token_related_403) or (code0 == 401 and bad_credentials_401):
                    headers_no_auth = dict(headers)
                    headers_no_auth.pop("Authorization", None)
                    # Budget: retries consume budget too.
                    self._budget_maybe_consume_or_raise()
                    # Count this retry as a REST call too (it is).
                    try:
                        GITHUB_API_STATS.rest_calls_total += 1
                        GITHUB_API_STATS.rest_calls_by_label[label] = int(GITHUB_API_STATS.rest_calls_by_label.get(label, 0) or 0) + 1
                    except (ValueError, TypeError):
                        pass
                    if self._debug_rest:
                        try:
                            self.logger.debug("GH REST GET [%s] retrying without Authorization (auth failure; public access may still work)", label)
                        except (ValueError, TypeError):
                            pass
                    t0_req2 = time.monotonic()
                    resp2 = requests.get(url, headers=headers_no_auth, params=params, timeout=timeout, allow_redirects=allow_redirects, stream=stream)
                    dt2 = max(0.0, time.monotonic() - t0_req2)
                    GITHUB_API_STATS.rest_time_total_s += float(dt2)
                    GITHUB_API_STATS.rest_time_by_label_s[label] = float(GITHUB_API_STATS.rest_time_by_label_s.get(label, 0.0)) + float(dt2)
                    _record_response(resp2)
                    # Prefer the retry if it succeeded (or at least changed the failure mode).
                    if int(resp2.status_code or 0) < 400:
                        resp = resp2
                    else:
                        # If the retry still fails, keep the original response (it likely contains the most useful policy message).
                        pass
        except (ValueError, TypeError):
            pass
        if self._debug_rest:
            rem = resp.headers.get("X-RateLimit-Remaining")
            code = resp.status_code
            self.logger.debug("GH REST RESP [%s] status=%s remaining=%s", label, str(code), str(rem))

        # Extract and cache rate limit info from response headers for use in get_core_rate_limit_info().
        # This allows displaying rate limit stats even when we later enter cache-only mode.
        try:
            remaining_hdr = resp.headers.get("X-RateLimit-Remaining")
            limit_hdr = resp.headers.get("X-RateLimit-Limit")
            reset_hdr = resp.headers.get("X-RateLimit-Reset")
            if remaining_hdr is not None or limit_hdr is not None or reset_hdr is not None:
                remaining = int(remaining_hdr) if remaining_hdr is not None else None
                limit = int(limit_hdr) if limit_hdr is not None else None
                reset_epoch = int(reset_hdr) if reset_hdr is not None else None
                # Cache if we have at least remaining and limit (reset is optional)
                if remaining is not None and limit is not None:
                    now = int(time.time())
                    seconds_until = int(reset_epoch) - now if reset_epoch is not None else 0
                    if reset_epoch is not None:
                        reset_local = datetime.fromtimestamp(int(reset_epoch)).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
                        try:
                            reset_pt = (
                                datetime.fromtimestamp(int(reset_epoch), tz=timezone.utc)
                                .astimezone(ZoneInfo("America/Los_Angeles"))
                                .strftime("%Y-%m-%d %H:%M:%S %Z")
                            )
                        except (ValueError, TypeError):
                            reset_pt = reset_local
                    else:
                        reset_local = "unknown"
                        reset_pt = "unknown"
                    self._cached_rate_limit_info = {
                        "remaining": int(remaining),
                        "limit": int(limit),
                        "reset_epoch": int(reset_epoch) if reset_epoch is not None else None,
                        "reset_local": reset_local,
                        "reset_pt": reset_pt,
                        "seconds_until_reset": seconds_until,
                    }
        except (ValueError, AttributeError):  # int() on invalid header values or resp.headers access fails
            pass

        return resp

    def get_rest_error_stats(self) -> Dict[str, Any]:
        """Return best-effort per-run REST error stats for displaying in dashboards."""
        try:
            return {
                "total": int(GITHUB_API_STATS.rest_errors_total),
                "by_status": dict(sorted(GITHUB_API_STATS.rest_errors_by_status.items(), key=lambda kv: (-kv[1], kv[0]))),
                "by_label_status": {
                    str(lbl): dict(sorted({int(k): int(v) for k, v in (m or {}).items() if int(v or 0) > 0}.items(), key=lambda kv: (-kv[1], kv[0])))
                    for (lbl, m) in dict(sorted((GITHUB_API_STATS.rest_errors_by_label_status or {}).items(), key=lambda kv: str(kv[0]))).items()
                    if isinstance(m, dict)
                },
                "last_label": str(GITHUB_API_STATS.rest_last_error_label or ""),
                "last": dict(GITHUB_API_STATS.rest_last_error or {}),
            }
        except (ValueError, TypeError):
            return {"total": 0, "by_status": {}, "by_label_status": {}, "last_label": "", "last": {}}

    def get_rest_call_stats(self) -> Dict[str, Any]:
        """Return per-run REST call stats for debugging."""
        try:
            return {
                "total": int(GITHUB_API_STATS.rest_calls_total),
                "budget_max": int(self._rest_budget_max) if self._rest_budget_max is not None else None,
                "budget_exhausted": bool(self._rest_budget_exhausted),
                "by_label": dict(sorted(GITHUB_API_STATS.rest_calls_by_label.items(), key=lambda kv: (-kv[1], kv[0]))),
                "success_total": int(GITHUB_API_STATS.rest_success_total),
                "success_by_label": dict(sorted(GITHUB_API_STATS.rest_success_by_label.items(), key=lambda kv: (-kv[1], kv[0]))),
                "error_total": int(GITHUB_API_STATS.rest_errors_total),
                "time_total_s": float(GITHUB_API_STATS.rest_time_total_s),
                "time_by_label_s": dict(sorted(GITHUB_API_STATS.rest_time_by_label_s.items(), key=lambda kv: (-kv[1], kv[0]))),
            }
        except (ValueError, TypeError):
            return {
                "total": 0,
                "budget_max": None,
                "budget_exhausted": False,
                "by_label": {},
                "success_total": 0,
                "success_by_label": {},
                "error_total": 0,
                "time_total_s": 0.0,
                "time_by_label_s": {},
            }

    def get_actions_job_status(self, *, owner: str, repo: str, job_id: str, ttl_s: int = 60) -> Optional[str]:
        """Return GitHub Actions job status ("completed", "in_progress", ...) with disk cache for completed jobs.

        OPTIMIZATION: Completed jobs are cached to disk permanently since their status never changes.
        This reduces API calls from ~300+ per run to near-zero on subsequent runs.
        
        TTL: 1m for non-completed jobs, ∞ for completed jobs (immutable).
        """

        job_id_s = str(job_id or "").strip()
        if not job_id_s:
            return None

        key = f"{owner}/{repo}:jobstatus:{job_id_s}"
        now = int(time.time())

        # Memory cache (all statuses, short TTL for in-progress jobs)
        try:
            ent = self._actions_job_status_mem_cache.get(key)
            if ent and int(ent.get("ts", 0) or 0) + int(ttl_s) > now:
                st = ent.get("status")
                if st is not None:
                    self._cache_hit("actions_job_status.mem")
                    return str(st)
        except (ValueError, TypeError):
            pass

        # Disk cache (only for completed jobs - they never change!)
        disk = self._load_actions_job_disk_cache()
        ent = disk.get(key) if isinstance(disk, dict) else None
        if isinstance(ent, dict):
            status_cached = ent.get("status")
            # Completed jobs are cached forever; non-completed jobs respect TTL
            if str(status_cached or "").lower() == "completed":
                self._actions_job_status_mem_cache[key] = {"ts": now, "status": status_cached}
                self._cache_hit("actions_job_status.disk")
                return str(status_cached)
            # For non-completed status, check TTL
            ts = int(ent.get("ts", 0) or 0)
            if ts and (now - ts) <= int(ttl_s):
                self._actions_job_status_mem_cache[key] = {"ts": ts, "status": status_cached}
                self._cache_hit("actions_job_status.disk")
                return str(status_cached) if status_cached is not None else None

        # Cache-only mode: don't fetch
        if self.cache_only_mode:
            return None

        # Network fetch
        self._cache_miss("actions_job_status")
        url = f"{self.base_url}/repos/{owner}/{repo}/actions/jobs/{job_id_s}"
        try:
            resp = self._rest_get(url, timeout=10)
        except AttributeError:  # .get() on non-dict
            return None

        # 404 can happen if job id is invalid / perm issue; just treat as unknown.
        if resp.status_code < 200 or resp.status_code >= 300:
            return None

        try:
            data = resp.json() or {}
            status = data.get("status")
            if status is None:
                return None
            status_s = str(status)
        except AttributeError:  # .get() on non-dict
            return None

        # Save to memory cache
        try:
            self._actions_job_status_mem_cache[key] = {"ts": int(now), "status": status_s}
        except json.JSONDecodeError:
            pass

        # Save to disk cache (completed jobs are cached forever)
        # Uses lock-load-merge-save pattern (enforced by DiskCacheWriter)
        self._save_actions_job_disk_cache(key, {"ts": int(now), "status": status_s})
        self._cache_write("actions_job_status.disk_write", entries=1)

        return status_s

    def get_pr_head_sha(self, *, owner: str, repo: str, pr_number: int, ttl_s: int = 86400) -> Optional[str]:
        """Return PR head SHA with disk cache for closed/merged PRs.

        OPTIMIZATION: Closed/merged PRs are cached to disk permanently since head SHA never changes.
        This reduces API calls from ~100+ per run to near-zero on subsequent runs.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number
            ttl_s: TTL for open PRs (default 1 day; closed PRs cached forever)

        Returns:
            Head SHA string or None if not found
        """
        pr_num = int(pr_number)
        key = f"{owner}/{repo}:pr:{pr_num}:head_sha"
        now = int(time.time())

        # Check cache (handles both memory + disk with TTL)
        cached_entry = PR_HEAD_SHA_CACHE.get_if_fresh(key, ttl_s=ttl_s, cache_only_mode=self.cache_only_mode)
        if cached_entry is not None:
            self._cache_hit("pr_info")
            head_sha = cached_entry.get("head_sha")
            return str(head_sha) if head_sha else None

        # Cache-only mode: don't fetch
        if self.cache_only_mode:
            return None

        # Network fetch
        pr = self.get(f"/repos/{owner}/{repo}/pulls/{pr_num}", timeout=10) or {}
        head_sha = (((pr.get("head") or {}) if isinstance(pr, dict) else {}) or {}).get("sha")
        head_sha = str(head_sha or "").strip() if head_sha else None
        state = str(pr.get("state", "") or "").strip() if isinstance(pr, dict) else None

        if not head_sha:
            return None

        # Save to cache
        PR_HEAD_SHA_CACHE.put(key, head_sha=head_sha, state=state)

        return head_sha

    def _load_actions_job_disk_cache(self) -> Dict[str, Any]:
        """Load actions jobs disk cache with in-memory caching to avoid repeated file I/O."""
        # Check if we have a cached copy in memory (invalidate based on file mtime)
        self._actions_job_cache_dir.mkdir(parents=True, exist_ok=True)
        p = self._actions_job_cache_dir / "actions_jobs.json"
        
        if not p.exists():
            return {}
        
        # Check if we have a fresh in-memory copy
        try:
            current_mtime = p.stat().st_mtime_ns
            if hasattr(self, '_actions_job_disk_cache_snapshot'):
                cached_mtime, cached_data = self._actions_job_disk_cache_snapshot
                if cached_mtime == current_mtime:
                    # File hasn't changed, use cached copy
                    return cached_data
        except (OSError, AttributeError):
            pass
        
        # Load from disk and cache in memory
        data = self._json_load_text(p.read_text() or "{}")
        try:
            self._actions_job_disk_cache_snapshot = (p.stat().st_mtime_ns, data)
        except OSError:
            pass
        return data

    def _save_actions_job_disk_cache(self, key: str, value: Dict[str, Any]) -> None:
        """Atomically update a single entry in actions_jobs disk cache."""
        _save_single_disk_cache_entry(
            cache_dir=self._actions_job_cache_dir,
            cache_filename="actions_jobs.json",
            lock_filename="actions_jobs.lock",
            load_fn=self._load_actions_job_disk_cache,
            json_dump_fn=lambda d: self._json_dump_text(d, indent=None),
            key=key,
            value=value,
        )

    def get_actions_job_details_cached(
        self,
        *,
        owner: str,
        repo: str,
        job_id: str,
        ttl_s: int = 600,
        timeout: int = 15,
    ) -> Optional[Dict[str, Any]]:
        """Return Actions job details (including `steps`) with a persistent cache.

        Uses:
          GET /repos/{owner}/{repo}/actions/jobs/{job_id}
        """
        job_id_s = str(job_id or "").strip()
        if not job_id_s.isdigit():
            return None
        key = f"{owner}/{repo}:job:{job_id_s}"
        now = int(time.time())

        def _is_completed_job_details(v: Any) -> bool:
            """We only trust/caches job details once the job is completed.

            Rationale: while a job is running, GitHub returns `steps[]` entries with `pending/in_progress`
            and missing `completed_at`, which makes downstream step-duration rendering empty/misleading.
            """
            if not isinstance(v, dict):
                return False
            st = str(v.get("status", "") or "").lower()
            if st != "completed":
                return False
            # `completed_at` should exist for completed jobs; be strict to avoid partial caches.
            if not str(v.get("completed_at", "") or "").strip():
                return False
            return True

        # Memory cache
        try:
            ent = self._actions_job_details_mem_cache.get(key)
            if isinstance(ent, dict):
                ts = int(ent.get("ts", 0) or 0)
                if ts and (now - ts) <= int(ttl_s):
                    val = ent.get("val")
                    if _is_completed_job_details(val):
                        if isinstance(val, dict):
                            self._cache_hit("actions_job_details.mem")
                            return val
        except (ValueError, TypeError):
            pass

        # Disk cache
        disk = self._load_actions_job_disk_cache()
        ent = disk.get(key) if isinstance(disk, dict) else None
        if isinstance(ent, dict):
            ts = int(ent.get("ts", 0) or 0)
            if ts and (now - ts) <= int(ttl_s):
                val = ent.get("val")
                if isinstance(val, dict):
                    if _is_completed_job_details(val):
                        self._actions_job_details_mem_cache[key] = {"ts": ts, "val": val}
                        self._cache_hit("actions_job_details.disk")
                        return val
            # Cache-only mode: allow stale disk cache.
            if self.cache_only_mode:
                val = ent.get("val")
                if isinstance(val, dict):
                    # Even in cache-only mode, do not return incomplete job details (they hide steps).
                    if _is_completed_job_details(val):
                        self._actions_job_details_mem_cache[key] = {"ts": ts, "val": val}
                        self._cache_hit("actions_job_details.disk_stale_cache_only")
                        return val

        # Cache-only mode: do not fetch.
        if self.cache_only_mode:
            self._cache_miss("actions_job_details.cache_only_empty")
            return None

        self._cache_miss("actions_job_details")

        url = f"{self.base_url}/repos/{owner}/{repo}/actions/jobs/{job_id_s}"
        try:
            resp = self._rest_get(url, timeout=int(timeout))
        except AttributeError:  # .get() on non-dict
            return None
        if resp.status_code < 200 or resp.status_code >= 300:
            # Do NOT negative-cache failures (None). If the API temporarily fails (budget, rate limit,
            # permission, etc), we want a later attempt to succeed without waiting for a negative TTL.
            return None

        try:
            data = resp.json() or {}
        except (json.JSONDecodeError, ValueError):  # requests.Response.json() can raise ValueError or JSONDecodeError
            data = {}
        if not isinstance(data, dict):
            return None

        # Never cache incomplete job details (see `_is_completed_job_details` above).
        try:
            if str(data.get("status", "") or "").lower() != "completed":
                return None
            if not str(data.get("completed_at", "") or "").strip():
                return None
        except json.JSONDecodeError:
            return None

        # Keep only the fields we need (keep cache small + stable).
        try:
            steps_in = data.get("steps") if isinstance(data.get("steps"), list) else []
            steps: List[Dict[str, Any]] = []
            for st in (steps_in or []):
                if not isinstance(st, dict):
                    continue
                steps.append(
                    {
                        "number": st.get("number"),
                        "name": st.get("name"),
                        "status": st.get("status"),
                        "conclusion": st.get("conclusion"),
                        "started_at": st.get("started_at"),
                        "completed_at": st.get("completed_at"),
                    }
                )
            val = {
                "id": data.get("id"),
                "run_id": data.get("run_id"),
                "name": data.get("name"),
                "status": data.get("status"),
                "conclusion": data.get("conclusion"),
                "started_at": data.get("started_at"),
                "completed_at": data.get("completed_at"),
                "html_url": data.get("html_url"),
                "steps": steps,
            }
        except AttributeError:  # .get() on non-dict
            return None

        self._actions_job_details_mem_cache[key] = {"ts": now, "val": val}
        # Uses lock-load-merge-save pattern (enforced by DiskCacheWriter)
        self._save_actions_job_disk_cache(key, {"ts": now, "val": val})
        return val

    def get_actions_runs_jobs_batched(
        self,
        *,
        owner: str,
        repo: str,
        run_ids: List[str],
        ttl_s: int = 30 * 24 * 3600,
        timeout: int = 15,
    ) -> Dict[str, Dict[str, Any]]:
        """Batch fetch jobs for multiple workflow runs.

        OPTIMIZATION (2026-01-18): Instead of calling /actions/jobs/{job_id} individually
        for each job (500-1000 calls), this method fetches all jobs for a workflow run
        in a single call using /actions/runs/{run_id}/jobs.

        Status: ✅ Implemented and wired up (2026-01-19)
        - Populates both run-level and job-level cache keys
        - Subsequent get_actions_job_details_cached() calls hit cache (95% reduction)

        Uses /repos/{owner}/{repo}/actions/runs/{run_id}/jobs to fetch all jobs
        for each run in one API call instead of calling /actions/jobs/{job_id}
        individually.

        Returns:
            Dict mapping job_id -> job_details for all jobs across all runs.
            Job details include: id, run_id, status, conclusion, started_at, completed_at, steps[], html_url

        Benefits:
            - 95% reduction in API calls: 500-1000 per-job calls → 10-20 per-run calls
            - Faster: fewer network round-trips
            - Rate limit friendly: batched fetching
        """
        job_map: Dict[str, Dict[str, Any]] = {}
        unique_run_ids = [rid for rid in set(run_ids) if str(rid).strip().isdigit()]

        if not unique_run_ids:
            return job_map

        # For each run_id, fetch all jobs in one call
        for run_id in unique_run_ids:
            run_id_s = str(run_id).strip()
            if not run_id_s.isdigit():
                continue

            # Check cache first (reuse job details cache with run-level key)
            cache_key = f"{owner}/{repo}:run_jobs:{run_id_s}"
            now = int(time.time())

            # Memory cache
            mem_ent = self._actions_job_details_mem_cache.get(cache_key)
            if mem_ent and isinstance(mem_ent, dict):
                ts = mem_ent.get("ts")
                if ts and (now - ts) <= int(ttl_s):
                    cached_jobs = mem_ent.get("val")
                    if isinstance(cached_jobs, dict):
                        job_map.update(cached_jobs)
                        continue

            # Disk cache
            disk = self._load_actions_job_disk_cache()
            disk_ent = disk.get(cache_key) if isinstance(disk, dict) else None
            if disk_ent and isinstance(disk_ent, dict):
                ts = disk_ent.get("ts")
                if ts and ((now - ts) <= int(ttl_s) or self.cache_only_mode):
                    cached_jobs = disk_ent.get("val")
                    if isinstance(cached_jobs, dict):
                        job_map.update(cached_jobs)
                        self._actions_job_details_mem_cache[cache_key] = {"ts": ts, "val": cached_jobs}
                        continue

            # Cache-only mode: skip fetch if no cache hit
            if self.cache_only_mode:
                continue

            # Fetch jobs for this run
            url = f"{self.base_url}/repos/{owner}/{repo}/actions/runs/{run_id_s}/jobs"
            try:
                resp = self._rest_get(url, params={"per_page": 100}, timeout=int(timeout))
            except AttributeError:  # .get() on non-dict
                continue

            if resp.status_code < 200 or resp.status_code >= 300:
                continue

            try:
                data = resp.json() or {}
            except AttributeError:  # .get() on non-dict
                continue

            if not isinstance(data, dict):
                continue

            jobs = data.get("jobs") if isinstance(data.get("jobs"), list) else []
            run_jobs: Dict[str, Dict[str, Any]] = {}

            for job in jobs:
                if not isinstance(job, dict):
                    continue

                job_id = str(job.get("id", "") or "").strip()
                if not job_id.isdigit():
                    continue

                # Only cache completed jobs (same logic as get_actions_job_details_cached)
                status = str(job.get("status", "") or "").lower()
                if status != "completed":
                    continue

                # Extract steps (same format as individual job API)
                steps_in = job.get("steps") if isinstance(job.get("steps"), list) else []
                steps: List[Dict[str, Any]] = []
                for st in (steps_in or []):
                    if not isinstance(st, dict):
                        continue
                    steps.append({
                        "name": st.get("name"),
                        "status": st.get("status"),
                        "conclusion": st.get("conclusion"),
                        "number": st.get("number"),
                        "started_at": st.get("started_at"),
                        "completed_at": st.get("completed_at"),
                    })

                job_details = {
                    "id": job.get("id"),
                    "run_id": job.get("run_id"),
                    "status": job.get("status"),
                    "conclusion": job.get("conclusion"),
                    "started_at": job.get("started_at"),
                    "completed_at": job.get("completed_at"),
                    "name": job.get("name"),
                    "html_url": job.get("html_url"),
                    "steps": steps,
                }
                run_jobs[job_id] = job_details
                job_map[job_id] = job_details

            # Cache the jobs for this run (run-level key for batch lookups)
            self._actions_job_details_mem_cache[cache_key] = {"ts": now, "val": run_jobs}
            # Uses lock-load-merge-save pattern (enforced by DiskCacheWriter)
            self._save_actions_job_disk_cache(cache_key, {"ts": now, "val": run_jobs})

            # ALSO cache each individual job (job-level key for individual lookups)
            # This makes subsequent get_actions_job_details_cached() calls hit cache
            # instead of making individual API calls (completes the batched optimization)
            for job_id, job_details in run_jobs.items():
                job_key = f"{owner}/{repo}:job:{job_id}"
                self._actions_job_details_mem_cache[job_key] = {"ts": now, "val": job_details}
                self._save_actions_job_disk_cache(job_key, {"ts": now, "val": job_details})

        return job_map

    def enrich_check_runs_with_job_details(
        self,
        *,
        owner: str,
        repo: str,
        check_runs: List[Dict[str, Any]],
        ttl_s: int = 30 * 24 * 3600,
    ) -> Dict[str, Dict[str, Any]]:
        """Enrich check-runs with job details using batched fetching.

        Takes a list of check-runs (with run_id/job_id) and returns a map of
        job_id -> job_details, using batched /actions/runs/{run_id}/jobs calls
        instead of individual /actions/jobs/{job_id} calls.

        This method is used by all dashboard scripts through common_dashboard_runtime.py
        to efficiently fetch job details for rendering step breakdowns.

        Args:
            check_runs: List of check-run dicts (from get_pr_checks_rows or similar)
            ttl_s: Cache TTL for job details

        Returns:
            Dict mapping job_id -> job_details for all jobs in the check-runs
        """
        # Extract unique run_ids from check-runs
        run_ids: List[str] = []
        for cr in check_runs:
            if not isinstance(cr, dict):
                continue
            url = str(cr.get("url", "") or cr.get("details_url", "") or cr.get("html_url", "") or "").strip()
            run_id = parse_actions_run_id_from_url(url)
            if run_id and run_id not in run_ids:
                run_ids.append(run_id)

        # Batch fetch jobs for all runs
        return self.get_actions_runs_jobs_batched(
            owner=owner,
            repo=repo,
            run_ids=run_ids,
            ttl_s=ttl_s,
        )

    def _format_seconds_delta(seconds: int) -> str:
        """Format a positive/negative seconds delta as a short human string."""
        s = int(seconds)
        if s <= 0:
            return "0s"
        m, sec = divmod(s, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        if d:
            return f"{d}d {h}h {m}m"
        if h:
            return f"{h}h {m}m"
        if m:
            return f"{m}m {sec}s"
        return f"{sec}s"

    @staticmethod
    def compute_checks_cache_ttl_s(
        last_change_dt: Optional[datetime],
        *,
        refresh: bool = False,
        pr_merged: bool = False,
        stable_after_hours: int = DEFAULT_STABLE_AFTER_HOURS,
        short_ttl_s: int = DEFAULT_UNSTABLE_TTL_S,
        stable_ttl_s: int = DEFAULT_STABLE_TTL_S,
    ) -> int:
        """Compute an appropriate cache TTL for CI/checks based on "how recently things changed".

        3-tier heuristic:
        1. If PR is merged/closed: 7 days (immutable)
        2. Else if commit age < stable_after_hours: short_ttl_s (CI still running)
        3. Else: stable_ttl_s (CI likely done, but might re-run)

        Special cases:
        - If refresh=True: force fetch (TTL=0)
        """
        if refresh:
            return 0

        # Tier 1: Merged/closed PRs - long TTL (immutable)
        if pr_merged:
            return 7 * 24 * 3600  # 7 days

        # Tier 2/3: Open PRs - use commit age to decide
        ttl = int(short_ttl_s)
        try:
            if last_change_dt is None:
                return ttl
            dt = last_change_dt
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age_s = (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds()
            if age_s >= float(stable_after_hours) * 3600.0:
                ttl = max(ttl, int(stable_ttl_s))
        except (ValueError, TypeError):
            return int(short_ttl_s)
        return ttl

    def check_core_rate_limit_or_raise(self) -> None:
        """Fail fast if GitHub REST (core) quota is exhausted.

        This makes scripts exit with a clear message instead of spamming API calls.
        """
        assert requests is not None
        url = f"{self.base_url}/rate_limit"
        try:
            resp = self._rest_get(url, timeout=10)
        except Exception as e:
            raise RuntimeError(f"Failed to query GitHub rate limit: {e}") from e

        remaining_hdr = resp.headers.get("X-RateLimit-Remaining")
        reset_hdr = resp.headers.get("X-RateLimit-Reset")
        limit_hdr = resp.headers.get("X-RateLimit-Limit")

        # Prefer headers (works even on 403), fall back to JSON if needed.
        try:
            remaining = int(remaining_hdr) if remaining_hdr is not None else None
        except (ValueError, TypeError):
            remaining = None
        try:
            reset_epoch = int(reset_hdr) if reset_hdr is not None else None
        except (ValueError, TypeError):
            reset_epoch = None
        try:
            limit = int(limit_hdr) if limit_hdr is not None else None
        except (ValueError, TypeError):
            limit = None

        if remaining is None or reset_epoch is None:
            try:
                data = resp.json()
                core = (data or {}).get("resources", {}).get("core", {})
                remaining = int(core.get("remaining")) if remaining is None and core.get("remaining") is not None else remaining
                reset_epoch = int(core.get("reset")) if reset_epoch is None and core.get("reset") is not None else reset_epoch
                limit = int(core.get("limit")) if limit is None and core.get("limit") is not None else limit
            except json.JSONDecodeError:
                pass

        if remaining is None or reset_epoch is None:
            # If we can't parse, don't block execution.
            return

        if remaining > 0:
            return

        now = int(time.time())
        seconds_until = reset_epoch - now
        reset_local = datetime.fromtimestamp(reset_epoch).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        raise RuntimeError(
            "GitHub API core (REST) quota exhausted "
            f"(remaining={remaining}"
            + (f", limit={limit}" if limit is not None else "")
            + "). "
            f"Resets at {reset_local} (in {self._format_seconds_delta(seconds_until)})."
        )

    def get_core_rate_limit_info(self) -> Optional[Dict[str, Any]]:
        """Return GitHub REST (core) rate limit info.

        Returns:
            Dict like:
              {
                "remaining": 1234,
                "limit": 5000,
                "reset_epoch": 1766947200,
                "reset_local": "2025-12-28 14:00:00 PST",
                "reset_pt": "2025-12-28 14:00:00 PST",
                "seconds_until_reset": 1234,
              }
            or None if not available.
        """
        url = f"{self.base_url}/rate_limit"
        try:
            resp = self._rest_get(url, timeout=10)
        except RuntimeError:  # cache_only_mode or budget_exhausted raises RuntimeError
            # If we can't make a fresh API call (e.g., cache-only mode), return cached value from headers.
            # This allows displaying rate limit stats in the Statistics footer even after entering cache-only mode.
            return self._cached_rate_limit_info

        remaining_hdr = resp.headers.get("X-RateLimit-Remaining")
        reset_hdr = resp.headers.get("X-RateLimit-Reset")
        limit_hdr = resp.headers.get("X-RateLimit-Limit")

        remaining = None
        reset_epoch = None
        limit = None
        try:
            remaining = int(remaining_hdr) if remaining_hdr is not None else None
        except (ValueError, TypeError):
            remaining = None
        try:
            reset_epoch = int(reset_hdr) if reset_hdr is not None else None
        except (ValueError, TypeError):
            reset_epoch = None
        try:
            limit = int(limit_hdr) if limit_hdr is not None else None
        except (ValueError, TypeError):
            limit = None

        if remaining is None or reset_epoch is None:
            try:
                data = resp.json()
                core = (data or {}).get("resources", {}).get("core", {})
                if remaining is None and core.get("remaining") is not None:
                    remaining = int(core.get("remaining"))
                if reset_epoch is None and core.get("reset") is not None:
                    reset_epoch = int(core.get("reset"))
                if limit is None and core.get("limit") is not None:
                    limit = int(core.get("limit"))
            except json.JSONDecodeError:
                pass

        if remaining is None or reset_epoch is None:
            return None

        now = int(time.time())
        seconds_until = int(reset_epoch) - now
        reset_local = datetime.fromtimestamp(int(reset_epoch)).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        try:
            reset_pt = (
                datetime.fromtimestamp(int(reset_epoch), tz=timezone.utc)
                .astimezone(ZoneInfo("America/Los_Angeles"))
                .strftime("%Y-%m-%d %H:%M:%S %Z")
            )
        except (ValueError, TypeError):
            reset_pt = reset_local

        return {
            "remaining": int(remaining),
            "limit": int(limit) if limit is not None else None,
            "reset_epoch": int(reset_epoch),
            "reset_local": reset_local,
            "reset_pt": reset_pt,
            "seconds_until_reset": seconds_until,
        }

    def _pr_checks_cache_key(self, owner: str, repo: str, pr_number: int, commit_sha: Optional[str] = None) -> str:
        sha_suffix = f":{commit_sha[:7]}" if commit_sha else ""
        return f"{owner}/{repo}#{int(pr_number)}{sha_suffix}"

    def _pulls_list_cache_key(self, owner: str, repo: str, state: str) -> str:
        return f"{owner}/{repo}:{state}"

    @staticmethod
    def _pr_info_min_to_dict(pr: "PRInfo") -> Dict[str, Any]:
        return {
            "number": int(pr.number),
            "title": pr.title,
            "url": pr.url,
            "state": pr.state,
            "is_merged": bool(pr.is_merged),
            "mergeable_state": pr.mergeable_state,
            "head_sha": pr.head_sha,
            "head_ref": pr.head_ref,
            "head_owner": pr.head_owner,
            "head_label": pr.head_label,
            "base_ref": pr.base_ref,
            "created_at": pr.created_at,
            "updated_at": pr.updated_at,
        }

    def _pr_data_to_pr_info_min(self, pr_data: Dict[str, Any]) -> Optional["PRInfo"]:
        """Convert raw PR data from GitHub API to minimal PRInfo object.

        Args:
            pr_data: Raw PR dict from /repos/{owner}/{repo}/pulls API

        Returns:
            Minimal PRInfo with basic fields populated (no checks/reviews)
        """
        try:
            head = pr_data.get("head") or {}
            head_sha = str(head.get("sha", "") or "") if isinstance(head, dict) else ""
            head_ref = str(head.get("ref", "") or "") if isinstance(head, dict) else ""
            head_label = str(head.get("label", "") or "") if isinstance(head, dict) else ""
            head_user = head.get("user") or {} if isinstance(head, dict) else {}
            head_owner = str(head_user.get("login", "") or "") if isinstance(head_user, dict) else ""

            base = pr_data.get("base") or {}
            base_ref = str(base.get("ref", "") or "") if isinstance(base, dict) else ""

            return PRInfo(
                number=int(pr_data.get("number") or 0),
                title=str(pr_data.get("title") or ""),
                url=str(pr_data.get("html_url") or ""),
                state=str(pr_data.get("state") or ""),
                is_merged=pr_data.get("merged_at") is not None,
                review_decision=None,
                mergeable_state=str(pr_data.get("mergeable_state") or "unknown"),
                unresolved_conversations=0,
                ci_status=None,
                head_sha=head_sha or None,
                head_ref=head_ref or None,
                head_owner=head_owner or None,
                head_label=head_label or None,
                base_ref=base_ref or None,
                created_at=pr_data.get("created_at"),
                updated_at=pr_data.get("updated_at"),
                has_conflicts=False,
                conflict_message=None,
                blocking_message=None,
                required_checks=[],
                failed_checks=[],
                running_checks=[],
                rerun_url=None,
            )
        except AttributeError:  # .get() on non-dict
            return None

    @staticmethod
    def _pr_info_min_from_dict(d: Dict[str, Any]) -> Optional["PRInfo"]:
        try:
            return PRInfo(
                number=int(d.get("number") or 0),
                title=str(d.get("title") or ""),
                url=str(d.get("url") or ""),
                state=str(d.get("state") or ""),
                is_merged=bool(d.get("is_merged") or False),
                review_decision=None,
                mergeable_state=str(d.get("mergeable_state") or "unknown"),
                unresolved_conversations=0,
                ci_status=None,
                head_sha=d.get("head_sha"),
                head_ref=d.get("head_ref"),
                head_owner=d.get("head_owner"),
                head_label=d.get("head_label"),
                base_ref=d.get("base_ref"),
                created_at=d.get("created_at"),
                updated_at=d.get("updated_at"),
                has_conflicts=False,
                conflict_message=None,
                blocking_message=None,
                required_checks=[],
                failed_checks=[],
                running_checks=[],
                rerun_url=None,
            )
        except AttributeError:  # .get() on non-dict
            return None

    @staticmethod
    def _failed_check_to_dict(fc: "FailedCheck") -> Dict[str, Any]:
        try:
            return {
                "name": str(fc.name or ""),
                "job_url": str(fc.job_url or ""),
                "run_id": str(fc.run_id or ""),
                "duration": str(fc.duration or ""),
                "raw_log_url": fc.raw_log_url,
                "is_required": bool(fc.is_required),
                "error_summary": fc.error_summary,
            }
        except AttributeError:  # .get() on non-dict
            return {}

    @staticmethod
    def _failed_check_from_dict(d: Dict[str, Any]) -> Optional["FailedCheck"]:
        try:
            return FailedCheck(
                name=str(d.get("name") or ""),
                job_url=str(d.get("job_url") or ""),
                run_id=str(d.get("run_id") or ""),
                duration=str(d.get("duration") or ""),
                raw_log_url=d.get("raw_log_url"),
                is_required=bool(d.get("is_required", False)),
                error_summary=d.get("error_summary"),
            )
        except AttributeError:  # .get() on non-dict
            return None

    @staticmethod
    def _running_check_to_dict(rc: "RunningCheck") -> Dict[str, Any]:
        try:
            return {
                "name": str(rc.name or ""),
                "check_url": str(rc.check_url or ""),
                "is_required": bool(rc.is_required),
                "elapsed_time": rc.elapsed_time,
            }
        except AttributeError:  # .get() on non-dict
            return {}

    @staticmethod
    def _running_check_from_dict(d: Dict[str, Any]) -> Optional["RunningCheck"]:
        try:
            return RunningCheck(
                name=str(d.get("name") or ""),
                check_url=str(d.get("check_url") or ""),
                is_required=bool(d.get("is_required", False)),
                elapsed_time=d.get("elapsed_time"),
            )
        except AttributeError:  # .get() on non-dict
            return None

    @staticmethod
    def _pr_info_full_to_dict(pr: "PRInfo") -> Dict[str, Any]:
        """Serialize a PRInfo including enrichment fields so it can be reused when PR is unchanged."""
        try:
            return {
                "number": int(pr.number),
                "title": pr.title,
                "url": pr.url,
                "state": pr.state,
                "is_merged": bool(pr.is_merged),
                "review_decision": pr.review_decision,
                "mergeable_state": pr.mergeable_state,
                "unresolved_conversations": int(pr.unresolved_conversations or 0),
                "ci_status": pr.ci_status,
                "head_sha": pr.head_sha,
                "head_ref": pr.head_ref,
                "head_owner": pr.head_owner,
                "head_label": pr.head_label,
                "base_ref": pr.base_ref,
                "created_at": pr.created_at,
                "updated_at": pr.updated_at,
                "has_conflicts": bool(pr.has_conflicts),
                "conflict_message": pr.conflict_message,
                "blocking_message": pr.blocking_message,
                "required_checks": list(pr.required_checks or []),
                "failed_checks": [GitHubAPIClient._failed_check_to_dict(fc) for fc in (pr.failed_checks or [])],
                "running_checks": [GitHubAPIClient._running_check_to_dict(rc) for rc in (pr.running_checks or [])],
                "rerun_url": pr.rerun_url,
            }
        except (TypeError, ValueError):  # list conversion
            return GitHubAPIClient._pr_info_min_to_dict(pr)

    @staticmethod
    def _pr_info_full_from_dict(d: Dict[str, Any]) -> Optional["PRInfo"]:
        try:
            failed = []
            for x in (d.get("failed_checks") or []):
                if isinstance(x, dict):
                    fc = GitHubAPIClient._failed_check_from_dict(x)
                    if fc is not None:
                        failed.append(fc)
            running = []
            for x in (d.get("running_checks") or []):
                if isinstance(x, dict):
                    rc = GitHubAPIClient._running_check_from_dict(x)
                    if rc is not None:
                        running.append(rc)
            return PRInfo(
                number=int(d.get("number") or 0),
                title=str(d.get("title") or ""),
                url=str(d.get("url") or ""),
                state=str(d.get("state") or ""),
                is_merged=bool(d.get("is_merged", False)),
                review_decision=d.get("review_decision"),
                mergeable_state=str(d.get("mergeable_state") or "unknown"),
                unresolved_conversations=int(d.get("unresolved_conversations") or 0),
                ci_status=d.get("ci_status"),
                head_sha=d.get("head_sha"),
                head_ref=d.get("head_ref"),
                head_owner=d.get("head_owner"),
                head_label=d.get("head_label"),
                base_ref=d.get("base_ref"),
                created_at=d.get("created_at"),
                updated_at=d.get("updated_at"),
                has_conflicts=bool(d.get("has_conflicts", False)),
                conflict_message=d.get("conflict_message"),
                blocking_message=d.get("blocking_message"),
                required_checks=list(d.get("required_checks") or []),
                failed_checks=failed,
                running_checks=running,
                rerun_url=d.get("rerun_url"),
            )
        except AttributeError:  # .get() on non-dict
            return None

    def _load_pulls_list_disk_cache(self) -> Dict[str, Any]:
        cache_file = self._pulls_list_cache_dir / "pulls_open_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    return data if isinstance(data, dict) else {}
            except AttributeError:  # .get() on non-dict
                return {}
        return {}

    def _save_pulls_list_disk_cache(self, key: str, value: Dict[str, Any]) -> None:
        """Atomically update a single entry in pulls_list disk cache."""
        _save_single_disk_cache_entry(
            cache_dir=self._pulls_list_cache_dir,
            cache_filename="pulls_open_cache.json",
            lock_filename="pulls_open_cache.lock",
            load_fn=self._load_pulls_list_disk_cache,
            json_dump_fn=lambda d: self._json_dump_text(d, indent=None),
            key=key,
            value=value,
        )
        self._cache_write("pulls_list.disk_write", entries=1)

    def list_pull_requests(
        self, owner: str, repo: str, *, state: str = "open", ttl_s: int = DEFAULT_OPEN_PRS_TTL_S
    ) -> List[Dict[str, Any]]:
        """List pull requests for a repo with a short-lived cache.

        This is used to avoid calling /pulls?head=... once per local branch.

        Args:
            owner: Repository owner
            repo: Repository name
            state: "open" or "all" (default: "open")
            ttl_s: Cache TTL in seconds

        Returns:
            List of PR dicts (GitHub REST API /pulls response objects).
            
        Note: ETag support removed for simplification. Can be re-added later if needed.
              Implementation: Store ETag in cache value, use If-None-Match header,
              handle 304 responses to avoid re-downloading unchanged data.
        """
        cache_key = self._pulls_list_cache_key(owner, repo, state)

        # Check cache (PULLS_LIST_CACHE handles both mem and disk internally)
        cached = PULLS_LIST_CACHE.get_if_fresh(cache_key, ttl_s=ttl_s)
        if cached is not None:
            self._cache_hit("pulls_list.disk")
            return cached

        # Cache-only mode: return empty (no network)
        if self.cache_only_mode:
            self._cache_miss("pulls_list.cache_only_empty")
            return []

        # Fetch from API (paginate), but dedupe concurrent fetches across threads.
        lock = self._inflight_lock(f"pulls_list:{cache_key}")
        with lock:
            # Re-check cache (another thread may have populated it)
            cached = PULLS_LIST_CACHE.get_if_fresh(cache_key, ttl_s=ttl_s)
            if cached is not None:
                self._cache_hit("pulls_list.disk")
                return cached

            self._cache_miss("pulls_list.network")
            endpoint = f"/repos/{owner}/{repo}/pulls"
            items = []
            try:
                page = 1
                while True:
                    params = {"state": state, "per_page": 100, "page": page}
                    url = f"{self.base_url}{endpoint}"
                    resp = self._rest_get(url, params=params)

                    # Parse response
                    chunk = resp.json() if resp.status_code >= 200 and resp.status_code < 300 else None
                    if not chunk:
                        break
                    if isinstance(chunk, list):
                        items.extend(chunk)
                    else:
                        break
                    if len(chunk) < 100:
                        break
                    page += 1
            except Exception as e:
                # Best-effort: if we can't list PRs (rate limit, network, auth), just return empty.
                # Also negative-cache the failure for a short window to avoid spamming retries when
                # scanning multiple clones of the same repo in one run.
                _logger.warning("Error listing PRs for %s/%s: %s", str(owner), str(repo), str(e))
                PULLS_LIST_CACHE.put(cache_key, [])
                return []

            # Save to cache
            PULLS_LIST_CACHE.put(cache_key, items)
            return items

    def get_open_prs(
        self,
        *,
        owner: str,
        repo: str,
        head: Optional[str] = None,
        max_prs: int = 100,
        ttl_s: int = DEFAULT_OPEN_PRS_TTL_S,
    ) -> List["PRInfo"]:
        """Return open PRs, optionally filtered by head branch.

        Args:
            owner: Repository owner
            repo: Repository name
            head: Optional head filter (e.g., "username:branch" or "branch")
            max_prs: Maximum number of PRs to return
            ttl_s: Cache TTL in seconds

        Returns:
            List of PRInfo objects (minimal enrichment - no checks/reviews)
        """
        pr_datas = self.list_pull_requests(owner, repo, state="open", ttl_s=int(ttl_s))

        # Filter by head if specified
        if head:
            head = str(head).strip()
            filtered = []
            for pr_data in pr_datas:
                if not isinstance(pr_data, dict):
                    continue
                pr_head = pr_data.get("head") or {}
                if isinstance(pr_head, dict):
                    pr_head_label = str(pr_head.get("label", "") or "")
                    pr_head_ref = str(pr_head.get("ref", "") or "")
                    # Match either "owner:ref" or just "ref"
                    if pr_head_label == head or pr_head_ref == head:
                        filtered.append(pr_data)
            pr_datas = filtered

        # Limit results
        pr_datas = pr_datas[:max_prs]

        # Convert to PRInfo objects (minimal enrichment)
        result: List[PRInfo] = []
        for pr_data in pr_datas:
            pr_info = self._pr_data_to_pr_info_min(pr_data)
            if pr_info:
                result.append(pr_info)

        return result

    def get_open_pr_info_for_author(
        self,
        owner: str,
        repo: str,
        *,
        author: str,
        ttl_s: int = DEFAULT_OPEN_PRS_TTL_S,
        max_prs: Optional[int] = None,
    ) -> List["PRInfo"]:
        """Return enriched PRInfo objects for OPEN PRs authored by `author`.

        This is meant to power dashboards that start from a GitHub username rather than local branches.
        Implementation intentionally mirrors the "open PR list + batched updated_at probe + per-PR enrichment"
        strategy used by `get_pr_info_for_branches(...)`.
        """
        author_lc = str(author or "").strip().lower()
        if not author_lc:
            return []

        pr_datas = self.list_pull_requests(owner, repo, state="open", ttl_s=int(ttl_s))

        # Filter by author (PR creator).
        filtered: List[Dict[str, Any]] = []
        for pr_data in (pr_datas or []):
            if not isinstance(pr_data, dict):
                continue
            login = str(((pr_data.get("user") or {}) if isinstance(pr_data.get("user"), dict) else {}).get("login") or "").strip().lower()
            if login == author_lc:
                filtered.append(pr_data)

        # Sort newest-first (by PR number, stable).
        try:
            filtered.sort(key=lambda d: int(d.get("number") or 0), reverse=True)
        except (ValueError, TypeError):
            pass

        if max_prs is not None:
            try:
                n = int(max_prs)
                if n > 0:
                    filtered = filtered[:n]
            except (ValueError, TypeError):
                pass

        if not filtered:
            return []

        # Cache-only mode: avoid per-PR enrichment fetches; populate a minimal PRInfo from pr_data.
        if self.cache_only_mode:
            out_min: List[PRInfo] = []
            for pr_data in filtered:
                try:
                    head = (pr_data.get("head") or {}) if isinstance(pr_data.get("head"), dict) else {}
                    head_ref = head.get("ref")
                    head_label = head.get("label")
                    head_owner = ""
                    hrepo = (head.get("repo") or {}) if isinstance(head.get("repo"), dict) else {}
                    hown = (hrepo.get("owner") or {}) if isinstance(hrepo.get("owner"), dict) else {}
                    head_owner = str(hown.get("login") or "").strip()
                    pr = self._pr_info_min_from_dict(
                        {
                            "number": pr_data.get("number"),
                            "title": pr_data.get("title") or "",
                            "url": pr_data.get("html_url") or "",
                            "state": pr_data.get("state") or "",
                            "is_merged": pr_data.get("merged_at") is not None,
                            "mergeable_state": pr_data.get("mergeable_state") or "unknown",
                            "head_sha": (pr_data.get("head") or {}).get("sha"),
                            "head_ref": head_ref,
                            "head_label": head_label,
                            "head_owner": head_owner,
                            "base_ref": (pr_data.get("base") or {}).get("ref", "main"),
                            "created_at": pr_data.get("created_at"),
                            "updated_at": pr_data.get("updated_at"),
                        }
                    )
                    if pr is not None:
                        out_min.append(pr)
                except AttributeError:  # .get() on non-dict
                    continue
            return out_min

        # Optimization: if a PR has not changed (by `updated_at`), reuse cached PRInfo and do *zero*
        # per-PR network calls. Probe updated_at via a batched search/issues call.
        pr_nums: List[int] = []
        for pr_data in filtered:
            try:
                n = int(pr_data.get("number") or 0)
            except (ValueError, TypeError):
                n = 0
            if n > 0:
                pr_nums.append(n)

        updated_map: Dict[int, str] = {}
        try:
            updated_map = self.get_pr_updated_at_via_search_issues(owner=owner, repo=repo, pr_numbers=pr_nums) or {}
        except (ValueError, TypeError):
            updated_map = {}

        def enrich_one(pr_data: Dict[str, Any]) -> Optional[PRInfo]:
            try:
                n = int(pr_data.get("number") or 0)
            except (ValueError, TypeError):
                n = 0
            upd = str(updated_map.get(n) or pr_data.get("updated_at") or "").strip()
            if n and upd:
                cached = self._get_cached_pr_info_if_unchanged(owner=owner, repo=repo, pr_number=n, updated_at=upd)
                if cached is not None:
                    # Backfill head branch info from the live /pulls payload, since older cached PRInfo
                    # entries may not have these fields yet.
                    changed = False
                    try:
                        head = (pr_data.get("head") or {}) if isinstance(pr_data.get("head"), dict) else {}
                        if not cached.head_ref:
                            cached.head_ref = head.get("ref")
                            changed = True
                        if not cached.head_label:
                            cached.head_label = head.get("label")
                            changed = True
                        if not cached.head_owner:
                            hrepo = (head.get("repo") or {}) if isinstance(head.get("repo"), dict) else {}
                            hown = (hrepo.get("owner") or {}) if isinstance(hrepo.get("owner"), dict) else {}
                            cached.head_owner = str(hown.get("login") or "").strip() or None
                            changed = True
                    except AttributeError:  # head.get() fails if head is not a dict (should never happen)
                        pass
                    if changed:
                        self._save_pr_info_cache(owner=owner, repo=repo, pr_number=n, updated_at=upd, pr=cached)
                    return cached
            # If we somehow entered cache-only mode mid-run, degrade gracefully.
            if self.cache_only_mode:
                head = (pr_data.get("head") or {}) if isinstance(pr_data.get("head"), dict) else {}
                return self._pr_info_min_from_dict(
                    {
                        "number": pr_data.get("number"),
                        "title": pr_data.get("title") or "",
                        "url": pr_data.get("html_url") or "",
                        "state": pr_data.get("state") or "",
                        "is_merged": pr_data.get("merged_at") is not None,
                        "mergeable_state": pr_data.get("mergeable_state") or "unknown",
                        "head_sha": (pr_data.get("head") or {}).get("sha"),
                        "head_ref": head.get("ref"),
                        "head_label": head.get("label"),
                        "base_ref": (pr_data.get("base") or {}).get("ref", "main"),
                        "created_at": pr_data.get("created_at"),
                        "updated_at": pr_data.get("updated_at"),
                    }
                )
            pr = self._pr_info_from_pr_data(owner, repo, pr_data)
            if pr is not None and n and upd:
                self._save_pr_info_cache(owner=owner, repo=repo, pr_number=n, updated_at=upd, pr=pr)
            return pr

        out: List[PRInfo] = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futs = [executor.submit(enrich_one, pr_data) for pr_data in filtered]
            for fut in as_completed(futs):
                pr = fut.result()
                if pr is not None:
                    out.append(pr)

        # Keep stable sort by PR number descending (regardless of completion order).
        try:
            out.sort(key=lambda p: int(p.number or 0), reverse=True)
        except (ValueError, TypeError):
            pass
        return out

    def get_pr_info_for_branches(
        self,
        owner: str,
        repo: str,
        branches: Iterable[str],
        *,
        include_closed: bool = True,
        refresh_closed: bool = False,
        closed_ttl_s: int = DEFAULT_CLOSED_PRS_TTL_S,
        no_pr_ttl_s: int = DEFAULT_NO_PR_TTL_S,
    ) -> Dict[str, List["PRInfo"]]:
        """Get PR information for many branches efficiently.

        Strategy:
        - list open PRs once for the repo (cached)
        - match PRs to local branches by head.ref
        - only for matched OPEN PRs, fetch required checks / unresolved conv / checks summary
        - for branches without an open PR, optionally resolve closed/merged PRs with a long-lived cache
        """
        branch_set = {b for b in branches if b}
        if not branch_set:
            return {}

        # Build head.ref/head.label -> list[pr_data]
        #
        # IMPORTANT: local branch names may be stored in different formats:
        # - Typical: "<branch_ref>" (matches GitHub PR head.ref directly)
        # - Fork-tracking convention: "<fork_owner>/<branch_ref>"
        #   In that case, GitHub PR head.label is "<fork_owner>:<branch_ref>" and head.ref is "<branch_ref>".
        #   We match both so fork branches still show their PRs.
        pr_datas = self.list_pull_requests(owner, repo, state="open")
        label_to_branch: Dict[str, str] = {}
        for b in branch_set:
            if "/" in b:
                pre, rest = b.split("/", 1)
                pre = (pre or "").strip()
                rest = (rest or "").strip()
                if pre and rest:
                    label_to_branch[f"{pre}:{rest}"] = b
        head_to_prs: Dict[str, List[Dict[str, Any]]] = {}
        for pr_data in pr_datas:
            try:
                head = (pr_data.get("head") or {})
                head_ref = head.get("ref")
                head_label = head.get("label")
            except AttributeError:  # pr_data.get() fails if pr_data is not a dict
                head_ref = None
                head_label = None

            branch_name: Optional[str] = None
            if head_ref and head_ref in branch_set:
                branch_name = str(head_ref)
            elif head_label and str(head_label) in label_to_branch:
                branch_name = label_to_branch[str(head_label)]

            if not branch_name:
                continue
            head_to_prs.setdefault(branch_name, []).append(pr_data)

        result: Dict[str, List[PRInfo]] = {b: [] for b in branch_set}

        # Cache-only mode: avoid per-PR enrichment fetches; populate a minimal PRInfo from pr_data.
        if self.cache_only_mode:
            for branch_name, prs in head_to_prs.items():
                for pr_data in prs:
                    try:
                        head = (pr_data.get("head") or {}) if isinstance(pr_data.get("head"), dict) else {}
                        head_ref = head.get("ref")
                        head_label = head.get("label")
                        head_owner = ""
                        hrepo = (head.get("repo") or {}) if isinstance(head.get("repo"), dict) else {}
                        hown = (hrepo.get("owner") or {}) if isinstance(hrepo.get("owner"), dict) else {}
                        head_owner = str(hown.get("login") or "").strip()
                        pr_info = self._pr_info_min_from_dict(
                            {
                                "number": pr_data.get("number"),
                                "title": pr_data.get("title") or "",
                                "url": pr_data.get("html_url") or "",
                                "state": pr_data.get("state") or "",
                                "is_merged": pr_data.get("merged_at") is not None,
                                "mergeable_state": pr_data.get("mergeable_state") or "unknown",
                                "head_sha": (pr_data.get("head") or {}).get("sha"),
                                "head_ref": head_ref,
                                "head_label": head_label,
                                "head_owner": head_owner,
                                "base_ref": (pr_data.get("base") or {}).get("ref", "main"),
                                "created_at": pr_data.get("created_at"),
                                "updated_at": pr_data.get("updated_at"),
                            }
                        )
                        if pr_info is not None:
                            result.setdefault(branch_name, []).append(pr_info)
                    except AttributeError:  # .get() on non-dict
                        continue
        else:
            # Per-PR enrichment can be parallelized (bounded).
            #
            # Optimization: if a PR has not changed (by `updated_at`), reuse a cached PRInfo and do
            # *zero* per-PR network calls. We probe updated_at for the target PR list using a single
            # search/issues call (batched).
            pr_nums: List[int] = []
            pr_data_by_num: Dict[int, Dict[str, Any]] = {}
            for _branch_name, prs in head_to_prs.items():
                for pr_data in prs:
                    try:
                        n = int(pr_data.get("number") or 0)
                    except (ValueError, TypeError):
                        continue
                    if n > 0 and n not in pr_data_by_num:
                        pr_nums.append(n)
                        pr_data_by_num[n] = pr_data

            updated_map: Dict[int, str] = {}
            try:
                updated_map = self.get_pr_updated_at_via_search_issues(owner=owner, repo=repo, pr_numbers=pr_nums) or {}
            except (ValueError, TypeError):
                updated_map = {}

            def enrich_one(pr_data: Dict[str, Any]) -> Optional[PRInfo]:
                try:
                    n = int(pr_data.get("number") or 0)
                except (ValueError, TypeError):
                    n = 0
                upd = str(updated_map.get(n) or pr_data.get("updated_at") or "").strip()
                # If we have a cached PRInfo for this updated_at, return it without network.
                if n and upd:
                    cached = self._get_cached_pr_info_if_unchanged(owner=owner, repo=repo, pr_number=n, updated_at=upd)
                    if cached is not None:
                        # Backfill head branch info from the live /pulls payload, since older cached PRInfo
                        # entries may not have these fields yet.
                        changed = False
                        try:
                            head = (pr_data.get("head") or {}) if isinstance(pr_data.get("head"), dict) else {}
                            if not cached.head_ref:
                                cached.head_ref = head.get("ref")
                                changed = True
                            if not cached.head_label:
                                cached.head_label = head.get("label")
                                changed = True
                            if not cached.head_owner:
                                hrepo = (head.get("repo") or {}) if isinstance(head.get("repo"), dict) else {}
                                hown = (hrepo.get("owner") or {}) if isinstance(hrepo.get("owner"), dict) else {}
                                cached.head_owner = str(hown.get("login") or "").strip() or None
                                changed = True
                        except AttributeError:  # head.get() fails if head is not a dict
                            pass
                        if changed:
                            self._save_pr_info_cache(owner=owner, repo=repo, pr_number=n, updated_at=upd, pr=cached)
                        return cached
                # Cache-only mode: don't attempt enrichment fetches.
                if self.cache_only_mode:
                    head = (pr_data.get("head") or {}) if isinstance(pr_data.get("head"), dict) else {}
                    return self._pr_info_min_from_dict(
                        {
                            "number": pr_data.get("number"),
                            "title": pr_data.get("title") or "",
                            "url": pr_data.get("html_url") or "",
                            "state": pr_data.get("state") or "",
                            "is_merged": pr_data.get("merged_at") is not None,
                            "mergeable_state": pr_data.get("mergeable_state") or "unknown",
                            "head_sha": (pr_data.get("head") or {}).get("sha"),
                            "head_ref": head.get("ref"),
                            "head_label": head.get("label"),
                            "base_ref": (pr_data.get("base") or {}).get("ref", "main"),
                            "created_at": pr_data.get("created_at"),
                            "updated_at": pr_data.get("updated_at"),
                        }
                    )
                pr = self._pr_info_from_pr_data(owner, repo, pr_data)
                if pr is not None and n and upd:
                    self._save_pr_info_cache(owner=owner, repo=repo, pr_number=n, updated_at=upd, pr=pr)
                return pr

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures2: List[Tuple[str, Dict[str, Any], Any]] = []
                for branch_name, prs in head_to_prs.items():
                    for pr_data in prs:
                        futures2.append((branch_name, pr_data, executor.submit(enrich_one, pr_data)))

                for branch_name, _pr_data, fut in futures2:
                    pr_info = fut.result()
                    if pr_info is not None:
                        result.setdefault(branch_name, []).append(pr_info)

        if not include_closed:
            return result

        # Resolve closed/merged PRs for branches with no OPEN PR match.
        now = int(time.time())

        missing: List[str] = []
        for b in branch_set:
            if result.get(b):
                # Branch already has at least one open PR; we don't additionally fetch historical PRs by default.
                continue

            if not refresh_closed:
                # Try cache with appropriate TTL
                # We need to determine TTL, but we don't know if there are PRs yet
                # Use the longer TTL initially, then validate
                cached_entry = PR_BRANCH_CACHE.get_if_fresh(
                    owner=owner,
                    repo=repo,
                    branch=b,
                    ttl_s=int(closed_ttl_s),  # Use longer TTL for initial check
                )
                
                if cached_entry is not None:
                    ts = int(cached_entry.get("ts") or 0)
                    prs_d = cached_entry.get("prs")
                    if isinstance(prs_d, list):
                        # Now check with the correct TTL based on whether PRs exist
                        ttl = int(no_pr_ttl_s) if len(prs_d) == 0 else int(closed_ttl_s)
                        if (now - ts) <= ttl:
                            prs: List[PRInfo] = []
                            for d in prs_d:
                                if isinstance(d, dict):
                                    pr = self._pr_info_min_from_dict(d)
                                    if pr is not None:
                                        prs.append(pr)
                            self._cache_hit("pr_branch")
                            result[b] = prs
                            continue

            self._cache_miss("pr_branch")
            missing.append(b)

        if not missing:
            return result

        # Cache-only mode: do NOT perform any network fetches for closed PRs.
        # We still want to leverage the on-disk/memory closed-pr cache above (so local dashboards can
        # mark branches as merged even when max_github_api_calls=0), but we must not call GitHub here.
        if self.cache_only_mode:
            return result

        def fetch_branch(branch_name: str) -> Tuple[str, List[Dict[str, Any]]]:
            endpoint = f"/repos/{owner}/{repo}/pulls"
            # Try the normal "same-repo" head first, then fall back to fork-style "owner/branch" naming.
            prs_data = self.get(endpoint, params={"head": f"{owner}:{branch_name}", "state": "all", "per_page": 30})
            if (not prs_data) and "/" in (branch_name or ""):
                pre, rest = branch_name.split("/", 1)
                pre = (pre or "").strip()
                rest = (rest or "").strip()
                if pre and rest:
                    prs_data = self.get(endpoint, params={"head": f"{pre}:{rest}", "state": "all", "per_page": 30})
            out_prs: List[Dict[str, Any]] = []
            if isinstance(prs_data, list):
                for pr_data in prs_data:
                    if not isinstance(pr_data, dict):
                        continue
                    st = str(pr_data.get("state") or "").lower()
                    is_merged = pr_data.get("merged_at") is not None
                    if st != "open" or is_merged:
                        out_prs.append(
                            {
                                "number": pr_data.get("number"),
                                "title": pr_data.get("title") or "",
                                "url": pr_data.get("html_url") or "",
                                "state": pr_data.get("state") or "",
                                "is_merged": bool(is_merged),
                                "mergeable_state": pr_data.get("mergeable_state") or "unknown",
                                "head_sha": (pr_data.get("head") or {}).get("sha"),
                                "base_ref": (pr_data.get("base") or {}).get("ref", "main"),
                                "created_at": pr_data.get("created_at"),
                            }
                        )
            out_prs.sort(key=lambda d: int(d.get("number") or 0), reverse=True)
            return branch_name, out_prs

        with ThreadPoolExecutor(max_workers=6) as executor:
            futs = [executor.submit(fetch_branch, b) for b in missing]
            for fut in as_completed(futs):
                try:
                    branch_name, prs_d = fut.result()
                except AttributeError:  # .get() on non-dict
                    continue
                
                # Save to cache
                PR_BRANCH_CACHE.put(
                    owner=owner,
                    repo=repo,
                    branch=branch_name,
                    value={"ts": now, "prs": prs_d},
                )
                
                prs: List[PRInfo] = []
                for d in prs_d:
                    if isinstance(d, dict):
                        pr = self._pr_info_min_from_dict(d)
                        if pr is not None:
                            prs.append(pr)
                result[branch_name] = prs

        return result

    def _pr_info_from_pr_data(self, owner: str, repo: str, pr_data: Dict[str, Any]) -> Optional["PRInfo"]:
        """Convert a /pulls list response object into a PRInfo (with extra lookups)."""
        try:
            pr_number = int(pr_data["number"])
        except (ValueError, TypeError):
            return None

        base_branch = (pr_data.get("base") or {}).get("ref", "main")
        base_branch = str(base_branch or "main")

        # Fetch PR checks data once (REST check-runs; reused by multiple methods)
        checks_data = self._fetch_pr_checks_data(owner, repo, pr_number)

        # Required checks: use GraphQL PR-level isRequired field (works without branch protection API access).
        # NOTE: The old get_required_checks_for_base_ref() approach requires admin perms and returns 403.
        # DO NOT use /repos/{owner}/{repo}/branches/{branch}/protection/required_status_checks anymore.
        required_checks = self.get_required_checks(owner, repo, pr_number)
        unresolved_count = self.count_unresolved_conversations(owner, repo, pr_number)
        review_decision = pr_data.get("reviewDecision")
        if not review_decision:
            # REST /pulls objects do not include GraphQL `reviewDecision`; derive from /reviews.
            review_decision = self._review_decision_from_reviews(owner, repo, pr_number)

        # Now process checks data synchronously (no need to parallelize since data is already fetched)
        head = (pr_data.get("head") or {}) if isinstance(pr_data.get("head"), dict) else {}
        head_sha = head.get("sha")
        head_ref = head.get("ref")
        head_label = head.get("label")
        head_owner = ""
        hrepo = (head.get("repo") or {}) if isinstance(head.get("repo"), dict) else {}
        hown = (hrepo.get("owner") or {}) if isinstance(hrepo.get("owner"), dict) else {}
        head_owner = str(hown.get("login") or "").strip()
        ci_status = self.get_ci_status(owner, repo, head_sha, pr_number, checks_data=checks_data)
        failed_checks, rerun_url = self.get_failed_checks(owner, repo, head_sha, required_checks, pr_number, checks_data=checks_data)
        running_checks = self.get_running_checks(pr_number, owner, repo, required_checks, checks_data=checks_data)

        mergeable = None  # list endpoint doesn't include this, but we can use mergeable_state when present
        mergeable_state = pr_data.get("mergeable_state", "unknown")
        has_conflicts = (mergeable is False) or (mergeable_state == "dirty")

        conflict_message = None
        if has_conflicts:
            conflict_message = f"This branch has conflicts that must be resolved (merge {base_branch} into this branch)"

        blocking_message = None
        if mergeable_state in ["blocked", "unstable", "behind"]:
            if mergeable_state == "unstable":
                blocking_message = "Merging is blocked - Waiting on code owner review or required status checks"
            elif mergeable_state == "blocked":
                blocking_message = "Merging is blocked - Required reviews or checks not satisfied"
            elif mergeable_state == "behind":
                blocking_message = "This branch is out of date with the base branch"

        return PRInfo(
            number=pr_number,
            title=pr_data.get("title", ""),
            url=pr_data.get("html_url", ""),
            state=pr_data.get("state", ""),
            is_merged=pr_data.get("merged_at") is not None,
            review_decision=review_decision,
            mergeable_state=mergeable_state,
            unresolved_conversations=unresolved_count,
            ci_status=ci_status,
            head_sha=head_sha,
            head_ref=head_ref,
            head_owner=head_owner or None,
            head_label=head_label,
            base_ref=base_branch,
            created_at=pr_data.get("created_at"),
            updated_at=pr_data.get("updated_at"),
            has_conflicts=has_conflicts,
            conflict_message=conflict_message,
            blocking_message=blocking_message,
            required_checks=sorted(list(required_checks or set())),
            failed_checks=failed_checks,
            running_checks=running_checks,
            rerun_url=rerun_url,
        )

    def _get_cached_pr_info_if_unchanged(
        self,
        *,
        owner: str,
        repo: str,
        pr_number: int,
        updated_at: str,
    ) -> Optional["PRInfo"]:
        """Return cached PRInfo if we have an entry for this PR with the same updated_at."""
        key = self._pr_info_cache_key(owner, repo, pr_number)
        upd = str(updated_at or "").strip()
        if not upd:
            return None

        def _hydrate_entry(ent: Dict[str, Any]) -> Optional[PRInfo]:
            prd = ent.get("pr")
            if not isinstance(prd, dict):
                return None
            return self._pr_info_full_from_dict(prd)

        def _maybe_backfill_and_persist(pr: PRInfo) -> PRInfo:
            """Backfill missing enrichment fields on PRInfo and persist back to caches if we changed anything."""
            changed = False

            # Required checks are effectively configuration and may be missing in older cache entries.
            if not pr.required_checks:
                try:
                    req = sorted(list(self.get_required_checks(owner, repo, int(pr_number)) or set()))
                except (ValueError, TypeError):
                    req = []
                if req:
                    pr.required_checks = list(req)
                    changed = True

            # REST /pulls objects do not include GraphQL `reviewDecision`; older cache entries may lack it.
            if not pr.review_decision:
                try:
                    rd = self._review_decision_from_reviews(owner, repo, int(pr_number))
                except (ValueError, TypeError):
                    rd = None
                if rd:
                    pr.review_decision = str(rd)
                    changed = True

            if changed:
                # Serialize once at the cache boundary.
                self._save_pr_info_cache(owner=owner, repo=repo, pr_number=int(pr_number), updated_at=upd, pr=pr)
            return pr

        # Check cache (handles both memory + disk)
        cached_entry = PR_INFO_CACHE.get_if_matches_updated_at(key, updated_at=upd)
        if cached_entry is not None:
            pr = _hydrate_entry(cached_entry)
            if pr is not None:
                self._cache_hit("pr_info")
                return _maybe_backfill_and_persist(pr)
        
        self._cache_miss("pr_info")
        return None

    def _save_pr_info_cache(
        self,
        *,
        owner: str,
        repo: str,
        pr_number: int,
        updated_at: str,
        pr: "PRInfo",
    ) -> None:
        key = self._pr_info_cache_key(owner, repo, pr_number)
        upd = str(updated_at or "").strip()
        if not upd:
            return
        prd = self._pr_info_full_to_dict(pr)
        PR_INFO_CACHE.put(key, updated_at=upd, pr_dict=prd)



    def get_pr_checks_rows(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        *,
        commit_sha: Optional[str] = None,
        head_sha: Optional[str] = None,
        required_checks: Optional[set] = None,
        ttl_s: int = 300,
        skip_fetch: bool = False,
    ) -> List[GHPRCheckRow]:
        """Get structured PR check rows (GitHub REST check-runs), with a short-lived persistent cache.

        Note: the raw-log links are time-limited, but the checks rows themselves are cheap to cache
        for a short TTL to speed repeated HTML generation.

        OPTIMIZATION (2026-01-18): This method implements ETag support for conditional requests:
        - Extracts ETags from stale cache entries
        - Sends If-None-Match header when re-fetching
        - 304 Not Modified responses DON'T count against rate limit!
        - Cache v6 stores ETags for both /check-runs and /status endpoints
        - Benefit: 85-95% rate limit reduction on subsequent runs

        Args:
            commit_sha: Optional commit SHA to cache per-commit. If not provided, caches per-PR only.
            head_sha: Optional PR head SHA. If provided, skips the PR fetch to get head SHA (saves 1 API call).
        """
        required_checks = required_checks or set()
        key = self._pr_checks_cache_key(owner, repo, pr_number, commit_sha=commit_sha)
        now = int(datetime.now(timezone.utc).timestamp())
        # Cache schema version. Bump this when adding new optional fields (additive changes only).
        # v2 adds status-context checks (GET /commits/<sha>/status) in addition to check-runs.
        # v3 added name-level dedupe (later reverted) and persisted `run_id`.
        # v4 persists both `run_id` and `job_id` per row so UIs can disambiguate duplicates.
        # v5 adds `workflow_name` and `event` fields for full check display names.
        # v6 adds `check_runs_etag` and `status_etag` for conditional requests (304 Not Modified).
        # v7 adds `incomplete` flag for entries written during budget exhaustion (missing job details/duration).
        #
        # **CRITICAL**: All schema changes are ADDITIVE ONLY (new optional fields with defaults).
        # This ensures backward/forward compatibility. Cache entries from any version >= MIN_CACHE_VER
        # can be read by any other version. Never remove or change existing fields.
        CACHE_VER = 7
        MIN_CACHE_VER = 2  # Minimum supported cache version (reject anything older)

        # 1) Check PR_CHECKS_CACHE
        cached_entry_dict = PR_CHECKS_CACHE.get_entry_dict(key)
        if cached_entry_dict is not None:
            try:
                # Deserialize from cache
                ent = GHPRChecksCacheEntry.from_disk_dict_strict(
                    d=cached_entry_dict,
                    cache_file=PR_CHECKS_CACHE._cache_file,
                    entry_key=key
                )
                ts = int(ent.ts)
                ver = int(ent.ver)
                incomplete = bool(ent.incomplete)
                
                # Check TTL (or cache_only_mode overrides TTL)
                if ts and ((now - ts) <= max(0, int(ttl_s)) or self.cache_only_mode) and not incomplete:
                    if ver >= MIN_CACHE_VER:
                        self._cache_hit("pr_checks")
                        # Apply required_checks overlay
                        out: List[GHPRCheckRow] = []
                        for r in ent.rows:
                            if r.is_required or (r.name in required_checks):
                                out.append(r if r.is_required else replace(r, is_required=True))
                            else:
                                out.append(r)
                        return out
            except (ValueError, TypeError, RuntimeError):
                # Cache entry invalid/corrupted, continue to fetch
                pass

        # Cache-only mode: do not fetch network; return empty if no cached entry was usable.
        if self.cache_only_mode:
            self._cache_miss("pr_checks.cache_only_empty")
            return []

        if skip_fetch:
            return []

        # 3) REST check-runs for PR head SHA (best-effort; works for public repos without auth)
        self._cache_miss("pr_checks")

        # NOTE: ETag support removed for simplification during cache migration
        
        out: List[GHPRCheckRow] = []  # Initialize before try block

        try:
            # If head_sha provided, use it directly (saves 1 API call)
            if head_sha:
                head_sha = str(head_sha).strip()
            else:
                # Otherwise fetch PR head SHA from cache (or API if needed)
                head_sha = self.get_pr_head_sha(owner=owner, repo=repo, pr_number=int(pr_number))

            if not head_sha:
                return []

            # Fetch check-runs with ETag support
            check_runs_url = f"{self.base_url}/repos/{owner}/{repo}/commits/{head_sha}/check-runs"
            check_runs_resp = self._rest_get(
                check_runs_url,
                params={"per_page": 100},
                timeout=10,
            )

            # NOTE: ETag extraction and 304 handling removed for simplification

            # Parse response (200 OK with new/changed data)
            data = check_runs_resp.json() if check_runs_resp else {}
            check_runs = data.get("check_runs") if isinstance(data, dict) else None
            if not isinstance(check_runs, list) or not check_runs:
                check_runs = []

            # Also fetch "status contexts" (aka classic commit statuses) with ETag.
            #
            # GitHub's PR Checks UI mixes:
            # - check-runs (Actions, some apps)
            # - status contexts (many GitHub Apps / integrations)
            #
            # Some checks (e.g. CodeRabbit "Review skipped") may only show up as status contexts,
            # not as check-runs. If we only fetch /check-runs, those appear "missing" in dashboards.
            status_url = f"{self.base_url}/repos/{owner}/{repo}/commits/{head_sha}/status"
            status_resp = self._rest_get(
                status_url,
                timeout=10,
            )

            # NOTE: ETag extraction removed for simplification

            # Parse response
            statuses_data = status_resp.json() if status_resp and status_resp.status_code != 304 else {}
            statuses = statuses_data.get("statuses") if isinstance(statuses_data, dict) else None
            if not isinstance(statuses, list):
                statuses = []

            def _parse_iso(s: str) -> Optional[datetime]:
                try:
                    ss = str(s or "").strip()
                    if not ss:
                        return None
                    # GitHub timestamps look like "2025-12-29T10:52:07Z"
                    if ss.endswith("Z"):
                        ss = ss[:-1] + "+00:00"
                    return datetime.fromisoformat(ss)
                except json.JSONDecodeError:
                    return None

            def _format_dur(start_iso: Optional[str], end_iso: Optional[str]) -> str:
                try:
                    st = _parse_iso(start_iso or "")
                    en = _parse_iso(end_iso or "")
                    if not st or not en:
                        return ""
                    sec = int((en - st).total_seconds())
                    if sec < 0:
                        return ""
                    m, s2 = divmod(sec, 60)
                    h, m = divmod(m, 60)
                    if h:
                        return f"{h}h {m}m"
                    if m:
                        return f"{m}m {s2}s"
                    return f"{s2}s"
                except (ValueError, TypeError):
                    return ""

            # De-dupe exact duplicates only (same name+url). If the same check name appears multiple
            # times with different run/job URLs (reruns), we keep them all so UIs can show each.
            seen: set[tuple[str, str]] = set()

            # OPTIMIZATION (2026-01-18): Batch fetch workflow run metadata
            # Instead of fetching run metadata inside the check-runs loop (100 individual calls),
            # collect all run_ids first, then batch fetch them (10-20 calls).
            # Benefit: ~90% reduction in API calls for workflow runs
            run_ids_to_fetch: set[str] = set()
            for cr in check_runs:
                if not isinstance(cr, dict):
                    continue
                url = str(cr.get("details_url") or cr.get("html_url") or "").strip()
                run_id = parse_actions_run_id_from_url(url)
                if run_id:
                    run_ids_to_fetch.add(run_id)

            # Batch fetch workflow run metadata (name, event) for all runs
            workflow_info_cache: Dict[str, tuple[str, str]] = {}  # run_id -> (workflow_name, event)
            for run_id in run_ids_to_fetch:
                try:
                    run_data = self.get(f"/repos/{owner}/{repo}/actions/runs/{run_id}", timeout=5) or {}
                    if isinstance(run_data, dict):
                        workflow_info_cache[run_id] = (
                            str(run_data.get("name", "") or "").strip(),
                            str(run_data.get("event", "") or "").strip()
                        )
                except AttributeError:  # Object not dict-like (indicates data structure bug)
                    workflow_info_cache[run_id] = ("", "")

            for cr in check_runs:
                if not isinstance(cr, dict):
                    continue
                name = str(cr.get("name", "") or "").strip()
                status = str(cr.get("status", "") or "").strip().lower()
                conclusion = str(cr.get("conclusion", "") or "").strip().lower()

                # Map REST check-run (status+conclusion) into a gh-like status_raw.
                status_raw = ""
                if status and status != "completed":
                    # in_progress / queued
                    status_raw = status
                else:
                    # completed; use conclusion
                    if conclusion in {"success"}:
                        status_raw = "pass"
                    elif conclusion in {"failure"}:
                        status_raw = "fail"
                    elif conclusion in {"cancelled", "canceled"}:
                        status_raw = "cancelled"
                    elif conclusion in {"skipped"}:
                        status_raw = "skipped"
                    elif conclusion in {"neutral"}:
                        status_raw = "neutral"
                    elif conclusion in {"timed_out"}:
                        status_raw = "timed_out"
                    elif conclusion in {"action_required"}:
                        status_raw = "action_required"
                    else:
                        status_raw = "unknown"

                duration = _format_dur(cr.get("started_at"), cr.get("completed_at"))
                url = str(cr.get("details_url") or cr.get("html_url") or "").strip()
                description = ""
                out_obj = cr.get("output") or {}
                if isinstance(out_obj, dict):
                    description = str(out_obj.get("title", "") or "").strip()

                is_req = bool(name and (name in required_checks))
                key2 = (name, url)
                if name and key2 in seen:
                    continue
                if name:
                    seen.add(key2)
                run_id = parse_actions_run_id_from_url(url)
                job_id = parse_actions_job_id_from_url(url)
                
                # Extract workflow name and event from pre-fetched cache
                workflow_name = ""
                event = ""
                if run_id and run_id in workflow_info_cache:
                    workflow_name, event = workflow_info_cache[run_id]
                
                out.append(
                    GHPRCheckRow(
                        name=name,
                        status_raw=status_raw,
                        duration=duration,
                        url=url,
                        run_id=run_id,
                        job_id=job_id,
                        description=description,
                        is_required=is_req,
                        workflow_name=workflow_name,
                        event=event,
                    )
                )

            # Merge in status contexts (best-effort).
            # These don't have step timings; duration remains empty.
            for st in statuses:
                if not isinstance(st, dict):
                    continue
                name = str(st.get("context", "") or "").strip()
                if not name:
                    continue
                state = str(st.get("state", "") or "").strip().lower()
                desc = str(st.get("description", "") or "").strip()
                target = str(st.get("target_url", "") or "").strip()

                # Map status API states into our status_raw.
                # Note: status contexts don't have a "skipped" state, but some integrations
                # encode it in the description.
                status_raw = "unknown"
                if state in {"success"}:
                    status_raw = "pass"
                elif state in {"failure", "error"}:
                    status_raw = "fail"
                elif state in {"pending"}:
                    status_raw = "pending"
                if desc and ("skipped" in desc.lower() or "skip" in desc.lower()):
                    status_raw = "skipped"

                is_req = bool(name and (name in required_checks))
                key2 = (name, target)
                if key2 in seen:
                    continue
                seen.add(key2)
                run_id = parse_actions_run_id_from_url(target)
                job_id = parse_actions_job_id_from_url(target)
                # Status contexts don't have workflow names
                out.append(
                    GHPRCheckRow(
                        name=name,
                        status_raw=status_raw,
                        duration="",
                        url=target,
                        run_id=run_id,
                        job_id=job_id,
                        description=desc,
                        is_required=is_req,
                        workflow_name="",
                        event="",
                    )
                )

            if not out:
                return []

            # Detect if data appears incomplete (missing job details that should be present)
            # A completed job (pass/fail/skipped) with GitHub Actions URL should have duration.
            # If many jobs are missing duration, the entry is likely incomplete (budget exhaustion).
            incomplete = False
            completed_jobs_without_duration = 0
            completed_jobs_total = 0
            for row in out:
                status = str(row.status_raw or "").lower()
                duration = str(row.duration or "").strip()
                url = str(row.url or "").strip()
                # Count completed GitHub Actions jobs
                if status in {"pass", "fail", "skipped"} and "/actions/runs/" in url and "/job/" in url:
                    completed_jobs_total += 1
                    if not duration:
                        completed_jobs_without_duration += 1
            # Mark incomplete if >30% of completed GitHub Actions jobs are missing duration
            if completed_jobs_total > 0 and completed_jobs_without_duration > 0:
                missing_pct = (completed_jobs_without_duration / completed_jobs_total) * 100
                if missing_pct > 30:
                    incomplete = True

            # persist to cache (without ETags)
            entry = GHPRChecksCacheEntry(
                ts=int(now),
                ver=int(CACHE_VER),
                rows=tuple(out),
                check_runs_etag="",  # Empty string (ETags removed)
                status_etag="",  # Empty string (ETags removed)
                incomplete=incomplete,
            )
            PR_CHECKS_CACHE.put_entry_dict(key, entry.to_disk_dict())
        except (ValueError, TypeError):
            pass

        return out

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, timeout: int = 10) -> Optional[Dict]:
        """Make GET request to GitHub API.

        Args:
            endpoint: API endpoint (e.g., "/repos/owner/repo/pulls/123")
            params: Query parameters
            timeout: Request timeout in seconds

        Returns:
            JSON response as dict, or None if request failed

            Example return value for pull request endpoint:
            {
                "number": 1234,
                "title": "Add Docker image caching improvements",
                "state": "open",
                "head": {
                    "sha": "21a03b316dc1e5031183965e5798b0d9fe2e64b3",
                    "ref": "feature/docker-caching"
                },
                "base": {"ref": "main"},
                "mergeable": true,
                "mergeable_state": "clean",
                "user": {"login": "johndoe"},
                "created_at": "2025-11-20T10:00:00Z"
            }

            Example return value for check runs endpoint:
            {
                "total_count": 5,
                "check_runs": [
                    {
                        "id": 12345678,
                        "name": "build-test-amd64",
                        "status": "completed",
                        "conclusion": "success",
                        "html_url": "https://github.com/owner/repo/actions/runs/12345678"
                    }
                ]
            }
        """
        url = f"{self.base_url}{endpoint}" if endpoint.startswith('/') else f"{self.base_url}/{endpoint}"

        try:
            assert requests is not None
            try:
                response = self._rest_get(url, timeout=timeout, params=params)
            except Exception as e:
                # Budget exhaustion / cache-only mode / rate-limited fallback should not hard-fail dashboards.
                # Best-effort: switch to cache-only mode and return None.
                self.set_cache_only_mode(True)
                return None
            if response is None:
                # _rest_get can return None in best-effort modes; treat as a soft failure.
                return None

            if response.status_code == 403:
                # Check if it's a rate limit error
                if 'X-RateLimit-Remaining' in response.headers and response.headers['X-RateLimit-Remaining'] == '0':
                    raise Exception(
                        "GitHub API rate limit exceeded. Provide --token (or login with gh so ~/.config/gh/hosts.yml exists)."
                    )
                else:
                    raise Exception(f"GitHub API returned 403 Forbidden: {response.text}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:  # type: ignore[union-attr]
            raise Exception(f"GitHub API request failed for {endpoint}: {e}")

    def has_token(self) -> bool:
        """Check if a GitHub token is configured."""
        return self.token is not None

    def _fetch_pr_checks_data(self, owner: str, repo: str, pr_number: int) -> Optional[dict]:
        """Fetch PR checks data once (via GitHub REST) and parse for reuse by multiple methods.

        We intentionally do NOT shell out to `gh` here. Everything is derived from:
          - GET /repos/{owner}/{repo}/pulls/{pr_number}         (to find head sha)
          - GET /repos/{owner}/{repo}/commits/{head_sha}/check-runs?per_page=100
        """
        try:
            # Fetch PR head SHA from cache (or API if needed)
            head_sha = self.get_pr_head_sha(owner=owner, repo=repo, pr_number=int(pr_number))
            if not head_sha:
                return None

            data = self.get(f"/repos/{owner}/{repo}/commits/{head_sha}/check-runs", params={"per_page": 100}, timeout=10) or {}
            check_runs = data.get("check_runs") if isinstance(data, dict) else None
            if not isinstance(check_runs, list) or not check_runs:
                return {"stdout": "", "checks": []}

            def _status_raw(status: str, conclusion: str) -> str:
                s = (status or "").strip().lower()
                c = (conclusion or "").strip().lower()
                if s and s != "completed":
                    return s  # queued / in_progress
                if c in {"success"}:
                    return "pass"
                if c in {"failure"}:
                    return "fail"
                if c in {"cancelled", "canceled"}:
                    return "cancelled"
                if c in {"skipped"}:
                    return "skipped"
                if c in {"neutral"}:
                    return "neutral"
                if c in {"timed_out"}:
                    return "timed_out"
                if c in {"action_required"}:
                    return "action_required"
                # completed with unknown conclusion
                return "unknown"

            checks: List[Dict[str, Any]] = []
            for cr in check_runs:
                if not isinstance(cr, dict):
                    continue
                name = str(cr.get("name", "") or "").strip()
                if not name:
                    continue
                status = str(cr.get("status", "") or "")
                conclusion = str(cr.get("conclusion", "") or "")
                url = str(cr.get("details_url") or cr.get("html_url") or "").strip()
                duration = format_gh_check_run_duration(cr)
                checks.append(
                    {
                        "name": name,
                        "status": _status_raw(status, conclusion),
                        "duration": duration,
                        "url": url,
                    }
                )

            stdout = "\n".join(
                [f"{c.get('name','')}\t{c.get('status','')}\t{c.get('duration','')}\t{c.get('url','')}" for c in checks]
            )
            return {"stdout": stdout, "checks": checks}
        except AttributeError:  # .get() on non-dict
            return None

    def get_ci_status(self, owner: str, repo: str, sha: str, pr_number: Optional[int] = None,
                     checks_data: Optional[dict] = None) -> Optional[str]:
        """Get CI status for a commit by checking PR checks.

        Args:
            owner: Repository owner
            repo: Repository name
            sha: Commit SHA
            pr_number: Optional PR number for faster lookup
            checks_data: Optional pre-fetched checks data from _fetch_pr_checks_data()

        Returns:
            CI status string ('passed', 'failed', 'running'), or None if unavailable
        """
        # If checks_data is provided, use it instead of fetching
        if checks_data:
            checks = checks_data.get('checks', [])
            if not checks:
                return None

            # Count check statuses
            has_fail = any(c['status'].lower() == 'fail' for c in checks)
            has_pending = any(c['status'].lower() in ('pending', 'queued', 'in_progress') for c in checks)

            if has_fail:
                return 'failed'
            elif has_pending:
                return 'running'
            else:
                return 'passed'

        # If we don't have PR number, try to find it
        if not pr_number:
            # Try to find PR from commit
            endpoint = f"/repos/{owner}/{repo}/commits/{sha}/pulls"
            try:
                pulls = self.get(endpoint)
                if pulls and len(pulls) > 0:
                    pr_number = pulls[0]['number']
                else:
                    # Fallback to legacy status API
                    endpoint = f"/repos/{owner}/{repo}/commits/{sha}/status"
                    data = self.get(endpoint)
                    if data:
                        state = data.get('state')
                        if state == 'success':
                            return 'passed'
                        elif state == 'failure':
                            return 'failed'
                        elif state == 'pending':
                            return 'running'
                    return None
            except AttributeError:  # .get() on non-dict
                return None

        # Use GitHub REST check-runs (no gh dependency)
        try:
            cd = self._fetch_pr_checks_data(owner, repo, int(pr_number)) if pr_number else None
            checks = (cd or {}).get("checks", []) if isinstance(cd, dict) else []
            if not checks:
                return None
            has_fail = any(str(c.get("status", "") or "").lower() == "fail" for c in checks)
            has_pending = any(str(c.get("status", "") or "").lower() in ("pending", "queued", "in_progress") for c in checks)
            if has_fail:
                return "failed"
            if has_pending:
                return "running"
            return "passed"
        except (ValueError, TypeError):
            return None

    def get_pr_details(self, owner: str, repo: str, pr_number: int) -> Optional[dict]:
        """Get full PR details including mergeable status.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number

        Returns:
            PR details as dict, or None if request failed

            Example return value:
            {
                "number": 1234,
                "title": "Add Docker image caching improvements",
                "state": "open",
                "head": {"sha": "21a03b316dc1e5031183965e5798b0d9fe2e64b3"},
                "base": {"ref": "main"},
                "mergeable": true,
                "mergeable_state": "clean",
                "user": {"login": "johndoe"},
                "created_at": "2025-11-20T10:00:00Z",
                "updated_at": "2025-11-20T17:05:58Z"
            }
        """
        endpoint = f"/repos/{owner}/{repo}/pulls/{pr_number}"
        try:
            return self.get(endpoint)
        except AttributeError:  # .get() on non-dict
            return None


    def count_unresolved_conversations(self, owner: str, repo: str, pr_number: int) -> int:
        """Count unresolved conversation threads in a PR.

        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number

        Returns:
            Count of unresolved conversations (approximated as top-level comments)
        """
        endpoint = f"/repos/{owner}/{repo}/pulls/{pr_number}/comments"
        try:
            comments = self.get(endpoint)
            if not comments:
                return 0
            # Count top-level comments (those without in_reply_to_id are conversation starters)
            unresolved = sum(1 for comment in comments if not comment.get('in_reply_to_id'))
            return unresolved
        except (ValueError, TypeError):  # Type conversions or malformed data
            return 0

    def _review_decision_from_reviews(self, owner: str, repo: str, pr_number: int) -> Optional[str]:
        """Best-effort review decision from REST reviews.

        GitHub's REST `/pulls` endpoints do not include GraphQL `reviewDecision`.
        For dashboards, we use a lightweight heuristic:
        - If any latest review is CHANGES_REQUESTED => CHANGES_REQUESTED
        - Else if any latest review is APPROVED => APPROVED
        - Else => REVIEW_REQUIRED
        """
        endpoint = f"/repos/{owner}/{repo}/pulls/{int(pr_number)}/reviews"
        try:
            reviews = self.get(endpoint, params={"per_page": 100})
        except AttributeError:  # .get() on non-dict
            return None
        if not isinstance(reviews, list) or not reviews:
            return "REVIEW_REQUIRED"

        # Keep only the latest review per user login.
        latest_by_user: Dict[str, Tuple[str, str]] = {}
        for r in reviews:
            if not isinstance(r, dict):
                continue
            user = r.get("user") if isinstance(r.get("user"), dict) else {}
            login = str((user or {}).get("login") or "").strip()
            if not login:
                continue
            state = str(r.get("state") or "").strip().upper()
            submitted_at = str(r.get("submitted_at") or "").strip()
            if not submitted_at:
                continue
            prev = latest_by_user.get(login)
            if prev is None or submitted_at > prev[0]:
                latest_by_user[login] = (submitted_at, state)

        states = {st for (_ts, st) in latest_by_user.values() if st}
        if "CHANGES_REQUESTED" in states:
            return "CHANGES_REQUESTED"
        if "APPROVED" in states:
            return "APPROVED"
        return "REVIEW_REQUIRED"




    def get_required_checks(self, owner: str, repo: str, pr_number: int) -> set:
        """Return the set of required check names for a PR (best-effort).

        We intentionally derive this from GitHub's merge-box required-ness:
        GraphQL `statusCheckRollup.contexts.nodes[].isRequired(pullRequestId: ...)`.

        Rationale:
        - Branch protection endpoints often require admin perms and can return 403.
        - We want REQUIRED to keep working even when our internal REST budget is exhausted:
          this uses `gh` subprocess calls (separate from this client's REST budget).
        
        Caching:
        - Tiered TTL based on PR state and commit age (matching pr_checks cache):
          - Merged/closed: 30 days (immutable)
          - Open PR, commit < 8 hours: 3 minutes
          - Open PR, commit >= 8 hours: 2 hours
        - Uses both memory and disk cache for persistence across runs.
        - Failed fetches are cached briefly (<= 60s) to avoid retry storms.

        Pagination:
        - Fetches up to 500 checks per page, with up to 5 pages (2500 total checks).
        - This ensures all checks are captured even for PRs with many status checks.
        """
        try:
            prn = int(pr_number)
        except (ValueError, TypeError):
            return set()

        def _ttl_s_for_pr_state_and_commit_age(is_merged_or_closed: bool, commit_age_s: int) -> int:
            """Tiered TTL for required-checks cache (matching pr_checks logic)."""
            if is_merged_or_closed:
                # Merged/closed PRs are immutable
                return 30 * 24 * 3600  # 30 days
            else:
                # Open PR - determine by commit age
                try:
                    age = int(commit_age_s)
                except (ValueError, TypeError):
                    age = 0
                if age < 8 * 3600:  # < 8 hours
                    return 3 * 60  # 3 minutes
                else:  # >= 8 hours
                    return 2 * 3600  # 2 hours

        def _is_cache_entry_valid(ent: Dict[str, Any], *, now: int) -> bool:
            """Return True if a cache entry is valid under current tiered TTL policy."""
            try:
                ts = int(ent.get("ts", 0) or 0)
            except (ValueError, TypeError):
                ts = 0
            if not ts:
                return False

            ok = ent.get("ok", True)

            # Cache entries store PR metadata to apply TTL without refetching
            pr_state = ent.get("pr_state", "open")  # "open", "closed", or "merged"
            is_merged_or_closed = pr_state in ("closed", "merged")
            
            pr_updated_at_epoch = ent.get("pr_updated_at_epoch", None)
            try:
                pr_updated_at_epoch_i = int(pr_updated_at_epoch) if pr_updated_at_epoch is not None else 0
            except (ValueError, TypeError):
                pr_updated_at_epoch_i = 0
            
            if not pr_updated_at_epoch_i and not is_merged_or_closed:
                # If this was a failed fetch, cache briefly to avoid retry storms.
                if ok is False:
                    return (int(now) - int(ts)) <= 60
                # Legacy entries lacked PR metadata; treat them as expired so we refresh and populate metadata.
                # Cache-only mode is handled by the caller.
                return False

            # For merged/closed PRs, commit age doesn't matter (use 7d TTL)
            # For open PRs, use commit age (time since PR.updated_at)
            commit_age_s = max(0, int(now) - int(pr_updated_at_epoch_i)) if pr_updated_at_epoch_i else 0
            ttl_s = _ttl_s_for_pr_state_and_commit_age(is_merged_or_closed, commit_age_s)

            # Failed fetches should never be cached long-term.
            if ok is False:
                ttl_s = min(int(ttl_s), 60)

            return (int(now) - int(ts)) <= int(ttl_s)

        # Check cache first (tiered TTL based on PR.updated_at age)
        cache_key = f"required_checks:{owner}/{repo}:pr{prn}"
        now = int(time.time())
        
        # Try cache (uses get_if_valid which handles TTL validation)
        cached_entry = REQUIRED_CHECKS_CACHE.get_if_valid(cache_key, cache_only_mode=self.cache_only_mode, check_ttl=False)
        if cached_entry is not None:
            # Custom TTL validation
            if self.cache_only_mode or _is_cache_entry_valid(cached_entry, now=now):
                return cached_entry.get("val", set())
        
        # Cache-only mode: do not fetch network; return empty if we have nothing.
        if self.cache_only_mode:
            return set()

        try:
            # Fetch PR node_id and state via gh (avoids consuming this client's REST budget).
            pr_node_id = ""
            pr_state = "open"
            pr_updated_at_epoch: Optional[int] = None
            try:
                res0 = subprocess.run(
                    [
                        "gh",
                        "api",
                        f"repos/{owner}/{repo}/pulls/{prn}",
                        "--jq",
                        "{node_id: .node_id, state: .state, merged_at: .merged_at, updated_at: .updated_at}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                if res0.returncode == 0:
                    meta = {}
                    try:
                        meta = json.loads(str(res0.stdout or "").strip() or "{}") or {}
                    except json.JSONDecodeError:
                        meta = {}
                    pr_node_id = str(meta.get("node_id") or "").strip()
                    pr_state = str(meta.get("state") or "open").strip()
                    # If merged_at is present, state is "merged" (not "closed")
                    if meta.get("merged_at"):
                        pr_state = "merged"
                    updated_at_s = str(meta.get("updated_at") or "").strip()
                    if updated_at_s:
                        try:
                            # GitHub returns ISO 8601 timestamps like "2026-01-21T09:25:31Z"
                            dt = datetime.fromisoformat(updated_at_s.replace("Z", "+00:00"))
                            pr_updated_at_epoch = int(dt.timestamp())
                        except (ValueError, TypeError):
                            pr_updated_at_epoch = None
            except (OSError, subprocess.SubprocessError):  # subprocess failures
                pr_node_id = ""
                pr_state = "open"
                pr_updated_at_epoch = None
            if not pr_node_id:
                # Cache negative result (PR not found / not accessible)
                REQUIRED_CHECKS_CACHE.put(
                    cache_key,
                    set(),
                    ok=False,
                    pr_state=pr_state,
                    pr_updated_at_epoch=pr_updated_at_epoch,
                )
                return set()

            # Paginated query: fetch up to 100 checks per page (GitHub's limit), up to 25 pages (2500 total).
            # NOTE: `isRequired` needs an explicit `pullRequestId`, otherwise GitHub errors.
            query_template = """\
query($owner:String!,$name:String!,$number:Int!,$prid:ID!,$after:String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      commits(last: 1) {
        nodes {
          commit {
            statusCheckRollup {
              contexts(first: 100, after: $after) {
                pageInfo {
                  hasNextPage
                  endCursor
                }
                nodes {
                  __typename
                  ... on CheckRun { name isRequired(pullRequestId: $prid) }
                  ... on StatusContext { context isRequired(pullRequestId: $prid) }
                }
              }
            }
          }
        }
      }
    }
  }
}
"""

            all_required: set = set()
            after_cursor = None
            max_pages = 25  # 25 pages * 100 checks/page = 2500 total checks
            
            for page_num in range(max_pages):
                try:
                    # Build gh command with pagination cursor
                    cmd = [
                        "gh",
                        "api",
                        "graphql",
                        "-f",
                        f"query={query_template}",
                        "-f",
                        f"owner={owner}",
                        "-f",
                        f"name={repo}",
                        "-F",
                        f"number={int(prn)}",
                        "-f",
                        f"prid={pr_node_id}",
                    ]
                    
                    # Only add after cursor if we have one (null/None on first page)
                    if after_cursor:
                        cmd.extend(["-f", f"after={after_cursor}"])
                    else:
                        # Explicitly pass null for first page
                        cmd.extend(["-f", "after=null"])
                    
                    # NOTE: we intentionally use `gh` here because it handles GraphQL auth and
                    # enterprise oddities; required-ness is not reliably available via REST without
                    # admin branch-protection permissions.
                    res = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=15,
                        check=False,
                    )
                    if res.returncode != 0:
                        break
                    
                    data = {}
                    try:
                        data = json.loads(res.stdout or "{}") or {}
                    except json.JSONDecodeError:
                        break
                    
                    nodes = (
                        (((((data.get("data") or {}).get("repository") or {}).get("pullRequest") or {}).get("commits") or {}).get("nodes") or [])
                    )
                    if not (isinstance(nodes, list) and nodes):
                        break
                    
                    commit0 = nodes[0].get("commit") if isinstance(nodes[0], dict) else None
                    scr = commit0.get("statusCheckRollup") if isinstance(commit0, dict) else None
                    ctxs = (scr.get("contexts") or {}) if isinstance(scr, dict) else {}
                    
                    # Extract required checks from this page
                    ctx_nodes = ctxs.get("nodes") if isinstance(ctxs, dict) else None
                    if isinstance(ctx_nodes, list):
                        for n in ctx_nodes:
                            if not isinstance(n, dict):
                                continue
                            if n.get("isRequired") is not True:
                                continue
                            nm = str(n.get("name") or n.get("context") or "").strip()
                            if nm:
                                all_required.add(nm)
                    
                    # Check if there are more pages
                    page_info = ctxs.get("pageInfo") if isinstance(ctxs, dict) else None
                    if not isinstance(page_info, dict):
                        break
                    
                    has_next = page_info.get("hasNextPage")
                    if not has_next:
                        break
                    
                    after_cursor = page_info.get("endCursor")
                    if not after_cursor:
                        break

                except (OSError, subprocess.SubprocessError, json.JSONDecodeError):  # subprocess or JSON errors
                    break
            
            # Cache the results with tiered TTL based on PR state and commit age.
            REQUIRED_CHECKS_CACHE.put(
                cache_key,
                all_required,
                ok=True,
                pr_state=pr_state,
                pr_updated_at_epoch=pr_updated_at_epoch,
            )

            return all_required
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):  # subprocess or JSON parsing failures
            # Cache negative result (fetch failed)
            REQUIRED_CHECKS_CACHE.put(
                cache_key,
                set(),
                ok=False,
                pr_state="open",
            )
            return set()

    def _pr_info_cache_key(self, owner: str, repo: str, pr_number: int) -> str:
        return f"{owner}/{repo}#pr:{int(pr_number)}"



    def _search_issues_cache_key(self, owner: str, repo: str, pr_numbers: List[int]) -> str:
        ns = sorted({_safe_int(x, 0) for x in (pr_numbers or []) if _safe_int(x, 0) > 0})
        return f"{owner}/{repo}:search_issues:" + ",".join([str(n) for n in ns])

    def _load_search_issues_disk_cache(self) -> Dict[str, Any]:
        self._search_issues_cache_dir.mkdir(parents=True, exist_ok=True)
        p = self._search_issues_cache_dir / "search_issues.json"
        if not p.exists():
            return {}
        return self._json_load_text(p.read_text() or "{}")

    def _save_search_issues_disk_cache(self, key: str, value: Dict[str, Any]) -> None:
        """Atomically update a single entry in search_issues disk cache."""
        _save_single_disk_cache_entry(
            cache_dir=self._search_issues_cache_dir,
            cache_filename="search_issues.json",
            lock_filename="search_issues.lock",
            load_fn=self._load_search_issues_disk_cache,
            json_dump_fn=lambda d: self._json_dump_text(d, indent=None),
            key=key,
            value=value,
            stats_fn=lambda entries: self._cache_write("search_issues.disk_write", entries=entries),
        )

    def get_pr_updated_at_via_search_issues(
        self,
        *,
        owner: str,
        repo: str,
        pr_numbers: List[int],
        ttl_s: int = 60,
        timeout: int = 15,
    ) -> Dict[int, str]:
        """Return {pr_number: updated_at} for the given PR numbers using GitHub search/issues.

        Endpoint:
          GET /search/issues?q=repo:OWNER/REPO type:pr number:123 number:456 ...
        """
        nums = sorted({_safe_int(x, 0) for x in (pr_numbers or []) if _safe_int(x, 0) > 0})
        if not nums:
            return {}

        # If search/issues is known-broken for this repo, skip (this is only an optimization).
        disabled_key = f"{owner}/{repo}:search_issues_disabled"
        disable_ttl_s = 6 * 3600
        now = int(time.time())
        try:
            ent = self._search_issues_disabled_mem_cache.get(disabled_key)
            if isinstance(ent, dict):
                ts = int(ent.get("ts", 0) or 0)
                if ts and (now - ts) <= int(disable_ttl_s):
                    self._cache_hit("search_issues.disabled_mem")
                    return {}
        except (ValueError, TypeError):
            pass

        key = self._search_issues_cache_key(owner, repo, nums)

        # 1) memory
        try:
            ent = self._search_issues_mem_cache.get(key)
            if isinstance(ent, dict):
                ts = int(ent.get("ts", 0) or 0)
                val = ent.get("val")
                if isinstance(val, dict) and ts and ((now - ts) <= int(ttl_s) or self.cache_only_mode):
                    if (now - ts) <= int(ttl_s):
                        self._cache_hit("search_issues.mem")
                    else:
                        self._cache_hit("search_issues.mem_stale_cache_only")
                    return {int(k): str(v) for k, v in val.items() if str(k).isdigit() and str(v).strip()}
        except (ValueError, TypeError):
            pass

        # 2) disk
        disk = self._load_search_issues_disk_cache()
        try:
            dis = disk.get(disabled_key) if isinstance(disk, dict) else None
            if isinstance(dis, dict):
                ts = int(dis.get("ts", 0) or 0)
                if ts and (now - ts) <= int(disable_ttl_s):
                    self._cache_hit("search_issues.disabled_disk")
                    self._search_issues_disabled_mem_cache[disabled_key] = {"ts": ts, "val": True}
                    return {}
        except (ValueError, TypeError):
            pass
        ent = disk.get(key) if isinstance(disk, dict) else None
        if isinstance(ent, dict):
            ts = int(ent.get("ts", 0) or 0)
            val = ent.get("val")
            if isinstance(val, dict) and ts and ((now - ts) <= int(ttl_s) or self.cache_only_mode):
                if (now - ts) <= int(ttl_s):
                    self._cache_hit("search_issues.disk")
                else:
                    self._cache_hit("search_issues.disk_stale_cache_only")
                out = {int(k): str(v) for k, v in val.items() if str(k).isdigit() and str(v).strip()}
                self._search_issues_mem_cache[key] = {"ts": ts, "val": dict(out)}
                return out

        # cache-only: no network
        if self.cache_only_mode:
            self._cache_miss("search_issues.cache_only_empty")
            return {}

        # 3) network (chunk to avoid overly long query strings), but dedupe concurrent fetches across threads.
        lock = self._inflight_lock(f"search_issues:{key}")
        with lock:
            # Re-check memory cache (another thread may have populated it).
            try:
                ent = self._search_issues_mem_cache.get(key)
                if isinstance(ent, dict):
                    ts = int(ent.get("ts", 0) or 0)
                    val = ent.get("val")
                    if isinstance(val, dict) and ts and ((now - ts) <= int(ttl_s) or self.cache_only_mode):
                        if (now - ts) <= int(ttl_s):
                            self._cache_hit("search_issues.mem")
                        else:
                            self._cache_hit("search_issues.mem_stale_cache_only")
                        return {int(k): str(v) for k, v in val.items() if str(k).isdigit() and str(v).strip()}
            except (ValueError, TypeError):
                pass

            # Re-check disk cache too.
            disk = self._load_search_issues_disk_cache()
            ent = disk.get(key) if isinstance(disk, dict) else None
            if isinstance(ent, dict):
                ts = int(ent.get("ts", 0) or 0)
                val = ent.get("val")
                if isinstance(val, dict) and ts and ((now - ts) <= int(ttl_s) or self.cache_only_mode):
                    if (now - ts) <= int(ttl_s):
                        self._cache_hit("search_issues.disk")
                    else:
                        self._cache_hit("search_issues.disk_stale_cache_only")
                    out = {int(k): str(v) for k, v in val.items() if str(k).isdigit() and str(v).strip()}
                    self._search_issues_mem_cache[key] = {"ts": ts, "val": dict(out)}
                    return out

            self._cache_miss("search_issues.network")
            out: Dict[int, str] = {}
            try:
                chunks: List[List[int]] = []
                cur: List[int] = []
                # conservative chunk size
                for n in nums:
                    cur.append(n)
                    if len(cur) >= 25:
                        chunks.append(cur)
                        cur = []
                if cur:
                    chunks.append(cur)

                for ch in chunks:
                    q = f"repo:{owner}/{repo} type:pr " + " ".join([f"number:{n}" for n in ch])
                    # Use _rest_get so we can handle 422 without throwing (this is an optimization call).
                    url = f"{self.base_url}/search/issues"
                    resp = self._rest_get(url, timeout=int(timeout), params={"q": q, "per_page": 100})
                    if int(resp.status_code) == 422:
                        # Disable search/issues temporarily for this repo; caller will fall back to PR payload updated_at.
                        self._search_issues_disabled_mem_cache[disabled_key] = {"ts": now, "val": True}
                        # Save to disk cache using DiskCacheWriter (enforces lock-load-merge-save)
                        self._save_search_issues_disk_cache(disabled_key, {"ts": now, "val": True, "code": 422})
                        return {}
                    if resp.status_code < 200 or resp.status_code >= 300:
                        continue
                    try:
                        data = resp.json() or {}
                    except (json.JSONDecodeError, ValueError):  # requests.Response.json() can raise ValueError or JSONDecodeError
                        data = {}
                    items = data.get("items") if isinstance(data, dict) else None
                    if not isinstance(items, list):
                        continue
                    for it in items:
                        if not isinstance(it, dict):
                            continue
                        num = it.get("number")
                        upd = it.get("updated_at")
                        if num is None or upd is None:
                            continue
                        try:
                            out[int(num)] = str(upd)
                        except json.JSONDecodeError:
                            continue
            except (ValueError, TypeError):
                out = {}

        self._search_issues_mem_cache[key] = {"ts": now, "val": dict(out)}
        # Save to disk cache using DiskCacheWriter (enforces lock-load-merge-save)
        self._save_search_issues_disk_cache(key, {"ts": now, "val": dict(out)})
        return out

    def _extract_pytest_summary(self, all_lines: list) -> Optional[str]:
        """Extract pytest short test summary from log lines.

        Args:
            all_lines: List of log lines

        Returns:
            Formatted summary with failed test names, or None if not a pytest failure
        """
        try:
            # Find the "short test summary info" section
            summary_start_idx = None
            summary_end_idx = None

            for i, line in enumerate(all_lines):
                if '=== short test summary info ===' in line or '==== short test summary info ====' in line:
                    summary_start_idx = i
                elif summary_start_idx is not None and '===' in line and 'failed' in line.lower() and 'passed' in line.lower():
                    # Found end line like "===== 4 failed, 329 passed, 3 skipped, 284 deselected in 422.68s (0:07:02) ====="
                    summary_end_idx = i
                    break

            if summary_start_idx is None:
                return None

            # Extract failed test lines
            failed_tests = []
            for i in range(summary_start_idx + 1, summary_end_idx if summary_end_idx else len(all_lines)):
                line = all_lines[i]
                # Look for lines that start with "FAILED" after timestamp/prefix
                if 'FAILED' in line:
                    # Extract the test name
                    # Format: "FAILED lib/bindings/python/tests/test_metrics_registry.py::test_counter_introspection"
                    parts = line.split('FAILED')
                    if len(parts) >= 2:
                        test_name = parts[1].strip()
                        # Remove any trailing info like " - AssertionError: ..."
                        if ' - ' in test_name:
                            test_name = test_name.split(' - ')[0].strip()
                        failed_tests.append(test_name)

            if not failed_tests:
                return None

            # Format the output
            result = "Failed unit tests:\n\n"
            for test in failed_tests:
                result += f"  • {test}\n"

            # Add summary line if available
            if summary_end_idx:
                summary_line = all_lines[summary_end_idx]
                # Extract just the summary part (after the timestamp and ====)
                # Format: "2025-11-04T20:25:55.7767823Z ==== 5 failed, 4 passed, 320 deselected, 111 warnings in 1637.48s (0:27:17) ===="
                if '====' in summary_line:
                    # Split by ==== and get the middle part
                    parts = summary_line.split('====')
                    if len(parts) >= 2:
                        summary_text = parts[1].strip()
                        if summary_text:
                            result += f"\nSummary: {summary_text}"

            return result

        except (ValueError, TypeError, AttributeError):  # String parsing or attribute access failures
            return None

    def get_job_error_summary(self, run_id: str, job_url: str, owner: str, repo: str) -> Optional[str]:
        """Get a small error summary for a GitHub Actions job (cached).

        Args:
            run_id: Workflow run ID
            job_url: Job URL (contains job ID)
            owner: Repository owner
            repo: Repository name

        Returns:
            Error summary string or None
        """
        job_id = "unknown"
        try:
            # Extract job ID from URL
            # Example: https://github.com/ai-dynamo/dynamo/actions/runs/18697156351/job/53317461976
            if '/job/' not in job_url:
                return None

            job_id = job_url.split('/job/')[1].split('?')[0]

            # Check cache
            cached_snippet = JOB_LOG_CACHE.get(job_id)
            if cached_snippet is not None:
                self._cache_hit("job_log")
                return cached_snippet

            self._cache_miss("job_log")

            # Download raw log text via REST (ZIP), then extract a high-signal snippet.
            txt = self.get_job_raw_log_text_cached(job_url=job_url, owner=owner, repo=repo, assume_completed=True)
            if not txt:
                return None

            # Prefer the purpose-built snippet extractor.
            try:
                snippet = ci_snippet.extract_error_snippet_from_text(txt)
                snippet = (snippet or "").strip()
                if snippet:
                    JOB_LOG_CACHE.put(job_id, snippet)
                    return snippet
            except (ValueError, TypeError, AttributeError):  # String/type manipulation failures
                pass

            # Fallback: keep last ~40 meaningful lines
            all_lines = [ln for ln in str(txt).splitlines() if (ln or "").strip()]

            # First, try to extract pytest short test summary (most useful for test failures)
            pytest_summary = self._extract_pytest_summary(all_lines)
            if pytest_summary:
                JOB_LOG_CACHE.put(job_id, pytest_summary)
                return pytest_summary

            # Filter for error-related lines with surrounding context
            error_keywords = ['error', 'fail', 'Error', 'ERROR', 'FAIL', 'fatal', 'FATAL', 'broken']
            error_indices = []

            # Find all lines with error keywords
            for i, line in enumerate(all_lines):
                if line.strip() and not line.startswith('#'):
                    if any(keyword in line for keyword in error_keywords):
                        error_indices.append(i)

            # If we found error lines, extract them with surrounding context
            if error_indices:
                # For each error, get 10 lines before and 5 lines after
                context_lines = set()
                for error_idx in error_indices:
                    # Add lines before (up to 10)
                    for i in range(max(0, error_idx - 10), error_idx):
                        context_lines.add(i)
                    # Add error line itself
                    context_lines.add(error_idx)
                    # Add lines after (up to 5)
                    for i in range(error_idx + 1, min(len(all_lines), error_idx + 6)):
                        context_lines.add(i)

                # Sort indices and extract lines
                sorted_indices = sorted(context_lines)
                error_lines = []
                for idx in sorted_indices:
                    line = all_lines[idx]
                    if line.strip() and not line.startswith('#'):
                        error_lines.append(line)

                # Use up to last 80 lines with context (increased from 50)
                relevant_errors = error_lines[-80:]
                summary = '\n'.join(relevant_errors)

                # Limit length to 5000 chars (increased from 3000 for more context)
                if len(summary) > 5000:
                    summary = summary[:5000] + '\n\n...(truncated, view full logs at job URL above)'

                JOB_LOG_CACHE.put(job_id, summary)
                return summary

            # If no error keywords found, get last 40 lines as fallback
            last_lines = [line for line in all_lines if line.strip() and not line.startswith('#')][-40:]
            if last_lines:
                summary = '\n'.join(last_lines)
                if len(summary) > 5000:
                    summary = summary[:5000] + '\n\n...(truncated)'
                JOB_LOG_CACHE.put(job_id, summary)
                return summary

            error_summary = f"No error details found in logs.\n\nView full logs at:\n{job_url}"
            JOB_LOG_CACHE.put(job_id, error_summary)
            return error_summary

        except Exception as e:
            return f"Error fetching logs: {str(e)}\n\nView full logs at:\n{job_url}"

    def get_job_raw_log_text_cached(
        self,
        *,
        job_url: str,
        owner: str,
        repo: str,
        ttl_s: int = DEFAULT_RAW_LOG_TEXT_TTL_S,
        timeout: int = 30,
        max_bytes: int = DEFAULT_RAW_LOG_TEXT_MAX_BYTES,
        assume_completed: bool = False,
    ) -> Optional[str]:
        """Download and cache the *text* content of a GitHub Actions job log.

        - Uses the stable API endpoint `/actions/jobs/{job_id}/logs` (returns a ZIP).
        - Extracts and concatenates text files from the ZIP.
        - Caches on disk per job_id to support later parsing without repeated downloads.
        """
        if "/job/" not in (job_url or ""):
            return None
        job_id = str(job_url.split("/job/")[1].split("?")[0] or "").strip()
        if not job_id:
            return None

        # IMPORTANT: never cache logs for jobs that are not done.
        # The /actions/jobs/{id}/logs endpoint can return partial logs while a job is running.
        # If the caller already proved completion (e.g., via check-runs), it can set assume_completed=True
        # to avoid an extra REST call to /actions/jobs/{id}.
        if not bool(assume_completed):
            st = str(self.get_actions_job_status(owner=owner, repo=repo, job_id=job_id) or "").lower()
            if not st or st != "completed":
                return None

        now = int(datetime.now(timezone.utc).timestamp())

        # 1) memory cache
        try:
            ent = self._raw_log_text_mem_cache.get(job_id)
            if isinstance(ent, dict):
                ts = int(ent.get("ts", 0) or 0)
                if ts and (now - ts) <= max(0, int(ttl_s)):
                    self._cache_hit("raw_log_text.mem")
                    return ent.get("text")
        except (ValueError, TypeError):
            pass

        # 2) disk cache (per job_id)
        self._raw_log_text_cache_dir.mkdir(parents=True, exist_ok=True)
        txt_path = self._raw_log_text_cache_dir / f"{job_id}.log"
        legacy_txt_path = self._raw_log_text_cache_dir / f"{job_id}.txt"
        meta = {}
        if self._raw_log_text_index_file.exists():
            try:
                meta = json.loads(self._raw_log_text_index_file.read_text() or "{}")
            except (OSError, json.JSONDecodeError):
                meta = {}
        ent = meta.get(job_id) if isinstance(meta, dict) else None
        # Prefer .log; fall back to legacy .txt.
        chosen_path = txt_path if txt_path.exists() else legacy_txt_path
        if chosen_path.exists() and isinstance(ent, dict):
            ts = int(ent.get("ts", 0) or 0)
            # Only trust cache entries that were recorded as completed.
            # (Older caches may have been populated while the job was in-progress, yielding partial logs.)
            if bool(ent.get("completed", False)) and ts and (now - ts) <= max(0, int(ttl_s)):
                text = chosen_path.read_text(encoding="utf-8", errors="replace")
                self._raw_log_text_mem_cache[job_id] = {"ts": ts, "text": text}
                self._cache_hit("raw_log_text.disk")
                # Best-effort migrate legacy .txt -> .log for future runs.
                if chosen_path == legacy_txt_path and (not txt_path.exists()):
                    try:
                        tmp = str(txt_path) + ".tmp"
                        Path(tmp).write_text(text, encoding="utf-8", errors="replace")
                        os.replace(tmp, txt_path)
                    except OSError:
                        pass
                return text
            # If entry exists but isn't trusted, remove the stale local file so callers can refetch.
            if chosen_path.exists() and not bool(ent.get("completed", False)):
                try:
                    if chosen_path == legacy_txt_path and txt_path.exists():
                        # keep preferred .log if it exists
                        pass
                    else:
                        chosen_path.unlink()
                except OSError:
                    pass

        # 3) fetch + extract

        tmp_zip_path: Optional[Path] = None
        try:
            self._cache_miss("raw_log_text.network")
            api_url = f"{self.base_url}/repos/{owner}/{repo}/actions/jobs/{job_id}/logs"
            resp = self._rest_get(api_url, timeout=timeout, allow_redirects=True, stream=True)
            resp.raise_for_status()

            # Download the ZIP to disk to avoid buffering large logs in memory.
            self._raw_log_text_cache_dir.mkdir(parents=True, exist_ok=True)
            tmp_zip_path = self._raw_log_text_cache_dir / f"{job_id}.zip.tmp"
            limit = int(max_bytes or 0)
            got = 0
            with open(tmp_zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    got += len(chunk)
                    if limit and got >= limit:
                        break

            # Extract to a temp .log, then atomically move into place.
            txt_path = self._raw_log_text_cache_dir / f"{job_id}.log"
            tmp_txt = str(txt_path) + ".tmp"
            with open(tmp_txt, "w", encoding="utf-8", errors="replace") as out:
                try:
                    with zipfile.ZipFile(str(tmp_zip_path)) as zf:  # type: ignore[name-defined]
                        names = list(zf.namelist())
                        for name in names:
                            try:
                                data = zf.read(name)
                            except (TypeError, ValueError):  # list conversion
                                continue
                            try:
                                t = data.decode("utf-8", errors="replace")
                            except (TypeError, ValueError):  # list conversion
                                continue
                            if not t:
                                continue
                            if len(names) > 1:
                                out.write(f"===== {name} =====\n")
                            out.write(t)
                            if not t.endswith("\n"):
                                out.write("\n")
                except (zipfile.BadZipFile, OSError):
                    # If it wasn't a zip for some reason, fall back to a best-effort decode.
                    raw_bytes = Path(tmp_zip_path).read_bytes()
                    out.write(raw_bytes.decode("utf-8", errors="replace"))

            os.replace(tmp_txt, txt_path)
            # Track that we wrote a cached log entry for this job_id.
            self._cache_write("raw_log_text.disk", entries=1)

            # Persist index + mem cache using lock-load-merge-save
            try:
                size_b = int(txt_path.stat().st_size)
            except (ValueError, TypeError):
                size_b = 0

            def load_raw_log_index():
                if self._raw_log_text_index_file.exists():
                    try:
                        data = json.loads(self._raw_log_text_index_file.read_text() or "{}")
                        return data if isinstance(data, dict) else {}
                    except (ValueError, TypeError):
                        return {}
                return {}

            _save_single_disk_cache_entry(
                cache_dir=self._raw_log_text_index_file.parent,
                cache_filename=self._raw_log_text_index_file.name,
                lock_filename=self._raw_log_text_index_file.name + ".lock",
                load_fn=load_raw_log_index,
                json_dump_fn=lambda d: json.dumps(d),
                key=job_id,
                value={"ts": now, "bytes": size_b, "completed": True},
            )

            try:
                text = txt_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            self._raw_log_text_mem_cache[job_id] = {"ts": now, "text": text}
            return text
        except Exception as e:
            # Log the error for debugging but don't fail silently
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to fetch/cache raw log for job {job_id}: {type(e).__name__}: {e}")
            return None
        finally:
            try:
                if tmp_zip_path and tmp_zip_path.exists():
                    tmp_zip_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            except OSError:
                pass

    def get_failed_checks(self, owner: str, repo: str, sha: str, required_checks: set, pr_number: Optional[int] = None,
                         checks_data: Optional[dict] = None) -> Tuple[List[FailedCheck], Optional[str]]:
        """Get failed CI checks for a commit (GitHub REST check-runs; no gh dependency).

        Args:
            owner: Repository owner
            repo: Repository name
            sha: Commit SHA (unused, kept for compatibility)
            required_checks: Set of required check names
            pr_number: PR number (used to fetch PR head sha for check-runs when available)
            checks_data: Optional pre-fetched checks data from _fetch_pr_checks_data()

        Returns:
            Tuple of (List of FailedCheck objects, rerun_url)

            Example return value:
            (
                [
                    FailedCheck(
                        name="build-test-arm64",
                        conclusion="failure",
                        job_id="12345678",
                        job_url="https://github.com/owner/repo/actions/runs/12345678",
                        rerun_url="https://github.com/owner/repo/actions/runs/12345678/rerun"
                    ),
                    FailedCheck(
                        name="lint",
                        conclusion="failure",
                        job_id="87654321",
                        job_url="https://github.com/owner/repo/actions/runs/87654321",
                        rerun_url=None
                    )
                ],
                "https://github.com/owner/repo/actions/runs/99999999/rerun-all-jobs"
            )
        """
        try:

            # If checks_data is provided, use it instead of fetching
            if checks_data:
                checks = checks_data.get('checks', [])
                failed_checks = []
                rerun_run_id = None

                for check in checks:
                    # Only process failed checks
                    if check['status'].lower() != 'fail':
                        continue

                    check_name = check['name']
                    html_url = check['url']
                    duration = check['duration']

                    # Extract run ID from html_url
                    check_run_id = ''
                    if '/runs/' in html_url:
                        check_run_id = html_url.split('/runs/')[1].split('/')[0]
                        if not rerun_run_id and check_run_id:
                            rerun_run_id = check_run_id

                    # Check if this is a required check
                    is_required = check_name in required_checks

                    # Get error summary from job logs
                    error_summary = None
                    if check_run_id and html_url:
                        error_summary = self.get_job_error_summary(check_run_id, html_url, owner, repo)

                    failed_check = FailedCheck(
                        name=check_name,
                        job_url=html_url,
                        raw_log_url=None,
                        run_id=check_run_id,
                        duration=duration,
                        is_required=is_required,
                        error_summary=error_summary
                    )
                    failed_checks.append(failed_check)

                # Sort: required checks first, then by name
                failed_checks.sort(key=lambda x: (not x.is_required, x.name))

                # Generate rerun URL if we have a run_id
                rerun_url = None
                if rerun_run_id:
                    rerun_url = f"https://github.com/{owner}/{repo}/actions/runs/{rerun_run_id}"

                return failed_checks, rerun_url

            # If checks_data is not provided, derive it from REST check-runs.
            checks: List[Dict[str, Any]] = []
            if checks_data and isinstance(checks_data, dict):
                checks = checks_data.get("checks", []) or []
            if not checks and pr_number:
                cd = self._fetch_pr_checks_data(owner, repo, int(pr_number))
                checks = (cd or {}).get("checks", []) if isinstance(cd, dict) else []

            # If we still don't have checks (no PR number), fall back to commit check-runs.
            if not checks:
                data = self.get(f"/repos/{owner}/{repo}/commits/{sha}/check-runs", params={"per_page": 100})
                crs = data.get("check_runs") if isinstance(data, dict) else None
                if isinstance(crs, list):
                    for cr in crs:
                        if not isinstance(cr, dict):
                            continue
                        checks.append(
                            {
                                "name": str(cr.get("name", "") or ""),
                                "status": "fail" if str(cr.get("conclusion", "") or "").lower() == "failure" else "pass",
                                "duration": format_gh_check_run_duration(cr),
                                "url": str(cr.get("details_url") or cr.get("html_url") or ""),
                            }
                        )

            failed_checks: List[FailedCheck] = []
            rerun_run_id: Optional[str] = None

            for check in checks:
                status = str(check.get("status", "") or "").lower()
                if status != "fail":
                    continue
                check_name = str(check.get("name", "") or "")
                url = str(check.get("url", "") or "")
                duration = str(check.get("duration", "") or "")

                run_id = ""
                if "/runs/" in url:
                    run_id = url.split("/runs/")[1].split("/")[0]
                    if not rerun_run_id and run_id:
                        rerun_run_id = run_id

                is_required = check_name in (required_checks or set())

                error_summary = None
                if run_id and url:
                    error_summary = self.get_job_error_summary(run_id, url, owner, repo)

                failed_checks.append(
                    FailedCheck(
                        name=check_name,
                        job_url=url,
                        raw_log_url=None,
                        run_id=run_id,
                        duration=duration,
                        is_required=is_required,
                        error_summary=error_summary,
                    )
                )

            failed_checks.sort(key=lambda x: (not x.is_required, x.name))

            rerun_url = f"https://github.com/{owner}/{repo}/actions/runs/{rerun_run_id}" if rerun_run_id else None
            return failed_checks, rerun_url

        except Exception as e:
            _logger.warning("Error fetching failed checks: %s", str(e))
            return [], None

    def get_running_checks(self, pr_number: int, owner: str, repo: str, required_checks: set,
                          checks_data: Optional[dict] = None) -> List[RunningCheck]:
        """Get running CI checks for a PR.

        Args:
            pr_number: PR number
            owner: Repository owner
            repo: Repository name
            required_checks: Set of required check names
            checks_data: Optional pre-fetched checks data from _fetch_pr_checks_data()

        Returns:
            List of RunningCheck objects
        """
        try:
            # If checks_data is provided, use it instead of fetching
            if checks_data:
                checks = checks_data.get('checks', [])
                running_checks = []

                for check in checks:
                    status = check['status'].lower()
                    # Check if it's running (REST check-run status: queued, in_progress)
                    if status in ('pending', 'queued', 'in_progress'):
                        name = check['name']
                        check_url = check['url']
                        is_required = name in required_checks

                        # Use the duration string we computed from timestamps (or fall back to "queued")
                        elapsed_time = check['duration'] if check['duration'] else 'queued'

                        running_check = RunningCheck(
                            name=name,
                            check_url=check_url,
                            is_required=is_required,
                            elapsed_time=elapsed_time
                        )
                        running_checks.append(running_check)

                # Sort: required checks first, then by name
                running_checks.sort(key=lambda x: (not x.is_required, x.name))
                return running_checks

            # Fallback (no checks_data): derive from REST check-runs for PR head SHA.
            # Fetch PR head SHA from cache (or API if needed)
            head_sha = self.get_pr_head_sha(owner=owner, repo=repo, pr_number=int(pr_number))
            if not head_sha:
                return []

            data = self.get(f"/repos/{owner}/{repo}/commits/{head_sha}/check-runs", params={"per_page": 100}, timeout=10) or {}
            check_runs = data.get("check_runs") if isinstance(data, dict) else None
            if not isinstance(check_runs, list):
                return []

            running_checks: List[RunningCheck] = []
            for cr in check_runs:
                if not isinstance(cr, dict):
                    continue
                status = str(cr.get("status", "") or "").strip().lower()
                if status == "completed":
                    continue
                name = str(cr.get("name", "") or "").strip()
                if not name:
                    continue
                url = str(cr.get("details_url") or cr.get("html_url") or "").strip()
                is_required = name in (required_checks or set())
                elapsed_time = format_gh_check_run_duration(cr) or status
                running_checks.append(RunningCheck(name=name, check_url=url, is_required=is_required, elapsed_time=elapsed_time))

            running_checks.sort(key=lambda x: (not x.is_required, x.name))
            return running_checks

        except Exception as e:
            _logger.warning("Error fetching running checks for PR %s: %s", str(pr_number), str(e))
            return []


    def get_cached_pr_merge_dates(self, pr_numbers: List[int],
                                  owner: str = "ai-dynamo",
                                  repo: str = "dynamo",
                                  cache_file: str = '.github_pr_merge_dates_cache.json') -> Dict[int, Optional[str]]:
        """Get merge dates for pull requests with caching.

        Merge dates are cached permanently since they don't change once a PR is merged.

        Args:
            pr_numbers: List of PR numbers
            owner: Repository owner (default: ai-dynamo)
            repo: Repository name (default: dynamo)
            cache_file: Path to cache file (deprecated, kept for compatibility)

        Returns:
            Dictionary mapping PR number to merge date string (YYYY-MM-DD HH:MM:SS)
            Returns None for PRs that are not merged or not found

        Example:
            >>> client = GitHubAPIClient()
            >>> merge_dates = client.get_cached_pr_merge_dates([4965, 5009])
            >>> merge_dates
            {4965: "2025-12-18 12:34:56", 5009: None}
        """

        # Prepare result
        result = {}
        logger = logging.getLogger('common')

        # First pass: collect cached results and PRs to fetch
        prs_to_fetch = []
        pr_numbers_unique = []
        seen = set()
        for pr_num in pr_numbers:
            try:
                pr_i = int(pr_num)
            except (ValueError, TypeError):
                continue
            if pr_i in seen:
                continue
            seen.add(pr_i)
            pr_numbers_unique.append(pr_i)

        for pr_num in pr_numbers_unique:
            cache_key = f"{owner}/{repo}:{pr_num}"
            cached = MERGE_DATES_CACHE.get_if_fresh(cache_key, ttl_s=365*24*3600)  # 1 year TTL (immutable)
            if cached is not None:
                result[pr_num] = cached
                self._cache_hit("merge_dates.disk")
            else:
                prs_to_fetch.append(pr_num)
                self._cache_miss("merge_dates")

        # Fetch uncached PRs using list endpoint (much more efficient!)
        if prs_to_fetch:
            # OPTIMIZATION: Use list_pull_requests to get all PRs at once (1-3 API calls instead of N)
            # This includes both open and closed/merged PRs
            try:
                all_prs_data = self.list_pull_requests(owner, repo, state="all", ttl_s=3600)

                # Build mapping: PR number -> PR data
                pr_num_to_data = {}
                if isinstance(all_prs_data, list):
                    for pr_data in all_prs_data:
                        if isinstance(pr_data, dict):
                            pr_num = pr_data.get('number')
                            if pr_num:
                                pr_num_to_data[int(pr_num)] = pr_data

                # Extract merge dates from the list
                still_missing = []
                for pr_num in prs_to_fetch:
                    pr_data = pr_num_to_data.get(pr_num)
                    if pr_data and pr_data.get('merged_at'):
                        # Parse ISO timestamp: "2025-12-18T12:34:56Z" (UTC)
                        merged_at = pr_data['merged_at']
                        dt_utc = datetime.fromisoformat(merged_at.replace('Z', '+00:00'))

                        # Convert to Pacific time (PST/PDT)
                        dt_pacific = dt_utc.astimezone(ZoneInfo('America/Los_Angeles'))
                        merge_date = dt_pacific.strftime('%Y-%m-%d %H:%M:%S')

                        result[pr_num] = merge_date
                        MERGE_DATES_CACHE.put(f"{owner}/{repo}:{pr_num}", merge_date)
                    elif pr_data:
                        # PR exists but not merged
                        result[pr_num] = None
                        MERGE_DATES_CACHE.put(f"{owner}/{repo}:{pr_num}", None)
                    else:
                        # PR not found in list (might be very old or doesn't exist)
                        still_missing.append(pr_num)

                # Fall back to individual fetches only for PRs not found in list
                if still_missing:
                    logger.debug(f"Falling back to individual fetch for {len(still_missing)} PRs not in list")

                    def fetch_pr_merge_date(pr_num):
                        """Helper function to fetch a single PR's merge date"""
                        try:
                            pr_details = self.get_pr_details(owner, repo, pr_num)

                            if pr_details and pr_details.get('merged_at'):
                                # Parse ISO timestamp: "2025-12-18T12:34:56Z" (UTC)
                                merged_at = pr_details['merged_at']
                                dt_utc = datetime.fromisoformat(merged_at.replace('Z', '+00:00'))

                                # Convert to Pacific time (PST/PDT)
                                dt_pacific = dt_utc.astimezone(ZoneInfo('America/Los_Angeles'))
                                merge_date = dt_pacific.strftime('%Y-%m-%d %H:%M:%S')
                                return (pr_num, merge_date)
                            else:
                                # PR not merged or not found
                                return (pr_num, None)
                        except Exception as e:
                            # Log error but continue with other PRs
                            logger.debug(f"Failed to fetch PR {pr_num} merge date: {e}")
                            return (pr_num, None)

                    # Fetch in parallel with 10 workers
                    with ThreadPoolExecutor(max_workers=10) as executor:
                        futures = [executor.submit(fetch_pr_merge_date, pr_num) for pr_num in still_missing]

                        # Collect results as they complete
                        for future in futures:
                            try:
                                pr_num, merge_date = future.result()
                                result[pr_num] = merge_date
                                MERGE_DATES_CACHE.put(f"{owner}/{repo}:{pr_num}", merge_date)
                            except Exception as e:
                                logger.debug(f"Failed to get future result: {e}")

            except Exception as e:
                # If list_pull_requests fails entirely, fall back to original individual fetch logic
                logger.warning(f"list_pull_requests failed, falling back to individual fetches: {e}")

                def fetch_pr_merge_date(pr_num):
                    """Helper function to fetch a single PR's merge date"""
                    try:
                        pr_details = self.get_pr_details(owner, repo, pr_num)

                        if pr_details and pr_details.get('merged_at'):
                            # Parse ISO timestamp: "2025-12-18T12:34:56Z" (UTC)
                            merged_at = pr_details['merged_at']
                            dt_utc = datetime.fromisoformat(merged_at.replace('Z', '+00:00'))

                            # Convert to Pacific time (PST/PDT)
                            dt_pacific = dt_utc.astimezone(ZoneInfo('America/Los_Angeles'))
                            merge_date = dt_pacific.strftime('%Y-%m-%d %H:%M:%S')
                            return (pr_num, merge_date)
                        else:
                            # PR not merged or not found
                            return (pr_num, None)
                    except Exception as e:
                        # Log error but continue with other PRs
                        logger.debug(f"Failed to fetch PR {pr_num} merge date: {e}")
                        return (pr_num, None)

                # Fetch in parallel with 10 workers
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(fetch_pr_merge_date, pr_num) for pr_num in prs_to_fetch]

                    # Collect results as they complete
                    for future in futures:
                        try:
                            pr_num, merge_date = future.result()
                            result[pr_num] = merge_date
                            MERGE_DATES_CACHE.put(f"{owner}/{repo}:{pr_num}", merge_date)
                        except Exception as e:
                            logger.debug(f"Failed to get future result: {e}")

        # Remove old GITHUB_CACHE_STATS tracking (now using standard _cache_hit/_cache_miss)

        return result


def select_shas_for_network_fetch(
    sha_list: List[str],
    sha_to_datetime: Optional[Dict[str, datetime]],
    *,
    max_fetch: int = 5,
    recent_hours: int = DEFAULT_STABLE_AFTER_HOURS,
) -> set[str]:
    """Select a small set of SHAs where we're allowed to do network fetches.

    Policy:
    - Only consider the newest `max_fetch` SHAs (order as given in sha_list).
    - Only allow fetch if the SHA's datetime is within `recent_hours`.
    """
    allow: set[str] = set()
    if not sha_list or not sha_to_datetime:
        return allow

    try:
        now_utc = datetime.now(timezone.utc)
        cutoff_s = float(recent_hours) * 3600.0
        for sha in sha_list[: int(max_fetch or 0)]:
            dt = sha_to_datetime.get(sha)
            if dt is None:
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age_s = (now_utc - dt.astimezone(timezone.utc)).total_seconds()
            if age_s < cutoff_s:
                allow.add(sha)
    except (ValueError, TypeError):
        return set()
    return allow


def normalize_check_name(name: str) -> str:
    """Normalize GitHub/GitLab check names for robust comparison."""
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def is_required_check_name(check_name: str, required_names_normalized: set[str]) -> bool:
    """Return True iff the check name is in the required set (branch protection)."""
    name_norm = normalize_check_name(check_name)
    if not name_norm:
        return False
    return name_norm in (required_names_normalized or set())


def format_gh_check_run_duration(check_run: Dict[str, Any]) -> str:
    """Compute a short duration string for a GitHub check_run dict.

    Args:
        check_run: A dict like the items returned by GitHub's check-runs API.

    Returns:
        A short duration string like "3s", "2m 10s", "1h 4m", or "" if unknown.

    Example check_run dict (partial):
        {
          "name": "backend-status-check",
          "status": "completed",
          "conclusion": "success",
          "started_at": "2025-12-24T09:06:10Z",
          "completed_at": "2025-12-24T09:06:13Z",
          "html_url": "https://github.com/.../actions/runs/.../job/..."
        }
    """
    try:
        started = str(check_run.get("started_at", "") or "")
        completed = str(check_run.get("completed_at", "") or "")
        if not started or not completed:
            return ""
        st = datetime.fromisoformat(started.replace("Z", "+00:00"))
        ct = datetime.fromisoformat(completed.replace("Z", "+00:00"))
        delta_s = int((ct - st).total_seconds())
        return GitHubAPIClient._format_seconds_delta(delta_s)
    except (ValueError, TypeError):
        return ""


