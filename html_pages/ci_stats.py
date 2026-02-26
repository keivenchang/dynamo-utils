# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CI statistics and API cache reporting: GitHub/GitLab stats rows and page-level stats."""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common_github import GITHUB_CACHE_STATS, GITHUB_API_STATS, COMMIT_HISTORY_PERF_STATS
from common_github.api.pr_branch_cached import TTL_POLICY_DESCRIPTION as PR_BRANCH_TTL_POLICY_DESCRIPTION
from common_github.api.pr_checks_cached import TTL_POLICY_DESCRIPTION as PR_CHECKS_TTL_POLICY_DESCRIPTION
from common_github.api.pr_comments_cached import TTL_POLICY_DESCRIPTION as PR_COMMENTS_TTL_POLICY_DESCRIPTION
from common_github.api.pr_details_cached import TTL_POLICY_DESCRIPTION as PR_DETAILS_TTL_POLICY_DESCRIPTION
from common_github.api.pr_reviews_cached import TTL_POLICY_DESCRIPTION as PR_REVIEWS_TTL_POLICY_DESCRIPTION
from common_github.api.pulls_list_cached import TTL_POLICY_DESCRIPTION as PULLS_LIST_TTL_POLICY_DESCRIPTION
from common_github.api.search_issues_cached import TTL_POLICY_DESCRIPTION as SEARCH_ISSUES_TTL_POLICY_DESCRIPTION
from common_github.api.required_checks_cached import TTL_POLICY_DESCRIPTION as REQUIRED_CHECKS_TTL_POLICY_DESCRIPTION
from common_github.api.pr_head_sha_cached import TTL_POLICY_DESCRIPTION as PR_HEAD_SHA_TTL_POLICY_DESCRIPTION
from common_gitlab.api.mr_pipelines_cached import TTL_POLICY_DESCRIPTION as GITLAB_MR_PIPELINES_TTL_POLICY_DESCRIPTION
from common_gitlab.api.pipeline_jobs_cached import TTL_POLICY_DESCRIPTION as GITLAB_PIPELINE_JOBS_TTL_POLICY_DESCRIPTION
from common_gitlab.api.pipeline_status_cached import TTL_POLICY_DESCRIPTION as GITLAB_PIPELINE_STATUS_TTL_POLICY_DESCRIPTION
from common_gitlab.api.registry_images_cached import TTL_POLICY_DESCRIPTION as GITLAB_REGISTRY_IMAGES_TTL_POLICY_DESCRIPTION
from cache_pytest_timings import PYTEST_TIMINGS_CACHE
from cache_snippet import SNIPPET_CACHE
from cache_commit_history import COMMIT_HISTORY_CACHE

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import DEFAULT_RAW_LOG_TEXT_TTL_S, format_duration_compact

logger = logging.getLogger(__name__)


def _format_ttl_duration(seconds: int) -> str:
    """Convert TTL seconds to human-readable format (e.g., '5m', '1h', '30d')."""
    return format_duration_compact(seconds)


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
    gh_stats = github_api.get_gh_call_stats() or {}

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
    # Rate limit values for the compact "Budget & mode" block:
    # - Prefer the "after" snapshot (if captured), to avoid extra /rate_limit calls.
    # - Otherwise fall back to get_core_rate_limit_info() (unless disabled).
    try:
        snaps0 = github_api.get_rate_limit_snapshots() if github_api is not None else {}
    except Exception:
        snaps0 = {}
    after0 = (snaps0.get("after") or {}) if isinstance(snaps0, dict) else {}
    after0_core = (after0.get("core") or {}) if isinstance(after0, dict) else {}
    rem = after0_core.get("remaining")
    lim = after0_core.get("limit")
    reset_pt = None
    # (reset_pt is optional; we keep it only in the budget block, not as a standalone row)
    try:
        if after0_core.get("reset") is not None:
            reset_epoch = int(after0_core.get("reset"))
            reset_pt = datetime.fromtimestamp(reset_epoch).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    except (ValueError, TypeError, OSError):
        reset_pt = None

    if rem is None and lim is None and not os.environ.get("DYNAMO_UTILS_DISABLE_INTERNAL_RATE_LIMIT_CHECKS"):
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

    # REST summary - reorganized to highlight ETag optimization
    rest_total = int(rest.get("total") or 0)
    rest_304 = int(GITHUB_API_STATS.etag_304_total or 0)
    rest_billable = max(0, rest_total - rest_304)
    
    rows.append(("github.rest.calls", str(rest_total), f"Total GitHub REST API calls made ({rest_total} total = {rest_304} free + {rest_billable} billable)"))
    rows.append(("github.rest.etag_304_total", str(rest_304), "🆓 FREE: 304 Not Modified responses (ETag hits - don't count against rate limit!)"))
    rows.append(("github.rest.billable_estimate", str(rest_billable), f"💰 BILLABLE: Calls counting against rate limit (total - 304 free)"))
    rows.append(("github.rest.ok", str(int(rest.get("success_total") or 0)), "Successful API calls"))
    rows.append(("github.rest.errors", str(int(rest.get("error_total") or 0)), "Failed API calls"))
    rows.append(("github.rest.time_total_secs", f"{float(rest.get('time_total_s') or 0.0):.2f}s", "Total time spent in API calls (SUM of all parallel threads - see note below)"))

    # `gh` CLI API stats (separate buckets: REST core vs GraphQL).
    gh_core_calls = int(gh_stats.get("core_calls_total") or 0)
    gh_core_304 = int(gh_stats.get("core_304_total") or 0)
    gh_core_billable = max(0, gh_core_calls - gh_core_304)
    gh_gql_calls = int(gh_stats.get("graphql_calls_total") or 0)
    gh_gql_cost = int(gh_stats.get("graphql_cost_total") or 0)

    rows.append(("github.gh.core.calls", str(gh_core_calls), "Number of `gh api <REST endpoint>` calls (core bucket)"))
    rows.append(("github.gh.core.etag_304_total", str(gh_core_304), "304 responses from `gh` REST calls (free; do not decrement core)"))
    rows.append(("github.gh.core.billable_calls", str(gh_core_billable), "Estimated billable core calls from `gh` REST (calls - 304)"))
    rows.append(("github.gh.graphql.calls", str(gh_gql_calls), "Number of `gh api graphql` calls (GraphQL bucket)"))
    rows.append(("github.gh.graphql.cost_total", str(gh_gql_cost), "Sum of GraphQL rateLimit.cost across `gh api graphql` calls"))

    # Helpful combined estimate for core consumption (excludes rate_limit polling).
    rows.append(("github.core.billable_calls_estimate", str(rest_billable + gh_core_billable), "💰 TOTAL BILLABLE: Estimated total billable core calls (python REST + gh REST; excludes 304s)"))

    # ETag stats breakdown by endpoint (show which endpoints benefit from caching)
    if rest_304 > 0:
        # Show top ETag 304 by endpoint
        etag_304_by_label = dict(GITHUB_API_STATS.etag_304_by_label or {})
        if etag_304_by_label:
            sorted_labels = sorted(etag_304_by_label.items(), key=lambda kv: (-kv[1], kv[0]))
            for lbl, cnt in sorted_labels[:5]:  # Top 5
                rows.append((f"github.rest.etag_304.{lbl}", str(int(cnt)), f"🆓 304 responses for {lbl} (cached, free)"))
            if len(sorted_labels) > 5:
                remaining = sum(cnt for lbl, cnt in sorted_labels[5:])
                rows.append(("github.rest.etag_304.other", str(remaining), "🆓 304 responses for other endpoints"))

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
    # NOTE: `github.core_remaining` / `github.core_resets` are intentionally omitted when
    # before/after snapshots are present (those rows are redundant). The compact budget block
    # still shows core_remaining/core_resets as a convenience.

    # Before/after rate limit snapshots (if captured by the caller).
    # These should NOT trigger any /rate_limit calls during rendering.
    try:
        snaps = github_api.get_rate_limit_snapshots() if github_api is not None else {}
    except Exception:
        snaps = {}
    before = (snaps.get("before") or {}) if isinstance(snaps, dict) else {}
    after = (snaps.get("after") or {}) if isinstance(snaps, dict) else {}
    if before or after:
        rows.append(("## Rate Limit (before/after)", None, ""))
        for bucket in ("core", "graphql"):
            b = (before.get(bucket) or {}) if isinstance(before, dict) else {}
            a = (after.get(bucket) or {}) if isinstance(after, dict) else {}
            b_rem = b.get("remaining")
            a_rem = a.get("remaining")
            b_used = b.get("used")
            a_used = a.get("used")
            if b_rem is not None:
                rows.append((f"github.rate_limit.before.{bucket}.remaining", str(b_rem), f"{bucket} remaining before"))
            if a_rem is not None:
                rows.append((f"github.rate_limit.after.{bucket}.remaining", str(a_rem), f"{bucket} remaining after"))
            if b_used is not None:
                rows.append((f"github.rate_limit.before.{bucket}.used", str(b_used), f"{bucket} used before"))
            if a_used is not None:
                rows.append((f"github.rate_limit.after.{bucket}.used", str(a_used), f"{bucket} used after"))
            try:
                # Prefer used-based delta when available (more robust for /rate_limit semantics).
                if b_used is not None and a_used is not None:
                    rows.append((
                        f"github.rate_limit.delta.{bucket}.consumed",
                        str(int(a_used) - int(b_used)),
                        f"{bucket} consumed between snapshots (excludes BEFORE call; includes AFTER call)",
                    ))
                elif b_rem is not None and a_rem is not None:
                    rows.append((
                        f"github.rate_limit.delta.{bucket}.consumed",
                        str(int(b_rem) - int(a_rem)),
                        f"{bucket} consumed between snapshots (excludes BEFORE call; includes AFTER call)",
                    ))
            except (ValueError, TypeError):
                pass

    # Cache summary (individual flat entries)
    rows.append(("github.cache.all.hits", str(int(cache.get("hits_total") or 0)), "Total cache hits across all operations"))
    rows.append(("github.cache.all.misses", str(int(cache.get("misses_total") or 0)), "Total cache misses"))
    rows.append(("github.cache.all.writes_ops", str(int(cache.get("writes_ops_total") or 0)), "Number of cache write operations"))
    rows.append(("github.cache.all.writes_entries", str(int(cache.get("writes_entries_total") or 0)), "Number of entries written to cache"))

    # TTL documentation (always show; independent of whether counters are non-zero in this run).
    rows.append(("github.cache.actions_job_details.ttl.completed", "30d", "Typical TTL used for completed job-details cache (dashboards pass 30d)"))
    rows.append(("github.cache.actions_job_details.ttl.in_progress", "adaptive", "Adaptive TTL for in-progress jobs: <1h=2m, <2h=4m, <4h=30m, <8h=60m, <12h=80m, >=12h=120m"))
    rows.append(("github.cache.actions_job_status.ttl.in_progress", "adaptive", "Adaptive TTL for non-completed jobs: <30m=1m, <2h=2m, <4h=10m, >=4h=15m (until completed)"))

    # In-progress job-details (now cached with adaptive TTL)
    #
    # As of recent changes, in-progress job details ARE cached with adaptive TTL.
    # This counter tracks how many in-progress fetches occurred (for monitoring).
    inprog = int(getattr(GITHUB_API_STATS, "actions_job_details_in_progress_uncached_total", 0) or 0)
    if inprog:
        rows.append((
            "github.cache.actions_job_details.in_progress_fetched",
            str(inprog),
            "Actions job-details fetches that returned in_progress (now cached with adaptive TTL)"
        ))
        try:
            job_ids = sorted(str(x) for x in (getattr(GITHUB_API_STATS, "actions_job_details_in_progress_uncached_job_ids", set()) or set()))
            if job_ids:
                rows.append(("github.cache.actions_job_details.in_progress_fetched.sample_job_ids", ",".join(job_ids[:8]), "Sample job_ids (max 8)"))
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
            "actions_jobs": "Actions workflow run jobs [key: owner/repo:run_jobs:run_id or owner/repo:job:job_id]",
            "actions_workflow": "Workflow run metadata [key: owner/repo:run_id]",
            "required_checks": "Required PR check names [key: owner/repo:pr#]",
            "pr_info": "PR metadata (author, labels, reviews) [key: owner/repo#PR]",
            "search_issues": "GitHub issue search results [key: query_hash]",
            "job_log": "Parsed job log content [key: job_id]",
        }

        cache_descriptions_disk = {
            "actions_job_status": "GitHub Actions job status [actions_jobs.json] [key: owner/repo:jobstatus:job_id]",
            "actions_job_details": "GitHub Actions job details [actions_jobs.json] [key: owner/repo:job:job_id]",
            "actions_jobs": "Actions workflow run jobs [actions_jobs.json] [key: owner/repo:run_jobs:run_id or owner/repo:job:job_id]",
            "pr_checks": "PR check runs [pr_checks.json] [key: owner/repo#PR or owner/repo#PR:sha]",
            "pr_comments": "PR comment threads and resolution status [pr_comments.json] [key: owner/repo#PR]",
            "pr_head_sha": "PR head SHA and merge metadata [pr_head_sha.json] [key: owner/repo:pr:PR:head_sha]",
            "pr_reviews": "PR review comments and states [pr_reviews.json] [key: owner/repo#PR]",
            "pulls_list": "Pull request list responses [pulls_list.json] [key: owner/repo:state]",
            "pull_request": "Individual PR details [pr_details.json] [key: owner/repo:pr:PR]",
            "pr_branch": "PR branch information [pr_branches.json] [key: owner/repo:branch]",
            "pr_info": "Full PR details with required checks and reviews [pr_info.json] [key: owner/repo#PR]",
            "search_issues": "GitHub search/issues API results [search_issues.json] [key: query_hash]",
            "job_log": "Job log error summaries and snippets [job_logs_cache.json] [key: job_id]",
            "raw_log_text": f"Raw CI log text content index [index.json] [key: job_id]",
            "required_checks": "Required PR check names [required_checks.json] [key: owner/repo:pr#]",
            "merge_dates": f"PR merge dates from GitHub API (extracted from pulls_list) [key: owner/repo#PR]",
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
            "actions_jobs": "30d (completed workflow runs never change; ETag-friendly)",
            "pr_checks": PR_CHECKS_TTL_POLICY_DESCRIPTION,
            "pr_comments": PR_COMMENTS_TTL_POLICY_DESCRIPTION,
            "pr_head_sha": PR_HEAD_SHA_TTL_POLICY_DESCRIPTION,
            "pr_reviews": PR_REVIEWS_TTL_POLICY_DESCRIPTION,
            "pulls_list": PULLS_LIST_TTL_POLICY_DESCRIPTION,
            "pull_request": PR_DETAILS_TTL_POLICY_DESCRIPTION,
            "pr_branch": PR_BRANCH_TTL_POLICY_DESCRIPTION,
            "pr_info": "by updated_at timestamp (invalidated when PR changes)",
            "search_issues": SEARCH_ISSUES_TTL_POLICY_DESCRIPTION,
            "job_log": "∞ (immutable, no TTL check)",
            "raw_log_text": f"{_format_ttl_duration(DEFAULT_RAW_LOG_TEXT_TTL_S)} (immutable once completed)",
            "required_checks": REQUIRED_CHECKS_TTL_POLICY_DESCRIPTION,
            "merge_dates": "extracted from pulls_list (merged_at field), immutable after merge",
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

    # Ordered list of literal API calls (REST URLs + `gh` commands) for debugging.
    #
    # Always render this section so it's obvious whether we recorded anything.
    api_calls_txt = ""
    if github_api is not None and hasattr(github_api, "get_actual_api_calls_text"):
        api_calls_txt = str(github_api.get_actual_api_calls_text() or "").strip()
    if not api_calls_txt:
        api_calls_txt = "(none recorded)"
    rows.append(("## API Calls (ordered)", None, ""))
    rows.append(("github.api.calls", api_calls_txt, "Literal API calls executed (in order)"))

    return rows


def gitlab_api_stats_rows(
    *,
    gitlab_api: Optional[Any],
    top_n: int = 15,
) -> List[Tuple[str, Optional[str], str]]:
    """Build GitLab API statistics rows using the same key conventions as GitHub.

    Keys:
      - gitlab.cache.all.(hits|misses|writes_ops|writes_entries)
      - gitlab.cache.<cache>.(disk|mem|ttl|hits|misses)
      - gitlab.rest.by_category.<label>.(calls|time_secs)
    """
    rows: List[Tuple[str, Optional[str], str]] = []
    if gitlab_api is None:
        return rows

    # Cache stats (operation-level) from GitLabAPIClient.get_cache_stats()
    try:
        cache = gitlab_api.get_cache_stats() if hasattr(gitlab_api, "get_cache_stats") else {}
    except Exception:
        cache = {}
    cache = cache if isinstance(cache, dict) else {}

    rows.append(("gitlab.cache.all.hits", str(int(cache.get("hits_total") or 0)), "Total cache hits across all GitLab cache operations"))
    rows.append(("gitlab.cache.all.misses", str(int(cache.get("misses_total") or 0)), "Total cache misses across all GitLab cache operations"))
    rows.append(("gitlab.cache.all.writes_ops", str(int(cache.get("writes_ops_total") or 0)), "Number of GitLab cache write operations"))
    rows.append(("gitlab.cache.all.writes_entries", str(int(cache.get("writes_entries_total") or 0)), "Number of GitLab cache entries written"))

    hits_by = dict(cache.get("hits_by") or {}) if isinstance(cache, dict) else {}
    misses_by = dict(cache.get("misses_by") or {}) if isinstance(cache, dict) else {}

    def _disk_entry_count(path: Path) -> int:
        try:
            if not path.exists():
                return 0
            raw = json.loads(path.read_text() or "{}")
            if not isinstance(raw, dict):
                return 0
            if "_metadata" in raw:
                raw = {k: v for k, v in raw.items() if k != "_metadata"}
            return int(len(raw))
        except Exception:
            return 0

    # Known GitLab caches (disk-backed JSON)
    try:
        from common import resolve_cache_path  # local import (avoid cycles)
    except Exception:
        resolve_cache_path = None  # type: ignore[assignment]

    caches: List[Tuple[str, str, str]] = [
        ("registry_images", "gitlab_commit_sha.json", "GitLab registry tags by commit SHA [key: <40-sha>]"),
        ("pipeline_status", "gitlab_pipeline_status.json", "GitLab pipeline status by commit SHA [key: <40-sha>]"),
        ("pipeline_jobs", "gitlab_pipeline_jobs.json", "GitLab pipeline job counts by pipeline_id [key: pipeline_id]"),
        ("pipeline_jobs_details", "gitlab_pipeline_jobs_details.json", "GitLab pipeline job details by pipeline_id [key: pipeline_id]"),
        ("mr_pipelines", "gitlab_mr_pipelines.json", "GitLab MR -> latest pipeline [key: mr_iid]"),
    ]
    ttl_by_cache = {
        "registry_images": str(GITLAB_REGISTRY_IMAGES_TTL_POLICY_DESCRIPTION),
        "pipeline_status": str(GITLAB_PIPELINE_STATUS_TTL_POLICY_DESCRIPTION),
        "pipeline_jobs": str(GITLAB_PIPELINE_JOBS_TTL_POLICY_DESCRIPTION),
        "pipeline_jobs_details": str(GITLAB_PIPELINE_JOBS_TTL_POLICY_DESCRIPTION),
        "mr_pipelines": str(GITLAB_MR_PIPELINES_TTL_POLICY_DESCRIPTION),
    }

    for cache_name, filename, desc in caches:
        disk = 0
        if resolve_cache_path is not None:
            disk = _disk_entry_count(resolve_cache_path(filename))
        rows.append((f"gitlab.cache.{cache_name}.disk", str(disk), f"{desc} [{filename}]"))
        rows.append((f"gitlab.cache.{cache_name}.mem", "0", ""))
        rows.append((f"gitlab.cache.{cache_name}.ttl", str(ttl_by_cache.get(cache_name, "varies")), "Cache TTL policy"))
        h = int(sum(int(v or 0) for k, v in hits_by.items() if str(k) == cache_name or str(k).startswith(f"{cache_name}.")))
        m = int(sum(int(v or 0) for k, v in misses_by.items() if str(k) == cache_name or str(k).startswith(f"{cache_name}.")))
        rows.append((f"gitlab.cache.{cache_name}.hits", str(h), ""))
        rows.append((f"gitlab.cache.{cache_name}.misses", str(m), ""))

    # REST by category (label) from GitLabAPIClient.get_rest_call_stats()
    try:
        st = gitlab_api.get_rest_call_stats() if hasattr(gitlab_api, "get_rest_call_stats") else {}
    except Exception:
        st = {}
    st = st if isinstance(st, dict) else {}
    by_label = dict(st.get("by_label") or {}) if isinstance(st, dict) else {}
    time_by_label_s = dict(st.get("time_by_label_s") or {}) if isinstance(st, dict) else {}
    labels = sorted(set(list(by_label.keys()) + list(time_by_label_s.keys())))
    if labels:
        labels.sort(key=lambda k: (-int(by_label.get(k, 0) or 0), -float(time_by_label_s.get(k, 0.0) or 0.0), k))
        for lbl in labels[: max(0, int(top_n))]:
            c = int(by_label.get(lbl, 0) or 0)
            t = float(time_by_label_s.get(lbl, 0.0) or 0.0)
            rows.append((f"gitlab.rest.by_category.{lbl}.calls", str(c), f"API calls for {lbl}"))
            rows.append((f"gitlab.rest.by_category.{lbl}.time_secs", f"{t:.2f}s", f"Time spent in {lbl} calls"))

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
        page_stats.append(("composite_sha.cache.disk", str(commit_disk_count), "Commit history cache entries (on disk)"))
        page_stats.append(("composite_sha.cache.hits", str(perf_stats.composite_sha_cache_hit), ""))
        page_stats.append(("composite_sha.cache.misses", str(perf_stats.composite_sha_cache_miss), ""))
        page_stats.append(("composite_sha.errors", str(perf_stats.composite_sha_errors), "Errors computing composite SHAs (commit history only)"))
        page_stats.append(("composite_sha.total_secs", f"{perf_stats.composite_sha_total_secs:.2f}s", "Total time computing composite SHAs (commit history only)"))
        page_stats.append(("composite_sha.compute_secs", f"{perf_stats.composite_sha_compute_secs:.2f}s", "Time spent in SHA computations (commit history only)"))
    else:
        commit_mem_count, commit_disk_count = COMMIT_HISTORY_CACHE.get_cache_sizes()
        page_stats.append(("composite_sha.cache.mem", str(commit_mem_count), "Commit history cache entries (in memory)"))
        page_stats.append(("composite_sha.cache.disk", str(commit_disk_count), "Commit history cache entries (on disk)"))
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

    # GitLab cache + REST statistics (GitHub-compatible key schema)
    if gitlab_client is not None:
        page_stats.extend(gitlab_api_stats_rows(gitlab_api=gitlab_client))

    # Sort all stats alphabetically by key (but preserve section headers with None values)
    section_headers = [s for s in page_stats if s[1] is None]
    regular_stats = [s for s in page_stats if s[1] is not None]
    regular_stats.sort(key=lambda x: x[0].lower())

    return regular_stats
