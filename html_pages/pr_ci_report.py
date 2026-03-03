#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PR & CI statistics report generator.

Data sources (GitHub REST API v3):
  - GET /repos/{owner}/{repo}/pulls?state=all         → PR metadata (author, dates, merge status)
  - GET /repos/{owner}/{repo}/actions/runs?event=...   → workflow run records

GitHub Actions event types and what they mean for this repo:
  - "push" (44% of runs): Triggered on every push.  Two sub-populations:
      * head_branch="main"              → real post-merge CI
      * head_branch="pull-request/{N}"  → fork-PR CI (a merge bot pushes the PR
        branch so CI can run with repo secrets; these are pre-merge, not post-merge)
  - "pull_request" (33%): Standard pre-merge CI. Runs in the PR head branch context
    with restricted permissions (no repo secrets). Covers same-repo PRs well.
  - "pull_request_target" (22%): Pre-merge CI that runs in the BASE branch context
    (e.g. main) so it has full permissions and secrets.  Used for workflows that
    need to post comments, labels, or access private APIs (Lint PR, Label PR,
    DCO Commenter, NVIDIA Dynamo Github Validation).

Matching workflow runs to PRs (hardest part):
  We need to connect each workflow run back to the PR it belongs to. Three strategies
  (union, deduped by run ID):
    1. head_branch == PR's head.ref   — works for same-repo PRs, breaks for forks
       that reuse common branch names like "main" or "fix-typo"
    2. head_branch == "pull-request/{N}" — reliable for fork PRs; the merge bot
       always uses this naming convention with the PR number embedded
    3. run.pull_requests[].number     — direct link from GitHub, but GitHub only
       populates this field ~39% of the time (known API limitation for forks)
  Strategy (2) is the most reliable for fork PRs and recovers the majority of
  previously unmatched runs.

Key metrics:
  - CI triggers/PR: distinct head_sha values across all runs for a PR.  Each push
    to a PR creates a new head_sha, so this counts "how many times the author
    pushed code" (including the initial push).
  - Pre-merge failure rate: (failed runs / total runs) for all PR-associated CI.
    A "failure" is conclusion="failure" on any workflow run.
  - Post-merge failure rate: same ratio but only for push runs on main.
  - Avg time-to-merge: hours between PR created_at and merged_at.

Caching:
  All workflow runs are cached on disk for 60 days (see actions_runs_list_cached.py).
  PR metadata is cached with a 1-hour TTL.  After the first run for a given time
  window, subsequent runs hit 100% cache with zero API calls.
"""

import argparse
import json
import logging
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
for p in [str(_project_root), str(_script_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from common_github import GitHubAPIClient, GITHUB_API_STATS
from common_github.api.actions_runs_list_cached import list_workflow_runs
from common_github.api.pulls_list_cached import list_pull_requests_cached

logger = logging.getLogger("pr_ci_report")


CODE_WORKFLOWS = {"PR", "Pre Merge", "Rust pre-merge checks", "NVIDIA Dynamo Github Validation", "NVIDIA Test Lab Validation"}


@dataclass
class SplitStats:
    """Metrics for one side of the code/non-code split."""
    prs_opened: int = 0
    prs_merged: int = 0
    unique_authors: int = 0
    avg_hours_to_merge: Optional[float] = None
    total_ci_triggers: int = 0
    prs_with_ci: int = 0
    mean_ci_triggers_per_pr: Optional[float] = None
    manual_retrigger_ci_runs: int = 0
    prs_with_manual_retrigger: int = 0
    worst_retrigger_pr: Optional[Tuple[int, int]] = None
    prs_with_ci_failure: int = 0
    pr_ci_failure_rate: float = 0.0


@dataclass
class PRStats:
    pr_number: int
    author: str
    title: str
    created_at: str
    merged_at: Optional[str]
    is_merged: bool
    head_sha: str
    total_ci_runs: int = 0
    ci_trigger_count: int = 0
    ci_retrigger_count: int = 0
    manual_retrigger_ci_runs: int = 0
    hours_to_merge: Optional[float] = None
    had_failure: bool = False
    failure_count: int = 0
    success_count: int = 0
    cancelled_count: int = 0


@dataclass
class BucketStats:
    bucket_label: str
    bucket_start: str
    prs_opened: int = 0
    prs_merged: int = 0
    unique_authors: int = 0
    author_counts: Dict[str, int] = field(default_factory=dict)
    total_ci_triggers: int = 0
    prs_with_ci: int = 0
    mean_ci_triggers_per_pr: Optional[float] = None
    total_ci_runs: int = 0
    manual_retrigger_ci_runs: int = 0
    prs_with_manual_retrigger: int = 0
    worst_retrigger_pr: Optional[Tuple[int, int]] = None
    prs_with_ci_failure: int = 0
    pr_ci_failure_rate: float = 0.0
    main_pushes_total: int = 0
    main_pushes_failed: int = 0
    main_failure_rate: float = 0.0
    avg_hours_to_merge: Optional[float] = None
    top_failing_workflows: List[Tuple[str, int]] = field(default_factory=list)
    code: SplitStats = field(default_factory=SplitStats)
    noncode: SplitStats = field(default_factory=SplitStats)


@dataclass
class OverallStats:
    days: int
    bucket_size: str
    repo: str
    total_prs_opened: int = 0
    total_prs_merged: int = 0
    unique_authors: int = 0
    total_ci_triggers: int = 0
    prs_with_ci: int = 0
    avg_ci_triggers_per_pr: float = 0.0
    total_ci_runs: int = 0
    total_manual_retrigger_ci_runs: int = 0
    prs_with_manual_retrigger: int = 0
    worst_retrigger_pr: Optional[Tuple[int, int]] = None
    prs_with_ci_failure: int = 0
    pr_ci_failure_rate: float = 0.0
    main_pushes_total: int = 0
    main_pushes_failed: int = 0
    main_failure_rate: float = 0.0
    avg_hours_to_merge: Optional[float] = None
    top_authors: List[Tuple[str, int]] = field(default_factory=list)
    top_failing_workflows: List[Tuple[str, int]] = field(default_factory=list)
    code: SplitStats = field(default_factory=SplitStats)
    noncode: SplitStats = field(default_factory=SplitStats)


def classify_pr_is_code(pr_runs: List[Dict[str, Any]], run_to_pr: Dict[int, int]) -> Dict[int, bool]:
    """Classify PRs as code (True) or non-code (False) by workflow names.

    A PR is "code" if any of its CI runs match CODE_WORKFLOWS.
    """
    pr_workflows: Dict[int, set] = defaultdict(set)
    for r in pr_runs:
        prn = run_to_pr.get(int(r.get("id", 0)))
        if prn is not None:
            pr_workflows[prn].add(r.get("name", ""))
    return {prn: bool(wfs & CODE_WORKFLOWS) for prn, wfs in pr_workflows.items()}


def _parse_iso(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _date_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def collect_pr_metadata(api: GitHubAPIClient, owner: str, repo: str, since: datetime, days: int = 30) -> Dict[int, Dict[str, Any]]:
    """Fetch all PRs active within the window via GET /repos/{owner}/{repo}/pulls?state=all.

    Returns dict keyed by PR number.  Includes PRs that were **created** OR
    **merged** within the window (a PR created 30 days ago but merged yesterday
    must appear so its merge is counted on the correct day).
    """
    pages_needed = max(5, (days + 6) // 7 * 3)
    all_prs = list_pull_requests_cached(api, owner=owner, repo=repo, state="all", ttl_s=3600, max_pages=pages_needed)
    result: Dict[int, Dict[str, Any]] = {}
    for pr in all_prs:
        if not isinstance(pr, dict):
            continue
        created = _parse_iso(pr.get("created_at", ""))
        merged = _parse_iso(pr.get("merged_at") or "")
        if (created and created >= since) or (merged and merged >= since):
            number = pr.get("number")
            if number is not None:
                result[int(number)] = pr
    return result


def collect_workflow_runs(
    api: GitHubAPIClient,
    owner: str,
    repo: str,
    event: str,
    since: datetime,
    branch: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch workflow runs via GET /repos/{owner}/{repo}/actions/runs?event={event}.

    Each run dict (after _slim_run extraction) contains: id, name, head_branch,
    head_sha, event, status, conclusion, run_attempt, created_at, actor_login,
    pr_numbers, html_url.  Runs are cached per-run for 60 days on disk.
    """
    since_str = since.strftime("%Y-%m-%dT%H:%M:%SZ")
    return list_workflow_runs(
        api,
        owner=owner,
        repo=repo,
        event=event,
        branch=branch,
        since_date=since_str,
        ttl_s=600,
    )


def build_pr_stats(
    pr_runs: List[Dict[str, Any]], pr_metadata: Dict[int, Dict[str, Any]]
) -> Tuple[List[PRStats], Dict[int, int]]:
    """Match workflow runs to PRs and compute per-PR statistics.

    Returns (pr_stats_list, run_to_pr) where run_to_pr maps run_id → PR number.

    Matching strategies (all three are unioned, deduped by run ID):
      1. head_branch == PR's head.ref — works for same-repo PRs.
      2. head_branch == "pull-request/{N}" — reliable for fork PRs.
      3. run.pr_numbers contains N — GitHub's direct link (~39% coverage).
    """
    runs_by_branch: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    runs_by_pr_num: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    _pr_branch_re = re.compile(r"^pull-request/(\d+)$")
    for run in pr_runs:
        branch = run.get("head_branch", "")
        if branch:
            runs_by_branch[branch].append(run)
            m = _pr_branch_re.match(branch)
            if m:
                runs_by_pr_num[int(m.group(1))].append(run)
        for n in run.get("pr_numbers") or []:
            runs_by_pr_num[int(n)].append(run)

    run_to_pr: Dict[int, int] = {}
    result: List[PRStats] = []
    for prn, pr_dict in pr_metadata.items():
        user = pr_dict.get("user") or {}
        head = pr_dict.get("head") or {}
        pr_branch = str(head.get("ref") or "")

        seen_ids: set = set()
        runs: List[Dict[str, Any]] = []
        for r in runs_by_branch.get(pr_branch, []):
            if r["id"] not in seen_ids:
                runs.append(r)
                seen_ids.add(r["id"])
        for r in runs_by_pr_num.get(prn, []):
            if r["id"] not in seen_ids:
                runs.append(r)
                seen_ids.add(r["id"])

        for r in runs:
            run_to_pr[int(r["id"])] = prn

        ps = PRStats(
            pr_number=prn,
            author=str(user.get("login") or "unknown"),
            title=str(pr_dict.get("title") or ""),
            created_at=str(pr_dict.get("created_at") or ""),
            merged_at=pr_dict.get("merged_at"),
            is_merged=pr_dict.get("merged_at") is not None,
            head_sha=str(head.get("sha") or ""),
            total_ci_runs=len(runs),
        )

        if ps.is_merged and ps.merged_at and ps.created_at:
            created_dt = _parse_iso(ps.created_at)
            merged_dt = _parse_iso(ps.merged_at)
            if created_dt and merged_dt:
                ps.hours_to_merge = (merged_dt - created_dt).total_seconds() / 3600.0

        shas = {r["head_sha"] for r in runs if r.get("head_sha")}
        ps.ci_trigger_count = len(shas)
        ps.ci_retrigger_count = max(0, ps.ci_trigger_count - 1)
        ps.manual_retrigger_ci_runs = sum(max(0, int(r.get("run_attempt", 1)) - 1) for r in runs)
        ps.failure_count = sum(1 for r in runs if r.get("conclusion") == "failure")
        ps.success_count = sum(1 for r in runs if r.get("conclusion") == "success")
        ps.cancelled_count = sum(1 for r in runs if r.get("conclusion") == "cancelled")
        ps.had_failure = ps.failure_count > 0
        result.append(ps)
    return result, run_to_pr


def _bucket_key(dt: datetime, bucket_size: str) -> Tuple[str, str]:
    if bucket_size == "day":
        d = _date_str(dt)
        return d, d
    if bucket_size == "week":
        iso_year, iso_week, _ = dt.isocalendar()
        label = f"{iso_year}-W{iso_week:02d}"
        monday = dt - timedelta(days=dt.weekday())
        return label, _date_str(monday)
    if bucket_size == "month":
        label = dt.strftime("%Y-%m")
        start = dt.replace(day=1)
        return label, _date_str(start)
    raise ValueError(f"Unknown bucket_size: {bucket_size}")


def bucket_stats(
    pr_metadata: Dict[int, Dict[str, Any]],
    pr_runs: List[Dict[str, Any]],
    main_runs: List[Dict[str, Any]],
    bucket_size: str,
    run_to_pr: Dict[int, int],
    since: datetime,
    pr_is_code: Optional[Dict[int, bool]] = None,
) -> List[BucketStats]:
    """Bucket metrics by **activity date** — the date something happened.

    - PRs opened: bucketed by ``created_at``
    - PRs merged: bucketed by ``merged_at`` (not the PR's creation day)
    - CI runs / failures / re-triggers: bucketed by the run's ``created_at``
    - Main (post-merge) runs: bucketed by the run's ``created_at``

    Only buckets on or after *since* are included.
    """
    since_label, _ = _bucket_key(since, bucket_size)
    all_labels: set = set()
    bucket_starts: Dict[str, str] = {}

    def _reg(label: str, start: str) -> None:
        if label >= since_label:
            all_labels.add(label)
            bucket_starts[label] = start

    prs_opened_by_bucket: Dict[str, List[Dict]] = defaultdict(list)
    for pr in pr_metadata.values():
        dt = _parse_iso(pr.get("created_at", ""))
        if not dt:
            continue
        label, start = _bucket_key(dt, bucket_size)
        prs_opened_by_bucket[label].append(pr)
        _reg(label, start)

    prs_merged_by_bucket: Dict[str, List[Dict]] = defaultdict(list)
    for pr in pr_metadata.values():
        merged_at = pr.get("merged_at")
        if not merged_at:
            continue
        dt = _parse_iso(merged_at)
        if not dt:
            continue
        label, start = _bucket_key(dt, bucket_size)
        prs_merged_by_bucket[label].append(pr)
        _reg(label, start)

    pr_runs_by_bucket: Dict[str, List[Dict]] = defaultdict(list)
    for run in pr_runs:
        dt = _parse_iso(run.get("created_at", ""))
        if not dt:
            continue
        label, start = _bucket_key(dt, bucket_size)
        pr_runs_by_bucket[label].append(run)
        _reg(label, start)

    main_runs_by_bucket: Dict[str, List[Dict]] = defaultdict(list)
    for run in main_runs:
        dt = _parse_iso(run.get("created_at", ""))
        if not dt:
            continue
        label, start = _bucket_key(dt, bucket_size)
        main_runs_by_bucket[label].append(run)
        _reg(label, start)

    result: List[BucketStats] = []
    for label in sorted(all_labels):
        bs = BucketStats(bucket_label=label, bucket_start=bucket_starts.get(label, ""))

        opened = prs_opened_by_bucket.get(label, [])
        bs.prs_opened = len(opened)
        authors: Dict[str, int] = defaultdict(int)
        for pr in opened:
            user = pr.get("user") or {}
            authors[str(user.get("login") or "unknown")] += 1
        bs.unique_authors = len(authors)
        bs.author_counts = dict(authors)

        merged = prs_merged_by_bucket.get(label, [])
        bs.prs_merged = len(merged)
        merge_hours = []
        for pr in merged:
            c_dt = _parse_iso(pr.get("created_at", ""))
            m_dt = _parse_iso(pr.get("merged_at", ""))
            if c_dt and m_dt:
                merge_hours.append((m_dt - c_dt).total_seconds() / 3600.0)
        bs.avg_hours_to_merge = statistics.mean(merge_hours) if merge_hours else None

        bucket_pr_runs = pr_runs_by_bucket.get(label, [])
        bs.total_ci_runs = len(bucket_pr_runs)

        shas: set = set()
        active_prs: set = set()
        pr_has_failure: set = set()
        for r in bucket_pr_runs:
            sha = r.get("head_sha")
            if sha:
                shas.add(sha)
            prn = run_to_pr.get(int(r.get("id", 0)))
            if prn is not None:
                active_prs.add(prn)
                if r.get("conclusion") == "failure":
                    pr_has_failure.add(prn)
        bs.total_ci_triggers = len(shas)
        bs.prs_with_ci = len(active_prs)
        bs.prs_with_ci_failure = len(pr_has_failure)
        bs.pr_ci_failure_rate = bs.prs_with_ci_failure / bs.prs_with_ci if bs.prs_with_ci > 0 else 0.0
        if bs.prs_with_ci > 0:
            bs.mean_ci_triggers_per_pr = bs.total_ci_triggers / bs.prs_with_ci

        retrigger_total = 0
        prs_with_retrigger: set = set()
        retrigger_per_pr: Dict[int, int] = defaultdict(int)
        for r in bucket_pr_runs:
            attempt = int(r.get("run_attempt", 1))
            if attempt > 1:
                extra = attempt - 1
                retrigger_total += extra
                prn = run_to_pr.get(int(r.get("id", 0)))
                if prn is not None:
                    prs_with_retrigger.add(prn)
                    retrigger_per_pr[prn] += extra
        bs.manual_retrigger_ci_runs = retrigger_total
        bs.prs_with_manual_retrigger = len(prs_with_retrigger)
        if retrigger_per_pr:
            worst_prn = max(retrigger_per_pr, key=retrigger_per_pr.get)
            bs.worst_retrigger_pr = (worst_prn, retrigger_per_pr[worst_prn])

        bucket_main_runs = main_runs_by_bucket.get(label, [])
        main_sha_has_failure: set = set()
        main_shas: set = set()
        for r in bucket_main_runs:
            sha = r.get("head_sha")
            if sha:
                main_shas.add(sha)
                if r.get("conclusion") == "failure":
                    main_sha_has_failure.add(sha)
        bs.main_pushes_total = len(main_shas)
        bs.main_pushes_failed = len(main_sha_has_failure)
        bs.main_failure_rate = bs.main_pushes_failed / bs.main_pushes_total if bs.main_pushes_total > 0 else 0.0

        bs.top_failing_workflows = compute_top_failing_workflows(bucket_pr_runs + bucket_main_runs)

        if pr_is_code is not None:
            for is_code_val, split in [(True, bs.code), (False, bs.noncode)]:
                split_prs = [pr for pr in opened if pr_is_code.get(pr.get("number"), False) == is_code_val]
                split.prs_opened = len(split_prs)
                split_authors: set = set()
                for pr in split_prs:
                    u = pr.get("user") or {}
                    split_authors.add(str(u.get("login") or "unknown"))
                split.unique_authors = len(split_authors)

                split_merged = [pr for pr in merged if pr_is_code.get(pr.get("number"), False) == is_code_val]
                split.prs_merged = len(split_merged)
                split_mh: List[float] = []
                for pr in split_merged:
                    c = _parse_iso(pr.get("created_at", ""))
                    m = _parse_iso(pr.get("merged_at", ""))
                    if c and m:
                        split_mh.append((m - c).total_seconds() / 3600.0)
                split.avg_hours_to_merge = statistics.mean(split_mh) if split_mh else None

                split_shas: set = set()
                split_active: set = set()
                split_fail: set = set()
                split_retrig_total = 0
                split_retrig_prs: set = set()
                split_retrig_per_pr: Dict[int, int] = defaultdict(int)
                for r in bucket_pr_runs:
                    prn = run_to_pr.get(int(r.get("id", 0)))
                    if prn is None or pr_is_code.get(prn) != is_code_val:
                        continue
                    sha = r.get("head_sha")
                    if sha:
                        split_shas.add(sha)
                    split_active.add(prn)
                    if r.get("conclusion") == "failure":
                        split_fail.add(prn)
                    attempt = int(r.get("run_attempt", 1))
                    if attempt > 1:
                        extra = attempt - 1
                        split_retrig_total += extra
                        split_retrig_prs.add(prn)
                        split_retrig_per_pr[prn] += extra

                split.total_ci_triggers = len(split_shas)
                split.prs_with_ci = len(split_active)
                split.prs_with_ci_failure = len(split_fail)
                split.pr_ci_failure_rate = split.prs_with_ci_failure / split.prs_with_ci if split.prs_with_ci > 0 else 0.0
                split.mean_ci_triggers_per_pr = split.total_ci_triggers / split.prs_with_ci if split.prs_with_ci > 0 else None
                split.manual_retrigger_ci_runs = split_retrig_total
                split.prs_with_manual_retrigger = len(split_retrig_prs)
                if split_retrig_per_pr:
                    wp = max(split_retrig_per_pr, key=split_retrig_per_pr.get)
                    split.worst_retrigger_pr = (wp, split_retrig_per_pr[wp])

        result.append(bs)
    return result


def compute_top_failing_workflows(runs: List[Dict[str, Any]], top_n: int = 5) -> List[Tuple[str, int]]:
    fail_counts: Dict[str, int] = defaultdict(int)
    for run in runs:
        if run.get("conclusion") == "failure":
            fail_counts[str(run.get("name") or "unknown")] += 1
    return sorted(fail_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]


def compute_overall_stats(
    pr_metadata: Dict[int, Dict[str, Any]],
    pr_runs: List[Dict[str, Any]],
    main_runs: List[Dict[str, Any]],
    run_to_pr: Dict[int, int],
    days: int,
    bucket_size: str,
    repo: str,
    since: datetime,
    pr_is_code: Optional[Dict[int, bool]] = None,
) -> OverallStats:
    """Compute window-wide aggregate stats using activity dates."""
    ov = OverallStats(days=days, bucket_size=bucket_size, repo=repo)

    # PRs opened (created in window)
    opened_authors: Dict[str, int] = defaultdict(int)
    for pr in pr_metadata.values():
        created = _parse_iso(pr.get("created_at", ""))
        if created and created >= since:
            ov.total_prs_opened += 1
            user = pr.get("user") or {}
            opened_authors[str(user.get("login") or "unknown")] += 1
    ov.unique_authors = len(opened_authors)
    ov.top_authors = sorted(opened_authors.items(), key=lambda x: x[1], reverse=True)[:10]

    # PRs merged (merged_at in window)
    merge_hours: List[float] = []
    for pr in pr_metadata.values():
        merged_at = pr.get("merged_at")
        if not merged_at:
            continue
        m_dt = _parse_iso(merged_at)
        if not m_dt or m_dt < since:
            continue
        ov.total_prs_merged += 1
        c_dt = _parse_iso(pr.get("created_at", ""))
        if c_dt and m_dt:
            merge_hours.append((m_dt - c_dt).total_seconds() / 3600.0)
    ov.avg_hours_to_merge = statistics.mean(merge_hours) if merge_hours else None

    # CI triggers / runs / per-PR failure rate
    shas: set = set()
    active_prs: set = set()
    pr_has_failure: set = set()
    for r in pr_runs:
        sha = r.get("head_sha")
        if sha:
            shas.add(sha)
        prn = run_to_pr.get(r["id"])
        if prn is not None:
            active_prs.add(prn)
            if r.get("conclusion") == "failure":
                pr_has_failure.add(prn)
    ov.total_ci_triggers = len(shas)
    ov.prs_with_ci = len(active_prs)
    ov.avg_ci_triggers_per_pr = ov.total_ci_triggers / ov.prs_with_ci if ov.prs_with_ci > 0 else 0.0
    ov.total_ci_runs = len(pr_runs)
    ov.prs_with_ci_failure = len(pr_has_failure)
    ov.pr_ci_failure_rate = ov.prs_with_ci_failure / ov.prs_with_ci if ov.prs_with_ci > 0 else 0.0

    # Manual re-triggers
    retrigger_per_pr: Dict[int, int] = defaultdict(int)
    for r in pr_runs:
        attempt = int(r.get("run_attempt", 1))
        if attempt > 1:
            prn = run_to_pr.get(r["id"])
            if prn is not None:
                retrigger_per_pr[prn] += attempt - 1
    ov.total_manual_retrigger_ci_runs = sum(retrigger_per_pr.values())
    ov.prs_with_manual_retrigger = len(retrigger_per_pr)
    if retrigger_per_pr:
        worst_prn = max(retrigger_per_pr, key=retrigger_per_pr.get)
        ov.worst_retrigger_pr = (worst_prn, retrigger_per_pr[worst_prn])

    # Main (post-merge) failure rate — per push (per distinct head_sha)
    main_shas: set = set()
    main_sha_fail: set = set()
    for r in main_runs:
        sha = r.get("head_sha")
        if sha:
            main_shas.add(sha)
            if r.get("conclusion") == "failure":
                main_sha_fail.add(sha)
    ov.main_pushes_total = len(main_shas)
    ov.main_pushes_failed = len(main_sha_fail)
    ov.main_failure_rate = ov.main_pushes_failed / ov.main_pushes_total if ov.main_pushes_total > 0 else 0.0
    ov.top_failing_workflows = compute_top_failing_workflows(pr_runs + main_runs, top_n=10)

    if pr_is_code is not None:
        for is_code_val, split in [(True, ov.code), (False, ov.noncode)]:
            split_mh: List[float] = []
            split_authors: Dict[str, int] = defaultdict(int)
            for pr in pr_metadata.values():
                prn = pr.get("number")
                if prn is None or pr_is_code.get(prn, False) != is_code_val:
                    continue
                created = _parse_iso(pr.get("created_at", ""))
                if created and created >= since:
                    split.prs_opened += 1
                    user = pr.get("user") or {}
                    split_authors[str(user.get("login") or "unknown")] += 1
                merged_at = pr.get("merged_at")
                if merged_at:
                    m_dt = _parse_iso(merged_at)
                    if m_dt and m_dt >= since:
                        split.prs_merged += 1
                        c_dt = _parse_iso(pr.get("created_at", ""))
                        if c_dt:
                            split_mh.append((m_dt - c_dt).total_seconds() / 3600.0)
            split.unique_authors = len(split_authors)
            split.avg_hours_to_merge = statistics.mean(split_mh) if split_mh else None

            split_shas: set = set()
            split_active: set = set()
            split_fail: set = set()
            split_retrig_per_pr: Dict[int, int] = defaultdict(int)
            for r in pr_runs:
                prn = run_to_pr.get(r["id"])
                if prn is None or pr_is_code.get(prn) != is_code_val:
                    continue
                sha = r.get("head_sha")
                if sha:
                    split_shas.add(sha)
                split_active.add(prn)
                if r.get("conclusion") == "failure":
                    split_fail.add(prn)
                attempt = int(r.get("run_attempt", 1))
                if attempt > 1:
                    split_retrig_per_pr[prn] += attempt - 1
            split.total_ci_triggers = len(split_shas)
            split.prs_with_ci = len(split_active)
            split.mean_ci_triggers_per_pr = split.total_ci_triggers / split.prs_with_ci if split.prs_with_ci > 0 else None
            split.prs_with_ci_failure = len(split_fail)
            split.pr_ci_failure_rate = split.prs_with_ci_failure / split.prs_with_ci if split.prs_with_ci > 0 else 0.0
            split.manual_retrigger_ci_runs = sum(split_retrig_per_pr.values())
            split.prs_with_manual_retrigger = len(split_retrig_per_pr)
            if split_retrig_per_pr:
                wp = max(split_retrig_per_pr, key=split_retrig_per_pr.get)
                split.worst_retrigger_pr = (wp, split_retrig_per_pr[wp])

    return ov


def _format_split_lines(tag: str, s: SplitStats) -> List[str]:
    """Format [code] or [non-code] lines for a SplitStats."""
    lines: List[str] = []
    ttm = f", avg time-to-merge: {s.avg_hours_to_merge:.1f}h" if s.avg_hours_to_merge is not None else ""
    lines.append(f"  [{tag}] New PRs: {s.prs_opened}, merged: {s.prs_merged}, authors: {s.unique_authors}{ttm}")
    ci_trig = f"{s.mean_ci_triggers_per_pr:.1f}" if s.mean_ci_triggers_per_pr is not None else "-"
    lines.append(f"  [{tag}] User pushes: {s.total_ci_triggers} across {s.prs_with_ci} PRs = {ci_trig}/PR")
    if s.prs_with_manual_retrigger > 0:
        mr_per_pr = f"{s.manual_retrigger_ci_runs / s.prs_with_manual_retrigger:.1f}"
        pct = f"{s.prs_with_manual_retrigger / s.prs_with_ci:.0%}" if s.prs_with_ci > 0 else "?"
        worst = ""
        if s.worst_retrigger_pr:
            worst = f", worst: #{s.worst_retrigger_pr[0]} ({s.worst_retrigger_pr[1]}x)"
        lines.append(f"  [{tag}] Manual re-triggers: {s.manual_retrigger_ci_runs} across {s.prs_with_manual_retrigger}/{s.prs_with_ci} PRs ({pct}) = {mr_per_pr}/PR{worst}")
    else:
        lines.append(f"  [{tag}] Manual re-triggers: 0")
    lines.append(f"  [{tag}] Failures: {s.prs_with_ci_failure}/{s.prs_with_ci} PRs ({s.pr_ci_failure_rate:.0%})")
    return lines


def format_terminal_report(buckets: List[BucketStats], overall: OverallStats) -> str:
    lines: List[str] = []
    bucket_label = "daily" if overall.bucket_size == "day" else ("weekly" if overall.bucket_size == "week" else "monthly")
    lines.append(f"PR CI Report: {overall.repo} (last {overall.days} days, {bucket_label})")
    lines.append("=" * 70)
    lines.append("")
    for bs in buckets:
        prefix = "Week of" if overall.bucket_size == "week" else ("Month" if overall.bucket_size == "month" else "")
        if overall.bucket_size == "day":
            dt = _parse_iso(bs.bucket_label + "T00:00:00+00:00")
            day_name = dt.strftime("%a") if dt else ""
            heading = f"{bs.bucket_label} {day_name}:"
        else:
            heading = f"{prefix} {bs.bucket_start}:".strip()
        lines.append(heading)
        lines.extend(_format_split_lines("code", bs.code))
        lines.extend(_format_split_lines("non-code", bs.noncode))
        if bs.top_failing_workflows:
            lines.append("  Top failing: " + ", ".join(f"{n} ({c})" for n, c in bs.top_failing_workflows[:3]))
        lines.append("")
    lines.append("-" * 70)
    lines.append(f"Overall ({overall.days} days):")
    lines.extend(_format_split_lines("code", overall.code))
    lines.extend(_format_split_lines("non-code", overall.noncode))
    if overall.top_authors:
        lines.append("  Most active: " + ", ".join(f"{a} ({c})" for a, c in overall.top_authors[:5]))
    if overall.top_failing_workflows:
        lines.append("  Top failing workflows:")
        for name, count in overall.top_failing_workflows[:5]:
            lines.append(f"    {name}: {count} failures")
    lines.append("")
    return "\n".join(lines)


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def format_html_report(buckets: List[BucketStats], overall: OverallStats, pr_stats_list: List[PRStats]) -> str:
    bucket_rows = []
    for bs in buckets:
        top_fail = ", ".join(f"{n} ({c})" for n, c in bs.top_failing_workflows[:3])
        ci_trig_str = f"{bs.mean_ci_triggers_per_pr:.1f}" if bs.mean_ci_triggers_per_pr is not None else "-"
        bucket_rows.append(
            "<tr>"
            f"<td>{bs.bucket_label}</td><td>{bs.prs_opened}</td><td>{bs.prs_merged}</td><td>{bs.unique_authors}</td>"
            f"<td>{bs.total_ci_triggers}</td><td>{ci_trig_str}</td>"
            f"<td>{bs.total_ci_runs}</td><td>{bs.manual_retrigger_ci_runs}</td>"
            f"<td>{bs.pr_ci_failure_rate:.0%}</td><td>{bs.main_failure_rate:.0%}</td><td>{top_fail or '-'}</td>"
            "</tr>"
        )
    pr_rows = []
    for ps in sorted(pr_stats_list, key=lambda p: p.created_at, reverse=True)[:200]:
        pr_rows.append(
            "<tr>"
            f"<td>#{ps.pr_number}</td><td>{_html_escape(ps.author)}</td>"
            f"<td title=\"{_html_escape(ps.title)}\">{_html_escape(ps.title[:60])}</td>"
            f"<td>{'merged' if ps.is_merged else 'open'}</td><td>{ps.ci_trigger_count}</td>"
            f"<td>{ps.total_ci_runs}</td><td>{ps.manual_retrigger_ci_runs}</td>"
            f"<td>{ps.failure_count}</td><td>{ps.success_count}</td>"
            "</tr>"
        )
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>PR CI Report: {_html_escape(overall.repo)}</title>"
        "<style>body{font-family:system-ui;margin:2em}table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #ccc;padding:6px 10px}th{background:#eee}</style></head><body>"
        f"<h1>PR CI Report: {_html_escape(overall.repo)}</h1>"
        f"<p>Last {overall.days} days, grouped by {overall.bucket_size}</p>"
        "<h2>Summary</h2><ul>"
        f"<li>Total PRs opened: {overall.total_prs_opened}</li>"
        f"<li>Total PRs merged: {overall.total_prs_merged}</li>"
        f"<li>Unique authors: {overall.unique_authors}</li>"
        f"<li>CI triggers: {overall.total_ci_triggers} across {overall.prs_with_ci} PRs = {overall.avg_ci_triggers_per_pr:.1f}/PR</li>"
        f"<li>Avg time-to-merge: {f'{overall.avg_hours_to_merge:.1f}h' if overall.avg_hours_to_merge is not None else 'N/A'}</li>"
        f"<li>CI runs: {overall.total_ci_runs} total, {overall.total_manual_retrigger_ci_runs} manual re-triggers</li>"
        f"<li>PR CI failure rate: {overall.pr_ci_failure_rate:.0%} ({overall.prs_with_ci_failure}/{overall.prs_with_ci} PRs)</li>"
        f"<li>Main failure rate: {overall.main_failure_rate:.0%} ({overall.main_pushes_failed}/{overall.main_pushes_total} merges)</li>"
        "</ul><h2>Buckets</h2><table><tr><th>Bucket</th><th>PRs opened</th><th>Merged</th>"
        "<th>Authors</th><th>CI triggers</th><th>CI triggers/PR</th>"
        "<th>CI runs</th><th>Manual re-triggers</th><th>PR CI fail</th><th>Main fail</th><th>Top failing</th></tr>"
        + "".join(bucket_rows)
        + "</table><h2>Per-PR Detail</h2><table><tr><th>PR</th><th>Author</th><th>Title</th><th>Status</th>"
        "<th>CI triggers</th><th>CI runs</th><th>Manual re-triggers</th><th>Failures</th><th>Successes</th></tr>"
        + "".join(pr_rows)
        + "</table></body></html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PR CI stats report (day/week/month).")
    parser.add_argument("--repo", required=True, help="owner/repo (e.g. ai-dynamo/dynamo)")
    parser.add_argument("--days", type=int, default=30, help="Lookback window in days")
    parser.add_argument("--bucket", choices=["day", "week", "month"], default="week")
    parser.add_argument("--output-dir", type=Path, help="Auto-generate txt/json/html in this dir (filenames include --days)")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-html", type=Path)
    parser.add_argument("--max-github-api-calls", type=int, default=500)
    parser.add_argument("--github-token")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING)
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    parts = args.repo.split("/")
    if len(parts) != 2:
        parser.error(f"--repo must be owner/repo, got {args.repo}")
    owner, repo = parts
    api = GitHubAPIClient(token=args.github_token, max_rest_calls=args.max_github_api_calls, debug_rest=args.debug)
    since = datetime.now(timezone.utc) - timedelta(days=args.days)

    pr_metadata = collect_pr_metadata(api, owner, repo, since, days=args.days)

    # Collect all three event types that carry PR-related CI runs.
    # For ai-dynamo/dynamo the breakdown is roughly:
    #   pull_request        33% — standard pre-merge CI (restricted perms, no secrets)
    #   pull_request_target 22% — pre-merge CI needing secrets (labels, lint, DCO)
    #   push                44% — split into two populations (see below)
    pr_runs = collect_workflow_runs(api, owner, repo, event="pull_request", since=since)
    prt_runs = collect_workflow_runs(api, owner, repo, event="pull_request_target", since=since)
    seen_ids = {r["id"] for r in pr_runs}
    for r in prt_runs:
        if r["id"] not in seen_ids:
            pr_runs.append(r)
            seen_ids.add(r["id"])

    # Push runs have two sub-populations that must be classified separately:
    #   head_branch="pull-request/{N}" (65% of push runs) — fork-PR CI.  A merge
    #     bot pushes the PR's code to a repo-owned branch so CI can run with full
    #     secrets.  These are PRE-merge and must be counted with pr_runs.
    #   head_branch="main" (25%) — real post-merge CI after a PR is merged.
    #   head_branch=other (10%) — release branches, etc. (ignored for now).
    push_runs = collect_workflow_runs(api, owner, repo, event="push", since=since)
    _PR_BRANCH_RE = re.compile(r"^pull-request/(\d+)$")
    main_runs: List[Dict[str, Any]] = []
    for r in push_runs:
        m = _PR_BRANCH_RE.match(r.get("head_branch", ""))
        if m:
            if r["id"] not in seen_ids:
                pr_runs.append(r)
                seen_ids.add(r["id"])
        elif r.get("head_branch") == "main":
            main_runs.append(r)

    pr_stats_list, run_to_pr = build_pr_stats(pr_runs, pr_metadata)
    pr_is_code = classify_pr_is_code(pr_runs, run_to_pr)
    buckets = bucket_stats(pr_metadata, pr_runs, main_runs, args.bucket, run_to_pr, since, pr_is_code=pr_is_code)
    overall = compute_overall_stats(pr_metadata, pr_runs, main_runs, run_to_pr, args.days, args.bucket, args.repo, since, pr_is_code=pr_is_code)

    terminal_text = format_terminal_report(buckets, overall)
    print(terminal_text)

    api_total = int(GITHUB_API_STATS.rest_calls_total)
    etag_304 = int(GITHUB_API_STATS.etag_304_total)
    billable = api_total - etag_304
    cache_hits = sum(int(v) for v in (GITHUB_API_STATS.cache_hits or {}).values())
    api_line = f"GitHub API: {billable} billable, {etag_304} ETag-304, {cache_hits} cached"
    print(api_line)

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        txt_path = args.output_dir / f"pr_ci_report_{args.days}d.txt"
        txt_path.write_text(terminal_text + "\n" + api_line + "\n")
        print(f"Text report saved to: {txt_path}")

    payload = {
        "overall": asdict(overall),
        "buckets": [asdict(b) for b in buckets],
        "pr_details": [asdict(p) for p in sorted(pr_stats_list, key=lambda x: x.created_at, reverse=True)],
    }
    json_path = args.output_json
    if not json_path and args.output_dir:
        json_path = args.output_dir / f"pr_ci_report_{args.days}d.json"
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2))
        print(f"JSON report saved to: {json_path}")

    html_path = args.output_html
    if not html_path and args.output_dir:
        html_path = args.output_dir / f"pr_ci_report_{args.days}d.html"
    if html_path:
        html = format_html_report(buckets, overall, pr_stats_list)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(html)
        print(f"HTML report saved to: {html_path}")


if __name__ == "__main__":
    main()
