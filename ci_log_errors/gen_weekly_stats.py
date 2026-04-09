#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate weekly post-merge CI stats files.

Usage:
    python3 ci_log_errors/gen_weekly_stats.py --week 2026-03-09
    python3 ci_log_errors/gen_weekly_stats.py --week 2026-03-09 --week 2026-03-16
"""
import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import time as _time

import requests as _requests

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common_github import GitHubAPIClient
from common_github.api.actions_runs_list_cached import list_workflow_runs
from common_github.api.pulls_list_cached import list_pull_requests_cached
from ci_log_errors.engine import categorize_error_log_lines


def _gh_token() -> str:
    """Read GitHub token from gh CLI config."""
    gh_hosts = Path.home() / ".config" / "gh" / "hosts.yml"
    if gh_hosts.exists():
        for line in gh_hosts.read_text().splitlines():
            if "oauth_token" in line:
                return line.split(":", 1)[1].strip()
    return os.environ.get("GH_TOKEN", os.environ.get("GITHUB_TOKEN", ""))


def _gh_get(url: str, **kwargs: Any) -> _requests.Response:
    headers = {"Authorization": f"token {_gh_token()}", "Accept": "application/vnd.github.v3+json"}
    headers.update(kwargs.pop("headers", {}))
    return _requests.get(url, headers=headers, timeout=kwargs.pop("timeout", 10), **kwargs)


OWNER = "ai-dynamo"
REPO = "dynamo"

BUCKET_MAP: Dict[str, str] = {
    "pytest-error": "Testing",
    "python-error": "Testing",
    "pytest-timeout-error": "Testing",
    "rust-error": "Testing",
    "ci-filter-coverage-error": "Testing",
    "backend-failure": "Infra",
    "huggingface-auth-error": "Infra",
    "exit-127-cmd-not-found": "Build",
    "network-error": "Infra",
    "network-timeout-generic": "Infra",
    "network-timeout-https": "Infra",
    "network-timeout-gitlab-mirror": "Infra",
    "network-timeout-github-action": "Infra",
    "network-download-error": "Infra",
    "network-port-conflict-error": "Infra",
    "k8s-error": "Infra",
    "k8s-network-timeout-pod": "Infra",
    "etcd-error": "Infra",
    "disk-space-error": "Infra",
    "oom": "Infra",
    "helm-error": "Build",
    "docker-daemon-error-response": "Build",
    "docker-build-error": "Build",
    "docker-upload-error": "Build",
    "docker-cli-error": "Build",
    "auth-token-expired": "Auth",
    "ci-status-check-error": "Gates",
    "invalid-task-type": "Testing",
}

GROUP_MAP: Dict[str, str] = {
    "pytest-error": "Tests",
    "python-error": "Tests",
    "pytest-timeout-error": "Tests",
    "rust-error": "Tests",
    "backend-failure": "Tests",
    "ci-filter-coverage-error": "Tests",
    "docker-build-error": "Docker / build",
    "docker-daemon-error-response": "Docker / build",
    "docker-upload-error": "Docker / build",
    "docker-cli-error": "Docker / build",
    "helm-error": "K8s / Helm",
    "k8s-error": "K8s / Helm",
    "k8s-network-timeout-pod": "K8s / Helm",
    "timeout-exit-124": "K8s / Helm",
    "network-error": "Network",
    "network-timeout-generic": "Network",
    "network-timeout-https": "Network",
    "network-timeout-gitlab-mirror": "Network",
    "network-timeout-github-action": "Network",
    "network-download-error": "Network",
    "network-port-conflict-error": "Network",
    "etcd-error": "Infra / system",
    "disk-space-error": "Infra / system",
    "oom": "Infra / system",
    "exit-127-cmd-not-found": "Infra / system",
    "github-action-unavailable": "Infra / system",
    "huggingface-auth-error": "Auth",
    "auth-token-expired": "Auth",
    "ci-status-check-error": "Gates / policy",
    "deploy-test-status-check": "Gates / policy",
    "invalid-task-type": "Gates / policy",
    "broken-links": "Docs",
    "go-operator-lint-error": "Build",
}

CATCH_ALLS = {"exit-139-sigsegv", "vllm-error", "sglang-error", "trtllm-error"}

_CACHE_DIR = Path.home() / ".cache" / "dynamo-utils"
_JOBS_CACHE_PATH = _CACHE_DIR / "weekly_stats_jobs.json"
_SCAN_CACHE_PATH = _CACHE_DIR / "weekly_stats_scan.json"


def _load_json_cache(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_json_cache(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _get_raw_log_dir() -> Path:
    override = os.environ.get("DYNAMO_UTILS_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser() / "raw-log-text"
    return Path.home() / ".cache" / "dynamo-utils" / "raw-log-text"


def _read_log_lines(job_id: str) -> Optional[List[str]]:
    log_path = _get_raw_log_dir() / f"{job_id}.log"
    if not log_path.exists():
        return None
    try:
        data = log_path.read_bytes()
        tail = data[-512 * 1024:] if len(data) > 512 * 1024 else data
        return tail.decode("utf-8", errors="replace").splitlines()
    except OSError:
        return None


_PR_BRANCH_RE = re.compile(r"^pull-request/(\d+)$")


def _parse_iso(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _compute_dev_metrics(
    api: GitHubAPIClient, week_start: date, week_end: date,
    post_merge_runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute Development Metrics (PRs, merge time, re-triggers, etc.) for a week.

    Uses PR metadata (fetched here) + pre-merge CI runs (pull_request +
    pull_request_target events, fetched here) + post_merge_runs (passed in
    from the caller to avoid a redundant push-event fetch).
    """
    since = datetime(week_start.year, week_start.month, week_start.day, tzinfo=timezone.utc)
    until = datetime(week_end.year, week_end.month, week_end.day, 23, 59, 59, tzinfo=timezone.utc)

    # PR metadata — need enough pages to reach back to week_start from today.
    weeks_back = max(1, (date.today() - week_start).days // 7 + 1)
    pages_needed = max(10, weeks_back * 3)
    print(f"    Fetching PR list (max_pages={pages_needed})...", flush=True)
    all_prs = list_pull_requests_cached(api, owner=OWNER, repo=REPO, state="all", ttl_s=3600, max_pages=pages_needed)
    print(f"    Got {len(all_prs)} PRs from cache/API", flush=True)
    pr_metadata: Dict[int, Dict[str, Any]] = {}
    for pr in all_prs:
        if not isinstance(pr, dict):
            continue
        created = _parse_iso(pr.get("created_at", ""))
        merged = _parse_iso(pr.get("merged_at") or "")
        if (created and created >= since) or (merged and merged >= since):
            number = pr.get("number")
            if number is not None:
                pr_metadata[int(number)] = pr

    # Pre-merge CI runs (pull_request + pull_request_target events only).
    # Skips the expensive push-event fetch; fork PR CI (push/pull-request/N)
    # is excluded but pull_request + pull_request_target cover the majority.
    since_str = since.strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"    Fetching pre-merge CI runs...", flush=True)
    pr_runs_raw = _retry(lambda: list_workflow_runs(api, owner=OWNER, repo=REPO, event="pull_request", since_date=since_str, ttl_s=600))
    prt_runs = _retry(lambda: list_workflow_runs(api, owner=OWNER, repo=REPO, event="pull_request_target", since_date=since_str, ttl_s=600))
    seen_ids = {r["id"] for r in pr_runs_raw}
    for r in prt_runs:
        if r["id"] not in seen_ids:
            pr_runs_raw.append(r)
            seen_ids.add(r["id"])

    # Filter runs to the week window
    pr_runs: List[Dict[str, Any]] = []
    for r in pr_runs_raw:
        dt = _parse_iso(r.get("created_at", ""))
        if dt and since <= dt <= until:
            pr_runs.append(r)

    # Match runs to PRs
    run_to_pr: Dict[int, int] = {}
    for r in pr_runs:
        pr_nums = r.get("pr_numbers") or []
        if pr_nums:
            run_to_pr[r["id"]] = pr_nums[0]

    # PRs submitted (created in window)
    prs_opened = 0
    for pr in pr_metadata.values():
        created = _parse_iso(pr.get("created_at", ""))
        if created and since <= created <= until:
            prs_opened += 1

    # PRs merged (merged_at in window)
    prs_merged = 0
    merged_pr_numbers: List[int] = []
    merge_hours: List[float] = []
    for pr in pr_metadata.values():
        merged_at_str = pr.get("merged_at")
        if not merged_at_str:
            continue
        m_dt = _parse_iso(merged_at_str)
        if not m_dt or m_dt < since or m_dt > until:
            continue
        prs_merged += 1
        merged_pr_numbers.append(int(pr["number"]))
        c_dt = _parse_iso(pr.get("created_at", ""))
        if c_dt and m_dt:
            merge_hours.append((m_dt - c_dt).total_seconds() / 3600.0)
    avg_days_to_merge = statistics.mean(merge_hours) / 24.0 if merge_hours else None

    # Line counts per merged PR (requires individual PR endpoint).
    lines_per_pr: List[int] = []
    print(f"    Fetching line counts for {len(merged_pr_numbers)} merged PRs...", flush=True)
    lc_errors = 0
    for i, prn in enumerate(merged_pr_numbers):
        if (i + 1) % 50 == 0:
            print(f"      {i+1}/{len(merged_pr_numbers)}...", flush=True)
        url = f"https://api.github.com/repos/{OWNER}/{REPO}/pulls/{prn}"
        try:
            resp = _gh_get(url, timeout=15)
            if resp.status_code == 200:
                d = resp.json()
                adds = d.get("additions", 0) or 0
                dels = d.get("deletions", 0) or 0
                lines_per_pr.append(adds + dels)
            elif resp.status_code == 403:
                lc_errors += 1
                if lc_errors == 1:
                    print(f"      Rate limited at PR {i+1}/{len(merged_pr_numbers)}", flush=True)
            else:
                lc_errors += 1
        except _requests.RequestException:
            lc_errors += 1
    if lc_errors:
        print(f"    Line count issues: {lc_errors} errors", flush=True)
    avg_lines = int(statistics.mean(lines_per_pr)) if lines_per_pr else None
    median_lines = int(statistics.median(lines_per_pr)) if lines_per_pr else None

    # Per-PR metrics: count unique SHAs only from runs matched to a PR.
    pr_shas: Dict[int, set] = defaultdict(set)
    pr_has_failure: set = set()
    for r in pr_runs:
        prn = run_to_pr.get(r["id"])
        if prn is None:
            continue
        sha = r.get("head_sha")
        if sha:
            pr_shas[prn].add(sha)
        if r.get("conclusion") == "failure":
            pr_has_failure.add(prn)

    prs_with_ci = len(pr_shas)
    total_ci_triggers = sum(len(s) for s in pr_shas.values())
    pushes_per_pr = total_ci_triggers / prs_with_ci if prs_with_ci > 0 else 0.0

    # PR submission failure rate
    pr_ci_failure_rate = len(pr_has_failure) / prs_with_ci * 100 if prs_with_ci > 0 else 0.0

    # Manual re-triggers: count run_attempt > 1 (standard GitHub Actions metric).
    # Note: undercounts vs pr_ci_report.py because push/pull-request/N events
    # (fork PR CI, where most manual re-runs happen) are excluded to avoid the
    # expensive all-push-events API fetch.
    retrigger_per_pr: Dict[int, int] = defaultdict(int)
    for r in pr_runs:
        attempt = int(r.get("run_attempt", 1))
        if attempt > 1:
            prn = run_to_pr.get(r["id"])
            if prn is not None:
                retrigger_per_pr[prn] += attempt - 1
    total_retriggers = sum(retrigger_per_pr.values())
    prs_with_retrigger = len(retrigger_per_pr)
    retrigger_rate = prs_with_retrigger / prs_with_ci * 100 if prs_with_ci > 0 else 0.0
    retriggers_per_affected = total_retriggers / prs_with_retrigger if prs_with_retrigger > 0 else 0.0

    # CI duration (avg minutes) — from post-merge heavy pipeline runs passed in by caller.
    heavy_post_merge = {"Post-Merge CI Pipeline", "NVIDIA Test Lab Validation", "Docker Build and Test"}
    durations_min: List[float] = []
    for r in post_merge_runs:
        if r.get("name", "") not in heavy_post_merge:
            continue
        started = _parse_iso(r.get("run_started_at", ""))
        updated = _parse_iso(r.get("updated_at", ""))
        if started and updated and updated > started:
            dur = (updated - started).total_seconds() / 60.0
            if dur < 600:
                durations_min.append(dur)
    avg_ci_duration = statistics.mean(durations_min) if durations_min else None

    return {
        "prs_submitted": prs_opened,
        "prs_merged": prs_merged,
        "avg_lines_per_pr": avg_lines,
        "median_lines_per_pr": median_lines,
        "days_to_merge": avg_days_to_merge,
        "pushes_per_pr": pushes_per_pr,
        "retrigger_rate": retrigger_rate,
        "retriggers_per_affected": retriggers_per_affected,
        "pr_failure_rate": pr_ci_failure_rate,
        "ci_duration_min": avg_ci_duration,
    }


def _format_dev_metrics(dm: Dict[str, Any]) -> List[str]:
    """Format the Development Metrics section lines."""
    lines = [
        "## Development Metrics",
        "",
    ]

    lines.append(f"PRs submitted:               {dm['prs_submitted']}")
    lines.append(f"PRs merged:                  {dm['prs_merged']}")

    avg_l = dm.get("avg_lines_per_pr")
    med_l = dm.get("median_lines_per_pr")
    lines.append(f"Avg lines/PR:                {avg_l:,}" if avg_l is not None else "Avg lines/PR:                N/A")
    lines.append(f"Median lines/PR:             {med_l:,}" if med_l is not None else "Median lines/PR:             N/A")

    dtm = dm["days_to_merge"]
    lines.append(f"Days to merge (avg):         {dtm:.1f}" if dtm is not None else "Days to merge (avg):         N/A")

    lines.append(f"Pushes per PR (avg):         {dm['pushes_per_pr']:.1f}")
    lines.append(f"Manual re-trigger rate:      {dm['retrigger_rate']:.0f}%")
    lines.append(f"Re-triggers per affected PR: {dm['retriggers_per_affected']:.1f}")
    lines.append(f"PR submission failure rate:  {dm['pr_failure_rate']:.0f}%")

    dur = dm["ci_duration_min"]
    lines.append(f"CI duration (avg, min):      {dur:.0f}" if dur is not None else "CI duration (avg, min):      N/A")

    return lines


def _retry(fn, *, retries: int = 3, backoff: float = 10.0):
    """Call *fn* with retries on ConnectionError / timeout."""
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except (_requests.ConnectionError, _requests.Timeout, OSError) as e:
            if attempt == retries:
                raise
            wait = backoff * attempt
            print(f"    Network error (attempt {attempt}/{retries}), retrying in {wait:.0f}s: {e}", flush=True)
            _time.sleep(wait)


def _fetch_post_merge_runs(api: GitHubAPIClient, week_start: date, week_end: date) -> List[Dict]:
    since = week_start.isoformat()
    runs = _retry(lambda: list_workflow_runs(
        api, owner=OWNER, repo=REPO, event="push", branch="main", since_date=since,
    ))
    filtered = []
    for r in runs:
        ca = r.get("created_at", "")
        if not ca:
            continue
        dt = datetime.fromisoformat(ca.replace("Z", "+00:00")).date()
        if week_start <= dt <= week_end:
            filtered.append(r)
    return filtered


def _get_failed_job_ids_for_runs(run_ids: List[int]) -> List[str]:
    """Get failed job IDs for a batch of workflow runs.

    Three-tier lookup: local write-back cache -> dashboard actions_jobs.json -> API.
    Results from API are persisted to the local cache so re-runs are free.
    """
    # Tier 1: local write-back cache (run_id -> [failed_job_id_str, ...])
    local_cache = _load_json_cache(_JOBS_CACHE_PATH)
    local_hits = 0

    # Tier 2: dashboard actions_jobs.json (read-only, different schema)
    jobs_cache_path = _CACHE_DIR / "actions_jobs.json"
    dashboard_run_to_jobs: Dict[int, List[str]] = {}
    if jobs_cache_path.exists():
        try:
            jd = json.loads(jobs_cache_path.read_text())
            for _jk, jv in jd.get("items", {}).items():
                jobs_dict = jv.get("jobs", {}) if isinstance(jv, dict) else {}
                if isinstance(jobs_dict, dict):
                    for job_id_str, job in jobs_dict.items():
                        if isinstance(job, dict):
                            rid = job.get("run_id")
                            if rid and job.get("conclusion") == "failure":
                                dashboard_run_to_jobs.setdefault(rid, []).append(job_id_str)
        except (json.JSONDecodeError, OSError):
            pass
    dashboard_hits = 0

    result: List[str] = []
    uncached_run_ids: List[int] = []
    dirty = False

    for rid in run_ids:
        rid_key = str(rid)
        if rid_key in local_cache:
            result.extend(local_cache[rid_key])
            local_hits += 1
        elif rid in dashboard_run_to_jobs:
            job_ids = dashboard_run_to_jobs[rid]
            result.extend(job_ids)
            local_cache[rid_key] = job_ids
            dashboard_hits += 1
            dirty = True
        else:
            uncached_run_ids.append(rid)

    print(f"    Job ID lookup: {local_hits} local cache, {dashboard_hits} dashboard cache, {len(uncached_run_ids)} need API", flush=True)

    if uncached_run_ids:
        print(f"    Fetching jobs for {len(uncached_run_ids)} runs via API...", flush=True)
        api_errors = 0
        api_rate_limited = 0
        for i, rid in enumerate(uncached_run_ids):
            if (i + 1) % 50 == 0:
                print(f"      {i+1}/{len(uncached_run_ids)}...", flush=True)
            url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/runs/{rid}/jobs?per_page=100"
            try:
                resp = _gh_get(url)
                if resp.status_code == 200:
                    failed_ids: List[str] = []
                    for job in resp.json().get("jobs", []):
                        if job.get("conclusion") == "failure":
                            failed_ids.append(str(job["id"]))
                    result.extend(failed_ids)
                    local_cache[str(rid)] = failed_ids
                    dirty = True
                elif resp.status_code == 403:
                    api_rate_limited += 1
                    if api_rate_limited == 1:
                        print(f"      Rate limited at run {i+1}/{len(uncached_run_ids)}", flush=True)
                else:
                    api_errors += 1
            except _requests.RequestException as e:
                api_errors += 1
                if api_errors <= 3:
                    print(f"      API error for run {rid}: {e}", flush=True)
        if api_errors or api_rate_limited:
            print(f"    API issues: {api_errors} errors, {api_rate_limited} rate-limited", flush=True)

    if dirty:
        _save_json_cache(_JOBS_CACHE_PATH, local_cache)
        print(f"    Saved {len(local_cache)} entries to jobs cache", flush=True)

    return result


def _download_missing_logs(job_ids: List[str]) -> int:
    """Download raw logs for jobs not already on disk. Returns count downloaded.

    Also registers each downloaded file in the global raw-log-text/index.json
    with ``completed=true`` so that the dashboard prune_partial_raw_log_caches()
    (runs via cron every few minutes) does not delete them.
    """
    raw_dir = _get_raw_log_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Load the shared index that the dashboard pruner checks.
    index_path = raw_dir / "index.json"
    try:
        index: Dict[str, Any] = json.loads(index_path.read_text()) if index_path.exists() else {}
    except (json.JSONDecodeError, OSError):
        index = {}
    index_dirty = False

    downloaded = 0
    missing = [jid for jid in job_ids if not (raw_dir / f"{jid}.log").exists()]
    if not missing:
        return 0
    print(f"    Downloading {len(missing)} missing raw logs...", flush=True)
    dl_errors = 0
    dl_rate_limited = 0
    for i, jid in enumerate(missing):
        if (i + 1) % 50 == 0:
            print(f"      {i+1}/{len(missing)}...", flush=True)
        url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/jobs/{jid}/logs"
        try:
            resp = _gh_get(url, headers={"Accept": "application/vnd.github.v3.raw"}, allow_redirects=True)
            if resp.status_code == 200:
                log_path = raw_dir / f"{jid}.log"
                log_path.write_bytes(resp.content)
                downloaded += 1
                index[jid] = {"ts": int(_time.time()), "bytes": len(resp.content), "completed": True}
                index_dirty = True
            elif resp.status_code == 403:
                dl_rate_limited += 1
                if dl_rate_limited == 1:
                    print(f"      Rate limited at log {i+1}/{len(missing)}", flush=True)
            elif resp.status_code == 410:
                pass  # logs expired (>90 days), expected
            else:
                dl_errors += 1
        except _requests.RequestException as e:
            dl_errors += 1
            if dl_errors <= 3:
                print(f"      Download error for job {jid}: {e}", flush=True)

        # Flush index after every download so the cron pruner (runs every 2-7 min)
        # never sees an unregistered file.  Writes are ~2ms for a 3.6MB index.
        if index_dirty:
            index_path.write_text(json.dumps(index))
            index_dirty = False

    if index_dirty:
        index_path.write_text(json.dumps(index))
    if dl_errors or dl_rate_limited:
        print(f"    Download issues: {dl_errors} errors, {dl_rate_limited} rate-limited", flush=True)
    print(f"    Downloaded {downloaded}/{len(missing)} logs", flush=True)
    return downloaded


def _scan_failed_logs(failed_job_ids: List[str]) -> Tuple[Counter, Counter, int]:
    """Scan raw log files for failed jobs. Returns (category_hits, catch_all_hits, scanned).

    Caches scan results to disk so re-runs skip re-scanning already-categorized logs.
    """
    scan_cache = _load_json_cache(_SCAN_CACHE_PATH)
    cat_hits: Counter = Counter()
    catch_all_hits: Counter = Counter()
    scanned = 0
    cache_hits = 0
    dirty = False

    for job_id in failed_job_ids:
        if job_id in scan_cache:
            cats = scan_cache[job_id]
            cache_hits += 1
        else:
            lines = _read_log_lines(job_id)
            if lines is None:
                continue
            cats = list(categorize_error_log_lines(lines))
            scan_cache[job_id] = cats
            dirty = True

        scanned += 1
        for c in cats:
            if c in CATCH_ALLS:
                catch_all_hits[c] += 1
            else:
                cat_hits[c] += 1

    if dirty:
        _save_json_cache(_SCAN_CACHE_PATH, scan_cache)

    print(f"    Scan: {cache_hits} cached, {scanned - cache_hits} new, {len(failed_job_ids) - scanned} missing logs", flush=True)
    return cat_hits, catch_all_hits, scanned


def generate_weekly_report(
    api: GitHubAPIClient,
    week_start: date,
    week_end: date,
) -> str:
    """Generate a weekly post-merge stats report."""
    runs = _fetch_post_merge_runs(api, week_start, week_end)

    # Compute Development Metrics, passing post-merge runs for CI duration calc.
    print("  Computing development metrics...", flush=True)
    dev_metrics = _compute_dev_metrics(api, week_start, week_end, post_merge_runs=runs)

    total = len(runs)
    success = sum(1 for r in runs if r.get("conclusion") == "success")
    failure = sum(1 for r in runs if r.get("conclusion") == "failure")
    cancelled = sum(1 for r in runs if r.get("conclusion") == "cancelled")

    # Unique commits
    commits = set()
    commits_with_failure = set()
    for r in runs:
        sha = r.get("head_sha", "")
        if sha:
            commits.add(sha)
            if r.get("conclusion") == "failure":
                commits_with_failure.add(sha)

    # Workflow-level breakdown
    wf_total: Counter = Counter()
    wf_fail: Counter = Counter()
    for r in runs:
        name = r.get("name", "Unknown")
        wf_total[name] += 1
        if r.get("conclusion") == "failure":
            wf_fail[name] += 1

    # Collect failed job IDs
    failed_runs = [r for r in runs if r.get("conclusion") == "failure"]
    failed_run_ids = [r.get("id") for r in failed_runs if r.get("id")]
    print(f"  {len(failed_runs)} failed runs, fetching job IDs...")
    failed_job_ids = _get_failed_job_ids_for_runs(failed_run_ids)
    print(f"  {len(failed_job_ids)} failed jobs found")

    # Download missing raw logs
    _download_missing_logs(failed_job_ids)

    # Diagnostic: how many logs exist on disk
    raw_dir = _get_raw_log_dir()
    on_disk = sum(1 for jid in failed_job_ids if (raw_dir / f"{jid}.log").exists())
    print(f"  Logs on disk: {on_disk}/{len(failed_job_ids)} failed jobs", flush=True)

    cat_hits, catch_all_hits, scanned = _scan_failed_logs(failed_job_ids)

    # Build groups
    group_hits: Dict[str, Counter] = defaultdict(Counter)
    for cat, count in cat_hits.items():
        grp = GROUP_MAP.get(cat, "Other")
        group_hits[grp][cat] += count

    total_cat_hits = sum(cat_hits.values())

    # Days info
    day_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    start_day = day_names[week_start.weekday()]
    end_day = day_names[week_end.weekday()]
    num_days = (week_end - week_start).days + 1
    partial_note = f" — partial week, {num_days} days" if num_days < 7 else ""

    def _fmt(n: int) -> str:
        return f"{n:,}"

    # Error rates as % of total runs
    group_pct: Dict[str, float] = {}
    for grp, cats in group_hits.items():
        group_pct[grp] = sum(cats.values()) / total * 100 if total > 0 else 0

    L: List[str] = []

    # Header (matches established format)
    L.append(f"# Post-Merge CI Stats — Week of {week_start.strftime('%b %-d')}")
    L.append("")
    L.append(f"Period: {week_start} ({start_day}) to {week_end} ({end_day}){partial_note}")
    L.append("Sources:")
    L.append("  - Workflow-level: actions_runs_list.json cache (GitHub Status conclusions)")
    L.append(f"  - Error-category-level: {scanned} raw post-merge job logs (local scan, no API)")
    L.append("")
    L.append("---")
    L.append("")

    # Development Metrics (matches established format from older files)
    L.extend(_format_dev_metrics(dev_metrics))
    L.append("")

    # Error Rate section (matches established order)
    L.append("## Error Rate (% of Total Runs)")
    L.append("")
    L.append(f"Total runs (success + failure + cancelled): {_fmt(total)}")
    L.append("Group error hits as percentage of total runs:")
    L.append("")
    group_order = ["Tests", "Docker / build", "K8s / Helm", "Network", "Infra / system", "Auth"]
    for grp in group_order:
        pct = group_pct.get(grp, 0.0)
        L.append(f"  {grp + ':':<22s}{pct:>5.1f}%")
    L.append("")
    L.append("---")
    L.append("")

    # Workflow-Level Overview
    L.append("## Workflow-Level Overview")
    L.append("")
    L.append(f"Total post-merge runs:     {_fmt(total):>5}")
    if total:
        L.append(f"  success:                 {_fmt(success):>5} ({success/total*100:.1f}%)")
        L.append(f"  failure:                 {_fmt(failure):>5} ({failure/total*100:.1f}%)")
        L.append(f"  cancelled:               {_fmt(cancelled):>5} ({cancelled/total*100:.1f}%)")
    L.append("")
    L.append(f"Unique commits on main:    {_fmt(len(commits)):>5}")
    if commits:
        L.append(f"Commits with >= 1 failure: {_fmt(len(commits_with_failure)):>5} ({len(commits_with_failure)/len(commits)*100:.1f}%)")
    L.append(f"Post-merge failure rate:     {failure/total*100:.1f}%" if total else "Post-merge failure rate:     N/A")
    L.append("")

    # Failures by Workflow
    L.append("## Failures by Workflow")
    L.append("")
    sorted_wf = sorted(wf_fail.items(), key=lambda x: -x[1])
    for i, (name, count) in enumerate(sorted_wf, 1):
        pct = count / failure * 100 if failure else 0
        wt = wf_total[name]
        fr = count / wt * 100 if wt else 0
        L.append(f"  {i}. {name:<40s} {count:>3} ({pct:>5.2f}%)  [{fr:>3.0f}% fail rate, {wt:>4} runs]")
    L.append("")
    L.append("---")
    L.append("")

    # Failure Distribution by Group
    L.append("## Failure Distribution by Group")
    L.append("")
    all_groups = group_order + ["Gates / policy", "Docs", "Build"]
    for grp in all_groups:
        if grp in group_hits:
            cats = group_hits[grp]
            grp_total = sum(cats.values())
            grp_pct = grp_total / total_cat_hits * 100 if total_cat_hits else 0
            detail_parts = [f"{c}: {n}" for c, n in cats.most_common()]
            # Wrap long detail lines
            detail = ", ".join(detail_parts)
            L.append(f"  {grp + ':':<26s} {grp_total:>3} ({grp_pct:.2f}%)")
            L.append(f"    {detail}")
    L.append("")

    # Failure Distribution Detail
    L.append("## Failure Distribution Detail")
    L.append("")
    if catch_all_hits:
        catch_str = ", ".join(f"{c} ({n})" for c, n in catch_all_hits.most_common())
        L.append(f"Catch-alls excluded: {catch_str}")
    L.append(f"Total error hits: {total_cat_hits}")
    L.append("")
    for i, (cat, count) in enumerate(cat_hits.most_common(), 1):
        pct = count / total_cat_hits * 100 if total_cat_hits else 0
        L.append(f" {i:>2}. {cat:<36s} {count:>5} ({pct:>5.2f}%)")
    L.append("")
    L.append("---")
    L.append("")

    return "\n".join(L) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate weekly post-merge CI stats")
    parser.add_argument("--week", action="append", required=True,
                        help="Monday date of week to generate (YYYY-MM-DD). Repeatable.")
    parser.add_argument("--days", type=int, default=7,
                        help="Number of days in week (default: 7)")
    parser.add_argument("--outdir", default=str(Path(__file__).parent / "stats"),
                        help="Output directory")
    args = parser.parse_args()

    api = GitHubAPIClient()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for week_str in args.week:
        week_start = date.fromisoformat(week_str)
        week_end = week_start + timedelta(days=args.days - 1)
        # Cap at today
        today = date.today()
        if week_end > today:
            week_end = today

        print(f"Generating stats for week of {week_start} to {week_end}...")
        report = generate_weekly_report(api, week_start, week_end)
        outfile = outdir / f"{week_start}.txt"
        outfile.write_text(report)
        print(f"  Written to {outfile}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
