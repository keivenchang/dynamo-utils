#!/usr/bin/env python3
"""Unified CI merge analysis: pre-merge, re-validate, and post-merge data
for merged main commits.

THREE DATA SOURCES, three caches:

  pre-merge   `~/.cache/dynamo-utils/pre-merge.json`
              Original CI runs on `pull-request/<N>` branch (multiple pushes
              x run_attempts). Captured by this script.

  re-validate `~/.cache/dynamo-utils/ci-health.json`
              Synthetic placebo-PR re-runs of merged SHAs. Managed by
              revalidate_pr.py's cron loop. This script READS it; it never
              writes (revalidate_pr is the owner).

  post-merge  `~/.cache/dynamo-utils/post-merge.json`
              Single push:main run after the squash-merge lands on main.
              Captured by this script.

Cache only entries that are TERMINAL (all attempts/runs reached `completed`
status) — in-progress entries are re-fetched until they settle, then frozen.

Subcommands:
  lookup <PR|SHA> [--source pre|post|revalidate|all]
  backfill [--past N] [--source pre|post|all]
  render-html (placeholder for the next phase)
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ci_lib import (  # noqa: E402
    REPO,
    _to_pt,
    extract_pytest_failures,
    fetch_job_log,
    gh_api,
    render_ci_attempts_page,
    short_sha,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRE_MERGE_DB = Path.home() / ".cache" / "dynamo-utils" / "pre-merge.json"
POST_MERGE_DB = Path.home() / ".cache" / "dynamo-utils" / "post-merge.json"
PRE_MERGE_WORKFLOW_NAME = "PR"
POST_MERGE_WORKFLOW_NAME = "Post-Merge CI Pipeline"

TERMINAL_CONCLUSIONS = {"success", "failure", "cancelled", "timed_out", "skipped"}
JOB_FAILED_CONCLUSIONS = {"failure", "timed_out", "cancelled"}

# Squash-merge subject ends with `(#NNNN)`.
_SUBJECT_PR_RE = re.compile(r"\(#(\d+)\)\s*$")


# ---------------------------------------------------------------------------
# Cache helpers (one pair per source; revalidate cache is read-only here)
# ---------------------------------------------------------------------------


def _load_db(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f)


def _save_db_atomic(path: Path, db: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(db, f, indent=2, sort_keys=True)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# SHA / PR resolution
# ---------------------------------------------------------------------------


def resolve_sha(arg: str) -> str:
    """Accept a PR # or hex SHA. Returns the merge commit SHA (lowercase)."""
    if arg.isdigit():
        pr_num = int(arg)
        data = gh_api(f"repos/{REPO}/pulls/{pr_num}")
        if not data:
            raise SystemExit(f"PR #{pr_num}: not found")
        sha = data.get("merge_commit_sha")
        if not sha or not data.get("merged"):
            raise SystemExit(f"PR #{pr_num} is not merged (state={data.get('state')})")
        return sha.lower()
    if len(arg) < 7 or not all(c in "0123456789abcdef" for c in arg.lower()):
        raise SystemExit(f"argument {arg!r}: not a PR number or hex SHA")
    s = arg.lower()
    if len(s) == 40:
        return s
    # Expand short prefix -> full SHA so cache keys are always 40-char.
    commit = gh_api(f"repos/{REPO}/commits/{s}")
    if commit and commit.get("sha"):
        return commit["sha"].lower()
    raise SystemExit(f"could not expand SHA prefix {s!r} to a full SHA")


def pr_for_sha(sha: str) -> int | None:
    """Pull PR # out of a merge commit subject. None if not a `(#NNNN)`-style merge."""
    commit = gh_api(f"repos/{REPO}/commits/{sha}")
    if not commit:
        return None
    msg = (commit.get("commit") or {}).get("message", "")
    first_line = msg.splitlines()[0] if msg else ""
    m = _SUBJECT_PR_RE.search(first_line)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Source: PRE-MERGE
# ---------------------------------------------------------------------------


def _list_pr_branch_runs(pr_num: int) -> list[dict]:
    """All `name=='PR'` workflow runs on `pull-request/<N>`, oldest-first."""
    runs: list[dict] = []
    page = 1
    while True:
        resp = gh_api(
            f"repos/{REPO}/actions/runs?branch=pull-request/{pr_num}"
            f"&per_page=100&page={page}"
        )
        if not resp:
            break
        batch = resp.get("workflow_runs") or []
        if not batch:
            break
        runs.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    pr_runs = [r for r in runs if r.get("name") == PRE_MERGE_WORKFLOW_NAME]
    pr_runs.sort(key=lambda r: r.get("created_at") or "")
    return pr_runs


def _list_attempt_jobs(run_id: int, attempt: int) -> list[dict]:
    """Jobs for a (run_id, run_attempt). Paginated."""
    jobs: list[dict] = []
    page = 1
    while True:
        resp = gh_api(
            f"repos/{REPO}/actions/runs/{run_id}/attempts/{attempt}/jobs"
            f"?per_page=100&page={page}"
        )
        if not resp:
            break
        batch = resp.get("jobs") or []
        if not batch:
            break
        jobs.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return jobs


def _job_record(j: dict, with_test_extraction: bool = True) -> dict:
    rec = {
        "name": j["name"],
        "id": j["id"],
        "url": j.get("html_url"),
        "conclusion": j.get("conclusion"),
        "status": j.get("status"),
        "started_at": j.get("started_at"),
        "completed_at": j.get("completed_at"),
        "failed_tests": [],
    }
    if (
        with_test_extraction
        and j.get("conclusion") in JOB_FAILED_CONCLUSIONS
        and j.get("id")
    ):
        log = fetch_job_log(j["id"])
        if log:
            rec["failed_tests"] = extract_pytest_failures(log)
    return rec


def _premerge_attempt(run: dict, attempt: int) -> dict:
    jobs = _list_attempt_jobs(run["id"], attempt)
    # Per-attempt metadata (incl. triggering_actor — who clicked Re-run for
    # attempt > 1; original CI trigger for attempt 1).
    att_meta = gh_api(f"repos/{REPO}/actions/runs/{run['id']}/attempts/{attempt}") or {}
    ta = att_meta.get("triggering_actor") or {}
    return {
        "run_id": run["id"],
        "head_sha": run.get("head_sha"),
        "run_attempt": attempt,
        "html_url": run.get("html_url"),
        "status": att_meta.get("status") or run.get("status"),
        "conclusion": run.get("conclusion") if attempt == run.get("run_attempt") else None,
        "created_at": att_meta.get("created_at") or run.get("created_at"),
        "updated_at": att_meta.get("updated_at") or run.get("updated_at"),
        "triggering_actor": ta.get("login"),
        "jobs": [_job_record(j) for j in jobs],
    }


def fetch_premerge(sha: str, pr_num: int) -> dict:
    """Build pre-merge entry: every (workflow_run, run_attempt) flattened."""
    runs = _list_pr_branch_runs(pr_num)
    attempts: list[dict] = []
    for run in runs:
        max_attempt = run.get("run_attempt") or 1
        for n in range(1, max_attempt + 1):
            attempts.append(_premerge_attempt(run, n))
    return {
        "pr": pr_num,
        "sha": sha,
        "n_workflow_runs": len(runs),
        "n_attempts": len(attempts),
        "attempts": attempts,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def is_premerge_terminal(entry: dict) -> bool:
    atts = entry.get("attempts") or []
    if not atts:
        return False
    return all(a.get("status") == "completed" for a in atts)


# ---------------------------------------------------------------------------
# Source: POST-MERGE
# ---------------------------------------------------------------------------


def _find_postmerge_run(sha: str) -> dict | None:
    runs = gh_api(f"repos/{REPO}/actions/runs?head_sha={sha}&event=push")
    if not runs:
        return None
    for r in runs.get("workflow_runs") or []:
        if r.get("name") == POST_MERGE_WORKFLOW_NAME:
            return r
    return None


def _list_run_jobs(run_id: int) -> list[dict]:
    jobs: list[dict] = []
    page = 1
    while True:
        resp = gh_api(
            f"repos/{REPO}/actions/runs/{run_id}/jobs?per_page=100"
            f"&filter=latest&page={page}"
        )
        if not resp:
            break
        batch = resp.get("jobs") or []
        if not batch:
            break
        jobs.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return jobs


def fetch_postmerge(sha: str) -> dict | None:
    """Build post-merge entry. Returns None if no post-merge run exists for the SHA."""
    run = _find_postmerge_run(sha)
    if run is None:
        return None
    jobs_raw = _list_run_jobs(run["id"])
    # Store ALL jobs (passing + failing). Test-extraction is gated on failure
    # to keep cache size bounded.
    all_jobs = [
        _job_record(j, with_test_extraction=j.get("conclusion") in JOB_FAILED_CONCLUSIONS)
        for j in jobs_raw
    ]
    failed_jobs = [j for j in all_jobs if j.get("conclusion") in JOB_FAILED_CONCLUSIONS]
    ta = run.get("triggering_actor") or {}
    return {
        "sha": sha,
        "run_id": run["id"],
        "html_url": run.get("html_url"),
        "status": run.get("status"),
        "conclusion": run.get("conclusion"),
        "run_attempt": run.get("run_attempt") or 1,
        "created_at": run.get("created_at"),
        "updated_at": run.get("updated_at"),
        "triggering_actor": ta.get("login"),
        "jobs": all_jobs,
        "failed_jobs": failed_jobs,  # kept for backward compat
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def is_postmerge_terminal(entry: dict) -> bool:
    return (
        entry.get("status") == "completed"
        and entry.get("conclusion") in TERMINAL_CONCLUSIONS
    )


# ---------------------------------------------------------------------------
# Cached lookups
# ---------------------------------------------------------------------------


def get_premerge(sha: str, *, refresh: bool = False) -> dict | None:
    db = _load_db(PRE_MERGE_DB)
    cached = db.get(sha)
    if cached and is_premerge_terminal(cached) and not refresh:
        return cached
    pr_num = pr_for_sha(sha)
    if pr_num is None:
        return None
    fresh = fetch_premerge(sha, pr_num)
    db[sha] = fresh
    _save_db_atomic(PRE_MERGE_DB, db)
    return fresh


def get_postmerge(sha: str, *, refresh: bool = False) -> dict | None:
    db = _load_db(POST_MERGE_DB)
    cached = db.get(sha)
    if cached and is_postmerge_terminal(cached) and not refresh:
        return cached
    fresh = fetch_postmerge(sha)
    if fresh is None:
        if cached:
            return cached
        return None
    db[sha] = fresh
    _save_db_atomic(POST_MERGE_DB, db)
    return fresh

# ---------------------------------------------------------------------------
# CLI: lookup
# ---------------------------------------------------------------------------


def _print_premerge(entry: dict) -> None:
    if not entry:
        print("  (no pre-merge data)")
        return
    print(f"  PR             : #{entry.get('pr')}")
    print(f"  Workflow runs  : {entry.get('n_workflow_runs')}")
    print(f"  Total attempts : {entry.get('n_attempts')}")
    for i, a in enumerate(entry.get("attempts") or [], 1):
        n_jobs = len(a.get("jobs") or [])
        n_failed = sum(
            1 for j in (a.get("jobs") or []) if j.get("conclusion") in JOB_FAILED_CONCLUSIONS
        )
        n_tests = sum(len(j.get("failed_tests") or []) for j in (a.get("jobs") or []))
        head = (a.get("head_sha") or "")[:9]
        c = a.get("conclusion") or "(eclipsed)"
        _dt = _to_pt(a.get("created_at"))
        _ts = _dt.strftime("%Y-%m-%d %H:%M:%S %Z") if _dt else "—"
        print(
            f"  att {i:2d}: run={a.get('run_id')}/{a.get('run_attempt')}  "
            f"head={head}  {_ts}  "
            f"verdict={c:<22}  {n_failed}/{n_jobs} jobs failed, {n_tests} tests"
        )


def _print_postmerge(entry: dict) -> None:
    if not entry:
        print("  (no post-merge data)")
        return
    print(f"  Run            : {entry.get('html_url')}")
    print(f"  Conclusion     : {entry.get('conclusion')}")
    failed = entry.get("failed_jobs") or []
    print(f"  Failed jobs    : {len(failed)}")
    for j in failed:
        tests = j.get("failed_tests") or []
        print(f"    [{j.get('conclusion')}] {j['name']}  ({len(tests)} test failures)")


def _resolve_flags(premerge: bool, postmerge: bool) -> tuple[bool, bool]:
    """If neither flag is set, default to BOTH on. Otherwise honor what's set."""
    if not premerge and not postmerge:
        return True, True
    return premerge, postmerge


def cmd_lookup(args: argparse.Namespace) -> int:
    sha = resolve_sha(args.target)
    do_pre, do_post = _resolve_flags(args.premerge, args.postmerge)
    print(f"SHA            : {sha}")
    if do_pre:
        print("\n=== pre-merge ===")
        _print_premerge(get_premerge(sha, refresh=args.refresh))
    if do_post:
        print("\n=== post-merge ===")
        _print_postmerge(get_postmerge(sha, refresh=args.refresh))
    return 0


# ---------------------------------------------------------------------------
# Backfill helpers (per-source loops, called by cmd_backfill / cmd_run)
# ---------------------------------------------------------------------------


def _backfill_premerge(commits: list[dict], refresh: bool) -> None:
    db = _load_db(PRE_MERGE_DB)
    refreshed = skipped = pending = no_pr = 0
    for c in commits:
        sha = c["sha"]
        cached = db.get(sha)
        if cached and is_premerge_terminal(cached) and not refresh:
            skipped += 1
            continue
        msg = ((c.get("commit") or {}).get("message") or "").splitlines()[0]
        m = _SUBJECT_PR_RE.search(msg)
        if not m:
            no_pr += 1
            continue
        entry = fetch_premerge(sha, int(m.group(1)))
        # Persist regardless of terminal state — in-progress entries surface
        # in the dashboard as "running" pills.
        db[sha] = entry
        # Flush after each entry so a crash mid-loop doesn't lose all work.
        _save_db_atomic(PRE_MERGE_DB, db)
        if is_premerge_terminal(entry):
            refreshed += 1
        else:
            pending += 1
    print(
        f"  pre-merge:  refreshed={refreshed}  cached-skip={skipped}  "
        f"in-progress={pending}  no-pr={no_pr}"
    )


def _backfill_postmerge(commits: list[dict], refresh: bool) -> None:
    db = _load_db(POST_MERGE_DB)
    refreshed = skipped = pending = no_run = 0
    for c in commits:
        sha = c["sha"]
        cached = db.get(sha)
        if cached and is_postmerge_terminal(cached) and not refresh:
            skipped += 1
            continue
        entry = fetch_postmerge(sha)
        if entry is None:
            no_run += 1
            continue
        db[sha] = entry
        _save_db_atomic(POST_MERGE_DB, db)
        if is_postmerge_terminal(entry):
            refreshed += 1
        else:
            pending += 1
    print(
        f"  post-merge: refreshed={refreshed}  cached-skip={skipped}  "
        f"in-progress={pending}  no-run={no_run}"
    )


def cmd_backfill(args: argparse.Namespace) -> int:
    n = args.past
    commits = gh_api(f"repos/{REPO}/commits?sha=main&per_page={n}")
    if not commits:
        print("no commits returned from API", file=sys.stderr)
        return 2
    commits = commits[:n]
    do_pre, do_post = _resolve_flags(args.premerge, args.postmerge)
    print(f"=== backfill past={n}  pre-merge={do_pre}  post-merge={do_post} ===")
    if do_pre:
        _backfill_premerge(commits, args.refresh)
    if do_post:
        _backfill_postmerge(commits, args.refresh)
    return 0


# ---------------------------------------------------------------------------
# CLI: render-html — per-SHA + summary HTML for pre-merge and post-merge
# ---------------------------------------------------------------------------


_HTML_STYLE = """<style>
  body { font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0; padding: 24px; background: #fafbfc; color: #24292e; max-width: 1400px; }
  h1 { font-size: 20px; margin: 0 0 4px 0; }
  h2 { font-size: 16px; margin: 24px 0 8px 0; padding-bottom: 4px; border-bottom: 1px solid #e1e4e8; }
  .meta { color: #586069; font-size: 13px; margin-bottom: 16px; }
  .meta span { margin-right: 16px; }
  code { background: #f6f8fa; padding: 2px 6px; border-radius: 3px; font-size: 12px;
         font-family: "SF Mono", Consolas, monospace; }
  a { color: #0366d6; text-decoration: none; }
  a:hover { text-decoration: underline; }
  a[target="_blank"]::after { content: " ↗"; font-size: 0.85em; color: #959da5; }
  .v-good { color: #28a745; font-weight: 600; }
  .v-bad  { color: #d73a49; font-weight: 600; }
  .v-other { color: #bf6c00; font-weight: 600; }
  .pill { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
  .pill-pass { background: #d4edda; color: #155724; }
  .pill-fail { background: #f8d7da; color: #721c24; }
  .pill-skip { background: #eaeef2; color: #586069; }
  .pill-run  { background: #cce5ff; color: #004085; }
  table { border-collapse: collapse; margin: 8px 0 4px 0; font-size: 13px; width: 100%; }
  th, td { text-align: left; padding: 5px 10px; border-bottom: 1px solid #eaecef; vertical-align: top; }
  th { background: #f6f8fa; font-weight: 600; font-size: 11px; text-transform: uppercase;
       letter-spacing: 0.05em; color: #586069; }
  tr.fail-final td { background: #ea868f; }
  tr.fail-eclipsed td { background: #fbe4e6; }
  tr.pass td { background: #c3e6cb; }
  tr.run td { background: #b8daff; }
  tr.skip td { background: #f0f0f0; color: #6a737d; }
  details { background: #fff; border: 1px solid #d0d7de; border-radius: 6px;
            margin: 12px 0; padding: 0; }
  details > summary { cursor: pointer; padding: 10px 14px; font-weight: 600;
                      list-style: none; user-select: none; }
  details > summary::-webkit-details-marker { display: none; }
  details > summary::before { content: "▶ "; color: #586069; font-size: 11px;
                              display: inline-block; transition: transform 0.1s; }
  details[open] > summary::before { transform: rotate(90deg); }
  details > .body { padding: 0 14px 12px 14px; }
  ul.tests { margin: 4px 0 4px 16px; padding: 0; font-family: "SF Mono", Consolas, monospace;
             font-size: 11px; color: #d73a49; }
  ul.tests li { padding: 1px 0; }
  td.col-job { width: 50%; }
  td.col-status { white-space: nowrap; }
</style>"""


def _esc(s) -> str:
    return html_mod.escape("" if s is None else str(s), quote=True)


def _verdict_class(c: str | None) -> str:
    if c == "success":
        return "v-good"
    if c in ("failure", "timed_out"):
        return "v-bad"
    return "v-other"


def _verdict_pill(c: str | None) -> str:
    if c == "success":
        return f"<span class='pill pill-pass'>{_esc(c)}</span>"
    if c in ("failure", "timed_out"):
        return f"<span class='pill pill-fail'>{_esc(c)}</span>"
    if c in ("running", "queued", "pending", "in_progress"):
        return f"<span class='pill pill-run'>{_esc(c)}</span>"
    return f"<span class='pill pill-skip'>{_esc(c or '?')}</span>"


def _row_class_for_job(job: dict, is_final_attempt: bool) -> str:
    """Color the job row: red on failure (darker if it's the final attempt),
    green on success, blue on running, grey on skip/cancel."""
    c = job.get("conclusion")
    if c in ("failure", "timed_out"):
        return "fail-final" if is_final_attempt else "fail-eclipsed"
    if c == "success":
        return "pass"
    if c in ("running", "queued", "pending"):
        return "run"
    return "skip"


def _date_dir_for(iso: str | None) -> str:
    dt = _to_pt(iso) if iso else None
    return dt.strftime("%Y-%m-%d") if dt else "unknown"


def _render_jobs_table(jobs: list[dict], is_final_attempt: bool) -> str:
    rows = []
    rows.append(
        "<tr><th>Job</th><th>Status</th><th>Started</th><th>Duration</th><th>Failed tests</th></tr>"
    )
    sorted_jobs = sorted(jobs, key=lambda j: j.get("name") or "")
    for j in sorted_jobs:
        cls = _row_class_for_job(j, is_final_attempt)
        url = j.get("url") or "#"
        name = _esc(j.get("name"))
        started = (j.get("started_at") or "")[:19]
        ended = j.get("completed_at") or ""
        dur = ""
        if started and ended:
            try:
                a = datetime.fromisoformat(j["started_at"])
                b = datetime.fromisoformat(j["completed_at"])
                secs = max(0, int((b - a).total_seconds()))
                if secs < 60:
                    dur = f"{secs}s"
                elif secs < 3600:
                    dur = f"{secs // 60}m{secs % 60:02d}s"
                else:
                    dur = f"{secs // 3600}h{(secs % 3600) // 60}m"
            except (ValueError, TypeError):
                dur = ""
        tests = j.get("failed_tests") or []
        tests_html = ""
        if tests:
            tests_html = "<ul class='tests'>" + "".join(
                f"<li>{_esc(t)}</li>" for t in tests
            ) + "</ul>"
        rows.append(
            f"<tr class='{cls}'>"
            f"<td class='col-job'><a href='{_esc(url)}' target='_blank'>{name}</a></td>"
            f"<td class='col-status'>{_verdict_pill(j.get('conclusion'))}</td>"
            f"<td class='col-status'>{_esc(started)}</td>"
            f"<td class='col-status'>{_esc(dur)}</td>"
            f"<td>{tests_html}</td>"
            f"</tr>"
        )
    return "<table>" + "".join(rows) + "</table>"


def _to_probe_attempt(my_attempt: dict, attempt_num: int) -> dict:
    """Convert a ci_merge_analysis attempt to revalidate's attempt shape.

    revalidate's per-SHA renderer expects:
      - jobs: dict[name -> {id, conclusion, started_at, completed_at, ...}]
      - failed_jobs: list[name] (used by the flake-banner counter)
      - failed_tests: dict[name -> [test_id, ...]]
      - log_meta: dict[name -> {categories, snippet}]
      - started_at / completed_at at attempt level

    Our cache stores jobs as a list and failed_tests inline on each job, so
    rebuild the dict-shaped structures here.
    """
    jobs_list = my_attempt.get("jobs") or []
    jobs_dict: dict[str, dict] = {}
    failed_tests: dict[str, list[str]] = {}
    failed_jobs: list[str] = []
    for j in jobs_list:
        name = j.get("name")
        if not name:
            continue
        jobs_dict[name] = {
            "id": j.get("id"),
            "url": j.get("url"),
            "conclusion": j.get("conclusion"),
            "status": j.get("status"),
            "started_at": j.get("started_at"),
            "completed_at": j.get("completed_at"),
        }
        if j.get("failed_tests"):
            failed_tests[name] = list(j["failed_tests"])
        if j.get("conclusion") in JOB_FAILED_CONCLUSIONS:
            failed_jobs.append(name)
    return {
        "attempt": attempt_num,
        "started_at": my_attempt.get("created_at"),
        "completed_at": my_attempt.get("updated_at"),
        "jobs": jobs_dict,
        "failed_tests": failed_tests,
        "failed_jobs": failed_jobs,
        "log_meta": {},
        "triggering_actor": my_attempt.get("triggering_actor"),
        # carried through for completeness; revalidate doesn't read these
        "run_id": my_attempt.get("run_id"),
        "run_attempt_n": my_attempt.get("run_attempt"),
        "head_sha": my_attempt.get("head_sha"),
        "html_url": my_attempt.get("html_url"),
    }


def _derive_probe_state(atts: list[dict]) -> tuple[str, str | None]:
    """(state, verdict) compatible with revalidate's _display_state/_state_class."""
    if not atts:
        return ("discovered", None)
    last = atts[-1]
    last_failed = bool(last.get("failed_jobs"))
    last_concl_ok = not last_failed
    if last_concl_ok:
        return ("passed", "passed")
    return ("failed", "failed")


def _render_premerge_page(sha: str, entry: dict) -> str:
    """Transform our pre-merge entry to revalidate's shape and reuse its renderer."""
    atts = entry.get("attempts") or []
    probe_atts = [_to_probe_attempt(a, i + 1) for i, a in enumerate(atts)]
    state, verdict = _derive_probe_state(probe_atts)
    last_ca = atts[-1].get("created_at") if atts else None
    probe_entry = {
        "pr": entry.get("pr"),
        "branch": f"pull-request/{entry.get('pr')}" if entry.get("pr") else None,
        "merge_date": last_ca,
        "image_sha256": "?",
        "attempts": probe_atts,
        "max_attempts": max(len(probe_atts), 1),
        "state": state,
        "verdict": verdict,
        "stuck_jobs": [],
    }
    return render_ci_attempts_page(
        sha, probe_entry, kind="Pre-merge", jobs_layout="pivoted",
    )


def _render_postmerge_page(sha: str, entry: dict) -> str:
    """Post-merge has one logical attempt. Reuse revalidate's renderer with a
    single-attempt entry."""
    # Prefer 'jobs' (all jobs); fall back to 'failed_jobs' for older cache
    # entries that pre-date the schema change.
    src_jobs = entry.get("jobs") or entry.get("failed_jobs") or []
    jobs_dict: dict[str, dict] = {}
    failed_tests: dict[str, list[str]] = {}
    failed_job_names: list[str] = []
    for j in src_jobs:
        name = j.get("name")
        if not name:
            continue
        jobs_dict[name] = {
            "id": j.get("id"),
            "url": j.get("url"),
            "conclusion": j.get("conclusion"),
            "status": j.get("status"),
            "started_at": j.get("started_at"),
            "completed_at": j.get("completed_at"),
        }
        if j.get("failed_tests"):
            failed_tests[name] = list(j["failed_tests"])
        if j.get("conclusion") in JOB_FAILED_CONCLUSIONS:
            failed_job_names.append(name)
    probe_atts = [{
        "attempt": 1,
        "started_at": entry.get("created_at"),
        "completed_at": entry.get("updated_at"),
        "jobs": jobs_dict,
        "failed_tests": failed_tests,
        "failed_jobs": failed_job_names,
        "log_meta": {},
        "triggering_actor": entry.get("triggering_actor"),
        "run_id": entry.get("run_id"),
        "run_attempt_n": entry.get("run_attempt"),
        "html_url": entry.get("html_url"),
    }]
    conc = entry.get("conclusion")
    state = "passed" if conc == "success" else "failed"
    probe_entry = {
        "pr": None,
        "branch": "main",
        "merge_date": entry.get("created_at"),
        "image_sha256": "?",
        "attempts": probe_atts,
        "max_attempts": 1,
        "state": state,
        "verdict": state,
        "stuck_jobs": [],
    }
    # Post-merge has no required-checks subset; verdict comes straight from
    # the run's overall conclusion.
    override = {1: "PASS" if conc == "success" else "FAIL"}
    return render_ci_attempts_page(
        sha, probe_entry, kind="Post-merge", jobs_layout="pivoted",
        att_verdict_override=override,
    )


def _render_summary(kind: str, entries_by_sha: dict, page_paths: dict) -> str:
    # kind: "pre-merge" or "post-merge"
    from ci_lib import _HTML_STYLE, _html_escape, commit_subject_pr_author
    far_past = datetime.min.replace(tzinfo=timezone.utc)
    if kind == "pre-merge":
        sort_key = lambda kv: (
            _to_pt((kv[1].get("attempts") or [{}])[-1].get("created_at")) or far_past, kv[0]
        )
    else:
        sort_key = lambda kv: (_to_pt(kv[1].get("created_at")) or far_past, kv[0])
    items = sorted(entries_by_sha.items(), key=sort_key, reverse=True)

    # Per-SHA metadata for both the index rows and the history-bar tooltips.
    sha_meta: dict[str, dict] = {}
    for _sha, _e in items:
        _subj, _orig_pr, _author = commit_subject_pr_author(_sha)
        if kind == "pre-merge":
            _atts = _e.get("attempts") or []
            _ca = (_atts[-1].get("created_at") if _atts else None)
        else:
            _ca = _e.get("created_at")
        _dt = _to_pt(_ca)
        _ts = _dt.strftime("%Y-%m-%d %H:%M:%S %Z") if _dt else "—"
        _pr = _e.get("pr") or _orig_pr
        sha_meta[_sha] = {
            "subj": _subj,
            "author": _author,
            "ts": _ts,
            "pr": _pr,
        }

    rows = []
    for sha, e in items:
        link = page_paths.get(sha, "#")
        s = short_sha(sha)
        _meta = sha_meta.get(sha) or {}
        subj = _meta.get("subj") or ""
        author = _meta.get("author") or ""
        subj_esc = _html_escape(subj) if subj else ""
        author_esc = _html_escape(author) if author else "—"
        if kind == "pre-merge":
            atts = e.get("attempts") or []
            n_atts = len(atts)
            last = atts[-1] if atts else {}
            conc = last.get("conclusion") or "(eclipsed)"
            n_failed_total = sum(
                sum(1 for j in (a.get("jobs") or []) if j.get("conclusion") in JOB_FAILED_CONCLUSIONS)
                for a in atts
            )
            n_tests = sum(
                sum(len(j.get("failed_tests") or []) for j in (a.get("jobs") or []))
                for a in atts
            )
            _dt_pt = _to_pt(last.get("created_at"))
            ts = _dt_pt.strftime("%Y-%m-%d %H:%M:%S") if _dt_pt else "—"
            pr = e.get("pr")
            rows.append(
                f"<tr style='white-space:nowrap;'><td>{_esc(ts)}</td>"
                f"<td><a href='{_esc(link)}'><code>{s}</code></a></td>"
                f"<td>PR <a href='https://github.com/{REPO}/pull/{pr}' target='_blank'>#{pr}</a></td>"
                f"<td style='text-align:left; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:600px;' title='{subj_esc}'>{subj_esc}</td>"
                f"<td style='text-align:left; white-space:nowrap;'>{author_esc}</td>"
                f"<td>{n_atts}</td>"
                f"<td>{_verdict_pill(conc)}</td>"
                f"<td>{n_failed_total}</td>"
                f"<td>{n_tests}</td>"
                f"</tr>"
            )
        else:
            conc = e.get("conclusion")
            failed_jobs = e.get("failed_jobs") or []
            n_failed = len(failed_jobs)
            n_tests = sum(len(j.get("failed_tests") or []) for j in failed_jobs)
            _dt_pt = _to_pt(e.get("created_at"))
            ts = _dt_pt.strftime("%Y-%m-%d %H:%M:%S") if _dt_pt else "—"
            rows.append(
                f"<tr style='white-space:nowrap;'><td>{_esc(ts)}</td>"
                f"<td><a href='{_esc(link)}'><code>{s}</code></a></td>"
                f"<td style='text-align:left; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:600px;' title='{subj_esc}'>{subj_esc}</td>"
                f"<td style='text-align:left; white-space:nowrap;'>{author_esc}</td>"
                f"<td>{_verdict_pill(conc)}</td>"
                f"<td>{n_failed}</td>"
                f"<td>{n_tests}</td>"
                f"</tr>"
            )

    if kind == "pre-merge":
        thead = ("<thead><tr><th>Last attempt (PT)</th><th>SHA</th><th>PR</th>"
                 "<th style='text-align:left;'>Title</th>"
                 "<th style='text-align:left;'>Author</th>"
                 "<th># attempts</th><th>Final verdict</th>"
                 "<th>Failed jobs (Σ)</th><th>Failed tests (Σ)</th></tr></thead>")
    else:
        thead = ("<thead><tr><th>Run started (PT)</th><th>SHA</th>"
                 "<th style='text-align:left;'>Title</th>"
                 "<th style='text-align:left;'>Author</th>"
                 "<th>Verdict</th><th># failed jobs</th><th># failed tests</th></tr></thead>")

    # Aggregate failing-tests tally — newest-first within each test row so the
    # most-recent occurrences surface as the first job links (drill-down).
    test_count: dict[str, int] = {}
    test_jobs: dict[str, list[tuple[str, str | None, str, int | None]]] = {}
    n_attempts = 0
    n_failed_jobs = 0
    n_with_failures = 0
    for sha, e in items:
        sha_had_failure = False
        if kind == "pre-merge":
            for a in e.get("attempts") or []:
                n_attempts += 1
                for j in a.get("jobs") or []:
                    jname = j.get("name") or ""
                    jurl = j.get("url")
                    jid = j.get("id")
                    if j.get("conclusion") in JOB_FAILED_CONCLUSIONS:
                        n_failed_jobs += 1
                        sha_had_failure = True
                    for t in j.get("failed_tests") or []:
                        if not t:
                            continue
                        test_count[t] = test_count.get(t, 0) + 1
                        test_jobs.setdefault(t, []).append((sha, jurl, jname, jid))
        else:
            n_attempts += 1
            src = e.get("jobs") or e.get("failed_jobs") or []
            for j in src:
                jname = j.get("name") or ""
                jurl = j.get("url")
                jid = j.get("id")
                if j.get("conclusion") in JOB_FAILED_CONCLUSIONS:
                    n_failed_jobs += 1
                    sha_had_failure = True
                for t in j.get("failed_tests") or []:
                    if not t:
                        continue
                    test_count[t] = test_count.get(t, 0) + 1
                    test_jobs.setdefault(t, []).append((sha, jurl, jname, jid))
        if sha_had_failure:
            n_with_failures += 1

    def _job_link(sha: str, url: str | None, jname: str, jid: int | None) -> str:
        target = url or page_paths.get(sha, "#")
        label = f"job#{jid}" if jid else f"job@{short_sha(sha)}"
        return (
            f"<a href='{_html_escape(target)}' target='_blank' rel='noopener noreferrer' "
            f"title='{_html_escape(jname)} ({short_sha(sha)})'>{label}</a>"
        )

    def _jobs_html(occurrences: list, limit: int = 10) -> str:
        seen, ordered = set(), []
        for sha, url, jname, jid in occurrences:
            key = (sha, url)
            if key in seen:
                continue
            seen.add(key)
            ordered.append((sha, url, jname, jid))
        head = ", ".join(_job_link(s, u, n, j) for s, u, n, j in ordered[:limit])
        more = (
            f", <span class='muted'>+{len(ordered) - limit} more</span>"
            if len(ordered) > limit else ""
        )
        return head + more

    ranked = sorted(test_count.items(), key=lambda kv: (-kv[1], kv[0]))
    total_test_occurrences = sum(test_count.values())

    # History bar: per-test, oldest→newest across cached commits.
    # 'g' (green) = test did not fail in this SHA, 'r' (red) = failed,
    # '.' (gray) = SHA has no jobs data at all (not yet fetched).
    items_old_first = list(reversed(items))

    # Map test_id → set of job names that have ever hosted this test (i.e. it
    # failed in those jobs at some point in the cached window). Used to
    # distinguish "test ran and passed" from "the host-job was cancelled /
    # didn't run, so we don't actually know if the test passed".
    test_host_jobs: dict[str, set[str]] = {}
    for sha, e in items:
        if kind == "pre-merge":
            for a in e.get("attempts") or []:
                for j in a.get("jobs") or []:
                    name = j.get("name") or ""
                    for t in j.get("failed_tests") or []:
                        if t and name:
                            test_host_jobs.setdefault(t, set()).add(name)
        else:
            for j in e.get("jobs") or e.get("failed_jobs") or []:
                name = j.get("name") or ""
                for t in j.get("failed_tests") or []:
                    if t and name:
                        test_host_jobs.setdefault(t, set()).add(name)

    def _did_test_fail_in(
        sha: str, e: dict, test_id: str
    ) -> tuple[bool, bool, str | None]:
        """Return (test_observed, test_failed, first_failed_job_url).
        test_observed=False means this test's host-job didn't actually complete
        successfully on this SHA — could be cancelled, missing, or just no
        data — so we paint gray ("no signal") instead of green."""
        host_jobs = test_host_jobs.get(test_id, set())
        observed = False
        if kind == "pre-merge":
            for a in e.get("attempts") or []:
                for j in a.get("jobs") or []:
                    name = j.get("name") or ""
                    if test_id in (j.get("failed_tests") or []):
                        return (True, True, j.get("url"))
                    if name in host_jobs and j.get("conclusion") == "success":
                        observed = True
        else:
            for j in e.get("jobs") or e.get("failed_jobs") or []:
                name = j.get("name") or ""
                if test_id in (j.get("failed_tests") or []):
                    return (True, True, j.get("url"))
                if name in host_jobs and j.get("conclusion") == "success":
                    observed = True
        return (observed, False, None)

    def _hb_tooltip(sha: str, status_text: str) -> str:
        """Compose a tooltip with PR, author, timestamp, status."""
        m = sha_meta.get(sha) or {}
        bits: list[str] = []
        pr = m.get("pr")
        if pr:
            bits.append(f"PR #{pr}")
        bits.append(short_sha(sha))
        author = m.get("author")
        if author:
            bits.append(author)
        ts = m.get("ts")
        if ts and ts != "—":
            bits.append(ts)
        bits.append(status_text)
        return _html_escape(" • ".join(bits))

    def _history_bar(test_id: str) -> str:
        cells: list[str] = []
        for sha, e in items_old_first:
            observed, failed, fail_url = _did_test_fail_in(sha, e, test_id)
            base = (
                "display:inline-block; width:9px; height:11px; "
                "vertical-align:middle; margin-right:1px;"
            )
            if not observed and not failed:
                _tip = _hb_tooltip(sha, "test did not run (host job cancelled or missing)")
                cells.append(
                    f"<span class='hb-cell' style='{base} background:#d0d7de;' "
                    f"title='{_tip}'></span>"
                )
            elif failed:
                inner = (
                    f"<span class='hb-cell' style='{base} background:#c83a3a;' "
                    f"title='{_hb_tooltip(sha, 'FAILED — click to open job log')}'></span>"
                )
                if fail_url:
                    cells.append(
                        f"<a class='hb-link' href='{_html_escape(fail_url)}' "
                        f"target='_blank' rel='noopener noreferrer'>{inner}</a>"
                    )
                else:
                    cells.append(inner)
            else:
                cells.append(
                    f"<span class='hb-cell' style='{base} background:#2da44e;' "
                    f"title='{_hb_tooltip(sha, 'passed')}'></span>"
                )
        return (
            "<span style='display:inline-block;white-space:nowrap;"
            "font-family:\"SF Mono\",Consolas,monospace;font-size:10px;'>"
            + "".join(cells)
            + "</span>"
        )

    test_rows: list[str] = []
    for rank, (test_id, n) in enumerate(ranked, 1):
        test_rows.append(
            f"<tr><td><strong>{rank}</strong></td>"
            f"<td class='num-nz'>{n} / {total_test_occurrences}</td>"
            f"<td><code>{_html_escape(test_id)}</code></td></tr>"
        )
        # History row directly below: bars (oldest left → newest right),
        # followed by a label. Each red cell links to the failing job log.
        test_rows.append(
            f"<tr style='background:#fafbfc;'><td></td><td></td>"
            f"<td style='padding-top:0; padding-bottom:6px; "
            f"color:#586069; font-size:11px;'>"
            f"{_history_bar(test_id)}"
            f" <span style='color:#586069;font-size:10px;'>most recent PR &#10145;</span>"
            f"</td></tr>"
        )
    if not test_rows:
        test_rows.append(
            "<tr><td colspan='3' style='color:#586069;text-align:center;padding:8px;'>"
            f"No pytest failures across the {len(items)} cached commit(s)."
            "</td></tr>"
        )

    body_style = """
<style>
  body.report-body { padding-left: 80px; padding-right: 48px; }
  body.report-body table.attempt-table th:nth-child(3),
  body.report-body table.attempt-table td:nth-child(3) { text-align: left; }
  body.report-body table.attempt-table th:nth-child(4),
  body.report-body table.attempt-table td:nth-child(4) {
    text-align: left; white-space: nowrap;
  }
  body.report-body table.report-table-tests td:nth-child(3) { white-space: normal; }
  /* History-bar links: no arrow, no underline, cells stay flush. */
  body.report-body a.hb-link { text-decoration: none; }
  body.report-body a.hb-link[target="_blank"]::after { content: none; }
</style>
"""

    return "\n".join([
        "<!DOCTYPE html>",
        f"<html><head><meta charset='utf-8'><title>{kind} report</title>",
        _HTML_STYLE,
        body_style,
        "</head><body class='report-body'>",
        f"<h1>Last {len(items)} {kind.capitalize()} Report &mdash; main branch</h1>",
        f"<div class='meta'>"
        f"Aggregated across {len(items)} cached commit(s); "
        f"{n_attempts} run(s) total, {n_with_failures} commit(s) with at least "
        f"one failure ({n_failed_jobs} failed job-run(s))."
        f"</div>",
        f"<h2>Failing tests <small style='color:#586069;font-weight:400'>"
        f"(pytest test-id occurrences across all cached commits; count / total)</small></h2>",
        "<table class='attempt-table report-table-tests'>",
        "<thead><tr><th>Rank</th><th>Count / Total</th>"
        "<th>Test &mdash; <small style='font-weight:400;color:#586069;'>red cell in history bar links to the failing job log</small></th></tr></thead>",
        "<tbody>",
        *test_rows,
        "</tbody></table>",
        "<h2>Per-SHA index</h2>",
        "<table>",
        thead,
        "<tbody>",
        *rows,
        "</tbody></table>",
        "</body></html>",
    ])


def _filename_premerge(sha: str) -> str:
    return f"{short_sha(sha)}-premerge.html"


def _filename_postmerge(sha: str) -> str:
    return f"{short_sha(sha)}-postmerge.html"


def cmd_render_html(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    do_pre, do_post = _resolve_flags(args.premerge, args.postmerge)
    n_pre = n_post = 0
    cap = max(1, int(args.past))

    far_past = datetime.min.replace(tzinfo=timezone.utc)

    if do_pre:
        db = _load_db(PRE_MERGE_DB)
        # Most-recent first by last attempt's created_at; cap at args.past.
        sorted_pre = sorted(
            db.items(),
            key=lambda kv: (
                _to_pt((kv[1].get("attempts") or [{}])[-1].get("created_at")) or far_past,
                kv[0],
            ),
            reverse=True,
        )[:cap]
        kept_pre = dict(sorted_pre)
        page_paths: dict[str, str] = {}
        for sha, e in kept_pre.items():
            atts = e.get("attempts") or []
            last_ca = atts[-1].get("created_at") if atts else None
            date_str = _date_dir_for(last_ca)
            ddir = output_root / date_str
            ddir.mkdir(parents=True, exist_ok=True)
            fname = _filename_premerge(sha)
            (ddir / fname).write_text(_render_premerge_page(sha, e))
            page_paths[sha] = f"{date_str}/{fname}"
            n_pre += 1
        (output_root / "pre_merge.html").write_text(
            _render_summary("pre-merge", kept_pre, page_paths)
        )

    if do_post:
        db = _load_db(POST_MERGE_DB)
        sorted_post = sorted(
            db.items(),
            key=lambda kv: (_to_pt(kv[1].get("created_at")) or far_past, kv[0]),
            reverse=True,
        )[:cap]
        kept_post = dict(sorted_post)
        page_paths = {}
        for sha, e in kept_post.items():
            date_str = _date_dir_for(e.get("created_at"))
            ddir = output_root / date_str
            ddir.mkdir(parents=True, exist_ok=True)
            fname = _filename_postmerge(sha)
            (ddir / fname).write_text(_render_postmerge_page(sha, e))
            page_paths[sha] = f"{date_str}/{fname}"
            n_post += 1
        (output_root / "post_merge.html").write_text(
            _render_summary("post-merge", kept_post, page_paths)
        )

    print(
        f"wrote pre-merge={n_pre} post-merge={n_post} per-SHA pages "
        f"under {output_root}"
    )
    return 0


_LOCK_PATH = Path.home() / ".cache" / "dynamo-utils" / "analyze_ci_merges.lock"


def _acquire_single_instance_lock() -> object | None:
    """Non-blocking flock on _LOCK_PATH. Returns the open fd to keep the lock
    held for the rest of the process; returns None and exits cleanly if another
    instance is already running."""
    import fcntl
    _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    f = open(_LOCK_PATH, "w")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        print(
            f"analyze_ci_merges: another instance is running (lock held at {_LOCK_PATH}); exiting.",
            file=sys.stderr,
        )
        sys.exit(0)
    f.write(f"{os.getpid()}\n")
    f.flush()
    return f


def main() -> int:
    p = argparse.ArgumentParser(prog="analyze_ci_merges")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_l = sub.add_parser("lookup", help="show CI data for one SHA")
    p_l.add_argument("target", help="PR number or merge SHA")
    p_l.add_argument("--premerge", action="store_true", help="include pre-merge")
    p_l.add_argument("--postmerge", action="store_true", help="include post-merge")
    p_l.add_argument("--refresh", action="store_true")
    p_l.set_defaults(func=cmd_lookup)

    p_b = sub.add_parser("backfill", help="walk recent main commits and cache")
    p_b.add_argument("--past", type=int, default=50)
    p_b.add_argument("--premerge", action="store_true", help="include pre-merge")
    p_b.add_argument("--postmerge", action="store_true", help="include post-merge")
    p_b.add_argument("--refresh", action="store_true")
    p_b.set_defaults(func=cmd_backfill)

    p_rh = sub.add_parser("render-html", help="render per-SHA + summary HTML")
    p_rh.add_argument("--premerge", action="store_true", help="render pre-merge")
    p_rh.add_argument("--postmerge", action="store_true", help="render post-merge")
    p_rh.add_argument(
        "--past", type=int, default=100,
        help="cap aggregate report + per-SHA pages at the N most-recent SHAs",
    )
    p_rh.add_argument(
        "--output-root",
        default=str(Path.home() / "dynamo" / "commits" / "logs"),
    )
    p_rh.set_defaults(func=cmd_render_html)

    args = p.parse_args()
    # Backfill is the only long-running write operation; gate it behind a
    # single-instance flock so cron + manual runs don't overlap. Other
    # subcommands (lookup, render-html) are short / read-only and can run
    # concurrently with a backfill in flight.
    if args.cmd == "backfill":
        _lock = _acquire_single_instance_lock()  # noqa: F841
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
