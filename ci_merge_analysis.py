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
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from revalidate_pr import (  # noqa: E402
    REPO,
    extract_pytest_failures,
    fetch_job_log,
    gh_api,
    short_sha,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRE_MERGE_DB = Path.home() / ".cache" / "dynamo-utils" / "pre-merge.json"
POST_MERGE_DB = Path.home() / ".cache" / "dynamo-utils" / "post-merge.json"
REVALIDATE_DB = Path.home() / ".cache" / "dynamo-utils" / "ci-health.json"

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
    return arg.lower()


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
    return {
        "run_id": run["id"],
        "head_sha": run.get("head_sha"),
        "run_attempt": attempt,
        "html_url": run.get("html_url"),
        "status": run.get("status"),
        "conclusion": run.get("conclusion") if attempt == run.get("run_attempt") else None,
        "created_at": run.get("created_at"),
        "updated_at": run.get("updated_at"),
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
    jobs = _list_run_jobs(run["id"])
    failed_jobs = [_job_record(j) for j in jobs if j.get("conclusion") in JOB_FAILED_CONCLUSIONS]
    return {
        "sha": sha,
        "run_id": run["id"],
        "html_url": run.get("html_url"),
        "status": run.get("status"),
        "conclusion": run.get("conclusion"),
        "run_attempt": run.get("run_attempt") or 1,
        "created_at": run.get("created_at"),
        "updated_at": run.get("updated_at"),
        "failed_jobs": failed_jobs,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def is_postmerge_terminal(entry: dict) -> bool:
    return (
        entry.get("status") == "completed"
        and entry.get("conclusion") in TERMINAL_CONCLUSIONS
    )


# ---------------------------------------------------------------------------
# Source: RE-VALIDATE (read-only view onto revalidate_pr's cache)
# ---------------------------------------------------------------------------


def revalidate_for_sha(sha: str) -> dict | None:
    """Look up the SHA in revalidate_pr's ci-health.json. Returns None if absent."""
    db = _load_db(REVALIDATE_DB)
    e = db.get(sha)
    return e if isinstance(e, dict) else None


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
    if is_premerge_terminal(fresh):
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
        return cached
    if is_postmerge_terminal(fresh):
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
        print(
            f"  att {i:2d}: run={a.get('run_id')}/{a.get('run_attempt')}  "
            f"head={head}  {(a.get('created_at') or '')[:19]}  "
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


def _print_revalidate(entry: dict) -> None:
    if not entry:
        print("  (no re-validate data)")
        return
    atts = entry.get("attempts") or []
    print(f"  PR             : #{entry.get('pr')}")
    print(f"  State          : {entry.get('state')}")
    print(f"  Attempts       : {len(atts)}")
    for i, a in enumerate(atts, 1):
        n_failed = sum(
            1
            for j in (a.get("jobs") or {}).values()
            if isinstance(j, dict) and j.get("conclusion") in JOB_FAILED_CONCLUSIONS
        )
        n_tests = sum(len(t or []) for t in (a.get("failed_tests") or {}).values())
        print(
            f"  att {a.get('attempt', i)}: {(a.get('started_at') or '')[:19]}  "
            f"{n_failed} jobs failed, {n_tests} tests"
        )


def cmd_lookup(args: argparse.Namespace) -> int:
    sha = resolve_sha(args.target)
    print(f"SHA            : {sha}")
    sources = (
        ["pre", "revalidate", "post"] if args.source == "all" else [args.source]
    )
    for src in sources:
        print(f"\n=== {src} ===")
        if src == "pre":
            _print_premerge(get_premerge(sha, refresh=args.refresh))
        elif src == "post":
            _print_postmerge(get_postmerge(sha, refresh=args.refresh))
        elif src == "revalidate":
            _print_revalidate(revalidate_for_sha(sha))
    return 0


# ---------------------------------------------------------------------------
# CLI: backfill
# ---------------------------------------------------------------------------


def cmd_backfill(args: argparse.Namespace) -> int:
    n = args.past
    commits = gh_api(f"repos/{REPO}/commits?sha=main&per_page={n}")
    if not commits:
        print("no commits returned from API", file=sys.stderr)
        return 2
    sources = ["pre", "post"] if args.source == "all" else [args.source]
    for src in sources:
        print(f"=== backfill source={src} past={n} ===")
        if src == "pre":
            db = _load_db(PRE_MERGE_DB)
            refreshed = skipped = pending = no_pr = 0
            for c in commits[:n]:
                sha = c["sha"]
                cached = db.get(sha)
                if cached and is_premerge_terminal(cached) and not args.refresh:
                    skipped += 1
                    continue
                msg = ((c.get("commit") or {}).get("message") or "").splitlines()[0]
                m = _SUBJECT_PR_RE.search(msg)
                if not m:
                    no_pr += 1
                    continue
                entry = fetch_premerge(sha, int(m.group(1)))
                if is_premerge_terminal(entry):
                    db[sha] = entry
                    refreshed += 1
                else:
                    pending += 1
            _save_db_atomic(PRE_MERGE_DB, db)
            print(
                f"  refreshed={refreshed}  cached-skip={skipped}  "
                f"in-progress={pending}  no-pr={no_pr}"
            )
        elif src == "post":
            db = _load_db(POST_MERGE_DB)
            refreshed = skipped = pending = no_run = 0
            for c in commits[:n]:
                sha = c["sha"]
                cached = db.get(sha)
                if cached and is_postmerge_terminal(cached) and not args.refresh:
                    skipped += 1
                    continue
                entry = fetch_postmerge(sha)
                if entry is None:
                    no_run += 1
                    continue
                if is_postmerge_terminal(entry):
                    db[sha] = entry
                    refreshed += 1
                else:
                    pending += 1
            _save_db_atomic(POST_MERGE_DB, db)
            print(
                f"  refreshed={refreshed}  cached-skip={skipped}  "
                f"in-progress={pending}  no-run={no_run}"
            )
    return 0


# ---------------------------------------------------------------------------
# CLI: render-html (placeholder; will mirror revalidate_pr's grid)
# ---------------------------------------------------------------------------


def cmd_render_html(args: argparse.Namespace) -> int:
    print("render-html: not yet implemented", file=sys.stderr)
    return 1


def main() -> int:
    p = argparse.ArgumentParser(prog="ci_merge_analysis")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_l = sub.add_parser("lookup", help="show CI data for one SHA")
    p_l.add_argument("target", help="PR number or merge SHA")
    p_l.add_argument(
        "--source",
        choices=["pre", "post", "revalidate", "all"],
        default="all",
    )
    p_l.add_argument("--refresh", action="store_true")
    p_l.set_defaults(func=cmd_lookup)

    p_b = sub.add_parser("backfill", help="walk recent main commits and cache")
    p_b.add_argument("--past", type=int, default=50)
    p_b.add_argument(
        "--source",
        choices=["pre", "post", "all"],
        default="all",
        help="re-validate is cron-managed, not backfilled here",
    )
    p_b.add_argument("--refresh", action="store_true")
    p_b.set_defaults(func=cmd_backfill)

    p_rh = sub.add_parser("render-html", help="render per-SHA HTML for all sources")
    p_rh.add_argument(
        "--output-root",
        default=str(Path.home() / "dynamo" / "commits" / "logs"),
    )
    p_rh.set_defaults(func=cmd_render_html)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
