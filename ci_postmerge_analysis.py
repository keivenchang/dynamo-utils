#!/usr/bin/env python3
"""Fetch + cache post-merge CI failures for merged main commits.

Subcommands:
  lookup <PR|SHA>            Fetch (or read from cache) post-merge results for one SHA.
  backfill [--past N]        Walk the last N main commits and populate the cache.
  render-html                (placeholder for Phase 3)

Scope: only the "Post-Merge CI Pipeline" workflow (push:main event).  Nightly
runs are intentionally out of scope -- they're cron-driven, not SHA-targeted.

Reuses helpers from revalidate_pr (gh_api / fetch_job_log /
extract_pytest_failures) so log caching and pytest-failure parsing stay
consistent with the re-validation flow.
"""

from __future__ import annotations

import argparse
import json
import os
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

POST_MERGE_WORKFLOW_NAME = "Post-Merge CI Pipeline"
POST_MERGE_WORKFLOW_FILE = "post-merge-ci.yml"
DB_PATH = Path.home() / ".cache" / "dynamo-utils" / "post-merge.json"

# Conclusions that mean the run is genuinely "done" -- safe to cache forever.
TERMINAL_CONCLUSIONS = {"success", "failure", "cancelled", "timed_out", "skipped"}
JOB_FAILED_CONCLUSIONS = {"failure", "timed_out", "cancelled"}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def load_db() -> dict:
    if not DB_PATH.exists():
        return {}
    with DB_PATH.open("r") as f:
        return json.load(f)


def save_db_atomic(db: dict) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = DB_PATH.with_suffix(f".tmp.{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(db, f, indent=2, sort_keys=True)
    tmp.replace(DB_PATH)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def resolve_sha(arg: str) -> str:
    """Accept a PR number or hex SHA. Returns the merge commit SHA."""
    if arg.isdigit():
        pr_num = int(arg)
        data = gh_api(f"repos/{REPO}/pulls/{pr_num}")
        if not data:
            raise SystemExit(f"PR #{pr_num}: not found")
        merge_sha = data.get("merge_commit_sha")
        if not merge_sha or not data.get("merged"):
            raise SystemExit(f"PR #{pr_num} is not merged (state={data.get('state')})")
        return merge_sha
    if len(arg) < 7 or not all(c in "0123456789abcdef" for c in arg.lower()):
        raise SystemExit(f"argument {arg!r}: not a PR number or hex SHA")
    return arg.lower()


# ---------------------------------------------------------------------------
# Fetch + classify
# ---------------------------------------------------------------------------


def find_post_merge_run(sha: str) -> dict | None:
    """Return the Post-Merge CI Pipeline run for `sha`, or None."""
    runs = gh_api(f"repos/{REPO}/actions/runs?head_sha={sha}&event=push")
    if not runs:
        return None
    for r in runs.get("workflow_runs") or []:
        if r.get("name") == POST_MERGE_WORKFLOW_NAME:
            return r
    return None


def list_failed_jobs(run_id: int) -> list[dict]:
    """Return failed/timed_out/cancelled jobs in the run, latest attempt only."""
    jobs: list[dict] = []
    page = 1
    while True:
        resp = gh_api(
            f"repos/{REPO}/actions/runs/{run_id}/jobs?per_page=100&filter=latest&page={page}"
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
    return [j for j in jobs if j.get("conclusion") in JOB_FAILED_CONCLUSIONS]


def collect_failures(run: dict) -> list[dict]:
    """Build the per-job failure list for cache/render."""
    failed = list_failed_jobs(run["id"])
    out = []
    for j in failed:
        log = fetch_job_log(j["id"])
        tests = extract_pytest_failures(log) if log else []
        out.append({
            "name": j["name"],
            "id": j["id"],
            "url": j.get("html_url"),
            "conclusion": j.get("conclusion"),
            "failed_tests": tests,
        })
    return out


def fetch_for_sha(sha: str) -> dict | None:
    """Fetch fresh post-merge data for one SHA. Returns None if no run exists."""
    run = find_post_merge_run(sha)
    if run is None:
        return None
    return {
        "run_id": run["id"],
        "html_url": run.get("html_url"),
        "status": run.get("status"),
        "conclusion": run.get("conclusion"),
        "created_at": run.get("created_at"),
        "updated_at": run.get("updated_at"),
        "failed_jobs": collect_failures(run),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def is_terminal(entry: dict) -> bool:
    return entry.get("status") == "completed" and entry.get("conclusion") in TERMINAL_CONCLUSIONS


def get_or_fetch(sha: str, db: dict, refresh: bool = False) -> dict | None:
    """Return cached data if terminal; otherwise re-fetch and cache when terminal."""
    cached = db.get(sha)
    if cached and is_terminal(cached) and not refresh:
        return cached
    fresh = fetch_for_sha(sha)
    if fresh is None:
        return cached  # Keep stale entry if API now empty (don't drop history).
    if is_terminal(fresh):
        db[sha] = fresh
        save_db_atomic(db)
    return fresh


# ---------------------------------------------------------------------------
# CLI: lookup
# ---------------------------------------------------------------------------


def print_human(sha: str, entry: dict) -> int:
    print(f"SHA          : {sha}")
    print(f"Run          : {entry.get('html_url')}")
    print(f"Status       : {entry.get('status')} / conclusion={entry.get('conclusion')}")
    failed = entry.get("failed_jobs") or []
    print(f"Failed jobs  : {len(failed)}")
    if not failed:
        print("(no failed jobs)")
        return 0 if entry.get("conclusion") == "success" else 1
    for j in failed:
        print()
        print(f"  [{j.get('conclusion')}] {j['name']}")
        print(f"    {j.get('url')}")
        tests = j.get("failed_tests") or []
        if tests:
            for t in tests:
                print(f"      FAILED  {t}")
        else:
            print("      (no pytest test IDs extracted; see log for non-test failure)")
    return 1


def cmd_lookup(args: argparse.Namespace) -> int:
    sha = resolve_sha(args.target)
    db = load_db()
    entry = get_or_fetch(sha, db, refresh=args.refresh)
    if entry is None:
        print(
            f"no Post-Merge CI Pipeline run found for {short_sha(sha)} "
            f"(SHA must be on main; push event indexes the run)",
            file=sys.stderr,
        )
        return 2
    if args.json:
        print(json.dumps({"sha": sha, **entry}, indent=2))
        return 0 if entry.get("conclusion") == "success" else 1
    return print_human(sha, entry)


# ---------------------------------------------------------------------------
# CLI: backfill
# ---------------------------------------------------------------------------


def cmd_backfill(args: argparse.Namespace) -> int:
    """Walk the most recent N main commits and populate the cache."""
    n = args.past
    commits = gh_api(f"repos/{REPO}/commits?sha=main&per_page={n}")
    if not commits:
        print("no commits returned from API", file=sys.stderr)
        return 2
    db = load_db()
    refreshed = skipped = none_run = pending = 0
    for c in commits[:n]:
        sha = c["sha"]
        cached = db.get(sha)
        if cached and is_terminal(cached) and not args.refresh:
            skipped += 1
            continue
        entry = fetch_for_sha(sha)
        if entry is None:
            none_run += 1
            continue
        if is_terminal(entry):
            db[sha] = entry
            refreshed += 1
        else:
            pending += 1
    save_db_atomic(db)
    print(
        f"refreshed: {refreshed}  cached-skip: {skipped}  "
        f"in-progress: {pending}  no-run: {none_run}  total: {n}"
    )
    return 0


# ---------------------------------------------------------------------------
# CLI: render-html (Phase 3 placeholder)
# ---------------------------------------------------------------------------


def cmd_render_html(args: argparse.Namespace) -> int:
    print("render-html: not yet implemented (Phase 3)", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(prog="ci_postmerge_analysis")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_lookup = sub.add_parser("lookup", help="fetch (or read cache) post-merge for one SHA")
    p_lookup.add_argument("target", help="PR number or commit SHA")
    p_lookup.add_argument("--refresh", action="store_true", help="bypass cache and re-fetch")
    p_lookup.add_argument("--json", action="store_true", help="JSON output")
    p_lookup.set_defaults(func=cmd_lookup)

    p_back = sub.add_parser("backfill", help="walk recent main commits and cache results")
    p_back.add_argument("--past", type=int, default=50, help="how many commits to walk (default: 50)")
    p_back.add_argument("--refresh", action="store_true", help="re-fetch even when cached")
    p_back.set_defaults(func=cmd_backfill)

    p_rh = sub.add_parser("render-html", help="render per-SHA + summary HTML (Phase 3)")
    p_rh.set_defaults(func=cmd_render_html)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
