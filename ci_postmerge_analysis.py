#!/usr/bin/env python3
"""Fetch post-merge CI failures for a given commit SHA or merged PR.

Phase 1 (this commit): CLI tool, no cache, no HTML.
- Resolves PR -> merge SHA when given a PR number.
- Finds the "Post-Merge CI Pipeline" run for that SHA (push event on main).
- Lists failed jobs and extracts pytest test IDs from each failed job's log.

Reuses helpers from revalidate_pr.py (gh_api, fetch_job_log,
extract_pytest_failures) so log caching and pytest-failure parsing stay
consistent with the re-validation flow.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Helpers from the re-validate flow.  Same package directory.
sys.path.insert(0, str(Path(__file__).parent))
from revalidate_pr import (  # noqa: E402
    REPO,
    extract_pytest_failures,
    fetch_job_log,
    gh_api,
    short_sha,
)

POST_MERGE_WORKFLOW_NAME = "Post-Merge CI Pipeline"


def resolve_sha(arg: str) -> str:
    """Accept a PR number or a SHA. Returns the merge commit SHA.

    PR number: digits only. Look up the merge commit via gh.
    SHA: at least 7 hex chars; pass through after a quick existence probe.
    """
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
    bad = {"failure", "timed_out", "cancelled"}
    return [j for j in jobs if j.get("conclusion") in bad]


def main() -> int:
    p = argparse.ArgumentParser(prog="ci_postmerge_analysis")
    p.add_argument("target", help="PR number or commit SHA")
    p.add_argument(
        "--json",
        action="store_true",
        help="emit JSON instead of human-readable output",
    )
    args = p.parse_args()

    sha = resolve_sha(args.target)
    run = find_post_merge_run(sha)
    if run is None:
        print(
            f"no Post-Merge CI Pipeline run found for {short_sha(sha)} "
            f"(SHA must be on main; push event indexes the run)",
            file=sys.stderr,
        )
        return 2

    run_id = run["id"]
    status = run.get("status")
    conclusion = run.get("conclusion")

    failed = list_failed_jobs(run_id)
    by_job: dict[str, list[str]] = {}
    for j in failed:
        log = fetch_job_log(j["id"])
        tests = extract_pytest_failures(log) if log else []
        by_job[j["name"]] = tests

    if args.json:
        out = {
            "sha": sha,
            "run_id": run_id,
            "run_url": run.get("html_url"),
            "status": status,
            "conclusion": conclusion,
            "failed_jobs": [
                {"name": j["name"], "id": j["id"], "url": j.get("html_url"),
                 "conclusion": j.get("conclusion"), "failed_tests": by_job[j["name"]]}
                for j in failed
            ],
        }
        print(json.dumps(out, indent=2))
        return 0 if conclusion == "success" else 1

    # Human-readable
    print(f"SHA          : {sha}")
    print(f"Run          : {run.get('html_url')}")
    print(f"Status       : {status} / conclusion={conclusion}")
    print(f"Failed jobs  : {len(failed)}")
    if not failed:
        print("(no failed jobs)")
        return 0 if conclusion == "success" else 1
    for j in failed:
        tests = by_job[j["name"]]
        print()
        print(f"  [{j.get('conclusion')}] {j['name']}")
        print(f"    {j.get('html_url')}")
        if tests:
            for t in tests:
                print(f"      FAILED  {t}")
        else:
            print("      (no pytest test IDs extracted; see log for non-test failure)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
