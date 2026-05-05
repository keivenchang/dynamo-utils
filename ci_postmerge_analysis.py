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
import html
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from revalidate_pr import (  # noqa: E402
    REPO,
    _to_pt,
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
# CLI: render-html
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
  .verdict-good { color: #28a745; font-weight: 600; }
  .verdict-bad  { color: #d73a49; font-weight: 600; }
  .verdict-other { color: #bf6c00; font-weight: 600; }
  .pill { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 12px;
          font-weight: 500; }
  .pill-fail { background: #f8d7da; color: #721c24; }
  table { border-collapse: collapse; margin: 8px 0 16px 0; font-size: 13px; width: 100%; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #eaecef; vertical-align: top; }
  th { background: #f6f8fa; font-weight: 600; font-size: 12px; text-transform: uppercase;
       letter-spacing: 0.05em; color: #586069; }
  tr.fail-final td { background: #ea868f; }
  tr.pass td { background: #c3e6cb; }
  details { background: #fff; border: 1px solid #d0d7de; border-radius: 6px;
            margin: 8px 0; padding: 0; }
  details > summary { cursor: pointer; padding: 10px 14px; font-weight: 600;
                      list-style: none; user-select: none; }
  details > summary::-webkit-details-marker { display: none; }
  details > summary::before { content: "▶ "; color: #586069; font-size: 11px;
                              display: inline-block; transition: transform 0.1s; }
  details[open] > summary::before { transform: rotate(90deg); }
  details > .details-body { padding: 0 14px 12px 14px; }
  ul.tests { margin: 6px 0 0 0; padding-left: 20px; }
  ul.tests li { font-family: "SF Mono", Consolas, monospace; font-size: 12px;
                color: #d73a49; padding: 2px 0; }
</style>"""


def _esc(s: str | None) -> str:
    return html.escape(s or "", quote=True)


def _verdict_class(conclusion: str | None) -> str:
    if conclusion == "success":
        return "verdict-good"
    if conclusion in ("failure", "timed_out"):
        return "verdict-bad"
    return "verdict-other"


def _filename_for(sha: str) -> str:
    return f"post_merge-{short_sha(sha)}.html"


def _date_dir_for(entry: dict) -> str:
    """Group per-SHA pages by the post-merge run's start date in PT."""
    dt = _to_pt(entry.get("created_at"))
    return dt.strftime("%Y-%m-%d") if dt else "unknown"


def _render_per_sha_page(sha: str, entry: dict) -> str:
    failed_jobs = entry.get("failed_jobs") or []
    n_tests = sum(len(j.get("failed_tests") or []) for j in failed_jobs)
    conc = entry.get("conclusion")
    vclass = _verdict_class(conc)
    run_url = entry.get("html_url") or "#"
    run_id = entry.get("run_id")
    created_pt = _to_pt(entry.get("created_at"))
    created_str = created_pt.strftime("%Y-%m-%d %H:%M:%S %Z") if created_pt else "—"
    sha_short = short_sha(sha)
    parts: list[str] = [
        "<!DOCTYPE html>",
        f"<html><head><meta charset='utf-8'><title>Post-merge {sha_short}</title>",
        _HTML_STYLE,
        "</head><body>",
        f"<h1>Post-merge run for <code>{sha_short}</code></h1>",
        "<div class='meta'>",
        f"<span>SHA: <code><a href='https://github.com/{REPO}/commit/{sha}' target='_blank'>{sha}</a></code></span>",
        f"<span>Run: <a href='{_esc(run_url)}' target='_blank'>#{run_id}</a></span>",
        f"<span class='{vclass}'>{_esc(conc or 'pending')}</span>",
        f"<span>Started: {_esc(created_str)}</span>",
        "</div>",
    ]
    if not failed_jobs:
        if conc == "success":
            parts.append("<p>All jobs passed.</p>")
        else:
            parts.append(
                f"<p>No failed jobs recorded "
                f"(run conclusion was <code>{_esc(conc or 'unknown')}</code>).</p>"
            )
    else:
        parts.append(
            f"<h2>Failed jobs ({len(failed_jobs)}) "
            f"&middot; {n_tests} pytest test{'s' if n_tests != 1 else ''}</h2>"
        )
        for j in failed_jobs:
            parts.append("<details open>")
            parts.append(
                f"<summary>{_esc(j['name'])} "
                f"<span class='pill pill-fail'>{_esc(j.get('conclusion'))}</span></summary>"
            )
            parts.append("<div class='details-body'>")
            parts.append(
                f"<p><a href='{_esc(j.get('url') or '#')}' target='_blank'>"
                f"view job log on GitHub</a></p>"
            )
            tests = j.get("failed_tests") or []
            if tests:
                parts.append("<ul class='tests'>")
                for t in tests:
                    parts.append(f"<li>{_esc(t)}</li>")
                parts.append("</ul>")
            else:
                parts.append(
                    "<p><em>No pytest test IDs extracted "
                    "(build/setup failure or non-pytest job).</em></p>"
                )
            parts.append("</div></details>")
    parts.append("</body></html>")
    return "\n".join(parts)


def _render_summary_index(db: dict, page_paths: dict[str, str]) -> str:
    # Sort newest first by created_at
    far_past = datetime.min.replace(tzinfo=timezone.utc)
    items = sorted(
        db.items(),
        key=lambda kv: (_to_pt(kv[1].get("created_at")) or far_past, kv[0]),
        reverse=True,
    )
    rows: list[str] = []
    for sha, entry in items:
        conc = entry.get("conclusion")
        vclass = _verdict_class(conc)
        n_failed = len(entry.get("failed_jobs") or [])
        n_tests = sum(len(j.get("failed_tests") or []) for j in (entry.get("failed_jobs") or []))
        created_pt = _to_pt(entry.get("created_at"))
        date_str = created_pt.strftime("%Y-%m-%d %H:%M") if created_pt else "—"
        sha_short = short_sha(sha)
        link = page_paths.get(sha, "#")
        run_url = entry.get("html_url") or "#"
        row_class = "fail-final" if conc in ("failure", "timed_out") else ("pass" if conc == "success" else "")
        rows.append(
            f"<tr class='{row_class}'>"
            f"<td>{_esc(date_str)}</td>"
            f"<td><a href='{_esc(link)}'><code>{sha_short}</code></a></td>"
            f"<td><span class='{vclass}'>{_esc(conc or '?')}</span></td>"
            f"<td>{n_failed}</td>"
            f"<td>{n_tests}</td>"
            f"<td><a href='{_esc(run_url)}' target='_blank'>GH Actions</a></td>"
            f"</tr>"
        )
    return "\n".join([
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>Post-merge summary</title>",
        _HTML_STYLE,
        "</head><body>",
        "<h1>Post-merge runs &mdash; main branch</h1>",
        f"<div class='meta'><span>{len(items)} cached SHA(s)</span></div>",
        "<table>",
        "<thead><tr>"
        "<th>Run started (PT)</th><th>SHA</th><th>Verdict</th>"
        "<th># failed jobs</th><th># failed tests</th><th>Run</th>"
        "</tr></thead>",
        "<tbody>",
        *rows,
        "</tbody></table>",
        "</body></html>",
    ])


def cmd_render_html(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    db = load_db()
    output_root.mkdir(parents=True, exist_ok=True)
    page_paths: dict[str, str] = {}
    n_pages = 0
    for sha, entry in db.items():
        date_str = _date_dir_for(entry)
        date_dir = output_root / date_str
        date_dir.mkdir(parents=True, exist_ok=True)
        out_path = date_dir / _filename_for(sha)
        out_path.write_text(_render_per_sha_page(sha, entry))
        page_paths[sha] = f"{date_str}/{_filename_for(sha)}"
        n_pages += 1
    summary_path = output_root / "post_merge.html"
    summary_path.write_text(_render_summary_index(db, page_paths))
    print(f"wrote {n_pages} per-SHA pages + {summary_path}")
    return 0


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

    p_rh = sub.add_parser("render-html", help="render per-SHA + summary HTML")
    p_rh.add_argument(
        "--output-root",
        default=str(Path.home() / "dynamo" / "commits" / "logs"),
        help="output directory (default: ~/dynamo/commits/logs)",
    )
    p_rh.set_defaults(func=cmd_render_html)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
