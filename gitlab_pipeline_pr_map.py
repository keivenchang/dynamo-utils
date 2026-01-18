#!/usr/bin/env python3
"""
Map GitLab pipeline URLs/IDs -> commit SHA -> Merge Request IID(s) ("PR#").

This is useful when you have a list of pipeline URLs like:
  https://gitlab-master.nvidia.com/dl/ai-dynamo/dynamo/-/pipelines/40743226
and want to link each pipeline back to the MR(s) that contain its SHA.

Auth:
  - Export GITLAB_TOKEN, or write it to ~/.config/gitlab-token

Examples:
  # Read pipeline URLs from stdin, output CSV
  cat pipelines.txt | python3 gitlab_pipeline_pr_map.py --format csv > out.csv

  # Mix URLs and IDs
  python3 gitlab_pipeline_pr_map.py 40743226 38895507 --format table

  # For a specific project
  python3 gitlab_pipeline_pr_map.py --project-id 169905 40743226
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from common_gitlab import GitLabAPIClient


DEFAULT_BASE_URL = "https://gitlab-master.nvidia.com"
# This repo hard-codes 169905 elsewhere (see common.py); keep it as the default.
DEFAULT_PROJECT_ID = "169905"


PIPELINE_ID_RE = re.compile(r"/-/pipelines/(?P<id>\d+)(?:\b|/|$)")
MR_REF_RE = re.compile(r"^refs/merge-requests/(?P<iid>\d+)/(?:head|merge)$")


@dataclass(frozen=True)
class MRInfo:
    iid: int
    title: str
    state: str
    web_url: str


@dataclass(frozen=True)
class PipelineToMRRow:
    pipeline_id: int
    pipeline_web_url: str
    status: str
    source: str
    ref: str
    sha: str
    created_at: str
    updated_at: str
    mr_iids: str
    mr_states: str
    mr_titles: str
    mr_web_urls: str


def _stringify_list(vals: Sequence[str]) -> str:
    return "; ".join([v for v in vals if v])


def _extract_pipeline_id(item: str) -> Optional[int]:
    item = item.strip()
    if not item:
        return None
    if item.isdigit():
        return int(item)
    m = PIPELINE_ID_RE.search(item)
    if m:
        return int(m.group("id"))
    return None


def _iter_input_items(positional: Sequence[str]) -> List[str]:
    if positional:
        return list(positional)
    # Read from stdin (one per line), but don't block if user forgot to pipe.
    if sys.stdin.isatty():
        return []
    return [ln.strip() for ln in sys.stdin.read().splitlines() if ln.strip()]


def _project_selector(project_id: Optional[str], project_path: Optional[str]) -> str:
    if project_id and project_path:
        raise ValueError("Use only one of --project-id or --project-path")
    if project_id:
        return project_id
    if project_path:
        # GitLab API expects URL-encoded path in /projects/:id
        # Slash must be encoded as %2F.
        return project_path.replace("/", "%2F")
    return DEFAULT_PROJECT_ID


def _get_with_retries(
    client: GitLabAPIClient,
    endpoint: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    retries: int = 4,
    backoff_s: float = 0.75,
) -> Optional[Any]:
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            return client.get(endpoint, params=params)
        except Exception as e:
            last_err = e
            msg = str(e)
            # Best-effort handling for rate limiting / transient issues.
            retryable = ("429" in msg) or ("502" in msg) or ("503" in msg) or ("504" in msg)
            if attempt >= retries or not retryable:
                break
            time.sleep(backoff_s * (2**attempt))
    if last_err:
        raise last_err
    return None


def fetch_pipeline_details(
    client: GitLabAPIClient,
    project: str,
    pipeline_id: int,
) -> Dict[str, Any]:
    endpoint = f"/api/v4/projects/{project}/pipelines/{pipeline_id}"
    data = _get_with_retries(client, endpoint)
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected pipeline response for {pipeline_id}: {type(data)}")
    return data


def fetch_merge_requests_for_sha(
    client: GitLabAPIClient,
    project: str,
    sha: str,
) -> List[MRInfo]:
    # GitLab supports this endpoint in modern versions; if it 404s, caller will fallback.
    endpoint = f"/api/v4/projects/{project}/repository/commits/{sha}/merge_requests"
    data = _get_with_retries(client, endpoint, params={"per_page": 100})
    if not data:
        return []
    if not isinstance(data, list):
        return []
    out: List[MRInfo] = []
    for mr in data:
        if not isinstance(mr, dict):
            continue
        iid = mr.get("iid")
        if not isinstance(iid, int):
            continue
        out.append(
            MRInfo(
                iid=iid,
                title=str(mr.get("title") or ""),
                state=str(mr.get("state") or ""),
                web_url=str(mr.get("web_url") or ""),
            )
        )
    return out


def mr_fallback_from_ref(pipeline_ref: str) -> List[MRInfo]:
    m = MR_REF_RE.match(pipeline_ref or "")
    if not m:
        return []
    iid = int(m.group("iid"))
    # We don't know title/state/url without more calls; keep minimally useful info.
    return [MRInfo(iid=iid, title="", state="", web_url="")]


def pipeline_to_row(
    client: GitLabAPIClient,
    project: str,
    pipeline_id: int,
) -> PipelineToMRRow:
    p = fetch_pipeline_details(client, project, pipeline_id)
    sha = str(p.get("sha") or "")
    ref = str(p.get("ref") or "")

    mrs: List[MRInfo] = []
    if sha:
        try:
            mrs = fetch_merge_requests_for_sha(client, project, sha)
        except Exception:
            # Fall back to parsing MR IID from refs/merge-requests/<iid>/* if present.
            mrs = mr_fallback_from_ref(ref)
    else:
        mrs = mr_fallback_from_ref(ref)

    mr_iids = _stringify_list([str(mr.iid) for mr in mrs])
    mr_states = _stringify_list([mr.state for mr in mrs])
    mr_titles = _stringify_list([mr.title for mr in mrs])
    mr_web_urls = _stringify_list([mr.web_url for mr in mrs])

    return PipelineToMRRow(
        pipeline_id=int(p.get("id") or pipeline_id),
        pipeline_web_url=str(p.get("web_url") or ""),
        status=str(p.get("status") or ""),
        source=str(p.get("source") or ""),
        ref=ref,
        sha=sha,
        created_at=str(p.get("created_at") or ""),
        updated_at=str(p.get("updated_at") or ""),
        mr_iids=mr_iids,
        mr_states=mr_states,
        mr_titles=mr_titles,
        mr_web_urls=mr_web_urls,
    )


def _print_table(rows: Sequence[PipelineToMRRow]) -> None:
    cols = [
        ("pipeline_id", 11),
        ("status", 10),
        ("source", 18),
        ("ref", 20),
        ("sha", 10),
        ("mr_iids", 12),
    ]
    header = "  ".join([name.ljust(width) for name, width in cols])
    print(header)
    print("-" * len(header))
    for r in rows:
        vals = {
            "pipeline_id": str(r.pipeline_id),
            "status": r.status,
            "source": r.source,
            "ref": r.ref,
            "sha": (r.sha[:10] if r.sha else ""),
            "mr_iids": r.mr_iids,
        }
        print("  ".join([vals[name][:width].ljust(width) for name, width in cols]))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Map GitLab pipeline URLs/IDs to MR IID(s).")
    ap.add_argument("items", nargs="*", help="Pipeline URLs or numeric IDs. If omitted, read from stdin.")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"GitLab base URL (default: {DEFAULT_BASE_URL})")
    ap.add_argument("--token", default=None, help="GitLab token (else use GITLAB_TOKEN or ~/.config/gitlab-token)")
    ap.add_argument("--project-id", default=None, help=f"GitLab project numeric ID (default: {DEFAULT_PROJECT_ID})")
    ap.add_argument("--project-path", default=None, help='GitLab project path, e.g. "dl/ai-dynamo/dynamo"')
    ap.add_argument("--format", choices=["csv", "json", "table"], default="csv", help="Output format")
    ap.add_argument("--workers", type=int, default=8, help="Parallel workers (default: 8)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    input_items = _iter_input_items(args.items)
    pipeline_ids: List[int] = []
    for it in input_items:
        pid = _extract_pipeline_id(it)
        if pid is not None:
            pipeline_ids.append(pid)

    if not pipeline_ids:
        ap.error("No pipeline IDs found. Provide URLs/IDs or pipe a list into stdin.")

    project = _project_selector(args.project_id, args.project_path)
    client = GitLabAPIClient(token=args.token, base_url=args.base_url)

    rows: List[PipelineToMRRow] = []
    # Keep stable-ish output ordering: we gather then sort by pipeline_id.
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(pipeline_to_row, client, project, pid): pid for pid in pipeline_ids}
        for fut in as_completed(futs):
            pid = futs[fut]
            try:
                rows.append(fut.result())
            except Exception as e:
                # Emit a best-effort row so CSV consumers can still join on pipeline_id.
                rows.append(
                    PipelineToMRRow(
                        pipeline_id=pid,
                        pipeline_web_url="",
                        status="error",
                        source="",
                        ref="",
                        sha="",
                        created_at="",
                        updated_at="",
                        mr_iids="",
                        mr_states="",
                        mr_titles=str(e),
                        mr_web_urls="",
                    )
                )

    rows.sort(key=lambda r: r.pipeline_id)

    if args.format == "json":
        print(json.dumps([asdict(r) for r in rows], indent=2))
        return 0

    if args.format == "table":
        _print_table(rows)
        return 0

    # CSV
    fieldnames = list(asdict(rows[0]).keys())
    w = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(asdict(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())















