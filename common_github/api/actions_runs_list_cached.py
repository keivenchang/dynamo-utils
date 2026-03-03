# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cached listing of GitHub Actions workflow runs (per-run granular cache).

Resource:
  GET /repos/{owner}/{repo}/actions/runs?event={event}&per_page=100

Caching strategy (per-run, incremental fetch):
  Each workflow run is cached individually, keyed by run_id.
  All runs cached for 60 days (this is a quarterly reporting tool, not a
  real-time dashboard). On fetch, we paginate newest-first and stop only
  when we reach the since_date cutoff or exhaust all pages. This means:
  - First call: full pagination back to since_date (~20 pages for 30 days).
  - Same window again: 0 API calls (all cached, oldest on page < since_date).
  - Wider window (e.g. 3d -> 14d): fetches only the uncovered tail pages.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from cache.cache_base import BaseDiskCache

if TYPE_CHECKING:
    from .. import GitHubAPIClient

CACHE_NAME = "actions_runs_list"
CACHE_FILE_DEFAULT = "actions_runs_list.json"

RUN_TTL_S = 60 * 24 * 3600  # 60 days for all runs (quarterly reporting tool)

logger = logging.getLogger(__name__)


def _slim_run(run: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only the fields we need from a workflow run dict."""
    actor = run.get("actor") or {}
    prs = run.get("pull_requests") or []
    pr_numbers = []
    for pr in prs:
        if isinstance(pr, dict):
            n = pr.get("number")
            if n is not None:
                pr_numbers.append(int(n))

    return {
        "id": run.get("id"),
        "name": str(run.get("name") or ""),
        "head_branch": str(run.get("head_branch") or ""),
        "head_sha": str(run.get("head_sha") or ""),
        "event": str(run.get("event") or ""),
        "status": str(run.get("status") or ""),
        "conclusion": run.get("conclusion"),
        "run_attempt": int(run.get("run_attempt", 1) or 1),
        "run_number": int(run.get("run_number", 0) or 0),
        "created_at": str(run.get("created_at") or ""),
        "updated_at": str(run.get("updated_at") or ""),
        "run_started_at": str(run.get("run_started_at") or ""),
        "actor_login": str(actor.get("login") or ""),
        "pr_numbers": pr_numbers,
        "html_url": str(run.get("html_url") or ""),
    }


class _ActionsRunsCache(BaseDiskCache):
    """Per-run disk cache. Keys: run:<run_id> for individual runs."""

    _SCHEMA_VERSION = 2

    def __init__(self, *, cache_file: Path):
        super().__init__(cache_file=cache_file, schema_version=self._SCHEMA_VERSION)

    def get_run(self, run_id: int) -> Optional[Dict[str, Any]]:
        """Get a cached run if within 60-day TTL."""
        key = f"run:{run_id}"
        with self._mu:
            self._load_once()
            ent = self._get_items().get(key)
            if not isinstance(ent, dict):
                return None
            run_data = ent.get("run")
            if not isinstance(run_data, dict):
                return None
            ts = int(ent.get("ts", 0) or 0)
            now = int(time.time())
            if ts and (now - ts) <= RUN_TTL_S:
                return run_data
            return None

    def put_run(self, run_id: int, run_data: Dict[str, Any]) -> None:
        key = f"run:{run_id}"
        with self._mu:
            self._load_once()
            self._set_item(key, {"ts": int(time.time()), "run": run_data})

    def get_all_runs_for_query(
        self, event: Optional[str], branch: Optional[str], since_date: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Return all cached runs matching filters, sorted newest-first."""
        with self._mu:
            self._load_once()
            items = self._get_items()

        now = int(time.time())
        results = []
        for k, v in items.items():
            if not k.startswith("run:"):
                continue
            if not isinstance(v, dict):
                continue
            run = v.get("run")
            if not isinstance(run, dict):
                continue
            ts = int(v.get("ts", 0) or 0)
            if ts and (now - ts) > RUN_TTL_S:
                continue
            if event and run.get("event") != event:
                continue
            if branch and run.get("head_branch") != branch:
                continue
            if since_date and run.get("created_at", "") < since_date:
                continue
            results.append(run)

        results.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return results

    def persist_now(self) -> None:
        with self._mu:
            self._dirty = True
            self._persist()


def _get_cache_file() -> Path:
    try:
        _module_dir = Path(__file__).resolve().parent.parent
        if str(_module_dir) not in sys.path:
            sys.path.insert(0, str(_module_dir))
        import common
        return common.dynamo_utils_cache_dir() / CACHE_FILE_DEFAULT
    except ImportError:
        return Path.home() / ".cache" / "dynamo-utils" / CACHE_FILE_DEFAULT


_CACHE = _ActionsRunsCache(cache_file=_get_cache_file())


def _fetch_window(
    api: "GitHubAPIClient",
    endpoint: str,
    event: Optional[str],
    branch: Optional[str],
    created_filter: str,
) -> Tuple[int, int]:
    """Fetch runs for a time window.  Returns (new_runs, total_count).

    *created_filter* can be a day (``2026-02-24``) or an hourly range
    (``2026-02-24T00:00:00Z..2026-02-24T00:59:59Z``).
    """
    new_runs = 0
    total_count = 0
    for page in range(1, 20):
        params: Dict[str, Any] = {"per_page": 100, "page": page, "created": created_filter}
        if event:
            params["event"] = event
        if branch:
            params["branch"] = branch

        resp = api._rest_get(endpoint, params=params, timeout=10)
        if resp.status_code < 200 or resp.status_code >= 300:
            break

        data = resp.json() if hasattr(resp, "json") else {}
        if page == 1:
            total_count = int(data.get("total_count", 0))

        runs_raw = data.get("workflow_runs")
        if not isinstance(runs_raw, list) or not runs_raw:
            break

        all_cached = True
        for raw_run in runs_raw:
            if not isinstance(raw_run, dict):
                continue
            slim = _slim_run(raw_run)
            run_id = slim.get("id")
            if run_id is None:
                continue
            if _CACHE.get_run(int(run_id)) is None:
                all_cached = False
                _CACHE.put_run(int(run_id), slim)
                new_runs += 1

        if len(runs_raw) < 100:
            break
        if all_cached:
            break

    if new_runs > 0:
        _CACHE.persist_now()

    return new_runs, total_count


def _fetch_day(
    api: "GitHubAPIClient",
    endpoint: str,
    event: Optional[str],
    branch: Optional[str],
    day_str: str,
) -> int:
    """Fetch all runs for a single day, auto-subdividing into hours if needed.

    GitHub caps paginated results at 1,000 per query.  For very active repos
    (ai-dynamo/dynamo has ~2,000 runs/day for some event types), a single day
    can exceed this limit.  We detect the cap via ``total_count`` from the API
    response and automatically re-fetch with hourly granularity.
    """
    new_runs, total_count = _fetch_window(api, endpoint, event, branch, day_str)

    if total_count > 900:
        logger.info(
            "Day %s has %d runs (event=%s), subdividing into hourly windows",
            day_str, total_count, event,
        )
        for hour in range(24):
            hour_filter = f"{day_str}T{hour:02d}:00:00Z..{day_str}T{hour:02d}:59:59Z"
            n, _ = _fetch_window(api, endpoint, event, branch, hour_filter)
            new_runs += n

    return new_runs


def list_workflow_runs(
    api: "GitHubAPIClient",
    *,
    owner: str,
    repo: str,
    event: Optional[str] = None,
    branch: Optional[str] = None,
    since_date: Optional[str] = None,
    max_pages: int = 100,
    ttl_s: int = 0,
) -> List[Dict[str, Any]]:
    """Fetch workflow runs with per-run caching and day-by-day pagination.

    GitHub's Actions API caps paginated results at 1,000 per query.  We
    fetch each day independently; if a day has >900 runs (detected via
    total_count from the first API response), it is automatically
    subdivided into 24 hourly windows to bypass the cap.

    Caching: each run is cached on disk for 60 days.  If a window's runs
    are all cached, the fetch stops after one page (the `all_cached`
    optimization in _fetch_window).  A fully-cached 7-day window costs ~7
    API calls (one per day to confirm all-cached) instead of 0, but
    guarantees correctness.

    Returns:
        List of slimmed workflow run dicts, newest first.
    """
    if api.cache_only_mode:
        results = _CACHE.get_all_runs_for_query(event, branch, since_date)
        if results:
            api._cache_hit(CACHE_NAME)
        return results

    endpoint = f"{api.base_url}/repos/{owner}/{repo}/actions/runs"
    total_new = 0

    # Build list of days to fetch: since_date .. today (inclusive)
    if since_date:
        start = datetime.fromisoformat(since_date.replace("Z", "+00:00")).date()
    else:
        start = (datetime.now(timezone.utc) - timedelta(days=30)).date()
    end = datetime.now(timezone.utc).date()

    day = start
    total_pages = 0
    while day <= end:
        day_str = day.isoformat()
        new = _fetch_day(api, endpoint, event, branch, day_str)
        total_new += new
        total_pages += 1
        day += timedelta(days=1)

    if total_new > 0:
        _CACHE.persist_now()
        api._cache_write(CACHE_NAME, entries=total_new)
    else:
        api._cache_hit(CACHE_NAME)

    results = _CACHE.get_all_runs_for_query(event, branch, since_date)
    logger.info(
        "list_workflow_runs event=%s branch=%s days=%d new=%d total=%d",
        event, branch, (end - start).days + 1, total_new, len(results),
    )
    return results


def get_cache_sizes() -> Tuple[int, int]:
    return _CACHE.get_cache_sizes()
