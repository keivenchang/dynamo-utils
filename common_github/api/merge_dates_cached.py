"""PR merge dates cached helper.

This is a pure "cache + transform" helper used by some dashboards to show when a PR was merged.
We keep the cache per-PR number since merge time is immutable once set.

Cache:
  MERGE_DATES_CACHE key: "{owner}/{repo}:{pr_number}"
  value: merge_date string (Pacific time, "%Y-%m-%d %H:%M:%S") or empty string for "not merged"
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, List, Optional, Set, TYPE_CHECKING
from zoneinfo import ZoneInfo

from ..cache_merge_dates import MERGE_DATES_CACHE

if TYPE_CHECKING:  # pragma: no cover
    from .. import GitHubAPIClient


TTL_POLICY_DESCRIPTION = "365d (immutable once merged; non-merged cached as empty)"


def _unique_ints(nums: Iterable[int]) -> List[int]:
    out: List[int] = []
    seen: Set[int] = set()
    for n in nums:
        try:
            i = int(n)
        except (ValueError, TypeError):
            continue
        if i <= 0 or i in seen:
            continue
        seen.add(i)
        out.append(i)
    return out


def _merge_date_pacific(merged_at_iso: str) -> str:
    # merged_at looks like "2025-12-18T12:34:56Z" (UTC)
    dt_utc = datetime.fromisoformat(str(merged_at_iso).replace("Z", "+00:00"))
    dt_pacific = dt_utc.astimezone(ZoneInfo("America/Los_Angeles"))
    return dt_pacific.strftime("%Y-%m-%d %H:%M:%S")


def get_cached_pr_merge_dates_cached(
    api: "GitHubAPIClient",
    *,
    pr_numbers: List[int],
    owner: str = "ai-dynamo",
    repo: str = "dynamo",
    ttl_s: int = 365 * 24 * 3600,
) -> Dict[int, Optional[str]]:
    nums = _unique_ints(pr_numbers or [])
    if not nums:
        return {}

    result: Dict[int, Optional[str]] = {}
    missing: List[int] = []

    for prn in nums:
        cache_key = f"{owner}/{repo}:{prn}"
        cached = MERGE_DATES_CACHE.get_if_fresh(cache_key, ttl_s=int(ttl_s))
        if cached is not None:
            # cache stores "" for "not merged"
            result[prn] = str(cached) if str(cached).strip() else None
            api._cache_hit("merge_dates.disk")
        else:
            missing.append(prn)
            api._cache_miss("merge_dates")

    if not missing:
        return result

    # Prefer list endpoint: 1-3 API calls instead of N.
    all_prs_data = api.list_pull_requests(owner, repo, state="all", ttl_s=3600)
    pr_num_to_data: Dict[int, dict] = {}
    if isinstance(all_prs_data, list):
        for pr_data in all_prs_data:
            if not isinstance(pr_data, dict):
                continue
            n = pr_data.get("number")
            try:
                ni = int(n)
            except (ValueError, TypeError):
                continue
            pr_num_to_data[ni] = pr_data

    still_missing: List[int] = []
    for prn in missing:
        pr_data = pr_num_to_data.get(prn)
        if isinstance(pr_data, dict) and pr_data.get("merged_at"):
            merge_date = _merge_date_pacific(str(pr_data.get("merged_at") or ""))
            result[prn] = merge_date
            MERGE_DATES_CACHE.put(f"{owner}/{repo}:{prn}", merge_date)
        elif isinstance(pr_data, dict):
            result[prn] = None
            MERGE_DATES_CACHE.put(f"{owner}/{repo}:{prn}", None)
        else:
            still_missing.append(prn)

    # Fallback: individual fetch for PRs not found in list.
    for prn in still_missing:
        pr_details = api.get_pr_details(owner, repo, prn)
        if isinstance(pr_details, dict) and pr_details.get("merged_at"):
            merge_date = _merge_date_pacific(str(pr_details.get("merged_at") or ""))
            result[prn] = merge_date
            MERGE_DATES_CACHE.put(f"{owner}/{repo}:{prn}", merge_date)
        else:
            result[prn] = None
            MERGE_DATES_CACHE.put(f"{owner}/{repo}:{prn}", None)

    return result

