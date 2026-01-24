"""Pulls list cached API (REST).

Resource:
  GET /repos/{owner}/{repo}/pulls?state={state}&per_page=100

Cache:
  PULLS_LIST_CACHE (disk + memory)

TTL: Adaptive based on most recent PR's updated_at (<1h=1m, <2h=2m, <4h=4m, >=4h=8m)
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .. import GITHUB_API_STATS
from ..cache_pulls_list import PULLS_LIST_CACHE
from .base_cached import CachedResourceBase

if TYPE_CHECKING:  # pragma: no cover
    from .. import GitHubAPIClient


TTL_POLICY_DESCRIPTION = "adaptive (<1h=1m, <2h=2m, <4h=4m, >=4h=8m)"


class PullsListCached(CachedResourceBase[List[Dict[str, Any]]]):
    def __init__(self, api: "GitHubAPIClient", *, ttl_s: int):
        super().__init__(api)
        self._ttl_s = int(ttl_s)

    @property
    def cache_name(self) -> str:
        return "pulls_list"

    def api_call_format(self) -> str:
        return (
            "REST GET /repos/{owner}/{repo}/pulls?state={state}&per_page=100 (paginated)\n"
            "Example response item (truncated):\n"
            "  {\n"
            "    \"number\": 5530,\n"
            "    \"title\": \"Add feature\",\n"
            "    \"state\": \"open\",\n"
            "    \"updated_at\": \"2026-01-24T03:12:34Z\",\n"
            "    \"head\": {\"ref\": \"feature-branch\", \"sha\": \"abc123...\"},\n"
            "    \"html_url\": \"https://github.com/OWNER/REPO/pull/5530\"\n"
            "  }"
        )

    def inflight_lock_key(self, **kwargs: Any) -> Optional[str]:
        return f"{self.cache_name}:{self.cache_key(**kwargs)}"

    def cache_key(self, **kwargs: Any) -> str:
        owner = str(kwargs["owner"])
        repo = str(kwargs["repo"])
        state = str(kwargs.get("state", "open"))
        # Use the helper from api client
        return self.api._pulls_list_cache_key(owner, repo, state)

    def empty_value(self) -> List[Dict[str, Any]]:
        return []

    def cache_read(self, *, key: str) -> Optional[Dict[str, Any]]:
        cached = PULLS_LIST_CACHE.get_if_fresh(key, ttl_s=int(self._ttl_s))
        if cached is not None:
            # Return a dict wrapper to match the base class interface
            return {"pulls": cached}
        return None

    def value_from_cache_entry(self, *, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        pulls = entry.get("pulls")
        return pulls if isinstance(pulls, list) else []

    def is_cache_entry_fresh(self, *, entry: Dict[str, Any], now: int) -> bool:
        # PULLS_LIST_CACHE.get_if_fresh already applied TTL; treat returned entries as fresh.
        return True

    def cache_write(self, *, key: str, value: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
        most_recent_updated_at = meta.get("updated_at_epoch")
        PULLS_LIST_CACHE.put(key, value, updated_at_epoch=most_recent_updated_at)

    def fetch(self, **kwargs: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        owner = str(kwargs["owner"])
        repo = str(kwargs["repo"])
        state = str(kwargs.get("state", "open"))

        cache_key = self.cache_key(owner=owner, repo=repo, state=state)
        now = int(time.time())

        # Attribute pulls_list calls to stale-cache age buckets.
        stale_ts = None
        try:
            # Best-effort: inspect existing on-disk entry without counting this as a cache hit/miss.
            with PULLS_LIST_CACHE._mu:  # type: ignore[attr-defined]
                PULLS_LIST_CACHE._load_once()  # type: ignore[attr-defined]
                ent = (PULLS_LIST_CACHE._get_items() or {}).get(cache_key)  # type: ignore[attr-defined]
                if isinstance(ent, dict):
                    ts_raw = ent.get("ts", None)
                    try:
                        ts_i = int(ts_raw) if ts_raw is not None else 0
                    except (ValueError, TypeError):
                        ts_i = 0
                    if ts_i > 0:
                        stale_ts = ts_i
        except Exception:
            stale_ts = None

        cache_age_bucket = "no_cache"
        if stale_ts is not None:
            age_s = max(0, int(now) - int(stale_ts))
            if age_s < 3600:
                cache_age_bucket = "<1h"
            elif age_s < 2 * 3600:
                cache_age_bucket = "<2h"
            elif age_s < 3 * 3600:
                cache_age_bucket = "<3h"
            else:
                cache_age_bucket = ">=3h"

        endpoint = f"/repos/{owner}/{repo}/pulls"
        items = []

        try:
            page = 1
            while True:
                params = {"state": state, "per_page": 100, "page": page}
                url = f"{self.api.base_url}{endpoint}"

                # Counters: each page fetch is one REST call (label: pulls_list).
                try:
                    GITHUB_API_STATS.pulls_list_network_page_calls_total += 1
                    st = str(state or "").strip().lower() or "open"
                    GITHUB_API_STATS.pulls_list_network_page_calls_by_state[st] = int(
                        GITHUB_API_STATS.pulls_list_network_page_calls_by_state.get(st, 0) or 0
                    ) + 1
                    b = str(cache_age_bucket)
                    GITHUB_API_STATS.pulls_list_network_page_calls_by_cache_age_bucket[b] = int(
                        GITHUB_API_STATS.pulls_list_network_page_calls_by_cache_age_bucket.get(b, 0) or 0
                    ) + 1
                    per_state = GITHUB_API_STATS.pulls_list_network_page_calls_by_state_cache_age_bucket.get(st)
                    if not isinstance(per_state, dict):
                        per_state = {}
                        GITHUB_API_STATS.pulls_list_network_page_calls_by_state_cache_age_bucket[st] = per_state
                    per_state[b] = int(per_state.get(b, 0) or 0) + 1
                except Exception:
                    pass

                resp = self.api._rest_get(url, params=params)

                # Parse response
                chunk = resp.json() if resp.status_code >= 200 and resp.status_code < 300 else None
                if not chunk:
                    break
                if isinstance(chunk, list):
                    items.extend(chunk)
                else:
                    break
                if len(chunk) < 100:
                    break
                page += 1
        except Exception:
            # Best-effort: if we can't list PRs (rate limit, network, auth), just return empty.
            # Also negative-cache the failure for a short window to avoid spamming retries when
            # scanning multiple clones of the same repo in one run.
            return [], {"updated_at_epoch": None}

        # Find most recent PR's updated_at for adaptive TTL
        most_recent_updated_at = None
        if items:
            for pr in items:
                if isinstance(pr, dict):
                    upd_str = pr.get("updated_at")
                    if upd_str:
                        try:
                            dt = datetime.fromisoformat(str(upd_str).replace("Z", "+00:00"))
                            upd_epoch = int(dt.timestamp())
                            if most_recent_updated_at is None or upd_epoch > most_recent_updated_at:
                                most_recent_updated_at = upd_epoch
                        except (ValueError, TypeError):
                            pass

        return items, {"updated_at_epoch": most_recent_updated_at}


def list_pull_requests_cached(
    api: "GitHubAPIClient",
    *,
    owner: str,
    repo: str,
    state: str = "open",
    ttl_s: int
) -> List[Dict[str, Any]]:
    """List pull requests for a repo with a short-lived cache.

    Args:
        api: GitHubAPIClient instance
        owner: Repository owner
        repo: Repository name
        state: "open" or "all" (default: "open")
        ttl_s: Cache TTL in seconds

    Returns:
        List of PR dicts (GitHub REST API /pulls response objects).
    """
    return PullsListCached(api, ttl_s=int(ttl_s)).get(owner=owner, repo=repo, state=state)
