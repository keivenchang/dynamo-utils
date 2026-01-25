"""Pulls list cached API (REST).

Resource:
  GET /repos/{owner}/{repo}/pulls?state={state}&per_page=100

Example API Response:
  [
    {
      "number": 1234,
      "title": "Add new feature",
      "state": "open",
      "user": {
        "login": "contributor123"
      },
      "head": {
        "ref": "feature-branch",
        "sha": "abc123..."
      },
      "base": {
        "ref": "main"
      },
      "created_at": "2026-01-20T10:00:00Z",
      "updated_at": "2026-01-24T10:30:00Z",
      "draft": false,
      "labels": [
        {"name": "enhancement"}
      ]
    },
    {
      "number": 1233,
      "title": "Fix bug",
      "state": "open",
      "user": {
        "login": "contributor456"
      },
      "updated_at": "2026-01-23T15:00:00Z"
    }
  ]

Cached Fields:
  - Full PR list (number, title, state, user, head, base, updated_at, etc.)
  - Limited to first 5 pages (500 most recent PRs)
  - Older PRs are fetched individually on-demand via get_pr_details fallback

Cache:
  Internal BaseDiskCache instance (disk + memory)

TTL:
  Fixed 2 minutes (120 seconds)

Cross-cache population:
  When fetching the pulls list, this module also populates pr_head_sha_cached
  for each PR to avoid redundant API calls when later checking individual PR
  head SHAs.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from cache.cache_base import BaseDiskCache
from .. import GITHUB_API_STATS
from ..cache_ttl_utils import pulls_list_adaptive_ttl_s
from .base_cached import CachedResourceBase

if TYPE_CHECKING:  # pragma: no cover
    from .. import GitHubAPIClient


TTL_POLICY_DESCRIPTION = "fixed 2m (120s)"


# =============================================================================
# Cache Implementation (private to this module)
# =============================================================================

class _PullsListCache(BaseDiskCache):
    """Cache for GitHub pulls list API results."""
    
    _SCHEMA_VERSION = 1
    
    def __init__(self, *, cache_file: Path):
        super().__init__(cache_file=cache_file, schema_version=self._SCHEMA_VERSION)
    
    def get_if_fresh(self, key: str, ttl_s: int) -> Optional[List[Dict[str, Any]]]:
        """Get cached pulls list if fresh (fixed 2m TTL)."""
        with self._mu:
            self._load_once()
            ent = self._check_item(key)
            
            if not isinstance(ent, dict):
                return None
            
            ts = int(ent.get("ts", 0) or 0)
            now = int(time.time())
            
            # Fixed TTL: 2 minutes (120 seconds)
            effective_ttl = 120
            
            if ts and (now - ts) <= max(0, int(effective_ttl)):
                pulls = ent.get("pulls")
                if isinstance(pulls, list):
                    return pulls
            
            return None
    
    def put(self, key: str, pulls: List[Dict[str, Any]], updated_at_epoch: Optional[int] = None) -> None:
        """Store pulls list."""
        now = int(time.time())
        with self._mu:
            self._load_once()
            entry = {
                "ts": now,
                "pulls": pulls,
            }
            if updated_at_epoch is not None:
                entry["updated_at_epoch"] = updated_at_epoch
            self._set_item(key, entry)
            self._persist()


def _get_cache_file() -> Path:
    """Get cache file path."""
    try:
        _module_dir = Path(__file__).resolve().parent.parent
        if str(_module_dir) not in sys.path:
            sys.path.insert(0, str(_module_dir))
        import common
        return common.dynamo_utils_cache_dir() / "pulls_list.json"
    except ImportError:
        return Path.home() / ".cache" / "dynamo-utils" / "pulls_list.json"


# Module-level singleton cache instance (private)
_CACHE = _PullsListCache(cache_file=_get_cache_file())


# =============================================================================
# Public API
# =============================================================================


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
        cached = _CACHE.get_if_fresh(key, ttl_s=int(self._ttl_s))
        if cached is not None:
            # Return a dict wrapper to match the base class interface
            return {"pulls": cached}
        return None

    def value_from_cache_entry(self, *, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        pulls = entry.get("pulls")
        return pulls if isinstance(pulls, list) else []

    def is_cache_entry_fresh(self, *, entry: Dict[str, Any], now: int) -> bool:
        # _CACHE.get_if_fresh already applied TTL; treat returned entries as fresh.
        return True

    def cache_write(self, *, key: str, value: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
        most_recent_updated_at = meta.get("updated_at_epoch")
        _CACHE.put(key, value, updated_at_epoch=most_recent_updated_at)
        
        # OPTIMIZATION: Cross-populate pr_head_sha cache
        # When we fetch the pulls list, populate pr_head_sha_cached for each PR
        # to avoid redundant API calls when later checking individual PR head SHAs
        _populate_pr_head_sha_from_pulls_list(key, value)

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
            with _CACHE._mu:  # type: ignore[attr-defined]
                _CACHE._load_once()  # type: ignore[attr-defined]
                ent = (_CACHE._get_items() or {}).get(cache_key)  # type: ignore[attr-defined]
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
            max_pages = 5  # Limit to 5 pages (500 most recent PRs)
            while page <= max_pages:
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


def _populate_pr_head_sha_from_pulls_list(list_key: str, pulls: List[Dict[str, Any]]) -> None:
    """Cross-populate pr_head_sha_cached from pulls list data.
    
    This optimization avoids redundant API calls when checking individual PR head SHAs
    after a pulls list fetch has already retrieved the data.
    
    Args:
        list_key: The pulls list cache key (e.g., "owner/repo:pulls:open")
        pulls: List of PR dicts from the pulls list API
    """
    try:
        # Import here to avoid circular dependency
        from . import pr_head_sha_cached
        
        # Extract owner/repo from list_key
        # Format: "owner/repo:pulls:state"
        parts = list_key.split(":")
        if len(parts) < 2:
            return
        owner_repo = parts[0]  # "owner/repo"
        
        for pr_dict in pulls:
            if not isinstance(pr_dict, dict):
                continue
            
            # Extract required fields
            pr_number = pr_dict.get("number")
            if not pr_number:
                continue
            
            head = pr_dict.get("head")
            if not isinstance(head, dict):
                continue
            
            head_sha = str(head.get("sha") or "").strip()
            if not head_sha:
                continue
            
            state = str(pr_dict.get("state") or "").strip().lower()
            if not state:
                state = "open"  # Default to open if missing
            
            # Populate pr_head_sha cache using its cache key format
            # Format: "{owner}/{repo}:pr:{pr_number}:head_sha"
            pr_head_sha_key = f"{owner_repo}:pr:{pr_number}:head_sha"
            pr_head_sha_cached._CACHE.put(pr_head_sha_key, head_sha=head_sha, state=state)
            
    except Exception:
        # Don't fail the main operation if cross-cache population fails
        pass


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


def get_cache_sizes() -> Tuple[int, int]:
    """Get cache sizes for stats reporting (memory count, initial disk count)."""
    return _CACHE.get_cache_sizes()

