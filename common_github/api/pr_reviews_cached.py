"""PR reviews cached API (REST).

Resource:
  GET /repos/{owner}/{repo}/pulls/{pr_number}/reviews

Example API Response:
  [
    {
      "id": 987654321,
      "user": {
        "login": "reviewer123"
      },
      "body": "LGTM!",
      "state": "APPROVED",
      "submitted_at": "2026-01-24T10:15:30Z",
      "commit_id": "abc123def456..."
    },
    {
      "id": 987654322,
      "user": {
        "login": "another-reviewer"
      },
      "body": "Needs changes",
      "state": "CHANGES_REQUESTED",
      "submitted_at": "2026-01-24T09:00:00Z",
      "commit_id": "abc123def456..."
    }
  ]

Cached Fields:
  - Full review list (id, user, body, state, submitted_at, commit_id)
  - Review states: "APPROVED", "CHANGES_REQUESTED", "COMMENTED", "DISMISSED"

Cache:
  Internal BaseDiskCache instance (disk + memory)

TTL:
  caller-provided (default 5m in existing code path)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from cache.cache_base import BaseDiskCache
from .base_cached import CachedResourceBase

if TYPE_CHECKING:  # pragma: no cover
    from .. import GitHubAPIClient


TTL_POLICY_DESCRIPTION = "ttl_s (default 5m)"


# =============================================================================
# Cache Implementation (private to this module)
# =============================================================================

class _PRReviewsCache(BaseDiskCache):
    """Cache for PR reviews (GitHub /pulls/{pr}/reviews endpoint)."""
    
    _SCHEMA_VERSION = 1
    
    def __init__(self, *, cache_file: Path):
        super().__init__(cache_file=cache_file, schema_version=self._SCHEMA_VERSION)
    
    def get_if_fresh(self, key: str, ttl_s: int) -> Optional[Dict[str, Any]]:
        """Get cached PR reviews if fresh."""
        with self._mu:
            self._load_once()
            ent = self._check_item(key)
            
            if not isinstance(ent, dict):
                return None
            
            ts = int(ent.get("ts", 0) or 0)
            now = int(time.time())
            
            if ts and (now - ts) <= max(0, int(ttl_s)):
                return ent
            
            return None
    
    def put(self, key: str, reviews: List[Dict[str, Any]], etag: Optional[str] = None) -> None:
        """Store PR reviews with optional ETag."""
        now = int(time.time())
        with self._mu:
            self._load_once()
            entry = {
                "ts": now,
                "reviews": reviews,
            }
            if etag:
                entry["etag"] = etag.strip()
            self._set_item(key, entry)
            self._persist()


def _get_cache_file() -> Path:
    """Get cache file path."""
    try:
        _module_dir = Path(__file__).resolve().parent.parent
        if str(_module_dir) not in sys.path:
            sys.path.insert(0, str(_module_dir))
        import common
        return common.dynamo_utils_cache_dir() / "pr_reviews.json"
    except ImportError:
        return Path.home() / ".cache" / "dynamo-utils" / "pr_reviews.json"


# Module-level singleton cache instance (private)
_CACHE = _PRReviewsCache(cache_file=_get_cache_file())


# =============================================================================
# Public API
# =============================================================================

class PRReviewsCached(CachedResourceBase[List[Dict[str, Any]]]):
    def __init__(self, api: "GitHubAPIClient", *, ttl_s: int):
        super().__init__(api)
        self._ttl_s = int(ttl_s)

    @property
    def cache_name(self) -> str:
        return "pr_reviews"

    def api_call_format(self) -> str:
        return (
            "REST GET /repos/{owner}/{repo}/pulls/{pr_number}/reviews?per_page=100\n"
            "Example response item (truncated):\n"
            "  {\n"
            "    \"id\": 987654321,\n"
            "    \"user\": {\"login\": \"octocat\"},\n"
            "    \"state\": \"APPROVED\",\n"
            "    \"submitted_at\": \"2026-01-24T01:02:03Z\"\n"
            "  }"
        )

    def inflight_lock_key(self, **kwargs: Any) -> Optional[str]:
        key = self.cache_key(**kwargs)
        return f"{self.cache_name}:{key}"

    def cache_key(self, **kwargs: Any) -> str:
        owner = str(kwargs["owner"])
        repo = str(kwargs["repo"])
        prn = int(kwargs["pr_number"])
        return f"{owner}/{repo}#{prn}"

    def empty_value(self) -> List[Dict[str, Any]]:
        return []

    def cache_read(self, *, key: str) -> Optional[Dict[str, Any]]:
        return _CACHE.get_if_fresh(key, ttl_s=int(self._ttl_s))

    def value_from_cache_entry(self, *, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        reviews = entry.get("reviews")
        return reviews if isinstance(reviews, list) else []

    def is_cache_entry_fresh(self, *, entry: Dict[str, Any], now: int) -> bool:
        # _CACHE.get_if_fresh already applied TTL; treat returned entries as fresh.
        return True

    def cache_write(self, *, key: str, value: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
        _CACHE.put(key, reviews=value)

    def fetch(self, **kwargs: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        owner = str(kwargs["owner"])
        repo = str(kwargs["repo"])
        prn = int(kwargs["pr_number"])
        endpoint = f"/repos/{owner}/{repo}/pulls/{prn}/reviews"
        reviews = self.api.get(endpoint, params={"per_page": 100})
        if not isinstance(reviews, list):
            reviews = []
        return reviews, {}


def get_pr_reviews_cached(
    api: "GitHubAPIClient", *, owner: str, repo: str, pr_number: int, ttl_s: int
) -> List[Dict[str, Any]]:
    return PRReviewsCached(api, ttl_s=int(ttl_s)).get(owner=owner, repo=repo, pr_number=int(pr_number))


def get_cache_sizes() -> Tuple[int, int]:
    """Get cache sizes for stats reporting (memory count, initial disk count)."""
    return _CACHE.get_cache_sizes()
