"""PR reviews cached API (REST).

Resource: GET /repos/{owner}/{repo}/pulls/{pr_number}/reviews
Cache: PR_REVIEWS_CACHE (disk + memory)
TTL: caller-provided (default 5m in existing code path)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from ..cache_pr_reviews import PR_REVIEWS_CACHE
from .base_cached import CachedResourceBase

if TYPE_CHECKING:  # pragma: no cover
    from .. import GitHubAPIClient


TTL_POLICY_DESCRIPTION = "ttl_s (default 5m)"


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
        return PR_REVIEWS_CACHE.get_if_fresh(key, ttl_s=int(self._ttl_s))

    def value_from_cache_entry(self, *, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        reviews = entry.get("reviews")
        return reviews if isinstance(reviews, list) else []

    def is_cache_entry_fresh(self, *, entry: Dict[str, Any], now: int) -> bool:
        # PR_REVIEWS_CACHE.get_if_fresh already applied TTL; treat returned entries as fresh.
        return True

    def cache_write(self, *, key: str, value: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
        PR_REVIEWS_CACHE.put(key, reviews=value)

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

