"""PR review comments cached API (REST).

Resource:
  GET /repos/{owner}/{repo}/pulls/{pr_number}/comments

Cache:
  PR_COMMENTS_CACHE (disk + memory)

This module returns the raw comment list; higher-level helpers can derive
unresolved conversation counts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from ..cache_pr_comments import PR_COMMENTS_CACHE
from .base_cached import CachedResourceBase

if TYPE_CHECKING:  # pragma: no cover
    from .. import GitHubAPIClient


TTL_POLICY_DESCRIPTION = "ttl_s (default 5m)"


class PRCommentsCached(CachedResourceBase[List[Dict[str, Any]]]):
    def __init__(self, api: "GitHubAPIClient", *, ttl_s: int):
        super().__init__(api)
        self._ttl_s = int(ttl_s)

    @property
    def cache_name(self) -> str:
        return "pr_comments"

    def api_call_format(self) -> str:
        return (
            "REST GET /repos/{owner}/{repo}/pulls/{pr_number}/comments\n"
            "Example response item (truncated):\n"
            "  {\n"
            "    \"id\": 123456789,\n"
            "    \"in_reply_to_id\": null,\n"
            "    \"user\": {\"login\": \"octocat\"},\n"
            "    \"created_at\": \"2026-01-24T01:02:03Z\",\n"
            "    \"html_url\": \"https://github.com/OWNER/REPO/pull/5530#discussion_r123\"\n"
            "  }"
        )

    def inflight_lock_key(self, **kwargs: Any) -> Optional[str]:
        return f"{self.cache_name}:{self.cache_key(**kwargs)}"

    def cache_key(self, **kwargs: Any) -> str:
        owner = str(kwargs["owner"])
        repo = str(kwargs["repo"])
        prn = int(kwargs["pr_number"])
        return f"{owner}/{repo}#{prn}"

    def empty_value(self) -> List[Dict[str, Any]]:
        return []

    def cache_read(self, *, key: str) -> Optional[Dict[str, Any]]:
        return PR_COMMENTS_CACHE.get_if_fresh(key, ttl_s=int(self._ttl_s))

    def value_from_cache_entry(self, *, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        comments = entry.get("comments")
        return comments if isinstance(comments, list) else []

    def is_cache_entry_fresh(self, *, entry: Dict[str, Any], now: int) -> bool:
        # PR_COMMENTS_CACHE.get_if_fresh already applied TTL; treat returned entries as fresh.
        return True

    def cache_write(self, *, key: str, value: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
        PR_COMMENTS_CACHE.put(key, comments=value)

    def fetch(self, **kwargs: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        owner = str(kwargs["owner"])
        repo = str(kwargs["repo"])
        prn = int(kwargs["pr_number"])
        endpoint = f"/repos/{owner}/{repo}/pulls/{prn}/comments"
        comments = self.api.get(endpoint)
        if not isinstance(comments, list):
            comments = []
        return comments, {}


def list_pr_review_comments_cached(
    api: "GitHubAPIClient", *, owner: str, repo: str, pr_number: int, ttl_s: int = 300
) -> List[Dict[str, Any]]:
    return PRCommentsCached(api, ttl_s=int(ttl_s)).get(owner=owner, repo=repo, pr_number=int(pr_number))


def count_unresolved_conversations_cached(
    api: "GitHubAPIClient", *, owner: str, repo: str, pr_number: int, ttl_s: int = 300
) -> int:
    comments = list_pr_review_comments_cached(api, owner=owner, repo=repo, pr_number=int(pr_number), ttl_s=int(ttl_s))
    # Approximation used in existing code: top-level comments (no in_reply_to_id) represent threads.
    n = 0
    for c in comments:
        if isinstance(c, dict) and not c.get("in_reply_to_id"):
            n += 1
    return n

