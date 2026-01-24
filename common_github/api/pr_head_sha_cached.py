"""PR head SHA cached API (REST).

Resource:
  GET /repos/{owner}/{repo}/pulls/{pr_number}

Cache:
  PR_HEAD_SHA_CACHE (disk + memory). Closed/merged PRs are cached effectively forever.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from ..cache_pr_info import PR_HEAD_SHA_CACHE
from .base_cached import CachedResourceBase

if TYPE_CHECKING:  # pragma: no cover
    from .. import GitHubAPIClient


TTL_POLICY_DESCRIPTION = "open: ttl_s (default 1d); closed/merged: ∞ (head SHA immutable)"


class PRHeadSHACached(CachedResourceBase[Optional[str]]):
    def __init__(self, api: "GitHubAPIClient", *, ttl_s: int):
        super().__init__(api)
        self._ttl_s = int(ttl_s)

    @property
    def cache_name(self) -> str:
        return "pr_head_sha"

    def api_call_format(self) -> str:
        return (
            "REST GET /repos/{owner}/{repo}/pulls/{pr_number} (extract head.sha)\n"
            "Example response fields used:\n"
            "  {\n"
            "    \"state\": \"open\",\n"
            "    \"head\": {\"sha\": \"21a03b3...\"}\n"
            "  }"
        )

    def inflight_lock_key(self, **kwargs: Any) -> Optional[str]:
        return f"{self.cache_name}:{self.cache_key(**kwargs)}"

    def cache_key(self, **kwargs: Any) -> str:
        owner = str(kwargs["owner"])
        repo = str(kwargs["repo"])
        prn = int(kwargs["pr_number"])
        return f"{owner}/{repo}:pr:{prn}:head_sha"

    def empty_value(self) -> Optional[str]:
        return None

    def cache_read(self, *, key: str) -> Optional[Dict[str, Any]]:
        return PR_HEAD_SHA_CACHE.get_if_fresh(key, ttl_s=int(self._ttl_s))

    def value_from_cache_entry(self, *, entry: Dict[str, Any]) -> Optional[str]:
        head_sha = entry.get("head_sha")
        hs = str(head_sha or "").strip()
        return hs or None

    def is_cache_entry_fresh(self, *, entry: Dict[str, Any], now: int) -> bool:
        # PR_HEAD_SHA_CACHE.get_if_fresh already applied TTL and "closed/merged forever" rules.
        return True

    def cache_write(self, *, key: str, value: Optional[str], meta: Dict[str, Any]) -> None:
        hs = str(value or "").strip()
        if not hs:
            return
        PR_HEAD_SHA_CACHE.put(key, head_sha=hs, state=str(meta.get("state") or "").strip())

    def fetch(self, **kwargs: Any) -> Tuple[Optional[str], Dict[str, Any]]:
        owner = str(kwargs["owner"])
        repo = str(kwargs["repo"])
        prn = int(kwargs["pr_number"])

        pr = self.api.get(f"/repos/{owner}/{repo}/pulls/{prn}", timeout=10) or {}
        head_sha = (((pr.get("head") or {}) if isinstance(pr, dict) else {}) or {}).get("sha")
        head_sha_s = str(head_sha or "").strip()
        state = str(pr.get("state", "") or "").strip() if isinstance(pr, dict) else ""
        return (head_sha_s or None), {"state": state}


def get_pr_head_sha_cached(
    api: "GitHubAPIClient", *, owner: str, repo: str, pr_number: int, ttl_s: int = 86400
) -> Optional[str]:
    return PRHeadSHACached(api, ttl_s=int(ttl_s)).get(owner=owner, repo=repo, pr_number=int(pr_number))

