"""PR branch cached API (REST).

This resource resolves CLOSED/MERGED PRs for a given branch name, used when
there is no matching OPEN PR found in the repo-wide pulls list.

API:
  - GET /repos/{owner}/{repo}/pulls?head={owner}:{branch}&state=all&per_page=30
  - If branch is in "<fork_owner>/<branch>" form, also try head={fork_owner}:{branch}
Cache:
  - PR_BRANCH_CACHE (disk + memory)
TTL policy:
  - If cached list is empty -> no_pr_ttl_s
  - If cached list is non-empty -> min(closed_ttl_s, 360d) (matches existing behavior)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from ..cache_pr_branch import PR_BRANCH_CACHE
from .base_cached import CachedResourceBase

if TYPE_CHECKING:  # pragma: no cover
    from .. import GitHubAPIClient


# Keep TTL documentation next to the actual TTL implementation (`is_cache_entry_fresh`).
TTL_POLICY_DESCRIPTION = "empty: no_pr_ttl_s; non-empty: min(closed_ttl_s, 360d)"


class PRBranchCached(CachedResourceBase[List[Dict[str, Any]]]):
    def __init__(
        self,
        api: "GitHubAPIClient",
        *,
        closed_ttl_s: int,
        no_pr_ttl_s: int,
        refresh: bool,
    ):
        super().__init__(api)
        self._closed_ttl_s = int(closed_ttl_s)
        self._no_pr_ttl_s = int(no_pr_ttl_s)
        self._refresh = bool(refresh)

    @property
    def cache_name(self) -> str:
        return "pr_branch"

    def api_call_format(self) -> str:
        return (
            "REST GET /repos/{owner}/{repo}/pulls?head={owner}:{branch}&state=all&per_page=30 (closed/merged only)\n"
            "Example response item (truncated):\n"
            "  {\n"
            "    \"number\": 4790,\n"
            "    \"state\": \"closed\",\n"
            "    \"merged_at\": \"2026-01-20T11:22:33Z\",\n"
            "    \"updated_at\": \"2026-01-20T11:22:33Z\",\n"
            "    \"head\": {\"ref\": \"user/feature\", \"sha\": \"21a03b3...\"},\n"
            "    \"html_url\": \"https://github.com/OWNER/REPO/pull/4790\"\n"
            "  }"
        )

    def inflight_lock_key(self, **kwargs: Any) -> Optional[str]:
        return f"{self.cache_name}:{self.cache_key(**kwargs)}"

    def cache_key(self, **kwargs: Any) -> str:
        owner = str(kwargs["owner"])
        repo = str(kwargs["repo"])
        branch = str(kwargs["branch"])
        return f"{owner}/{repo}:{branch}"

    def empty_value(self) -> List[Dict[str, Any]]:
        return []

    def cache_read(self, *, key: str) -> Optional[Dict[str, Any]]:
        if self._refresh:
            return None
        try:
            owner_repo, branch = key.split(":", 1)
            owner, repo = owner_repo.split("/", 1)
        except ValueError:
            return None
        return PR_BRANCH_CACHE.get_entry_dict(owner=owner, repo=repo, branch=branch)

    def value_from_cache_entry(self, *, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        prs = entry.get("prs")
        return prs if isinstance(prs, list) else []

    def is_cache_entry_fresh(self, *, entry: Dict[str, Any], now: int) -> bool:
        try:
            ts = int(entry.get("ts", 0) or 0)
        except (ValueError, TypeError):
            ts = 0
        if not ts:
            return False

        prs = entry.get("prs")
        if not isinstance(prs, list):
            return False

        if len(prs) == 0:
            ttl = max(0, int(self._no_pr_ttl_s))
        else:
            ttl = max(0, min(int(self._closed_ttl_s), 360 * 24 * 3600))
        return (now - ts) <= ttl

    def cache_write(self, *, key: str, value: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
        try:
            owner_repo, branch = key.split(":", 1)
            owner, repo = owner_repo.split("/", 1)
        except ValueError:
            return
        ts = int(meta.get("ts") or 0)
        PR_BRANCH_CACHE.put(owner=owner, repo=repo, branch=branch, value={"ts": ts, "prs": value})

    def fetch(self, **kwargs: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        owner = str(kwargs["owner"])
        repo = str(kwargs["repo"])
        branch_name = str(kwargs["branch"])
        now = int(time.time())

        endpoint = f"/repos/{owner}/{repo}/pulls"
        prs_data = self.api.get(endpoint, params={"head": f"{owner}:{branch_name}", "state": "all", "per_page": 30})
        if (not prs_data) and "/" in (branch_name or ""):
            pre, rest = branch_name.split("/", 1)
            pre = (pre or "").strip()
            rest = (rest or "").strip()
            if pre and rest:
                prs_data = self.api.get(endpoint, params={"head": f"{pre}:{rest}", "state": "all", "per_page": 30})

        out_prs: List[Dict[str, Any]] = []
        if isinstance(prs_data, list):
            for pr_data in prs_data:
                if not isinstance(pr_data, dict):
                    continue
                st = str(pr_data.get("state") or "").lower()
                is_merged = pr_data.get("merged_at") is not None
                if st != "open" or is_merged:
                    out_prs.append(
                        {
                            "number": pr_data.get("number"),
                            "title": pr_data.get("title") or "",
                            "url": pr_data.get("html_url") or "",
                            "state": pr_data.get("state") or "",
                            "is_merged": bool(is_merged),
                            "mergeable_state": pr_data.get("mergeable_state") or "unknown",
                            "head_sha": (pr_data.get("head") or {}).get("sha"),
                            "base_ref": (pr_data.get("base") or {}).get("ref", "main"),
                            "created_at": pr_data.get("created_at"),
                            "updated_at": pr_data.get("updated_at"),
                        }
                    )

        try:
            out_prs.sort(key=lambda d: int(d.get("number") or 0), reverse=True)
        except (ValueError, TypeError):
            pass

        return out_prs, {"ts": now}


def get_pr_branch_cached(
    api: "GitHubAPIClient",
    *,
    owner: str,
    repo: str,
    branch: str,
    closed_ttl_s: int,
    no_pr_ttl_s: int,
    refresh: bool,
) -> List[Dict[str, Any]]:
    return PRBranchCached(
        api,
        closed_ttl_s=int(closed_ttl_s),
        no_pr_ttl_s=int(no_pr_ttl_s),
        refresh=bool(refresh),
    ).get(owner=owner, repo=repo, branch=branch)

