# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PR review comments cached API (REST).

Resource:
  GET /repos/{owner}/{repo}/pulls/{pr_number}/comments

Example API Response:
  [
    {
      "id": 123456789,
      "user": {
        "login": "reviewer123"
      },
      "body": "Consider refactoring this method",
      "path": "src/main.py",
      "position": 15,
      "line": 42,
      "commit_id": "abc123def456...",
      "created_at": "2026-01-24T09:30:00Z",
      "updated_at": "2026-01-24T09:35:00Z",
      "in_reply_to_id": null
    },
    {
      "id": 123456790,
      "user": {
        "login": "author456"
      },
      "body": "Good point, will fix",
      "path": "src/main.py",
      "position": 15,
      "line": 42,
      "commit_id": "abc123def456...",
      "created_at": "2026-01-24T10:00:00Z",
      "updated_at": "2026-01-24T10:00:00Z",
      "in_reply_to_id": 123456789
    }
  ]

Cached Fields:
  - Full comment list (id, user, body, path, position, line, in_reply_to_id)
  - Used to track review conversations and unresolved threads

Cache:
  Internal BaseDiskCache instance (disk + memory)

This module returns the raw comment list; higher-level helpers can derive
unresolved conversation counts.
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

CACHE_NAME = "pr_comments"
API_CALL_FORMAT = (
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
CACHE_KEY_FORMAT = "{owner}/{repo}#{pr_number}"
CACHE_FILE_DEFAULT = "pr_comments.json"


# =============================================================================
# Cache Implementation (private to this module)
# =============================================================================

class _PRCommentsCache(BaseDiskCache):
    """Cache for PR review comments (GitHub /pulls/{pr}/comments endpoint).
    
    ETag Support:
        This cache stores ETags from GitHub API responses to enable conditional requests.
        When re-fetching, use get_etag() to get the cached ETag, send it in If-None-Match header,
        and GitHub will return 304 Not Modified if unchanged (doesn't count against rate limit).
    """
    
    _SCHEMA_VERSION = 1
    
    def __init__(self, *, cache_file: Path):
        super().__init__(cache_file=cache_file, schema_version=self._SCHEMA_VERSION)
    
    def get_if_fresh(self, key: str, ttl_s: int) -> Optional[Dict[str, Any]]:
        """Get cached PR comments if fresh."""
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
    
    def put(self, key: str, comments: List[Dict[str, Any]], etag: Optional[str] = None) -> None:
        """Store PR comments with optional ETag."""
        now = int(time.time())
        with self._mu:
            self._load_once()
            entry = {
                "ts": now,
                "comments": comments,
            }
            if etag:
                entry["etag"] = etag.strip()
            self._set_item(key, entry)
            self._persist()


def _get_cache_file() -> Path:
    """Get cache file path."""
    try:
        # Try to import common module to get cache dir
        _module_dir = Path(__file__).resolve().parent.parent
        if str(_module_dir) not in sys.path:
            sys.path.insert(0, str(_module_dir))
        import common
        return common.dynamo_utils_cache_dir() / CACHE_FILE_DEFAULT
    except ImportError:
        return Path.home() / ".cache" / "dynamo-utils" / CACHE_FILE_DEFAULT


# Module-level singleton cache instance (private)
_CACHE = _PRCommentsCache(cache_file=_get_cache_file())


# =============================================================================
# Public API
# =============================================================================


class PRCommentsCached(CachedResourceBase[List[Dict[str, Any]]]):
    def __init__(self, api: "GitHubAPIClient", *, ttl_s: int):
        super().__init__(api)
        self._ttl_s = int(ttl_s)

    @property
    def cache_name(self) -> str:
        return CACHE_NAME

    def api_call_format(self) -> str:
        return API_CALL_FORMAT

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
        return _CACHE.get_if_fresh(key, ttl_s=int(self._ttl_s))

    def value_from_cache_entry(self, *, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        comments = entry.get("comments")
        return comments if isinstance(comments, list) else []

    def is_cache_entry_fresh(self, *, entry: Dict[str, Any], now: int) -> bool:
        # _CACHE.get_if_fresh already applied TTL; treat returned entries as fresh.
        return True

    def cache_write(self, *, key: str, value: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
        _CACHE.put(key, comments=value)

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


def get_cache_sizes() -> Tuple[int, int]:
    """Get cache sizes for stats reporting (memory count, initial disk count)."""
    return _CACHE.get_cache_sizes()

