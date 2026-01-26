# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PR head SHA cached API (REST).

Resource:
  GET /repos/{owner}/{repo}/pulls/{pr_number}

Example API Response:
  {
    "number": 1234,
    "state": "open",
    "head": {
      "sha": "abc123def456789...",
      "ref": "feature-branch"
    },
    "base": {
      "ref": "main"
    },
    "updated_at": "2026-01-24T10:30:00Z",
    "merged": false,
    "merged_at": "2026-01-24T10:45:30Z"  // Only present if merged
  }

Cached Fields:
  - head_sha: The head SHA of the PR (from head.sha)
  - state: "open", "closed", or "merged"

MERGE DATES (Consolidated from deprecated merge_dates_cached module):
  The PR API response includes "merged_at" (ISO timestamp) for merged PRs.
  
  How to get merge dates:
    1. Fetch PR using this module or pulls_list_cached
    2. Extract pr_dict.get("merged_at") - returns ISO timestamp or None
    3. Convert to Pacific time if needed:
       from datetime import datetime
       from zoneinfo import ZoneInfo
       dt_utc = datetime.fromisoformat(merged_at.replace("Z", "+00:00"))
       dt_pacific = dt_utc.astimezone(ZoneInfo("America/Los_Angeles"))
       merge_date_str = dt_pacific.strftime("%Y-%m-%d %H:%M:%S")
  
  Note about merged_at vs updated_at:
    - merged_at: Set ONCE when PR is merged, never changes (immutable)
    - updated_at: Changes with ANY activity (comments, reviews, labels, etc.)
    - In practice, they are often within 1-2 seconds of each other if there's
      no post-merge activity, but can diverge by hours/days if there are
      comments or other updates after merge.
    - For "when was this merged?" use merged_at (immutable)
    - For cache invalidation, use updated_at (tracks recent activity)

Cross-cache population:
  This cache is automatically populated by pulls_list_cached when fetching
  the pulls list, avoiding redundant API calls for individual PR head SHA lookups.

Cache:
  Internal BaseDiskCache instance (disk + memory). Closed/merged PRs are cached effectively forever.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from cache.cache_base import BaseDiskCache
from .base_cached import CachedResourceBase

if TYPE_CHECKING:  # pragma: no cover
    from .. import GitHubAPIClient


TTL_POLICY_DESCRIPTION = "open: ttl_s (default 1d); closed/merged: ∞ (head SHA immutable)"

CACHE_NAME = "pr_head_sha"
API_CALL_FORMAT = (
    "REST GET /repos/{owner}/{repo}/pulls/{pr_number} (extract head.sha)\n"
    "Example response fields used:\n"
    "  {\n"
    "    \"state\": \"open\",\n"
    "    \"head\": {\"sha\": \"21a03b3...\"}\n"
    "  }"
)
CACHE_KEY_FORMAT = "{owner}/{repo}:pr:{pr_number}:head_sha"
CACHE_FILE_DEFAULT = "pr_head_sha.json"


# =============================================================================
# Cache Implementation (private to this module)
# =============================================================================

class _PRHeadSHACache(BaseDiskCache):
    """Cache for PR head SHA lookups."""
    
    _SCHEMA_VERSION = 1
    
    def __init__(self, *, cache_file: Path):
        super().__init__(cache_file=cache_file, schema_version=self._SCHEMA_VERSION)
    
    def get_if_fresh(self, key: str, ttl_s: int) -> Optional[Dict[str, Any]]:
        """Get cached head SHA if fresh."""
        with self._mu:
            self._load_once()
            ent = self._check_item(key)
            
            if not isinstance(ent, dict):
                return None
            
            state = str(ent.get("state", "") or "").strip().lower()
            
            # Closed/merged PRs are cached forever
            if state in ("closed", "merged"):
                return {
                    "head_sha": ent.get("head_sha"),
                    "state": ent.get("state"),
                }
            
            # For open PRs, check TTL
            ts = int(ent.get("ts", 0) or 0)
            now = int(time.time())
            
            if ts and (now - ts) <= max(0, int(ttl_s)):
                return {
                    "head_sha": ent.get("head_sha"),
                    "state": ent.get("state"),
                }
            
            return None
    
    def put(self, key: str, head_sha: str, state: str) -> None:
        """Store head SHA."""
        now = int(time.time())
        with self._mu:
            self._load_once()
            entry = {
                "ts": now,
                "head_sha": head_sha,
                "state": state,
            }
            self._set_item(key, entry)
            self._persist()


def _get_cache_file() -> Path:
    """Get cache file path."""
    try:
        _module_dir = Path(__file__).resolve().parent.parent
        if str(_module_dir) not in sys.path:
            sys.path.insert(0, str(_module_dir))
        import common
        cache_dir = common.dynamo_utils_cache_dir() / "pr-info"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / CACHE_FILE_DEFAULT
    except ImportError:
        cache_dir = Path.home() / ".cache" / "dynamo-utils" / "pr-info"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / CACHE_FILE_DEFAULT


# Module-level singleton cache instance (private)
_CACHE = _PRHeadSHACache(cache_file=_get_cache_file())


# =============================================================================
# Public API
# =============================================================================


class PRHeadSHACached(CachedResourceBase[Optional[str]]):
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
        return f"{owner}/{repo}:pr:{prn}:head_sha"

    def empty_value(self) -> Optional[str]:
        return None

    def cache_read(self, *, key: str) -> Optional[Dict[str, Any]]:
        return _CACHE.get_if_fresh(key, ttl_s=int(self._ttl_s))

    def value_from_cache_entry(self, *, entry: Dict[str, Any]) -> Optional[str]:
        head_sha = entry.get("head_sha")
        hs = str(head_sha or "").strip()
        return hs or None

    def is_cache_entry_fresh(self, *, entry: Dict[str, Any], now: int) -> bool:
        # _CACHE.get_if_fresh already applied TTL and "closed/merged forever" rules.
        return True

    def cache_write(self, *, key: str, value: Optional[str], meta: Dict[str, Any]) -> None:
        hs = str(value or "").strip()
        if not hs:
            return
        _CACHE.put(key, head_sha=hs, state=str(meta.get("state") or "").strip())

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


def get_cache_sizes() -> Tuple[int, int]:
    """Get cache sizes for stats reporting (memory count, initial disk count)."""
    return _CACHE.get_cache_sizes()

