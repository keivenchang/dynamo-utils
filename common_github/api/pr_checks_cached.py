# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PR checks rows cached API (REST).

This is the heavy hitter for dashboards: it builds `GHPRCheckRow` entries from a mix of:
  - check-runs: GET /repos/{owner}/{repo}/commits/{sha}/check-runs
  - status contexts: GET /repos/{owner}/{repo}/commits/{sha}/status
  - (best-effort) workflow metadata: GET /repos/{owner}/{repo}/actions/runs/{run_id}

Example API Response (check-runs):
  {
    "total_count": 3,
    "check_runs": [
      {
        "id": 12345678,
        "name": "build",
        "status": "completed",
        "conclusion": "success",
        "started_at": "2026-01-24T10:00:00Z",
        "completed_at": "2026-01-24T10:05:00Z",
        "html_url": "https://github.com/owner/repo/actions/runs/98765/jobs/12345678",
        "check_suite": {
          "id": 87654321
        }
      }
    ]
  }

Example API Response (status):
  {
    "state": "success",
    "statuses": [
      {
        "id": 11111111,
        "context": "ci/circleci",
        "state": "success",
        "description": "All tests passed",
        "target_url": "https://circleci.com/gh/owner/repo/123",
        "created_at": "2026-01-24T10:00:00Z",
        "updated_at": "2026-01-24T10:05:00Z"
      }
    ]
  }

Cached Fields:
  - GHPRCheckRow objects combining check-runs + statuses
  - Fields: name, status, conclusion, started_at, completed_at, html_url, workflow_name

Example API Response (check-runs):
  {
    "total_count": 3,
    "check_runs": [
      {
        "id": 12345678,
        "name": "build",
        "status": "completed",
        "conclusion": "success",
        "started_at": "2026-01-24T10:00:00Z",
        "completed_at": "2026-01-24T10:05:00Z",
        "html_url": "https://github.com/owner/repo/actions/runs/98765/jobs/12345678",
        "check_suite": {
          "id": 87654321
        }
      }
    ]
  }

Example API Response (status):
  {
    "state": "success",
    "statuses": [
      {
        "id": 11111111,
        "context": "ci/circleci",
        "state": "success",
        "description": "All tests passed",
        "target_url": "https://circleci.com/gh/owner/repo/123",
        "created_at": "2026-01-24T10:00:00Z",
        "updated_at": "2026-01-24T10:05:00Z"
      }
    ]
  }

Cached Fields:
  - GHPRCheckRow objects combining check-runs + statuses
  - Fields: name, status, conclusion, started_at, completed_at, html_url, workflow_name

Owns:
  - caching (internal BaseDiskCache) + TTL policy (adaptive based on PR updated_at age)
  - ETag/304 handling
  - consistent cache stats attribution
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from cache.cache_base import BaseDiskCache


def _normalize_check_name(name: str) -> str:
    return " ".join((name or "").strip().lower().split())


def _is_required_check_name(check_name: str, required_names: set[str]) -> bool:
    name_norm = _normalize_check_name(check_name)
    if not name_norm:
        return False

    req = {_normalize_check_name(x) for x in (required_names or set()) if _normalize_check_name(x)}
    if not req:
        return False

    if name_norm in req:
        return True

    # Substring match to handle workflow-prefixed contexts
    return any(r and (r in name_norm or name_norm in r) for r in req)

from .. import GITHUB_API_STATS
from ..pr_checks_types import (
    GHPRCheckRow,
    GHPRChecksCacheEntry,
    parse_actions_job_id_from_url,
    parse_actions_run_id_from_url,
)
from .base_cached import CachedResourceBase

if TYPE_CHECKING:  # pragma: no cover
    from .. import GitHubAPIClient


# Used by _get_cache_file() below (must be defined before import-time call).
CACHE_FILE_DEFAULT = "pr_checks.json"


# =============================================================================
# Cache Implementation (private to this module)
# =============================================================================

class _PRChecksCache(BaseDiskCache):
    """Cache for PR check runs and statuses."""
    
    _SCHEMA_VERSION = 1
    
    def __init__(self, *, cache_file: Path):
        super().__init__(cache_file=cache_file, schema_version=self._SCHEMA_VERSION)
    
    def _load_once(self) -> None:
        """Load cache with migration from old format."""
        if self._loaded:
            return
        self._loaded = True

        if not self._cache_file.exists():
            self._data = self._create_empty_cache()
            self._initial_disk_count = 0
            return

        try:
            data = json.loads(self._cache_file.read_text() or "{}")
        except Exception:
            data = {}

        if not isinstance(data, dict):
            data = {}

        # Migration: Check for any top-level entries
        items_dict = data.get("items")
        if not isinstance(items_dict, dict):
            items_dict = {}
        
        migrated_count = 0
        for key, value in list(data.items()):
            if key not in ["version", "items"]:
                items_dict[key] = value
                migrated_count += 1
        
        if migrated_count > 0:
            data = {
                "version": self._schema_version,
                "items": items_dict
            }
            self._dirty = True
        else:
            data["items"] = items_dict
            data["version"] = self._schema_version

        self._initial_disk_count = len(data.get("items", {}))
        self._data = data
    
    def get_entry_dict(self, key: str) -> Optional[Dict[str, Any]]:
        """Get raw cache entry dict."""
        with self._mu:
            self._load_once()
            return self._check_item(key)
    
    def put_entry_dict(self, key: str, entry_dict: Dict[str, Any]) -> None:
        """Store entry dict."""
        with self._mu:
            self._load_once()
            self._set_item(key, entry_dict)
            self._persist()


def _get_cache_file() -> Path:
    """Get cache file path."""
    try:
        _module_dir = Path(__file__).resolve().parent.parent
        if str(_module_dir) not in sys.path:
            sys.path.insert(0, str(_module_dir))
        import common
        return common.dynamo_utils_cache_dir() / CACHE_FILE_DEFAULT
    except ImportError:
        return Path.home() / ".cache" / "dynamo-utils" / CACHE_FILE_DEFAULT


# Module-level singleton cache instance (private)
_CACHE = _PRChecksCache(cache_file=_get_cache_file())


# =============================================================================
# Public API
# =============================================================================


# Keep TTL documentation next to the actual TTL implementation (`api._adaptive_ttl_s` usage).
TTL_POLICY_DESCRIPTION = (
    "adaptive (<1h=2m, <2h=4m, <4h=10m, <8h=15m, <12h=20m, >=12h=30m) when pr_updated_at_epoch is provided; "
    "otherwise ttl_s (default 5m)"
)

CACHE_NAME = "pr_checks"
API_CALL_FORMAT = (
    "REST GET /repos/{owner}/{repo}/commits/{sha}/check-runs?per_page=100 (etag)\n"
    "Example output fields used (truncated):\n"
    "  {\n"
    "    \"check_runs\": [\n"
    "      {\"name\":\"pre-commit\",\"status\":\"completed\",\"conclusion\":\"success\",\"details_url\":\"...\",\"started_at\":\"...\",\"completed_at\":\"...\",\"output\":{\"title\":\"...\"}}\n"
    "    ]\n"
    "  }\n"
    "\n"
    "REST GET /repos/{owner}/{repo}/commits/{sha}/status (etag)\n"
    "Example output fields used (truncated):\n"
    "  {\"statuses\":[{\"context\":\"codecov/project\",\"state\":\"success\",\"description\":\"...\",\"target_url\":\"...\"}]}\n"
    "\n"
    "REST GET /repos/{owner}/{repo}/actions/runs/{run_id} (best-effort)\n"
    "Example output fields used (truncated):\n"
    "  {\"name\":\"CI\",\"event\":\"pull_request\"}"
)
CACHE_KEY_FORMAT = "{owner}/{repo}:pr:{pr_number}:{head_sha}"


class PRChecksCached(CachedResourceBase[List[GHPRCheckRow]]):
    def __init__(
        self,
        api: "GitHubAPIClient",
        *,
        required_checks: Set[str],
        ttl_s: int,
        pr_updated_at_epoch: Optional[int],
        skip_fetch: bool,
    ):
        super().__init__(api)
        self._required_checks = set(required_checks or set())
        self._ttl_s = int(ttl_s)
        self._pr_updated_at_epoch = int(pr_updated_at_epoch) if pr_updated_at_epoch is not None else None
        self._skip_fetch = bool(skip_fetch)

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
        commit_sha = kwargs.get("commit_sha")
        sha_suffix = f":{str(commit_sha)[:7]}" if commit_sha else ""
        return f"{owner}/{repo}#{prn}{sha_suffix}"

    def empty_value(self) -> List[GHPRCheckRow]:
        return []

    # The base class helpers aren't expressive enough for the ETag/304 fast-paths,
    # so we keep this resource self-contained and override get().
    def cache_read(self, *, key: str) -> Optional[Dict[str, Any]]:
        return _CACHE.get_entry_dict(key)

    def value_from_cache_entry(self, *, entry: Dict[str, Any]) -> List[GHPRCheckRow]:
        # For this resource, we always deserialize strictly.
        # NOTE: callers should go through get() which applies TTL.
        raise RuntimeError("PRChecksCached.value_from_cache_entry should not be called directly")

    def is_cache_entry_fresh(self, *, entry: Dict[str, Any], now: int) -> bool:
        raise RuntimeError("PRChecksCached.is_cache_entry_fresh should not be called directly")

    def cache_write(self, *, key: str, value: List[GHPRCheckRow], meta: Dict[str, Any]) -> None:
        raise RuntimeError("PRChecksCached.cache_write should not be called directly")

    def fetch(self, **kwargs: Any) -> Tuple[List[GHPRCheckRow], Dict[str, Any]]:
        raise RuntimeError("PRChecksCached.fetch should not be called directly")

    def get(self, **kwargs: Any) -> List[GHPRCheckRow]:
        owner = str(kwargs["owner"])
        repo = str(kwargs["repo"])
        prn = int(kwargs["pr_number"])
        commit_sha = kwargs.get("commit_sha")
        head_sha = kwargs.get("head_sha")

        cn = self.cache_name
        key = self.cache_key(owner=owner, repo=repo, pr_number=prn, commit_sha=commit_sha)
        now = int(datetime.now(timezone.utc).timestamp())

        effective_ttl_s = int(self._ttl_s)
        if self._pr_updated_at_epoch is not None:
            effective_ttl_s = int(self.api._adaptive_ttl_s(timestamp_epoch=int(self._pr_updated_at_epoch), default_ttl_s=int(self._ttl_s)))

        CACHE_VER = 7
        MIN_CACHE_VER = 2

        cached_entry_dict = _CACHE.get_entry_dict(key)
        has_etag = False
        if cached_entry_dict is not None:
            try:
                ent = GHPRChecksCacheEntry.from_disk_dict_strict(
                    d=cached_entry_dict,
                    cache_file=_CACHE._cache_file,
                    entry_key=key,
                )
                ts = int(ent.ts)
                ver = int(ent.ver)
                incomplete = bool(ent.incomplete)
                has_etag = bool(ent.check_runs_etag or ent.status_etag)
                
                # CRITICAL: If we have ETags, ALWAYS query GitHub with them (even if within TTL).
                # - If data changed: GitHub returns 200 + new data → immediately reflects new CI runs
                # - If unchanged: GitHub returns 304 → cheap, just refreshes timestamp
                # Only skip GitHub query if: (1) no ETag available, AND (2) within TTL
                if ts and ((now - ts) <= max(0, int(effective_ttl_s)) or self.api.cache_only_mode) and not incomplete and not has_etag:
                    if ver >= MIN_CACHE_VER:
                        self.api._cache_hit(cn)
                        return _apply_required_overlay(ent.rows, required=self._required_checks)
            except (ValueError, TypeError, RuntimeError):
                # Invalid/corrupt cache entry: fall through to fetch.
                pass

        stale_check_runs_etag = ""
        stale_status_etag = ""
        if cached_entry_dict is not None:
            try:
                ent2 = GHPRChecksCacheEntry.from_disk_dict_strict(
                    d=cached_entry_dict,
                    cache_file=_CACHE._cache_file,
                    entry_key=key,
                )
                stale_check_runs_etag = str(ent2.check_runs_etag or "")
                stale_status_etag = str(ent2.status_etag or "")
            except (ValueError, TypeError, RuntimeError):
                pass

        if self.api.cache_only_mode:
            self.api._cache_miss(f"{cn}.cache_only_empty")
            return []

        if self._skip_fetch:
            return []

        self.api._cache_miss(cn)

        # If head_sha provided, use it directly (saves 1 API call).
        if head_sha:
            head_sha_s = str(head_sha).strip()
        else:
            head_sha_s = str(self.api.get_pr_head_sha(owner=owner, repo=repo, pr_number=int(prn)) or "").strip()
        if not head_sha_s:
            return []

        check_runs_url = f"{self.api.base_url}/repos/{owner}/{repo}/commits/{head_sha_s}/check-runs"
        check_runs_resp = self.api._rest_get(
            check_runs_url,
            params={"per_page": 100},
            timeout=10,
            etag=stale_check_runs_etag if stale_check_runs_etag else None,
        )
        new_check_runs_etag = ""
        if hasattr(check_runs_resp, "headers"):
            new_check_runs_etag = str(check_runs_resp.headers.get("ETag", "") or "")

        # 304: return cached rows with refreshed timestamp.
        if check_runs_resp is not None and int(getattr(check_runs_resp, "status_code", 0) or 0) == 304:
            if cached_entry_dict is not None:
                ent3 = GHPRChecksCacheEntry.from_disk_dict_strict(
                    d=cached_entry_dict,
                    cache_file=_CACHE._cache_file,
                    entry_key=key,
                )
                refreshed_entry = GHPRChecksCacheEntry(
                    ts=int(now),
                    ver=int(ent3.ver),
                    rows=ent3.rows,
                    check_runs_etag=stale_check_runs_etag,
                    status_etag=ent3.status_etag,
                    incomplete=ent3.incomplete,
                )
                _CACHE.put_entry_dict(key, refreshed_entry.to_disk_dict())
                self.api._cache_write(cn, entries=1)
                return _apply_required_overlay(ent3.rows, required=self._required_checks)
            return []

        data = check_runs_resp.json() if check_runs_resp is not None else {}
        check_runs = data.get("check_runs") if isinstance(data, dict) else None
        if not isinstance(check_runs, list):
            check_runs = []

        status_url = f"{self.api.base_url}/repos/{owner}/{repo}/commits/{head_sha_s}/status"
        status_resp = self.api._rest_get(
            status_url,
            timeout=10,
            etag=stale_status_etag if stale_status_etag else None,
        )
        new_status_etag = ""
        if status_resp is not None and hasattr(status_resp, "headers"):
            new_status_etag = str(status_resp.headers.get("ETag", "") or "")

        statuses: List[Any] = []
        if status_resp is not None and int(getattr(status_resp, "status_code", 0) or 0) != 304:
            statuses_data = status_resp.json() if status_resp is not None else {}
            statuses = statuses_data.get("statuses") if isinstance(statuses_data, dict) else []
            if not isinstance(statuses, list):
                statuses = []

        # Collect run_ids and prefetch workflow info (best-effort).
        run_ids_to_fetch: set[str] = set()
        for cr in check_runs:
            if not isinstance(cr, dict):
                continue
            url = str(cr.get("details_url") or cr.get("html_url") or "").strip()
            rid = parse_actions_run_id_from_url(url)
            if rid:
                run_ids_to_fetch.add(rid)

        workflow_info_cache: Dict[str, tuple[str, str]] = {}
        for run_id in run_ids_to_fetch:
            GITHUB_API_STATS.actions_run_prefetch_total += 1
            run_data = self.api.get(f"/repos/{owner}/{repo}/actions/runs/{run_id}", timeout=5) or {}
            if isinstance(run_data, dict):
                workflow_info_cache[run_id] = (
                    str(run_data.get("name", "") or "").strip(),
                    str(run_data.get("event", "") or "").strip(),
                )
            else:
                workflow_info_cache[run_id] = ("", "")

        rows: List[GHPRCheckRow] = []
        seen: set[tuple[str, str]] = set()

        for cr in check_runs:
            if not isinstance(cr, dict):
                continue
            name = str(cr.get("name", "") or "").strip()
            if not name:
                continue
            status = str(cr.get("status", "") or "").strip().lower()
            conclusion = str(cr.get("conclusion", "") or "").strip().lower()

            status_raw = ""
            if status and status != "completed":
                status_raw = status
            else:
                if conclusion in {"success"}:
                    status_raw = "pass"
                elif conclusion in {"failure"}:
                    status_raw = "fail"
                elif conclusion in {"cancelled", "canceled"}:
                    status_raw = "cancelled"
                elif conclusion in {"skipped"}:
                    status_raw = "skipped"
                elif conclusion in {"neutral"}:
                    status_raw = "neutral"
                elif conclusion in {"timed_out"}:
                    status_raw = "timed_out"
                elif conclusion in {"action_required"}:
                    status_raw = "action_required"
                else:
                    status_raw = "unknown"

            url = str(cr.get("details_url") or cr.get("html_url") or "").strip()
            key2 = (name, url)
            if key2 in seen:
                continue
            seen.add(key2)

            description = ""
            out_obj = cr.get("output") or {}
            if isinstance(out_obj, dict):
                description = str(out_obj.get("title", "") or "").strip()

            started_at = str(cr.get("started_at") or "").strip()
            completed_at = str(cr.get("completed_at") or "").strip()
            duration = _format_dur(started_at, completed_at)

            run_id = parse_actions_run_id_from_url(url)
            job_id = parse_actions_job_id_from_url(url)
            workflow_name, event = workflow_info_cache.get(run_id, ("", "")) if run_id else ("", "")

            rows.append(
                GHPRCheckRow(
                    name=name,
                    status_raw=status_raw,
                    duration=duration,
                    url=url,
                    run_id=run_id,
                    job_id=job_id,
                    description=description,
                    is_required=_is_required_check_name(name, self._required_checks),
                    workflow_name=workflow_name,
                    event=event,
                )
            )

        for st in statuses:
            if not isinstance(st, dict):
                continue
            name = str(st.get("context", "") or "").strip()
            if not name:
                continue
            state = str(st.get("state", "") or "").strip().lower()
            desc = str(st.get("description", "") or "").strip()
            target = str(st.get("target_url", "") or "").strip()

            status_raw = "unknown"
            if state in {"success"}:
                status_raw = "pass"
            elif state in {"failure", "error"}:
                status_raw = "fail"
            elif state in {"pending"}:
                status_raw = "pending"
            if desc and ("skipped" in desc.lower() or "skip" in desc.lower()):
                status_raw = "skipped"

            key2 = (name, target)
            if key2 in seen:
                continue
            seen.add(key2)

            run_id = parse_actions_run_id_from_url(target)
            job_id = parse_actions_job_id_from_url(target)
            rows.append(
                GHPRCheckRow(
                    name=name,
                    status_raw=status_raw,
                    duration="",
                    url=target,
                    run_id=run_id,
                    job_id=job_id,
                    description=desc,
                    is_required=_is_required_check_name(name, self._required_checks),
                    workflow_name="",
                    event="",
                )
            )

        if not rows:
            return []

        incomplete = _is_incomplete(rows)
        entry = GHPRChecksCacheEntry(
            ts=int(now),
            ver=int(CACHE_VER),
            rows=tuple(rows),
            check_runs_etag=str(new_check_runs_etag or ""),
            status_etag=str(new_status_etag or ""),
            incomplete=bool(incomplete),
        )
        _CACHE.put_entry_dict(key, entry.to_disk_dict())
        self.api._cache_write(cn, entries=1)
        return rows


def _apply_required_overlay(rows: Tuple[GHPRCheckRow, ...], *, required: Set[str]) -> List[GHPRCheckRow]:
    out: List[GHPRCheckRow] = []
    for r in rows:
        if r.is_required or (r.name in required):
            out.append(r if r.is_required else replace(r, is_required=True))
        else:
            out.append(r)
    return out


def _parse_iso(s: str) -> Optional[datetime]:
    ss = str(s or "").strip()
    if not ss:
        return None
    if ss.endswith("Z"):
        ss = ss[:-1] + "+00:00"
    return datetime.fromisoformat(ss)


def _format_dur(start_iso: str, end_iso: str) -> str:
    try:
        st = _parse_iso(start_iso)
        en = _parse_iso(end_iso)
        if not st or not en:
            return ""
        sec = int((en - st).total_seconds())
        if sec < 0:
            return ""
        m, s2 = divmod(sec, 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h {m}m"
        if m:
            return f"{m}m {s2}s"
        return f"{s2}s"
    except (ValueError, TypeError):
        return ""


def _is_incomplete(rows: List[GHPRCheckRow]) -> bool:
    completed_jobs_total = 0
    completed_jobs_without_duration = 0
    for row in rows:
        status = str(row.status_raw or "").lower()
        duration = str(row.duration or "").strip()
        url = str(row.url or "").strip()
        if status in {"pass", "fail", "skipped"} and "/actions/runs/" in url and "/job/" in url:
            completed_jobs_total += 1
            if not duration:
                completed_jobs_without_duration += 1
    if completed_jobs_total > 0 and completed_jobs_without_duration > 0:
        missing_pct = (completed_jobs_without_duration / completed_jobs_total) * 100
        return missing_pct > 30
    return False


def get_pr_checks_rows_cached(
    api: "GitHubAPIClient",
    *,
    owner: str,
    repo: str,
    pr_number: int,
    commit_sha: Optional[str] = None,
    head_sha: Optional[str] = None,
    required_checks: Optional[set] = None,
    ttl_s: int = 300,
    pr_updated_at_epoch: Optional[int] = None,
    skip_fetch: bool = False,
) -> List[GHPRCheckRow]:
    return PRChecksCached(
        api,
        required_checks=set(required_checks or set()),
        ttl_s=int(ttl_s),
        pr_updated_at_epoch=pr_updated_at_epoch,
        skip_fetch=bool(skip_fetch),
    ).get(owner=owner, repo=repo, pr_number=int(pr_number), commit_sha=commit_sha, head_sha=head_sha)



def get_cache_sizes() -> Tuple[int, int]:
    """Get cache sizes for stats reporting (memory count, initial disk count)."""
    return _CACHE.get_cache_sizes()


def get_entry_dict(key: str) -> Optional[Dict[str, Any]]:
    """Get entry dict from cache (for legacy __init__.py usage).
    
    Args:
        key: Cache key
        
    Returns:
        Entry dict or None if not found
    """
    return _CACHE.get_entry_dict(key)


def get_cache_file() -> Path:
    """Get cache file path (for legacy __init__.py usage).
    
    Returns:
        Path to cache file
    """
    return _CACHE._cache_file


def put_entry_dict(key: str, entry_dict: Dict[str, Any]) -> None:
    """Store entry dict in cache (for legacy __init__.py usage).
    
    Args:
        key: Cache key
        entry_dict: Entry dictionary to store
    """
    _CACHE.put_entry_dict(key, entry_dict)
