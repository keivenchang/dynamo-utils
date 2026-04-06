# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Commit check-runs cached API (REST).

Fetches check-runs for a specific commit SHA (not a PR). This is the building
block for post-merge CI display: after a PR is merged, post-merge workflows
run on the merge commit SHA (triggered by `on: push` to main). Those results
are only visible via `GET /repos/{owner}/{repo}/commits/{sha}/check-runs`,
NOT via the PR's head SHA.

This module mirrors the architecture of pr_checks_cached.py but is keyed by
commit SHA instead of PR number, and does not need a head_sha lookup step.

Cache file: commit_checks.json
Cache key:  {owner}/{repo}:commit:{sha}
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from cache.cache_base import BaseDiskCache

logger = logging.getLogger(__name__)

from .. import GITHUB_API_STATS
from ..pr_checks_types import (
    GHPRCheckRow,
    GHPRChecksCacheEntry,
    parse_actions_job_id_from_url,
    parse_actions_run_id_from_url,
)
from .base_cached import CachedResourceBase

if TYPE_CHECKING:
    from .. import GitHubAPIClient


CACHE_FILE_DEFAULT = "commit_checks.json"
CACHE_NAME = "commit_checks"
API_CALL_FORMAT = (
    "REST GET /repos/{owner}/{repo}/commits/{sha}/check-runs?per_page=100 (etag)\n"
    "REST GET /repos/{owner}/{repo}/commits/{sha}/status (etag)"
)


class _CommitChecksCache(BaseDiskCache):
    _SCHEMA_VERSION = 1

    def __init__(self, *, cache_file: Path):
        super().__init__(cache_file=cache_file, schema_version=self._SCHEMA_VERSION)

    def get_entry_dict(self, key: str) -> Optional[Dict[str, Any]]:
        with self._mu:
            self._load_once()
            return self._check_item(key)

    def put_entry_dict(self, key: str, entry_dict: Dict[str, Any]) -> None:
        with self._mu:
            self._load_once()
            self._set_item(key, entry_dict)
            self._persist()


def _get_cache_file() -> Path:
    try:
        _module_dir = Path(__file__).resolve().parent.parent
        if str(_module_dir) not in sys.path:
            sys.path.insert(0, str(_module_dir))
        import common
        return common.dynamo_utils_cache_dir() / CACHE_FILE_DEFAULT
    except ImportError:
        return Path.home() / ".cache" / "dynamo-utils" / CACHE_FILE_DEFAULT


_CACHE = _CommitChecksCache(cache_file=_get_cache_file())


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


class CommitChecksCached(CachedResourceBase[List[GHPRCheckRow]]):
    """Fetch and cache check-runs for a specific commit SHA."""

    def __init__(
        self,
        api: "GitHubAPIClient",
        *,
        ttl_s: int,
        skip_fetch: bool,
    ):
        super().__init__(api)
        self._ttl_s = int(ttl_s)
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
        commit_sha = str(kwargs["commit_sha"])
        return f"{owner}/{repo}:commit:{commit_sha[:12]}"

    def empty_value(self) -> List[GHPRCheckRow]:
        return []

    def cache_read(self, *, key: str) -> Optional[Dict[str, Any]]:
        return _CACHE.get_entry_dict(key)

    def value_from_cache_entry(self, *, entry: Dict[str, Any]) -> List[GHPRCheckRow]:
        raise RuntimeError("CommitChecksCached.value_from_cache_entry should not be called directly")

    def is_cache_entry_fresh(self, *, entry: Dict[str, Any], now: int) -> bool:
        raise RuntimeError("CommitChecksCached.is_cache_entry_fresh should not be called directly")

    def cache_write(self, *, key: str, value: List[GHPRCheckRow], meta: Dict[str, Any]) -> None:
        raise RuntimeError("CommitChecksCached.cache_write should not be called directly")

    def fetch(self, **kwargs: Any) -> Tuple[List[GHPRCheckRow], Dict[str, Any]]:
        raise RuntimeError("CommitChecksCached.fetch should not be called directly")

    def _check_for_timeout_annotation(self, owner: str, repo: str, check_run_id: str) -> bool:
        annots_url = f"{self.api.base_url}/repos/{owner}/{repo}/check-runs/{check_run_id}/annotations"
        try:
            old_cache_only = self.api.cache_only_mode
            old_budget_max = self.api._rest_budget_max
            old_budget_exhausted = self.api._rest_budget_exhausted
            self.api.cache_only_mode = False
            self.api._rest_budget_max = None
            self.api._rest_budget_exhausted = False
            try:
                annots_resp = self.api._rest_get(annots_url, timeout=5)
                if annots_resp is not None:
                    annots_data = annots_resp.json() if annots_resp else []
                    if isinstance(annots_data, list):
                        return any(
                            isinstance(annot, dict) and
                            "maximum execution time" in str(annot.get("message", "") or "").lower()
                            for annot in annots_data
                        )
            finally:
                self.api.cache_only_mode = old_cache_only
                self.api._rest_budget_max = old_budget_max
                self.api._rest_budget_exhausted = old_budget_exhausted
        except Exception:
            pass
        return False

    def get(self, **kwargs: Any) -> List[GHPRCheckRow]:
        owner = str(kwargs["owner"])
        repo = str(kwargs["repo"])
        commit_sha = str(kwargs["commit_sha"]).strip()
        if not commit_sha:
            return []

        cn = self.cache_name
        key = self.cache_key(owner=owner, repo=repo, commit_sha=commit_sha)
        now = int(datetime.now(timezone.utc).timestamp())
        effective_ttl_s = int(self._ttl_s)

        CACHE_VER = 1
        MIN_CACHE_VER = 1

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

                if ts and ((now - ts) <= max(0, int(effective_ttl_s)) or self.api.cache_only_mode) and not incomplete and not has_etag:
                    if ver >= MIN_CACHE_VER:
                        self.api._cache_hit(cn)
                        return list(ent.rows)
            except (ValueError, TypeError, RuntimeError):
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
                if ent2.ver >= MIN_CACHE_VER:
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

        # Fetch check-runs
        check_runs_url = f"{self.api.base_url}/repos/{owner}/{repo}/commits/{commit_sha}/check-runs"
        check_runs_resp = self.api._rest_get(
            check_runs_url,
            params={"per_page": 100},
            timeout=10,
            etag=stale_check_runs_etag if stale_check_runs_etag else None,
        )
        new_check_runs_etag = ""
        if hasattr(check_runs_resp, "headers"):
            new_check_runs_etag = str(check_runs_resp.headers.get("ETag", "") or "")

        # 304: return cached rows with refreshed timestamp
        if check_runs_resp is not None and int(getattr(check_runs_resp, "status_code", 0) or 0) == 304:
            if cached_entry_dict is not None:
                try:
                    ent304 = GHPRChecksCacheEntry.from_disk_dict_strict(
                        d=cached_entry_dict,
                        cache_file=_CACHE._cache_file,
                        entry_key=key,
                    )
                    if ent304.ver >= MIN_CACHE_VER:
                        refreshed = GHPRChecksCacheEntry(
                            ts=int(now),
                            ver=int(ent304.ver),
                            rows=ent304.rows,
                            check_runs_etag=stale_check_runs_etag,
                            status_etag=ent304.status_etag,
                            incomplete=ent304.incomplete,
                        )
                        _CACHE.put_entry_dict(key, refreshed.to_disk_dict())
                        self.api._cache_write(cn, entries=1)
                        return list(ent304.rows)
                except (ValueError, TypeError, RuntimeError):
                    pass
            return []

        try:
            data = check_runs_resp.json() if check_runs_resp is not None else {}
        except (ValueError, RuntimeError):
            data = {}
        check_runs = data.get("check_runs") if isinstance(data, dict) else None
        if not isinstance(check_runs, list):
            check_runs = []

        # Fetch status contexts
        status_url = f"{self.api.base_url}/repos/{owner}/{repo}/commits/{commit_sha}/status"
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
            try:
                statuses_data = status_resp.json() if status_resp is not None else {}
            except (ValueError, RuntimeError):
                statuses_data = {}
            statuses = statuses_data.get("statuses") if isinstance(statuses_data, dict) else []
            if not isinstance(statuses, list):
                statuses = []

        # Prefetch workflow metadata
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

        # Build rows
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

            has_timeout_annotation = False
            check_run_id = str(cr.get("id", "") or "").strip()
            if conclusion in ("cancelled", "canceled") and check_run_id and check_run_id.isdigit():
                has_timeout_annotation = self._check_for_timeout_annotation(owner, repo, check_run_id)

            rows.append(
                GHPRCheckRow(
                    name=name,
                    status_raw=status_raw,
                    duration=duration,
                    url=url,
                    run_id=run_id,
                    job_id=job_id,
                    description=description,
                    is_required=False,
                    workflow_name=workflow_name,
                    event=event,
                    has_timeout_annotation=has_timeout_annotation,
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
                    is_required=False,
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


def get_commit_checks_rows_cached(
    api: "GitHubAPIClient",
    *,
    owner: str,
    repo: str,
    commit_sha: str,
    ttl_s: int = 300,
    skip_fetch: bool = False,
) -> List[GHPRCheckRow]:
    return CommitChecksCached(
        api,
        ttl_s=int(ttl_s),
        skip_fetch=bool(skip_fetch),
    ).get(owner=owner, repo=repo, commit_sha=str(commit_sha))


def get_cache_sizes() -> Tuple[int, int]:
    return _CACHE.get_cache_sizes()
