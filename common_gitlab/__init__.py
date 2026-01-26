# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GitLab API client and cached API resources for dynamo-utils.

This mirrors the `common_github/` structure:
- `common_gitlab/` defines the API client + shared stats helpers
- `common_gitlab/api/*_cached.py` contains per-resource caching + fetch logic
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .exceptions import (
    GitLabAPIError,
    GitLabAuthError,
    GitLabForbiddenError,
    GitLabNotFoundError,
    GitLabRequestError,
)

from .api.mr_pipelines_cached import get_cached_merge_request_pipelines
from .api.pipeline_jobs_cached import (
    get_cached_pipeline_job_counts,
    get_cached_pipeline_job_details,
)
from .api.pipeline_status_cached import get_cached_pipeline_status
from .api.registry_images_cached import get_cached_registry_images_for_shas

_logger = logging.getLogger(__name__)
_MR_RE = re.compile(r"#(?P<iid>\d+)")


# ======================================================================================
# GLOBAL API + CACHE STATISTICS (GitLab)
# ======================================================================================
# Mirror the GitHub statistics shape so dashboards can render GitLab with the same
# key conventions: gitlab.cache.* and gitlab.rest.by_category.*.


class _GitLabCacheStats:
    """Kept for symmetry with GitHub; reserved for non-resource cache stats."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._reserved = 0

    def to_dict(self) -> Dict[str, dict]:
        return {}


class _GitLabAPIStats:
    """Global singleton for tracking GitLab API REST + cache statistics."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        # REST call stats (by logical label/category).
        self.rest_calls_total = 0
        self.rest_calls_by_label: Dict[str, int] = {}
        self.rest_success_total = 0
        self.rest_success_by_label: Dict[str, int] = {}
        self.rest_time_total_s = 0.0
        self.rest_time_by_label_s: Dict[str, float] = {}

        # Error stats
        self.rest_errors_total = 0
        self.rest_errors_by_status: Dict[int, int] = {}
        self.rest_errors_by_label_status: Dict[str, Dict[int, int]] = {}
        self.rest_last_error: Dict[str, Any] = {}
        self.rest_last_error_label = ""

        # Generic cache stats (operations-level)
        self.cache_hits: Dict[str, int] = {}
        self.cache_misses: Dict[str, int] = {}
        self.cache_writes_ops: Dict[str, int] = {}
        self.cache_writes_entries: Dict[str, int] = {}


# Global instances (match GitHub naming pattern)
GITLAB_CACHE_STATS = _GitLabCacheStats()
GITLAB_API_STATS = _GitLabAPIStats()


class GitLabAPIClient:
    """GitLab REST API client (lightweight; caching lives in `common_gitlab/api/`)."""

    @staticmethod
    def get_gitlab_token_from_file() -> Optional[str]:
        """Get GitLab token from `~/.config/gitlab-token` (best-effort)."""
        try:
            token_file = Path.home() / ".config" / "gitlab-token"
            if token_file.exists():
                return token_file.read_text().strip()
        except OSError:
            pass
        return None

    def __init__(self, token: Optional[str] = None, base_url: str = "https://gitlab-master.nvidia.com"):
        # Token priority: 1) provided token, 2) environment variable, 3) config file
        self.token = token or os.environ.get("GITLAB_TOKEN") or self.get_gitlab_token_from_file()
        self.base_url = str(base_url or "").rstrip("/")
        self.headers: Dict[str, str] = {}
        if self.token:
            self.headers["PRIVATE-TOKEN"] = self.token

        # Per-run REST stats (best-effort; label-based to match GitHub dashboard conventions).
        self._rest_calls_total: int = 0
        self._rest_calls_by_label: Dict[str, int] = {}
        self._rest_success_total: int = 0
        self._rest_errors_total: int = 0
        self._rest_time_total_s: float = 0.0
        self._rest_time_by_label_s: Dict[str, float] = {}
        self._rest_errors_by_status: Dict[int, int] = {}

        # Match GitHub client shape: allow resource.get(cache_only_mode=...) override.
        self.cache_only_mode: bool = False

        # Inflight request deduplication (mirror GitHub client).
        self._inflight_locks_mu = threading.Lock()
        self._inflight_locks: Dict[str, threading.Lock] = {}

    def has_token(self) -> bool:
        return self.token is not None

    def _inflight_lock(self, key: str) -> "threading.Lock":
        """Return a per-key lock to dedupe concurrent network fetches across threads."""
        k = str(key or "")
        if not k:
            k = "__default__"
        with self._inflight_locks_mu:
            lk = self._inflight_locks.get(k)
            if lk is None:
                lk = threading.Lock()
                self._inflight_locks[k] = lk
            return lk

    def _rest_record(self, *, label: str, endpoint: str, status_code: Optional[int], dt_s: float) -> None:
        """Record one REST call in GitLab stats (GitHub-compatible shape)."""
        lbl = str(label or "").strip() or "unknown"
        ep = str(endpoint or "")
        dt = max(0.0, float(dt_s or 0.0))

        # Instance-local aggregation
        self._rest_calls_total += 1
        self._rest_calls_by_label[lbl] = int(self._rest_calls_by_label.get(lbl, 0) or 0) + 1
        self._rest_time_total_s += float(dt)
        self._rest_time_by_label_s[lbl] = float(self._rest_time_by_label_s.get(lbl, 0.0) or 0.0) + float(dt)

        # Global aggregation
        try:
            GITLAB_API_STATS.rest_calls_total += 1
            GITLAB_API_STATS.rest_calls_by_label[lbl] = int(GITLAB_API_STATS.rest_calls_by_label.get(lbl, 0) or 0) + 1
            GITLAB_API_STATS.rest_time_total_s += float(dt)
            GITLAB_API_STATS.rest_time_by_label_s[lbl] = float(GITLAB_API_STATS.rest_time_by_label_s.get(lbl, 0.0) or 0.0) + float(dt)
        except Exception:
            pass

        if status_code is None:
            return
        try:
            sc = int(status_code)
        except (ValueError, TypeError):
            sc = -1
        if 200 <= sc < 300:
            self._rest_success_total += 1
            try:
                GITLAB_API_STATS.rest_success_total += 1
                GITLAB_API_STATS.rest_success_by_label[lbl] = int(GITLAB_API_STATS.rest_success_by_label.get(lbl, 0) or 0) + 1
            except Exception:
                pass
        elif sc >= 400:
            self._rest_errors_total += 1
            self._rest_errors_by_status[sc] = int(self._rest_errors_by_status.get(sc, 0) or 0) + 1
            try:
                GITLAB_API_STATS.rest_errors_total += 1
                GITLAB_API_STATS.rest_errors_by_status[sc] = int(GITLAB_API_STATS.rest_errors_by_status.get(sc, 0) or 0) + 1
                by_lbl = GITLAB_API_STATS.rest_errors_by_label_status.get(lbl)
                if not isinstance(by_lbl, dict):
                    by_lbl = {}
                    GITLAB_API_STATS.rest_errors_by_label_status[lbl] = by_lbl
                by_lbl[sc] = int(by_lbl.get(sc, 0) or 0) + 1
                GITLAB_API_STATS.rest_last_error = {"status": int(sc), "endpoint": ep}
                GITLAB_API_STATS.rest_last_error_label = lbl
            except Exception:
                pass

    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 10,
        *,
        label: Optional[str] = None,
    ) -> Optional[Any]:
        """Make a GET request to the GitLab API and return JSON (dict/list) or raise."""
        ep = str(endpoint or "")
        t0 = time.monotonic()
        status_code: Optional[int] = None
        lbl = str(label or "").strip() or "unknown"
        url = f"{self.base_url}{ep}" if ep.startswith("/") else f"{self.base_url}/{ep}"

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=timeout)
            try:
                status_code = int(response.status_code)
            except (ValueError, TypeError):
                status_code = None

            if response.status_code == 401:
                raise GitLabAuthError(status_code=401, endpoint=ep, message="GitLab API returned 401 Unauthorized. Check your token.")
            if response.status_code == 403:
                raise GitLabForbiddenError(status_code=403, endpoint=ep, message="GitLab API returned 403 Forbidden. Token may lack permissions.")
            if response.status_code == 404:
                raise GitLabNotFoundError(status_code=404, endpoint=ep, message=f"GitLab API returned 404 Not Found for {endpoint}")

            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:  # type: ignore[union-attr]
            if status_code is not None:
                GITLAB_API_STATS.rest_last_error = {"status": int(status_code), "endpoint": ep}
                GITLAB_API_STATS.rest_last_error_label = lbl
            raise GitLabRequestError(status_code=int(status_code or 0), endpoint=ep, message=f"GitLab API request failed for {endpoint}: {e}")
        finally:
            dt = max(0.0, time.monotonic() - t0)
            self._rest_record(label=lbl, endpoint=ep, status_code=status_code, dt_s=float(dt))

    def get_rest_call_stats(self) -> Dict[str, Any]:
        """Return best-effort REST call stats for the current process/run."""
        return {
            "total": int(self._rest_calls_total),
            "success_total": int(self._rest_success_total),
            "error_total": int(self._rest_errors_total),
            "time_total_s": float(self._rest_time_total_s),
            "by_label": dict(sorted(dict(self._rest_calls_by_label or {}).items(), key=lambda kv: (-int(kv[1] or 0), kv[0]))),
            "time_by_label_s": dict(sorted(dict(self._rest_time_by_label_s or {}).items(), key=lambda kv: (-float(kv[1] or 0.0), kv[0]))),
            "errors_by_status": dict(
                sorted(dict(self._rest_errors_by_status or {}).items(), key=lambda kv: (-int(kv[1] or 0), int(kv[0] or 0)))
            ),
        }

    def _cache_hit(self, name: str) -> None:
        k = str(name or "").strip() or "unknown"
        GITLAB_API_STATS.cache_hits[k] = int(GITLAB_API_STATS.cache_hits.get(k, 0) or 0) + 1

    def _cache_miss(self, name: str) -> None:
        k = str(name or "").strip() or "unknown"
        GITLAB_API_STATS.cache_misses[k] = int(GITLAB_API_STATS.cache_misses.get(k, 0) or 0) + 1

    def _cache_write(self, name: str, *, entries: int = 0) -> None:
        k = str(name or "").strip() or "unknown"
        GITLAB_API_STATS.cache_writes_ops[k] = int(GITLAB_API_STATS.cache_writes_ops.get(k, 0) or 0) + 1
        if int(entries or 0) > 0:
            GITLAB_API_STATS.cache_writes_entries[k] = int(GITLAB_API_STATS.cache_writes_entries.get(k, 0) or 0) + int(entries or 0)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return best-effort per-run cache hit/miss stats for dashboards (GitHub-compatible)."""
        hits_by = dict(GITLAB_API_STATS.cache_hits or {})
        misses_by = dict(GITLAB_API_STATS.cache_misses or {})
        writes_ops_by = dict(GITLAB_API_STATS.cache_writes_ops or {})
        writes_entries_by = dict(GITLAB_API_STATS.cache_writes_entries or {})
        return {
            "hits_total": int(sum(int(v or 0) for v in hits_by.values())),
            "misses_total": int(sum(int(v or 0) for v in misses_by.values())),
            "writes_ops_total": int(sum(int(v or 0) for v in writes_ops_by.values())),
            "writes_entries_total": int(sum(int(v or 0) for v in writes_entries_by.values())),
            "hits_by": hits_by,
            "misses_by": misses_by,
            "writes_ops_by": writes_ops_by,
            "writes_entries_by": writes_entries_by,
            "entries": {},
        }

    @staticmethod
    def parse_mr_number_from_message(message: str) -> Optional[int]:
        """Parse MR/PR number from commit subject (e.g. '... (#1234)')."""
        m = _MR_RE.search(str(message or ""))
        if not m:
            return None
        try:
            return int(m.group("iid"))
        except (ValueError, TypeError):
            return None

    # -----------------------------------------------------------------------------
    # Cached resources (delegated to common_gitlab/api/*_cached.py)
    # -----------------------------------------------------------------------------

    def get_cached_registry_images_for_shas(
        self,
        *,
        project_id: str,
        registry_id: str,
        sha_list: List[str],
        sha_to_datetime: Optional[Dict[str, Any]] = None,
        cache_file: str = "gitlab_commit_sha.json",
        skip_fetch: bool = False,
    ) -> Dict[str, List[Dict[str, Any]]]:
        return get_cached_registry_images_for_shas(
            self,
            project_id=project_id,
            registry_id=registry_id,
            sha_list=sha_list,
            sha_to_datetime=sha_to_datetime,
            cache_file=cache_file,
            skip_fetch=skip_fetch,
        )

    def get_cached_pipeline_status(
        self, sha_list: List[str], *, cache_file: str = "gitlab_pipeline_status.json", skip_fetch: bool = False
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        return get_cached_pipeline_status(self, sha_list=sha_list, cache_file=cache_file, skip_fetch=skip_fetch)

    def get_cached_pipeline_job_counts(
        self, pipeline_ids: List[int], *, cache_file: str = "gitlab_pipeline_jobs.json", skip_fetch: bool = False
    ) -> Dict[int, Optional[Dict[str, int]]]:
        return get_cached_pipeline_job_counts(self, pipeline_ids=pipeline_ids, cache_file=cache_file, skip_fetch=skip_fetch)

    def get_cached_pipeline_job_details(
        self,
        pipeline_ids: List[int],
        *,
        cache_file: str = "gitlab_pipeline_jobs_details.json",
        skip_fetch: bool = False,
    ) -> Dict[int, Optional[Dict[str, Any]]]:
        return get_cached_pipeline_job_details(self, pipeline_ids=pipeline_ids, cache_file=cache_file, skip_fetch=skip_fetch)

    def get_cached_merge_request_pipelines(
        self,
        mr_numbers: List[int],
        *,
        project_id: str = "169905",
        cache_file: str = "gitlab_mr_pipelines.json",
        skip_fetch: bool = False,
    ) -> Dict[int, Optional[Dict[str, Any]]]:
        return get_cached_merge_request_pipelines(
            self, mr_numbers=mr_numbers, project_id=project_id, cache_file=cache_file, skip_fetch=skip_fetch
        )


__all__ = ["GitLabAPIClient"]

