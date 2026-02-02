"""Types and helpers for PR checks caching.

This module exists to keep `common_github/__init__.py` from becoming even more monolithic and
to avoid circular imports when implementing `common_github/api/pr_checks_cached.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

from common_types import CIStatus


def parse_actions_run_id_from_url(url: str) -> str:
    """Extract a GitHub Actions run_id from a typical Actions/check URL.

    Examples:
      - https://github.com/owner/repo/actions/runs/18697156351
      - https://github.com/owner/repo/actions/runs/18697156351/job/53317461976
    """
    s = str(url or "")
    if "/actions/runs/" not in s:
        return ""
    rest = s.split("/actions/runs/", 1)[1]
    run_id = rest.split("/", 1)[0].split("?", 1)[0].strip()
    return run_id if run_id.isdigit() else ""


def parse_actions_job_id_from_url(url: str) -> str:
    """Extract a GitHub Actions job_id from a typical Actions job URL.

    Examples:
      https://github.com/OWNER/REPO/actions/runs/20732129035/job/59522167110 -> 59522167110
    """
    s = str(url or "").strip()
    if "/job/" not in s:
        return ""
    rest = s.split("/job/", 1)[1]
    job_id = rest.split("/", 1)[0].split("?", 1)[0].strip()
    return job_id if job_id.isdigit() else ""


@dataclass(frozen=True)
class GHPRCheckRow:
    """A single PR check row (derived from GitHub REST check-runs/status contexts)."""

    name: str
    status_raw: str
    duration: str = ""
    url: str = ""
    run_id: str = ""
    job_id: str = ""
    description: str = ""
    is_required: bool = False
    workflow_name: str = ""
    event: str = ""
    has_timeout_annotation: bool = False  # True if check-run has "exceeded the maximum execution time" annotation

    @property
    def status_norm(self) -> str:
        s = (self.status_raw or "").strip().lower()
        if s in {"pass", CIStatus.SUCCESS.value}:
            return CIStatus.SUCCESS.value
        if s in {CIStatus.SKIPPED.value, "skip", CIStatus.NEUTRAL.value}:
            return CIStatus.SUCCESS.value
        if s in {"fail", "failure", "timed_out", "action_required"}:
            return CIStatus.FAILURE.value
        # CRITICAL: Treat timeout-cancelled as FAILURE (not cancelled)
        if s in {"cancelled", "canceled"}:
            if self.has_timeout_annotation:
                return CIStatus.FAILURE.value  # Timeout = failure
            return CIStatus.CANCELLED.value  # User cancellation
        if s in {"in_progress", "running"}:
            return CIStatus.IN_PROGRESS.value
        if s in {"pending", "queued"}:
            return CIStatus.PENDING.value
        return CIStatus.UNKNOWN.value


_GHPR_CHECK_ROW_DISK_KEYS: set[str] = {
    "name",
    "status_raw",
    "duration",
    "url",
    "run_id",
    "job_id",
    "description",
    "is_required",
    "workflow_name",
    "event",
    "has_timeout_annotation",
}


def _ghpr_check_row_to_disk_dict(row: GHPRCheckRow) -> Dict[str, Any]:
    return {
        "name": row.name,
        "status_raw": row.status_raw,
        "duration": row.duration,
        "url": row.url,
        "run_id": row.run_id,
        "job_id": row.job_id,
        "description": row.description,
        "is_required": bool(row.is_required),
        "workflow_name": row.workflow_name,
        "event": row.event,
        "has_timeout_annotation": bool(row.has_timeout_annotation),
    }


def _ghpr_check_row_from_disk_dict_strict(*, d: Any, cache_file: Path, entry_key: str) -> GHPRCheckRow:
    if not isinstance(d, dict):
        raise RuntimeError(f"Invalid pr_checks cache entry row type in {cache_file}: key={entry_key!r} type={type(d)}")
    extra = set(d.keys()) - _GHPR_CHECK_ROW_DISK_KEYS
    if extra:
        raise RuntimeError(
            f"Unknown pr_checks row fields in {cache_file}: key={entry_key!r} extra={sorted(extra)}; "
            f"expected subset of {sorted(_GHPR_CHECK_ROW_DISK_KEYS)}"
        )
    return GHPRCheckRow(
        name=str(d.get("name", "") or ""),
        status_raw=str(d.get("status_raw", "") or ""),
        duration=str(d.get("duration", "") or ""),
        url=str(d.get("url", "") or ""),
        run_id=str(d.get("run_id", "") or ""),
        job_id=str(d.get("job_id", "") or ""),
        description=str(d.get("description", "") or ""),
        is_required=bool(d.get("is_required", False)),
        workflow_name=str(d.get("workflow_name", "") or ""),
        event=str(d.get("event", "") or ""),
        has_timeout_annotation=bool(d.get("has_timeout_annotation", False)),
    )


@dataclass(frozen=True, slots=True)
class GHPRChecksCacheEntry:
    """Typed cache entry for PR check rows (additive schema only)."""

    ts: int
    ver: int
    rows: Tuple[GHPRCheckRow, ...]
    check_runs_etag: str = ""
    status_etag: str = ""
    incomplete: bool = False

    _DISK_KEYS: set[str] = field(
        default_factory=lambda: {"ts", "ver", "rows", "check_runs_etag", "status_etag", "incomplete"},
        init=False,
        repr=False,
    )

    def to_disk_dict(self) -> Dict[str, Any]:
        return {
            "ts": int(self.ts),
            "ver": int(self.ver),
            "rows": [_ghpr_check_row_to_disk_dict(r) for r in self.rows],
            "check_runs_etag": str(self.check_runs_etag or ""),
            "status_etag": str(self.status_etag or ""),
            "incomplete": bool(self.incomplete),
        }

    @classmethod
    def from_disk_dict_strict(cls, *, d: Any, cache_file: Path, entry_key: str) -> "GHPRChecksCacheEntry":
        if not isinstance(d, dict):
            raise RuntimeError(f"Invalid pr_checks cache entry type in {cache_file}: key={entry_key!r} type={type(d)}")
        extra = set(d.keys()) - {"ts", "ver", "rows", "check_runs_etag", "status_etag", "incomplete"}
        if extra:
            raise RuntimeError(
                f"Unknown pr_checks cache entry fields in {cache_file}: key={entry_key!r} extra={sorted(extra)}; "
                f"expected subset of {sorted({'ts','ver','rows','check_runs_etag','status_etag','incomplete'})}"
            )
        ts = int(d.get("ts", 0) or 0)
        ver = int(d.get("ver", 0) or 0)
        rows_in = d.get("rows") or []
        if not isinstance(rows_in, list):
            raise RuntimeError(
                f"Invalid pr_checks cache entry rows type in {cache_file}: key={entry_key!r} type={type(rows_in)}"
            )
        rows = tuple(_ghpr_check_row_from_disk_dict_strict(d=r, cache_file=cache_file, entry_key=entry_key) for r in rows_in)
        check_runs_etag = str(d.get("check_runs_etag", "") or "")
        status_etag = str(d.get("status_etag", "") or "")
        incomplete = bool(d.get("incomplete", False))
        return cls(ts=ts, ver=ver, rows=rows, check_runs_etag=check_runs_etag, status_etag=status_etag, incomplete=incomplete)

