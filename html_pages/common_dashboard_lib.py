#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Common dashboard helpers shared by HTML generators under dynamo-utils/html_pages/.

This file intentionally groups previously-split helper modules into one place to:
- avoid UI drift between dashboards
- reduce small-module sprawl
- keep <pre>-safe tree rendering + check-line rendering consistent
- centralize workflow-graph parsing (derived from .github/workflows/*.yml)
"""

from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

from common import GitHubAPIClient, classify_ci_kind

# ======================================================================================
# Shared ordering + default-expand policies
# ======================================================================================

def sort_github_check_runs_by_name(check_runs: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Return check runs sorted by job name (stable), then job id/url for disambiguation."""
    def _label(name: str) -> str:
        k = classify_ci_kind(name)
        return f"{k}: {name}" if k and k != "check" else name

    def _key(cr: Dict[str, object]) -> Tuple[str, str, str]:
        try:
            name = str(cr.get("name", "") or "").strip()
        except Exception:
            name = ""
        try:
            url = str(cr.get("html_url", "") or cr.get("details_url", "") or "").strip()
        except Exception:
            url = ""
        jid = extract_actions_job_id_from_url(url)
        return (_label(name).lower(), str(jid or ""), url)

    try:
        xs = [cr for cr in (check_runs or []) if isinstance(cr, dict)]
        return sorted(xs, key=_key)
    except Exception:
        return [cr for cr in (check_runs or []) if isinstance(cr, dict)]


def sort_pr_check_rows_by_name(rows: List[object]) -> List[object]:
    """Return PR check rows sorted by display label (stable): 'kind: name' (e.g. lint/test/build)."""
    def _label(name: str) -> str:
        k = classify_ci_kind(name)
        return f"{k}: {name}" if k and k != "check" else name

    def _key(r: object) -> Tuple[str, str]:
        nm = ""
        url = ""
        try:
            nm = str(getattr(r, "name", "") or "").strip()
        except Exception:
            nm = ""
        try:
            url = str(getattr(r, "url", "") or "").strip()
        except Exception:
            url = ""
        jid = extract_actions_job_id_from_url(url)
        return (_label(nm).lower(), str(jid or url))

    try:
        return sorted(list(rows or []), key=_key)
    except Exception:
        return list(rows or [])


def ci_should_expand_by_default(*, rollup_status: str, has_required_failure: bool) -> bool:
    """Shared UX rule: expand only when something truly needs attention.

    - expand for required failures (red ✗)
    - do NOT auto-expand long/step-heavy jobs by default (even if they have subsections)
    - do NOT auto-expand for optional failures (⚠), in-progress/pending/cancelled, unknown-only leaves, or all-green trees
    """
    if bool(has_required_failure):
        return True
    return False

# ======================================================================================
# Shared UI snippets
# ======================================================================================

# Style for the optional pass count in the "REQ+OPT✓" compact CI summary.
PASS_PLUS_STYLE = "font-size: 10px; font-weight: 600; opacity: 0.9;"


# ======================================================================================
# Shared tree UI rendering (<pre>-safe)
# ======================================================================================

@dataclass(frozen=True)
class TreeNodeVM:
    """View-model for a single tree node line.

    - label_html: full HTML for the line content (excluding the tree connectors).
    - children: child nodes.
    - collapsible: if True, render a triangle placeholder (▶/▼) and allow toggling children.
    - default_expanded: initial state for collapsible nodes.
    - triangle_tooltip: optional title for the triangle element.
    - node_key: a stable key for the logical node (used for debugging/caching, not DOM ids).
    """

    node_key: str
    label_html: str
    children: List["TreeNodeVM"] = field(default_factory=list)
    collapsible: bool = False
    default_expanded: bool = False
    triangle_tooltip: Optional[str] = None


def _hash10(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


_ACTIONS_JOB_ID_RE = re.compile(r"/job/([0-9]+)(?:$|[/?#])")


def extract_actions_job_id_from_url(url: str) -> str:
    """Best-effort extraction of the numeric job id from GitHub Actions job URLs."""
    try:
        m = _ACTIONS_JOB_ID_RE.search(str(url or ""))
        return str(m.group(1)) if m else ""
    except Exception:
        return ""


def disambiguate_check_run_name(name: str, url: str, *, name_counts: Dict[str, int]) -> str:
    """If multiple runs share the same name, add a stable suffix so the UI doesn't show duplicates."""
    try:
        n = str(name or "")
        if int(name_counts.get(n, 0) or 0) <= 1:
            return n
        jid = extract_actions_job_id_from_url(str(url or ""))
        if jid:
            return f"{n} [job {jid}]"
        u = str(url or "")
        if u:
            return f"{n} [{_hash10(u)}]"
        return n
    except Exception:
        return str(name or "")


def _triangle_html(*, expanded: bool, children_id: str, tooltip: Optional[str], parent_children_id: Optional[str]) -> str:
    ch = "▼" if expanded else "▶"
    title_attr = f' title="{html.escape(tooltip or "", quote=True)}"' if tooltip else ""
    parent_attr = (
        f' data-parent-children-id="{html.escape(parent_children_id, quote=True)}"'
        if parent_children_id
        else ""
    )
    return (
        f'<span style="display: inline-block; width: 12px; margin-right: 2px; color: #0969da; '
        f'cursor: pointer; user-select: none;"{title_attr} '
        f'data-children-id="{html.escape(children_id, quote=True)}"{parent_attr} '
        f'onclick="toggleTreeChildren(\'{html.escape(children_id, quote=True)}\', this)">{ch}</span>'
    )


def _triangle_placeholder_html() -> str:
    return '<span style="display: inline-block; width: 12px; margin-right: 2px;"></span>'


def render_tree_pre_lines(root_nodes: List[TreeNodeVM]) -> List[str]:
    """Render a forest into lines suitable to be joined with \n and placed inside <pre>."""

    out: List[str] = []
    # NOTE: This function may be called multiple times per HTML page (e.g. per-repo / per-PR).
    # DOM ids must be unique across the whole document, not just per call.
    global _TREE_RENDER_CALL_SEQ
    try:
        _TREE_RENDER_CALL_SEQ += 1
    except NameError:
        _TREE_RENDER_CALL_SEQ = 1
    render_call_id = _TREE_RENDER_CALL_SEQ
    next_dom_id = 0

    def alloc_children_id() -> str:
        # Must be unique within the document. Deterministic IDs (e.g., hashed from labels)
        # are not safe because the same logical node label/template can appear many times.
        nonlocal next_dom_id
        next_dom_id += 1
        return f"tree_children_{render_call_id:x}_{next_dom_id:x}"

    def render_node(
        node: TreeNodeVM,
        prefix: str,
        is_last: bool,
        is_root: bool,
        parent_children_id: Optional[str],
    ) -> None:
        # Tree connector
        if not is_root:
            connector = "└─" if is_last else "├─"
            current_prefix = prefix + connector + " "
        else:
            current_prefix = ""

        # Triangle UI
        children_id: Optional[str] = None
        if node.collapsible:
            if node.children:
                children_id = alloc_children_id()
                tri = _triangle_html(
                    expanded=bool(node.default_expanded),
                    children_id=children_id,
                    tooltip=node.triangle_tooltip,
                    parent_children_id=parent_children_id,
                )
            else:
                tri = _triangle_placeholder_html()
        else:
            tri = ""

        line = (node.label_html or "").strip()
        if line:
            out.append(current_prefix + tri + line)

        # Children rendering
        if not node.children:
            return

        # Compute child prefix continuation
        if is_root:
            child_prefix = ""
        else:
            child_prefix = prefix + ("   " if is_last else "│  ")

        # If collapsible: wrap children in a span appended to the end of the parent line.
        if node.collapsible and children_id and out:
            disp = "inline" if node.default_expanded else "none"
            out[-1] = out[-1] + f'<span id="{children_id}" style="display: {disp};">'
            child_lines_start_idx = len(out)

            for idx, ch in enumerate(node.children):
                render_node(
                    ch,
                    child_prefix,
                    idx == len(node.children) - 1,
                    False,
                    children_id,
                )

            # Close children span on the last rendered child line, or on the parent line if no child line rendered.
            if len(out) > child_lines_start_idx:
                out[-1] = out[-1] + "</span>"
            else:
                out[-1] = out[-1] + "</span>"
            return

        # Non-collapsible: always render children normally.
        for idx, ch in enumerate(node.children):
            render_node(
                ch,
                child_prefix,
                idx == len(node.children) - 1,
                False,
                parent_children_id,
            )

    for i, n in enumerate(root_nodes):
        render_node(n, prefix="", is_last=(i == len(root_nodes) - 1), is_root=True, parent_children_id=None)

    return out


# ======================================================================================
# Shared GitHub/GitLab check/job line rendering (HTML)
# ======================================================================================

def _parse_utc_ts_prefix(line: str) -> Optional[datetime]:
    """Parse a GitHub Actions log timestamp prefix like '2025-11-29T21:02:44.7091912Z ...'."""
    try:
        s = str(line or "")
        if len(s) < 22 or "T" not in s or "Z" not in s:
            return None
        head = s.split(" ", 1)[0].strip()
        if not head.endswith("Z"):
            return None
        # datetime.fromisoformat doesn't accept 'Z' in older versions; normalize to +00:00.
        return datetime.fromisoformat(head[:-1] + "+00:00")
    except Exception:
        return None


def _format_duration_short(seconds: float) -> str:
    """Format seconds as a short duration like '3s', '2m 10s', '1h 4m'."""
    try:
        s = int(round(float(seconds)))
    except Exception:
        return ""
    if s <= 0:
        return "0s"
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h}h {m}m" if m else f"{h}h"
    if m > 0:
        return f"{m}m {sec}s" if sec else f"{m}m"
    return f"{sec}s"


def parse_build_and_test_dynamo_phase_timings(*, raw_log_path: Path) -> List[Tuple[str, str]]:
    """Best-effort parse of 'Build and Test - dynamo' raw log into phase timing rows.

    Returns a list of (phase_name, duration_str), in execution order.
    """
    try:
        p = Path(raw_log_path)
        if not p.exists() or not p.is_file():
            return []
        txt = p.read_text(errors="replace")
    except Exception:
        return []

    lines = (txt or "").splitlines()
    # Collect timestamped lines for quick scanning.
    ts_lines: List[Tuple[datetime, str]] = []
    for ln in lines:
        ts = _parse_utc_ts_prefix(ln)
        if ts is None:
            continue
        ts_lines.append((ts, ln))
    if not ts_lines:
        return []

    # Helper to find first/last timestamp for patterns within a window.
    def _first_ts(pred) -> Optional[datetime]:
        for ts, ln in ts_lines:
            if pred(ln):
                return ts
        return None

    def _last_ts(pred, *, after: Optional[datetime] = None, before: Optional[datetime] = None) -> Optional[datetime]:
        out: Optional[datetime] = None
        for ts, ln in ts_lines:
            if after is not None and ts < after:
                continue
            if before is not None and ts > before:
                continue
            if pred(ln):
                out = ts
        return out

    # Phase 1: Build image
    #
    # The log format varies:
    # - old: "+ docker build ... --tag dynamo:latest"
    # - current: "Run ./container/build.sh --tag dynamo:latest ..." + buildkit "#N [..]" lines
    # We key off the earliest stable "build started" marker we can find.
    build_start = _first_ts(
        lambda ln: (
            ("Building Dynamo Image:" in ln and "dynamo:latest" in ln)
            or ("##[group]Run ./container/build.sh" in ln and "--tag" in ln and "dynamo:latest" in ln)
            or ("./container/build.sh" in ln and "--tag" in ln and "dynamo:latest" in ln)
            or ("+ docker build" in ln and "--tag dynamo:latest" in ln)
            or ("/usr/bin/docker buildx" in ln and " build " in ln and "dynamo:latest" in ln)
        )
    )
    build_end = None
    if build_start is not None:
        # Prefer buildkit export markers near the end of the build.
        build_end = _last_ts(
            lambda ln: (
                ("exporting config sha256:" in ln)
                or ("exporting manifest sha256:" in ln)
                or ("exporting layers" in ln and " done" in ln)
                or ("exporting to docker image format" in ln and " done" in ln)
                or ("writing image sha" in ln)
                or (" naming to " in ln and "dynamo:latest" in ln)
            ),
            after=build_start,
        )

    # Phase 2: Rust checks (docker run ... rust_checks -> just before pytest-parallel starts)
    rust_start = _first_ts(lambda ln: ("_rust_checks" in ln) or ("cargo fmt -- --check" in ln))

    # Phase 3a: Pytest "serial" (best-effort). Not always present in logs.
    pytest_serial_start = _first_ts(
        lambda ln: (
            ("bash -c \"pytest" in ln and "pytest_test_report.xml" in ln)
            or ("--junitxml=pytest_test_report.xml" in ln)
            or ("--name" in ln and "build-test_dynamo_pytest" in ln and "pytest_parallel" not in ln)
        )
    )

    pytest_parallel_start = _first_ts(lambda ln: ("pytest-parallel" in ln) or ("_pytest_parallel" in ln) or ("PYTEST_MARKS:" in ln and "parallel" in ln))
    # Prefer the explicit pytest command line if present.
    if pytest_parallel_start is None:
        pytest_parallel_start = _first_ts(lambda ln: ("pytest --basetemp=/tmp/pytest-parallel" in ln) or ("pre_merge and parallel" in ln and "pytest" in ln))

    rust_end = None
    if rust_start is not None:
        # End at the last timestamp before pytest parallel starts (or end-of-file).
        before = pytest_parallel_start
        rust_end = _last_ts(lambda _ln: True, after=rust_start, before=before) if before else _last_ts(lambda _ln: True, after=rust_start)

    # Phase 3b: Pytest parallel (pytest cmd -> generated xml / docker cp)
    pytest_par_end = None
    if pytest_parallel_start is not None:
        pytest_par_end = _last_ts(lambda ln: ("generated xml file:" in ln and "pytest_parallel.xml" in ln), after=pytest_parallel_start)
        if pytest_par_end is None:
            pytest_par_end = _first_ts(lambda ln: ("docker cp" in ln and "pytest_parallel.xml" in ln))
        if pytest_par_end is None:
            pytest_par_end = _last_ts(lambda _ln: True, after=pytest_parallel_start)

    # End marker for "serial" pytest: docker cp / path line for pytest_test_report.xml.
    pytest_serial_end = None
    if pytest_serial_start is not None:
        pytest_serial_end = _last_ts(lambda ln: ("docker cp" in ln and "pytest_test_report.xml" in ln), after=pytest_serial_start)
        if pytest_serial_end is None:
            pytest_serial_end = _last_ts(lambda ln: (" path: " in ln and "pytest_test_report.xml" in ln), after=pytest_serial_start)
        if pytest_serial_end is None:
            pytest_serial_end = _last_ts(lambda _ln: True, after=pytest_serial_start, before=pytest_parallel_start)

    out: List[Tuple[str, str]] = []
    try:
        if build_start and build_end and build_end > build_start:
            out.append(("Build Image", _format_duration_short((build_end - build_start).total_seconds())))
    except Exception:
        pass
    try:
        if rust_start and rust_end and rust_end > rust_start:
            out.append(("Rust checks", _format_duration_short((rust_end - rust_start).total_seconds())))
    except Exception:
        pass
    try:
        if pytest_serial_start and pytest_serial_end and pytest_serial_end > pytest_serial_start:
            out.append(("pytest (serial)", _format_duration_short((pytest_serial_end - pytest_serial_start).total_seconds())))
    except Exception:
        pass
    try:
        if pytest_parallel_start and pytest_par_end and pytest_par_end > pytest_parallel_start:
            out.append(("pytest (parallel)", _format_duration_short((pytest_par_end - pytest_parallel_start).total_seconds())))
    except Exception:
        pass

    return out


def _parse_iso_utc(s: str) -> Optional[datetime]:
    try:
        x = str(s or "").strip()
        if not x:
            return None
        if x.endswith("Z"):
            return datetime.fromisoformat(x[:-1] + "+00:00")
        return datetime.fromisoformat(x)
    except Exception:
        return None


def _status_norm_from_actions_step(status: str, conclusion: str) -> str:
    s = (status or "").strip().lower()
    c = (conclusion or "").strip().lower()
    if c in ("success", "neutral", "skipped"):
        return "success"
    if c in ("failure", "timed_out", "action_required"):
        return "failure"
    if c in ("cancelled", "canceled"):
        return "cancelled"
    # Some API responses omit `conclusion` even for completed successful steps.
    # Treat "completed + empty conclusion" as success so we don't auto-expand clean trees.
    if s == "completed" and c in ("", "null", "none"):
        return "success"
    if s in ("in_progress", "in progress"):
        return "in_progress"
    if s in ("queued", "pending"):
        return "pending"
    return "unknown"


def build_and_test_dynamo_phases_from_actions_job(job: Dict[str, object]) -> List[Tuple[str, str, str]]:
    """Extract phase rows from the Actions job `steps` array.

    Returns (phase_name, duration_str, status_norm).
    """
    try:
        steps = job.get("steps") if isinstance(job, dict) else None
        if not isinstance(steps, list) or not steps:
            return []
    except Exception:
        return []

    def _dur(st: Dict[str, object]) -> str:
        a = _parse_iso_utc(str(st.get("started_at", "") or ""))
        b = _parse_iso_utc(str(st.get("completed_at", "") or ""))
        if not a or not b:
            return ""
        try:
            return _format_duration_short((b - a).total_seconds())
        except Exception:
            return ""

    out: List[Tuple[str, str, str]] = []

    # We match the canonical step names used in the workflow, but keep it fuzzy.
    for st in steps:
        if not isinstance(st, dict):
            continue
        nm = str(st.get("name", "") or "")
        nm_lc = nm.lower()
        status_norm = _status_norm_from_actions_step(
            status=str(st.get("status", "") or ""),
            conclusion=str(st.get("conclusion", "") or ""),
        )
        dur = _dur(st)

        if "build image" in nm_lc:
            out.append(("Build Image", dur, status_norm))
        elif "rust" in nm_lc and "check" in nm_lc:
            out.append(("Rust checks", dur, status_norm))
        elif "pytest" in nm_lc and ("parallel" in nm_lc or "xdist" in nm_lc):
            out.append(("pytest (parallel)", dur, status_norm))
        elif nm_lc.startswith("run pytest") or (("pytest" in nm_lc) and ("parallel" not in nm_lc) and ("xdist" not in nm_lc)):
            # If the workflow has an explicit non-parallel pytest step, call it serial.
            out.append(("pytest (serial)", dur, status_norm))

    # De-dup while keeping order (some jobs echo repeated step names via composites).
    seen = set()
    uniq: List[Tuple[str, str, str]] = []
    for ph in out:
        k = ph[0]
        if k in seen:
            continue
        seen.add(k)
        uniq.append(ph)
    return uniq


def actions_job_steps_over_threshold_from_actions_job(
    job: Dict[str, object], *, min_seconds: float = 30.0
) -> List[Tuple[str, str, str]]:
    """Return (step_name, duration_str, status_norm) for steps we want to display.

    Policy:
    - show steps with duration >= min_seconds
    - always show failing steps (even if < min_seconds or duration is missing)
    """
    try:
        steps = job.get("steps") if isinstance(job, dict) else None
        if not isinstance(steps, list) or not steps:
            return []
    except Exception:
        return []

    out: List[Tuple[str, str, str]] = []
    for st in steps:
        if not isinstance(st, dict):
            continue
        nm = str(st.get("name", "") or "").strip()
        if not nm:
            continue
        status_norm = _status_norm_from_actions_step(
            status=str(st.get("status", "") or ""),
            conclusion=str(st.get("conclusion", "") or ""),
        )
        # Duration is best-effort; some steps may not include timestamps.
        dt_s = None
        try:
            a = _parse_iso_utc(str(st.get("started_at", "") or ""))
            b = _parse_iso_utc(str(st.get("completed_at", "") or ""))
            if a and b:
                dt_s = float((b - a).total_seconds())
        except Exception:
            dt_s = None

        # Selection rule:
        # - always include failures
        # - if min_seconds <= 0: include all non-failing steps (even if duration is missing)
        # - otherwise include only steps >= threshold
        if status_norm != "failure":
            try:
                if float(min_seconds) <= 0.0:
                    pass
                elif dt_s is None or float(dt_s) < float(min_seconds):
                    continue
            except Exception:
                continue

        dur_s = _format_duration_short(float(dt_s)) if dt_s is not None else ""
        out.append((nm, dur_s, status_norm))

    # De-dup while keeping order (some composite actions can repeat step names).
    seen = set()
    uniq: List[Tuple[str, str, str]] = []
    for (nm, dur, st) in out:
        if nm in seen:
            continue
        seen.add(nm)
        uniq.append((nm, dur, st))
    return uniq


def actions_job_step_tuples(
    *,
    github_api: Optional["GitHubAPIClient"],
    job_url: str,
    min_seconds: float = 30.0,
    ttl_s: int = 7 * 24 * 3600,
) -> List[Tuple[str, str, str]]:
    """Fetch job details (cached) and return long-running steps (duration >= min_seconds)."""
    if not github_api:
        return []
    jid = extract_actions_job_id_from_url(str(job_url or ""))
    if not jid:
        return []
    try:
        job = github_api.get_actions_job_details_cached(
            owner="ai-dynamo", repo="dynamo", job_id=jid, ttl_s=int(ttl_s)
        ) or {}
    except Exception:
        job = {}
    if not isinstance(job, dict):
        return []
    return actions_job_steps_over_threshold_from_actions_job(job, min_seconds=float(min_seconds))


def ci_subsection_tuples_for_job(
    *,
    github_api: Optional["GitHubAPIClient"],
    job_name: str,
    job_url: str,
    raw_log_path: Optional[Path],
    duration_seconds: float,
    is_required: bool,
    long_job_threshold_s: float = 10.0 * 60.0,
    step_min_s: float = 30.0,
) -> List[Tuple[str, str, str]]:
    """Shared rule: return child tuples for phase/step subsections.

    - Build and Test - dynamo: return the dedicated phases (status+duration) (steps API first; raw log fallback).
    - Other long-running Actions jobs: return job steps >= step_min_s.
    """
    nm = str(job_name or "").strip()
    if not nm:
        return []
    if nm == "Build and Test - dynamo":
        try:
            phases = build_and_test_dynamo_phase_tuples(
                github_api=github_api,
                job_url=str(job_url or ""),
                raw_log_path=raw_log_path,
                is_required=bool(is_required),
            )
            # Also include steps (like other Actions jobs).
            # Policy for REQUIRED jobs: show all failing steps + steps >= 30s; ignore the rest.
            steps = actions_job_step_tuples(github_api=github_api, job_url=str(job_url or ""), min_seconds=float(step_min_s))
            out = [(p[0], p[1], p[2]) for p in (phases or [])]
            out.extend([(s[0], s[1], s[2]) for s in (steps or [])])
            return out
        except Exception:
            return []

    # REQUIRED jobs: always show failing steps + steps >= threshold (even if job isn't "long-running").
    if bool(is_required):
        return actions_job_step_tuples(github_api=github_api, job_url=str(job_url or ""), min_seconds=float(step_min_s))

    # Non-required jobs: show steps only for long-running jobs (avoid noise).
    try:
        if float(duration_seconds or 0.0) < float(long_job_threshold_s):
            return []
    except Exception:
        return []

    return actions_job_step_tuples(github_api=github_api, job_url=str(job_url or ""), min_seconds=float(step_min_s))


def step_window_snippet_from_cached_raw_log(
    *,
    job: Dict[str, object],
    step_name: str,
    raw_log_path: Optional[Path],
) -> str:
    """Extract an error snippet scoped to a specific Actions step time window (best-effort).

    We do not have per-step log URLs. Instead, we:
    - locate the step's started_at/completed_at from the cached job `steps[]`
    - slice the cached raw log by timestamp
    - run the common snippet extractor on the sliced text
    """
    if not raw_log_path:
        return ""
    p = Path(raw_log_path)
    if not p.exists() or not p.is_file():
        return ""
    step = None
    try:
        steps = job.get("steps") if isinstance(job, dict) else None
        if isinstance(steps, list):
            for st in steps:
                if isinstance(st, dict) and str(st.get("name", "") or "") == str(step_name or ""):
                    step = st
                    break
    except Exception:
        step = None
    if not isinstance(step, dict):
        return ""

    a = _parse_iso_utc(str(step.get("started_at", "") or ""))
    b = _parse_iso_utc(str(step.get("completed_at", "") or ""))
    if not a or not b:
        return ""

    try:
        from common_log_errors import extract_error_snippet_from_text  # local import (avoid circulars)
    except Exception:
        return ""

    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""

    # Filter lines by timestamp window (raw logs include ISO-8601 timestamps).
    kept: List[str] = []
    for ln in (text.splitlines() or []):
        # Most lines are prefixed with an ISO timestamp; ignore unparsable lines.
        ts = None
        try:
            # Heuristic: take the first token, strip any trailing 'Z'.
            head = (ln.split(" ", 1)[0] if " " in ln else ln).strip()
            ts = _parse_iso_utc(head)
        except Exception:
            ts = None
        if not ts:
            continue
        if ts < a or ts > b:
            continue
        kept.append(ln)
    if not kept:
        return ""
    return extract_error_snippet_from_text("\n".join(kept))

def required_badge_html(*, is_required: bool, status_norm: str) -> str:
    """Render a [REQUIRED] badge with shared semantics."""
    if not is_required:
        return ""

    s = (status_norm or "").strip().lower()
    if s == "failure":
        color = "#d73a49"
        weight = "700"
    elif s == "success":
        color = "#2da44e"
        weight = "400"
    else:
        color = "#57606a"
        weight = "400"

    return f' <span style="color: {color}; font-weight: {weight};">[REQUIRED]</span>'


def mandatory_badge_html(*, is_mandatory: bool, status_norm: str) -> str:
    """Render a [MANDATORY] badge (GitLab) following the same color convention as [REQUIRED]."""
    if not is_mandatory:
        return ""

    s = (status_norm or "").strip().lower()
    if s == "failure":
        color = "#d73a49"
        weight = "700"
    elif s == "success":
        color = "#2da44e"
        weight = "400"
    else:
        color = "#57606a"
        weight = "400"

    return f' <span style="color: {color}; font-weight: {weight};">[MANDATORY]</span>'


def status_icon_html(
    *,
    status_norm: str,
    is_required: bool,
    required_failure: bool = False,
    warning_present: bool = False,
) -> str:
    """Shared status icon HTML (match show_dynamo_branches)."""
    s = (status_norm or "").strip().lower()

    if s == "success":
        # UX: required successes use the green circle-check (like GitHub's required checks),
        # optional successes use a simpler green check (no circle).
        if bool(is_required):
            out = (
                '<span style="color: #2da44e; display: inline-flex; vertical-align: text-bottom;">'
                '<svg aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="12" height="12" '
                'data-view-component="true" class="octicon octicon-check-circle-fill" fill="currentColor">'
                '<path fill-rule="evenodd" '
                'd="M8 16A8 8 0 108 0a8 8 0 000 16zm3.78-9.78a.75.75 0 00-1.06-1.06L7 9.94 5.28 8.22a.75.75 0 10-1.06 1.06l2 2a.75.75 0 001.06 0l4-4z">'
                "</path></svg></span>"
            )
        else:
            out = '<span style="color: #2da44e; font-weight: 900;">✓</span>'
        if bool(warning_present):
            out += '<span style="color: #f59e0b; font-size: 13px; font-weight: 900; line-height: 1; margin-left: 2px;">⚠</span>'
        return out
    if s in {"skipped", "neutral"}:
        # GitHub-like "skipped": grey circle with a slash.
        return (
            '<span style="color: #8c959f; display: inline-flex; vertical-align: text-bottom;">'
            '<svg aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="12" height="12" '
            'data-view-component="true" class="octicon octicon-circle-slash" fill="currentColor">'
            '<path fill-rule="evenodd" '
            'd="M8 16A8 8 0 108 0a8 8 0 000 16ZM1.5 8a6.5 6.5 0 0110.364-5.083l-8.947 8.947A6.473 6.473 0 011.5 8Zm3.136 5.083 8.947-8.947A6.5 6.5 0 014.636 13.083Z">'
            "</path></svg></span>"
        )
    if s == "failure":
        if is_required or required_failure:
            return (
                '<span style="display: inline-flex; align-items: center; justify-content: center; '
                'width: 12px; height: 12px; border-radius: 999px; background-color: #d73a49; '
                'color: #ffffff; font-size: 10px; font-weight: 900; line-height: 1;">✗</span>'
            )
        return '<span style="color: #f59e0b; font-weight: 900;">⚠</span>'
    if s == "in_progress":
        return '<span style="color: #dbab09;">⏳</span>'
    if s == "pending":
        return (
            '<span style="display: inline-flex; align-items: center; justify-content: center; '
            'width: 12px; height: 12px; border-radius: 999px; background-color: #8c959f; '
            'color: #ffffff; font-size: 10px; font-weight: 900; line-height: 1;">•</span>'
        )
    if s == "cancelled":
        return (
            '<span style="display: inline-flex; align-items: center; justify-content: center; '
            'width: 12px; height: 12px; border-radius: 999px; background-color: #8c959f; '
            'color: #ffffff; font-size: 10px; font-weight: 900; line-height: 1;">×</span>'
        )
    return '<span style="color: #8c959f;">•</span>'


def _small_link_html(*, url: str, label: str) -> str:
    if not url:
        return ""
    return (
        f' <a href="{html.escape(url, quote=True)}" target="_blank" '
        f'style="color: #0969da; font-size: 11px; margin-left: 5px; text-decoration: none;">{html.escape(label)}</a>'
    )


from common_log_errors import render_error_snippet_html as _format_snippet_html  # shared implementation
from common_log_errors import categorize_error_snippet_text as _snippet_categories


def _tag_pill_html(*, text: str, monospace: bool = False, kind: str = "category") -> str:
    """Render a small tag/pill for inline display next to links.

    kind:
      - "category": orange-ish pill (high-level error category)
      - "command": gray pill (detected command; not the root cause)
    """
    t = (text or "").strip()
    if not t:
        return ""
    font = "SFMono-Regular, Consolas, Liberation Mono, Menlo, monospace" if monospace else "inherit"
    if kind == "command":
        border = "#d0d7de"
        bg = "#f6f8fa"
        fg = "#57606a"
    else:
        border = "#d4a72c"
        bg = "#fff8c5"
        fg = "#7d4e00"
    return (
        f' <span style="display: inline-block; vertical-align: baseline; '
        f'border: 1px solid {border}; background: {bg}; color: {fg}; '
        f'border-radius: 999px; padding: 1px 6px; font-size: 10px; '
        f'font-weight: 600; line-height: 1; font-family: {font};">'
        f"{html.escape(t)}</span>"
    )


def _snippet_first_command(snippet_text: str) -> str:
    """Extract the first command (1-line) from the snippet header, if present."""
    text = (snippet_text or "").strip()
    if not text:
        return ""
    lines = text.splitlines()
    in_cmd = False
    collected: list[str] = []
    for ln in lines:
        s = ln.rstrip("\n")
        if not in_cmd:
            if s.strip().lower().startswith("commands:"):
                in_cmd = True
            continue
        # Stop at the first blank line (headers are separated from snippet body by a blank line).
        if not s.strip():
            break
        collected.append(s)
        # We only need the first command. If the next line looks like a continuation, we'll
        # still compress it into a single line with " …".
        if len(collected) >= 1:
            break
    if not collected:
        return ""
    first = collected[0].lstrip()
    # Commands are formatted like "  <cmd>".
    if first.startswith("- "):
        first = first[2:].lstrip()
    # Normalize whitespace for inline display.
    first = re.sub(r"\s+", " ", first).strip()
    if len(first) > 90:
        first = first[:90].rstrip() + "…"
    return first


def _error_snippet_toggle_html(*, dom_id_seed: str, snippet_text: str) -> str:
    global _ERROR_SNIP_SEQ
    try:
        _ERROR_SNIP_SEQ += 1
    except NameError:
        _ERROR_SNIP_SEQ = 1
    err_id = f"err_snip_{_ERROR_SNIP_SEQ:x}"
    shown = _format_snippet_html(snippet_text or "")
    if not shown:
        shown = '<span style="color: #57606a;">(no snippet found)</span>'
    return (
        f' <span style="cursor: pointer; color: #0969da; font-size: 11px; margin-left: 5px; '
        f'text-decoration: none; font-weight: 500; user-select: none;" '
        f'onclick="toggleErrorSnippet(\'{html.escape(err_id, quote=True)}\', this)">▶ Snippet</span>'
        f'<span id="{html.escape(err_id, quote=True)}" style="display: none;">'
        f"<br>"
        # Full-width snippet box (within the <pre> container) so it expands to available screen width.
        f'<span style="display: block; width: 100%; max-width: 100%; box-sizing: border-box; '
        f'border: 1px solid #d0d7de; background: #f6f8fa; '
        f'border-radius: 6px; padding: 6px 8px; white-space: pre-wrap; overflow-wrap: anywhere; color: #24292f;">'
        f'<span style="color: #d73a49; font-weight: 700;">Snippet:</span> {shown}'
        f"</span>"
        f"</span>"
    )


def _format_bytes_short(n_bytes: int) -> str:
    try:
        n = int(n_bytes or 0)
    except Exception:
        return ""
    if n <= 0:
        return ""
    gb = 1024**3
    mb = 1024**2
    kb = 1024
    if n >= gb:
        v = n / gb
        s = f"{v:.0f}GB" if v >= 10 else f"{v:.1f}GB"
        return s.replace(".0GB", "GB")
    if n >= mb:
        v = n / mb
        s = f"{v:.0f}MB" if v >= 10 else f"{v:.1f}MB"
        return s.replace(".0MB", "MB")
    if n >= kb:
        v = n / kb
        s = f"{v:.0f}KB" if v >= 10 else f"{v:.1f}KB"
        return s.replace(".0KB", "KB")
    return f"{n}B"


def check_line_html(
    *,
    job_id: str,
    display_name: str = "",
    status_norm: str,
    is_required: bool,
    duration: str = "",
    log_url: str = "",
    raw_log_href: str = "",
    raw_log_size_bytes: int = 0,
    error_snippet_text: str = "",
    required_failure: bool = False,
    warning_present: bool = False,
) -> str:
    icon = status_icon_html(
        status_norm=status_norm,
        is_required=is_required,
        required_failure=required_failure,
        warning_present=warning_present,
    )
    def _format_arch_text(raw_text: str) -> str:
        """Format the job text with arch styling.

        - If the job contains an explicit `(arm64)` token, grey out the *entire* string.
        - Otherwise, keep normal styling (no special casing).
        """
        raw = str(raw_text or "")
        esc = html.escape(raw)
        if "arm64" in raw and "(" in raw and ")" in raw:
            # Grey out the whole label (requested).
            return f'<span style="color: #8c959f;">{esc}</span>'
        return esc

    id_html = (
        '<span style="font-family: SFMono-Regular, Consolas, Liberation Mono, Menlo, monospace; font-size: 12px;">'
        + _format_arch_text(job_id or "")
        + "</span>"
    )
    req_html = required_badge_html(is_required=is_required, status_norm=status_norm)

    name_html = ""
    if display_name and display_name != job_id:
        name_html = f'<span style="color: #57606a; font-size: 12px;"> — {_format_arch_text(display_name)}</span>'

    dur_html = f'<span style="color: #57606a; font-size: 12px;"> ({html.escape(duration)})</span>' if duration else ""

    links = ""
    if log_url:
        links += _small_link_html(url=log_url, label="[log]")
    # Raw-log link (only when present).
    if raw_log_href:
        size_s = _format_bytes_short(raw_log_size_bytes)
        label = f"[cached raw log {size_s}]" if size_s else "[cached raw log]"
        links += _small_link_html(url=raw_log_href, label=label)

    # Error tags/snippet can be shown either:
    # - after [cached raw log] when present (preferred), OR
    # - on its own (e.g. for step subsections) when we have a snippet but no dedicated raw-log link.
    if (error_snippet_text or "").strip():
        cats = _snippet_categories(error_snippet_text or "")
        for c in cats[:3]:
            links += _tag_pill_html(text=c, monospace=False, kind="category")
        cmd = _snippet_first_command(error_snippet_text or "")
        if cmd:
            links += _tag_pill_html(text=cmd, monospace=True, kind="command")
        links += _error_snippet_toggle_html(dom_id_seed=f"{job_id}|{raw_log_href}", snippet_text=error_snippet_text)

    return f"{icon} {id_html}{req_html}{name_html}{dur_html}{links}"


def render_gl_job_line_html(*, status_norm: str, name: str, url: str = "", duration: str = "") -> str:
    icon = status_icon_html(status_norm=status_norm, is_required=False)
    name_html = (
        '<span style="font-family: SFMono-Regular, Consolas, Liberation Mono, Menlo, monospace; font-size: 12px;">'
        + html.escape(name or "")
        + "</span>"
    )
    dur_html = f'<span style="color: #57606a; font-size: 12px;"> ({html.escape(duration)})</span>' if duration else ""
    links = _small_link_html(url=url, label="[log]") if url else ""
    return f"{icon} {name_html}{dur_html}{links}"


def build_and_test_dynamo_phase_tuples(
    *,
    github_api: Optional["GitHubAPIClient"],
    job_url: str,
    raw_log_path: Optional[Path] = None,
    is_required: bool = False,
) -> List[Tuple[str, str, str]]:
    """Return phase tuples for the Build-and-Test phase breakdown (best-effort).

    Shared helper used by both dashboards to keep logic identical.
    """
    phases3: List[Tuple[str, str, str]] = []
    try:
        jid = extract_actions_job_id_from_url(str(job_url or ""))
        if github_api and jid:
            job = github_api.get_actions_job_details_cached(owner="ai-dynamo", repo="dynamo", job_id=jid, ttl_s=600) or {}
            if isinstance(job, dict):
                phases3 = build_and_test_dynamo_phases_from_actions_job(job) or []
    except Exception:
        phases3 = []

    if not phases3 and raw_log_path is not None:
        try:
            phases2 = parse_build_and_test_dynamo_phase_timings(raw_log_path=raw_log_path)
            phases3 = [(n, d, "unknown") for (n, d) in phases2]
        except Exception:
            phases3 = []

    return [(str(n), str(d), str(s)) for (n, d, s) in (phases3 or [])]


