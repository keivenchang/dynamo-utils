#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
revalidate_pr.py — probe CI health by pushing per-SHA placebo PRs.

For every commit merged to main since --starting-sha, push a tiny placebo
diff on a per-SHA branch and open a Draft PR. Watch CI; on failure, retry
up to N times via gh run rerun --failed. On all-pass, close PR + delete
branch. Persist verdicts and per-attempt failures to a JSON cache.

Subcommands:
  run        one cycle (cron-friendly, idempotent)
  status     print DB state per SHA
  report     verdict + flake leaderboard
  reset SHA  drop a SHA's entry to re-probe from scratch

Run flags:
  --starting-sha SHA      first SHA to probe (required first run; persisted)
  --parallelism N         max in-flight probes (default 4)
  --max-attempts N        retries per SHA (default 3)
  --stalled-after-hours H stall threshold (default 3)
  --dry-run / --dryrun    no writes; reads OK, prints what would happen

Cache: ~/.cache/dynamo-utils/ci-health.json
Clone: /tmp/ci_health/repo
Lock:  /tmp/ci_health/launch.pid
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")

REPO = "ai-dynamo/dynamo"
RAW_LOG_DIR = Path(
    os.environ.get("DYNAMO_UTILS_CACHE_DIR")
    or str(Path.home() / ".cache" / "dynamo-utils")
) / "raw-log-text"
# Prefer the dashboard's commit checkout (kept up-to-date by update_html_pages.sh)
# over revalidate's stale /tmp clone.
_DEFAULT_CLONES = [
    Path.home() / "dev" / "commits",
    Path("/tmp/ci_health/repo"),
]
CLONE_PATH = next((p for p in _DEFAULT_CLONES if (p / ".git").exists()), _DEFAULT_CLONES[0])

# Match either:
#   FAILED tests/foo/test_bar.py::TestClass::test_method[params] - reason   (assertion fail)
#   ERROR tests/foo/test_bar.py::test_method - RuntimeError: ...            (fixture/setup error)
# with optional GH Actions timestamp prefix and ANSI color codes.
_PYTEST_FAILED_RE = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2}T[\d:.]+Z\s+)?(?:\x1b\[[0-9;]*m)?(?:FAILED|ERROR)\s+(\S+::\S+)"
)

# Required status checks on `main` (from repo rulesets, integration_id 15368).
# Everything else (runtime tests, docker builds, …) is optional but rolls up
# into the status-check aggregators. Refresh via:
#   gh api repos/ai-dynamo/dynamo/rules/branches/main
# Status icons matching show_commit_history Legend & Key
# (see html_pages/ci_status_icons.py — kept inline here so this script
# stays a single file).
_ICON_GREEN = "#2da44e"
_ICON_RED = "#c83a3a"
_ICON_GREY = "#8c959f"

ICON_REQ_PASS = (
    f'<span class="legend-icon" style="color:{_ICON_GREEN};" title="required passed">'
    '<svg aria-hidden="true" viewBox="0 0 16 16" width="14" height="14" '
    'class="octicon octicon-check-circle-fill" fill="currentColor">'
    '<path fill-rule="evenodd" d="M8 16A8 8 0 108 0a8 8 0 000 16zm3.78-9.78a.75.75 0 00-1.06-1.06L7 9.94 5.28 8.22a.75.75 0 10-1.06 1.06l2 2a.75.75 0 001.06 0l4-4z"/>'
    "</svg></span>"
)
ICON_OPT_PASS = (
    f'<span class="legend-icon" style="color:{_ICON_GREEN};" title="optional passed">'
    '<svg aria-hidden="true" viewBox="0 0 16 16" width="14" height="14" '
    'class="octicon octicon-check" fill="currentColor">'
    '<path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 00-1.06 0L6.75 10.19 3.28 6.72a.75.75 0 10-1.06 1.06l4 4a.75.75 0 001.06 0l7.5-7.5a.75.75 0 000-1.06z"/>'
    "</svg></span>"
)
ICON_REQ_FAIL = (
    f'<span class="legend-icon" style="color:{_ICON_RED};" title="required failed">'
    '<svg aria-hidden="true" viewBox="0 0 16 16" width="14" height="14" '
    'class="octicon octicon-x-circle-fill" fill="currentColor">'
    '<circle cx="8" cy="8" r="8" fill="currentColor"/>'
    '<path d="M4.5 4.5l7 7m-7 0l7-7" stroke="#fff" stroke-width="2" stroke-linecap="round"/>'
    "</svg></span>"
)
ICON_OPT_FAIL = (
    f'<span class="legend-icon" style="color:{_ICON_RED};" title="optional failed">'
    '<svg aria-hidden="true" viewBox="0 0 16 16" width="14" height="14" '
    'class="octicon octicon-x" fill="currentColor">'
    '<path fill-rule="evenodd" d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 11-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z"/>'
    "</svg></span>"
)
ICON_RUN = (
    f'<span class="legend-icon" style="color:{_ICON_GREY};" title="in progress">'
    '<svg aria-hidden="true" viewBox="0 0 16 16" width="14" height="14" '
    'class="octicon octicon-clock" fill="currentColor">'
    '<path d="M8 1C4.1 1 1 4.1 1 8s3.1 7 7 7 7-3.1 7-7-3.1-7-7-7zm0 12c-2.8 0-5-2.2-5-5s2.2-5 5-5 5 2.2 5 5-2.2 5-5 5z"/>'
    '<path d="M8 4v5l3 2"/></svg></span>'
)

REQUIRED_CHECKS = frozenset({
    "copyright-checks",
    "DCO",
    "backend-status-check",
    "dynamo-status-check",
    "pre-merge-status-check",
    "deploy-status-check",
})

# DEFAULT_MAX_ATTEMPTS: legacy fallback used in the per-SHA "Runs: N/M" header.
# Plumbed through entry["max_attempts"] when present; otherwise defaults to 1.
DEFAULT_MAX_ATTEMPTS = 1

logger = logging.getLogger("ci_lib")


# ---------- helpers ----------


def short_sha(sha: str) -> str:
    return sha[:11]


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    logger.debug("$ %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
        env=env,
    )


def gh_api(
    path: str,
    *,
    method: str = "GET",
    fields: list[tuple[str, str]] | None = None,
) -> Any:
    cmd = ["gh", "api", path]
    if method != "GET":
        cmd += ["--method", method]
    if fields:
        for k, v in fields:
            cmd += ["-f", f"{k}={v}"]
    proc = run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"gh api {path} failed: {proc.stderr.strip()}")
    if not proc.stdout.strip():
        return None
    return json.loads(proc.stdout)


# ---------- DB I/O ----------


# ---------- git ----------


def commit_subject_pr_author(sha: str) -> tuple[str, int | None, str]:
    """Return (subject_without_pr_suffix, pr_number, author) for `sha`."""
    proc = run(
        ["git", "show", "-s", "--format=%s%x1f%an", sha],
        cwd=CLONE_PATH,
        check=False,
    )
    if proc.returncode != 0:
        return ("", None, "")
    raw = proc.stdout.strip()
    if "\x1f" not in raw:
        return (raw, None, "")
    subject_full, author = raw.split("\x1f", 1)
    pr: int | None = None
    subject = subject_full
    m = re.search(r"\s*\(#(\d+)\)\s*$", subject_full)
    if m:
        pr = int(m.group(1))
        subject = subject_full[: m.start()].rstrip()
    return (subject, pr, author)


# ---------- gh ----------


# ---------- log fetching + pytest failure extraction ----------


def fetch_job_log(job_id: int) -> Path | None:
    """Fetch a job's log text into ~/.cache/dynamo-utils/raw-log-text/<job_id>.log.

    Returns the cached path on success, None on failure. Cache-hits skip the fetch.
    Reuses the existing dynamo-utils raw-log-text convention so logs cached by
    other tools (commit dashboards, ci_log_errors) are reused.
    """
    RAW_LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_LOG_DIR / f"{job_id}.log"
    if out.exists() and out.stat().st_size > 0:
        return out
    # `gh api .../logs` follows the Azure-blob redirect and emits plain text.
    proc = subprocess.run(
        ["gh", "api", f"repos/{REPO}/actions/jobs/{job_id}/logs"],
        capture_output=True,
        text=False,
        check=False,
    )
    if proc.returncode != 0:
        logger.warning(
            "fetch log job_id=%s failed: %s",
            job_id,
            proc.stderr.decode("utf-8", errors="replace")[:200],
        )
        return None
    if not proc.stdout:
        return None
    out.write_bytes(proc.stdout)
    return out


def extract_pytest_failures(log_path: Path) -> list[str]:
    """Return sorted unique pytest test IDs from `FAILED tests/...::test` lines."""
    found: set[str] = set()
    try:
        with log_path.open("r", errors="replace") as fh:
            for line in fh:
                m = _PYTEST_FAILED_RE.match(line)
                if m:
                    found.add(m.group(1))
    except OSError as e:
        logger.warning("read %s: %s", log_path, e)
    return sorted(found)


# Lazy-imported once per process. Reuses the production ci_log_errors engine.
_CI_LOG_ERRORS_LOADED = False


def _ensure_ci_log_errors():
    global _CI_LOG_ERRORS_LOADED
    if _CI_LOG_ERRORS_LOADED:
        return True
    utils_root = str(Path.home() / "utils")
    if utils_root not in sys.path:
        sys.path.insert(0, utils_root)
    try:
        global _categorize_error_log_lines, _extract_error_snippet_from_log_file, _html_highlight_error_keywords
        from ci_log_errors.engine import categorize_error_log_lines as _categorize_error_log_lines  # type: ignore
        from ci_log_errors.snippet import extract_error_snippet_from_log_file as _extract_error_snippet_from_log_file  # type: ignore
        from ci_log_errors.render import html_highlight_error_keywords as _html_highlight_error_keywords  # type: ignore
        _CI_LOG_ERRORS_LOADED = True
        return True
    except Exception as e:
        logger.warning("ci_log_errors unavailable: %s", e)
        return False


# Strip GH Actions ISO timestamp prefix like "2026-04-29T17:15:55.4973999Z "
_TS_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s+")
# Strip ANSI escape sequences (color codes)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# Detect any line that contains an echo'd string — `echo "..."` or `echo '...'` —
# whether at line start (xtrace `+ echo "X"`) or inside a longer shell command
# (Docker `RUN ... echo "ERROR: ..."`). We skip error-keyword highlighting on
# these lines because the keyword is *literal text being echoed*, not a real error.
_ECHO_LINE_RE = re.compile(r"""\becho\s+["']""")


def render_snippet_html(snippet_text: str) -> str:
    """Render a snippet with red error-keyword highlighting only.

    Deliberately does NOT use the production renderer's command-block / blue-line
    detection — those highlight things like `git version`, `PYTEST_CMD=...`,
    `docker run ...` which add noise without signal.

    Strips: ANSI escapes, GH Actions timestamps, [[CMD]]...[[/CMD]] cut-paste
    blocks, and lines that are just shell command-noise prelude (Run, group
    markers, env: blocks).
    """
    if not snippet_text:
        return ""
    # 1. Strip the cut-pasteable command boxes
    cleaned = re.sub(r"\[\[CMD\]\].*?\[\[/CMD\]\]\n?", "", snippet_text, flags=re.DOTALL)
    out_lines: list[str] = []
    for raw in cleaned.splitlines():
        # 2. Strip GH Actions timestamp prefix and ANSI codes
        line = _ANSI_RE.sub("", _TS_PREFIX_RE.sub("", raw))
        # 3. Drop pure noise lines (best-effort — keep error context)
        s = line.strip()
        if not s:
            out_lines.append("")
            continue
        if s.startswith("##[group]") or s == "##[endgroup]":
            continue
        if s.startswith("Categories:") or s == "...":
            continue
        # 4. Plain-escape echo lines — keywords inside literal echo args are noise
        # Matches: 'echo "X"',  '+ echo "X"' (xtrace),  '  echo X', etc.
        if _ECHO_LINE_RE.match(line):
            out_lines.append(_html_escape(line))
            continue
        # 5. Highlight error keywords (red), default-render rest as plain text
        if _ensure_ci_log_errors():
            try:
                out_lines.append(_html_highlight_error_keywords(line))
                continue
            except Exception:
                pass
        out_lines.append(_html_escape(line))
    # Collapse runs of empty lines
    collapsed: list[str] = []
    blank = False
    for ln in out_lines:
        if ln == "":
            if not blank:
                collapsed.append("")
            blank = True
        else:
            collapsed.append(ln)
            blank = False
    return "\n".join(collapsed).strip("\n")


# ---------- state machine ----------


# ---------- subcommands ----------


def _job_conclusion(j) -> str:
    """Read conclusion from a jobs[name] entry (handles legacy str + new dict)."""
    if isinstance(j, str):
        return j
    if isinstance(j, dict):
        return j.get("conclusion") or "?"
    return "?"


def _job_url(j) -> str | None:
    """Read html_url from a jobs[name] entry. None for legacy str format."""
    if isinstance(j, dict):
        return j.get("url")
    return None


def _fmt_duration(secs: int) -> str:
    """Compact human-readable duration: Ns / MmSSs / HhMm / DdHhMm.

    Single-digit minor components stay unpadded ('4h3m', '1d4h5m') except for
    the seconds in MmSSs which we keep zero-padded so 5m09s and 5m59s line up
    visually in stacked tables. Examples: 45s, 5m31s, 5m09s, 4h3m, 1d4h5m.
    """
    if secs < 0:
        return "—"
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m{secs % 60:02d}s"
    if secs < 86400:
        return f"{secs // 3600}h{(secs % 3600) // 60}m"
    days, rem = divmod(secs, 86400)
    return f"{days}d{rem // 3600}h{(rem % 3600) // 60}m"


def _job_timing(j, conclusion: str = "") -> tuple[str, str]:
    """Return (started_str, duration_str) for display.

    For running jobs (conclusion in running/queued/pending) and a known
    started_at, emits a `<span class='live-duration' data-started='<iso>'>`
    that the page's JS ticker updates every second.
    """
    if not isinstance(j, dict):
        return "—", "—"
    s = j.get("started_at")
    c = j.get("completed_at")
    started = "—"
    duration = "—"
    if s:
        dt = _to_pt(s)
        if dt:
            started = dt.strftime("%H:%M:%S")
    if s and c:
        try:
            ds = datetime.fromisoformat(s)
            dc = datetime.fromisoformat(c)
            duration = _fmt_duration(int((dc - ds).total_seconds()))
        except Exception:
            pass
    elif s and conclusion in ("running", "queued", "pending"):
        try:
            ds = datetime.fromisoformat(s)
            now = datetime.now(timezone.utc)
            initial = _fmt_duration(int((now - ds).total_seconds()))
        except Exception:
            initial = "—"
        duration = (
            f"<span class='live-duration' data-started='{s}'>{initial}</span>"
        )
    return started, duration


_LIVE_DURATION_JS = """
<script>
(function() {
  function pad(n) { return n < 10 ? "0" + n : "" + n; }
  function fmt(secs) {
    if (secs < 0) return "—";
    if (secs < 60) return secs + "s";
    if (secs < 3600) return Math.floor(secs / 60) + "m" + pad(secs % 60) + "s";
    if (secs < 86400) return Math.floor(secs / 3600) + "h" + Math.floor((secs % 3600) / 60) + "m";
    var days = Math.floor(secs / 86400);
    var rem = secs % 86400;
    return days + "d" + Math.floor(rem / 3600) + "h" + Math.floor((rem % 3600) / 60) + "m";
  }
  function _setOver90m(el, secs) {
    // Toggle the red-class on the cell that contains this live span (and on
    // the span itself in case CSS targets it directly). >90m == > 5400s.
    var over = secs > 5400;
    el.classList.toggle("duration-over-90m", over);
    var td = el.closest("td");
    if (td) td.classList.toggle("duration-over-90m", over);
  }
  function tick() {
    var now = Date.now();
    document.querySelectorAll(".live-duration").forEach(function(el) {
      var s = el.getAttribute("data-started");
      if (!s) return;
      var t = Date.parse(s);
      if (isNaN(t)) return;
      var secs = Math.floor((now - t) / 1000);
      el.textContent = fmt(secs);
      _setOver90m(el, secs);
    });
    document.querySelectorAll(".live-duration-total").forEach(function(el) {
      var fixed = parseInt(el.getAttribute("data-fixed") || "0", 10) || 0;
      var liveAttr = el.getAttribute("data-live") || "";
      var liveSecs = 0;
      if (liveAttr) {
        liveAttr.split(",").forEach(function(s) {
          if (!s) return;
          var t = Date.parse(s);
          if (isNaN(t)) return;
          liveSecs += Math.floor((now - t) / 1000);
        });
      }
      var total = fixed + liveSecs;
      el.textContent = fmt(total);
      _setOver90m(el, total);
    });
  }
  tick();
  setInterval(tick, 1000);
})();
function toggleSnip(ev, id) {
  var row = document.getElementById(id);
  if (row) row.classList.toggle('show');
  var tgt = ev.currentTarget || ev.target;
  if (tgt) {
    var tri = tgt.querySelector('.triangle-toggle');
    if (tri) tri.classList.toggle('expanded');
  }
  ev.stopPropagation();
}
</script>
"""


def _descriptive_counts(entry: dict) -> str:
    """Long form for HTML: '76 pass, 2 fail, 12 running, 4 skipped'."""
    attempts = entry.get("attempts", [])
    if not attempts:
        return "—"
    jobs = attempts[-1].get("jobs", {})
    if not jobs:
        return "—"
    cons = [_job_conclusion(v) for v in jobs.values()]
    p = sum(1 for c in cons if c == "success")
    f = sum(1 for c in cons if c in ("failure", "timed_out"))
    r = sum(1 for c in cons if c in ("running", "queued", "pending"))
    s = sum(1 for c in cons if c in ("skipped", "cancelled", "neutral"))
    parts = [f"{p} pass", f"{f} fail", f"{r} running"]
    if s:
        parts.append(f"{s} skipped")
    return ", ".join(parts)


def _pr_url(pr: int | None) -> str:
    if not pr or pr == "-" or pr == -1:
        return "-"
    return f"https://github.com/{REPO}/pull/{pr}"


def _to_pt(iso: str | None) -> datetime | None:
    """Parse an ISO-8601 string and convert to Pacific time."""
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso).astimezone(PT)
    except Exception:
        return None


def _short_merge_date(iso: str | None) -> str:
    """'YYYY-MM-DD HH:MM:SS PT' in Pacific time."""
    dt = _to_pt(iso)
    if dt is None:
        return "?"
    return dt.strftime("%Y-%m-%d %H:%M:%S PT")


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# History bar primitives — shared between the per-SHA Run-history and the
# aggregate-report PR-history. Each "cell" represents a single run or SHA;
# the bar is a horizontal sequence of cells (oldest left → newest right).
# ---------------------------------------------------------------------------

_HB_BG = {
    "failed": "#c83a3a",
    "passed": "#2da44e",
    "other": "#d0d7de",
    "missing": "#d0d7de",
}

# CSS for both bars + popup. Inject once per page that uses the bar.
HB_CSS = (
    "<style>"
    "a.hb-link { text-decoration: none; } "
    "a.hb-link[target=\"_blank\"]::after { content: none; } "
    ".hb-pop-menu { visibility: hidden; opacity: 0; "
    "transition: visibility 0s linear 500ms, opacity 150ms ease 350ms; "
    "position: absolute; top: 11px; left: 0; z-index: 1000; background: #fff; "
    "border: 1px solid #d0d7de; border-radius: 4px; "
    "box-shadow: 0 4px 8px rgba(0,0,0,0.12); padding: 6px 8px; line-height: 1.5; "
    "font-size: 12px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; "
    "min-width: 280px; white-space: nowrap; } "
    ".hb-pop:hover .hb-pop-menu, .hb-pop-menu:hover { "
    "visibility: visible; opacity: 1; "
    "transition: visibility 0s linear 0s, opacity 100ms ease 0s; } "
    ".hb-pop-menu .hb-pop-head { display: block; color: #586069; font-size: 11px; "
    "padding-bottom: 4px; border-bottom: 1px solid #eaecef; margin-bottom: 4px; } "
    ".hb-pop-menu .hb-pop-item { display: block; padding: 2px 4px; color: #24292e; "
    "text-decoration: none; border-radius: 3px; } "
    ".hb-pop-menu a.hb-pop-item:hover { background: #f6f8fa; } "
    ".hb-pop-menu a.hb-pop-item[target=\"_blank\"]::after { content: \" \\2197\"; color: #959da5; }"
    "</style>"
)


def hb_popup_menu(head: str, items: list[dict]) -> str:
    """items: each dict has status, label_text, sublabels, url."""
    rows = []
    for it in items:
        bg = _HB_BG.get(it.get("status", "other"), "#d0d7de")
        label_color = (
            "#c83a3a" if it.get("status") == "failed"
            else ("#2da44e" if it.get("status") == "passed" else "#586069")
        )
        line = (
            f"<span style='display:inline-block;width:8px;height:8px;"
            f"border-radius:2px;background:{bg};margin-right:6px;"
            f"vertical-align:middle;'></span>"
            f"<span style='color:{label_color};font-weight:600;'>{_html_escape(it.get('label_text', ''))}</span>"
        )
        for sub in it.get("sublabels") or []:
            line += f" <span style='color:#586069;'>{_html_escape(sub)}</span>"
        url = it.get("url")
        if url:
            rows.append(
                f"<a class='hb-pop-item' href='{_html_escape(url)}' "
                f"target='_blank' rel='noopener noreferrer'>{line}</a>"
            )
        else:
            rows.append(f"<span class='hb-pop-item'>{line}</span>")
    return (
        f"<span class='hb-pop-menu'>"
        f"<span class='hb-pop-head'>{_html_escape(head)}</span>"
        + "".join(rows)
        + "</span>"
    )


def hb_cell(
    *,
    status: str,                       # 'failed'|'passed'|'other'|'missing'
    title: str,                        # plain-text tooltip (will be escaped)
    href: str | None = None,
    count_in_cell: int | None = None,  # show inside cell when > 1
    popup_html: str = "",              # output of hb_popup_menu, optional
    width: int = 9,
    height: int = 11,
) -> str:
    bg = _HB_BG.get(status, "#d0d7de")
    base = (
        f"display:inline-block; width:{width}px; height:{height}px; "
        "vertical-align:middle;"
    )
    count_label = (
        f"<span style='font-size:8px;color:#fff;font-weight:700;"
        f"line-height:{height}px;text-align:center;display:block;'>"
        f"{'+' if count_in_cell > 9 else count_in_cell}"
        f"</span>"
        if count_in_cell and count_in_cell > 1 else ""
    )
    cell_inner = (
        f"<span class='hb-cell' style='{base} background:{bg};' "
        f"title='{_html_escape(title)}'>{count_label}</span>"
    )
    # Hover-popup wrapper (preferred for multi-host cells).
    if popup_html:
        return (
            f"<span class='hb-pop' style='display:inline-block;position:relative;"
            f"margin-right:1px;line-height:0;'>{cell_inner}{popup_html}</span>"
        )
    # Single-link wrapper.
    if href:
        return (
            f"<a class='hb-link' href='{_html_escape(href)}' "
            f"target='_blank' rel='noopener noreferrer'>"
            + cell_inner.replace(base, base + " margin-right:1px;")
            + "</a>"
        )
    # Plain cell with margin.
    return cell_inner.replace(base, base + " margin-right:1px;")


def hb_bar(cells: list[str], font_size: int = 10) -> str:
    return (
        f"<span style='display:inline-block;white-space:nowrap;"
        f"font-family:\"SF Mono\",Consolas,monospace;font-size:{font_size}px;'>"
        + "".join(cells)
        + "</span>"
    )


THEME_BOOTSTRAP_SCRIPT = """<script>
  (function() {
    var cookieName = 'dynamo-dashboard-theme';
    var mode = 'auto';
    try {
      String(document.cookie || '').split(';').forEach(function(cookie) {
        var parts = cookie.trim().split('=');
        if (decodeURIComponent(parts[0] || '') === cookieName) {
          mode = decodeURIComponent(parts.slice(1).join('=') || '') || 'auto';
        }
      });
    } catch (e) {}
    function resolvedTheme(value) {
      if (value === 'light' || value === 'dark') return value;
      try {
        return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
      } catch (e) {
        return 'light';
      }
    }
    document.documentElement.setAttribute('data-theme-mode', mode);
    document.documentElement.setAttribute('data-theme', resolvedTheme(mode));
  })();
</script>"""


THEME_CONTROL_HTML = """<span class="theme-switcher" role="group" aria-label="Dashboard theme">
  <span class="theme-label">Theme</span>
  <button type="button" data-theme-choice="auto" aria-pressed="false">Auto</button>
  <button type="button" data-theme-choice="light" aria-pressed="false">Light</button>
  <button type="button" data-theme-choice="dark" aria-pressed="false">Dark</button>
</span>"""


THEME_RUNTIME_SCRIPT = """<script>
  (function() {
    var cookieName = 'dynamo-dashboard-theme';
    var media = null;
    try {
      media = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
    } catch (e) {}

    function resolvedTheme(value) {
      if (value === 'light' || value === 'dark') return value;
      return media && media.matches ? 'dark' : 'light';
    }

    function currentMode() {
      try {
        var found = 'auto';
        String(document.cookie || '').split(';').forEach(function(cookie) {
          var parts = cookie.trim().split('=');
          if (decodeURIComponent(parts[0] || '') === cookieName) {
            found = decodeURIComponent(parts.slice(1).join('=') || '') || 'auto';
          }
        });
        return found;
      } catch (e) {}
      return 'auto';
    }

    function rememberMode(mode) {
      try {
        document.cookie = cookieName + '=' + encodeURIComponent(mode) + '; Max-Age=31536000; Path=/; SameSite=Lax';
      } catch (e) {}
    }

    function applyTheme(mode) {
      if (mode !== 'light' && mode !== 'dark') mode = 'auto';
      document.documentElement.setAttribute('data-theme-mode', mode);
      document.documentElement.setAttribute('data-theme', resolvedTheme(mode));
      document.querySelectorAll('[data-theme-choice]').forEach(function(btn) {
        btn.setAttribute('aria-pressed', String(btn.getAttribute('data-theme-choice') === mode));
      });
    }

    document.addEventListener('DOMContentLoaded', function() {
      applyTheme(currentMode());
      document.querySelectorAll('[data-theme-choice]').forEach(function(btn) {
        btn.addEventListener('click', function() {
          var mode = btn.getAttribute('data-theme-choice') || 'auto';
          rememberMode(mode);
          applyTheme(mode);
        });
      });
    });

    if (media) {
      var onChange = function() {
        if ((document.documentElement.getAttribute('data-theme-mode') || 'auto') === 'auto') {
          applyTheme('auto');
        }
      };
      try { media.addEventListener('change', onChange); }
      catch (e) { try { media.addListener(onChange); } catch (ignored) {} }
    }
  })();
</script>"""


_HTML_STYLE = """
<style>
  body { font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0; padding: 24px; background: #fafbfc; color: #24292e; max-width: 1400px; }
  h1 { font-size: 20px; margin: 0 0 4px 0; }
  h2 { font-size: 16px; margin: 24px 0 8px 0; padding-bottom: 4px; border-bottom: 1px solid #e1e4e8; }
  .meta { color: #586069; font-size: 13px; margin-bottom: 16px; }
  .meta span { margin-right: 16px; }
  .meta code, code { background: #f6f8fa; padding: 2px 6px; border-radius: 3px; font-size: 12px;
                     font-family: "SF Mono", Consolas, monospace; }
  a { color: #0366d6; text-decoration: none; }
  a:hover { text-decoration: underline; }
  a[target="_blank"]::after { content: " ↗"; font-size: 0.85em; color: #959da5; }
  .verdict-good { color: #28a745; font-weight: 600; }
  .verdict-bad  { color: #d73a49; font-weight: 600; }
  .verdict-pending { color: #bf6c00; font-weight: 600; }
  .pill { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 12px;
          font-weight: 500; }
  .pill-pass { background: #d4edda; color: #155724; }
  .pill-fail { background: #f8d7da; color: #721c24; }
  .pill-run  { background: #cce5ff; color: #004085; }
  table { border-collapse: collapse; margin: 8px 0 16px 0; font-size: 13px; width: 100%; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #eaecef; vertical-align: top; }
  th { background: #f6f8fa; font-weight: 600; font-size: 12px; text-transform: uppercase;
       letter-spacing: 0.05em; color: #586069; }
  tr.fail td { background: #f5c6cb; }
  /* Final attempt of the job failed → unrecovered regression. Darker red. */
  tr.fail-final td { background: #ea868f; }
  /* Earlier failure that the job later recovered from. Lighter red. */
  tr.fail-flake td { background: #fbe4e6; }
  tr.pass td { background: #c3e6cb; }
  tr.run  td { background: #b8daff; }
  tr.skip td { background: #f0f0f0; color: #6a737d; }
  details { background: #fff; border: 1px solid #d0d7de; border-radius: 6px;
            margin: 8px 0; padding: 0; }
  details > summary { cursor: pointer; padding: 10px 14px; font-weight: 600;
                      list-style: none; user-select: none; }
  details > summary::-webkit-details-marker { display: none; }
  details > summary::before { content: "▶ "; color: #586069; font-size: 11px;
                              display: inline-block; transition: transform 0.1s; }
  details[open] > summary::before { transform: rotate(90deg); }
  details > summary:hover { background: #f6f8fa; border-radius: 6px 6px 0 0; }
  details[open] > summary { border-bottom: 1px solid #eaecef; border-radius: 6px 6px 0 0; }
  details > .details-body { padding: 0 14px 12px 14px; }
  .cat-list { font-size: 11px; color: #586069; font-family: "SF Mono", Consolas, monospace; }
  .attempt-badge { display: inline-block; padding: 1px 7px; margin: 0 4px 2px 0;
                   border-radius: 10px; font-size: 11px; font-weight: 600;
                   white-space: nowrap;
                   font-variant-numeric: tabular-nums; font-family: "SF Mono", Consolas, monospace; }
  td.attempts-cell { white-space: nowrap; min-width: 180px; }
  td.cat-cell { white-space: nowrap; min-width: 200px; }
  td.started-cell, td.duration-cell, td.status-cell, td.attempt-cell { white-space: nowrap; }
  td.duration-outlier { font-weight: 700; }
  /* Anything > 90 min is unusually slow — red on the per-SHA pages only
     (per-job row Duration cell, attempt-overview Duration column, Total CI
     time). JS tickers toggle the class as the clock crosses 5400s. The
     commit-history index.html keeps its own (unrelated) styling. */
  .duration-over-90m, td.duration-over-90m { color: #d73a49; }
  .attempt-summary { margin: 12px 0 4px 0; }
  .attempt-summary .pill { margin-right: 6px; }
  .snip-toggle { cursor: pointer; user-select: none; display: inline-block;
                 margin-left: 6px; vertical-align: middle; }
  .triangle-toggle { display: inline-block; transition: transform 300ms ease;
                     transform-origin: center; color: #586069; font-size: 14px;
                     margin-right: 4px; }
  .triangle-toggle.expanded { transform: rotate(90deg); }
  .legend-icon { display: inline-flex; vertical-align: text-bottom; margin: 0 1px; }
  table.attempt-table { width: auto; min-width: 540px; }
  table.attempt-table th, table.attempt-table td { text-align: center; white-space: nowrap; }
  table.attempt-table td.num-zero { color: #959da5; font-variant-numeric: tabular-nums; }
  table.attempt-table td.num-nz   { color: #24292e; font-variant-numeric: tabular-nums; font-weight: 600; }
  table.attempt-table td:first-child, table.attempt-table th:first-child { text-align: left; }
  .req-badge { display: inline-block; padding: 0 5px; margin-left: 6px;
               border-radius: 3px; font-size: 10px; font-weight: 600;
               background: #d73a49; color: #fff; vertical-align: middle;
               letter-spacing: 0.04em; text-transform: uppercase; }
  .opt-badge { display: inline-block; padding: 0 5px; margin-left: 6px;
               border-radius: 3px; font-size: 10px; font-weight: 600;
               background: #e1e4e8; color: #586069; vertical-align: middle;
               letter-spacing: 0.04em; text-transform: uppercase; }
  .status-x { display: inline-block; width: 14px; height: 14px; line-height: 14px;
              text-align: center; background: #d73a49; color: #fff;
              border-radius: 50%; font-weight: 700; font-size: 10px;
              font-family: "SF Mono", Consolas, monospace; }
  .status-check { display: inline-block; width: 14px; height: 14px; line-height: 14px;
                  text-align: center; background: #28a745; color: #fff;
                  border-radius: 50%; font-weight: 700; font-size: 10px; }
  .status-dot { display: inline-block; width: 14px; height: 14px; line-height: 14px;
                text-align: center; color: #586069; font-size: 14px; }
  tr.snippet-row { display: none; }
  tr.snippet-row.show { display: table-row; }
  tr.snippet-row > td { padding: 4px 14px 8px 14px; }
  tr.snippet-row.fail-final > td { background: #ea868f; }
  tr.snippet-row.fail-flake > td { background: #fbe4e6; }
  tr.snippet-row > td { background: #f5c6cb; }
  pre.snip { background: #0d1117; color: #e6edf3; font-size: 11px;
             padding: 10px 12px; border-radius: 4px; overflow-x: auto;
             margin: 4px 0 0 0; max-height: 320px; overflow-y: auto;
             white-space: pre-wrap; word-break: break-word;
             font-family: "SF Mono", Consolas, monospace; line-height: 1.45; }
  .job-name { font-family: "SF Mono", Consolas, monospace; font-size: 12px; white-space: nowrap; }
  .test-list { margin: 4px 0 0 16px; padding: 0; font-size: 12px;
               color: #586069; font-family: "SF Mono", Consolas, monospace; }
  .test-list li { margin: 2px 0; }
  .summary { display: flex; gap: 16px; margin: 8px 0 16px 0; }
  .summary-box { background: #fff; border: 1px solid #e1e4e8; border-radius: 6px;
                 padding: 12px 16px; min-width: 100px; }
  .summary-box .label { color: #586069; font-size: 11px; text-transform: uppercase;
                        letter-spacing: 0.05em; }
  .summary-box .value { font-size: 22px; font-weight: 600; margin-top: 4px; }
  .index-table tr:hover td { background: #f6f8fa; }
  :root {
    color-scheme: light dark;
    --bg: #fafbfc;
    --surface: #fff;
    --surface-muted: #f6f8fa;
    --border: #d0d7de;
    --border-muted: #eaecef;
    --text: #24292e;
    --text-inverse: #fff;
    --muted: #586069;
    --link: #0366d6;
    --success: #28a745;
    --danger: #d73a49;
    --warning: #bf6c00;
    --success-bg: #c3e6cb;
    --success-chip-bg: #d4edda;
    --success-chip-text: #155724;
    --danger-bg: #f5c6cb;
    --danger-final-bg: #ea868f;
    --danger-flake-bg: #fbe4e6;
    --danger-chip-bg: #f8d7da;
    --danger-chip-text: #721c24;
    --run-bg: #b8daff;
    --run-chip-bg: #cce5ff;
    --run-chip-text: #004085;
    --skip-bg: #f0f0f0;
    --skip-chip-bg: #eaeef2;
    --sha-zero: #959da5;
    --theme-chip-bg: rgba(36, 41, 46, 0.08);
  }
  html[data-theme="dark"] {
    color-scheme: dark;
    --bg: #0d1117;
    --surface: #161b22;
    --surface-muted: #21262d;
    --border: #30363d;
    --border-muted: #30363d;
    --text: #e6edf3;
    --text-inverse: #fff;
    --muted: #8b949e;
    --link: #58a6ff;
    --success: #3fb950;
    --danger: #ff7b72;
    --warning: #d29922;
    --success-bg: #17351f;
    --success-chip-bg: #17351f;
    --success-chip-text: #7ee787;
    --danger-bg: #4a1f24;
    --danger-final-bg: #7d2f38;
    --danger-flake-bg: #3d2027;
    --danger-chip-bg: #4a1f24;
    --danger-chip-text: #ffb3ad;
    --run-bg: #0d2d4d;
    --run-chip-bg: #0d2d4d;
    --run-chip-text: #79c0ff;
    --skip-bg: #21262d;
    --skip-chip-bg: #21262d;
    --sha-zero: #6e7681;
    --theme-chip-bg: rgba(139, 148, 158, 0.18);
  }
  h1 { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
  .theme-switcher {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    margin-left: auto;
    padding: 2px;
    border-radius: 999px;
    background: var(--theme-chip-bg);
    color: var(--text);
    font-size: 11px;
    font-weight: 400;
  }
  .theme-switcher .theme-label {
    padding: 2px 6px;
    color: var(--muted);
    font-weight: 600;
  }
  .theme-switcher button {
    border: 0;
    border-radius: 999px;
    padding: 2px 8px;
    background: transparent;
    color: var(--text);
    font: inherit;
    cursor: pointer;
  }
  .theme-switcher button[aria-pressed="true"] {
    background: var(--surface);
    color: var(--text);
    font-weight: 700;
    box-shadow: inset 0 0 0 1px var(--border);
  }
  .theme-switcher button:hover { background: var(--surface-muted); }
  html[data-theme="dark"] body { background: var(--bg); color: var(--text); }
  html[data-theme="dark"] h2 { border-bottom-color: var(--border); }
  html[data-theme="dark"] .meta,
  html[data-theme="dark"] .cat-list,
  html[data-theme="dark"] .triangle-toggle,
  html[data-theme="dark"] .test-list,
  html[data-theme="dark"] .summary-box .label { color: var(--muted); }
  html[data-theme="dark"] a { color: var(--link); }
  html[data-theme="dark"] code,
  html[data-theme="dark"] .meta code { background: var(--surface-muted); color: var(--text); }
  html[data-theme="dark"] th {
    background: var(--surface-muted);
    color: var(--muted);
  }
  html[data-theme="dark"] th,
  html[data-theme="dark"] td { border-bottom-color: var(--border-muted); }
  html[data-theme="dark"] details,
  html[data-theme="dark"] .summary-box {
    background: var(--surface);
    border-color: var(--border);
  }
  html[data-theme="dark"] details > summary:hover,
  html[data-theme="dark"] .index-table tr:hover td { background: var(--surface-muted); }
  html[data-theme="dark"] details[open] > summary { border-bottom-color: var(--border-muted); }
  html[data-theme="dark"] .v-good,
  html[data-theme="dark"] .verdict-good { color: var(--success); }
  html[data-theme="dark"] .v-bad,
  html[data-theme="dark"] .verdict-bad,
  html[data-theme="dark"] .duration-over-90m,
  html[data-theme="dark"] td.duration-over-90m,
  html[data-theme="dark"] ul.tests { color: var(--danger); }
  html[data-theme="dark"] .v-other,
  html[data-theme="dark"] .verdict-pending { color: var(--warning); }
  html[data-theme="dark"] .pill-pass { background: var(--success-chip-bg); color: var(--success-chip-text); }
  html[data-theme="dark"] .pill-fail { background: var(--danger-chip-bg); color: var(--danger-chip-text); }
  html[data-theme="dark"] .pill-run { background: var(--run-chip-bg); color: var(--run-chip-text); }
  html[data-theme="dark"] .pill-skip { background: var(--skip-chip-bg); color: var(--muted); }
  html[data-theme="dark"] tr.fail td,
  html[data-theme="dark"] tr.snippet-row > td { background: var(--danger-bg); }
  html[data-theme="dark"] tr.fail-final td,
  html[data-theme="dark"] tr.snippet-row.fail-final > td { background: var(--danger-final-bg); }
  html[data-theme="dark"] tr.fail-flake td,
  html[data-theme="dark"] tr.fail-eclipsed td,
  html[data-theme="dark"] tr.snippet-row.fail-flake > td { background: var(--danger-flake-bg); }
  html[data-theme="dark"] tr.pass td { background: var(--success-bg); }
  html[data-theme="dark"] tr.run td { background: var(--run-bg); }
  html[data-theme="dark"] tr.skip td { background: var(--skip-bg); color: var(--muted); }
  html[data-theme="dark"] table.attempt-table td.num-zero { color: var(--sha-zero); }
  html[data-theme="dark"] table.attempt-table td.num-nz { color: var(--text); }
  html[data-theme="dark"] [style*="color:#586069"],
  html[data-theme="dark"] [style*="color: #586069"] { color: var(--muted) !important; }
  html[data-theme="dark"] [style*="background:#fff8c5"] {
    background: #3a330a !important;
    border-color: #d29922 !important;
  }
  html[data-theme="dark"] [style*="background:#cce5ff"],
  html[data-theme="dark"] [style*="background: #cce5ff"] {
    background: var(--run-chip-bg) !important;
    color: var(--run-chip-text) !important;
  }
</style>
"""


def _display_state(entry: dict) -> str:
    """Human-readable state. Folds passed/cleaned→PASSED, failed→FAILED, discovered→queued."""
    s = entry.get("state", "?")
    if s in ("passed", "cleaned"):
        return "PASSED"
    if s == "failed":
        return "FAILED"
    if s == "discovered":
        return "queued"
    return s


def _state_class(entry: dict) -> str:
    s = entry.get("state")
    if s in ("passed", "cleaned"):
        return "verdict-good"
    if s == "failed":
        return "verdict-bad"
    return "verdict-pending"


def _render_jobs_pivoted_html(
    attempts_all: list[dict],
    job_runs: dict,
    longest_keys: set,
    _job_grad_stop: dict,
    next_att_start_for_job: dict,
    job_last_attempt: dict,
    snip_uid: list,
    att_verdict_map: dict | None = None,
) -> str:
    """Pivoted Jobs table: one row per unique job; cols = Attempt 1..N + Last
    failed (attempt #X) detail. Each attempt cell = status icon + duration,
    wrapped in <a> to the GH Actions job URL.
    """
    if not job_runs:
        return ""
    max_att = max(
        (max((a for a, _ in runs), default=0) for runs in job_runs.values()),
        default=0,
    )
    if max_att == 0:
        return ""

    def job_priority(name: str):
        runs = job_runs[name]
        last_conc = _job_conclusion(runs[-1][1]) if runs else ""
        last_failed = last_conc in ("failure", "timed_out")
        is_req = 0 if name in REQUIRED_CHECKS else 1
        return (0 if last_failed else 1, is_req, name.lower())

    sorted_jobs = sorted(job_runs.keys(), key=job_priority)

    av = att_verdict_map or {}
    def _verdict_pill(n: int) -> str:
        v = av.get(n)
        if v == "PASS":
            return "<span class='pill pill-pass'>PASS</span>"
        if v == "FAIL":
            return "<span class='pill pill-fail'>FAIL</span>"
        if v == "PENDING":
            return "<span class='pill pill-run'>PENDING</span>"
        return ""

    header_row1 = (
        ["<th>Job</th>"]
        + [f"<th>Run #{n}</th>" for n in range(1, max_att + 1)]
    )
    header_row2 = (
        ["<th></th>"]
        + [
            f"<th style='font-weight:400; padding-top:2px;'>{_verdict_pill(n)}</th>"
            for n in range(1, max_att + 1)
        ]
    )

    rows_html: list[str] = []
    for name in sorted_jobs:
        runs = job_runs[name]
        runs_by_att: dict = {a: j for a, j in runs}
        is_req = name in REQUIRED_CHECKS

        # Pick the most useful job URL for the name link (prefer a failed run,
        # else last run).
        link_url = None
        for _, j in runs:
            if _job_conclusion(j) in ("failure", "timed_out"):
                link_url = _job_url(j) or link_url
                break
        if not link_url and runs:
            link_url = _job_url(runs[-1][1])

        name_disp = _html_escape(name)
        name_html = (
            f"<a href='{link_url}' target='_blank' rel='noopener noreferrer'>{name_disp}</a>"
            if link_url else name_disp
        )
        if is_req:
            name_html += " <span class='req-badge' title='required for merge'>required</span>"

        # Run cells are built first; the Job cell (with optional expand
        # toggle) is prepended once we know whether the row has details.
        run_cells: list[str] = []

        for n in range(1, max_att + 1):
            j = runs_by_att.get(n)
            if j is None:
                run_cells.append(
                    "<td class='attempt-pivot-cell' style='color:#d0d7de; text-align:center;'>—</td>"
                )
                continue
            conc = _job_conclusion(j)
            # Stale-running guard (same as flat layout).
            if (
                conc in ("running", "queued", "pending")
                and n < job_last_attempt.get(name, n)
            ):
                conc = "cancelled"
                _proxy_end = next_att_start_for_job.get((name, n))
                if _proxy_end and isinstance(j, dict) and not j.get("completed_at"):
                    j = {**j, "completed_at": _proxy_end}

            if conc in ("failure", "timed_out"):
                icon = ICON_REQ_FAIL if is_req else ICON_OPT_FAIL
            elif conc == "success":
                icon = ICON_REQ_PASS if is_req else ICON_OPT_PASS
            elif conc in ("running", "queued", "pending"):
                icon = ICON_RUN
            elif conc in ("skipped", "cancelled", "neutral"):
                icon = f"<span class='status-dot' title='{conc}'>⊘</span>"
            else:
                _conc_disp = conc or "?"
                icon = f"<span class='status-dot' title='{_conc_disp}'>?</span>"

            _, duration = _job_timing(j, conc)
            url = _job_url(j)
            grad = _job_grad_stop.get((name, n))
            cell_style = f" style='background:{grad};'" if grad else ""
            dur_marks = " ⏱" if (name, n) in longest_keys else ""

            inner = f"{icon} <span style='font-variant-numeric:tabular-nums;'>{duration}{dur_marks}</span>"
            if url:
                inner = (
                    f"<a href='{url}' target='_blank' rel='noopener noreferrer' "
                    f"style='text-decoration:none; color:inherit;' "
                    f"title='run {n}: {conc}'>{inner}</a>"
                )
            run_cells.append(
                f"<td class='attempt-pivot-cell' style='white-space:nowrap;'{cell_style}>{inner}</td>"
            )

        # Find the last (highest-numbered) attempt with a failure for this job.
        # Its details (test list + snippet) get a triangle toggle next to the
        # job name itself.
        last_failed_att = None
        for n in range(max_att, 0, -1):
            j = runs_by_att.get(n)
            if j is None:
                continue
            if _job_conclusion(j) in ("failure", "timed_out"):
                last_failed_att = n
                break

        toggle_html = ""
        detail_expand_row = ""
        if last_failed_att is not None:
            for a in attempts_all:
                if a.get("attempt") != last_failed_att:
                    continue
                log_info = (a.get("log_meta") or {}).get(name, {})
                cats = log_info.get("categories") or []
                snippet = log_info.get("snippet") or ""
                tests = (a.get("failed_tests") or {}).get(name, [])
                if not (cats or tests or snippet):
                    break
                snip_uid[0] += 1
                sid = f"snip-pivot-{snip_uid[0]}"
                bits: list[str] = []
                if tests:
                    n_tests = len(tests)
                    bits.append(f"{n_tests} failed test{'s' if n_tests != 1 else ''}")
                if cats:
                    bits.append(f"{len(cats)} categor{'ies' if len(cats) != 1 else 'y'}")
                if snippet:
                    bits.append(f"snippet ({len(snippet)} chars)")
                summary_text = _html_escape(" • ".join(bits))
                toggle_html = (
                    f" <span class='snip-toggle' "
                    f"onclick=\"toggleSnip(event, '{sid}')\" "
                    f"title='last failed: run #{last_failed_att} — click to expand'>"
                    f"<span class='triangle-toggle'>▶</span>"
                    f"</span>"
                    f" <span style='color:#586069; font-size:11px;'>"
                    f"last failed: run #{last_failed_att} ({summary_text})</span>"
                )
                body_parts: list[str] = []
                if cats:
                    body_parts.append(
                        f"<div class='cat-list'>{_html_escape(', '.join(cats))}</div>"
                    )
                if tests:
                    items = "".join(
                        f"<li>{_html_escape(t)}</li>" for t in tests[:50]
                    )
                    more = (
                        f"<li>… +{len(tests) - 50} more</li>"
                        if len(tests) > 50 else ""
                    )
                    body_parts.append(
                        f"<ul class='test-list'>{items}{more}</ul>"
                    )
                if snippet:
                    body_parts.append(
                        f"<pre class='snip'>{render_snippet_html(snippet)}</pre>"
                    )
                col_span = max_att + 1  # Job + N runs
                detail_expand_row = (
                    f"<tr id='{sid}' class='snippet-row'>"
                    f"<td colspan='{col_span}'>{''.join(body_parts)}</td>"
                    f"</tr>"
                )
                break

        job_cell = f"<td class='job-name'>{name_html}{toggle_html}</td>"
        rows_html.append(f"<tr>{job_cell}{''.join(run_cells)}</tr>")
        if detail_expand_row:
            rows_html.append(detail_expand_row)

    sec = [
        "<h2>Jobs (one row per unique job; runs as columns)</h2>",
        "<table><thead>"
        + "<tr>" + "".join(header_row1) + "</tr>"
        + "<tr>" + "".join(header_row2) + "</tr>"
        + "</thead><tbody>",
    ]
    sec.extend(rows_html)
    sec.append("</tbody></table>")
    return "\n".join(sec)


def render_ci_attempts_page(
    sha: str,
    entry: dict,
    kind: str = "Pre-merge",
    att_verdict_override: dict | None = None,
) -> str:
    """Per-SHA CI attempts detail page.

    kind drives the page title/H1 ("Pre-merge" / "Post-merge" / ...).
    Jobs are rendered as a pivoted table: one row per unique job, runs as
    columns (with a hover popup that lists every host job per run).
    att_verdict_override: optional {run_n: "PASS"|"FAIL"|"PENDING"} that
    overrides the mandatory-only verdict computation (used by post-merge,
    where there's no required/optional distinction).
    """
    pr = entry.get("pr")
    pr_url = _pr_url(pr) if pr else "-"
    commit_url = f"https://github.com/{REPO}/commit/{sha}"
    state_disp = _display_state(entry)
    sclass = _state_class(entry)
    img = entry.get("image_sha256") or "?"
    merge_dt = _short_merge_date(entry.get("merge_date"))
    n_att = len(entry.get("attempts", []))
    subj, orig_pr, author = commit_subject_pr_author(sha)
    subj_esc = _html_escape(subj) if subj else ""
    author_esc = _html_escape(author) if author else ""
    orig_pr_html = (
        f', PR <a href="https://github.com/{REPO}/pull/{orig_pr}" target="_blank" rel="noopener noreferrer">#{orig_pr}</a>'
        if orig_pr else ""
    )
    title_html = f' &ldquo;{subj_esc}&rdquo;' if subj_esc else ""
    author_html = f' by {author_esc}' if author_esc else ""

    # Use entry.pr (pre-merge) when present; otherwise fall back to the PR
    # number parsed from the commit subject (post-merge), so post-merge titles
    # show "PR #N Post-merge Jobs and Runs Summary" too.
    _title_pr = pr or orig_pr
    _title_prefix = f"PR #{_title_pr}" if _title_pr else short_sha(sha)
    _h1_pr_link = (
        f'<a href="{_pr_url(_title_pr)}" target="_blank" rel="noopener noreferrer">PR #{_title_pr}</a>'
        if _title_pr else short_sha(sha)
    )
    head = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{_title_prefix} {kind} Jobs and Runs Summary</title>
{THEME_BOOTSTRAP_SCRIPT}
{_HTML_STYLE}
</head><body>
<h1><span>{_h1_pr_link} {kind} Jobs and Runs Summary</span>{THEME_CONTROL_HTML}</h1>
<div class="meta">
  <span><a href="{commit_url}" target="_blank" rel="noopener noreferrer">{short_sha(sha)}</a> on GitHub{orig_pr_html}{title_html}{author_html}</span>
</div>
<div class="meta">
  <span>Merged (PT): <code>{merge_dt}</code></span>
  <span>State: <span class="{sclass}">{state_disp}</span></span>
  <span>Runs: <code>{n_att}/{entry.get('max_attempts', DEFAULT_MAX_ATTEMPTS)}</code></span>
</div>
"""
    # If overall PASSED but earlier attempts had failures → flake banner.
    attempts_list = entry.get("attempts", [])
    if state_disp == "PASSED" and any(a.get("failed_jobs") for a in attempts_list):
        n_flaked = sum(len(a.get("failed_jobs", [])) for a in attempts_list[:-1])
        head += (
            f'<div style="background:#fff8c5; border:1px solid #d4a72c; '
            f'border-radius:6px; padding:10px 14px; margin:8px 0 16px 0;">'
            f'<strong>Confirmed flake.</strong> Final verdict: '
            f'<span class="{sclass}">PASSED</span>. '
            f'Earlier run(s) had {n_flaked} job-level failure(s); '
            f'cleared automatically by <code>gh run rerun --failed</code>. '
            f'Per-attempt history shown below.'
            f'</div>'
        )
    # Live counts summary
    counts = _descriptive_counts(entry)
    summary = f"""
<div class="summary">
  <div class="summary-box" style="min-width: 240px;"><div class="label">Latest counts</div><div class="value" style="font-size: 16px;">{counts}</div></div>
  <div class="summary-box"><div class="label">State</div><div class="value"><span class="{sclass}">{state_disp}</span></div></div>
</div>
"""

    # ------- View: group by JOB, with attempt history badges -------
    sections = []
    attempts_all = entry.get("attempts", [])
    snip_uid = [0]

    # 1. Build job-history dedup'd by *physical* run.
    # GitHub's rerun-failed-jobs creates new job IDs in each attempt's view,
    # but for unchanged (carried-over) jobs those new IDs point at the same
    # physical run with the same timestamps. Dedup on (started_at, completed_at)
    # — two entries with identical timestamps for the same job name ARE the
    # same physical run, regardless of GitHub's per-attempt ID.
    job_runs: dict[str, list[tuple[int, dict]]] = {}
    for a in attempts_all:
        att_num = a.get("attempt", 0)
        for name, j in a.get("jobs", {}).items():
            runs = job_runs.setdefault(name, [])
            if not isinstance(j, dict):
                runs.append((att_num, j))
                continue
            sig = (j.get("started_at"), j.get("completed_at"))
            existing = {
                (r[1].get("started_at"), r[1].get("completed_at"))
                for r in runs if isinstance(r[1], dict)
            }
            # If both ends are None (no timing yet), treat as unique per attempt.
            if sig != (None, None) and sig in existing:
                continue  # same physical run — skip dup
            runs.append((att_num, j))

    # 2. Per-attempt overview now reflects what ACTUALLY ran in each attempt
    # (new physical runs, not carried-over status), split into mandatory
    # (REQUIRED_CHECKS) vs optional. Verdict is *mandatory-only*: if any
    # required check failed → FAIL; else if all required checks succeeded
    # → PASS; else → PENDING.
    runs_per_att: dict[int, list[tuple[str, dict]]] = {}
    for name, runs in job_runs.items():
        for att_num, j in runs:
            runs_per_att.setdefault(att_num, []).append((name, j))

    summary_lines = [
        "<h2>Runs overview <small style='color:#586069;font-weight:400'>(verdict = mandatory-only)</small></h2>",
        "<table class='attempt-table'><thead><tr>"
        "<th>Run</th>"
        "<th>Verdict</th>"
        f"<th title='required passed'>Req {ICON_REQ_PASS}</th>"
        f"<th title='required failed'>Req {ICON_REQ_FAIL}</th>"
        f"<th title='optional passed'>Opt {ICON_OPT_PASS}</th>"
        f"<th title='optional failed'>Opt {ICON_OPT_FAIL}</th>"
        f"<th title='in progress' style='background:#cce5ff;color:#004085;'>RUNNING {ICON_RUN}</th>"
        "<th>Skipped</th>"
        "<th>Started (PT)</th>"
        "<th title='wall-clock: earliest job start → latest job end in this run'>Duration</th>"
        "<th style='text-align:left;' title='who triggered this run (initial CI for run #1; re-run clicker for runs > 1)'>Run / Re-run triggered by</th>"
        "</tr></thead><tbody>",
    ]
    # Track previous attempt's verdict so attempts that re-ran ONLY optional
    # jobs (no required activity) inherit the previous PASS/FAIL — the
    # required jobs still hold their old conclusion via GitHub's job
    # carry-over, the rerun just didn't re-touch them.
    prev_verdict = "PENDING"
    fixed_total_secs = 0
    live_starts: list[str] = []  # iso-8601 starts of attempts still running
    # Only the highest-numbered attempt is "currently live"; older attempts are
    # frozen by definition. Probe writes attempts[current_attempt-1] only and
    # `get_run_jobs` uses `filter=latest`, so an old attempt's job state may be
    # stale (e.g. a job cancelled-during-rerun still appears as "running").
    # Treat older attempts as fixed: prefer max(end), else use the next
    # attempt's earliest start as a proxy end.
    latest_att_num = max(runs_per_att) if runs_per_att else None
    next_att_start_iso_by_att: dict[int, str] = {}
    sorted_atts = sorted(runs_per_att)
    for i, att_num in enumerate(sorted_atts[:-1]):
        nxt = runs_per_att[sorted_atts[i + 1]]
        nxt_starts = [
            j.get("started_at") for _, j in nxt
            if isinstance(j, dict) and j.get("started_at")
        ]
        if nxt_starts:
            next_att_start_iso_by_att[att_num] = min(nxt_starts)
    _attempt_rows: list[tuple] = []
    for att_num in sorted_atts:
        these = runs_per_att[att_num]
        is_latest = att_num == latest_att_num
        # Wall-clock duration for this attempt: earliest job start → latest end
        # if all jobs completed, else live (now - earliest start) and ticking.
        # Only the latest attempt may live-tick; older attempts are frozen.
        running_jobs = is_latest and any(
            isinstance(j, dict) and _job_conclusion(j) in ("running", "queued", "pending")
            for _, j in these
        )
        starts = [j.get("started_at") for _, j in these if isinstance(j, dict) and j.get("started_at")]
        ends = [j.get("completed_at") for _, j in these if isinstance(j, dict) and j.get("completed_at")]
        att_dur_str = "—"
        att_dur_secs = 0  # used for the >90m red-class decision (frozen rows)
        att_is_live = False
        if starts:
            try:
                t0_iso = min(starts)
                _t0 = datetime.fromisoformat(t0_iso)
                if running_jobs:
                    now = datetime.now(timezone.utc)
                    _secs = max(0, int((now - _t0).total_seconds()))
                    att_dur_secs = _secs
                    att_is_live = True
                    live_starts.append(t0_iso)
                    att_dur_str = (
                        f"<span class='live-duration' data-started='{t0_iso}'>"
                        f"{_fmt_duration(_secs)}</span>"
                    )
                else:
                    end_iso = max(ends) if ends else None
                    proxy = next_att_start_iso_by_att.get(att_num)
                    if proxy and (not end_iso or proxy > end_iso):
                        end_iso = proxy
                    if end_iso:
                        _t1 = datetime.fromisoformat(end_iso)
                        _secs = max(0, int((_t1 - _t0).total_seconds()))
                        att_dur_secs = _secs
                        fixed_total_secs += _secs
                        att_dur_str = _fmt_duration(_secs)
            except Exception:
                pass
        req_pass = req_fail = req_run = 0
        opt_pass = opt_fail = opt_run = 0
        n_skip = 0
        for nm, j in these:
            conc = _job_conclusion(j)
            is_req = nm in REQUIRED_CHECKS
            # Older attempts can have stale "running" jobs in the DB (probe
            # only refreshes the latest attempt). Treat those as cancelled
            # for verdict + counts — when attempt N+1 spun up, attempt N's
            # running jobs were effectively abandoned.
            if conc in ("running", "queued", "pending") and not is_latest:
                conc = "cancelled"
            if conc == "success":
                if is_req: req_pass += 1
                else: opt_pass += 1
            elif conc in ("failure", "timed_out"):
                if is_req: req_fail += 1
                else: opt_fail += 1
            elif conc in ("running", "queued", "pending"):
                if is_req: req_run += 1
                else: opt_run += 1
            elif conc in ("skipped", "cancelled", "neutral"):
                n_skip += 1
        if req_fail > 0:
            verdict = "FAIL"
        elif req_run + opt_run > 0:
            # Any running job (required OR optional) keeps the attempt PENDING.
            # We only call PASS when every job in this attempt is terminal —
            # otherwise users see "PASS" next to a non-zero Run count, which
            # is confusing (a job that's still building hasn't passed yet).
            verdict = "PENDING"
        elif req_pass > 0:
            verdict = "PASS"
        else:
            # No required job in this attempt's runs → carry the prior verdict.
            # If this is attempt 1 (no prior), keep PENDING.
            verdict = prev_verdict if att_num > 1 else "PENDING"
        # Caller-supplied override (e.g. post-merge, where there's no
        # required/optional distinction — the run's overall conclusion is
        # the verdict).
        if att_verdict_override and att_num in att_verdict_override:
            verdict = att_verdict_override[att_num]
        prev_verdict = verdict
        if verdict == "PASS":
            verdict_html = "<span class='pill pill-pass'>PASS</span>"
        elif verdict == "FAIL":
            verdict_html = "<span class='pill pill-fail'>FAIL</span>"
        else:
            verdict_html = "<span class='pill pill-run'>PENDING</span>"

        def _cell(n: int) -> str:
            cls = "num-zero" if n == 0 else "num-nz"
            return f"<td class='{cls}'>{n}</td>"

        # Red the Duration cell when frozen value > 90m. Live rows let JS toggle.
        _att_dur_cls = " duration-over-90m" if (not att_is_live and att_dur_secs > 5400) else ""
        # Defer row emission so we can post-decide a progressive-red gradient
        # if the LAST attempt ends up FAIL: each row gets a slightly redder
        # background, attempt 1 nearly white, the final attempt fail-red.
        _attempt_rows.append((att_num, verdict, verdict_html, req_pass, req_fail,
                              opt_pass, opt_fail, req_run + opt_run, n_skip,
                              _att_dur_cls, att_dur_str))

    # If the final attempt ended in FAIL, render the attempt rows with a
    # progressive-red gradient so the eye reads "this got worse with each
    # retry": attempt 1 nearly white, the last (failed) attempt fail-red.
    # Otherwise render plain.
    _last_verdict = _attempt_rows[-1][1] if _attempt_rows else None
    _N = len(_attempt_rows)
    # Disabled: progressive-red gradient on Runs overview was hiding the per-cell
    # verdict pills. Verdict column already encodes pass/fail per row.
    _grad_active = False
    # Endpoints: very-light pink (~white-pink hint) → fail-final red.
    _grad_light = (0xfe, 0xf2, 0xf3)
    _grad_dark = (0xea, 0x86, 0x8f)
    def _row_style(_idx: int) -> str:
        if not _grad_active:
            return ""
        _t = _idx / max(1, _N - 1)  # 0..1 across attempts
        _r = int(_grad_light[0] + _t * (_grad_dark[0] - _grad_light[0]))
        _g = int(_grad_light[1] + _t * (_grad_dark[1] - _grad_light[1]))
        _b = int(_grad_light[2] + _t * (_grad_dark[2] - _grad_light[2]))
        return f" style='background:#{_r:02x}{_g:02x}{_b:02x};'"
    def _cell_static(n: int) -> str:
        cls = "num-zero" if n == 0 else "num-nz"
        return f"<td class='{cls}'>{n}</td>"
    # Map attempt# → triggering_actor (login or None) for the new column.
    _att_trig: dict[int, str | None] = {
        a.get("attempt"): a.get("triggering_actor")
        for a in attempts_all if a.get("attempt") is not None
    }
    # Map attempt# → started_at (PT-formatted) for the Started column.
    _att_started: dict[int, str] = {}
    for a in attempts_all:
        n = a.get("attempt")
        if n is None:
            continue
        ca = a.get("started_at")
        dt = _to_pt(ca) if ca else None
        _att_started[n] = dt.strftime("%Y-%m-%d %H:%M:%S PT") if dt else "—"

    def _started_str(att_num: int) -> str:
        return _att_started.get(att_num, "—")
    for _idx, (att_num, _verdict, verdict_html, rp, rf, op, of, run_n, sk,
              dur_cls, dur_str) in enumerate(_attempt_rows):
        _login = _att_trig.get(att_num)
        if _login:
            _is_bot = _login == "copy-pr-bot[bot]"
            _wrap_open = "" if _is_bot else "<strong>"
            _wrap_close = "" if _is_bot else "</strong>"
            _trig_html = (
                f"{_wrap_open}<a href='https://github.com/{_login}' target='_blank' "
                f"rel='noopener noreferrer'>{_html_escape(_login)}</a>{_wrap_close}"
            )
        else:
            _trig_html = "<span style='color:#959da5'>—</span>"
        summary_lines.append(
            f"<tr{_row_style(_idx)}>"
            f"<td><strong>#{att_num}</strong></td>"
            f"<td>{verdict_html}</td>"
            f"{_cell_static(rp)}{_cell_static(rf)}"
            f"{_cell_static(op)}{_cell_static(of)}"
            f"{_cell_static(run_n)}"
            f"{_cell_static(sk)}"
            f"<td style='font-variant-numeric: tabular-nums;'>{_started_str(att_num)}</td>"
            f"<td class='att-dur-cell{dur_cls}' style='font-variant-numeric: tabular-nums;'>{dur_str}</td>"
            f"<td style='text-align:left;'>{_trig_html}</td>"
            f"</tr>"
        )
    if fixed_total_secs > 0 or live_starts:
        # Total = fixed completed time + live (now - earliest live start) summed.
        total_is_live = bool(live_starts)
        if live_starts:
            try:
                now = datetime.now(timezone.utc)
                live_secs_initial = sum(
                    max(0, int((now - datetime.fromisoformat(s)).total_seconds()))
                    for s in live_starts
                )
            except Exception:
                live_secs_initial = 0
            total_init = fixed_total_secs + live_secs_initial
            live_starts_attr = ",".join(live_starts)
            total_html = (
                f"<span class='live-duration-total' "
                f"data-fixed='{fixed_total_secs}' data-live='{live_starts_attr}'>"
                f"{_fmt_duration(total_init)}</span>"
            )
        else:
            total_init = fixed_total_secs
            total_html = _fmt_duration(fixed_total_secs)
        # Red when frozen total > 90m. Live cells let JS toggle.
        _total_cls = " duration-over-90m" if (not total_is_live and total_init > 5400) else ""
        summary_lines.append(
            "<tr style='border-top: 2px solid #d0d7de; font-weight:600;'>"
            "<td colspan='9' style='text-align:right; color:#586069;'>Total CI time</td>"
            f"<td class='total-ci-cell{_total_cls}' style='font-variant-numeric: tabular-nums;'>{total_html}</td>"
            "<td></td>"
            "</tr>"
        )
    summary_lines.append("</tbody></table>")
    sections.append("\n".join(summary_lines))

    # Compute shared locals consumed by _render_jobs_pivoted_html below.
    # (job_last_attempt, next_att_start_for_job, longest_keys, _job_grad_stop)

    flat_rows: list[tuple[str, int, dict]] = []
    for name, runs in job_runs.items():
        for att_num, j in runs:
            flat_rows.append((name, att_num, j))
    flat_rows.sort(key=lambda x: (x[0], x[1]))

    # Per job: which attempt # is the LAST run? Only that row renders dark
    # if it's a failure (= unrecovered final). Earlier failure rows of the
    # same job render light ("recovered or eclipsed by a later attempt").
    job_last_attempt: dict[str, int] = {}
    for name, runs in job_runs.items():
        if runs:
            job_last_attempt[name] = runs[-1][0]

    # For stale-running jobs (att N has conclusion=running but a later att N+1
    # exists for the same job), use att N+1's started_at as a proxy end. The
    # job was effectively abandoned when the next attempt spun up; that gives
    # us a frozen duration instead of a forever-ticking "now - start_at".
    next_att_start_for_job: dict[tuple[str, int], str] = {}
    for _name, _runs in job_runs.items():
        for _i, (_an, _) in enumerate(_runs[:-1]):
            _nxt_j = _runs[_i + 1][1]
            _nxt_s = _nxt_j.get("started_at") if isinstance(_nxt_j, dict) else None
            if _nxt_s:
                next_att_start_for_job[(_name, _an)] = _nxt_s

    # Outlier (job, attempt) pairs by wall-clock duration → mark with ⏱.
    # Genuinely-running jobs use "now" as end. Stale-running rows (older
    # attempts of a job that was rerun) are excluded from outlier
    # consideration entirely — their duration is artificial (start until
    # external cancel/supersede), not a fair signal of "slow build".
    # Tukey fence: duration > Q3 + 1.5*IQR. Only flags jobs that are
    # *meaningfully* slower than the rest; if everything is roughly the
    # same speed, no clock is shown.
    _durations: list[tuple[float, str, int]] = []
    _now_utc = datetime.now(timezone.utc)
    for _name, _att_num, _j in flat_rows:
        if not isinstance(_j, dict):
            continue
        _s = _j.get("started_at")
        if not _s:
            continue
        _c = _j.get("completed_at")
        # Skip stale-running rows from outlier/over-1h detection.
        if not _c and (_name, _att_num) in next_att_start_for_job:
            continue
        try:
            _t0 = datetime.fromisoformat(_s)
            _t1 = datetime.fromisoformat(_c) if _c else _now_utc
            _secs = (_t1 - _t0).total_seconds()
        except Exception:
            continue
        _durations.append((_secs, _name, _att_num))

    # Outlier rule: a job is flagged only if its duration is >2× the next
    # slowest job. Walk top-down: as long as durs[i] > 2 * durs[i+1] AND
    # > 60s (don't bother with tiny jobs), keep marking. Stop on the first
    # gap that's ≤2× — everything below clusters with the rest.
    # Outlier rule: find the biggest *ratio gap* between consecutive
    # sorted durations. Mark everything above that gap if the ratio is
    # > 2× and the marked entries are ≥ 60s. This handles tied/clustered
    # outliers (e.g. 4 jobs all stuck at 228m together): they all sit
    # above the gap to the next-fastest job, so all 4 get flagged.
    longest_keys: set[tuple[str, int]] = set()
    if len(_durations) >= 2:
        ranked = sorted(_durations, reverse=True)  # desc by duration
        best_gap_idx = -1
        best_ratio = 0.0
        for i in range(len(ranked) - 1):
            d_i, _, _ = ranked[i]
            d_next, _, _ = ranked[i + 1]
            if d_i < 60.0:
                break
            ratio = d_i / max(d_next, 1.0)
            if ratio > best_ratio:
                best_ratio = ratio
                best_gap_idx = i
        if best_gap_idx >= 0 and best_ratio > 2.0:
            for d_i, n_i, a_i in ranked[: best_gap_idx + 1]:
                if d_i >= 60.0:
                    longest_keys.add((n_i, a_i))

    # Per-job progressive-red gradient: when a job has multiple runs AND
    # its last run is a failure/timed_out, paint every (job, attempt) row
    # for that job along a gradient from very-light pink (first attempt)
    # to fail-final red (last attempt). Mirrors the Attempts-overview
    # gradient, scoped per-job. Stale-running attempts are NOT counted as
    # the "last" — they're treated as cancelled.
    _job_grad_stop: dict[tuple[str, int], str] = {}
    _grad_light = (0xfe, 0xf2, 0xf3)
    _grad_dark = (0xea, 0x86, 0x8f)
    for _name, _runs in job_runs.items():
        if len(_runs) < 2:
            continue
        # Resolve the LAST run's effective conclusion using the same stale
        # guard the per-row loop applies, so an old stuck "running" attempt
        # does not count as the failing terminal state.
        _last_att, _last_j = _runs[-1]
        _last_conc = _job_conclusion(_last_j)
        if _last_conc not in ("failure", "timed_out"):
            continue
        _N = len(_runs)
        for _idx, (_an, _) in enumerate(_runs):
            _t = _idx / max(1, _N - 1)
            _r = int(_grad_light[0] + _t * (_grad_dark[0] - _grad_light[0]))
            _g = int(_grad_light[1] + _t * (_grad_dark[1] - _grad_light[1]))
            _b = int(_grad_light[2] + _t * (_grad_dark[2] - _grad_light[2]))
            _job_grad_stop[(_name, _an)] = f"#{_r:02x}{_g:02x}{_b:02x}"

    # Build the pivoted Jobs section directly (no more flat layout).
    att_verdict_map = {row[0]: row[1] for row in _attempt_rows}
    if att_verdict_override:
        att_verdict_map.update(att_verdict_override)
    pivoted = _render_jobs_pivoted_html(
        attempts_all,
        job_runs,
        longest_keys,
        _job_grad_stop,
        next_att_start_for_job,
        job_last_attempt,
        snip_uid,
        att_verdict_map=att_verdict_map,
    )
    if pivoted:
        sections.append(pivoted)


    # Failing-test tally: list every failing pytest test-id seen across all
    # attempts/runs on this SHA, ordered by occurrence count desc.
    test_tally: dict[str, dict] = {}
    # test_id -> set of job names that have hosted this test (failed in them
    # at some point). Used to distinguish "ran and passed" from "didn't run".
    test_host_jobs: dict[str, set[str]] = {}
    for a in attempts_all:
        run_n = a.get("attempt")
        jobs_dict = a.get("jobs") or {}
        for job_name, tests in (a.get("failed_tests") or {}).items():
            j = jobs_dict.get(job_name) if isinstance(jobs_dict.get(job_name), dict) else None
            job_url = j.get("url") if j else None
            for t in tests or []:
                row = test_tally.setdefault(
                    t,
                    {"count": 0, "runs": set(), "run_to_url": {}},
                )
                row["count"] += 1
                if run_n is not None:
                    row["runs"].add(run_n)
                    row["run_to_url"].setdefault(run_n, job_url)
                test_host_jobs.setdefault(t, set()).add(job_name)

    if test_tally:
        ranked = sorted(
            test_tally.items(),
            key=lambda kv: (-kv[1]["count"], kv[0]),
        )
        n_unique = len(ranked)
        n_total = sum(info["count"] for _, info in ranked)

        # Per-run history bar, one cell per run (oldest left → newest right).
        # red = failed, green = host-job ran and passed, gray = didn't run.
        max_att = max(
            (a.get("attempt") for a in attempts_all if a.get("attempt") is not None),
            default=0,
        )

        def _hosts_in_run(test_id: str, run_n: int) -> list[tuple[dict, str, str]]:
            """Return list of (job, jname, status) for all host jobs of the test
            in run N. status ∈ {"failed", "passed", "other"}."""
            host = test_host_jobs.get(test_id, set())
            att = next(
                (a for a in attempts_all if a.get("attempt") == run_n), None
            )
            if not att:
                return []
            jd = att.get("jobs") or {}
            failed_tests_in_att = att.get("failed_tests") or {}
            out: list[tuple[dict, str, str]] = []
            for jname, j in jd.items():
                if jname not in host or not isinstance(j, dict):
                    continue
                if test_id in (failed_tests_in_att.get(jname) or []):
                    out.append((j, jname, "failed"))
                elif j.get("conclusion") == "success":
                    out.append((j, jname, "passed"))
                else:
                    out.append((j, jname, "other"))
            return out

        def _per_run_bar(test_id: str, info: dict) -> str:
            cells: list[str] = []
            for n in range(1, max_att + 1):
                hosts = _hosts_in_run(test_id, n)
                if not hosts:
                    cells.append(hb_cell(
                        status="missing",
                        title=f"Run #{n}: did not run (host job cancelled or missing)",
                    ))
                    continue
                statuses = {s for _, _, s in hosts}
                cell_status = (
                    "failed" if "failed" in statuses
                    else "passed" if "passed" in statuses
                    else "other"
                )
                items = []
                for j, jname, s in hosts:
                    label = (
                        "FAILED" if s == "failed"
                        else "passed" if s == "passed"
                        else (j.get("conclusion") or "unknown")
                    )
                    jid = j.get("id")
                    items.append({
                        "status": s,
                        "label_text": label,
                        "sublabels": [f"job#{jid or '?'}", jname],
                        "url": j.get("url"),
                    })
                cells.append(hb_cell(
                    status=cell_status,
                    title=f"Run #{n}: hover to see {len(hosts)} host job(s)",
                    count_in_cell=len(hosts),
                    popup_html=hb_popup_menu(
                        f"Run #{n} — {len(hosts)} host job(s)", items,
                    ),
                ))
            return hb_bar(cells)

        rows: list[str] = []
        for rank, (test_id, info) in enumerate(ranked, 1):
            rows.append(
                f"<tr><td style='text-align:left;'>{rank}</td>"
                f"<td style='text-align:left; font-variant-numeric:tabular-nums;'>{info['count']} / {n_total}</td>"
                f"<td style='text-align:left;'><code>{_html_escape(test_id)}</code></td>"
                f"<td style='text-align:left; white-space:nowrap;'>{_per_run_bar(test_id, info)}</td></tr>"
            )
        tally_html = (
            f"<h2>Failing tests <small style='color:#586069;font-weight:400'>"
            f"({n_unique} unique • {n_total} occurrences across runs)</small></h2>"
            + HB_CSS
            + "<table class='attempt-table'>"
            "<thead><tr>"
            "<th style='text-align:left;'>Rank</th>"
            "<th style='text-align:left;'>Count / Total</th>"
            "<th style='text-align:left;'>Test</th>"
            "<th style='text-align:left;' title='one cell per run (oldest left → newest right): red=failed, green=passed, gray=did not run; click colored cells to open the job log'>"
            "Run history &mdash; <small style='font-weight:400;color:#586069;'>oldest &#8594; newest</small></th>"
            "</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )
        # Surface failing tests at the top of the page (above Runs overview).
        sections.insert(0, tally_html)

    if not sections:
        sections.append("<p><em>No attempts recorded yet.</em></p>")

    foot = _LIVE_DURATION_JS + THEME_RUNTIME_SCRIPT + "</body></html>"
    return head + summary + "\n".join(sections) + foot


# ---------- main ----------


