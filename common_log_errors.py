#!/usr/bin/env python3
"""
Shared log error detection + categorization + snippet formatting utilities.

This module is intentionally dependency-light so it can be shared by:
- `dynamo-utils/common.py` (log/snippet extraction, cache logic)
- `dynamo-utils/html_pages/common_dashboard_lib.py` (HTML rendering for snippets/tags)
"""

from __future__ import annotations

import argparse
import html
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Pattern, Sequence

#
# Error snippet selection (text-only)
# =============================================================================
#

# Lines that should anchor an "error snippet" extraction from raw logs.
# Keep this conservative and high-signal to avoid pulling unrelated noise.
ERROR_SNIPPET_LINE_RE: Pattern[str] = re.compile(
    r"(?:"
    r"\b(?:error|failed|failure|exception|traceback|fatal)\b"
    # Pytest failures (the exact failing test id line is the most useful snippet anchor).
    # Example: "FAILED tests/x.py::test_name[param]"
    r"|(?:^|\s)FAILED(?:\s+|$).*::"
    # Avoid matching dependency strings like "pytest-timeout==2.4.0" or "timeout-2.4.0".
    r"|\b(?:time\s*out|timed\s*out)\b"
    # "timeout" is often a benign parameter (e.g. "--timeout 20", "timeout=120s"); don't treat those
    # as snippet anchors. Still keep real "timed out"/"time out" above, and allow other failure tokens.
    r"|(?<!-)\btimeout\b(?!\s*[=:]\s*\d)(?!\s+\d+(?:\.\d+)?\s*(?:ms|s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)?\b)"
    r"|\b[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)\b"
    r"|\b(?:broken\s+links?|broken\s+link|dead\s+links?)\b"
    r"|\b(?:network\s+error|connection\s+failed)\b"
    # Multi-line backend result blocks (JSON-ish) often show:
    #   "trtllm": { ... "result": "failure", ... }
    # Anchor on the high-signal failure field so the snippet includes the surrounding block.
    r"|\"result\"\s*:\s*\"failure\""
    r")",
    re.IGNORECASE,
)


#
# Categorization (text-only)
# =============================================================================
#

# Backend result blocks are printed as JSON-ish text in logs, e.g.:
#   "sglang": { ... "result": "failure", ... }
_BACKEND_BLOCK_START_RE: Pattern[str] = re.compile(r"\"(trtllm|sglang|vllm)\"\s*:\s*\{", re.IGNORECASE)
_BACKEND_RESULT_FAILURE_RE: Pattern[str] = re.compile(r"\"result\"\s*:\s*\"failure\"", re.IGNORECASE)


def _backend_failure_engines_from_lines(lines: Sequence[str]) -> set[str]:
    """Detect which backend engine blocks report `"result": "failure"` (multi-line aware).

    Example block (ANSI/timestamps may wrap lines):
        "sglang": {
          "result": "failure",
          ...
        },
    """
    engines: set[str] = set()
    try:
        current: Optional[str] = None
        for raw in (lines or []):
            s = str(raw or "")
            # Strip common timestamp prefixes and ANSI escapes.
            s = re.sub(r"^\d{4}-\d{2}-\d{2}T[0-9:.]+Z\s+", "", s)
            s = re.sub(r"\x1b\[[0-9;]*m", "", s)

            m = _BACKEND_BLOCK_START_RE.search(s)
            if m:
                current = str(m.group(1) or "").strip().lower() or None

            if current and _BACKEND_RESULT_FAILURE_RE.search(s):
                engines.add(current)

            if current:
                st = (s or "").strip()
                if st in ("}", "},"):
                    current = None
    except Exception:
        return engines
    return engines

def categorize_log_text(lines: Sequence[str]) -> List[str]:
    """Return high-level error categories found in log lines (best-effort).

    This is intentionally lightweight and heuristic; it powers the "Categories:" header in snippets
    and the inline category tags.
    """
    cats: List[str] = []
    try:
        text = "\n".join(lines[-4000:]) if lines else ""
        t = text.lower()

        def add(name: str) -> None:
            if name and name not in cats:
                cats.append(name)

        # Pytest / Python
        # Only tag "pytest" when the log looks like an actual pytest run failure,
        # not when it merely mentions pytest packages/versions (e.g., "pytest==9.0.2").
        if (
            re.search(r"===+\s*short test summary info\s*===+", t)
            or re.search(r"(?:^|\s)FAILED(?:\s+|$).*::", t, re.IGNORECASE)
            or re.search(r"(?:^|\s)E\s+\w+(?:Error|Exception)\b", t, re.IGNORECASE)
            or re.search(r"\berror\s+collecting\b", t, re.IGNORECASE)
        ):
            add("pytest")
        if re.search(r"\b(traceback|exception|assertionerror)\b", t):
            add("python-exception")

        # Git / GitHub LFS
        if "failed to fetch some objects" in t:
            if "/info/lfs" in t or "git lfs" in t:
                add("github-lfs")
            add("git-fetch")

        # Downloads (Rust/cargo, pip, curl, etc.)
        if re.search(r"\bcaused by:\s*failed to download\b|\bfailed to download\b|\bdownload error\b", t):
            add("download-error")

        # Build failures (Docker/buildkit/etc.)
        if re.search(r"\berror:\s*failed\s+to\s+build\b|\bfailed\s+to\s+solve\b", t):
            add("build-error")

        # CUDA / GPU toolchain
        if re.search(
            r"(?:"
            r"unsupported\s+cuda\s+version\s+for\s+vllm\s+installation"
            r"|\bcuda\b[^\n]{0,120}\bunsupported\b"
            r")",
            t,
        ):
            add("cuda-error")

        # HTTP timeouts / gateway errors (wget/curl/HTTP clients)
        if re.search(r"awaiting\s+response\.\.\.\s*(?:504|503|502)\b|gateway\s+time-?out|\bhttp\s+(?:504|503|502)\b", t):
            add("http-timeout")

        # Network connectivity
        if re.search(r"\bnetwork\s+error:\s*connection\s+failed\b|\bconnection\s+failed\.\s*check\s+network\s+connectivity\b|\bfirewall\s+settings\b", t):
            add("network-error")

        # Etcd / lease
        # Avoid tagging `etcd-error` just because the word "etcd" appears (it often shows up in benign logs).
        # Require a lease/status failure signature.
        if re.search(
            r"\bunable\s+to\s+create\s+lease\b|\bcheck\s+etcd\s+server\s+status\b|\betcd[^\n]{0,80}\blease\b|\blease\b[^\n]{0,80}\betcd\b",
            t,
        ):
            add("etcd-error")

        # Docker
        # Only tag "docker" when the log indicates a docker *infrastructure* failure (daemon/CLI),
        # not merely because the run executed `docker build/run` commands.
        #
        # IMPORTANT: BuildKit "failed to solve"/"ERROR: failed to build" are almost always *build* failures
        # (often due to CUDA/Python deps/etc) and should not be attributed to "docker".
        if re.search(
            r"(?:"
            r"cannot\s+connect\s+to\s+the\s+docker\s+daemon"
            # Don't tag docker for the common post-failure noise:
            #   "Error response from daemon: No such container: ..."
            r"|error\s+response\s+from\s+daemon:(?!.*no\s+such\s+container)"
            r"|\bdocker:\s+.*\berror\b"
            r")",
            t,
        ):
            add("docker")

        # Backend result JSON-ish blocks (vllm/sglang/trtllm): multi-line aware.
        engines = _backend_failure_engines_from_lines(lines[-4000:] if lines else [])
        if engines:
            add("backend-failure")
            for e in sorted(engines):
                add(f"{e}-error")

        # Broken links
        if re.search(r"\bbroken\s+links?\b|\bdead\s+links?\b", t):
            add("broken-links")

        # Timeout / infra flake
        if re.search(r"\b(time\s*out|timed\s*out|timeout)\b", t):
            add("timeout")

        # OOM / kill
        if re.search(r"\bout\s+of\s+memory\b|\boom\b|killed\s+process", t):
            add("oom")
    except Exception:
        return cats

    return cats


#
# HTML highlighting + snippet formatting
# =============================================================================
#

ERROR_HIGHLIGHT_RE: Pattern[str] = re.compile(
    r"(?:"
    r"\b(?:error|failed|failure|exception|traceback|fatal)\b"
    r"|\bno\s+module\s+named\b"
    r"|\b(?:time\s*out|timed\s*out)\b"
    # "timeout" can be either an error token OR a benign parameter like "--timeout 20" / "timeout=120s".
    # Treat as error only when it doesn't look like an option/assignment/explicit duration.
    r"|(?<!-)\btimeout\b(?!\s*[=:]\s*\d)(?!\s+\d+(?:\.\d+)?\s*(?:ms|s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)?\b)"
    r"|\b(?:gateway\s+time-?out)\b"
    r"|\b(?:http\s*)?(?:502|503|504)\b"
    r"|\b(?:network\s+error|connection\s+failed|check\s+network\s+connectivity|firewall\s+settings)\b"
    # Don't highlight bare "etcd"/"lease" (too many false positives). Highlight the actual error phrases.
    r"|\b(?:unable\s+to\s+create\s+lease|check\s+etcd\s+server\s+status)\b"
    r"|\b[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)\b"
    r"|\b(?:broken\s+links?|broken\s+link|dead\s+links?)\b"
    r")",
    re.IGNORECASE,
)

PYTEST_FAILED_LINE_RE: Pattern[str] = re.compile(
    r"(?:^|\s)FAILED(?:\s+|$).*::",  # e.g. "... FAILED tests/x.py::test_name"
    re.IGNORECASE,
)

# Docker daemon errors we want to surface explicitly in snippets.
DOCKER_DAEMON_ERROR_LINE_RE: Pattern[str] = re.compile(
    r"^.*\berror response from daemon:.*$",
    re.IGNORECASE,
)

DOCKER_NO_SUCH_CONTAINER_RE: Pattern[str] = re.compile(
    r"\berror response from daemon:\s*no\s+such\s+container\b",
    re.IGNORECASE,
)

# Network connectivity failures we want to surface explicitly in snippets.
NETWORK_ERROR_LINE_RE: Pattern[str] = re.compile(
    r"\bnetwork\s+error:\s*connection\s+failed\b|\bconnection\s+failed\.\s*check\s+network\s+connectivity\b",
    re.IGNORECASE,
)

# CUDA / vLLM install failures
UNSUPPORTED_CUDA_VLLM_RE: Pattern[str] = re.compile(
    r"unsupported\s+cuda\s+version\s+for\s+vllm\s+installation",
    re.IGNORECASE,
)

FAILED_TO_BUILD_RE: Pattern[str] = re.compile(
    r"\berror:\s*failed\s+to\s+build\b",
    re.IGNORECASE,
)

# Backend status JSON-ish summary lines (multi-line blocks).
BACKEND_RESULT_FAILURE_LINE_RE: Pattern[str] = re.compile(
    r"\"result\"\s*:\s*\"failure\"",
    re.IGNORECASE,
)

# Pytest block markers we want to preserve around failures.
PYTEST_FAILURES_HEADER_RE: Pattern[str] = re.compile(r"=+\s*FAILURES\s*=+", re.IGNORECASE)
PYTEST_PROGRESS_100_RE: Pattern[str] = re.compile(r"^.*\[100%\].*$")
# Example:
#   "_____ test_mocker_two_kv_router[file] _____"
PYTEST_UNDERSCORE_TITLE_RE: Pattern[str] = re.compile(r"_{5,}\s*test_[A-Za-z0-9_\[\]-]+\s*_{5,}", re.IGNORECASE)
PYTEST_TIMEOUT_E_LINE_RE: Pattern[str] = re.compile(
    r"\bE\s+Failed:\s+Timeout\b.*\bpytest-timeout\b",
    re.IGNORECASE,
)

# Lines where the user wants the *entire line* red (not just keyword highlighting).
FULL_LINE_ERROR_REDS_RE: List[Pattern[str]] = [
    # Git fetch failures (common infra issue); user wants the whole line red.
    re.compile(r"\berror:\s*failed\s+to\s+fetch\s+some\s+objects\s+from\b", re.IGNORECASE),
    # HTTP gateway timeouts (wget/curl/etc); user wants 504 Gateway Time-out red.
    re.compile(r"\b504\s+gateway\s+time-?out\b|\bgateway\s+time-?out\b", re.IGNORECASE),
    # Network connectivity failures.
    re.compile(r"\bnetwork\s+error:\s*connection\s+failed\b|\bconnection\s+failed\.\s*check\s+network\s+connectivity\b", re.IGNORECASE),
    # Etcd lease creation failures.
    re.compile(r"\bunable\s+to\s+create\s+lease\b|\bcheck\s+etcd\s+server\s+status\b", re.IGNORECASE),
    # Docker daemon errors.
    # Don't full-line-highlight the common post-failure noise:
    #   "Error response from daemon: No such container: ..."
    re.compile(r"\berror response from daemon:(?!.*no\s+such\s+container)", re.IGNORECASE),
    # Mirror sync infra errors (user wants the entire line red).
    re.compile(r"\bmirror sync failed or timed out\b", re.IGNORECASE),
    # CUDA / vLLM install errors.
    UNSUPPORTED_CUDA_VLLM_RE,
    # Python import errors are high-signal; make the entire line red.
    re.compile(r"\bModuleNotFoundError:\s*No\s+module\s+named\b", re.IGNORECASE),
    # CI sentinel variables indicating test failure. Example:
    #   FAILED_TESTS=1  # Treat missing XML as failure
    re.compile(r"\bFAILED_TESTS\s*=\s*1\b", re.IGNORECASE),
    # Pytest collection errors are high-signal and typically the true root cause.
    # Example:
    #   ________________ ERROR collecting tests/... _________________
    re.compile(r"\berror\s+collecting\b", re.IGNORECASE),
    # Multi-line backend result blocks: full-line highlight the failure field.
    re.compile(r"\"result\"\s*:\s*\"failure\"", re.IGNORECASE),
    # Pytest failure block lines (high-signal).
    PYTEST_FAILURES_HEADER_RE,
    PYTEST_UNDERSCORE_TITLE_RE,
    PYTEST_TIMEOUT_E_LINE_RE,
    # The 100% progress line that contains the failing "F" is useful context.
    re.compile(r"\[100%\].*F", re.IGNORECASE),
]


# Snippet category rules (HTML-side, to tag a snippet without re-reading the full raw log).
# A snippet can have multiple categories; keep these high-signal and relatively stable.
SNIPPET_CATEGORY_RULES: list[tuple[str, Pattern[str]]] = [
    ("pytest", re.compile(r"(?:^|\s)FAILED(?:\s+|$).*::|short test summary info|\\berror\\s+collecting\\b", re.IGNORECASE)),
    ("download-error", re.compile(r"caused by:\s*failed to download|failed to download|download error", re.IGNORECASE)),
    ("build-error", re.compile(r"\berror:\s*failed\s+to\s+build\b|\bfailed\s+to\s+solve\b", re.IGNORECASE)),
    ("cuda-error", re.compile(r"unsupported\s+cuda\s+version\s+for\s+vllm\s+installation|\bcuda\b[^\n]{0,120}\bunsupported\b", re.IGNORECASE)),
    (
        "http-timeout",
        re.compile(
            r"awaiting\s+response\.\.\.\s*(?:504|503|502)\b|gateway\s+time-?out|http\s+504\b|http\s+503\b|http\s+502\b",
            re.IGNORECASE,
        ),
    ),
    ("network-error", re.compile(r"network\s+error:\s*connection\s+failed|connection\s+failed\.\s*check\s+network\s+connectivity|firewall\s+settings", re.IGNORECASE)),
    ("etcd-error", re.compile(r"unable\s+to\s+create\s+lease|check\s+etcd\s+server\s+status|\betcd[^\n]{0,80}\blease\b|\blease\b[^\n]{0,80}\betcd\b", re.IGNORECASE)),
    ("git-fetch", re.compile(r"failed to fetch some objects from|RPC failed|early EOF|remote end hung up|fetch-pack", re.IGNORECASE)),
    ("github-api", re.compile(r"Failed to query GitHub API|secondary rate limit|API rate limit exceeded|HTTP 403|HTTP 429", re.IGNORECASE)),
    ("timeout", re.compile(r"\b(time\s*out|timed\s*out|timeout)\b", re.IGNORECASE)),
    ("oom", re.compile(r"\b(out of memory|CUDA out of memory|Killed process|oom)\b", re.IGNORECASE)),
    (
        "docker",
        re.compile(
            r"cannot\s+connect\s+to\s+the\s+docker\s+daemon"
            r"|error\s+response\s+from\s+daemon:(?!.*no\s+such\s+container)"
            r"|\bdocker:\s+.*\berror\b",
            re.IGNORECASE,
        ),
    ),
    ("k8s", re.compile(r"\bkubectl\b|\bkubernetes\b|CrashLoopBackOff|ImagePullBackOff|ErrImagePull", re.IGNORECASE)),
    ("python-exception", re.compile(r"Traceback \(most recent call last\)|\b(AssertionError|ValueError|TypeError)\b", re.IGNORECASE)),
    ("broken-links", re.compile(r"\bbroken\s+links?\b|\bdead\s+links?\b|\blychee\b", re.IGNORECASE)),
]


def highlight_error_keywords_html(text: str) -> str:
    """Escape HTML and highlight error keywords."""
    # Don't keyword-highlight this common post-failure docker noise.
    # It's useful to *show* in snippets sometimes, but shouldn't draw attention.
    if DOCKER_NO_SUCH_CONTAINER_RE.search(text or ""):
        return html.escape(text or "")

    escaped = html.escape(text or "")
    if not escaped:
        return ""

    def repl(m: re.Match) -> str:
        return f'<span style="color: #d73a49; font-weight: 700;">{m.group(0)}</span>'

    return ERROR_HIGHLIGHT_RE.sub(repl, escaped)


def snippet_categories(snippet_text: str) -> List[str]:
    """Return a stable list of categories for a snippet (best-effort)."""
    text = (snippet_text or "").strip()
    if not text:
        return []

    out: List[str] = []
    seen: set[str] = set()

    # Multi-line backend JSON-ish blocks: tag both engines when both blocks fail.
    try:
        engines = _backend_failure_engines_from_lines((snippet_text or "").splitlines())
        if engines:
            for name in (["backend-failure"] + [f"{e}-error" for e in sorted(engines)]):
                if name not in seen:
                    seen.add(name)
                    out.append(name)
    except Exception:
        pass

    for name, rx in SNIPPET_CATEGORY_RULES:
        try:
            if rx.search(text) and name not in seen:
                seen.add(name)
                out.append(name)
        except Exception:
            continue
    return out


def format_snippet_html(snippet_text: str) -> str:
    """Format an error snippet for HTML display.

    - Preserve line breaks (container uses `white-space: pre-wrap`).
    - For pytest "FAILED ...::test_..." summary lines, color the *entire line* red.
    - Otherwise, keep keyword-level highlighting for common failure tokens.
    """
    if not (snippet_text or "").strip():
        return ""

    out_lines: List[str] = []
    for raw_line in (snippet_text or "").splitlines():
        # Don't highlight the synthetic snippet header line(s).
        if raw_line.strip().lower().startswith("categories:") or raw_line.strip().lower().startswith("commands:"):
            out_lines.append(html.escape(raw_line))
            continue

        # Keep empty lines (they matter for readability) but don't highlight them.
        if raw_line == "":
            out_lines.append("")
            continue

        if PYTEST_FAILED_LINE_RE.search(raw_line) or any(r.search(raw_line) for r in FULL_LINE_ERROR_REDS_RE):
            out_lines.append(
                f'<span style="color: #d73a49; font-weight: 700;">{html.escape(raw_line)}</span>'
            )
        else:
            out_lines.append(highlight_error_keywords_html(raw_line))

    return "\n".join(out_lines)


#
# Snippet extraction (text-only; shared by dashboards)
# =============================================================================
#

def extract_error_snippet_from_text(
    text: str,
    *,
    context_before: int = 10,
    context_after: int = 5,
    max_lines: int = 80,
    max_chars: int = 5000,
) -> str:
    """Extract a short, high-signal error snippet from raw log text.

    Used by dashboards to populate the "▶ Snippet" toggle (HTML rendering happens elsewhere).
    """
    try:
        all_lines = (text or "").splitlines()
        if not all_lines:
            return ""

        def strip_prefix(line: str) -> str:
            # Drop common timestamp prefixes like: "2025-12-25T06:54:51.4973999Z "
            try:
                s = re.sub(r"^\d{4}-\d{2}-\d{2}T[0-9:.]+Z\s+", "", line)
                # Strip common ANSI color sequences.
                s = re.sub(r"\x1b\[[0-9;]*m", "", s)
                return s
            except Exception:
                return line

        def extract_commands(lines: List[str]) -> List[str]:
            """Best-effort extraction of interesting commands (pytest/docker/run.sh/build.sh)."""
            # Capture multi-line commands like:
            #   docker run ... \n  --flag ... \n  --flag ...
            start_res = [
                # Require whitespace/end after "pytest" so we don't match package/version strings like
                # "pytest==9.0.2" or "pytest-timeout==2.4.0".
                re.compile(r"(?:^|\\s)(?:python\\s+-m\\s+pytest|pytest)(?:\\s|$)", re.IGNORECASE),
                re.compile(r"\\bdocker\\s+(?:build|run)\\b", re.IGNORECASE),
                re.compile(r"(?:^|\\s)(?:\\./)?run\\.sh\\b", re.IGNORECASE),
                re.compile(r"(?:^|\\s)(?:\\./)?build\\.sh\\b", re.IGNORECASE),
            ]

            def normalize_cmd_line(raw: str) -> str:
                s = strip_prefix(raw).strip()
                if not s:
                    return ""
                # Common shell prefixes
                s = s.lstrip("+").strip()
                # GitHub Actions often wraps commands as "Run <cmd>"
                if s.startswith("##[group]Run "):
                    s = s.split("##[group]Run ", 1)[1].strip()
                elif s.startswith("Run "):
                    s = s.split("Run ", 1)[1].strip()
                return s

            cleaned = [normalize_cmd_line(x) for x in lines]

            def collect_continuation_block(start_idx: int) -> str:
                block: List[str] = []
                i = start_idx
                # Limit how far we read to avoid giant blocks.
                max_block_lines = 16
                while i < len(cleaned) and len(block) < max_block_lines:
                    ln = cleaned[i]
                    if not ln:
                        break
                    # Stop if we hit a new unrelated "Run ..." group marker
                    if i != start_idx and ln.startswith("##[group]"):
                        break
                    block.append(ln)
                    # Continue if this line ends with a backslash OR next line is an obvious continuation.
                    if ln.rstrip().endswith("\\\\"):
                        i += 1
                        continue
                    # If next line starts with common option indentation, treat it as continuation too.
                    if i + 1 < len(cleaned):
                        nxt = cleaned[i + 1].lstrip()
                        if (
                            nxt.startswith("--")
                            or nxt.startswith("-v ")
                            or nxt.startswith("-e ")
                            or nxt.startswith("-w ")
                            or nxt.startswith("-p ")
                            or nxt.startswith("--name")
                            or nxt.startswith("--network")
                        ):
                            i += 1
                            continue
                    break
                return "\\n".join(block).strip()

            # Scan from the end (more likely to catch the actual executed command).
            out: List[str] = []
            seen: set[str] = set()
            for i in range(len(cleaned) - 1, -1, -1):
                s = cleaned[i]
                if not s:
                    continue
                if not any(r.search(s) for r in start_res):
                    continue
                blk = collect_continuation_block(i)
                if not blk:
                    continue
                # Skip trivial fragments.
                low = blk.strip().lower()
                if low in {"pytest", "docker build", "docker run"}:
                    continue
                # Cap size per command
                if len(blk) > 1200:
                    blk = blk[:1200] + "\\n...(truncated)"
                if blk not in seen:
                    seen.add(blk)
                    out.append(blk)
                if len(out) >= 4:
                    break

            # We scanned from the end; display in chronological order.
            out.reverse()
            return out

        # Pick a single best anchor so we don't drown out the important line when there are many matches.
        # Priority:
        #   1) the *last* pytest "FAILED ...::..." line
        #   2) CUDA/vLLM "Unsupported CUDA version ..." (this is typically the real root cause)
        #   3) "ERROR: failed to build" (high-level build summary)
        #   4) backend status JSON block failure (`"result": "failure"`) — keeps the engine block visible
        #   5) the *last* docker daemon error ("Error response from daemon: ...")
        #   6) the *last* generic error line match
        anchor_idx: Optional[int] = None
        last_generic: Optional[int] = None
        last_pytest_failed: Optional[int] = None
        last_docker_daemon_err: Optional[int] = None
        last_cuda_err: Optional[int] = None
        last_failed_to_build: Optional[int] = None
        last_network_err: Optional[int] = None
        last_backend_result_failure: Optional[int] = None

        for i, line in enumerate(all_lines):
            if not line or not line.strip():
                continue
            if line.startswith("#"):
                continue
            if PYTEST_FAILED_LINE_RE.search(line):
                last_pytest_failed = i
            if DOCKER_DAEMON_ERROR_LINE_RE.search(line):
                last_docker_daemon_err = i
            if UNSUPPORTED_CUDA_VLLM_RE.search(line):
                last_cuda_err = i
            if FAILED_TO_BUILD_RE.search(line):
                last_failed_to_build = i
            if NETWORK_ERROR_LINE_RE.search(line):
                last_network_err = i
            if BACKEND_RESULT_FAILURE_LINE_RE.search(line):
                last_backend_result_failure = i
            if ERROR_SNIPPET_LINE_RE.search(line):
                last_generic = i

        anchor_idx = (
            last_pytest_failed
            if last_pytest_failed is not None
            else (
                last_cuda_err
                if last_cuda_err is not None
                else (
                    last_failed_to_build
                    if last_failed_to_build is not None
                    else (
                        last_backend_result_failure
                        if last_backend_result_failure is not None
                        else (last_docker_daemon_err if last_docker_daemon_err is not None else last_generic)
                    )
                )
            )
        )

        snippet_lines: List[str] = []
        if anchor_idx is not None:
            before = max(0, int(context_before))
            after = max(0, int(context_after))
            start = max(0, anchor_idx - before)
            end = min(len(all_lines), anchor_idx + after + 1)
            for k in range(start, end):
                line = all_lines[k]
                if line and line.strip() and not line.startswith("#"):
                    snippet_lines.append(line)
        else:
            # Fallback: last lines with signal.
            snippet_lines = [ln for ln in all_lines if ln and ln.strip() and not ln.startswith("#")][-40:]

        # If this is a pytest failure, ensure we include the core pytest failure block lines the user cares about.
        if last_pytest_failed is not None:
            # Pytest prints the FAILURES section noticeably earlier than the final "FAILED ...::..." line.
            # Use a larger lookback window to reliably capture the FAILURES header / test title / timeout line.
            w_start = max(0, last_pytest_failed - 1200)
            w_end = min(len(all_lines), last_pytest_failed + 80)
            window = all_lines[w_start:w_end]

            def add_last(rx: Pattern[str]) -> None:
                try:
                    last: Optional[str] = None
                    for ln in window:
                        if ln and ln.strip() and rx.search(ln):
                            last = ln
                    if last and last not in snippet_lines:
                        snippet_lines.append(last)
                except Exception:
                    return

            add_last(PYTEST_PROGRESS_100_RE)
            add_last(PYTEST_FAILURES_HEADER_RE)
            add_last(PYTEST_UNDERSCORE_TITLE_RE)
            # The explicit FAILED test id line is the anchor itself, but ensure it’s present.
            add_last(PYTEST_FAILED_LINE_RE)
            add_last(PYTEST_TIMEOUT_E_LINE_RE)

        # Ensure we include the last docker daemon error line if present (high-signal and easy to miss).
        if last_docker_daemon_err is not None:
            docker_line = all_lines[last_docker_daemon_err]
            if docker_line and docker_line.strip() and docker_line not in snippet_lines:
                snippet_lines.append(docker_line)

        # Ensure we include the backend failure line if present (so the engine failure block is visible).
        if last_backend_result_failure is not None:
            bf_line = all_lines[last_backend_result_failure]
            if bf_line and bf_line.strip() and bf_line not in snippet_lines:
                snippet_lines.append(bf_line)

        # Ensure we include the last network error line if present (high-signal infra failure).
        if last_network_err is not None:
            net_line = all_lines[last_network_err]
            if net_line and net_line.strip() and net_line not in snippet_lines:
                snippet_lines.append(net_line)

        # Ensure we include the CUDA/vLLM root-cause line if present (it often occurs before buildkit's final error).
        if last_cuda_err is not None:
            cuda_line = all_lines[last_cuda_err]
            if cuda_line and cuda_line.strip() and cuda_line not in snippet_lines:
                snippet_lines.append(cuda_line)

        # Ensure we include the "ERROR: failed to build" line if present (useful high-level build failure summary).
        if last_failed_to_build is not None:
            build_line = all_lines[last_failed_to_build]
            if build_line and build_line.strip() and build_line not in snippet_lines:
                snippet_lines.append(build_line)

        # Cap size and add explicit ellipsis markers when we cut off leading/trailing log content.
        #
        # The goal is to make it obvious when the snippet is a window into a larger log:
        # - If there are lines before the captured window (or we drop earlier lines due to max_lines),
        #   prepend a literal "..." line.
        # - If there are lines after the captured window, append a literal "..." line.
        #
        # Note: We intentionally cap from the tail to preserve the highest-signal failure lines.
        max_lines_i = max(1, int(max_lines))
        omitted_before_window = False
        omitted_after_window = False
        if anchor_idx is not None:
            omitted_before_window = bool(start > 0)  # type: ignore[name-defined]
            omitted_after_window = bool(end < len(all_lines))  # type: ignore[name-defined]

        # Reserve space for ellipsis lines if needed, so we never exceed max_lines.
        omitted_before = bool(omitted_before_window)
        omitted_after = bool(omitted_after_window)
        for _ in range(2):
            reserve = (1 if omitted_before else 0) + (1 if omitted_after else 0)
            content_cap = max(1, max_lines_i - reserve)
            omitted_before = bool(omitted_before_window) or (len(snippet_lines) > content_cap)

        reserve = (1 if omitted_before else 0) + (1 if omitted_after else 0)
        content_cap = max(1, max_lines_i - reserve)
        content = snippet_lines[-content_cap:]
        if omitted_before:
            content = ["..."] + content
        if omitted_after:
            content = content + ["..."]
        snippet_lines = content

        body = "\n".join(snippet_lines).strip()
        if not body:
            return ""

        cats = categorize_log_text(all_lines)
        cmds = extract_commands(all_lines)

        cats_line = ("Categories: " + ", ".join(cats)) if cats else ""
        cmds_block = ("Commands:\n" + "\n".join(f"  {c}" for c in cmds)) if cmds else ""

        def build_header(include_cmds: bool) -> str:
            parts: List[str] = []
            if cats_line:
                parts.append(cats_line)
            if include_cmds and cmds_block:
                parts.append(cmds_block)
            return ("\n\n".join(parts) + "\n\n") if parts else ""

        header = build_header(include_cmds=True)
        snippet = header + body

        # If too large, first drop the commands block (it can be very long), then keep the *tail* of the body
        # so we preserve the highest-signal failure lines (which are typically near the end).
        max_chars_i = int(max_chars)
        if len(snippet) > max_chars_i:
            header = build_header(include_cmds=False)
            snippet = header + body

        if len(snippet) > max_chars_i:
            trunc_marker = "\n\n...(truncated)"
            keep_body = max(0, max_chars_i - len(header) - len(trunc_marker))
            if keep_body <= 0:
                return body[-max_chars_i:]
            if len(body) > keep_body:
                body = body[-keep_body:]
                snippet = header + body + trunc_marker
            else:
                snippet = header + body

        return snippet
    except Exception:
        return ""


def _read_text_tail(path: Path, *, max_bytes: int) -> str:
    """Read the tail of a text file efficiently (best-effort)."""
    try:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return ""
        max_b = int(max_bytes or 0)
        if max_b <= 0:
            return p.read_text(encoding="utf-8", errors="replace")
        size = int(p.stat().st_size)
        with p.open("rb") as f:
            if size > max_b:
                f.seek(-max_b, os.SEEK_END)
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def extract_error_snippet_from_log_file(
    log_path: Path,
    *,
    tail_bytes: int = 512 * 1024,
    context_before: int = 10,
    context_after: int = 5,
    max_lines: int = 80,
    max_chars: int = 5000,
) -> str:
    """Extract an error snippet from a local raw log file (best-effort, tail-read)."""
    txt = _read_text_tail(Path(log_path), max_bytes=int(tail_bytes))
    return extract_error_snippet_from_text(
        txt,
        context_before=context_before,
        context_after=context_after,
        max_lines=max_lines,
        max_chars=max_chars,
    )


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract and format a high-signal error snippet from a CI log file.",
    )
    parser.add_argument("log_path", help="Path to a local raw log file (e.g., raw-log-text/<job_id>.log)")
    parser.add_argument("--tail-bytes", type=int, default=512 * 1024, help="Read only the last N bytes (default: 524288)")
    parser.add_argument("--no-tail", action="store_true", help="Read the entire file (disables --tail-bytes).")
    parser.add_argument("--context-before", type=int, default=10, help="Lines of context before anchor (default: 10)")
    parser.add_argument("--context-after", type=int, default=5, help="Lines of context after anchor (default: 5)")
    parser.add_argument("--max-lines", type=int, default=80, help="Max snippet lines (default: 80)")
    parser.add_argument("--max-chars", type=int, default=5000, help="Max snippet characters (default: 5000)")
    parser.add_argument(
        "--html",
        action="store_true",
        help="Print HTML-formatted snippet (no surrounding <pre>; just per-line HTML).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    log_path = Path(args.log_path).expanduser()
    if not log_path.exists():
        print(f"ERROR: file not found: {log_path}", file=sys.stderr)
        return 2
    if not log_path.is_file():
        print(f"ERROR: not a file: {log_path}", file=sys.stderr)
        return 2

    tail_bytes = 0 if args.no_tail else int(args.tail_bytes)
    snippet = extract_error_snippet_from_log_file(
        log_path,
        tail_bytes=tail_bytes,
        context_before=int(args.context_before),
        context_after=int(args.context_after),
        max_lines=int(args.max_lines),
        max_chars=int(args.max_chars),
    )

    if not (snippet or "").strip():
        print("(no snippet found)")
        return 0

    print(format_snippet_html(snippet) if args.html else snippet)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

