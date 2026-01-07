"""Snippet extraction for ci_log_errors (shared library)."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Pattern

# pyright: reportUndefinedVariable=false
# ^ This module intentionally shares a large set of regex/constants/helpers with `ci_log_errors.engine`
#   via `globals().update(engine.__dict__)` (see below). Static analyzers can’t easily follow that,
#   so we suppress undefined-variable noise here.

# To avoid import cycles, `engine.py` must not import this module at import time.

# Populate this module namespace with the shared helpers/constants from `engine.py`
# so the extracted code can remain unchanged.
from . import engine as _engine  # noqa: E402

globals().update(_engine.__dict__)

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

        def _format_suggested_cmd(cmd: str) -> str:
            """Render a suggested command as a commented line; copy strips markers in HTML renderer."""
            s = str(cmd or "").strip()
            if not s:
                return ""
            return f"# {s}   # suggested"

        def extract_commands(lines: List[str]) -> List[str]:
            """Best-effort extraction of interesting commands (pytest/docker/run.sh/build.sh)."""
            # Capture multi-line commands like:
            #   docker run ... \n  --flag ... \n  --flag ...
            start_res = [
                # Pytest commands: only treat these as execution context when they look like actual commands.
                # Avoid prose matches like "request: Pytest request fixture ...".
                re.compile(r"^(?:python\\s+-m\\s+pytest|pytest)\\b", re.IGNORECASE),
                # Shell-wrapped payloads (common: docker run ... bash/sh ... -c "<payload>").
                re.compile(r"\\b(?:/bin/)?(?:bash|sh)\\b.*\\bpytest\\b", re.IGNORECASE),
                # Prefer the explicit PYTEST_CMD line when present (highest-signal, most complete).
                PYTEST_CMD_LINE_RE,
                # Rust/cargo commands (common in CI).
                re.compile(r"^cargo\\s+(?:test|build|check|clippy|fmt|rustfmt)\\b", re.IGNORECASE),
                # Docker commands (often multi-line with backslashes).
                re.compile(r"^docker\\s+(?:buildx|build|run)\\b", re.IGNORECASE),
                re.compile(r"^(?:\\./)?run\\.sh\\b", re.IGNORECASE),
                re.compile(r"^(?:\\./)?build\\.sh\\b", re.IGNORECASE),
            ]

            def _extract_shell_c_payload(line: str) -> str:
                """Extract the quoted payload from shell `... -c "<payload>"` (bash/sh variants)."""
                try:
                    s = str(line or "").strip()
                    if not s:
                        return ""
                    # Handle:
                    # - bash -c "..."
                    # - /bin/bash -lc "..."
                    # - sh -c '...'
                    # - bash -eux -c "..."
                    for rx in (
                        re.compile(
                            r"\\b(?:/bin/)?(?:bash|sh)\\b.*?\\s+-c\\s+(['\"])(.+?)\\1\\s*$",
                            re.IGNORECASE,
                        ),
                        re.compile(
                            r"\\b(?:/bin/)?(?:bash|sh)\\b.*?\\s+-[A-Za-z]*c[A-Za-z]*\\s+(['\"])(.+?)\\1\\s*$",
                            re.IGNORECASE,
                        ),
                    ):
                        m = rx.search(s)
                        if m:
                            return str(m.group(2) or "")
                    return ""
                except Exception:
                    return ""

            def extract_vanilla_pytest_from_shell(line: str) -> str:
                """If line contains shell `-c "<payload>"` with pytest, extract the inner payload."""
                try:
                    s = str(line or "")
                    inner = _extract_shell_c_payload(s)
                    if not inner:
                        return ""
                    if not re.search(r"\\bpytest\\b", inner, flags=re.IGNORECASE):
                        return ""
                    return _unescape_nested_shell_quotes(inner).strip()
                except Exception:
                    return ""

            def normalize_cmd_line(raw: str) -> str:
                s = _strip_ts_and_ansi(raw).strip()
                if not s:
                    return ""
                # Common shell prefixes
                s = s.lstrip("+").strip()
                # GitHub Actions "command" wrappers.
                # Examples:
                #   [command]/usr/bin/docker buildx ...
                #   ##[command]/usr/bin/docker buildx ...
                s = re.sub(r"^(?:##\\[(?:command)\\]|\\[(?:command)\\])\\s*/usr/bin/", "", s, flags=re.IGNORECASE)
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
                # Skip truly trivial fragments (single token / no args).
                low = blk.strip().lower()
                if low in {"pytest", "cargo", "docker build", "docker run"}:
                    continue
                # Keep the *entire* multi-line invocation in the snippet (docker/pytest/cargo/etc).
                try:
                    parts = [x.rstrip() for x in blk.splitlines() if x.strip()]
                    if parts:
                        blk = "\n".join(parts).strip()
                except Exception:
                    pass
                # Cap length so commands don't drown out the actual failure lines.
                cap = 2200
                if len(blk) > cap:
                    blk = blk[:cap].rstrip() + "…"
                if blk not in seen:
                    seen.add(blk)
                    out.append(blk)
                    # Also include a suggested inner payload extracted from shell `-c "<payload>"`.
                    # This makes it easy to copy/paste without the `bash/sh ... -c` wrapper.
                    py = extract_vanilla_pytest_from_shell(blk)
                    py_sug = _format_suggested_cmd(py) if py else ""
                    if py_sug and py_sug not in seen:
                        seen.add(py_sug)
                        out.append(py_sug)
                if len(out) >= 4:
                    break

            # We scanned from the end; display in chronological order.
            out.reverse()
            return out

        # Pick a single best anchor so we don't drown out the important line when there are many matches.
        #
        # We choose the first available match from this priority list (highest-signal first):
        #   1) pytest "FAILED ...::..." (best locator)
        #   2) CUDA runner missing libcuda (ImportError: libcuda.so.1 ...) (often the real root cause)
        #   3) python import failure (ModuleNotFoundError: No module named ...) (often the real root cause)
        #   4) other Python exception lines (Traceback / AssertionError / etc)
        #   5) pytest file-level ERROR marker (points to the failing suite quickly)
        #   6) CUDA/vLLM "Unsupported CUDA version ..." (often the real root cause)
        #   7) Rust "test result: FAILED." (cargo test)
        #   8) Exit code 139 (SIGSEGV) (common "mystery" failure signature)
        #   9) Dockerfile context block header (BuildKit prints the snippet right after this)
        #  10) Git LFS dependency fetch failure block (often the root cause of build failures)
        #  11) "ERROR: failed to build" (high-level build summary)
        #  12) backend status JSON failure (`"result": "failure"`) (engine failure block)
        #  13) docker daemon error ("Error response from daemon: ...")
        #  14) last generic error line match
        anchor_idx: Optional[int] = None
        last_generic: Optional[int] = None
        last_pytest_failed: Optional[int] = None
        last_docker_daemon_err: Optional[int] = None
        last_cuda_err: Optional[int] = None
        last_failed_to_build: Optional[int] = None
        last_network_err: Optional[int] = None
        last_backend_result_failure: Optional[int] = None
        last_libcuda_import_err: Optional[int] = None
        last_pytest_error_file: Optional[int] = None
        last_pytest_cmd: Optional[int] = None
        pytest_cmd_idxs: List[int] = []
        last_pytest_exec_cmd: Optional[int] = None
        last_module_not_found: Optional[int] = None
        last_e_module_not_found: Optional[int] = None
        last_pytest_short_summary: Optional[int] = None
        last_python_exception_line: Optional[int] = None
        last_dockerfile_ctx_hdr: Optional[int] = None
        last_rust_test_result_failed: Optional[int] = None
        last_rust_failures_header: Optional[int] = None
        last_git_lfs_anchor: Optional[int] = None
        last_exit_code_139: Optional[int] = None
        etcd_sigs: List[int] = []
        last_hf_auth_sig: Optional[int] = None
        last_copyright_sig: Optional[int] = None
        last_docker_exec_cmd: Optional[int] = None
        last_cargo_exec_cmd: Optional[int] = None
        # broken-links: keep high-signal report blocks in the snippet (file + problematic symlink details)
        last_broken_links_file_hdr: Optional[int] = None
        last_broken_links_count: Optional[int] = None
        last_broken_link_error: Optional[int] = None
        last_problematic_symlink_error: Optional[int] = None
        # Docker/BuildKit: when a build fails, BuildKit often prints the exact failing shell payload:
        #   process "/bin/sh -c <payload...>" did not complete successfully
        # We surface this as a copyable command block in the snippet.
        last_buildkit_process_cmd: str = ""

        # Detect executed pytest invocations (not package/version lines like "pytest==8.4.2",
        # and not prose like "request: Pytest request fixture ...").
        PYTEST_EXEC_AT_START_RE: Pattern[str] = re.compile(
            r"^(?:python\s+-m\s+pytest|pytest)\b",
            re.IGNORECASE,
        )
        PYTEST_EXEC_IN_BASH_RE: Pattern[str] = re.compile(
            r"\b(?:/bin/)?(?:bash|sh)\b.*\bpytest\b",
            re.IGNORECASE,
        )

        for i, line in enumerate(all_lines):
            if not line or not line.strip():
                continue
            if line.startswith("#"):
                # Keep BuildKit step lines like "#48 ..." (they often contain the root-cause command).
                # Skip other "#" lines (usually shell comments / noise).
                if not re.match(r"^#\d+\b", line):
                    continue
            s_norm = _strip_ts_and_ansi(line)
            if PYTEST_FAILED_LINE_RE.search(s_norm):
                last_pytest_failed = i
            if SNIPPET_PYTEST_SHORT_TEST_SUMMARY_RE.search(s_norm):
                last_pytest_short_summary = i
            if PYTEST_ERROR_FILE_LINE_RE.search(s_norm):
                last_pytest_error_file = i
            if PYTEST_CMD_LINE_RE.search(s_norm):
                last_pytest_cmd = i
                pytest_cmd_idxs.append(i)
            # Also capture generic "pytest ..." invocations (e.g. docker run ... bash -c "pytest ...")
            # since many CI logs do not emit PYTEST_CMD=... but still include the executed command.
            if PYTEST_EXEC_AT_START_RE.search(s_norm) or PYTEST_EXEC_IN_BASH_RE.search(s_norm):
                last_pytest_exec_cmd = i
            if DOCKER_DAEMON_ERROR_LINE_RE.search(s_norm):
                last_docker_daemon_err = i
            # Capture actual execution commands for snippet context (user wants these shown).
            if re.search(r"\bdocker\s+(?:run|build)\b", s_norm, flags=re.IGNORECASE):
                last_docker_exec_cmd = i
            if re.search(r"\bcargo\s+(?:test|build|check|clippy|fmt|rustfmt)\b", s_norm, flags=re.IGNORECASE):
                last_cargo_exec_cmd = i
            if SNIPPET_UNSUPPORTED_CUDA_VLLM_RE.search(s_norm):
                last_cuda_err = i
            if SNIPPET_CUDA_LIBCUDA_IMPORT_ERROR_RE.search(s_norm):
                last_libcuda_import_err = i
            if SNIPPET_PYTHON_MODULE_NOT_FOUND_RE.search(s_norm):
                last_module_not_found = i
                # Prefer the pytest traceback-style line:
                #   E   ModuleNotFoundError: No module named 'sniffio'
                # This is often clearer than the short summary `ERROR tests/... - ModuleNotFoundError: ...`.
                try:
                    if re.search(r"^E\s+ModuleNotFoundError:\s*No\s+module\s+named\b", s_norm):
                        last_e_module_not_found = i
                except Exception:
                    pass
            if SNIPPET_PYTHON_EXCEPTION_LINE_RE.search(s_norm):
                last_python_exception_line = i
            if SNIPPET_DOCKERFILE_CONTEXT_HEADER_RE.search(s_norm):
                last_dockerfile_ctx_hdr = i
            if SNIPPET_RUST_TEST_FAILURES_HEADER_RE.search(_strip_ts_and_ansi(line)):
                last_rust_failures_header = i
            if SNIPPET_RUST_TEST_RESULT_FAILED_RE.search(s_norm):
                last_rust_test_result_failed = i
            if SNIPPET_GIT_LFS_SNIPPET_ANCHOR_RE.search(s_norm):
                last_git_lfs_anchor = i
            if SNIPPET_EXIT_CODE_139_LINE_RE.search(s_norm):
                last_exit_code_139 = i
            # Some categories are often only visible as a single high-signal line that can get
            # pushed out of the snippet window. Track them explicitly so we can force-include.
            if CAT_ETCD_ERROR_RE.search(line.lower()):
                etcd_sigs.append(i)
            if CAT_HUGGINGFACE_AUTH_ERROR_RE.search(_strip_ts_and_ansi(line)) and not _line_is_warn_or_lower(line):
                last_hf_auth_sig = i
            if CAT_COPYRIGHT_HEADER_ERROR_RE.search(line):
                last_copyright_sig = i
            if SNIPPET_FAILED_TO_BUILD_RE.search(s_norm):
                last_failed_to_build = i

            # Capture the exact failing BuildKit payload for copy/paste.
            # Examples:
            #   #48 ERROR: process "/bin/sh -c cp ... && ..." did not complete successfully
            #   ERROR: failed to solve: process "/bin/sh -c ..." did not complete successfully
            if not last_buildkit_process_cmd:
                try:
                    m = re.search(r'process\s+"?/bin/sh\s+-c\s+([^"]+)"', s_norm, flags=re.IGNORECASE)
                    if m:
                        cmd = str(m.group(1) or "").strip()
                        if cmd:
                            last_buildkit_process_cmd = _unescape_nested_shell_quotes(cmd)
                except Exception:
                    pass
            if NETWORK_ERROR_LINE_RE.search(s_norm):
                last_network_err = i
            if SNIPPET_BACKEND_RESULT_FAILURE_LINE_RE.search(s_norm):
                last_backend_result_failure = i
            if ERROR_SNIPPET_LINE_RE.search(s_norm):
                last_generic = i

            # broken-links / symlink-check report markers
            try:
                st = (s_norm or "").strip()
                if st.startswith("📄 File:"):
                    last_broken_links_file_hdr = i
                if re.search(r"\b\d+\s+broken\s+link\(s\)\s+found\b", st, flags=re.IGNORECASE):
                    last_broken_links_count = i
                if st.startswith("##[error]Broken link:"):
                    last_broken_link_error = i
                if st.startswith("##[error]Problematic symlink:"):
                    last_problematic_symlink_error = i
            except Exception:
                pass

        # Choose anchor by priority (first non-None index wins).
        for idx in (
            # broken-links: anchor on the detailed report lines (broken link / problematic symlink).
            last_problematic_symlink_error,
            last_broken_link_error,
            last_pytest_failed,
            last_libcuda_import_err,
            last_module_not_found,
            last_python_exception_line,
            last_pytest_error_file,
            last_cuda_err,
            last_rust_test_result_failed,
            last_exit_code_139,
            last_dockerfile_ctx_hdr,
            last_git_lfs_anchor,
            last_failed_to_build,
            last_backend_result_failure,
            last_docker_daemon_err,
            last_generic,
        ):
            if idx is not None:
                anchor_idx = idx
                break

        # For logs where docker runs `bash -c "${PYTEST_CMD}"`, we want the PYTEST_CMD definition
        # that was visible *before the failure* (not a later env dump).
        closest_pytest_cmd_before_anchor: Optional[int] = None
        try:
            if anchor_idx is not None and pytest_cmd_idxs:
                best = None
                for j in pytest_cmd_idxs:
                    if j is None:
                        continue
                    if int(j) <= int(anchor_idx):
                        best = int(j)
                closest_pytest_cmd_before_anchor = best
        except Exception:
            closest_pytest_cmd_before_anchor = None

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
            # If everything in the anchor window is a GitHub Actions marker (##[...]) or similar,
            # we still want *something* in the snippet.
            if not snippet_lines:
                for k in range(start, end):
                    line = all_lines[k]
                    if line and line.strip():
                        snippet_lines.append(line)
        else:
            # Fallback: last lines with signal.
            snippet_lines = [ln for ln in all_lines if ln and ln.strip() and not ln.startswith("#")][-40:]
            if not snippet_lines:
                # As a last resort, include the last non-empty lines even if they start with "#".
                snippet_lines = [ln for ln in all_lines if ln and ln.strip()][-40:]

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
                        if not (ln and ln.strip()):
                            continue
                        # Match on a normalized view so ANSI color escapes don't break patterns like
                        # `E Failed: Timeout (...) from pytest-timeout.`
                        s = _strip_ts_and_ansi(ln)
                        if s and rx.search(s):
                            last = s
                    if last and last not in snippet_lines:
                        snippet_lines.append(last)
                except Exception:
                    return

            add_last(SNIPPET_PYTEST_PROGRESS_100_RE)
            add_last(SNIPPET_PYTEST_FAILURES_HEADER_RE)
            add_last(SNIPPET_PYTEST_UNDERSCORE_TITLE_RE)
            # The explicit FAILED test id line is the anchor itself, but ensure it’s present.
            add_last(PYTEST_FAILED_LINE_RE)
            add_last(SNIPPET_PYTEST_TIMEOUT_E_LINE_RE)

        # Ensure we include the last docker daemon error line if present (high-signal and easy to miss).
        if last_docker_daemon_err is not None:
            docker_line = all_lines[last_docker_daemon_err]
            if docker_line and docker_line.strip() and docker_line not in snippet_lines:
                snippet_lines.append(docker_line)

        # Ensure we include the last PYTEST_CMD line if present (requested; critical for debugging).
        if last_pytest_cmd is not None:
            cmd_line = all_lines[last_pytest_cmd]
            if cmd_line and cmd_line.strip() and cmd_line not in snippet_lines:
                snippet_lines.append(cmd_line)
        # If we saw a generic "pytest ..." execution line, include it too (often more reliable than PYTEST_CMD).
        if last_pytest_exec_cmd is not None:
            cmd_line2 = all_lines[last_pytest_exec_cmd]
            if cmd_line2 and cmd_line2.strip() and cmd_line2 not in snippet_lines:
                snippet_lines.append(cmd_line2)

        # Ensure we include the libcuda ImportError line if present (often the true root cause).
        if last_libcuda_import_err is not None:
            lib_line = all_lines[last_libcuda_import_err]
            if lib_line and lib_line.strip() and lib_line not in snippet_lines:
                snippet_lines.append(lib_line)

        # Ensure we include the pytest ERROR file line if present (high-signal locator).
        if last_pytest_error_file is not None:
            err_file_line = all_lines[last_pytest_error_file]
            if err_file_line and err_file_line.strip() and err_file_line not in snippet_lines:
                snippet_lines.append(err_file_line)

        # Ensure we include the actual `E   ModuleNotFoundError: No module named '...'` line when present.
        # This often appears in pytest traceback output and can be truncated away by summary lines like:
        #   ERROR tests/... - ModuleNotFoundError: No module named 'sn...'
        if last_e_module_not_found is not None:
            try:
                mn_line = _strip_ts_and_ansi(all_lines[int(last_e_module_not_found)]).rstrip("\n")
                if mn_line and mn_line.strip() and mn_line not in snippet_lines:
                    snippet_lines.append(mn_line)
            except Exception:
                pass

        # If we anchored on a python ModuleNotFoundError inside pytest, ensure we include the
        # "short test summary info" header (it often provides useful context like skipped tests).
        if last_module_not_found is not None and last_pytest_short_summary is not None:
            hdr = all_lines[last_pytest_short_summary]
            if hdr and hdr.strip() and hdr not in snippet_lines:
                snippet_lines.append(hdr)

        # If we anchored on a Python exception line in a pytest log, include the short summary header
        # if it exists (it’s often a compact “what happened” index).
        if (
            last_python_exception_line is not None
            and last_pytest_short_summary is not None
            and last_pytest_short_summary <= last_python_exception_line
        ):
            hdr = all_lines[last_pytest_short_summary]
            if hdr and hdr.strip() and hdr not in snippet_lines:
                snippet_lines.append(hdr)

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

        # CI tooling: files not covered by CI filters.
        #
        # Make the key line stand out in HTML by:
        #  - ensuring it’s present in the snippet, and
        #  - inserting a synthetic marker that we can full-line-highlight red.
        #
        # Example log: 59652435193.log
        try:
            ci_filter_phrase_re = re.compile(r"\bnot\s+covered\s+by\s+any\s+ci\s+filter\b", re.IGNORECASE)
            marker = "[CI_FILTER_UNCOVERED]"
            insert_at: Optional[int] = None
            for i, ln in enumerate(list(snippet_lines or [])):
                s = _strip_ts_and_ansi(str(ln or ""))
                if s and ci_filter_phrase_re.search(s):
                    insert_at = int(i)
                    break
            if insert_at is not None and marker not in snippet_lines:
                snippet_lines.insert(insert_at, marker)
        except Exception:
            pass

        # Git LFS dependency fetch failures: include the whole uv/pip/LFS error block when present.
        #
        # These failures often occur *before* BuildKit prints its generic "ERROR: failed to build" summary,
        # so they can be easy to miss if we anchor too late.
        if last_git_lfs_anchor is not None:
            # Attempt to scope to the BuildKit step number ("#52") if present.
            step_id: Optional[str] = None
            try:
                m = re.search(r"\s#(\d+)\b", all_lines[last_git_lfs_anchor])
                if m:
                    step_id = str(m.group(1))
            except Exception:
                step_id = None

            def same_step(ln: str) -> bool:
                if not step_id:
                    return True
                try:
                    return bool(re.search(rf"\s#{re.escape(step_id)}\b", ln))
                except Exception:
                    return True

            # Walk backward to find a good block start.
            start_i = last_git_lfs_anchor
            for k in range(last_git_lfs_anchor, max(-1, last_git_lfs_anchor - 300), -1):
                ln = all_lines[k]
                if not (ln or "").strip():
                    continue
                if ln.startswith("#"):
                    continue
                if not same_step(ln):
                    continue
                if SNIPPET_GIT_LFS_BLOCK_START_RE.search(ln):
                    start_i = k
                    break
                # If we see the uv command line for this step, treat that as a good start too.
                if "UV_GIT_LFS=1" in ln and "uv pip install" in ln:
                    start_i = k
                    # Keep searching for an even better start marker, but don’t go beyond this step.
                    # (We don’t break here on purpose.)

            # Walk forward to include the whole block (cap to avoid huge snippets).
            end_i = min(len(all_lines) - 1, last_git_lfs_anchor + 140)
            for k in range(max(start_i, last_git_lfs_anchor), min(len(all_lines), start_i + 220)):
                ln = all_lines[k]
                if not (ln or "").strip():
                    continue
                if ln.startswith("#"):
                    continue
                # Stop once we leave the step and we've already captured some of the block.
                if not same_step(ln) and k > last_git_lfs_anchor:
                    end_i = k - 1
                    break
                if SNIPPET_GIT_LFS_BLOCK_END_RE.search(_strip_ts_and_ansi(ln)):
                    end_i = k
                    break

            for k in range(start_i, end_i + 1):
                ln = all_lines[k]
                if not (ln or "").strip():
                    continue
                if ln.startswith("#"):
                    continue
                if ln not in snippet_lines:
                    snippet_lines.append(ln)

        # Docker build context blocks: if BuildKit printed a Dockerfile snippet, include it.
        # These look like:
        #   Dockerfile.foo:190
        #   --------------------
        #    189 | ...
        #    190 | >>> RUN ...
        if last_dockerfile_ctx_hdr is not None:
            hdr_i = last_dockerfile_ctx_hdr
            # Include a small forward window: header + divider + numbered lines.
            for k in range(hdr_i, min(len(all_lines), hdr_i + 40)):
                ln = all_lines[k]
                if not ln or not ln.strip():
                    continue
                if ln.startswith("#"):
                    continue
                if (
                    k == hdr_i
                    or SNIPPET_DOCKERFILE_CONTEXT_DIVIDER_RE.search(ln)
                    or SNIPPET_DOCKERFILE_CONTEXT_LINE_RE.search(ln)
                ):
                    if ln not in snippet_lines:
                        snippet_lines.append(ln)
                    continue
                # Stop once we leave the Dockerfile block.
                if (
                    k > hdr_i
                    and not SNIPPET_DOCKERFILE_CONTEXT_LINE_RE.search(ln)
                    and not SNIPPET_DOCKERFILE_CONTEXT_DIVIDER_RE.search(ln)
                ):
                    break

        # Ensure we include representative lines for some “single-line” failure categories.
        #
        # These are common in CI gate checks and linters, and without forcing them in, the snippet
        # can accidentally miss the line even though the full log clearly indicates the failure.
        # For etcd, include *all* matched etcd error lines (these are typically few, and seeing
        # the whole set is useful for debugging).
        try:
            for idx in (etcd_sigs or []):
                if idx is None:
                    continue
                ln = _strip_ts_and_ansi(all_lines[idx])
                if ln and ln.strip() and ln not in snippet_lines:
                    snippet_lines.append(ln)
        except Exception:
            pass
        for idx in (last_hf_auth_sig, last_copyright_sig):
            if idx is None:
                continue
            ln = _strip_ts_and_ansi(all_lines[idx])
            if ln and ln.strip() and ln not in snippet_lines:
                snippet_lines.append(ln)

        # Rust test failures: include the `failures:` block (failed test names) if present.
        if last_rust_failures_header is not None:
            hdr_i = last_rust_failures_header
            # Include header + subsequent indented test names until blank line (or a small cap).
            for k in range(hdr_i, min(len(all_lines), hdr_i + 25)):
                ln = all_lines[k]
                if k == hdr_i:
                    if ln not in snippet_lines:
                        snippet_lines.append(ln)
                    continue
                # Stop on first blank line after header.
                if not (ln or "").strip():
                    break
                s = _strip_ts_and_ansi(ln)
                if RUST_TEST_FAILED_TEST_NAME_RE.search(s):
                    if ln not in snippet_lines:
                        snippet_lines.append(ln)
                    continue
                # Stop if we leave the simple failures list.
                break

        # Ensure we include the exit code 139 line if present (it’s often the only useful clue).
        #
        # IMPORTANT: do this *late*, right before capping, so tail-capping doesn't accidentally drop it
        # when we add other helpful blocks (Dockerfile/LFS/etc).
        if last_exit_code_139 is not None:
            ln = all_lines[last_exit_code_139]
            if ln and ln.strip() and ln not in snippet_lines:
                snippet_lines.append(ln)

        # broken-links: force-include the high-signal report blocks users need to fix the issue.
        # Without this, the snippet can degrade into the verbose script footer ("what to do next")
        # and omit the actual broken link / suspicious symlink details.
        try:
            cats_for_snip = categorize_error_log_lines(all_lines[-4000:] if all_lines else [])
            if "broken-links" in cats_for_snip:
                def _add_line_idx(idx: Optional[int]) -> None:
                    if idx is None:
                        return
                    ln0 = all_lines[int(idx)]
                    if ln0 and ln0.strip() and ln0 not in snippet_lines:
                        snippet_lines.append(ln0)

                _add_line_idx(last_broken_links_file_hdr)
                _add_line_idx(last_broken_links_count)

                if last_problematic_symlink_error is not None:
                    start_i = int(last_problematic_symlink_error)
                    end_i = min(len(all_lines), start_i + 10)
                    for k in range(start_i, end_i):
                        ln = all_lines[k]
                        s = _strip_ts_and_ansi(ln or "").strip()
                        if not s:
                            if k > start_i:
                                break
                            continue
                        if ln not in snippet_lines:
                            snippet_lines.append(ln)
        except Exception:
            pass

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

        # Include .github/actions/* and .github/workflows/* context lines right before the failure.
        # These are often the clearest "what was running" breadcrumbs when debugging CI.
        try:
            gh_path_re: Pattern[str] = re.compile(r"\.github/(?:actions|workflows)/[A-Za-z0-9_.\-/]+", re.IGNORECASE)
            anchor_for_ctx = int(anchor_idx) if anchor_idx is not None else (len(all_lines) - 1)
            ctx_start = max(0, anchor_for_ctx - 6000)
            ctx_lines: List[str] = []
            seen_ctx: set[str] = set()
            for ln in all_lines[ctx_start : anchor_for_ctx + 1]:
                s = _strip_ts_and_ansi(ln or "").strip()
                if not s:
                    continue
                if not gh_path_re.search(s):
                    continue
                if s not in seen_ctx:
                    seen_ctx.add(s)
                    ctx_lines.append(s)
            ctx_lines = ctx_lines[-6:]
            if ctx_lines:
                snippet_lines = ctx_lines + ["..."] + list(snippet_lines or [])
        except Exception:
            pass

        # Prepend "execution context" commands into the snippet body, separated by ellipses.
        # These are the best breadcrumbs for debugging (docker run/build, cargo, pytest).
        #
        # We intentionally extract these from GitHub Actions "##[group]Run ..." blocks so we can
        # preserve *multi-line* invocations (docker run/build + wrapped bash -c "pytest ...").
        #
        # IMPORTANT (strong UX invariant):
        # For pytest/rust failures, only show the **single closest** command that immediately precedes
        # the failure block. Many CI jobs run multiple test phases (unit + e2e + parallel + etc).
        # Showing ALL commands is noisy and frequently misleading; users want "what ran right before
        # this error" (plus derived `bash -c "..."` and the plain `pytest ...`/`cargo ...`).
        try:
            cmd_blocks: List[str] = []
            rerun_only_failed_pytest_cmd: str = ""
            try:
                cleaned = [_strip_ts_and_ansi(x).rstrip("\n") for x in (all_lines or [])]
                anchor_for_cmds = int(anchor_idx) if anchor_idx is not None else (len(cleaned) - 1)
                want_single_closest_execution_cmd = bool(
                    last_pytest_failed is not None
                    or last_rust_test_result_failed is not None
                    or last_broken_link_error is not None
                    or last_problematic_symlink_error is not None
                )

                # Also capture docker buildx commands that appear as Actions "[command]" lines,
                # e.g. "[command]/usr/bin/docker buildx create ...". These are often critical
                # for debugging docker-build failures.
                try:
                    buildx_re = re.compile(
                        # Actions often uses /usr/bin/docker, but some runners use /usr/local/bin/docker.
                        r"^(?:##\[(?:command)\]|\[(?:command)\])\s*/usr/(?:local/)?bin/docker\s+buildx\b",
                        re.IGNORECASE,
                    )
                    buildx_lines: List[str] = []
                    for ln in cleaned[: anchor_for_cmds + 1]:
                        s = (ln or "").strip()
                        if not s:
                            continue
                        if not buildx_re.search(s):
                            continue
                        # Normalize: strip "[command]/usr/(local/)?bin/" or "##[command]/usr/(local/)?bin/" prefix.
                        s2 = re.sub(
                            r"^(?:##\[(?:command)\]|\[(?:command)\])\s*/usr/(?:local/)?bin/",
                            "",
                            s,
                            flags=re.IGNORECASE,
                        )
                        if s2 and s2 not in buildx_lines:
                            buildx_lines.append(s2)
                    # Keep only the last few to avoid drowning the snippet.
                    buildx_lines = buildx_lines[-6:]
                    # For pytest/rust failures, keep the prelude laser-focused; don't add unrelated buildx noise.
                    if buildx_lines and not want_single_closest_execution_cmd:
                        cmd_blocks.append("\n".join(buildx_lines))
                except Exception:
                    pass

                # Also surface the BuildKit failing "/bin/sh -c ..." payload if we captured it.
                # This is often the most actionable "command preceding the error" for docker-build failures.
                try:
                    if last_buildkit_process_cmd and last_buildkit_process_cmd not in cmd_blocks:
                        cmd_blocks.append(last_buildkit_process_cmd)
                except Exception:
                    pass

                # Find the last few "Run ..." command groups *before the anchor* and keep the most relevant ones.
                # Many workflows use "Run # <comment>" headers, so we don't try to match the header text.
                #
                # IMPORTANT: iterate backward from the anchor. If we scan from EOF, we can accidentally
                # pick post-failure bookkeeping groups (artifact upload / docker cp / etc) and miss the
                # *actual* execution command that preceded the failures.
                idxs: List[int] = []
                scan_start = min(len(cleaned) - 1, max(0, anchor_for_cmds))
                for idx in range(scan_start, -1, -1):
                    s0 = (cleaned[idx] or "").strip()
                    if not s0.startswith("##[group]Run "):
                        continue
                    idxs.append(idx)
                    if len(idxs) >= 24:
                        break
                idxs.reverse()

                def _is_pytest_module_dump(ln: str) -> bool:
                    return bool(re.search(r"^\s*pytest\s*=\s*<module\s+'pytest'\b", ln or "", flags=re.IGNORECASE))

                want_cmd_start_re = re.compile(
                    r"^(?:"
                    r"docker\s+(?:run|buildx|build)\b"
                    r"|cargo\s+(?:test|build|check|clippy|fmt|rustfmt)\b"
                    r"|python\s+-m\s+pytest\b"
                    r"|pytest\b"
                    r"|bash\s+-c\s+['\"][^'\"]*\bpytest\b"
                    r"|python3\s+\.github/workflows/detect_broken_links\.py\b"
                    r")",
                    re.IGNORECASE,
                )

                # For pytest/rust failures, only keep the single closest execution command prelude.
                idx_iter = list(reversed(idxs)) if want_single_closest_execution_cmd else list(idxs)
                for idx in idx_iter:
                    # Grab the group body until endgroup/shell/env; then extract the meaningful command lines.
                    raw_block: List[str] = []
                    j = idx
                    while j < len(cleaned) and len(raw_block) < 80:
                        s = (cleaned[j] or "").rstrip()
                        if not s:
                            # GitHub Actions groups often include blank lines; skip them.
                            j += 1
                            continue
                        if s.startswith("##[endgroup]"):
                            break
                        if s.startswith("shell:") or s.startswith("env:"):
                            break
                        if s.startswith("##[group]Run "):
                            # Keep the header text as a hint, but we'll filter it out unless it looks like a command.
                            s = s.split("##[group]Run ", 1)[1].strip()
                        raw_block.append(s)
                        j += 1

                    # Find the first real command line in the block.
                    start_k: Optional[int] = None
                    for k, ln in enumerate(raw_block):
                        s = (ln or "").strip()
                        if not s or s.startswith("#"):
                            continue
                        if _is_pytest_module_dump(s):
                            continue
                        if want_cmd_start_re.search(s):
                            start_k = k
                            break
                        # docker run lines in some logs are indented but still start with docker once stripped
                        if want_cmd_start_re.search(s.lstrip()):
                            start_k = k
                            raw_block[k] = s.lstrip()
                            break
                    if start_k is None:
                        continue

                    # For some workflows, the "Run ..." group contains a short, helpful preamble right
                    # before the command (e.g. a comment explaining what it's doing, and `set +e` to
                    # tolerate failures). By default we skip comments to keep snippets tight, but for
                    # certain commands (notably detect_broken_links.py) users want to see these lines.
                    #
                    # Example (from broken-links workflow):
                    #   # Run the broken links detection script and capture exit code
                    #   set +e  # Don't exit immediately on error
                    #   python3 .github/workflows/detect_broken_links.py \
                    pre_k = start_k
                    try:
                        s_cmd = (raw_block[start_k] or "").strip()
                        if re.search(r"\bpython3\s+\.github/workflows/detect_broken_links\.py\b", s_cmd, flags=re.IGNORECASE):
                            # Include up to 3 preamble lines if they are comments or `set +/-e`.
                            for _ in range(3):
                                if pre_k <= 0:
                                    break
                                prev = (raw_block[pre_k - 1] or "").strip()
                                if not prev:
                                    break
                                if prev.startswith("#") or prev.startswith("set +e") or prev.startswith("set -e"):
                                    pre_k -= 1
                                    continue
                                break
                    except Exception:
                        pre_k = start_k

                    # Collect the command block including continuation lines until a clear stop marker.
                    block: List[str] = []
                    k = pre_k
                    while k < len(raw_block) and len(block) < 40:
                        s = (raw_block[k] or "").rstrip()
                        if not s:
                            break
                        if s.startswith("shell:") or s.startswith("env:") or s.startswith("TEST_EXIT_CODE="):
                            break
                        if _is_pytest_module_dump(s):
                            break
                        # Stop once we hit post-run bookkeeping that isn't part of the command itself.
                        if s.startswith("echo ") or s.startswith("exit "):
                            break
                        block.append(s)
                        # Preamble lines (comments / shell safety) should not terminate the block.
                        # We want to show them immediately before the actual command they describe.
                        try:
                            st = s.strip()
                            if st.startswith("#") or st.startswith("set +e") or st.startswith("set -e"):
                                k += 1
                                continue
                        except Exception:
                            pass
                        # If line ends with "\" keep consuming obvious continuations.
                        if s.rstrip().endswith("\\"):
                            k += 1
                            continue
                        nxt = (raw_block[k + 1] if (k + 1) < len(raw_block) else "") or ""
                        nxt_s = nxt.lstrip()
                        if (
                            nxt_s.startswith("--")
                            or nxt_s.startswith("-")
                            or re.match(r"^(?:/bin/)?(?:bash|sh)\b", nxt_s, flags=re.IGNORECASE)
                        ):
                            k += 1
                            raw_block[k] = nxt_s
                            continue
                        break

                    # Drop consecutive dupes.
                    dedup: List[str] = []
                    for ln in block:
                        if dedup and dedup[-1] == ln:
                            continue
                        dedup.append(ln)
                    blk = "\n".join([x for x in dedup if x.strip()]).strip()
                    if blk:
                        cmd_blocks.append(blk)
                        if want_single_closest_execution_cmd:
                            break
            except Exception:
                cmd_blocks = []

            # De-dup blocks
            try:
                uniq: List[str] = []
                seenb: set[str] = set()
                for blk in cmd_blocks:
                    s = (blk or "").strip()
                    if s and s not in seenb:
                        seenb.add(s)
                        uniq.append(s)
                cmd_blocks = uniq
            except Exception:
                pass

            # If the closest execution command references ${PYTEST_CMD}, inject the resolved PYTEST_CMD="pytest ..."
            # *before* the failure block so users can see the real pytest invocation even when docker wraps it.
            try:
                if cmd_blocks:
                    joined = "\n".join(cmd_blocks[-2:])  # cheap check (usually 1 block in focused mode)
                    if "${PYTEST_CMD}" in joined or "PYTEST_CMD" in joined:
                        # Prefer a PYTEST_CMD definition before the failure anchor, but if the only
                        # available PYTEST_CMD appears later (e.g. in an `env:` dump), still surface it
                        # at the top of the snippet. Users care about the real command line even if CI
                        # prints it after the error.
                        pick_i = (
                            int(closest_pytest_cmd_before_anchor)
                            if closest_pytest_cmd_before_anchor is not None
                            else (int(last_pytest_cmd) if last_pytest_cmd is not None else None)
                        )
                        if pick_i is not None:
                            ln0 = _strip_ts_and_ansi(all_lines[int(pick_i)]).strip()
                            if ln0:
                                # Add the assignment line as its own copyable cmd block.
                                if ln0 not in cmd_blocks:
                                    cmd_blocks.insert(0, ln0)
                                # Also add the plain pytest command extracted from the assignment value.
                                m = re.search(r"\bPYTEST_CMD\s*=\s*(['\"])(.+)\1", ln0)
                                if m:
                                    inner = str(m.group(2) or "").strip()
                                    if inner and inner not in cmd_blocks:
                                        cmd_blocks.insert(1, inner)
            except Exception:
                pass

            # Also add a "vanilla pytest ..." block when the command is wrapped in `bash -c "... pytest ..."`.
            # This is especially common inside `docker run ... bash -c "<prep> && pytest ..."` and makes copy/paste easy.
            try:
                def _extract_failed_pytest_nodeids() -> List[str]:
                    """Extract FAILED nodeids like `tests/x.py::test_name[param]` near the failure anchor."""
                    try:
                        if last_pytest_failed is None:
                            return []
                        center = int(last_pytest_failed)
                        w0 = max(0, center - 2500)
                        w1 = min(len(all_lines), center + 250)
                        nodeids: List[str] = []
                        seen: set[str] = set()
                        rx = re.compile(r"\bFAILED\s+([^\s]+::[^\s]+)", re.IGNORECASE)
                        for ln in all_lines[w0:w1]:
                            s = _strip_ts_and_ansi(ln or "")
                            m = rx.search(s)
                            if not m:
                                continue
                            nid = str(m.group(1) or "").strip()
                            if nid and nid not in seen:
                                seen.add(nid)
                                nodeids.append(nid)
                        return nodeids
                    except Exception:
                        return []

                def _extract_shell_c_payload_from_cmd_block(blk: str) -> str:
                    """Extract shell `-c "<payload>"` payload from a command block (bash/sh variants)."""
                    s = (blk or "").strip()
                    if not s:
                        return ""
                    for ln in s.splitlines():
                        ln_s = (ln or "").strip()
                        if not ln_s:
                            continue
                        m = None
                        try:
                            for rx in (
                                re.compile(
                                    r"\b(?:/bin/)?(?:bash|sh)\b.*?\s+-c\s+(['\"])(.+?)\1\s*$",
                                    re.IGNORECASE,
                                ),
                                re.compile(
                                    r"\b(?:/bin/)?(?:bash|sh)\b.*?\s+-[A-Za-z]*c[A-Za-z]*\s+(['\"])(.+?)\1\s*$",
                                    re.IGNORECASE,
                                ),
                            ):
                                m = rx.search(ln_s)
                                if m:
                                    break
                        except Exception:
                            m = None
                        if not m:
                            continue
                        inner = str(m.group(2) or "")
                        if not inner:
                            continue
                        return _unescape_nested_shell_quotes(inner).strip()
                    return ""

                def _extract_vanilla_pytest_from_cmd_block(blk: str) -> str:
                    s = (blk or "").strip()
                    if not s:
                        return ""
                    # Look for a shell `-c "<payload>"` payload on any line.
                    for ln in s.splitlines():
                        ln_s = (ln or "").strip()
                        if not ln_s:
                            continue
                        inner = ""
                        try:
                            for rx in (
                                re.compile(
                                    r"\b(?:/bin/)?(?:bash|sh)\b.*?\s+-c\s+(['\"])(.+?)\1\s*$",
                                    re.IGNORECASE,
                                ),
                                re.compile(
                                    r"\b(?:/bin/)?(?:bash|sh)\b.*?\s+-[A-Za-z]*c[A-Za-z]*\s+(['\"])(.+?)\1\s*$",
                                    re.IGNORECASE,
                                ),
                            ):
                                m = rx.search(ln_s)
                                if m:
                                    inner = str(m.group(2) or "")
                                    break
                        except Exception:
                            inner = ""
                        if not inner:
                            continue
                        last = None
                        for mm in re.finditer(r"(?:^|&&\s*|;\s*|\|\|\s*)(pytest\b.*)$", inner, flags=re.IGNORECASE):
                            last = mm
                        if not last:
                            continue
                        cmd = str(last.group(1) or "").strip()
                        if cmd.lower().startswith("pytest"):
                            return _unescape_nested_shell_quotes(cmd)
                    return ""

                expanded: List[str] = []
                seen_exp: set[str] = set()
                # Track a canonical plain `pytest ...` command so we can synthesize
                # `pytest ... <failed_nodeids...>` for quick reruns.
                best_vanilla_pytest: str = ""
                for blk in (cmd_blocks or []):
                    b = (blk or "").strip()
                    if not b:
                        continue
                    if b not in seen_exp:
                        seen_exp.add(b)
                        expanded.append(b)
                    # If we already have a plain `pytest ...` command block (common when we expand
                    # a `PYTEST_CMD="pytest ..."` variable), use it as the canonical rerun base.
                    #
                    # This fixes a gap where `bash -c "${PYTEST_CMD}"` contains no inline `pytest ...`
                    # payload, so `_extract_vanilla_pytest_from_cmd_block()` can't recover it.
                    if not best_vanilla_pytest:
                        try:
                            # Use the first line only; rerun command must be a single line.
                            first = (b.splitlines()[0] if b.splitlines() else "").strip()
                            # Prefer the actual underlying `pytest ...` for `PYTEST_CMD="pytest ..."` style logs.
                            m_cmd = re.match(r"^PYTEST_CMD=(['\"])(.*)\1\s*$", first, flags=re.IGNORECASE)
                            if m_cmd:
                                inner = str(m_cmd.group(2) or "").strip()
                                if inner.lower().startswith("pytest"):
                                    best_vanilla_pytest = _unescape_nested_shell_quotes(inner)
                            elif first.lower().startswith("pytest"):
                                best_vanilla_pytest = _unescape_nested_shell_quotes(first)
                        except Exception:
                            pass
                    # If executed via `bash/sh ... -c "<payload>"`, suggest the inner payload (without shell wrapper).
                    inner_payload = _extract_shell_c_payload_from_cmd_block(b)
                    inner_sug = _format_suggested_cmd(inner_payload) if inner_payload else ""
                    if inner_sug and inner_sug not in seen_exp:
                        seen_exp.add(inner_sug)
                        expanded.append(inner_sug)
                    py = _extract_vanilla_pytest_from_cmd_block(b)
                    # Even if we already emitted the suggested `# pytest ...` line elsewhere,
                    # still capture the plain pytest command as the canonical rerun base.
                    if py and not best_vanilla_pytest:
                        best_vanilla_pytest = py
                    py_sug = _format_suggested_cmd(py) if py else ""
                    if py_sug and py_sug not in seen_exp:
                        seen_exp.add(py_sug)
                        expanded.append(py_sug)

                # If we saw failed nodeids and we have a plain pytest command, synthesize a command
                # that reruns ONLY those failing tests.
                try:
                    failed_nodeids = _extract_failed_pytest_nodeids()
                    if best_vanilla_pytest and failed_nodeids:
                        # Keep the list bounded; huge failure sets become unreadable.
                        failed_nodeids = failed_nodeids[:25]
                        # IMPORTANT: nodeids can contain shell-glob characters like `[...]` (parametrization).
                        # In bash, unquoted `[`/`]` can trigger pathname expansion. Quote each nodeid so
                        # copy/paste is safe and works reliably.
                        rerun = (
                            best_vanilla_pytest.rstrip()
                            + " "
                            + " ".join(shlex.quote(x) for x in failed_nodeids if x)
                        ).strip()
                        # IMPORTANT UX: place the suggested rerun command *after* the contiguous FAILED
                        # lines chunk in the snippet body, not in the command prelude.
                        rerun_only_failed_pytest_cmd = rerun
                except Exception:
                    pass
                cmd_blocks = expanded
            except Exception:
                pass

            # Keep only the last few extracted command blocks so the snippet stays readable.
            # (We only need enough context to see "what ran" before the error.)
            try:
                cmd_blocks = (cmd_blocks or [])[-10:]
            except Exception:
                pass

            cmd_lines: List[str] = []
            for blk in (cmd_blocks or []):
                s = (blk or "").strip("\n")
                if not s.strip():
                    continue
                # Mark command blocks so HTML renderer can add copy buttons + multiline formatting.
                cmd_lines.append("[[CMD]]")
                cmd_lines.extend(s.splitlines())
                cmd_lines.append("[[/CMD]]")

            if cmd_lines:
                tail = list(snippet_lines or [])

                # If we synthesized a "rerun only failed tests" pytest command, insert it immediately
                # BEFORE the FAILED-chunk *summary* line when available, e.g.:
                #   "============= 4 failed, 27 passed, 1 skipped in ... =============="
                # Otherwise, fall back to inserting right after the last contiguous `FAILED ...` chunk.
                try:
                    if rerun_only_failed_pytest_cmd and tail:
                        summary_re = re.compile(r"^=+\s*\d+\s+failed\b.*=+\s*$", re.IGNORECASE)

                        # Find contiguous FAILED chunks and prefer the last one that is followed by a
                        # pytest summary line (e.g. "==== 4 failed, 27 passed ... ====").
                        best_insert_at: Optional[int] = None
                        last_failed_chunk_end: Optional[int] = None
                        i0 = 0
                        while i0 < len(tail):
                            s0 = _strip_ts_and_ansi(tail[i0] or "").strip()
                            if not s0.startswith("FAILED "):
                                i0 += 1
                                continue
                            # Consume the chunk.
                            j0 = i0
                            while j0 < len(tail):
                                sj0 = _strip_ts_and_ansi(tail[j0] or "").strip()
                                if not sj0.startswith("FAILED "):
                                    break
                                j0 += 1
                            chunk_end = j0 - 1
                            last_failed_chunk_end = chunk_end

                            # Look ahead for the pytest summary line; if found, place BEFORE it.
                            look_end = min(len(tail), j0 + 35)
                            for k in range(j0, look_end):
                                sk = _strip_ts_and_ansi(tail[k] or "").strip()
                                if summary_re.search(sk):
                                    best_insert_at = k
                                    break
                            i0 = j0

                        insert_at: Optional[int] = None
                        if best_insert_at is not None:
                            insert_at = int(best_insert_at)
                        elif last_failed_chunk_end is not None:
                            insert_at = int(last_failed_chunk_end) + 1

                        if insert_at is not None:
                            # Avoid inserting duplicates if the rerun cmd is already present.
                            if rerun_only_failed_pytest_cmd not in "\n".join(tail):
                                tail[insert_at:insert_at] = [
                                    "[[CMD]]",
                                    _format_suggested_cmd(rerun_only_failed_pytest_cmd) or rerun_only_failed_pytest_cmd,
                                    "[[/CMD]]",
                                ]
                except Exception:
                    pass

                # Use a single ellipsis to separate the command prelude from the failure context,
                # but do NOT insert ellipses between adjacent command blocks.
                if tail:
                    if tail[0].strip() != "...":
                        cmd_lines.append("...")
                combined = cmd_lines + tail
                if len(combined) > max_lines_i:
                    keep_tail = max(0, max_lines_i - len(cmd_lines))
                    combined = cmd_lines + (tail[-keep_tail:] if keep_tail > 0 else [])
                snippet_lines = combined
        except Exception:
            pass

        # Normalize body lines: never show noisy GitHub Actions timestamp prefixes or ANSI codes.
        try:
            normalized_lines: List[str] = []
            for ln in (snippet_lines or []):
                if ln == "...":
                    normalized_lines.append(ln)
                    continue
                # Handle multi-line entries (e.g., docker run blocks) by normalizing each line.
                try:
                    if "\n" in (ln or ""):
                        normalized_lines.append(
                            "\n".join(_strip_ts_and_ansi(x) for x in (ln or "").splitlines())
                        )
                    else:
                        normalized_lines.append(_strip_ts_and_ansi(ln))
                except Exception:
                    normalized_lines.append(_strip_ts_and_ansi(ln))
            snippet_lines = normalized_lines
        except Exception:
            pass

        body = "\n".join(snippet_lines).strip()
        if not body:
            return ""

        cats = categorize_error_log_lines(all_lines)
        cats_line = ("Categories: " + ", ".join(cats)) if cats else ""
        header = (cats_line + "\n") if cats_line else ""
        snippet = header + body

        # If too large, preserve BOTH:
        # - the beginning of the body (often contains docker/cargo/pytest invocations)
        # - the tail of the body (often contains the actual failure)
        #
        # This matches the desired UX:
        #   docker run ...
        #   ...
        #   PYTEST_CMD=...
        #   ...
        #   <failure block>
        max_chars_i = int(max_chars)
        if len(snippet) > max_chars_i:
            trunc_marker = "\n...\n"
            keep_body = max(0, max_chars_i - len(header) - len(trunc_marker))
            if keep_body <= 0:
                return (header + body)[-max_chars_i:]
            if len(body) > keep_body:
                # Keep a small head to retain command context, and the remaining budget for tail.
                head_keep = min(len(body), 900)
                tail_keep = max(0, keep_body - head_keep)
                if tail_keep > 0 and (head_keep + tail_keep + len(trunc_marker)) <= keep_body + len(trunc_marker):
                    body = body[:head_keep].rstrip() + trunc_marker + body[-tail_keep:].lstrip()
                    snippet = header + body
                else:
                    # Fallback: tail-only.
                    body = body[-keep_body:]
                    snippet = header + body
            else:
                snippet = header + body

        return snippet
    except Exception:
        # Debugging aid: snippet extraction is intentionally best-effort and swallows many errors to
        # avoid breaking dashboards. During refactors, enable this to surface the root cause:
        #   CI_LOG_ERRORS_DEBUG=1 python3 -m ci_log_errors <log>
        if os.getenv("CI_LOG_ERRORS_DEBUG"):
            import traceback

            traceback.print_exc()
        return ""


def _read_text_tail(path: Path, *, max_bytes: int) -> str:
    from . import engine as _E
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
    """Extract an error snippet from a local raw log file (best-effort, tail-read).

    Important: For very large CI logs, the *command that ran* can appear earlier than the default tail window.
    For pytest/rust failures, missing that command makes the snippet significantly less useful, so we do a
    best-effort "second pass" with a larger tail window when the first pass yields no command blocks.
    """
    p = Path(log_path)
    tb = int(tail_bytes)
    txt = _read_text_tail(p, max_bytes=int(tb))
    snip = extract_error_snippet_from_text(
        txt,
        context_before=context_before,
        context_after=context_after,
        max_lines=max_lines,
        max_chars=max_chars,
    )

    # If we didn't find command blocks for a pytest/rust failure, retry with a larger tail window.
    try:
        if tb > 0 and "[[CMD]]" not in (snip or ""):
            cats = categorize_error_log_lines((txt or "").splitlines())
            if ("pytest-error" in cats) or ("rust-error" in cats):
                # 8 MiB is usually enough to include the "Run docker run ... pytest/cargo ..." block
                # while still keeping reads bounded.
                tb2 = max(tb, 8 * 1024 * 1024)
                txt2 = _read_text_tail(p, max_bytes=int(tb2))
                snip2 = extract_error_snippet_from_text(
                    txt2,
                    context_before=context_before,
                    context_after=context_after,
                    max_lines=max_lines,
                    max_chars=max_chars,
                )
                if "[[CMD]]" in (snip2 or ""):
                    return snip2
    except Exception:
        pass

    return snip


def _audit_snippet_commands(*, logs_root: Path, tail_bytes: int) -> int:
    from . import engine as _E
    """
    Scan all *.log under logs_root and report pytest-error / rust-error logs whose
    extracted snippet does NOT contain a preceding command (pytest/cargo) in the command prelude.
    """
    logs_root = Path(logs_root)
    if not logs_root.exists() or not logs_root.is_dir():
        print(f"ERROR: --logs-root is not a directory: {logs_root}", file=sys.stderr)
        return 2

    files = sorted(logs_root.glob("*.log"))
    if not files:
        print(f"(no logs found under {logs_root})")
        return 0

    want_pytest = re.compile(r"\bpytest\b|python\s+-m\s+pytest\b|PYTEST_CMD\s*=", re.IGNORECASE)
    want_cargo = re.compile(r"^\s*cargo\s+(?:test|build|check|clippy|fmt|rustfmt)\b", re.IGNORECASE | re.MULTILINE)
    fail_line = re.compile(
        r"(?:^|\b)(?:FAILED\b|=+ FAILURES =+|test result:\s*FAILED\.|##\[error\]Process completed with exit code)",
        re.IGNORECASE,
    )

    total = 0
    relevant = 0
    missing: list[tuple[str, str]] = []  # (filename, reason)

    for p in files:
        total += 1
        try:
            txt = _read_text_tail(p, max_bytes=int(tail_bytes))
            lines = (txt or "").splitlines()
            cats = categorize_error_log_lines(lines)
        except Exception:
            continue

        is_pytest = "pytest-error" in cats
        is_rust = "rust-error" in cats
        if not (is_pytest or is_rust):
            continue
        relevant += 1

        snip = extract_error_snippet_from_log_file(p, tail_bytes=int(tail_bytes), max_lines=120)
        if not snip:
            missing.append((p.name, "empty snippet"))
            continue

        # Extract command blocks (the snippet prelude uses [[CMD]] blocks).
        cmd_blocks: list[str] = []
        try:
            cur: list[str] = []
            in_cmd = False
            for ln in snip.splitlines():
                if ln.strip() == "[[CMD]]":
                    in_cmd = True
                    cur = []
                    continue
                if ln.strip() == "[[/CMD]]":
                    if in_cmd:
                        cmd_blocks.append("\n".join(cur).strip())
                    in_cmd = False
                    cur = []
                    continue
                if in_cmd:
                    cur.append(ln)
        except Exception:
            cmd_blocks = []

        prelude = "\n".join([b for b in cmd_blocks if b]).strip()

        # Determine whether we have a "preceding" command:
        # we require it to appear before the first failure marker in the snippet.
        try:
            first_fail_idx = None
            for i, ln in enumerate(snip.splitlines()):
                if fail_line.search(ln or ""):
                    first_fail_idx = i
                    break
            pre_fail_text = "\n".join(snip.splitlines()[:first_fail_idx]) if first_fail_idx is not None else snip
        except Exception:
            pre_fail_text = snip

        ok = True
        if is_pytest:
            if not cmd_blocks:
                ok = False
                missing.append((p.name, "pytest-error: no [[CMD]] blocks"))
            elif not want_pytest.search(prelude) and not want_pytest.search(pre_fail_text):
                ok = False
                missing.append((p.name, "pytest-error: no pytest command in prelude"))

        if ok and is_rust:
            if not cmd_blocks:
                ok = False
                missing.append((p.name, "rust-error: no [[CMD]] blocks"))
            elif not want_cargo.search(prelude) and not want_cargo.search(pre_fail_text):
                ok = False
                missing.append((p.name, "rust-error: no cargo command in prelude"))

    print(f"audit: logs_root={logs_root}")
    print(f"audit: total_logs={total} relevant(pytest/rust)={relevant} missing={len(missing)}")
    if missing:
        print("audit: missing command prelude (first 200):")
        for name, reason in missing[:200]:
            print(f"- {name}: {reason}")
        if len(missing) > 200:
            print(f"... and {len(missing) - 200} more")
        return 1
    return 0
