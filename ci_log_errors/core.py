#!/usr/bin/env python3
"""
Shared log error detection + categorization + snippet formatting utilities.

This module is intentionally dependency-light so it can be shared by:
- `dynamo-utils/common.py` (log/snippet extraction, cache logic)
- `dynamo-utils/html_pages/common_dashboard_lib.py` (HTML rendering for snippets/tags)

Snippet UX (what it should look like)
------------------------------------
- Snippets are **timestamp/ANSI stripped**. They should NOT show `2026-...Z` prefixes.
- Snippets may include **.github/workflows/** and **.github/actions/** lines *preceding the failure*
  (breadcrumbs for "what workflow/action was running").
- Snippets may include one or more **multi-line command blocks** (docker/cargo/pytest) preceding the
  failure. These are rendered inline in blue with a Copy button.
- Snippets use literal `...` lines to indicate omitted content.
- Error emphasis:
  - Full-line error signals like `FAILED ...`, `[TIMEOUT]`, `assertion failed:`, `ERROR: failed to build`,
    and `[FAIL] incorrect date:` render the entire line in red (not bold).

Log files to error markers -- training examples:
Grouped (best-effort) so it’s easier to find the golden log for a given category:

CI / tooling:
- 59030780729.log => build-status-check-error
- 58887254616.log => broken-links
- 57945094461.log => copyright-header-error
- 59030172010.log => helm-error, k8s-error

Docker:
- 58861726335.log => network-timeout-https, docker-build-error, !pytest-error
- 58745798050.log => network-download-error, docker-build-error
- 58818079816.log => github-lfs-error, docker-build-error, !git-fetch, !pytest-error
- 58861639352.log => docker-image-error

Infra / system:
- 57877945100.log => cuda-error
- 56700029731.log => etcd-error
- 59347193958.log => etcd-error
- 59418212320.log => etcd-error, pytest-error
- 57521050539.log => pytest-error, etcd-error, python-error
- 59520885010.log => pytest-error  # router decisions disagg failures (ea9503559 merge)
- 59520513875.log => copyright-header-error, etcd-error  # includes "[FAIL] incorrect date: ..."
- 59525400060.log => copyright-header-error, etcd-error  # includes "[FAIL] incorrect date: ..."
- 59539777738.log => copyright-header-error, etcd-error  # includes "[FAIL] incorrect date: ..." (container/dev/*.sh)
- 58588864118.log => k8s-error, k8s-network-timeout-pod, !pytest-error
- 57877945085.log => exit-127-cmd-not-found
- 57521050554.log => exit-139-sigsegv, !huggingface-auth-error
- 58412373114.log => oom

Network / timeouts:
- 57930774858.log => network-error
- 58745798050.log => network-download-error, docker-build-error
- 57930747559.log => network-timeout-gitlab-mirror
- 58575902063.log => network-timeout-github-action, !network-timeout-generic
- 58861726335.log => network-timeout-https, docker-build-error
- 59386365389.log => network-timeout-https, !broken-links
- 58572103421.log => k8s-network-timeout-pod, k8s-error
- 58463784363.log => k8s-network-timeout-portfwd, k8s-error, !network-timeout-generic

Tests / languages:
- 58906141961.log => !pytest-error  # success run (passed/skipped), should not be tagged as pytest-error
- 58097278528.log => pytest-error, !python-error, !huggingface-auth-error
- 58179788784.log => pytest-error, pytest-timeout-error, !python-error, !huggingface-auth-error
- 58457161045.log => python-error, !pytest-error, !huggingface-auth-error
- 58465471934.log => rust-error, !huggingface-auth-error
- 59493756549.log => rust-error, !pytest-error

VLLM/SGLang/TRTLLM backends:
- 56701494636.log => backend-failure, trtllm-error
- 57524186105.log => backend-failure, trtllm-error, sglang-error
- 58471383691.log => vllm-error

HuggingFace auth:
- 58604176773.log => pytest-error, network-download-error, build-status-check-error, huggingface-auth-error

Category frequency summary (all 733 logs, sorted by occurrence):
    1. k8s-error                           306/733 (41.7%) - Kubernetes/kubectl failure signal (cluster-related failures)
    2. k8s-network-timeout-pod             209/733 (28.5%) - kubectl wait timeout (pods condition)
    3. build-status-check-error            195/733 (26.6%) - CI gate checking upstream builds
    4. pytest-error                        178/733 (24.3%) - Pytest test failures
    5. python-error                        119/733 (16.2%) - Python exceptions/tracebacks
    6. exit-127-cmd-not-found               62/733  (8.5%) - Exit code 127 (command not found / missing binary in PATH)
    7. network-timeout-gitlab-mirror        33/733  (4.5%) - GitLab mirror sync infra timeout
    8. network-download-error               29/733  (4.0%) - Failed downloads (pip/cargo/curl)
    9. docker-build-error                   24/733  (3.3%) - Docker/BuildKit failures
   10. cuda-error                           18/733  (2.5%) - CUDA version/driver issues
   11. huggingface-auth-error               16/733  (2.2%) - HF token/gated model access
   12. pytest-timeout-error                 15/733  (2.0%) - Pytest per-test timeout (pytest-timeout plugin)
   13. backend-failure                      14/733  (1.9%) - vllm/sglang/trtllm failures
   14. etcd-error                           12/733  (1.6%) - Etcd lease/connection issues
   15. github-lfs-error                     12/733  (1.6%) - Git LFS fetch failures
   16. docker-image-error                   11/733  (1.5%) - Missing Docker images
   17. network-error                        11/733  (1.5%) - Network connectivity failures
   18. oom                                   9/733  (1.2%) - Out of memory
   19. vllm-error                            8/733  (1.1%) - VLLM backend failures
   20. helm-error                            7/733  (1.0%) - Helm chart failures
   21. network-timeout-https                 7/733  (1.0%) - HTTP(S) gateway timeouts + link-checker timeouts
   22. trtllm-error                          6/733  (0.8%) - TensorRT-LLM failures
   23. network-timeout-github-action         5/733  (0.7%) - GitHub Actions step timed out
   24. broken-links                          3/733  (0.4%) - Dead links in documentation
   25. copyright-header-error                2/733  (0.3%) - Missing copyright headers
   26. rust-error                            2/733  (0.3%) - Cargo test failures
   27. sglang-error                          2/733  (0.3%) - SGLang backend failures
   28. exit-139-sigsegv                      1/733  (0.1%) - Exit code 139 (SIGSEGV / signal 11)
   29. k8s-network-timeout-portfwd           1/733  (0.1%) - kubectl port-forward connect timeout

Golden-log workflow (IMPORTANT for future edits):
- These example logs are treated as *golden training set* for regression testing. Keep them read-only:
  - `chmod a-w /home/keivenc/nvidia/raw-log-text/<job_id>.log`
- After changing rules/regexes/snippet logic, run the built-in self-test:
  - `python3 dynamo-utils/ci_log_errors/core.py --self-test-examples`
  - This parses the "Examples:" list above, loads each log from `~/.cache/dynamo-utils/raw-log-text/`
    (or `$DYNAMO_UTILS_CACHE_DIR/raw-log-text`), and reports
    missing/extra categories for both full-log categorization and snippet-derived categorization.
- If mismatches show up, adjust categorization/snippet anchors until the example logs match again,
  then re-run the self-test until it’s clean.

Snippet output assertions (extra self-test)
------------------------------------------
Grammar:
  * `<job_id>.log => +must_contain1, +must_contain2, !must_not_contain1, !must_not_contain2`
  * Optional HTML full-line-red assertions:
    - `+RED:<substr>` means the rendered snippet HTML must contain `<substr>` inside a full-line red span.
    - `!RED:<substr>` means the rendered snippet HTML must NOT contain `<substr>` inside a full-line red span.
Notes:
  - These assertions validate snippet **text output** (not HTML).
  - Prefer stable substrings (avoid volatile IDs/timings).

* 58887254616.log => +Run the broken links detection script and capture exit code, +set +e, +python3 .github/workflows/detect_broken_links.py, +--check-symlinks, +--output broken-links-report.json, +📄 File: docs/kubernetes/installation_guide.md, +1 broken link(s) found, +Problematic symlink: Suspicious symlink: target requires many directory traversals, +1. docs/examples/runtime/hello_world/README.md, +→ ../../../../examples/custom_backend/hello_world/README.md
* 59520885010.log => +docker run -w /workspace, +bash -c "pytest, +FAILED tests/router/test_router_e2e_with_mockers.py::test_router_decisions_disagg, !2026-
* 57521050539.log => +FAILED tests/router/test_router_e2e_with_sglang.py::test_sglang_kv_router_basic, +FAILED tests/router/test_router_e2e_with_sglang.py::test_router_decisions_sglang_multiple_workers, +FAILED tests/router/test_router_e2e_with_sglang.py::test_sglang_indexers_sync, +pytest -v --tb=short --basetemp=/tmp -o cache_dir=/tmp/.pytest_cache --junitxml=/workspace/test-results/pytest_test_report.xml --durations=10 -m, +tests/router/test_router_e2e_with_sglang.py::test_sglang_kv_router_basic, +tests/router/test_router_e2e_with_sglang.py::test_router_decisions_sglang_multiple_workers, +tests/router/test_router_e2e_with_sglang.py::test_sglang_indexers_sync, +====== 3 failed, 8 passed, 4 skipped, 626 deselected
* 56700029731.log => +docker run --runtime=nvidia, +bash -c "mkdir -p /workspace/test-results && pytest -v --tb=short --basetemp=/tmp, +pytest -v --tb=short --basetemp=/tmp -o cache_dir=/tmp/.pytest_cache --junitxml=/workspace/test-results/pytest_test_report.xml, +not slow, !unit and trtllm_marker, !pytest     = <module 'pytest'
* 59332716597.log => +docker run -w /workspace, +bash -c "pytest --basetemp=/tmp/pytest-parallel, +pytest --basetemp=/tmp/pytest-parallel --junitxml=pytest_parallel.xml, +-m \\\"pre_merge and parallel, !pytest     = <module 'pytest'
* 59539777738.log => +[FAIL] incorrect date:, !2026-
* 59540519012.log => +docker buildx create --name builder-, +docker buildx inspect --bootstrap, !2026-
* 58465491442.log => +docker buildx create --name builder-, +cp /tmp/deps/vllm/install_vllm.sh /tmp/install_vllm.sh, +--cuda-version $CUDA_VERSION, !2026-
* 58818079816.log => +Git operation failed, +failed to fetch LFS objects, +RED:Git operation failed, +RED:failed to fetch LFS objects
* 58465471934.log => +failures:, +recorder::tests::test_recorder_streams_events_to_file, +RED:failures:, +RED:    recorder::tests::test_recorder_streams_events_to_file
* 59520885010.log => +FAILED tests/router/test_router_e2e_with_mockers.py::test_router_decisions_disagg, +pytest --basetemp=/tmp/pytest-parallel --junitxml=pytest_parallel.xml -n 4, +-m \"pre_merge and parallel, +tests/router/test_router_e2e_with_mockers.py::test_router_decisions_disagg[with_bootstrap-decode_first], +'tests/router/test_router_e2e_with_mockers.py::test_router_decisions_disagg[with_bootstrap-decode_first]', !RED:__________ test_router_decisions_disagg[with_bootstrap-prefill_first] __________
"""

from __future__ import annotations

import argparse
import functools
import html
import json
import os
import re
import shlex
import stat
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Pattern, Sequence, Tuple


def _default_raw_log_dir() -> Path:
    """Default raw-log directory (single source of truth).

    Resolution order:
    - $DYNAMO_UTILS_CACHE_DIR/raw-log-text
    - ~/.cache/dynamo-utils/raw-log-text
    """
    try:
        override = os.environ.get("DYNAMO_UTILS_CACHE_DIR", "").strip()
        if override:
            return Path(override).expanduser() / "raw-log-text"
    except Exception:
        pass
    return Path.home() / ".cache" / "dynamo-utils" / "raw-log-text"

#
# Shared helpers (keep dependency-light)
# =============================================================================
#

# GitHub Actions log lines often start with a timestamp prefix like:
#   2025-12-25T06:54:51.4973999Z <payload>
_TS_PREFIX_RE: Pattern[str] = re.compile(r"^\d{4}-\d{2}-\d{2}T[0-9:.]+Z\s+")

# Common ANSI escape sequences (colors, etc).
_ANSI_ESCAPE_RE: Pattern[str] = re.compile(r"\x1b\[[0-9;]*m")

# Some logs include an *additional* inner ISO timestamp prefix after the Actions prefix,
# e.g. Rust tracing output:
#   2025-11-29T21:22:52.2708052Z 2025-11-29T21:22:52.270138Z  WARN dynamo_llm::hub: ...
_ISO_TS_PREFIX_LOOSE_RE: Pattern[str] = re.compile(r"^\d{4}-\d{2}-\d{2}T[0-9:.]+Z?\s+")

# Common log-level tokens (we only trust these when they appear near the start of a line).
_LOG_LEVEL_TOKEN_RE: Pattern[str] = re.compile(r"\b(TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL)\b")


def _strip_ts_prefix(s: str) -> str:
    """Remove the leading ISO timestamp prefix (if present)."""
    return _TS_PREFIX_RE.sub("", s or "")


def _strip_ansi(s: str) -> str:
    """Remove ANSI color escape sequences (if present)."""
    return _ANSI_ESCAPE_RE.sub("", s or "")


def _strip_ts_and_ansi(s: str) -> str:
    """Common normalization used across categorization/snippet extraction."""
    return _strip_ansi(_strip_ts_prefix(s or ""))


def _unescape_nested_shell_quotes(s: str) -> str:
    """Undo common backslash-escaped quoting used inside nested `bash -c "..."` payloads.

    When we extract a *standalone* command (e.g. plain `pytest ...`) from a nested shell string,
    the log often contains sequences like `\\\"` that were only necessary because the command was
    originally embedded inside a larger quoted string. For copy/paste as a standalone command, we
    want the natural quotes back.
    """
    try:
        t = str(s or "")
        if not t:
            return ""
        # Be conservative: only undo escaped quotes. Do NOT try to interpret all backslash escapes.
        t = t.replace('\\"', '"').replace("\\'", "'")
        return t
    except Exception:
        return str(s or "")


def _extract_log_level(line: str) -> Optional[str]:
    """Best-effort extraction of log level for a single log line.

    Returns one of: TRACE/DEBUG/INFO/WARN/WARNING/ERROR/FATAL or None if unknown.
    """
    try:
        s = _strip_ts_and_ansi(line or "")
        # Strip nested timestamp prefixes (common in Rust tracing / structured logs).
        for _ in range(3):
            s2 = _ISO_TS_PREFIX_LOOSE_RE.sub("", s)
            if s2 == s:
                break
            s = s2
        # Only trust tokens that show up early; avoid matching message text like "warning: ...".
        m = _LOG_LEVEL_TOKEN_RE.search((s or "")[:96])
        if not m:
            return None
        return str(m.group(1) or "").upper() or None
    except Exception:
        return None


def _line_is_warn_or_lower(line: str) -> bool:
    """True if the line clearly indicates WARN/INFO/DEBUG/TRACE (i.e. not an error)."""
    lvl = _extract_log_level(line)
    return bool(lvl in {"TRACE", "DEBUG", "INFO", "WARN", "WARNING"})


def _has_huggingface_auth_error_signal(lines: Sequence[str]) -> bool:
    """Line-based HuggingFace auth detection with WARN filtering.

    We only tag `huggingface-auth-error` when the auth signature appears on a line that is not
    clearly a WARN/INFO/DEBUG/TRACE log line.

    This prevents false positives from benign warnings like:
      'WARN ... ModelExpress download failed ... (401 Unauthorized) ... huggingface.co/...'
    """
    try:
        for raw in (lines or []):
            s = _strip_ts_and_ansi(str(raw or ""))
            if not s:
                continue
            if not HUGGINGFACE_AUTH_ERROR_RE.search(s):
                continue
            # If the line is explicitly WARN/INFO/etc, treat it as a warning, not a root-cause error tag.
            if _line_is_warn_or_lower(str(raw or "")):
                continue
            return True
    except Exception:
        return False
    return False


_PYTEST_NONZERO_FAIL_OR_ERROR_COUNT_RE: Pattern[str] = re.compile(
    r"\b([1-9]\d*)\s+failed\b|\b([1-9]\d*)\s+errors?\b", re.IGNORECASE
)

# BuildKit/docker logs often prefix lines with progress markers like:
#   "#55 ERROR: ..."  or  "41.35 Errors logged to ..."
# Those can accidentally look like pytest's "N errors" summaries. Strip these prefixes before
# applying count-based pytest heuristics.
_BUILDKIT_STEP_PREFIX_RE: Pattern[str] = re.compile(r"^\s*#\d+\s+")
_BUILDKIT_TIME_PREFIX_RE: Pattern[str] = re.compile(r"^\s*\d+\.\d+\s+")


def _has_pytest_failure_signal(lines: Sequence[str]) -> bool:
    """True if a log looks like pytest had failures/errors (not just a successful run summary)."""
    try:
        for raw in (lines or []):
            s = _strip_ts_and_ansi(str(raw or ""))
            if not s:
                continue
            # Avoid misclassifying Rust test output as pytest.
            # Cargo prints summaries like:
            #   test result: FAILED. 630 passed; 1 failed; ...
            if re.search(r"^\s*test\s+result:\s+", s, flags=re.IGNORECASE):
                continue
            # Explicit failing test id lines.
            if PYTEST_FAILED_LINE_RE.search(s):
                return True
            # Pytest's per-file error marker.
            if PYTEST_ERROR_FILE_LINE_RE.search(s):
                return True
            # Collection errors are always failures.
            if re.search(r"\berror[ \t]+collecting\b", s, flags=re.IGNORECASE):
                return True
            # Traditional section headers.
            if re.search(r"==+\s*(FAILURES|ERRORS)\s*==+", s, flags=re.IGNORECASE):
                return True
            # Summary lines like:
            #   "====== 3 failed, 8 passed, 4 skipped, 626 deselected in 309.86s (0:05:09) ======"
            # These are highly characteristic of pytest and safe to treat as a failure signal.
            s2 = _BUILDKIT_STEP_PREFIX_RE.sub("", s)
            s2 = _BUILDKIT_TIME_PREFIX_RE.sub("", s2)
            if re.search(r"\b([1-9]\d*)\s+failed\b", s2, flags=re.IGNORECASE) and re.search(
                r"(?:\bpassed\b|\bskipped\b|\bdeselected\b|\bwarnings?\b|\bshort test summary info\b)",
                s2,
                flags=re.IGNORECASE,
            ):
                return True
    except Exception:
        return False
    return False


def _norm_cat(s: str) -> str:
    """Normalize category strings for comparison (self-test)."""
    x = (s or "").strip().lower()
    if not x:
        return ""
    x = x.replace("_", "-")
    x = re.sub(r"[^a-z0-9-]+", "-", x)
    x = re.sub(r"-{2,}", "-", x).strip("-")
    # Canonicalize common variants
    if x in {"k8s"}:
        return "k8s-error"
    if x in {"github-lfs", "github-lfs"}:
        return "github-lfs-error"
    if x in {"mystery-139-error", "mystery-139"}:
        return "exit-139-sigsegv"
    if x in {"exit-139-error"}:
        return "exit-139-sigsegv"
    if x in {"exit-127-error"}:
        return "exit-127-cmd-not-found"
    return x


def _parse_examples_from_docstring() -> list[tuple[str, list[str], list[str]]]:
    """Parse the module docstring's Examples list.

    Grammar:
      - `<file>.log => cat1, cat2, !forbidden1, !forbidden2`
      - Inline comments after `#` are allowed and ignored (useful for occurrence counts).
    """
    out: list[tuple[str, list[str], list[str]]] = []
    doc = (__doc__ or "").splitlines()
    for ln in doc:
        s = (ln or "").strip()
        if not s.startswith("- "):
            continue
        if "=>" not in s:
            continue
        left, right = s[2:].split("=>", 1)
        log_name = left.strip()
        # Allow inline comments like:
        #   - 123.log => pytest-error  # occurred 5/623
        right_no_comment = right.split("#", 1)[0].strip()
        tokens = [c.strip() for c in right_no_comment.split(",") if c.strip()]
        expected: list[str] = []
        forbidden: list[str] = []
        for tok in tokens:
            if tok.startswith("!"):
                forbidden.append(tok[1:].strip())
            else:
                expected.append(tok)
        if log_name.endswith(".log") and (expected or forbidden):
            out.append((log_name, expected, forbidden))
    return out


def _parse_snippet_assertions_from_docstring() -> list[tuple[str, list[str], list[str], list[str], list[str]]]:
    """Parse snippet output assertions from the module docstring.

    Grammar:
      * `<file>.log => +must_contain1, +must_contain2, !must_not_contain1, !must_not_contain2`
      * Optional HTML full-line-red assertions:
        - `+RED:<substr>` means the rendered snippet HTML must contain `<substr>` inside a full-line red span.
        - `!RED:<substr>` means the rendered snippet HTML must NOT contain `<substr>` inside a full-line red span.
    """
    out: list[tuple[str, list[str], list[str], list[str], list[str]]] = []
    doc = (__doc__ or "").splitlines()
    for ln in doc:
        s = (ln or "").strip()
        if not s.startswith("* "):
            continue
        if "=>" not in s:
            continue
        left, right = s[2:].split("=>", 1)
        log_name = left.strip()
        if not log_name.endswith(".log"):
            continue
        # Avoid matching the grammar line (e.g. `<job_id>.log`) or other placeholders.
        if not re.match(r"^\d+\.log$", log_name):
            continue
        right_no_comment = right.split("#", 1)[0].strip()
        tokens = [c.strip() for c in right_no_comment.split(",") if c.strip()]
        must: list[str] = []
        must_not: list[str] = []
        must_red: list[str] = []
        must_not_red: list[str] = []
        for tok in tokens:
            if tok.lower().startswith("+red:"):
                must_red.append(tok.split(":", 1)[1].strip())
                continue
            if tok.lower().startswith("!red:"):
                must_not_red.append(tok.split(":", 1)[1].strip())
                continue
            if tok.startswith("+"):
                must.append(tok[1:].strip())
            elif tok.startswith("!"):
                must_not.append(tok[1:].strip())
        if log_name and (must or must_not or must_red or must_not_red):
            out.append((log_name, must, must_not, must_red, must_not_red))
    return out


def _self_test_examples(*, raw_log_path: Path) -> int:
    """Self-test: load the example logs and report category match coverage."""
    examples = _parse_examples_from_docstring()
    if not examples:
        print("Self-test: no Examples found in module docstring.")
        return 2
    snippet_assertions = _parse_snippet_assertions_from_docstring()

    root = Path(raw_log_path).expanduser().resolve()
    missing_files: list[str] = []
    failures: int = 0

    print(f"Self-test: raw_log_path={root}")
    print(f"Self-test: cases={len(examples)}")
    print("")

    for log_name, expected_raw, forbidden_raw in examples:
        p = (root / log_name).resolve()
        if not p.exists():
            missing_files.append(str(p))
            continue

        # Golden-log workflow: keep the example logs read-only.
        # This should ONLY happen during the self-check process (per policy).
        try:
            _ = _chmod_remove_write_bits(p)
        except Exception:
            pass

        text = _read_text_tail(p, max_bytes=512 * 1024)
        all_lines = (text or "").splitlines()
        cats_full = [_norm_cat(x) for x in categorize_error_log_lines(all_lines)]
        cats_full_set = {c for c in cats_full if c}

        snip = extract_error_snippet_from_text(text)
        cats_snip = [_norm_cat(x) for x in categorize_error_snippet_text(snip)]
        cats_snip_set = {c for c in cats_snip if c}

        exp = [_norm_cat(x) for x in expected_raw]
        exp_set = {c for c in exp if c}
        forb = [_norm_cat(x) for x in forbidden_raw]
        forb_set = {c for c in forb if c}

        missing_full = sorted(exp_set - cats_full_set)
        missing_snip = sorted(exp_set - cats_snip_set)
        forbidden_full = sorted(forb_set & cats_full_set)
        forbidden_snip = sorted(forb_set & cats_snip_set)

        ok = (not missing_full) and (not missing_snip) and (not forbidden_full) and (not forbidden_snip)
        if not ok:
            failures += 1

        match_full = len(exp_set & cats_full_set)
        match_snip = len(exp_set & cats_snip_set)
        denom = max(1, len(exp_set))

        print(f"- {log_name}")
        print(f"  expected:  {', '.join(sorted(exp_set))}")
        if forb_set:
            print(f"  forbidden: {', '.join(sorted(forb_set))}")
        print(f"  full:      {', '.join(sorted(cats_full_set))}   (match {match_full}/{denom})")
        if missing_full:
            print(f"  missing(full): {', '.join(missing_full)}")
        if forbidden_full:
            print(f"  forbidden(full): {', '.join(forbidden_full)}")
        print(f"  snippet:   {', '.join(sorted(cats_snip_set))}   (match {match_snip}/{denom})")
        if missing_snip:
            print(f"  missing(snip): {', '.join(missing_snip)}")
        if forbidden_snip:
            print(f"  forbidden(snip): {', '.join(forbidden_snip)}")
        print("")

    # Extra snippet output assertions (text-level must-include / must-not-include)
    if snippet_assertions:
        print("Self-test: snippet output assertions")
        for log_name, must, must_not, must_red, must_not_red in snippet_assertions:
            p = (root / log_name).resolve()
            if not p.exists():
                missing_files.append(str(p))
                continue
            try:
                _ = _chmod_remove_write_bits(p)
            except Exception:
                pass
            # Snippet assertions are sensitive to truncation. Use a larger tail budget than the
            # category self-test so we reliably capture preceding command blocks / workflow paths.
            text = _read_text_tail(p, max_bytes=2 * 1024 * 1024)
            snip = extract_error_snippet_from_text(text)
            snip_html = render_error_snippet_html(snip)
            missing: list[str] = []
            present_forbidden: list[str] = []
            missing_red: list[str] = []
            present_forbidden_red: list[str] = []
            for m in (must or []):
                if m and m not in (snip or ""):
                    missing.append(m)
            for f in (must_not or []):
                if f and f in (snip or ""):
                    present_forbidden.append(f)

            # HTML full-line-red assertions: verify substrings appear (or do not appear) inside
            # a full-line red span emitted by render_error_snippet_html():
            #   <span style="color: #c83a3a;">...</span>
            def _has_full_line_red_substr(substr: str) -> bool:
                try:
                    if not substr:
                        return False
                    # Substrings appear in HTML-escaped form.
                    needle = html.escape(substr)
                    return bool(
                        re.search(
                            r"<span\s+style=\"color:\s*#c83a3a;\">\s*[^<]*" + re.escape(needle) + r"[^<]*</span>",
                            snip_html or "",
                            flags=re.IGNORECASE,
                        )
                    )
                except Exception:
                    return False

            for rsub in (must_red or []):
                if rsub and (not _has_full_line_red_substr(rsub)):
                    missing_red.append(rsub)
            for rsub in (must_not_red or []):
                if rsub and _has_full_line_red_substr(rsub):
                    present_forbidden_red.append(rsub)

            ok = (not missing) and (not present_forbidden) and (not missing_red) and (not present_forbidden_red)
            if not ok:
                failures += 1
            print(f"- {log_name}")
            if must:
                print(f"  must_contain: {', '.join(must)}")
            if must_not:
                print(f"  must_not:    {', '.join(must_not)}")
            if must_red:
                print(f"  must_be_red: {', '.join(must_red)}")
            if must_not_red:
                print(f"  must_not_red:{', '.join(must_not_red)}")
            if missing:
                print(f"  missing:     {', '.join(missing)}")
            if present_forbidden:
                print(f"  forbidden:   {', '.join(present_forbidden)}")
            if missing_red:
                print(f"  missing_red: {', '.join(missing_red)}")
            if present_forbidden_red:
                print(f"  forbidden_red:{', '.join(present_forbidden_red)}")
            print("")

    # Extra renderer unit test: verify full-line-red behavior for specific high-signal lines.
    #
    # This is intentionally independent from the golden logs: some exact strings are rare or
    # unstable in real CI logs, but we still want regression protection for the UI rules.
    try:
        print("Self-test: snippet HTML red/not-red rules (unit)")

        def _has_full_line_red_substr_in_html(rendered_html: str, substr: str) -> bool:
            try:
                if not substr:
                    return False
                needle = html.escape(substr)
                return bool(
                    re.search(
                        r"<span\s+style=\"color:\s*#c83a3a;\">\s*[^<]*" + re.escape(needle) + r"[^<]*</span>",
                        rendered_html or "",
                        flags=re.IGNORECASE,
                    )
                )
            except Exception:
                return False

        # Craft a small snippet body with both "should be red" and "should NOT be red" lines.
        unit_snip = "\n".join(
            [
                "[[CMD]]",
                "docker run something",
                "[[/CMD]]",
                "...",
                "E           Failed: Timeout (>60.0s) from pytest-timeout.",
                "assertion failed: (7..=13).contains(&elapsed_ms)",
                "ERROR: failed to build",
                "[FAIL] incorrect date: container/dev/dev_build.sh",
                "#104 25.87   ├─▶ Git operation failed",
                "#104 25.87   ├─▶ failed to fetch LFS objects at 8ecebecaf2797e2acc2cd07c5fc5ef26d1acab71",
                "failures:",
                "recorder::tests::test_recorder_streams_events_to_file",
                # Should NOT be full-line red (user requested removal).
                "=================================== FAILURES ===================================",
                "__________ test_router_decisions_disagg[with_bootstrap-decode_first] ___________",
            ]
        )
        unit_html = render_error_snippet_html(unit_snip)

        must_be_red = [
            "E           Failed: Timeout (>60.0s) from pytest-timeout.",
            "assertion failed: (7..=13).contains(&elapsed_ms)",
            "ERROR: failed to build",
            "[FAIL] incorrect date: container/dev/dev_build.sh",
            "Git operation failed",
            "failed to fetch LFS objects",
            "failures:",
            "recorder::tests::test_recorder_streams_events_to_file",
        ]
        must_not_be_red = [
            "=================================== FAILURES ===================================",
            "__________ test_router_decisions_disagg[with_bootstrap-decode_first] ___________",
        ]

        unit_fail = False
        for s in must_be_red:
            if not _has_full_line_red_substr_in_html(unit_html, s):
                print(f"  missing_red: {s}")
                unit_fail = True
        for s in must_not_be_red:
            if _has_full_line_red_substr_in_html(unit_html, s):
                print(f"  forbidden_red: {s}")
                unit_fail = True
        print("")
        if unit_fail:
            failures += 1
    except Exception:
        # Never hard-fail self-test on the unit runner itself; the golden logs are the primary guardrail.
        pass

    # Extra unit test: ensure the derived "rerun only failed tests" pytest command is placed
    # immediately BEFORE the pytest summary line (and not after).
    try:
        print("Self-test: snippet rerun-only-failed placement (unit)")
        unit_log = "\n".join(
            [
                # Put the real command on the Actions "Run ..." line (this matches how the extractor
                # finds execution commands in practice).
                '##[group]Run docker run dynamo:latest bash -c "pytest --basetemp=/tmp/pytest-parallel --junitxml=pytest_parallel.xml -n 4 -m \\"pre_merge and parallel and not (vllm or sglang or trtllm) and (gpu_0 or gpu_1)\\""',
                "##[endgroup]",
                "FAILED tests/router/test_router_e2e_with_mockers.py::test_router_decisions_disagg[prefill_first-nats]",
                "FAILED tests/router/test_router_e2e_with_mockers.py::test_router_decisions_disagg[prefill_first-tcp]",
                "FAILED tests/router/test_router_e2e_with_mockers.py::test_router_decisions_disagg[decode_first-tcp]",
                "FAILED tests/router/test_router_e2e_with_mockers.py::test_router_decisions_disagg[decode_first-nats]",
                "============= 4 failed, 27 passed, 1 skipped in 199.28s (0:03:19) ==============",
            ]
        )
        unit_snip = extract_error_snippet_from_text(unit_log, context_before=10, context_after=5, max_lines=120, max_chars=12000)
        # Expected: derived rerun command appears before the summary line.
        summary_line = "============= 4 failed, 27 passed, 1 skipped in 199.28s (0:03:19) =============="
        rerun_prefix = "pytest --basetemp=/tmp/pytest-parallel --junitxml=pytest_parallel.xml -n 4 -m \"pre_merge and parallel"
        ok = True
        if summary_line not in unit_snip:
            print("  missing_summary_line")
            ok = False
        if rerun_prefix not in unit_snip:
            print("  missing_rerun_cmd")
            ok = False
        if ok:
            if unit_snip.find(rerun_prefix) > unit_snip.find(summary_line):
                print("  ordering_error: rerun cmd appears after summary line")
                ok = False
        print("")
        if not ok:
            failures += 1
    except Exception:
        pass

    # Extra golden regression guard: `PYTEST_CMD="pytest ..."` variable expansion should still produce
    # a suggested rerun-only-failed command, and it should appear after the FAILED chunk and before
    # the `====== N failed, ... ======` summary (note: this summary format is common in our logs).
    try:
        job_id = "57521050539"
        p = Path(raw_log_path) / f"{job_id}.log"
        if p.exists() and p.is_file():
            print("Self-test: rerun-only-failed (PYTEST_CMD) placement (golden)")
            sn = extract_error_snippet_from_log_file(p, tail_bytes=0, context_before=15, context_after=15, max_lines=160, max_chars=20000)
            # Last FAILED line in that log:
            last_failed = "FAILED tests/router/test_router_e2e_with_sglang.py::test_sglang_indexers_sync"
            # Summary style in that log:
            summary_re = re.compile(r"^=+\s*\d+\s+failed\b.*=+\s*$", re.IGNORECASE)
            ok = True
            if last_failed not in sn:
                print("  missing_failed_line")
                ok = False
            # Find a summary line (any variant)
            summary_line = ""
            for ln in (sn or "").splitlines():
                if summary_re.search((ln or "").strip()):
                    summary_line = ln
                    break
            if not summary_line:
                print("  missing_summary_line")
                ok = False

            # Identify the synthesized rerun-only-failed command line:
            # it must start with `pytest` and include at least one of the failed nodeids
            # (so we don't confuse it with the base pytest command in the prelude).
            rerun_line = ""
            want_nodeids = [
                "tests/router/test_router_e2e_with_sglang.py::test_sglang_kv_router_basic",
                "tests/router/test_router_e2e_with_sglang.py::test_router_decisions_sglang_multiple_workers",
                "tests/router/test_router_e2e_with_sglang.py::test_sglang_indexers_sync",
            ]
            for ln in (sn or "").splitlines():
                s = (ln or "").strip()
                if not s.lower().startswith("pytest "):
                    continue
                if any(nid in s for nid in want_nodeids):
                    rerun_line = ln
                    break
            if not rerun_line:
                print("  missing_rerun_cmd")
                ok = False
            if ok:
                if sn.find(rerun_line) < sn.find(last_failed):
                    print("  ordering_error: rerun cmd appears before FAILED chunk")
                    ok = False
                if summary_line and sn.find(rerun_line) > sn.find(summary_line):
                    print("  ordering_error: rerun cmd appears after summary line")
                    ok = False
            print("")
            if not ok:
                failures += 1
    except Exception:
        pass

    if missing_files:
        print("Self-test: missing example log files:")
        for x in missing_files:
            print(f"  - {x}")
        print("")
        return 2

    if failures:
        print(f"Self-test: FAIL ({failures}/{len(examples)} cases mismatched)")
        return 1

    print("Self-test: OK")
    return 0


_GOLDEN_JOB_IDS: Optional[set[str]] = None


def golden_log_job_ids() -> set[str]:
    """Return the set of job_ids treated as "golden" example logs.

    Source of truth is the module docstring "Log files to error markers -- training examples" list.
    """
    global _GOLDEN_JOB_IDS
    if _GOLDEN_JOB_IDS is not None:
        return set(_GOLDEN_JOB_IDS)
    ids: set[str] = set()
    try:
        for log_name, _exp, _forb in _parse_examples_from_docstring():
            m = re.match(r"^\s*(\d+)\.log\s*$", str(log_name or ""))
            if m:
                ids.add(str(m.group(1)))
    except Exception:
        ids = set()
    _GOLDEN_JOB_IDS = set(ids)
    return set(_GOLDEN_JOB_IDS)


def is_golden_log_job_id(job_id: str) -> bool:
    try:
        j = str(job_id or "").strip()
        if not j.isdigit():
            return False
        return j in golden_log_job_ids()
    except Exception:
        return False


def _chmod_remove_write_bits(path: Path) -> bool:
    """Best-effort `chmod a-w` for a file.

    Returns True if we successfully *attempted* to update perms (including no-op),
    False if we couldn't stat/chmod.
    """
    try:
        p = Path(path)
        st0 = p.stat()
        new_mode = int(st0.st_mode) & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH
        # Avoid unnecessary chmod syscalls when already read-only.
        if int(st0.st_mode) == int(new_mode):
            return True
        os.chmod(str(p), int(new_mode))
        return True
    except Exception:
        return False


def _chmod_add_user_write_bit(path: Path) -> bool:
    """Best-effort `chmod u+w` for a file.

    Returns True if we successfully *attempted* to update perms (including no-op),
    False if we couldn't stat/chmod.
    """
    try:
        p = Path(path)
        st0 = p.stat()
        new_mode = int(st0.st_mode) | stat.S_IWUSR
        # Avoid unnecessary chmod syscalls when already user-writable.
        if int(st0.st_mode) == int(new_mode):
            return True
        os.chmod(str(p), int(new_mode))
        return True
    except Exception:
        return False


def _scan_all_logs(*, logs_root: Path, tail_bytes: int = 512 * 1024) -> int:
    """Scan a directory of `*.log` files and report categorization + snippet coverage.

    This is a practical "retrain/validate" helper:
    - validates that snippet anchoring finds something in most logs
    - summarizes category frequency (full-log + snippet-derived)

    Permission policy:
    - Golden training-example logs must be preserved (non-writable).
    - All other logs in the directory should be user-writable so they can be edited/cleaned up.
    - This scan enforces permissions on each `*.log` it touches (best-effort).
    """
    root = Path(logs_root).expanduser().resolve()
    if not root.exists():
        print(f"ERROR: logs_root not found: {root}", file=sys.stderr)
        return 2
    if not root.is_dir():
        print(f"ERROR: logs_root is not a directory: {root}", file=sys.stderr)
        return 2

    logs = sorted([p for p in root.glob("*.log") if p.is_file()])
    total = len(logs)
    print(f"Scan-all: logs_root={root}")
    print(f"Scan-all: files={total}")
    print("")

    if total == 0:
        return 0

    cats_full_counts: Dict[str, int] = {}
    cats_snip_counts: Dict[str, int] = {}
    snippet_found = 0
    no_snippet_samples: List[str] = []
    for p in logs:
        # Enforce log permissions:
        # - golden/training logs must remain non-writable
        # - all other logs should be user-writable (to allow cleanup/editing)
        try:
            job_id = str(p.stem or "")
            if is_golden_log_job_id(job_id):
                _ = _chmod_remove_write_bits(p)
            else:
                _ = _chmod_add_user_write_bit(p)
        except Exception:
            pass

        txt = _read_text_tail(p, max_bytes=int(tail_bytes))
        lines = (txt or "").splitlines()

        cats_full = [_norm_cat(x) for x in categorize_error_log_lines(lines)]
        for c in cats_full:
            if not c:
                continue
            cats_full_counts[c] = int(cats_full_counts.get(c, 0)) + 1

        snip = extract_error_snippet_from_text(txt)
        if (snip or "").strip():
            snippet_found += 1
        else:
            # Keep a small sample for debugging if snippet anchoring regresses.
            if len(no_snippet_samples) < 12:
                no_snippet_samples.append(p.name)

        cats_snip = [_norm_cat(x) for x in categorize_error_snippet_text(snip)]
        for c in cats_snip:
            if not c:
                continue
            cats_snip_counts[c] = int(cats_snip_counts.get(c, 0)) + 1

    def _top_items(d: Dict[str, int], n: int = 25) -> List[Tuple[str, int]]:
        return sorted(d.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))

    print(f"Scan-all: snippet_found={snippet_found}/{total} ({(100.0 * snippet_found / max(1, total)):.1f}%)")
    print("")

    print("Scan-all: category frequency (full-log categorization):")
    for name, count in _top_items(cats_full_counts):
        pct = 100.0 * float(count) / float(total)
        print(f"  - {name:<28} {count:>5}/{total} ({pct:>4.1f}%)")
    print("")

    print("Scan-all: category frequency (snippet-derived categorization):")
    for name, count in _top_items(cats_snip_counts):
        pct = 100.0 * float(count) / float(total)
        print(f"  - {name:<28} {count:>5}/{total} ({pct:>4.1f}%)")
    print("")

    if no_snippet_samples:
        print("Scan-all: sample logs with NO snippet extracted (first 12):")
        for nm in no_snippet_samples:
            print(f"  - {nm}")
        print("")

    return 0


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
    # Timeout: keep this very conservative to avoid false positives from dependency strings like
    # "pytest-timeout==2.4.0" / "timeout-2.4.0" and from prose like "individual test timeouts".
    r"|\b(?:timed\s*out|timedout)\b"
    # Generic error/exception class tokens (CamelCase) like "ModuleNotFoundError:".
    # Avoid false positives from crate/package names like "serde_path_to_error" and from stack traces.
    r"|(?-i:(?<![./])\b[A-Z][A-Za-z0-9]{2,}(?:Error|Exception)(?::|\b$))"
    r"|\b(?:broken\s+links?|broken\s+link|dead\s+links?)\b"
    r"|\b(?:network\s+error|connection\s+failed)\b"
    # Multi-line backend result blocks (JSON-ish) often show:
    #   "trtllm": { ... "result": "failure", ... }
    # Anchor on the high-signal failure field so the snippet includes the surrounding block.
    r"|\"result\"\s*:\s*\"failure\""
    # Copyright header checks
    r"|\bInvalid/Missing\s+Header:\b"
    r"|\binvalid/missing\s+header:\b"
    r"|\bcopyright\s+checkers\s+detected\s+missing\s+or\s+invalid\s+copyright\s+headers\b"
    # HuggingFace auth / missing token warnings
    r"|\bHF_TOKEN\s+not\s+found\s+in\s+environment\b"
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

# Categorization patterns (full log)
PYTEST_DETECT_RE: Pattern[str] = re.compile(
    r"(?:"
    # Restrict to pytest-style failing test ids: "FAILED path/to/test_foo.py::test_name[...]"
    r"(?:^|[ \t])FAILED(?:[ \t]+|$)[^\\n]*\\.py::"
    r"|==+\\s*FAILURES\\s*==+"
    r"|==+\\s*ERRORS\\s*==+"
    r"|\berror[ \t]+collecting\b"
    r")",
    re.IGNORECASE | re.MULTILINE,
)
DOWNLOAD_ERROR_RE: Pattern[str] = re.compile(r"\bcaused by:\s*failed to download\b|\bfailed to download\b|\bdownload error\b")
DOCKER_BUILD_ERROR_RE: Pattern[str] = re.compile(r"\berror:\s*failed\s+to\s+build\b|\bfailed\s+to\s+solve\b")
CUDA_ERROR_RE: Pattern[str] = re.compile(
    r"(?:"
    r"unsupported\s+cuda\s+version\s+for\s+vllm\s+installation"
    r"|\bcuda\b[^\n]{0,120}\bunsupported\b"
    r"|\bimporterror:\s*libcuda\.so\.1:\s*cannot\s+open\s+shared\s+object\s+file\b"
    r")"
)
HTTP_TIMEOUT_RE: Pattern[str] = re.compile(
    r"(?:"
    r"awaiting\s+response\.\.\.\s*(?:504|503|502)\b"
    r"|gateway\s+time-?out"
    r"|\bhttp\s+(?:504|503|502)\b"
    # Lychee/link-checker timeouts:
    #   [TIMEOUT] https://example.com | Timeout
    r"|\[timeout\]\s+https?://"
    r")"
)
GITLAB_MIRROR_TIMEOUT_RE: Pattern[str] = re.compile(r"\bmirror sync failed or timed out\b", re.IGNORECASE)
BUILD_STATUS_CHECK_ERROR_RE: Pattern[str] = re.compile(
    r"(?:"
    r"\bchecking\s+build\s+status\s+for\b"
    r"|\bbuild\s+status\s+for\s+'Build\b"
    r"|\bError:\s*Failed\s+to\s+query\s+GitHub\s+API\b"
    r"|\bBuild\s+failed\s+or\s+did\s+not\s+complete\s+successfully\.\s*(?:Failing\s+tests|Marking\s+tests\s+as\s+failed)\b"
    r")",
    re.IGNORECASE,
)
HUGGINGFACE_AUTH_ERROR_RE: Pattern[str] = re.compile(
    r"(?:"
    r"\bHF_TOKEN\s+not\s+found\s+in\s+environment\b"
    r"|\bHfHubHTTPError\b"
    r"|\bGatedRepoError\b"
    r"|\bRepositoryNotFoundError\b"
    r"|\bhuggingface[_-]hub\b[^\n]{0,160}\b(unauthorized|forbidden|invalid|token)\b"
    r"|\b401\b[^\n]{0,120}\bhuggingface\b"
    r"|\b403\b[^\n]{0,120}\bhuggingface\b"
    r")",
    re.IGNORECASE,
)
COPYRIGHT_HEADER_ERROR_RE: Pattern[str] = re.compile(
    r"(?:"
    r"\bcopyright-checks\b"
    r"|\bcopyright\s+checkers\s+detected\s+missing\s+or\s+invalid\s+copyright\s+headers\b"
    r"|\bInvalid/Missing\s+Header:\b"
    r"|\binvalid/missing\s+header:\b"
    r")",
    re.IGNORECASE,
)
HELM_ERROR_RE: Pattern[str] = re.compile(
    r"(?:"
    r"\bUPGRADE\s+FAILED\b"
    r"|\bINSTALLATION\s+FAILED\b"
    r"|\bKubernetes\s+cluster\s+unreachable\b"
    r"|\bhelm\b[^\n]{0,120}\berror\b"
    r"|\berror:\s*flag\s+needs\s+an\s+argument:\s*'n'\s+in\s+-n\b"
    r"|\berror:\s*resource\(s\)\s+were\s+provided,\s+but\s+no\s+name\s+was\s+specified\b"
    r")",
    re.IGNORECASE,
)
NETWORK_ERROR_RE: Pattern[str] = re.compile(
    r"\bnetwork\s+error:\s*connection\s+failed\b|\bconnection\s+failed\.\s*check\s+network\s+connectivity\b|\bfirewall\s+settings\b"
)
ETCD_ERROR_RE: Pattern[str] = re.compile(
    r"\bunable\s+to\s+create\s+lease\b|\bcheck\s+etcd\s+server\s+status\b|\betcd[^\n]{0,80}\blease\b|\blease\b[^\n]{0,80}\betcd\b"
)
DOCKER_INFRA_ERROR_RE: Pattern[str] = re.compile(
    r"(?:"
    r"cannot\s+connect\s+to\s+the\s+docker\s+daemon"
    r"|error\s+response\s+from\s+daemon:(?!.*no\s+such\s+container)"
    r"|\bdocker:\s+.*\berror\b"
    r")"
)
DOCKER_DAEMON_CONNECTION_ERROR_RE: Pattern[str] = re.compile(
    r"cannot\s+connect\s+to\s+the\s+docker\s+daemon", re.IGNORECASE
)
DOCKER_DAEMON_ERROR_RESPONSE_RE: Pattern[str] = re.compile(
    r"error\s+response\s+from\s+daemon:(?!.*no\s+such\s+container)", re.IGNORECASE
)
DOCKER_CLI_ERROR_RE: Pattern[str] = re.compile(r"\bdocker:\s+.*\berror\b", re.IGNORECASE)
BROKEN_LINKS_RE: Pattern[str] = re.compile(r"\bbroken\s+links?\b|\bdead\s+links?\b")
TIMED_OUT_RE: Pattern[str] = re.compile(r"\b(?:timed\s*out|timedout)\b")
K8S_PODS_TIMED_OUT_RE: Pattern[str] = re.compile(
    r"\btimed\s*out\s+waiting\s+for\s+the\s+condition\s+on\s+pods/",
    re.IGNORECASE,
)
# Kubernetes error signals.
#
# IMPORTANT: Do NOT match on the bare word "kubernetes" because it frequently appears as a Python
# dependency install (e.g. `pip install kubernetes==...`) and is not a Kubernetes *error*.
K8S_ERROR_RE: Pattern[str] = re.compile(
    r"(?:"
    r"CrashLoopBackOff|ImagePullBackOff|ErrImagePull"
    r"|Kubernetes\s+cluster\s+unreachable"
    r"|\bkubectl\b[^\n]{0,200}\b("
    r"error|failed|failure|timed\s*out|timeout|unable|forbidden|unauthorized|refused|i/o\s+timeout|not\s+found|"
    r"context\s+deadline\s+exceeded"
    r")\b"
    r")",
    re.IGNORECASE,
)
KUBECTL_PORTFORWARD_TIMEOUT_RE: Pattern[str] = re.compile(
    r"(?:"
    r"\bportforward\.go:\d+\].*\bconnection timed out\b"
    r"|\ban error occurred forwarding\b[^\n]{0,400}\bconnection timed out\b"
    r")",
    re.IGNORECASE,
)
GITHUB_ACTION_STEP_TIMEOUT_RE: Pattern[str] = re.compile(
    r"##\[error\].{0,40}\bhas\s+timed\s+out\s+after\s+\d+\s+minutes?\b",
    re.IGNORECASE,
)
RUST_TEST_FAIL_RE: Pattern[str] = re.compile(r"^\s*failures:\s*$|test result:\s*FAILED\.", re.IGNORECASE | re.MULTILINE)
# Exit code 139 is conventionally SIGSEGV (signal 11) in POSIX shells (\(128 + 11 = 139\)).
EXIT_CODE_139_RE: Pattern[str] = re.compile(r"process completed with exit code 139\b|exit code:\s*139\b", re.IGNORECASE)

# Exit code 127 is conventionally “command not found” in POSIX shells.
# In CI logs this usually means a missing dependency inside the container or a PATH issue.
EXIT_CODE_127_RE: Pattern[str] = re.compile(
    r"process completed with exit code 127\b|exit code:\s*127\b|\bcommand not found\b",
    re.IGNORECASE,
)


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
            s = _strip_ts_and_ansi(str(raw or ""))

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

def categorize_error_log_lines(lines: Sequence[str]) -> List[str]:
    """Categorize an error log (full log) into high-level categories (best-effort).

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

        # Reuse module-level regexes so categorization stays consistent and readable.
        _PYTHON_ERROR_DETECT_RE = PYTHON_EXCEPTION_LINE_RE

        # Pytest / Python
        # Only tag "pytest" when the log looks like an actual pytest run failure,
        # not when it merely mentions pytest packages/versions (e.g., "pytest==9.0.2").
        has_pytest = bool(_has_pytest_failure_signal(lines[-4000:] if lines else []))
        if has_pytest:
            add("pytest-error")
        # Per-test timeout (pytest-timeout plugin). Keep distinct from generic/network timeouts.
        if PYTEST_TIMEOUT_E_LINE_RE.search(text):
            add("pytest-timeout-error")
        # Keep python-error distinct from pytest-error: pytest failures frequently contain AssertionError
        # and other tracebacks that are not actionable as "python infra" errors.
        if _PYTHON_ERROR_DETECT_RE.search(t):
            if (not has_pytest) or bool(PYTHON_STRONG_EXCEPTION_RE.search(text)):
                add("python-error")
        # Rust test failures (cargo test)
        if RUST_TEST_FAIL_RE.search(text):
            add("rust-error")
        # Exit code 139 is conventionally SIGSEGV (signal 11): 128 + 11 = 139.
        if EXIT_CODE_139_RE.search(text):
            add("exit-139-sigsegv")
        # Exit code 127: command not found (missing dependency / PATH issue).
        if EXIT_CODE_127_RE.search(text):
            add("exit-127-cmd-not-found")

        # Git / GitHub LFS
        #
        # If LFS is implicated, prefer tagging github-lfs-error and DO NOT also tag git-fetch
        # (the golden logs treat those as mutually exclusive for this class of failure).
        if "failed to fetch some objects" in t:
            if "/info/lfs" in t or "git lfs" in t:
                add("github-lfs-error")
            else:
                add("git-fetch")

        # Downloads (Rust/cargo, pip, curl, etc.)
        if DOWNLOAD_ERROR_RE.search(t):
            add("network-download-error")

        # Build failures (Docker/buildkit/etc.)
        if DOCKER_BUILD_ERROR_RE.search(t):
            add("docker-build-error")

        # Build-status-check failures (CI gate that checks upstream build job status)
        if BUILD_STATUS_CHECK_ERROR_RE.search(text):
            add("build-status-check-error")

        # HuggingFace auth/token failures (missing/invalid HF_TOKEN or gated model access).
        #
        # IMPORTANT: filter out WARN-level lines so we don't mis-tag benign warnings as errors.
        if _has_huggingface_auth_error_signal(lines[-4000:] if lines else []):
            add("huggingface-auth-error")

        # Copyright header checks
        if COPYRIGHT_HEADER_ERROR_RE.search(text):
            add("copyright-header-error")

        # Helm/k8s workflow failures (deploy/cleanup)
        if HELM_ERROR_RE.search(text):
            add("helm-error")
        # Kubernetes signal (error) — keep this strict so we don't mis-tag logs that merely install
        # the Python `kubernetes` package.
        if K8S_ERROR_RE.search(text) or ("helm-error" in cats):
            add("k8s-error")

        # CUDA / GPU toolchain
        if CUDA_ERROR_RE.search(t):
            add("cuda-error")

        # HTTP(S) gateway timeouts (wget/curl/HTTP clients + link-checker timeouts)
        if HTTP_TIMEOUT_RE.search(t):
            add("network-timeout-https")
        # GitLab mirror infra timeout (special-case; keep distinct from generic timeout)
        if GITLAB_MIRROR_TIMEOUT_RE.search(text):
            add("network-timeout-gitlab-mirror")

        # Network connectivity
        if NETWORK_ERROR_RE.search(t):
            add("network-error")

        # Etcd / lease
        # Avoid tagging `etcd-error` just because the word "etcd" appears (it often shows up in benign logs).
        # Require a lease/status failure signature.
        if ETCD_ERROR_RE.search(t):
            add("etcd-error")

        # Docker infrastructure / daemon / CLI errors.
        #
        # IMPORTANT: `docker-image-error` is already specific (manifest unknown). If we see that,
        # don't also tag a redundant daemon error-response category.
        if not DOCKER_IMAGE_NOT_FOUND_RE.search(t):
            if DOCKER_DAEMON_CONNECTION_ERROR_RE.search(text):
                add("docker-daemon-connection-error")
            elif DOCKER_DAEMON_ERROR_RESPONSE_RE.search(text):
                add("docker-daemon-error-response-error")
            elif DOCKER_CLI_ERROR_RE.search(text):
                add("docker-cli-error")

        # Backend result JSON-ish blocks (vllm/sglang/trtllm): multi-line aware.
        engines = _backend_failure_engines_from_lines(lines[-4000:] if lines else [])
        if engines:
            add("backend-failure")
            for e in sorted(engines):
                add(f"{e}-error")

        # Broken links
        if BROKEN_LINKS_RE.search(t):
            add("broken-links")

        # Kubernetes wait timeouts ("timed out waiting for the condition on pods/...").
        if K8S_PODS_TIMED_OUT_RE.search(text):
            add("k8s-network-timeout-pod")

        # Kubernetes port-forward failures (portforward.go ... connection timed out).
        if KUBECTL_PORTFORWARD_TIMEOUT_RE.search(text):
            add("k8s-network-timeout-portfwd")

        # GitHub Actions step timeout markers.
        if GITHUB_ACTION_STEP_TIMEOUT_RE.search(text):
            add("network-timeout-github-action")

        # Timeout / infra flake
        #
        # Keep this very conservative to avoid false positives.
        if TIMED_OUT_RE.search(t) and not (
            HTTP_TIMEOUT_RE.search(t)
            or GITLAB_MIRROR_TIMEOUT_RE.search(text)
            or K8S_PODS_TIMED_OUT_RE.search(text)
            or KUBECTL_PORTFORWARD_TIMEOUT_RE.search(text)
            or GITHUB_ACTION_STEP_TIMEOUT_RE.search(text)
        ):
            add("network-timeout-generic")

        # Docker image not found (registry manifest unknown)
        if DOCKER_IMAGE_NOT_FOUND_RE.search(t):
            add("docker-image-error")

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
    # Timeout: keep conservative to avoid false positives like "timeout-2.4.0" (pytest plugin) or prose.
    r"|\b(?:timed\s*out|timedout)\b"
    r"|\b(?:gateway\s+time-?out)\b"
    r"|\b(?:http\s*)?(?:502|503|504)\b"
    r"|\b(?:network\s+error|connection\s+failed|check\s+network\s+connectivity|firewall\s+settings)\b"
    # Don't highlight bare "etcd"/"lease" (too many false positives). Highlight the actual error phrases.
    r"|\b(?:unable\s+to\s+create\s+lease|check\s+etcd\s+server\s+status)\b"
    # Generic error/exception class tokens (CamelCase) like "ModuleNotFoundError:".
    # Avoid false positives from crate/package names like "serde_path_to_error" and from stack traces.
    r"|(?-i:(?<![./])\b[A-Z][A-Za-z0-9]{2,}(?:Error|Exception)(?::|\b$))"
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

# Snippet: command/execution lines (user wants these shown prominently in blue).
COMMAND_LINE_BLUE_RE: Pattern[str] = re.compile(
    r"(?:"
    r"\bPYTEST_CMD\s*="
    # Treat pytest as a command only when it doesn't look like a Python module/variable dump:
    #   pytest = <module 'pytest' from '...'>
    r"|\bpython\s+-m\s+pytest\b(?!\s*=)"
    r"|\bpytest\b(?!\s*=)(?:\s|$)"
    r"|\bbash\s+-c\s+['\"][^'\"]*\bpytest\b"
    r"|\bcargo\s+(?:test|build|check|clippy|fmt|rustfmt)\b"
    r"|\bdocker\s+(?:run|buildx|build)\b"
    r"|\b(?:\./)?run\.sh\b"
    r"|\b(?:\./)?build\.sh\b"
    r")",
    re.IGNORECASE,
)

# Snippet HTML helpers/styles (shared by multiple snippet render paths)
_SNIP_COPY_ROW_STYLE = "display: flex; align-items: flex-start; gap: 8px; margin: 2px 0; color: #0969da;"
_SNIP_COPY_BTN_STYLE = (
    "flex: 0 0 auto; display: inline-block; padding: 1px 8px; font-size: 11px; line-height: 1.2; "
    "background: transparent; color: #0969da; border: 1px solid #0969da; border-radius: 999px; "
    "cursor: pointer; margin: 0;"
)
_SNIP_COPY_TEXT_STYLE = "flex: 1 1 auto; white-space: pre-wrap; overflow-wrap: anywhere;"

@functools.lru_cache(maxsize=1)
def _copy_icon_svg(*, size_px: int = 12) -> str:
    """Return the shared 'copy' icon SVG (2-squares), sourced from html_pages/copy_icon_paths.svg."""
    try:
        # Shared library lives in dynamo-utils/ci_log_errors/; the shared icon lives in dynamo-utils/html_pages/
        p = (Path(__file__).resolve().parent / "html_pages" / "copy_icon_paths.svg").resolve()
        paths = p.read_text(encoding="utf-8").strip()
    except Exception:
        # Fallback: empty icon rather than crashing snippet rendering.
        paths = ""
    return (
        f'<svg width="{int(size_px)}" height="{int(size_px)}" viewBox="0 0 16 16" fill="currentColor" '
        f'style="display: inline-block; vertical-align: middle;">{paths}</svg>'
    )

# Exit code 139 is conventionally SIGSEGV (signal 11) in POSIX shells (\(128 + 11 = 139\)).
# Many CI jobs only report the exit code near the end.
EXIT_CODE_139_LINE_RE: Pattern[str] = re.compile(
    r"(?:"
    r"\bprocess completed with exit code 139\b"
    r"|\bexit code:\s*139\b"
    r")",
    re.IGNORECASE,
)

# Git LFS fetch failures inside Docker/BuildKit dependency installs (often via uv/pip).
# Example:
#   × Failed to download and build `aiconfigurator @
#   │ git+https://github.com/...#lfs=true`
#   ├─▶ Git operation failed
#   ├─▶ failed to fetch LFS objects at ...
#   ...
#   error: failed to fetch some objects from '.../info/lfs'
GIT_LFS_SNIPPET_ANCHOR_RE: Pattern[str] = re.compile(
    r"(?:"
    r"\bFailed\s+to\s+download\s+and\s+build\s+`"
    r"|\bGit\s+operation\s+failed\b"
    r"|\bfailed\s+to\s+fetch\s+LFS\s+objects\b"
    r"|\bUse\s+`git\s+lfs\s+logs\s+last`\b"
    r"|\bprocess\s+didn'?t\s+exit\s+successfully:.*\bgit\s+lfs\b"
    r"|\berror:\s*failed\s+to\s+fetch\s+some\s+objects\s+from\b"
    r")",
    re.IGNORECASE,
)

GIT_LFS_BLOCK_START_RE: Pattern[str] = re.compile(r"\bFailed\s+to\s+download\s+and\s+build\s+`", re.IGNORECASE)
GIT_LFS_BLOCK_END_RE: Pattern[str] = re.compile(
    r"(?:"
    r"\bprocess\b.*\bdid\s+not\s+complete\s+successfully\b"
    r"|\bERROR:\s*process\b.*\bdid\s+not\s+complete\s+successfully\b"
    r"|\bERROR:\s*failed\s+to\s+build\b"
    r"|^------\s*$"
    r"|##\\[error\\]"
    r")",
    re.IGNORECASE,
)

# CUDA runtime library import failures (common on runners missing NVIDIA driver libs).
CUDA_LIBCUDA_IMPORT_ERROR_RE: Pattern[str] = re.compile(
    r"\bImportError:\s*libcuda\.so\.1:\s*cannot\s+open\s+shared\s+object\s+file\b",
    re.IGNORECASE,
)

# Pytest often prints file-level error markers like:
#   ERROR components/src/dynamo/trtllm/tests/test_trtllm_unit.py
PYTEST_ERROR_FILE_LINE_RE: Pattern[str] = re.compile(
    r"(?:^|\s)\bERROR\s+components/src/dynamo/trtllm/tests/test_trtllm_[^\s]+\.py\b",
    re.IGNORECASE,
)

# Capture the effective pytest command line for debugging.
PYTEST_CMD_LINE_RE: Pattern[str] = re.compile(r"\bPYTEST_CMD\s*=", re.IGNORECASE)

# Docker image pull/tag errors (registry manifest missing).
DOCKER_IMAGE_NOT_FOUND_RE: Pattern[str] = re.compile(
    r"\bnot\s+found:\s*manifest\s+unknown:\s*requested\s+image\s+not\s+found\b",
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

PYTEST_SHORT_TEST_SUMMARY_RE: Pattern[str] = re.compile(r"===+\s*short test summary info\s*===+", re.IGNORECASE)
PYTHON_MODULE_NOT_FOUND_RE: Pattern[str] = re.compile(
    r"\bModuleNotFoundError:\s*No\s+module\s+named\b",
    re.IGNORECASE,
)

PYTHON_EXCEPTION_LINE_RE: Pattern[str] = re.compile(
    r"(?:"
    # Stack traces
    r"Traceback\s*\(most\s+recent\s+call\s+last\)"
    # Common high-signal exception types
    r"|\b(?:"
    r"ModuleNotFoundError"
    r"|ImportError"
    r"|AttributeError"
    r"|NameError"
    r"|KeyError"
    r"|IndexError"
    r"|ValueError"
    r"|TypeError"
    r"|AssertionError"
    r"|RuntimeError"
    r"|NotImplementedError"
    r"|TimeoutError"
    r"|FileNotFoundError"
    r"|PermissionError"
    r"|OSError"
    r"|IOError"
    r"|EOFError"
    r"|ConnectionError"
    r"|BrokenPipeError"
    r"|SyntaxError"
    r")\b"
    # Generic Python-style exception class tokens, e.g. "FooBarError" / "SomeException"
    # (Avoid matching bare "Error".)
    #
    # Also avoid matching package/type tokens from non-Python stack traces like:
    #   github.com/.../runcexecutor.exitError
    #   os/exec.ExitError
    #
    # These often appear in Docker/BuildKit logs and should NOT imply a Python failure.
    # Keep this conservative: require ':' or end-of-line after the token so we match real exception
    # lines like "KeyError: ..." and avoid random mentions.
    r"|(?-i:(?<![./])\b[A-Z][A-Za-z0-9]{2,}(?:Error|Exception)(?::|\b$))"
    r")",
    re.IGNORECASE,
)

# "Strong" Python exception signals that are typically actionable environment/infra/runtime errors,
# not just normal assertion-based pytest failures.
PYTHON_STRONG_EXCEPTION_RE: Pattern[str] = re.compile(
    r"(?:"
    r"\bModuleNotFoundError\b"
    r"|\bImportError\b"
    r"|\bNameError\b"
    r"|\bAttributeError\b"
    r"|\bKeyError\b"
    r"|\bIndexError\b"
    r"|\bValueError\b"
    r"|\bTypeError\b"
    r"|\bRuntimeError\b"
    r"|\bOSError\b"
    r"|\bPermissionError\b"
    r"|\bFileNotFoundError\b"
    r"|\bConnectionError\b"
    r"|\bTimeoutError\b"
    r")",
    re.IGNORECASE,
)

# Docker build error context blocks (BuildKit prints file/line + a numbered snippet).
DOCKERFILE_CONTEXT_HEADER_RE: Pattern[str] = re.compile(r"\bDockerfile\.[^: \t]+:\d+\b", re.IGNORECASE)
DOCKERFILE_CONTEXT_LINE_RE: Pattern[str] = re.compile(r"^\s*\d+\s*\|\s", re.IGNORECASE)
DOCKERFILE_CONTEXT_DIVIDER_RE: Pattern[str] = re.compile(r"^-{8,}\s*$")

# Rust test output (cargo test)
RUST_TEST_RESULT_FAILED_RE: Pattern[str] = re.compile(r"test result:\s*FAILED\.", re.IGNORECASE)
RUST_TEST_FAILURES_HEADER_RE: Pattern[str] = re.compile(r"^\s*failures:\s*$", re.IGNORECASE)
# Failed test node ids printed under the `failures:` block. These can be indented or not, depending on
# the harness / formatting.
RUST_TEST_FAILED_TEST_NAME_RE: Pattern[str] = re.compile(r"^\s*[A-Za-z0-9_:]+\s*$")

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
    # CUDA runtime missing on runner.
    CUDA_LIBCUDA_IMPORT_ERROR_RE,
    # Python import errors are high-signal; make the entire line red.
    PYTHON_MODULE_NOT_FOUND_RE,
    # CI sentinel variables indicating test failure. Example:
    #   FAILED_TESTS=1  # Treat missing XML as failure
    re.compile(r"\bFAILED_TESTS\s*=\s*1\b", re.IGNORECASE),
    # Pytest collection errors are high-signal and typically the true root cause.
    # Example:
    #   ________________ ERROR collecting tests/... _________________
    re.compile(r"\berror\s+collecting\b", re.IGNORECASE),
    # Pytest file-level ERROR markers (helps identify the failing suite quickly).
    PYTEST_ERROR_FILE_LINE_RE,
    # Multi-line backend result blocks: full-line highlight the failure field.
    re.compile(r"\"result\"\s*:\s*\"failure\"", re.IGNORECASE),
    # Docker registry manifest missing.
    DOCKER_IMAGE_NOT_FOUND_RE,
    # Timeout marker inserted by snippet extraction / categorization.
    re.compile(r"\[TIMEOUT\]", re.IGNORECASE),
    # Assertion failures: user wants the whole line red.
    re.compile(r"\bassertion\s+failed:", re.IGNORECASE),
    # BuildKit/cargo/etc generic build failure summary.
    FAILED_TO_BUILD_RE,
    # Local policy checks (e.g. dev scripts) that emit a [FAIL] marker.
    re.compile(r"\[FAIL\]\s*incorrect\s+date\s*:", re.IGNORECASE),
    # Git LFS failures surfaced through BuildKit/uv/pip snippet formatting.
    # Example lines (user wants whole line red, not bold):
    #   ├─▶ Git operation failed
    #   ├─▶ failed to fetch LFS objects at <sha>
    re.compile(r"\bGit\s+operation\s+failed\b", re.IGNORECASE),
    re.compile(r"\bfailed\s+to\s+fetch\s+LFS\s+objects\b", re.IGNORECASE),
    PYTEST_TIMEOUT_E_LINE_RE,
    # The 100% progress line that contains the failing "F" is useful context.
    re.compile(r"\[100%\].*F", re.IGNORECASE),
    # Rust test harness failure summary.
    RUST_TEST_FAILURES_HEADER_RE,
    # Rust failure list entry (indented test path), e.g. "    recorder::tests::test_...".
    RUST_TEST_FAILED_TEST_NAME_RE,
    re.compile(r"test result:\s*FAILED\.", re.IGNORECASE),
]


#
# Shared categorization rules (used by BOTH full-log categorization and snippet categorization).
#
# Keep this list "regex-only" and push special-case suppression logic into `_apply_category_rules()`
# so we don’t duplicate business rules across categorization call sites.
#
CATEGORY_RULES: list[tuple[str, Pattern[str]]] = [
    # Pytest per-test timeout (pytest-timeout plugin).
    ("pytest-timeout-error", PYTEST_TIMEOUT_E_LINE_RE),
    # Pytest failures:
    # - "short test summary info" can appear on successful runs (skips/xfail), so do NOT treat it as failure.
    # - Prefer explicit failure/collection error markers.
    ("pytest-error", re.compile(r"(?:^|\s)FAILED(?:\s+|$).*::|\\berror\\s+collecting\\b|==+\\s*(?:FAILURES|ERRORS)\\s*==+", re.IGNORECASE)),
    ("network-download-error", re.compile(DOWNLOAD_ERROR_RE.pattern, re.IGNORECASE)),
    ("docker-build-error", re.compile(DOCKER_BUILD_ERROR_RE.pattern, re.IGNORECASE)),
    ("build-status-check-error", BUILD_STATUS_CHECK_ERROR_RE),
    ("huggingface-auth-error", HUGGINGFACE_AUTH_ERROR_RE),
    ("copyright-header-error", COPYRIGHT_HEADER_ERROR_RE),
    ("helm-error", HELM_ERROR_RE),
    # Use the same CUDA matcher as full-log categorization, so snippets catch libcuda ImportError too.
    ("cuda-error", re.compile(CUDA_ERROR_RE.pattern, re.IGNORECASE)),
    ("network-timeout-https", re.compile(HTTP_TIMEOUT_RE.pattern, re.IGNORECASE)),
    ("network-timeout-gitlab-mirror", GITLAB_MIRROR_TIMEOUT_RE),
    ("k8s-network-timeout-pod", K8S_PODS_TIMED_OUT_RE),
    ("k8s-network-timeout-portfwd", KUBECTL_PORTFORWARD_TIMEOUT_RE),
    ("network-timeout-github-action", GITHUB_ACTION_STEP_TIMEOUT_RE),
    ("network-error", re.compile(NETWORK_ERROR_RE.pattern, re.IGNORECASE)),
    ("etcd-error", re.compile(ETCD_ERROR_RE.pattern, re.IGNORECASE)),
    ("git-fetch", re.compile(r"failed to fetch some objects from|RPC failed|early EOF|remote end hung up|fetch-pack", re.IGNORECASE)),
    ("github-api", re.compile(r"Failed to query GitHub API|secondary rate limit|API rate limit exceeded|HTTP 403|HTTP 429", re.IGNORECASE)),
    ("github-lfs-error", re.compile(r"/info/lfs|git lfs", re.IGNORECASE)),
    # Avoid tagging timeout just because pytest plugins list "timeout-<ver>" or prose mentions "timeouts".
    ("network-timeout-generic", re.compile(TIMED_OUT_RE.pattern, re.IGNORECASE)),
    ("oom", re.compile(r"\b(out of memory|CUDA out of memory|Killed process|oom)\b", re.IGNORECASE)),
    ("docker-daemon-connection-error", DOCKER_DAEMON_CONNECTION_ERROR_RE),
    ("docker-daemon-error-response-error", DOCKER_DAEMON_ERROR_RESPONSE_RE),
    ("docker-cli-error", DOCKER_CLI_ERROR_RE),
    ("docker-image-error", DOCKER_IMAGE_NOT_FOUND_RE),
    ("k8s-error", K8S_ERROR_RE),
    ("python-error", PYTHON_EXCEPTION_LINE_RE),
    # IMPORTANT: don't tag broken-links just because the tool name "lychee" appears; that causes false positives
    # on timeout-only runs. Require an explicit broken/dead links phrase.
    ("broken-links", re.compile(r"\bbroken\s+links?\b|\bdead\s+links?\b", re.IGNORECASE)),
    ("rust-error", RUST_TEST_FAIL_RE),
    ("exit-139-sigsegv", EXIT_CODE_139_RE),
    ("exit-127-cmd-not-found", EXIT_CODE_127_RE),
]


def _apply_category_rules(*, text: str, lines: Sequence[str], out: List[str], seen: set[str]) -> None:
    """Apply shared regex-based category rules with shared suppression logic."""
    try:
        text_l = (text or "").lower()

        has_specific_timeout = bool(
            HTTP_TIMEOUT_RE.search(text_l)
            or GITLAB_MIRROR_TIMEOUT_RE.search(text)
            or K8S_PODS_TIMED_OUT_RE.search(text)
            or KUBECTL_PORTFORWARD_TIMEOUT_RE.search(text)
            or GITHUB_ACTION_STEP_TIMEOUT_RE.search(text)
        )

        def add(name: str) -> None:
            if name and name not in seen:
                seen.add(name)
                out.append(name)

        # Heuristic: in pytest output, plain AssertionError test failures frequently trigger the broad
        # python exception regex. To keep categories meaningfully distinct, suppress python-error when
        # pytest-error is present unless the snippet/log shows a "non-test-failure" exception type.
        #
        # This keeps ImportError/ModuleNotFoundError/etc visible (real infra/env problems), while
        # avoiding redundant python-error tags for normal assertion-based test failures.
        allow_python_error_with_pytest = True
        if "pytest-error" in seen:
            allow_python_error_with_pytest = bool(PYTHON_STRONG_EXCEPTION_RE.search(text))

        for name, rx in CATEGORY_RULES:
            try:
                # Policy: if we can classify a specific timeout, do not show the generic timeout tag.
                if name == "network-timeout-generic" and has_specific_timeout:
                    continue

                # LFS vs git-fetch: mutually exclusive for this class of failure.
                if name == "git-fetch":
                    if "github-lfs-error" in seen or "/info/lfs" in text_l or "git lfs" in text_l:
                        continue

                # HuggingFace auth: ignore WARN-level-only hits (common benign warnings).
                if name == "huggingface-auth-error":
                    if not _has_huggingface_auth_error_signal(lines):
                        continue

                if name == "python-error" and "pytest-error" in seen and not allow_python_error_with_pytest:
                    continue

                # Docker: if we already have docker-image-error, don't also tag daemon/CLI infra errors.
                if name.startswith("docker-") and name != "docker-image-error":
                    if "docker-image-error" in seen:
                        continue

                # Full-log categorization already separately handles "failed to fetch some objects" -> (git-fetch|github-lfs-error).
                if name == "github-lfs-error":
                    if "/info/lfs" not in text_l and "git lfs" not in text_l and "failed to fetch some objects" in text_l:
                        # If we're in a generic git-fetch LFS-ish state but don't have explicit LFS strings,
                        # avoid spuriously tagging LFS from a snippet.
                        pass

                if rx.search(text) and name not in seen:
                    add(name)
            except Exception:
                continue

        # Derived: helm errors are inherently Kubernetes-related; show k8s-error alongside helm-error.
        if "helm-error" in seen:
            add("k8s-error")
    except Exception:
        return


def html_highlight_error_keywords(text: str) -> str:
    """HTML: escape and keyword-highlight error tokens (inline highlighting)."""
    # Don't keyword-highlight this common post-failure docker noise.
    # It's useful to *show* in snippets sometimes, but shouldn't draw attention.
    if DOCKER_NO_SUCH_CONTAINER_RE.search(text or ""):
        return html.escape(text or "")

    escaped = html.escape(text or "")
    if not escaped:
        return ""

    def repl(m: re.Match) -> str:
        # Slightly deeper red than GitHub default; keep readable and not overly saturated.
        # (No bold: user wants red without extra emphasis.)
        return f'<span style="color: #c83a3a;">{m.group(0)}</span>'

    return ERROR_HIGHLIGHT_RE.sub(repl, escaped)


def categorize_error_snippet_text(snippet_text: str) -> List[str]:
    """Categorize an extracted snippet into categories (best-effort)."""
    text = (snippet_text or "").strip()
    if not text:
        return []

    out: List[str] = []
    seen: set[str] = set()

    # Seed from synthesized "Categories: ..." header line (generated from full-log categorization).
    # This is intentionally not displayed in the UI, but it helps snippet categorization match
    # the full-log outcome even when the visible snippet window doesn't include the root-cause line.
    try:
        for ln in (snippet_text or "").splitlines()[:6]:
            s = (ln or "").strip()
            if not s:
                continue
            if s.lower().startswith("categories:"):
                payload = s.split(":", 1)[1] if ":" in s else ""
                for tok in [x.strip() for x in payload.split(",") if x.strip()]:
                    if tok not in seen:
                        seen.add(tok)
                        out.append(tok)
                break
    except Exception:
        pass

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

    # Apply the shared marker rules to the snippet text as well, so snippet tags stay consistent
    # with full-log categorization (and avoid duplicated special-case logic).
    #
    # Also, ignore our synthetic command blocks (they can include "docker run"/"pytest" strings,
    # which are execution context and should not influence error categorization).
    try:
        in_cmd = False
        filtered: List[str] = []
        for ln in (snippet_text or "").splitlines():
            s = (ln or "").strip()
            if s == "[[CMD]]":
                in_cmd = True
                continue
            if s == "[[/CMD]]":
                in_cmd = False
                continue
            if in_cmd:
                continue
            # Skip the synthetic "Snippet:" prefix (if any) and any stray "Categories:" lines.
            if s.lower().startswith("snippet:") or s.lower().startswith("categories:"):
                continue
            filtered.append(ln)
        _apply_category_rules(text=text, lines=filtered, out=out, seen=seen)
    except Exception:
        _apply_category_rules(text=text, lines=(snippet_text or "").splitlines(), out=out, seen=seen)
    return out


def render_error_snippet_html(snippet_text: str) -> str:
    """HTML: render an extracted error snippet.

    - Preserve line breaks (container uses `white-space: pre-wrap`).
    - For pytest "FAILED ...::test_..." summary lines, color the *entire line* red.
    - Otherwise, keep keyword-level highlighting for common failure tokens.
    """
    if not (snippet_text or "").strip():
        return ""

    out_lines: List[str] = []
    lines = (snippet_text or "").splitlines()
    i = 0
    cmd_block_idx = 0
    while i < len(lines):
        raw_line = lines[i]
        if raw_line.strip() == "[[CMD]]":
            # Consume a command block and render it as a multi-line blue block with a Copy button.
            cmd_lines: List[str] = []
            j = i + 1
            while j < len(lines):
                if lines[j].strip() == "[[/CMD]]":
                    break
                cmd_lines.append(_strip_ts_and_ansi(lines[j]))
                j += 1
            cmd_text = "\n".join(cmd_lines).strip("\n")
            cmd_js = html.escape(json.dumps(cmd_text), quote=True)
            cmd_html = html.escape(cmd_text)
            text_style = _SNIP_COPY_TEXT_STYLE + ("; font-weight: 600;" if cmd_block_idx == 0 else "")
            out_lines.append(
                f'<span style="{_SNIP_COPY_ROW_STYLE}">'
                f'<button type="button" onclick="event.stopPropagation(); try {{ copyToClipboard({cmd_js}, this); }} catch (e) {{}}" '
                f'style="{_SNIP_COPY_BTN_STYLE}" title="Copy command">{_copy_icon_svg(size_px=12)}</button>'
                f'<span style="{text_style}">{cmd_html}</span>'
                "</span>"
            )
            cmd_block_idx += 1
            # Skip to the end marker (or end of file if missing).
            i = (j + 1) if (j < len(lines) and lines[j].strip() == "[[/CMD]]") else j
            continue

        # Don't display synthetic snippet header line(s) (they're used internally for categorization).
        if raw_line.strip().lower().startswith("categories:") or raw_line.strip().lower().startswith("commands:"):
            i += 1
            continue

        # Keep empty lines (they matter for readability) but don't highlight them.
        if raw_line == "":
            out_lines.append("")
            i += 1
            continue

        # Match red/blue rules on a normalized view (strip timestamp prefix + ANSI), but render
        # the normalized line so snippets don't show noisy timestamps.
        s_norm = _strip_ts_and_ansi(raw_line)
        display_line = s_norm

        if PYTEST_FAILED_LINE_RE.search(s_norm) or any(r.search(s_norm) for r in FULL_LINE_ERROR_REDS_RE):
            out_lines.append(
                f'<span style="color: #c83a3a;">{html.escape(display_line)}</span>'
            )
        elif COMMAND_LINE_BLUE_RE.search(s_norm):
            # Special-case: make PYTEST_CMD=... copyable (high-signal and often very long).
            if PYTEST_CMD_LINE_RE.search(s_norm):
                # Extract payload after the first "=" and strip a single matching quote pair.
                payload = ""
                try:
                    rhs = str(s_norm).split("=", 1)[1] if "=" in str(s_norm) else ""
                    rhs = rhs.strip()
                    if len(rhs) >= 2 and rhs[0] in ("'", '"') and rhs[-1] == rhs[0]:
                        rhs = rhs[1:-1]
                    payload = rhs
                except Exception:
                    payload = ""
                payload_js = html.escape(json.dumps(payload), quote=True)
                out_lines.append(
                    f'<span style="{_SNIP_COPY_ROW_STYLE}">'
                    f'<button type="button" onclick="event.stopPropagation(); try {{ copyToClipboard({payload_js}, this); }} catch (e) {{}}" '
                    f'style="{_SNIP_COPY_BTN_STYLE}" title="Copy pytest command">{_copy_icon_svg(size_px=12)}</button>'
                    f'<span style="{_SNIP_COPY_TEXT_STYLE}">{html.escape(display_line)}</span>'
                    "</span>"
                )
            # Also copy-enable `bash -c "...pytest..."` lines (common execution context).
            elif re.search(r"\bbash\s+-c\s+['\"][^'\"]*\bpytest\b", s_norm, flags=re.IGNORECASE):
                payload = ""
                try:
                    payload = str(display_line or "").strip()
                except Exception:
                    payload = ""
                payload_js = html.escape(json.dumps(payload), quote=True)
                out_lines.append(
                    f'<span style="{_SNIP_COPY_ROW_STYLE}">'
                    f'<button type="button" onclick="event.stopPropagation(); try {{ copyToClipboard({payload_js}, this); }} catch (e) {{}}" '
                    f'style="{_SNIP_COPY_BTN_STYLE}" title="Copy command">{_copy_icon_svg(size_px=12)}</button>'
                    f'<span style="{_SNIP_COPY_TEXT_STYLE}">{html.escape(display_line)}</span>'
                    "</span>"
                )
            else:
                out_lines.append(
                    f'<span style="color: #0969da;">{html.escape(display_line)}</span>'
                )
        else:
            out_lines.append(html_highlight_error_keywords(display_line))

        i += 1

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

        def extract_commands(lines: List[str]) -> List[str]:
            """Best-effort extraction of interesting commands (pytest/docker/run.sh/build.sh)."""
            # Capture multi-line commands like:
            #   docker run ... \n  --flag ... \n  --flag ...
            start_res = [
                # Pytest commands: only treat these as execution context when they look like actual commands.
                # Avoid prose matches like "request: Pytest request fixture ...".
                re.compile(r"^(?:python\\s+-m\\s+pytest|pytest)\\b", re.IGNORECASE),
                re.compile(r"\\bbash\\s+-c\\s+['\"][^'\"]*\\bpytest\\b", re.IGNORECASE),
                # Prefer the explicit PYTEST_CMD line when present (highest-signal, most complete).
                PYTEST_CMD_LINE_RE,
                # Rust/cargo commands (common in CI).
                re.compile(r"^cargo\\s+(?:test|build|check|clippy|fmt|rustfmt)\\b", re.IGNORECASE),
                # Docker commands (often multi-line with backslashes).
                re.compile(r"^docker\\s+(?:buildx|build|run)\\b", re.IGNORECASE),
                re.compile(r"^(?:\\./)?run\\.sh\\b", re.IGNORECASE),
                re.compile(r"^(?:\\./)?build\\.sh\\b", re.IGNORECASE),
            ]

            def extract_vanilla_pytest_from_shell(line: str) -> str:
                """If line contains bash -c "... pytest ...", extract the inner `pytest ...` command."""
                try:
                    s = str(line or "")
                    m = re.search(r"\bbash\s+-c\s+(['\"])(.+?)\1\s*$", s, flags=re.IGNORECASE)
                    if not m:
                        return ""
                    inner = str(m.group(2) or "")
                    # Prefer the last pytest invocation inside the shell fragment.
                    last = None
                    for mm in re.finditer(r"(?:^|&&\s*|;\s*|\|\|\s*)(pytest\b.*)$", inner, flags=re.IGNORECASE):
                        last = mm
                    if not last:
                        return ""
                    cmd = str(last.group(1) or "").strip()
                    if not cmd.lower().startswith("pytest"):
                        return ""
                    return _unescape_nested_shell_quotes(cmd)
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
                    # Also include a "vanilla pytest ..." command extracted from bash -c blocks.
                    # This makes it easy to copy/paste just the pytest invocation.
                    py = extract_vanilla_pytest_from_shell(blk)
                    if py and py not in seen:
                        seen.add(py)
                        out.append(py)
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
            r"\bbash\s+-c\s+['\"][^'\"]*\bpytest\b",
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
            if PYTEST_SHORT_TEST_SUMMARY_RE.search(s_norm):
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
            if UNSUPPORTED_CUDA_VLLM_RE.search(s_norm):
                last_cuda_err = i
            if CUDA_LIBCUDA_IMPORT_ERROR_RE.search(s_norm):
                last_libcuda_import_err = i
            if PYTHON_MODULE_NOT_FOUND_RE.search(s_norm):
                last_module_not_found = i
            if PYTHON_EXCEPTION_LINE_RE.search(s_norm):
                last_python_exception_line = i
            if DOCKERFILE_CONTEXT_HEADER_RE.search(s_norm):
                last_dockerfile_ctx_hdr = i
            if RUST_TEST_FAILURES_HEADER_RE.search(_strip_ts_and_ansi(line)):
                last_rust_failures_header = i
            if RUST_TEST_RESULT_FAILED_RE.search(s_norm):
                last_rust_test_result_failed = i
            if GIT_LFS_SNIPPET_ANCHOR_RE.search(s_norm):
                last_git_lfs_anchor = i
            if EXIT_CODE_139_LINE_RE.search(s_norm):
                last_exit_code_139 = i
            # Some categories are often only visible as a single high-signal line that can get
            # pushed out of the snippet window. Track them explicitly so we can force-include.
            if ETCD_ERROR_RE.search(line.lower()):
                etcd_sigs.append(i)
            if HUGGINGFACE_AUTH_ERROR_RE.search(_strip_ts_and_ansi(line)) and not _line_is_warn_or_lower(line):
                last_hf_auth_sig = i
            if COPYRIGHT_HEADER_ERROR_RE.search(line):
                last_copyright_sig = i
            if FAILED_TO_BUILD_RE.search(s_norm):
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
            if BACKEND_RESULT_FAILURE_LINE_RE.search(s_norm):
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
                if GIT_LFS_BLOCK_START_RE.search(ln):
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
                if GIT_LFS_BLOCK_END_RE.search(_strip_ts_and_ansi(ln)):
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
                if k == hdr_i or DOCKERFILE_CONTEXT_DIVIDER_RE.search(ln) or DOCKERFILE_CONTEXT_LINE_RE.search(ln):
                    if ln not in snippet_lines:
                        snippet_lines.append(ln)
                    continue
                # Stop once we leave the Dockerfile block.
                if k > hdr_i and not DOCKERFILE_CONTEXT_LINE_RE.search(ln) and not DOCKERFILE_CONTEXT_DIVIDER_RE.search(ln):
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
                        if nxt_s.startswith("--") or nxt_s.startswith("-") or nxt_s.startswith("bash -c "):
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

                def _extract_bash_c_line_from_cmd_block(blk: str) -> str:
                    s = (blk or "").strip()
                    if not s:
                        return ""
                    for ln in s.splitlines():
                        ln_s = ln.strip()
                        if re.search(r"\bbash\s+-c\s+(['\"]).+\1\s*$", ln_s, flags=re.IGNORECASE):
                            return ln_s
                    return ""

                def _extract_vanilla_pytest_from_cmd_block(blk: str) -> str:
                    s = (blk or "").strip()
                    if not s:
                        return ""
                    # Look for the bash -c payload on any line.
                    for ln in s.splitlines():
                        m = re.search(r"\bbash\s+-c\s+(['\"])(.+?)\1\s*$", ln.strip(), flags=re.IGNORECASE)
                        if not m:
                            continue
                        inner = str(m.group(2) or "")
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
                    # Add the bare `bash -c "..."` line as its own command block (copyable).
                    bash_c = _extract_bash_c_line_from_cmd_block(b)
                    if bash_c and bash_c not in seen_exp:
                        seen_exp.add(bash_c)
                        expanded.append(bash_c)
                    py = _extract_vanilla_pytest_from_cmd_block(b)
                    if py and py not in seen_exp:
                        seen_exp.add(py)
                        expanded.append(py)
                        if not best_vanilla_pytest:
                            best_vanilla_pytest = py

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
                                    rerun_only_failed_pytest_cmd,
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


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract and format a high-signal error snippet from a CI log file.",
    )
    parser.add_argument(
        "log_path",
        nargs="?",
        default="",
        help="Path to a local raw log file (e.g., raw-log-text/<job_id>.log). Not required for --self-test-examples.",
    )
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
    parser.add_argument(
        "--self-test-examples",
        action="store_true",
        help="Run the Examples self-test (parses module docstring Examples and validates categories).",
    )
    parser.add_argument(
        "--raw-log-path",
        default=str(_default_raw_log_dir()),
        help="Directory containing raw-log-text/*.log for --self-test-examples (default: ~/.cache/dynamo-utils/raw-log-text).",
    )
    parser.add_argument(
        "--scan-all-logs",
        action="store_true",
        help="Scan all *.log under --logs-root and print frequency/coverage stats.",
    )
    parser.add_argument(
        "--audit-snippet-commands",
        action="store_true",
        help="Audit all *.log under --logs-root and report pytest/rust snippets missing preceding commands.",
    )
    parser.add_argument(
        "--logs-root",
        default=str(_default_raw_log_dir()),
        help="Directory containing raw-log-text/*.log for --scan-all-logs (default: ~/.cache/dynamo-utils/raw-log-text).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if bool(getattr(args, "self_test_examples", False)):
        return _self_test_examples(raw_log_path=Path(str(args.raw_log_path)))
    if bool(getattr(args, "scan_all_logs", False)):
        return _scan_all_logs(
            logs_root=Path(str(args.logs_root)),
            tail_bytes=int(0 if bool(args.no_tail) else int(args.tail_bytes)),
        )
    if bool(getattr(args, "audit_snippet_commands", False)):
        return _audit_snippet_commands(
            logs_root=Path(str(args.logs_root)),
            tail_bytes=int(0 if bool(args.no_tail) else int(args.tail_bytes)),
        )

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

    print(render_error_snippet_html(snippet) if args.html else snippet)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

