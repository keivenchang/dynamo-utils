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
  - `python3 dynamo-utils/common_log_errors.py --self-test-examples`
  - This parses the "Examples:" list above, loads each log from `../raw-log-text/`, and reports
    missing/extra categories for both full-log categorization and snippet-derived categorization.
- If mismatches show up, adjust categorization/snippet anchors until the example logs match again,
  then re-run the self-test until it’s clean.

Snippet output assertions (extra self-test)
------------------------------------------
Grammar:
  * `<job_id>.log => +must_contain1, +must_contain2, !must_not_contain1, !must_not_contain2`
Notes:
  - These assertions validate snippet **text output** (not HTML).
  - Prefer stable substrings (avoid volatile IDs/timings).

* 59520885010.log => +docker run -w /workspace, +bash -c "pytest, +FAILED tests/router/test_router_e2e_with_mockers.py::test_router_decisions_disagg, !2026-
* 56700029731.log => +docker run --runtime=nvidia, +mkdir -p /workspace/test-results && pytest -v --tb=short, !pytest     = <module 'pytest'
* 59539777738.log => +[FAIL] incorrect date:, !2026-
* 59540519012.log => +docker buildx create --name builder-, +docker buildx inspect --bootstrap, !2026-
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Pattern, Sequence, Tuple

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
            # Explicit failing test id lines.
            if PYTEST_FAILED_LINE_RE.search(s):
                return True
            # Pytest's per-file error marker.
            if PYTEST_ERROR_FILE_LINE_RE.search(s):
                return True
            # Summary-style output: only treat non-zero failures/errors as a failure signal.
            s2 = _BUILDKIT_STEP_PREFIX_RE.sub("", s)
            s2 = _BUILDKIT_TIME_PREFIX_RE.sub("", s2)
            if _PYTEST_NONZERO_FAIL_OR_ERROR_COUNT_RE.search(s2):
                # Guard against false positives from k8s tables like:
                #   "0/1     Error              5 (3m ago)"
                # which can match "1 Error" as "1 error".
                low = s2.lower()
                if (
                    ("passed" in low)
                    or ("skipped" in low)
                    or ("deselected" in low)
                    or ("warnings" in low)
                    or re.search(r"\bin\s+\d", low)  # e.g. "in 12.34s"
                    or ("short test summary info" in low)
                    or ("====" in low)
                ):
                    return True
            # Collection errors are always failures.
            if re.search(r"\berror[ \t]+collecting\b", s, flags=re.IGNORECASE):
                return True
            # Traditional section headers.
            if re.search(r"==+\s*(FAILURES|ERRORS)\s*==+", s, flags=re.IGNORECASE):
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


def _parse_snippet_assertions_from_docstring() -> list[tuple[str, list[str], list[str]]]:
    """Parse snippet output assertions from the module docstring.

    Grammar:
      * `<file>.log => +must_contain1, +must_contain2, !must_not_contain1, !must_not_contain2`
    """
    out: list[tuple[str, list[str], list[str]]] = []
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
        for tok in tokens:
            if tok.startswith("+"):
                must.append(tok[1:].strip())
            elif tok.startswith("!"):
                must_not.append(tok[1:].strip())
        if log_name and (must or must_not):
            out.append((log_name, must, must_not))
    return out


def _self_test_examples(*, examples_root: Path) -> int:
    """Self-test: load the example logs and report category match coverage."""
    examples = _parse_examples_from_docstring()
    if not examples:
        print("Self-test: no Examples found in module docstring.")
        return 2
    snippet_assertions = _parse_snippet_assertions_from_docstring()

    root = Path(examples_root).expanduser().resolve()
    missing_files: list[str] = []
    failures: int = 0

    print(f"Self-test: examples_root={root}")
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
        for log_name, must, must_not in snippet_assertions:
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
            missing: list[str] = []
            present_forbidden: list[str] = []
            for m in (must or []):
                if m and m not in (snip or ""):
                    missing.append(m)
            for f in (must_not or []):
                if f and f in (snip or ""):
                    present_forbidden.append(f)
            ok = (not missing) and (not present_forbidden)
            if not ok:
                failures += 1
            print(f"- {log_name}")
            if must:
                print(f"  must_contain: {', '.join(must)}")
            if must_not:
                print(f"  must_not:    {', '.join(must_not)}")
            if missing:
                print(f"  missing:     {', '.join(missing)}")
            if present_forbidden:
                print(f"  forbidden:   {', '.join(present_forbidden)}")
            print("")

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
RUST_TEST_FAILED_TEST_NAME_RE: Pattern[str] = re.compile(r"^\s+[A-Za-z0-9_:]+\s*$")

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
    re.compile(r"\\[TIMEOUT\\]", re.IGNORECASE),
    # Assertion failures: user wants the whole line red.
    re.compile(r"\\bassertion\\s+failed:\\b", re.IGNORECASE),
    # BuildKit/cargo/etc generic build failure summary.
    FAILED_TO_BUILD_RE,
    # Local policy checks (e.g. dev scripts) that emit a [FAIL] marker.
    re.compile(r"\\[FAIL\\]\\s*incorrect\\s+date\\s*:", re.IGNORECASE),
    # Pytest failure block lines (high-signal).
    PYTEST_FAILURES_HEADER_RE,
    PYTEST_UNDERSCORE_TITLE_RE,
    PYTEST_TIMEOUT_E_LINE_RE,
    # The 100% progress line that contains the failing "F" is useful context.
    re.compile(r"\[100%\].*F", re.IGNORECASE),
    # Rust test harness failure summary.
    re.compile(r"^\s*failures:\s*$", re.IGNORECASE),
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
    ("pytest-error", re.compile(r"(?:^|\s)FAILED(?:\s+|$).*::|short test summary info|\\berror\\s+collecting\\b", re.IGNORECASE)),
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
            out_lines.append(
                '<span style="display: block; margin: 2px 0; white-space: pre-wrap; overflow-wrap: anywhere; color: #0969da;">'
                f'<button type="button" onclick="event.stopPropagation(); try {{ copyToClipboard({cmd_js}, this); }} catch (e) {{}}" '
                'style="display: inline-block; padding: 1px 8px; font-size: 11px; line-height: 1.2; '
                'background: transparent; color: #0969da; border: 1px solid #0969da; border-radius: 999px; '
                'cursor: pointer; margin: 0 0 2px 0;" title="Copy command">Copy</button>\n'
                f"{cmd_html}"
                "</span>"
            )
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
                re.compile(r"^docker\\s+(?:build|run)\\b", re.IGNORECASE),
                re.compile(r"^(?:\\./)?run\\.sh\\b", re.IGNORECASE),
                re.compile(r"^(?:\\./)?build\\.sh\\b", re.IGNORECASE),
            ]

            def normalize_cmd_line(raw: str) -> str:
                s = _strip_ts_and_ansi(raw).strip()
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
            if NETWORK_ERROR_LINE_RE.search(s_norm):
                last_network_err = i
            if BACKEND_RESULT_FAILURE_LINE_RE.search(s_norm):
                last_backend_result_failure = i
            if ERROR_SNIPPET_LINE_RE.search(s_norm):
                last_generic = i

        # Choose anchor by priority (first non-None index wins).
        for idx in (
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
        try:
            cmd_blocks: List[str] = []
            try:
                cleaned = [_strip_ts_and_ansi(x).rstrip("\n") for x in (all_lines or [])]
                anchor_for_cmds = int(anchor_idx) if anchor_idx is not None else (len(cleaned) - 1)

                # Also capture docker buildx commands that appear as Actions "[command]" lines,
                # e.g. "[command]/usr/bin/docker buildx create ...". These are often critical
                # for debugging docker-build failures.
                try:
                    buildx_re = re.compile(
                        r"^(?:##\[(?:command)\]|\[(?:command)\])\s*/usr/bin/docker\s+buildx\b",
                        re.IGNORECASE,
                    )
                    buildx_lines: List[str] = []
                    for ln in cleaned[: anchor_for_cmds + 1]:
                        s = (ln or "").strip()
                        if not s:
                            continue
                        if not buildx_re.search(s):
                            continue
                        # Normalize: strip "[command]/usr/bin/" or "##[command]/usr/bin/" prefix.
                        s2 = re.sub(r"^(?:##\[(?:command)\]|\[(?:command)\])\s*/usr/bin/", "", s, flags=re.IGNORECASE)
                        if s2 and s2 not in buildx_lines:
                            buildx_lines.append(s2)
                    # Keep only the last few to avoid drowning the snippet.
                    buildx_lines = buildx_lines[-6:]
                    if buildx_lines:
                        cmd_blocks.append("\n".join(buildx_lines))
                except Exception:
                    pass

                # Find the last few "Run ..." command groups and keep the most relevant ones.
                # Many workflows use "Run # <comment>" headers, so we don't try to match the header text.
                idxs: List[int] = []
                for idx in range(len(cleaned) - 1, -1, -1):
                    s0 = (cleaned[idx] or "").strip()
                    if not s0.startswith("##[group]Run "):
                        continue
                    idxs.append(idx)
                    if len(idxs) >= 8:
                        break
                idxs.reverse()

                def _is_pytest_module_dump(ln: str) -> bool:
                    return bool(re.search(r"^\s*pytest\s*=\s*<module\s+'pytest'\b", ln or "", flags=re.IGNORECASE))

                want_cmd_start_re = re.compile(
                    r"^(?:docker\s+(?:run|buildx|build)\b|cargo\s+(?:test|build|check|clippy|fmt|rustfmt)\b|python\s+-m\s+pytest\b|pytest\b|bash\s+-c\s+['\"][^'\"]*\bpytest\b)",
                    re.IGNORECASE,
                )

                for idx in idxs:
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

                    # Collect the command block including continuation lines until a clear stop marker.
                    block: List[str] = []
                    k = start_k
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

            cmd_lines: List[str] = []
            for blk in (cmd_blocks or []):
                s = (blk or "").strip("\n")
                if not s.strip():
                    continue
                # Mark command blocks so HTML renderer can add copy buttons + multiline formatting.
                cmd_lines.append("[[CMD]]")
                cmd_lines.extend(s.splitlines())
                cmd_lines.append("[[/CMD]]")
                cmd_lines.append("...")

            if cmd_lines:
                cmd_lines.pop()  # trailing ellipsis
                tail = list(snippet_lines or [])
                if tail and tail[0].strip() != "...":
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
        "--examples-root",
        default=str((Path(__file__).resolve().parent.parent / "raw-log-text")),
        help="Root directory containing raw-log-text/*.log for --self-test-examples (default: ../raw-log-text).",
    )
    parser.add_argument(
        "--scan-all-logs",
        action="store_true",
        help="Scan all *.log under --logs-root and print frequency/coverage stats.",
    )
    parser.add_argument(
        "--logs-root",
        default=str((Path(__file__).resolve().parent.parent / "raw-log-text")),
        help="Directory containing raw-log-text/*.log for --scan-all-logs (default: ../raw-log-text).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if bool(getattr(args, "self_test_examples", False)):
        return _self_test_examples(examples_root=Path(str(args.examples_root)))
    if bool(getattr(args, "scan_all_logs", False)):
        return _scan_all_logs(
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

