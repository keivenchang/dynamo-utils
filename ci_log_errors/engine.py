#!/usr/bin/env python3
"""
Shared log error detection + categorization + snippet formatting utilities.

This module is intentionally dependency-light so it can be shared by:
- `dynamo-utils/common.py` (log/snippet extraction, cache logic)
- `dynamo-utils/html_pages/common_dashboard_lib.py` (HTML rendering for snippets/tags)

For golden examples, training data, and self-test documentation, see:
- ci_log_errors/golden_examples.txt

Run `python3 -m ci_log_errors --self-test-examples` to validate categorization rules
against the golden training set.
"""


from __future__ import annotations

import argparse
import functools
import html
import json
import multiprocessing
import os
import re
import shlex
import stat
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Pattern, Sequence, Tuple

# Grouped, prefixed regex definitions live in `ci_log_errors/regexes.py`.
from .regexes import (
    CAT_BACKEND_BLOCK_START_RE,
    CAT_BACKEND_RESULT_FAILURE_RE,
    CAT_BROKEN_LINKS_RE,
    CAT_BUILD_STATUS_CHECK_ERROR_RE,
    CAT_CI_FILTER_UNCOVERED_RE,
    CAT_COPYRIGHT_HEADER_ERROR_RE,
    CAT_CUDA_ERROR_RE,
    CAT_DOCKER_BUILD_ERROR_RE,
    CAT_DOCKER_CLI_ERROR_RE,
    CAT_DOCKER_DAEMON_CONNECTION_ERROR_RE,
    CAT_DOCKER_DAEMON_ERROR_RESPONSE_RE,
    CAT_DOCKER_INFRA_ERROR_RE,
    CAT_DOCKER_UPLOAD_ERROR_RE,
    CAT_DOWNLOAD_ERROR_RE,
    CAT_ETCD_ERROR_RE,
    CAT_EXIT_CODE_127_RE,
    CAT_EXIT_CODE_139_RE,
    CAT_GITHUB_ACTION_STEP_TIMEOUT_RE,
    CAT_GITHUB_API_RE,
    CAT_GITHUB_FETCH_RE,
    CAT_GITHUB_LFS_RE,
    CAT_GITLAB_MIRROR_TIMEOUT_RE,
    CAT_HELM_ERROR_RE,
    CAT_HUGGINGFACE_AUTH_ERROR_RE,
    CAT_HTTP_TIMEOUT_RE,
    CAT_INVALID_TASK_TYPE_RE,
    CAT_K8S_ERROR_RE,
    CAT_K8S_PODS_TIMED_OUT_RE,
    CAT_KUBECTL_PORTFORWARD_TIMEOUT_RE,
    CAT_NETWORK_ERROR_RE,
    CAT_NETWORK_PORT_CONFLICT_ERROR_RE,
    CAT_OOM_RE,
    CAT_PYTEST_DETECT_RE,
    CAT_PYTEST_ERROR_RE,
    CAT_RUST_TEST_FAIL_RE,
    CAT_RULES,
    CAT_TIMED_OUT_RE,
    RED_DOCKER_DAEMON_ERROR_LINE_RE,
    RED_DOCKER_NO_SUCH_CONTAINER_RE,
    RED_DOCKER_UPLOAD_ERROR_LINE_RE,
    RED_FULL_LINE_RES,
    RED_KEYWORD_HIGHLIGHT_RE,
    RED_NETWORK_ERROR_LINE_RE,
    SNIPPET_ANCHOR_LINE_RE,
    SNIPPET_BACKEND_RESULT_FAILURE_LINE_RE,
    SNIPPET_COMMAND_LINE_BLUE_RE,
    SNIPPET_CUDA_LIBCUDA_IMPORT_ERROR_RE,
    SNIPPET_DOCKERFILE_CONTEXT_DIVIDER_RE,
    SNIPPET_DOCKERFILE_CONTEXT_HEADER_RE,
    SNIPPET_DOCKERFILE_CONTEXT_LINE_RE,
    SNIPPET_DOCKER_IMAGE_NOT_FOUND_RE,
    SNIPPET_EXIT_CODE_139_LINE_RE,
    SNIPPET_SIGSEGV_LINE_RE,
    SNIPPET_FAILED_TO_BUILD_RE,
    SNIPPET_GIT_LFS_BLOCK_END_RE,
    SNIPPET_GIT_LFS_BLOCK_START_RE,
    SNIPPET_GIT_LFS_SNIPPET_ANCHOR_RE,
    SNIPPET_PYTEST_CMD_LINE_RE,
    SNIPPET_PYTEST_ERROR_FILE_LINE_RE,
    SNIPPET_PYTEST_FAILURES_HEADER_RE,
    SNIPPET_PYTEST_FAILED_LINE_RE,
    SNIPPET_PYTEST_PROGRESS_100_RE,
    SNIPPET_PYTEST_SHORT_TEST_SUMMARY_RE,
    SNIPPET_PYTEST_TIMEOUT_E_LINE_RE,
    SNIPPET_PYTEST_UNDERSCORE_TITLE_RE,
    SNIPPET_PYTHON_EXCEPTION_LINE_RE,
    SNIPPET_PYTHON_MODULE_NOT_FOUND_RE,
    SNIPPET_RUST_TEST_FAILURES_HEADER_RE,
    SNIPPET_RUST_TEST_FAILED_TEST_NAME_RE,
    SNIPPET_RUST_TEST_RESULT_FAILED_RE,
    SNIPPET_UNSUPPORTED_CUDA_VLLM_RE,
 )


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
    except (OSError, ValueError):
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
    except (AttributeError, ValueError):
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
    except (AttributeError, ValueError):
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
            if not CAT_HUGGINGFACE_AUTH_ERROR_RE.search(s):
                continue
            # If the line is explicitly WARN/INFO/etc, treat it as a warning, not a root-cause error tag.
            if _line_is_warn_or_lower(str(raw or "")):
                continue
            return True
    except (AttributeError, ValueError):
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
    except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
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
    """Parse the golden examples from ci_log_errors/golden_examples.txt.

    Grammar:
      - `<file>.log => cat1, cat2, !forbidden1, !forbidden2`
      - Inline comments after `#` are allowed and ignored (useful for occurrence counts).
    """
    out: list[tuple[str, list[str], list[str]]] = []
    
    # Read from external file instead of module docstring
    try:
        examples_file = Path(__file__).parent / "golden_examples.txt"
        doc = examples_file.read_text(encoding="utf-8").splitlines()
    except Exception:
        # Fall back to empty list if file doesn't exist
        return out
    
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
    """Parse snippet output assertions from ci_log_errors/golden_examples.txt.

    Grammar:
      * `<file>.log => +must_contain1, +must_contain2, !must_not_contain1, !must_not_contain2`
      * Optional HTML full-line-red assertions:
        - `+RED:<substr>` means the rendered snippet HTML must contain `<substr>` inside a full-line red span.
        - `!RED:<substr>` means the rendered snippet HTML must NOT contain `<substr>` inside a full-line red span.
    """
    out: list[tuple[str, list[str], list[str], list[str], list[str]]] = []
    
    # Read from external file instead of module docstring
    try:
        examples_file = Path(__file__).parent / "golden_examples.txt"
        doc = examples_file.read_text(encoding="utf-8").splitlines()
    except Exception:
        # Fall back to empty list if file doesn't exist
        return out
    
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


def _download_missing_golden_log(*, job_id: str, output_path: Path, owner: str = "ai-dynamo", repo: str = "dynamo") -> bool:
    """Download a missing golden log file from GitHub Actions.
    
    Args:
        job_id: The GitHub Actions job ID (e.g., "59030780729")
        output_path: Path where the log should be saved
        owner: GitHub repository owner (default: "ai-dynamo")
        repo: GitHub repository name (default: "dynamo")
        
    Returns:
        True if download succeeded, False otherwise
    """
    try:
        # Import here to avoid circular dependency
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from common_github import GitHubAPIClient
        
        # Get GitHub token using the same method as the rest of the codebase
        # (prefers ~/.github_token, falls back to GitHub CLI config)
        github_token = GitHubAPIClient.get_github_token_from_file()
        if not github_token:
            print(f"  [download] GitHub token not found (checked ~/.github_token and gh CLI), skipping download for {job_id}")
            return False
        
        print(f"  [download] Downloading log for job {job_id} from GitHub Actions...")
        api = GitHubAPIClient(token=github_token)
        
        # Construct a fake job_url (only the job ID is needed by get_job_raw_log_text_cached)
        job_url = f"https://github.com/{owner}/{repo}/actions/runs/0/job/{job_id}"
        
        # Download the raw log (assume_completed=True since these are golden historical logs)
        log_text = api.get_job_raw_log_text_cached(
            job_url=job_url,
            owner=owner,
            repo=repo,
            ttl_s=365 * 24 * 3600,  # Long TTL for golden logs
            assume_completed=True,  # Skip status check for historical logs
            max_bytes=32 * 1024 * 1024,  # 32MB max
        )
        
        if not log_text:
            print(f"  [download] Failed to download log for job {job_id} (API returned empty)")
            return False
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the log to disk
        output_path.write_text(log_text, encoding="utf-8")
        print(f"  [download] Successfully downloaded log for job {job_id} ({len(log_text)} bytes)")
        
        # Make it read-only (golden log policy)
        try:
            _ = _chmod_remove_write_bits(output_path)
        except Exception:
            pass
        
        return True
    except Exception as e:
        print(f"  [download] Error downloading log for job {job_id}: {type(e).__name__}: {e}")
        return False


def _self_test_examples(*, raw_log_path: Path, auto_download: bool = True) -> int:
    """Self-test: load the example logs and report category match coverage.
    
    Args:
        raw_log_path: Directory containing golden log files
        auto_download: If True, attempt to download missing golden logs from GitHub Actions
    """
    examples = _parse_examples_from_docstring()
    if not examples:
        print("Self-test: no Examples found in module docstring.")
        return 2
    snippet_assertions = _parse_snippet_assertions_from_docstring()

    root = Path(raw_log_path).expanduser().resolve()
    missing_files: list[str] = []
    downloaded_files: list[str] = []
    failures: int = 0

    print(f"Self-test: raw_log_path={root}")
    print(f"Self-test: cases={len(examples)}")
    print(f"Self-test: auto_download={'enabled' if auto_download else 'disabled'}")
    print("")

    for log_name, expected_raw, forbidden_raw in examples:
        p = (root / log_name).resolve()
        if not p.exists():
            # Try to download if auto_download is enabled
            if auto_download:
                # Extract job_id from log_name (e.g., "59030780729.log" -> "59030780729")
                job_id = str(log_name).replace(".log", "").strip()
                if job_id.isdigit():
                    if _download_missing_golden_log(job_id=job_id, output_path=p):
                        downloaded_files.append(log_name)
                        # Continue with testing if download succeeded
                    else:
                        missing_files.append(str(p))
                        continue
                else:
                    missing_files.append(str(p))
                    continue
            else:
                missing_files.append(str(p))
                continue

        # Golden-log workflow: keep the example logs read-only.
        # This should ONLY happen during the self-check process (per policy).
        try:
            _ = _chmod_remove_write_bits(p)
        except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
            pass

        from . import snippet as _snippet
        text = _snippet._read_text_tail(p, max_bytes=512 * 1024)
        all_lines = (text or "").splitlines()
        cats_full = [_norm_cat(x) for x in categorize_error_log_lines(all_lines)]
        cats_full_set = {c for c in cats_full if c}
        snip = _snippet.extract_error_snippet_from_text(text)
        from . import render as _render
        cats_snip = [_norm_cat(x) for x in _render.categorize_error_snippet_text(snip)]
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
                # Try to download if auto_download is enabled
                if auto_download:
                    job_id = str(log_name).replace(".log", "").strip()
                    if job_id.isdigit():
                        if _download_missing_golden_log(job_id=job_id, output_path=p):
                            if log_name not in downloaded_files:
                                downloaded_files.append(log_name)
                            # Continue with testing if download succeeded
                        else:
                            if str(p) not in missing_files:
                                missing_files.append(str(p))
                            continue
                    else:
                        if str(p) not in missing_files:
                            missing_files.append(str(p))
                        continue
                else:
                    if str(p) not in missing_files:
                        missing_files.append(str(p))
                    continue
            try:
                _ = _chmod_remove_write_bits(p)
            except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
                pass
            # Snippet assertions are sensitive to truncation. Use a larger tail budget than the
            # category self-test so we reliably capture preceding command blocks / workflow paths.
            from . import snippet as _snippet
            text = _snippet._read_text_tail(p, max_bytes=2 * 1024 * 1024)
            snip = _snippet.extract_error_snippet_from_text(text)
            from . import render as _render
            snip_html = _render.render_error_snippet_html(snip)
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
                except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
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
            except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
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
                "Exception: Failed to start HTTP server: port 8081 already in use. Use --http-port to specify a different port.",
                "E   Exception: Unable to create lease. Check etcd server status at http://localhost:2379",
                "Caught signal 11 (Segmentation fault: invalid permissions for mapped object at address 0x6aed2c30)",
                "Signal: Segmentation fault (11)",
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
        from . import render as _render
        unit_html = _render.render_error_snippet_html(unit_snip)

        must_be_red = [
            "E           Failed: Timeout (>60.0s) from pytest-timeout.",
            "assertion failed: (7..=13).contains(&elapsed_ms)",
            "ERROR: failed to build",
            "[FAIL] incorrect date: container/dev/dev_build.sh",
            "Git operation failed",
            "failed to fetch LFS objects",
            # Port bind failures (common e2e root cause)
            "Failed to start HTTP server: port 8081 already in use.",
            # Etcd lease failures
            "Unable to create lease. Check etcd server status",
            # SIGSEGV / caught signal lines
            "Caught signal 11",
            "Segmentation fault (11)",
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
    except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
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
        from . import snippet as _snippet
        unit_snip = _snippet.extract_error_snippet_from_text(unit_log, context_before=10, context_after=5, max_lines=120, max_chars=12000)
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
    except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
        pass

    # Extra categorization unit test: network port conflicts.
    try:
        print("Self-test: categorize (unit) network-port-conflict-error")
        unit_lines = [
            "ERROR    KVRouterProcess:managed_process.py:332 [PYTHON3] Exception: Failed to start HTTP server: port 8081 already in use. Use --http-port to specify a different port.",
        ]
        cats = categorize_error_log_lines(unit_lines)
        if "network-port-conflict-error" not in set(cats or []):
            failures += 1
            print("  missing: network-port-conflict-error")
            print(f"  got: {cats}")
        print("")
    except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
        pass

    # Extra golden regression guard: `PYTEST_CMD="pytest ..."` variable expansion should still produce
    # a suggested rerun-only-failed command, and it should appear after the FAILED chunk and before
    # the `====== N failed, ... ======` summary (note: this summary format is common in our logs).
    try:
        job_id = "57521050539"
        p = Path(raw_log_path) / f"{job_id}.log"
        if p.exists() and p.is_file():
            print("Self-test: rerun-only-failed (PYTEST_CMD) placement (golden)")
            from . import snippet as _snippet
            sn = _snippet.extract_error_snippet_from_log_file(
                p,
                tail_bytes=0,
                context_before=15,
                context_after=15,
                max_lines=160,
                max_chars=20000,
            )
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
                s0 = (ln or "").strip()
                # Suggested command lines are rendered as shell comments in snippet text:
                #   # suggested: pytest ...
                # Accept both plain and suggested formats (back-compat).
                s = re.sub(r"#\s*auto suggested\s*$", "", s0, flags=re.IGNORECASE).strip()
                # New format: "# [suggested] <cmd>" (also accept the old colon form for back-compat).
                s = re.sub(r"^\s*#\s*\[\s*suggested\s*\]\s*:?\s*", "", s, flags=re.IGNORECASE).strip()
                # Back-compat: "# suggested: <cmd>"
                s = re.sub(r"^\s*#\s*suggested\s*:\s*", "", s, flags=re.IGNORECASE).strip()
                # Old/plain: "# <cmd>" (or non-comment)
                s = re.sub(r"^\s*#\s*", "", s).strip()
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
    except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
        pass

    if downloaded_files:
        print(f"Self-test: downloaded {len(downloaded_files)} missing golden log(s):")
        for x in sorted(downloaded_files):
            print(f"  - {x}")
        print("")

    if missing_files:
        print("Self-test: missing example log files (could not download):")
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
    except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
        ids = set()
    _GOLDEN_JOB_IDS = set(ids)
    return set(_GOLDEN_JOB_IDS)


def is_golden_log_job_id(job_id: str) -> bool:
    try:
        j = str(job_id or "").strip()
        if not j.isdigit():
            return False
        return j in golden_log_job_ids()
    except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
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
    except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
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
    except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
        return False


def _scan_single_log(*, log_path: Path, tail_bytes: int, cache_file_path: Optional[str] = None) -> Tuple[List[str], List[str], bool, str]:
    """Process a single log file and return categorization results.
    
    Args:
        log_path: Path to the log file
        tail_bytes: Number of bytes to read from end
        cache_file_path: If provided, use snippet cache at this path (SHA-keyed, preserves old versions)
    
    Returns:
        (cats_full, cats_snip, has_snippet, log_name)
    """
    # Enforce log permissions:
    # - golden/training logs must remain non-writable
    # - all other logs should be user-writable (to allow cleanup/editing)
    try:
        job_id = str(log_path.stem or "")
        if is_golden_log_job_id(job_id):
            _ = _chmod_remove_write_bits(log_path)
        else:
            _ = _chmod_add_user_write_bit(log_path)
    except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
        pass

    from . import snippet as _snippet
    
    # Read full log for full-log categorization (always needed)
    txt = _snippet._read_text_tail(log_path, max_bytes=int(tail_bytes))
    lines = (txt or "").splitlines()
    cats_full = [_norm_cat(x) for x in categorize_error_log_lines(lines) if _norm_cat(x)]

    # For snippet: use cache if provided, otherwise compute directly
    if cache_file_path:
        try:
            # Import cache_snippet in worker process (acceptable: each worker needs its own cache instance).
            # This avoids serializing the cache object across process boundaries and ensures
            # each worker independently reads/writes the shared cache file with proper file locking.
            html_pages_dir = Path(__file__).parent.parent / "html_pages"
            if str(html_pages_dir) not in sys.path:
                sys.path.insert(0, str(html_pages_dir))
            from cache_snippet import SNIPPET_CACHE
            
            # get_or_compute handles: cache lookup (SHA-keyed), compute if miss, store result
            snippet_body, cats_snip = SNIPPET_CACHE.get_or_compute(raw_log_path=log_path)
            has_snippet = bool((snippet_body or "").strip())
            return (cats_full, cats_snip, has_snippet, log_path.name)
        except Exception as e:
            # Fall back to direct computation if cache fails
            print(f"Cache error for {log_path.name}: {e}", file=sys.stderr)
    
    # Direct computation without cache
    snip = _snippet.extract_error_snippet_from_text(txt)
    has_snippet = bool((snip or "").strip())
    
    from . import render as _render
    cats_snip = [_norm_cat(x) for x in _render.categorize_error_snippet_text(snip) if _norm_cat(x)]

    return (cats_full, cats_snip, has_snippet, log_path.name)


def _scan_all_logs(*, logs_root: Path, tail_bytes: int = 512 * 1024) -> int:
    """Scan a directory of `*.log` files and report categorization + snippet coverage.

    This is a practical "retrain/validate" helper:
    - validates that snippet anchoring finds something in most logs
    - summarizes category frequency (full-log + snippet-derived)

    Permission policy:
    - Golden training-example logs must be preserved (non-writable).
    - All other logs in the directory should be user-writable so they can be edited/cleaned up.
    - This scan enforces permissions on each `*.log` it touches (best-effort).
    
    Caching:
    - Uses SNIPPET_CACHE (keyed by ci_log_errors SHA) for snippet extraction
    - Preserves cache entries from different code versions (no invalidation)
    - Each worker process reads from and writes to the shared cache file (with file locking)
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
    
    # Detect CPU count
    try:
        cpu_count = multiprocessing.cpu_count()
    except NotImplementedError:
        cpu_count = 1
    
    # Check if snippet cache is available and get its SHA
    cache_file_path = None
    cache_sha = ""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "html_pages"))
        from cache_snippet import SNIPPET_CACHE
        # Force cache to load so we can get the SHA
        SNIPPET_CACHE._load_once()
        cache_sha = SNIPPET_CACHE._ci_log_errors_sha or ""
        cache_file_path = str(SNIPPET_CACHE._cache_file)
    except Exception as e:
        print(f"Warning: snippet cache unavailable ({e}), running without cache", file=sys.stderr)
    
    print(f"Scan-all: logs_root={root}")
    print(f"Scan-all: files={total}")
    print(f"Scan-all: workers={cpu_count}")
    print(f"Scan-all: cache={'enabled' if cache_file_path else 'disabled'}")
    if cache_file_path:
        print(f"Scan-all: cache_sha={cache_sha[:16]}..." if len(cache_sha) > 16 else f"Scan-all: cache_sha={cache_sha}")
        print(f"Scan-all: cache_file={cache_file_path}")
    print("")

    if total == 0:
        return 0

    cats_full_counts: Dict[str, int] = {}
    cats_snip_counts: Dict[str, int] = {}
    snippet_found = 0
    no_snippet_samples: List[str] = []

    # Parallel processing
    with multiprocessing.Pool(processes=cpu_count) as pool:
        # Create tasks for all log files
        tasks = [
            pool.apply_async(_scan_single_log, kwds={"log_path": p, "tail_bytes": tail_bytes, "cache_file_path": cache_file_path})
            for p in logs
        ]
        
        # Collect results with progress indicator
        processed = 0
        for task in tasks:
            cats_full, cats_snip, has_snippet, log_name = task.get()
            
            # Aggregate full-log categories
            for c in cats_full:
                cats_full_counts[c] = int(cats_full_counts.get(c, 0)) + 1

            # Aggregate snippet results
            if has_snippet:
                snippet_found += 1
            else:
                if len(no_snippet_samples) < 12:
                    no_snippet_samples.append(log_name)

            # Aggregate snippet-derived categories
            for c in cats_snip:
                cats_snip_counts[c] = int(cats_snip_counts.get(c, 0)) + 1
            
            processed += 1
            if processed % 100 == 0:
                print(f"Scan-all: processed {processed}/{total} logs...", file=sys.stderr, flush=True)

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

# Back-compat alias: the canonical name is `SNIPPET_ANCHOR_LINE_RE`.
ERROR_SNIPPET_LINE_RE: Pattern[str] = SNIPPET_ANCHOR_LINE_RE


#
# Categorization (text-only)
# =============================================================================
#

# Categorization regexes are defined in `ci_log_errors/regexes.py` under the `CAT_*` prefix.
# Keep legacy names here as aliases to avoid churn in call sites.
_BACKEND_BLOCK_START_RE: Pattern[str] = CAT_BACKEND_BLOCK_START_RE
_BACKEND_RESULT_FAILURE_RE: Pattern[str] = CAT_BACKEND_RESULT_FAILURE_RE


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
    except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
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
        _PYTHON_ERROR_DETECT_RE = SNIPPET_PYTHON_EXCEPTION_LINE_RE

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
        if CAT_RUST_TEST_FAIL_RE.search(text):
            add("rust-error")
        # Exit code 139 is conventionally SIGSEGV (signal 11): 128 + 11 = 139.
        if CAT_EXIT_CODE_139_RE.search(text):
            add("exit-139-sigsegv")
        # Exit code 127: command not found (missing dependency / PATH issue).
        if CAT_EXIT_CODE_127_RE.search(text):
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
        if CAT_DOWNLOAD_ERROR_RE.search(t):
            add("network-download-error")

        # Build failures (Docker/buildkit/etc.)
        if CAT_DOCKER_BUILD_ERROR_RE.search(t):
            add("docker-build-error")
        
        # Docker upload/push failures (registry PUT errors)
        if CAT_DOCKER_UPLOAD_ERROR_RE.search(t):
            add("docker-upload-error")

        # Build-status-check failures (CI gate that checks upstream build job status)
        if CAT_BUILD_STATUS_CHECK_ERROR_RE.search(text):
            add("build-status-check-error")
        
        # CI filters coverage validation (workflows/.github/filters.yaml)
        if CAT_CI_FILTER_UNCOVERED_RE.search(text):
            add("ci-filter-coverage-error")

        # Commit message policy (conventional commits / task type).
        if CAT_INVALID_TASK_TYPE_RE.search(text):
            add("invalid-task-type")

        # HuggingFace auth/token failures (missing/invalid HF_TOKEN or gated model access).
        #
        # IMPORTANT: filter out WARN-level lines so we don't mis-tag benign warnings as errors.
        if _has_huggingface_auth_error_signal(lines[-4000:] if lines else []):
            add("huggingface-auth-error")

        # Copyright header checks
        if CAT_COPYRIGHT_HEADER_ERROR_RE.search(text):
            add("copyright-header-error")

        # Helm/k8s workflow failures (deploy/cleanup)
        if CAT_HELM_ERROR_RE.search(text):
            add("helm-error")
        # Kubernetes signal (error) — keep this strict so we don't mis-tag logs that merely install
        # the Python `kubernetes` package.
        if CAT_K8S_ERROR_RE.search(text) or ("helm-error" in cats):
            add("k8s-error")

        # CUDA / GPU toolchain
        if CAT_CUDA_ERROR_RE.search(t):
            add("cuda-error")

        # HTTP(S) gateway timeouts (wget/curl/HTTP clients + link-checker timeouts)
        if CAT_HTTP_TIMEOUT_RE.search(t):
            add("network-timeout-https")
        # Local service bind failures / port conflicts (common root cause for e2e tests).
        if CAT_NETWORK_PORT_CONFLICT_ERROR_RE.search(text):
            add("network-port-conflict-error")
        # GitLab mirror infra timeout (special-case; keep distinct from generic timeout)
        if CAT_GITLAB_MIRROR_TIMEOUT_RE.search(text):
            add("network-timeout-gitlab-mirror")

        # Network connectivity
        if CAT_NETWORK_ERROR_RE.search(t):
            add("network-error")

        # Etcd / lease
        # Avoid tagging `etcd-error` just because the word "etcd" appears (it often shows up in benign logs).
        # Require a lease/status failure signature.
        if CAT_ETCD_ERROR_RE.search(t):
            add("etcd-error")

        # Docker infrastructure / daemon / CLI errors.
        #
        # IMPORTANT: `docker-image-error` is already specific (manifest unknown). If we see that,
        # don't also tag a redundant daemon error-response category.
        if not DOCKER_IMAGE_NOT_FOUND_RE.search(t):
            if CAT_DOCKER_DAEMON_CONNECTION_ERROR_RE.search(text):
                add("docker-daemon-connection-error")
            elif CAT_DOCKER_DAEMON_ERROR_RESPONSE_RE.search(text):
                add("docker-daemon-error-response-error")
            elif CAT_DOCKER_CLI_ERROR_RE.search(text):
                add("docker-cli-error")

        # Backend result JSON-ish blocks (vllm/sglang/trtllm): multi-line aware.
        engines = _backend_failure_engines_from_lines(lines[-4000:] if lines else [])
        if engines:
            add("backend-failure")
            for e in sorted(engines):
                add(f"{e}-error")

        # Broken links
        if CAT_BROKEN_LINKS_RE.search(t):
            add("broken-links")

        # Kubernetes wait timeouts ("timed out waiting for the condition on pods/...").
        if CAT_K8S_PODS_TIMED_OUT_RE.search(text):
            add("k8s-network-timeout-pod")

        # Kubernetes port-forward failures (portforward.go ... connection timed out).
        if CAT_KUBECTL_PORTFORWARD_TIMEOUT_RE.search(text):
            add("k8s-network-timeout-portfwd")

        # GitHub Actions step timeout markers.
        if CAT_GITHUB_ACTION_STEP_TIMEOUT_RE.search(text):
            add("network-timeout-github-action")

        # Timeout / infra flake
        #
        # Keep this very conservative to avoid false positives.
        if CAT_TIMED_OUT_RE.search(t) and not (
            CAT_HTTP_TIMEOUT_RE.search(t)
            or CAT_GITLAB_MIRROR_TIMEOUT_RE.search(text)
            or CAT_K8S_PODS_TIMED_OUT_RE.search(text)
            or CAT_KUBECTL_PORTFORWARD_TIMEOUT_RE.search(text)
            or CAT_GITHUB_ACTION_STEP_TIMEOUT_RE.search(text)
        ):
            add("network-timeout-generic")

        # Docker image not found (registry manifest unknown)
        if DOCKER_IMAGE_NOT_FOUND_RE.search(t):
            add("docker-image-error")

        # OOM / kill
        if re.search(r"\bout\s+of\s+memory\b|\boom\b|killed\s+process", t):
            add("oom")
    except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
        return cats

    return cats


#
# NOTE: Avoid forwarding wrappers here. Import/call the real implementations directly:
# - `ci_log_errors.snippet.*` for snippet extraction
# - `ci_log_errors.render.*` for HTML rendering + snippet categorization


#
# HTML highlighting + snippet formatting
# =============================================================================
#

# Back-compat alias: canonical name lives in `ci_log_errors/regexes.py`.
ERROR_HIGHLIGHT_RE: Pattern[str] = RED_KEYWORD_HIGHLIGHT_RE

PYTEST_FAILED_LINE_RE: Pattern[str] = SNIPPET_PYTEST_FAILED_LINE_RE

DOCKER_DAEMON_ERROR_LINE_RE: Pattern[str] = RED_DOCKER_DAEMON_ERROR_LINE_RE

DOCKER_NO_SUCH_CONTAINER_RE: Pattern[str] = RED_DOCKER_NO_SUCH_CONTAINER_RE

NETWORK_ERROR_LINE_RE: Pattern[str] = RED_NETWORK_ERROR_LINE_RE

UNSUPPORTED_CUDA_VLLM_RE: Pattern[str] = SNIPPET_UNSUPPORTED_CUDA_VLLM_RE
FAILED_TO_BUILD_RE: Pattern[str] = SNIPPET_FAILED_TO_BUILD_RE

COMMAND_LINE_BLUE_RE: Pattern[str] = SNIPPET_COMMAND_LINE_BLUE_RE

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
        # Shared icon lives in `dynamo-utils/html_pages/copy_icon_paths.svg` (sibling of this package).
        p = (Path(__file__).resolve().parent.parent / "html_pages" / "copy_icon_paths.svg").resolve()
        paths = p.read_text(encoding="utf-8").strip()
    except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
        # Fallback: empty icon rather than crashing snippet rendering.
        paths = ""
    return (
        f'<svg width="{int(size_px)}" height="{int(size_px)}" viewBox="0 0 16 16" fill="currentColor" '
        f'style="display: inline-block; vertical-align: middle;">{paths}</svg>'
    )

# Exit code 139 is conventionally SIGSEGV (signal 11) in POSIX shells (\(128 + 11 = 139\)).
# Many CI jobs only report the exit code near the end.
EXIT_CODE_139_LINE_RE: Pattern[str] = SNIPPET_EXIT_CODE_139_LINE_RE

# Git LFS fetch failures inside Docker/BuildKit dependency installs (often via uv/pip).
# Example:
#   × Failed to download and build `aiconfigurator @
#   │ git+https://github.com/...#lfs=true`
#   ├─▶ Git operation failed
#   ├─▶ failed to fetch LFS objects at ...
#   ...
#   error: failed to fetch some objects from '.../info/lfs'
GIT_LFS_SNIPPET_ANCHOR_RE: Pattern[str] = SNIPPET_GIT_LFS_SNIPPET_ANCHOR_RE
GIT_LFS_BLOCK_START_RE: Pattern[str] = SNIPPET_GIT_LFS_BLOCK_START_RE
GIT_LFS_BLOCK_END_RE: Pattern[str] = SNIPPET_GIT_LFS_BLOCK_END_RE

# CUDA runtime library import failures (common on runners missing NVIDIA driver libs).
CUDA_LIBCUDA_IMPORT_ERROR_RE: Pattern[str] = SNIPPET_CUDA_LIBCUDA_IMPORT_ERROR_RE

# Pytest often prints file-level error markers like:
#   ERROR components/src/dynamo/trtllm/tests/test_trtllm_unit.py
PYTEST_ERROR_FILE_LINE_RE: Pattern[str] = SNIPPET_PYTEST_ERROR_FILE_LINE_RE

PYTEST_CMD_LINE_RE: Pattern[str] = SNIPPET_PYTEST_CMD_LINE_RE

# Docker image pull/tag errors (registry manifest missing).
DOCKER_IMAGE_NOT_FOUND_RE: Pattern[str] = SNIPPET_DOCKER_IMAGE_NOT_FOUND_RE

# Backend status JSON-ish summary lines (multi-line blocks).
BACKEND_RESULT_FAILURE_LINE_RE: Pattern[str] = SNIPPET_BACKEND_RESULT_FAILURE_LINE_RE

# Pytest block markers we want to preserve around failures.
PYTEST_FAILURES_HEADER_RE: Pattern[str] = SNIPPET_PYTEST_FAILURES_HEADER_RE
PYTEST_PROGRESS_100_RE: Pattern[str] = SNIPPET_PYTEST_PROGRESS_100_RE
# Example:
#   "_____ test_mocker_two_kv_router[file] _____"
PYTEST_UNDERSCORE_TITLE_RE: Pattern[str] = SNIPPET_PYTEST_UNDERSCORE_TITLE_RE
PYTEST_TIMEOUT_E_LINE_RE: Pattern[str] = SNIPPET_PYTEST_TIMEOUT_E_LINE_RE
PYTEST_SHORT_TEST_SUMMARY_RE: Pattern[str] = SNIPPET_PYTEST_SHORT_TEST_SUMMARY_RE
PYTHON_MODULE_NOT_FOUND_RE: Pattern[str] = SNIPPET_PYTHON_MODULE_NOT_FOUND_RE

PYTHON_EXCEPTION_LINE_RE: Pattern[str] = SNIPPET_PYTHON_EXCEPTION_LINE_RE

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
DOCKERFILE_CONTEXT_HEADER_RE: Pattern[str] = SNIPPET_DOCKERFILE_CONTEXT_HEADER_RE
DOCKERFILE_CONTEXT_LINE_RE: Pattern[str] = SNIPPET_DOCKERFILE_CONTEXT_LINE_RE
DOCKERFILE_CONTEXT_DIVIDER_RE: Pattern[str] = SNIPPET_DOCKERFILE_CONTEXT_DIVIDER_RE

# Rust test output (cargo test)
RUST_TEST_RESULT_FAILED_RE: Pattern[str] = SNIPPET_RUST_TEST_RESULT_FAILED_RE
RUST_TEST_FAILURES_HEADER_RE: Pattern[str] = SNIPPET_RUST_TEST_FAILURES_HEADER_RE
# Failed test node ids printed under the `failures:` block. These can be indented or not, depending on
# the harness / formatting.
RUST_TEST_FAILED_TEST_NAME_RE: Pattern[str] = SNIPPET_RUST_TEST_FAILED_TEST_NAME_RE

# Lines where the user wants the *entire line* red (not just keyword highlighting).
FULL_LINE_ERROR_REDS_RE: List[Pattern[str]] = list(RED_FULL_LINE_RES)


#
# Shared categorization rules (used by BOTH full-log categorization and snippet categorization).
#
# Keep this list "regex-only" and push special-case suppression logic into `_apply_category_rules()`
# so we don’t duplicate business rules across categorization call sites.
#
CATEGORY_RULES: list[tuple[str, Pattern[str]]] = list(CAT_RULES)

def _apply_category_rules(*, text: str, lines: Sequence[str], out: List[str], seen: set[str]) -> None:
    """Apply shared regex-based category rules with shared suppression logic."""
    try:
        text_l = (text or "").lower()

        has_specific_timeout = bool(
            CAT_HTTP_TIMEOUT_RE.search(text_l)
            or CAT_GITLAB_MIRROR_TIMEOUT_RE.search(text)
            or CAT_K8S_PODS_TIMED_OUT_RE.search(text)
            or CAT_KUBECTL_PORTFORWARD_TIMEOUT_RE.search(text)
            or CAT_GITHUB_ACTION_STEP_TIMEOUT_RE.search(text)
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
            except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
                continue

        # Derived: helm errors are inherently Kubernetes-related; show k8s-error alongside helm-error.
        if "helm-error" in seen:
            add("k8s-error")
    except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
        return


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
        "--no-auto-download",
        action="store_true",
        help="Disable automatic download of missing golden logs during self-test (requires --self-test-examples).",
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

    if bool(args.self_test_examples):
        return _self_test_examples(
            raw_log_path=Path(str(args.raw_log_path)),
            auto_download=(not bool(args.no_auto_download)),
        )
    if bool(args.scan_all_logs):
        return _scan_all_logs(
            logs_root=Path(str(args.logs_root)),
            tail_bytes=int(0 if bool(args.no_tail) else int(args.tail_bytes)),
        )
    if bool(args.audit_snippet_commands):
        from . import snippet as _snippet
        return int(
            _snippet._audit_snippet_commands(
            logs_root=Path(str(args.logs_root)),
            tail_bytes=int(0 if bool(args.no_tail) else int(args.tail_bytes)),
            )
        )

    log_path = Path(args.log_path).expanduser()
    if not log_path.exists():
        print(f"ERROR: file not found: {log_path}", file=sys.stderr)
        return 2
    if not log_path.is_file():
        print(f"ERROR: not a file: {log_path}", file=sys.stderr)
        return 2

    tail_bytes = 0 if args.no_tail else int(args.tail_bytes)
    from . import snippet as _snippet
    snippet = _snippet.extract_error_snippet_from_log_file(
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

    from . import render as _render
    print(_render.render_error_snippet_html(snippet) if args.html else snippet)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

