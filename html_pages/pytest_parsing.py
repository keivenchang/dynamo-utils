# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest output parsing: log analysis, test result extraction, and snippet caching."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datetime import datetime
from common_types import CIStatus

logger = logging.getLogger(__name__)


def _parse_iso_utc(s: str):
    x = str(s or "").strip()
    if not x:
        return None
    if x.endswith("Z"):
        return datetime.fromisoformat(x[:-1] + "+00:00")
    return datetime.fromisoformat(x)

# ======================================================================================
# Grafana URL Templates
# ======================================================================================

# Grafana Test Details dashboard URL template (for individual pytest tests)
# Example: https://grafana.nvidia.com/d/bf0set70vqygwb/test-details?orgId=283&var-branch=All&var-test_status=All&var-test=test_serve_deployment%5Baggregated%5D&from=now-30d&to=now
# Note: Multiple var-test parameters can be present, but only the last one is used for single-select variables
GRAFANA_TEST_URL_TEMPLATE = "https://grafana.nvidia.com/d/bf0set70vqygwb/test-details?orgId=283&var-branch=All&var-test_status=All&var-test={test_name}&from=now-30d&to=now"


# ======================================================================================
# Pytest Log Parsing Regex Patterns
# ======================================================================================

# Regex for parsing pytest slowest durations section (from --durations=N output)
# Example lines (with GitHub Actions timestamp):
#   "2026-01-31T20:03:47.4743233Z 160.53s call     tests/serve/test_vllm.py::test_serve_deployment[agg]"
#   "160.53s call     tests/serve/test_vllm.py::test_serve_deployment[agg]" (without timestamp)
# CRITICAL: Requires test path to start with "tests/" to avoid matching error messages
PYTEST_SLOWEST_DURATIONS_REGEX = re.compile(
    r'^(?:\d{4}-\d{2}-\d{2}T[\d:.]+Z\s+)?'  # Optional GitHub Actions timestamp
    r'(\d+\.?\d*)s\s+'                       # Duration (captured)
    r'(setup|call|teardown)\s+'              # Phase (captured)
    r'(tests/.+)$'                           # Test path must start with "tests/" (captured)
)

# Regex for parsing pytest summary lines (FAILED/ERROR/PASSED lines)
# Example lines:
#   "2026-01-31T20:03:47.4749199Z ERROR tests/kvbm_integration/test_kvbm.py::test_onboarding_determinism[llm_server_kvbm0]"
#   "FAILED tests/serve/test_vllm.py::test_serve_deployment[agg] - AssertionError: ..."
#   "PASSED tests/router/test_basic.py::test_something"
# CRITICAL: Requires test path to start with "tests/" to avoid matching error messages like:
#   "[call] ManagedProcess:managed_process.py:345 ERROR ..." (contains ERROR but not a test)
#   "[call] at setup of test_onboarding_determinism..." (contains test name but not a summary line)
PYTEST_SUMMARY_REGEX = re.compile(
    r'^.*?'                                      # Any prefix (including empty)
    r'(FAILED|ERROR|XPASS|XFAIL|SKIPPED|PASSED)'  # Status word (captured)
    r'\s+'                                       # Whitespace after status
    r'(tests/.+?)'                               # Test path must start with "tests/" (captured)
    r'(?:\s+-\s+.*)?\s*$'                        # Optional " - error message" suffix
)


def pytest_slowest_tests_from_raw_log(
    *,
    raw_log_path: Optional[Path],
    min_seconds: float = 10.0,
    include_all: bool = False,
    step_name: str = "",
    step_dict: Optional[Dict[str, object]] = None,
) -> List[Tuple[str, str, str]]:
    """Parse pytest per-test durations from cached raw log file.
    
    This is based on pytest's "slowest N durations" section (`--durations=N`).
    If CI is configured with `--durations=0`, this section contains all tests.
    
    Args:
        raw_log_path: Path to the raw log file
        min_seconds: Minimum duration to include (default: 10s). Ignored when include_all=True.
        include_all: If True, include all entries in the durations section regardless of duration threshold.
    
    Returns:
        List of (test_name, duration_str, status_norm) tuples, in the same order as the log section.
        
    Example output format:
        [
            ("[call] tests/serve/test_vllm.py::test_serve_deployment[agg]", "1m 43s", "success"),
            ("[setup] tests/kvbm_integration/test_kvbm.py::test_offload_and_onboard[llm_server_kvbm0]", "1m 50s", "failure"),
            ...
        ]
    """
    if not raw_log_path:
        return []

    # Parsed pytest timings cache (disk-backed). Cache boundary is JSON-on-disk; in-memory uses dataclasses.
    from cache_pytest_timings import PYTEST_TIMINGS_CACHE  # local file import

    # Normalize step_name for caching: Build-and-test phases often show up as 'pytest (parallel)'
    # but we want a stable cache key that still maps to our pytest parsing heuristics.
    step_name_for_cache = str(step_name or "").strip()
    step_name_lc = step_name_for_cache.lower()
    if step_name_lc.startswith("pytest") and not step_name_lc.startswith("test:"):
        step_name_for_cache = f"test: {step_name_for_cache}"

    cached = PYTEST_TIMINGS_CACHE.get_if_fresh(raw_log_path=Path(raw_log_path), step_name=step_name_for_cache)
    if cached is not None:
        return list(cached)
    
    try:
        t0_parse = time.monotonic()
        p = Path(raw_log_path)
        if not p.exists() or not p.is_file():
            return []
        
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            log_text = f.read()
        
        lines = log_text.split('\n')
        
        logger.info(f"[pytest_slowest_tests_from_raw_log] Called for '{step_name}', step_dict={type(step_dict).__name__}, is_dict={isinstance(step_dict, dict) if step_dict else False}")
        
        # Filter lines by timestamp if step_dict has started_at/completed_at
        if step_dict and isinstance(step_dict, dict):
            logger.info(f"[pytest_slowest_tests_from_raw_log] step_dict provided for '{step_name}': {bool(step_dict)}, keys={list(step_dict.keys())}")
            started_at_str = str(step_dict.get("started_at", "") or "")
            completed_at_str = str(step_dict.get("completed_at", "") or "")
            logger.info(f"[pytest_slowest_tests_from_raw_log] Timestamps: started={started_at_str}, completed={completed_at_str}")
            if started_at_str and completed_at_str:
                started_at = _parse_iso_utc(started_at_str)
                completed_at = _parse_iso_utc(completed_at_str)
                if started_at and completed_at:
                    filtered_lines = []
                    for line in lines:
                        # Extract timestamp from GitHub Actions log line
                        # Format: 2026-02-06T00:51:17.3815132Z ...
                        if len(line) > 28 and line[27] == 'Z':
                            line_ts_str = line[:28]
                            line_ts = _parse_iso_utc(line_ts_str)
                            if line_ts and started_at <= line_ts <= completed_at:
                                filtered_lines.append(line)
                        else:
                            # No timestamp, include it (safer)
                            filtered_lines.append(line)
                    lines = filtered_lines
                    logger.info(
                        f"[pytest_slowest_tests_from_raw_log] Filtered to {len(lines)} lines "
                        f"between {started_at_str} and {completed_at_str} for step '{step_name}'"
                    )

        # Build a map from test-id -> status using pytest summary lines.
        # Example lines:
        #   FAILED tests/foo.py::test_bar[param] - AssertionError: ...
        #   ERROR  tests/foo.py::test_baz - ...
        #   SKIPPED tests/foo.py::test_qux - ...
        #   XFAIL tests/foo.py::test_x - ...
        status_by_test: Dict[str, str] = {}

        def _norm_test_id(s: str) -> str:
            return str(s or "").strip()

        # Match status lines with optional timestamp prefix (GitHub Actions format)
        # Example: "2025-11-29T21:55:17.1891443Z FAILED tests/foo.py::test_bar - AssertionError: ..."
        # CRITICAL: Require test path to start with "tests/" to avoid matching error messages like:
        #   "[call] ManagedProcess:managed_process.py:354 ERROR ..." (contains ERROR but not a test)
        #   "[call] at setup of test_onboarding_determinism..." (contains test name but not a pytest summary line)
        for ln in lines:
            msum = PYTEST_SUMMARY_REGEX.match(str(ln or ""))
            if not msum:
                continue
            st_word = str(msum.group(1) or "").strip().upper()
            test_id = _norm_test_id(msum.group(2) or "")
            if not test_id:
                continue
            if st_word in {"FAILED", "ERROR", "XPASS"}:
                status_by_test[test_id] = CIStatus.FAILURE.value
            elif st_word in {"SKIPPED", "XFAIL"}:
                status_by_test[test_id] = CIStatus.SKIPPED.value
            elif st_word == "PASSED":
                status_by_test[test_id] = CIStatus.SUCCESS.value
        
        # Determine which test_type to filter for based on step_name
        # Examples: "Run unit tests" -> "unit", "Run e2e tests" -> "e2e"
        target_test_type = ""
        step_lower = str(step_name or "").lower()
        if "unit" in step_lower:
            target_test_type = "unit"
        elif "e2e" in step_lower:
            target_test_type = "e2e"

        # Simply parse all slowest durations from the log
        # The step structure in the UI makes it clear which step each test belongs to
        
        # Look for the "slowest N durations" section(s)
        # Format: "============================= slowest 10 durations ============================="
        # Followed by lines like (with GitHub Actions timestamp prefix):
        # "2026-01-15T22:01:23.5641223Z 110.16s setup    tests/kvbm_integration/test_kvbm.py::test_offload_and_onboard[llm_server_kvbm0]"
        # "2026-01-15T22:01:23.5641223Z 103.05s call     tests/serve/test_vllm.py::test_serve_deployment[agg-request-plane-tcp]"

        test_times: List[Tuple[str, str, str]] = []
        in_slowest_section = False
        current_test_type = ""  # Track which test_type section we're in
        threshold = 0.0 if bool(include_all) else float(min_seconds or 0.0)

        for i, line in enumerate(lines):
            # Track which test_type section we're in by looking for pytest action markers
            # Example: "  test_type: unit" or "  test_type: e2e, gpu_1"
            # Match only "test_type:" at the start of a word (not "STR_TEST_TYPE:")
            type_match = re.search(r'\btest_type:\s*(\w+)', line, re.IGNORECASE)
            if type_match:
                current_test_type = type_match.group(1).strip().lower()
            # Start of slowest section
            if 'slowest' in line.lower() and 'duration' in line.lower() and '=====' in line:
                in_slowest_section = True
                continue

            # End of slowest section (next ===== line)
            # Don't break - there may be multiple "slowest durations" sections (multiple pytest runs)
            if in_slowest_section and '=====' in line:
                in_slowest_section = False
                continue

            if in_slowest_section:
                # Skip this section if we're filtering by test_type and it doesn't match
                if target_test_type and current_test_type != target_test_type:
                    continue
                # Parse line format (with GitHub Actions timestamp prefix):
                # These lines are generated by pytest with --durations=N flag (e.g., --durations=10 or --durations=0)
                # Example pytest command: pytest --durations=10 --tb=short --basetemp=/tmp tests/
                #
                # VALID MATCHES (these should match):
                #   "2026-01-15T22:01:23.5641223Z 110.16s setup    tests/kvbm_integration/test_kvbm.py::test_offload_and_onboard[llm_server_kvbm0]"
                #   "2026-01-15T22:01:23.5641223Z 103.05s call     tests/serve/test_vllm.py::test_serve_deployment[aggregated]"
                #   "160.53s call     tests/router/test_router_e2e_with_vllm.py::test_vllm_kv_router_basic[tcp]"  (without timestamp)
                #
                # INVALID MATCHES (these should NOT match - they're error messages, not pytest test lines):
                #   "2026-01-31T20:03:47.4718825Z ERROR    ManagedProcess:managed_process.py:345 [VLLM] [0;36m(APIServer pid=1306)[0;0m     self.engine_core = ..."
                #   "[call] [ 62%] (?)"  (pytest progress indicator, not a test)
                #   "[call] at setup of test_onboarding_determinism[llm_server_kvbm0] ________"  (pytest error header, not a test)
                #   "2026-01-31T20:03:47.4708208Z INFO     ManagedProcess:managed_process.py:219 Running command: vllm serve --block-size 16 ..."
                #
                # CRITICAL: Must ensure duration+phase appears right after timestamp (not in error messages)
                # Pattern: optional timestamp, then IMMEDIATELY duration + phase + test path starting with "tests/"
                m = PYTEST_SLOWEST_DURATIONS_REGEX.match(str(line or ""))
                if m:
                    duration = float(m.group(1))
                    phase = m.group(2)
                    test_id = str(m.group(3) or "").strip()
                    
                    # Filter by minimum duration unless include_all is set
                    if duration >= threshold:
                        # Format duration as "1m 50s" or "110s"
                        if duration >= 60:
                            mins = int(duration // 60)
                            secs = int(duration % 60)
                            dur_str = f"{mins}m {secs}s"
                        else:
                            dur_str = f"{int(duration)}s"
                        
                        # Include phase in the test name for clarity
                        full_name = f"[{phase}] {test_id}"
                        
                        # Determine status (best-effort) from summary lines; default to success.
                        status_norm = status_by_test.get(test_id, CIStatus.SUCCESS.value)
                        test_times.append((full_name, dur_str, status_norm))
        
        
        # Add failed/error tests that didn't appear in slowest durations section
        # This ensures failed tests are always shown, even without --durations flag
        tests_already_shown = {t[0].split('] ', 1)[-1] for t in test_times if '] ' in t[0]}
        for test_id, test_status in status_by_test.items():
            # Only add if it's a failure/error and not already in the list
            if test_status == CIStatus.FAILURE.value and test_id not in tests_already_shown:
                # Add with unknown duration (will be shown as "?")
                full_name = f"[call] {test_id}"
                test_times.append((full_name, "?", test_status))
        # Persist parsed rows (best-effort).
        from cache_pytest_timings import PYTEST_TIMINGS_CACHE  # local file import

        # Record parse timing on the concrete cache object.
        PYTEST_TIMINGS_CACHE.stats.parse_calls += 1
        PYTEST_TIMINGS_CACHE.stats.parse_secs += max(0.0, float(time.monotonic() - t0_parse))

        PYTEST_TIMINGS_CACHE.put(raw_log_path=p, step_name=step_name, rows=test_times)

        logger.info(f"[pytest_slowest_tests_from_raw_log] Returning {len(test_times)} tests for step_name='{step_name}'")
        return test_times

    except Exception as e:
        logger.debug(f"Failed to parse pytest slowest durations from {raw_log_path}: {e}")
        return []


def pytest_results_from_raw_log(
    *,
    raw_log_path: Optional[Path],
    min_seconds: float = 0.0,
    step_name: str = "",
    step_dict: Optional[Dict[str, object]] = None,
) -> List[Tuple[str, str, str]]:
    """Parse pytest individual test results (PASSED/FAILED/SKIPPED) from raw log.
    
    This parses lines like:
        2026-02-03T17:46:32.8254182Z tests/test_foo.py::test_bar PASSED [  5%]
        2026-02-03T17:49:14.7655058Z tests/router/test_e2e.py::test_basic[tcp] FAILED [ 61%]
        
    Also detects pytest-timeout markers:
        2026-02-03T17:52:47.0974817Z tests/router/test_e2e.py::test_basic[tcp] +++++++++++++++++++++++++++++++++++ Timeout ++++++++++++++++++++++++++++++++++++
        ... stack traces ...
        2026-02-03T17:52:51.1096979Z FAILED                                                                   [ 61%]
    
    CRITICAL: Uses timestamp-based filtering when job object is provided.
    For synthetic steps like "pytest (parallel)" or "pytest (serial)", this function will:
    1. Find the real step in the job API that matches the synthetic name
    2. Use that step's started_at/completed_at timestamps to filter log lines
    3. Only parse tests within that time window
    
    Args:
        raw_log_path: Path to the raw log file
        min_seconds: Minimum duration to include (calculated from timestamp diff). Default: 0.0 (include all)
        step_name: Name of the step (used to find matching step in job API for timestamp filtering)
        job: Optional job dict from GitHub API (contains steps with timestamps)
    
    Returns:
        List of (test_name, duration_str, status_norm) tuples, in order of execution.
        Test names with timeout will have " [pytest-timeout]" appended.
        
    Example output:
        [
            ("tests/test_foo.py::test_bar", "0.5s", "success"),
            ("tests/router/test_e2e.py::test_basic[tcp] [pytest-timeout]", "2m 39s", "failure"),
            ...
        ]
    """
    if not raw_log_path:
        return []
    
    try:
        p = Path(raw_log_path)
        if not p.exists() or not p.is_file():
            return []
        
        import re
        from datetime import datetime
        
        # Read entire log file
        with open(p, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # Filter lines by timestamp if step_dict has started_at/completed_at
        if step_dict and isinstance(step_dict, dict):
            started_at_str = str(step_dict.get("started_at", "") or "")
            completed_at_str = str(step_dict.get("completed_at", "") or "")
            if started_at_str and completed_at_str:
                started_at = _parse_iso_utc(started_at_str)
                completed_at = _parse_iso_utc(completed_at_str)
                if started_at and completed_at:
                    filtered_lines = []
                    for line in lines:
                        # Extract timestamp from GitHub Actions log line
                        # Format: 2026-02-06T00:51:17.3815132Z ...
                        if len(line) > 28 and line[27] == 'Z':
                            line_ts_str = line[:28]
                            line_ts = _parse_iso_utc(line_ts_str)
                            if line_ts and started_at <= line_ts <= completed_at:
                                filtered_lines.append(line)
                        else:
                            # No timestamp, include it (safer)
                            filtered_lines.append(line)
                    lines = filtered_lines
                    logger.info(
                        f"[pytest_results_from_raw_log] Filtered to {len(lines)} lines "
                        f"between {started_at_str} and {completed_at_str} for step '{step_name}'"
                    )
        
        results = []
        prev_timestamp = None
        timeout_tests = set()  # Track which tests timed out
        
        # First pass: detect timeout markers
        # Pattern: tests/test_foo.py::test_bar +++++++++++++++++++++++++++++++++++ Timeout ++++++++++++++++++++++++++++++++++++
        timeout_pattern = re.compile(
            r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s+(\S+)\s+\+{20,}\s+Timeout\s+\+{20,}'
        )
        
        for line in lines:
            m = timeout_pattern.match(line)
            if m:
                timeout_tests.add(m.group(1))
        
        # Second pass: parse test results
        # Pattern: 2026-02-03T17:46:32.8254182Z tests/test_foo.py::test_bar PASSED [  5%]
        pattern = re.compile(
            r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\s+(\S+)\s+(PASSED|FAILED|SKIPPED|ERROR)\s+\[\s*\d+%\]'
        )
        
        for line in lines:
            m = pattern.match(line)
            if not m:
                continue
            
            timestamp_str, test_name, status_raw = m.groups()
            
            # Parse timestamp
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                timestamp = None
            
            # Calculate duration from previous test
            duration_s = 0.0
            if prev_timestamp and timestamp:
                duration_s = (timestamp - prev_timestamp).total_seconds()
            
                # Normalize status
                status_norm = {
                    'PASSED': 'success',
                    'FAILED': 'failure',
                    'SKIPPED': 'skipped',
                    'ERROR': 'failure',
                }.get(status_raw, 'unknown')
                
                # Format duration
                if duration_s < 1.0:
                    dur_str = f"{duration_s:.2f}s"
                elif duration_s < 60:
                    dur_str = f"{duration_s:.1f}s"
                elif duration_s < 3600:
                    mins = int(duration_s // 60)
                    secs = int(duration_s % 60)
                    dur_str = f"{mins}m {secs}s"
                else:
                    hours = int(duration_s // 3600)
                    mins = int((duration_s % 3600) // 60)
                    dur_str = f"{hours}h {mins}m"
                
                # Apply duration filter
                if duration_s >= min_seconds:
                    results.append((test_name, dur_str, status_norm))
                
                prev_timestamp = timestamp
        
        # Now add timeout tests that were detected but didn't have PASSED/FAILED/SKIPPED lines
        # (these appear as a test name followed by Timeout marker, then FAILED with no test name)
        for timeout_test in timeout_tests:
            # Check if this test is already in results
            if not any(timeout_test in r[0] for r in results):
                # Add it as a failed test with timeout marker
                results.append((f"{timeout_test} [pytest-timeout]", "unknown", "failure"))
        
        # Add timeout markers to tests that are in results and timed out
        results_with_markers = []
        for test_name, dur_str, status_norm in results:
            # Check if this test timed out and doesn't already have the marker
            if any(timeout_test in test_name for timeout_test in timeout_tests) and "[pytest-timeout]" not in test_name:
                test_name_with_marker = f"{test_name} [pytest-timeout]"
                results_with_markers.append((test_name_with_marker, dur_str, status_norm))
            else:
                results_with_markers.append((test_name, dur_str, status_norm))
        
        return results_with_markers
    
    except Exception as e:
        logger.debug(f"Failed to parse pytest results from {raw_log_path}: {e}")
        return []


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
    steps = job.get("steps") if isinstance(job, dict) else None
    if isinstance(steps, list):
        for st in steps:
            if isinstance(st, dict) and str(st.get("name", "") or "") == str(step_name or ""):
                step = st
                break
    if not isinstance(step, dict):
        return ""

    a = _parse_iso_utc(str(step.get("started_at", "") or ""))
    b = _parse_iso_utc(str(step.get("completed_at", "") or ""))
    if not a or not b:
        return ""

    # Shared library (dependency-light): `dynamo-utils/ci_log_errors/`
    from ci_log_errors import snippet as ci_snippet  # local import (avoid circulars)

    text = p.read_text(encoding="utf-8", errors="replace")

    # Filter lines by timestamp window (raw logs include ISO-8601 timestamps).
    kept: List[str] = []
    for ln in (text.splitlines() or []):
        # Most lines are prefixed with an ISO timestamp; ignore unparsable lines.
        ts = None
        # Heuristic: take the first token, strip any trailing 'Z'.
        head = (ln.split(" ", 1)[0] if " " in ln else ln).strip()
        ts = _parse_iso_utc(head)
        if not ts:
            continue
        if ts < a or ts > b:
            continue
        kept.append(ln)
    if not kept:
        return ""
    return ci_snippet.extract_error_snippet_from_text("\n".join(kept))
