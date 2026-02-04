"""
Pytest tests for common_dashboard_lib.py functions.

Run from dynamo-utils.dev directory:
    cd /path/to/dynamo-utils.dev
    PYTHONPATH=.:html_pages pytest html_pages/test_common_dashboard_lib.py -v
"""

import sys
import tempfile
from pathlib import Path
import pytest

# Set up path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from html_pages.common_dashboard_lib import (
    pytest_results_from_raw_log,
    PYTEST_SLOWEST_DURATIONS_REGEX,
    PYTEST_SUMMARY_REGEX,
)


# ============================================================================
# Tests for pytest_slowest_tests_from_raw_log() - slowest durations regex
# ============================================================================

def test_pytest_slowest_regex_valid_lines():
    """Test that valid pytest --durations lines are matched correctly."""
    
    # VALID lines from actual pytest --durations=10 output
    valid_lines = [
        # With GitHub Actions timestamp
        "2026-01-31T20:03:47.4743233Z 160.53s call     tests/serve/test_vllm.py::test_serve_deployment[multimodal_agg_qwen2vl_2b_epd]",
        "2026-01-31T20:03:47.4743778Z 128.98s call     tests/serve/test_vllm.py::test_serve_deployment[multimodal_agg_qwen]",
        "2026-01-31T20:03:47.4744340Z 94.34s setup    tests/kvbm_integration/test_kvbm.py::test_offload_and_onboard[llm_server_kvbm0]",
        # Without timestamp
        "160.53s call     tests/serve/test_vllm.py::test_serve_deployment[aggregated]",
        "94.34s setup    tests/kvbm_integration/test_kvbm.py::test_offload_and_onboard[llm_server_kvbm0]",
        "48.67s teardown tests/cleanup/test_cleanup.py::test_cleanup_resources",
    ]
    
    for line in valid_lines:
        m = PYTEST_SLOWEST_DURATIONS_REGEX.match(line)
        assert m is not None, f"Should match: {line}"
        # Verify captured groups
        assert m.group(1)  # duration
        assert m.group(2) in ['setup', 'call', 'teardown']  # phase
        assert m.group(3).startswith('tests/')  # test path


def test_pytest_slowest_regex_invalid_lines():
    """Test that invalid lines are correctly rejected by slowest durations regex."""
    
    # INVALID lines from actual CI error logs (should NOT match)
    invalid_lines = [
        "[call] [ 62%] (?)",
        "[setup] [ 15%]",
        "[call] at setup of test_onboarding_determinism[llm_server_kvbm0] ________",
        "ERROR at setup of test_serve_deployment[agg] ________",
        "2026-01-31T20:03:47.4718825Z ERROR    ManagedProcess:managed_process.py:354 Main server process died with exit code 1",
        "[call] ManagedProcess:managed_process.py:354 Main server process died with exit code 1 while waiting for port 8000",
        "FAILED tests/serve/test_vllm.py::test_serve_deployment[agg] - AssertionError: ...",
        "[call] Some other text with call in it",
        "This line has setup in it but not a test",
    ]
    
    for line in invalid_lines:
        m = PYTEST_SLOWEST_DURATIONS_REGEX.match(line)
        assert m is None, f"Should NOT match: {line}"


def test_pytest_summary_regex_valid_lines():
    """Test that valid pytest summary lines (FAILED/ERROR/PASSED) are matched correctly."""
    
    # VALID summary lines (should match)
    valid_lines = [
        "2026-01-31T20:03:47.4749199Z ERROR tests/kvbm_integration/test_kvbm.py::test_onboarding_determinism[llm_server_kvbm0]",
        "FAILED tests/serve/test_vllm.py::test_serve_deployment[agg] - AssertionError: ...",
        "2025-11-29T21:55:17.1891443Z FAILED tests/foo.py::test_bar - AssertionError: ...",
        "PASSED tests/router/test_basic.py::test_something",
    ]
    
    for line in valid_lines:
        m = PYTEST_SUMMARY_REGEX.match(line)
        assert m is not None, f"Should match: {line}"
        # Verify captured groups
        assert m.group(1) in ['FAILED', 'ERROR', 'PASSED', 'SKIPPED', 'XPASS', 'XFAIL']  # status
        assert m.group(2).startswith('tests/')  # test path


def test_pytest_summary_regex_invalid_lines():
    """Test that invalid lines are correctly rejected by summary regex."""
    
    # INVALID summary lines (should NOT match - error messages that contain status words)
    invalid_lines = [
        "[call] at setup of test_onboarding_determinism[llm_server_kvbm0] ________",
        "ERROR at setup of test_serve_deployment[agg] ________",
        "2026-01-31T20:03:47.4718825Z ERROR    ManagedProcess:managed_process.py:354 Main server process died",
        "[call] ManagedProcess:managed_process.py:345 [VLLM] ERROR something",
        "This line has ERROR in it but not a test",
        "[ 62%] PASSED but not a real test line",
    ]
    
    for line in invalid_lines:
        m = PYTEST_SUMMARY_REGEX.match(line)
        assert m is None, f"Should NOT match: {line}"


# ============================================================================
# Tests for pytest_results_from_raw_log() - PASSED/FAILED/SKIPPED lines
# ============================================================================

def test_pytest_results_parser_basic():
    """Test parsing pytest PASSED/FAILED/SKIPPED lines from GitHub Actions log."""
    
    # Real snippet from GitHub Actions log (job 62377203610)
    log_content = """2026-02-03T17:46:32.8254182Z tests/test_predownload_models.py::test_predownload_models[predownload_models_sglang_gpu1] PASSED [  5%]
2026-02-03T17:46:32.8275238Z components/src/dynamo/sglang/tests/test_sglang_prometheus_utils.py::TestGetPrometheusExpfmt::test_sglang_use_case PASSED [ 11%]
2026-02-03T17:46:32.8288908Z components/src/dynamo/sglang/tests/test_sglang_prometheus_utils.py::TestGetPrometheusExpfmt::test_error_handling PASSED [ 16%]
2026-02-03T17:46:32.8385693Z components/src/dynamo/sglang/tests/test_sglang_unit.py::test_custom_jinja_template_invalid_path PASSED [ 22%]
2026-02-03T17:46:35.2303648Z components/src/dynamo/sglang/tests/test_sglang_unit.py::test_custom_jinja_template_valid_path PASSED [ 27%]
2026-02-03T17:46:35.5571016Z components/src/dynamo/sglang/tests/test_sglang_unit.py::test_custom_jinja_template_env_var_expansion PASSED [ 33%]
2026-02-03T17:46:36.1647534Z tests/basic/test_cuda_version_consistency.py::test_cuda_major_consistency PASSED [ 38%]
2026-02-03T17:46:36.2028537Z tests/profiler/test_profile_sla_dryrun.py::TestProfileSLADryRun::test_sglang_dryrun PASSED [ 44%]
2026-02-03T17:46:36.2477707Z tests/profiler/test_profile_sla_dryrun.py::TestProfileSLADryRun::test_sglang_moe_dryrun PASSED [ 50%]
2026-02-03T17:46:36.6617364Z tests/profiler/test_profile_sla_dryrun.py::TestProfileSLADryRun::test_sglang_profile_with_autogen_search_space_h100 PASSED [ 55%]
2026-02-03T17:49:14.7655058Z tests/router/test_router_e2e_with_sglang.py::test_sglang_kv_router_basic[tcp] +++++++++++++++++++++++++++++++++++ Timeout ++++++++++++++++++++++++++++++++++++
2026-02-03T17:49:18.7774115Z FAILED                                                                   [ 61%]
2026-02-03T17:49:18.7781035Z tests/router/test_router_e2e_with_sglang.py::test_router_decisions_sglang_multiple_workers[tcp] SKIPPED [ 66%]
"""
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        f.write(log_content)
        temp_path = Path(f.name)
    
    try:
        # Parse the log
        results = pytest_results_from_raw_log(raw_log_path=temp_path)
        
        # Verify count
        assert len(results) == 11, f"Expected 11 results, got {len(results)}"
        
        # Verify status counts
        passed = sum(1 for _, _, status in results if status == 'success')
        failed = sum(1 for _, _, status in results if status == 'failure')
        skipped = sum(1 for _, _, status in results if status == 'skipped')
        
        assert passed == 10, f"Expected 10 passed, got {passed}"
        assert failed == 0, f"Expected 0 failed, got {failed}"
        assert skipped == 1, f"Expected 1 skipped, got {skipped}"
        
    finally:
        # Clean up temp file
        temp_path.unlink()


def test_pytest_results_parser_first_test_instant():
    """Test that first test is parsed correctly with instant duration."""
    
    log_content = """2026-02-03T17:46:32.8254182Z tests/test_predownload_models.py::test_predownload_models[predownload_models_sglang_gpu1] PASSED [  5%]
2026-02-03T17:46:32.8275238Z components/src/dynamo/sglang/tests/test_sglang_prometheus_utils.py::TestGetPrometheusExpfmt::test_sglang_use_case PASSED [ 11%]
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        f.write(log_content)
        temp_path = Path(f.name)
    
    try:
        results = pytest_results_from_raw_log(raw_log_path=temp_path)
        
        # First test should be instant (0.00s)
        assert len(results) == 2
        test_name, duration, status = results[0]
        assert "test_predownload_models" in test_name
        assert status == "success"
        assert duration == "0.00s"
        
    finally:
        temp_path.unlink()


def test_pytest_results_parser_slow_test():
    """Test that slow tests are parsed with correct timing."""
    
    log_content = """2026-02-03T17:46:32.8385693Z components/src/dynamo/sglang/tests/test_sglang_unit.py::test_custom_jinja_template_invalid_path PASSED [ 22%]
2026-02-03T17:46:35.2303648Z components/src/dynamo/sglang/tests/test_sglang_unit.py::test_custom_jinja_template_valid_path PASSED [ 27%]
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        f.write(log_content)
        temp_path = Path(f.name)
    
    try:
        results = pytest_results_from_raw_log(raw_log_path=temp_path)
        
        # Second test should take ~2.4 seconds
        assert len(results) == 2
        test_name, duration, status = results[1]
        assert "test_custom_jinja_template_valid_path" in test_name
        assert status == "success"
        assert duration.startswith("2.")  # 2.4s or similar
        
    finally:
        temp_path.unlink()


def test_pytest_results_parser_skipped_test():
    """Test that SKIPPED tests are parsed correctly."""
    
    log_content = """2026-02-03T17:46:36.6617364Z tests/profiler/test_profile_sla_dryrun.py::TestProfileSLADryRun::test_sglang_profile_with_autogen_search_space_h100 PASSED [ 55%]
2026-02-03T17:49:18.7781035Z tests/router/test_router_e2e_with_sglang.py::test_router_decisions_sglang_multiple_workers[tcp] SKIPPED [ 66%]
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        f.write(log_content)
        temp_path = Path(f.name)
    
    try:
        results = pytest_results_from_raw_log(raw_log_path=temp_path)
        
        # Last test should be skipped with ~2m 42s duration
        assert len(results) == 2
        test_name, duration, status = results[1]
        assert "test_router_decisions_sglang_multiple_workers" in test_name
        assert status == "skipped"
        # Duration should be ~162 seconds (2m 42s)
        assert "2m" in duration
        
    finally:
        temp_path.unlink()


def test_pytest_results_parser_empty_log():
    """Test parsing an empty log returns empty list."""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        f.write("")
        temp_path = Path(f.name)
    
    try:
        results = pytest_results_from_raw_log(raw_log_path=temp_path)
        assert results == []
    finally:
        temp_path.unlink()


def test_pytest_results_parser_no_pytest_output():
    """Test parsing a log with no pytest output returns empty list."""
    
    log_content = """2026-02-03T17:46:32.8254182Z Some random log line
2026-02-03T17:46:32.8275238Z Another log line without pytest
2026-02-03T17:46:32.8288908Z Building something...
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        f.write(log_content)
        temp_path = Path(f.name)
    
    try:
        results = pytest_results_from_raw_log(raw_log_path=temp_path)
        assert results == []
    finally:
        temp_path.unlink()


def test_pytest_results_parser_min_duration_filter():
    """Test that min_seconds filter works correctly."""
    
    log_content = """2026-02-03T17:46:32.8254182Z tests/test_fast.py::test_instant PASSED [  5%]
2026-02-03T17:46:32.8275238Z tests/test_fast.py::test_quick PASSED [ 10%]
2026-02-03T17:46:35.2303648Z tests/test_slow.py::test_slow PASSED [ 15%]
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        f.write(log_content)
        temp_path = Path(f.name)
    
    try:
        # Without filter: should get all 3 tests
        results_all = pytest_results_from_raw_log(raw_log_path=temp_path, min_seconds=0.0)
        assert len(results_all) == 3
        
        # With filter: should only get tests >= 1 second (last test only, 2.4s)
        # First test: 0.00s, Second test: 0.00s, Third test: 2.4s
        results_filtered = pytest_results_from_raw_log(raw_log_path=temp_path, min_seconds=1.0)
        assert len(results_filtered) == 1
        assert "test_slow" in results_filtered[0][0]
        
    finally:
        temp_path.unlink()
