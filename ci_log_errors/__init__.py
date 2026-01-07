"""
Shared log error detection + snippet extraction library (dynamo-utils).

This package contains the *implementation* for:
- log categorization (e.g. pytest-error, broken-links, etc)
- snippet extraction (high-signal error snippets)
- snippet HTML rendering (copyable command blocks, highlighting)

It is designed to be dependency-light so it can be reused by:
- `dynamo-utils/common.py`
- `dynamo-utils/html_pages/*` dashboards

Public API is re-exported from `ci_log_errors.engine` (implementation) and `ci_log_errors.core` (CLI entrypoint).
"""

from .engine import (  # noqa: F401
    categorize_error_log_lines,
    categorize_error_snippet_text,
    extract_error_snippet_from_log_file,
    extract_error_snippet_from_text,
    golden_log_job_ids,
    is_golden_log_job_id,
    render_error_snippet_html,
)

__all__ = [
    "categorize_error_log_lines",
    "categorize_error_snippet_text",
    "extract_error_snippet_from_log_file",
    "extract_error_snippet_from_text",
    "render_error_snippet_html",
    "golden_log_job_ids",
    "is_golden_log_job_id",
]


