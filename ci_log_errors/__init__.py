"""
Shared log error detection + snippet extraction library (dynamo-utils).

This package contains the *implementation* for:
- log categorization (e.g. pytest-error, broken-links, etc)
- snippet extraction (high-signal error snippets)
- snippet HTML rendering (copyable command blocks, highlighting)

It is designed to be dependency-light so it can be reused by:
- `dynamo-utils/common.py`
- `dynamo-utils/html_pages/*` dashboards

Public API is re-exported from:
- `ci_log_errors.engine` for log categorization primitives
- `ci_log_errors.snippet` for snippet extraction
- `ci_log_errors.render` for snippet HTML rendering
"""

from .engine import (  # noqa: F401
    categorize_error_log_lines,
    golden_log_job_ids,
    is_golden_log_job_id,
)

# Rendering helpers live in `ci_log_errors/render.py`. Prefer importing from there directly:
#   from ci_log_errors import render as ci_render
#   ci_render.render_error_snippet_html(...)
from .render import (  # noqa: F401
    categorize_error_snippet_text,
    render_error_snippet_html,
)

# Snippet extraction lives in `ci_log_errors/snippet.py`. Prefer importing from there directly:
#   from ci_log_errors import snippet as ci_snippet
#   ci_snippet.extract_error_snippet_from_text(...)
from .snippet import (  # noqa: F401
    extract_error_snippet_from_log_file,
    extract_error_snippet_from_text,
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


