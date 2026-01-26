"""
CLI wrapper for ci_log_errors.

We keep CLI glue in its own module so the shared library implementation (`engine.py`)
stays easier to navigate and reuse from dashboards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import argparse
import logging
import sys

from . import engine
from . import snippet

logger = logging.getLogger(__name__)


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Delegate to the implementation for the heavy lifting.
    parser = argparse.ArgumentParser(
        description="Extract and format a high-signal error snippet from a CI log file.",
        epilog="Examples:\n"
               "  %(prog)s 59975400792  # by job ID\n"
               "  %(prog)s ~/.cache/dynamo-utils/raw-log-text/59975400792.log  # by file path\n"
               "  %(prog)s 59975400792 --html  # HTML output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "log_path",
        nargs="?",
        default="",
        help="Path to a local raw log file (e.g., raw-log-text/<job_id>.log) OR job ID number (e.g., 59975400792). Not required for --self-test-examples.",
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
        help="Run the Examples self-test (parses golden examples and validates categories).",
    )
    parser.add_argument(
        "--raw-log-path",
        default=str(engine._default_raw_log_dir()),
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
        default=str(engine._default_raw_log_dir()),
        help="Directory containing raw-log-text/*.log for --scan-all-logs (default: ~/.cache/dynamo-utils/raw-log-text).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if bool(args.self_test_examples):
        return int(engine._self_test_examples(raw_log_path=Path(str(args.raw_log_path))))
    if bool(args.scan_all_logs):
        return int(
            engine._scan_all_logs(
                logs_root=Path(str(args.logs_root)),
                tail_bytes=int(0 if bool(args.no_tail) else int(args.tail_bytes)),
            )
        )
    if bool(args.audit_snippet_commands):
        return int(
            engine._audit_snippet_commands(
                logs_root=Path(str(args.logs_root)),
                tail_bytes=int(0 if bool(args.no_tail) else int(args.tail_bytes)),
            )
        )

    # Handle both file paths and job IDs
    log_input = args.log_path
    log_path = Path(log_input).expanduser()
    
    # If the input looks like a job ID (numeric string), try to find it in the default log directory
    if log_input.isdigit() and not log_path.exists():
        default_log_dir = Path(args.logs_root).expanduser()
        log_path = default_log_dir / f"{log_input}.log"
        if not log_path.exists():
            logger.error(f"ERROR: log file not found for job ID {log_input}: {log_path}")
            return 2
    
    if not log_path.exists():
        logger.error(f"ERROR: file not found: {log_path}")
        return 2
    if not log_path.is_file():
        logger.error(f"ERROR: not a file: {log_path}")
        return 2

    tail_bytes = 0 if args.no_tail else int(args.tail_bytes)
    snippet_text = snippet.extract_error_snippet_from_log_file(
        log_path,
        tail_bytes=tail_bytes,
        context_before=int(args.context_before),
        context_after=int(args.context_after),
        max_lines=int(args.max_lines),
        max_chars=int(args.max_chars),
    )

    if not (snippet_text or "").strip():
        logger.info("(no snippet found)")
        return 0

    out = engine.render_error_snippet_html(snippet_text) if args.html else snippet_text
    sys.stdout.write(str(out or "") + ("\n" if not str(out or "").endswith("\n") else ""))
    return 0


