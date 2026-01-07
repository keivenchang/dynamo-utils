#!/usr/bin/env python3
"""
CI log error categorization + snippet extraction (CLI entrypoint).

The implementation lives in `dynamo-utils/ci_log_errors/engine.py`.

Run the golden self-test:
- `python3 dynamo-utils/ci_log_errors/core.py --self-test-examples`
"""

from __future__ import annotations

import sys
from pathlib import Path


def _load_cli():
    """Import `ci_log_errors.cli` in both module and "run as script" modes."""
    if __package__:
        from . import cli as _cli_mod  # type: ignore

        return _cli_mod
    # Running as a script: add `dynamo-utils/` to sys.path so `import ci_log_errors` works.
    utils_dir = Path(__file__).resolve().parent.parent
    if str(utils_dir) not in sys.path:
        sys.path.insert(0, str(utils_dir))
    import ci_log_errors.cli as _cli_mod  # type: ignore

    return _cli_mod


def _cli(argv=None) -> int:
    m = _load_cli()
    return int(m._cli(argv))  # type: ignore[attr-defined]


if __name__ == "__main__":
    raise SystemExit(_cli())


