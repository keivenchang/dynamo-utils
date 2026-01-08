#!/usr/bin/env python3
"""Module entrypoint for `ci_log_errors`.

Usage (from `dynamo-utils/`):
  - `python3 -m ci_log_errors --self-test-examples`
"""

from __future__ import annotations

from .cli import _cli


if __name__ == "__main__":
    raise SystemExit(_cli())



