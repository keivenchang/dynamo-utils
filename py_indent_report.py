#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Print a quick indentation report for Python files.

Goal: make indentation/continuation-indent problems obvious during review.

This script is intentionally self-contained (single file) so it's easy to invoke
and doesn't rely on additional local modules.

Typical usage:
  python3 py_indent_report.py path/to/file.py
  python3 py_indent_report.py --all path/to/file.py
  python3 py_indent_report.py --only-problems path/to/file.py

Notes:
- This is *not* a formatter. Prefer `ruff format` for auto-fixing formatting.
- This script *subsumes* the usual indentation gates by running:
  - `python -tt -m py_compile <file>`
  - `tabnanny <file>`
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import sys
import tempfile
import tokenize
import tabnanny
import subprocess
from dataclasses import dataclass
from typing import Iterable, Iterator

LEADING_WS_RE = re.compile(r"^[ \t]*")


@dataclass(frozen=True)
class LineInfo:
    lineno: int
    text: str
    leading_ws: str
    spaces: int
    tabs: int
    indent_cols: int


@dataclass(frozen=True)
class LineCheck:
    lineno: int
    indent_cols: int
    expected_cols: int | None
    paren_level: int | None
    problems: tuple[str, ...]

    @property
    def is_problem(self) -> bool:
        return bool(self.problems)


def _compute_line_info(lines: list[str], *, tabsize: int) -> list[LineInfo]:
    infos: list[LineInfo] = []
    for i, line in enumerate(lines, start=1):
        leading = LEADING_WS_RE.match(line).group(0)  # type: ignore[union-attr]
        spaces = leading.count(" ")
        tabs = leading.count("\t")
        indent_cols = len(leading.expandtabs(tabsize))
        infos.append(
            LineInfo(
                lineno=i,
                text=line.rstrip("\n"),
                leading_ws=leading,
                spaces=spaces,
                tabs=tabs,
                indent_cols=indent_cols,
            )
        )
    return infos


def _iter_tokens(source: str) -> Iterator[tokenize.TokenInfo]:
    # tokenize works on readline, preserve exact newlines from the file.
    return tokenize.generate_tokens(io.StringIO(source).readline)


def _compute_line_paren_levels(tokens: Iterable[tokenize.TokenInfo]) -> dict[int, int]:
    """
    Map line -> paren nesting level at the start of that line (best-effort).
    """
    paren_level = 0
    line_paren_level: dict[int, int] = {}
    current_line = 1

    for tok in tokens:
        (sline, _scol) = tok.start

        # Fill in missing lines up to sline with current paren_level.
        while current_line < sline:
            line_paren_level.setdefault(current_line, paren_level)
            current_line += 1

        line_paren_level.setdefault(sline, paren_level)

        if tok.type == tokenize.OP:
            if tok.string in ("(", "[", "{"):
                paren_level += 1
            elif tok.string in (")", "]", "}"):
                paren_level = max(paren_level - 1, 0)

    # Add a final mapping for the last observed line.
    line_paren_level.setdefault(current_line, paren_level)
    return line_paren_level


def _compute_lines_in_multiline_strings(tokens: Iterable[tokenize.TokenInfo]) -> set[int]:
    """
    Return the set of line numbers that are part of a multi-line STRING token.
    """
    lines: set[int] = set()
    for tok in tokens:
        if tok.type != tokenize.STRING:
            continue
        start_line = tok.start[0]
        end_line = tok.end[0]
        if end_line > start_line:
            for ln in range(start_line, end_line + 1):
                lines.add(ln)
    return lines


def _compute_expected_indent_per_line(
    tokens: Iterable[tokenize.TokenInfo], *, tabsize: int
) -> dict[int, int]:
    """
    Best-effort expected indent columns per line for "statement-start" lines.
    """
    indent_stack = [0]
    expected: dict[int, int] = {}

    current_line = 1
    for tok in tokens:
        sline, _scol = tok.start

        while current_line < sline:
            expected.setdefault(current_line, indent_stack[-1])
            current_line += 1

        if tok.type == tokenize.INDENT:
            cols = len(tok.string.expandtabs(tabsize))
            indent_stack.append(cols)
            continue

        if tok.type == tokenize.DEDENT:
            if len(indent_stack) > 1:
                indent_stack.pop()
            continue

        expected.setdefault(sline, indent_stack[-1])

    expected.setdefault(current_line, indent_stack[-1])
    return expected


def _compute_indent_deltas_per_line(
    tokens: Iterable[tokenize.TokenInfo], *, tabsize: int
) -> dict[int, int]:
    """
    Best-effort indentation delta per INDENT line (flags indent "jumps").
    """
    indent_stack = [0]
    delta_by_line: dict[int, int] = {}

    for tok in tokens:
        if tok.type == tokenize.DEDENT:
            if len(indent_stack) > 1:
                indent_stack.pop()
            continue
        if tok.type != tokenize.INDENT:
            continue

        new_cols = len(tok.string.expandtabs(tabsize))
        old_cols = indent_stack[-1]
        delta_by_line[tok.start[0]] = new_cols - old_cols
        indent_stack.append(new_cols)

    return delta_by_line


def _is_blank_or_comment_only(text: str) -> bool:
    stripped = text.strip()
    return stripped == "" or stripped.startswith("#")


def check_file(
    path: str, *, indent_width: int, tabsize: int
) -> tuple[list[LineInfo], list[LineCheck], str | None]:
    """
    Returns: (line_infos, line_checks, tokenize_error)
    """
    data = tokenize.open(path).read()
    lines = data.splitlines(keepends=True)
    infos = _compute_line_info(lines, tabsize=tabsize)

    try:
        tokens = list(_iter_tokens(data))
    except (tokenize.TokenError, IndentationError) as e:
        checks = [
            LineCheck(
                lineno=li.lineno,
                indent_cols=li.indent_cols,
                expected_cols=None,
                paren_level=None,
                problems=(),
            )
            for li in infos
        ]
        return infos, checks, f"{type(e).__name__}: {e}"

    line_paren = _compute_line_paren_levels(tokens)
    string_lines = _compute_lines_in_multiline_strings(tokens)
    expected_indent = _compute_expected_indent_per_line(tokens, tabsize=tabsize)
    indent_deltas = _compute_indent_deltas_per_line(tokens, tabsize=tabsize)

    checks: list[LineCheck] = []
    for li in infos:
        problems: list[str] = []
        paren_level = line_paren.get(li.lineno)
        exp = expected_indent.get(li.lineno)

        in_multiline_string = li.lineno in string_lines

        if li.tabs and not in_multiline_string:
            problems.append("TAB_IN_INDENT")

        if (
            not in_multiline_string
            and paren_level == 0
            and li.indent_cols % indent_width != 0
            and not _is_blank_or_comment_only(li.text)
        ):
            problems.append(f"INDENT_NOT_MULTIPLE_OF_{indent_width}")

        if (
            not in_multiline_string
            and paren_level == 0
            and li.lineno in indent_deltas
            and indent_deltas[li.lineno] != indent_width
            and not _is_blank_or_comment_only(li.text)
        ):
            problems.append(
                f"INDENT_DELTA_NOT_{indent_width}_GOT_{indent_deltas[li.lineno]}"
            )

        if (
            not in_multiline_string
            and paren_level == 0
            and exp is not None
            and not _is_blank_or_comment_only(li.text)
            and li.indent_cols != exp
        ):
            problems.append(f"EXPECTED_INDENT_{exp}")

        checks.append(
            LineCheck(
                lineno=li.lineno,
                indent_cols=li.indent_cols,
                expected_cols=exp,
                paren_level=paren_level,
                problems=tuple(problems),
            )
        )

    return infos, checks, None


def _format_line(
    li: LineInfo,
    lc: LineCheck,
    *,
    show_text: bool,
    show_expected: bool,
    max_text: int,
) -> str:
    parts = [f"L{li.lineno:>4} indent={lc.indent_cols:>3}"]
    if li.tabs or li.spaces:
        parts.append(f"(spaces={li.spaces}, tabs={li.tabs})")
    if show_expected and lc.expected_cols is not None and lc.paren_level == 0:
        parts.append(f"expected={lc.expected_cols:>3}")
    if lc.paren_level is not None:
        parts.append(f"paren={lc.paren_level}")
    if lc.problems:
        parts.append("PROBLEM=" + ",".join(lc.problems))

    if show_text:
        text = li.text.rstrip("\n")
        if len(text) > max_text:
            text = text[: max_text - 3] + "..."
        parts.append("| " + text)
    return " ".join(parts)


def _run_py_compile(path: str) -> tuple[bool, str]:
    """
    Compile using `python -tt -m py_compile` to match the recommended hard gate.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-tt", "-m", "py_compile", path],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return True, ""
        msg = (proc.stderr or proc.stdout or "").strip()
        return False, msg or f"py_compile failed (exit={proc.returncode})"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def _run_tabnanny(path: str) -> tuple[bool, str]:
    """
    Run tabnanny against a file and return (ok, message).
    """
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            tabnanny.check(path)
    except SystemExit as e:
        out = buf.getvalue().strip()
        msg = out if out else f"SystemExit: {e}"
        return False, msg
    except Exception as e:  # noqa: BLE001
        out = buf.getvalue().strip()
        msg = out if out else f"{type(e).__name__}: {e}"
        return False, msg

    out = buf.getvalue().strip()
    if out:
        return False, out
    return True, ""


@dataclass(frozen=True)
class PreflightResult:
    py_compile_ok: bool
    py_compile_msg: str
    tabnanny_ok: bool
    tabnanny_msg: str

    @property
    def ok(self) -> bool:
        return self.py_compile_ok and self.tabnanny_ok


def _self_check(indent_width: int, tabsize: int) -> int:
    cases: list[
        tuple[
            str,
            str,
            tuple[str, ...],
            bool,
            bool | None,
            bool,
        ]
    ] = [
        (
            "else-too-much-indent-jump",
            "def f(x):\n"
            "    if x:\n"
            "        return 1\n"
            "    else:\n"
            "            return 2\n",
            ("INDENT_DELTA_NOT_4_GOT_8",),
            True,
            True,
            False,
        ),
        (
            "tabs-vs-spaces-indent",
            "def f():\n"
            "    if True:\n"
            "\tif True:\n"
            "\t\treturn 1\n"
            "        return 2\n",
            (),
            False,
            False,
            True,
        ),
        (
            "bad-dedent",
            "def f():\n"
            "    if True:\n"
            "        x = 1\n"
            "      y = 2\n",
            (),
            False,
            None,
            True,
        ),
    ]

    overall_ok = True
    for name, src, required, expect_compile_ok, expect_tabnanny_ok, expect_tokenize_error in cases:
        case_ok = True
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                tmp_path = f.name
                f.write(src)

            compile_ok, _compile_msg = _run_py_compile(tmp_path)
            tab_ok, _tab_msg = _run_tabnanny(tmp_path)
            if compile_ok != expect_compile_ok:
                if expect_compile_ok:
                    print("FAIL: compile failed unexpectedly")
                else:
                    print("FAIL: expected compile to fail, but it succeeded")
                case_ok = False

            if expect_tabnanny_ok is not None and tab_ok != expect_tabnanny_ok:
                if expect_tabnanny_ok:
                    print("FAIL: expected tabnanny to pass, but it complained")
                else:
                    print("FAIL: expected tabnanny to complain, but it passed")
                case_ok = False

            infos, checks, tokenize_error = check_file(
                tmp_path, indent_width=indent_width, tabsize=tabsize
            )
            print(f"== self-check: {name} ==")
            if tokenize_error is not None:
                print(f"Tokenize error: {tokenize_error}")

            if expect_tokenize_error and tokenize_error is None:
                print("FAIL: expected a tokenize/indentation error, but got none")
                case_ok = False
            if not expect_tokenize_error and tokenize_error is not None:
                print("FAIL: unexpected tokenize/indentation error")
                case_ok = False

            formatted = "\n".join(
                _format_line(
                    li,
                    checks[li.lineno - 1],
                    show_text=True,
                    show_expected=True,
                    max_text=200,
                )
                for li in infos
                if not _is_blank_or_comment_only(li.text)
            )
            for req in required:
                if req not in formatted:
                    print(f"FAIL: did not find expected marker: {req}")
                    case_ok = False

            if case_ok:
                print("PASS")
            else:
                overall_ok = False
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    return 0 if overall_ok else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Print indentation report for Python files.",
        epilog=(
            "Examples:\n"
            "  python3 py_indent_report.py --only-problems path/to/file.py\n"
            "  python3 py_indent_report.py --all path/to/file.py\n"
            "  python3 py_indent_report.py --self-check\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("paths", nargs="*", help="Python file paths to analyze")
    ap.add_argument(
        "--indent-width", type=int, default=4, help="Indent width in spaces (default: 4)"
    )
    ap.add_argument(
        "--tabsize", type=int, default=8, help="Tab expansion size for column counts (default: 8)"
    )
    ap.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip py_compile preflight (not recommended)",
    )
    ap.add_argument(
        "--no-tabnanny",
        action="store_true",
        help="Skip tabnanny preflight (not recommended)",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Show all non-blank lines (default shows only statement-start + problems)",
    )
    ap.add_argument(
        "--only-problems",
        action="store_true",
        help="Show only lines that look suspicious",
    )
    ap.add_argument(
        "--self-check",
        action="store_true",
        help="Run built-in examples and verify the detector flags common indentation mistakes",
    )
    ap.add_argument(
        "--max-text",
        type=int,
        default=140,
        help="Max characters of source to show per line (default: 140)",
    )
    ap.add_argument(
        "--no-text",
        action="store_true",
        help="Do not print source text (just line numbers / indentation stats)",
    )
    args = ap.parse_args(argv)

    if args.self_check:
        return _self_check(args.indent_width, args.tabsize)

    if not args.paths:
        ap.error("paths are required unless --self-check is used")

    exit_code = 0

    for path in args.paths:
        if not os.path.exists(path):
            print(f"{path}: missing", file=sys.stderr)
            exit_code = 2
            continue

        pre = PreflightResult(True, "", True, "")
        if not args.no_compile:
            ok, msg = _run_py_compile(path)
            pre = PreflightResult(ok, msg, pre.tabnanny_ok, pre.tabnanny_msg)
        if not args.no_tabnanny:
            ok, msg = _run_tabnanny(path)
            pre = PreflightResult(pre.py_compile_ok, pre.py_compile_msg, ok, msg)

        infos, checks, tokenize_error = check_file(
            path, indent_width=args.indent_width, tabsize=args.tabsize
        )

        print(f"== {path} ==")
        if not args.no_compile:
            if pre.py_compile_ok:
                print("py_compile: OK")
            else:
                print(f"py_compile: FAIL ({pre.py_compile_msg})")
                exit_code = max(exit_code, 1)
        if not args.no_tabnanny:
            if pre.tabnanny_ok:
                print("tabnanny: OK")
            else:
                print("tabnanny: FAIL")
                print(pre.tabnanny_msg)
                exit_code = max(exit_code, 1)

        if tokenize_error is not None:
            print(f"Tokenize error: {tokenize_error}")
            exit_code = max(exit_code, 1)

        line_checks = {c.lineno: c for c in checks}
        printed = 0
        problems = 0

        for li in infos:
            lc = line_checks[li.lineno]
            if _is_blank_or_comment_only(li.text):
                continue

            if args.only_problems:
                should_print = lc.is_problem
            elif args.all:
                should_print = True
            else:
                should_print = lc.is_problem or (
                    lc.paren_level == 0 and not _is_blank_or_comment_only(li.text)
                )

            if should_print:
                printed += 1
                if lc.is_problem:
                    problems += 1
                print(
                    _format_line(
                        li,
                        lc,
                        show_text=not args.no_text,
                        show_expected=True,
                        max_text=args.max_text,
                    )
                )

        if problems:
            exit_code = max(exit_code, 1)

        print(f"-- summary: printed={printed} suspicious={problems} --")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


