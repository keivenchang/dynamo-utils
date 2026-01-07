"""HTML rendering for ci_log_errors snippets (shared library)."""

from __future__ import annotations

# pyright: reportUndefinedVariable=false
# ^ This module intentionally shares a large set of regex/constants/helpers with `ci_log_errors.engine`
#   via `globals().update(engine.__dict__)` (see below). Static analyzers can’t easily follow that,
#   so we suppress undefined-variable noise here.

import functools
import html
import json
import re
from pathlib import Path
from typing import List, Pattern

# To avoid import cycles, `engine.py` must not import this module at import time.

# Populate this module namespace with the shared helpers/constants from `engine.py`
# so the extracted code can remain unchanged.
from . import engine as _engine  # noqa: E402

globals().update(_engine.__dict__)

def html_highlight_error_keywords(text: str) -> str:
    from . import engine as _E
    """HTML: escape and keyword-highlight error tokens (inline highlighting)."""
    # Don't keyword-highlight this common post-failure docker noise.
    # It's useful to *show* in snippets sometimes, but shouldn't draw attention.
    if RED_DOCKER_NO_SUCH_CONTAINER_RE.search(text or ""):
        return html.escape(text or "")

    escaped = html.escape(text or "")
    if not escaped:
        return ""

    def repl(m: re.Match) -> str:
        # Slightly deeper red than GitHub default; keep readable and not overly saturated.
        # (No bold: user wants red without extra emphasis.)
        return f'<span style="color: #c83a3a;">{m.group(0)}</span>'

    return RED_KEYWORD_HIGHLIGHT_RE.sub(repl, escaped)


def categorize_error_snippet_text(snippet_text: str) -> List[str]:
    from . import engine as _E
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
    from . import engine as _E
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
    cmd_block_idx = 0
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
            # UX: some command blocks are *suggestions* and we render them as shell comments:
            #   # [suggested]: <cmd>
            # When copying, strip the comment marker + suggested marker so the clipboard contains
            # a runnable command.
            cmd_copy_text = cmd_text
            try:
                cleaned: List[str] = []
                for ln in (cmd_text or "").splitlines():
                    s = str(ln or "")
                    # New format: "# [suggested]: <cmd>"
                    m = re.match(r"^\s*#\s*\[\s*suggested\s*\]\s*:\s*(.*)$", s, flags=re.IGNORECASE)
                    if m:
                        cleaned.append(str(m.group(1) or "").rstrip())
                        continue
                    # Back-compat: "# suggested: <cmd>"
                    m2 = re.match(r"^\s*#\s*suggested\s*:\s*(.*)$", s, flags=re.IGNORECASE)
                    if m2:
                        cleaned.append(str(m2.group(1) or "").rstrip())
                        continue
                    # Back-compat: "# <cmd>   # suggested"
                    if re.search(r"#\s*suggested\s*$", s, flags=re.IGNORECASE):
                        ln2 = re.sub(r"#\s*suggested\s*$", "", s, flags=re.IGNORECASE).rstrip()
                        ln2 = re.sub(r"^\s*#\s*", "", ln2).rstrip()
                        cleaned.append(ln2)
                        continue
                    else:
                        cleaned.append(ln)
                cmd_copy_text = "\n".join(cleaned).strip("\n")
            except Exception:
                cmd_copy_text = cmd_text

            cmd_js = html.escape(json.dumps(cmd_copy_text), quote=True)
            cmd_html = html.escape(cmd_text)
            text_style = _SNIP_COPY_TEXT_STYLE + ("; font-weight: 600;" if cmd_block_idx == 0 else "")
            out_lines.append(
                f'<span style="{_SNIP_COPY_ROW_STYLE}">'
                f'<button type="button" onclick="event.stopPropagation(); try {{ copyToClipboard({cmd_js}, this); }} catch (e) {{}}" '
                f'style="{_SNIP_COPY_BTN_STYLE}" title="Copy command">{_copy_icon_svg(size_px=12)}</button>'
                f'<span style="{text_style}">{cmd_html}</span>'
                "</span>"
            )
            cmd_block_idx += 1
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

        if SNIPPET_PYTEST_FAILED_LINE_RE.search(s_norm) or any(r.search(s_norm) for r in RED_FULL_LINE_RES):
            out_lines.append(
                f'<span style="color: #c83a3a;">{html.escape(display_line)}</span>'
            )
        elif SNIPPET_COMMAND_LINE_BLUE_RE.search(s_norm):
            # Special-case: make PYTEST_CMD=... copyable (high-signal and often very long).
            if SNIPPET_PYTEST_CMD_LINE_RE.search(s_norm):
                # Extract payload after the first "=" and strip a single matching quote pair.
                payload = ""
                try:
                    rhs = str(s_norm).split("=", 1)[1] if "=" in str(s_norm) else ""
                    rhs = rhs.strip()
                    if len(rhs) >= 2 and rhs[0] in ("'", '"') and rhs[-1] == rhs[0]:
                        rhs = rhs[1:-1]
                    payload = rhs
                except Exception:
                    payload = ""
                payload_js = html.escape(json.dumps(payload), quote=True)
                out_lines.append(
                    f'<span style="{_SNIP_COPY_ROW_STYLE}">'
                    f'<button type="button" onclick="event.stopPropagation(); try {{ copyToClipboard({payload_js}, this); }} catch (e) {{}}" '
                    f'style="{_SNIP_COPY_BTN_STYLE}" title="Copy pytest command">{_copy_icon_svg(size_px=12)}</button>'
                    f'<span style="{_SNIP_COPY_TEXT_STYLE}">{html.escape(display_line)}</span>'
                    "</span>"
                )
            # Also copy-enable `bash -c "...pytest..."` lines (common execution context).
            elif re.search(r"\bbash\s+-c\s+['\"][^'\"]*\bpytest\b", s_norm, flags=re.IGNORECASE):
                payload = ""
                try:
                    payload = str(display_line or "").strip()
                except Exception:
                    payload = ""
                payload_js = html.escape(json.dumps(payload), quote=True)
                out_lines.append(
                    f'<span style="{_SNIP_COPY_ROW_STYLE}">'
                    f'<button type="button" onclick="event.stopPropagation(); try {{ copyToClipboard({payload_js}, this); }} catch (e) {{}}" '
                    f'style="{_SNIP_COPY_BTN_STYLE}" title="Copy command">{_copy_icon_svg(size_px=12)}</button>'
                    f'<span style="{_SNIP_COPY_TEXT_STYLE}">{html.escape(display_line)}</span>'
                    "</span>"
                )
            else:
                out_lines.append(
                    f'<span style="color: #0969da;">{html.escape(display_line)}</span>'
                )
        else:
            out_lines.append(html_highlight_error_keywords(display_line))

        i += 1

    return "\n".join(out_lines)


#
