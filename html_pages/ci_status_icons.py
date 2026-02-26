# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CI status icon rendering: SVG generators, badges, and compact summary HTML."""

from __future__ import annotations

import html
from typing import Dict, List

from common_types import CIStatus

COLOR_GREEN = "#2da44e"
COLOR_RED = "#c83a3a"
COLOR_GREY = "#8c959f"
COLOR_YELLOW = "#bf8700"


def _octicon_svg(*, path_d: str, name: str, width: int = 12, height: int = 12) -> str:
    """Return a fixed-size Octicon-like SVG (16x16 viewBox) using currentColor fill."""
    pd = str(path_d or "").strip()
    if not pd:
        return ""
    nm = html.escape(str(name or "octicon"), quote=True)
    return (
        f'<svg aria-hidden="true" viewBox="0 0 16 16" version="1.1" '
        f'width="{int(width)}" height="{int(height)}" data-view-component="true" '
        f'class="octicon {nm}" fill="currentColor">'
        f'<path fill-rule="evenodd" d="{pd}"></path></svg>'
    )


def _circle_x_fill_svg(*, color: str, width: int = 12, height: int = 12, extra_style: str = "") -> str:
    """Filled circle with a white X (SVG)."""
    st = f"color: {html.escape(str(color or ''), quote=True)}; display: inline-flex; vertical-align: text-bottom;"
    if str(extra_style or "").strip():
        st += " " + str(extra_style).strip()
    return (
        f'<span style="{st}">'
        f'<svg aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="{int(width)}" height="{int(height)}" '
        f'data-view-component="true" class="octicon octicon-x-circle-fill" fill="currentColor">'
        '<circle cx="8" cy="8" r="8" fill="currentColor"></circle>'
        '<path d="M4.5 4.5l7 7m-7 0l7-7" stroke="#fff" stroke-width="2" stroke-linecap="round"></path>'
        "</svg></span>"
    )


def _circle_dot_fill_svg(*, color: str, width: int = 12, height: int = 12, extra_style: str = "") -> str:
    """Filled circle with a white dot (SVG) for 'pending'."""
    st = f"color: {html.escape(str(color or ''), quote=True)}; display: inline-flex; vertical-align: text-bottom;"
    if str(extra_style or "").strip():
        st += " " + str(extra_style).strip()
    return (
        f'<span style="{st}">'
        f'<svg aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="{int(width)}" height="{int(height)}" '
        f'data-view-component="true" class="octicon octicon-dot-circle-fill" fill="currentColor">'
        '<circle cx="8" cy="8" r="8" fill="currentColor"></circle>'
        '<circle cx="8" cy="8" r="2.2" fill="#fff"></circle>'
        "</svg></span>"
    )


def _clock_ring_svg(*, color: str, width: int = 12, height: int = 12, extra_style: str = "") -> str:
    """Clock/ring icon (SVG) for 'in progress'."""
    st = f"color: {html.escape(str(color or ''), quote=True)}; display: inline-flex; vertical-align: text-bottom;"
    if str(extra_style or "").strip():
        st += " " + str(extra_style).strip()
    return (
        f'<span style="{st}">'
        f'<svg aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="{int(width)}" height="{int(height)}" '
        f'data-view-component="true" class="octicon octicon-clock" fill="currentColor">'
        '<path d="M8 1C4.1 1 1 4.1 1 8s3.1 7 7 7 7-3.1 7-7-3.1-7-7-7zm0 12c-2.8 0-5-2.2-5-5s2.2-5 5-5 5 2.2 5 5-2.2 5-5 5z"></path>'
        '<path d="M8 4v5l3 2"></path>'
        "</svg></span>"
    )


def _dot_svg(*, color: str, width: int = 12, height: int = 12, extra_style: str = "") -> str:
    """Small dot (SVG) for 'unknown/other'."""
    st = f"color: {html.escape(str(color or ''), quote=True)}; display: inline-flex; vertical-align: text-bottom;"
    if str(extra_style or "").strip():
        st += " " + str(extra_style).strip()
    return (
        f'<span style="{st}">'
        f'<svg aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="{int(width)}" height="{int(height)}" '
        f'data-view-component="true" class="octicon octicon-dot" fill="currentColor">'
        '<circle cx="8" cy="8" r="2.6" fill="currentColor"></circle>'
        "</svg></span>"
    )


def status_icon_html(
    *,
    status_norm: str,
    is_required: bool,
    required_failure: bool = False,
    warning_present: bool = False,
    icon_px: int = 12,
) -> str:
    """Shared status icon HTML (match all dashboards).

    Conventions:
    - REQUIRED success: green filled circle-check
    - REQUIRED failure: red filled circle X
    - non-required success: small check
    - non-required failure: small X
    - Synthetic items (icon_px=7): colored dots instead of checkmarks/X
    """
    s = (status_norm or "").strip().lower()

    icon_px_i = int(icon_px or 12)
    is_synthetic = (icon_px_i == 7)

    if s == CIStatus.SUCCESS:
        if is_synthetic:
            return _circle_dot_fill_svg(color=COLOR_GREEN, width=12, height=12)
        if bool(is_required):
            out = (
                f'<span style="color: {COLOR_GREEN}; display: inline-flex; vertical-align: text-bottom;">'
                f'<svg aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="{icon_px_i}" height="{icon_px_i}" '
                'data-view-component="true" class="octicon octicon-check-circle-fill" fill="currentColor">'
                '<path fill-rule="evenodd" '
                'd="M8 16A8 8 0 108 0a8 8 0 000 16zm3.78-9.78a.75.75 0 00-1.06-1.06L7 9.94 5.28 8.22a.75.75 0 10-1.06 1.06l2 2a.75.75 0 001.06 0l4-4z">'
                "</path></svg></span>"
            )
        else:
            out = (
                f'<span style="color: {COLOR_GREEN}; display: inline-flex; vertical-align: text-bottom;">'
                f'{_octicon_svg(path_d="M13.78 4.22a.75.75 0 00-1.06 0L6.75 10.19 3.28 6.72a.75.75 0 10-1.06 1.06l4 4a.75.75 0 001.06 0l7.5-7.5a.75.75 0 000-1.06z", name="octicon-check", width=icon_px_i, height=icon_px_i)}'
                "</span>"
            )
        if bool(warning_present):
            out += '<span style="color: #57606a; font-size: 11px; margin: 0 2px;">/</span>'
            if bool(required_failure):
                out += _circle_x_fill_svg(color=COLOR_RED, width=icon_px_i, height=icon_px_i, extra_style="margin-left: 2px;")
            else:
                out += (
                    f'<span style="color: {COLOR_RED}; display: inline-flex; vertical-align: text-bottom; margin-left: 2px;">'
                    f'{_octicon_svg(path_d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 11-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z", name="octicon-x", width=icon_px_i, height=icon_px_i)}'
                    "</span>"
                )
        return out
    if s in {CIStatus.SKIPPED, CIStatus.NEUTRAL}:
        if is_synthetic:
            return _circle_dot_fill_svg(color=COLOR_GREY, width=12, height=12)
        return (
            '<span style="color: #8c959f; display: inline-flex; vertical-align: text-bottom;">'
            f'<svg aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="{icon_px_i}" height="{icon_px_i}" '
            'data-view-component="true" class="octicon octicon-circle-slash" fill="currentColor">'
            '<path fill-rule="evenodd" '
            'd="M8 16A8 8 0 108 0a8 8 0 000 16ZM1.5 8a6.5 6.5 0 0110.364-5.083l-8.947 8.947A6.473 6.473 0 011.5 8Zm3.136 5.083 8.947-8.947A6.5 6.5 0 014.636 13.083Z">'
            "</path></svg></span>"
        )
    if s == CIStatus.FAILURE:
        if is_synthetic:
            return _circle_dot_fill_svg(color=COLOR_RED, width=12, height=12)
        if bool(is_required or required_failure):
            return _circle_x_fill_svg(color=COLOR_RED, width=icon_px_i, height=icon_px_i)
        return (
            f'<span style="color: {COLOR_RED}; display: inline-flex; vertical-align: text-bottom;">'
            f'{_octicon_svg(path_d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 11-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z", name="octicon-x", width=icon_px_i, height=icon_px_i)}'
            "</span>"
        )
    if s == CIStatus.IN_PROGRESS:
        if is_synthetic:
            return _circle_dot_fill_svg(color=COLOR_YELLOW, width=12, height=12)
        return _clock_ring_svg(color=COLOR_YELLOW, width=icon_px_i, height=icon_px_i)
    if s == CIStatus.PENDING:
        return _circle_dot_fill_svg(color=COLOR_GREY, width=icon_px_i, height=icon_px_i)
    if s == CIStatus.CANCELLED:
        if is_synthetic:
            return _circle_dot_fill_svg(color=COLOR_GREY, width=12, height=12)
        return _circle_x_fill_svg(color=COLOR_GREY, width=icon_px_i, height=icon_px_i)
    if s == "cancelled-timeout":
        if bool(is_required):
            return _circle_x_fill_svg(color=COLOR_RED, width=icon_px_i, height=icon_px_i)
        return (
            f'<span style="color: {COLOR_RED}; display: inline-flex; vertical-align: text-bottom;">'
            f'{_octicon_svg(path_d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 11-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z", name="octicon-x", width=icon_px_i, height=icon_px_i)}'
            "</span>"
        )
    if is_synthetic:
        return _circle_dot_fill_svg(color=COLOR_GREY, width=12, height=12)
    return _dot_svg(color=COLOR_GREY, width=icon_px_i, height=icon_px_i)


def ci_status_icon_context() -> Dict[str, str]:
    """Template context: consistent icon HTML used by all dashboards (legend, tooltips, status bar)."""
    return {
        "success_icon_html": status_icon_html(status_norm="success", is_required=False),
        "success_required_icon_html": status_icon_html(status_norm="success", is_required=True),
        "failure_required_icon_html": status_icon_html(status_norm="failure", is_required=True),
        "failure_optional_icon_html": status_icon_html(status_norm="failure", is_required=False),
        "in_progress_icon_html": status_icon_html(status_norm="in_progress", is_required=False),
        "pending_icon_html": status_icon_html(status_norm="pending", is_required=False),
        "cancelled_icon_html": status_icon_html(status_norm="cancelled", is_required=False),
        "skipped_icon_html": status_icon_html(status_norm="skipped", is_required=False),
    }


EXPECTED_CHECK_PLACEHOLDER_SYMBOL = "◇"

KNOWN_ERROR_MARKERS = frozenset({
    "pytest-timeout",
    "exceed-action-timeout",
    "backend-failure",
    "trtllm-error",
    "vllm-error",
    "sglang-error",
    "cuda-error",
    "oom-error",
    "import-error",
    "timeout",
})

PASS_PLUS_STYLE = "font-size: 10px; font-weight: 600; opacity: 0.9;"


def compact_ci_summary_html(
    *,
    success_required: int = 0,
    success_optional: int = 0,
    failure_required: int = 0,
    failure_optional: int = 0,
    in_progress_required: int = 0,
    in_progress_optional: int = 0,
    pending: int = 0,
    cancelled: int = 0,
) -> str:
    """Render the compact CI summary used in the GitHub column (shared across dashboards)."""
    sr = int(success_required or 0)
    so = int(success_optional or 0)
    fr = int(failure_required or 0)
    fo = int(failure_optional or 0)
    ip = int(in_progress_required or 0) + int(in_progress_optional or 0)
    pd = int(pending or 0)
    cx = int(cancelled or 0)

    parts: List[str] = []

    if sr > 0:
        parts.append(
            f'<span style="color: {COLOR_GREEN};" title="Passed (required)">'
            f'{status_icon_html(status_norm=CIStatus.SUCCESS.value, is_required=True)}'
            f"<strong>{sr}</strong></span>"
        )
    if so > 0:
        parts.append(
            f'<span style="color: {COLOR_GREEN};" title="Passed (optional)">'
            f'{status_icon_html(status_norm=CIStatus.SUCCESS.value, is_required=False)}'
            f"<strong>{so}</strong></span>"
        )
    if fr > 0:
        parts.append(
            f'<span style="color: {COLOR_RED};" title="Failed (required)">'
            f'{status_icon_html(status_norm=CIStatus.FAILURE.value, is_required=True)}'
            f"<strong>{fr}</strong></span>"
        )
    if fo > 0:
        parts.append(
            f'<span style="color: {COLOR_RED};" title="Failed (optional)">'
            f'{status_icon_html(status_norm=CIStatus.FAILURE.value, is_required=False)}'
            f"<strong>{fo}</strong></span>"
        )
    if ip > 0:
        parts.append(
            f'<span style="color: {COLOR_GREY};" title="In progress">'
            f'{status_icon_html(status_norm=CIStatus.IN_PROGRESS.value, is_required=False)}'
            f"<strong>{ip}</strong></span>"
        )
    if pd > 0:
        parts.append(
            f'<span style="color: {COLOR_GREY};" title="Pending">'
            f'{status_icon_html(status_norm=CIStatus.PENDING.value, is_required=False)}'
            f"<strong>{pd}</strong></span>"
        )
    if cx > 0:
        parts.append(
            f'<span style="color: {COLOR_GREY};" title="Canceled">'
            f'{status_icon_html(status_norm=CIStatus.CANCELLED.value, is_required=False)}'
            f"<strong>{cx}</strong></span>"
        )

    return " ".join([p for p in parts if str(p or "").strip()])


def required_badge_html(*, is_required: bool, status_norm: str) -> str:
    """Render a [REQUIRED] badge with shared semantics."""
    if not is_required:
        return ""

    s = (status_norm or "").strip().lower()
    if s == CIStatus.FAILURE:
        color = COLOR_RED
        weight = "400"
    elif s == CIStatus.SUCCESS:
        color = COLOR_GREEN
        weight = "400"
    else:
        color = "#57606a"
        weight = "400"

    return f' <span style="color: {color}; font-weight: {weight};">[REQUIRED]</span> '


def mandatory_badge_html(*, is_mandatory: bool, status_norm: str) -> str:
    """Render a [MANDATORY] badge (GitLab) following the same color convention as [REQUIRED]."""
    if not is_mandatory:
        return ""

    s = (status_norm or "").strip().lower()
    if s == CIStatus.FAILURE:
        color = COLOR_RED
        weight = "700"
    elif s == CIStatus.SUCCESS:
        color = COLOR_GREEN
        weight = "400"
    else:
        color = "#57606a"
        weight = "400"

    return f' <span style="color: {color}; font-weight: {weight};">[MANDATORY]</span>'
