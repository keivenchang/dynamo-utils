#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read, search, and continue agent transcripts from Cursor, Claude, or Codex.

Sources:
    --source cursor           Cursor  (~/.cursor/projects/*/agent-transcripts)
    --source claude           Claude  (~/.claude/projects)
    --source codex            Codex   (~/.codex/sessions)
    --source all              All of the above
    (default: cursor)

Modes:
    --list                    List recent transcripts
    --latest                  Show / continue the most recent transcript
    --id UUID                 Show / continue a specific transcript (prefix match OK)
    --search KEYWORD          Find transcripts containing a keyword

Output styles (combine with --latest or --id):
    (default)                 Reader view: user messages + last assistant response
    --continue                Continuation prompt (interleaved user + assistant)
    --json                    Machine-readable JSON
    --raw                     Plain [USER]/[ASSISTANT] blocks

Examples:
    %(prog)s --list
    %(prog)s --source claude --list
    %(prog)s --source codex --latest --continue
    %(prog)s --source all --search "mlperf"
    %(prog)s --latest --continue --tail 10
    %(prog)s --id 806f --full
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

CURSOR_ROOT = Path.home() / ".cursor" / "projects"
CLAUDE_ROOT = Path.home() / ".claude" / "projects"
CODEX_ROOT = Path.home() / ".codex" / "sessions"

SOURCES = ("cursor", "claude", "codex")

MAX_TEXT_LEN = 300


@dataclass
class Message:
    role: str
    text: str


@dataclass
class Transcript:
    uuid: str
    path: Path
    project: str
    source: str
    messages: list[Message] = field(default_factory=list)
    mtime: float = 0.0


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _load_cursor_transcript(jsonl_path: Path, project: str) -> Transcript:
    messages: list[Message] = []
    for obj in _iter_jsonl(jsonl_path):
        role = obj.get("role", "unknown")
        content_parts = obj.get("message", {}).get("content", [])
        texts = []
        for part in content_parts if isinstance(content_parts, list) else []:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text", ""))
        if texts:
            messages.append(Message(role=role, text="\n".join(texts)))

    uuid = jsonl_path.stem
    return Transcript(
        uuid=uuid,
        path=jsonl_path,
        project=project,
        source="cursor",
        messages=messages,
        mtime=jsonl_path.stat().st_mtime,
    )


def _load_claude_transcript(jsonl_path: Path, project: str) -> Transcript:
    """Claude transcripts: each line is a typed entry (user/assistant/etc)."""
    messages: list[Message] = []
    for obj in _iter_jsonl(jsonl_path):
        if obj.get("isSidechain"):
            continue  # sub-agent output; skip for the main thread
        kind = obj.get("type")
        if kind not in ("user", "assistant"):
            continue
        msg = obj.get("message") or {}
        role = msg.get("role", kind)
        content = msg.get("content")

        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    parts.append(part.get("text", ""))
                elif ptype == "input_text" or ptype == "output_text":
                    parts.append(part.get("text", ""))
                # skip thinking, tool_use, tool_result, image, etc.
            text = "\n".join(p for p in parts if p)
        else:
            text = ""

        if text.strip():
            messages.append(Message(role=role, text=text))

    uuid = jsonl_path.stem
    return Transcript(
        uuid=uuid,
        path=jsonl_path,
        project=project,
        source="claude",
        messages=messages,
        mtime=jsonl_path.stat().st_mtime,
    )


def _load_codex_transcript(jsonl_path: Path) -> Transcript:
    """Codex rollout files: wrapped event objects with timestamp/type/payload."""
    messages: list[Message] = []
    cwd = ""
    uuid = ""
    for obj in _iter_jsonl(jsonl_path):
        etype = obj.get("type")
        payload = obj.get("payload") or {}

        if etype == "session_meta":
            cwd = payload.get("cwd") or payload.get("working_directory") or cwd
            uuid = payload.get("id") or payload.get("session_id") or uuid
            continue

        # Prefer event_msg (authoritative user/assistant prompts), fall back to
        # response_item messages when event_msg isn't present.
        if etype == "event_msg":
            subtype = payload.get("type")
            if subtype == "user_message":
                text = payload.get("message") or payload.get("text") or ""
                if text.strip():
                    messages.append(Message(role="user", text=text))
            elif subtype == "agent_message":
                text = payload.get("message") or payload.get("text") or ""
                if text.strip():
                    messages.append(Message(role="assistant", text=text))
            continue

        if etype == "response_item":
            if payload.get("type") != "message":
                continue
            role = payload.get("role", "unknown")
            content = payload.get("content") or []
            texts: list[str] = []
            for part in content if isinstance(content, list) else []:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype in ("input_text", "output_text", "text"):
                    texts.append(part.get("text", ""))
            text = "\n".join(t for t in texts if t)
            if text.strip():
                messages.append(Message(role=role, text=text))

    # Deduplicate: if both event_msg and response_item were used, consecutive
    # duplicates may appear. Drop adjacent exact repeats.
    deduped: list[Message] = []
    for m in messages:
        if deduped and deduped[-1].role == m.role and deduped[-1].text == m.text:
            continue
        deduped.append(m)

    if not uuid:
        # Fall back to a deterministic id from the filename
        # pattern: rollout-YYYY-MM-DDTHH-MM-SS-<id>.jsonl
        m = re.search(r"rollout-[^-]+-[^-]+-[^-]+T[^-]+-[^-]+-[^-]+-(.+)", jsonl_path.stem)
        uuid = m.group(1) if m else jsonl_path.stem

    project = cwd or jsonl_path.parent.as_posix()
    return Transcript(
        uuid=uuid,
        path=jsonl_path,
        project=project,
        source="codex",
        messages=deduped,
        mtime=jsonl_path.stat().st_mtime,
    )


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------

def _discover_cursor(project_filter: str | None) -> list[Transcript]:
    if not CURSOR_ROOT.is_dir():
        return []
    results: list[Transcript] = []
    for proj in sorted(CURSOR_ROOT.iterdir()):
        tdir = proj / "agent-transcripts"
        if not tdir.is_dir():
            continue
        if project_filter and not (proj.name == project_filter or proj.name.startswith(project_filter)):
            continue
        for entry in tdir.iterdir():
            if not entry.is_dir() or entry.name == "subagents":
                continue
            jsonl = entry / f"{entry.name}.jsonl"
            if jsonl.is_file():
                results.append(_load_cursor_transcript(jsonl, project=proj.name))
    return results


def _discover_claude(project_filter: str | None) -> list[Transcript]:
    if not CLAUDE_ROOT.is_dir():
        return []
    results: list[Transcript] = []
    for proj in sorted(CLAUDE_ROOT.iterdir()):
        if not proj.is_dir():
            continue
        if project_filter and not (proj.name == project_filter or proj.name.startswith(project_filter)):
            continue
        for jsonl in proj.glob("*.jsonl"):
            if not jsonl.is_file():
                continue
            results.append(_load_claude_transcript(jsonl, project=proj.name))
    return results


def _discover_codex(project_filter: str | None) -> list[Transcript]:
    if not CODEX_ROOT.is_dir():
        return []
    results: list[Transcript] = []
    for jsonl in CODEX_ROOT.rglob("rollout-*.jsonl"):
        if not jsonl.is_file():
            continue
        t = _load_codex_transcript(jsonl)
        if project_filter and not (t.project == project_filter or project_filter in t.project):
            continue
        results.append(t)
    return results


SOURCE_LOADERS = {
    "cursor": _discover_cursor,
    "claude": _discover_claude,
    "codex": _discover_codex,
}


def _discover(sources: list[str], project_filter: str | None) -> list[Transcript]:
    results: list[Transcript] = []
    for src in sources:
        results.extend(SOURCE_LOADERS[src](project_filter))
    results.sort(key=lambda t: t.mtime, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _extract_user_query(text: str) -> str:
    """Pull text inside <user_query> tags, or strip system-injected blocks."""
    m = re.search(r"<user_query>\s*(.*?)\s*</user_query>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    cleaned = re.sub(r"<open_and_recently_viewed_files>.*?</open_and_recently_viewed_files>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<user_info>.*?</user_info>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<system_reminder>.*?</system_reminder>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<attached_files>.*?</attached_files>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    return cleaned.strip()


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _first_user_query(t: Transcript) -> str:
    for m in t.messages:
        if m.role == "user":
            q = _extract_user_query(m.text)
            if q:
                return q[:120]
    return "(no user messages)"


def _last_exchange(t: Transcript) -> tuple[str, str]:
    """Return (last_user_msg, last_assistant_snippet) for quick context."""
    last_user = ""
    last_asst = ""
    for m in reversed(t.messages):
        if m.role == "user" and not last_user:
            last_user = _extract_user_query(m.text).replace("\n", " ").strip()
        elif m.role == "assistant" and not last_asst:
            last_asst = m.text.replace("\n", " ").strip()
        if last_user and last_asst:
            break
    return last_user, last_asst


def _recent_user_messages(t: Transcript, count: int = 3) -> list[str]:
    """Return last N non-trivial user messages (cleaned, deduplicated)."""
    seen: set[str] = set()
    result: list[str] = []
    for m in reversed(t.messages):
        if m.role != "user":
            continue
        q = _extract_user_query(m.text).replace("\n", " ").strip()
        if not q or q in seen:
            continue
        seen.add(q)
        result.append(q)
        if len(result) >= count:
            break
    result.reverse()
    return result


def _recent_exchanges(t: Transcript, count: int) -> list[tuple[str, str]]:
    """Return the last N (user, assistant) exchange pairs, oldest first.

    Walks backwards pairing each non-empty user message with the nearest
    following assistant reply. Deduplicates adjacent user messages that were
    already shown.
    """
    pairs: list[tuple[str, str]] = []
    pending_asst = ""
    seen_user: set[str] = set()
    for m in reversed(t.messages):
        if m.role == "assistant":
            if not pending_asst and m.text.strip():
                pending_asst = m.text.replace("\n", " ").strip()
        elif m.role == "user":
            q = _extract_user_query(m.text).replace("\n", " ").strip()
            if not q or q in seen_user:
                continue
            seen_user.add(q)
            pairs.append((q, pending_asst))
            pending_asst = ""
            if len(pairs) >= count:
                break
    pairs.reverse()
    return pairs


def _time_ago(mtime: float, now: float | None = None) -> str:
    """Human-friendly 'how long ago' suffix, e.g. '2h ago', '3d ago'."""
    if now is None:
        now = time.time()
    delta = max(0, int(now - mtime))
    if delta < 60:
        return f"{delta}s ago"
    if delta < 3600:
        return f"{delta // 60}m ago"
    if delta < 86400:
        return f"{delta // 3600}h ago"
    if delta < 86400 * 30:
        return f"{delta // 86400}d ago"
    if delta < 86400 * 365:
        return f"{delta // (86400 * 30)}mo ago"
    return f"{delta // (86400 * 365)}y ago"


def _term_width(default: int = 160) -> int:
    try:
        return max(80, shutil.get_terminal_size((default, 24)).columns)
    except (OSError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Output: --list / --search
# ---------------------------------------------------------------------------

def _print_transcript_table(
    transcripts: list[Transcript],
    *,
    verbose: bool = False,
    exchanges: int = 3,
    width: int | None = None,
) -> None:
    width = width or _term_width()
    body_width = max(80, width - 10)  # indent for '       > '
    topic_width = max(40, width - 80)
    project_width = 24

    header = (
        f"{'#':>3}  {'Date':16}  {'Age':>8}  {'Src':6}  "
        f"{'U':>3}  {'A':>3}  {'Project':{project_width}}  {'UUID':36}  Topic"
    )
    print(header)
    print("-" * min(width, len(header) + topic_width))
    now = time.time()
    for i, t in enumerate(transcripts):
        ts = datetime.fromtimestamp(t.mtime).strftime("%Y-%m-%d %H:%M")
        age = _time_ago(t.mtime, now)
        n_user = sum(1 for m in t.messages if m.role == "user")
        n_asst = sum(1 for m in t.messages if m.role == "assistant")
        topic = _first_user_query(t)
        print(
            f"{i:3}  {ts}  {age:>8}  {t.source:6}  "
            f"{n_user:3}  {n_asst:3}  "
            f"{_truncate(t.project, project_width):{project_width}}  "
            f"{t.uuid:36}  {_truncate(topic, topic_width)}"
        )

        if verbose:
            print(f"       path: {t.path}")

        pair_limit = exchanges if not verbose else max(exchanges, 5)
        pairs = _recent_exchanges(t, count=pair_limit)
        if not pairs:
            print()
            continue

        user_trunc = body_width if verbose else min(body_width, 160)
        asst_trunc = body_width if verbose else min(body_width, 160)

        for user_q, asst_a in pairs:
            print(f"       > {_truncate(user_q, user_trunc)}")
            if asst_a:
                print(f"       = {_truncate(asst_a, asst_trunc)}")
        print()


def cmd_list(
    all_transcripts: list[Transcript],
    limit: int,
    *,
    verbose: bool = False,
    exchanges: int = 3,
    width: int | None = None,
) -> None:
    transcripts = all_transcripts[:limit]
    if not transcripts:
        print("No transcripts found.")
        return
    total = len(all_transcripts)
    suffix = f" of {total}" if total > len(transcripts) else ""
    print(f"Recent transcripts ({len(transcripts)}{suffix} shown):\n")
    _print_transcript_table(
        transcripts, verbose=verbose, exchanges=exchanges, width=width,
    )


def cmd_search(
    all_transcripts: list[Transcript],
    keyword: str,
    limit: int,
    *,
    verbose: bool = False,
    exchanges: int = 3,
    width: int | None = None,
) -> None:
    keyword_lower = keyword.lower()
    matches: list[Transcript] = []
    for t in all_transcripts:
        for m in t.messages:
            if keyword_lower in m.text.lower():
                matches.append(t)
                break
    matches = matches[:limit]
    if not matches:
        print(f"No transcripts matching '{keyword}'.", file=sys.stderr)
        sys.exit(1)
    print(f"Transcripts matching '{keyword}' ({len(matches)} found):\n")
    _print_transcript_table(
        matches, verbose=verbose, exchanges=exchanges, width=width,
    )


# ---------------------------------------------------------------------------
# Output: default reader view
# ---------------------------------------------------------------------------

def cmd_show(t: Transcript, full: bool, tail: int | None) -> None:
    user_msgs: list[tuple[int, str]] = []
    for i, m in enumerate(t.messages):
        if m.role == "user":
            user_msgs.append((i, _extract_user_query(m.text)))

    title = user_msgs[0][1] if user_msgs else "(no user messages)"
    max_len = 9999 if full else MAX_TEXT_LEN
    n_user = len(user_msgs)
    n_asst = sum(1 for m in t.messages if m.role == "assistant")

    print("=" * 70)
    print(f"Transcript: {t.uuid}")
    print(f"Source: {t.source}")
    print(f"Project: {t.project}")
    print(f"Messages: {len(t.messages)}  |  User: {n_user}  |  Assistant: {n_asst}")
    print(f"Started with: {_truncate(title, 120)}")
    print("=" * 70)

    display_msgs = user_msgs[-tail:] if tail is not None else user_msgs
    print(f"\n--- User Messages ({len(display_msgs)}/{n_user}) ---\n")
    for line_num, text in display_msgs:
        print(f"  [{line_num:3}] {_truncate(text, max_len)}")

    last_assistant = ""
    for m in reversed(t.messages):
        if m.role == "assistant" and m.text.strip():
            last_assistant = m.text
            break
    if last_assistant:
        print(f"\n--- Last Assistant Response ---\n")
        print(_truncate(last_assistant, 800 if not full else 9999))

    if user_msgs:
        _, last_text = user_msgs[-1]
        print(f"\n{'=' * 70}")
        print(f"LAST USER REQUEST:")
        print(f"{'=' * 70}")
        print(last_text)
    print()


# ---------------------------------------------------------------------------
# Output: --continue (continuation prompt)
# ---------------------------------------------------------------------------

def cmd_continue(t: Transcript, tail: int) -> None:
    topic = _first_user_query(t)
    n_user = sum(1 for m in t.messages if m.role == "user")
    n_asst = sum(1 for m in t.messages if m.role == "assistant")
    recent = t.messages[-tail:]

    if len(t.messages) > tail:
        header_note = f"(showing last {tail} of {len(t.messages)} messages; {len(t.messages) - tail} earlier omitted)"
    else:
        header_note = f"(full conversation: {len(t.messages)} messages)"

    print(f"# Continuing previous conversation\n")
    print(f"**Source:** {t.source}")
    print(f"**Transcript:** {t.uuid}")
    print(f"**Project:** {t.project}")
    print(f"**Messages:** {n_user} user, {n_asst} assistant {header_note}")
    print(f"**Topic:** {topic}")
    print(f"\n---\n## Conversation so far\n")

    for msg in recent:
        role_label = "USER" if msg.role == "user" else "ASSISTANT"
        text = _extract_user_query(msg.text) if msg.role == "user" else msg.text
        if msg.role == "assistant" and len(text) > 2000:
            text = text[:2000] + "\n... [truncated]"
        print(f"### {role_label}")
        print(text)
        print()

    print("---\n")
    print("Please continue from where we left off. Review the conversation above and pick up where the previous session ended.")


# ---------------------------------------------------------------------------
# Output: --raw
# ---------------------------------------------------------------------------

def cmd_raw(t: Transcript, tail: int) -> None:
    for msg in t.messages[-tail:]:
        role_label = "USER" if msg.role == "user" else "ASSISTANT"
        text = _extract_user_query(msg.text) if msg.role == "user" else msg.text
        print(f"[{role_label}] {text}\n")


# ---------------------------------------------------------------------------
# Output: --json
# ---------------------------------------------------------------------------

def cmd_json(t: Transcript, full: bool) -> None:
    user_msgs = [_extract_user_query(m.text) for m in t.messages if m.role == "user"]
    last_assistant = ""
    for m in reversed(t.messages):
        if m.role == "assistant" and m.text.strip():
            last_assistant = m.text
            break
    out = {
        "uuid": t.uuid,
        "source": t.source,
        "project": t.project,
        "path": str(t.path),
        "total_messages": len(t.messages),
        "user_messages": user_msgs if full else user_msgs[-20:],
        "last_assistant": last_assistant if full else last_assistant[:2000],
        "last_user": user_msgs[-1] if user_msgs else "",
    }
    print(json.dumps(out, indent=2))


# ---------------------------------------------------------------------------
# Resolve target transcript
# ---------------------------------------------------------------------------

def _resolve_target(all_transcripts: list[Transcript], args) -> Transcript:
    if not all_transcripts:
        print("No transcripts found.", file=sys.stderr)
        sys.exit(1)

    if args.latest:
        t = all_transcripts[0]
        # Auto-skip tiny transcripts (likely the current session asking to continue)
        if len(t.messages) <= 2 and len(all_transcripts) > 1:
            t = all_transcripts[1]
        return t

    matches = [t for t in all_transcripts if t.uuid == args.id or t.uuid.startswith(args.id)]
    if not matches:
        print(f"No transcript matching '{args.id}'", file=sys.stderr)
        sys.exit(1)
    unique_ids = {t.uuid for t in matches}
    if len(unique_ids) > 1:
        print(f"Multiple transcripts match '{args.id}':", file=sys.stderr)
        for t in matches[:10]:
            print(f"  [{t.source}] {t.uuid}  [{t.project}]  {_first_user_query(t)}", file=sys.stderr)
        print("Use a longer UUID prefix or pass --project / --source.", file=sys.stderr)
        sys.exit(1)
    return matches[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read, search, and continue agent transcripts (Cursor / Claude / Codex).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                              List recent Cursor transcripts
  %(prog)s --source claude --list              List recent Claude transcripts
  %(prog)s --source all --list                 Mix all sources, newest first
  %(prog)s --latest                            Reader view of most recent Cursor transcript
  %(prog)s --source codex --latest --continue  Continuation prompt for most recent Codex session
  %(prog)s --id 806f --full                    Full transcript by UUID prefix (default source)
  %(prog)s --source all --search "mlperf"      Find transcripts mentioning "mlperf" anywhere
  %(prog)s --latest --json                     JSON output for scripting

Source defaults:
  cursor  : ~/.cursor/projects/<proj>/agent-transcripts/<uuid>/<uuid>.jsonl
  claude  : ~/.claude/projects/<encoded-cwd>/<uuid>.jsonl
  codex   : ~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl
""",
    )
    parser.add_argument(
        "--source",
        choices=(*SOURCES, "all"),
        default="cursor",
        help="Transcript source (default: cursor). Use 'all' to combine.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--latest", action="store_true", help="Most recent transcript")
    group.add_argument("--id", metavar="UUID", help="Transcript by UUID (prefix match OK)")
    group.add_argument("--list", action="store_true", help="List recent transcripts")
    group.add_argument("--search", metavar="KEYWORD", help="Search transcripts for a keyword")

    parser.add_argument("--continue", action="store_true", dest="do_continue",
                        help="Generate a continuation prompt (interleaved user + assistant)")
    parser.add_argument("--raw", action="store_true", help="Plain [USER]/[ASSISTANT] output")
    parser.add_argument("--json", action="store_true", dest="output_json",
                        help="JSON output for scripting")
    parser.add_argument("--full", action="store_true", help="Don't truncate messages")
    parser.add_argument("--tail", type=int, default=None,
                        help="Last N messages (default: all for reader, 20 for --continue/--raw)")
    parser.add_argument("--limit", type=int, default=20,
                        help="Number of transcripts to list/search (default: 20)")
    parser.add_argument(
        "--project",
        metavar="NAME",
        help="Restrict to a project name (prefix match; for codex, substring match on cwd).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="For --list/--search: show more exchanges, full width, and file paths.",
    )
    parser.add_argument(
        "--exchanges", type=int, default=3,
        help="For --list/--search: number of recent (user, assistant) pairs per "
             "transcript (default: 3; --verbose bumps to at least 5).",
    )
    parser.add_argument(
        "--width", type=int, default=None,
        help="Override terminal width for --list/--search output (default: auto-detect).",
    )

    args = parser.parse_args()

    sources = list(SOURCES) if args.source == "all" else [args.source]
    all_transcripts = _discover(sources, project_filter=args.project)

    if not all_transcripts:
        detail = f" for project '{args.project}'" if args.project else ""
        print(f"ERROR: No transcripts found{detail} in sources: {', '.join(sources)}.",
              file=sys.stderr)
        return 1

    if args.list:
        cmd_list(
            all_transcripts,
            limit=args.limit,
            verbose=args.verbose,
            exchanges=args.exchanges,
            width=args.width,
        )
        return 0

    if args.search:
        cmd_search(
            all_transcripts,
            args.search,
            limit=args.limit,
            verbose=args.verbose,
            exchanges=args.exchanges,
            width=args.width,
        )
        return 0

    t = _resolve_target(all_transcripts, args)
    if not t.messages:
        print(f"Transcript {t.uuid} has no messages.", file=sys.stderr)
        return 1

    if args.output_json:
        cmd_json(t, full=args.full)
    elif args.do_continue:
        cmd_continue(t, tail=args.tail or 20)
    elif args.raw:
        cmd_raw(t, tail=args.tail or 20)
    else:
        cmd_show(t, full=args.full, tail=args.tail)

    return 0


if __name__ == "__main__":
    sys.exit(main())
