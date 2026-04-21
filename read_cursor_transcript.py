#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read, search, and continue Cursor agent transcripts.

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
    %(prog)s --latest
    %(prog)s --latest --continue
    %(prog)s --latest --continue --tail 10
    %(prog)s --id 806f --full
    %(prog)s --search "mlperf"
    %(prog)s --latest --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

TRANSCRIPTS_ROOT = Path.home() / ".cursor" / "projects"
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
    messages: list[Message] = field(default_factory=list)
    mtime: float = 0.0


def _find_transcripts_dirs(project: str | None = None) -> list[Path]:
    """Find agent-transcripts directories under ~/.cursor/projects/."""
    if not TRANSCRIPTS_ROOT.is_dir():
        return []

    candidates: list[Path] = []
    for proj in sorted(TRANSCRIPTS_ROOT.iterdir()):
        candidate = proj / "agent-transcripts"
        if candidate.is_dir():
            candidates.append(candidate)

    if project is None:
        return candidates

    matches = [
        candidate
        for candidate in candidates
        if candidate.parent.name == project or candidate.parent.name.startswith(project)
    ]
    return matches


def _load_transcript(tdir: Path, uuid: str) -> Transcript:
    jsonl_path = tdir / uuid / f"{uuid}.jsonl"
    if not jsonl_path.is_file():
        print(f"ERROR: Transcript not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    messages: list[Message] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            role = obj.get("role", "unknown")
            content_parts = obj.get("message", {}).get("content", [])
            texts = []
            for part in content_parts if isinstance(content_parts, list) else []:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part["text"])
            if texts:
                messages.append(Message(role=role, text="\n".join(texts)))

    return Transcript(
        uuid=uuid,
        path=jsonl_path,
        project=tdir.parent.name,
        messages=messages,
        mtime=jsonl_path.stat().st_mtime,
    )


def _list_all(tdirs: list[Path]) -> list[Transcript]:
    """Load all transcripts, sorted newest-first."""
    results: list[Transcript] = []
    for tdir in tdirs:
        for entry in tdir.iterdir():
            if not entry.is_dir() or entry.name == "subagents":
                continue
            jsonl = entry / f"{entry.name}.jsonl"
            if jsonl.is_file():
                results.append(_load_transcript(tdir, entry.name))
    results.sort(key=lambda t: t.mtime, reverse=True)
    return results


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


# ---------------------------------------------------------------------------
# Output: --list
# ---------------------------------------------------------------------------

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


def _print_transcript_table(transcripts: list[Transcript]) -> None:
    """Print a table of transcripts with recent context lines."""
    print(f"{'#':>3}  {'Date':16}  {'User':>4}  {'Asst':>4}  {'Project':20}  {'UUID':36}  {'Topic'}")
    print("-" * 137)
    for i, t in enumerate(transcripts):
        ts = datetime.fromtimestamp(t.mtime).strftime("%Y-%m-%d %H:%M")
        n_user = sum(1 for m in t.messages if m.role == "user")
        n_asst = sum(1 for m in t.messages if m.role == "assistant")
        topic = _first_user_query(t)
        print(f"{i:3}  {ts}  {n_user:4}  {n_asst:4}  {_truncate(t.project, 20):20}  {t.uuid}  {_truncate(topic, 60)}")
        recent = _recent_user_messages(t, count=3)
        for q in recent:
            print(f"       > {_truncate(q, 140)}")
        _, last_asst = _last_exchange(t)
        if last_asst:
            print(f"       = {_truncate(last_asst, 140)}")
        print()


def cmd_list(all_transcripts: list[Transcript], limit: int) -> None:
    transcripts = all_transcripts[:limit]
    if not transcripts:
        print("No transcripts found.")
        return
    print(f"Recent transcripts ({len(transcripts)} shown):\n")
    _print_transcript_table(transcripts)


# ---------------------------------------------------------------------------
# Output: --search
# ---------------------------------------------------------------------------

def cmd_search(all_transcripts: list[Transcript], keyword: str, limit: int) -> None:
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
    _print_transcript_table(matches)


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
        "project": t.project,
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

    # --id with prefix matching
    matches = [t for t in all_transcripts if t.uuid == args.id or t.uuid.startswith(args.id)]
    if not matches:
        print(f"No transcript matching '{args.id}'", file=sys.stderr)
        sys.exit(1)
    unique_ids = {t.uuid for t in matches}
    if len(unique_ids) > 1:
        print(f"Multiple transcripts match '{args.id}':", file=sys.stderr)
        for t in matches[:10]:
            print(f"  {t.uuid}  [{t.project}]  {_first_user_query(t)}", file=sys.stderr)
        print("Use a longer UUID prefix or pass --project.", file=sys.stderr)
        sys.exit(1)
    return matches[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read, search, and continue Cursor agent transcripts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                      List recent transcripts
  %(prog)s --list --project home-keivenc-dynamo
  %(prog)s --latest                    Reader view of most recent transcript
  %(prog)s --latest --continue         Continuation prompt for most recent
  %(prog)s --latest --continue --tail 10
  %(prog)s --id 806f --full            Full transcript by UUID prefix
  %(prog)s --id 459ac4ca --project home-keivenc-dynamo --continue
  %(prog)s --search "mlperf"           Find transcripts mentioning "mlperf"
  %(prog)s --latest --json             JSON output for scripting
  %(prog)s --latest --raw --tail 5     Raw interleaved messages
""",
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
    parser.add_argument("--tail", type=int, default=None, help="Last N messages (default: all for reader, 20 for --continue/--raw)")
    parser.add_argument("--limit", type=int, default=20, help="Number of transcripts to list/search (default: 20)")
    parser.add_argument(
        "--project",
        metavar="NAME",
        help="Restrict to a specific Cursor project name (prefix match OK)",
    )

    args = parser.parse_args()

    tdirs = _find_transcripts_dirs(project=args.project)
    if not tdirs:
        print("ERROR: Could not find agent-transcripts directory.", file=sys.stderr)
        return 1
    all_transcripts = _list_all(tdirs)
    if not all_transcripts:
        if args.project:
            print(f"ERROR: No transcripts found for project '{args.project}'.", file=sys.stderr)
        else:
            print("ERROR: No transcripts found.", file=sys.stderr)
        return 1

    if args.list:
        cmd_list(all_transcripts, limit=args.limit)
        return 0

    if args.search:
        cmd_search(all_transcripts, args.search, limit=args.limit)
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
