#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Auto-approve Cursor CLI permission prompts in tmux sessions.

Watches one or more tmux panes for permission prompts and sends Enter to
approve them.  Blocks only genuinely dangerous commands (rm, rmdir, etc.).
Everything else is approved automatically.

Detected prompt patterns:
  1. "Do you want to proceed?"       (bash command confirmation)
  2. "Do you want to make this edit"  (file edit confirmation)
  3. "Do you want to create"          (file create confirmation)

Usage:
  ./auto_approve_tmux.py dynamo1                  # single session
  ./auto_approve_tmux.py dynamo1,dynamo2          # comma-separated
  ./auto_approve_tmux.py "dynamo*"                # wildcard (glob)
  ./auto_approve_tmux.py dynamo1:0.1              # specific pane
  ./auto_approve_tmux.py --dry-run dynamo1
  ./auto_approve_tmux.py --list
"""

import argparse
import fnmatch
import hashlib
import logging
import os
import re
import signal
import subprocess
import sys
import time

log = logging.getLogger("auto_approve")

# ---------------------------------------------------------------------------
# Denylist: block only genuinely dangerous commands
# ---------------------------------------------------------------------------

DANGEROUS_COMMANDS = frozenset({
    "rm", "rmdir", "shred", "mkfs", "fdisk", "parted", "wipefs", "dd", "format",
})

DANGEROUS_PATTERNS = [
    re.compile(r">\s*/dev/sd"),
    re.compile(r">\s*/dev/nvme"),
    re.compile(r">\s*/dev/vd"),
    re.compile(r":\(\)\{.*:\|:&\};:"),   # fork bomb
    re.compile(r"chmod\s+-R\s+777\s+/"),
    re.compile(r"chown\s+-R\s+\S+\s+/"),
    re.compile(r"sudo\s+rm\s"),
    re.compile(r"sudo\s+rmdir\s"),
]


def is_dangerous(cmd_line: str) -> bool:
    """Return True if cmd_line contains a dangerous command."""
    cmd_line = cmd_line.strip()
    if not cmd_line:
        return False

    for pat in DANGEROUS_PATTERNS:
        if pat.search(cmd_line):
            return True

    # Split on shell operators to get individual command segments
    segments = re.split(r"[|&;]+", cmd_line)
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        # Skip leading env var assignments (FOO=bar cmd ...)
        words = segment.split()
        while words and re.match(r"^[A-Za-z_]\w*=", words[0]):
            words = words[1:]
        if not words:
            continue

        first = words[0]
        if first == "sudo" and len(words) > 1:
            first = words[1]

        base = os.path.basename(first)

        for dangerous in DANGEROUS_COMMANDS:
            if base == dangerous or base.startswith(dangerous + "."):
                return True

    return False


# ---------------------------------------------------------------------------
# Command extraction from tmux pane text
# ---------------------------------------------------------------------------

# Lines to skip when collecting command text
_SKIP_LINE = re.compile(
    r"^("
    r"─+$"                              # separator bars
    r"|Bash command$"                   # label
    r"|Permission rule\b"              # permission line
    r"|Do you want"                    # prompt line
    r"|Running"                        # status
    r"|Esc to cancel"                  # footer hint
    r")",
    re.IGNORECASE,
)

# Lines that look like commands (contain special shell chars, flags, or are long)
_CMD_CHARS = re.compile(r"[/|&;$=(>`~]|--|\s-[a-zA-Z]")


def extract_command(pane_text: str) -> str | None:
    """Extract the pending command from pane text above a permission prompt.

    Walks backward from the trigger line ("Permission rule" / "Do you want to")
    to the nearest separator bar (────) or bullet (●), then collects lines that
    look like shell commands.
    """
    lines = pane_text.splitlines()

    # Find trigger line index
    trigger_idx = None
    for i, line in enumerate(lines):
        if "Permission rule" in line or "Do you want to proceed" in line or "Do you want to make this edit" in line:
            trigger_idx = i
            break

    if trigger_idx is None:
        return None

    # Walk backwards to find the top boundary
    top_idx = 0
    found_content = False
    for i in range(trigger_idx - 1, -1, -1):
        stripped = lines[i].strip()

        # Separator bar
        if re.match(r"^─+$", stripped):
            top_idx = i + 1
            break

        # Bullet line from previous tool output
        if stripped.startswith("●"):
            top_idx = i + 1
            break

        # Blank line after we've seen content — that's the boundary
        if not stripped and found_content:
            top_idx = i + 1
            break

        if stripped:
            found_content = True

    # Collect command lines from the window
    cmd_parts: list[str] = []
    for i in range(top_idx, trigger_idx):
        stripped = lines[i].strip()
        if not stripped:
            continue
        if _SKIP_LINE.match(stripped):
            continue
        if _CMD_CHARS.search(stripped) or len(stripped) > 60:
            cmd_parts.append(stripped)

    if not cmd_parts:
        return None

    return " ".join(cmd_parts)


# Patterns for finding full file paths in pane context
_FULL_PATH_RE = re.compile(
    r"(?:Write|Edit|Create|Read)\(([^)]+)\)"  # Write(/full/path/to/file)
    r"|(?:^|\s)(/\S+)"                        # or a bare absolute path
    r"|(?:^|\s)((?:\w[\w.-]*/)+\w[\w.-]*)",   # or a relative path with slashes
    re.MULTILINE,
)


def _find_full_path(pane_text: str, short_name: str) -> str:
    """Try to resolve a short filename to its full path from pane context.

    Looks for Write(path), Edit(path), or bare paths in the lines above the
    prompt that end with the same basename.
    """
    if "/" in short_name:
        return short_name

    basename = short_name.rstrip("?").strip()

    for m in _FULL_PATH_RE.finditer(pane_text):
        path = m.group(1) or m.group(2) or m.group(3)
        if path and path.rstrip(")").endswith(basename):
            return path.rstrip(")")

    return short_name


# ---------------------------------------------------------------------------
# Prompt detection
# ---------------------------------------------------------------------------

# Matches any file-related prompt: edit, create, overwrite, replace, etc.
_FILE_PROMPT_RE = re.compile(
    r"Do you want to (?:make this )?(edit|create|overwrite|replace|rename|move)\b[^?]*\?",
    re.IGNORECASE,
)


def detect_prompt(pane_text: str) -> str | None:
    """Detect which kind of permission prompt is visible.

    Returns "bash", "file", "tool", or None.
    """
    if "Do you want to proceed" in pane_text:
        return "bash"
    m = _FILE_PROMPT_RE.search(pane_text)
    if m:
        return "file"
    if "Do you want to allow" in pane_text:
        return "tool"
    return None


def yes_is_selected(pane_text: str) -> bool:
    """Check that the first option (Yes) is currently highlighted."""
    return bool(re.search(r"❯ 1\. Yes", pane_text))


# ---------------------------------------------------------------------------
# tmux helpers
# ---------------------------------------------------------------------------

def tmux_run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["tmux", *args],
        capture_output=True, text=True, check=check,
    )


def tmux_list_sessions() -> str | None:
    result = tmux_run("list-sessions", check=False)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def tmux_session_names() -> list[str]:
    """Return list of all tmux session names."""
    result = tmux_run("list-sessions", "-F", "#{session_name}", check=False)
    if result.returncode != 0:
        return []
    return [s.strip() for s in result.stdout.splitlines() if s.strip()]


def tmux_has_session(session: str) -> bool:
    return tmux_run("has-session", "-t", session, check=False).returncode == 0


def tmux_capture_pane(target: str, lines: int = 80) -> str | None:
    result = tmux_run("capture-pane", "-t", target, "-p", "-S", f"-{lines}", check=False)
    if result.returncode != 0:
        return None
    return result.stdout


def tmux_send_enter(target: str) -> None:
    tmux_run("send-keys", "-t", target, "Enter")


# ---------------------------------------------------------------------------
# Target resolution (comma-separated, wildcards)
# ---------------------------------------------------------------------------

def resolve_targets(spec: str) -> list[str]:
    """Expand a target spec into a list of tmux targets.

    Supports:
      - Single target:   "dynamo1"
      - Comma-separated: "dynamo1,dynamo2"
      - Wildcards:       "dynamo*"  (matched against session names via fnmatch)
    """
    all_sessions = tmux_session_names()
    targets: list[str] = []
    seen: set[str] = set()

    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue

        session_part = part.split(":")[0]

        if any(c in session_part for c in "*?[]"):
            # Wildcard — match against known session names
            for name in all_sessions:
                if fnmatch.fnmatch(name, session_part) and name not in seen:
                    # Preserve any :window.pane suffix from the spec
                    suffix = part[len(session_part):]
                    targets.append(name + suffix)
                    seen.add(name)
        else:
            if session_part not in seen:
                targets.append(part)
                seen.add(session_part)

    return targets


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def prompt_hash(pane_text: str) -> str:
    """Hash the lines around the Yes selector to deduplicate repeated polls."""
    context_lines: list[str] = []
    for line in pane_text.splitlines():
        if "❯ 1. Yes" in line:
            context_lines.append(line)
        elif context_lines:
            context_lines.append(line)
            if len(context_lines) >= 3:
                break
    # Also include a few lines before
    all_lines = pane_text.splitlines()
    for i, line in enumerate(all_lines):
        if "❯ 1. Yes" in line:
            start = max(0, i - 5)
            context_lines = all_lines[start : i + 3]
            break

    blob = "\n".join(context_lines).encode()
    return hashlib.md5(blob).hexdigest()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-approve Cursor CLI permission prompts in tmux sessions.",
        epilog=(
            "Target formats:\n"
            '  dynamo1                single session\n'
            '  dynamo1,dynamo2        comma-separated\n'
            '  "dynamo*"             wildcard (glob against session names)\n'
            '  dynamo1:0.1            specific window.pane\n'
            "\n"
            "To create/attach a named tmux session:\n"
            "  tmux new-session -s mysession      # create new\n"
            "  tmux attach -t mysession           # reattach to existing\n"
            "  tmux new-session -A -s mysession   # attach if exists, else create"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("target", nargs="?", default=None,
                        help='tmux target(s): session, "s1,s2", or "pattern*"')
    parser.add_argument("--dry-run", action="store_true",
                        help="show what would be approved without sending keys")
    parser.add_argument("--verbose", action="store_true",
                        help="print every poll cycle")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="poll interval in seconds (default: 2)")
    parser.add_argument("--list", action="store_true",
                        help="list available tmux sessions and exit")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

class SessionState:
    """Per-session tracking for dedup and counters."""

    MAX_RETRIES = 3  # resend Enter if prompt persists after approval

    def __init__(self, target: str) -> None:
        self.target = target
        self.label = target.split(":")[0]  # short name for log output
        self.last_hash = ""
        self.retry_count = 0
        self.approved = 0
        self.blocked = 0


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    if args.list:
        sessions = tmux_list_sessions()
        print(sessions or "No tmux sessions found.")
        sys.exit(0)

    if args.target is None:
        print("No tmux target specified. Available sessions:")
        print()
        sessions = tmux_list_sessions()
        print(sessions or "  (none)")
        print()
        print('Usage: auto_approve_tmux.py <target>  (e.g. dynamo1, "dynamo*", d1,d2)')
        print()
        print("To create/attach a named tmux session:")
        print("  tmux new-session -s mysession      # create new")
        print("  tmux attach -t mysession           # reattach to existing")
        print("  tmux new-session -A -s mysession   # attach if exists, else create")
        sys.exit(1)

    # Check if the spec uses wildcards — if so, we'll re-resolve each cycle
    is_dynamic = any(c in args.target for c in "*?[]")

    # Initial resolution
    targets = resolve_targets(args.target)
    if not targets:
        if is_dynamic:
            log.info("No sessions match '%s' yet — waiting for them to appear...", args.target)
        else:
            print(f"Error: no tmux sessions match '{args.target}'.")
            print("Available sessions:")
            sessions = tmux_list_sessions()
            print(sessions or "  (none)")
            sys.exit(1)

    # For non-dynamic specs, verify sessions exist upfront
    if not is_dynamic:
        bad: list[str] = []
        for t in targets:
            session_name = t.split(":")[0]
            if not tmux_has_session(session_name):
                bad.append(session_name)
        if bad:
            print(f"Error: tmux session(s) not found: {', '.join(bad)}")
            print("Available sessions:")
            sessions = tmux_list_sessions()
            print(sessions or "  (none)")
            sys.exit(1)

    states: dict[str, SessionState] = {t: SessionState(t) for t in targets}

    def refresh_targets() -> None:
        """Re-resolve dynamic targets, adding new sessions and removing gone ones."""
        current_targets = resolve_targets(args.target)
        current_set = set(current_targets)
        existing_set = set(states.keys())

        for t in current_set - existing_set:
            states[t] = SessionState(t)
            log.info("New session detected: %s", states[t].label)

        for t in existing_set - current_set:
            st = states.pop(t)
            log.info("Session gone: %s (was %d approved, %d blocked)", st.label, st.approved, st.blocked)

    # Dedup for log output (across all sessions)
    last_log_msg = ""
    repeat_count = 0

    def log_dedup(level: int, msg: str) -> None:
        """Log a message, collapsing consecutive identical lines into '......'."""
        nonlocal last_log_msg, repeat_count
        if msg == last_log_msg:
            repeat_count += 1
            if repeat_count == 1:
                log.log(level, "......")
            return
        if repeat_count > 1:
            log.log(level, "...... (%d repeats)", repeat_count)
        last_log_msg = msg
        repeat_count = 0
        log.log(level, "%s", msg)

    def on_exit(signum: int, _frame: object) -> None:
        sig_name = signal.Signals(signum).name
        parts = [f"{s.label}: {s.approved} approved, {s.blocked} blocked" for s in states.values()]
        log.info("Caught %s — exiting. %s", sig_name, " | ".join(parts))
        sys.exit(0)

    signal.signal(signal.SIGINT, on_exit)
    signal.signal(signal.SIGTERM, on_exit)

    # Adaptive polling: start at base interval, ramp up when idle, reset on activity
    base_interval = args.interval
    max_interval = max(5.0, base_interval)
    current_interval = base_interval

    sys.stderr.write("=" * 72 + "\n")
    sys.stderr.write(" AUTO-APPROVE TMUX\n")
    sys.stderr.write("\n")
    sys.stderr.write(" WARNING: This script automatically presses Enter on Cursor CLI\n")
    sys.stderr.write(" permission prompts. It will approve ALL commands except:\n")
    sys.stderr.write("   - rm, rmdir, shred, mkfs, fdisk, parted, wipefs, dd, format\n")
    sys.stderr.write("   - sudo rm/rmdir, writes to block devices, fork bombs\n")
    sys.stderr.write("   - File deletion prompts\n")
    sys.stderr.write("\n")
    sys.stderr.write(" USE AT YOUR OWN RISK. Review the denylist before relying on this.\n")
    sys.stderr.write(" The author is not responsible for unintended side effects.\n")
    sys.stderr.write("=" * 72 + "\n")
    sys.stderr.write("\n")
    sys.stderr.flush()

    if states:
        target_names = ", ".join(s.label for s in states.values())
        log.info("Watching %d session(s): %s", len(states), target_names)
    if is_dynamic:
        log.info("Dynamic mode: will auto-detect new sessions matching '%s'", args.target)
    log.info("Poll interval: %ss (ramps to %ss when idle)", base_interval, max_interval)
    if args.dry_run:
        log.info("DRY RUN — will not send keys")
    log.info("Press Ctrl+C to stop")
    print()

    while True:
        if is_dynamic:
            refresh_targets()

        acted = False

        for st in list(states.values()):
            pane_text = tmux_capture_pane(st.target)
            if pane_text is None:
                log.warning("[%s] Failed to capture pane. Session still alive?", st.label)
                continue

            prompt_type = detect_prompt(pane_text)

            if prompt_type is None:
                st.last_hash = ""  # reset so next prompt is always fresh
                log_dedup(logging.DEBUG, f"[{st.label}] No prompt (approved={st.approved} blocked={st.blocked})")
                continue

            if not yes_is_selected(pane_text):
                log_dedup(logging.DEBUG, f"[{st.label}] Prompt found but 'Yes' not selected")
                continue

            current_hash = prompt_hash(pane_text)
            if current_hash == st.last_hash:
                st.retry_count += 1
                if st.retry_count <= SessionState.MAX_RETRIES:
                    log.info("[%s] Prompt still visible after Enter — retry %d/%d",
                             st.label, st.retry_count, SessionState.MAX_RETRIES)
                    if not args.dry_run:
                        tmux_send_enter(st.target)
                    time.sleep(1)
                else:
                    log_dedup(logging.DEBUG, f"[{st.label}] Same prompt persists after {SessionState.MAX_RETRIES} retries")
                continue

            acted = True

            if prompt_type == "file":
                # Generic: "Do you want to [make this] <action> [to] <filename>?"
                match = re.search(
                    r"Do you want to (?:make this )?(\w+)\s+(?:to\s+)?([^?\n]+)\?",
                    pane_text, re.IGNORECASE,
                )
                action = match.group(1).strip() if match else "file"
                short_name = match.group(2).strip() if match else "(file)"
                desc = _find_full_path(pane_text, short_name)
                if args.dry_run:
                    log.info("[%s] WOULD APPROVE (%s): %s", st.label, action, desc)
                else:
                    log.info("[%s] APPROVE (%s): %s", st.label, action, desc)
                    tmux_send_enter(st.target)
                st.last_hash = current_hash
                st.retry_count = 0
                st.approved += 1
                time.sleep(3)

            elif prompt_type == "tool":
                # "Permission rule <Tool> requires confirmation" / "Do you want to allow Claude to <action>?"
                match = re.search(r"Permission rule (\w+) requires confirmation", pane_text)
                tool_name = match.group(1) if match else "tool"
                if args.dry_run:
                    log.info("[%s] WOULD APPROVE (tool): %s", st.label, tool_name)
                else:
                    log.info("[%s] APPROVE (tool): %s", st.label, tool_name)
                    tmux_send_enter(st.target)
                st.last_hash = current_hash
                st.retry_count = 0
                st.approved += 1
                time.sleep(3)

            else:  # bash prompt
                cmd = extract_command(pane_text)

                if cmd is None:
                    if last_log_msg != f"[{st.label}] APPROVE (no cmd extracted, defaulting yes)":
                        pane_lines = pane_text.splitlines()
                        for i, line in enumerate(pane_lines):
                            if "Do you want to proceed" in line:
                                start = max(0, i - 10)
                                context = pane_lines[start : i + 1]
                                log.warning("[%s] Could not extract command. Context:", st.label)
                                for ctx in context:
                                    print(f"  | {ctx}")
                                break
                    if args.dry_run:
                        log_dedup(logging.INFO, f"[{st.label}] WOULD APPROVE (no cmd extracted)")
                    else:
                        log_dedup(logging.INFO, f"[{st.label}] APPROVE (no cmd extracted, defaulting yes)")
                        tmux_send_enter(st.target)
                    st.last_hash = current_hash
                    st.retry_count = 0
                    st.approved += 1
                    time.sleep(3)

                elif is_dangerous(cmd):
                    log.info("[%s] BLOCKED (dangerous): %s", st.label, cmd)
                    st.last_hash = current_hash
                    st.retry_count = 0
                    st.blocked += 1

                else:
                    if args.dry_run:
                        log.info("[%s] WOULD APPROVE: %s", st.label, cmd)
                    else:
                        log.info("[%s] APPROVE: %s", st.label, cmd)
                        tmux_send_enter(st.target)
                    st.last_hash = current_hash
                    st.retry_count = 0
                    st.approved += 1
                    time.sleep(3)

        if acted:
            current_interval = base_interval
        else:
            current_interval = min(current_interval + base_interval, max_interval)

        time.sleep(current_interval)


if __name__ == "__main__":
    main()
