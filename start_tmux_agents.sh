#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ==============================================================================
# start_tmux_agents.sh - Bring up the standard agent tmux sessions (+ dev servers)
# ==============================================================================
#
# Creates (only if missing) these tmux sessions, cd's each into its working
# directory, and launches the right agent CLI. For the YOLOmux dev ports it
# also (nohup-)launches the YOLOmux server from that worktree if one is not
# already bound on the port.
#
#   Session  Directory                  Agent    Server
#   -------  -------------------------  -------  ----------------------------
#   7777     ~/yolomux                  codex    -  (prod; managed elsewhere)
#   8001     ~/yolomux.dev8001          codex    yolomux.py --port 8001
#   8002     ~/yolomux.dev8002          codex    yolomux.py --port 8002
#   8003     ~/yolomux.dev8003          codex    yolomux.py --port 8003
#   1        ~/dynamo/frontend-crates1  claude   -  (+ dyn-continue-transcript)
#   2        ~/dynamo/frontend-crates2  claude   -  (+ dyn-continue-transcript)
#   3        ~/dynamo/frontend-crates3  claude   -  (+ dyn-continue-transcript)
#   4        ~/dynamo/frontend-crates4  claude   -  (+ dyn-continue-transcript)
#
# Idempotent: a session that already exists is left untouched (we do NOT re-cd
# it or relaunch its agent, since it may already be doing work). A dev server
# is launched only if nothing is already bound on its port. Re-running is safe.
#
# Server launch mirrors the proven shape from the yolo-dev-start skill:
# `nohup setsid -f` (plain nohup children get reaped by the command harness),
# augmented PATH so the server's shutil.which() resolves claude/codex, and the
# logic lives in this script FILE so `pgrep -f "...--port 800N"` cannot
# self-match this process's argv (the exit-144 footgun).
# ==============================================================================

set -uo pipefail

CLAUDE_CMD="claude --dangerously-skip-permissions"
CODEX_CMD="codex --dangerously-bypass-approvals-and-sandbox --dangerously-bypass-hook-trust"
BASH_SHELL="/bin/bash"

# Augmented PATH (matches yolo-dev-start) so the YOLOmux server's
# shutil.which("claude"/"codex") resolves and new panes find their agent.
export PATH="$HOME/.local/bin:$HOME/.local/node-v22.11.0-linux-x64/bin:$PATH"
export SHELL="$BASH_SHELL"

# Session table: "<session>|<dir>|<agent>|<server_port>"  (server_port "-" = none)
SESSIONS=(
  "7777|$HOME/yolomux|codex|-"
  "8001|$HOME/yolomux.dev8001|codex|8001"
  "8002|$HOME/yolomux.dev8002|codex|8002"
  "8003|$HOME/yolomux.dev8003|codex|8003"
  "1|$HOME/dynamo/frontend-crates1|claude|-"
  "2|$HOME/dynamo/frontend-crates2|claude|-"
  "3|$HOME/dynamo/frontend-crates3|claude|-"
  "4|$HOME/dynamo/frontend-crates4|claude|-"
)

# Launch the YOLOmux server for $port from worktree $dir, unless one is already
# bound on the port. Idempotent and non-destructive: an already-running server
# is left alone (use the yolo-dev-start skill for a forced restart).
launch_server() {
  local port="$1" dir="$2"
  if pgrep -f "yolomux.py --host 0.0.0.0 --port $port" >/dev/null 2>&1; then
    echo "  srv $port: already running, leaving it alone"
    return 0
  fi
  ( cd "$dir" && nohup setsid -f env SHELL="$BASH_SHELL" \
      python3 -u yolomux.py --host 0.0.0.0 --port "$port" --dangerously-yolo --self-signed \
      >> "/tmp/yolomux-$port.log" 2>&1 < /dev/null & )
  echo "  srv $port: launched from $dir (log=/tmp/yolomux-$port.log)"
}

# Wait until the claude TUI is ready (its footer prints "? for shortcuts"),
# so the follow-up prompt isn't swallowed during startup. Bounded poll.
wait_for_claude_ready() {
  local sess="$1" deadline=$((SECONDS + 45))
  while (( SECONDS < deadline )); do
    # Footer differs by mode: "? for shortcuts" normally, "bypass permissions"
    # under --dangerously-skip-permissions. Match either, plus the welcome box.
    if tmux capture-pane -p -t "=$sess:" 2>/dev/null \
        | grep -qE 'shortcuts|bypass permissions|Welcome back'; then
      return 0
    fi
    sleep 1
  done
  echo "  WARN: claude in '$sess' did not show a ready prompt within 45s; sending anyway" >&2
  return 1
}

for entry in "${SESSIONS[@]}"; do
  IFS='|' read -r sess dir agent server_port <<< "$entry"

  if [[ ! -d "$dir" ]]; then
    echo "  WARN: directory '$dir' for session '$sess' does not exist, skipping" >&2
    continue
  fi

  # --- tmux session + agent --------------------------------------------------
  # Use exact-match targets ("=name") everywhere: a bare numeric target like
  # "-t 1" collides with window index 1 of the attached session, so it would
  # type into the wrong pane. "=1:" forces session-name resolution.
  if tmux has-session -t "=$sess" 2>/dev/null; then
    echo "= session '$sess' already exists, leaving untouched"
  else
    echo "+ creating session '$sess' in $dir (agent: $agent)"
    tmux new-session -d -s "$sess" -c "$dir"

    if [[ "$agent" == "claude" ]]; then
      tmux send-keys -t "=$sess:" "$CLAUDE_CMD" Enter
      wait_for_claude_ready "$sess"
      # Clear any autosuggestion text in the input box before sending the prompt.
      tmux send-keys -t "=$sess:" C-u
      dir_name="$(basename "$dir")"
      tmux send-keys -t "=$sess:" \
        "/dyn-continue-transcript resume the most recent prior transcript for this tmux session ($dir_name in $dir)"
      tmux send-keys -t "=$sess:" Enter
    else
      tmux send-keys -t "=$sess:" "$CODEX_CMD" Enter
    fi
  fi

  # --- YOLOmux dev server (runs regardless of whether the session existed) ---
  if [[ "$server_port" != "-" ]]; then
    launch_server "$server_port" "$dir"
  fi
done

echo
echo "Current tmux sessions:"
tmux ls 2>/dev/null
