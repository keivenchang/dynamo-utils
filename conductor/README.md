# Conductor tmux tools

Browser tools for watching and attaching to Dynamo tmux sessions.

## Interactive web terminals

`tmux_webterm.py` is the first interactive slice:

- Two side-by-side tmux terminals by default: `dynamo1` and `dynamo2`.
- Expand/collapse controls for each terminal.
- PTY-backed WebSocket bridge to `tmux attach-session`.
- Transcript discovery for the Claude/Codex process attached to each tmux pane.
- Local-only binding by default.

Run:

```bash
python3 ~/dynamo/dynamo-utils.dev/conductor/tmux_webterm.py
```

Then open:

```text
http://localhost:9998/
```

To expose it beyond localhost:

```bash
python3 ~/dynamo/dynamo-utils.dev/conductor/tmux_webterm.py --host 0.0.0.0 --port 9998
```

Auth defaults to `dynamo` / `conductor`. Override it with `CONDUCTOR_AUTH_USER` and `CONDUCTOR_AUTH_PASSWORD`, or put `{"user": "...", "password": "..."}` in `~/.config/conductor/auth.json`.

Inspect the transcript mapping without starting the server:

```bash
python3 ~/dynamo/dynamo-utils.dev/conductor/tmux_webterm.py --print-transcripts
```

AI-facing endpoints:

- `GET /api/transcripts` returns pane, process, and transcript-path metadata.
- `GET /api/transcript?session=dynamo1&lines=120` returns the transcript tail for one session.
- `GET /api/context?session=dynamo1&messages=40` returns a compact, message-oriented transcript tail.

## Read-only wall

`tmux_wall.py` is a read-only dashboard:

- Stdlib HTTP server.
- Server-Sent Events for live terminal snapshots.
- `tmux capture-pane` as the terminal source.
- Existing `container/show_dynamo_containers.py` as optional container metadata.
- JSON endpoints that can feed a future AI summarizer without scraping the browser.

```bash
python3 ~/dynamo/dynamo-utils.dev/conductor/tmux_wall.py --host 0.0.0.0 --port 8765
```

Then open:

```text
http://localhost:8765/
```

Without `--targets`, the server discovers panes from `dynamo1` through `dynamo4`, picks one agent pane per session first, then fills the remaining six slots with other panes from those sessions.

Current target selection can be inspected without starting the server:

```bash
python3 ~/dynamo/dynamo-utils.dev/conductor/tmux_wall.py --print-targets
```

To override:

```bash
python3 ~/dynamo/dynamo-utils.dev/conductor/tmux_wall.py --targets dynamo1:0.0,dynamo2:0.0,dynamo3:1.0,dynamo4:0.0 --slots 6
```

## AI-facing endpoints

- `GET /api/snapshot` returns the current six-pane dashboard payload.
- `GET /api/transcript?target=dynamo1:0.0&lines=2000` returns one tmux pane transcript.
- `GET /api/summary-input?lines=1200` returns the active dashboard panes and container metadata as one JSON payload.

These endpoints do not call an LLM. They are the stable input surface for a later summarizer.
