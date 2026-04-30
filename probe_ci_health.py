#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
probe_ci_health.py — probe CI health by pushing per-SHA placebo PRs.

For every commit merged to main since --starting-sha, push a tiny placebo
diff on a per-SHA branch and open a Draft PR. Watch CI; on failure, retry
up to N times via gh run rerun --failed. On all-pass, close PR + delete
branch. Persist verdicts and per-attempt failures to a JSON cache.

Subcommands:
  run        one cycle (cron-friendly, idempotent)
  status     print DB state per SHA
  report     verdict + flake leaderboard
  reset SHA  drop a SHA's entry to re-probe from scratch

Run flags:
  --starting-sha SHA      first SHA to probe (required first run; persisted)
  --parallelism N         max in-flight probes (default 4)
  --max-attempts N        retries per SHA (default 3)
  --stalled-after-hours H stall threshold (default 3)
  --dry-run / --dryrun    no writes; reads OK, prints what would happen

Cache: ~/.cache/dynamo-utils/ci-health.json
Clone: /tmp/ci_health/repo
Lock:  /tmp/ci_health/launch.pid
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")  # PST/PDT, handles DST automatically

REPO = "ai-dynamo/dynamo"
REPO_URL = "git@github.com:ai-dynamo/dynamo.git"
DB_PATH = Path.home() / ".cache" / "dynamo-utils" / "ci-health.json"
RAW_LOG_DIR = Path(
    os.environ.get("DYNAMO_UTILS_CACHE_DIR")
    or str(Path.home() / ".cache" / "dynamo-utils")
) / "raw-log-text"
WORK_DIR = Path("/tmp/ci_health")
CLONE_PATH = WORK_DIR / "repo"
LOCK_PATH = WORK_DIR / "launch.pid"

# Match either:
#   FAILED tests/foo/test_bar.py::TestClass::test_method[params] - reason   (assertion fail)
#   ERROR tests/foo/test_bar.py::test_method - RuntimeError: ...            (fixture/setup error)
# with optional GH Actions timestamp prefix and ANSI color codes.
_PYTEST_FAILED_RE = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2}T[\d:.]+Z\s+)?(?:\x1b\[[0-9;]*m)?(?:FAILED|ERROR)\s+(\S+::\S+)"
)

PROBE_BRANCH_PREFIX = "keivenchang/ci_health/"
PROBE_FILES = [
    "components/src/dynamo/common/__init__.py",
    "lib/runtime/src/runtime.rs",
]

DEFAULT_PARALLELISM = 4
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_STALLED_AFTER_HOURS = 3

logger = logging.getLogger("probe_ci_health")


# ---------- helpers ----------


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def short_sha(sha: str) -> str:
    return sha[:11]


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    logger.debug("$ %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
        env=env,
    )


def gh_api(
    path: str,
    *,
    method: str = "GET",
    fields: list[tuple[str, str]] | None = None,
) -> Any:
    cmd = ["gh", "api", path]
    if method != "GET":
        cmd += ["--method", method]
    if fields:
        for k, v in fields:
            cmd += ["-f", f"{k}={v}"]
    proc = run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"gh api {path} failed: {proc.stderr.strip()}")
    if not proc.stdout.strip():
        return None
    return json.loads(proc.stdout)


# ---------- DB I/O ----------


def load_db() -> dict[str, Any]:
    if not DB_PATH.exists():
        return {"_meta": {}}
    with DB_PATH.open("r") as f:
        return json.load(f)


def save_db_atomic(db: dict[str, Any]) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = DB_PATH.with_suffix(f".tmp.{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(db, f, indent=2, sort_keys=True)
    tmp.replace(DB_PATH)


# ---------- single-instance lock ----------


@contextlib.contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise RuntimeError(f"another instance holds {path}; exiting")
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        yield
    finally:
        with contextlib.suppress(Exception):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# ---------- git ----------


def ensure_clone() -> None:
    """Clone is a read-only network op (no remote writes). Always allowed."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if (CLONE_PATH / ".git").is_dir():
        return
    logger.info("clone %s → %s", REPO_URL, CLONE_PATH)
    run(["git", "clone", "--quiet", REPO_URL, str(CLONE_PATH)])
    hooks = CLONE_PATH / ".git" / "hooks"
    for h in hooks.iterdir():
        if not h.name.endswith(".sample"):
            with contextlib.suppress(FileNotFoundError):
                h.unlink()


def fetch_main() -> None:
    run(["git", "fetch", "--quiet", "origin", "main"], cwd=CLONE_PATH)


def resolve_sha(sha_or_prefix: str) -> str:
    proc = run(["git", "rev-parse", sha_or_prefix], cwd=CLONE_PATH)
    return proc.stdout.strip()


def commit_date(sha: str) -> str | None:
    """Return ISO-8601 committer date for `sha`, or None if not resolvable."""
    proc = run(
        ["git", "show", "-s", "--format=%cI", sha],
        cwd=CLONE_PATH,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def commit_subject_pr_author(sha: str) -> tuple[str, int | None, str]:
    """Return (subject_without_pr_suffix, pr_number, author) for `sha`."""
    proc = run(
        ["git", "show", "-s", "--format=%s%x1f%an", sha],
        cwd=CLONE_PATH,
        check=False,
    )
    if proc.returncode != 0:
        return ("", None, "")
    raw = proc.stdout.strip()
    if "\x1f" not in raw:
        return (raw, None, "")
    subject_full, author = raw.split("\x1f", 1)
    pr: int | None = None
    subject = subject_full
    m = re.search(r"\s*\(#(\d+)\)\s*$", subject_full)
    if m:
        pr = int(m.group(1))
        subject = subject_full[: m.start()].rstrip()
    return (subject, pr, author)


def image_sha256(commit_sha: str) -> str:
    """Compute the ImageSHA256 for a commit (SHA-256 of `git ls-tree -r container/`).

    Returns first 6 chars UPPERCASE, matching the convention in
    show_commit_history.py / common.py::generate_docker_image_sha_for_commit.

    Returns 'NO_CTR' if container/ doesn't exist at that commit, 'ERROR' on
    git failure.
    """
    import hashlib

    proc = run(
        ["git", "ls-tree", "-r", commit_sha, "--", "container/"],
        cwd=CLONE_PATH,
        check=False,
    )
    if proc.returncode != 0:
        return "ERROR"
    if not proc.stdout.strip():
        return "NO_CTR"
    return hashlib.sha256(proc.stdout.encode()).hexdigest()[:6].upper()


def list_shas_since(starting_sha: str) -> list[str]:
    """Return main commits from starting_sha (inclusive) → HEAD, oldest first."""
    # `<sha>^..main` includes <sha> itself (excludes its parent).
    proc = run(
        ["git", "rev-list", "--reverse", f"{starting_sha}^..origin/main"],
        cwd=CLONE_PATH,
        check=False,
    )
    if proc.returncode == 0:
        return [s.strip() for s in proc.stdout.splitlines() if s.strip()]
    # First commit (no parent): fall back to including starting + descendants.
    proc = run(
        ["git", "rev-list", "--reverse", f"{starting_sha}..origin/main"],
        cwd=CLONE_PATH,
    )
    shas = [s.strip() for s in proc.stdout.splitlines() if s.strip()]
    return [starting_sha] + shas


def remote_branch_exists(branch: str) -> bool:
    proc = run(
        ["git", "ls-remote", "--exit-code", "origin", f"refs/heads/{branch}"],
        cwd=CLONE_PATH,
        check=False,
    )
    return proc.returncode == 0


def push_probe(sha: str, dry_run: bool) -> tuple[str, str | None]:
    """Push a probe branch for `sha`. Returns (branch_name, probe_head_sha)."""
    branch = PROBE_BRANCH_PREFIX + short_sha(sha)
    if dry_run:
        logger.info("[dry-run] would push probe branch %s for %s", branch, short_sha(sha))
        return branch, None

    logger.info("push probe %s → %s", short_sha(sha), branch)
    run(["git", "fetch", "--quiet", "origin", sha], cwd=CLONE_PATH)
    run(["git", "checkout", "--detach", sha], cwd=CLONE_PATH)
    run(["git", "reset", "--hard", sha], cwd=CLONE_PATH)

    ts = now_iso()
    for rel in PROBE_FILES:
        f = CLONE_PATH / rel
        if not f.exists():
            logger.warning("probe file missing: %s", rel)
            continue
        prefix = "# " if rel.endswith(".py") else "// "
        with f.open("a") as fh:
            fh.write(f"\n{prefix}ci-health probe {short_sha(sha)} {ts}\n")
    run(["git", "add", "--", *PROBE_FILES], cwd=CLONE_PATH)
    env = {**os.environ, "GIT_SSH_COMMAND": "ssh -o BatchMode=yes"}
    run(
        [
            "git",
            "commit",
            "--signoff",
            "--no-verify",
            "-m",
            f"ci-health: probe {short_sha(sha)}",
        ],
        cwd=CLONE_PATH,
        env=env,
    )
    run(
        ["git", "push", "--force", "origin", f"HEAD:refs/heads/{branch}"],
        cwd=CLONE_PATH,
        env=env,
    )
    probe_head = run(["git", "rev-parse", "HEAD"], cwd=CLONE_PATH).stdout.strip()
    return branch, probe_head


# ---------- gh ----------


def open_pr(sha: str, branch: str, dry_run: bool) -> int:
    if dry_run:
        logger.info("[dry-run] would open Draft PR for %s on %s", short_sha(sha), branch)
        return -1

    # Check if PR already exists for this branch (resilience after a crash)
    existing = gh_api(f"repos/{REPO}/pulls?head=keivenchang:{branch}&state=open")
    if existing:
        pr = int(existing[0]["number"])
        logger.info("existing PR #%s for %s; reusing", pr, branch)
        return pr

    subject, orig_pr, author = commit_subject_pr_author(sha)
    intro = (
        "Re-validating that a previous PR did not regress, by re-running "
        "against a trivial placebo diff:"
    )
    if orig_pr and author and subject:
        # Wrap the PR ref + subject in a fenced code block so GitHub doesn't
        # auto-link `#NNNN` (which spams the original PR with cross-refs) and
        # doesn't treat `(gh-NNNN)`-style refs as links.
        ref_block = (
            "```\n"
            f"PR #{orig_pr} by {author}: {subject}\n"
            "```"
        )
    else:
        ref_block = (
            "```\n"
            f"commit {short_sha(sha)}\n"
            "```"
        )
    body = (
        f"{intro}\n\n"
        f"{ref_block}\n\n"
        "Branch deleted on green; kept on failure for diagnosis.\n\n"
        "Drafts are skipped by CodeRabbit per repo `.coderabbit.yaml`."
    )
    proc = run(
        [
            "gh",
            "pr",
            "create",
            "--repo",
            REPO,
            "--draft",
            "--title",
            f"chore(ci-health): re-validate {short_sha(sha)}",
            "--body",
            body,
            "--head",
            branch,
            "--base",
            "main",
        ],
    )
    url = proc.stdout.strip().splitlines()[-1]
    return int(url.rsplit("/", 1)[-1])


def remove_auto_reviewers(pr: int, dry_run: bool) -> None:
    """Strip any reviewers GitHub auto-added (CODEOWNERS, labelers, team rules).
    Probe PRs are noise; we don't want them pinging humans."""
    if dry_run or pr <= 0:
        return
    try:
        data = gh_api(f"repos/{REPO}/pulls/{pr}/requested_reviewers") or {}
        users = [u["login"] for u in data.get("users", [])]
        teams = [t["slug"] for t in data.get("teams", [])]
        if not users and not teams:
            return
        cmd = [
            "gh", "api", f"repos/{REPO}/pulls/{pr}/requested_reviewers",
            "--method", "DELETE",
        ]
        for u in users:
            cmd += ["-f", f"reviewers[]={u}"]
        for t in teams:
            cmd += ["-f", f"team_reviewers[]={t}"]
        run(cmd, check=False)
        logger.info(
            "removed %d auto-reviewer(s) + %d team(s) from PR #%s: users=%s teams=%s",
            len(users), len(teams), pr, users, teams,
        )
    except Exception as e:
        logger.warning("remove_auto_reviewers PR #%s: %s", pr, e)


def workflow_runs_for_sha(sha: str) -> list[dict]:
    data = gh_api(f"repos/{REPO}/actions/runs?head_sha={sha}&per_page=50")
    return (data or {}).get("workflow_runs", [])


def get_run_jobs(run_id: int) -> list[dict]:
    data = gh_api(f"repos/{REPO}/actions/runs/{run_id}/jobs?per_page=100&filter=latest")
    return (data or {}).get("jobs", [])


def rerun_failed_jobs(run_id: int, dry_run: bool) -> None:
    if dry_run:
        logger.info("[dry-run] would rerun-failed-jobs run_id=%s", run_id)
        return
    gh_api(f"repos/{REPO}/actions/runs/{run_id}/rerun-failed-jobs", method="POST")


def _build_status_comment(entry: dict) -> str:
    """Build a Markdown summary of probe attempts for posting on PR close."""
    verdict = entry.get("verdict", "?")
    icon = {"good": "✓", "bad": "✗"}.get(verdict, "")
    attempts = entry.get("attempts", [])
    n_att = len(attempts)
    lines = [f"**ci-health probe — verdict: {verdict}** {icon}", ""]
    if verdict == "good":
        if n_att == 1:
            lines.append(f"Passed on first attempt ({n_att}/3).")
        else:
            lines.append(
                f"Passed on attempt {n_att}/3 — earlier attempt(s) flaked, "
                f"`gh run rerun --failed` cleared them."
            )
    elif verdict == "bad":
        lines.append(f"Failed all {n_att}/3 attempts.")
    lines.append("")
    lines.append("| Attempt | Passed | Failed | Outcome |")
    lines.append("|---|---|---|---|")
    for a in attempts:
        jobs = a.get("jobs", {})
        n_pass = sum(1 for v in jobs.values() if _job_conclusion(v) == "success")
        failed = a.get("failed_jobs", [])
        n_fail = len(failed)
        if n_fail == 0:
            outcome = "green"
        else:
            shown = failed[:3]
            outcome = ", ".join(f"`{j}`" for j in shown)
            if n_fail > 3:
                outcome += f" (+{n_fail - 3} more)"
        lines.append(f"| {a['attempt']} | {n_pass} | {n_fail} | {outcome} |")
    lines.append("")
    if verdict == "good":
        branch = entry.get("branch", "")
        if branch:
            lines.append(f"Branch `{branch}` deleted.")
    elif verdict == "bad":
        stuck = entry.get("stuck_jobs", [])
        if stuck:
            lines.append("Stuck jobs (failed every attempt):")
            for j in stuck:
                lines.append(f"- `{j}`")
        lines.append("")
        lines.append("PR + branch left open for diagnosis.")
    return "\n".join(lines)


def post_status_comment(pr: int, entry: dict, dry_run: bool) -> None:
    body = _build_status_comment(entry)
    if dry_run:
        logger.info("[dry-run] would post status comment to PR #%s:\n%s", pr, body)
        return
    run(["gh", "pr", "comment", "--repo", REPO, str(pr), "--body", body])


def close_pr_and_delete_branch(pr: int, entry: dict, dry_run: bool) -> None:
    if dry_run:
        logger.info("[dry-run] would post status comment + close PR #%s + delete branch", pr)
        return
    # Post the summary first so it survives even if the close fails.
    try:
        post_status_comment(pr, entry, dry_run=False)
    except subprocess.CalledProcessError as e:
        logger.warning("status comment on PR #%s failed: %s", pr, e.stderr or e)
    run(["gh", "pr", "close", "--repo", REPO, str(pr), "--delete-branch"])


# ---------- log fetching + pytest failure extraction ----------


def fetch_job_log(job_id: int) -> Path | None:
    """Fetch a job's log text into ~/.cache/dynamo-utils/raw-log-text/<job_id>.log.

    Returns the cached path on success, None on failure. Cache-hits skip the fetch.
    Reuses the existing dynamo-utils raw-log-text convention so logs cached by
    other tools (commit dashboards, ci_log_errors) are reused.
    """
    RAW_LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_LOG_DIR / f"{job_id}.log"
    if out.exists() and out.stat().st_size > 0:
        return out
    # `gh api .../logs` follows the Azure-blob redirect and emits plain text.
    proc = subprocess.run(
        ["gh", "api", f"repos/{REPO}/actions/jobs/{job_id}/logs"],
        capture_output=True,
        text=False,
        check=False,
    )
    if proc.returncode != 0:
        logger.warning(
            "fetch log job_id=%s failed: %s",
            job_id,
            proc.stderr.decode("utf-8", errors="replace")[:200],
        )
        return None
    if not proc.stdout:
        return None
    out.write_bytes(proc.stdout)
    return out


def extract_pytest_failures(log_path: Path) -> list[str]:
    """Return sorted unique pytest test IDs from `FAILED tests/...::test` lines."""
    found: set[str] = set()
    try:
        with log_path.open("r", errors="replace") as fh:
            for line in fh:
                m = _PYTEST_FAILED_RE.match(line)
                if m:
                    found.add(m.group(1))
    except OSError as e:
        logger.warning("read %s: %s", log_path, e)
    return sorted(found)


# Lazy-imported once per process. Reuses the production ci_log_errors engine.
_CI_LOG_ERRORS_LOADED = False


def _ensure_ci_log_errors():
    global _CI_LOG_ERRORS_LOADED
    if _CI_LOG_ERRORS_LOADED:
        return True
    utils_root = str(Path.home() / "utils")
    if utils_root not in sys.path:
        sys.path.insert(0, utils_root)
    try:
        global _categorize_error_log_lines, _extract_error_snippet_from_log_file, _html_highlight_error_keywords
        from ci_log_errors.engine import categorize_error_log_lines as _categorize_error_log_lines  # type: ignore
        from ci_log_errors.snippet import extract_error_snippet_from_log_file as _extract_error_snippet_from_log_file  # type: ignore
        from ci_log_errors.render import html_highlight_error_keywords as _html_highlight_error_keywords  # type: ignore
        _CI_LOG_ERRORS_LOADED = True
        return True
    except Exception as e:
        logger.warning("ci_log_errors unavailable: %s", e)
        return False


# Strip GH Actions ISO timestamp prefix like "2026-04-29T17:15:55.4973999Z "
_TS_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s+")
# Strip ANSI escape sequences (color codes)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# Detect any line that contains an echo'd string — `echo "..."` or `echo '...'` —
# whether at line start (xtrace `+ echo "X"`) or inside a longer shell command
# (Docker `RUN ... echo "ERROR: ..."`). We skip error-keyword highlighting on
# these lines because the keyword is *literal text being echoed*, not a real error.
_ECHO_LINE_RE = re.compile(r"""\becho\s+["']""")


def render_snippet_html(snippet_text: str) -> str:
    """Render a snippet with red error-keyword highlighting only.

    Deliberately does NOT use the production renderer's command-block / blue-line
    detection — those highlight things like `git version`, `PYTEST_CMD=...`,
    `docker run ...` which add noise without signal.

    Strips: ANSI escapes, GH Actions timestamps, [[CMD]]...[[/CMD]] cut-paste
    blocks, and lines that are just shell command-noise prelude (Run, group
    markers, env: blocks).
    """
    if not snippet_text:
        return ""
    # 1. Strip the cut-pasteable command boxes
    cleaned = re.sub(r"\[\[CMD\]\].*?\[\[/CMD\]\]\n?", "", snippet_text, flags=re.DOTALL)
    out_lines: list[str] = []
    for raw in cleaned.splitlines():
        # 2. Strip GH Actions timestamp prefix and ANSI codes
        line = _ANSI_RE.sub("", _TS_PREFIX_RE.sub("", raw))
        # 3. Drop pure noise lines (best-effort — keep error context)
        s = line.strip()
        if not s:
            out_lines.append("")
            continue
        if s.startswith("##[group]") or s == "##[endgroup]":
            continue
        if s.startswith("Categories:") or s == "...":
            continue
        # 4. Plain-escape echo lines — keywords inside literal echo args are noise
        # Matches: 'echo "X"',  '+ echo "X"' (xtrace),  '  echo X', etc.
        if _ECHO_LINE_RE.match(line):
            out_lines.append(_html_escape(line))
            continue
        # 5. Highlight error keywords (red), default-render rest as plain text
        if _ensure_ci_log_errors():
            try:
                out_lines.append(_html_highlight_error_keywords(line))
                continue
            except Exception:
                pass
        out_lines.append(_html_escape(line))
    # Collapse runs of empty lines
    collapsed: list[str] = []
    blank = False
    for ln in out_lines:
        if ln == "":
            if not blank:
                collapsed.append("")
            blank = True
        else:
            collapsed.append(ln)
            blank = False
    return "\n".join(collapsed).strip("\n")


def classify_log(log_path: Path) -> tuple[list[str], str]:
    """Return (categories, snippet) using the production ci_log_errors engine.

    categories: list of normalized category strings (e.g. ['timeout'], ['pytest-error']).
    snippet: short error context (≤80 lines, ≤5000 chars), trimmed.
    """
    if not _ensure_ci_log_errors():
        return [], ""
    cats: list[str] = []
    snippet = ""
    try:
        with log_path.open("r", errors="replace") as fh:
            lines = fh.read().splitlines()
        cats = list({c for c in _categorize_error_log_lines(lines) if c})
        snippet = _extract_error_snippet_from_log_file(log_path) or ""
    except Exception as e:
        logger.warning("classify_log %s: %s", log_path, e)
    return cats, snippet.strip()


# ---------- state machine ----------


def is_terminal(run_data: dict) -> bool:
    return run_data.get("status") == "completed"


def cycle(
    *,
    starting_sha: str | None,
    parallelism: int,
    max_attempts: int,
    stalled_after_hours: int,
    dry_run: bool,
    drain: bool = False,
) -> None:
    db = load_db()
    meta = db.setdefault("_meta", {})

    ensure_clone()
    fetch_main()

    if starting_sha:
        full = resolve_sha(starting_sha)
        if "starting_sha" not in meta:
            meta["starting_sha"] = full
            meta["first_run_at"] = now_iso()
            logger.info("first run; starting_sha=%s", short_sha(full))
        elif meta["starting_sha"] != full:
            logger.warning(
                "DB starting_sha=%s, ignoring CLI %s",
                short_sha(meta["starting_sha"]),
                short_sha(full),
            )
    starting = meta.get("starting_sha")
    if not starting:
        logger.error("no starting_sha; pass --starting-sha on first invocation")
        return

    # ---- 1. discover ----
    if drain:
        logger.info("[drain] skipping discover + launch; only advancing existing SHAs")
    else:
        new_shas = list_shas_since(starting)
        logger.info(
            "main has %d SHAs since %s", len(new_shas), short_sha(starting)
        )
        for sha in new_shas:
            if sha not in db:
                db[sha] = {
                    "state": "discovered",
                    "discovered_at": now_iso(),
                    "merge_date": commit_date(sha),
                    "image_sha256": image_sha256(sha),
                }
                logger.info("discovered %s", short_sha(sha))
            else:
                # Backfill for legacy rows
                if not db[sha].get("merge_date"):
                    md = commit_date(sha)
                    if md:
                        db[sha]["merge_date"] = md
                if not db[sha].get("image_sha256"):
                    db[sha]["image_sha256"] = image_sha256(sha)
        sha_order = {sha: i for i, sha in enumerate(new_shas)}

        # ---- 2. launch (oldest discovered first, up to budget) ----
        in_flight = sum(
            1
            for k, v in db.items()
            if k != "_meta"
            and isinstance(v, dict)
            and v.get("state") in ("launched", "in_progress")
        )
        discovered = sorted(
            (
                k
                for k, v in db.items()
                if k != "_meta"
                and isinstance(v, dict)
                and v.get("state") == "discovered"
            ),
            key=lambda s: sha_order.get(s, 1 << 30),
        )
        budget = max(0, parallelism - in_flight)
        logger.info(
            "in_flight=%d budget=%d discovered=%d",
            in_flight,
            budget,
            len(discovered),
        )
        for sha in discovered[:budget]:
            try:
                branch, probe_head = push_probe(sha, dry_run=dry_run)
                pr = open_pr(sha, branch, dry_run=dry_run)
                remove_auto_reviewers(pr, dry_run=dry_run)
            except subprocess.CalledProcessError as e:
                logger.error(
                    "push/open failed for %s: %s",
                    short_sha(sha),
                    e.stderr or e,
                )
                continue
            except Exception as e:
                logger.error(
                    "push/open exception for %s: %s", short_sha(sha), e
                )
                continue
            db[sha].update(
                {
                    "state": "launched",
                    "branch": branch,
                    "pr": pr,
                    "probe_head_sha": probe_head,
                    "pushed_at": now_iso(),
                    "attempts": [],
                }
            )
            logger.info(
                "launched %s as PR #%s (probe head %s)",
                short_sha(sha),
                pr,
                short_sha(probe_head or ""),
            )

    # ---- 3. collect / advance launched + in_progress ----
    for sha, entry in list(db.items()):
        if sha == "_meta" or not isinstance(entry, dict):
            continue
        if entry.get("state") not in ("launched", "in_progress"):
            continue

        if dry_run:
            logger.info("[dry-run] would query workflow runs for %s", short_sha(sha))
            continue

        # Look up runs against the PROBE PR's head sha, not the main sha.
        # Backfill probe_head_sha if missing (legacy DB rows).
        probe_head = entry.get("probe_head_sha")
        if not probe_head and entry.get("pr"):
            try:
                pr_data = gh_api(f"repos/{REPO}/pulls/{entry['pr']}")
                probe_head = (pr_data or {}).get("head", {}).get("sha")
                if probe_head:
                    entry["probe_head_sha"] = probe_head
                    logger.info(
                        "backfilled probe_head_sha for %s: %s",
                        short_sha(sha),
                        short_sha(probe_head),
                    )
            except Exception as e:
                logger.warning("probe_head_sha backfill for %s failed: %s", sha, e)

        if not probe_head:
            logger.warning("no probe_head_sha for %s; skipping", short_sha(sha))
            continue

        runs = workflow_runs_for_sha(probe_head)
        if not runs:
            continue

        # latest attempt's run per workflow name
        latest: dict[str, dict] = {}
        for r in runs:
            name = r["name"]
            if name not in latest or r["run_attempt"] > latest[name]["run_attempt"]:
                latest[name] = r

        current_attempt = max((r["run_attempt"] for r in latest.values()), default=1)
        attempts = entry.setdefault("attempts", [])
        while len(attempts) < current_attempt:
            attempts.append(
                {
                    "attempt": len(attempts) + 1,
                    "started_at": now_iso(),
                    "completed_at": None,
                    "workflow_runs": {},
                    "jobs": {},
                    "failed_jobs": [],
                }
            )
        att = attempts[current_attempt - 1]
        att["workflow_runs"] = {r["name"]: r["id"] for r in latest.values()}

        # Always refresh jobs — gives `status` live counts mid-run.
        # Schema: jobs[name] = {conclusion, id, url}. Old DBs may have str values;
        # readers handle both via _job_conclusion().
        all_jobs: dict[str, dict] = {}
        all_job_meta: list[tuple[str, int, str]] = []  # (name, id, conclusion)
        for r in latest.values():
            for j in get_run_jobs(r["id"]):
                if j.get("status") != "completed":
                    conclusion = "running"
                else:
                    conclusion = j.get("conclusion") or "completed"
                all_jobs[j["name"]] = {
                    "conclusion": conclusion,
                    "id": j.get("id"),
                    "url": j.get("html_url"),
                    "started_at": j.get("started_at"),
                    "completed_at": j.get("completed_at"),
                }
                all_job_meta.append((j["name"], int(j.get("id") or 0), conclusion))
        att["jobs"] = all_jobs
        att["failed_jobs"] = sorted(
            n
            for n, m in all_jobs.items()
            if m["conclusion"] in ("failure", "timed_out")
        )

        # Drill into logs for failed jobs and pull pytest test IDs.
        # Skip only if we already have a non-empty result (lets regex fixes
        # re-run against the cached log on the next cycle for free).
        failed_tests: dict[str, list[str]] = att.setdefault("failed_tests", {})
        log_meta: dict[str, dict] = att.setdefault("log_meta", {})
        for name, jid, conclusion in all_job_meta:
            if conclusion not in ("failure", "timed_out") or not jid:
                continue
            log_path = fetch_job_log(jid)
            if log_path is None:
                continue
            if not failed_tests.get(name):
                tests = extract_pytest_failures(log_path)
                failed_tests[name] = tests
                logger.info(
                    "extracted %d pytest failure(s) from %s (job %s, %s)",
                    len(tests),
                    name,
                    jid,
                    conclusion,
                )
            if name not in log_meta:
                cats, snippet = classify_log(log_path)
                log_meta[name] = {"categories": cats, "snippet": snippet}
                if cats or snippet:
                    logger.info(
                        "classified %s: cats=%s snippet=%dch",
                        name,
                        ",".join(cats) or "—",
                        len(snippet),
                    )

        if not all(is_terminal(r) for r in latest.values()):
            entry["state"] = "in_progress"
            continue

        # all workflows terminal — finalize this attempt + classify
        att["completed_at"] = now_iso()

        any_failed = any(
            r.get("conclusion") in ("failure", "timed_out") for r in latest.values()
        )
        sha_max_attempts = entry.get("max_attempts", max_attempts)
        if not any_failed:
            entry["state"] = "passed"
            entry["verdict"] = "good"
            logger.info("PASSED %s on attempt %d", short_sha(sha), current_attempt)
        elif current_attempt < sha_max_attempts:
            for r in latest.values():
                if r.get("conclusion") in ("failure", "timed_out"):
                    try:
                        rerun_failed_jobs(r["id"], dry_run=dry_run)
                    except Exception as e:
                        logger.error("rerun-failed-jobs %s: %s", r["id"], e)
            entry["state"] = "in_progress"
            logger.info(
                "RETRY %s attempt %d → %d (cap=%d)",
                short_sha(sha), current_attempt, current_attempt + 1, sha_max_attempts,
            )
        else:
            stuck: set[str] | None = None
            for a in attempts:
                f = set(a.get("failed_jobs", []))
                stuck = f if stuck is None else stuck & f
            entry["stuck_jobs"] = sorted(stuck or [])
            entry["state"] = "failed"
            entry["verdict"] = "bad"
            logger.info(
                "FAILED %s after %d attempts; stuck=%s",
                short_sha(sha),
                sha_max_attempts,
                entry["stuck_jobs"],
            )

    # ---- 4. cleanup passed ----
    for sha, entry in db.items():
        if sha == "_meta" or not isinstance(entry, dict):
            continue
        if entry.get("state") == "passed" and not entry.get("cleaned_at"):
            try:
                close_pr_and_delete_branch(entry["pr"], entry, dry_run=dry_run)
            except subprocess.CalledProcessError as e:
                logger.error("cleanup PR #%s: %s", entry.get("pr"), e.stderr or e)
                continue
            if not dry_run:
                entry["cleaned_at"] = now_iso()
                entry["state"] = "cleaned"

    # ---- 5. write DB ----
    if dry_run:
        n = sum(1 for k in db if k != "_meta")
        logger.info("[dry-run] would write DB with %d entries", n)
    else:
        save_db_atomic(db)

    # ---- 6. always re-render HTML so dashboards reflect current state ----
    if not dry_run:
        try:
            cmd_render_html()
        except Exception as e:
            logger.warning("auto-render-html failed: %s", e)


# ---------- subcommands ----------


def _job_conclusion(j) -> str:
    """Read conclusion from a jobs[name] entry (handles legacy str + new dict)."""
    if isinstance(j, str):
        return j
    if isinstance(j, dict):
        return j.get("conclusion") or "?"
    return "?"


def _job_url(j) -> str | None:
    """Read html_url from a jobs[name] entry. None for legacy str format."""
    if isinstance(j, dict):
        return j.get("url")
    return None


def _fmt_duration(secs: int) -> str:
    if secs < 0:
        return "—"
    if secs >= 60:
        return f"{secs // 60}m{secs % 60:02d}s"
    return f"{secs}s"


def _job_timing(j, conclusion: str = "") -> tuple[str, str]:
    """Return (started_str, duration_str) for display.

    For running jobs (conclusion in running/queued/pending) and a known
    started_at, emits a `<span class='live-duration' data-started='<iso>'>`
    that the page's JS ticker updates every second.
    """
    if not isinstance(j, dict):
        return "—", "—"
    s = j.get("started_at")
    c = j.get("completed_at")
    started = "—"
    duration = "—"
    if s:
        dt = _to_pt(s)
        if dt:
            started = dt.strftime("%H:%M:%S")
    if s and c:
        try:
            ds = datetime.fromisoformat(s)
            dc = datetime.fromisoformat(c)
            duration = _fmt_duration(int((dc - ds).total_seconds()))
        except Exception:
            pass
    elif s and conclusion in ("running", "queued", "pending"):
        try:
            ds = datetime.fromisoformat(s)
            now = datetime.now(timezone.utc)
            initial = _fmt_duration(int((now - ds).total_seconds()))
        except Exception:
            initial = "—"
        duration = (
            f"<span class='live-duration' data-started='{s}'>{initial}</span>"
        )
    return started, duration


_LIVE_DURATION_JS = """
<script>
(function() {
  function pad(n) { return n < 10 ? "0" + n : "" + n; }
  function fmt(secs) {
    if (secs < 0) return "—";
    if (secs >= 60) return Math.floor(secs / 60) + "m" + pad(secs % 60) + "s";
    return secs + "s";
  }
  function tick() {
    var now = Date.now();
    document.querySelectorAll(".live-duration").forEach(function(el) {
      var s = el.getAttribute("data-started");
      if (!s) return;
      var t = Date.parse(s);
      if (isNaN(t)) return;
      el.textContent = fmt(Math.floor((now - t) / 1000));
    });
  }
  tick();
  setInterval(tick, 1000);
})();
function toggleSnip(ev, id) {
  var row = document.getElementById(id);
  if (row) row.classList.toggle('show');
  var tgt = ev.currentTarget || ev.target;
  if (tgt) {
    var tri = tgt.querySelector('.triangle-toggle');
    if (tri) tri.classList.toggle('expanded');
  }
  ev.stopPropagation();
}
</script>
"""


def _live_counts(entry: dict) -> str:
    """Compact 'Xp/Yf/Zr' (passed/failed/running) — used in terminal output."""
    attempts = entry.get("attempts", [])
    if not attempts:
        return "-"
    jobs = attempts[-1].get("jobs", {})
    if not jobs:
        return "-"
    cons = [_job_conclusion(v) for v in jobs.values()]
    p = sum(1 for c in cons if c == "success")
    f = sum(1 for c in cons if c == "failure")
    r = sum(1 for c in cons if c in ("running", "queued", "pending"))
    return f"{p}p/{f}f/{r}r"


def _descriptive_counts(entry: dict) -> str:
    """Long form for HTML: '76 pass, 2 fail, 12 running, 4 skipped'."""
    attempts = entry.get("attempts", [])
    if not attempts:
        return "—"
    jobs = attempts[-1].get("jobs", {})
    if not jobs:
        return "—"
    cons = [_job_conclusion(v) for v in jobs.values()]
    p = sum(1 for c in cons if c == "success")
    f = sum(1 for c in cons if c in ("failure", "timed_out"))
    r = sum(1 for c in cons if c in ("running", "queued", "pending"))
    s = sum(1 for c in cons if c in ("skipped", "cancelled", "neutral"))
    parts = [f"{p} pass", f"{f} fail", f"{r} running"]
    if s:
        parts.append(f"{s} skipped")
    return ", ".join(parts)


def _pr_url(pr: int | None) -> str:
    if not pr or pr == "-" or pr == -1:
        return "-"
    return f"https://github.com/{REPO}/pull/{pr}"


def _to_pt(iso: str | None) -> datetime | None:
    """Parse an ISO-8601 string and convert to Pacific Time (PST/PDT)."""
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso).astimezone(PT)
    except Exception:
        return None


def _sha_entries_sorted(db: dict) -> list[tuple[str, dict]]:
    """Return [(sha, entry)] sorted by merge time (newest first), unknown last."""
    items = [
        (s, e)
        for s, e in db.items()
        if s != "_meta" and isinstance(e, dict)
    ]
    # Newest-first: compare by tz-aware datetime so mixed committer timezones
    # sort correctly. Entries without merge_date fall to the bottom.
    far_past = datetime.min.replace(tzinfo=timezone.utc)
    items.sort(
        key=lambda kv: (_to_pt(kv[1].get("merge_date")) or far_past, kv[0]),
        reverse=True,
    )
    return items


def _short_merge_date(iso: str | None) -> str:
    """'YYYY-MM-DD HH:MM:SS' in Pacific Time."""
    dt = _to_pt(iso)
    if dt is None:
        return "?"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def cmd_status() -> None:
    db = load_db()
    meta = db.get("_meta", {})
    print(f"DB: {DB_PATH}")
    print(f"starting_sha: {short_sha(meta.get('starting_sha', '')) or '(none)'}")
    print(f"first_run_at: {meta.get('first_run_at', '(none)')}")
    print()
    fmt = "{:<12} {:<13} {:<21} {:<13} {:<10} {:<11} {}"
    print(
        fmt.format(
            "ImageSHA256",
            "SHA",
            "Merged (PT)",
            "State",
            "Attempts",
            "Jobs(p/f/r)",
            "PR",
        )
    )
    print("-" * 120)
    for sha, e in _sha_entries_sorted(db):
        n_att = len(e.get("attempts", []))
        print(
            fmt.format(
                e.get("image_sha256") or "?",
                short_sha(sha),
                _short_merge_date(e.get("merge_date")),
                _display_state(e),
                f"{n_att}/{e.get('max_attempts', DEFAULT_MAX_ATTEMPTS)}",
                _live_counts(e),
                _pr_url(e.get("pr")),
            )
        )
    print()
    print("Sorted by merge date (newest first). Jobs(p/f/r) = passed/failed/running")
    print("from last cycle (≤3 min stale). Verdict finalizes after all retries.")


def cmd_report(csv_path: str | None = None) -> None:
    db = load_db()
    sha_entries = _sha_entries_sorted(db)

    # ---- 1. per-SHA state ----
    print("=== Per-SHA State ===")
    fmt = "{:<13} {:<14} {:<10} {:<55} {}"
    print(fmt.format("SHA", "State", "Attempts", "PR", "Stuck Jobs"))
    for sha, e in sha_entries:
        n_att = len(e.get("attempts", []))
        stuck = ", ".join(e.get("stuck_jobs", [])) or "—"
        print(
            fmt.format(
                short_sha(sha),
                _display_state(e),
                f"{n_att}/{e.get('max_attempts', DEFAULT_MAX_ATTEMPTS)}",
                _pr_url(e.get("pr")),
                stuck,
            )
        )

    # ---- 2. per-SHA failure detail ----
    print("\n=== Per-Attempt Failures ===")
    any_fail = False
    for sha, e in sha_entries:
        sha_failed_anywhere = any(a.get("failed_jobs") for a in e.get("attempts", []))
        if not sha_failed_anywhere:
            continue
        any_fail = True
        commit_url = f"https://github.com/{REPO}/commit/{sha}"
        pr_url = _pr_url(e.get("pr"))
        print(f"  {short_sha(sha)}")
        print(f"    commit: {commit_url}")
        print(f"    PR:     {pr_url}")
        for a in e.get("attempts", []):
            failed = a.get("failed_jobs", [])
            if not failed:
                continue
            print(f"    attempt {a['attempt']}  ({len(failed)} failed):")
            jobs_dict = a.get("jobs", {})
            failed_tests = a.get("failed_tests") or {}
            for j in failed:
                print(f"      ✗ {j}")
                url = _job_url(jobs_dict.get(j))
                if url:
                    print(f"        {url}")
                tests = failed_tests.get(j, [])
                if tests:
                    shown = tests[:3]
                    for t in shown:
                        print(f"        - {t}")
                    if len(tests) > 3:
                        print(f"        ... +{len(tests) - 3} more")
    if not any_fail:
        print("  (no failures recorded)")

    # ---- 3. failure frequency across SHAs ----
    fail_count: dict[str, int] = {}
    sha_set_per_job: dict[str, set] = {}
    for sha, e in sha_entries:
        for a in e.get("attempts", []):
            for j in a.get("failed_jobs", []):
                fail_count[j] = fail_count.get(j, 0) + 1
                sha_set_per_job.setdefault(j, set()).add(sha)

    print("\n=== Failure Frequency (any attempt) ===")
    if not fail_count:
        print("  (no failures yet)")
    else:
        print(f"  {'Failures':>8}  {'SHAs':>5}  Job")
        for job, cnt in sorted(fail_count.items(), key=lambda kv: -kv[1]):
            n_shas = len(sha_set_per_job[job])
            print(f"  {cnt:>8}  {n_shas:>5}  {job}")

    # ---- 4. flake leaderboard (failed once + passed once on same SHA) ----
    flake_count: dict[str, int] = {}
    for sha, e in sha_entries:
        attempts = e.get("attempts", [])
        if len(attempts) < 2:
            continue
        seen_jobs = {j for a in attempts for j in a.get("jobs", {})}
        for job in seen_jobs:
            ever_failed = any(job in a.get("failed_jobs", []) for a in attempts)
            ever_passed = any(
                _job_conclusion(a.get("jobs", {}).get(job)) == "success"
                for a in attempts
            )
            if ever_failed and ever_passed:
                flake_count[job] = flake_count.get(job, 0) + 1

    print("\n=== Flake Leaderboard (failed + recovered on same SHA) ===")
    if not flake_count:
        print("  (no confirmed flakes yet — needs ≥2 attempts on a SHA)")
    else:
        for job, cnt in sorted(flake_count.items(), key=lambda kv: -kv[1]):
            print(f"  {cnt:>3}  {job}")

    # ---- 5. test-level failure frequency (from log parsing) ----
    test_count: dict[str, int] = {}
    test_shas: dict[str, set] = {}
    for sha, e in sha_entries:
        for a in e.get("attempts", []):
            for _job, tests in (a.get("failed_tests") or {}).items():
                for t in tests:
                    test_count[t] = test_count.get(t, 0) + 1
                    test_shas.setdefault(t, set()).add(sha)

    print("\n=== Test-Level Failure Frequency ===")
    if not test_count:
        print("  (no pytest test IDs extracted yet)")
    else:
        print(f"  {'Failures':>8}  {'SHAs':>5}  Test")
        for t, cnt in sorted(test_count.items(), key=lambda kv: -kv[1]):
            print(f"  {cnt:>8}  {len(test_shas[t]):>5}  {t}")

    # ---- 6. optional CSV export ----
    if csv_path:
        import csv

        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sha", "attempt", "job", "conclusion", "failed_tests"])
            for sha, e in sha_entries:
                for a in e.get("attempts", []):
                    failed_tests = a.get("failed_tests") or {}
                    for j, v in a.get("jobs", {}).items():
                        tests = ";".join(failed_tests.get(j, []))
                        w.writerow([sha, a["attempt"], j, _job_conclusion(v), tests])
        print(f"\nCSV written: {csv_path}")


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


_HTML_STYLE = """
<style>
  body { font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0; padding: 24px; background: #fafbfc; color: #24292e; max-width: 1400px; }
  h1 { font-size: 20px; margin: 0 0 4px 0; }
  h2 { font-size: 16px; margin: 24px 0 8px 0; padding-bottom: 4px; border-bottom: 1px solid #e1e4e8; }
  .meta { color: #586069; font-size: 13px; margin-bottom: 16px; }
  .meta span { margin-right: 16px; }
  .meta code, code { background: #f6f8fa; padding: 2px 6px; border-radius: 3px; font-size: 12px;
                     font-family: "SF Mono", Consolas, monospace; }
  a { color: #0366d6; text-decoration: none; }
  a:hover { text-decoration: underline; }
  a[target="_blank"]::after { content: " ↗"; font-size: 0.85em; color: #959da5; }
  .verdict-good { color: #28a745; font-weight: 600; }
  .verdict-bad  { color: #d73a49; font-weight: 600; }
  .verdict-pending { color: #bf6c00; font-weight: 600; }
  .pill { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 12px;
          font-weight: 500; }
  .pill-pass { background: #d4edda; color: #155724; }
  .pill-fail { background: #f8d7da; color: #721c24; }
  .pill-run  { background: #cce5ff; color: #004085; }
  table { border-collapse: collapse; margin: 8px 0 16px 0; font-size: 13px; width: 100%; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #eaecef; vertical-align: top; }
  th { background: #f6f8fa; font-weight: 600; font-size: 12px; text-transform: uppercase;
       letter-spacing: 0.05em; color: #586069; }
  tr.fail td { background: #f5c6cb; }
  /* Final attempt of the job failed → unrecovered regression. Darker red. */
  tr.fail-final td { background: #ea868f; }
  /* Earlier failure that the job later recovered from. Lighter red. */
  tr.fail-flake td { background: #fbe4e6; }
  tr.pass td { background: #c3e6cb; }
  tr.run  td { background: #b8daff; }
  tr.skip td { background: #f0f0f0; color: #6a737d; }
  details { background: #fff; border: 1px solid #d0d7de; border-radius: 6px;
            margin: 8px 0; padding: 0; }
  details > summary { cursor: pointer; padding: 10px 14px; font-weight: 600;
                      list-style: none; user-select: none; }
  details > summary::-webkit-details-marker { display: none; }
  details > summary::before { content: "▶ "; color: #586069; font-size: 11px;
                              display: inline-block; transition: transform 0.1s; }
  details[open] > summary::before { transform: rotate(90deg); }
  details > summary:hover { background: #f6f8fa; border-radius: 6px 6px 0 0; }
  details[open] > summary { border-bottom: 1px solid #eaecef; border-radius: 6px 6px 0 0; }
  details > .details-body { padding: 0 14px 12px 14px; }
  .cat-list { font-size: 11px; color: #586069; font-family: "SF Mono", Consolas, monospace; }
  .attempt-badge { display: inline-block; padding: 1px 7px; margin: 0 4px 2px 0;
                   border-radius: 10px; font-size: 11px; font-weight: 600;
                   white-space: nowrap;
                   font-variant-numeric: tabular-nums; font-family: "SF Mono", Consolas, monospace; }
  td.attempts-cell { white-space: nowrap; min-width: 180px; }
  td.cat-cell { white-space: nowrap; min-width: 200px; }
  td.started-cell, td.duration-cell, td.status-cell, td.attempt-cell { white-space: nowrap; }
  .attempt-summary { margin: 12px 0 4px 0; }
  .attempt-summary .pill { margin-right: 6px; }
  .snip-toggle { cursor: pointer; user-select: none; display: inline-block;
                 margin-left: 6px; vertical-align: middle; }
  .triangle-toggle { display: inline-block; transition: transform 300ms ease;
                     transform-origin: center; color: #586069; font-size: 14px;
                     margin-right: 4px; }
  .triangle-toggle.expanded { transform: rotate(90deg); }
  .status-x { display: inline-block; width: 14px; height: 14px; line-height: 14px;
              text-align: center; background: #d73a49; color: #fff;
              border-radius: 50%; font-weight: 700; font-size: 10px;
              font-family: "SF Mono", Consolas, monospace; }
  .status-check { display: inline-block; width: 14px; height: 14px; line-height: 14px;
                  text-align: center; background: #28a745; color: #fff;
                  border-radius: 50%; font-weight: 700; font-size: 10px; }
  .status-dot { display: inline-block; width: 14px; height: 14px; line-height: 14px;
                text-align: center; color: #586069; font-size: 14px; }
  tr.snippet-row { display: none; }
  tr.snippet-row.show { display: table-row; }
  tr.snippet-row > td { padding: 4px 14px 8px 14px; }
  tr.snippet-row.fail-final > td { background: #ea868f; }
  tr.snippet-row.fail-flake > td { background: #fbe4e6; }
  tr.snippet-row > td { background: #f5c6cb; }
  pre.snip { background: #0d1117; color: #e6edf3; font-size: 11px;
             padding: 10px 12px; border-radius: 4px; overflow-x: auto;
             margin: 4px 0 0 0; max-height: 320px; overflow-y: auto;
             white-space: pre-wrap; word-break: break-word;
             font-family: "SF Mono", Consolas, monospace; line-height: 1.45; }
  .job-name { font-family: "SF Mono", Consolas, monospace; font-size: 12px; white-space: nowrap; }
  .test-list { margin: 4px 0 0 16px; padding: 0; font-size: 12px;
               color: #586069; font-family: "SF Mono", Consolas, monospace; }
  .test-list li { margin: 2px 0; }
  .summary { display: flex; gap: 16px; margin: 8px 0 16px 0; }
  .summary-box { background: #fff; border: 1px solid #e1e4e8; border-radius: 6px;
                 padding: 12px 16px; min-width: 100px; }
  .summary-box .label { color: #586069; font-size: 11px; text-transform: uppercase;
                        letter-spacing: 0.05em; }
  .summary-box .value { font-size: 22px; font-weight: 600; margin-top: 4px; }
  .index-table tr:hover td { background: #f6f8fa; }
</style>
"""


def _display_state(entry: dict) -> str:
    """Human-readable state. Folds passed/cleaned→PASSED, failed→FAILED, discovered→queued."""
    s = entry.get("state", "?")
    if s in ("passed", "cleaned"):
        return "PASSED"
    if s == "failed":
        return "FAILED"
    if s == "discovered":
        return "queued"
    return s


def _state_class(entry: dict) -> str:
    s = entry.get("state")
    if s in ("passed", "cleaned"):
        return "verdict-good"
    if s == "failed":
        return "verdict-bad"
    return "verdict-pending"


def _render_probe_page(sha: str, entry: dict) -> str:
    """Per-SHA detail page."""
    pr = entry.get("pr")
    pr_url = _pr_url(pr) if pr else "-"
    commit_url = f"https://github.com/{REPO}/commit/{sha}"
    state_disp = _display_state(entry)
    sclass = _state_class(entry)
    img = entry.get("image_sha256") or "?"
    merge_dt = _short_merge_date(entry.get("merge_date"))
    n_att = len(entry.get("attempts", []))

    head = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Probe {short_sha(sha)} — ci-health</title>
{_HTML_STYLE}
</head><body>
<h1>Probe <code>{short_sha(sha)}</code></h1>
<div class="meta">
  <span>Merged (PT): <code>{merge_dt}</code></span>
  <span>ImageSHA256: <code>{img}</code></span>
  <span>State: <span class="{sclass}">{state_disp}</span></span>
  <span>Attempts: <code>{n_att}/{entry.get('max_attempts', DEFAULT_MAX_ATTEMPTS)}</code></span>
</div>
<div class="meta">
  <span><a href="{commit_url}" target="_blank" rel="noopener noreferrer">commit on GitHub</a></span>
  <span><a href="{pr_url}" target="_blank" rel="noopener noreferrer">probe PR{f" #{pr}" if pr else ""}</a></span>
</div>
"""
    # If overall PASSED but earlier attempts had failures → flake banner.
    attempts_list = entry.get("attempts", [])
    if state_disp == "PASSED" and any(a.get("failed_jobs") for a in attempts_list):
        n_flaked = sum(len(a.get("failed_jobs", [])) for a in attempts_list[:-1])
        head += (
            f'<div style="background:#fff8c5; border:1px solid #d4a72c; '
            f'border-radius:6px; padding:10px 14px; margin:8px 0 16px 0;">'
            f'<strong>Confirmed flake.</strong> Final verdict: '
            f'<span class="{sclass}">PASSED</span>. '
            f'Earlier attempt(s) had {n_flaked} job-level failure(s); '
            f'cleared automatically by <code>gh run rerun --failed</code>. '
            f'Per-attempt history shown below.'
            f'</div>'
        )
    # Live counts summary
    counts = _descriptive_counts(entry)
    summary = f"""
<div class="summary">
  <div class="summary-box" style="min-width: 240px;"><div class="label">Latest counts</div><div class="value" style="font-size: 16px;">{counts}</div></div>
  <div class="summary-box"><div class="label">State</div><div class="value"><span class="{sclass}">{state_disp}</span></div></div>
</div>
"""

    # ------- View: group by JOB, with attempt history badges -------
    sections = []
    attempts_all = entry.get("attempts", [])
    snip_uid = [0]

    # 1. Build job-history dedup'd by *physical* run.
    # GitHub's rerun-failed-jobs creates new job IDs in each attempt's view,
    # but for unchanged (carried-over) jobs those new IDs point at the same
    # physical run with the same timestamps. Dedup on (started_at, completed_at)
    # — two entries with identical timestamps for the same job name ARE the
    # same physical run, regardless of GitHub's per-attempt ID.
    job_runs: dict[str, list[tuple[int, dict]]] = {}
    for a in attempts_all:
        att_num = a.get("attempt", 0)
        for name, j in a.get("jobs", {}).items():
            runs = job_runs.setdefault(name, [])
            if not isinstance(j, dict):
                runs.append((att_num, j))
                continue
            sig = (j.get("started_at"), j.get("completed_at"))
            existing = {
                (r[1].get("started_at"), r[1].get("completed_at"))
                for r in runs if isinstance(r[1], dict)
            }
            # If both ends are None (no timing yet), treat as unique per attempt.
            if sig != (None, None) and sig in existing:
                continue  # same physical run — skip dup
            runs.append((att_num, j))

    # 2. Per-attempt overview now reflects what ACTUALLY ran in each attempt
    # (i.e., new physical runs, not carried-over status).
    runs_per_att: dict[int, list[dict]] = {}
    for runs in job_runs.values():
        for att_num, j in runs:
            runs_per_att.setdefault(att_num, []).append(j)

    summary_lines = ["<h2>Attempts overview <small style='color:#586069;font-weight:400'>(what actually ran each attempt)</small></h2>",
                     "<div class='attempt-summary'>"]
    for att_num in sorted(runs_per_att):
        these = runs_per_att[att_num]
        n_pass = sum(1 for j in these if _job_conclusion(j) == "success")
        n_fail = sum(1 for j in these if _job_conclusion(j) in ("failure", "timed_out"))
        n_run = sum(1 for j in these if _job_conclusion(j) in ("running", "queued", "pending"))
        n_skip = sum(1 for j in these if _job_conclusion(j) in ("skipped", "cancelled", "neutral"))
        summary_lines.append(
            f"<div><strong>Attempt {att_num}:</strong> "
            f"<span class='pill pill-pass'>{n_pass} pass</span> "
            f"<span class='pill pill-fail'>{n_fail} fail</span> "
            f"<span class='pill pill-run'>{n_run} run</span>"
            + (f" <span style='color:#959da5;font-size:11px'>({n_skip} skipped)</span>" if n_skip else "")
            + "</div>"
        )
    summary_lines.append("</div>")
    sections.append("\n".join(summary_lines))

    # Flat: one row per (job, attempt-run). No more per-job grouping or
    # per-attempt badges — if a job ran 3 times, it's 3 rows.
    sec = [
        "<h2>Jobs (one row per attempt-run; failed first)</h2>",
        "<table><thead><tr>"
        "<th></th>"
        "<th>Job</th>"
        "<th>Attempt</th>"
        "<th>Started (PT)</th>"
        "<th>Duration</th>"
        "<th>Failure Detail</th>"
        "</tr></thead><tbody>",
    ]

    flat_rows: list[tuple[str, int, dict]] = []
    for name, runs in job_runs.items():
        for att_num, j in runs:
            flat_rows.append((name, att_num, j))
    flat_rows.sort(key=lambda x: (x[0], x[1]))

    # Per job: which attempt # is the LAST run? Only that row renders dark
    # if it's a failure (= unrecovered final). Earlier failure rows of the
    # same job render light ("recovered or eclipsed by a later attempt").
    job_last_attempt: dict[str, int] = {}
    for name, runs in job_runs.items():
        if runs:
            job_last_attempt[name] = runs[-1][0]

    for name, att_num, j in flat_rows:
        conc = _job_conclusion(j)
        if conc in ("failure", "timed_out"):
            is_last = att_num == job_last_attempt.get(name)
            row_class = "fail-final" if is_last else "fail-flake"
            status_html = f"<span class='status-x' title='{conc}'>✗</span>"
        elif conc == "success":
            row_class = "pass"
            status_html = "<span class='status-check' title='success'>✓</span>"
        elif conc in ("running", "queued", "pending"):
            row_class = "run"
            status_html = f"<span class='status-dot' title='{conc}'>⏳</span>"
        elif conc in ("skipped", "cancelled", "neutral"):
            row_class = "skip"
            status_html = f"<span class='status-dot' title='{conc}'>⊘</span>"
        else:
            row_class = ""
            status_html = f"<span class='status-dot' title='{conc or "?"}'>?</span>"

        started_pt = "—"
        s = j.get("started_at") if isinstance(j, dict) else None
        if s:
            dt = _to_pt(s)
            if dt:
                started_pt = dt.strftime("%Y-%m-%d %H:%M:%S")
        _short_started, duration = _job_timing(j, conc)

        url = _job_url(j)
        name_html = (
            f"<a href='{url}' target='_blank' rel='noopener noreferrer'>{_html_escape(name)}</a>"
            if url else _html_escape(name)
        )

        detail_parts: list[str] = []
        snippet_toggle_html = ""
        snippet_rows_inline: list[str] = []
        if conc in ("failure", "timed_out"):
            for a in attempts_all:
                if a.get("attempt") != att_num:
                    continue
                log_info = (a.get("log_meta") or {}).get(name, {})
                cats = log_info.get("categories") or []
                snippet = log_info.get("snippet") or ""
                tests = (a.get("failed_tests") or {}).get(name, [])
                if cats:
                    detail_parts.append(
                        f"<div class='cat-list'>{_html_escape(', '.join(cats))}</div>"
                    )
                if tests:
                    items = "".join(f"<li>{_html_escape(t)}</li>" for t in tests[:50])
                    more = (
                        f"<li>… +{len(tests) - 50} more</li>" if len(tests) > 50 else ""
                    )
                    detail_parts.append(f"<ul class='test-list'>{items}{more}</ul>")
                if snippet:
                    snip_uid[0] += 1
                    sid = f"snip-job-{snip_uid[0]}"
                    snippet_html = render_snippet_html(snippet)
                    snippet_toggle_html = (
                        f" <span class='snip-toggle' "
                        f"onclick=\"toggleSnip(event, '{sid}')\" "
                        f"title='show failure snippet ({len(snippet)} chars)'>"
                        f"<span class='triangle-toggle'>▶</span>"
                        f"</span>"
                    )
                    snippet_rows_inline.append(
                        f"<tr id='{sid}' class='snippet-row {row_class}'><td colspan='6'>"
                        f"<pre class='snip'>{snippet_html}</pre>"
                        f"</td></tr>"
                    )
                break
        detail_html = "".join(detail_parts) or '<span style="color:#959da5">—</span>'

        sec.append(
            f"<tr class='{row_class}'>"
            f"<td class='status-cell'>{status_html}</td>"
            f"<td class='job-name'>{name_html}{snippet_toggle_html}</td>"
            f"<td class='attempt-cell'>{att_num}</td>"
            f"<td class='started-cell'>{started_pt}</td>"
            f"<td class='duration-cell'>{duration}</td>"
            f"<td>{detail_html}</td>"
            f"</tr>"
        )
        sec.extend(snippet_rows_inline)
    sec.append("</tbody></table>")
    sections.append("\n".join(sec))

    if not sections:
        sections.append("<p><em>No attempts recorded yet.</em></p>")

    foot = _LIVE_DURATION_JS + "</body></html>"
    return head + summary + "\n".join(sections) + foot


def _render_probe_index(db: dict, page_paths: dict[str, str]) -> str:
    """Index page listing every probed SHA with link to its detail page."""
    rows = []
    for sha, e in _sha_entries_sorted(db):
        state_disp = _display_state(e)
        sclass = _state_class(e)
        link = page_paths.get(sha, "#")
        pr = e.get("pr")
        pr_link = (
            f"<a href='{_pr_url(pr)}' target='_blank' rel='noopener noreferrer'>#{pr}</a>"
            if pr else "—"
        )
        rows.append(
            f"<tr>"
            f"<td><code>{short_sha(sha)}</code></td>"
            f"<td><code>{e.get('image_sha256') or '?'}</code></td>"
            f"<td>{_short_merge_date(e.get('merge_date'))}</td>"
            f"<td><span class='{sclass}'>{state_disp}</span></td>"
            f"<td>{len(e.get('attempts', []))}/{e.get('max_attempts', DEFAULT_MAX_ATTEMPTS)}</td>"
            f"<td>{_descriptive_counts(e)}</td>"
            f"<td>{pr_link}</td>"
            f"<td><a href='{link}' target='_blank' rel='noopener noreferrer'>detail</a></td>"
            f"</tr>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>ci-health probe index</title>
{_HTML_STYLE}
</head><body>
<h1>ci-health probe index</h1>
<div class="meta">
  Generated {now_iso()}. Sorted by merge date (newest first).
</div>
<table class="index-table">
<thead><tr>
<th>SHA</th><th>ImageSHA256</th><th>Merged (PT)</th><th>State</th>
<th>Attempts</th><th>Jobs</th><th>PR</th><th></th>
</tr></thead>
<tbody>
{''.join(rows)}
</tbody></table>
</body></html>
"""


def _probe_filename(sha: str) -> str:
    """Short, human-readable filename: 'CI-<short_sha>.html'."""
    return f"CI-{short_sha(sha)}.html"


def cmd_render_html(output_root: Path | None = None) -> None:
    """Render per-SHA detail pages + index under output_root."""
    if output_root is None:
        output_root = Path.home() / "dynamo" / "commits" / "logs"
    db = load_db()
    output_root.mkdir(parents=True, exist_ok=True)
    page_paths: dict[str, str] = {}
    n_pages = 0
    for sha, e in _sha_entries_sorted(db):
        merge_dt = _to_pt(e.get("merge_date"))
        date_str = merge_dt.strftime("%Y-%m-%d") if merge_dt else "unknown"
        date_dir = output_root / date_str
        date_dir.mkdir(parents=True, exist_ok=True)
        out_path = date_dir / _probe_filename(sha)
        out_path.write_text(_render_probe_page(sha, e))
        page_paths[sha] = f"{date_str}/{_probe_filename(sha)}"
        n_pages += 1
        # Cleanup any old long-named files for this same sha
        old = date_dir / f"{sha}.html"
        if old.exists():
            old.unlink()
    index_path = output_root / "index.html"
    index_path.write_text(_render_probe_index(db, page_paths))
    print(f"wrote {n_pages} per-SHA pages + {index_path}")


def cmd_retry(sha_or_prefix: str, new_max_attempts: int) -> None:
    """Bump per-SHA max_attempts; if currently FAILED, flip to in_progress
    and trigger another `rerun-failed-jobs` to actually start attempt N+1."""
    db = load_db()
    matches = [
        k
        for k in db
        if k != "_meta" and (k == sha_or_prefix or k.startswith(sha_or_prefix))
    ]
    if not matches:
        print(f"no entry matching {sha_or_prefix}")
        return
    if len(matches) > 1:
        print(f"ambiguous: {[short_sha(k) for k in matches]}")
        return
    sha = matches[0]
    entry = db[sha]
    if entry.get("state") == "cleaned":
        print(f"{short_sha(sha)} is PASSED + cleaned — PR closed, branch deleted, can't retry")
        return
    cur_max = entry.get("max_attempts", DEFAULT_MAX_ATTEMPTS)
    if new_max_attempts <= cur_max:
        print(f"max_attempts {new_max_attempts} ≤ current {cur_max} — no change")
        return
    entry["max_attempts"] = new_max_attempts
    print(f"{short_sha(sha)}: max_attempts {cur_max} → {new_max_attempts}")

    # If FAILED (terminal): flip back + actively kick a rerun on each failed workflow.
    if entry.get("state") == "failed":
        attempts = entry.get("attempts", [])
        if attempts:
            last = attempts[-1]
            for wf_name, run_id in (last.get("workflow_runs") or {}).items():
                try:
                    rerun_failed_jobs(run_id, dry_run=False)
                    print(f"  triggered rerun-failed-jobs on {wf_name} (run {run_id})")
                except Exception as e:
                    logger.warning("rerun-failed-jobs %s: %s", run_id, e)
        entry.pop("verdict", None)
        entry.pop("stuck_jobs", None)
        entry["state"] = "in_progress"
        print(f"  flipped state: failed → in_progress")
    else:
        print(f"  state unchanged ({entry['state']}); next cycle will use new cap")
    save_db_atomic(db)
    # Re-render HTML so the dashboard reflects the new cap immediately.
    try:
        cmd_render_html()
    except Exception as e:
        logger.warning("auto-render-html after retry failed: %s", e)


def cmd_reset(sha_or_prefix: str) -> None:
    db = load_db()
    matches = [
        k
        for k in db
        if k != "_meta" and (k == sha_or_prefix or k.startswith(sha_or_prefix))
    ]
    if not matches:
        print(f"no entry matching {sha_or_prefix}")
        return
    for k in matches:
        del db[k]
    save_db_atomic(db)
    print(f"reset {len(matches)} entries: {[short_sha(k) for k in matches]}")


# ---------- main ----------


def main() -> None:
    p = argparse.ArgumentParser(prog="probe_ci_health")
    p.add_argument("-v", "--verbose", action="count", default=0)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run")
    pr.add_argument("--starting-sha")
    pr.add_argument("--parallelism", type=int, default=DEFAULT_PARALLELISM)
    pr.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    pr.add_argument(
        "--stalled-after-hours", type=int, default=DEFAULT_STALLED_AFTER_HOURS
    )
    pr.add_argument(
        "--dry-run", "--dryrun", dest="dry_run", action="store_true"
    )
    pr.add_argument(
        "--drain",
        action="store_true",
        help="Advance existing in-flight SHAs only; skip discover + launch. "
        "Useful for winding down without queueing more probes.",
    )

    sub.add_parser("status")
    p_rep = sub.add_parser("report")
    p_rep.add_argument("--csv", help="write per-job rows to this CSV path")
    p_rh = sub.add_parser("render-html")
    p_rh.add_argument(
        "--output-root",
        default=str(Path.home() / "dynamo" / "commits" / "logs"),
        help="root dir for per-SHA pages + index.html",
    )
    p_retry = sub.add_parser(
        "retry",
        help="bump per-SHA max_attempts and (if FAILED) trigger another rerun",
    )
    p_retry.add_argument("sha", help="SHA or prefix")
    p_retry.add_argument(
        "--max-attempts", type=int, required=True,
        help="new per-SHA max_attempts cap (must be > current)",
    )
    pre = sub.add_parser("reset")
    pre.add_argument("sha")

    args = p.parse_args()

    level = logging.WARNING
    if args.verbose >= 1:
        level = logging.INFO
    if args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.cmd == "run":
        try:
            with file_lock(LOCK_PATH):
                cycle(
                    starting_sha=args.starting_sha,
                    parallelism=args.parallelism,
                    max_attempts=args.max_attempts,
                    stalled_after_hours=args.stalled_after_hours,
                    dry_run=args.dry_run,
                    drain=args.drain,
                )
        except RuntimeError as e:
            logger.warning("%s", e)
            sys.exit(0)
    elif args.cmd == "status":
        cmd_status()
    elif args.cmd == "report":
        cmd_report(csv_path=args.csv)
    elif args.cmd == "render-html":
        cmd_render_html(output_root=Path(args.output_root))
    elif args.cmd == "retry":
        cmd_retry(args.sha, args.max_attempts)
    elif args.cmd == "reset":
        cmd_reset(args.sha)


if __name__ == "__main__":
    main()
