#!/usr/bin/env python3
"""
Quota experiment (strict order):
  1) /rate_limit (before)
  2) run show_commit_history.py (work only; no internal /rate_limit)
  3) /rate_limit (after)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def _load_github_token() -> str:
    # Use the same token that `gh` uses, so the before/after quota reflects
    # the `gh api ...` calls performed inside the subprocess.
    gh_hosts = Path.home() / ".config" / "gh" / "hosts.yml"
    if gh_hosts.exists():
        cfg = yaml.safe_load(gh_hosts.read_text() or "") or {}
        gh = cfg.get("github.com") or {}
        tok = (gh.get("oauth_token") or "").strip()
        if tok:
            return tok
        users = gh.get("users") or {}
        if isinstance(users, dict):
            for _user, ucfg in users.items():
                if not isinstance(ucfg, dict):
                    continue
                tok2 = (ucfg.get("oauth_token") or "").strip()
                if tok2:
                    return tok2

    raise RuntimeError("No GitHub token found in ~/.config/github-token or ~/.config/gh/hosts.yml")


def get_rate_limit_snapshot_via_gh(*, token: str) -> dict:
    # One API call: GET /rate_limit (via `gh`) returning ALL buckets.
    env = dict(os.environ)
    env["GH_TOKEN"] = token
    res = subprocess.run(
        ["gh", "api", "rate_limit"],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(f"gh api rate_limit failed: {res.stderr.strip()}")
    try:
        return json.loads(str(res.stdout or "").strip() or "{}") or {}
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Unexpected gh rate_limit JSON: {res.stdout!r}") from e


def _get_remaining(snapshot: dict, *, bucket: str) -> int:
    try:
        return int((((snapshot.get("resources") or {}).get(bucket) or {}).get("remaining") or 0))
    except (TypeError, ValueError):
        return 0


def main() -> int:
    token = _load_github_token()
    max_commits = 0
    if len(sys.argv) >= 2:
        try:
            max_commits = int(sys.argv[1])
        except ValueError as e:
            raise RuntimeError(f"Invalid max_commits arg: {sys.argv[1]!r}") from e

    # 1) before (EXACTLY 1 API call)
    before_snap = get_rate_limit_snapshot_via_gh(token=token)
    before_core = _get_remaining(before_snap, bucket="core")
    before_gql = _get_remaining(before_snap, bucket="graphql")
    print(f"BEFORE core.remaining={before_core}")
    print(f"BEFORE graphql.remaining={before_gql}")

    # 2) work (no internal /rate_limit calls)
    env = dict(os.environ)
    env["DYNAMO_UTILS_DISABLE_INTERNAL_RATE_LIMIT_CHECKS"] = "1"
    # Force the Python REST client in the subprocess to use the same token as `gh`,
    # so core.remaining deltas reflect *all* GitHub API calls (REST + gh).
    env["GH_TOKEN"] = token
    proc = subprocess.run(
        [
            "python3",
            "html_pages/show_commit_history.py",
            "--repo-path",
            "/home/keivenc/dynamo/dynamo_ci",
            "--output",
            f"/tmp/quota_commits{max_commits}.html",
            "--max-commits",
            str(max_commits),
            "--skip-gitlab-api",
            "--debug",
            "--github-token",
            token,
        ],
        cwd="/home/keivenc/dynamo/dynamo-utils.dev",
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print("show_commit_history.py failed:", proc.returncode)
        print(proc.stderr)
        return proc.returncode

    # 3) after (EXACTLY 1 API call)
    after_snap = get_rate_limit_snapshot_via_gh(token=token)
    after_core = _get_remaining(after_snap, bucket="core")
    after_gql = _get_remaining(after_snap, bucket="graphql")
    print(f"AFTER  core.remaining={after_core}")
    print(f"AFTER  graphql.remaining={after_gql}")
    delta_core_raw = before_core - after_core
    delta_gql = before_gql - after_gql
    # Note: the AFTER snapshot call itself consumes 1 core call; adjust to estimate subprocess-only usage.
    delta_core_subproc = max(0, delta_core_raw - 1)
    print(f"DELTA  core(raw_includes_after_rate_limit_call)={delta_core_raw}")
    print(f"DELTA  core(subprocess_only_estimate)={delta_core_subproc}")
    print(f"DELTA  graphql={delta_gql}")

    # Print subprocess debug stderr (for auditing call order)
    if proc.stderr:
        print("\n=== show_commit_history.py stderr ===")
        print(proc.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

