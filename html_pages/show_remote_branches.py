#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generate a "remote" branches/PR dashboard for a GitHub user.

Goal: same look & feel as `show_local_branches.py`, but the entrypoint is:
  GitHub username -> open PRs -> same PR/CI subtree rendering.

This intentionally reuses:
- `PRNode`, `PRURLNode`, `PRStatusNode`, `SectionNode`, `_build_ci_hierarchy_nodes`, `generate_html`
  from `show_local_branches.py`
- `GitHubAPIClient` from `common.py`
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

# Ensure we can import sibling utilities (common.py) from the parent dynamo-utils directory
_THIS_DIR = Path(__file__).resolve().parent
_UTILS_DIR = _THIS_DIR.parent
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from common import GitHubAPIClient  # noqa: E402
from html_pages.common_dashboard_runtime import prune_dashboard_raw_logs, prune_partial_raw_log_caches  # noqa: E402
from show_local_branches import (  # noqa: E402
    DYNAMO_OWNER,
    DYNAMO_REPO,
    DYNAMO_REPO_SLUG,
    BranchNode,
    BranchInfoNode,
    CommitMessageNode,
    MetadataNode,
    PRNode,
    PRStatusNode,
    PRURLNode,
    SectionNode,
    _build_ci_hierarchy_nodes,
    _format_age_compact,
    _format_branch_metadata_suffix,
    _format_base_branch_inline,
    _pr_needs_attention,
    _strip_repo_prefix_for_clipboard,
    generate_html,
)


class RemoteBranchInfoNode(BranchInfoNode):
    """Branch line for remote PRs (inherits new structure from BranchInfoNode).

    Structure (same as local):
    - Branch line: copy + label + → base + [SHA]
    - Child 1: Commit message (PR#)
    - Child 2: (modified, created, age)
    - Child 3+: PRStatusNode
    """
    # No override needed - BranchInfoNode.to_tree_vm() already implements the correct structure.


class RemotePRStatusNode(PRStatusNode):
    """Remote PR status node (inherits collapse logic from PRStatusNode).
    
    Policy: collapsed for PASSED, expanded for FAILED/RUNNING (same as local).
    """
    # No override needed - PRStatusNode.to_tree_vm() already implements the correct logic.


def _looks_like_git_repo_dir(p: Path) -> bool:
    try:
        if not p.is_dir():
            return False
        git_marker = p / ".git"
        return git_marker.is_dir() or git_marker.is_file()
    except Exception:
        return False


def _gitdir_from_git_file(p: Path) -> Optional[Path]:
    """Handle worktrees where .git is a file containing 'gitdir: <path>'."""
    try:
        txt = (p / ".git").read_text(encoding="utf-8", errors="ignore").strip()
        if not txt.startswith("gitdir:"):
            return None
        rest = txt.split("gitdir:", 1)[1].strip()
        if not rest:
            return None
        gd = Path(rest)
        if not gd.is_absolute():
            gd = (p / gd).resolve()
        return gd
    except Exception:
        return None


def _origin_url_from_git_config(repo_dir: Path) -> str:
    """Best-effort parse of origin URL from .git/config without GitPython."""
    try:
        repo_dir = Path(repo_dir)
        git_path = repo_dir / ".git"
        config_path: Optional[Path] = None
        if git_path.is_dir():
            config_path = git_path / "config"
        elif git_path.is_file():
            gd = _gitdir_from_git_file(repo_dir)
            if gd is not None:
                config_path = gd / "config"
        if config_path is None or (not config_path.exists()):
            return ""
        lines = config_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        in_origin = False
        for ln in lines:
            s = ln.strip()
            if s.startswith("["):
                in_origin = (s.lower() == '[remote "origin"]')
                continue
            if not in_origin:
                continue
            if s.lower().startswith("url"):
                try:
                    _k, v = s.split("=", 1)
                    return (v or "").strip()
                except Exception:
                    continue
        return ""
    except Exception:
        return ""


def _find_local_clone_of_repo(base_dir: Path, *, repo_slug: str) -> Optional[Path]:
    """Find a local git clone whose origin URL mentions `repo_slug` (e.g. 'ai-dynamo/dynamo')."""
    base_dir = Path(base_dir)
    candidates = []
    try:
        if _looks_like_git_repo_dir(base_dir):
            candidates.append(base_dir)
    except Exception:
        pass
    try:
        for d in base_dir.iterdir():
            try:
                if d.is_dir() and _looks_like_git_repo_dir(d):
                    candidates.append(d)
            except Exception:
                continue
    except Exception:
        pass

    repo_slug_lc = str(repo_slug or "").strip().lower()
    for d in candidates:
        try:
            url = _origin_url_from_git_config(d).lower()
            if repo_slug_lc and repo_slug_lc in url:
                return d
        except Exception:
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Show remote PRs for a GitHub user (HTML-only)")
    parser.add_argument("--github-user", required=True, help="GitHub username (author of PRs)")
    parser.add_argument(
        "--sort",
        choices=["latest", "branch"],
        default="latest",
        help="Sort order: latest (default) or branch (by head branch name)",
    )
    parser.add_argument("--owner", default=DYNAMO_OWNER, help=f"GitHub owner/org (default: {DYNAMO_OWNER})")
    parser.add_argument("--repo", default=DYNAMO_REPO, help=f"GitHub repo (default: {DYNAMO_REPO})")
    parser.add_argument("--repo-root", type=Path, default=None, help="Path to a local clone of the repo (for workflow YAML inference)")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd(), help="Directory to search for a local clone (default: cwd)")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path (default: <base-dir>/remote_prs_<user>.html)")
    parser.add_argument("--token", help="GitHub personal access token (or login with gh so ~/.config/gh/hosts.yml exists)")
    parser.add_argument("--allow-anonymous-github", action="store_true", help="Allow anonymous GitHub REST calls (60/hr core rate limit)")
    parser.add_argument("--max-github-api-calls", type=int, default=100, help="Hard cap on GitHub REST API network calls per invocation")
    parser.add_argument("--max-prs", type=int, default=50, help="Cap PRs shown (default: 50)")
    parser.add_argument("--refresh-checks", action="store_true", help="Force-refresh checks cache TTLs (more GitHub calls)")
    args = parser.parse_args()

    user = str(args.github_user or "").strip()
    owner = str(args.owner or "").strip()
    repo = str(args.repo or "").strip()
    base_dir = Path(args.base_dir).resolve()

    output = args.output
    if output is None:
        safe_user = "".join([c for c in user if c.isalnum() or c in {"-", "_"}]) or "user"
        output = base_dir / f"remote_prs_{safe_user}.html"
    output = Path(output).resolve()

    # Determine local repo root for workflow YAML inference (best-effort).
    repo_root = Path(args.repo_root).resolve() if args.repo_root is not None else None
    if repo_root is None:
        repo_root = _find_local_clone_of_repo(base_dir, repo_slug=f"{owner}/{repo}")
    if repo_root is None:
        # Degrade gracefully: YAML inference will be unavailable, but PR/CI rows still render from APIs.
        repo_root = base_dir

    page_root_dir = output.parent

    # Prune locally-served raw logs to avoid unbounded growth and delete any partial/unverified artifacts.
    try:
        _ = prune_dashboard_raw_logs(page_root_dir=page_root_dir, max_age_days=30)
        _ = prune_partial_raw_log_caches(page_root_dirs=[page_root_dir])
    except Exception:
        pass

    gh = GitHubAPIClient(
        token=args.token,
        require_auth=(not bool(args.allow_anonymous_github)),
        allow_anonymous_fallback=bool(args.allow_anonymous_github),
        max_rest_calls=int(args.max_github_api_calls),
    )

    # Cache-only fallback if exhausted.
    cache_only_reason = ""
    try:
        gh.check_core_rate_limit_or_raise()
    except Exception as e:
        cache_only_reason = str(e)
        try:
            gh.set_cache_only_mode(True)
        except Exception:
            pass

    t0 = time.monotonic()
    prs = gh.get_open_pr_info_for_author(owner, repo, author=user, max_prs=int(args.max_prs))

    root = BranchNode(label="")

    allow_fetch_checks = not bool(getattr(gh, "cache_only_mode", False))

    def _branch_display_for_pr(pr) -> str:
        try:
            head_owner = str(getattr(pr, "head_owner", "") or "").strip()
        except Exception:
            head_owner = ""
        try:
            head_ref = str(getattr(pr, "head_ref", "") or "").strip()
        except Exception:
            head_ref = ""
        try:
            head_label = str(getattr(pr, "head_label", "") or "").strip()
        except Exception:
            head_label = ""
        if head_owner and head_ref:
            return f"{head_owner}/{head_ref}"
        if head_label:
            return head_label.replace(":", "/", 1)
        if head_ref:
            return head_ref
        return f"pr#{getattr(pr, 'number', '')}"

    def _dt_from_iso(s: str) -> Optional[datetime]:
        try:
            dt = datetime.fromisoformat(str(s or "").replace("Z", "+00:00"))
            if getattr(dt, "tzinfo", None) is None:
                dt = dt.replace(tzinfo=ZoneInfo("UTC"))
            return dt
        except Exception:
            return None

    def _pt_str(dt: Optional[datetime]) -> Optional[str]:
        try:
            if dt is None:
                return None
            return dt.astimezone(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M PT")
        except Exception:
            return None

    def build_root(prs_ordered: List[object]) -> BranchNode:
        r = BranchNode(label="")
        for pr in (prs_ordered or []):
            branch_name = _branch_display_for_pr(pr)
            sha7 = ""
            try:
                sha7 = str(getattr(pr, "head_sha", "") or "").strip()[:7]
            except Exception:
                sha7 = ""
            created_dt = _dt_from_iso(str(getattr(pr, "created_at", "") or ""))
            updated_dt = _dt_from_iso(str(getattr(pr, "updated_at", "") or "")) or created_dt
            commit_time_pt = _pt_str(updated_dt)
            commit_url = ""
            try:
                head_sha_full = str(getattr(pr, "head_sha", "") or "").strip()
                if head_sha_full:
                    commit_url = f"https://github.com/{owner}/{repo}/commit/{head_sha_full}"
            except Exception:
                commit_url = ""

            # Use PR title as commit message (since we don't have local git access for remote).
            commit_msg = str(getattr(pr, "title", "") or "").strip()

            branch_node = RemoteBranchInfoNode(
                label=branch_name,
                sha=sha7 or None,
                is_current=False,
                commit_url=commit_url or None,
                commit_time_pt=commit_time_pt,
                commit_datetime=updated_dt,
                commit_message=commit_msg,
            )
            r.add_child(branch_node)

            pr_node = PRNode(label="", pr=pr)
            branch_node.add_child(pr_node)

            status_node = RemotePRStatusNode(
                label="",
                pr=pr,
                github_api=gh,
                refresh_checks=bool(args.refresh_checks),
                branch_commit_dt=updated_dt,
                allow_fetch_checks=bool(allow_fetch_checks),
                context_key=f"remote:{owner}/{repo}:{branch_name}:{sha7}:pr{getattr(pr, 'number', '')}",
            )
            branch_node.add_child(status_node)  # Add directly to branch_node, not pr_node

            # CI hierarchy as children of the PR status line.
            try:
                for ci_node in _build_ci_hierarchy_nodes(
                    Path(repo_root),
                    pr,
                    github_api=gh,
                    page_root_dir=page_root_dir,
                    checks_ttl_s=int(GitHubAPIClient.compute_checks_cache_ttl_s(None, refresh=bool(args.refresh_checks))),
                    skip_fetch=(not bool(allow_fetch_checks)),
                ):
                    try:
                        if hasattr(ci_node, "context_key"):
                            setattr(ci_node, "context_key", str(status_node.context_key or ""))
                    except Exception:
                        pass
                    status_node.add_child(ci_node)
            except Exception:
                pass

            # Conflict/blocking messages (same policy as local page).
            try:
                msg = getattr(pr, "conflict_message", None)
                if msg:
                    pr_node.add_child(BranchNode(label=str(msg)))
            except Exception:
                pass
            try:
                msg = getattr(pr, "blocking_message", None)
                if msg:
                    pr_node.add_child(BranchNode(label=str(msg)))
            except Exception:
                pass
        return r

    prs_list = list(prs or [])
    prs_latest = prs_list
    prs_branch = prs_list
    try:
        prs_branch = sorted(prs_list, key=lambda p: _branch_display_for_pr(p).lower())
    except Exception:
        prs_branch = list(prs_list)
    try:
        prs_latest = sorted(
            prs_list,
            key=lambda p: (
                (_dt_from_iso(getattr(p, "updated_at", None) or "") or _dt_from_iso(getattr(p, "created_at", None) or "") or datetime.min.replace(tzinfo=ZoneInfo("UTC"))),
                int(getattr(p, "number", 0) or 0),
            ),
            reverse=True,
        )
    except Exception:
        prs_latest = list(prs_list)

    # Tree HTML for both sort orders (JS toggles which one is shown; URL param 'sort' persists it).
    from common_dashboard_lib import render_tree_pre_lines  # local import

    root_latest = build_root(prs_latest)
    root_branch = build_root(prs_branch)

    def _tree_html_for_root(rr: BranchNode) -> str:
        lines: List[str] = []
        for i, child in enumerate(rr.children):
            is_last = i == len(rr.children) - 1
            lines.extend(render_tree_pre_lines([child.to_tree_vm()]))
            if not is_last:
                lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    tree_html_latest = _tree_html_for_root(root_latest) if prs_latest else "(no PRs)\n"
    tree_html_branch = _tree_html_for_root(root_branch) if prs_branch else "(no PRs)\n"

    if not prs:
        root.add_child(BranchNode(label=f"(no open PRs found for {user} in {owner}/{repo})"))

    elapsed_s = max(0.0, time.monotonic() - t0)
    page_stats = [
        ("Generation time", f"{elapsed_s:.2f}s"),
        ("Repo", f"{owner}/{repo}"),
        ("GitHub user", user),
        ("PRs shown", str(len(prs))),
    ]
    try:
        from common_dashboard_lib import github_api_stats_rows  # local import

        mode = "cache-only" if bool(getattr(gh, "cache_only_mode", False)) else "normal"
        page_stats.extend(
            list(
                github_api_stats_rows(
                    github_api=gh,
                    max_github_api_calls=int(args.max_github_api_calls),
                    mode=mode,
                    mode_reason=cache_only_reason or "",
                    top_n=15,
                )
                or []
            )
        )
    except Exception:
        pass

    # If the CLI requested a particular sort, reflect it as the default (JS still allows switching).
    tree_sort_default = str(args.sort or "latest").strip().lower()
    html = generate_html(
        root_latest,
        page_stats=page_stats,
        page_title=f"Remote PR Info ({user})",
        header_title=f"Remote PR Info ({user})",
        tree_html_override=tree_html_latest,
        tree_html_alt=tree_html_branch,
        tree_sort_default=tree_sort_default,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


