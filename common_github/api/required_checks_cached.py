# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Required-checks cached API.

This resource is intentionally implemented via `gh` because:
- branch protection REST endpoints can 403 without admin perms
- required-ness lives in GraphQL (merge-box semantics)

Resource (current implementation):
  1) gh api repos/{owner}/{repo}/pulls/{pr_number} --jq <expr>
     - used to fetch PR `node_id` + state + updated_at/merged_at (for TTL meta)
  2) gh api graphql -f query=<omitted> ... (paginated)
     - used to compute required-ness via `isRequired(pullRequestId: ...)`

Do NOT use:
  - `gh pr view {pr_number} --repo {owner}/{repo} --json statusCheckRollup`
    Empirically this can return an empty required set for `isRequired` even when the
    equivalent GraphQL query (same as `_fetch_required_checks()` below) returns
    required checks (e.g. PR 5635 in ai-dynamo/dynamo). Prefer GraphQL.

Implementation note (how to migrate off `gh`, later; do NOT change behavior now):
  TODO: Replace `gh` subprocess calls with direct REST+GraphQL HTTP (Python/requests),
  keeping the same semantics and stats visibility.

  This module can be implemented via direct GitHub GraphQL HTTP instead of the `gh` CLI.
  The key requirement is the PR node id (`node_id`) because GraphQL uses
  `isRequired(pullRequestId: $prid)` to compute required-ness per PR.

  Outline (Python + requests):
    1) REST: GET https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}
       - Extract `node_id` as `$prid` (and optionally `updated_at` / `merged_at` for TTL meta).
    2) GraphQL: POST https://api.github.com/graphql
       - Body: {"query": <QUERY>, "variables": {"owner":..., "name":..., "number":..., "prid":..., "after":...}}
       - Use the same query template as `_fetch_required_checks()` below; page with `after` until
         `pageInfo.hasNextPage == false`.
    3) Collect required check names where `isRequired == true`, using:
       - CheckRun: `name`
       - StatusContext: `context`

Example API Response (truncated; key fields used by this module):
  - REST (gh core) PR meta:
      {"node_id":"PR_kwDO...","state":"OPEN","merged_at":null,"updated_at":"2026-01-24T03:12:34Z"}
  - GraphQL required-ness (gh graphql):
      {
        "data": {
          "repository": {
            "pullRequest": {
              "commits": {
                "nodes": [
                  {
                    "commit": {
                      "statusCheckRollup": {
                        "contexts": {
                          "nodes": [
                            {"__typename":"CheckRun","name":"pre-commit","isRequired":true},
                            {"__typename":"StatusContext","context":"license/cla","isRequired":false}
                          ]
                        }
                      }
                    }
                  }
                ]
              }
            }
          }
        }
      }

Cached Fields:
  - Set of required check names/contexts (e.g., {"pre-commit"})
  - Extracted from GraphQL nodes where `isRequired == true`
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple, TYPE_CHECKING

from cache.cache_base import BaseDiskCache
from .. import GITHUB_API_STATS
from .base_cached import CachedResourceBase

if TYPE_CHECKING:  # pragma: no cover
    from .. import GitHubAPIClient


# Keep TTL documentation next to the actual TTL implementation (`is_cache_entry_fresh`).
TTL_POLICY_DESCRIPTION = (
    "adaptive (<1h=2m, <2h=4m, <4h=30m, <8h=60m, <12h=80m, >=12h=120m); closed/merged: 360d"
)

CACHE_NAME = "required_checks"
API_CALL_FORMAT = (
    "gh api repos/{owner}/{repo}/pulls/{pr} --jq <expr> (core)\n"
    "Example output (truncated):\n"
    "  {\"node_id\":\"PR_kwDO...\",\"state\":\"OPEN\",\"merged_at\":null,\"updated_at\":\"2026-01-24T03:12:34Z\"}\n"
    "\n"
    "gh api graphql -f query=<omitted> ... (graphql; paginated)\n"
    "Example output fields used (truncated):\n"
    "  {\n"
    "    \"data\": {\n"
    "      \"repository\": {\n"
    "        \"pullRequest\": {\n"
    "          \"commits\": {\n"
    "            \"nodes\": [\n"
    "              {\"commit\": {\"statusCheckRollup\": {\"contexts\": {\"nodes\": [{\"__typename\":\"CheckRun\",\"name\":\"pre-commit\",\"isRequired\":true}]}}}}\n"
    "            ]\n"
    "          }\n"
    "        }\n"
    "      },\n"
    "      \"rateLimit\": {\"cost\": 1, \"remaining\": 4999, \"resetAt\": \"2026-01-24T04:00:00Z\"}\n"
    "    }\n"
    "  }"
)
CACHE_KEY_FORMAT = "{cache_name}:{owner}/{repo}:pr{pr_number}"
CACHE_FILE_DEFAULT = "required_checks.json"


# =============================================================================
# Cache Implementation (private to this module)
# =============================================================================

class _RequiredChecksCache(BaseDiskCache):
    """Cache for required status checks from branch protection."""
    
    _SCHEMA_VERSION = 1
    
    def __init__(self, *, cache_file: Path):
        super().__init__(cache_file=cache_file, schema_version=self._SCHEMA_VERSION)
    
    def get_if_valid(self, key: str, *, cache_only_mode: bool = False, check_ttl: bool = True) -> Optional[Dict[str, Any]]:
        """Get cached required checks if valid."""
        with self._mu:
            self._load_once()
            ent = self._check_item(key)
            
            if not isinstance(ent, dict):
                return None
            
            # Extract value (stored as list on disk, return as set)
            val = ent.get("val")
            if isinstance(val, list):
                val_set = set(val)
            elif isinstance(val, set):
                val_set = val
            else:
                return None
            
            # In cache_only_mode, always return cached value
            if cache_only_mode or not check_ttl:
                return {
                    "val": val_set,
                    "ok": ent.get("ok", True),
                    "pr_state": ent.get("pr_state", "open"),
                    "pr_updated_at_epoch": ent.get("pr_updated_at_epoch"),
                }
            
            # Otherwise, check if entry is still fresh
            ts = int(ent.get("ts", 0) or 0)
            if ts:
                return {
                    "val": val_set,
                    "ok": ent.get("ok", True),
                    "pr_state": ent.get("pr_state", "open"),
                    "pr_updated_at_epoch": ent.get("pr_updated_at_epoch"),
                    "ts": ts,
                }
            
            return None
    
    def put(
        self,
        key: str,
        val: Set[str],
        *,
        ok: bool = True,
        pr_state: str = "open",
        pr_updated_at_epoch: Optional[int] = None,
    ) -> None:
        """Store required checks."""
        now = int(time.time())
        with self._mu:
            self._load_once()
            entry = {
                "ts": now,
                "val": list(val) if isinstance(val, set) else val,
                "ok": ok,
                "pr_state": pr_state,
                "pr_updated_at_epoch": pr_updated_at_epoch,
            }
            self._set_item(key, entry)
            self._persist()


def _get_cache_file() -> Path:
    """Get cache file path."""
    try:
        _module_dir = Path(__file__).resolve().parent.parent
        if str(_module_dir) not in sys.path:
            sys.path.insert(0, str(_module_dir))
        import common
        return common.dynamo_utils_cache_dir() / CACHE_FILE_DEFAULT
    except ImportError:
        return Path.home() / ".cache" / "dynamo-utils" / CACHE_FILE_DEFAULT


# Module-level singleton cache instance (private)
_CACHE = _RequiredChecksCache(cache_file=_get_cache_file())


# =============================================================================
# Public API
# =============================================================================


class RequiredChecksCached(CachedResourceBase[Set[str]]):
    @property
    def cache_name(self) -> str:
        return CACHE_NAME

    def api_call_format(self) -> str:
        return API_CALL_FORMAT

    def cache_key(self, **kwargs: Any) -> str:
        owner = str(kwargs["owner"])
        repo = str(kwargs["repo"])
        prn = int(kwargs["pr_number"])
        return f"{self.cache_name}:{owner}/{repo}:pr{prn}"

    def empty_value(self) -> Set[str]:
        return set()

    def cache_read(self, *, key: str) -> Optional[Dict[str, Any]]:
        return _CACHE.get_if_valid(key, cache_only_mode=self.api.cache_only_mode, check_ttl=True)

    def value_from_cache_entry(self, *, entry: Dict[str, Any]) -> Set[str]:
        val = entry.get("val", set())
        if isinstance(val, set):
            return set(val)
        if isinstance(val, list):
            return {str(x) for x in val if str(x).strip()}
        return set()

    def is_cache_entry_fresh(self, *, entry: Dict[str, Any], now: int) -> bool:
        try:
            ts = int(entry.get("ts", 0) or 0)
        except (ValueError, TypeError):
            ts = 0
        if not ts:
            return False

        ok = entry.get("ok", True)
        pr_state = entry.get("pr_state", "open")
        is_merged_or_closed = pr_state in ("closed", "merged")

        pr_updated_at_epoch = entry.get("pr_updated_at_epoch", None)
        try:
            pr_updated_at_epoch_i = int(pr_updated_at_epoch) if pr_updated_at_epoch is not None else 0
        except (ValueError, TypeError):
            pr_updated_at_epoch_i = 0

        if not pr_updated_at_epoch_i and not is_merged_or_closed:
            if ok is False:
                return (int(now) - int(ts)) <= 60
            return False

        if is_merged_or_closed:
            ttl_s = 360 * 24 * 3600
        else:
            ttl_s = self.api._adaptive_ttl_s(timestamp_epoch=pr_updated_at_epoch_i, default_ttl_s=180)

        if ok is False:
            ttl_s = min(int(ttl_s), 60)

        return (int(now) - int(ts)) <= int(ttl_s)

    def cache_write(self, *, key: str, value: Set[str], meta: Dict[str, Any]) -> None:
        _CACHE.put(
            key,
            set(value),
            ok=bool(meta.get("ok", True)),
            pr_state=str(meta.get("pr_state", "open")),
            pr_updated_at_epoch=meta.get("pr_updated_at_epoch", None),
        )

    def fetch(self, **kwargs: Any) -> Tuple[Set[str], Dict[str, Any]]:
        owner = str(kwargs["owner"])
        repo = str(kwargs["repo"])
        prn = int(kwargs["pr_number"])

        if self.api._debug_rest:
            stack = "".join(traceback.format_stack(limit=12))
            print(f"[REQUIRED_CHECKS_CALL] owner={owner} repo={repo} pr={prn}\n{stack}", file=sys.stderr, flush=True)

        pr_node_id, pr_state, pr_updated_at_epoch = self._fetch_pr_meta(owner=owner, repo=repo, prn=prn)
        if not pr_node_id:
            return set(), {"ok": False, "pr_state": pr_state, "pr_updated_at_epoch": pr_updated_at_epoch}

        required = self._fetch_required_checks(owner=owner, repo=repo, prn=prn, pr_node_id=pr_node_id)
        return required, {"ok": True, "pr_state": pr_state, "pr_updated_at_epoch": pr_updated_at_epoch}

    def _fetch_pr_meta(self, *, owner: str, repo: str, prn: int) -> Tuple[str, str, Optional[int]]:
        pr_node_id = ""
        pr_state = "open"
        pr_updated_at_epoch: Optional[int] = None

        cmd0 = [
            "gh",
            "api",
            f"repos/{owner}/{repo}/pulls/{prn}",
            "--jq",
            "{node_id: .node_id, state: .state, merged_at: .merged_at, updated_at: .updated_at}",
        ]
        GITHUB_API_STATS.gh_core_calls_total = int(getattr(GITHUB_API_STATS, "gh_core_calls_total", 0) or 0) + 1
        GITHUB_API_STATS.log_actual_api_call(kind="gh.core", text=f"gh api repos/{owner}/{repo}/pulls/{prn} --jq <expr>")

        if self.api._debug_rest:
            cmd0 = list(cmd0) + ["--include"]
            print(f"[GH_CLI_CALL] {' '.join(cmd0)}", file=sys.stderr, flush=True)

        try:
            res0 = subprocess.run(cmd0, capture_output=True, text=True, timeout=15, check=False)
        except (OSError, subprocess.SubprocessError):
            return ("", "open", None)

        if res0.returncode != 0:
            return ("", "open", None)

        out0 = str(res0.stdout or "")
        if self.api._debug_rest and "\n\n" in out0:
            hdr0, body0 = out0.split("\n\n", 1)
            status_line = (hdr0.splitlines()[0] if hdr0.splitlines() else "").strip()
            if status_line:
                try:
                    code_i = int(status_line.split()[1])
                except (IndexError, ValueError, TypeError):
                    code_i = 0
                if code_i == 304:
                    GITHUB_API_STATS.gh_core_304_total = int(getattr(GITHUB_API_STATS, "gh_core_304_total", 0) or 0) + 1
            remaining_line = ""
            for ln in hdr0.splitlines():
                if ln.lower().startswith("x-ratelimit-remaining:"):
                    remaining_line = ln.strip()
                    break
            if status_line or remaining_line:
                print(f"[GH_CLI_RESP] {status_line} {remaining_line}".strip(), file=sys.stderr, flush=True)
            out0 = body0

        try:
            meta = json.loads(out0.strip() or "{}") or {}
        except json.JSONDecodeError:
            meta = {}

        pr_node_id = str(meta.get("node_id") or "").strip()
        pr_state = str(meta.get("state") or "open").strip()
        if meta.get("merged_at"):
            pr_state = "merged"
        updated_at_s = str(meta.get("updated_at") or "").strip()
        if updated_at_s:
            try:
                dt = datetime.fromisoformat(updated_at_s.replace("Z", "+00:00"))
                pr_updated_at_epoch = int(dt.timestamp())
            except (ValueError, TypeError):
                pr_updated_at_epoch = None
        return (pr_node_id, pr_state, pr_updated_at_epoch)

    def _fetch_required_checks(self, *, owner: str, repo: str, prn: int, pr_node_id: str) -> Set[str]:
        query_template = """\
query($owner:String!,$name:String!,$number:Int!,$prid:ID!,$after:String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      commits(last: 1) {
        nodes {
          commit {
            statusCheckRollup {
              contexts(first: 100, after: $after) {
                pageInfo {
                  hasNextPage
                  endCursor
                }
                nodes {
                  __typename
                  ... on CheckRun { name isRequired(pullRequestId: $prid) }
                  ... on StatusContext { context isRequired(pullRequestId: $prid) }
                }
              }
            }
          }
        }
      }
    }
  }
  rateLimit { cost remaining resetAt }
}
"""
        all_required: Set[str] = set()
        after_cursor = None
        max_pages = 25

        for page_num in range(max_pages):
            cn = self.cache_name
            cmd = [
                "gh",
                "api",
                "graphql",
                "-f",
                f"query={query_template}",
                "-f",
                f"owner={owner}",
                "-f",
                f"name={repo}",
                "-F",
                f"number={int(prn)}",
                "-f",
                f"prid={pr_node_id}",
            ]
            if after_cursor:
                cmd.extend(["-f", f"after={after_cursor}"])
            else:
                cmd.extend(["-f", "after=null"])

            GITHUB_API_STATS.gh_graphql_calls_total = int(getattr(GITHUB_API_STATS, "gh_graphql_calls_total", 0) or 0) + 1
            cursor_desc = after_cursor if after_cursor else "null"
            cmd_redacted = []
            for a in cmd:
                s = str(a)
                if s.startswith("query="):
                    cmd_redacted.append("query=<omitted>")
                else:
                    cmd_redacted.append(s)
            GITHUB_API_STATS.log_actual_api_call(
                kind="gh.graphql",
                text=f"{' '.join(cmd_redacted)}  # {cn} page={page_num + 1} after={cursor_desc}",
            )

            if self.api._debug_rest:
                cmd_desc = " ".join(cmd)
                print(
                    f"[GH_CLI_CALL] {cmd_desc}  # {cn} page={page_num + 1} after={cursor_desc}",
                    file=sys.stderr,
                    flush=True,
                )

            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
            except (OSError, subprocess.SubprocessError):
                break
            if res.returncode != 0:
                break

            out = str(res.stdout or "")
            if self.api._debug_rest and "\n\n" in out:
                hdr, body = out.split("\n\n", 1)
                status_line = (hdr.splitlines()[0] if hdr.splitlines() else "").strip()
                remaining_line = ""
                for ln in hdr.splitlines():
                    if ln.lower().startswith("x-ratelimit-remaining:"):
                        remaining_line = ln.strip()
                        break
                if status_line or remaining_line:
                    print(f"[GH_CLI_RESP] {status_line} {remaining_line}".strip(), file=sys.stderr, flush=True)
                out = body

            try:
                data = json.loads(out or "{}") or {}
            except json.JSONDecodeError:
                break

            try:
                rl = ((data.get("data") or {}).get("rateLimit") or {})
                cost = int(rl.get("cost") or 0)
                if cost > 0:
                    GITHUB_API_STATS.gh_graphql_cost_total = int(getattr(GITHUB_API_STATS, "gh_graphql_cost_total", 0) or 0) + cost
            except (ValueError, TypeError, AttributeError):
                pass

            nodes = (((((data.get("data") or {}).get("repository") or {}).get("pullRequest") or {}).get("commits") or {}).get("nodes") or [])
            if not (isinstance(nodes, list) and nodes):
                break

            commit0 = nodes[0].get("commit") if isinstance(nodes[0], dict) else None
            scr = commit0.get("statusCheckRollup") if isinstance(commit0, dict) else None
            ctxs = (scr.get("contexts") or {}) if isinstance(scr, dict) else {}

            ctx_nodes = ctxs.get("nodes") if isinstance(ctxs, dict) else None
            if isinstance(ctx_nodes, list):
                for n in ctx_nodes:
                    if not isinstance(n, dict):
                        continue
                    if n.get("isRequired") is not True:
                        continue
                    nm = str(n.get("name") or n.get("context") or "").strip()
                    if nm:
                        all_required.add(nm)

            page_info = ctxs.get("pageInfo") if isinstance(ctxs, dict) else None
            if not isinstance(page_info, dict):
                break
            if not page_info.get("hasNextPage"):
                break
            after_cursor = page_info.get("endCursor")
            if not after_cursor:
                break

        return all_required


def get_required_checks_cached(api: "GitHubAPIClient", *, owner: str, repo: str, pr_number: int) -> Set[str]:
    """Back-compat wrapper."""
    return RequiredChecksCached(api).get(owner=owner, repo=repo, pr_number=pr_number)


def get_cache_sizes() -> Tuple[int, int]:
    """Get cache sizes for stats reporting (memory count, initial disk count)."""
    return _CACHE.get_cache_sizes()

