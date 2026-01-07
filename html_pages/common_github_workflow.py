#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GitHub workflow helpers for the HTML dashboards.

This module is intentionally small and best-effort:
- Parse `.github/workflows/*.yml|*.yaml` for `jobs.*.needs` (dependency graph).
- Optionally group CI check nodes based on those dependencies (e.g. backend-status-check -> matrix jobs).

We do NOT attempt to fully emulate GitHub's runtime job naming rules. Instead, we match check-run names
against workflow job `name:` templates in a conservative way.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Protocol, TypeVar

# YAML is used to parse GitHub workflow graphs (jobs.*.needs) for CI grouping.
try:
    import yaml  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

class SupportsChildren(Protocol):
    """Minimal interface required for workflow grouping."""

    children: List[object]


TNode = TypeVar("TNode", bound=SupportsChildren)


@dataclass
class WorkflowJobDef:
    """A job definition from a single workflow file."""

    job_id: str
    display_name: str
    needs: List[str] = field(default_factory=list)
    # Compiled regex for matching check-run names against this job (best-effort).
    name_regex: Optional[re.Pattern[str]] = None


@dataclass
class WorkflowDef:
    """Parsed subset of a workflow file: just jobs + needs graph."""

    workflow_path: Path
    jobs_by_id: Dict[str, WorkflowJobDef] = field(default_factory=dict)


def _compile_job_name_regex(template: str) -> Optional[re.Pattern[str]]:
    """Convert a workflow job `name:` template into a regex that matches rendered check-run names."""
    s = str(template or "").strip()
    if not s:
        return None
    # Replace GitHub expressions like ${{ matrix.foo }} with a wildcard.
    # Keep it conservative: match within a line and ignore case.
    parts: List[str] = []
    i = 0
    while i < len(s):
        j = s.find("${{", i)
        if j < 0:
            parts.append(re.escape(s[i:]))
            break
        parts.append(re.escape(s[i:j]))
        k = s.find("}}", j + 3)
        if k < 0:
            # Unterminated expression; treat remainder literally.
            parts.append(re.escape(s[j:]))
            break
        # wildcard for expression
        parts.append(r".+?")
        i = k + 2
    pat = "".join(parts)
    # Normalize whitespace: template spaces may expand; match any run of whitespace.
    pat = re.sub(r"\\ +", r"\\s+", pat)
    try:
        return re.compile(r"^" + pat + r"$", re.IGNORECASE)
    except Exception:
        return None


def load_workflow_defs(repo_root: Path) -> List[WorkflowDef]:
    """Parse .github/workflows/*.yml|*.yaml and return workflow defs (best-effort)."""
    if yaml is None:
        return []
    root = Path(repo_root)
    wf_dir = root / ".github" / "workflows"
    if not wf_dir.exists() or not wf_dir.is_dir():
        return []

    # PERF: This function is called once-per-commit when grouping check runs.
    # YAML parsing of workflow files is relatively expensive; cache within a process.
    #
    # Cache key includes the workflows directory mtime so edits invalidate the cache.
    try:
        wf_dir_mtime_ns = int(wf_dir.stat().st_mtime_ns)
    except Exception:
        wf_dir_mtime_ns = 0
    return _load_workflow_defs_cached(str(wf_dir.resolve()), wf_dir_mtime_ns)


@functools.lru_cache(maxsize=8)
def _load_workflow_defs_cached(wf_dir_resolved: str, wf_dir_mtime_ns: int) -> List[WorkflowDef]:
    """Cached implementation for `load_workflow_defs()` (see caller for keying)."""
    if yaml is None:
        return []
    wf_dir = Path(str(wf_dir_resolved or ""))
    if not wf_dir.exists() or not wf_dir.is_dir():
        return []
    out: List[WorkflowDef] = []
    for p in sorted(list(wf_dir.glob("*.yml")) + list(wf_dir.glob("*.yaml"))):
        try:
            raw = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        try:
            data = yaml.safe_load(raw)  # type: ignore[attr-defined]
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        jobs = data.get("jobs")
        if not isinstance(jobs, dict):
            continue

        wf = WorkflowDef(workflow_path=p, jobs_by_id={})
        for job_id, j in jobs.items():
            if not isinstance(job_id, str):
                continue
            if not isinstance(j, dict):
                j = {}
            name = str(j.get("name") or job_id)
            needs_val = j.get("needs")
            needs: List[str] = []
            if isinstance(needs_val, str) and needs_val.strip():
                needs = [needs_val.strip()]
            elif isinstance(needs_val, list):
                needs = [str(x).strip() for x in needs_val if str(x).strip()]
            # Note: needs can be an expression in rare cases; ignore non-str/list.

            wf.jobs_by_id[job_id] = WorkflowJobDef(
                job_id=job_id,
                display_name=name,
                needs=needs,
                name_regex=_compile_job_name_regex(name),
            )
        out.append(wf)

    return out


def match_check_to_workflow_job(
    *,
    check_name: str,
    workflow_defs: List[WorkflowDef],
) -> Optional[Tuple[Path, str]]:
    """Return (workflow_path, job_id) that best matches the check-run name (best-effort)."""
    nm0 = str(check_name or "").strip()
    if not nm0:
        return None

    # Build a small set of name candidates to account for common CI naming variations.
    # Example: GitHub check-runs may be named "Build vllm (amd64)" while workflow `name:` is "vllm (amd64)".
    candidates: List[str] = []
    for s in (nm0,):
        s = str(s or "").strip()
        if not s:
            continue
        candidates.append(s)
        # Strip common verbs/prefixes
        candidates.append(re.sub(r"^(?:build|test|deploy|lint)\s+", "", s, flags=re.IGNORECASE).strip())
        candidates.append(re.sub(r"^(?:build|test|deploy|lint)\s*:\s*", "", s, flags=re.IGNORECASE).strip())
    # Deduplicate while preserving order.
    seen: set[str] = set()
    candidates = [s for s in candidates if s and not (s in seen or seen.add(s))]  # type: ignore[misc]

    # Prefer exact matches (job name or job id), then regex.
    best: Optional[Tuple[int, Path, str]] = None

    def _job_id_prefix_bonus(*, job_id: str, name: str) -> int:
        # Heuristic: if the rendered check name begins with the workflow job_id (e.g. "vllm (amd64)"),
        # prefer mapping to that job_id over more "decorated" jobs like "build-vllm".
        try:
            jid = str(job_id or "").strip().lower()
            nm = str(name or "").strip().lower()
            if not jid or not nm:
                return 0
            if nm.startswith(jid + " ") or nm.startswith(jid + "("):
                return 2000
            return 0
        except Exception:
            return 0

    for wf in workflow_defs:
        for job_id, jd in (wf.jobs_by_id or {}).items():
            for nm in candidates:
                # Exact match against job id or display name
                if nm.lower() == str(job_id).strip().lower() or nm.lower() == str(jd.display_name or "").strip().lower():
                    score = 10_000 + len(str(jd.display_name or "")) + _job_id_prefix_bonus(job_id=str(job_id), name=nm)
                    cand = (score, wf.workflow_path, job_id)
                    if best is None or cand[0] > best[0]:
                        best = cand
                    break
                rx = jd.name_regex
                if rx and rx.match(nm):
                    # Prefer more specific display names.
                    score = 1_000 + len(str(jd.display_name or "")) + _job_id_prefix_bonus(job_id=str(job_id), name=nm)
                    cand = (score, wf.workflow_path, job_id)
                    if best is None or cand[0] > best[0]:
                        best = cand

    if not best:
        return None
    return (best[1], best[2])


def group_ci_nodes_by_workflow_needs(
    *,
    repo_root: Path,
    items: List[Tuple[str, TNode]],
) -> List[TNode]:
    """Group CI nodes into a parent/child hierarchy based on workflow `jobs.*.needs`.

    `items` is a list of (check_name, node). `node` must have a mutable `.children` list attribute.
    This is best-effort and will only group when parent+children can be mapped to the SAME workflow file.
    """
    wfs = load_workflow_defs(repo_root)
    if not wfs:
        return [n for (_nm, n) in (items or [])]

    # Map each node to (workflow_path, job_id). Keep only matches.
    node_meta: List[Tuple[str, TNode, Optional[Tuple[Path, str]]]] = []
    for (nm, node) in (items or []):
        meta = match_check_to_workflow_job(check_name=str(nm or ""), workflow_defs=wfs)
        node_meta.append((str(nm or ""), node, meta))

    # Index nodes by workflow+job_id.
    by_wf_job: Dict[Tuple[str, str], List[TNode]] = {}
    for (_nm, node, meta) in node_meta:
        if not meta:
            continue
        wf_path, job_id = meta
        key = (str(wf_path), str(job_id))
        by_wf_job.setdefault(key, []).append(node)

    wf_by_path: Dict[str, WorkflowDef] = {str(w.workflow_path): w for w in wfs if w and w.workflow_path}

    def _parent_score(job_id: str) -> int:
        """Higher score means 'more appropriate' parent when multiple parents want the same children."""
        s = str(job_id or "").strip().lower()
        if s == "backend-status-check":
            return 100
        if "status-check" in s or s.endswith("-status-check"):
            return 50
        return 0

    # Build best parent assignment per child node, then attach once (prevents "lost" nodes).
    child_best: Dict[int, Tuple[int, TNode]] = {}

    for (_nm, parent_node, meta) in node_meta:
        if not meta:
            continue
        wf_path, parent_job_id = meta
        wf = wf_by_path.get(str(wf_path))
        if not wf:
            continue

        parent_nodes = by_wf_job.get((str(wf_path), str(parent_job_id)), [])
        if not parent_nodes:
            continue
        # Canonical parent instance for this workflow+job_id (avoid duplicating the subtree).
        canonical_parent = parent_nodes[0]
        if canonical_parent is not parent_node:
            continue

        jd = (wf.jobs_by_id or {}).get(str(parent_job_id))
        needs = list(getattr(jd, "needs", None) or [])
        if not needs:
            continue

        p_score = _parent_score(str(parent_job_id))

        for child_job_id in needs:
            for child_node in by_wf_job.get((str(wf_path), str(child_job_id)), []):
                if child_node is canonical_parent:
                    continue
                cid = id(child_node)
                prior = child_best.get(cid)
                if prior is None or int(p_score) > int(prior[0]):
                    child_best[cid] = (int(p_score), canonical_parent)

    # Attach children to chosen parents.
    parent_to_children: Dict[int, List[TNode]] = {}
    for cid, (_score, pnode) in child_best.items():
        try:
            parent_to_children.setdefault(id(pnode), []).append(next(n for (_nm, n, _m) in node_meta if id(n) == cid))
        except Exception:
            # Best-effort: if we can't locate the node instance, skip attaching.
            pass

    for (_nm, pnode, _meta) in node_meta:
        kids = parent_to_children.get(id(pnode))
        if not kids:
            continue
        try:
            existing_ref = getattr(pnode, "children", None)
            existing = list(existing_ref or [])
        except Exception:
            existing_ref = None
            existing = []
        try:
            # Avoid duplicates if a child is already present.
            existing_ids = {id(c) for c in existing}
            add = [c for c in kids if id(c) not in existing_ids]
            # TreeNodeVM is a frozen dataclass, but its `children` is a mutable list.
            # Prefer in-place mutation (works for both frozen and non-frozen nodes).
            if isinstance(existing_ref, list):
                existing_ref.extend(add)
            else:
                pnode.children = existing + add  # type: ignore[attr-defined]
        except Exception:
            pass

    # Emit top-level nodes in original order, skipping those assigned as children.
    out: List[TNode] = []
    for (_nm, node, _meta) in node_meta:
        if id(node) in child_best:
            continue
        out.append(node)
    return out


