#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Common GitHub workflow parser helpers (derived from .github/workflows/*.yml).

Canonical implementation used by the dashboards to:
- Parse workflow job metadata (job_id, display name, needs)
- Build matchers to map GitHub check names back to workflow job_ids

We prefer PyYAML when available, but intentionally keep a fallback parser
for environments where PyYAML isn't installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


@dataclass(frozen=True)
class WorkflowJobSpec:
    job_id: str
    display_name: str
    name_template: str
    needs: Tuple[str, ...]


class GitHubWorkflowParser:
    """Parser and cache for workflow job metadata."""

    _WORKFLOW_SPECS_CACHE: ClassVar[Dict[str, Dict[str, WorkflowJobSpec]]] = {}

    @staticmethod
    def parse_workflow_jobs(workflow_path: Path) -> Dict[str, WorkflowJobSpec]:
        """Parse minimal job metadata from a workflow YAML (job_id, name, needs)."""
        if yaml is not None:
            jobs = GitHubWorkflowParser._parse_workflow_jobs_via_yaml(workflow_path)
            if jobs:
                return jobs
        return GitHubWorkflowParser._parse_workflow_jobs_via_lines(workflow_path)

    @staticmethod
    def load_workflow_specs(repo_path: Path) -> Dict[str, WorkflowJobSpec]:
        """Load workflow job specs from a repo checkout (cached per repo path)."""
        key = str(repo_path.resolve())
        if key in GitHubWorkflowParser._WORKFLOW_SPECS_CACHE:
            return GitHubWorkflowParser._WORKFLOW_SPECS_CACHE[key]

        specs: Dict[str, WorkflowJobSpec] = {}
        wf_dir = repo_path / ".github" / "workflows"
        if wf_dir.exists():
            for wf in sorted(wf_dir.glob("*.yml")) + sorted(wf_dir.glob("*.yaml")):
                for job_id, spec in GitHubWorkflowParser.parse_workflow_jobs(wf).items():
                    if job_id not in specs:
                        specs[job_id] = spec
                    else:
                        prev = specs[job_id]
                        if prev.display_name == prev.job_id and spec.display_name != spec.job_id:
                            specs[job_id] = spec

        GitHubWorkflowParser._WORKFLOW_SPECS_CACHE[key] = specs
        return specs

    @staticmethod
    def build_check_name_matchers(specs: Dict[str, WorkflowJobSpec]) -> List[Tuple[str, re.Pattern]]:
        """Return (job_id, regex) pairs that match GitHub check names back to workflow job_ids."""
        out: List[Tuple[str, re.Pattern]] = []
        for job_id, spec in specs.items():
            out.append((job_id, re.compile(r"^" + re.escape(job_id) + r"$")))
            if spec.display_name and "${{" not in spec.display_name:
                out.append((job_id, re.compile(r"^" + re.escape(spec.display_name) + r"$")))
            rx = GitHubWorkflowParser._template_to_regex(spec.name_template)
            if rx is not None:
                out.append((job_id, rx))
        return out

    # -------------------------
    # Implementation details
    # -------------------------

    @staticmethod
    def _parse_workflow_jobs_via_yaml(workflow_path: Path) -> Dict[str, WorkflowJobSpec]:
        try:
            text = workflow_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return {}

        try:
            data = yaml.safe_load(text) if yaml is not None else None
        except Exception:
            return {}

        if not isinstance(data, dict):
            return {}
        jobs_obj = data.get("jobs")
        if not isinstance(jobs_obj, dict):
            return {}

        out: Dict[str, WorkflowJobSpec] = {}
        for job_id, job in jobs_obj.items():
            if not isinstance(job_id, str) or not job_id:
                continue
            if not isinstance(job, dict):
                continue

            name = job.get("name")
            display_name = name.strip() if isinstance(name, str) and name.strip() else job_id
            name_template = display_name if "${{" in display_name else ""

            needs_val = job.get("needs")
            needs_list: List[str] = []
            if isinstance(needs_val, str) and needs_val.strip():
                needs_list = [needs_val.strip()]
            elif isinstance(needs_val, list):
                for item in needs_val:
                    if isinstance(item, str) and item.strip():
                        needs_list.append(item.strip())

            out[job_id] = WorkflowJobSpec(
                job_id=job_id,
                display_name=display_name,
                name_template=name_template,
                needs=tuple(needs_list),
            )

        return out

    @staticmethod
    def _parse_workflow_jobs_via_lines(workflow_path: Path) -> Dict[str, WorkflowJobSpec]:
        """Fallback parser (no PyYAML): parse only the minimal subset we need."""
        try:
            lines = workflow_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return {}

        in_jobs = False
        cur_job: Optional[str] = None
        cur_name: str = ""
        cur_needs: List[str] = []
        reading_needs_list = False

        jobs: Dict[str, WorkflowJobSpec] = {}

        def flush() -> None:
            nonlocal cur_job, cur_name, cur_needs, reading_needs_list
            if not cur_job:
                return
            name_clean = cur_name.strip().strip('"').strip("'")
            jobs[cur_job] = WorkflowJobSpec(
                job_id=cur_job,
                display_name=(name_clean if name_clean else cur_job),
                name_template=(name_clean if ("${{" in name_clean) else ""),
                needs=tuple(cur_needs),
            )
            cur_job = None
            cur_name = ""
            cur_needs = []
            reading_needs_list = False

        for line in lines:
            if not in_jobs:
                if line.strip() == "jobs:":
                    in_jobs = True
                continue

            m_job = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", line)
            if m_job:
                flush()
                cur_job = m_job.group(1)
                continue

            if not cur_job:
                continue

            m_name = re.match(r"^    name:\s*(.+?)\s*$", line)
            if m_name:
                cur_name = m_name.group(1)
                continue

            m_needs_inline = re.match(r"^    needs:\s*\[(.*?)\]\s*$", line)
            if m_needs_inline:
                reading_needs_list = False
                inner = m_needs_inline.group(1).strip()
                cur_needs = [p.strip() for p in inner.split(",") if p.strip()] if inner else []
                continue

            if re.match(r"^    needs:\s*$", line):
                reading_needs_list = True
                cur_needs = []
                continue

            if reading_needs_list:
                m_item = re.match(r"^      -\s*([A-Za-z0-9_-]+)\s*$", line)
                if m_item:
                    cur_needs.append(m_item.group(1))
                    continue
                if line.startswith("    ") and not line.startswith("      -"):
                    reading_needs_list = False

        flush()
        return jobs

    @staticmethod
    def _template_to_regex(name_template: str) -> Optional[re.Pattern]:
        if not name_template or "${{" not in name_template:
            return None
        parts: List[str] = []
        idx = 0
        for m in re.finditer(r"\$\{\{.*?\}\}", name_template):
            parts.append(re.escape(name_template[idx : m.start()]))
            parts.append(r"(.+?)")
            idx = m.end()
        parts.append(re.escape(name_template[idx:]))
        return re.compile("^" + "".join(parts) + "$")


