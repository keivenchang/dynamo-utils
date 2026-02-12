# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed dataclasses for build report JSON serialization.

This module defines the schema for build report JSON files written alongside
the HTML report by build_images.py and consumed by show_commit_history.py.

File naming convention:
    YYYY-MM-DD.<sha9>.json   (alongside YYYY-MM-DD.<sha9>.report.html)

Usage (producer -- build_images.py):
    report = BuildReport(...)
    report.to_file(log_dir / f"{date_str}.{sha}.json")

Usage (consumer -- show_commit_history.py):
    report = BuildReport.from_file(json_path)
    for fw in report.frameworks:
        for img in fw.registry_images:
            print(img.location, img.push_status)
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Bump when the schema changes in a backward-incompatible way.
SCHEMA_VERSION = 1


@dataclass
class TaskResult:
    """Result of a single build step (build, compilation, or sanity)."""

    status: str  # "PASS", "FAIL", "SKIP", "KILLED", "RUNNING", "QUEUED"
    duration_s: Optional[float] = None
    log_file: Optional[str] = None
    prev_status: Optional[str] = None  # for skipped tasks: previous run's status


@dataclass
class TargetResult:
    """Build results for a single target (runtime, dev, local-dev)."""

    target: str  # "runtime", "dev", "local-dev"
    build: Optional[TaskResult] = None
    compilation: Optional[TaskResult] = None
    sanity: Optional[TaskResult] = None
    image_name: Optional[str] = None  # e.g. "dynamo:v0.8.0.dev.c8ad4aa67-vllm-dev"
    image_size_bytes: Optional[int] = None  # from local docker
    input_image: Optional[str] = None  # base image used


@dataclass
class RegistryImage:
    """A container image pushed to a remote registry."""

    location: str  # full pull URL
    image_name: str  # repo:tag portion, e.g. "dynamo:c8ad4aa67-vllm-dev"
    tag: str  # just the tag, e.g. "c8ad4aa67-vllm-dev"
    framework: str  # "none", "vllm", "sglang", "trtllm"
    target: str  # "dev" (currently only dev images are uploaded)
    push_status: str  # "PASS", "FAIL", "SKIP", "KILLED", "RUNNING", "QUEUED"
    size_bytes: Optional[int] = None  # image size from local docker at push time
    pushed_at: Optional[str] = None  # ISO 8601 timestamp when push completed


@dataclass
class FrameworkResult:
    """Build results for a single framework (none, vllm, sglang, trtllm)."""

    framework: str  # "none", "vllm", "sglang", "trtllm"
    targets: List[TargetResult] = field(default_factory=list)
    registry_images: List[RegistryImage] = field(default_factory=list)


@dataclass
class CommitInfo:
    """Git commit metadata."""

    sha_short: str  # 9-char short SHA
    sha_full: str  # 40-char full SHA
    author: Optional[str] = None  # "Name <email>"
    date: Optional[str] = None  # ISO 8601 or human-readable
    message: Optional[str] = None  # full commit message
    pr_number: Optional[int] = None
    insertions: Optional[int] = None  # lines added
    deletions: Optional[int] = None  # lines removed


@dataclass
class BuildReport:
    """Top-level build report structure.

    Serialized as JSON alongside the HTML report file:
        logs/2026-02-11/2026-02-11.c8ad4aa67.json
    """

    schema_version: int
    sha_short: str  # 9-char
    sha_full: str  # 40-char
    build_date: str  # "YYYY-MM-DD"
    report_generated: str  # ISO 8601 when the report was generated
    overall_status: str  # "PASS", "FAIL", "PARTIAL", "KILLED", "RUNNING"
    total_tasks: int
    succeeded: int
    failed: int
    skipped: int
    killed: int
    commit: CommitInfo
    frameworks: List[FrameworkResult] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict (suitable for json.dumps)."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_file(self, path: Path) -> None:
        """Write JSON to *path*, creating parent directories if needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        logger.debug("BuildReport written to %s", path)

    # ------------------------------------------------------------------
    # Deserialization
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BuildReport":
        """Reconstruct a BuildReport from a plain dict."""
        commit_d = d.get("commit") or {}
        commit = CommitInfo(
            sha_short=commit_d.get("sha_short", ""),
            sha_full=commit_d.get("sha_full", ""),
            author=commit_d.get("author"),
            date=commit_d.get("date"),
            message=commit_d.get("message"),
            pr_number=commit_d.get("pr_number"),
            insertions=commit_d.get("insertions"),
            deletions=commit_d.get("deletions"),
        )

        frameworks: List[FrameworkResult] = []
        for fw_d in d.get("frameworks") or []:
            targets: List[TargetResult] = []
            for tgt_d in fw_d.get("targets") or []:
                targets.append(
                    TargetResult(
                        target=tgt_d.get("target", ""),
                        build=_task_result_from_dict(tgt_d.get("build")),
                        compilation=_task_result_from_dict(tgt_d.get("compilation")),
                        sanity=_task_result_from_dict(tgt_d.get("sanity")),
                        image_name=tgt_d.get("image_name"),
                        image_size_bytes=tgt_d.get("image_size_bytes"),
                        input_image=tgt_d.get("input_image"),
                    )
                )
            registry_images: List[RegistryImage] = []
            for img_d in fw_d.get("registry_images") or []:
                registry_images.append(
                    RegistryImage(
                        location=img_d.get("location", ""),
                        image_name=img_d.get("image_name", ""),
                        tag=img_d.get("tag", ""),
                        framework=img_d.get("framework", ""),
                        target=img_d.get("target", ""),
                        push_status=img_d.get("push_status", ""),
                        size_bytes=img_d.get("size_bytes"),
                        pushed_at=img_d.get("pushed_at"),
                    )
                )
            frameworks.append(
                FrameworkResult(
                    framework=fw_d.get("framework", ""),
                    targets=targets,
                    registry_images=registry_images,
                )
            )

        return cls(
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            sha_short=d.get("sha_short", ""),
            sha_full=d.get("sha_full", ""),
            build_date=d.get("build_date", ""),
            report_generated=d.get("report_generated", ""),
            overall_status=d.get("overall_status", ""),
            total_tasks=d.get("total_tasks", 0),
            succeeded=d.get("succeeded", 0),
            failed=d.get("failed", 0),
            skipped=d.get("skipped", 0),
            killed=d.get("killed", 0),
            commit=commit,
            frameworks=frameworks,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "BuildReport":
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_file(cls, path: Path) -> "BuildReport":
        """Read and deserialize from a JSON file."""
        path = Path(path)
        return cls.from_json(path.read_text())

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def all_registry_images(self) -> List[RegistryImage]:
        """Return a flat list of all registry images across all frameworks."""
        result: List[RegistryImage] = []
        for fw in self.frameworks:
            result.extend(fw.registry_images)
        return result

    def successful_registry_images(self) -> List[RegistryImage]:
        """Return only registry images that were successfully pushed."""
        return [img for img in self.all_registry_images() if img.push_status == "PASS"]


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _task_result_from_dict(d: Optional[Dict[str, Any]]) -> Optional[TaskResult]:
    """Convert a dict to a TaskResult, or None if the dict is None/empty."""
    if not d:
        return None
    return TaskResult(
        status=d.get("status", ""),
        duration_s=d.get("duration_s"),
        log_file=d.get("log_file"),
        prev_status=d.get("prev_status"),
    )
