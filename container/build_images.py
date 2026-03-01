#!/usr/bin/env python3
"""
Build Images - Docker Image Build Orchestration System

A Docker image build orchestration system for the Dynamo inference framework.
Supports building images for multiple frameworks (VLLM, SGLANG, TRTLLM) with
dependency management, parallel execution, and HTML reporting.

Architecture:
    - BaseTask: Abstract base class for all task types
    - BuildTask: Handles Docker image building
    - CompilationTask: Handles workspace compilation inside containers
    - ChownTask: Fixes file ownership after compilation
    - SanityCheckTask: Runs validation scripts in containers
"""

import argparse
import atexit
import glob
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import yaml

# Add parent directory to path to import common.py
sys.path.insert(0, str(Path(__file__).parent.parent))

# Third-party imports
import git
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Global set to track running subprocesses for signal handling
_running_subprocesses: Set[subprocess.Popen] = set()
_subprocesses_lock = threading.Lock()

try:
    select_autoescape  # type: ignore
except NameError:
    select_autoescape = None  # type: ignore

# Import utilities from common.py
from common import (
    normalize_framework,
    get_framework_display_name,
    DynamoImageInfo,
    DockerImageInfo,
    get_terminal_width,
    DynamoRepositoryUtils,
    DockerUtils,
    GitUtils,
    MARKER_RUNNING,
    MARKER_PASSED,
    MARKER_FAILED,
    MARKER_KILLED,
)
from common_build_report import (
    BuildReport,
    CommitInfo,
    FrameworkResult,
    RegistryImage,
    TargetResult,
    TaskResult,
    SCHEMA_VERSION,
)


# ==============================================================================
# CONSTANTS
# ==============================================================================

FRAMEWORKS = ["none", "vllm", "sglang", "trtllm"]

# Default values for HTML report generation
DEFAULT_HOSTNAME = "keivenc-linux"
DEFAULT_HTML_PATH = "/nvidia/dynamo_ci/logs"

# Global frameworks data cache for HTML reporting (initialized once, updated as tasks run)
_frameworks_data_cache: Optional[Dict[str, Dict[str, Any]]] = None
_html_out_file: Optional[Path] = None
# Set in main() so HTML report can show "Disabled (--no-compress)" / "Disabled (--no-upload)" / "Disabled (--no-build)" when applicable
_report_no_compress: bool = False
_report_no_upload: bool = False
_report_no_build: bool = False

# Global initial commit SHA tracking for detecting mid-build commits
_initial_commit_sha_9: Optional[str] = None

# Blocklisted commit SHAs: must never be checked out or built (problematic).
BLOCKLISTED_COMMIT_SHAS = ["222c2e85c"]
_BLOCKLISTED_9 = {s[:9].lower() for s in BLOCKLISTED_COMMIT_SHAS}


def _rewrite_html_links_for_repo_root(html_content: str, image_and_commit_sha: str = "", date_str: str = "") -> str:
    """
    The HTML report lives in logs/{date}/{image_and_commit_sha}/, so log links are simple basenames
    like "{task}.log".  Cross-date links use "../../{date}/{image_and_commit_sha}/{task}.log".

    When we also write the same report to repo root (e.g. <repo>/build.html),
    those must be rewritten so they resolve correctly from the repo root URL.
    """
    # Cross-date links: "../../{date}/{commit_sha_9}/{task}.log" -> "logs/{date}/{commit_sha_9}/{task}.log"
    html_content = html_content.replace('href="../../', 'href="logs/')

    # Current-date links: bare "{task}.log" -> "logs/{date}/{image_and_commit_sha}/{task}.log"
    if date_str and image_and_commit_sha:
        prefix = f"logs/{date_str}/{image_and_commit_sha}/"
        html_content = re.sub(
            r'href="([^/"]+?\.log)"',
            rf'href="{prefix}\1"',
            html_content,
        )
    return html_content


def _write_html_report_files(html_content: str, primary_html_file: Path, image_and_commit_sha: str = "", date_str: str = "") -> None:
    """
    Write the HTML report to the primary location (logs/{date}/{image_and_commit_sha}/report.html)
    and also to an optional secondary location (e.g., <repo>/build.html) if configured.
    """
    primary_html_file.write_text(html_content)
    if _html_out_file is not None:
        try:
            _html_out_file.parent.mkdir(parents=True, exist_ok=True)
            _html_out_file.write_text(_rewrite_html_links_for_repo_root(html_content, image_and_commit_sha=image_and_commit_sha, date_str=date_str))
        except (OSError, IOError) as e:
            logger = logging.getLogger("html")
            logger.warning(f"Failed to write secondary HTML file {_html_out_file}: {e}")


def _write_report_and_json(
    html_content: str,
    all_tasks: Dict[str, 'BaseTask'],
    repo_path: Path,
    image_and_commit_sha: str,
    log_dir: Path,
    date_str: str,
) -> None:
    """Write HTML report, update status marker, and save JSON build report.

    Files are written to: logs/{date_str}/{image_and_commit_sha}/report.html and report.json
    """
    report_dir = log_dir / image_and_commit_sha
    report_dir.mkdir(parents=True, exist_ok=True)
    html_file = report_dir / "report.html"
    _write_html_report_files(html_content, html_file, image_and_commit_sha=image_and_commit_sha, date_str=date_str)
    update_report_status_marker(html_file, all_tasks)

    # Best-effort JSON alongside the HTML
    try:
        report = generate_build_report(
            all_tasks=all_tasks,
            repo_path=repo_path,
            image_and_commit_sha=image_and_commit_sha,
            log_dir=log_dir,
            date_str=date_str,
        )
        json_file = report_dir / "report.json"
        report.to_file(json_file)
    except Exception as exc:
        logging.getLogger("html").warning(f"Failed to write JSON build report: {exc}")


def check_sha_changed(repo_path: Path, task_id: str) -> Optional[str]:
    """
    Check if repository HEAD SHA changed since build started.

    Returns:
        Warning message if SHA changed, None otherwise
    """
    global _initial_commit_sha_9

    if _initial_commit_sha_9 is None:
        return None

    if git is None:
        return None

    try:
        repo = git.Repo(repo_path)
        current_commit_sha_9 = repo.head.commit.hexsha[:9]

        if current_commit_sha_9 != _initial_commit_sha_9:
            # Get new commits
            try:
                new_commits = repo.git.log(f"{_initial_commit_sha_9}..HEAD", "--oneline").strip()
            except git.exc.GitCommandError:
                new_commits = "(unable to list commits - git log failed)"

            warning = (
                f"⚠️  CRITICAL: Repository SHA changed during build!\n"
                f"   Build started with: {_initial_commit_sha_9}\n"
                f"   Current HEAD:       {current_commit_sha_9}\n"
                f"   \n"
                f"   New commits:\n"
                f"   {new_commits}\n"
                f"   \n"
                f"   This causes image tag mismatches:\n"
                f"   - Earlier tasks built images with SHA {_initial_commit_sha_9}\n"
                f"   - Later tasks (like {task_id}) expect {_initial_commit_sha_9} but build.sh may use {current_commit_sha_9}\n"
                f"   - Result: 'Input image not found' errors\n"
                f"   \n"
                f"   RECOMMENDATION: Avoid committing code while builds are running.\n"
                f"   To fix: Kill build, checkout {_initial_commit_sha_9}, and rebuild.\n"
            )
            return warning
    except (git.exc.InvalidGitRepositoryError, git.exc.NoSuchPathError):
        # Not a git repo or path doesn't exist - can't check, return None
        return None

    return None


class TaskStatus(Enum):
    """Status of a task in the pipeline"""
    QUEUED = "queued"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    KILLED = "killed"


@dataclass
class LogParseResult:
    """Results from parsing a task log file"""
    status: TaskStatus
    duration: Optional[float] = None
    exit_code: Optional[int] = None


@dataclass
class TaskData:
    """Data structure for a task (build, compilation, or sanity) in HTML report"""
    status: Optional[str] = None
    time: Optional[str] = None  # Duration for any task type
    log_file: Optional[str] = None
    prev_status: Optional[str] = None  # Previous status (for SKIPPED tasks)
    start_epoch: Optional[float] = None  # Absolute start time (Unix epoch) for RUNNING tasks


@dataclass
class FrameworkTargetData:
    """Data structure for a framework target (runtime, dev, local-dev) in HTML report"""
    build: Optional[TaskData] = None
    compilation: Optional[TaskData] = None
    sanity: Optional[TaskData] = None
    image_size: Optional[str] = None
    input_image: Optional[str] = None
    output_image: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Jinja2 template compatibility"""
        return {
            'build': self.build.__dict__ if self.build else None,
            'compilation': self.compilation.__dict__ if self.compilation else None,
            'sanity': self.sanity.__dict__ if self.sanity else None,
            'image_size': self.image_size,
            'input_image': self.input_image,
            'output_image': self.output_image,
        }


# ==============================================================================
# TASK GRAPH DEFINITION
# ==============================================================================

# ==============================================================================
# DOCKER COMMAND FILTER FUNCTIONS
# ==============================================================================

# NOTE: Docker command filter functions removed in simplification (2025-12-10)
# build.sh is now executed directly instead of parsing --dry-run output


# Framework name mapping: build_images.py uses "none" but render.py uses "dynamo"
RENDER_FRAMEWORK_MAP = {
    "none": "dynamo",
    "vllm": "vllm",
    "sglang": "sglang",
    "trtllm": "trtllm",
}

# Build methods
BUILD_METHOD_BUILD_SH = "build_sh"
BUILD_METHOD_RENDER_PY = "render_py"


def detect_build_method(repo_path: Path) -> str:
    """
    Detect which build method is available at the checked-out SHA.

    Priority: render.py > build.sh (render.py is the newer approach).

    Returns:
        BUILD_METHOD_RENDER_PY or BUILD_METHOD_BUILD_SH

    Raises:
        RuntimeError: If neither render.py nor build.sh exists
    """
    render_py = repo_path / "container" / "render.py"
    build_sh = repo_path / "container" / "build.sh"

    if render_py.exists():
        return BUILD_METHOD_RENDER_PY
    if build_sh.exists():
        return BUILD_METHOD_BUILD_SH

    raise RuntimeError(
        f"Neither container/render.py nor container/build.sh found at {repo_path}. "
        "Cannot determine how to build Docker images at this SHA."
    )


def extract_version_from_build_sh(framework: str, repo_path: Path) -> str:
    """
    Extract the VERSION from build.sh by running it with --dry-run.

    The build.sh script dynamically determines VERSION based on git tags and commits:
    - If on a tagged commit: v{tag}
    - Otherwise: v{latest_tag}.dev.{commit_id}

    This function runs build.sh --dry-run and parses the --tag argument to extract
    the actual version being used.

    Args:
        framework: Framework name (vllm, sglang, trtllm)
        repo_path: Path to the Dynamo repository

    Returns:
        The version string from build.sh (legacy; render_py uses IMAGE_SHA.commit_sha_9 instead)

    Raises:
        RuntimeError: If unable to extract version from build.sh output
    """
    try:
        # Run build.sh --dry-run to get docker commands
        result = subprocess.run(
            f"{repo_path}/container/build.sh --dry-run --framework {framework} --target runtime",
            shell=True,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30
        )

        output = result.stdout.strip()

        # First try: Parse from "Building Dynamo Image: '--tag dynamo:vX.Y.Z-...-framework-target'"
        # This captures the actual tag that build.sh will use, including complex version strings
        building_match = re.search(r"Building Dynamo Image:.*'--tag\s+(?:dynamo|dynamo-base):(v.+?)-(?:none|vllm|sglang|trtllm)(?:-(?:runtime|dev|local-dev))?'", output)
        if building_match:
            return building_match.group(1)

        # Fallback: Look for --tag arguments in docker command (original simple regex)
        # Expected format: --tag dynamo:v{VERSION}-{framework}-{target}
        tag_match = re.search(r'--tag\s+(?:dynamo|dynamo-base):(v[^-\s]+)', output)
        if tag_match:
            return tag_match.group(1)

        # Fallback: if we can't find version, raise error
        raise RuntimeError(f"Could not extract version from build.sh output. Output: {output[:200]}")

    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"build.sh --dry-run timed out while extracting version: {e}")
    except (subprocess.CalledProcessError, OSError) as e:
        raise RuntimeError(f"Failed to extract version from build.sh: {e}")


def _generate_error_html_and_exit(
    frameworks: List[str],
    image_and_commit_sha: str,
    repo_path: Path,
    error_message: str,
    log_dir: Path,
    log_date: str,
    html_out_file: Optional[Path] = None,
    email: Optional[str] = None,
    hostname: str = DEFAULT_HOSTNAME,
    html_path: str = DEFAULT_HTML_PATH,
) -> int:
    """
    Generate an error HTML report when a fatal error occurs before task execution
    (e.g., build.sh doesn't exist at the checked-out SHA).

    Creates a task graph with a placeholder version, marks all tasks as FAILED
    with the error message, generates the HTML report, and returns exit code 1.

    Args:
        frameworks: List of framework names that were requested
        image_and_commit_sha: Git commit SHA (9 chars)
        repo_path: Path to the repository
        error_message: Human-readable error description
        log_dir: Directory for log files
        log_date: Date string (YYYY-MM-DD)
        html_out_file: Optional secondary HTML output path (e.g., repo/build.html)
        email: Optional email address to send report to
        hostname: Hostname for absolute URLs
        html_path: Base path for absolute URLs

    Returns:
        1 (always -- caller should return this)
    """
    global _frameworks_data_cache, _html_out_file

    logger = logging.getLogger("main")
    logger.error(f"Fatal error: {error_message}")

    _html_out_file = html_out_file

    # Create task graphs with a placeholder version so we have tasks to mark as FAILED
    all_tasks: Dict[str, BaseTask] = {}
    for framework in frameworks:
        framework_tasks = create_task_graph(framework, image_and_commit_sha, repo_path, version="error")
        all_tasks.update(framework_tasks)

    # Set log file paths and mark every task as FAILED
    for task_id, task in all_tasks.items():
        task.set_log_file_path(log_dir, log_date, image_and_commit_sha)
        task.mark_status_as(TaskStatus.FAILED, error_message)

        # Write a log file for each task so log links in the HTML report work
        try:
            with open(task.log_file, 'w') as log_fh:
                log_fh.write(f"Task: {task_id}\n")
                log_fh.write(f"Description: {task.description}\n")
                log_fh.write(f"Error: {error_message}\n")
                log_fh.write("=" * 80 + "\n")
        except (OSError, IOError) as e:
            logger.warning(f"Failed to write error log for {task_id}: {e}")

    # Initialize frameworks data cache -- pass version="error" so temp task graphs
    # for non-active frameworks don't re-invoke build.sh (which is missing).
    _frameworks_data_cache = initialize_frameworks_data_cache(
        all_tasks=all_tasks,
        log_dir=log_dir,
        date_str=log_date,
        image_and_commit_sha=image_and_commit_sha,
        repo_path=repo_path,
        use_absolute_urls=email is not None,
        version="error",
    )

    # Update cache with FAILED status for all tasks
    for task in all_tasks.values():
        update_frameworks_data_cache(task, use_absolute_urls=email is not None)

    # Generate HTML report
    html_file = log_dir / image_and_commit_sha / "report.html"
    try:
        html_content = generate_html_report(
            all_tasks=all_tasks,
            repo_path=repo_path,
            image_and_commit_sha=image_and_commit_sha,
            log_dir=log_dir,
            date_str=log_date,
            use_absolute_urls=False,
        )
        _write_report_and_json(html_content, all_tasks, repo_path, image_and_commit_sha, log_dir, log_date)
        logger.info(f"Error HTML report written: {html_file}")
        if _html_out_file:
            logger.info(f"Error HTML report also written: {_html_out_file}")
    except (OSError, IOError, ValueError, RuntimeError, KeyError) as e:
        logger.error(f"Failed to generate error HTML report: {e}")

    # Send email if requested
    # send_email_notification handles its own errors internally and returns a bool,
    # so only generate_html_report needs guarding here.
    if email:
        try:
            html_content_email = generate_html_report(
                all_tasks=all_tasks,
                repo_path=repo_path,
                image_and_commit_sha=image_and_commit_sha,
                log_dir=log_dir,
                date_str=log_date,
                use_absolute_urls=True,
                hostname=hostname,
                html_path=html_path,
            )
        except (OSError, IOError, ValueError, RuntimeError, KeyError) as e:
            logger.error(f"Failed to generate email HTML content: {e}")
            return 1

        failed_task_names = [t.task_id for t in all_tasks.values()]
        send_email_notification(
            email=email,
            html_content=html_content_email,
            image_and_commit_sha=image_and_commit_sha[:7],
            failed_tasks=failed_task_names,
        )

    return 1


# Keys in context.yaml that define CUDA versions (e.g. cuda12.9, cuda13.0, cuda13.1)
_CUDA_KEY_PATTERN = re.compile(r"^cuda(\d+\.\d+)$")


def get_latest_cuda_version(repo_path: Path, framework: str) -> str:
    """
    Parse container/context.yaml and return the latest CUDA version for the given framework.
    Returns the highest cudaX.Y key present (e.g. "13.0" for vllm/sglang/none, "13.1" for trtllm).
    Fallback: "13.1" for trtllm, "12.9" for others if context is missing or has no cuda keys.
    """
    context_path = repo_path / "container" / "context.yaml"
    if not context_path.exists():
        return "13.1" if framework == "trtllm" else "12.9"
    try:
        with open(context_path) as f:
            context = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return "13.1" if framework == "trtllm" else "12.9"
    if not context:
        return "13.1" if framework == "trtllm" else "12.9"
    section = "dynamo" if framework == "none" else framework
    fw_ctx = context.get(section)
    if not fw_ctx:
        return "13.1" if framework == "trtllm" else "12.9"
    versions = []
    for key in fw_ctx:
        m = _CUDA_KEY_PATTERN.match(key)
        if m:
            versions.append(m.group(1))
    if not versions:
        return "13.1" if framework == "trtllm" else "12.9"
    # Sort by (major, minor) and take latest
    def version_key(v: str) -> tuple:
        parts = v.split(".")
        return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    return max(versions, key=version_key)


def get_runtime_base_image(repo_path: Path, framework: str) -> Optional[str]:
    """
    Parse container/context.yaml and return the base image for the runtime target
    (the first FROM ... AS runtime). Used to show "Input Image" in the report for runtime.
    """
    context_path = repo_path / "container" / "context.yaml"
    if not context_path.exists():
        return None
    try:
        with open(context_path) as f:
            context = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return None
    if not context:
        return None
    # Use latest CUDA version from context (same as create_task_graph)
    cuda_version = get_latest_cuda_version(repo_path, framework)
    cuda_key = f"cuda{cuda_version}"
    if framework == "none":
        # dynamo_runtime.Dockerfile uses FROM dynamo_base AS runtime; dynamo_base is FROM BASE_IMAGE:BASE_IMAGE_TAG
        dyn = context.get("dynamo") or {}
        cuda_ctx = dyn.get(cuda_key) or dyn.get("cuda12.9") or {}
        base = dyn.get("base_image")
        tag = cuda_ctx.get("base_image_tag")
        if base and tag:
            return f"{base}:{tag}"
        return None
    fw_ctx = context.get(framework)
    if not fw_ctx:
        return None
    cuda_ctx = fw_ctx.get(cuda_key) or fw_ctx.get("cuda12.9") or {}
    img = fw_ctx.get("runtime_image")
    tag = cuda_ctx.get("runtime_image_tag")
    if img and tag:
        return f"{img}:{tag}"
    return None


def detect_latest_image_sha(repo_path: Path, framework: str) -> Optional[str]:
    """
    Auto-detect the latest local Docker image SHA for a framework.
    Looks for images matching dynamo:*-{framework}-cuda*-local-dev or dynamo:*-{framework}-cuda*-dev
    and returns the SHA portion (version prefix in the tag).
    """
    import subprocess as _sp
    result = _sp.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}", "--filter", "reference=dynamo"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        return None

    cuda_version = get_latest_cuda_version(repo_path, framework)
    cuda_tag = f"cuda{cuda_version}"

    # Prefer local-dev, then dev, then runtime
    for target in ("local-dev", "dev", "runtime"):
        pattern = f"-{framework}-{cuda_tag}-{target}"
        for line in result.stdout.strip().splitlines():
            if pattern in line:
                # Tag format: dynamo:{IMAGE_SHA}.{commit_sha_9}-{framework}-cuda{ver}-{target}
                tag = line.split(":", 1)[1] if ":" in line else line
                tag_version = tag.split(f"-{framework}-")[0]
                if tag_version:
                    return tag_version
    return None


def _load_previous_report_images(repo_path: Path, image_sha_6: str) -> Dict[str, Dict[str, Dict[str, Optional[str]]]]:
    """
    Find a previous report.json with the same Image SHA and extract image info.
    Returns: {framework: {target: {'input_image': ..., 'output_image': ...}}}
    """
    import glob as _glob
    logger = logging.getLogger(__name__)
    logs_dir = repo_path / "logs"
    if not logs_dir.is_dir():
        return {}
    pattern = str(logs_dir / "*" / f"{image_sha_6}.*" / "report.json")
    candidates = sorted(_glob.glob(pattern), reverse=True)
    for candidate in candidates:
        try:
            with open(candidate) as f:
                data = json.load(f)
            result: Dict[str, Dict[str, Dict[str, Optional[str]]]] = {}
            for fw in data.get("frameworks", []):
                fw_name = fw["framework"]
                result[fw_name] = {}
                for tgt in fw.get("targets", []):
                    tgt_name = tgt["target"]
                    result[fw_name][tgt_name] = {
                        "input_image": tgt.get("input_image"),
                        "output_image": tgt.get("image_name"),
                    }
            logger.info(f"Loaded image info from previous report: {candidate}")
            return result
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.debug(f"Skipping {candidate}: {e}")
            continue
    return {}


def _docker_image_exists(tag: str) -> bool:
    """Return True if the given Docker image tag exists locally."""
    try:
        result = subprocess.run(
            ["docker", "inspect", tag],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
        return False


def _find_dev_image(image_sha_6: str, framework: str, cuda_tag: str) -> Optional[str]:
    """
    Find an existing dev image with the given Image SHA.
    Returns the full tag string if found, None otherwise.
    """
    try:
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}",
             "--filter", f"reference=dynamo:{image_sha_6}.*-{framework}-{cuda_tag}-dev"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        existing_tags = [t.strip() for t in result.stdout.strip().splitlines() if t.strip()]
        return existing_tags[0] if existing_tags else None
    except (subprocess.TimeoutExpired, OSError):
        return None


def _check_reuse_dev_images_exist(repo_path: Path, tag_version: str, frameworks: List[str]) -> Optional[Dict[str, str]]:
    """
    Check if dev images with the same Image SHA exist locally for all frameworks.
    Returns {framework: existing_tag} if all found, None otherwise.
    No retagging — reuses existing images as-is.
    """
    logger = logging.getLogger(__name__)
    image_sha_6 = tag_version.split('.')[0] if '.' in tag_version else tag_version
    found: Dict[str, str] = {}
    for framework in frameworks:
        cuda_version = get_latest_cuda_version(repo_path, framework)
        cuda_tag = f"cuda{cuda_version}"
        existing = _find_dev_image(image_sha_6, framework, cuda_tag)
        if not existing:
            logger.info(f"Reuse skipped: no dev image with Image SHA {image_sha_6} for {framework}-{cuda_tag}")
            return None
        found[framework] = existing
        logger.info(f"  Reuse: found {existing}")
    return found


# Factory function to create task instances for a specific framework
def create_task_graph(
    framework: str,
    commit_sha_9: str,
    repo_path: Path,
    version: Optional[str] = None,
    build_method: str = BUILD_METHOD_BUILD_SH,
) -> Dict[str, 'BaseTask']:
    """
    Create task instances for a specific framework.

    Args:
        framework: Framework name (none, vllm, sglang, trtllm)
        commit_sha_9: Git commit SHA (short form, 9 chars)
        repo_path: Path to the Dynamo repository
        version: Version string for image tags. For render_py method this is
                 IMAGE_SHA.commit_sha_9 (e.g., "C62194.6d3e0137c"). For build_sh
                 method this comes from build.sh --dry-run. If None, will be
                 extracted from build.sh output (only works with build_sh method).
        build_method: BUILD_METHOD_BUILD_SH or BUILD_METHOD_RENDER_PY

    Returns:
        Dictionary mapping task IDs to task instances

    Image Dependency Chain (build_sh):
        build.sh is executed directly for each target.
        1. Runtime: build.sh --target runtime → dynamo:{version}-{framework}-runtime
        2. Dev: build.sh --target dev → dynamo:{version}-{framework}-dev
        3. Local-dev: build.sh --target local-dev → dynamo:{version}-{framework}-local-dev

    Image Dependency Chain (render_py):
        render.py generates a Dockerfile, then docker build produces the image.
        1. Runtime: render.py + docker build → dynamo:{commit_sha_9}-{framework}-cuda{cuda_version}-runtime
        2. Dev: render.py + docker build → dynamo:{commit_sha_9}-{framework}-cuda{cuda_version}-dev
        3. Local-dev: render.py + docker build → dynamo:{commit_sha_9}-{framework}-cuda{cuda_version}-local-dev
    """
    # Task ID format: framework-target-type (all lowercase)
    # Example: vllm-runtime-build, vllm-dev-compilation, vllm-runtime-sanity

    # Determine version if not provided
    if version is None:
        if build_method == BUILD_METHOD_RENDER_PY:
            # render.py method: use IMAGE_SHA.commit_sha_9 as version (tag: dynamo:{IMAGE_SHA}.{commit_sha_9}-{fw}-cuda{ver}-{target})
            version = commit_sha_9
        else:
            version = extract_version_from_build_sh(framework, repo_path)

    tasks: Dict[str, BaseTask] = {}
    sanity_no_framework_flag = " --no-framework-check" if framework == "none" else ""
    home_dir = str(Path.home())

    # render.py uses "dynamo" instead of "none"
    render_fw = RENDER_FRAMEWORK_MAP.get(framework, framework)
    # CUDA version from context.yaml (latest cudaX.Y per framework)
    render_cuda_version = get_latest_cuda_version(repo_path, framework)

    def _build_cmd(target: str, image_tag: str) -> str:
        """Generate the build command for a given target and image tag."""
        if build_method == BUILD_METHOD_RENDER_PY:
            # Two-step:
            #   1. render.py generates Dockerfile, prints "INFO: Generated Dockerfile written to <path>"
            #   2. Parse that output to get the actual path, then docker build with it
            render_cmd = f"python3 {repo_path}/container/render.py --framework {render_fw} --target {target} --cuda-version {render_cuda_version}"
            extra_args = ""
            if target == "local-dev":
                extra_args = " --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g)"
            # Chain: run render.py, parse its output for the Dockerfile path, then docker build
            return (
                f"RENDER_OUTPUT=$({render_cmd}) && "
                f"RENDERED_DF=$(echo \"$RENDER_OUTPUT\" | grep -oP '(?<=written to ).*') && "
                f"echo \"$RENDER_OUTPUT\" && "
                f"docker build --progress=plain -t {image_tag}{extra_args} -f \"$RENDERED_DF\" {repo_path}"
            )
        # build_sh method
        return f"{repo_path}/container/build.sh --no-tag-latest --framework {framework} --target {target}"

    # Embed CUDA version in image tags so the tag encodes framework, CUDA, and target
    cuda_tag = f"cuda{render_cuda_version}"

    # Level 0: Runtime image build (builds directly from CUDA base image).
    # Set input_image for report display only; check_input_image_exists() skips runtime-build (docker build pulls base).
    runtime_base_image = get_runtime_base_image(repo_path, framework)
    runtime_image_tag = f"dynamo:{version}-{framework}-{cuda_tag}-runtime"
    tasks[f"{framework}-runtime-build"] = BuildTask(
        task_id=f"{framework}-runtime-build",
        description=f"Build {framework.upper()} runtime image",
        command=_build_cmd("runtime", runtime_image_tag),
        output_image=runtime_image_tag,
        input_image=runtime_base_image,
        timeout=3600.0,  # 60 minutes for builds
    )

    # Level 2: Runtime sanity check
    tasks[f"{framework}-runtime-sanity"] = CommandTask(
        task_id=f"{framework}-runtime-sanity",
        description=f"Run sanity_check.py in {framework.upper()} runtime container",
        command=f"{repo_path}/container/run.sh --image {runtime_image_tag} --hf-home {home_dir}/.cache/huggingface -- bash -c 'id && python3 /workspace/deploy/sanity_check.py --runtime-check{sanity_no_framework_flag}'",
        input_image=runtime_image_tag,
        parents=[f"{framework}-runtime-build"],
        timeout=240.0,  # 4 minutes for sanity checks (trtllm can exceed 3 min)
        ignore_exit_code=False,
    )

    # Level 1: Dev image build (waits for runtime to complete)
    # Builds as -dev-orig; compress step will later produce the final -dev image.
    dev_image_tag = f"dynamo:{version}-{framework}-{cuda_tag}-dev"
    dev_orig_image_tag = f"{dev_image_tag}-orig"
    tasks[f"{framework}-dev-build"] = BuildTask(
        task_id=f"{framework}-dev-build",
        description=f"Build {framework.upper()} dev image",
        command=_build_cmd("dev", dev_orig_image_tag),
        input_image=runtime_image_tag,
        output_image=dev_orig_image_tag,
        parents=[f"{framework}-runtime-build"],
        timeout=3600.0,  # 60 minutes for builds
    )

    # Level 3: Dev compilation
    # Use fast build flags from compile.sh:
    # - CARGO_INCREMENTAL=1: Enable incremental compilation
    # - CARGO_PROFILE_DEV_OPT_LEVEL=0: No optimizations (faster compile)
    # - CARGO_BUILD_JOBS=$(nproc): Parallel compilation
    # - CARGO_PROFILE_DEV_CODEGEN_UNITS=256: More parallel code generation
    cargo_cmd = "CARGO_INCREMENTAL=1 CARGO_PROFILE_DEV_OPT_LEVEL=0 CARGO_BUILD_JOBS=$(nproc) CARGO_PROFILE_DEV_CODEGEN_UNITS=256 cargo build --profile dev --features dynamo-llm/block-manager && cd /workspace/lib/bindings/python && CARGO_INCREMENTAL=1 CARGO_PROFILE_DEV_OPT_LEVEL=0 CARGO_BUILD_JOBS=$(nproc) CARGO_PROFILE_DEV_CODEGEN_UNITS=256 maturin develop --uv && uv pip install -e ."
    chown_cmd = f"sudo chown -R $(id -u):$(id -g) {repo_path}/target/.{framework} {repo_path}/lib/bindings/python {home_dir}/.cargo || true"

    # Level 3a: Pre-chown before dev compilation (fix ownership from previous root-owned builds)
    tasks[f"{framework}-dev-pre-chown"] = CommandTask(
        task_id=f"{framework}-dev-pre-chown",
        description=f"Fix file ownership before {framework.upper()} dev compilation",
        command=chown_cmd,
        parents=[f"{framework}-dev-build"],
        timeout=60.0,
    )

    # Compilation runs after pre-chown (before sanity), same for all frameworks
    tasks[f"{framework}-dev-compilation"] = CommandTask(
        task_id=f"{framework}-dev-compilation",
        description=f"Run workspace compilation in {framework.upper()} dev container",
        command=f"{repo_path}/container/run.sh --image {dev_orig_image_tag} --mount-workspace -v {home_dir}/.cargo:/root/.cargo -v {repo_path}/target/.{framework}:/workspace/target -- bash -c '{cargo_cmd}'",
        input_image=dev_orig_image_tag,
        parents=[f"{framework}-dev-pre-chown"],
        timeout=1800.0,  # 30 minutes for compilation
    )

    # Level 4: Dev post-chown (runs after compilation, always runs even if compilation fails)
    tasks[f"{framework}-dev-chown"] = CommandTask(
        task_id=f"{framework}-dev-chown",
        description=f"Fix file ownership after {framework.upper()} dev compilation",
        command=chown_cmd,
        parents=[f"{framework}-dev-compilation"],
        run_even_if_deps_fail=True,
        timeout=60.0,
    )

    # Level 4: Dev sanity check (runs after compilation, same for all frameworks)
    tasks[f"{framework}-dev-sanity"] = CommandTask(
        task_id=f"{framework}-dev-sanity",
        description=f"Run sanity_check.py in {framework.upper()} dev container",
        command=f"{repo_path}/container/run.sh --image {dev_orig_image_tag} --mount-workspace --hf-home {home_dir}/.cache/huggingface -v {home_dir}/.cargo:/root/.cargo -- bash -c 'id && python3 /workspace/deploy/sanity_check.py{sanity_no_framework_flag}'",
        input_image=dev_orig_image_tag,
        parents=[f"{framework}-dev-compilation"],
        timeout=240.0,  # 4 minutes for sanity checks (trtllm can exceed 3 min)
    )

    # Level 8: Dev upload (runs after compress-sanity, in parallel with local-dev build)
    gitlab_dev_image_tag = f"gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo/dev/{dev_image_tag}"
    # Use "docker rmi ... || true" so a second run or already-removed tag does not fail the task
    tasks[f"{framework}-dev-upload"] = CommandTask(
        task_id=f"{framework}-dev-upload",
        description=f"Upload {framework.upper()} dev image to GitLab registry",
        command=(
            f"docker tag {dev_image_tag} {gitlab_dev_image_tag} && "
            f"docker push {gitlab_dev_image_tag} && "
            f"( docker rmi {gitlab_dev_image_tag} || true )"
        ),
        input_image=dev_image_tag,
        output_image=gitlab_dev_image_tag,
        parents=[f"{framework}-dev-compress-sanity"],
        timeout=14400.0,  # 4 hours for upload
    )

    # Level 6: Dev compress (squash dev-orig image to produce dev)
    # Takes the -orig image (built by dev-build), removes /workspace, .git dirs,
    # /tmp, __pycache__, squashes all layers into a single layer via
    # docker export/import, and outputs the final -dev tag.
    # The -orig image is preserved after build (not deleted).
    #
    # docker export/import loses all image metadata (ENV, ENTRYPOINT, WORKDIR, USER,
    # CMD), so we reconstruct it: inspect the original, emit --change flags, then
    # pass them to docker import.
    compress_script_path = f"/tmp/_compress_{framework}_{version}.sh"
    compress_import_script_path = f"/tmp/_compress_import_{framework}_{version}.sh"
    compress_script_content = f"""#!/usr/bin/env bash
set -e

echo "Compressing {dev_orig_image_tag} -> {dev_image_tag}"

# 1. Extract metadata and write a helper import script.
#    We pipe docker export into this script so stdin flows to docker import.
docker inspect --format '{{{{json .Config}}}}' {dev_orig_image_tag} | \\
  python3 -c '
import sys, json, shlex
c = json.load(sys.stdin)
parts = []
for e in (c.get("Env") or []):
    # Use ENV key="value" form: quotes protect both spaces in values
    # (e.g. CUDA_ARCH_LIST) and empty values (e.g. TRTOSS_VERSION=).
    k, _, v = e.partition("=")
    parts.extend(["--change", shlex.quote("ENV " + k + "=" + chr(34) + v.replace(chr(34), chr(92)+chr(34)) + chr(34))])
ep = c.get("Entrypoint")
if ep:
    parts.extend(["--change", shlex.quote("ENTRYPOINT " + json.dumps(ep))])
cmd = c.get("Cmd")
if cmd:
    parts.extend(["--change", shlex.quote("CMD " + json.dumps(cmd))])
wd = c.get("WorkingDir")
if wd:
    parts.extend(["--change", shlex.quote("WORKDIR " + wd)])
u = c.get("User")
if u:
    parts.extend(["--change", shlex.quote("USER " + u)])
with open("{compress_import_script_path}", "w") as f:
    f.write("#!/usr/bin/env bash\\n")
    f.write("docker import " + " ".join(parts) + " - \\"$1\\"\\n")
'
chmod +x {compress_import_script_path}

# 2. Create container from -orig that removes bloat, run it to completion
CONTAINER_ID=$(docker create --entrypoint /bin/bash {dev_orig_image_tag} \\
  -c 'rm -rf /workspace /tmp/* && find / -name .git -type d -exec rm -rf {{}} + 2>/dev/null; find / -name __pycache__ -type d -exec rm -rf {{}} + 2>/dev/null; true')
docker start -a "$CONTAINER_ID"

# 3. Flatten and import as the final dev tag
docker export "$CONTAINER_ID" | bash {compress_import_script_path} {dev_image_tag}
docker rm -f "$CONTAINER_ID"
rm -f {compress_import_script_path}

echo "Compress complete: {dev_image_tag}"
docker images {dev_image_tag} --format 'Size: {{{{.Size}}}}'
"""
    # Write script to temp file so shell=True can execute it correctly
    Path(compress_script_path).write_text(compress_script_content)
    Path(compress_script_path).chmod(0o755)

    tasks[f"{framework}-dev-compress"] = CommandTask(
        task_id=f"{framework}-dev-compress",
        description=f"Compress {framework.upper()} dev image (remove bloat + squash layers)",
        command=f"bash {compress_script_path}",
        input_image=dev_orig_image_tag,
        output_image=dev_image_tag,
        parents=[f"{framework}-dev-sanity"],
        timeout=5400.0,  # 90 minutes for compress (export/import)
    )

    # Level 7: Pre-chown before compressed dev compilation
    tasks[f"{framework}-dev-compress-pre-chown"] = CommandTask(
        task_id=f"{framework}-dev-compress-pre-chown",
        description=f"Fix file ownership before compressed {framework.upper()} dev compilation",
        command=chown_cmd,
        parents=[f"{framework}-dev-compress"],
        timeout=60.0,
    )

    # Level 7: Compilation in the compressed dev image (verify cargo build works after squash)
    tasks[f"{framework}-dev-compress-compilation"] = CommandTask(
        task_id=f"{framework}-dev-compress-compilation",
        description=f"Run workspace compilation in compressed {framework.upper()} dev container",
        command=f"{repo_path}/container/run.sh --image {dev_image_tag} --mount-workspace -v {home_dir}/.cargo:/root/.cargo -v {repo_path}/target/.{framework}:/workspace/target -- bash -c '{cargo_cmd}'",
        input_image=dev_image_tag,
        parents=[f"{framework}-dev-compress-pre-chown"],
        timeout=1800.0,  # 30 minutes for compilation
    )

    # Level 7b: Post-chown after compressed dev compilation
    tasks[f"{framework}-dev-compress-chown"] = CommandTask(
        task_id=f"{framework}-dev-compress-chown",
        description=f"Fix file ownership after compressed {framework.upper()} dev compilation",
        command=chown_cmd,
        parents=[f"{framework}-dev-compress-compilation"],
        run_even_if_deps_fail=True,
        timeout=60.0,
    )

    # Level 7b: Sanity check on the compressed dev image (after compilation)
    tasks[f"{framework}-dev-compress-sanity"] = CommandTask(
        task_id=f"{framework}-dev-compress-sanity",
        description=f"Run sanity_check.py on compressed {framework.upper()} dev image",
        command=f"{repo_path}/container/run.sh --image {dev_image_tag} --mount-workspace --hf-home {home_dir}/.cache/huggingface -v {home_dir}/.cargo:/root/.cargo -- bash -c 'id && python3 /workspace/deploy/sanity_check.py{sanity_no_framework_flag}'",
        input_image=dev_image_tag,
        parents=[f"{framework}-dev-compress-compilation"],
        timeout=240.0,  # 4 minutes for sanity checks (trtllm can exceed 3 min)
    )

    # Level 8: Local-dev image build (runs after compress-sanity, in parallel with dev-upload)
    local_dev_image_tag = f"dynamo:{version}-{framework}-{cuda_tag}-local-dev"
    tasks[f"{framework}-local-dev-build"] = BuildTask(
        task_id=f"{framework}-local-dev-build",
        description=f"Build {framework.upper()} local-dev image",
        command=_build_cmd("local-dev", local_dev_image_tag),
        input_image=dev_image_tag,
        output_image=local_dev_image_tag,
        parents=[f"{framework}-dev-compress-sanity"],
        timeout=3600.0,  # 60 minutes for builds
    )

    # Level 9a: Pre-chown before local-dev compilation
    tasks[f"{framework}-local-dev-pre-chown"] = CommandTask(
        task_id=f"{framework}-local-dev-pre-chown",
        description=f"Fix file ownership before {framework.upper()} local-dev compilation",
        command=chown_cmd,
        parents=[f"{framework}-local-dev-build", f"{framework}-dev-compress-chown"],
        timeout=60.0,
    )

    # Level 9: Local-dev compilation (runs after local-dev-build + pre-chown)
    tasks[f"{framework}-local-dev-compilation"] = CommandTask(
        task_id=f"{framework}-local-dev-compilation",
        description=f"Run workspace compilation in {framework.upper()} local-dev container",
        command=f"{repo_path}/container/run.sh --image {local_dev_image_tag} --mount-workspace -v {home_dir}/.cargo:/home/dynamo/.cargo -v {repo_path}/target/.{framework}:/workspace/target -- bash -c '{cargo_cmd}'",
        input_image=local_dev_image_tag,
        parents=[f"{framework}-local-dev-pre-chown"],
        timeout=1800.0,  # 30 minutes for compilation
    )

    # Level 10: Local-dev sanity check (runs after compilation, same for all frameworks)
    tasks[f"{framework}-local-dev-sanity"] = CommandTask(
        task_id=f"{framework}-local-dev-sanity",
        description=f"Run sanity_check.py in {framework.upper()} local-dev container",
        command=f"{repo_path}/container/run.sh --image {local_dev_image_tag} --mount-workspace --hf-home {home_dir}/.cache/huggingface -v {home_dir}/.cargo:/home/dynamo/.cargo -- bash -c 'id && (sudo id || true) && python3 /workspace/deploy/sanity_check.py{sanity_no_framework_flag}'",
        input_image=local_dev_image_tag,
        parents=[f"{framework}-local-dev-compilation"],
        timeout=240.0,  # 4 minutes for sanity checks (trtllm can exceed 3 min)
    )

    return tasks


def create_task_graph_reuse(
    framework: str,
    commit_sha_9: str,
    repo_path: Path,
    dev_image_tag: str,
    build_method: str = BUILD_METHOD_RENDER_PY,
) -> Dict[str, "BaseTask"]:
    """
    Create dev-compress compilation and sanity-check tasks for reuse mode.
    Uses the existing dev image directly — no build, no retag.
    """
    tasks: Dict[str, BaseTask] = {}
    sanity_no_framework_flag = " --no-framework-check" if framework == "none" else ""
    home_dir = str(Path.home())

    cargo_cmd = "CARGO_INCREMENTAL=1 CARGO_PROFILE_DEV_OPT_LEVEL=0 CARGO_BUILD_JOBS=$(nproc) CARGO_PROFILE_DEV_CODEGEN_UNITS=256 cargo build --profile dev --features dynamo-llm/block-manager && cd /workspace/lib/bindings/python && CARGO_INCREMENTAL=1 CARGO_PROFILE_DEV_OPT_LEVEL=0 CARGO_BUILD_JOBS=$(nproc) CARGO_PROFILE_DEV_CODEGEN_UNITS=256 maturin develop --uv && uv pip install -e ."
    chown_cmd = f"sudo chown -R $(id -u):$(id -g) {repo_path}/target/.{framework} {repo_path}/lib/bindings/python {home_dir}/.cargo || true"

    tasks[f"{framework}-dev-compress-pre-chown"] = CommandTask(
        task_id=f"{framework}-dev-compress-pre-chown",
        description=f"Fix file ownership before compressed {framework.upper()} dev compilation (reuse)",
        command=chown_cmd,
        parents=[],
        timeout=60.0,
    )

    tasks[f"{framework}-dev-compress-compilation"] = CommandTask(
        task_id=f"{framework}-dev-compress-compilation",
        description=f"Run workspace compilation in compressed {framework.upper()} dev container (reuse)",
        command=f"{repo_path}/container/run.sh --image {dev_image_tag} --mount-workspace -v {home_dir}/.cargo:/root/.cargo -v {repo_path}/target/.{framework}:/workspace/target -- bash -c '{cargo_cmd}'",
        input_image=dev_image_tag,
        parents=[f"{framework}-dev-compress-pre-chown"],
        timeout=1800.0,  # 30 minutes for compilation
    )

    tasks[f"{framework}-dev-compress-chown"] = CommandTask(
        task_id=f"{framework}-dev-compress-chown",
        description=f"Fix file ownership after compressed {framework.upper()} dev compilation (reuse)",
        command=chown_cmd,
        parents=[f"{framework}-dev-compress-compilation"],
        run_even_if_deps_fail=True,
        timeout=60.0,
    )

    tasks[f"{framework}-dev-compress-sanity"] = CommandTask(
        task_id=f"{framework}-dev-compress-sanity",
        description=f"Run sanity_check.py on compressed {framework.upper()} dev image (reuse)",
        command=f"{repo_path}/container/run.sh --image {dev_image_tag} --mount-workspace --hf-home {home_dir}/.cache/huggingface -v {home_dir}/.cargo:/root/.cargo -- bash -c 'id && python3 /workspace/deploy/sanity_check.py{sanity_no_framework_flag}'",
        input_image=dev_image_tag,
        parents=[f"{framework}-dev-compress-chown"],
        timeout=240.0,  # 4 minutes for sanity checks (trtllm can exceed 3 min)
    )

    return tasks


# ==============================================================================
# BASE TASK CLASS
# ==============================================================================

@dataclass
class BaseTask(ABC):
    """
    Abstract base class for all task types.

    All tasks share common attributes and behaviors:
    - Unique identification (task_id, framework, target)
    - Dependency management (parents, children)
    - Execution state (status, start_time, end_time)
    - Logging (log_file)
    - Configuration (timeout, run_on_dep_failure)
    """

    # Identity
    task_id: str
    description: str

    # Command and execution
    command: str = ""
    timeout: float = 900.0  # Default 15 minutes

    # Docker image tracking
    input_image: Optional[str] = None   # Base/input Docker image (for builds or container runs)
    output_image: Optional[str] = None  # Output Docker image (for builds)

    # Dependency management
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    run_even_if_deps_fail: bool = False  # Run even if dependencies fail

    # Execution state
    _status: TaskStatus = TaskStatus.QUEUED
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    log_file: Optional[Path] = None

    # Results
    exit_code: Optional[int] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        """Initialize task-specific attributes"""
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{self.task_id}")

    @property
    def status(self) -> TaskStatus:
        """Get the current status of the task"""
        return self._status

    def set_log_file_path(self, log_dir: Path, log_date: str, image_and_commit_sha: str) -> Path:
        """
        Set and return the log file path for this task.

        Log files live under: logs/{date}/{image_and_commit_sha}/{task_id}.log

        Args:
            log_dir: Date-level directory (e.g. logs/2026-02-20)
            log_date: Date string (YYYY-MM-DD) -- unused, kept for API compat
            image_and_commit_sha: File SHA string (e.g. "C62194.6d3e0137c")

        Returns:
            The log file path
        """
        report_dir = log_dir / image_and_commit_sha
        report_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = report_dir / f"{self.task_id}.log"
        return self.log_file

    @staticmethod
    def get_log_file_path(log_dir: Path, log_date: str, image_and_commit_sha: str, task_id: str) -> Path:
        """
        Get the log file path for a task ID without a task instance.

        Args:
            log_dir: Date-level directory (e.g. logs/2026-02-20)
            log_date: Date string (YYYY-MM-DD) -- unused, kept for API compat
            image_and_commit_sha: File SHA string (e.g. "C62194.6d3e0137c")
            task_id: Task identifier

        Returns:
            The log file path
        """
        return log_dir / image_and_commit_sha / f"{task_id}.log"

    @abstractmethod
    def prepare(self, repo_path: Path, commit_sha_9: str) -> None:
        """
        Prepare task for execution.

        This method should:
        - Generate the command to execute
        - Set up any required paths or environment
        - Validate prerequisites

        Args:
            repo_path: Path to the repository
            commit_sha_9: Git commit SHA
        """
        pass

    def _has_network_error(self) -> bool:
        """
        Check if the log file contains network-related errors.

        Returns:
            True if network errors are detected in the log file
        """
        if self.log_file is None or not self.log_file.exists():
            return False

        network_error_patterns = [
            "network is unreachable",
            "dial tcp",
            "failed to do request",
            "connection refused",
            "connection timeout",
            "temporary failure in name resolution",
            "could not resolve host",
        ]

        try:
            with open(self.log_file, 'r') as f:
                log_content = f.read().lower()
                for pattern in network_error_patterns:
                    if pattern in log_content:
                        return True
        except (OSError, IOError):
            pass

        return False

    def _run_command_once(self, command: str, repo_path: Path) -> int:
        """
        Run a command once and log output (internal method).

        Args:
            command: Command to execute
            repo_path: Path to the repository

        Returns:
            Exit code of the command (0 for success, non-zero for failure)
        """
        try:
            # Open log file in append mode
            if self.log_file is None:
                raise ValueError("log_file is not set")
            with open(self.log_file, 'a') as log_fh:
                log_fh.write(f"\n→ {command}\n\n")
                log_fh.flush()

                # Run the command
                process = subprocess.Popen(
                    command,
                    shell=True,
                    cwd=repo_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                log_fh.write(f"PID: {process.pid}\n")
                log_fh.flush()

                # Register subprocess for signal handling
                with _subprocesses_lock:
                    _running_subprocesses.add(process)

                try:
                    # Stream output to log file; enforce task timeout via a timer that kills the process
                    if process.stdout is None:
                        raise ValueError("process.stdout is None")
                    timeout_sec = self.timeout
                    timed_out = threading.Event()

                    def kill_after_timeout():
                        if not timed_out.is_set():
                            timed_out.set()
                            try:
                                process.kill()
                            except ProcessLookupError:
                                pass

                    timer = threading.Timer(timeout_sec, kill_after_timeout)
                    timer.daemon = True
                    timer.start()
                    try:
                        for line in process.stdout:
                            log_fh.write(line)
                            log_fh.flush()
                        process.wait()
                    finally:
                        timer.cancel()
                    if timed_out.is_set():
                        self.exit_code = -1
                        self.error_message = f"Task timed out after {timeout_sec:.0f}s"
                        self.logger.error(self.error_message)
                        log_fh.write(f"\n\n[TIMEOUT] Killed after {timeout_sec:.0f}s\n")
                        log_fh.flush()
                        return -1
                    self.exit_code = process.returncode
                    return self.exit_code
                finally:
                    # Unregister subprocess
                    with _subprocesses_lock:
                        _running_subprocesses.discard(process)

        except (OSError, IOError, ValueError) as e:
            self.error_message = str(e)
            self.exit_code = -1
            self.logger.error(f"Command execution failed: {e}")
            raise

    def _run_command(self, command: str, repo_path: Path) -> int:
        """
        Run a command with retry logic for network errors.

        Retries up to 3 times if network-related errors are detected.

        Args:
            command: Command to execute
            repo_path: Path to the repository

        Returns:
            Exit code of the command (0 for success, non-zero for failure)
        """
        max_retries = 3
        retry_delay = 10  # seconds

        for attempt in range(1, max_retries + 1):
            exit_code = self._run_command_once(command, repo_path)

            # If command succeeded, return immediately
            if exit_code == 0:
                return exit_code

            # Check if failure was due to network error
            if self._has_network_error():
                if attempt < max_retries:
                    msg = f"Network error detected (attempt {attempt}/{max_retries}). Retrying in {retry_delay}s..."
                    self.logger.warning(msg)
                    if self.log_file:
                        with open(self.log_file, 'a') as log_fh:
                            log_fh.write(f"\n[RETRY] {msg}\n\n")
                            log_fh.flush()
                    time.sleep(retry_delay)
                    continue
                else:
                    msg = f"Network error persisted after {max_retries} attempts. Giving up."
                    self.logger.error(msg)
                    if self.log_file:
                        with open(self.log_file, 'a') as log_fh:
                            log_fh.write(f"\n[FAILED] {msg}\n\n")
                            log_fh.flush()
                    return exit_code
            else:
                # Non-network error, don't retry
                return exit_code

        return exit_code

    @abstractmethod
    def execute(self, repo_path: Path, dry_run: bool = False) -> bool:
        """
        Execute the task.

        Args:
            repo_path: Path to the repository
            dry_run: If True, don't actually run commands

        Returns:
            True if execution succeeded, False otherwise
        """
        pass

    def can_run(self, all_tasks: Dict[str, 'BaseTask']) -> bool:
        """
        Check if this task can run based on dependency status.

        Args:
            all_tasks: Dictionary of all tasks in the pipeline

        Returns:
            True if all dependencies are satisfied
        """
        if self.status != TaskStatus.QUEUED:
            return False

        for dep_id in self.parents:
            if dep_id not in all_tasks:
                self.logger.warning(f"Dependency {dep_id} not found")
                return False

            dep_task = all_tasks[dep_id]

            # If dependency is still queued or running, we can't run yet
            if dep_task.status in [TaskStatus.QUEUED, TaskStatus.RUNNING]:
                return False

            # If dependency failed and we don't run on failure, we can't run
            if dep_task.status == TaskStatus.FAILED and not self.run_even_if_deps_fail:
                return False

        return True

    def duration(self) -> Optional[float]:
        """Get task duration in seconds"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    def mark_status_as(self, status: TaskStatus, reason: str = "") -> None:
        """
        Mark task with a specific status and optional reason.
        Also creates/removes marker files on disk.

        Args:
            status: The status to set
            reason: Optional reason/message for the status change

        Examples:
            task.mark_status_as(TaskStatus.RUNNING)  # Creates .RUNNING marker
            task.mark_status_as(TaskStatus.SKIPPED, "Image already exists")
            task.mark_status_as(TaskStatus.FAILED, "Build timeout")  # Creates .FAILED, removes .RUNNING
        """
        self._status = status
        if reason:
            self.error_message = reason

        # Set end_time for terminal statuses (SUCCESS, FAILED, KILLED)
        if status in [TaskStatus.PASSED, TaskStatus.FAILED, TaskStatus.KILLED]:
            if not self.end_time:
                self.end_time = time.time()

        # Create/cleanup marker files based on status
        if self.log_file:
            if status == TaskStatus.RUNNING:
                # Create .RUNNING marker
                running_marker = self.log_file.with_suffix(f'.{MARKER_RUNNING}')
                running_marker.touch()
                # If we are re-running a task, ensure stale terminal markers don't linger
                self._cleanup_marker(f'.{MARKER_PASSED}')
                self._cleanup_marker(f'.{MARKER_FAILED}')
                self._cleanup_marker(f'.{MARKER_KILLED}')
            elif status == TaskStatus.PASSED:
                # Create .PASSED marker, remove .RUNNING
                passed_marker = self.log_file.with_suffix(f'.{MARKER_PASSED}')
                passed_marker.touch()
                self._cleanup_marker(f'.{MARKER_RUNNING}')
                # Ensure mutually exclusive terminal markers
                self._cleanup_marker(f'.{MARKER_FAILED}')
                self._cleanup_marker(f'.{MARKER_KILLED}')
            elif status == TaskStatus.FAILED:
                # Create .FAILED marker, remove .RUNNING
                fail_marker = self.log_file.with_suffix(f'.{MARKER_FAILED}')
                fail_marker.touch()
                self._cleanup_marker(f'.{MARKER_RUNNING}')
                # Ensure mutually exclusive terminal markers
                self._cleanup_marker(f'.{MARKER_PASSED}')
                self._cleanup_marker(f'.{MARKER_KILLED}')
            elif status == TaskStatus.KILLED:
                # Create .KILLED marker, remove .RUNNING
                killed_marker = self.log_file.with_suffix(f'.{MARKER_KILLED}')
                killed_marker.touch()
                self._cleanup_marker(f'.{MARKER_RUNNING}')
                # Ensure mutually exclusive terminal markers
                self._cleanup_marker(f'.{MARKER_PASSED}')
                self._cleanup_marker(f'.{MARKER_FAILED}')

        # Log based on status
        status_logs = {
            TaskStatus.SKIPPED: f"Skipped: {reason}",
            TaskStatus.RUNNING: f"Running: {self.task_id}",
            TaskStatus.PASSED: f"Success: {self.task_id}",
            TaskStatus.FAILED: f"Failed: {reason}" if reason else f"Failed: {self.task_id}",
            TaskStatus.KILLED: f"Killed: {reason}" if reason else f"Killed: {self.task_id}",
            TaskStatus.QUEUED: f"Queued: {self.task_id}",
        }
        if status in status_logs:
            self.logger.info(status_logs[status])

    @abstractmethod
    def get_command(self, repo_path: Path) -> str:
        """
        Get the Unix command that would be executed for this task.

        Args:
            repo_path: Path to the repository

        Returns:
            The full Unix command string
        """
        pass

    def extract_docker_command_from_build_dryrun(self, repo_path: Path) -> Optional[str]:
        """
        Extract the underlying docker command (optional, for tasks that use docker).

        Args:
            repo_path: Path to the repository

        Returns:
            The docker command string, or None if not applicable
        """
        return None

    def check_input_image_exists(self) -> bool:
        """
        Check if the input Docker image exists locally.

        Returns:
            True if input image exists or no input image required, False otherwise
        """
        if not self.input_image:
            return True
        # BuildTask Dockerfiles are self-contained; docker build pulls base images internally.
        if self.task_id and str(self.task_id).endswith('-runtime-build'):
            return True

        try:
            result = subprocess.run(
                ["docker", "inspect", self.input_image],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
            return False

    def image_exists(self) -> bool:
        """Check if output Docker image already exists locally.

        For dev-build tasks (output ends with -dev-orig), also returns True
        if the final compressed image (-dev) already exists, since rebuilding
        dev-orig would be pointless when the downstream product is present.
        """
        if not self.output_image:
            return False

        def _docker_image_exists(tag: str) -> bool:
            try:
                result = subprocess.run(
                    ["docker", "inspect", tag],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
                return result.returncode == 0
            except (subprocess.CalledProcessError, OSError):
                return False

        if _docker_image_exists(self.output_image):
            return True

        if self.output_image.endswith("-dev-orig"):
            final_dev = self.output_image.replace("-dev-orig", "-dev")
            if _docker_image_exists(final_dev):
                self.logger.info(f"  Final image {final_dev} exists (skipping {self.output_image} rebuild)")
                return True

        return False

    def format_log_header(self, repo_path: Path) -> str:
        """
        Format the log file header for this task.
        Should be overridden by subclasses to customize format.

        Returns:
            Formatted header string
        """
        header = f"Task:        {self.task_id}\n"
        header += f"Description: {self.description}\n"
        header += f"Command:     {sanitize_token(self.get_command(repo_path))}\n"
        header += f"Started:     {datetime.now().isoformat()}\n"
        header += "-" * 80 + "\n"
        return header

    def format_log_footer(self, success: bool) -> str:
        """
        Format the log file footer for this task.

        Args:
            success: Whether the task succeeded

        Returns:
            Formatted footer string
        """
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
        else:
            duration = 0.0

        ended_str = datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else datetime.now().isoformat()

        footer = f"\n"
        footer += f"-" * 80 + "\n"
        footer += f"Task:     {self.task_id}\n"
        footer += f"Duration: {duration:.2f}s\n"
        footer += f"Ended:    {ended_str}\n"
        footer += f"Status:   {'SUCCESS' if success else 'FAILED'} (exit code: {self.exit_code if self.exit_code is not None else (0 if success else 1)})\n"
        footer += f"-" * 80 + "\n"
        return footer

    def passed_previously(self) -> bool:
        """
        Check if this task passed in a previous run by looking for .PASSED marker.

        Returns:
            True if task passed previously, False otherwise
        """
        if not self.log_file:
            return False

        pass_marker = self.log_file.with_suffix(f'.{MARKER_PASSED}')
        return pass_marker.exists()

    def load_previous_results(self) -> Optional[Dict[str, Any]]:
        """
        Load results from a previous run by parsing the log file.

        Returns:
            Dictionary with 'status', 'duration', 'exit_code' or None if no previous run
        """
        if not self.log_file or not self.log_file.exists():
            return None

        return parse_log_file_results(self.log_file)

    def cleanup_markers(self) -> None:
        """
        Clean up all marker files (.PASSED, .FAILED, .RUNNING, .KILLED) and log file for this task.
        Should be called before starting task execution.
        """
        if not self.log_file:
            return

        files_to_clean = [
            self.log_file,
            self.log_file.with_suffix(f'.{MARKER_PASSED}'),
            self.log_file.with_suffix(f'.{MARKER_FAILED}'),
            self.log_file.with_suffix(f'.{MARKER_RUNNING}'),
            self.log_file.with_suffix(f'.{MARKER_KILLED}'),
        ]

        for file_to_clean in files_to_clean:
            if file_to_clean.exists():
                try:
                    file_to_clean.unlink()
                except (OSError, PermissionError) as e:
                    logger = logging.getLogger("task")
                    logger.warning(f"Failed to remove {file_to_clean}: {e}")

    def _cleanup_marker(self, suffix: str) -> None:
        """Helper to remove a marker file if it exists."""
        if not self.log_file:
            return
        marker = self.log_file.with_suffix(suffix)
        if marker.exists():
            marker.unlink()


# ==============================================================================
# BUILD TASK
# ==============================================================================

@dataclass
class BuildTask(BaseTask):
    """
    Task for building Docker images.

    Executes build.sh directly (no longer parses --dry-run output).
    """

    def __post_init__(self):
        super().__post_init__()

    def prepare(self, repo_path: Path, commit_sha_9: str) -> None:
        """
        Prepare build command.

        Generates the docker build command with appropriate arguments.
        """
        # TODO: Implement command generation
        # Will use build.sh script from the repo
        pass

    def execute(self, repo_path: Path, dry_run: bool = False) -> bool:
        """
        Execute Docker build.

        Runs build.sh directly (no longer extracts/filters commands).
        """
        if dry_run:
            self.logger.info("Dry-run mode: skipping actual execution")
            return True

        # Run build.sh directly
        command = self.get_command(repo_path)
        exit_code = self._run_command(command, repo_path)
        return exit_code == 0

    def get_image_size(self) -> Optional[str]:
        """Get the size of the Docker image in human-readable format"""
        if not self.output_image:
            return None

        return get_docker_image_size(self.output_image)

    def get_command(self, repo_path: Path) -> str:
        """Get the build.sh command to execute"""
        # Return the build.sh command stored in self.command
        # e.g., "/path/to/build.sh --framework vllm --target runtime --no-tag-latest"
        return self.command

    def format_log_header(self, repo_path: Path) -> str:
        """
        Format the log file header for BuildTask.

        Returns:
            Formatted header string
        """
        header = f"Task:        {self.task_id}\n"
        header += f"Description: {self.description}\n"
        header += f"Command:     {sanitize_token(self.get_command(repo_path))}\n"
        header += f"Started:     {datetime.now().isoformat()}\n"
        header += "-" * 80 + "\n"
        return header


# ==============================================================================
# COMMAND TASK (Generic task for executing commands)
# ==============================================================================

@dataclass
class CommandTask(BaseTask):
    """
    Generic task for executing commands (replaces CompilationTask, ChownTask, SanityCheckTask).

    This is a simplified task that just runs a command and logs output.
    Can be used for compilation, chown, sanity checks, or any other command execution.

    Additional attributes:
        ignore_exit_code: If True, task succeeds even if command exits with non-zero
    """

    ignore_exit_code: bool = False

    def __post_init__(self):
        super().__post_init__()
        # Task type is already set by the factory function

    def prepare(self, repo_path: Path, commit_sha_9: str) -> None:
        """Prepare command for execution."""
        pass

    def execute(self, repo_path: Path, dry_run: bool = False) -> bool:
        """Execute the command."""
        if dry_run:
            self.logger.info("Dry-run mode: skipping actual execution")
            return True

        command = self.get_command(repo_path)
        exit_code = self._run_command(command, repo_path)

        # If ignore_exit_code is set, always return success
        if self.ignore_exit_code:
            return True

        return exit_code == 0

    def get_command(self, repo_path: Path) -> str:
        """Get the command to execute (from self.command)."""
        return self.command


# ==============================================================================
# PIPELINE BUILDER
# ==============================================================================

class BuildPipeline:
    """
    Manages the complete build pipeline with dependency resolution.

    Responsibilities:
    - Store all tasks
    - Resolve dependencies
    - Determine execution order
    - Track overall pipeline status
    """

    def __init__(self):
        self.tasks: Dict[str, BaseTask] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_task(self, task: BaseTask) -> None:
        """Add a task to the pipeline"""
        self.tasks[task.task_id] = task

    def _build_children_relationships(self) -> None:
        """Build children relationships after all tasks are added"""
        for task in self.tasks.values():
            for parent_id in task.parents:
                if parent_id in self.tasks:
                    if task.task_id not in self.tasks[parent_id].children:
                        self.tasks[parent_id].children.append(task.task_id)

    def add_dependency(self, child_id: str, parent_id: str) -> None:
        """Establish parent-child dependency"""
        if child_id in self.tasks and parent_id in self.tasks:
            if parent_id not in self.tasks[child_id].parents:
                self.tasks[child_id].parents.append(parent_id)
            if child_id not in self.tasks[parent_id].children:
                self.tasks[parent_id].children.append(child_id)

    def get_ready_tasks(self) -> List[BaseTask]:
        """Get all tasks ready to run (dependencies satisfied)"""
        return [task for task in self.tasks.values() if task.can_run(self.tasks)]

    def get_levels(self) -> List[List[BaseTask]]:
        """
        Group tasks by execution level (for parallel execution).

        Level 0: Tasks with no dependencies
        Level 1: Tasks depending only on level 0
        Level N: Tasks depending on level N-1 or earlier

        Returns:
            List of task lists, one per level
        """
        levels: List[List[BaseTask]] = []
        remaining = set(self.tasks.keys())

        while remaining:
            # Find tasks whose dependencies are all satisfied
            level_tasks = []
            for task_id in list(remaining):
                task = self.tasks[task_id]
                if all(dep_id not in remaining for dep_id in task.parents):
                    level_tasks.append(task)
                    remaining.remove(task_id)

            if not level_tasks:
                # Circular dependency or missing dependency
                self.logger.error(f"Cannot resolve dependencies for: {remaining}")
                break

            levels.append(level_tasks)

        return levels

    def visualize_tree(self) -> str:
        """
        Generate ASCII tree visualization of the pipeline.

        Returns:
            String containing the tree visualization
        """
        levels = self.get_levels()
        output = []
        output.append("=" * 80)
        output.append("DEPENDENCY TREE")
        output.append("=" * 80)

        for level_num, level_tasks in enumerate(levels):
            output.append(f"\nLevel {level_num}:")
            for task in level_tasks:
                status_symbol = {
                    TaskStatus.QUEUED: "⏳",
                    TaskStatus.RUNNING: "🔄",
                    TaskStatus.PASSED: "✅",
                    TaskStatus.FAILED: "❌",
                    TaskStatus.SKIPPED: "⏭️",
                }[task.status]

                output.append(f"  {status_symbol} {task.task_id} ({task.__class__.__name__})")

                if task.parents:
                    output.append(f"      └─ depends on: {', '.join(task.parents)}")

        return "\n".join(output)

    def get_status_summary(self) -> Dict[str, int]:
        """Get count of tasks in each status"""
        summary = {status.value: 0 for status in TaskStatus}
        for task in self.tasks.values():
            summary[task.status.value] += 1
        return summary


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="DynamoDockerBuilder V2 - Build orchestration for Dynamo Docker images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Repository configuration
    parser.add_argument(
        "--repo-path",
        type=Path,
        required=True,
        help="Path to the Dynamo repository",
    )
    parser.add_argument(
        "--commit-sha",
        type=str,
        dest="commit_sha_9",
        help="Git commit SHA to build (default: current HEAD)",
    )
    parser.add_argument(
        "--pull-latest",
        action="store_true",
        help="Pull latest code from main branch before building",
    )

    # Report generation
    parser.add_argument(
        "--generate-html-only",
        action="store_true",
        help="Generate the HTML report and exit without running any builds/sanity/compilation",
    )
    parser.add_argument(
        "--html-out",
        type=Path,
        default=None,
        help="Path to write the HTML report (default: <repo-path>/build.html)",
    )

    # Framework and target selection
    parser.add_argument(
        "-f", "--framework",
        action="append",
        help=(
            "Framework(s) to build/test (none, vllm, sglang, trtllm). "
            "Can specify multiple times and/or as a comma-separated list. "
            "Default: all (but if --sanity-check-only is set and --framework is omitted, defaults to 'none')."
        ),
    )
    parser.add_argument(
        "--target",
        default="runtime,dev,local-dev",
        help="Comma-separated list of build targets (default: all)",
    )

    # Execution options
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Execute tasks in parallel when possible",
    )
    parser.add_argument(
        "--dry-run", "--dryrun",
        action="store_true",
        dest="dry_run",
        help="Show what would be executed without running commands",
    )
    parser.add_argument(
        "--run-no-matter-what", "--run-ignore-lock",
        action="store_true",
        dest="run_no_matter_what",
        help="Force run regardless of: (1) lock file from a previous run, (2) unchanged Docker image SHA, (3) unchanged compilation SHA (.last_compilation_sha)",
    )
    parser.add_argument(
        "--reuse-dev-if-image-exists",
        action="store_true",
        dest="reuse_dev_if_image_exists",
        help=(
            "Compute image SHA; if the local-dev image for that SHA exists locally, "
            "only run local-dev compilation and sanity check (skip all builds). "
            "Otherwise run the full build."
        ),
    )

    # Build options
    parser.add_argument(
        "--sanity-check-only",
        action="store_true",
        help="Only run sanity checks, skip builds (defaults to framework 'none' if --framework is omitted)",
    )
    parser.add_argument(
        "--skip-action-if-already-passed",
        "--skip",
        action="store_true",
        help="Skip any task if it has already passed (checks for .PASSED marker) or if build output image already exists",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip all compilation tasks",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        default=False,
        help="Skip pushing dev images to GitLab registry (upload is enabled by default).",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        default=False,
        help="Skip compressing dev images (compress is enabled by default).",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        default=False,
        help="Skip all build tasks; run only compilation and sanity checks against existing images.",
    )
    parser.add_argument(
        "--image-sha",
        dest="image_tag_override",
        type=str,
        default=None,
        help=(
            "Image SHA to use for existing Docker image tags "
            "with --no-build. E.g. --image-sha C62194.6d3e0137c uses dynamo:C62194.6d3e0137c-vllm-cuda12.9-dev. "
            "If omitted with --no-build, auto-detects from the latest local Docker images."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers (default: CPU count, currently %(default)s)",
    )

    # Output options
    parser.add_argument(
        "--html-path",
        type=str,
        default=DEFAULT_HTML_PATH,
        help=f"Base URL path for HTML reports in emails (default: {DEFAULT_HTML_PATH})",
    )
    parser.add_argument(
        "--hostname",
        type=str,
        default=DEFAULT_HOSTNAME,
        help=f"Hostname for absolute URLs in email reports (default: {DEFAULT_HOSTNAME})",
    )
    parser.add_argument(
        "--email",
        help="Email address to send report to",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--show-tree",
        action="store_true",
        help="Show dependency tree visualization",
    )

    return parser.parse_args()


def sanitize_token(text: str) -> str:
    """
    Replace sensitive tokens with asterisks for security.

    Args:
        text: Text that may contain tokens

    Returns:
        Text with tokens replaced by asterisks
    """
    # Pattern for GitHub tokens (gho_..., ghp_..., etc.)
    text = re.sub(r'(gho_|ghp_|ghs_|ghr_)[A-Za-z0-9_]{36,}', r'\1' + '*' * 40, text)

    # Pattern for --build-arg GITHUB_TOKEN=...
    text = re.sub(r'(--build-arg\s+GITHUB_TOKEN=)[^\s]+', r'\1' + '*' * 40, text)

    return text


def get_docker_image_size(image_name: str) -> Optional[str]:
    """
    Get Docker image size using DockerUtils from common.py.

    Args:
        image_name: Docker image name (e.g., "dynamo-base:f2a3c638d")

    Returns:
        Human-readable size string (e.g., "8.5 GB") or None if image not found
    """
    try:
        docker_utils = DockerUtils(dry_run=False, verbose=False)
        image_info = docker_utils.get_image_info(image_name)
        if image_info:
            return image_info.size_human
        return None
    except (subprocess.CalledProcessError, OSError, ValueError):
        # Image doesn't exist or docker command failed
        return None


def setup_logging(verbose: bool = False) -> None:
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %a %H:%M:%S %Z",
    )


def parse_log_file_results(log_file: Path) -> Optional[LogParseResult]:
    """
    Parse a log file to extract task results (status, duration, exit code).

    Args:
        log_file: Path to log file

    Returns:
        LogParseResult with status, duration, exit_code or None if parsing fails
    """
    if not log_file.exists():
        return None

    try:
        # Check for marker files first (.PASSED, .FAILED, .KILLED)
        pass_marker = log_file.with_suffix(f'.{MARKER_PASSED}')
        fail_marker = log_file.with_suffix(f'.{MARKER_FAILED}')
        killed_marker = log_file.with_suffix(f'.{MARKER_KILLED}')

        status = None
        if pass_marker.exists():
            status = TaskStatus.PASSED
        elif fail_marker.exists():
            status = TaskStatus.FAILED
        elif killed_marker.exists():
            status = TaskStatus.KILLED

        # Read the last 30 lines to find the results section
        with open(log_file, 'r') as f:
            lines = f.readlines()
            last_lines = lines[-30:] if len(lines) > 30 else lines

        # Look for duration and exit code
        duration = None
        exit_code = None

        for line in last_lines:
            line = line.strip()
            if line.startswith('Duration:'):
                # Duration: 3.83s
                duration_str = line.split('Duration:')[1].strip().rstrip('s')
                try:
                    duration = float(duration_str)
                except ValueError:
                    pass
            elif line.startswith('Exit code:'):
                # Exit code: 0
                try:
                    exit_code = int(line.split('Exit code:')[1].strip())
                except ValueError:
                    pass
            elif line.startswith('Status:') and status is None:
                # Status: SUCCESS or FAILED (only if not already determined by marker files)
                status_str = line.split('Status:')[1].strip()
                status = TaskStatus.PASSED if status_str == 'SUCCESS' else TaskStatus.FAILED

        if status is not None:
            return LogParseResult(
                status=status,
                duration=duration,
                exit_code=exit_code,
            )

        return None

    except (OSError, ValueError) as e:
        logger = logging.getLogger("log_parser")
        logger.debug(f"Failed to parse log file {log_file}: {e}")
        return None


def execute_task_sequential(
    all_tasks: Dict[str, 'BaseTask'],
    executed_tasks: Set[str],
    failed_tasks: Set[str],
    task_id: str,
    repo_path: Path,
    image_and_commit_sha: str,
    log_dir: Optional[Path] = None,
    log_date: Optional[str] = None,
    dry_run: bool = False,
    skip_action_if_already_passed: bool = False,
    no_compile: bool = False,
    enable_dev_upload: bool = False,
    enable_dev_compress: bool = False,
) -> bool:
    """
    Execute a single task and its dependencies recursively (sequential mode).

    Args:
        all_tasks: Dictionary of all tasks
        executed_tasks: Set of already executed task IDs (modified in place)
        failed_tasks: Set of failed task IDs (modified in place)
        task_id: ID of task to execute
        repo_path: Path to repository
        image_and_commit_sha: Git commit SHA
        log_dir: Directory for log files (None in dry-run)
        log_date: Date string for log files (None in dry-run)
        dry_run: If True, only print commands without executing
        skip_action_if_already_passed: If True, skip any task if .PASSED marker exists or if build image exists
        no_compile: If True, skip all compilation tasks
        enable_dev_upload: If True, run dev-upload tasks; otherwise skip them and report as skipped

    Returns:
        True if task succeeded, False if failed
    """
    logger = logging.getLogger("executor")

    # Skip if already executed
    if task_id in executed_tasks:
        return task_id not in failed_tasks

    task = all_tasks[task_id]

    # Execute dependencies first
    for parent_id in task.parents:
        if not execute_task_sequential(
            all_tasks, executed_tasks, failed_tasks, parent_id,
            repo_path, image_and_commit_sha, log_dir, log_date, dry_run, skip_action_if_already_passed,
            no_compile, enable_dev_upload, enable_dev_compress
        ):
            # Parent failed
            if not task.run_even_if_deps_fail:
                if dry_run:
                    logger.info(f"Would skip {task_id} due to failed dependency {parent_id}")
                else:
                    logger.info(f"Skipping {task_id} due to failed dependency {parent_id}")
                task.mark_status_as(TaskStatus.SKIPPED, f"Failed dependency: {parent_id}")
                executed_tasks.add(task_id)
                failed_tasks.add(task_id)
                return False

    executed_tasks.add(task_id)

    # Check if we should skip compilation tasks (--no-compile)
    if no_compile and isinstance(task, CommandTask) and 'compilation' in task_id.lower():
        logger.info(f"⊘ Skipping {task_id}: --no-compile flag set")
        task.mark_status_as(TaskStatus.SKIPPED, "--no-compile flag set")

        # Process children
        for child_id in task.children:
            if child_id not in executed_tasks:
                execute_task_sequential(
                    all_tasks, executed_tasks, failed_tasks, child_id,
                    repo_path, image_and_commit_sha, log_dir, log_date, dry_run, skip_action_if_already_passed,
                    no_compile, enable_dev_upload, enable_dev_compress
                )

        return True

    # Check if we should skip dev-compress or dev-compress-sanity (--no-compress)
    if not enable_dev_compress and (task_id.endswith("-dev-compress") or task_id.endswith("-dev-compress-sanity")):
        logger.info(f"⊘ Skipping {task_id}: compress disabled (--no-compress)")
        # When skipping dev-compress, retag -dev-orig to -dev so downstream tasks have the expected image
        if task_id.endswith("-dev-compress") and task.input_image and task.output_image:
            orig_tag = task.input_image
            dev_tag = task.output_image
            logger.info(f"  Retagging {orig_tag} -> {dev_tag} (compress skipped)")
            rc = subprocess.call(["docker", "tag", orig_tag, dev_tag])
            if rc != 0:
                logger.warning(f"  Failed to retag {orig_tag} -> {dev_tag} (exit code {rc})")
        task.mark_status_as(TaskStatus.SKIPPED, "Compress disabled (use default to enable)")
        executed_tasks.add(task_id)
        for child_id in task.children:
            if child_id not in executed_tasks:
                execute_task_sequential(
                    all_tasks, executed_tasks, failed_tasks, child_id,
                    repo_path, image_and_commit_sha, log_dir, log_date, dry_run, skip_action_if_already_passed,
                    no_compile, enable_dev_upload, enable_dev_compress
                )
        return True

    # Check if we should skip dev-upload (--no-upload)
    if not enable_dev_upload and task_id.endswith("-dev-upload"):
        logger.info(f"⊘ Skipping {task_id}: upload disabled (--no-upload)")
        task.mark_status_as(TaskStatus.SKIPPED, "Upload disabled (pass default to enable)")
        if log_dir and log_date:
            update_frameworks_data_cache(task, use_absolute_urls=False)

        # Process children
        for child_id in task.children:
            if child_id not in executed_tasks:
                execute_task_sequential(
                    all_tasks, executed_tasks, failed_tasks, child_id,
                    repo_path, image_and_commit_sha, log_dir, log_date, dry_run, skip_action_if_already_passed,
                    no_compile, enable_dev_upload, enable_dev_compress
                )

        return True

    # DRY-RUN: Just print what would be executed
    if dry_run:
        if task.status == TaskStatus.SKIPPED:
            logger.info(f"[{task_id}] SKIPPED: {task.error_message or 'pre-skipped'}")
        else:
            logger.info(f"[{task_id}] {task.description}")

            # For BuildTask, show both build.sh command AND docker build command
            if isinstance(task, BuildTask):
                logger.info(f"  1. {sanitize_token(task.command)}")
                docker_cmd = task.extract_docker_command_from_build_dryrun(repo_path)
                if docker_cmd:
                    docker_lines = docker_cmd.split('\n')
                    for i, line in enumerate(docker_lines):
                        if i == 0:
                            logger.info(f"  2. {sanitize_token(line)}")
                        else:
                            logger.info(f"     {sanitize_token(line)}")
            else:
                logger.info(f"→ {sanitize_token(task.get_command(repo_path))}")
            logger.info("")

            # Mark as success for dry-run traversal
            task.mark_status_as(TaskStatus.PASSED)

        # Process children
        for child_id in task.children:
            if child_id not in executed_tasks:
                execute_task_sequential(
                    all_tasks, executed_tasks, failed_tasks, child_id,
                    repo_path, image_and_commit_sha, log_dir, log_date, dry_run, skip_action_if_already_passed,
                    no_compile, enable_dev_upload, enable_dev_compress
                )

        return True

    # ACTUAL EXECUTION
    # Setup log file: YYYY-MM-DD.{image_and_commit_sha}.{task-id}.log
    if log_dir is None or log_date is None:
        raise ValueError("log_dir and log_date must be set for actual execution")
    task.set_log_file_path(log_dir, log_date, image_and_commit_sha)

    # Check if we should skip any task if it has already passed.
    # This must run BEFORE the input image check: if the task already passed,
    # we don't need the input image (it may have been cleaned up).
    # IMPORTANT: A stale .PASSED marker is NOT sufficient to skip if the task produces
    # a Docker image. The output image must still exist locally (users may have pruned).
    if skip_action_if_already_passed:
        if isinstance(task, BuildTask):
            if task.image_exists():
                skip_task = True
                reason = "Docker image already exists"
            elif task.passed_previously():
                # Don't skip: marker may be stale if images were pruned.
                skip_task = False
                reason = ""
                logger.info(f"Found .PASSED marker for {task_id} but output image is missing; rebuilding")
            else:
                skip_task = False
                reason = ""
        else:
            has_output = bool(task.output_image)
            output_ok = task.image_exists() if has_output else True
            if task.passed_previously() and output_ok:
                skip_task = True
                reason = "Already passed previously"
            elif task.passed_previously() and not output_ok:
                skip_task = False
                reason = ""
                logger.info(f"Found .PASSED marker for {task_id} but output image is missing; re-running")
            else:
                skip_task = False
                reason = ""

        if skip_task:
            logger.info(f"⊘ Skipping {task_id}: {reason}")
            task.mark_status_as(TaskStatus.SKIPPED, reason)
            executed_tasks.add(task_id)

            # Process children
            for child_id in task.children:
                if child_id not in executed_tasks:
                    execute_task_sequential(
                        all_tasks, executed_tasks, failed_tasks, child_id,
                        repo_path, image_and_commit_sha, log_dir, log_date, dry_run, skip_action_if_already_passed,
                        no_compile, enable_dev_upload
                    )

            return True

    # Check if input image exists (required for tasks that depend on previous builds)
    if not task.check_input_image_exists():
        head_changed_warning = check_sha_changed(repo_path, task_id)
        if head_changed_warning:
            logger.warning(head_changed_warning)

        logger.error(f"✗ Skipping {task_id}: Input image missing ({task.input_image})")
        task.mark_status_as(TaskStatus.FAILED, f"Input image not found: {task.input_image}")
        executed_tasks.add(task_id)
        failed_tasks.add(task_id)

        with open(task.log_file, 'w') as log_fh:
            log_fh.write(f"Task: {task_id}\n")
            log_fh.write(f"Description: {task.description}\n")
            log_fh.write(f"Error: Input image not found: {task.input_image}\n")
            if head_changed_warning:
                log_fh.write("\n")
                log_fh.write(head_changed_warning)
                log_fh.write("\n")
            log_fh.write("=" * 80 + "\n")

        update_frameworks_data_cache(task, use_absolute_urls=False)

        if log_dir and log_date:
            try:
                html_content = generate_html_report(
                    all_tasks=all_tasks,
                    repo_path=repo_path,
                    image_and_commit_sha=image_and_commit_sha,
                    log_dir=log_dir,
                    date_str=log_date,
                    use_absolute_urls=False,
                )
                _write_report_and_json(html_content, all_tasks, repo_path, image_and_commit_sha, log_dir, log_date)
            except Exception as e:
                logger.warning(f"Failed to generate HTML report after image-not-found failure: {e}")

        return False

    # Execute this task
    logger.info(f"Executing: {task_id} ({task.description})")
    logger.info(f"  Command:           {sanitize_token(task.get_command(repo_path))}")
    if task.input_image:
        logger.info(f"  Input image:       {task.input_image}")
    if task.output_image:
        logger.info(f"  Output image:      {task.output_image}")
    logger.info(f"  Log:               {task.log_file}")

    # Clean up existing files for THIS specific task only (now that we know it will run)
    task.cleanup_markers()

    task.mark_status_as(TaskStatus.RUNNING)  # Also creates .RUNNING marker file
    task.start_time = time.time()

    # Write log file header first (so users see content immediately when clicking log link)
    with open(task.log_file, 'w') as log_fh:
        log_fh.write(task.format_log_header(repo_path))
        log_fh.flush()

    # Update frameworks data cache with RUNNING status
    update_frameworks_data_cache(task, use_absolute_urls=False)

    # Generate HTML report showing task as RUNNING (with log link)
    if log_dir and log_date:
        try:
            html_content = generate_html_report(
                all_tasks=all_tasks,
                repo_path=repo_path,
                image_and_commit_sha=image_and_commit_sha,
                log_dir=log_dir,
                date_str=log_date,
                use_absolute_urls=False,
            )
            _write_report_and_json(html_content, all_tasks, repo_path, image_and_commit_sha, log_dir, log_date)
        except Exception as e:
            logger.warning(f"Failed to generate HTML report before task execution: {e}")

    try:

        # Execute the task (will append to log file)
        success = task.execute(repo_path, dry_run=False)

        task.end_time = time.time()

        # Append execution summary to log file
        with open(task.log_file, 'a') as log_fh:
            log_fh.write(task.format_log_footer(success))

        if success:
            task.mark_status_as(TaskStatus.PASSED)  # Also creates .PASSED marker, removes .RUNNING
            duration = task.end_time - task.start_time if task.start_time and task.end_time else 0.0
            logger.info(f"✓ Completed: {task_id} ({duration:.2f}s)")
        else:
            task.mark_status_as(TaskStatus.FAILED)  # Also creates .FAILED marker, removes .RUNNING
            logger.error(f"✗ Failed: {task_id}")
            failed_tasks.add(task_id)
    except Exception as e:
        task.end_time = time.time()
        task.mark_status_as(TaskStatus.FAILED, str(e))  # Also creates .FAILED marker, removes .RUNNING
        logger.error(f"✗ Failed: {task_id} - {e}")
        failed_tasks.add(task_id)

    # Update frameworks data cache with task results
    update_frameworks_data_cache(task, use_absolute_urls=False)

    # Generate incremental HTML report after task completes
    if log_dir and log_date:
        try:
            html_content = generate_html_report(
                all_tasks=all_tasks,
                repo_path=repo_path,
                image_and_commit_sha=image_and_commit_sha,
                log_dir=log_dir,
                date_str=log_date,
                use_absolute_urls=False,
            )
            _write_report_and_json(html_content, all_tasks, repo_path, image_and_commit_sha, log_dir, log_date)
        except Exception as e:
            logger.warning(f"Failed to generate incremental HTML report: {e}")

    # After successfully executing this task, execute all children
    if task.status == TaskStatus.PASSED:
        for child_id in task.children:
            if child_id not in executed_tasks:
                execute_task_sequential(
                    all_tasks, executed_tasks, failed_tasks, child_id,
                    repo_path, image_and_commit_sha, log_dir, log_date, dry_run, skip_action_if_already_passed,
                    no_compile, enable_dev_upload, enable_dev_compress
                )

    return task.status == TaskStatus.PASSED


def execute_task_parallel(
    all_tasks: Dict[str, 'BaseTask'],
    root_tasks: List[str],
    repo_path: Path,
    image_and_commit_sha: str,
    log_dir: Optional[Path] = None,
    log_date: Optional[str] = None,
    dry_run: bool = False,
    skip_if_passed: bool = False,
    no_compile: bool = False,
    enable_dev_upload: bool = False,
    enable_dev_compress: bool = False,
    max_workers: int = 4,
) -> Tuple[Set[str], Set[str]]:
    """
    Execute tasks in parallel respecting dependencies.

    Uses a thread pool to execute tasks that have all dependencies satisfied.
    Tasks are executed as soon as their dependencies complete.

    Args:
        all_tasks: Dictionary of all tasks
        root_tasks: List of root task IDs (no dependencies)
        repo_path: Path to repository
        image_and_commit_sha: Git commit SHA
        log_dir: Directory for log files (None in dry-run)
        log_date: Date string for log files (None in dry-run)
        dry_run: If True, only print commands without executing
        skip_if_passed: If True, skip any task if .PASSED marker exists or if build image exists
        no_compile: If True, skip all compilation tasks
        enable_dev_upload: If True, run dev-upload tasks; otherwise skip them and report as skipped
        max_workers: Maximum number of parallel threads

    Returns:
        Tuple of (executed_tasks, failed_tasks) sets
    """
    logger = logging.getLogger("executor")

    executed_tasks = set()
    failed_tasks = set()
    lock = threading.Lock()

    # Flag to stop the periodic HTML updater thread
    stop_html_updater = threading.Event()

    def periodic_html_updater():
        """Background thread to update HTML report every 5 seconds"""
        while not stop_html_updater.is_set():
            # Wait for 5 seconds (or until stop signal)
            if stop_html_updater.wait(5):
                break  # Stop signal received

            # Generate HTML report
            if log_dir and log_date and not dry_run:
                try:
                    # Update frameworks cache with fresh elapsed times for running tasks
                    for task_id, task in all_tasks.items():
                        if task.status == TaskStatus.RUNNING:
                            update_frameworks_data_cache(task, use_absolute_urls=False)

                    html_content = generate_html_report(
                        all_tasks=all_tasks,
                        repo_path=repo_path,
                        image_and_commit_sha=image_and_commit_sha,
                        log_dir=log_dir,
                        date_str=log_date,
                        use_absolute_urls=False,
                    )
                    _write_report_and_json(html_content, all_tasks, repo_path, image_and_commit_sha, log_dir, log_date)
                except Exception as e:
                    # Silently continue on error to avoid spam
                    pass

    # Start periodic HTML updater thread
    html_updater_thread = None
    if log_dir and log_date and not dry_run:
        html_updater_thread = threading.Thread(target=periodic_html_updater, daemon=True)
        html_updater_thread.start()

    # Setup signal handlers to mark running tasks as KILLED on interrupt
    def signal_handler(signum, frame):
        """Handle SIGTERM/SIGINT by marking running tasks as KILLED"""
        logger = logging.getLogger("signal")
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        logger.warning(f"\nReceived {sig_name}, marking running tasks as KILLED...")

        # Stop the background HTML updater thread first
        if html_updater_thread is not None:
            logger.info("  Stopping background HTML updater...")
            stop_html_updater.set()
            html_updater_thread.join(timeout=2.0)  # Wait up to 2 seconds

        # Terminate all running subprocesses
        with _subprocesses_lock:
            subprocesses_to_terminate = list(_running_subprocesses)

        if subprocesses_to_terminate:
            logger.info(f"  Terminating {len(subprocesses_to_terminate)} running subprocess(es)...")
            for proc in subprocesses_to_terminate:
                try:
                    proc.terminate()
                except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
                    pass  # Ignore errors if process already exited

            # Wait up to 2 seconds for processes to terminate
            time.sleep(2)

            # Force kill any remaining processes
            for proc in subprocesses_to_terminate:
                try:
                    if proc.poll() is None:  # Still running
                        proc.kill()
                except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
                    pass

        # Mark all running tasks as KILLED
        for task_id, task in all_tasks.items():
            if task.status == TaskStatus.RUNNING:
                task.mark_status_as(TaskStatus.KILLED, f"Interrupted by {sig_name}")
                update_frameworks_data_cache(task, use_absolute_urls=False)
                logger.info(f"  Marked {task_id} as KILLED")

        # Generate final HTML report
        if log_dir and log_date and not dry_run:
            try:
                html_content = generate_html_report(
                    all_tasks=all_tasks,
                    repo_path=repo_path,
                    image_and_commit_sha=image_and_commit_sha,
                    log_dir=log_dir,
                    date_str=log_date,
                    use_absolute_urls=False,
                )
                _write_report_and_json(html_content, all_tasks, repo_path, image_and_commit_sha, log_dir, log_date)
                logger.info(f"  Generated final HTML report: {log_dir / image_and_commit_sha / 'report.html'}")
            except Exception as e:
                logger.error(f"  Failed to generate final HTML report: {e}")

        sys.exit(1)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    def can_execute(task_id: str) -> bool:
        """Check if all dependencies have finished (passed, failed, or skipped)"""
        task = all_tasks[task_id]
        for parent_id in task.parents:
            if parent_id not in executed_tasks:
                return False
            # Don't block here on failed parents -- let execute_single_task
            # handle the skip/fail propagation so tasks don't get stuck in pending.
        return True

    def execute_single_task(task_id: str) -> bool:
        """Execute a single task (thread-safe)"""
        task = all_tasks[task_id]

        # Check if we should skip due to failed dependency
        with lock:
            for parent_id in task.parents:
                if parent_id in failed_tasks and not task.run_even_if_deps_fail:
                    if dry_run:
                        logger.info(f"Would skip {task_id} due to failed dependency {parent_id}")
                    else:
                        logger.info(f"Skipping {task_id} due to failed dependency {parent_id}")
                    task.mark_status_as(TaskStatus.SKIPPED, f"Failed dependency: {parent_id}")
                    executed_tasks.add(task_id)
                    failed_tasks.add(task_id)
                    return False

        # Check if we should skip compilation tasks (--no-compile)
        if no_compile and isinstance(task, CommandTask) and 'compilation' in task_id.lower():
            with lock:
                logger.info(f"⊘ Skipping {task_id}: --no-compile flag set")
            task.mark_status_as(TaskStatus.SKIPPED, "--no-compile flag set")
            with lock:
                executed_tasks.add(task_id)
            return True

        # Check if we should skip dev-compress or dev-compress-sanity (--no-compress)
        if not enable_dev_compress and (task_id.endswith("-dev-compress") or task_id.endswith("-dev-compress-sanity")):
            with lock:
                logger.info(f"⊘ Skipping {task_id}: compress disabled (--no-compress)")
            # When skipping dev-compress, retag -dev-orig to -dev so downstream tasks have the expected image
            if task_id.endswith("-dev-compress") and task.input_image and task.output_image:
                orig_tag = task.input_image
                dev_tag = task.output_image
                with lock:
                    logger.info(f"  Retagging {orig_tag} -> {dev_tag} (compress skipped)")
                rc = subprocess.call(["docker", "tag", orig_tag, dev_tag])
                if rc != 0:
                    with lock:
                        logger.warning(f"  Failed to retag {orig_tag} -> {dev_tag} (exit code {rc})")
            task.mark_status_as(TaskStatus.SKIPPED, "Compress disabled (use default to enable)")
            with lock:
                executed_tasks.add(task_id)
            return True

        # Check if we should skip dev-upload (--no-upload)
        if not enable_dev_upload and task_id.endswith("-dev-upload"):
            with lock:
                logger.info(f"⊘ Skipping {task_id}: upload disabled (--no-upload)")
            task.mark_status_as(TaskStatus.SKIPPED, "Upload disabled (pass default to enable)")
            if log_dir and log_date:
                update_frameworks_data_cache(task, use_absolute_urls=False)
            with lock:
                executed_tasks.add(task_id)
            return True

        # DRY-RUN: Just print
        if dry_run:
            if task.status == TaskStatus.SKIPPED:
                with lock:
                    logger.info(f"[{task_id}] SKIPPED: {task.error_message or 'pre-skipped'}")
                    executed_tasks.add(task_id)
                return True

            with lock:
                logger.info(f"[{task_id}] {task.description}")
                if isinstance(task, BuildTask):
                    logger.info(f"  1. {sanitize_token(task.command)}")
                    docker_cmd = task.extract_docker_command_from_build_dryrun(repo_path)
                    if docker_cmd:
                        for i, line in enumerate(docker_cmd.split('\n')):
                            logger.info(f"  2. {sanitize_token(line)}" if i == 0 else f"     {sanitize_token(line)}")
                else:
                    logger.info(f"→ {sanitize_token(task.get_command(repo_path))}")
                logger.info("")

            task.mark_status_as(TaskStatus.PASSED)
            with lock:
                executed_tasks.add(task_id)
            return True

        # ACTUAL EXECUTION
        if log_dir is None or log_date is None:
            raise ValueError("log_dir and log_date must be set for actual execution")
        task.set_log_file_path(log_dir, log_date, image_and_commit_sha)

        # Check if we should skip any task if it has already passed.
        # This must run BEFORE the input image check: if the task already passed,
        # we don't need the input image (it may have been cleaned up).
        # IMPORTANT: A stale .PASSED marker is NOT sufficient to skip if the task produces
        # a Docker image. The output image must still exist locally (users may have pruned).
        if skip_if_passed:
            if isinstance(task, BuildTask):
                if task.image_exists():
                    skip_task = True
                    reason = "Docker image already exists"
                elif task.passed_previously():
                    # Don't skip: marker may be stale if images were pruned.
                    skip_task = False
                    reason = ""
                    with lock:
                        logger.info(f"Found .PASSED marker for {task_id} but output image is missing; rebuilding")
                else:
                    skip_task = False
                    reason = ""
            else:
                has_output = bool(task.output_image)
                output_ok = task.image_exists() if has_output else True
                if task.passed_previously() and output_ok:
                    skip_task = True
                    reason = "Already passed previously"
                elif task.passed_previously() and not output_ok:
                    skip_task = False
                    reason = ""
                    with lock:
                        logger.info(f"Found .PASSED marker for {task_id} but output image is missing; re-running")
                else:
                    skip_task = False
                    reason = ""

            if skip_task:
                with lock:
                    logger.info(f"⊘ Skipping {task_id}: {reason}")
                task.mark_status_as(TaskStatus.SKIPPED, reason)
                with lock:
                    executed_tasks.add(task_id)
                return True

        # Check if input image exists (required for tasks that depend on previous builds)
        if not task.check_input_image_exists():
            head_changed_warning = check_sha_changed(repo_path, task_id)
            if head_changed_warning:
                with lock:
                    logger.warning(head_changed_warning)

            with lock:
                logger.error(f"✗ Skipping {task_id}: Input image missing ({task.input_image})")
            task.mark_status_as(TaskStatus.FAILED, f"Input image not found: {task.input_image}")

            with open(task.log_file, 'w') as log_fh:
                log_fh.write(f"Task: {task_id}\n")
                log_fh.write(f"Description: {task.description}\n")
                log_fh.write(f"Error: Input image not found: {task.input_image}\n")
                if head_changed_warning:
                    log_fh.write("\n")
                    log_fh.write(head_changed_warning)
                    log_fh.write("\n")
                log_fh.write("=" * 80 + "\n")

            with lock:
                executed_tasks.add(task_id)
                failed_tasks.add(task_id)

            update_frameworks_data_cache(task, use_absolute_urls=False)

            if log_dir and log_date:
                try:
                    html_content = generate_html_report(
                        all_tasks=all_tasks,
                        repo_path=repo_path,
                        image_and_commit_sha=image_and_commit_sha,
                        log_dir=log_dir,
                        date_str=log_date,
                        use_absolute_urls=False,
                    )
                    _write_report_and_json(html_content, all_tasks, repo_path, image_and_commit_sha, log_dir, log_date)
                except Exception as e:
                    with lock:
                        logger.warning(f"Failed to generate HTML report after image-not-found failure: {e}")

            return False

        with lock:
            logger.info(f"Executing: {task_id} ({task.description})")
            logger.info(f"  Command:           {sanitize_token(task.get_command(repo_path))}")
            if task.input_image:
                logger.info(f"  Input image:       {task.input_image}")
            if task.output_image:
                logger.info(f"  Output image:      {task.output_image}")
            logger.info(f"  Log:               {task.log_file}")

        # Clean up existing files for THIS specific task only (now that we know it will run)
        task.cleanup_markers()

        task.mark_status_as(TaskStatus.RUNNING)  # Also creates .RUNNING marker file
        task.start_time = time.time()

        # Write log file header first (so users see content immediately when clicking log link)
        with open(task.log_file, 'w') as log_fh:
            log_fh.write(task.format_log_header(repo_path))
            log_fh.flush()

        # Update frameworks data cache with RUNNING status
        update_frameworks_data_cache(task, use_absolute_urls=False)

        # Generate HTML report showing task as RUNNING (with log link)
        if log_dir and log_date:
            try:
                html_content = generate_html_report(
                    all_tasks=all_tasks,
                    repo_path=repo_path,
                    image_and_commit_sha=image_and_commit_sha,
                    log_dir=log_dir,
                    date_str=log_date,
                    use_absolute_urls=False,
                )
                _write_report_and_json(html_content, all_tasks, repo_path, image_and_commit_sha, log_dir, log_date)
            except Exception as e:
                with lock:
                    logger.warning(f"Failed to generate HTML report before task execution: {e}")

        try:

            success = task.execute(repo_path, dry_run=False)
            task.end_time = time.time()

            # Append execution summary to log file
            with open(task.log_file, 'a') as log_fh:
                log_fh.write(task.format_log_footer(success))

            with lock:
                if success:
                    task.mark_status_as(TaskStatus.PASSED)  # Also creates .PASSED marker, removes .RUNNING
                    duration = task.end_time - task.start_time

                    # Show running tasks and their elapsed time
                    running_tasks = [
                        (tid, t) for tid, t in all_tasks.items()
                        if t.status == TaskStatus.RUNNING and t.start_time
                    ]
                    if running_tasks:
                        running_info = []
                        for tid, t in running_tasks:
                            elapsed = time.time() - t.start_time
                            running_info.append(f"{tid} ({elapsed:.0f}s)")
                        logger.info(f"✓ Completed: {task_id} ({duration:.2f}s) | Running: {', '.join(running_info)}")
                    else:
                        logger.info(f"✓ Completed: {task_id} ({duration:.2f}s)")
                else:
                    task.mark_status_as(TaskStatus.FAILED)  # Also creates .FAILED marker, removes .RUNNING
                    logger.error(f"✗ Failed: {task_id}")
                    failed_tasks.add(task_id)
        except Exception as e:
            task.end_time = time.time()
            task.mark_status_as(TaskStatus.FAILED, str(e))  # Also creates .FAILED marker, removes .RUNNING
            with lock:
                logger.error(f"✗ Failed: {task_id} - {e}")
                failed_tasks.add(task_id)

        # Update frameworks data cache with task results
        update_frameworks_data_cache(task, use_absolute_urls=False)

        # Generate incremental HTML report after task completes
        if log_dir and log_date:
            try:
                html_content = generate_html_report(
                    all_tasks=all_tasks,
                    repo_path=repo_path,
                    image_and_commit_sha=image_and_commit_sha,
                    log_dir=log_dir,
                    date_str=log_date,
                    use_absolute_urls=False,
                )
                _write_report_and_json(html_content, all_tasks, repo_path, image_and_commit_sha, log_dir, log_date)
            except Exception as e:
                with lock:
                    logger.warning(f"Failed to generate incremental HTML report: {e}")

        with lock:
            executed_tasks.add(task_id)

        return task.status == TaskStatus.PASSED

    # Execute tasks dynamically based on dependencies
    pending_tasks = set(all_tasks.keys())
    active_futures = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Initial submission of root tasks
        for task_id in list(pending_tasks):
            if can_execute(task_id):
                future = executor.submit(execute_single_task, task_id)
                active_futures[future] = task_id
                pending_tasks.remove(task_id)

        # Process tasks as they complete
        while active_futures or pending_tasks:
            if not active_futures:
                # No active tasks but pending tasks remain - check for deadlock
                if pending_tasks:
                    logger.error(f"Deadlock detected! Remaining tasks: {pending_tasks}")
                    for task_id in pending_tasks:
                        with lock:
                            failed_tasks.add(task_id)
                            executed_tasks.add(task_id)
                break

            # Wait for next task to complete
            done, active_futures_set = wait(active_futures.keys(), return_when='FIRST_COMPLETED')

            for future in done:
                task_id = active_futures.pop(future)
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Exception in task {task_id}: {e}")
                    with lock:
                        failed_tasks.add(task_id)

                # Check if any new tasks can now execute
                for pending_task_id in list(pending_tasks):
                    if can_execute(pending_task_id):
                        new_future = executor.submit(execute_single_task, pending_task_id)
                        active_futures[new_future] = pending_task_id
                        pending_tasks.remove(pending_task_id)

    # Stop the periodic HTML updater thread
    if html_updater_thread and html_updater_thread.is_alive():
        stop_html_updater.set()
        html_updater_thread.join(timeout=2)  # Wait up to 2 seconds for thread to stop

    return executed_tasks, failed_tasks




def find_previous_log_file(
    repo_path: Path,
    current_date_str: str,
    image_and_commit_sha: str,
    task_id: str,
    max_days_back: int = 7,
) -> Optional[Tuple[Path, Path]]:
    """
    Find a previous log file for a given task, searching backwards from current date.

    Args:
        repo_path: Repository root path
        current_date_str: Current date string (YYYY-MM-DD)
        image_and_commit_sha: Git commit SHA
        task_id: Task identifier
        max_days_back: Maximum number of days to search back

    Returns:
        Tuple of (log_file_path, log_dir) if found, None otherwise
    """
    logs_base = repo_path / "logs"
    if not logs_base.exists():
        return None

    current_date = datetime.strptime(current_date_str, "%Y-%m-%d")

    # Search backwards from current date
    for days_ago in range(max_days_back + 1):
        search_date = current_date - timedelta(days=days_ago)
        search_date_str = search_date.strftime("%Y-%m-%d")
        search_log_dir = logs_base / search_date_str

        if not search_log_dir.exists():
            continue

        log_file = BaseTask.get_log_file_path(search_log_dir, search_date_str, image_and_commit_sha, task_id)
        if log_file.exists():
            return (log_file, search_log_dir)

    return None


def initialize_frameworks_data_cache(
    all_tasks: Dict[str, 'BaseTask'],
    log_dir: Path,
    date_str: str,
    image_and_commit_sha: str,
    repo_path: Path,
    use_absolute_urls: bool = False,
    version: Optional[str] = None,
    build_method: str = BUILD_METHOD_BUILD_SH,
) -> Dict[str, Dict[str, Any]]:
    """
    Initialize the frameworks data cache once with image info and previous logs.
    This is called once at the start, before any tasks run.
    Searches for previous logs in current date and up to 7 days back.
    """
    hostname = DEFAULT_HOSTNAME
    html_path = DEFAULT_HTML_PATH
    frameworks_data: Dict[str, Dict[str, Any]] = {}
    targets = ['runtime', 'dev', 'dev-compress', 'dev-upload', 'local-dev']

    # Determine which frameworks are actually being run (present in all_tasks)
    frameworks_in_all_tasks = set()
    for task_id in all_tasks.keys():
        parts = task_id.split('-', 1)
        if len(parts) >= 1 and parts[0] in FRAMEWORKS:
            frameworks_in_all_tasks.add(parts[0])

    # Load image info from a previous report.json with the same Image SHA (for reuse mode fallback)
    image_sha_6 = image_and_commit_sha.split('.')[0] if '.' in image_and_commit_sha else image_and_commit_sha
    prev_report_images = _load_previous_report_images(repo_path, image_sha_6)

    # Prefill ALL frameworks with image info from all_tasks + previous log data (if exists)
    for framework in FRAMEWORKS:
        frameworks_data[framework] = {}

        # For frameworks not in all_tasks, create temporary task graph to get image info
        temp_tasks = None
        if framework not in frameworks_in_all_tasks:
            # Create a temporary task graph just to extract image names.
            # Pass version and build_method through so callers in error paths
            # don't re-invoke build.sh (which may not exist at this SHA).
            temp_tasks = create_task_graph(framework, image_and_commit_sha, repo_path, version=version, build_method=build_method)

        for target in targets:
            # Initialize target structure
            frameworks_data[framework][target] = FrameworkTargetData()

            # Special handling for dev-compress: compress (build), compress-compilation, compress-sanity
            if target == 'dev-compress':
                compress_task_id = f"{framework}-dev-compress"
                compress_compilation_task_id = f"{framework}-dev-compress-compilation"
                compress_sanity_task_id = f"{framework}-dev-compress-sanity"
                compress_task = all_tasks.get(compress_task_id) or (temp_tasks.get(compress_task_id) if temp_tasks else None)
                compress_compilation_task = all_tasks.get(compress_compilation_task_id) or (temp_tasks.get(compress_compilation_task_id) if temp_tasks else None)
                compress_sanity_task = all_tasks.get(compress_sanity_task_id) or (temp_tasks.get(compress_sanity_task_id) if temp_tasks else None)

                # Extract image info from compress task, or fall back to compilation task (reuse mode)
                image_source = compress_task or compress_compilation_task
                if image_source:
                    frameworks_data[framework][target].input_image = image_source.input_image
                    frameworks_data[framework][target].output_image = image_source.output_image or image_source.input_image
                    check_tag = image_source.output_image or image_source.input_image
                    if check_tag:
                        image_size = get_docker_image_size(check_tag)
                        if image_size:
                            frameworks_data[framework][target].image_size = image_size

                # Map compress→build, compress-compilation→compilation, compress-sanity→sanity
                for task_id_suffix, col in [
                    (compress_task_id, 'build'),
                    (compress_compilation_task_id, 'compilation'),
                    (compress_sanity_task_id, 'sanity'),
                ]:
                    prev_log = find_previous_log_file(repo_path, date_str, image_and_commit_sha, task_id_suffix)
                    if prev_log:
                        log_file, found_log_dir = prev_log
                        log_results = parse_log_file_results(log_file)
                        if log_results:
                            log_link = (
                                f"../../{found_log_dir.name}/{log_file.parent.name}/{log_file.name}" if found_log_dir.name != date_str
                                else log_file.name
                            )
                            td = TaskData(
                                status='skipped',
                                time=f"{log_results.duration:.1f}s" if log_results.duration else None,
                                log_file=log_link if not use_absolute_urls else f"http://{hostname}{html_path}/{found_log_dir.name}/{log_file.parent.name}/{log_file.name}",
                                prev_status=log_results.status.value,
                            )
                            if col == 'build':
                                frameworks_data[framework][target].build = td
                            elif col == 'compilation':
                                frameworks_data[framework][target].compilation = td
                            else:
                                frameworks_data[framework][target].sanity = td
                continue  # Skip normal BuildTask handling for dev-compress

            # Special handling for dev-upload: it's a CommandTask, not a BuildTask
            if target == 'dev-upload':
                task_id_for_upload = f"{framework}-dev-upload"
                upload_task = None
                
                # Try to get from all_tasks (frameworks currently being run)
                if task_id_for_upload in all_tasks and isinstance(all_tasks[task_id_for_upload], CommandTask):
                    upload_task = all_tasks[task_id_for_upload]
                # Otherwise, get from temp_tasks (frameworks NOT being run)
                elif temp_tasks and task_id_for_upload in temp_tasks and isinstance(temp_tasks[task_id_for_upload], CommandTask):
                    upload_task = temp_tasks[task_id_for_upload]
                
                if upload_task:
                    frameworks_data[framework][target].input_image = upload_task.input_image
                    frameworks_data[framework][target].output_image = upload_task.output_image
                    # Don't check image size for dev-upload (will be set when upload succeeds)
                
                # Load previous upload log (if exists)
                prev_log_result = find_previous_log_file(repo_path, date_str, image_and_commit_sha, f"{framework}-dev-upload")
                if prev_log_result:
                    log_file, found_log_dir = prev_log_result
                    log_results = parse_log_file_results(log_file)
                    if log_results:
                        if found_log_dir.name != date_str:
                            log_link = f"../../{found_log_dir.name}/{log_file.parent.name}/{log_file.name}"
                        else:
                            log_link = log_file.name

                        frameworks_data[framework][target].build = TaskData(
                            status='skipped',
                            time=f"{log_results.duration:.1f}s" if log_results.duration else None,
                            log_file=log_link if not use_absolute_urls else f"http://{hostname}{html_path}/{found_log_dir.name}/{log_file.parent.name}/{log_file.name}",
                            prev_status=log_results.status.value,
                        )
                continue  # Skip the normal BuildTask handling for dev-upload

            # Get BuildTask for this target to extract image info
            # BuildTask IDs now have -build suffix: framework-target-build
            task_id_for_build = f"{framework}-{target}-build"

            # Try to get from all_tasks (frameworks currently being run)
            if task_id_for_build in all_tasks and isinstance(all_tasks[task_id_for_build], BuildTask):
                build_task = all_tasks[task_id_for_build]
            # Otherwise, get from temp_tasks (frameworks NOT being run)
            elif temp_tasks and task_id_for_build in temp_tasks and isinstance(temp_tasks[task_id_for_build], BuildTask):
                build_task = temp_tasks[task_id_for_build]
            else:
                build_task = None

            if build_task:
                # For runtime target, show base image (FROM ... AS runtime) in report only; task has no input_image check
                if target == "runtime":
                    frameworks_data[framework][target].input_image = get_runtime_base_image(repo_path, framework) or build_task.input_image
                else:
                    frameworks_data[framework][target].input_image = build_task.input_image
                frameworks_data[framework][target].output_image = build_task.output_image

                # Try to get image size if image exists
                if build_task.output_image:
                    image_size = get_docker_image_size(build_task.output_image)
                    if image_size:
                        frameworks_data[framework][target].image_size = image_size



            # Load previous build log (if exists) - search current and previous dates
            prev_log_result = find_previous_log_file(repo_path, date_str, image_and_commit_sha, f"{framework}-{target}-build")
            if prev_log_result:
                    log_file, found_log_dir = prev_log_result
                    log_results = parse_log_file_results(log_file)
                    if log_results:
                        if found_log_dir.name != date_str:
                            log_link = f"../../{found_log_dir.name}/{log_file.parent.name}/{log_file.name}"
                        else:
                            log_link = log_file.name

                        frameworks_data[framework][target].build = TaskData(
                            status='skipped',
                            time=f"{log_results.duration:.1f}s" if log_results.duration else None,
                            log_file=log_link if not use_absolute_urls else f"http://{hostname}{html_path}/{found_log_dir.name}/{log_file.parent.name}/{log_file.name}",
                            prev_status=log_results.status.value,
                        )

            # Load previous compilation log (if exists) - search current and previous dates
            if target in ['dev', 'local-dev']:
                prev_log_result = find_previous_log_file(repo_path, date_str, image_and_commit_sha, f"{framework}-{target}-compilation")
                if prev_log_result:
                    log_file_comp, found_log_dir = prev_log_result
                    log_results = parse_log_file_results(log_file_comp)
                    if log_results:
                        if found_log_dir.name != date_str:
                            log_link = f"../../{found_log_dir.name}/{log_file_comp.parent.name}/{log_file_comp.name}"
                        else:
                            log_link = log_file_comp.name

                        frameworks_data[framework][target].compilation = TaskData(
                            status='skipped',
                            time=f"{log_results.duration:.1f}s" if log_results.duration else None,
                            log_file=log_link if not use_absolute_urls else f"http://{hostname}{html_path}/{found_log_dir.name}/{log_file_comp.parent.name}/{log_file_comp.name}",
                            prev_status=log_results.status.value,
                        )

            # Load previous sanity log (if exists) - search current and previous dates
            if target in ['runtime', 'dev', 'local-dev']:
                prev_log_result = find_previous_log_file(repo_path, date_str, image_and_commit_sha, f"{framework}-{target}-sanity")
                if prev_log_result:
                    log_file_sanity, found_log_dir = prev_log_result
                    log_results = parse_log_file_results(log_file_sanity)
                    if log_results:
                        if found_log_dir.name != date_str:
                            log_link = f"../../{found_log_dir.name}/{log_file_sanity.parent.name}/{log_file_sanity.name}"
                        else:
                            log_link = log_file_sanity.name

                        frameworks_data[framework][target].sanity = TaskData(
                            status='skipped',
                            time=f"{log_results.duration:.1f}s" if log_results.duration else None,
                            log_file=log_link if not use_absolute_urls else f"http://{hostname}{html_path}/{found_log_dir.name}/{log_file_sanity.parent.name}/{log_file_sanity.name}",
                            prev_status=log_results.status.value,
                        )

    # Backfill missing input_image/output_image from previous report with same Image SHA
    if prev_report_images:
        for framework in FRAMEWORKS:
            fw_prev = prev_report_images.get(framework, {})
            for target in targets:
                if target not in frameworks_data[framework]:
                    continue
                td = frameworks_data[framework][target]
                tgt_prev = fw_prev.get(target, {})
                if not td.input_image and tgt_prev.get("input_image"):
                    td.input_image = tgt_prev["input_image"]
                if not td.output_image and tgt_prev.get("output_image"):
                    td.output_image = tgt_prev["output_image"]

    return frameworks_data


def update_frameworks_data_cache(task: 'BaseTask', use_absolute_urls: bool = False) -> None:
    """
    Update the frameworks data cache when a task completes.
    This is called after each task runs.
    """
    global _frameworks_data_cache
    if _frameworks_data_cache is None:
        raise RuntimeError("Frameworks data cache not initialized! This should never happen.")

    # Update cache for all statuses (including SKIPPED) so the HTML report shows correct state
    # Parse task_id to extract framework and target
    parts = task.task_id.split('-', 1)
    if len(parts) < 2:
        return

    framework = parts[0]
    rest = parts[1]

    if framework not in _frameworks_data_cache:
        return

    # Helper to create task data
    def create_task_data_from_task(t: 'BaseTask') -> 'TaskData':
        task_start_epoch: Optional[float] = None
        if t.status == TaskStatus.RUNNING:
            # Use absolute timestamp from .RUNNING marker file if available,
            # falling back to in-memory start_time. This survives process restarts.
            start_ts = t.start_time
            if t.log_file:
                running_marker = t.log_file.with_suffix(f'.{MARKER_RUNNING}')
                if running_marker.exists():
                    start_ts = running_marker.stat().st_mtime
            if start_ts:
                task_start_epoch = start_ts
                elapsed = time.time() - start_ts
                task_time = f"elapsed {elapsed:.1f}s"
            else:
                task_time = None
        elif t.start_time and t.end_time:
            task_time = f"{t.end_time - t.start_time:.1f}s"
        else:
            task_time = None

        log_url = None
        if t.log_file:
            if use_absolute_urls:
                # logs/{date}/{commit_sha_9}/{task}.log -> {date}/{commit_sha_9}/{task}.log
                rel_path = t.log_file.relative_to(t.log_file.parent.parent.parent)
                log_url = f"http://{DEFAULT_HOSTNAME}{DEFAULT_HTML_PATH}/{rel_path}"
            else:
                # report.html is in the same commit_sha_9 dir, so just the filename
                log_url = t.log_file.name

        return TaskData(
            status=t.status.value,
            time=task_time,
            log_file=log_url,
            prev_status=None,
            start_epoch=task_start_epoch,
        )

    # Determine target and task type, then update cache
    if isinstance(task, BuildTask):
        # BuildTask IDs are like: framework-target-build
        # Strip -build suffix to get target
        target = rest.replace('-build', '') if rest.endswith('-build') else rest
        _frameworks_data_cache[framework][target].build = create_task_data_from_task(task)
        _frameworks_data_cache[framework][target].image_size = task.get_image_size()
        _frameworks_data_cache[framework][target].input_image = task.input_image
        _frameworks_data_cache[framework][target].output_image = task.output_image
    elif isinstance(task, CommandTask):
        if rest == 'dev-compress':
            # Compress task maps to 'build' column of 'dev-compress' target
            _frameworks_data_cache[framework]['dev-compress'].build = create_task_data_from_task(task)
            if task.input_image:
                _frameworks_data_cache[framework]['dev-compress'].input_image = task.input_image
            if task.output_image:
                _frameworks_data_cache[framework]['dev-compress'].output_image = task.output_image
                image_size = get_docker_image_size(task.output_image)
                if image_size:
                    _frameworks_data_cache[framework]['dev-compress'].image_size = image_size
        elif rest == 'dev-compress-sanity':
            # Compress-sanity maps to 'sanity' column of 'dev-compress' target
            _frameworks_data_cache[framework]['dev-compress'].sanity = create_task_data_from_task(task)
        elif 'compilation' in rest:
            target = rest.replace('-compilation', '')
            _frameworks_data_cache[framework][target].compilation = create_task_data_from_task(task)
            if not _frameworks_data_cache[framework][target].input_image and task.input_image:
                _frameworks_data_cache[framework][target].input_image = task.input_image
        elif 'sanity' in rest:
            target = rest.replace('-sanity', '')
            _frameworks_data_cache[framework][target].sanity = create_task_data_from_task(task)
        elif 'upload' in rest:
            # Don't strip '-upload' - upload tasks update the 'dev-upload' target, not 'dev'
            target = rest
            _frameworks_data_cache[framework][target].build = create_task_data_from_task(task)
            # Update image info for upload task
            if task.output_image:
                _frameworks_data_cache[framework][target].output_image = task.output_image
            if task.input_image:
                _frameworks_data_cache[framework][target].input_image = task.input_image
            # Set image size to the dev image size when upload succeeds
            if task.status == TaskStatus.PASSED and framework in _frameworks_data_cache:
                if 'dev' in _frameworks_data_cache[framework]:
                    _frameworks_data_cache[framework][target].image_size = _frameworks_data_cache[framework]['dev'].image_size


def update_report_status_marker(
    html_file: Path,
    all_tasks: Dict[str, 'BaseTask'],
) -> None:
    """
    Create or update status marker files for the HTML report (.PASSED, .FAILED, .RUNNING).

    Args:
        html_file: Path to the HTML report file
        all_tasks: Dictionary of all tasks that were executed
    """
    # Count task statistics (excluding none-compilation and none-sanity)
    succeeded = sum(1 for task_id, t in all_tasks.items()
                   if t.status == TaskStatus.PASSED
                   and not (task_id.startswith('none-') and ('compilation' in task_id or 'sanity' in task_id)))
    failed = sum(1 for task_id, t in all_tasks.items()
                if t.status == TaskStatus.FAILED
                and not (task_id.startswith('none-') and ('compilation' in task_id or 'sanity' in task_id)))
    killed = sum(1 for t in all_tasks.values() if t.status == TaskStatus.KILLED)
    running = sum(1 for t in all_tasks.values() if t.status == TaskStatus.RUNNING)
    queued = sum(1 for t in all_tasks.values() if t.status == TaskStatus.QUEUED)

    # Determine overall status (priority: failed > killed > running/queued > success)
    if failed > 0:
        status_marker = MARKER_FAILED
    elif killed > 0:
        status_marker = MARKER_KILLED
    elif running > 0 or queued > 0:
        status_marker = MARKER_RUNNING
    else:
        status_marker = MARKER_PASSED

    # Remove old marker files
    for marker in [MARKER_PASSED, MARKER_FAILED, MARKER_RUNNING, MARKER_KILLED]:
        old_marker = html_file.with_suffix(f'.{marker}')
        if old_marker.exists():
            old_marker.unlink()

    # Create new marker file
    marker_file = html_file.with_suffix(f'.{status_marker}')
    marker_file.touch()


def generate_html_report(
    all_tasks: Dict[str, 'BaseTask'],
    repo_path: Path,
    image_and_commit_sha: str,
    log_dir: Path,
    date_str: str,
    use_absolute_urls: bool = False,
    hostname: str = DEFAULT_HOSTNAME,
    html_path: str = DEFAULT_HTML_PATH,
) -> str:
    """
    Generate HTML report using Jinja2 template.

    Args:
        all_tasks: Dictionary of all tasks that were executed
        repo_path: Path to the repository
        image_and_commit_sha: "E04427.a798e08c8" (image_sha_6.commit_sha_9) or plain commit_sha_9
        log_dir: Directory containing log files
        date_str: Date string (YYYY-MM-DD)
        use_absolute_urls: If True, use http://hostname/... URLs for email
        hostname: Hostname for absolute URLs
        html_path: Base path for absolute URLs

    Returns:
        HTML string
    """

    # Count task statistics
    # Exclude none-compilation and none-sanity from pass/fail counts
    # (they still run but don't affect overall status)
    total_tasks = len(all_tasks)
    succeeded = sum(1 for task_id, t in all_tasks.items()
                   if t.status == TaskStatus.PASSED
                   and not (task_id.startswith('none-') and ('compilation' in task_id or 'sanity' in task_id)))
    failed = sum(1 for task_id, t in all_tasks.items()
                if t.status == TaskStatus.FAILED
                and not (task_id.startswith('none-') and ('compilation' in task_id or 'sanity' in task_id)))
    skipped = sum(1 for t in all_tasks.values() if t.status == TaskStatus.SKIPPED)
    killed = sum(1 for t in all_tasks.values() if t.status == TaskStatus.KILLED)
    running = sum(1 for t in all_tasks.values() if t.status == TaskStatus.RUNNING)
    queued = sum(1 for t in all_tasks.values() if t.status == TaskStatus.QUEUED)

    # Calculate overall build elapsed time using absolute timestamps.
    # Prefer .RUNNING marker mtimes (survive process restarts) over in-memory start_time.
    absolute_start_times: List[float] = []
    for t in all_tasks.values():
        if t.log_file:
            running_marker = t.log_file.with_suffix(f'.{MARKER_RUNNING}')
            if running_marker.exists():
                absolute_start_times.append(running_marker.stat().st_mtime)
                continue
            # For completed tasks, use log file mtime as a proxy for when the task ran
            if t.log_file.exists() and t.status in (TaskStatus.PASSED, TaskStatus.FAILED, TaskStatus.KILLED):
                absolute_start_times.append(t.log_file.stat().st_mtime)
                continue
        if t.start_time:
            absolute_start_times.append(t.start_time)

    overall_elapsed_str = ""
    if absolute_start_times:
        earliest_start = min(absolute_start_times)

        # If build is still in progress, calculate elapsed time
        if running > 0 or queued > 0:
            elapsed_seconds = time.time() - earliest_start
            minutes = int(elapsed_seconds // 60)
            seconds = int(elapsed_seconds % 60)
            if minutes > 0:
                overall_elapsed_str = f" (elapsed {minutes}m {seconds}s)"
            else:
                overall_elapsed_str = f" (elapsed {seconds}s)"

    # Determine overall status (priority: failed > killed > running/queued > success)
    if failed > 0:
        overall_status = f"❌ TESTS FAILED{overall_elapsed_str}"
        header_color = "#dc3545"  # Red
    elif killed > 0:
        overall_status = f"⚠️ BUILD KILLED{overall_elapsed_str}"
        header_color = "#ff9800"  # Orange
    elif running > 0 or queued > 0:
        overall_status = f"🔄 BUILD IN PROGRESS{overall_elapsed_str}"
        header_color = "#ffc107"  # Yellow
    else:
        overall_status = f"✅ ALL TESTS PASSED{overall_elapsed_str}"
        header_color = "#28a745"  # Green

    # Extract commit_sha_9 and image_sha_6 from image_and_commit_sha (format: "E04427.a798e08c8")
    if '.' in image_and_commit_sha and re.match(r'^[0-9A-F]{6}$', image_and_commit_sha.split('.', 1)[0]):
        image_sha_6 = image_and_commit_sha.split('.', 1)[0]
        commit_sha_9 = image_and_commit_sha.split('.', 1)[1]
    else:
        commit_sha_9 = image_and_commit_sha
        image_sha_6 = None
    image_sha_url: Optional[str] = None
    if image_sha_6 and repo_path.exists():
        try:
            dru = DynamoRepositoryUtils(repo_path)
            origin_commit_sha_40, _ = dru.find_docker_image_sha_origin(commit_sha_9)
            image_sha_url = f"https://github.com/ai-dynamo/dynamo/commit/{origin_commit_sha_40}"
        except ValueError:
            pass
    commit_info: Dict[str, Any] = {}
    if git:
        try:
            repo = git.Repo(repo_path)
            commit = repo.commit(commit_sha_9)
            commit_info = {
                'commit_sha_9': commit_sha_9[:9],
                'commit_sha_40': commit_sha_9,
                'author': f"{commit.author.name} <{commit.author.email}>",
                'date': commit.committed_datetime.strftime('%Y-%m-%d %H:%M:%S %Z'),
                'message': commit.message.strip(),
            }

            # Extract PR number
            commit_message = str(commit.message) if commit.message else ""
            first_line = commit_message.split('\n')[0]
            pr_match = re.search(r'\(#(\d+)\)', first_line)
            if pr_match:
                commit_info['pr_number'] = pr_match.group(1)
                commit_info['pr_link'] = f"https://github.com/ai-dynamo/dynamo/pull/{pr_match.group(1)}"

            # Get file changes
            try:
                stats = commit.stats
                commit_info['insertions'] = int(stats.total.get('insertions', 0))
                commit_info['deletions'] = int(stats.total.get('deletions', 0))
                commit_info['files_changed'] = dict(stats.files)
            except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
                pass

        except Exception as e:
            logging.getLogger("html").warning(f"Could not get git info: {e}")
            commit_info = {'commit_sha_9': commit_sha_9[:9], 'commit_sha_40': commit_sha_9}
    else:
        commit_info = {'commit_sha_9': commit_sha_9[:9], 'commit_sha_40': commit_sha_9}

    # Use the global frameworks data cache (already initialized and updated by tasks)
    global _frameworks_data_cache
    if _frameworks_data_cache is None:
        # Fallback: initialize it now if not already done
        _frameworks_data_cache = initialize_frameworks_data_cache(
            all_tasks, log_dir, date_str, commit_sha_9, repo_path, use_absolute_urls
        )

    frameworks_data = _frameworks_data_cache

    # No need for STEP 1 and STEP 2 anymore - cache is already maintained!
    # Just convert FrameworkTargetData objects to dicts for Jinja2
    frameworks_dict = {}
    for framework in FRAMEWORKS:
        frameworks_dict[framework] = {}
        for target in ['runtime', 'dev', 'local-dev']:
            frameworks_dict[framework][target] = frameworks_data[framework][target].to_dict()

    # Load Jinja2 template
    template_dir = Path(__file__).parent
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(['html', 'xml'])
    )
    template = env.get_template('build_images_report.html.j2')

    # Render HTML
    # Example data structure passed to Jinja2 template:
    # {
    #   'commit_sha_9': 'f2a3c638d',
    #   'commit_sha_40': 'f2a3c638d',

    #   'image_sha_6': 'E04427',
    #   'image_sha_url': 'https://github.com/ai-dynamo/dynamo/commit/...',
    #   'overall_status': '✅ ALL TESTS PASSED' or '❌ TESTS FAILED',
    #   'commit': {
    #       'commit_sha_9': 'f2a3c63',
    #       'commit_sha_40': 'f2a3c638d',
    #       'author': 'John Doe <john@example.com>',
    #       'date': '2025-10-30 10:15:30 UTC',
    #       'message': 'Fix bug in inference pipeline (#123)',
    #       'pr_number': '123',
    #       'pr_link': 'https://github.com/ai-dynamo/dynamo/pull/123',
    #       'insertions': 42,
    #       'deletions': 15,
    #       'files_changed': {'src/main.py': {...}, 'tests/test_main.py': {...}},
    #   },
    #   'frameworks': {
    #       'vllm': {
    #           'runtime': {
    #               'build': {'status': 'SUCCESS', 'time': '45.2s', 'log_file': '...'},
    #               'compilation': None,
    #               'sanity': {'status': 'SUCCESS', 'time': '3.8s', 'log_file': '...'},
    #               'image_size': '12.3GB',
    #               'input_image': 'dynamo-base:f2a3c638d',
    #               'output_image': 'dynamo-vllm-runtime:f2a3c638d',
    #           },
    #           'dev': {
    #               'build': {'status': 'RUNNING', 'time': None, 'log_file': None},
    #               'compilation': {'status': 'QUEUED', 'time': '120.5s', 'log_file': None},  # prev time shown
    #               'sanity': {'status': 'QUEUED', 'time': '5.2s', 'log_file': None},
    #               'image_size': '15.8GB',
    #               'input_image': 'dynamo-vllm-runtime:f2a3c638d',
    #               'output_image': 'dynamo-vllm-dev:f2a3c638d',
    #           },
    #           'local-dev': {
    #               'build': {'status': 'SKIPPED', 'time': '89.3s', 'log_file': '...'},  # prev time shown
    #               'compilation': {'status': 'SKIPPED', 'time': '95.1s', 'log_file': '...'},
    #               'sanity': {'status': 'SKIPPED', 'time': '4.7s', 'log_file': '...'},
    #               'image_size': '18.2GB',
    #               'input_image': 'dynamo-vllm-dev:f2a3c638d',
    #               'output_image': 'dynamo-vllm-local-dev:f2a3c638d',
    #           },
    #       },
    #       'SGLANG': {
    #           # Similar structure for SGLANG...
    #       },
    #       'TRTLLM': {
    #           # Similar structure for TRTLLM...
    #       },
    #   }
    # }
    # Convert dataclasses to dictionaries for Jinja2 template
    frameworks_dict = {}
    for framework, targets in frameworks_data.items():
        frameworks_dict[framework] = {}
        for target, target_data in targets.items():
            frameworks_dict[framework][target] = target_data.to_dict()

    now = datetime.now()
    html = template.render(
        commit_sha_9=commit_info.get('commit_sha_9', commit_sha_9[:9]),
        commit_sha_40=commit_info.get('commit_sha_40', commit_sha_9),
        image_sha_6=image_sha_6,
        image_sha_url=image_sha_url,
        overall_status=overall_status,
        header_color=header_color,
        total_tasks=total_tasks,
        succeeded=succeeded,
        failed=failed,
        skipped=skipped,
        killed=killed,
        running=running,
        queued=queued,
        build_date=now.strftime('%Y-%m-%d %H:%M:%S'),
        report_generated=now.strftime('%Y-%m-%d %H:%M:%S'),
        commit=commit_info,
        frameworks=frameworks_dict,
        no_compress=_report_no_compress,
        no_upload=_report_no_upload,
        no_build=_report_no_build,
    )

    return html


def _parse_size_to_bytes(size_str: Optional[str]) -> Optional[int]:
    """Parse human-readable size like '21.2 GB' to bytes."""
    if not size_str:
        return None
    # Try to get size_bytes directly from DockerUtils (more accurate)
    # but this function handles the string form stored in FrameworkTargetData.image_size
    size_str = size_str.strip()
    multipliers = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
    for unit, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if size_str.upper().endswith(unit):
            num_part = size_str[: -len(unit)].strip()
            try:
                return int(float(num_part) * mult)
            except ValueError:
                return None
    return None


def _parse_duration_str(time_str: Optional[str]) -> Optional[float]:
    """Parse duration string like '120.5s' or 'elapsed 30.2s' to float seconds."""
    if not time_str:
        return None
    # Strip 'elapsed ' prefix
    s = time_str.strip()
    if s.startswith("elapsed "):
        s = s[len("elapsed "):]
    # Strip trailing 's'
    if s.endswith("s"):
        s = s[:-1]
    try:
        return float(s)
    except ValueError:
        return None


def _normalize_status(status: Optional[str]) -> str:
    """Normalize task status strings to the BuildReport convention."""
    if not status:
        return "SKIP"
    s = status.strip().upper()
    # Map build_images.py status values to BuildReport convention
    mapping = {
        "PASSED": "PASS",
        "PASS": "PASS",
        "FAILED": "FAIL",
        "FAIL": "FAIL",
        "SKIPPED": "SKIP",
        "SKIP": "SKIP",
        "KILLED": "KILLED",
        "RUNNING": "RUNNING",
        "QUEUED": "QUEUED",
        "UPLOADED": "PASS",
    }
    return mapping.get(s, s)


def generate_build_report(
    all_tasks: Dict[str, 'BaseTask'],
    repo_path: Path,
    image_and_commit_sha: str,
    log_dir: Path,
    date_str: str,
) -> BuildReport:
    """Generate a BuildReport from the current build state.

    Converts _frameworks_data_cache, commit info, and task statistics into
    the typed BuildReport structure for JSON serialization.

    Args:
        all_tasks: Dictionary of all tasks that were executed
        repo_path: Path to the repository
        image_and_commit_sha: "E04427.a798e08c8" (image_sha_6.commit_sha_9) or plain commit_sha_9
        log_dir: Directory containing log files
        date_str: Date string (YYYY-MM-DD)

    Returns:
        BuildReport ready for serialization
    """
    # Task statistics (same logic as generate_html_report)
    total_tasks = len(all_tasks)
    succeeded = sum(1 for task_id, t in all_tasks.items()
                   if t.status == TaskStatus.PASSED
                   and not (task_id.startswith('none-') and ('compilation' in task_id or 'sanity' in task_id)))
    failed = sum(1 for task_id, t in all_tasks.items()
                if t.status == TaskStatus.FAILED
                and not (task_id.startswith('none-') and ('compilation' in task_id or 'sanity' in task_id)))
    skipped = sum(1 for t in all_tasks.values() if t.status == TaskStatus.SKIPPED)
    killed = sum(1 for t in all_tasks.values() if t.status == TaskStatus.KILLED)

    # Determine overall status
    if failed > 0:
        overall_status = "FAIL"
    elif killed > 0:
        overall_status = "KILLED"
    elif sum(1 for t in all_tasks.values() if t.status in (TaskStatus.RUNNING, TaskStatus.QUEUED)) > 0:
        overall_status = "RUNNING"
    else:
        overall_status = "PASS"

    # Extract the bare commit_sha_9 from image_and_commit_sha (e.g., "E04427.a798e08c8" → "a798e08c8")
    commit_sha_9 = image_and_commit_sha.split('.', 1)[1] if '.' in image_and_commit_sha and re.match(r'^[0-9A-F]{6}$', image_and_commit_sha.split('.', 1)[0]) else image_and_commit_sha
    commit_info_dict: Dict[str, Any] = {}
    if git:
        try:
            repo = git.Repo(repo_path)
            commit_obj = repo.commit(commit_sha_9)
            commit_info_dict = {
                'commit_sha_9': commit_sha_9[:9],
                'commit_sha_40': str(commit_obj.hexsha),
                'author': f"{commit_obj.author.name} <{commit_obj.author.email}>",
                'date': commit_obj.committed_datetime.isoformat(),
                'message': commit_obj.message.strip(),
            }
            first_line = str(commit_obj.message).split('\n')[0]
            pr_match = re.search(r'\(#(\d+)\)', first_line)
            if pr_match:
                commit_info_dict['pr_number'] = int(pr_match.group(1))
            try:
                stats = commit_obj.stats
                commit_info_dict['insertions'] = int(stats.total.get('insertions', 0))
                commit_info_dict['deletions'] = int(stats.total.get('deletions', 0))
            except Exception:
                pass
        except Exception:
            commit_info_dict = {'commit_sha_9': commit_sha_9[:9], 'commit_sha_40': commit_sha_9}
    else:
        commit_info_dict = {'commit_sha_9': commit_sha_9[:9], 'commit_sha_40': commit_sha_9}

    commit = CommitInfo(
        commit_sha_9=commit_info_dict.get('commit_sha_9', commit_sha_9[:9]),
        commit_sha_40=commit_info_dict.get('commit_sha_40', commit_sha_9),
        author=commit_info_dict.get('author'),
        date=commit_info_dict.get('date'),
        message=commit_info_dict.get('message'),
        pr_number=commit_info_dict.get('pr_number'),
        insertions=commit_info_dict.get('insertions'),
        deletions=commit_info_dict.get('deletions'),
    )

    # Build frameworks data from _frameworks_data_cache
    global _frameworks_data_cache
    if _frameworks_data_cache is None:
        _frameworks_data_cache = initialize_frameworks_data_cache(
            all_tasks, log_dir, date_str, commit_sha_9, repo_path, False
        )
    frameworks_data = _frameworks_data_cache

    framework_results: List[FrameworkResult] = []
    for framework in FRAMEWORKS:
        fw_data = frameworks_data.get(framework, {})
        targets: List[TargetResult] = []
        registry_images: List[RegistryImage] = []

        for target_name in ['runtime', 'dev', 'dev-compress', 'local-dev']:
            target_data: FrameworkTargetData = fw_data.get(target_name, FrameworkTargetData())

            def _to_task_result(td: Optional[TaskData]) -> Optional[TaskResult]:
                if td is None:
                    return None
                return TaskResult(
                    status=_normalize_status(td.status),
                    duration_s=_parse_duration_str(td.time),
                    log_file=td.log_file,
                    prev_status=_normalize_status(td.prev_status) if td.prev_status else None,
                )

            targets.append(TargetResult(
                target=target_name,
                build=_to_task_result(target_data.build),
                compilation=_to_task_result(target_data.compilation),
                sanity=_to_task_result(target_data.sanity),
                image_name=target_data.output_image,
                image_size_bytes=_parse_size_to_bytes(target_data.image_size),
                input_image=target_data.input_image,
            ))

        # Registry images from dev-upload target -- only record if the upload
        # actually ran and succeeded.  Reuse builds skip dev-upload but still
        # have output_image set at task-creation time; recording those would
        # create bogus entries for tags that don't exist in GitLab.
        upload_task_id = f"{framework}-dev-upload"
        upload_task = all_tasks.get(upload_task_id)
        if upload_task and upload_task.status == TaskStatus.PASSED:
            upload_data: FrameworkTargetData = fw_data.get('dev-upload', FrameworkTargetData())
            if upload_data.output_image:
                gitlab_location = upload_data.output_image
                image_name = gitlab_location.rsplit("/", 1)[-1] if "/" in gitlab_location else gitlab_location
                tag = image_name.split(":", 1)[-1] if ":" in image_name else image_name
                pushed_at = (
                    datetime.fromtimestamp(upload_task.end_time).isoformat()
                    if upload_task.end_time else None
                )

                compress_data: FrameworkTargetData = fw_data.get('dev-compress', FrameworkTargetData())
                if compress_data.image_size:
                    dev_size_bytes = _parse_size_to_bytes(compress_data.image_size)
                else:
                    dev_data: FrameworkTargetData = fw_data.get('dev', FrameworkTargetData())
                    dev_size_bytes = _parse_size_to_bytes(dev_data.image_size)

                registry_images.append(RegistryImage(
                    location=gitlab_location,
                    image_name=image_name,
                    tag=tag,
                    framework=framework,
                    target="dev",
                    push_status="PASS",
                    size_bytes=dev_size_bytes,
                    pushed_at=pushed_at,
                ))

        framework_results.append(FrameworkResult(
            framework=framework,
            targets=targets,
            registry_images=registry_images,
        ))

    now = datetime.now()
    return BuildReport(
        schema_version=SCHEMA_VERSION,
        commit_sha_9=commit_sha_9[:9],
        commit_sha_40=commit_info_dict.get('commit_sha_40', commit_sha_9),
        build_date=date_str,
        report_generated=now.isoformat(),
        overall_status=overall_status,
        total_tasks=total_tasks,
        succeeded=succeeded,
        failed=failed,
        skipped=skipped,
        killed=killed,
        commit=commit,
        frameworks=framework_results,
    )


def send_email_notification(
    email: str,
    html_content: str,
    commit_sha_9: str,
    failed_tasks: List[str],
) -> bool:
    """
    Send HTML email notification via SMTP using curl.

    Args:
        email: Recipient email address
        html_content: HTML content to send
        commit_sha_9: Git commit SHA (short, 7 chars)
        failed_tasks: List of failed task IDs

    Returns:
        True if email sent successfully, False otherwise
    """
    logger = logging.getLogger("email")

    try:
        # Determine overall status
        overall_status = "SUCCESS" if not failed_tasks else "FAILURE"

        # Create subject line
        status_prefix = "SUCC" if overall_status == "SUCCESS" else "FAIL"

        # Include failed task names in subject if any
        if failed_tasks:
            failure_summary = ", ".join(failed_tasks[:3])  # Limit to first 3
            if len(failed_tasks) > 3:
                failure_summary += f" (+{len(failed_tasks) - 3} more)"
            subject = f"{status_prefix}: DynamoDockerBuilder - {commit_sha_9} ({failure_summary})"
        else:
            subject = f"{status_prefix}: DynamoDockerBuilder - {commit_sha_9}"

        # Create email file with proper CRLF formatting
        email_file = Path(f"/tmp/dynamo_email_{os.getpid()}.txt")

        # Write email content directly
        email_content = (
            f'Subject: {subject}\r\n'
            f'From: DynamoDockerBuilder <dynamo-docker-builder@nvidia.com>\r\n'
            f'To: {email}\r\n'
            f'MIME-Version: 1.0\r\n'
            f'Content-Type: text/html; charset=UTF-8\r\n'
            f'\r\n'
            f'{html_content}\r\n'
        )

        with open(email_file, 'w', encoding='utf-8') as f:
            f.write(email_content)

        # Send email using curl
        result = subprocess.run([
            'curl', '--url', 'smtp://smtp.nvidia.com:25',
            '--mail-from', 'dynamo-docker-builder@nvidia.com',
            '--mail-rcpt', email,
            '--upload-file', str(email_file)
        ], capture_output=True, text=True, timeout=30)

        # Clean up
        email_file.unlink(missing_ok=True)

        if result.returncode == 0:
            logger.info(f"📧 Email notification sent to {email}")
            logger.info(f"   Subject: {subject}")
            if failed_tasks:
                logger.info(f"   Failed tasks: {', '.join(failed_tasks)}")
            return True
        else:
            logger.error(f"⚠️  Failed to send email: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"⚠️  Error sending email: {e}")
        return False


def print_dependency_tree(frameworks: List[str], commit_sha_9: str, repo_path: Path, verbose: bool = False) -> None:
    """
    Print dependency tree visualization for given frameworks.

    Args:
        frameworks: List of framework names (normalized, lowercase)
        commit_sha_9: Git commit SHA (9 chars)
        repo_path: Path to the repository
        verbose: If True, show commands in tree
    """
    print("\n" + "=" * 80)
    print("DEPENDENCY TREE VISUALIZATION")
    print("=" * 80)
    print(f"\nRepository: {repo_path}")
    print(f"Commit SHA: {commit_sha_9}")
    print(f"Frameworks: {', '.join(f.upper() for f in frameworks)}")
    print()

    for framework in frameworks:
        # Create task graph for this framework
        tasks = create_task_graph(framework, commit_sha_9, repo_path, build_method=BUILD_METHOD_RENDER_PY)

        # Create pipeline and add tasks
        pipeline = BuildPipeline()
        for task in tasks.values():
            pipeline.add_task(task)

        # Build children relationships after all tasks are added
        pipeline._build_children_relationships()

        # Show framework header
        framework_display = get_framework_display_name(framework)
        print("\n" + "=" * 80)
        print(f"{framework_display} PIPELINE")
        print("=" * 80)
        print()

        # Helper function to print tree recursively
        def print_tree(task_id: str, prefix: str = "", is_last: bool = True, visited: Optional[set] = None) -> None:
            if visited is None:
                visited = set()

            if task_id in visited:
                return
            visited.add(task_id)

            task = pipeline.tasks[task_id]

            # Determine the connector for this task
            connector = "└─ " if is_last else "├─ "

            # Print this task
            suffix = ""
            if task.run_even_if_deps_fail:
                suffix += " [runs even if dependencies fail]"
            if len(task.parents) > 1:
                parents_str = ", ".join(task.parents)
                suffix += f" [parents: {parents_str}]"
            print(f"{prefix}{connector}{task.task_id} ({task.__class__.__name__}){suffix}")

            # Print commands if verbose mode is enabled
            if verbose:
                continuation = "   " if is_last else "│  "

                # Get both commands to determine numbering
                command = task.get_command(repo_path)
                docker_cmd = task.extract_docker_command_from_build_dryrun(repo_path)

                # Only show "1." if there's also a "2." (docker command exists)
                if docker_cmd:
                    # Two-level command structure
                    print(f"{prefix}{continuation}1. {sanitize_token(command)}")

                    # 2. Print the underlying docker command
                    # Docker command may be multiline (multiple docker build commands)
                    docker_lines = docker_cmd.split('\n')
                    for i, line in enumerate(docker_lines):
                        if i == 0:
                            print(f"{prefix}{continuation}2. {sanitize_token(line)}")
                        else:
                            # Additional lines with same continuation prefix
                            print(f"{prefix}{continuation}   {sanitize_token(line)}")
                else:
                    # Single command - no numbering needed
                    print(f"{prefix}{continuation}{sanitize_token(command)}")

            # Print children (tasks that depend on this one)
            children = task.children
            for i, child_id in enumerate(children):
                is_last_child = (i == len(children) - 1)
                # Determine the prefix for the child
                extension = "   " if is_last else "│  "
                print_tree(child_id, prefix + extension, is_last_child, visited)

        # Find root tasks (tasks with no dependencies)
        root_tasks = [task_id for task_id, task in pipeline.tasks.items() if not task.parents]

        # Print tree starting from each root
        for root_id in root_tasks:
            print_tree(root_id)
            print()  # Blank line between root trees

    print()  # Blank line after tree
    sys.stdout.flush()  # Ensure tree prints before any logger output


def check_running_build_processes() -> list[str]:
    """
    Check for actually running docker build processes using ps.

    This is more reliable than checking .RUNNING marker files, which can become
    stale if a build crashes or is killed without cleanup.

    Returns:
        List of running build process descriptions (empty if none found)
    """
    try:
        # Check for running docker build processes
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True,
            timeout=5
        )

        running_builds = []
        for line in result.stdout.splitlines():
            # Look for docker build commands and build_images.py processes
            if 'docker build' in line or ('python' in line and 'build_images.py' in line and '--parallel' not in line):
                # Exclude the current process (this check) and grep itself
                if 'ps aux' not in line and 'grep' not in line:
                    # Extract just the relevant part of the command
                    parts = line.split()
                    if len(parts) >= 11:
                        cmd = ' '.join(parts[10:])[:80]  # First 80 chars of command
                        running_builds.append(cmd)

        return running_builds
    except Exception as e:
        # If ps check fails, return empty list to allow build to proceed
        return []


def main() -> int:
    """Main entry point.

    SHA naming convention used throughout this module:
      commit_sha_9:          9-char git commit hash (e.g., "a798e08c8")
      image_sha_6:           6-char uppercase hash of container/ dir content (e.g., "E04427")
      image_and_commit_sha:  combined image_sha_6.commit_sha_9 (e.g., "E04427.a798e08c8")
    """
    if "--repo-sha" in sys.argv:
        print("ERROR: --repo-sha has been renamed to --commit-sha. Please update your command.", file=sys.stderr)
        sys.exit(1)

    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger("main")
    global _frameworks_data_cache
    global _html_out_file

    # Get repository info
    repo_path = args.repo_path.resolve()
    if not repo_path.exists():
        logger.error(f"Repository path does not exist: {repo_path}")
        return 1

    # Validate conflicting flags early
    if args.pull_latest and args.commit_sha_9:
        logger.error("Cannot use --pull-latest and --commit-sha together. They conflict in intent.")
        return 1

    # Blocklisted commits: never allow user to specify or checkout
    if args.commit_sha_9 and args.commit_sha_9[:9].lower() in _BLOCKLISTED_9:
        logger.error(
            f"Commit SHA {args.commit_sha_9[:9]} is blocklisted ({BLOCKLISTED_COMMIT_SHAS}). "
            "Cannot be built or checked out. Use a different commit."
        )
        return 1

    # HTML-only mode: generate report without running tasks, lockfiles, or rebuild checks
    if args.generate_html_only:
        if args.pull_latest:
            logger.warning("--pull-latest ignored in --generate-html-only mode")

        # Get commit SHA and checkout if needed (consistent with normal mode)
        try:
            if git is None:
                raise RuntimeError("GitPython not installed; cannot resolve repo SHA")
            repo = git.Repo(repo_path)

            if args.commit_sha_9:
                current_commit_sha_9 = repo.head.commit.hexsha[:9]
                target_commit_sha_9 = args.commit_sha_9[:9]
                if current_commit_sha_9 != target_commit_sha_9:
                    logger.info(f"Checking out SHA: {args.commit_sha_9}")
                    repo.git.checkout(args.commit_sha_9)
                commit_sha_9 = repo.head.commit.hexsha[:9]
            else:
                commit_sha_9 = repo.head.commit.hexsha[:9]

                # Resolve to the commit that first introduced this Docker image SHA
                dynamo_repo_utils_resolve = DynamoRepositoryUtils(repo_path)
                resolved_commit_sha_9, image_sha_6 = dynamo_repo_utils_resolve.find_docker_image_sha_origin(commit_sha_9)
                if resolved_commit_sha_9 != commit_sha_9:
                    logger.info(f"Auto-resolved build commit: {commit_sha_9} -> {resolved_commit_sha_9} "
                                f"(Docker image SHA {image_sha_6})")
                    if resolved_commit_sha_9[:9].lower() in _BLOCKLISTED_9:
                        logger.error(
                            f"Resolved commit {resolved_commit_sha_9} is blocklisted ({BLOCKLISTED_COMMIT_SHAS}). "
                            "Cannot checkout or build."
                        )
                        return 1
                    repo.git.checkout(resolved_commit_sha_9)
                    commit_sha_9 = resolved_commit_sha_9
                else:
                    logger.info(f"HEAD {commit_sha_9} is the commit that introduced Docker image SHA {image_sha_6}")
        except Exception as e:
            logger.error(f"Failed to get/checkout commit SHA: {e}")
            return 1

        # Determine which frameworks to include
        if args.framework:
            frameworks: List[str] = []
            for f in args.framework:
                frameworks.extend([normalize_framework(fw.strip()) for fw in f.split(',')])
        else:
            frameworks = FRAMEWORKS

        # Prepare log dir (used for linking to historical logs + marker discovery)
        log_date = datetime.now().strftime("%Y-%m-%d")
        log_dir = repo_path / "logs" / log_date
        log_dir.mkdir(parents=True, exist_ok=True)

        # Detect build method (render.py vs build.sh)
        try:
            build_method = detect_build_method(repo_path)
        except RuntimeError as e:
            html_out = args.html_out if args.html_out is not None else (repo_path / "build.html")
            return _generate_error_html_and_exit(
                frameworks=frameworks,
                commit_sha_9=commit_sha_9,
                repo_path=repo_path,
                error_message=str(e),
                log_dir=log_dir,
                log_date=log_date,
                html_out_file=html_out,
            )
        logger.info(f"Build method: {build_method}")

        # Create task graphs (also determines image names and versioning)
        all_tasks: Dict[str, BaseTask] = {}
        version_extraction_failed = False
        version_error_message = ""
        for framework in frameworks:
            if build_method == BUILD_METHOD_RENDER_PY:
                # render.py method: use IMAGE_SHA.commit_sha_9 as version (tag: dynamo:{IMAGE_SHA}.{commit_sha_9}-{fw}-cuda{ver}-{target})
                version = commit_sha_9
                logger.info(f"  Using SHA as version for {framework.upper()}: {version}")
            else:
                logger.info(f"Extracting version from build.sh for {framework.upper()}...")
                try:
                    version = extract_version_from_build_sh(framework, repo_path)
                    logger.info(f"  Version: {version}")
                except RuntimeError as e:
                    version_error_message = str(e)
                    version_extraction_failed = True
                    break
            framework_tasks = create_task_graph(framework, commit_sha_9, repo_path, version=version, build_method=build_method)
            all_tasks.update(framework_tasks)

        if version_extraction_failed:
            html_out = args.html_out if args.html_out is not None else (repo_path / "build.html")
            return _generate_error_html_and_exit(
                frameworks=frameworks,
                commit_sha_9=commit_sha_9,
                repo_path=repo_path,
                error_message=version_error_message,
                log_dir=log_dir,
                log_date=log_date,
                html_out_file=html_out,
            )

        # Compute image_and_commit_sha (IMAGE_SHA.commit_sha_9 format)
        dru_html = DynamoRepositoryUtils(repo_path)
        image_sha_6 = dru_html.generate_docker_image_sha_for_commit(commit_sha_9)
        if image_sha_6 not in ("ERROR", "NO_CONTAINER_DIR"):
            image_and_commit_sha = f"{image_sha_6}.{commit_sha_9}"
        else:
            image_and_commit_sha = commit_sha_9
            logger.warning(f"Could not compute Docker image SHA; using plain SHA for filenames")
        logger.info(f"File SHA: {image_and_commit_sha}")

        # Set log_file paths for all tasks (needed for HTML generation)
        for task_id, task in all_tasks.items():
            task.set_log_file_path(log_dir, log_date, image_and_commit_sha)

        # Initialize frameworks data cache once (loads previous logs and image info)
        _frameworks_data_cache = initialize_frameworks_data_cache(
            all_tasks=all_tasks,
            log_dir=log_dir,
            date_str=log_date,
            image_and_commit_sha=image_and_commit_sha,
            repo_path=repo_path,
            use_absolute_urls=False,
            build_method=build_method,
        )

        # Generate and write HTML report
        html_content = generate_html_report(
            all_tasks=all_tasks,
            repo_path=repo_path,
            image_and_commit_sha=image_and_commit_sha,
            log_dir=log_dir,
            date_str=log_date,
            use_absolute_urls=False,
        )

        html_out = args.html_out if args.html_out is not None else (repo_path / "build.html")
        html_out.parent.mkdir(parents=True, exist_ok=True)
        html_out.write_text(_rewrite_html_links_for_repo_root(html_content, image_and_commit_sha=image_and_commit_sha, date_str=log_date))
        logger.info(f"HTML report written: {html_out}")
        return 0

    # Lock file lives outside the repo so git operations (reset, pull, checkout) can't destroy it.
    _lock_hash = hashlib.md5(str(repo_path.resolve()).encode()).hexdigest()[:8]
    lock_file = Path(f"/tmp/build_images.{_lock_hash}.lock")
    if lock_file.exists() and not args.run_no_matter_what:
        try:
            with open(lock_file, 'r') as f:
                lock_info = f.read().strip().split('\n')
                lock_pid = int(lock_info[0])
                lock_time = lock_info[1] if len(lock_info) > 1 else "unknown"

            # Check if the process is still running
            try:
                os.kill(lock_pid, 0)  # Signal 0 checks if process exists
                logger.warning(f"Another instance (PID {lock_pid}) is already running since {lock_time}")
                logger.info("Exiting to avoid concurrent builds")
                return 0  # Exit gracefully, not an error
            except OSError:
                # Process doesn't exist, stale lock file
                logger.warning(f"Found stale lock file from PID {lock_pid}, removing it")
                lock_file.unlink()
        except (ValueError, IndexError, IOError) as e:
            logger.warning(f"Invalid lock file, removing it: {e}")
            lock_file.unlink()

    # Create lock file with current PID and timestamp
    logger.info(f"Acquiring lock: {lock_file}")
    try:
        with open(lock_file, 'w') as f:
            f.write(f"{os.getpid()}\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    except IOError as e:
        logger.error(f"Failed to create lock file: {e}")
        return 1

    # Ensure lock file is removed on exit (only if we own it)
    _our_pid = os.getpid()

    def cleanup_lock():
        try:
            if lock_file.exists():
                with open(lock_file, 'r') as f:
                    stored_pid = int(f.readline().strip())
                if stored_pid == _our_pid:
                    lock_file.unlink()
        except (ValueError, IOError, OSError):
            pass
    atexit.register(cleanup_lock)

    # Pull latest code if requested
    if args.pull_latest:
        # Check if another build process is actually running using ps
        running_processes = check_running_build_processes()
        if running_processes:
            logger.warning(f"Skipping pull - found {len(running_processes)} running build process(es):")
            for proc in running_processes[:5]:  # Show first 5
                logger.warning(f"  {proc}")
            if len(running_processes) > 5:
                logger.warning(f"  ... and {len(running_processes) - 5} more")
            logger.info("Will retry pull on next run when builds complete")
            # Don't return error - just skip the pull and continue
            args.pull_latest = False  # Disable pull for this run

        if args.pull_latest:  # Only proceed if not disabled above
            logger.info("Pulling latest code from main branch...")
            try:
                # Initialize GitUtils to check repo status
                git_utils_temp = GitUtils(repo_path)

                # Hard reset to clean up any uncommitted changes
                logger.info("Doing hard reset to clean repository state...")
                git_utils_temp.repo.git.reset('--hard', 'HEAD')

                # Clean untracked files (except Docker image SHA tracking files)
                sha_files_keep = {DynamoRepositoryUtils.DOCKER_IMAGE_SHA_FILE, DynamoRepositoryUtils.COMPILATION_SHA_FILE}
                untracked = git_utils_temp.repo.untracked_files
                if untracked:
                    logger.info(f"Removing {len(untracked)} untracked file(s)...")
                    for file in untracked:
                        if file not in sha_files_keep:
                            file_path = repo_path / file
                            try:
                                if file_path.is_file():
                                    file_path.unlink()
                                elif file_path.is_dir():
                                    shutil.rmtree(file_path)
                                logger.debug(f"  Removed {file}")
                            except Exception as e:
                                logger.warning(f"  Could not remove {file}: {e}")

                # Checkout main branch and pull latest
                logger.info("Checking out main branch...")
                git_utils_temp.repo.git.checkout('main')
                origin = git_utils_temp.repo.remotes.origin
                logger.info("Pulling latest from origin/main...")
                origin.pull('main')
                pulled_sha = git_utils_temp.repo.head.commit.hexsha[:9]
                logger.info(f"✓ Successfully pulled latest code from main (HEAD: {pulled_sha})")
                args.commit_sha_9 = pulled_sha
            except Exception as e:
                logger.error(f"Failed to pull latest code: {e}")
                return 1

    # Get commit SHA and checkout if needed
    try:
        git_utils = GitUtils(repo_path)
        repo = git.Repo(repo_path)

        effective_commit_sha_9 = args.commit_sha_9

        if effective_commit_sha_9:
            # Check if we need to checkout the specified SHA
            current_commit_sha_9 = repo.head.commit.hexsha[:9]
            target_commit_sha_9 = effective_commit_sha_9[:9]

            if current_commit_sha_9 != target_commit_sha_9:
                # Reset tracked *.lock files and delete untracked ones before checkout
                lock_files = list(repo_path.glob('**/*.lock'))
                if lock_files:
                    tracked_files = set(repo.git.ls_files('--full-name').splitlines())
                    restored, deleted = 0, 0
                    for lock_file in lock_files:
                        rel = str(lock_file.relative_to(repo_path))
                        if rel in tracked_files:
                            try:
                                repo.git.restore(rel)
                                restored += 1
                            except Exception as e:
                                logger.warning(f"  Could not reset {rel}: {e}")
                        else:
                            lock_file.unlink(missing_ok=True)
                            deleted += 1
                    logger.info(f"Lock files: {restored} restored (tracked), {deleted} deleted (untracked)")

                logger.info(f"Checking out SHA: {effective_commit_sha_9}")
                repo.git.checkout(effective_commit_sha_9)
                logger.info(f"Checked out {effective_commit_sha_9}")
                commit_sha_9 = repo.head.commit.hexsha[:9]
            else:
                logger.info(f"Already at SHA: {effective_commit_sha_9}")
                commit_sha_9 = effective_commit_sha_9[:9]
        else:
            commit_sha_40 = git_utils.get_current_commit()
            commit_sha_9 = commit_sha_40[:9]

            # Resolve to the commit that first introduced this Docker image SHA
            dynamo_repo_utils_resolve = DynamoRepositoryUtils(repo_path)
            resolved_commit_sha_9, image_sha_6 = dynamo_repo_utils_resolve.find_docker_image_sha_origin(commit_sha_9)
            if resolved_commit_sha_9 != commit_sha_9:
                logger.info(f"Auto-resolved build commit: {commit_sha_9} -> {resolved_commit_sha_9} "
                            f"(Docker image SHA {image_sha_6})")
                if resolved_commit_sha_9[:9].lower() in _BLOCKLISTED_9:
                    logger.error(
                        f"Resolved commit {resolved_commit_sha_9} is blocklisted ({BLOCKLISTED_COMMIT_SHAS}). "
                        "Cannot checkout or build."
                    )
                    return 1
                repo.git.checkout(resolved_commit_sha_9)
                commit_sha_9 = resolved_commit_sha_9
            else:
                logger.info(f"HEAD {commit_sha_9} is the commit that introduced Docker image SHA {image_sha_6}")
    except Exception as e:
        logger.error(f"Failed to get/checkout commit SHA: {e}")
        return 1

    # Store initial SHA globally for detecting mid-build commits
    global _initial_commit_sha_9
    _initial_commit_sha_9 = commit_sha_9
    logger.info(f"Initial SHA: {commit_sha_9}")

    # Determine which frameworks to build
    if args.framework:
        # Support both comma-separated and multiple --framework flags
        # e.g., --framework vllm,sglang OR --framework vllm --framework sglang
        frameworks = []
        for f in args.framework:
            # Split on comma in case user provided comma-separated list
            frameworks.extend([normalize_framework(fw.strip()) for fw in f.split(',')])
    else:
        # In sanity-check-only mode, default to the lightweight "none" sanity checks
        # unless the user explicitly requested frameworks via --framework.
        frameworks = ["none"] if args.sanity_check_only else FRAMEWORKS

    # Handle --show-tree flag: show dependency tree
    if args.show_tree:
        print_dependency_tree(frameworks, commit_sha_9, repo_path, verbose=args.verbose)

    # Handle --no-build + --image-sha: determine which image tags to use
    # image_version overrides the version in Docker tags (dynamo:<version>-fw-cuda-target)
    # but commit_sha_9 stays as the repo commit SHA for logs, reports, and filenames.
    no_build = getattr(args, 'no_build', False)
    image_tag_override = getattr(args, 'image_tag_override', None)
    image_version = None  # None = use default (commit_sha_9 for render_py, build.sh output otherwise)
    if image_tag_override:
        image_version = image_tag_override
        logger.info(f"Using --image-sha override for image tags: {image_version}")
    elif no_build:
        detected = detect_latest_image_sha(repo_path, frameworks[0])
        if detected:
            image_version = detected
            logger.info(f"Auto-detected image SHA from local Docker images: {image_version}")
        else:
            logger.error(f"--no-build requires existing images but none found for {frameworks[0]}. "
                         "Use --image-sha to specify explicitly.")
            return 1

    # image_and_commit_sha: used in log/report filenames and Docker tags.
    # Format: "<IMAGE_SHA_UPPER>.<commit_sha_9>" where IMAGE_SHA is a 6-char uppercase
    # content hash of container/ files. Image SHA first for visual dominance.
    dru = DynamoRepositoryUtils(repo_path)
    image_sha_6 = dru.generate_docker_image_sha_for_commit(
        image_version if (image_version and image_version != commit_sha_9) else commit_sha_9
    )
    if image_sha_6 not in ("ERROR", "NO_CONTAINER_DIR"):
        image_and_commit_sha = f"{image_sha_6}.{commit_sha_9}"
    else:
        image_and_commit_sha = commit_sha_9
        logger.warning(f"Could not compute Docker image SHA (got {image_sha_6}); using plain SHA for filenames")
    logger.info(f"File SHA: {image_and_commit_sha}")
    logger.info(f"PID: {os.getpid()}")

    # --reuse-dev-if-image-exists: if dev image with same Image SHA exists, reuse it (no builds, no retag)
    reuse_mode = False
    reuse_dev_tags: Optional[Dict[str, str]] = None
    if getattr(args, "reuse_dev_if_image_exists", False):
        reuse_dev_tags = _check_reuse_dev_images_exist(repo_path, image_and_commit_sha, frameworks)
        if reuse_dev_tags:
            reuse_mode = True
            logger.info("Reuse mode: all images present — running compilation and sanity only (no builds)")

    # Execution mode header
    if args.dry_run:
        logger.info("DRY-RUN MODE: Showing commands that would be executed")
    elif no_build:
        logger.info("DynamoDockerBuilder V2 - Starting (--no-build: compilation + sanity only)")
    elif reuse_mode:
        logger.info("DynamoDockerBuilder V2 - Starting (reuse mode: compilation + sanity only)")
    else:
        logger.info("DynamoDockerBuilder V2 - Starting")

        # Check if rebuild is needed based on Docker image SHA
        # Only check in non-dry-run mode to avoid writing .last_docker_image_sha
        # Skip this check in --no-build and reuse mode (no builds to skip)
        dynamo_repo_utils = DynamoRepositoryUtils(repo_path)
        if not dynamo_repo_utils.check_if_rebuild_needed(force_run=bool(args.run_no_matter_what)):
            logger.info("✅ No rebuild needed - exiting")
            return 0

    # Track overall execution time
    execution_start_time = time.time()

    # Keep a stable "latest report" output path updated during execution
    # (defaults to <repo-path>/build.html)
    _html_out_file = args.html_out if args.html_out is not None else (repo_path / "build.html")

    # Setup log directory: logs/YYYY-MM-DD/ (skip in dry-run)
    if not args.dry_run:
        log_date = datetime.now().strftime("%Y-%m-%d")
        log_dir = repo_path / "logs" / log_date
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Log directory: {log_dir}")

        # Clean up existing log files and marker files for this SHA and frameworks
        # Only delete files for tasks that will be executed (very specific)
        # Pattern: YYYY-MM-DD.{commit_sha_9}.{task-name}.{log|PASS|FAIL}
        cleaned_count = 0
        # We'll populate this after creating task graphs, so skip cleanup for now
        # Cleanup will happen per-task before execution

    # Detect build method (render.py vs build.sh)
    try:
        build_method = detect_build_method(repo_path)
    except RuntimeError as e:
        if not args.dry_run:
            return _generate_error_html_and_exit(
                frameworks=frameworks,
                commit_sha_9=commit_sha_9,
                repo_path=repo_path,
                error_message=str(e),
                log_dir=log_dir,
                log_date=log_date,
                html_out_file=_html_out_file,
                email=args.email,
                hostname=args.hostname,
                html_path=args.html_path,
            )
        else:
            logger.error(f"Build method detection failed: {e}")
            return 1
    logger.info(f"Build method: {build_method}")

    # Create task graph for each framework
    # For build_sh: extract version once per framework to avoid repeated build.sh calls
    # For render_py: use IMAGE_SHA.commit_sha_9 as version (e.g. dynamo:C62194.6d3e0137c-none-cuda13.0-runtime)
    all_tasks = {}
    version_extraction_failed = False
    version_error_message = ""
    for framework in frameworks:
        if reuse_mode:
            version = image_and_commit_sha
            logger.info(f"  Reuse mode: {framework.upper()} version {version}")
        elif image_version:
            version = image_version
            logger.info(f"  Using image version for {framework.upper()}: {version} (repo SHA: {commit_sha_9})")
        elif build_method == BUILD_METHOD_RENDER_PY:
            version = image_and_commit_sha
            logger.info(f"  Using SHA as version for {framework.upper()}: {version}")
        else:
            logger.info(f"Extracting version from build.sh for {framework.upper()}...")
            try:
                version = extract_version_from_build_sh(framework, repo_path)
                logger.info(f"  Version: {version}")
            except RuntimeError as e:
                # Version extraction failed (build.sh missing or broken at this SHA).
                # Generate an error HTML report instead of silently exiting.
                version_error_message = str(e)
                version_extraction_failed = True
                break
        framework_tasks = (
            create_task_graph_reuse(framework, commit_sha_9, repo_path, dev_image_tag=reuse_dev_tags[framework], build_method=build_method)
            if reuse_mode and reuse_dev_tags
            else create_task_graph(framework, commit_sha_9, repo_path, version=version, build_method=build_method)
        )
        all_tasks.update(framework_tasks)

    if version_extraction_failed:
        if not args.dry_run:
            return _generate_error_html_and_exit(
                frameworks=frameworks,
                commit_sha_9=commit_sha_9,
                repo_path=repo_path,
                error_message=version_error_message,
                log_dir=log_dir,
                log_date=log_date,
                html_out_file=_html_out_file,
                email=args.email,
                hostname=args.hostname,
                html_path=args.html_path,
            )
        else:
            logger.error(f"Version extraction failed: {version_error_message}")
            return 1

    # Set log_file paths for all tasks (needed for HTML generation to find existing logs)
    if not args.dry_run:
        for task_id, task in all_tasks.items():
            task.set_log_file_path(log_dir, log_date, image_and_commit_sha)

        # Initialize frameworks data cache once (this loads previous logs and image info)
        _frameworks_data_cache = initialize_frameworks_data_cache(
            all_tasks=all_tasks,
            log_dir=log_dir,
            date_str=log_date,
            image_and_commit_sha=image_and_commit_sha,
            repo_path=repo_path,
            use_absolute_urls=args.email is not None,
            build_method=build_method,
        )
        logger.info("Frameworks data cache initialized")
        # Note: HTML report will be generated when first task starts running

    # Filter tasks based on --target (default: all targets)
    requested_targets = {t.strip() for t in args.target.split(",")}
    all_targets = {"runtime", "dev", "local-dev"}
    if requested_targets != all_targets:
        def _task_target(task_id: str) -> str:
            """Map task_id to its target: 'local-dev', 'dev', or 'runtime'."""
            # Strip framework prefix (e.g. "vllm-" from "vllm-runtime-build")
            rest = task_id.split("-", 1)[1] if "-" in task_id else task_id
            if rest.startswith("local-dev"):
                return "local-dev"
            if rest.startswith("dev"):
                return "dev"
            return "runtime"

        needed_targets = set(requested_targets)

        remove_ids = set()
        for task_id in all_tasks:
            if _task_target(task_id) not in needed_targets:
                remove_ids.add(task_id)

        if remove_ids:
            excluded = all_targets - needed_targets
            logger.info(f"--target {args.target}: removing {len(remove_ids)} tasks for excluded targets ({', '.join(sorted(excluded))})")
            for task_id in remove_ids:
                del all_tasks[task_id]
            for task in all_tasks.values():
                task.parents = [p for p in task.parents if p not in remove_ids]
                task.children = [c for c in task.children if c not in remove_ids]
                # Dockerfiles are self-contained (render.py); no cross-target image deps
                task.input_image = None

    # Filter tasks based on --sanity-check-only
    if args.sanity_check_only:
        logger.info("Sanity-check-only mode: skipping builds and compilation")

        # Mark build and compilation tasks as SKIPPED, keep sanity checks
        sanity_tasks = {}
        for task_id, task in all_tasks.items():
            if 'sanity' in task_id:
                # Keep sanity check tasks
                sanity_tasks[task_id] = task
                # Clear dependencies since we're not building
                task.parents = []
                task.children = []
            elif 'compilation' in task_id or 'chown' in task_id:
                # Mark compilation and chown tasks as skipped (for HTML report)
                task.mark_status_as(TaskStatus.SKIPPED, "Not included in sanity-only mode")
                sanity_tasks[task_id] = task
            elif isinstance(task, BuildTask):
                # Mark build tasks as skipped
                task.mark_status_as(TaskStatus.SKIPPED, "Not included in sanity-only mode")
                # Add to sanity_tasks so they show in HTML report with image size
                sanity_tasks[task_id] = task

        all_tasks = sanity_tasks
        logger.info(f"Filtered to {len([t for t in all_tasks.values() if 'sanity' in t.task_id])} sanity check tasks")

    # Filter tasks based on --no-build (skip builds, run compilation + sanity against existing images)
    if getattr(args, 'no_build', False):
        logger.info("No-build mode: skipping builds, running compilation and sanity only")

        build_task_ids = set()
        for task_id, task in all_tasks.items():
            if isinstance(task, BuildTask):
                build_task_ids.add(task_id)
            # Also skip compress, upload, and cleanup tasks (they depend on building)
            elif any(x in task_id for x in ('compress', 'upload', 'orig-cleanup')):
                build_task_ids.add(task_id)

        for task_id in build_task_ids:
            all_tasks[task_id].mark_status_as(TaskStatus.SKIPPED, "Skipped (--no-build)")

        # Remove build dependencies from compilation/sanity/chown tasks
        for task_id, task in all_tasks.items():
            if task_id not in build_task_ids:
                task.parents = [p for p in task.parents if p not in build_task_ids]

        runnable = [tid for tid, t in all_tasks.items() if tid not in build_task_ids]
        logger.info(f"Will run {len(runnable)} tasks (compilation + sanity + chown)")

    # Skip compilation if stored .last_compilation_sha matches current commit
    dynamo_repo_utils_comp = DynamoRepositoryUtils(repo_path)
    stored_comp_sha = dynamo_repo_utils_comp.get_stored_compilation_sha()
    if stored_comp_sha and stored_comp_sha == commit_sha_9 and not args.run_no_matter_what:
        comp_skip_ids = set()
        for task_id in all_tasks:
            if 'compilation' in task_id or 'chown' in task_id:
                comp_skip_ids.add(task_id)

        if comp_skip_ids:
            logger.info(f"Compilation SHA unchanged ({commit_sha_9}) — skipping {len(comp_skip_ids)} compilation/chown tasks")
            for task_id in comp_skip_ids:
                all_tasks[task_id].mark_status_as(TaskStatus.SKIPPED, f"Compilation SHA unchanged ({commit_sha_9})")

            for task_id, task in all_tasks.items():
                if task_id not in comp_skip_ids:
                    task.parents = [p for p in task.parents if p not in comp_skip_ids]
    elif stored_comp_sha and stored_comp_sha != commit_sha_9 and not args.dry_run:
        # Compilation SHA changed — remove stale PASSED markers for compilation/chown/sanity
        # so --skip-action-if-already-passed doesn't skip tasks that must re-run.
        stale_cleared = 0
        for task_id, task in all_tasks.items():
            if 'compilation' in task_id or 'chown' in task_id or 'sanity' in task_id:
                if task.log_file:
                    passed_marker = task.log_file.with_suffix(f'.{MARKER_PASSED}')
                    if passed_marker.exists():
                        passed_marker.unlink()
                        stale_cleared += 1
        if stale_cleared:
            logger.info(f"Compilation SHA changed ({stored_comp_sha} → {commit_sha_9}) — cleared {stale_cleared} stale PASSED markers")

    # Execute tasks in dependency order
    mode = "parallel" if args.parallel else "sequential"
    if args.dry_run:
        logger.info(f"Would execute {len(all_tasks)} tasks {mode}ly")
        logger.info("")
    else:
        logger.info(f"Executing {len(all_tasks)} tasks {mode}ly")

    # Find root tasks (tasks with no dependencies)
    root_tasks = [task_id for task_id, task in all_tasks.items() if not task.parents]

    # Build children relationships first
    for task_id, task in all_tasks.items():
        for parent_id in task.parents:
            if parent_id in all_tasks and task_id not in all_tasks[parent_id].children:
                all_tasks[parent_id].children.append(task_id)

    # Setup basic signal handler for both parallel and sequential modes
    # (will be enhanced in parallel mode to stop background HTML updater)
    def basic_signal_handler(signum, frame):
        """Handle SIGTERM/SIGINT by marking running tasks as KILLED"""
        logger = logging.getLogger("signal")
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        logger.warning(f"\nReceived {sig_name}, marking running tasks as KILLED...")

        # Terminate all running subprocesses first
        with _subprocesses_lock:
            subprocesses_to_terminate = list(_running_subprocesses)

        if subprocesses_to_terminate:
            logger.info(f"  Terminating {len(subprocesses_to_terminate)} running subprocess(es)...")
            for proc in subprocesses_to_terminate:
                try:
                    proc.terminate()
                except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
                    pass  # Ignore errors if process already exited

            # Wait up to 2 seconds for processes to terminate
            time.sleep(2)

            # Force kill any remaining processes
            for proc in subprocesses_to_terminate:
                try:
                    if proc.poll() is None:  # Still running
                        proc.kill()
                except Exception:  # THIS IS A HORRIBLE ANTI-PATTERN, FIX IT
                    pass

        # Mark all running tasks as KILLED
        for task_id, task in all_tasks.items():
            if task.status == TaskStatus.RUNNING:
                task.mark_status_as(TaskStatus.KILLED, f"Interrupted by {sig_name}")
                update_frameworks_data_cache(task, use_absolute_urls=False)
                logger.info(f"  Marked {task_id} as KILLED")

        # Generate final HTML report
        if log_dir and log_date and not args.dry_run:
            try:
                html_content = generate_html_report(
                    all_tasks=all_tasks,
                    repo_path=repo_path,
                    image_and_commit_sha=image_and_commit_sha,
                    log_dir=log_dir,
                    date_str=log_date,
                    use_absolute_urls=False,
                )
                _write_report_and_json(html_content, all_tasks, repo_path, image_and_commit_sha, log_dir, log_date)
                logger.info(f"  Generated final HTML report: {log_dir / image_and_commit_sha / 'report.html'}")
            except Exception as e:
                logger.error(f"  Failed to generate final HTML report: {e}")

        sys.exit(1)

    signal.signal(signal.SIGTERM, basic_signal_handler)
    signal.signal(signal.SIGINT, basic_signal_handler)

    # Set report flags so HTML can show "Disabled (--no-compress)" / "Disabled (--no-upload)" when applicable
    global _report_no_compress, _report_no_upload, _report_no_build
    _report_no_compress = getattr(args, 'no_compress', False)
    _report_no_upload = getattr(args, 'no_upload', False)
    _report_no_build = getattr(args, 'no_build', False)

    # Execute based on mode
    if args.parallel:
        # Parallel execution
        # Determine max_workers: use CLI arg, or default to CPU count
        max_workers = args.max_workers if args.max_workers is not None else os.cpu_count()
        logger.info(f"Using {max_workers} worker threads for parallel execution")
        
        executed_tasks, failed_tasks = execute_task_parallel(
            all_tasks=all_tasks,
            root_tasks=root_tasks,
            repo_path=repo_path,
            image_and_commit_sha=image_and_commit_sha,
            log_dir=log_dir if not args.dry_run else None,
            log_date=log_date if not args.dry_run else None,
            dry_run=args.dry_run,
            skip_if_passed=args.skip_action_if_already_passed,
            no_compile=args.no_compile,
            enable_dev_upload=not args.no_upload,
            enable_dev_compress=not args.no_compress,
            max_workers=max_workers,
        )
    else:
        # Sequential execution
        executed_tasks = set()
        failed_tasks = set()
        for root_id in root_tasks:
            execute_task_sequential(
                all_tasks=all_tasks,
                executed_tasks=executed_tasks,
                failed_tasks=failed_tasks,
                task_id=root_id,
                repo_path=repo_path,
                image_and_commit_sha=image_and_commit_sha,
                log_dir=log_dir if not args.dry_run else None,
                log_date=log_date if not args.dry_run else None,
                dry_run=args.dry_run,
                skip_action_if_already_passed=args.skip_action_if_already_passed,
                no_compile=args.no_compile,
                enable_dev_upload=not args.no_upload,
                enable_dev_compress=not args.no_compress,
            )

    # Calculate total execution time
    execution_end_time = time.time()
    total_duration = execution_end_time - execution_start_time

    # Format duration nicely
    hours, remainder = divmod(int(total_duration), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        duration_str = f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        duration_str = f"{minutes}m {seconds}s"
    else:
        duration_str = f"{seconds}s"

    # Report summary
    if args.dry_run:
        logger.info(f"\nDry-run Summary:")
        logger.info(f"  Total tasks: {len(all_tasks)}")
        logger.info(f"  Time: {duration_str} ({total_duration:.2f}s)")

        # In dry-run mode, still generate HTML report and send email if requested
        # (but don't execute actual tasks)
        if args.email:
            logger.info(f"  Note: Email notifications are not sent in dry-run mode")

        return 0
    else:
        success_count = len([t for t in all_tasks.values() if t.status == TaskStatus.PASSED])
        failed_count = len([t for t in all_tasks.values() if t.status == TaskStatus.FAILED])
        skipped_count = len([t for t in all_tasks.values() if t.status == TaskStatus.SKIPPED])

        # Exclude none-compilation and none-sanity tasks from exit status calculation
        # (they're expected to fail since framework=none has no inference frameworks)
        critical_failures = [
            task_id for task_id, t in all_tasks.items()
            if t.status == TaskStatus.FAILED and not (task_id.startswith('none-') and ('compilation' in task_id or 'sanity' in task_id))
        ]
        exit_status = 0 if len(critical_failures) == 0 else 1

        # Store compilation SHA on success (tracks HEAD at last successful compilation)
        if exit_status == 0:
            compilation_tasks = [tid for tid in all_tasks if 'compilation' in tid and all_tasks[tid].status == TaskStatus.PASSED]
            if compilation_tasks:
                dynamo_repo_utils = DynamoRepositoryUtils(repo_path)
                dynamo_repo_utils.store_compilation_sha(commit_sha_9)

        logger.info(f"\nExecution Summary:")
        logger.info(f"  ✓ Success: {success_count}")
        logger.info(f"  ✗ Failed: {failed_count}")
        logger.info(f"  ⊘ Skipped: {skipped_count}")

        # Show detailed task breakdown
        logger.info(f"\n  Task Details:")
        for task_id, task in all_tasks.items():
            if task.status == TaskStatus.PASSED and task.start_time and task.end_time:
                duration = task.end_time - task.start_time
                logger.info(f"    ✓ {task_id}: {duration:.1f}s")
            elif task.status == TaskStatus.FAILED:
                if task.start_time and task.end_time:
                    duration = task.end_time - task.start_time
                    logger.info(f"    ✗ {task_id}: {duration:.1f}s")
                else:
                    logger.info(f"    ✗ {task_id}: failed")
            elif task.status == TaskStatus.SKIPPED:
                logger.info(f"    ⊘ {task_id}: skipped")

        logger.info(f"\n  Logs: {log_dir}")
        logger.info(f"  Total time: {duration_str} ({total_duration:.2f}s)")
        logger.info(f"  Exit status: {exit_status}")
        logger.info(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Generate HTML report (always, regardless of --framework flag)
        # This ensures all frameworks/targets are included in the report
        try:
            # Always generate HTML report with relative paths for file
            html_content_file = generate_html_report(
                all_tasks=all_tasks,
                repo_path=repo_path,
                image_and_commit_sha=image_and_commit_sha,
                log_dir=log_dir,
                date_str=log_date,
                use_absolute_urls=False,
            )

            # Save HTML report, status marker, and JSON build report together
            _write_report_and_json(html_content_file, all_tasks, repo_path, image_and_commit_sha, log_dir, log_date)
            logger.info(f"  HTML + JSON Report saved to: {log_dir}")
        except Exception as e:
            logger.error(f"Failed to generate reports: {e}")
            # Continue to try sending email even if report generation failed

        # Send email notification (separate try-except to avoid hiding email errors)
        if args.email:
            try:
                # Generate report with absolute URLs for email
                html_content_email = generate_html_report(
                    all_tasks=all_tasks,
                    repo_path=repo_path,
                    image_and_commit_sha=image_and_commit_sha,
                    log_dir=log_dir,
                    date_str=log_date,
                    use_absolute_urls=True,
                    hostname=args.hostname,
                    html_path=args.html_path,
                )

                # Collect failed task names for the email subject.
                # Keep this consistent with exit-status calculation above: some failures are expected
                # (framework=none compilation/sanity) and should not flip the overall status to FAIL.
                failed_task_names = list(critical_failures)

                # Send email
                email_sent = send_email_notification(
                    email=args.email,
                    html_content=html_content_email,
                    commit_sha_9=commit_sha_9[:7],
                    failed_tasks=failed_task_names,
                )

                if not email_sent:
                    logger.warning(f"⚠️  Email notification was not sent successfully")
            except Exception as e:
                logger.error(f"Failed to send email notification: {e}")
                logger.debug(traceback.format_exc())

        return exit_status


if __name__ == "__main__":
    sys.exit(main())
