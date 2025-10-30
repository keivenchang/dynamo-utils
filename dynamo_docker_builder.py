#!/usr/bin/env python3
"""
DynamoDockerBuilder V2 - Clean OOP Architecture

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
import glob
import logging
import os
import re
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Third-party imports (optional, with error handling)
try:
    import git
except ImportError:
    git = None  # type: ignore

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
except ImportError:
    Environment = None  # type: ignore
    FileSystemLoader = None  # type: ignore
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
)


# ==============================================================================
# CONSTANTS
# ==============================================================================

FRAMEWORKS_UPPER = ["VLLM", "SGLANG", "TRTLLM"]
FRAMEWORKS_LOWER = ["vllm", "sglang", "trtllm"]


class TaskStatus(Enum):
    """Status of a task in the pipeline"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


# ==============================================================================
# TASK GRAPH DEFINITION
# ==============================================================================

# ==============================================================================
# DOCKER COMMAND FILTER FUNCTIONS
# ==============================================================================

def select_command(index: int) -> Callable[[List[str]], Optional[str]]:
    """
    Create a function that selects a specific command by index.
    
    Args:
        index: 0-based index (0 for first, 1 for second, etc.)
        
    Returns:
        Function that selects the command at the given index
    """
    def selector(commands: List[str]) -> Optional[str]:
        if len(commands) <= index:
            # Not enough commands - return what we have
            return '\n'.join(commands) if commands else None
        return commands[index]
    return selector


def filter_out_latest_tag(select_func: Callable[[List[str]], Optional[str]]) -> Callable[[List[str]], Optional[str]]:
    """
    Wrap a selection function to also filter out 'latest' tags.
    
    Returns a function that does: filter_out_latest(select_func(commands))
    """
    def wrapper(commands: List[str]) -> Optional[str]:
        command = select_func(commands)
        if command is None:
            return None
        # Remove --tag arguments that contain 'latest'
        command = re.sub(r'--tag\s+[^\s]*latest[^\s]*', '', command)
        # Clean up whitespace
        command = re.sub(r'\s+', ' ', command).strip()
        return command
    return wrapper


def rename_output_tag(new_tag: str, select_func: Callable[[List[str]], Optional[str]]) -> Callable[[List[str]], Optional[str]]:
    """
    Wrap a selection function to also rename output tags (--tag).
    
    Returns a function that does: rename_output_tag_to(new_tag, select_func(commands))
    """
    def wrapper(commands: List[str]) -> Optional[str]:
        command = select_func(commands)
        if command is None:
            return None
        # Remove all existing --tag arguments
        command = re.sub(r'--tag\s+[^\s-][^\s]*', '', command)
        # Clean up whitespace
        command = re.sub(r'\s+', ' ', command).strip()

        # Add our new tag before the final path argument
        parts = command.rsplit(maxsplit=1)
        if len(parts) == 2:
            return f"{parts[0]} --tag {new_tag} {parts[1]}"
        else:
            return f"{command} --tag {new_tag}"
    return wrapper


def rename_input_tag(new_tag: str, select_func: Callable[[List[str]], Optional[str]]) -> Callable[[List[str]], Optional[str]]:
    """
    Wrap a selection function to also rename input base image (--build-arg DYNAMO_BASE_IMAGE=).
    
    Returns a function that does: rename_input_tag_to(new_tag, select_func(commands))
    """
    def wrapper(commands: List[str]) -> Optional[str]:
        command = select_func(commands)
        if command is None:
            return None
        # Replace DYNAMO_BASE_IMAGE value
        command = re.sub(
            r'--build-arg\s+DYNAMO_BASE_IMAGE=[^\s]+',
            f'--build-arg DYNAMO_BASE_IMAGE={new_tag}',
            command
        )
        return command
    return wrapper


# Factory function to create task instances for a specific framework
def create_task_graph(framework: str, sha: str, repo_path: Path) -> Dict[str, 'BaseTask']:
    """
    Create task instances for a specific framework.

    Args:
        framework: Framework name (vllm, sglang, trtllm)
        sha: Git commit SHA (short form, 9 chars)
        repo_path: Path to the Dynamo repository

    Returns:
        Dictionary mapping task IDs to task instances
        
    Image Dependency Chain:
        The docker_command_filter uses function composition to transform build commands:
        
        1. Base image (produces: dynamo-base:v0.1.0.dev.f1552864b-vllm):
           rename_output_tag(base_image_tag, 
               filter_out_latest_tag(select_command(0)))
           
        2. Runtime image (consumes: base, produces: dynamo:v0.1.0.dev.f1552864b-vllm-runtime):
           rename_output_tag(runtime_image_tag,
               rename_input_tag(base_image_tag,
                   filter_out_latest_tag(select_command(1))))
           
        3. Dev image (consumes: base, produces: dynamo:v0.1.0.dev.f1552864b-vllm-dev):
           rename_output_tag(dev_image_tag,
               rename_input_tag(base_image_tag,
                   filter_out_latest_tag(select_command(1))))
           
        4. Local-dev image (consumes: dev, produces: dynamo:v0.1.0.dev.f1552864b-vllm-local-dev):
           rename_output_tag(local_dev_image_tag,
               rename_input_tag(dev_image_tag,
                   filter_out_latest_tag(select_command(1))))
        
        Each function in the chain wraps the next:
        - select_command(index): Selects docker command from build.sh output
        - filter_out_latest_tag(): Removes --tag arguments containing 'latest'
        - rename_input_tag(tag): Sets --build-arg DYNAMO_BASE_IMAGE=<tag>
        - rename_output_tag(tag): Sets --tag <tag>
        
        Result: Input and output images match parent/child dependencies perfectly.
    """
    framework_upper = framework.upper()
    framework_lower = framework.lower()

    # Task ID format: FRAMEWORK-target
    # Example: VLLM-base, VLLM-dev-compilation

    tasks: Dict[str, BaseTask] = {}

    # Level 0: Base image build
    base_image_tag = f"dynamo-base:v0.1.0.dev.{sha}-{framework_lower}"
    tasks[f"{framework_upper}-base"] = BuildTask(
        task_id=f"{framework_upper}-base",
        description=f"Build {framework_upper} base image",
        command=f"{repo_path}/container/build.sh --framework {framework_lower} --target dynamo_base --dry-run",
        output_image=base_image_tag,
        docker_command_filter=rename_output_tag(base_image_tag, filter_out_latest_tag(select_command(0))),
        timeout=600.0,
    )

    # Level 1: Runtime image build
    runtime_image_tag = f"dynamo:v0.1.0.dev.{sha}-{framework_lower}-runtime"
    tasks[f"{framework_upper}-runtime"] = BuildTask(
        task_id=f"{framework_upper}-runtime",
        description=f"Build {framework_upper} runtime image",
        command=f"{repo_path}/container/build.sh --framework {framework_lower} --target runtime --dry-run",
        input_image=base_image_tag,
        output_image=runtime_image_tag,
        docker_command_filter=rename_output_tag(runtime_image_tag, 
            rename_input_tag(base_image_tag, 
                filter_out_latest_tag(select_command(1)))),
        parents=[f"{framework_upper}-base"],
        timeout=600.0,
    )

    # Level 2: Runtime sanity check
    tasks[f"{framework_upper}-runtime-sanity"] = CommandTask(
        task_id=f"{framework_upper}-runtime-sanity",
        description=f"Run sanity_check.py in {framework_upper} runtime container",
        command=f"{repo_path}/container/run.sh --image {runtime_image_tag} -- python3 /workspace/deploy/sanity_check.py",
        input_image=runtime_image_tag,
        parents=[f"{framework_upper}-runtime"],
        timeout=120.0,
        ignore_exit_code=True,  # Runtime may fail some checks, we only care about Dynamo paths
    )

    # Level 2: Dev image build
    dev_image_tag = f"dynamo:v0.1.0.dev.{sha}-{framework_lower}-dev"
    tasks[f"{framework_upper}-dev"] = BuildTask(
        task_id=f"{framework_upper}-dev",
        description=f"Build {framework_upper} dev image",
        command=f"{repo_path}/container/build.sh --framework {framework_lower} --target dev --dry-run",
        input_image=base_image_tag,
        output_image=dev_image_tag,
        docker_command_filter=rename_output_tag(dev_image_tag,
            rename_input_tag(base_image_tag,
                filter_out_latest_tag(select_command(1)))),
        parents=[f"{framework_upper}-base"],  # Parent is base, not runtime
        timeout=600.0,
    )

    # Level 3: Dev compilation
    # Use fast build flags from compile.sh:
    # - CARGO_INCREMENTAL=1: Enable incremental compilation
    # - CARGO_PROFILE_DEV_OPT_LEVEL=0: No optimizations (faster compile)
    # - CARGO_BUILD_JOBS=$(nproc): Parallel compilation
    # - CARGO_PROFILE_DEV_CODEGEN_UNITS=256: More parallel code generation
    cargo_cmd = "CARGO_INCREMENTAL=1 CARGO_PROFILE_DEV_OPT_LEVEL=0 CARGO_BUILD_JOBS=$(nproc) CARGO_PROFILE_DEV_CODEGEN_UNITS=256 cargo build --locked --profile dev --features dynamo-llm/block-manager && cd /workspace/lib/bindings/python && CARGO_INCREMENTAL=1 CARGO_PROFILE_DEV_OPT_LEVEL=0 CARGO_BUILD_JOBS=$(nproc) CARGO_PROFILE_DEV_CODEGEN_UNITS=256 maturin develop --uv --features block-manager && uv pip install -e ."
    home_dir = str(Path.home())
    tasks[f"{framework_upper}-dev-compilation"] = CommandTask(
        task_id=f"{framework_upper}-dev-compilation",
        description=f"Run workspace compilation in {framework_upper} dev container",
        command=f"{repo_path}/container/run.sh --image {dev_image_tag} --mount-workspace -v {home_dir}/.cargo:/root/.cargo -- bash -c '{cargo_cmd}'",
        input_image=dev_image_tag,
        parents=[f"{framework_upper}-dev"],
        timeout=1800.0,
    )

    # Level 4: Dev chown (always runs, even if compilation fails)
    tasks[f"{framework_upper}-dev-chown"] = CommandTask(
        task_id=f"{framework_upper}-dev-chown",
        description=f"Fix file ownership after {framework_upper} dev compilation",
        command=f"sudo chown -R $(id -u):$(id -g) {repo_path}/target {repo_path}/lib/bindings/python {home_dir}/.cargo",
        parents=[f"{framework_upper}-dev-compilation"],
        run_even_if_deps_fail=True,
        timeout=60.0,
    )

    # Level 5: Dev sanity check
    tasks[f"{framework_upper}-dev-sanity"] = CommandTask(
        task_id=f"{framework_upper}-dev-sanity",
        description=f"Run sanity_check.py in {framework_upper} dev container",
        command=f"{repo_path}/container/run.sh --image {dev_image_tag} --mount-workspace -v {home_dir}/.cargo:/root/.cargo -- python3 /workspace/deploy/sanity_check.py",
        input_image=dev_image_tag,
        parents=[f"{framework_upper}-dev-chown"],
        timeout=120.0,
    )

    # Level 3: Local-dev image build
    local_dev_image_tag = f"dynamo:v0.1.0.dev.{sha}-{framework_lower}-local-dev"
    tasks[f"{framework_upper}-local-dev"] = BuildTask(
        task_id=f"{framework_upper}-local-dev",
        description=f"Build {framework_upper} local-dev image",
        command=f"{repo_path}/container/build.sh --framework {framework_lower} --target local-dev --dry-run",
        input_image=dev_image_tag,
        output_image=local_dev_image_tag,
        docker_command_filter=rename_output_tag(local_dev_image_tag,
            rename_input_tag(dev_image_tag,
                filter_out_latest_tag(select_command(1)))),
        parents=[f"{framework_upper}-dev"],
        timeout=600.0,
    )

    # Level 5: Local-dev compilation
    tasks[f"{framework_upper}-local-dev-compilation"] = CommandTask(
        task_id=f"{framework_upper}-local-dev-compilation",
        description=f"Run workspace compilation in {framework_upper} local-dev container",
        command=f"{repo_path}/container/run.sh --image {local_dev_image_tag} --mount-workspace -v {home_dir}/.cargo:/home/ubuntu/.cargo -- bash -c '{cargo_cmd}'",
        input_image=local_dev_image_tag,
        parents=[f"{framework_upper}-local-dev", f"{framework_upper}-dev-chown"],
        timeout=1800.0,
    )

    # Level 6: Local-dev sanity check
    tasks[f"{framework_upper}-local-dev-sanity"] = CommandTask(
        task_id=f"{framework_upper}-local-dev-sanity",
        description=f"Run sanity_check.py in {framework_upper} local-dev container",
        command=f"{repo_path}/container/run.sh --image {local_dev_image_tag} --mount-workspace -v {home_dir}/.cargo:/home/ubuntu/.cargo -- python3 /workspace/deploy/sanity_check.py",
        input_image=local_dev_image_tag,
        parents=[f"{framework_upper}-local-dev-compilation"],
        timeout=120.0,
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
    timeout: float = 600.0  # Default 10 minutes

    # Docker image tracking
    input_image: Optional[str] = None   # Base/input Docker image (for builds or container runs)
    output_image: Optional[str] = None  # Output Docker image (for builds)

    # Dependency management
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    run_even_if_deps_fail: bool = False  # Run even if dependencies fail

    # Execution state
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    log_file: Optional[Path] = None

    # Results
    exit_code: Optional[int] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        """Initialize task-specific attributes"""
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{self.task_id}")

    @abstractmethod
    def prepare(self, repo_path: Path, sha: str) -> None:
        """
        Prepare task for execution.

        This method should:
        - Generate the command to execute
        - Set up any required paths or environment
        - Validate prerequisites

        Args:
            repo_path: Path to the repository
            sha: Git commit SHA
        """
        pass

    def _run_command(self, command: str, repo_path: Path) -> int:
        """
        Common method to run a command and log output.
        
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
                log_fh.write(f"\n{'='*80}\n")
                log_fh.write(f"Executing: {command}\n")
                log_fh.write(f"{'='*80}\n\n")
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
                
                # Stream output to log file
                if process.stdout is None:
                    raise ValueError("process.stdout is None")
                for line in process.stdout:
                    log_fh.write(line)
                    log_fh.flush()
                
                # Wait for process to complete
                process.wait()
                self.exit_code = process.returncode
                
                return self.exit_code
                
        except Exception as e:
            self.error_message = str(e)
            self.logger.error(f"Command execution failed: {e}")
            self.exit_code = -1
            return self.exit_code

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
        if self.status != TaskStatus.PENDING:
            return False

        for dep_id in self.parents:
            if dep_id not in all_tasks:
                self.logger.warning(f"Dependency {dep_id} not found")
                return False

            dep_task = all_tasks[dep_id]

            # If dependency is still pending or running, we can't run yet
            if dep_task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                return False

            # If dependency failed and we don't run on failure, we can't run
            if dep_task.status == TaskStatus.FAILED and not self.run_even_if_deps_fail:
                return False

        return True

    def should_skip(self, all_tasks: Dict[str, 'BaseTask']) -> bool:
        """
        Check if this task should be skipped due to dependency failures.

        Args:
            all_tasks: Dictionary of all tasks in the pipeline

        Returns:
            True if task should be skipped
        """
        if self.run_even_if_deps_fail:
            return False

        for dep_id in self.parents:
            if dep_id in all_tasks:
                dep_task = all_tasks[dep_id]
                if dep_task.status == TaskStatus.FAILED:
                    return True

        return False

    def duration(self) -> Optional[float]:
        """Get task duration in seconds"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    def mark_skipped(self, reason: str = ""):
        """Mark task as skipped"""
        self.status = TaskStatus.SKIPPED
        if reason:
            self.error_message = reason
        self.logger.info(f"Skipped: {reason}")

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

    def get_docker_command(self, repo_path: Path) -> Optional[str]:
        """
        Get the underlying docker command (optional, for tasks that use docker).

        Args:
            repo_path: Path to the repository

        Returns:
            The docker command string, or None if not applicable
        """
        return None


# ==============================================================================
# BUILD TASK
# ==============================================================================

@dataclass
class BuildTask(BaseTask):
    """
    Task for building Docker images.

    Additional attributes:
        docker_command_filter: Optional function to filter/select and transform docker commands
                              Takes List[str] of commands, returns Optional[str] (selected and transformed command)
                              Can chain: selection -> filter_out_latest_tag -> rename_tag
        original_build_command: Full output from build.sh --dry-run (all docker commands)
        actual_build_command: The extracted and transformed docker command to execute
    """

    docker_command_filter: Optional[Callable[[List[str]], Optional[str]]] = None
    original_build_command: Optional[str] = field(default=None, init=False, repr=False)
    actual_build_command: Optional[str] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()

    def prepare(self, repo_path: Path, sha: str) -> None:
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

        Runs docker build command and captures output to log file.
        """
        if dry_run:
            self.logger.info("Dry-run mode: skipping actual execution")
            return True
        
        # Get the actual docker build command (without --dry-run)
        # Remove --dry-run from the command
        actual_command = self.command.replace(' --dry-run', '')
        
        exit_code = self._run_command(actual_command, repo_path)
        return exit_code == 0

    def image_exists(self) -> bool:
        """Check if Docker image already exists"""
        if not self.output_image:
            return False

        try:
            # Use docker inspect to check if image exists
            result = subprocess.run(
                ["docker", "inspect", self.output_image],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return result.returncode == 0
        except Exception as e:
            self.logger.warning(f"Error checking image existence: {e}")
            return False

    def get_image_size(self) -> Optional[str]:
        """Get the size of the Docker image in human-readable format"""
        if not self.output_image:
            return None

        try:
            # Use docker images to get size
            result = subprocess.run(
                ["docker", "images", self.output_image, "--format", "{{.Size}}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            return None
        except Exception as e:
            self.logger.warning(f"Error getting image size: {e}")
            return None

    def get_command(self, repo_path: Path) -> str:
        """Get the build.sh wrapper command (not the underlying docker command)"""
        # Return the build.sh command stored in self.command
        # e.g., "/path/to/build.sh --framework vllm --target runtime --dry-run"
        return self.command

    def get_docker_command(self, repo_path: Path) -> Optional[str]:
        """Get the underlying docker build command by running build.sh --dry-run

        Uses the explicit command stored in self.command (e.g., "build.sh --framework vllm --target dev --dry-run")

        This populates:
        - original_build_command: All docker commands from build.sh --dry-run
        - Returns: The selected and transformed docker command

        Returns:
            The docker command after applying docker_command_filter (which can chain selection + transformations)
        """
        if not self.command:
            return None

        try:
            result = subprocess.run(
                self.command,
                shell=True,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30  # Increased timeout for slower systems
            )
            # build.sh --dry-run prints docker build commands
            output = result.stdout.strip()
            docker_commands = []
            for line in output.split('\n'):
                line = line.strip()
                # Look for lines with "docker build" (may be prefixed with "echo")
                if 'docker build' in line:
                    # Remove "echo" prefix if present
                    if line.startswith('echo '):
                        line = line[5:]
                    docker_commands.append(line)

            # Store original commands
            self.original_build_command = '\n'.join(docker_commands) if docker_commands else None

            # Apply filter function if provided (handles selection + transformation)
            if self.docker_command_filter:
                selected_command = self.docker_command_filter(docker_commands)
            else:
                # Default: return all commands joined with newlines
                selected_command = '\n'.join(docker_commands) if docker_commands else None

            return selected_command

        except subprocess.TimeoutExpired:
            # Timeout - build.sh --dry-run took too long
            return None
        except Exception:
            # Silently fail - don't break tree visualization
            return None


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

    def prepare(self, repo_path: Path, sha: str) -> None:
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
                    TaskStatus.PENDING: "⏳",
                    TaskStatus.RUNNING: "🔄",
                    TaskStatus.SUCCESS: "✅",
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
        "--repo-sha",
        type=str,
        help="Git commit SHA to build (default: current HEAD)",
    )
    parser.add_argument(
        "--pull-latest",
        action="store_true",
        help="Pull latest code from main branch before building",
    )

    # Framework and target selection
    parser.add_argument(
        "-f", "--framework",
        action="append",
        help="Framework(s) to build (vllm, sglang, trtllm). Can specify multiple times. Default: all",
    )
    parser.add_argument(
        "--target",
        default="base,runtime,dev,local-dev",
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
        "--force-run",
        action="store_true",
        help="Force run even if composite SHA hasn't changed (bypasses lock check)",
    )

    # Build options
    parser.add_argument(
        "--skip-build-if-image-exists",
        "--skip",
        action="store_true",
        help="Skip building if Docker image already exists",
    )
    parser.add_argument(
        "--sanity-check-only",
        action="store_true",
        help="Only run sanity checks, skip builds",
    )

    # Output options
    parser.add_argument(
        "--html-path",
        type=str,
        default="/nvidia/dynamo_ci/logs",
        help="Base URL path for HTML reports in emails (default: /nvidia/dynamo_ci/logs)",
    )
    parser.add_argument(
        "--hostname",
        type=str,
        default="keivenc-linux",
        help="Hostname for absolute URLs in email reports (default: keivenc-linux)",
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
        "--tree",
        action="store_true",
        help="Show dependency tree visualization",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML report after execution",
    )

    return parser.parse_args()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def execute_task_sequential(
    all_tasks: Dict[str, 'BaseTask'],
    executed_tasks: Set[str],
    failed_tasks: Set[str],
    task_id: str,
    repo_path: Path,
    sha: str,
    log_dir: Optional[Path] = None,
    log_date: Optional[str] = None,
    dry_run: bool = False,
    skip_if_image_exists: bool = False,
) -> bool:
    """
    Execute a single task and its dependencies recursively (sequential mode).
    
    Args:
        all_tasks: Dictionary of all tasks
        executed_tasks: Set of already executed task IDs (modified in place)
        failed_tasks: Set of failed task IDs (modified in place)
        task_id: ID of task to execute
        repo_path: Path to repository
        sha: Git commit SHA
        log_dir: Directory for log files (None in dry-run)
        log_date: Date string for log files (None in dry-run)
        dry_run: If True, only print commands without executing
        skip_if_image_exists: If True, skip build tasks if Docker image already exists
        
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
            repo_path, sha, log_dir, log_date, dry_run, skip_if_image_exists
        ):
            # Parent failed
            if not task.run_even_if_deps_fail:
                if dry_run:
                    logger.info(f"Would skip {task_id} due to failed dependency {parent_id}")
                else:
                    logger.info(f"Skipping {task_id} due to failed dependency {parent_id}")
                task.status = TaskStatus.SKIPPED
                executed_tasks.add(task_id)
                failed_tasks.add(task_id)
                return False

    executed_tasks.add(task_id)

    # DRY-RUN: Just print what would be executed
    if dry_run:
        logger.info(f"[{task_id}] {task.description}")
        
        # For BuildTask, show both build.sh command AND docker build command
        if isinstance(task, BuildTask):
            logger.info(f"  1. {task.command}")
            docker_cmd = task.get_docker_command(repo_path)
            if docker_cmd:
                docker_lines = docker_cmd.split('\n')
                for i, line in enumerate(docker_lines):
                    if i == 0:
                        logger.info(f"  2. {line}")
                    else:
                        logger.info(f"     {line}")
        else:
            logger.info(f"→ {task.get_command(repo_path)}")
        logger.info("")
        
        # Mark as success for dry-run traversal
        task.status = TaskStatus.SUCCESS
        
        # Process children
        for child_id in task.children:
            if child_id not in executed_tasks:
                execute_task_sequential(
                    all_tasks, executed_tasks, failed_tasks, child_id,
                    repo_path, sha, log_dir, log_date, dry_run, skip_if_image_exists
                )
        
        return True

    # ACTUAL EXECUTION
    # Setup log file: YYYY-MM-DD.{sha}.{task-name}.log
    # Convert task_id (e.g., "VLLM-base") to log name (e.g., "vllm-dynamo-base")
    if log_dir is None or log_date is None:
        raise ValueError("log_dir and log_date must be set for actual execution")
    log_task_name = task_id.lower().replace("-base", "-dynamo-base")
    log_file = log_dir / f"{log_date}.{sha}.{log_task_name}.log"
    task.log_file = log_file

    # Check if we should skip this build task if image already exists
    if skip_if_image_exists and isinstance(task, BuildTask):
        if task.image_exists():
            logger.info(f"⊘ Skipping {task_id}: Docker image already exists ({task.output_image})")
            task.status = TaskStatus.SKIPPED
            executed_tasks.add(task_id)
            
            # Process children
            for child_id in task.children:
                if child_id not in executed_tasks:
                    execute_task_sequential(
                        all_tasks, executed_tasks, failed_tasks, child_id,
                        repo_path, sha, log_dir, log_date, dry_run, skip_if_image_exists
                    )
            
            return True

    # Execute this task
    logger.info(f"Executing: {task_id} ({task.description})")
    logger.info(f"  Command: {task.get_command(repo_path)}")
    logger.info(f"  Log: {log_file}")

    task.status = TaskStatus.RUNNING
    task.start_time = time.time()

    try:
        # Open log file for writing header
        with open(log_file, 'w') as log_fh:
            log_fh.write(f"Task: {task_id}\n")
            log_fh.write(f"Description: {task.description}\n")
            log_fh.write(f"Command: {task.get_command(repo_path)}\n")
            log_fh.write(f"Started: {datetime.now().isoformat()}\n")
            log_fh.write("=" * 80 + "\n\n")
            log_fh.flush()

        # Execute the task (will append to log file)
        success = task.execute(repo_path, dry_run=False)

        task.end_time = time.time()
        duration = task.end_time - task.start_time

        # Append execution summary to log file
        with open(log_file, 'a') as log_fh:
            log_fh.write(f"\n")
            log_fh.write(f"{'='*80}\n")
            log_fh.write(f"Task: {task_id}\n")
            log_fh.write(f"Duration: {duration:.2f}s\n")
            log_fh.write(f"Exit code: {task.exit_code if hasattr(task, 'exit_code') else (0 if success else 1)}\n")
            log_fh.write(f"Status: {'SUCCESS' if success else 'FAILED'}\n")
            log_fh.write(f"{'='*80}\n")

        if success:
            task.status = TaskStatus.SUCCESS
            logger.info(f"✓ Completed: {task_id} ({duration:.2f}s)")
            # Create .PASS marker file
            pass_marker = log_file.with_suffix('.PASS')
            pass_marker.touch()
        else:
            task.status = TaskStatus.FAILED
            logger.error(f"✗ Failed: {task_id}")
            failed_tasks.add(task_id)
            # Create .FAIL marker file
            fail_marker = log_file.with_suffix('.FAIL')
            fail_marker.touch()
    except Exception as e:
        task.end_time = time.time()
        task.status = TaskStatus.FAILED
        task.error_message = str(e)
        logger.error(f"✗ Failed: {task_id} - {e}")
        failed_tasks.add(task_id)
        # Create .FAIL marker file
        fail_marker = log_file.with_suffix('.FAIL')
        fail_marker.touch()
    
    # After successfully executing this task, execute all children
    if task.status == TaskStatus.SUCCESS:
        for child_id in task.children:
            if child_id not in executed_tasks:
                execute_task_sequential(
                    all_tasks, executed_tasks, failed_tasks, child_id,
                    repo_path, sha, log_dir, log_date, dry_run, skip_if_image_exists
                )
    
    return task.status == TaskStatus.SUCCESS


def execute_task_parallel(
    all_tasks: Dict[str, 'BaseTask'],
    root_tasks: List[str],
    repo_path: Path,
    sha: str,
    log_dir: Optional[Path] = None,
    log_date: Optional[str] = None,
    dry_run: bool = False,
    skip_if_image_exists: bool = False,
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
        sha: Git commit SHA
        log_dir: Directory for log files (None in dry-run)
        log_date: Date string for log files (None in dry-run)
        dry_run: If True, only print commands without executing
        skip_if_image_exists: If True, skip build tasks if Docker image already exists
        max_workers: Maximum number of parallel threads
        
    Returns:
        Tuple of (executed_tasks, failed_tasks) sets
    """
    logger = logging.getLogger("executor")
    
    executed_tasks = set()
    failed_tasks = set()
    lock = threading.Lock()
    
    def can_execute(task_id: str) -> bool:
        """Check if all dependencies are satisfied"""
        task = all_tasks[task_id]
        for parent_id in task.parents:
            if parent_id not in executed_tasks:
                return False
            if parent_id in failed_tasks and not task.run_even_if_deps_fail:
                return False
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
                    task.status = TaskStatus.SKIPPED
                    executed_tasks.add(task_id)
                    failed_tasks.add(task_id)
                    return False
        
        # DRY-RUN: Just print
        if dry_run:
            with lock:
                logger.info(f"[{task_id}] {task.description}")
                if isinstance(task, BuildTask):
                    logger.info(f"  1. {task.command}")
                    docker_cmd = task.get_docker_command(repo_path)
                    if docker_cmd:
                        for i, line in enumerate(docker_cmd.split('\n')):
                            logger.info(f"  2. {line}" if i == 0 else f"     {line}")
                else:
                    logger.info(f"→ {task.get_command(repo_path)}")
                logger.info("")
            
            task.status = TaskStatus.SUCCESS
            with lock:
                executed_tasks.add(task_id)
            return True
        
        # ACTUAL EXECUTION
        if log_dir is None or log_date is None:
            raise ValueError("log_dir and log_date must be set for actual execution")
        log_task_name = task_id.lower().replace("-base", "-dynamo-base")
        log_file = log_dir / f"{log_date}.{sha}.{log_task_name}.log"
        task.log_file = log_file
        
        # Check if we should skip this build task if image already exists
        if skip_if_image_exists and isinstance(task, BuildTask):
            if task.image_exists():
                with lock:
                    logger.info(f"⊘ Skipping {task_id}: Docker image already exists ({task.output_image})")
                task.status = TaskStatus.SKIPPED
                with lock:
                    executed_tasks.add(task_id)
                return True
        
        with lock:
            logger.info(f"Executing: {task_id} ({task.description})")
            logger.info(f"  Command: {task.get_command(repo_path)}")
            logger.info(f"  Log: {log_file}")
        
        task.status = TaskStatus.RUNNING
        task.start_time = time.time()
        
        try:
            with open(log_file, 'w') as log_fh:
                log_fh.write(f"Task: {task_id}\n")
                log_fh.write(f"Description: {task.description}\n")
                log_fh.write(f"Command: {task.get_command(repo_path)}\n")
                log_fh.write(f"Started: {datetime.now().isoformat()}\n")
                log_fh.write("=" * 80 + "\n\n")
                log_fh.flush()
            
            success = task.execute(repo_path, dry_run=False)
            task.end_time = time.time()
            duration = task.end_time - task.start_time
            
            # Append execution summary to log file
            with open(log_file, 'a') as log_fh:
                log_fh.write(f"\n")
                log_fh.write(f"{'='*80}\n")
                log_fh.write(f"Task: {task_id}\n")
                log_fh.write(f"Duration: {duration:.2f}s\n")
                log_fh.write(f"Exit code: {task.exit_code if hasattr(task, 'exit_code') else (0 if success else 1)}\n")
                log_fh.write(f"Status: {'SUCCESS' if success else 'FAILED'}\n")
                log_fh.write(f"{'='*80}\n")
            
            with lock:
                if success:
                    task.status = TaskStatus.SUCCESS
                    
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
                    
                    pass_marker = log_file.with_suffix('.PASS')
                    pass_marker.touch()
                else:
                    task.status = TaskStatus.FAILED
                    logger.error(f"✗ Failed: {task_id}")
                    failed_tasks.add(task_id)
                    fail_marker = log_file.with_suffix('.FAIL')
                    fail_marker.touch()
        except Exception as e:
            task.end_time = time.time()
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            with lock:
                logger.error(f"✗ Failed: {task_id} - {e}")
                failed_tasks.add(task_id)
                fail_marker = log_file.with_suffix('.FAIL')
                fail_marker.touch()
        
        with lock:
            executed_tasks.add(task_id)
        
        return task.status == TaskStatus.SUCCESS
    
    # Execute tasks in waves based on dependencies
    pending_tasks = set(all_tasks.keys())
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while pending_tasks:
            # Find tasks ready to execute
            ready_tasks = [t for t in pending_tasks if can_execute(t)]
            
            if not ready_tasks:
                # No ready tasks but pending tasks remain - check for deadlock
                if pending_tasks:
                    logger.error(f"Deadlock detected! Remaining tasks: {pending_tasks}")
                    for task_id in pending_tasks:
                        with lock:
                            failed_tasks.add(task_id)
                            executed_tasks.add(task_id)
                break
            
            # Submit ready tasks
            futures = {executor.submit(execute_single_task, t): t for t in ready_tasks}
            pending_tasks -= set(ready_tasks)
            
            # Wait for at least one to complete
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Exception in task {task_id}: {e}")
                    with lock:
                        failed_tasks.add(task_id)
    
    return executed_tasks, failed_tasks


def generate_html_report(
    all_tasks: Dict[str, 'BaseTask'],
    repo_path: Path,
    sha: str,
    log_dir: Path,
    date_str: str,
    use_absolute_urls: bool = False,
    hostname: str = "keivenc-linux",
    html_path: str = "/nvidia/dynamo_ci/logs",
) -> str:
    """
    Generate HTML report using Jinja2 template.
    
    Args:
        all_tasks: Dictionary of all tasks that were executed
        repo_path: Path to the repository
        sha: Git commit SHA (9 chars)
        log_dir: Directory containing log files
        date_str: Date string (YYYY-MM-DD)
        use_absolute_urls: If True, use http://hostname/... URLs for email
        hostname: Hostname for absolute URLs
        html_path: Base path for absolute URLs
        
    Returns:
        HTML string
    """
    # Check if jinja2 is available (imported at top of file)
    if Environment is None:
        logging.getLogger("html").error("Jinja2 not installed. Install with: pip install jinja2")
        return "<html><body><h1>Error: Jinja2 not installed</h1></body></html>"
    
    # Check if git is available (imported at top of file)
    if git is None:
        logging.getLogger("html").warning("GitPython not installed, skipping git information")
    
    # Count task statistics
    total_tasks = len(all_tasks)
    succeeded = sum(1 for t in all_tasks.values() if t.status == TaskStatus.SUCCESS)
    failed = sum(1 for t in all_tasks.values() if t.status == TaskStatus.FAILED)
    skipped = sum(1 for t in all_tasks.values() if t.status == TaskStatus.SKIPPED)
    
    # Determine overall status
    overall_status = "✅ ALL TESTS PASSED" if failed == 0 else "❌ TESTS FAILED"
    header_color = "#28a745" if failed == 0 else "#dc3545"
    
    # Get git information
    commit_info: Dict[str, Any] = {}
    if git:
        try:
            repo = git.Repo(repo_path)
            commit = repo.commit(sha)
            commit_info = {
                'sha_short': sha[:7],
                'sha_full': sha,
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
            except Exception:
                pass
                
        except Exception as e:
            logging.getLogger("html").warning(f"Could not get git info: {e}")
            commit_info = {'sha_short': sha[:7], 'sha_full': sha}
    else:
        commit_info = {'sha_short': sha[:7], 'sha_full': sha}
    
    # Helper function to generate log URL
    def get_log_url(task: 'BaseTask') -> Optional[str]:
        """Generate URL/path for task log file"""
        if not task.log_file:
            return None
        try:
            if use_absolute_urls:
                # Email: absolute URL with http://
                # Get path relative to logs root (e.g., "2025-10-29/file.log")
                rel_path = task.log_file.relative_to(log_dir.parent)
                # Construct full URL: http://hostname/path/to/logs/date/file.log
                return f"http://{hostname}{html_path}/{rel_path}"
            else:
                # File: relative path (just filename)
                return task.log_file.name
        except Exception as e:
            logging.getLogger("html").warning(f"Error generating log URL: {e}")
            return None
    
    # Organize tasks by framework
    frameworks_data: Dict[str, Dict[str, Any]] = {}
    
    for task_id, task in all_tasks.items():
        # Parse task_id to extract framework and target
        # Format: VLLM-base, VLLM-runtime, VLLM-dev-compilation, etc.
        parts = task_id.split('-', 1)
        if len(parts) < 2:
            continue
            
        framework = parts[0]
        rest = parts[1]
        
        if framework not in frameworks_data:
            frameworks_data[framework] = {}
        
        # Determine target and task type
        if isinstance(task, BuildTask):
            # BuildTask: VLLM-base, VLLM-runtime, VLLM-dev, VLLM-local-dev
            target = rest  # base, runtime, dev, local-dev
            if target not in frameworks_data[framework]:
                frameworks_data[framework][target] = {'build': None, 'compilation': None, 'sanity': None, 'image_size': None}
            frameworks_data[framework][target]['build'] = {
                'status': task.status.name,
                'build_time': f"{task.end_time - task.start_time:.1f}s" if task.start_time and task.end_time else None,
                'log_file': get_log_url(task) if task.status != TaskStatus.SKIPPED else None,
            }
            # Get image size for build tasks
            if isinstance(task, BuildTask):
                frameworks_data[framework][target]['image_size'] = task.get_image_size()
        elif isinstance(task, CommandTask):
            # CommandTask: could be compilation, chown, or sanity
            if 'compilation' in rest:
                target = rest.replace('-compilation', '')  # dev, local-dev
                if target not in frameworks_data[framework]:
                    frameworks_data[framework][target] = {'build': None, 'compilation': None, 'sanity': None, 'image_size': None}
                frameworks_data[framework][target]['compilation'] = {
                    'status': task.status.name,
                    'time': f"{task.end_time - task.start_time:.1f}s" if task.start_time and task.end_time else None,
                    'log_file': get_log_url(task),
                }
            elif 'sanity' in rest:
                target = rest.replace('-sanity', '')  # runtime, dev, local-dev
                if target not in frameworks_data[framework]:
                    frameworks_data[framework][target] = {'build': None, 'compilation': None, 'sanity': None, 'image_size': None}
                frameworks_data[framework][target]['sanity'] = {
                    'status': task.status.name,
                    'sanity_time': f"{task.end_time - task.start_time:.1f}s" if task.start_time and task.end_time else None,
                    'log_file': get_log_url(task),
                }
    
    # Load Jinja2 template
    template_dir = Path(__file__).parent
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(['html', 'xml'])
    )
    template = env.get_template('dynamo_docker_builder.html.j2')
    
    # Render HTML
    html = template.render(
        sha_display=commit_info.get('sha_short', sha[:7]),
        sha_link=commit_info.get('sha_full', sha),
        overall_status=overall_status,
        header_color=header_color,
        total_tasks=total_tasks,
        succeeded=succeeded,
        failed=failed,
        skipped=skipped,
        build_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        commit=commit_info,
        frameworks=frameworks_data,
    )
    
    return html


def send_email_notification(
    email: str,
    html_content: str,
    sha: str,
    failed_tasks: List[str],
) -> bool:
    """
    Send HTML email notification via SMTP using curl.
    
    Args:
        email: Recipient email address
        html_content: HTML content to send
        sha: Git commit SHA (short, 7 chars)
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
            subject = f"{status_prefix}: DynamoDockerBuilder - {sha} ({failure_summary})"
        else:
            subject = f"{status_prefix}: DynamoDockerBuilder - {sha}"
        
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


def print_dependency_tree(frameworks: List[str], sha: str, repo_path: Path, verbose: bool = False) -> None:
    """
    Print dependency tree visualization for given frameworks.
    
    Args:
        frameworks: List of framework names (normalized, lowercase)
        sha: Git commit SHA (9 chars)
        repo_path: Path to the repository
        verbose: If True, show commands in tree
    """
    print("\n" + "=" * 80)
    print("DEPENDENCY TREE VISUALIZATION")
    print("=" * 80)
    print(f"\nRepository: {repo_path}")
    print(f"Commit SHA: {sha}")
    print(f"Frameworks: {', '.join(f.upper() for f in frameworks)}")
    print()

    for framework in frameworks:
        # Create task graph for this framework
        tasks = create_task_graph(framework, sha, repo_path)

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
                docker_cmd = task.get_docker_command(repo_path)

                # Only show "1." if there's also a "2." (docker command exists)
                if docker_cmd:
                    # Two-level command structure
                    print(f"{prefix}{continuation}1. {command}")

                    # 2. Print the underlying docker command
                    # Docker command may be multiline (multiple docker build commands)
                    docker_lines = docker_cmd.split('\n')
                    for i, line in enumerate(docker_lines):
                        if i == 0:
                            print(f"{prefix}{continuation}2. {line}")
                        else:
                            # Additional lines with same continuation prefix
                            print(f"{prefix}{continuation}   {line}")
                else:
                    # Single command - no numbering needed
                    print(f"{prefix}{continuation}{command}")

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


def main() -> int:
    """Main entry point"""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger("main")

    # Get repository info
    repo_path = args.repo_path.resolve()
    if not repo_path.exists():
        logger.error(f"Repository path does not exist: {repo_path}")
        return 1

    # Pull latest code if requested
    if args.pull_latest:
        logger.info("Pulling latest code from main branch...")
        try:
            # Initialize GitUtils to check repo status
            git_utils_temp = GitUtils(repo_path)
            
            # Check if there are uncommitted changes
            if git_utils_temp.is_dirty() or git_utils_temp.repo.untracked_files:
                logger.warning("Repository has uncommitted changes. Stashing them before pull...")
                git_utils_temp.repo.git.stash('save', '--include-untracked', 'Auto-stash before pull')
            
            # Pull latest from main using GitPython
            origin = git_utils_temp.repo.remotes.origin
            origin.pull('main')
            logger.info("✓ Successfully pulled latest code from main")
        except Exception as e:
            logger.error(f"Failed to pull latest code: {e}")
            return 1

    # Get commit SHA
    try:
        git_utils = GitUtils(repo_path)
        if args.repo_sha:
            sha = args.repo_sha
        else:
            full_sha = git_utils.get_current_commit()
            sha = full_sha[:9]  # Use 9 chars to match build.sh format
    except Exception as e:
        logger.error(f"Failed to get commit SHA: {e}")
        return 1

    # Determine which frameworks to build
    if args.framework:
        frameworks = [normalize_framework(f) for f in args.framework]
    else:
        frameworks = FRAMEWORKS_LOWER

    # Handle --tree flag: show dependency tree
    if args.tree:
        print_dependency_tree(frameworks, sha, repo_path, verbose=args.verbose)

    # Execution mode header
    if args.dry_run:
        logger.info("DRY-RUN MODE: Showing commands that would be executed")
    else:
        logger.info("DynamoDockerBuilder V2 - Starting")

    # Track overall execution time
    execution_start_time = time.time()

    # Setup log directory: logs/YYYY-MM-DD/ (skip in dry-run)
    if not args.dry_run:
        log_date = datetime.now().strftime("%Y-%m-%d")
        log_dir = repo_path / "logs" / log_date
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Log directory: {log_dir}")
        
        # Clean up existing log files and marker files for this SHA and frameworks
        # Pattern: YYYY-MM-DD.{sha}.{framework}-*
        cleaned_count = 0
        for framework in frameworks:
            cleanup_pattern = str(log_dir / f"{log_date}.{sha}.{framework}-*")
            framework_files = glob.glob(cleanup_pattern)
            if framework_files:
                for file_path in framework_files:
                    try:
                        Path(file_path).unlink()
                        cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to remove {file_path}: {e}")
        
        # Also clean up HTML report files for this SHA
        html_report_pattern = str(log_dir / f"{log_date}.{sha}.report.html")
        html_files = glob.glob(html_report_pattern)
        for file_path in html_files:
            try:
                Path(file_path).unlink()
                cleaned_count += 1
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} existing log/marker files for SHA {sha}")

    # Create task graph for each framework
    all_tasks = {}
    for framework in frameworks:
        framework_tasks = create_task_graph(framework, sha, repo_path)
        all_tasks.update(framework_tasks)

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
                task.status = TaskStatus.SKIPPED
                sanity_tasks[task_id] = task
            elif isinstance(task, BuildTask):
                # Mark build tasks as skipped
                task.status = TaskStatus.SKIPPED
                # Add to sanity_tasks so they show in HTML report with image size
                sanity_tasks[task_id] = task
        
        all_tasks = sanity_tasks
        logger.info(f"Filtered to {len([t for t in all_tasks.values() if 'sanity' in t.task_id])} sanity check tasks")

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

    # Execute based on mode
    if args.parallel:
        # Parallel execution
        executed_tasks, failed_tasks = execute_task_parallel(
            all_tasks=all_tasks,
            root_tasks=root_tasks,
            repo_path=repo_path,
            sha=sha,
            log_dir=log_dir if not args.dry_run else None,
            log_date=log_date if not args.dry_run else None,
            dry_run=args.dry_run,
            skip_if_image_exists=args.skip_build_if_image_exists,
            max_workers=4,  # TODO: make this configurable
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
                sha=sha,
                log_dir=log_dir if not args.dry_run else None,
                log_date=log_date if not args.dry_run else None,
                dry_run=args.dry_run,
                skip_if_image_exists=args.skip_build_if_image_exists,
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
        return 0
    else:
        success_count = len([t for t in all_tasks.values() if t.status == TaskStatus.SUCCESS])
        failed_count = len([t for t in all_tasks.values() if t.status == TaskStatus.FAILED])
        skipped_count = len([t for t in all_tasks.values() if t.status == TaskStatus.SKIPPED])
        
        exit_status = 0 if failed_count == 0 else 1

        logger.info(f"\nExecution Summary:")
        logger.info(f"  ✓ Success: {success_count}")
        logger.info(f"  ✗ Failed: {failed_count}")
        logger.info(f"  ⊘ Skipped: {skipped_count}")
        
        # Show detailed task breakdown
        logger.info(f"\n  Task Details:")
        for task_id, task in all_tasks.items():
            if task.status == TaskStatus.SUCCESS and task.start_time and task.end_time:
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
        
        # Generate HTML report if requested
        if args.html or args.email:
            try:
                # Generate report with relative paths for file
                html_content_file = generate_html_report(
                    all_tasks=all_tasks,
                    repo_path=repo_path,
                    sha=sha,
                    log_dir=log_dir,
                    date_str=log_date,
                    use_absolute_urls=False,
                )
                
                if args.html:
                    html_file = log_dir / f"{log_date}.{sha}.report.html"
                    html_file.write_text(html_content_file)
                    logger.info(f"  HTML Report: {html_file}")
                
                # Generate report with absolute URLs for email
                if args.email:
                    html_content_email = generate_html_report(
                        all_tasks=all_tasks,
                        repo_path=repo_path,
                        sha=sha,
                        log_dir=log_dir,
                        date_str=log_date,
                        use_absolute_urls=True,
                        hostname=args.hostname,
                        html_path=args.html_path,
                    )
                    
                    # Collect failed task names
                    failed_task_names = [
                        task_id for task_id, task in all_tasks.items()
                        if task.status == TaskStatus.FAILED
                    ]
                    
                    # Send email
                    send_email_notification(
                        email=args.email,
                        html_content=html_content_email,
                        sha=sha[:7],
                        failed_tasks=failed_task_names,
                    )
                    
            except Exception as e:
                logger.error(f"Failed to generate HTML report: {e}")

        return exit_status


if __name__ == "__main__":
    sys.exit(main())
