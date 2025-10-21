#!/usr/bin/env python3
"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

DynamoDockerBuilder V2 - Simplified Parallel Docker Build System

A cleaner, more elegant implementation focusing on:
- Native parallel execution with asyncio
- Simpler code structure with functional patterns
- Type-safe dataclasses
- Same behavior as V1 but with cleaner code
"""

import argparse
import asyncio
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set

from common import FRAMEWORKS_UPPER, BaseUtils, DockerUtils, DynamoRepositoryUtils, get_terminal_width
import subprocess
import docker


class TaskType(Enum):
    """Task types in the build pipeline"""
    BUILD = "build"
    COMPILATION = "compilation"
    SANITY_CHECK = "sanity_check"


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """Represents a single task in the build pipeline"""
    task_id: str
    task_type: TaskType
    framework: Optional[str]
    target: Optional[str]
    description: str
    command: str

    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)

    # Execution state
    status: TaskStatus = TaskStatus.PENDING
    duration: float = 0.0
    error_message: Optional[str] = None
    log_file: Optional[Path] = None

    # Build-specific (for BUILD tasks)
    image_tag: Optional[str] = None
    base_image: Optional[str] = None
    image_size: Optional[str] = None  # Human-readable image size (e.g., "18.2 GB")

    # Execute-specific (for COMPILATION/SANITY_CHECK tasks)
    timeout: float = 3600.0

    def can_run(self, task_tree: Dict[str, 'Task']) -> bool:
        """Check if all dependencies are completed successfully"""
        if self.status != TaskStatus.PENDING:
            return False

        for dep_id in self.depends_on:
            dep_task = task_tree.get(dep_id)
            if not dep_task or dep_task.status not in (TaskStatus.SUCCESS, TaskStatus.SKIPPED):
                return False

        return True


class BuildPipeline:
    """Manages the complete build pipeline with dependency resolution"""

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_task(self, task: Task) -> None:
        """Add a task to the pipeline"""
        self.tasks[task.task_id] = task

    def add_dependency(self, child_id: str, parent_id: str) -> None:
        """Establish parent-child dependency"""
        if child_id in self.tasks and parent_id in self.tasks:
            self.tasks[child_id].depends_on.append(parent_id)
            self.tasks[parent_id].children.append(child_id)

    def get_ready_tasks(self) -> List[Task]:
        """Get all tasks ready to run (dependencies satisfied)"""
        return [task for task in self.tasks.values() if task.can_run(self.tasks)]

    def get_levels(self) -> List[List[Task]]:
        """
        Group tasks by execution level (for parallel execution).
        Tasks in same level can run in parallel.
        """
        levels: List[List[Task]] = []
        processed: Set[str] = set()  # Track which tasks we've already added to levels

        # Mark skipped tasks as processed
        for task in self.tasks.values():
            if task.status == TaskStatus.SKIPPED:
                processed.add(task.task_id)

        while len(processed) < len(self.tasks):
            # Find tasks ready to run that haven't been processed yet
            ready = [
                task for task in self.tasks.values()
                if task.task_id not in processed and
                task.status == TaskStatus.PENDING and
                all(dep in processed for dep in task.depends_on)
            ]

            if not ready:
                # Check if there are unprocessed pending tasks (circular dependency)
                unprocessed = [t for t in self.tasks.values()
                              if t.task_id not in processed and t.status == TaskStatus.PENDING]
                if unprocessed:
                    self.logger.error(f"Circular dependency detected: {[t.task_id for t in unprocessed]}")
                break

            levels.append(ready)
            # Mark these tasks as processed so they don't appear in future levels
            processed.update(task.task_id for task in ready)

        return levels

    def visualize_tree(self) -> str:
        """Generate tree visualization with box-drawing characters"""
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("DEPENDENCY TREE")
        lines.append("=" * 80)
        lines.append("\nRoot tasks (can run in parallel):")
        lines.append("")

        # Find root tasks (no dependencies)
        roots = [t for t in self.tasks.values() if not t.depends_on]

        def print_task(task: Task, prefix: str = "", is_last_sibling: bool = True, is_root: bool = False):
            # Determine tree connector
            if is_root:
                connector = ""
            elif is_last_sibling:
                connector = "└─ "
            else:
                connector = "├─ "

            # Build task info line (compact, single line)
            info_parts = [task.task_id]
            info_parts.append(f"({task.task_type.value}")
            if task.framework:
                info_parts.append(f"{task.framework}")
            if task.target:
                info_parts.append(f"{task.target})")
            else:
                info_parts[-1] += ")"

            # Print compact task line
            lines.append(f"{prefix}{connector}{' '.join(info_parts)}")

            # Print children with proper tree lines
            if task.children:
                for idx, child_id in enumerate(task.children):
                    child_task = self.tasks[child_id]
                    is_last_child = idx == len(task.children) - 1

                    # Calculate child prefix - very compact
                    if is_root:
                        child_prefix = ""
                    elif is_last_sibling:
                        child_prefix = prefix + "   "
                    else:
                        child_prefix = prefix + "│  "

                    print_task(child_task, child_prefix, is_last_child, is_root=False)

        for idx, task in enumerate(roots):
            is_last = idx == len(roots) - 1
            print_task(task, "", is_last, is_root=True)

        return "\n".join(lines)


class TaskExecutor:
    """Executes tasks with async parallel support"""

    def __init__(self, dry_run: bool = False, verbose: bool = False, log_dir: Optional[Path] = None, repo_sha: Optional[str] = None, repo_path: Optional[Path] = None):
        self.dry_run = dry_run
        self.verbose = verbose
        self.log_dir = log_dir
        self.repo_sha = repo_sha
        self.repo_path = repo_path
        self.logger = logging.getLogger(self.__class__.__name__)

    async def execute_task(self, task: Task, worker_num: int = None) -> bool:
        """Execute a single task"""
        task.status = TaskStatus.RUNNING
        start_time = asyncio.get_event_loop().time()

        # Succinct one-line output with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        worker_prefix = f"#{worker_num} " if worker_num is not None else ""

        # Calculate log file path early for display
        log_file_path_str = None
        log_file = None
        if self.log_dir and not self.dry_run:
            date_str = datetime.now().strftime('%Y-%m-%d')
            sha_prefix = self.repo_sha[:7] if self.repo_sha else "unknown"
            log_suffix = f"{sha_prefix}.{task.task_id.lower()}"
            log_file = self.log_dir / f"{date_str}.{log_suffix}.log"
            log_file_path_str = str(log_file)
            # Store log file in task for later reference
            task.log_file = log_file

        # Print execution message with log path
        exec_msg = f"[{timestamp}] {worker_prefix}Executing: {task.task_id} ({task.task_type.value})"
        if log_file_path_str:
            exec_msg += f" → {log_file_path_str}"
        self.logger.info(exec_msg)

        # Show complete command in verbose mode
        if self.verbose:
            cmd_display = task.command
            # Pretty print multi-line commands
            if len(cmd_display) > 100:
                self.logger.info(f"  Command:")
                # Split on common delimiters for readability
                if ' -- ' in cmd_display:
                    parts = cmd_display.split(' -- ')
                    for i, part in enumerate(parts):
                        if i == 0:
                            self.logger.info(f"    {part} -- \\")
                        else:
                            self.logger.info(f"      {part}")
                else:
                    self.logger.info(f"    {cmd_display}")
            else:
                self.logger.info(f"  Command: {cmd_display}")

        if self.dry_run:
            cmd_display = task.command[:200] + "..." if len(task.command) > 200 else task.command
            self.logger.info(f"  Would execute: {cmd_display}")
            task.status = TaskStatus.SUCCESS
            task.duration = 0.0
            timestamp = datetime.now().strftime("%H:%M:%S")
            worker_prefix = f"#{worker_num} " if worker_num is not None else ""
            self.logger.info(f"[{timestamp}] {worker_prefix}✅ {task.task_id}: Success (0.0s)")
            return True

        try:
            # Use log file already set in task (set earlier in this function)
            if log_file:
                log_file.parent.mkdir(parents=True, exist_ok=True)

            # Determine working directory - use repo_path for compilation/sanity tasks
            cwd = self.repo_path if task.task_type in (TaskType.COMPILATION, TaskType.SANITY_CHECK) else None

            if self.dry_run:
                # Dry-run: capture output
                process = await asyncio.create_subprocess_shell(
                    task.command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=cwd,
                )
                stdout, _ = await asyncio.wait_for(
                    process.communicate(),
                    timeout=task.timeout
                )
            elif log_file:
                # Real run with log file: write to log file only (not console)
                with open(log_file, 'w') as f:
                    process = await asyncio.create_subprocess_shell(
                        task.command,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                        cwd=cwd,
                    )

                    # Stream output to log file only
                    while True:
                        line = await process.stdout.readline()
                        if not line:
                            break
                        decoded_line = line.decode('utf-8')
                        f.write(decoded_line)

                    await asyncio.wait_for(process.wait(), timeout=task.timeout)
            else:
                # Real run without log file: stream to console only
                process = await asyncio.create_subprocess_shell(
                    task.command,
                    stdout=None,
                    stderr=None,
                    cwd=cwd,
                )
                await asyncio.wait_for(process.wait(), timeout=task.timeout)

            task.duration = asyncio.get_event_loop().time() - start_time

            if process.returncode == 0:
                task.status = TaskStatus.SUCCESS
                timestamp = datetime.now().strftime("%H:%M:%S")
                worker_prefix = f"#{worker_num} " if worker_num is not None else ""
                self.logger.info(f"[{timestamp}] {worker_prefix}✅ {task.task_id}: Success ({task.duration:.1f}s)")
                # Create success marker file
                if log_file:
                    success_marker = log_file.with_suffix('.SUCC')
                    success_marker.touch()
                return True
            else:
                task.status = TaskStatus.FAILED
                if self.dry_run:
                    task.error_message = stdout.decode('utf-8')[-1000:]
                timestamp = datetime.now().strftime("%H:%M:%S")
                worker_prefix = f"#{worker_num} " if worker_num is not None else ""
                self.logger.error(f"[{timestamp}] {worker_prefix}❌ {task.task_id}: Failed ({task.duration:.1f}s)")
                # Create failure marker file
                if log_file:
                    fail_marker = log_file.with_suffix('.FAIL')
                    fail_marker.touch()
                return False

        except asyncio.TimeoutError:
            process.kill()
            task.status = TaskStatus.FAILED
            task.error_message = f"Timeout after {task.timeout}s"
            self.logger.error(f"  ❌ {task.task_id}: Timeout")
            return False

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            self.logger.error(f"  ❌ {task.task_id}: Error - {e}")
            return False

    async def execute_pipeline(self, pipeline: BuildPipeline, parallel: bool = False) -> tuple:
        """Execute pipeline with optional parallel execution"""
        successful = 0
        failed = 0

        self.logger.info("\n" + "=" * 80)
        self.logger.info("EXECUTING TASKS")
        self.logger.info("=" * 80)

        if parallel:
            # Greedy/immediate scheduling: start tasks as soon as dependencies complete
            return await self._execute_pipeline_greedy(pipeline)
        else:
            # Serial execution: use level-based approach
            levels = pipeline.get_levels()
            for level_idx, level in enumerate(levels):
                for task in level:
                    result = await self.execute_task(task)
                    if result:
                        successful += 1
                    else:
                        failed += 1

                # Stop on failure
                if failed > 0:
                    # Mark remaining as skipped
                    for remaining_level in levels[level_idx + 1:]:
                        for task in remaining_level:
                            task.status = TaskStatus.SKIPPED
                    break

            return successful, failed

    async def _execute_pipeline_greedy(self, pipeline: BuildPipeline) -> tuple:
        """Execute pipeline with greedy scheduling - start tasks immediately when ready"""
        successful = 0
        failed = 0
        completed_tasks = set()  # Track completed task IDs (including skipped)
        running_tasks = {}  # Map task_id -> asyncio.Task
        task_workers = {}  # Map task_id -> worker_num
        worker_counter = 0  # For assigning worker numbers

        # Get all tasks
        all_tasks = list(pipeline.tasks.values())
        pending_tasks = {task.task_id for task in all_tasks if task.status == TaskStatus.PENDING}

        # Mark skipped tasks as completed (for dependency resolution)
        for task in all_tasks:
            if task.status == TaskStatus.SKIPPED:
                completed_tasks.add(task.task_id)

        async def run_task_wrapper(task: Task, worker_num: int):
            """Wrapper to execute task and return result"""
            result = await self.execute_task(task, worker_num=worker_num)
            return task.task_id, result, worker_num

        while pending_tasks or running_tasks:
            # Find tasks that are ready to run (all dependencies completed)
            ready_tasks = [
                task for task in all_tasks
                if task.task_id in pending_tasks and
                all(dep in completed_tasks for dep in task.depends_on)
            ]

            # Start all ready tasks
            for task in ready_tasks:
                worker_counter += 1
                pending_tasks.remove(task.task_id)
                task_workers[task.task_id] = worker_counter
                running_tasks[task.task_id] = asyncio.create_task(
                    run_task_wrapper(task, worker_counter)
                )

            # Wait for at least one task to complete (or all if none are ready to start)
            if running_tasks:
                done, pending = await asyncio.wait(
                    running_tasks.values(),
                    return_when=asyncio.FIRST_COMPLETED
                )

                # Process completed tasks
                for completed_future in done:
                    task_id, result, worker_num = await completed_future
                    del running_tasks[task_id]
                    del task_workers[task_id]
                    completed_tasks.add(task_id)

                    if result:
                        successful += 1
                        # Show which workers are still running (if any)
                        if running_tasks:
                            still_running = sorted(task_workers.values())
                            workers_str = ', '.join(f"#{w}" for w in still_running)
                            self.logger.info(f"  ⏳ Still running: {workers_str}")
                    else:
                        failed += 1
                        # On failure, mark all remaining tasks as skipped
                        for task in all_tasks:
                            if task.task_id in pending_tasks or task.task_id in running_tasks:
                                task.status = TaskStatus.SKIPPED
                        # Cancel running tasks
                        for running_task in running_tasks.values():
                            running_task.cancel()
                        return successful, failed
            else:
                # No tasks running and none ready - check for circular dependencies
                if pending_tasks:
                    self.logger.error(f"Circular dependency detected: {pending_tasks}")
                break

        return successful, failed


class PipelineBuilder:
    """Builds task pipeline matching V1 logic exactly"""

    def __init__(self, repo_path: Path, frameworks: List[str], targets: List[str],
                 dry_run: bool = False, verbose: bool = False):
        self.repo_path = repo_path
        self.frameworks = frameworks
        self.targets = targets
        self.dry_run = dry_run
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)
        self.docker_utils = DockerUtils(dry_run, verbose)
        self.seen_commands: Dict[str, str] = {}  # command -> task_id

    def _get_build_commands(self, framework: str, target: Optional[str]) -> List[str]:
        """Get docker build commands from build.sh --dry-run and filter out latest tags"""
        script_path = self.repo_path / "container" / "build.sh"

        try:
            cmd = [str(script_path), "--framework", framework.lower(), "--dry-run"]
            if target:
                cmd.extend(["--target", target])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.repo_path / "container",
                timeout=30
            )

            if result.returncode != 0:
                return []

            # Extract and filter docker commands to remove latest tags
            commands = []
            import re
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.startswith("docker build"):
                    # Remove --tag arguments containing ":latest" using regex
                    line = re.sub(r'--tag\s+\S*:latest\S*(?:\s|$)', '', line)
                    commands.append(line)

            return commands

        except Exception as e:
            self.logger.error(f"Failed to get build commands: {e}")
            return []

    def build_pipeline(self) -> BuildPipeline:
        """Build pipeline matching V1's exact logic"""
        pipeline = BuildPipeline()

        self.logger.info("Building dependency tree...")
        self.logger.info("")

        # Step 1: Create build tasks (matching V1 logic)
        for framework in self.frameworks:
            self.logger.info(f"Creating tasks for {framework}...")

            for target in self.targets:
                # Get commands (V1 uses None for "dev")
                target_arg = None if target == "dev" else target
                commands = self._get_build_commands(framework, target_arg)

                for docker_cmd in commands:
                    # Skip duplicates
                    if docker_cmd in self.seen_commands:
                        self.logger.debug(f"  Skipping duplicate command")
                        continue

                    # Extract metadata
                    base_image = self.docker_utils.extract_base_image_from_command(docker_cmd)
                    image_tag = self.docker_utils.extract_image_tag_from_command(docker_cmd)

                    if not image_tag:
                        continue

                    # Determine task_id based on image_tag (matching V1 logic)
                    if "dynamo-base" in image_tag:
                        task_id = f"{framework}-dynamo-base"
                        task_target = "base"
                    elif "local-dev" in image_tag:
                        task_id = f"{framework}-local-dev"
                        task_target = "local-dev"
                    else:
                        task_id = f"{framework}-dev"
                        task_target = "dev"

                    task = Task(
                        task_id=task_id,
                        task_type=TaskType.BUILD,
                        framework=framework,
                        target=task_target,
                        description=f"Build {framework} {task_target} image",
                        command=docker_cmd,
                        image_tag=image_tag,
                        base_image=base_image
                    )

                    pipeline.add_task(task)
                    self.seen_commands[docker_cmd] = task_id
                    self.logger.info(f"  Created task: {task_id}")

        # Step 2: Establish dependencies based on base_image
        self.logger.info("")
        self.logger.info("Establishing dependencies...")

        for task_id, task in pipeline.tasks.items():
            if task.task_type != TaskType.BUILD or not task.base_image:
                continue

            # Find parent task that produces this base_image
            for other_id, other_task in pipeline.tasks.items():
                if other_id == task_id:
                    continue
                if other_task.task_type == TaskType.BUILD and task.base_image == other_task.image_tag:
                    # Prefer same framework
                    if task.framework == other_task.framework:
                        pipeline.add_dependency(task_id, other_id)
                        self.logger.info(f"  {task_id} depends on {other_id}")
                        break

        # Step 3: Create compilation tasks
        self.logger.info("")
        self.logger.info("Creating compilation tasks...")

        for framework in self.frameworks:
            local_dev_id = f"{framework}-local-dev"
            if local_dev_id not in pipeline.tasks:
                continue

            local_dev_task = pipeline.tasks[local_dev_id]
            image_tag = local_dev_task.image_tag

            compilation_id = f"{framework}-compilation"
            compilation_task = Task(
                task_id=compilation_id,
                task_type=TaskType.COMPILATION,
                framework=framework,
                target="local-dev",
                description=f"Run workspace compilation in {framework} local-dev container",
                command=f"./container/run.sh --image {image_tag} --mount-workspace -- bash -c 'cargo build --locked --profile dev --features dynamo-llm/block-manager && cd /workspace/lib/bindings/python && maturin develop && uv pip install -e .'",
                timeout=1800.0
            )

            pipeline.add_task(compilation_task)
            pipeline.add_dependency(compilation_id, local_dev_id)
            self.logger.info(f"  Created task: {compilation_id} (depends on {local_dev_id})")

        # Step 4: Create sanity check tasks
        self.logger.info("")
        self.logger.info("Creating sanity check tasks...")

        for framework in self.frameworks:
            compilation_id = f"{framework}-compilation"
            if compilation_id not in pipeline.tasks:
                continue

            for target in ['dev', 'local-dev']:
                task_id = f"{framework}-{target}"
                if task_id not in pipeline.tasks:
                    continue

                build_task = pipeline.tasks[task_id]
                image_tag = build_task.image_tag

                sanity_id = f"{framework}-{target}-sanity"
                sanity_task = Task(
                    task_id=sanity_id,
                    task_type=TaskType.SANITY_CHECK,
                    framework=framework,
                    target=target,
                    description=f"Run sanity_check.py in {framework} {target} container (after compilation)",
                    command=f"./container/run.sh --image {image_tag} --mount-workspace --entrypoint deploy/sanity_check.py",
                    timeout=120.0
                )

                pipeline.add_task(sanity_task)
                pipeline.add_dependency(sanity_id, compilation_id)
                self.logger.info(f"  Created task: {sanity_id} (depends on {compilation_id})")

        self.logger.info("")
        self.logger.info(f"Dependency tree built: {len(pipeline.tasks)} tasks total")

        # Resolve image tag conflicts (matching V1 logic)
        self._resolve_image_tag_conflicts(pipeline)

        return pipeline

    def _resolve_image_tag_conflicts(self, pipeline: BuildPipeline) -> None:
        """
        Detect and resolve image tag conflicts when multiple tasks produce the same tag.
        Matching V1's logic.
        """
        self.logger.info("")
        self.logger.info("Checking for image tag conflicts...")

        # Step 1: Build mapping of image_tag -> [task_ids that produce it]
        image_tag_to_tasks: Dict[str, List[str]] = {}
        for task_id, task in pipeline.tasks.items():
            if task.task_type != TaskType.BUILD or not task.image_tag:
                continue

            if task.image_tag not in image_tag_to_tasks:
                image_tag_to_tasks[task.image_tag] = []
            image_tag_to_tasks[task.image_tag].append(task_id)

        # Step 2: Find conflicts (tags produced by multiple tasks)
        conflicts: Set[str] = set()
        for image_tag, task_ids in image_tag_to_tasks.items():
            if len(task_ids) > 1:
                conflicts.add(image_tag)
                self.logger.info(f"  Conflict detected: {image_tag}")
                for task_id in task_ids:
                    self.logger.info(f"    - {task_id}")

        if not conflicts:
            self.logger.info("  No image tag conflicts found")
            return

        # Step 3: Rename conflicting tags by appending framework suffix
        task_to_new_tag: Dict[str, str] = {}  # task_id -> new_tag

        for task_id, task in pipeline.tasks.items():
            if task.task_type != TaskType.BUILD:
                continue

            if not task.image_tag or task.image_tag not in conflicts:
                continue

            # Check if tag already has framework suffix
            if task.framework and task.image_tag.endswith(f"-{task.framework.lower()}"):
                continue

            # Create new tag with framework suffix
            old_tag = task.image_tag
            if task.framework:
                new_tag = f"{old_tag}-{task.framework.lower()}"
            else:
                # Fallback: use part of task_id
                new_tag = f"{old_tag}-{task_id.split('-')[0].lower()}"

            task_to_new_tag[task_id] = new_tag

            # Update the task's image_tag
            task.image_tag = new_tag

            # Update the docker command to use the new tag
            if task.command:
                import re
                task.command = re.sub(
                    rf'--tag\s+{re.escape(old_tag)}(?=\s|$)',
                    f'--tag {new_tag}',
                    task.command
                )

            self.logger.info(f"  Renamed: {task_id}")
            self.logger.info(f"    Old: {old_tag}")
            self.logger.info(f"    New: {new_tag}")

        # Step 4: Update base_image references in dependent tasks
        for task_id, task in pipeline.tasks.items():
            if task.task_type != TaskType.BUILD:
                continue

            if not task.base_image or task.base_image not in conflicts:
                continue

            # Find which parent task this depends on
            for parent_task_id in task.depends_on:
                parent_task = pipeline.tasks.get(parent_task_id)
                if not parent_task or parent_task.task_type != TaskType.BUILD:
                    continue

                # Check if parent was renamed
                if parent_task_id in task_to_new_tag:
                    old_base = task.base_image
                    new_base = task_to_new_tag[parent_task_id]
                    task.base_image = new_base
                    self.logger.info(f"  Updated base_image: {task_id}")
                    self.logger.info(f"    {old_base} -> {new_base}")

                    # Also update docker command if it references the old tag
                    if task.command and old_base in task.command:
                        task.command = task.command.replace(old_base, new_base)
                    break

        # Step 5: Update image references in COMPILATION and SANITY_CHECK tasks
        for task_id, task in pipeline.tasks.items():
            if task.task_type not in (TaskType.COMPILATION, TaskType.SANITY_CHECK):
                continue

            if not task.command:
                continue

            # Find parent build task and update command if its tag was renamed
            for parent_task_id in task.depends_on:
                parent_task = pipeline.tasks.get(parent_task_id)
                if not parent_task:
                    continue

                # Check if parent was renamed
                if parent_task_id in task_to_new_tag:
                    old_tag = None
                    # Find the old tag by checking what the parent's original tag was
                    for conflict_tag in conflicts:
                        if task_to_new_tag[parent_task_id].startswith(conflict_tag):
                            old_tag = conflict_tag
                            break

                    if old_tag:
                        new_tag = task_to_new_tag[parent_task_id]
                        if old_tag in task.command:
                            task.command = task.command.replace(old_tag, new_tag)
                            self.logger.info(f"  Updated image reference in {task_id}")
                            self.logger.info(f"    {old_tag} -> {new_tag}")


class ReportGenerator:
    """Generate HTML reports from build pipeline results"""

    @staticmethod
    def generate_html_report(
        pipeline: BuildPipeline,
        repo_path: Path,
        repo_sha: str,
        log_dir: Path,
        date_str: str,
        hostname: str = "keivenc-linux",
        html_path: str = "/nvidia/dynamo_ci/logs",
        use_absolute_urls: bool = False
    ) -> str:
        """Generate elegant HTML report matching V1 format with clickable log links

        Args:
            use_absolute_urls: If True, use http://hostname/... URLs. If False, use relative paths.
        """
        import git

        # Helper function to generate log URL from task
        def get_log_url(task: Task) -> Optional[str]:
            """Generate URL/path for task log file"""
            if not task.log_file:
                return None
            try:
                if use_absolute_urls:
                    # Email: absolute URL with http://
                    # Get relative path from logs directory (includes date subdirectory)
                    rel_path = task.log_file.relative_to(log_dir.parent)
                    return f"http://{hostname}{html_path}/{rel_path}"
                else:
                    # File: relative path from HTML report location
                    # HTML is in same directory as log files, so just use filename
                    return str(task.log_file.name)
            except ValueError:
                return None

        # Get git information
        repo = git.Repo(repo_path)
        commit = repo.commit(repo_sha)
        sha_short = repo_sha[:7]

        # Extract PR number from commit message (format: "title (#1234)")
        import re
        pr_number = None
        pr_link = None
        first_line = commit.message.split('\n')[0] if commit.message else ""
        pr_match = re.search(r'\(#(\d+)\)', first_line)
        if pr_match:
            pr_number = pr_match.group(1)
            pr_link = f"https://github.com/ai-dynamo/dynamo/pull/{pr_number}"

        # Get file changes (stats for this commit)
        try:
            stats = commit.stats
            files_changed = stats.files
            total_insertions = stats.total['insertions']
            total_deletions = stats.total['deletions']
        except Exception:
            files_changed = {}
            total_insertions = 0
            total_deletions = 0

        # Count task statistics
        total_tasks = len(pipeline.tasks)
        succeeded = sum(1 for t in pipeline.tasks.values() if t.status == TaskStatus.SUCCESS)
        failed = sum(1 for t in pipeline.tasks.values() if t.status == TaskStatus.FAILED)
        skipped = sum(1 for t in pipeline.tasks.values() if t.status == TaskStatus.SKIPPED)

        # Determine overall status
        overall_status = "✅ ALL TESTS PASSED" if failed == 0 else "❌ TESTS FAILED"
        header_color = "#28a745" if failed == 0 else "#dc3545"

        # Organize tasks by framework
        frameworks = {}
        compilation_tasks = []

        for task_id, task in pipeline.tasks.items():
            if task.task_type == TaskType.COMPILATION:
                compilation_tasks.append(task)
            elif task.framework and task.target:
                if task.framework not in frameworks:
                    frameworks[task.framework] = {}
                if task.target not in frameworks[task.framework]:
                    frameworks[task.framework][task.target] = {
                        'build': None, 'sanity': None
                    }
                if task.task_type == TaskType.BUILD:
                    frameworks[task.framework][task.target]['build'] = task
                elif task.task_type == TaskType.SANITY_CHECK:
                    frameworks[task.framework][task.target]['sanity'] = task

        # Build HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>DynamoDockerBuilder - {sha_short} - {overall_status}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 10px; line-height: 1.3; }}
.header {{ background-color: {header_color}; color: white; padding: 15px 20px; border-radius: 4px; margin-bottom: 10px; text-align: center; }}
.summary {{ background-color: #f8f9fa; padding: 4px 6px; border-radius: 2px; margin: 3px 0; }}
.summary-boxes {{ width: 100%; margin: 15px 0; }}
.summary-boxes table {{ width: 100%; border-collapse: separate; border-spacing: 10px; }}
.summary-box {{ padding: 12px; border-radius: 6px; text-align: center; width: 25%; }}
.summary-box.total {{ background: #e8f4f8; border-left: 4px solid #3498db; }}
.summary-box.success {{ background: #e8f8f5; border-left: 4px solid #27ae60; }}
.summary-box.failed {{ background: #fde8e8; border-left: 4px solid #e74c3c; }}
.summary-box.skipped {{ background: #f9f9f9; border-left: 4px solid #95a5a6; }}
.summary-box .number {{ font-size: 28px; font-weight: bold; margin: 5px 0; display: block; }}
.summary-box .label {{ font-size: 13px; color: #7f8c8d; text-transform: uppercase; font-weight: 600; display: block; }}
.framework {{ margin: 10px 0; padding: 8px; border: 1px solid #dee2e6; border-radius: 4px; background-color: #ffffff; }}
.framework-header {{ background-color: #007bff; color: white; padding: 8px 12px; margin: -8px -8px 8px -8px; border-radius: 4px 4px 0 0; font-weight: bold; }}
.results-chart {{ display: table; width: 100%; border-collapse: collapse; margin: 8px 0; }}
.chart-row {{ display: table-row; }}
.chart-cell {{ display: table-cell; padding: 6px 12px; border: 1px solid #dee2e6; vertical-align: middle; }}
.chart-header {{ background-color: #f8f9fa; font-weight: bold; text-align: center; }}
.chart-target {{ font-weight: bold; background-color: #f1f3f4; }}
.chart-status {{ text-align: center; }}
.chart-timing {{ text-align: right; font-family: monospace; font-size: 0.9em; }}
.success {{ color: #28a745; font-weight: bold; }}
.failure {{ color: #dc3545; font-weight: bold; }}
.git-info {{ background-color: #e9ecef; padding: 4px 6px; border-radius: 2px; font-family: monospace; font-size: 0.9em; }}
a {{ color: #007bff; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
a.log-link {{ font-size: 0.85em; margin-left: 4px; }}
p {{ margin: 1px 0; }}
h2 {{ margin: 0; font-size: 1.2em; font-weight: bold; }}
</style>
</head>
<body>
<div class="header">
<h2>DynamoDockerBuilder - <a href="https://github.com/ai-dynamo/dynamo/commit/{repo_sha}" style="color: white; text-decoration: underline;">{sha_short}</a> - {overall_status}</h2>
</div>

<div class="summary-boxes">
<table>
<tr>
<td class="summary-box total">
<div class="number">{total_tasks}</div>
<div class="label">Total Tasks</div>
</td>
<td class="summary-box success">
<div class="number">{succeeded}</div>
<div class="label">Succeeded</div>
</td>
<td class="summary-box failed">
<div class="number">{failed}</div>
<div class="label">Failed</div>
</td>
<td class="summary-box skipped">
<div class="number">{skipped}</div>
<div class="label">Skipped</div>
</td>
</tr>
</table>
</div>

<div class="summary">
<p><strong>Build Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}</p>
</div>

<div class="git-info">
<p><strong>Commit SHA:</strong> <a href="https://github.com/ai-dynamo/dynamo/commit/{repo_sha}">{sha_short}</a></p>"""

        # Add PR link if found
        if pr_link:
            html += f"""<p><strong>Pull Request:</strong> <a href="{pr_link}">#{pr_number}</a></p>"""

        html += f"""<p><strong>Commit Date:</strong> {commit.committed_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')}</p>
<p><strong>Author:</strong> {commit.author.name} &lt;{commit.author.email}&gt;</p>
<div style="background-color: #f8f9fa; padding: 6px; border-radius: 3px; margin: 3px 0;">
<strong>Commit Message:</strong>
<pre style="margin: 3px 0; white-space: pre-wrap; font-family: monospace; font-size: 0.9em;">{commit.message.strip()}</pre>
</div>

<p><strong>Changes Summary:</strong> +{total_insertions}/-{total_deletions} lines</p>
"""

        # Add files changed section if there are any changes
        if files_changed:
            html += """
<p><strong>Files Changed with Line Counts:</strong></p>
<div style="background-color: #f8f9fa; padding: 6px; border-radius: 3px; font-family: monospace; font-size: 0.9em; margin: 3px 0;">
"""
            for file_path, file_stats in sorted(files_changed.items()):
                insertions = file_stats.get('insertions', 0)
                deletions = file_stats.get('deletions', 0)
                if insertions and deletions:
                    html += f"\n• {file_path} +{insertions}/-{deletions}<br>"
                elif insertions:
                    html += f"\n• {file_path} +{insertions}<br>"
                elif deletions:
                    html += f"\n• {file_path} -{deletions}<br>"
                else:
                    html += f"\n• {file_path}<br>"
            html += "\n</div>\n"

        html += "</div>\n"
        html += """
"""

        # Compilation section
        if compilation_tasks:
            html += """
<div class="framework">
<div class="framework-header">Compilation</div>
<div class="results-chart">
<div class="chart-row">
<div class="chart-cell chart-header">Framework</div>
<div class="chart-cell chart-header">Status</div>
<div class="chart-cell chart-header">Time</div>
</div>
"""
            for task in compilation_tasks:
                status_class = "success" if task.status == TaskStatus.SUCCESS else "failure"
                status_text = "✅ PASS" if task.status == TaskStatus.SUCCESS else "❌ FAIL"
                time_text = f"{task.duration:.1f}s" if task.duration else "-"
                # Add log link if available
                log_url = get_log_url(task)
                if log_url:
                    status_text += f' <a href="{log_url}" class="log-link">[log]</a>'
                html += f"""
<div class="chart-row">
<div class="chart-cell chart-target">{task.framework}</div>
<div class="chart-cell chart-status {status_class}">{status_text}</div>
<div class="chart-cell chart-timing">{time_text}</div>
</div>
"""
            html += "</div></div>\n"

        # Framework sections
        for framework_name in sorted(frameworks.keys()):
            targets = frameworks[framework_name]
            html += f"""
<div class="framework">
<div class="framework-header">{framework_name}</div>
<div class="results-chart">
<div class="chart-row">
<div class="chart-cell chart-header">Target</div>
<div class="chart-cell chart-header">Build</div>
<div class="chart-cell chart-header">Build Time</div>
<div class="chart-cell chart-header">Sanity Check</div>
<div class="chart-cell chart-header">Sanity Time</div>
<div class="chart-cell chart-header">Image Size</div>
</div>
"""
            for target_name in sorted(targets.keys()):
                target_data = targets[target_name]
                build_task = target_data['build']
                sanity_task = target_data['sanity']

                # Get image size from either build or sanity task
                image_size = "-"
                if build_task and build_task.image_size:
                    image_size = build_task.image_size
                elif sanity_task and sanity_task.image_size:
                    image_size = sanity_task.image_size

                # Build status
                if build_task:
                    if build_task.status == TaskStatus.SUCCESS:
                        build_status = '<span class="success">✅ PASS</span>'
                        build_time = f"{build_task.duration:.1f}s"
                    elif build_task.status == TaskStatus.SKIPPED:
                        build_status = "⏭️ SKIP"
                        build_time = "skipped"
                    else:
                        build_status = '<span class="failure">❌ FAIL</span>'
                        build_time = f"{build_task.duration:.1f}s" if build_task.duration else "failed"
                    # Add log link if available
                    log_url = get_log_url(build_task)
                    if log_url:
                        build_status += f' <a href="{log_url}" class="log-link">[log]</a>'
                else:
                    build_status = "-"
                    build_time = "-"

                # Sanity status
                if sanity_task:
                    if sanity_task.status == TaskStatus.SUCCESS:
                        sanity_status = '<span class="success">✅ PASS</span>'
                        sanity_time = f"{sanity_task.duration:.1f}s"
                    elif sanity_task.status == TaskStatus.SKIPPED:
                        sanity_status = "⏭️ SKIP"
                        sanity_time = "skipped"
                    else:
                        sanity_status = '<span class="failure">❌ FAIL</span>'
                        sanity_time = f"{sanity_task.duration:.1f}s" if sanity_task.duration else "failed"
                    # Add log link if available
                    log_url = get_log_url(sanity_task)
                    if log_url:
                        sanity_status += f' <a href="{log_url}" class="log-link">[log]</a>'
                else:
                    sanity_status = "-"
                    sanity_time = "-"

                html += f"""
<div class="chart-row">
<div class="chart-cell chart-target">{target_name}</div>
<div class="chart-cell chart-status">{build_status}</div>
<div class="chart-cell chart-timing">{build_time}</div>
<div class="chart-cell chart-status">{sanity_status}</div>
<div class="chart-cell chart-timing">{sanity_time}</div>
<div class="chart-cell chart-timing">{image_size}</div>
</div>
"""
            html += "</div></div>\n"

        html += "</body></html>"
        return html


class DynamoDockerBuilderV2:
    """Main V2 orchestrator - simpler than V1 but same behavior"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.repo_path = Path("../dynamo_ci").resolve()
        self.docker_client = None
        self.lock_file = Path(__file__).parent / f".{Path(__file__).name}.lock"
        # Initialize repository utils (will be set properly in run())
        self.repo_utils = None

    def check_if_running(self, force_run: bool = False) -> None:
        """Check if another instance is already running"""
        import os
        import atexit
        import psutil

        script_name = Path(__file__).name
        current_pid = os.getpid()

        # Skip lock check if force_run is specified
        if force_run:
            self.logger.warning("FORCE-RUN MODE: Bypassing process lock check")
            self.lock_file.write_text(str(current_pid))
            self.logger.info(f"Created process lock file: {self.lock_file} (PID: {current_pid})")
            atexit.register(lambda: self.lock_file.unlink(missing_ok=True))
            return

        # Check if lock file exists
        if self.lock_file.exists():
            try:
                existing_pid = int(self.lock_file.read_text().strip())

                # Check if the process is still running
                if psutil.pid_exists(existing_pid):
                    try:
                        proc = psutil.Process(existing_pid)
                        if script_name in " ".join(proc.cmdline()):
                            self.logger.error(f"Another instance of {script_name} is already running (PID: {existing_pid})")
                            self.logger.error(f"If you're sure no other instance is running, remove the lock file:")
                            self.logger.error(f"  rm '{self.lock_file}'")
                            self.logger.error(f"Or use --force-run to bypass the check")
                            sys.exit(1)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        # Process exists but not accessible, remove stale lock
                        self.logger.warning(f"Removing stale lock file (PID {existing_pid})")
                        self.lock_file.unlink()
                else:
                    # Process doesn't exist, remove stale lock file
                    self.logger.warning(f"Removing stale lock file (PID {existing_pid} no longer exists)")
                    self.lock_file.unlink()
            except (ValueError, FileNotFoundError):
                # Invalid lock file content, remove it
                self.logger.warning("Removing invalid lock file")
                self.lock_file.unlink(missing_ok=True)

        # Create lock file with current PID
        self.lock_file.write_text(str(current_pid))
        self.logger.info(f"Created process lock file: {self.lock_file} (PID: {current_pid})")

        # Set up cleanup on exit
        atexit.register(lambda: self.lock_file.unlink(missing_ok=True))

    def _mark_existing_images_skipped(self, pipeline: BuildPipeline) -> None:
        """Mark build tasks as skipped if their images already exist"""
        if not self.docker_client:
            self.docker_client = docker.from_env()

        skipped_count = 0
        for task_id, task in pipeline.tasks.items():
            # Only check build tasks
            if task.task_type != TaskType.BUILD:
                continue

            # Check if image exists
            if task.image_tag:
                try:
                    self.docker_client.images.get(task.image_tag)
                    # Image exists - mark as skipped
                    task.status = TaskStatus.SKIPPED
                    skipped_count += 1
                    self.logger.info(f"  Skipping {task_id}: image {task.image_tag} already exists")
                except docker.errors.ImageNotFound:
                    # Image doesn't exist - will need to build
                    pass
                except Exception as e:
                    self.logger.warning(f"  Error checking image {task.image_tag}: {e}")

        if skipped_count > 0:
            self.logger.info(f"Skipped {skipped_count} existing images")
            self.logger.info("")

    def _apply_sanity_check_only_mode(self, pipeline: BuildPipeline) -> None:
        """Mark build tasks as skipped when in sanity-check-only mode (but keep compilation)"""
        if not self.docker_client:
            self.docker_client = docker.from_env()

        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("SANITY-CHECK-ONLY MODE")
        self.logger.info("=" * 80)
        self.logger.info("Skipping build tasks (compilation and sanity checks will run)...")
        self.logger.info("")

        # Track required images for sanity checks and map images to tasks
        required_images: Set[str] = set()
        image_to_tasks: Dict[str, List[Task]] = {}  # Map image name to tasks that use it
        sanity_tasks = []
        compilation_tasks = []

        # Find all sanity check tasks and skip ONLY build tasks
        for task_id, task in pipeline.tasks.items():
            if task.task_type == TaskType.SANITY_CHECK:
                sanity_tasks.append(task)
                # Extract image name from command
                import re
                match = re.search(r'--image\s+(\S+)', task.command)
                if match:
                    image_name = match.group(1)
                    required_images.add(image_name)
                    if image_name not in image_to_tasks:
                        image_to_tasks[image_name] = []
                    image_to_tasks[image_name].append(task)
            elif task.task_type == TaskType.COMPILATION:
                compilation_tasks.append(task)
                # Extract image name from compilation command
                import re
                match = re.search(r'--image\s+(\S+)', task.command)
                if match:
                    image_name = match.group(1)
                    required_images.add(image_name)
                    if image_name not in image_to_tasks:
                        image_to_tasks[image_name] = []
                    image_to_tasks[image_name].append(task)
            elif task.task_type == TaskType.BUILD:
                task.status = TaskStatus.SKIPPED
                self.logger.info(f"  Skipping: {task_id} (build)")

        # Verify required images exist and store sizes
        self.logger.info("")
        self.logger.info("Verifying required images exist...")
        missing_images = []

        for image_name in sorted(required_images):
            try:
                img = self.docker_client.images.get(image_name)
                size_bytes = img.attrs.get('Size', 0)
                size_gb = size_bytes / (1024**3)
                size_str = f"{size_gb:.1f} GB"
                self.logger.info(f"  ✅ Found: {image_name} ({size_str})")

                # Store size in all tasks that use this image
                if image_name in image_to_tasks:
                    for task in image_to_tasks[image_name]:
                        task.image_size = size_str
            except docker.errors.ImageNotFound:
                missing_images.append(image_name)
                self.logger.error(f"  ❌ Missing: {image_name}")
            except Exception as e:
                missing_images.append(image_name)
                self.logger.error(f"  ❌ Error checking {image_name}: {e}")

        if missing_images:
            self.logger.error("")
            self.logger.error("=" * 80)
            self.logger.error("ERROR: Missing required Docker images")
            self.logger.error("=" * 80)
            self.logger.error("The following images are required but not found:")
            for img in missing_images:
                self.logger.error(f"  - {img}")
            self.logger.error("")
            self.logger.error("Please build the images first by running without --sanity-check-only")
            sys.exit(1)

        self.logger.info("")
        self.logger.info(f"Sanity check mode: {len(compilation_tasks)} compilation + {len(sanity_tasks)} sanity checks will run")
        self.logger.info("")


    def show_commit_history(self, max_commits: int = 50, verbose: bool = False, html_output: bool = False) -> int:
        """Show recent commit history with composite SHAs

        Args:
            max_commits: Maximum number of commits to show
            verbose: Enable verbose output
            html_output: Generate HTML output instead of terminal output
        """
        import git
        import time
        import re
        import json

        # Initialize repo utils for this operation
        repo_utils = DynamoRepositoryUtils(self.repo_path, dry_run=False, verbose=verbose)

        # Load cache
        cache_file = Path("~/nvidia/dynamo_ci/.commit_history_cache.json")
        cache = {}
        if cache_file.exists():
            try:
                cache = json.loads(cache_file.read_text())
                if verbose:
                    print(f"Loaded cache with {len(cache)} entries")
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")

        try:
            repo = git.Repo(self.repo_path)
            commits = list(repo.iter_commits('HEAD', max_count=max_commits))
            original_head = repo.head.commit.hexsha

            # Collect commit data
            commit_data = []
            cache_updated = False

            if not html_output:
                # Terminal output mode
                term_width = get_terminal_width(padding=2, default=118)
                sha_width = 10
                composite_width = 13
                date_width = 20
                author_width = 20
                separator_width = 3
                fixed_width = sha_width + composite_width + date_width + author_width + separator_width
                message_width = max(30, term_width - fixed_width)

                print(f"\nCommit History with Composite SHAs")
                print(f"Repository: {self.repo_path}")
                print(f"Showing {len(commits)} most recent commits:\n")
                print(f"{'Commit SHA':<{sha_width}} {'Composite SHA':<{composite_width}} {'Date':<{date_width}} {'Author':<{author_width}} Message")
                print("-" * term_width)

            try:
                for i, commit in enumerate(commits):
                    sha_short = commit.hexsha[:9]
                    sha_full = commit.hexsha
                    date_str = commit.committed_datetime.strftime('%Y-%m-%d %H:%M:%S')
                    author_name = commit.author.name
                    message_first_line = commit.message.strip().split('\n')[0]

                    if not html_output:
                        # Terminal: show progress with placeholder
                        author_str = author_name[:author_width-1]
                        message_str = message_first_line[:message_width]
                        print(f"{sha_short:<{sha_width}} {'CALCULATING...':<{composite_width}} {date_str:<{date_width}} {author_str:<{author_width}} {message_str}", end='', flush=True)

                    # Check cache first
                    if sha_full in cache:
                        composite_sha = cache[sha_full]
                        if verbose:
                            print(f"Cache hit for {sha_short}: {composite_sha}")
                    else:
                        # Checkout commit and calculate composite SHA
                        try:
                            repo.git.checkout(commit.hexsha)
                            composite_sha = repo_utils.generate_composite_sha()
                            # Update cache
                            cache[sha_full] = composite_sha
                            cache_updated = True
                            if verbose:
                                print(f"Calculated and cached {sha_short}: {composite_sha}")
                        except Exception as e:
                            composite_sha = "ERROR"

                    if html_output:
                        # Collect data for HTML generation
                        commit_data.append({
                            'sha_short': sha_short,
                            'sha_full': sha_full,
                            'composite_sha': composite_sha,
                            'date': date_str,
                            'author': author_name,
                            'message': message_first_line
                        })
                        if verbose:
                            print(f"Processed commit {i+1}/{len(commits)}: {sha_short}")
                    else:
                        # Terminal: overwrite line with actual composite SHA
                        author_str = author_name[:author_width-1]
                        message_str = message_first_line[:message_width]
                        print(f"\r{sha_short:<{sha_width}} {composite_sha:<{composite_width}} {date_str:<{date_width}} {author_str:<{author_width}} {message_str}")

            finally:
                # Restore original HEAD
                repo.git.checkout(original_head)
                if not html_output:
                    print(f"\nRestored HEAD to {original_head[:9]}")

            # Generate HTML if requested
            if html_output:
                html_content = self._generate_commit_history_html(commit_data)
                output_path = Path("~/nvidia/dynamo_ci/logs/commit-history.html")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(html_content)
                print(f"\nHTML report generated: {output_path}")
                print(f"Restored HEAD to {original_head[:9]}")

            # Save cache if updated
            if cache_updated:
                try:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(json.dumps(cache, indent=2))
                    if verbose:
                        print(f"Cache saved with {len(cache)} entries")
                except Exception as e:
                    self.logger.warning(f"Failed to save cache: {e}")

            return 0
        except KeyboardInterrupt:
            print("\n\nOperation interrupted by user")
            # Try to restore HEAD
            try:
                repo.git.checkout(original_head)
            except:
                pass
            return 1
        except Exception as e:
            self.logger.error(f"Failed to get commit history: {e}")
            return 1

    def _generate_commit_history_html(self, commit_data: List[dict]) -> str:
        """Generate HTML report for commit history with Docker image detection

        Args:
            commit_data: List of commit dictionaries with sha_short, sha_full, composite_sha, date, author, message
        """
        import re
        import subprocess

        # Get Docker images containing SHAs
        docker_images = self._get_docker_images_by_sha([c['sha_short'] for c in commit_data])

        html = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Dynamo Commit History</title>
<style>
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
    margin: 10px;
    line-height: 1.3;
    background-color: #f6f8fa;
    font-size: 13px;
}
.header {
    background-color: #24292f;
    color: white;
    padding: 10px 15px;
    border-radius: 4px;
    margin-bottom: 10px;
}
.header h1 {
    margin: 0;
    font-size: 18px;
}
.container {
    background-color: white;
    border: 1px solid #d0d7de;
    border-radius: 4px;
    overflow: hidden;
}
table {
    width: 100%;
    border-collapse: collapse;
}
th {
    background-color: #f6f8fa;
    padding: 6px 8px;
    text-align: left;
    font-weight: 600;
    border-bottom: 1px solid #d0d7de;
    position: sticky;
    top: 0;
    font-size: 12px;
}
td {
    padding: 6px 8px;
    border-bottom: 1px solid #d0d7de;
    vertical-align: top;
}
tr:hover {
    background-color: #f6f8fa;
}
.sha {
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
    font-size: 12px;
}
.sha a {
    color: #0969da;
    text-decoration: none;
}
.sha a:hover {
    text-decoration: underline;
}
.message {
    color: #24292f;
}
.pr-link {
    color: #0969da;
    text-decoration: none;
    font-weight: 500;
}
.pr-link:hover {
    text-decoration: underline;
}
.docker-images {
    margin-top: 8px;
}
details {
    margin-top: 4px;
}
summary {
    cursor: pointer;
    color: #0969da;
    font-size: 13px;
    font-weight: 500;
}
summary:hover {
    text-decoration: underline;
}
.image-list {
    margin: 8px 0;
    padding-left: 20px;
}
.image-tag {
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
    font-size: 12px;
    background-color: #f6f8fa;
    padding: 2px 6px;
    border-radius: 3px;
    display: block;
    margin: 4px 0;
}
.date {
    color: #57606a;
    font-size: 14px;
}
.author {
    color: #24292f;
    font-size: 14px;
}
.composite-sha-link {
    color: #0969da;
    text-decoration: none;
    cursor: pointer;
    border-bottom: 1px dotted #0969da;
}
.composite-sha-link:hover {
    text-decoration: underline;
}
</style>
<script>
function toggleDockerImages(event, linkElement) {
    event.preventDefault();
    // Find the current row
    var currentRow = linkElement.closest('tr');
    // Find the next row (which should be the docker-images-row)
    var dockerRow = currentRow.nextElementSibling;

    if (dockerRow && dockerRow.classList.contains('docker-images-row')) {
        // Toggle visibility
        if (dockerRow.style.display === 'none') {
            dockerRow.style.display = 'table-row';
        } else {
            dockerRow.style.display = 'none';
        }
    }
}
</script>
</head>
<body>
"""
        # Add generation timestamp in PDT at the top
        from datetime import datetime
        import pytz
        pdt = pytz.timezone('America/Los_Angeles')
        generated_time = datetime.now(pdt).strftime('%Y-%m-%d %H:%M:%S %Z')

        html += f"""
<div class="header">
<h1>Dynamo Commit History</h1>
<p style="margin: 5px 0 0 0; opacity: 0.9;">Recent commits with composite SHAs and Docker images</p>
<p style="margin: 5px 0 0 0; opacity: 0.7; font-size: 12px;">Page generated: {generated_time}</p>
</div>
"""

        html += """
<div class="container">
<table>
<thead>
<tr>
<th style="width: 120px;">Commit SHA</th>
<th style="width: 140px;">Composite SHA</th>
<th style="width: 180px;">Date/Time (PDT)</th>
<th style="width: 150px;">Author</th>
<th>Message</th>
</tr>
</thead>
<tbody>
"""

        for commit in commit_data:
            sha_short = commit['sha_short']
            sha_full = commit['sha_full']
            composite_sha = commit['composite_sha']
            date_str = commit['date']
            author = commit['author']
            message = commit['message']

            # Create GitHub commit link
            commit_link = f"https://github.com/ai-dynamo/dynamo/commit/{sha_full}"

            # Extract PR number and create PR link
            pr_match = re.search(r'\(#(\d+)\)', message)
            if pr_match:
                pr_number = pr_match.group(1)
                pr_link = f"https://github.com/ai-dynamo/dynamo/pull/{pr_number}"
                # Replace (#1234) with clickable link
                message = re.sub(
                    r'\(#(\d+)\)',
                    f'(<a href="{pr_link}" class="pr-link" target="_blank">#{pr_number}</a>)',
                    message
                )

            # Check if Docker images exist for this commit
            has_docker_images = sha_short in docker_images and docker_images[sha_short]

            # Make composite SHA clickable if Docker images exist
            if has_docker_images:
                composite_sha_html = f'<a href="#" class="composite-sha-link" onclick="toggleDockerImages(event, this); return false;">{composite_sha}</a>'
            else:
                composite_sha_html = composite_sha

            html += f"""
<tr>
<td class="sha"><a href="{commit_link}" target="_blank">{sha_short}</a></td>
<td class="sha">{composite_sha_html}</td>
<td class="date">{date_str}</td>
<td class="author">{author}</td>
<td>
<div class="message">{message}</div>
</td>
</tr>
"""

            # Add Docker images row if any exist for this SHA
            if has_docker_images:
                images = docker_images[sha_short]
                html += f"""
<tr class="docker-images-row" style="display: none;">
<td colspan="5" style="padding: 0; background-color: #f6f8fa;">
<div style="padding: 4px 8px; border-top: 1px solid #d0d7de;">
<table style="width: 100%; border: none; background-color: white; border-collapse: collapse;">
<thead>
<tr>
<th style="width: 50%; padding: 3px 6px; border: none;">Image Name:Tag</th>
<th style="width: 15%; padding: 3px 6px; border: none;">Image ID</th>
<th style="width: 10%; padding: 3px 6px; border: none;">Size</th>
<th style="width: 25%; padding: 3px 6px; border: none;">Created (PDT)</th>
</tr>
</thead>
<tbody>
"""
                for image in sorted(images, key=lambda x: x['tag']):
                    html += f"""
<tr>
<td style="font-family: monospace; font-size: 11px; padding: 2px 6px; border: none;">{image['tag']}</td>
<td style="font-family: monospace; font-size: 11px; padding: 2px 6px; border: none;">{image['id']}</td>
<td style="font-size: 11px; padding: 2px 6px; border: none;">{image['size']}</td>
<td style="font-size: 11px; padding: 2px 6px; border: none;">{image['created']}</td>
</tr>
"""
                html += """
</tbody>
</table>
</div>
</td>
</tr>
"""

        html += """
</tbody>
</table>
</div>
</body>
</html>
"""

        return html

    def _get_docker_images_by_sha(self, sha_list: List[str]) -> dict:
        """Get Docker images containing each SHA in their tag

        Args:
            sha_list: List of short SHAs (9 characters)

        Returns:
            Dictionary mapping SHA to list of image details (dicts with tag, id, size, created)
        """
        import subprocess

        sha_to_images = {sha: [] for sha in sha_list}

        try:
            # Get all Docker images with detailed information
            result = subprocess.run(
                ['docker', 'images', '--format', '{{.Repository}}:{{.Tag}}|{{.ID}}|{{.Size}}|{{.CreatedAt}}'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                images = result.stdout.strip().split('\n')

                # Filter images containing each SHA and parse details
                for image_line in images:
                    if not image_line or '<none>:<none>' in image_line:
                        continue

                    parts = image_line.split('|')
                    if len(parts) < 4:
                        continue

                    tag = parts[0]
                    image_id = parts[1]
                    size = parts[2]
                    # Parse created timestamp: "2025-10-18 09:27:29 -0700 PDT"
                    # Extract date and time (first two parts)
                    created_parts = parts[3].split() if parts[3] else []
                    if len(created_parts) >= 2:
                        created = f"{created_parts[0]} {created_parts[1]}"  # Date and time
                    else:
                        created = parts[3].strip() if parts[3] else ''

                    for sha in sha_list:
                        if sha in tag:
                            sha_to_images[sha].append({
                                'tag': tag,
                                'id': image_id,
                                'size': size,
                                'created': created
                            })

        except Exception as e:
            self.logger.warning(f"Failed to get Docker images: {e}")

        return sha_to_images

    def _send_html_email_via_smtp(
        self,
        email: str,
        html_content: str,
        subject_prefix: str,
        git_sha: str,
        failed_tasks: List[str]
    ) -> None:
        """Send HTML email notification via SMTP using curl"""
        try:
            # Determine overall status
            overall_status = "SUCCESS" if not failed_tasks else "FAILURE"

            # Create subject line
            status_prefix = "SUCC" if overall_status == "SUCCESS" else "FAIL"

            # Include failed task names in subject if any
            if failed_tasks:
                failure_summary = ", ".join(failed_tasks)
                subject = f"{status_prefix}: {subject_prefix} - {git_sha} ({failure_summary})"
            else:
                subject = f"{status_prefix}: {subject_prefix} - {git_sha}"

            # Create email file with proper CRLF formatting
            email_file = Path(f"/tmp/dynamo_email_{os.getpid()}.txt")

            # Write email content directly
            email_content = f'Subject: {subject}\r\nFrom: {subject_prefix} <dynamo-docker-builder@nvidia.com>\r\nTo: {email}\r\nMIME-Version: 1.0\r\nContent-Type: text/html; charset=UTF-8\r\n\r\n{html_content}\r\n'

            with open(email_file, 'w', encoding='utf-8') as f:
                f.write(email_content)

            # Send email using curl
            result = subprocess.run([
                'curl', '--url', 'smtp://smtp.nvidia.com:25',
                '--mail-from', 'dynamo-docker-builder@nvidia.com',
                '--mail-rcpt', email,
                '--upload-file', str(email_file)
            ], capture_output=True, text=True)

            # Clean up
            email_file.unlink(missing_ok=True)

            if result.returncode == 0:
                self.logger.info(f"\n📧 Email notification sent to {email}")
                self.logger.info(f"   Subject: {subject}")
                if failed_tasks:
                    self.logger.info(f"   Failed tasks: {', '.join(failed_tasks)}")
            else:
                self.logger.error(f"\n⚠️  Failed to send email: {result.stderr}")

        except Exception as e:
            self.logger.error(f"\n⚠️  Error sending email notification: {e}")

    def _populate_image_sizes(self, pipeline: BuildPipeline) -> None:
        """Populate image sizes for all BUILD tasks by querying Docker.

        This is called after task execution to populate image sizes for both:
        - Successfully built images
        - Skipped images (when using --skip-build-if-image-exists)

        Args:
            pipeline: The build pipeline with tasks to populate
        """
        import docker

        try:
            docker_client = docker.from_env()
        except Exception as e:
            self.logger.warning(f"Could not connect to Docker to get image sizes: {e}")
            return

        # Collect all BUILD tasks that have an image_tag
        build_tasks = [
            task for task in pipeline.tasks.values()
            if task.task_type == TaskType.BUILD and task.image_tag
        ]

        if not build_tasks:
            return

        self.logger.info("")
        self.logger.info("Populating image sizes...")

        for task in build_tasks:
            try:
                img = docker_client.images.get(task.image_tag)
                size_bytes = img.attrs.get('Size', 0)
                size_gb = size_bytes / (1024**3)
                task.image_size = f"{size_gb:.1f} GB"
                self.logger.info(f"  ✅ {task.image_tag}: {task.image_size}")
            except docker.errors.ImageNotFound:
                # Image doesn't exist - this is expected for failed/skipped builds
                pass
            except Exception as e:
                self.logger.warning(f"  ⚠️  Could not get size for {task.image_tag}: {e}")

    def run(self, args: argparse.Namespace) -> int:
        """Main entry point"""
        dry_run = args.dry_run
        verbose = args.verbose
        parallel = args.parallel
        skip_build_if_image_exists = args.skip_build_if_image_exists if hasattr(args, 'skip_build_if_image_exists') else False
        sanity_check_only = args.sanity_check_only if hasattr(args, 'sanity_check_only') else False
        repo_sha = args.repo_sha if hasattr(args, 'repo_sha') else None
        force_run = args.force_run if hasattr(args, 'force_run') else False

        if args.repo_path:
            self.repo_path = Path(args.repo_path).resolve()

        # Handle --show-commit-history flag
        if hasattr(args, 'show_commit_history') and args.show_commit_history:
            max_commits = args.max_commits if hasattr(args, 'max_commits') else 50
            html_output = args.html if hasattr(args, 'html') else False
            return self.show_commit_history(max_commits, verbose=verbose, html_output=html_output)

        # Initialize repository utils
        self.repo_utils = DynamoRepositoryUtils(self.repo_path, dry_run=dry_run, verbose=verbose)

        # Validate mutually exclusive flags
        if args.repo_sha and args.no_checkout:
            print("❌ Error: --repo-sha and --no-checkout are mutually exclusive")
            return 1

        # Get or checkout specific SHA (BEFORE composite SHA check)
        import git
        try:
            repo = git.Repo(self.repo_path)
            if args.no_checkout:
                # Use current HEAD without checkout
                repo_sha = repo.head.commit.hexsha
                print(f"NO-CHECKOUT MODE: Using current HEAD {repo_sha[:9]}")
            elif repo_sha:
                current_sha = repo.head.commit.hexsha[:9]
                if current_sha != repo_sha[:9]:
                    print(f"Checking out SHA: {repo_sha}")
                    repo.git.checkout(repo_sha)
                    print(f"✅ Checked out {repo_sha}")
                    repo_sha = repo.head.commit.hexsha
            else:
                # Checkout main and pull latest
                print("Checking out main branch and pulling latest...")
                repo.git.checkout('main')
                repo.remotes.origin.pull()
                repo_sha = repo.head.commit.hexsha
                print(f"✅ Using latest main: {repo_sha[:9]}")
        except Exception as e:
            print(f"❌ Failed to get/checkout SHA: {e}")
            return 1

        # Check if another instance is running (skip in sanity-check-only mode)
        if not sanity_check_only:
            self.check_if_running(force_run=force_run)

            # Check if rebuild is needed based on composite SHA (AFTER pulling latest)
            if not self.repo_utils.check_if_rebuild_needed(force_run=force_run):
                return 0  # Exit early - no rebuild needed

        # Determine frameworks - support both --framework vllm --framework sglang and --framework vllm,sglang
        if args.framework:
            # Flatten and split comma-separated values
            frameworks = []
            for f in args.framework:
                frameworks.extend([fw.strip().upper() for fw in f.split(',')])
        else:
            frameworks = list(FRAMEWORKS_UPPER)

        # Determine targets
        targets = [t.strip() for t in args.target.split(',')]

        # Print header
        print("=" * 80)
        print(f"DynamoDockerBuilder V2 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Simplified Parallel Build System")
        print("=" * 80)

        if dry_run:
            print("DRY-RUN MODE: Commands will be shown but not executed")
        if sanity_check_only:
            print("SANITY-CHECK-ONLY MODE: Will only run sanity checks, skipping builds and compilation")
        if repo_sha:
            print(f"REPO-SHA MODE: Building for SHA {repo_sha}")
        if skip_build_if_image_exists:
            print("SKIP-BUILD-IF-IMAGE-EXISTS MODE: Will skip building images that already exist")
        if parallel:
            print("PARALLEL MODE: Tasks will execute in parallel when dependencies allow")
        else:
            print("SERIAL MODE: Tasks will execute sequentially (use --parallel for parallel execution)")

        print(f"Dynamo CI Directory: {self.repo_path}")
        print("")

        # Build pipeline
        builder = PipelineBuilder(self.repo_path, frameworks, targets, dry_run, verbose)
        pipeline = builder.build_pipeline()

        # Apply sanity-check-only mode if specified
        if sanity_check_only:
            self._apply_sanity_check_only_mode(pipeline)
        # Otherwise, mark existing images as skipped if --skip-build-if-image-exists
        elif skip_build_if_image_exists:
            self._mark_existing_images_skipped(pipeline)

        # Visualize
        print(pipeline.visualize_tree())

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total tasks: {len(pipeline.tasks)}")
        print("")
        print("By type:")
        for task_type in TaskType:
            count = sum(1 for t in pipeline.tasks.values() if t.task_type == task_type)
            if count > 0:
                print(f"  {task_type.value}: {count}")
        print("")
        print("By status:")
        for status in TaskStatus:
            count = sum(1 for t in pipeline.tasks.values() if t.status == status)
            if count > 0:
                print(f"  {status.value}: {count}")
        print("=" * 80)

        # Setup log directory (matching V1 behavior: logs/<date>/)
        log_dir = None
        if not dry_run:
            date_str = datetime.now().strftime('%Y-%m-%d')
            log_dir = self.repo_path / "logs" / date_str
            log_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nLog directory: {log_dir}")

            # Clean up existing log files ONLY for tasks that will actually run
            sha_prefix = repo_sha[:7] if repo_sha else "unknown"
            removed_files = []
            for task_id, task in pipeline.tasks.items():
                # Only clean up logs for tasks that will run (not skipped)
                if task.status != TaskStatus.SKIPPED:
                    task_id_lower = task_id.lower()
                    log_patterns = [
                        f"{date_str}.{sha_prefix}.{task_id_lower}.log",
                        f"{date_str}.{sha_prefix}.{task_id_lower}.SUCC",
                        f"{date_str}.{sha_prefix}.{task_id_lower}.FAIL"
                    ]
                    for pattern in log_patterns:
                        file_path = log_dir / pattern
                        if file_path.exists() and file_path.is_file():
                            file_path.unlink()
                            removed_files.append(file_path.name)

            if removed_files:
                print(f"Removed {len(removed_files)} existing log file(s) for tasks that will run")
            print("")

        # Execute
        executor = TaskExecutor(dry_run, verbose, log_dir=log_dir, repo_sha=repo_sha, repo_path=self.repo_path)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        successful, failed = loop.run_until_complete(executor.execute_pipeline(pipeline, parallel))
        loop.close()

        # Populate image sizes for all BUILD tasks (for both built and skipped images)
        if not dry_run:
            self._populate_image_sizes(pipeline)

        # Final summary
        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Skipped: {sum(1 for t in pipeline.tasks.values() if t.status == TaskStatus.SKIPPED)}")

        # Generate HTML report with relative paths for file
        html_report_file = None
        html_report_email = None
        if not dry_run and log_dir:
            try:
                # Generate report with relative paths for the HTML file
                html_report_file = ReportGenerator.generate_html_report(
                    pipeline=pipeline,
                    repo_path=self.repo_path,
                    repo_sha=repo_sha,
                    log_dir=log_dir,
                    date_str=date_str,
                    hostname=getattr(args, 'hostname', 'keivenc-linux'),
                    html_path=getattr(args, 'html_path', '/nvidia/dynamo_ci/logs'),
                    use_absolute_urls=False  # Use relative paths for file
                )
                report_file = log_dir / f"{date_str}.{repo_sha[:7]}.report.html"
                report_file.write_text(html_report_file)
                print(f"\n📊 HTML report generated: {report_file}")

                # If email is requested, generate a second version with absolute URLs
                if hasattr(args, 'email') and args.email:
                    html_report_email = ReportGenerator.generate_html_report(
                        pipeline=pipeline,
                        repo_path=self.repo_path,
                        repo_sha=repo_sha,
                        log_dir=log_dir,
                        date_str=date_str,
                        hostname=getattr(args, 'hostname', 'keivenc-linux'),
                        html_path=getattr(args, 'html_path', '/dynamo_ci/logs'),
                        use_absolute_urls=True  # Use absolute URLs for email
                    )
            except Exception as e:
                print(f"\n⚠️  Failed to generate HTML report: {e}")

        # Send email notification if requested
        if hasattr(args, 'email') and args.email and html_report_email:
            # Collect failed task names
            failed_task_names = [
                task_id for task_id, task in pipeline.tasks.items()
                if task.status == TaskStatus.FAILED
            ]

            # Send email with HTML report (using absolute URLs)
            self._send_html_email_via_smtp(
                email=args.email,
                html_content=html_report_email,
                subject_prefix="DynamoDockerBuilder V2",
                git_sha=repo_sha[:7] if repo_sha else "unknown",
                failed_tasks=failed_task_names
            )

        return 0 if failed == 0 else 1


def main():
    """Entry point"""
    parser = argparse.ArgumentParser(
        description="DynamoDockerBuilder V2 - Simplified Build System"
    )
    parser.add_argument("-f", "--framework", action="append",
                        help="Framework to build (can specify multiple)")
    parser.add_argument("--target", default="dev,local-dev",
                        help="Comma-separated targets (default: dev,local-dev)")
    parser.add_argument("--repo-path", type=Path,
                        help="Path to dynamo repository (default: ../dynamo_ci)")
    parser.add_argument("--repo-sha", type=str,
                        help="Git SHA to checkout (default: current HEAD)")
    parser.add_argument("--no-checkout", action="store_true",
                        help="Use current HEAD without checking out (mutually exclusive with --repo-sha)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show commands without executing")
    parser.add_argument("--skip-build-if-image-exists", action="store_true",
                        help="Skip building images that already exist locally")
    parser.add_argument("--sanity-check-only", action="store_true",
                        help="Only run sanity checks, skip all builds and compilation")
    parser.add_argument("--parallel", action="store_true",
                        help="Execute tasks in parallel when possible")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("--force-run", action="store_true",
                        help="Force run even if another instance is running (bypasses lock check)")
    parser.add_argument("--show-commit-history", action="store_true",
                        help="Show recent commit history and exit")
    parser.add_argument("--max-commits", type=int, default=50,
                        help="Maximum number of commits to show in commit history (default: 50)")
    parser.add_argument("--html", action="store_true",
                        help="Generate HTML output for commit history (use with --show-commit-history)")
    parser.add_argument("--email", type=str,
                        help="Email address for notifications (sends email if specified)")
    parser.add_argument("--hostname", type=str, default="keivenc-linux",
                        help="Hostname for log file URLs (default: keivenc-linux)")
    parser.add_argument("--html-path", type=str, default="/nvidia/dynamo_ci/logs",
                        help="Web-accessible path prefix for log files (default: /nvidia/dynamo_ci/logs)")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(message)s'
    )

    builder = DynamoDockerBuilderV2()
    return builder.run(args)


if __name__ == "__main__":
    sys.exit(main())
