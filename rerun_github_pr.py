#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Monitor GitHub Actions workflows and re-run failed jobs automatically.

This script monitors one or more PRs and automatically re-runs failed jobs
if they appear to be infrastructure failures (not code-related errors).
"""

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from common_github import GitHubAPIClient


# Code-related error patterns to detect
CODE_ERROR_PATTERNS = [
    r'SyntaxError',
    r'ImportError',
    r'ModuleNotFoundError',
    r'IndentationError',
    r'NameError',
    r'TypeError',
    r'AttributeError',
    r'pytest.*FAILED',
    r'mypy.*error',
    r'ruff.*error',
    r'black.*error',
    r'isort.*error',
    r'flake8.*error',
    r'pre-commit.*failed',
    r'sphinx.*warning',
    r'sphinx.*error',
    r'broken link',
    r'toctree'
]


@dataclass
class CheckStatus:
    """Status of a single check."""
    name: str
    status: str  # 'pass', 'fail', 'pending', 'skipping'
    duration: str
    url: str
    run_id: Optional[str] = None  # Extracted from URL if available


class PRMonitor:
    """Monitor and manage PR workflow runs."""

    def __init__(self, repo: str = "ai-dynamo/dynamo", token: Optional[str] = None):
        self.repo = repo
        self.owner, self.repo_name = repo.split('/')
        self.github_client = GitHubAPIClient(token=token)
        self.completed_prs: Set[int] = set()
        self.rerun_triggered: Dict[str, bool] = {}  # Track which runs we've already triggered rerun for

    def get_pr_checks(self, pr_number: int) -> List[CheckStatus]:
        """Get all checks for a PR by parsing gh pr checks output."""
        try:
            result = subprocess.run(
                ['gh', 'pr', 'checks', str(pr_number), '--repo', self.repo],
                capture_output=True,
                text=True,
                timeout=10
            )

            checks = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue

                # Parse tab-separated output: name \t status \t duration \t url \t [message]
                parts = line.split('\t')
                if len(parts) < 4:
                    continue

                name = parts[0].strip()
                status = parts[1].strip().lower()
                duration = parts[2].strip()
                url = parts[3].strip()

                # Extract run ID from URL if it's a GitHub Actions URL
                run_id = None
                match = re.search(r'runs/(\d+)', url)
                if match:
                    run_id = match.group(1)

                checks.append(CheckStatus(
                    name=name,
                    status=status,
                    duration=duration,
                    url=url,
                    run_id=run_id
                ))

            return checks
        except Exception as e:
            print(f"  ⚠️  Error getting PR checks: {e}", file=sys.stderr)
            return []

    def get_failed_jobs_for_run(self, run_id: str) -> List[Dict[str, str]]:
        """Get list of failed jobs for a specific workflow run."""
        try:
            result = subprocess.run(
                ['gh', 'run', 'view', run_id, '--repo', self.repo, '--json', 'jobs'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)
                jobs = data.get('jobs', [])
                return [job for job in jobs if job.get('conclusion') == 'failure']
            return []
        except Exception:
            return []

    def check_for_code_errors_in_run(self, run_id: str, failed_jobs: List[Dict[str, str]]) -> bool:
        """Check if failed jobs contain code-related errors."""
        for job in failed_jobs:
            job_name = job.get('name', '')
            job_url = job.get('url', '')

            if not job_url:
                continue

            # Use the GitHub API client to get job error summary
            error_summary = self.github_client.get_job_error_summary(
                run_id, job_url, self.owner, self.repo_name
            )

            if error_summary:
                # Check for code-related error patterns
                for pattern in CODE_ERROR_PATTERNS:
                    if re.search(pattern, error_summary, re.IGNORECASE):
                        print(f"      🚫 Code-related error in {job_name}")
                        return True

        return False

    def rerun_failed_jobs_for_run(self, run_id: str) -> bool:
        """Re-run failed jobs for a specific workflow run."""
        try:
            result = subprocess.run(
                ['gh', 'run', 'rerun', run_id, '--repo', self.repo, '--failed'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False

    def get_run_status(self, run_id: str) -> Dict[str, str]:
        """Get workflow run status and conclusion."""
        try:
            result = subprocess.run(
                ['gh', 'run', 'view', run_id, '--repo', self.repo, '--json', 'status,conclusion'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                return json.loads(result.stdout)
            return {}
        except Exception:
            return {}

    def get_job_elapsed_time(self, run_id: str, job_name: str) -> Optional[str]:
        """Get elapsed time for a job (works for in-progress jobs)."""
        try:
            result = subprocess.run(
                ['gh', 'run', 'view', run_id, '--repo', self.repo, '--json', 'jobs'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)
                jobs = data.get('jobs', [])
                for job in jobs:
                    if job.get('name') == job_name:
                        started_at = job.get('startedAt')
                        completed_at = job.get('completedAt')

                        if not started_at or started_at == '0001-01-01T00:00:00Z':
                            return 'queued'

                        start_time = datetime.fromisoformat(started_at.replace('Z', '+00:00'))

                        if completed_at and completed_at != '0001-01-01T00:00:00Z':
                            end_time = datetime.fromisoformat(completed_at.replace('Z', '+00:00'))
                        else:
                            end_time = datetime.now(timezone.utc)

                        elapsed = end_time - start_time
                        total_seconds = int(elapsed.total_seconds())

                        if total_seconds < 60:
                            return f'{total_seconds}s'
                        else:
                            minutes = total_seconds // 60
                            seconds = total_seconds % 60
                            return f'{minutes}m{seconds}s'

                return None
        except Exception:
            return None

    def monitor_pr(self, pr_number: int) -> bool:
        """
        Monitor a single PR and re-run failed jobs if needed.

        Returns:
            True if PR is completed successfully, False otherwise
        """
        # Skip if already completed
        if pr_number in self.completed_prs:
            return True

        print(f"  PR#{pr_number}:")

        # Get all checks for this PR
        checks = self.get_pr_checks(pr_number)
        if not checks:
            print(f"    ⚠️  Could not retrieve checks")
            return False

        # Count check statuses
        passed = sum(1 for c in checks if c.status == 'pass')
        failed = sum(1 for c in checks if c.status == 'fail')
        pending = sum(1 for c in checks if c.status == 'pending')
        skipped = sum(1 for c in checks if c.status in ('skipping', 'skipped'))
        total = len(checks)

        print(f"    Checks: {passed} passed, {failed} failed, {pending} pending, {skipped} skipped (total: {total})")

        # If there are failed checks, try to rerun them
        if failed > 0:
            print(f"    ❌ Found {failed} failed check(s):")

            # Group failed checks by workflow run
            failed_runs: Dict[str, List[CheckStatus]] = {}
            for check in checks:
                if check.status == 'fail' and check.run_id:
                    if check.run_id not in failed_runs:
                        failed_runs[check.run_id] = []
                    failed_runs[check.run_id].append(check)

            # Also show failed checks without run_id (like GitLab)
            failed_external = [c for c in checks if c.status == 'fail' and not c.run_id]
            if failed_external:
                print(f"      External failures (cannot rerun):")
                for check in failed_external[:5]:
                    print(f"        - {check.name}")

            # Process each failed workflow run
            for run_id, failed_checks in failed_runs.items():
                check_names = ', '.join(c.name for c in failed_checks[:3])
                if len(failed_checks) > 3:
                    check_names += f', ... (+{len(failed_checks) - 3} more)'
                print(f"      Run {run_id}: {check_names}")

                # Check if run is completed
                run_status = self.get_run_status(run_id)
                status = run_status.get('status', 'unknown')

                if status != 'completed':
                    print(f"        ⏳ Run still in progress, waiting...")
                    continue

                # Check if we already triggered a rerun
                if self.rerun_triggered.get(run_id, False):
                    print(f"        ⏳ Re-run already triggered, waiting...")
                    continue

                # Check for code errors
                print(f"        🔍 Checking for code errors...")
                failed_jobs = self.get_failed_jobs_for_run(run_id)
                has_code_errors = self.check_for_code_errors_in_run(run_id, failed_jobs)

                if has_code_errors:
                    print(f"        ⛔ Code errors detected - skipping re-run")
                    print(f"        💡 Fix the code issues before re-running")
                else:
                    print(f"        ✅ No code errors detected")
                    print(f"        🔄 Re-running failed jobs...")

                    if self.rerun_failed_jobs_for_run(run_id):
                        print(f"        ✅ Re-run triggered")
                        self.rerun_triggered[run_id] = True
                    else:
                        print(f"        ❌ Failed to trigger re-run")

        # Show pending checks (even if there are failures)
        if pending > 0:
            print(f"    ⏳ Found {pending} pending check(s):")
            # Show which checks are pending with elapsed time
            pending_check_objs = [c for c in checks if c.status == 'pending'][:5]
            for check in pending_check_objs:
                elapsed = 'unknown'
                if check.run_id:
                    elapsed_time = self.get_job_elapsed_time(check.run_id, check.name)
                    if elapsed_time:
                        elapsed = elapsed_time
                print(f"      {check.name} ({elapsed})")
            if len(pending_check_objs) < pending:
                print(f"      ... (+{pending - len(pending_check_objs)} more)")

        # Determine completion status
        if failed > 0 or pending > 0:
            return False
        else:
            # All checks passed or skipped
            print(f"    ✅ All checks completed successfully")
            self.completed_prs.add(pr_number)
            return True

    def monitor_prs(self, pr_numbers: List[int], max_iterations: int = 10, sleep_seconds: int = 300):
        """
        Monitor multiple PRs and re-run failed jobs as needed.

        Args:
            pr_numbers: List of PR numbers to monitor
            max_iterations: Maximum number of monitoring iterations
            sleep_seconds: Seconds to sleep between iterations
        """
        print(f"=== Monitoring PRs: {pr_numbers} ===")
        print(f"Max iterations: {max_iterations}")
        print(f"Check interval: {sleep_seconds} seconds")
        print()

        for iteration in range(1, max_iterations + 1):
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Iteration {iteration}/{max_iterations}")
            print()

            all_passed = True

            for pr_number in pr_numbers:
                if not self.monitor_pr(pr_number):
                    all_passed = False
                print()

            # Check if all PRs are completed successfully
            if all_passed and len(self.completed_prs) == len(pr_numbers):
                print("🎉 All PRs completed successfully! Monitoring complete.")
                return

            print("---")
            print()

            # Don't sleep on last iteration
            if iteration < max_iterations:
                time.sleep(sleep_seconds)

        print(f"⏰ Reached maximum iterations ({max_iterations})")
        print()
        print("Final status for all PRs:")
        for pr_number in pr_numbers:
            print()
            print(f"PR#{pr_number}:")
            checks = self.get_pr_checks(pr_number)
            if checks:
                passed = sum(1 for c in checks if c.status == 'pass')
                failed = sum(1 for c in checks if c.status == 'fail')
                pending = sum(1 for c in checks if c.status == 'pending')
                print(f"  {passed} passed, {failed} failed, {pending} pending")
                if failed > 0:
                    failed_checks = [c.name for c in checks if c.status == 'fail']
                    for name in failed_checks[:10]:
                        print(f"    ❌ {name}")
            else:
                print(f"  Could not retrieve status")


def main():
    parser = argparse.ArgumentParser(
        description='Monitor GitHub Actions workflows and re-run failed jobs automatically',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor single PR
  %(prog)s 3688

  # Monitor multiple PRs
  %(prog)s 3688 3689 3690

  # Monitor with custom settings
  %(prog)s 3688 3689 --max-iterations 20 --sleep 600

  # Monitor with custom repository
  %(prog)s 3688 --repo owner/repo
"""
    )

    parser.add_argument(
        'pr_numbers',
        type=int,
        nargs='+',
        help='PR numbers to monitor'
    )

    parser.add_argument(
        '--repo',
        default='ai-dynamo/dynamo',
        help='Repository in format owner/repo (default: ai-dynamo/dynamo)'
    )

    parser.add_argument(
        '--max-iterations',
        type=int,
        default=10,
        help='Maximum number of monitoring iterations (default: 10)'
    )

    parser.add_argument(
        '--sleep',
        type=int,
        default=300,
        help='Seconds to sleep between iterations (default: 300)'
    )

    parser.add_argument(
        '--token',
        help='GitHub token (optional, will use GITHUB_TOKEN env or gh CLI config)'
    )

    args = parser.parse_args()

    # Create monitor and run
    monitor = PRMonitor(repo=args.repo, token=args.token)
    monitor.monitor_prs(args.pr_numbers, args.max_iterations, args.sleep)


if __name__ == '__main__':
    main()
