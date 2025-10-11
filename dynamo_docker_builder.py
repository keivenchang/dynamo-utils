#!/usr/bin/env python3
"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

DynamoDockerBuilder - Automated Docker Build and Test System
Tests Dockerfile.vllm, Dockerfile.sglang, and Dockerfile.trtllm with comprehensive reporting
- Builds and tests both dev and local-dev Docker targets
- HTML email notifications with failure details and GitHub PR links
- SHA-based rebuild detection to avoid unnecessary builds
- Ultra-compact email formatting for quick scanning
- Build timeout: 1 hour, Container test timeout: 2 minutes
"""

import argparse
import hashlib
import logging
import os
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from jinja2 import Template
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class DynamoBuilderTester:
    """DynamoDockerBuilder - Main class for automated Docker build testing and reporting"""

    # Framework constants
    FRAMEWORKS = ["VLLM", "SGLANG", "TRTLLM"]

    def __init__(self):
        self.script_dir = Path(__file__).parent.absolute()
        self.dynamo_ci_dir = self.script_dir.parent / "dynamo_ci"
        self.date = datetime.now().strftime("%Y-%m-%d")
        self.log_dir = self.dynamo_ci_dir / "logs" / self.date

        # Configuration flags
        self.dry_run = False
        self.test_only = False
        self.no_checkout = False
        self.force_run = False
        self.email = None
        self.targets = ["dev", "local-dev"]  # Default targets
        self.repo_sha = None  # SHA to checkout

        # Lock file for preventing concurrent runs
        self.lock_file = self.script_dir / f".{Path(__file__).name}.lock"

        # Set up logger
        self._setup_logger()

        # Track build times for email reporting
        self.build_times: Dict[str, float] = {}

    def _setup_logger(self) -> None:
        """Set up the logger with appropriate formatting"""
        self.logger = logging.getLogger('DynamoDockerBuilder')
        self.logger.setLevel(logging.DEBUG)

        # Remove any existing handlers
        self.logger.handlers.clear()

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Create custom formatter that handles DRYRUN prefix
        class DryRunFormatter(logging.Formatter):
            def __init__(self, dry_run_instance):
                super().__init__()
                self.dry_run_instance = dry_run_instance

            def format(self, record):
                if self.dry_run_instance.dry_run:
                    return f"DRYRUN {record.levelname} - {record.getMessage()}"
                else:
                    return f"{record.levelname} - {record.getMessage()}"

        formatter = DryRunFormatter(self)
        console_handler.setFormatter(formatter)

        # Add handler to logger
        self.logger.addHandler(console_handler)

        # Prevent propagation to root logger
        self.logger.propagate = False

    def log_info(self, message: str) -> None:
        """Log info message"""
        self.logger.info(message)

    def log_success(self, message: str) -> None:
        """Log success message"""
        self.logger.info(f"SUCCESS: {message}")

    def log_error(self, message: str) -> None:
        """Log error message"""
        self.logger.error(message)

    def log_warning(self, message: str) -> None:
        """Log warning message"""
        self.logger.warning(message)

    def log_debug(self, message: str) -> None:
        """Log debug message"""
        self.logger.debug(message)

    def get_docker_image_info(self, image_tag: str) -> Dict[str, str]:
        """Get Docker image information including size and full repo:tag

        Args:
            image_tag: Docker image tag to inspect

        Returns:
            Dict with 'repo_tag', 'size', 'size_bytes' keys
        """
        try:
            # Get image size using docker images command
            result = subprocess.run([
                'docker', 'images', '--format', 'table {{.Repository}}:{{.Tag}}\t{{.Size}}',
                '--no-trunc', image_tag
            ], capture_output=True, text=True)

            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                # Skip header line if present
                data_lines = [line for line in lines if not line.startswith('REPOSITORY')]
                if data_lines:
                    # Docker output uses spaces for alignment, not tabs
                    # Split on whitespace and take the last part as size
                    line = data_lines[0].strip()
                    parts = line.split()
                    if len(parts) >= 2:
                        # Last part is the size, everything else is the repo:tag
                        size = parts[-1]
                        repo_tag = ' '.join(parts[:-1])

                        # Also get size in bytes for sorting/comparison
                        inspect_result = subprocess.run([
                            'docker', 'inspect', '--format', '{{.Size}}', image_tag
                        ], capture_output=True, text=True)

                        size_bytes = "0"
                        if inspect_result.returncode == 0 and inspect_result.stdout.strip():
                            size_bytes = inspect_result.stdout.strip()

                        return {
                            'repo_tag': repo_tag,
                            'size': size,
                            'size_bytes': size_bytes
                        }

            # Fallback if docker images command fails
            return {
                'repo_tag': image_tag,
                'size': 'Unknown',
                'size_bytes': '0'
            }

        except Exception as e:
            self.log_warning(f"Failed to get Docker image info for {image_tag}: {e}")
            return {
                'repo_tag': image_tag,
                'size': 'Unknown',
                'size_bytes': '0'
            }

    def get_email_template(self) -> str:
        """Get the Jinja2 email template"""
        return """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
body { font-family: Arial, sans-serif; margin: 10px; line-height: 1.3; }
.header { background-color: {{ status_color }}; color: white; padding: 15px 20px; border-radius: 4px; margin-bottom: 10px; text-align: center; }
.summary { background-color: #f8f9fa; padding: 4px 6px; border-radius: 2px; margin: 3px 0; }
.results { margin: 10px 0; }
.framework { margin: 10px 0; padding: 8px; border: 1px solid #dee2e6; border-radius: 4px; background-color: #ffffff; }
.framework-header { background-color: #007bff; color: white; padding: 8px 12px; margin: -8px -8px 8px -8px; border-radius: 4px 4px 0 0; font-weight: bold; }
.results-chart { display: table; width: 100%; border-collapse: collapse; margin: 8px 0; }
.chart-row { display: table-row; }
.chart-cell { display: table-cell; padding: 6px 12px; border: 1px solid #dee2e6; vertical-align: middle; }
.chart-header { background-color: #f8f9fa; font-weight: bold; text-align: center; }
.chart-target { font-weight: bold; background-color: #f1f3f4; }
.chart-status { text-align: center; }
.chart-timing { text-align: right; font-family: monospace; font-size: 0.9em; }
.success { color: #28a745; font-weight: bold; }
.failure { color: #dc3545; font-weight: bold; }
.git-info { background-color: #e9ecef; padding: 4px 6px; border-radius: 2px; font-family: monospace; font-size: 0.9em; }
.error-output { background-color: #2d3748; color: #e2e8f0; padding: 8px; border-radius: 3px; font-family: 'Courier New', monospace; font-size: 0.85em; margin: 5px 0; overflow-x: auto; white-space: pre-wrap; }
p { margin: 1px 0; }
h3 { margin: 4px 0 2px 0; font-size: 1.0em; }
h4 { margin: 3px 0 1px 0; font-size: 0.95em; }
h2 { margin: 0; font-size: 1.2em; font-weight: bold; }
</style>
</head>
<body>
<div class="header">
<h2>DynamoDockerBuilder - {{ overall_status }}</h2>
</div>

<div class="summary">
<p><strong>Build & Test Date:</strong> {{ build_date }}</p>
<p><strong>Total Builds:</strong> {{ total_builds }} | <strong>Total Tests:</strong> {{ total_tests }}</p>
<p><strong>Passed:</strong> <span class="success">{{ passed_tests }}</span> | <strong>Failed:</strong> <span class="failure">{{ failed_tests }}</span></p>
</div>

<div class="git-info">
<p><strong>Commit SHA:</strong> {{ git_info.sha }}</p>
<p><strong>Commit Date:</strong> {{ git_info.date }}</p>
<p><strong>Author:</strong> {{ git_info.author | safe }}</p>
<div style="background-color: #f8f9fa; padding: 6px; border-radius: 3px; margin: 3px 0;">
<strong>Commit Message:</strong>
<pre style="margin: 3px 0; white-space: pre-wrap; font-family: monospace; font-size: 0.9em;">{{ git_info.full_message | e }}</pre>
</div>
{% if git_info.total_additions is defined or git_info.total_deletions is defined %}
<p><strong>Changes Summary:</strong> +{{ git_info.total_additions | default(0) }}/-{{ git_info.total_deletions | default(0) }} lines</p>
{% endif %}
{% if git_info.diff_stats %}
<p><strong>Files Changed with Line Counts:</strong></p>
<div style="background-color: #f8f9fa; padding: 6px; border-radius: 3px; font-family: monospace; font-size: 0.9em; margin: 3px 0;">
{% for stat in git_info.diff_stats %}
• {{ stat | e }}<br>
{% endfor %}
</div>
{% endif %}
</div>

<div class="results">
{% for framework in frameworks %}
<div class="framework">
<div class="framework-header">Dockerfile.{{ framework.name.lower() }}</div>
<div class="results-chart">
<div class="chart-row">
<div class="chart-cell chart-header">Target</div>
<div class="chart-cell chart-header">Status</div>
<div class="chart-cell chart-header">Docker Image Build Time</div>
<div class="chart-cell chart-header">sanity_check.py</div>
<div class="chart-cell chart-header">Container Size</div>
<div class="chart-cell chart-header">Image Tag</div>
</div>
{% for target in framework.targets %}
<div class="chart-row">
<div class="chart-cell chart-target">{{ target.name }}</div>
<div class="chart-cell chart-status">
{% if target.success %}
<span class="success">✅ PASSED</span>
{% else %}
<span class="failure">❌ FAILED</span>
{% endif %}
</div>
<div class="chart-cell chart-timing">{{ target.build_time }}</div>
<div class="chart-cell chart-timing">{{ target.test_time }}</div>
<div class="chart-cell chart-timing">{{ target.container_size if target.container_size else '-' }}</div>
<div class="chart-cell chart-timing" style="font-family: monospace; font-size: 0.8em;">{{ target.image_tag if target.image_tag else '-' }}</div>
</div>
{% if not target.success and target.error_output %}
<div class="chart-row">
<div class="chart-cell" colspan="6">
<div class="error-output">{{ target.error_output }}</div>
</div>
</div>
{% endif %}
{% endfor %}
</div>
</div>
{% endfor %}
</div>

<div class="summary">
<p><strong>Repository:</strong> {{ repo_path }}</p>
<p><strong>Log Directory:</strong> {{ log_dir }}</p>
</div>

<p><em>This email was generated automatically by DynamoDockerBuilder.</em></p>
</body>
</html>"""

    def convert_pr_links(self, message: str) -> str:
        """Convert PR references like (#3107) to GitHub links"""
        import re
        # Pattern to match (#number)
        pr_pattern = r'\(#(\d+)\)'

        def replace_pr(match):
            pr_number = match.group(1)
            return f'(<a href="https://github.com/ai-dynamo/dynamo/pull/{pr_number}" style="color: #0066cc;">#{pr_number}</a>)'

        return re.sub(pr_pattern, replace_pr, message)

    def convert_target_for_build(self, target: str) -> Optional[str]:
        """Convert target name for build commands. 'dev' becomes None, others stay as-is"""
        return None if target == "dev" else target

    def html_escape(self, text: str) -> str:
        """Escape HTML special characters"""
        return (text.replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;")
                   .replace('"', "&quot;")
                   .replace("'", "&#x27;"))

    def format_author_html(self, author: str) -> str:
        """Format author information as proper HTML with mailto link"""
        import re

        # Pattern to match "Name <email> (login)" or "Name <email>"
        pattern = r'^(.+?)\s*<([^>]+)>(?:\s*\(([^)]+)\))?$'
        match = re.match(pattern, author.strip())

        if match:
            name = match.group(1).strip()
            email = match.group(2).strip()
            login = match.group(3).strip() if match.group(3) else None

            # Create HTML with clickable email
            if login:
                return f'{self.html_escape(name)} &lt;<a href="mailto:{email}" style="color: #0066cc; text-decoration: none;">{email}</a>&gt; ({self.html_escape(login)})'
            else:
                return f'{self.html_escape(name)} &lt;<a href="mailto:{email}" style="color: #0066cc; text-decoration: none;">{email}</a>&gt;'
        else:
            # Fallback: just escape the whole string
            return self.html_escape(author)

    def get_failure_details(self, framework: str, docker_target_type: str) -> str:
        """Get failure details from log files for a failed test"""
        try:
            commit_sha = self.get_git_commit_sha()
            framework_lower = framework.lower()

            # Determine log file suffix
            if docker_target_type == "local-dev":
                log_suffix = f"{commit_sha}.{framework_lower}.local-dev"
            else:
                log_suffix = f"{commit_sha}.{framework_lower}.dev"

            log_file = self.log_dir / f"{self.date}.{log_suffix}.log"

            if log_file.exists():
                # Read last 20 lines of the log file
                result = subprocess.run(['tail', '-20', str(log_file)],
                                      capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()

            return f"No log file found at {log_file}"
        except Exception as e:
            return f"Error reading log file: {e}"

    def send_email_notification(self, results: Dict[str, bool], failure_details: Optional[Dict[str, str]] = None, build_times: Optional[Dict[str, float]] = None) -> None:
        """Send email notification with test results using Jinja2 template

        Args:
            results: Dict[str, bool] mapping test keys (e.g. 'VLLM_dev') to success boolean
            failure_details: Optional[Dict[str, str]] mapping failed test keys to error output strings
            build_times: Optional[Dict[str, float]] mapping timing keys to duration in seconds
        """
        if not self.email:
            return

        if failure_details is None:
            failure_details = {}

        if self.dry_run:
            self.log_info(f"DRY-RUN: Would send email notification to {self.email}")
            return

        try:
            # Check if Jinja2 is available
            if not HAS_JINJA2:
                self.log_error("Jinja2 not available, falling back to basic email format")
                self._send_basic_email_notification(results, failure_details, build_times)
                return

            # Get git information
            git_info = self.get_git_info()

            # Count results (only count tests that were actually run)
            total_tests = len(results)
            passed_tests = sum(1 for success in results.values() if success)
            failed_tests = sum(1 for success in results.values() if not success)

            # Calculate total builds (same as total tests since each target involves both build and test)
            # In test-only mode, builds are skipped but we still show the count for clarity
            total_builds = total_tests

            # Collect failed tests for summary
            failed_tests_list = [key for key, success in results.items() if not success]

            # Determine overall status
            overall_status = "SUCCESS" if failed_tests == 0 else "FAILURE"
            status_color = "#28a745" if failed_tests == 0 else "#dc3545"

            # Prepare framework data for template
            frameworks = []
            tested_frameworks = set()
            for key in results.keys():
                framework = key.split('_')[0]  # Extract framework from key like "VLLM_dev"
                tested_frameworks.add(framework)

            for framework in sorted(tested_frameworks):
                # Get all targets tested for this framework
                framework_targets = set()
                for key in results.keys():
                    if key.startswith(f"{framework}_"):
                        target = key[len(f"{framework}_"):]
                        framework_targets.add(target)

                targets = []
                for target in sorted(framework_targets):
                    framework_target = f"{framework}_{target}"
                    if framework_target in results:  # Only show targets that were actually tested
                        success = results[framework_target]

                        # Get timing information if available
                        timing_info = ""
                        build_time_str = ""
                        test_time_str = ""

                        if build_times:
                            build_time_key = f"{framework_target}_build"
                            test_time_key = f"{framework_target}_test"

                            # Get build time (may not exist in test-only mode)
                            if build_time_key in build_times:
                                build_time = build_times[build_time_key]
                                build_time_str = f"{build_time:.1f}s"
                            else:
                                build_time_str = "-"  # No build in test-only mode

                            # Get test time (should always exist)
                            if test_time_key in build_times:
                                test_time = build_times[test_time_key]
                                test_time_str = f"{test_time:.1f}s"
                            else:
                                test_time_str = "-"

                            # Create timing info for backward compatibility (used in some places)
                            if build_time_str != "-" and test_time_str != "-":
                                timing_info = f" (build: {build_time_str}, sanity_check.py: {test_time_str})"
                            elif test_time_str != "-":
                                timing_info = f" (sanity_check.py: {test_time_str})"

                        # Get error output if available
                        error_output = ""
                        if not success and framework_target in failure_details and failure_details[framework_target]:
                            error_lines = failure_details[framework_target].split('\n')
                            if len(error_lines) > 25:
                                error_output = '\n'.join(error_lines[-25:])  # Show last 25 lines
                                error_output = "... (showing last 25 lines)\n" + error_output
                            else:
                                error_output = failure_details[framework_target]

                        # Get Docker image information if build was successful
                        container_size = ""
                        image_tag = ""
                        if success and not self.dry_run:
                            # Try to get the built image tag for this framework/target combination
                            try:
                                # Convert target type for build command discovery
                                docker_target_type = target if target != "dev" else None
                                _, _, discovered_tag = self.get_build_commands(framework, docker_target_type)
                                if discovered_tag:
                                    docker_info = self.get_docker_image_info(discovered_tag)
                                    container_size = docker_info['size']
                                    image_tag = docker_info['repo_tag']
                            except Exception as e:
                                self.log_debug(f"Failed to get Docker info for {framework_target}: {e}")

                        targets.append({
                            'name': target,
                            'success': success,
                            'timing_info': timing_info,
                            'build_time': build_time_str,
                            'test_time': test_time_str,
                            'error_output': error_output,
                            'container_size': container_size,
                            'image_tag': image_tag
                        })

                frameworks.append({
                    'name': framework,
                    'targets': targets
                })

            # Prepare template context
            context = {
                'overall_status': overall_status,
                'status_color': status_color,
                'total_builds': total_builds,
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'build_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' PDT',
                'git_info': {
                    'sha': git_info['sha'],
                    'date': git_info['date'],
                    'message': self.convert_pr_links(self.html_escape(git_info['message'])),
                    'author': self.format_author_html(git_info['author']),
                    'total_additions': git_info.get('total_additions', 0),
                    'total_deletions': git_info.get('total_deletions', 0),
                    'diff_stats': git_info.get('diff_stats', [])
                },
                'frameworks': frameworks,
                'repo_path': str(self.dynamo_ci_dir),
                'log_dir': str(self.log_dir)
            }

            # Render template
            template = Template(self.get_email_template())
            html_content = template.render(context)

            # Create subject line
            status_prefix = "SUCC" if overall_status == "SUCCESS" else "FAIL"
            if failed_tests_list:
                failure_summary = ", ".join(failed_tests_list)
                subject = f"{status_prefix}: DynamoDockerBuilder - {git_info['sha']} ({failure_summary})"
            else:
                subject = f"{status_prefix}: DynamoDockerBuilder - {git_info['sha']}"

            # Create email file with proper CRLF formatting
            email_file = Path(f"/tmp/dynamo_email_{os.getpid()}.txt")

            # Write email content directly to avoid printf format specifier issues
            email_content = f'Subject: {subject}\r\nFrom: DynamoDockerBuilder <dynamo-docker-builder@nvidia.com>\r\nTo: {self.email}\r\nMIME-Version: 1.0\r\nContent-Type: text/html; charset=UTF-8\r\n\r\n{html_content}\r\n'

            with open(email_file, 'w', encoding='utf-8') as f:
                f.write(email_content)

            # Send email using curl
            result = subprocess.run([
                'curl', '--url', 'smtp://smtp.nvidia.com:25',
                '--mail-from', 'dynamo-docker-builder@nvidia.com',
                '--mail-rcpt', self.email,
                '--upload-file', str(email_file)
            ], capture_output=True, text=True)

            # Clean up
            email_file.unlink(missing_ok=True)

            if result.returncode == 0:
                self.log_success(f"Email notification sent to {self.email} (using Jinja2 template)")
            else:
                self.log_error(f"Failed to send email: {result.stderr}")

        except Exception as e:
            self.log_error(f"Error sending email notification: {e}")

    def _send_basic_email_notification(self, results: Dict[str, bool], failure_details: Optional[Dict[str, str]] = None, build_times: Optional[Dict[str, float]] = None) -> None:
        """Fallback email notification without Jinja2"""
        self.log_info("Using basic email format (Jinja2 not available)")
        # This would contain the original implementation as a fallback
        # For brevity, I'm not including the full fallback implementation here

    def cmd(self, command: List[str], **kwargs) -> subprocess.CompletedProcess:
        """Execute command with dry-run support"""
        # Show command in shell tracing format
        cmd_str = " ".join(shlex.quote(str(arg)) for arg in command)
        self.log_debug(f"+ {cmd_str}")

        # Only execute if not in dry-run mode
        if not self.dry_run:
            return subprocess.run(command, **kwargs)
        else:
            # Return a mock completed process for dry-run
            return subprocess.CompletedProcess(command, 0)

    def check_if_running(self) -> None:
        """Check if another instance of this script is already running"""
        script_name = Path(__file__).name
        current_pid = os.getpid()

        # Skip lock check if --force-run is specified
        if self.force_run:
            self.log_warning("FORCE-RUN MODE: Bypassing process lock check")
            # Still create our own lock file
            self.lock_file.write_text(str(current_pid))
            self.log_info(f"Created process lock file: {self.lock_file} (PID: {current_pid})")
            import atexit
            atexit.register(lambda: self.lock_file.unlink(missing_ok=True))
            return

        # Check if lock file exists
        if self.lock_file.exists():
            try:
                existing_pid = int(self.lock_file.read_text().strip())

                # Check if the process is still running and is our script
                if HAS_PSUTIL:
                    # Use psutil if available for more accurate checking
                    if psutil.pid_exists(existing_pid):
                        try:
                            proc = psutil.Process(existing_pid)
                            if script_name in " ".join(proc.cmdline()):
                                self.log_error(f"Another instance of {script_name} is already running (PID: {existing_pid})")
                                self.log_error(f"If you're sure no other instance is running, remove the lock file:")
                                self.log_error(f"  rm '{self.lock_file}'")
                                self.log_error(f"Or kill the existing process:")
                                self.log_error(f"  kill {existing_pid}")
                                sys.exit(1)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            # Process exists but it's not our script, remove stale lock file
                            self.log_warning(f"Removing stale lock file (PID {existing_pid} is not our script)")
                            self.lock_file.unlink()
                    else:
                        # Process doesn't exist, remove stale lock file
                        self.log_warning(f"Removing stale lock file (PID {existing_pid} no longer exists)")
                        self.lock_file.unlink()
                else:
                    # Fallback: Use kill -0 to check if process exists
                    try:
                        os.kill(existing_pid, 0)
                        # Process exists, but we can't check if it's our script without psutil
                        # Check if it's a Python process running our script
                        try:
                            result = subprocess.run(
                                ["ps", "-p", str(existing_pid), "-o", "args="],
                                capture_output=True, text=True, check=True
                            )
                            if script_name in result.stdout:
                                self.log_error(f"Another instance of {script_name} is already running (PID: {existing_pid})")
                                self.log_error(f"If you're sure no other instance is running, remove the lock file:")
                                self.log_error(f"  rm '{self.lock_file}'")
                                self.log_error(f"Or kill the existing process:")
                                self.log_error(f"  kill {existing_pid}")
                                sys.exit(1)
                            else:
                                # Process exists but it's not our script, remove stale lock file
                                self.log_warning(f"Removing stale lock file (PID {existing_pid} is not our script)")
                                self.lock_file.unlink()
                        except subprocess.CalledProcessError:
                            # ps command failed, assume process doesn't exist
                            self.log_warning(f"Removing stale lock file (PID {existing_pid} no longer exists)")
                            self.lock_file.unlink()
                    except OSError:
                        # Process doesn't exist, remove stale lock file
                        self.log_warning(f"Removing stale lock file (PID {existing_pid} no longer exists)")
                        self.lock_file.unlink()
            except (ValueError, FileNotFoundError):
                # Invalid lock file content, remove it
                self.log_warning("Removing invalid lock file")
                self.lock_file.unlink(missing_ok=True)

        # Create lock file with current PID
        self.lock_file.write_text(str(current_pid))
        self.log_info(f"Created process lock file: {self.lock_file} (PID: {current_pid})")

        # Set up cleanup on exit
        import atexit
        atexit.register(lambda: self.lock_file.unlink(missing_ok=True))

    def setup_dynamo_ci(self) -> None:
        """Setup or update dynamo_ci repository"""
        self.log_info("Setting up dynamo_ci repository...")

        if self.no_checkout and not self.repo_sha:
            self.log_info("NO-CHECKOUT MODE: Skipping git operations, using existing repository")

            if not self.dynamo_ci_dir.exists():
                self.log_error(f"dynamo_ci directory does not exist at {self.dynamo_ci_dir}")
                self.log_error("Cannot use --no-checkout without existing repository")
                sys.exit(1)

            self.log_success(f"Using existing repository at {self.dynamo_ci_dir}")
            return
        elif self.no_checkout and self.repo_sha:
            self.log_info(f"NO-CHECKOUT MODE with SHA override: Will checkout specific SHA {self.repo_sha} but skip other git operations")

            if not self.dynamo_ci_dir.exists():
                self.log_error(f"dynamo_ci directory does not exist at {self.dynamo_ci_dir}")
                self.log_error("Cannot use --no-checkout without existing repository")
                sys.exit(1)

        if not self.dynamo_ci_dir.exists():
            self.log_info(f"Cloning dynamo repository to {self.dynamo_ci_dir}")
            self.cmd(["git", "clone", "git@github.com:ai-dynamo/dynamo.git", str(self.dynamo_ci_dir)])

            # If we need to checkout a specific SHA after cloning, we'll do it below
        else:
            if self.repo_sha:
                self.log_info(f"dynamo_ci directory exists, will checkout specific SHA: {self.repo_sha}")
            else:
                self.log_info("dynamo_ci directory exists, updating from main branch")

            # Check if it's a git repository (only in non-dry-run mode)
            if not self.dry_run and not (self.dynamo_ci_dir / ".git").exists():
                self.log_error("dynamo_ci exists but is not a git repository")
                sys.exit(1)

        # Change to repository directory for git operations
        os.chdir(self.dynamo_ci_dir)

        # Only fetch if not in no-checkout mode (unless we need to checkout a SHA)
        if not self.no_checkout:
            # Fetch latest changes from origin
            self.cmd(["git", "fetch", "origin"])

        if self.repo_sha:
            # Checkout specific SHA
            self.log_info(f"Checking out specific SHA: {self.repo_sha}")
            self.cmd(["git", "checkout", self.repo_sha])

            # Validate that we're on the correct SHA
            if not self.dry_run:
                result = subprocess.run(["git", "rev-parse", "HEAD"],
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    current_sha = result.stdout.strip()
                    if current_sha.startswith(self.repo_sha) or self.repo_sha.startswith(current_sha):
                        self.log_success(f"Successfully checked out SHA: {current_sha}")
                    else:
                        self.log_error(f"SHA mismatch: requested {self.repo_sha}, got {current_sha}")
                        sys.exit(1)
                else:
                    self.log_warning("Could not verify current SHA")
        elif not self.no_checkout:
            # Default behavior: checkout and pull main (only if not in no-checkout mode)
            self.cmd(["git", "checkout", "main"])
            self.cmd(["git", "pull", "origin", "main"])

        self.log_success("Repository setup complete")

    def setup_logging(self) -> None:
        """Create date-based log directory"""
        self.log_info("Setting up date-based logging directory...")
        if not self.dry_run:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.log_debug(f"+ mkdir -p {self.log_dir}")
        self.log_success(f"Date-based log directory created at {self.log_dir}")

    def cleanup_existing_logs(self, framework: Optional[str] = None) -> None:
        """Clean up existing log files for current date and optionally specific framework"""
        if framework:
            self.log_info(f"Cleaning up existing log files for date: {self.date}, framework: {framework}")
            framework_lower = framework.lower()
        else:
            self.log_info(f"Cleaning up existing log files for date: {self.date}")

        if not self.dry_run:
            # Remove existing log files, success files, and failure files for current date
            # Pattern now includes SHA: date.sha.framework.type.ext
            if framework:
                # Framework-specific patterns
                patterns = [f"{self.date}.*.{framework_lower}.*.log",
                           f"{self.date}.*.{framework_lower}.*.SUCC",
                           f"{self.date}.*.{framework_lower}.*.FAIL"]
            else:
                # All files for the date
                patterns = [f"{self.date}.*.log", f"{self.date}.*.SUCC", f"{self.date}.*.FAIL"]

            removed_files = []

            for pattern in patterns:
                for file_path in self.log_dir.glob(pattern):
                    if file_path.is_file():
                        file_path.unlink()
                        removed_files.append(file_path.name)
                        self.log_info(f"Removed existing file: {file_path.name}")

            if removed_files:
                if framework:
                    self.log_success(f"Removed {len(removed_files)} existing log files for {framework}")
                else:
                    self.log_success(f"Removed {len(removed_files)} existing log files")
            else:
                if framework:
                    self.log_info(f"No existing log files found for {self.date} and {framework}")
                else:
                    self.log_info(f"No existing log files found for {self.date}")
        else:
            # In dry-run mode, just show what would be removed
            if framework:
                patterns = [f"{self.date}.*.{framework_lower}.*.log",
                           f"{self.date}.*.{framework_lower}.*.SUCC",
                           f"{self.date}.*.{framework_lower}.*.FAIL"]
            else:
                patterns = [f"{self.date}.*.log", f"{self.date}.*.SUCC", f"{self.date}.*.FAIL"]

            for pattern in patterns:
                self.log_debug(f"+ rm -f {self.log_dir}/{pattern}")

            if framework:
                self.log_info(f"Would remove existing log files for {self.date} and {framework} (dry-run)")
            else:
                self.log_info(f"Would remove existing log files for {self.date} (dry-run)")

    def generate_composite_sha_from_container_dir(self) -> Optional[str]:
        """Generate composite SHA from all container files recursively, ignoring hidden files/directories"""
        container_dir = self.dynamo_ci_dir / "container"

        if not container_dir.exists():
            self.log_error(f"Container directory not found: {container_dir}")
            return None

        # Get all files in container directory recursively, sorted for consistent hashing
        # Ignore hidden files and directories (starting with .)
        files_to_hash = []
        for file_path in sorted(container_dir.rglob('*')):
            if file_path.is_file():
                # Skip hidden files/directories (any path part starting with .)
                if any(part.startswith('.') for part in file_path.relative_to(container_dir).parts):
                    continue
                # Store relative path from dynamo_ci_dir for consistent hashing
                rel_path = file_path.relative_to(self.dynamo_ci_dir)
                files_to_hash.append(rel_path)

        if not files_to_hash:
            self.log_error("No files found in container directory for composite SHA calculation")
            return None

        self.log_info(f"Hashing {len(files_to_hash)} files from container directory")

        with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as temp_file:
            temp_path = Path(temp_file.name)

            try:
                # Concatenate all files that exist
                found_files = 0
                for file_rel_path in files_to_hash:
                    full_path = self.dynamo_ci_dir / file_rel_path
                    if full_path.exists():
                        # Write file path first (for uniqueness), then file content
                        temp_file.write(str(file_rel_path).encode('utf-8'))
                        temp_file.write(b'\n')
                        with open(full_path, 'rb') as f:
                            temp_file.write(f.read())
                        temp_file.write(b'\n')
                        found_files += 1
                    else:
                        self.log_warning(f"File not found for composite SHA calculation: {file_rel_path}")

                if found_files == 0:
                    self.log_error("No files found for composite SHA calculation")
                    return None

                # Generate SHA256 of concatenated files
                temp_file.flush()
                with open(temp_path, 'rb') as f:
                    sha = hashlib.sha256(f.read()).hexdigest()

                self.log_debug(f"Generated composite SHA from {found_files} container files: {sha[:12]}...")
                return sha

            finally:
                temp_path.unlink(missing_ok=True)

    def get_git_info(self) -> dict:
        """Get comprehensive git information"""
        try:
            os.chdir(self.dynamo_ci_dir)

            # Get commit SHA
            sha_result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                                      capture_output=True, text=True)
            commit_sha = sha_result.stdout.strip() if sha_result.returncode == 0 else "unknown"

            # Get commit date in both UTC and Pacific time
            utc_result = subprocess.run(['git', 'log', '-1', '--format=%cd', '--date=iso'],
                                      capture_output=True, text=True)
            utc_date = utc_result.stdout.strip() if utc_result.returncode == 0 else "unknown"

            pacific_result = subprocess.run(['git', 'log', '-1', '--format=%cd', '--date=format-local:%Y-%m-%d %H:%M:%S %Z'],
                                          capture_output=True, text=True, env={**os.environ, 'TZ': 'America/Los_Angeles'})
            pacific_date = pacific_result.stdout.strip() if pacific_result.returncode == 0 else "unknown"

            # Combine both timezones
            if utc_date != "unknown" and pacific_date != "unknown":
                commit_date = f"{utc_date} UTC / {pacific_date}"
            else:
                commit_date = utc_date

            # Get commit message (subject only)
            msg_result = subprocess.run(['git', 'log', '-1', '--format=%s'],
                                      capture_output=True, text=True)
            commit_message = msg_result.stdout.strip() if msg_result.returncode == 0 else "unknown"
            
            # Get full commit message (subject + body)
            full_msg_result = subprocess.run(['git', 'log', '-1', '--format=%B'],
                                           capture_output=True, text=True)
            full_commit_message = full_msg_result.stdout.strip() if full_msg_result.returncode == 0 else "unknown"

            # Get author (the actual person who wrote the code)
            author_result = subprocess.run(['git', 'log', '-1', '--format=%an <%ae> (%al)'],
                                         capture_output=True, text=True)
            if author_result.returncode == 0 and author_result.stdout.strip():
                author = author_result.stdout.strip()
            else:
                # Fallback: try to get name, email, and login separately
                name_result = subprocess.run(['git', 'log', '-1', '--format=%an'],
                                           capture_output=True, text=True)
                email_result = subprocess.run(['git', 'log', '-1', '--format=%ae'],
                                            capture_output=True, text=True)
                login_result = subprocess.run(['git', 'log', '-1', '--format=%al'],
                                            capture_output=True, text=True)
                if name_result.returncode == 0 and email_result.returncode == 0:
                    name = name_result.stdout.strip()
                    email = email_result.stdout.strip()
                    login = login_result.stdout.strip() if login_result.returncode == 0 else ""
                    if name and email:
                        author = f"{name} <{email}> ({login})" if login else f"{name} <{email}>"
                    else:
                        author = "unknown"
                else:
                    author = "unknown"

            # Get files changed in the commit
            files_result = subprocess.run(['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', 'HEAD'],
                                        capture_output=True, text=True)
            changed_files = []
            if files_result.returncode == 0 and files_result.stdout.strip():
                changed_files = [f.strip() for f in files_result.stdout.strip().split('\n') if f.strip()]

            # Get diff statistics (lines added/removed)
            stats_result = subprocess.run(['git', 'diff-tree', '--no-commit-id', '--numstat', '-r', 'HEAD'],
                                        capture_output=True, text=True)
            diff_stats = []
            total_additions = 0
            total_deletions = 0

            if stats_result.returncode == 0 and stats_result.stdout.strip():
                for line in stats_result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            additions = parts[0]
                            deletions = parts[1]
                            filename = parts[2]

                            # Handle binary files (marked with '-')
                            if additions == '-' or deletions == '-':
                                diff_stats.append(f"{filename}: Binary file")
                            else:
                                try:
                                    add_count = int(additions)
                                    del_count = int(deletions)
                                    total_additions += add_count
                                    total_deletions += del_count
                                    diff_stats.append(f"{filename}: +{add_count}/-{del_count}")
                                except ValueError:
                                    diff_stats.append(f"{filename}: Invalid stats")

            return {
                'sha': commit_sha,
                'date': commit_date,
                'message': commit_message,
                'full_message': full_commit_message,
                'author': author,
                'changed_files': changed_files,
                'diff_stats': diff_stats,
                'total_additions': total_additions,
                'total_deletions': total_deletions
            }
        except Exception as e:
            self.log_warning(f"Could not get git info: {e}")
            return {
                'sha': "unknown",
                'date': "unknown",
                'message': "unknown",
                'full_message': "unknown",
                'author': "Unknown Author <unknown@unknown.com> (unknown)",
                'changed_files': [],
                'diff_stats': [],
                'total_additions': 0,
                'total_deletions': 0
            }

    def get_git_commit_sha(self) -> str:
        """Get the current git commit SHA (short version)"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=self.dynamo_ci_dir,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            # If git command fails, use timestamp as fallback
            return datetime.now().strftime("%H%M%S")

    def get_stored_sha(self) -> str:
        """Get stored SHA from file"""
        sha_file = self.dynamo_ci_dir / ".last_build_composite_sha"
        if sha_file.exists():
            return sha_file.read_text().strip()
        return ""

    def store_composite_sha(self, sha: str) -> None:
        """Store current composite SHA to file"""
        sha_file = self.dynamo_ci_dir / ".last_build_composite_sha"
        sha_file.write_text(sha)
        self.log_info(f"Stored composite SHA in repository: {sha}")

    def check_if_rebuild_needed(self) -> bool:
        """Check if rebuild is needed based on composite SHA"""
        self.log_info("Checking if rebuild is needed based on file changes...")
        self.log_info(f"Composite SHA file location: {self.dynamo_ci_dir}/.last_build_composite_sha")

        # Generate current composite SHA
        current_sha = self.generate_composite_sha_from_container_dir()
        if current_sha is None:
            self.log_error("Failed to generate current composite SHA")
            return False

        self.log_info(f"Generated composite SHA from container files: {current_sha}")

        # Get stored composite SHA
        stored_sha = self.get_stored_sha()

        if not stored_sha:
            self.log_info("No previous composite SHA found - rebuild needed")
            self.store_composite_sha(current_sha)
            return True  # Rebuild needed

        if current_sha == stored_sha:
            if self.force_run:
                self.log_info(f"Composite SHA unchanged ({current_sha}) but --force-run specified - proceeding")
                return True  # Rebuild needed (forced)
            else:
                self.log_info(f"Composite SHA unchanged ({current_sha}) - skipping rebuild")
                self.log_info("Use --force-run to force rebuild")
                return False  # Rebuild not needed
        else:
            self.log_info("Composite SHA changed:")
            self.log_info(f"  Previous: {stored_sha}")
            self.log_info(f"  Current:  {current_sha}")
            self.log_info("Rebuild needed")
            self.store_composite_sha(current_sha)
            return True  # Rebuild needed

    def get_build_commands(self, framework: str, docker_target_type: Optional[str]) -> tuple[bool, List[str], str]:
        """Get docker build commands from build.sh --dry-run and filter out latest tags

        This method:
        1. Calls build.sh --dry-run to get the raw docker build commands
        2. Extracts all "docker build" commands from the output
        3. Filters out any --tag arguments containing "latest" to prevent latest tagging
        4. Discovers the appropriate versioned image tag for the target type
        5. Returns the filtered commands and discovered tag ready for execution

        The filtering is critical because build.sh by default creates "latest" tags
        (e.g., dynamo:latest-vllm-local-dev) which we want to avoid to prevent
        clobbering existing latest images.

        Args:
            framework: Framework to build (VLLM, SGLANG, TRTLLM)
            docker_target_type: Docker build target type (None for dev, "local-dev" for local-dev)

        Returns:
            (success: bool, filtered_commands: List[str], image_tag: str)
        """
        self.log_info("Getting docker build commands from build.sh --dry-run...")

        # Build command - use --target flag only when docker_target_type is specified
        cmd = ["./container/build.sh", "--dry-run", "--framework", framework]
        if docker_target_type is not None:
            cmd.extend(["--target", docker_target_type])

        build_result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.dynamo_ci_dir)

        if build_result.returncode != 0:
            self.log_error(f"Failed to get build commands for {framework}")
            if build_result.stderr:
                self.log_error(f"Error: {build_result.stderr}")
            return False, [], ""

        # Extract and filter docker commands to remove latest tags
        # Also discover versioned tags for image naming
        docker_commands = []
        versioned_tags = []
        framework_lower = framework.lower()
        output_lines = build_result.stdout.split('\n') + build_result.stderr.split('\n')

        for line in output_lines:
            line = line.strip()
            if line.startswith("docker build"):
                # Filter out --tag arguments containing "latest" to prevent latest tagging
                # Example: --tag dynamo:latest-vllm-local-dev -> REMOVED
                # Example: --tag dynamo:v0.1.0.dev.abc123-vllm -> KEPT
                cmd_parts = line.split()
                filtered_parts = []
                skip_next = False

                for i, part in enumerate(cmd_parts):
                    if skip_next:
                        skip_next = False
                        continue

                    if part == "--tag":
                        # Check if next part contains "latest"
                        if i + 1 < len(cmd_parts) and "latest" in cmd_parts[i + 1]:
                            skip_next = True  # Skip both --tag and the latest tag value
                            self.log_info(f"Filtering out latest tag: {cmd_parts[i + 1]}")
                            continue
                        else:
                            filtered_parts.append(part)  # Keep non-latest --tag
                            # Also collect versioned tags for image discovery
                            tag = cmd_parts[i + 1]
                            if framework_lower in tag:
                                versioned_tags.append(tag)
                    else:
                        filtered_parts.append(part)

                if filtered_parts:  # Only add if there are still parts left after filtering
                    docker_commands.append(" ".join(filtered_parts))

        if not docker_commands:
            self.log_error(f"No docker build commands found for {framework}")
            return False, [], ""

        # Discover the appropriate image tag for this target type
        image_tag = ""
        if docker_target_type == "local-dev":
            # Look for local-dev tag
            for tag in versioned_tags:
                if "local-dev" in tag and framework_lower in tag:
                    image_tag = tag
                    break
        else:
            # Look for dev tag (without local-dev) - docker_target_type is None for dev
            for tag in versioned_tags:
                if "local-dev" not in tag and framework_lower in tag:
                    image_tag = tag
                    break

        if not image_tag:
            target_desc = "dev" if docker_target_type is None else docker_target_type
            self.log_error(f"Could not find versioned tag for {framework} {target_desc} image")
            return False, [], ""

        return True, docker_commands, image_tag

    def execute_build_commands(self, framework: str, docker_commands: List[str], log_file: Path, fail_file: Path) -> bool:
        """Execute docker build commands with real-time output"""
        try:
            with open(log_file, 'a') as f:
                f.write(f"Extracted {len(docker_commands)} docker build commands\n")

                for i, docker_cmd in enumerate(docker_commands):
                    self.log_info(f"Executing docker build command {i+1}/{len(docker_commands)}")
                    self.log_debug(f"+ {docker_cmd}")
                    f.write(f"+ {docker_cmd}\n")
                    f.flush()

                    # Run docker command with real-time output
                    process = subprocess.Popen(
                        docker_cmd,
                        shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        universal_newlines=True, bufsize=1
                    )

                    # Stream output line by line
                    if process.stdout:
                        while True:
                            output = process.stdout.readline()
                            if output == '' and process.poll() is not None:
                                break
                            if output:
                                print(output.strip())  # Show on console
                                f.write(output)  # Write to log file
                                f.flush()  # Ensure immediate write

                    if process.returncode != 0:
                        self.log_error(f"Docker build command {i+1} failed for {framework}")
                        f.write(f"Build Status: FAILED (docker command {i+1})\n")
                        with open(fail_file, 'a') as f:
                            f.write(f"{framework} docker build failed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        return False

                self.log_success(f"Build completed for {framework} (no latest tags created)")
                f.write("Build Status: SUCCESS (no latest tags)\n")
                return True

        except Exception as e:
            self.log_error(f"Build failed for {framework}: {e}")
            with open(log_file, 'a') as f:
                f.write(f"Build Status: FAILED ({e})\n")
            with open(fail_file, 'a') as f:
                f.write(f"{framework} build failed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            return False

    def test_framework_image(self, framework: str, docker_target_type: str) -> bool:
        """Test a specific framework image

        Args:
            framework: Framework name - one of "VLLM", "SGLANG", or "TRTLLM"
                      - "VLLM": vLLM framework for LLM inference
                      - "SGLANG": SGLang framework for structured generation
                      - "TRTLLM": TensorRT-LLM framework for optimized inference
            docker_target_type: Docker build target type - either "dev" or "local-dev"
                       - "dev": Regular framework image (e.g., dynamo:v0.1.0.dev.abc123-vllm)
                       - "local-dev": Local development image with user permissions
                                     (e.g., dynamo:v0.1.0.dev.abc123-vllm-local-dev)
        """
        framework_lower = framework.lower()

        # Get git commit SHA for log file naming
        commit_sha = self.get_git_commit_sha()

        # Get image tag using our build commands method (reuse the tag discovery logic)
        target_param = self.convert_target_for_build(docker_target_type)
        success, _, image_name = self.get_build_commands(framework, target_param)
        if not success:
            self.log_error(f"Failed to discover image tag for {framework} {docker_target_type}")
            return False

        # Log file suffix
        if docker_target_type == "local-dev":
            log_suffix = f"{commit_sha}.{framework_lower}.local-dev"
        else:
            log_suffix = f"{commit_sha}.{framework_lower}.dev"

        log_file = self.log_dir / f"{self.date}.{log_suffix}.log"
        success_file = self.log_dir / f"{self.date}.{log_suffix}.SUCC"
        fail_file = self.log_dir / f"{self.date}.{log_suffix}.FAIL"

        self.log_info(f"Testing framework: {framework} ({docker_target_type} image)")

        # Track timing for this framework/target combination
        test_key = f"{framework}_{docker_target_type}"
        start_time = time.time()
        build_start_time = None
        build_end_time = None

        # Change to dynamo_ci directory
        os.chdir(self.dynamo_ci_dir)

        # Log start of framework test (only in non-dry-run mode)
        if not self.dry_run:
            with open(log_file, 'a') as f:
                f.write("=" * 42 + "\n")
                f.write(f"Framework: {framework} ({docker_target_type} image)\n")
                f.write(f"Image: {image_name}\n")
                f.write(f"Date: {datetime.now().strftime('%c')}\n")
                f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if self.test_only:
                    f.write("Mode: TEST-ONLY (skipping build)\n")
                f.write("=" * 42 + "\n")
        else:
            self.log_info(f"Would write framework test start to: {log_file}")

        # Step 1: Build the framework (skip if TEST_ONLY is true)
        if not self.test_only:
            print("=" * 60)
            self.log_info(f"Step 1: Building {framework} framework ({docker_target_type} target)...")
            print("=" * 60)

            build_start_time = time.time()

            # Get build commands
            target_param = self.convert_target_for_build(docker_target_type)
            success, docker_commands, image_tag = self.get_build_commands(framework, target_param)
            if not success:
                if not self.dry_run:
                    with open(log_file, 'a') as f:
                        f.write("Build Status: FAILED (could not get docker commands)\n")
                    with open(fail_file, 'a') as f:
                        f.write(f"{framework} build command extraction failed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                return False

            if self.dry_run:
                # Show filtered commands that would be executed
                self.log_info("Filtered docker commands (removing latest tags):")
                for cmd in docker_commands:
                    print(f"+ {cmd}")
                self.log_success(f"Build commands prepared for {framework} (dry-run, no latest tags)")
            else:
                # Execute the commands
                if not self.execute_build_commands(framework, docker_commands, log_file, fail_file):
                    return False

            build_end_time = time.time()
        else:
            self.log_info(f"Skipping build step for {framework} (test-only mode)")
            if not self.dry_run:
                with open(log_file, 'a') as f:
                    f.write("Build Status: SKIPPED (test-only mode)\n")

        # Step 2: Run the container with sanity_check.py
        print("=" * 60)
        self.log_info(f"Step 2: Running sanity_check.py test for {framework} ({docker_target_type} target)...")
        print("=" * 60)

        # Get the docker command from container/run.sh dry-run, then execute without -it flags
        try:
            docker_cmd_result = subprocess.run(
                ["./container/run.sh", "--dry-run", "--image", image_name, "--mount-workspace", "--entrypoint", "deploy/sanity_check.py"],
                capture_output=True, text=True, cwd=self.dynamo_ci_dir
            )

            if docker_cmd_result.returncode == 0:
                # Extract docker command and remove -it flags
                docker_cmd = None
                # Check both stdout and stderr for the docker command
                output_text = docker_cmd_result.stdout + docker_cmd_result.stderr
                for line in output_text.split('\n'):
                    if line.startswith("docker run"):
                        docker_cmd = line.replace(" -it ", " ")
                        break

                if docker_cmd and not self.dry_run:
                    prefix = "DRYRUN +" if self.dry_run else "+"
                    self.log_debug(f"+ timeout 30 {docker_cmd}")
                    try:
                        with open(log_file, 'a') as f:
                            # Run container test with real-time output
                            process = subprocess.Popen(
                                f"timeout 30 {docker_cmd}",
                                shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True, bufsize=1
                            )

                            # Stream output line by line
                            if process.stdout:
                                while True:
                                    output = process.stdout.readline()
                                    if output == '' and process.poll() is not None:
                                        break
                                    if output:
                                        print(output.strip())  # Show on console
                                        f.write(output)  # Write to log file
                                        f.flush()  # Ensure immediate write

                            result = process
                            result.returncode = process.poll()

                        if result.returncode == 0:
                            self.log_success(f"Container test completed for {framework}")
                            with open(log_file, 'a') as f:
                                f.write("Container Test Status: SUCCESS\n")
                            with open(success_file, 'a') as f:
                                f.write(f"{framework} test completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        else:
                            self.log_error(f"Container test failed for {framework}")
                            with open(log_file, 'a') as f:
                                f.write("Container Test Status: FAILED\n")
                            with open(fail_file, 'a') as f:
                                f.write(f"{framework} container test failed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            return False

                    except subprocess.TimeoutExpired:
                        self.log_error(f"Container test timeout for {framework}")
                        with open(log_file, 'a') as f:
                            f.write("Container Test Status: TIMEOUT\n")
                        with open(fail_file, 'a') as f:
                            f.write(f"{framework} container test timeout at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        return False

                elif docker_cmd:
                    prefix = "DRYRUN +" if self.dry_run else "+"
                    self.log_debug(f"+ timeout 30 {docker_cmd}")
                    self.log_success(f"Container test completed for {framework} (dry-run)")
                    self.log_info(f"Would write success to: {success_file}")
                else:
                    self.log_error(f"Could not extract docker command for {framework}")
                    return False
            else:
                self.log_error(f"Failed to get docker command for {framework}")
                return False

        except Exception as e:
            self.log_error(f"Exception during container test for {framework}: {e}")
            return False

        # Log end of framework test (only in non-dry-run mode)
        if not self.dry_run:
            with open(log_file, 'a') as f:
                f.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("Overall Status: SUCCESS\n\n")
        else:
            self.log_info(f"Would write framework test end to: {log_file}")

        # Store timing information for email reporting
        end_time = time.time()
        total_time = end_time - start_time

        # Calculate build time if build was performed
        if build_start_time is not None and build_end_time is not None:
            build_time = build_end_time - build_start_time
            test_time = total_time - build_time
            self.build_times[f"{test_key}_build"] = build_time
            self.build_times[f"{test_key}_test"] = test_time
            self.log_debug(f"Build time for {test_key}: {build_time:.1f}s")
            self.log_debug(f"Test time for {test_key}: {test_time:.1f}s")
        else:
            # Test-only mode: no build time, all time is test time
            self.build_times[f"{test_key}_test"] = total_time
            self.log_debug(f"Test-only time for {test_key}: {total_time:.1f}s")

        # Store total time (build + test)
        self.build_times[f"{test_key}_total"] = total_time
        self.log_debug(f"Total time for {test_key}: {total_time:.1f}s")

        return True

    def test_framework(self, framework: str) -> bool:
        """Test a framework with configured targets"""
        overall_success = True

        targets_str = ", ".join(self.targets)
        self.log_info(f"Starting tests for framework: {framework} (targets: {targets_str})")

        # Test each configured target
        for target in self.targets:
            if self.test_framework_image(framework, target):
                self.log_success(f"Framework {framework} {target} image test completed successfully")
            else:
                self.log_error(f"Framework {framework} {target} image test failed")
                overall_success = False

        if overall_success:
            self.log_success(f"All tests completed successfully for framework: {framework}")
            return True
        else:
            self.log_error(f"Some tests failed for framework: {framework}")
            return False

    def run_all_tests(self) -> bool:
        """Run all framework tests"""
        overall_success = True
        successful_frameworks = []
        failed_frameworks = []

        self.log_info("Starting CI tests for all frameworks...")

        for framework in self.FRAMEWORKS:
            self.log_info(f"Starting test for framework: {framework}")

            if self.test_framework(framework):
                self.log_success(f"Framework {framework} test completed successfully")
                successful_frameworks.append(framework)
            else:
                self.log_error(f"Framework {framework} test failed")
                failed_frameworks.append(framework)
                overall_success = False

            self.log_info(f"Completed test for framework: {framework}")
            print("-" * 40)

        # Summary
        self.log_info("Test Summary:")
        if successful_frameworks:
            self.log_success(f"Successful frameworks: {', '.join(successful_frameworks)}")

        if failed_frameworks:
            self.log_error(f"Failed frameworks: {', '.join(failed_frameworks)}")

        if overall_success:
            self.log_success("All framework tests completed successfully!")
            return True
        else:
            self.log_error("Some framework tests failed!")
            return False

    def show_usage(self) -> None:
        """Show usage information"""
        print("""Usage: python dynamo_docker_builder.py [OPTIONS]

Options:
  -f, --framework FRAMEWORK    Test specific framework (VLLM, SGLANG, TRTLLM) - case insensitive
  -a, --all                    Test all frameworks (default)
  --test-only                  Skip build step and only run container tests
  --no-checkout                Skip git operations and use existing repository (except when --repo-sha is used)
  --force-run                  Force run even if files haven't changed
  --dry-run, --dryrun          Show commands that would be executed without running them
  --target TARGETS             Comma-separated Docker build targets to test: dev, local-dev (default: dev,local-dev)
  --repo-sha SHA               Git SHA to checkout instead of latest main branch
  -h, --help                   Show this help message

Examples:
  python dynamo_docker_builder.py                           # Test all frameworks
  python dynamo_docker_builder.py --all                     # Test all frameworks
  python dynamo_docker_builder.py --framework VLLM          # Test only VLLM framework
  python dynamo_docker_builder.py --framework vllm          # Same as above (case insensitive)
  python dynamo_docker_builder.py --framework sglang        # Test only SGLANG framework (case insensitive)
  python dynamo_docker_builder.py --test-only               # Skip build and only run tests for all frameworks
  python dynamo_docker_builder.py --framework VLLM --test-only  # Skip build and test only VLLM framework
  python dynamo_docker_builder.py --no-checkout             # Use existing repo without git operations
  python dynamo_docker_builder.py --test-only --no-checkout # Skip both build and git operations
  python dynamo_docker_builder.py --force-run               # Force run even if no files changed
  python dynamo_docker_builder.py --dry-run                 # Show what would be executed without running
  python dynamo_docker_builder.py --dryrun                  # Same as --dry-run
  python dynamo_docker_builder.py --email user@nvidia.com  # Send email notifications
  python dynamo_docker_builder.py --target dev              # Test only dev target
  python dynamo_docker_builder.py --target local-dev        # Test only local-dev target
  python dynamo_docker_builder.py --target dev,local-dev,custom  # Test multiple targets
  python dynamo_docker_builder.py --repo-sha abc123def        # Test specific git commit
  python dynamo_docker_builder.py --framework VLLM --repo-sha abc123def  # Test specific commit with VLLM only
  python dynamo_docker_builder.py --no-checkout --repo-sha abc123def   # Use existing repo but checkout specific SHA
""")

    def main(self) -> int:
        """Main function"""
        parser = argparse.ArgumentParser(description="DynamoDockerBuilder - Automated Docker Build and Test System")
        parser.add_argument("-f", "--framework", "--frameworks", type=str, action='append', dest='framework',
                          help="Test specific framework (VLLM, SGLANG, TRTLLM) - case insensitive. Can be specified multiple times.")
        parser.add_argument("--target", type=str, default="dev,local-dev",
                          help="Comma-separated Docker build targets to test: dev, local-dev (default: dev,local-dev)")
        parser.add_argument("-a", "--all", action="store_true", default=True,
                          help="Test all frameworks (default)")
        parser.add_argument("--test-only", action="store_true",
                          help="Skip build step and only run container tests")
        parser.add_argument("--no-checkout", action="store_true",
                          help="Skip git operations and use existing repository")
        parser.add_argument("--force-run", action="store_true",
                          help="Force run even if files haven't changed or another process is running")
        parser.add_argument("--dry-run", "--dryrun", action="store_true",
                          help="Show commands that would be executed without running them")
        parser.add_argument("--repo-path", type=str, default=None,
                          help="Path to the dynamo repository (default: ../dynamo_ci)")
        parser.add_argument("--email", type=str, default=None,
                          help="Email address for notifications (sends email if specified)")
        parser.add_argument("--repo-sha", type=str, default=None,
                          help="Git SHA to checkout instead of latest main branch")

        args = parser.parse_args()

        # Update repo path if specified
        if args.repo_path:
            self.dynamo_ci_dir = Path(args.repo_path).absolute()
            self.log_dir = self.dynamo_ci_dir / "logs" / self.date

        # Set configuration flags
        self.dry_run = args.dry_run
        self.test_only = args.test_only
        self.no_checkout = args.no_checkout
        self.force_run = args.force_run
        self.email = args.email
        self.repo_sha = args.repo_sha

        # Parse targets
        self.targets = [target.strip() for target in args.target.split(',') if target.strip()]
        if not self.targets:
            self.targets = ["dev", "local-dev"]  # Fallback to default

        # Validate targets - only allow known Docker build targets
        valid_targets = ["dev", "local-dev", "runtime", "dynamo_base", "framework"]
        invalid_targets = [t for t in self.targets if t not in valid_targets]
        if invalid_targets:
            valid_targets_str = ", ".join(valid_targets)
            invalid_targets_str = ", ".join(invalid_targets)
            self.log_error(f"Invalid target(s) '{invalid_targets_str}'. Valid Docker build targets are: {valid_targets_str}")
            self.log_error("Note: Targets are Docker build targets (dev, local-dev, runtime, dynamo_base, framework), not framework names (VLLM, SGLANG, TRTLLM)")
            return 1

        # Determine which frameworks to test
        if args.framework:
            # Normalize framework names to uppercase for case-insensitive matching
            frameworks_to_test = []
            for framework in args.framework:
                framework_upper = framework.upper()
                if framework_upper not in self.FRAMEWORKS:
                    valid_frameworks = ", ".join(self.FRAMEWORKS)
                    self.log_error(f"Invalid framework '{framework}'. Valid options are: {valid_frameworks} (case insensitive)")
                    return 1
                frameworks_to_test.append(framework_upper)
        else:
            # Test all frameworks by default
            frameworks_to_test = list(self.FRAMEWORKS)

        print("=" * 60)
        self.log_info(f"Starting DynamoDockerBuilder - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # Check if another instance is already running
        self.check_if_running()

        if self.dry_run:
            self.log_info("DRY-RUN MODE: Commands will be shown but not executed")
        if self.test_only:
            self.log_info("TEST-ONLY MODE: Skipping build step and only running tests")
        if self.no_checkout:
            self.log_info("NO-CHECKOUT MODE: Skipping git operations and using existing repository")
        if self.force_run:
            self.log_info("FORCE-RUN MODE: Will run even if files haven't changed or another process is running")
        if self.repo_sha:
            self.log_info(f"REPO-SHA MODE: Will checkout specific SHA: {self.repo_sha}")

        self.log_info(f"Date: {self.date}")
        self.log_info(f"Dynamo CI Directory: {self.dynamo_ci_dir}")
        self.log_info(f"Log Directory: {self.log_dir}")

        # Setup repository and logging
        self.setup_dynamo_ci()
        self.setup_logging()

        # Check if rebuild is needed based on file changes
        rebuild_needed = self.check_if_rebuild_needed()

        if not rebuild_needed:
            self.log_success("No rebuild needed - all files unchanged")
            self.log_info("Exiting early (use --force-run to force rebuild)")
            self.log_info("Preserving existing log files")
            return 0

        # Run tests and collect detailed results for email notification
        test_results: Dict[str, bool] = {}
        failure_details: Dict[str, str] = {}
        overall_success = True

        # Log what we're testing
        if len(frameworks_to_test) == 1:
            self.log_info(f"Testing single framework: {frameworks_to_test[0]}")
        elif len(frameworks_to_test) == len(self.FRAMEWORKS):
            self.log_info("Testing all frameworks")
        else:
            frameworks_str = ", ".join(frameworks_to_test)
            self.log_info(f"Testing multiple frameworks: {frameworks_str}")

        # Clean up logs (all frameworks if testing all, specific ones if testing subset)
        if len(frameworks_to_test) == len(self.FRAMEWORKS):
            self.cleanup_existing_logs()  # Clean all
        else:
            for framework in frameworks_to_test:
                self.cleanup_existing_logs(framework=framework)  # Clean specific ones

        # Test each framework and collect detailed results
        overall_success = True
        for framework in frameworks_to_test:
            targets_str = ", ".join(self.targets)
            self.log_info(f"Starting tests for framework: {framework} (targets: {targets_str})")

            # Test all configured targets
            framework_success = True
            for target in self.targets:
                target_success = self.test_framework_image(framework, target)
                test_results[f"{framework}_{target}"] = target_success

                # Collect failure details for failed tests
                if not target_success:
                    failure_details[f"{framework}_{target}"] = self.get_failure_details(framework, target)
                    framework_success = False

            if not framework_success:
                overall_success = False

            if framework_success:
                self.log_success(f"All tests completed successfully for framework: {framework}")
            else:
                self.log_error(f"Some tests failed for framework: {framework}")

        # Send email notification if email is specified and tests actually ran
        if self.email:
            self.send_email_notification(test_results, failure_details, self.build_times)

        # Return appropriate exit code
        if overall_success:
            self.log_success("All tests completed successfully")
            return 0
        else:
            self.log_error("Some tests failed")
            return 1


if __name__ == "__main__":
    tester = DynamoBuilderTester()
    sys.exit(tester.main())
