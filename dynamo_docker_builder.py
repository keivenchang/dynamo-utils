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
import os
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

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
        self.dest_email = None
        
        # Lock file for preventing concurrent runs
        self.lock_file = self.script_dir / f".{Path(__file__).name}.lock"
    
    def log_info(self, message: str) -> None:
        """Log info message"""
        print(f"[INFO] {message}")
    
    def log_success(self, message: str) -> None:
        """Log success message"""
        print(f"[SUCCESS] {message}")
    
    def log_error(self, message: str) -> None:
        """Log error message"""
        print(f"[ERROR] {message}")
    
    def log_warning(self, message: str) -> None:
        """Log warning message"""
        print(f"[WARNING] {message}")
    
    def convert_pr_links(self, message: str) -> str:
        """Convert PR references like (#3107) to GitHub links"""
        import re
        # Pattern to match (#number)
        pr_pattern = r'\(#(\d+)\)'
        
        def replace_pr(match):
            pr_number = match.group(1)
            return f'(<a href="https://github.com/ai-dynamo/dynamo/pull/{pr_number}" style="color: #0066cc;">#{pr_number}</a>)'
        
        return re.sub(pr_pattern, replace_pr, message)
    
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
    
    def send_email_notification(self, results: dict, failure_details: dict = None) -> None:
        """Send email notification with test results
        
        Args:
            results: Dict mapping test keys (e.g. 'VLLM_dev') to success boolean
            failure_details: Dict mapping failed test keys to error output strings
        """
        if not self.dest_email:
            return
            
        if failure_details is None:
            failure_details = {}
            
        if self.dry_run:
            self.log_info(f"DRY-RUN: Would send email notification to {self.dest_email}")
            return
            
        try:
            # Get git information
            git_info = self.get_git_info()
            
            # Count results (only count tests that were actually run)
            total_tests = len(results)
            passed_tests = sum(1 for success in results.values() if success)
            failed_tests = sum(1 for success in results.values() if not success)
            
            # Collect failed tests for summary
            failed_tests_list = [key for key, success in results.items() if not success]
            
            # Determine overall status
            overall_status = "SUCCESS" if failed_tests == 0 else "FAILURE"
            status_color = "#28a745" if failed_tests == 0 else "#dc3545"
            
            # Create HTML email content
            status_prefix = "SUCC" if overall_status == "SUCCESS" else "FAIL"
            
            # Add failure summary to subject if there are failures
            if failed_tests_list:
                failure_summary = ", ".join(failed_tests_list)
                subject = f"{status_prefix}: DynamoDockerBuilder - {git_info['sha']} ({failure_summary})"
            else:
                subject = f"{status_prefix}: DynamoDockerBuilder - {git_info['sha']}"
            
            html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
body {{ font-family: Arial, sans-serif; margin: 10px; line-height: 1.3; }}
.header {{ background-color: {status_color}; color: white; padding: 4px 6px; border-radius: 2px; margin-bottom: 5px; }}
.summary {{ background-color: #f8f9fa; padding: 4px 6px; border-radius: 2px; margin: 3px 0; }}
.results {{ margin: 3px 0; }}
.framework {{ margin: 3px 0; padding: 3px; border-left: 2px solid #007bff; }}
.success {{ color: #28a745; font-weight: bold; }}
.failure {{ color: #dc3545; font-weight: bold; }}
.git-info {{ background-color: #e9ecef; padding: 4px 6px; border-radius: 2px; font-family: monospace; font-size: 0.9em; }}
.error-output {{ background-color: #2d3748; color: #e2e8f0; padding: 8px; border-radius: 3px; font-family: 'Courier New', monospace; font-size: 0.85em; margin: 5px 0; overflow-x: auto; white-space: pre-wrap; }}
p {{ margin: 1px 0; }}
h3 {{ margin: 4px 0 2px 0; font-size: 1.0em; }}
h4 {{ margin: 3px 0 1px 0; font-size: 0.95em; }}
h2 {{ margin: 0; font-size: 1.1em; font-weight: normal; }}
</style>
</head>
<body>
<div class="header">
<h2>DynamoDockerBuilder - {overall_status}</h2>
</div>

<div class="summary">
<p><strong>Total Tests:</strong> {total_tests}</p>
<p><strong>Passed:</strong> <span class="success">{passed_tests}</span></p>
<p><strong>Failed:</strong> <span class="failure">{failed_tests}</span></p>
<p><strong>Build & Test Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} PDT</p>
</div>"""

            html_content += f"""

<div class="git-info">
<p><strong>Commit SHA:</strong> {git_info['sha']}</p>
<p><strong>Commit Date:</strong> {git_info['date']}</p>
<p><strong>Commit Message:</strong> {self.convert_pr_links(git_info['message'])}</p>
<p><strong>Author:</strong> {git_info['author']}</p>
</div>

<div class="results">
"""
            
            # Add framework results (only show frameworks that were actually tested)
            tested_frameworks = set()
            for key in results.keys():
                framework = key.split('_')[0]  # Extract framework from key like "VLLM_dev"
                tested_frameworks.add(framework)
            
            for framework in sorted(tested_frameworks):
                html_content += f'<div class="framework"><h4>Dockerfile.{framework.lower()} targets:</h4>'
                
                for target in ['dev', 'local-dev']:
                    key = f"{framework}_{target}"
                    if key in results:  # Only show targets that were actually tested
                        status = "SUCCESS" if results[key] else "FAILURE"
                        css_class = "success" if status == "SUCCESS" else "failure"
                        target_display = "dev" if target == "dev" else "local-dev"
                        
                        if status == "SUCCESS":
                            html_content += f'<p>✅ {target_display} target: <span class="{css_class}">PASSED</span></p>'
                        else:
                            html_content += f'<p>❌ {target_display} target: <span class="{css_class}">FAILED</span></p>'
                            # Add error output if available
                            if key in failure_details and failure_details[key]:
                                error_lines = failure_details[key].split('\\n')
                                if len(error_lines) > 25:
                                    error_output = '\\n'.join(error_lines[-25:])  # Show last 25 lines
                                    error_output = "... (showing last 25 lines)\\n" + error_output
                                else:
                                    error_output = failure_details[key]
                                html_content += f'<div class="error-output">{error_output}</div>'
                
                html_content += '</div>'
            
            html_content += f"""
</div>

<div class="summary">
<p><strong>Repository:</strong> {self.dynamo_ci_dir}</p>
<p><strong>Log Directory:</strong> {self.log_dir}</p>
</div>

<p><em>This email was generated automatically by DynamoDockerBuilder.</em></p>
</body>
</html>"""
            
            # Create email file with proper CRLF formatting
            email_file = Path(f"/tmp/dynamo_email_{os.getpid()}.txt")
            
            # Use printf to create proper CRLF formatting like our successful test
            subprocess.run([
                'printf', 
                f'Subject: {subject}\\r\\nFrom: DynamoDockerBuilder <dynamo-docker-builder@nvidia.com>\\r\\nTo: {self.dest_email}\\r\\nMIME-Version: 1.0\\r\\nContent-Type: text/html; charset=UTF-8\\r\\n\\r\\n{html_content}\\r\\n'
            ], stdout=open(email_file, 'w'))
            
            # Send email using curl
            result = subprocess.run([
                'curl', '--url', 'smtp://smtp.nvidia.com:25',
                '--mail-from', 'dynamo-docker-builder@nvidia.com',
                '--mail-rcpt', self.dest_email,
                '--upload-file', str(email_file)
            ], capture_output=True, text=True)
            
            # Clean up
            email_file.unlink(missing_ok=True)
            
            if result.returncode == 0:
                self.log_success(f"Email notification sent to {self.dest_email}")
            else:
                self.log_error(f"Failed to send email: {result.stderr}")
                
        except Exception as e:
            self.log_error(f"Error sending email notification: {e}")
    
    def cmd(self, command: List[str], **kwargs) -> subprocess.CompletedProcess:
        """Execute command with dry-run support"""
        # Show command in shell tracing format
        cmd_str = " ".join(shlex.quote(str(arg)) for arg in command)
        print(f"+ {cmd_str}")
        
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
        
        if self.no_checkout:
            self.log_info("NO-CHECKOUT MODE: Skipping git operations, using existing repository")
            
            if not self.dynamo_ci_dir.exists():
                self.log_error(f"dynamo_ci directory does not exist at {self.dynamo_ci_dir}")
                self.log_error("Cannot use --no-checkout without existing repository")
                sys.exit(1)
            
            self.log_success(f"Using existing repository at {self.dynamo_ci_dir}")
            return
        
        if not self.dynamo_ci_dir.exists():
            self.log_info(f"Cloning dynamo repository to {self.dynamo_ci_dir}")
            self.cmd(["git", "clone", "git@github.com:ai-dynamo/dynamo.git", str(self.dynamo_ci_dir)])
        else:
            self.log_info("dynamo_ci directory exists, updating from main branch")
            
            # Check if it's a git repository (only in non-dry-run mode)
            if not self.dry_run and not (self.dynamo_ci_dir / ".git").exists():
                self.log_error("dynamo_ci exists but is not a git repository")
                sys.exit(1)
            
            # Fetch and pull from main
            os.chdir(self.dynamo_ci_dir)
            self.cmd(["git", "fetch", "origin"])
            self.cmd(["git", "checkout", "main"])
            self.cmd(["git", "pull", "origin", "main"])
        
        self.log_success("Repository setup complete")
    
    def setup_logging(self) -> None:
        """Create date-based log directory"""
        self.log_info("Setting up date-based logging directory...")
        if not self.dry_run:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        else:
            print(f"+ mkdir -p {self.log_dir}")
        self.log_success(f"Date-based log directory created at {self.log_dir}")
    
    def cleanup_existing_logs(self, framework: str = None) -> None:
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
                print(f"+ rm -f {self.log_dir}/{pattern}")
            
            if framework:
                self.log_info(f"Would remove existing log files for {self.date} and {framework} (dry-run)")
            else:
                self.log_info(f"Would remove existing log files for {self.date} (dry-run)")
    
    def generate_composite_sha(self) -> Optional[str]:
        """Generate composite SHA from key container files"""
        files_to_hash = [
            "container/Dockerfile.sglang",
            "container/Dockerfile.trtllm", 
            "container/Dockerfile.vllm",
            "container/run.sh",
            "container/build.sh"
        ]
        
        with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            
            try:
                # Concatenate all files that exist
                found_files = 0
                for file_rel_path in files_to_hash:
                    full_path = self.dynamo_ci_dir / file_rel_path
                    if full_path.exists():
                        with open(full_path, 'rb') as f:
                            temp_file.write(f.read())
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
            
            # Get commit message
            msg_result = subprocess.run(['git', 'log', '-1', '--format=%s'], 
                                      capture_output=True, text=True)
            commit_message = msg_result.stdout.strip() if msg_result.returncode == 0 else "unknown"
            
            # Get author (the actual person who wrote the code)
            author_result = subprocess.run(['git', 'log', '-1', '--format=%an <%ae>'], 
                                         capture_output=True, text=True)
            author = author_result.stdout.strip() if author_result.returncode == 0 else "unknown"
            
            return {
                'sha': commit_sha,
                'date': commit_date,
                'message': commit_message,
                'author': author
            }
        except Exception as e:
            self.log_warning(f"Could not get git info: {e}")
            return {
                'sha': "unknown",
                'date': "unknown", 
                'message': "unknown",
                'author': "unknown"
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
        sha_file = self.dynamo_ci_dir / ".last_build_sha"
        if sha_file.exists():
            return sha_file.read_text().strip()
        return ""
    
    def store_composite_sha(self, sha: str) -> None:
        """Store current composite SHA to file"""
        sha_file = self.dynamo_ci_dir / ".last_build_sha"
        sha_file.write_text(sha)
        self.log_info(f"Stored composite SHA in repository: {sha}")
    
    def check_if_rebuild_needed(self) -> bool:
        """Check if rebuild is needed based on composite SHA"""
        self.log_info("Checking if rebuild is needed based on file changes...")
        self.log_info(f"Composite SHA file location: {self.dynamo_ci_dir}/.last_build_sha")
        
        # Generate current composite SHA
        current_sha = self.generate_composite_sha()
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
                    print(f"+ {docker_cmd}")
                    f.write(f"+ {docker_cmd}\n")
                    f.flush()
                    
                    # Run docker command with real-time output
                    process = subprocess.Popen(
                        docker_cmd,
                        shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        universal_newlines=True, bufsize=1
                    )
                    
                    # Stream output line by line
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
        target_param = None if docker_target_type == "dev" else docker_target_type
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
            
            # Get build commands
            target_param = None if docker_target_type == "dev" else docker_target_type
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
        else:
            self.log_info(f"Skipping build step for {framework} (test-only mode)")
            if not self.dry_run:
                with open(log_file, 'a') as f:
                    f.write("Build Status: SKIPPED (test-only mode)\n")
        
        # Step 2: Run the container with dynamo_check.py
        print("=" * 60)
        self.log_info(f"Step 2: Running container test for {framework} ({docker_target_type} target)...")
        print("=" * 60)
        
        # Get the docker command from container/run.sh dry-run, then execute without -it flags
        try:
            docker_cmd_result = subprocess.run(
                ["./container/run.sh", "--dry-run", "--image", image_name, "--mount-workspace", "--entrypoint", "deploy/dynamo_check.py"],
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
                    print(f"+ timeout 30 {docker_cmd}")
                    try:
                        with open(log_file, 'a') as f:
                            # Run container test with real-time output
                            process = subprocess.Popen(
                                f"timeout 30 {docker_cmd}",
                                shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True, bufsize=1
                            )
                            
                            # Stream output line by line
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
                    self.cmd(["timeout", "30", "bash", "-c", docker_cmd])
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
        
        return True
    
    def test_framework(self, framework: str) -> bool:
        """Test a framework (both dev and local-dev images)"""
        overall_success = True
        
        self.log_info(f"Starting tests for framework: {framework} (both dev and local-dev images)")
        
        # Test the regular dev image
        if self.test_framework_image(framework, "dev"):
            self.log_success(f"Framework {framework} dev image test completed successfully")
        else:
            self.log_error(f"Framework {framework} dev image test failed")
            overall_success = False
        
        # Test the local-dev image
        if self.test_framework_image(framework, "local-dev"):
            self.log_success(f"Framework {framework} local-dev image test completed successfully")
        else:
            self.log_error(f"Framework {framework} local-dev image test failed")
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
        print("Usage: python dynamo_docker_builder.py [OPTIONS]")
        print("")
        print("Options:")
        print("  -f, --framework FRAMEWORK    Test specific framework (VLLM, SGLANG, TRTLLM)")
        print("  -a, --all                    Test all frameworks (default)")
        print("  --test-only                  Skip build step and only run container tests")
        print("  --no-checkout                Skip git operations and use existing repository")
        print("  --force-run                  Force run even if files haven't changed")
        print("  --dry-run, --dryrun          Show commands that would be executed without running them")
        print("  -h, --help                   Show this help message")
        print("")
        print("Examples:")
        print("  python dynamo_docker_builder.py                           # Test all frameworks")
        print("  python dynamo_docker_builder.py --all                     # Test all frameworks")
        print("  python dynamo_docker_builder.py --framework VLLM          # Test only VLLM framework")
        print("  python dynamo_docker_builder.py --test-only               # Skip build and only run tests for all frameworks")
        print("  python dynamo_docker_builder.py --framework VLLM --test-only  # Skip build and test only VLLM framework")
        print("  python dynamo_docker_builder.py --no-checkout             # Use existing repo without git operations")
        print("  python dynamo_docker_builder.py --test-only --no-checkout # Skip both build and git operations")
        print("  python dynamo_docker_builder.py --force-run               # Force run even if no files changed")
        print("  python dynamo_docker_builder.py --dry-run                 # Show what would be executed without running")
        print("  python dynamo_docker_builder.py --dryrun                  # Same as --dry-run")
        print("  python dynamo_docker_builder.py --dest-email user@nvidia.com  # Send email notifications")
        print("")
    
    def main(self) -> int:
        """Main function"""
        parser = argparse.ArgumentParser(description="DynamoDockerBuilder - Automated Docker Build and Test System")
        parser.add_argument("-f", "--framework", choices=self.FRAMEWORKS,
                          help="Test specific framework")
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
        parser.add_argument("--dest-email", type=str, default=None,
                          help="Destination email address for notifications (sends email if specified)")
        
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
        self.dest_email = args.dest_email
        
        # If framework is specified, disable --all
        if args.framework:
            test_all = False
            test_framework_only = args.framework
        else:
            test_all = True
            test_framework_only = None
        
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
        test_results = {}
        failure_details = {}
        overall_success = True
        
        if test_framework_only:
            # Only clean up logs for the specific framework being tested
            self.cleanup_existing_logs(framework=test_framework_only)
            self.log_info(f"Testing single framework: {test_framework_only}")
            
            # Collect results for both dev and local-dev targets
            dev_success = self.test_framework_image(test_framework_only, "dev")
            local_dev_success = self.test_framework_image(test_framework_only, "local-dev")
            
            test_results[f"{test_framework_only}_dev"] = dev_success
            test_results[f"{test_framework_only}_local-dev"] = local_dev_success
            
            # Collect failure details for failed tests
            if not dev_success:
                failure_details[f"{test_framework_only}_dev"] = self.get_failure_details(test_framework_only, "dev")
            if not local_dev_success:
                failure_details[f"{test_framework_only}_local-dev"] = self.get_failure_details(test_framework_only, "local-dev")
            
            framework_success = dev_success and local_dev_success
            overall_success = framework_success
            
            if framework_success:
                self.log_success(f"Framework {test_framework_only} test completed successfully")
            else:
                self.log_error(f"Framework {test_framework_only} test failed")
                
        elif test_all:
            # Clean up all logs when testing all frameworks
            self.cleanup_existing_logs()
            self.log_info("Testing all frameworks")
            
            # Test all frameworks and collect detailed results
            for framework in self.FRAMEWORKS:
                self.log_info(f"Starting tests for framework: {framework} (both dev and local-dev images)")
                
                # Test both dev and local-dev targets
                dev_success = self.test_framework_image(framework, "dev")
                local_dev_success = self.test_framework_image(framework, "local-dev")
                
                test_results[f"{framework}_dev"] = dev_success
                test_results[f"{framework}_local-dev"] = local_dev_success
                
                # Collect failure details for failed tests
                if not dev_success:
                    failure_details[f"{framework}_dev"] = self.get_failure_details(framework, "dev")
                if not local_dev_success:
                    failure_details[f"{framework}_local-dev"] = self.get_failure_details(framework, "local-dev")
                
                framework_success = dev_success and local_dev_success
                if not framework_success:
                    overall_success = False
                
                if framework_success:
                    self.log_success(f"All tests completed successfully for framework: {framework}")
                else:
                    self.log_error(f"Some tests failed for framework: {framework}")
        
        # Send email notification if destination email is specified and tests actually ran
        if self.dest_email:
            self.send_email_notification(test_results, failure_details)
        
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
