#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test Growth Statistics Analyzer

Analyzes pytest test growth over time by checking out historical commits
and counting tests with their markers (pre_merge, post_merge, parallel, etc.).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class TestStats:
    """Statistics for tests at a specific point in time."""
    
    date: str
    commit_hash: str
    py_files: int
    test_files: int
    total_tests: int
    markers: Dict[str, int]  # marker_name -> count


def run_command(cmd: List[str], cwd: str = None, check: bool = True) -> Tuple[int, str, str]:
    """Run a command and return (exit_code, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=check,
            cwd=cwd,
            timeout=300  # 5 minute timeout for pytest collection
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout or "", e.stderr or ""
    except subprocess.TimeoutExpired:
        print(f"  ⚠️  Command timed out after 300s: {' '.join(cmd)}", file=sys.stderr)
        return -1, "", "Timeout"


def get_commit_at_date(repo_path: str, target_date: str) -> Optional[str]:
    """Get the latest commit hash at or before a specific date."""
    cmd = [
        "git", "rev-list", 
        "-n", "1", 
        f"--before={target_date} 23:59:59",
        "--all"
    ]
    exit_code, stdout, stderr = run_command(cmd, cwd=repo_path)
    
    if exit_code != 0 or not stdout.strip():
        return None
    
    return stdout.strip()


def checkout_commit(repo_path: str, commit_hash: str) -> bool:
    """Checkout a specific commit. Returns True on success."""
    # Stash any local changes first
    run_command(["git", "stash"], cwd=repo_path, check=False)
    
    cmd = ["git", "checkout", "-f", commit_hash]
    exit_code, stdout, stderr = run_command(cmd, cwd=repo_path, check=False)
    
    return exit_code == 0


def find_test_files(repo_path: str, test_dir: str = "tests") -> Tuple[List[Path], List[Path]]:
    """
    Find Python files in the test directory.
    Returns (all_py_files, test_only_files)
    """
    test_path = Path(repo_path) / test_dir
    
    if not test_path.exists():
        return [], []
    
    # Get ALL .py files in tests directory
    all_py_files = list(test_path.rglob("*.py"))
    
    # Filter to get only actual test files (test_*.py or *_test.py)
    test_only_files = []
    for py_file in all_py_files:
        filename = py_file.name
        if filename.startswith('test_') or filename.endswith('_test.py'):
            test_only_files.append(py_file)
    
    return sorted(set(all_py_files)), sorted(set(test_only_files))


def collect_tests_with_pytest(repo_path: str, test_dir: str = "tests") -> Optional[Dict[str, any]]:
    """
    Parse pytest markers from a test file by looking for @pytest.mark decorators.
    Returns dict mapping test_name -> set of markers.
    """
    markers_by_test = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Look for test functions and their markers
        lines = content.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check if this line is a test function definition
            test_match = re.match(r'\s*(async\s+)?def\s+(test_\w+)', line)
            if test_match:
                test_name = test_match.group(2)
                
                # Look backwards to collect all markers above this test
                markers = set()
                j = i - 1
                while j >= 0:
                    prev_line = lines[j].strip()
                    
                    # Check for pytest.mark decorator
                    marker_match = re.match(r'@pytest\.mark\.(\w+)', prev_line)
                    if marker_match:
                        markers.add(marker_match.group(1))
                        j -= 1
                        continue
                    
                    # Stop if we hit a non-decorator line (except blank lines and comments)
                    if prev_line and not prev_line.startswith('@') and not prev_line.startswith('#'):
                        break
                    
                    j -= 1
                
                markers_by_test[test_name] = markers
            
            i += 1
    
    except Exception as e:
        print(f"  ⚠️  Error parsing {file_path}: {e}", file=sys.stderr)
    
    return markers_by_test


def count_tests_and_markers_with_pytest(repo_path: str, test_dir: str = "tests") -> Optional[Dict]:
    """
    Use pytest collection to get accurate test and marker counts.
    Returns None if pytest fails.
    """
    all_py_files, test_only_files = find_test_files(repo_path, test_dir)
    
    # First, collect all tests to get total count
    cmd = [
        "python3", "-m", "pytest",
        "--collect-only",
        "-q",
        "--disable-warnings",
        "-o", "addopts=",
        test_dir
    ]
    
    exit_code, stdout, stderr = run_command(cmd, cwd=repo_path, check=False)
    
    # Count tests from collection
    test_count = 0
    for line in stdout.split('\n'):
        if '<Function ' in line or '<Method ' in line:
            test_count += 1
    
    # If pytest failed, return None (don't fallback)
    if exit_code not in [0, 5]:
        print(f"pytest failed (exit {exit_code})", file=sys.stderr)
        return None
    
    # If no tests collected, also return None
    if test_count == 0:
        print(f"no tests collected", file=sys.stderr)
        return None
    
    # Now count markers by collecting tests with specific markers
    marker_counts = {}
    markers_to_check = ['pre_merge', 'post_merge', 'parallel', 'integration', 'skip']
    
    for marker in markers_to_check:
        cmd_marker = [
            "python3", "-m", "pytest",
            "--collect-only",
            "-q",
            "--disable-warnings",
            "-o", "addopts=",
            "-m", marker,
            test_dir
        ]
        
        exit_code_m, stdout_m, stderr_m = run_command(cmd_marker, cwd=repo_path, check=False)
        
        # Count tests with this marker
        count = 0
        for line in stdout_m.split('\n'):
            if '<Function ' in line or '<Method ' in line:
                count += 1
        
        if count > 0:
            marker_counts[marker] = count
    
    # Count unmarked tests (tests with none of the tracked markers)
    marked_total = sum(marker_counts.values())
    # Note: A test can have multiple markers, so this is approximate
    # We'll still calculate unmarked as best effort
    if test_count > marked_total:
        marker_counts['unmarked'] = test_count - marked_total
    
    return {
        'py_files': len(all_py_files),
        'test_files': len(test_only_files),
        'total_tests': test_count,
        'markers': marker_counts
    }


def count_tests_and_markers_fallback(repo_path: str, test_dir: str = "tests") -> Dict:
    """
    Count tests and their markers by parsing test files directly.
    This is more reliable than pytest --collect-only for old commits.
    """
    all_py_files, test_only_files = find_test_files(repo_path, test_dir)
    
    marker_counts = defaultdict(int)
    total_tests = 0
    tests_with_markers = 0
    
    for test_file in test_only_files:
        markers_by_test = parse_test_markers_from_file(test_file)
        
        for test_name, markers in markers_by_test.items():
            total_tests += 1
            
            if markers:
                tests_with_markers += 1
                for marker in markers:
                    marker_counts[marker] += 1
    
    # Count unmarked tests
    unmarked = total_tests - tests_with_markers
    if unmarked > 0:
        marker_counts['unmarked'] = unmarked
    
    return {
        'py_files': len(all_py_files),
        'test_files': len(test_only_files),
        'total_tests': total_tests,
        'markers': dict(marker_counts)
    }


def generate_monthly_dates(start_date: str, end_date: str) -> List[str]:
    """Generate a list of dates for the 1st of each month from start to end, plus end date."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    dates = []
    
    # Start from the 1st of the month containing start_date
    current = datetime(start.year, start.month, 1)
    
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        
        # Move to the 1st of next month
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)
    
    # Add the end date if it's not already included (not the 1st of month)
    if dates and dates[-1] != end_date:
        dates.append(end_date)
    
    return dates


def print_table(stats_history: List[dict], tracked_markers: List[str]) -> None:
    """Print test statistics table in clean format."""
    
    # Print header
    header = "| Date       | py files | pytest files | pytests     |"
    for marker in tracked_markers:
        header += f" {marker:<11} |"
    print(header)
    
    # Print separator
    separator = "|------------|----------|--------------|-------------|"
    for _ in tracked_markers:
        separator += "-------------|"
    print(separator)
    
    # Print data rows
    for stats in stats_history:
        date = stats['date']
        py_files = stats.get('py_files', 0)
        test_files = stats.get('test_files', 0)
        total_tests = stats['total_tests']
        markers = stats.get('markers', {})
        
        # Format row (no deltas)
        row = f"| {date} | {py_files:<8} | {test_files:<12} | {total_tests:<11} |"
        
        for marker in tracked_markers:
            count = markers.get(marker, 0)
            row += f" {count:<11} |"
        
        print(row)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze pytest test growth over time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze test growth for last 6 months
  %(prog)s --repo /path/to/dynamo2 --months 6

  # Analyze test growth for last 2 months
  %(prog)s --repo /path/to/dynamo2 --months 2

  # Custom date range with monthly snapshots
  %(prog)s --repo /path/to/dynamo2 --start 2024-06-01 --end 2025-01-01

  # Track specific markers
  %(prog)s --repo /path/to/dynamo2 --months 12 --markers pre_merge post_merge parallel integration
        """,
    )
    
    parser.add_argument(
        "--repo", 
        type=str, 
        required=True,
        help="Path to git repository"
    )
    parser.add_argument(
        "--months",
        type=int,
        help='Number of months to go back from today (e.g., --months 6 for last 6 months)'
    )
    parser.add_argument(
        "--start",
        type=str,
        help='Start date (format: YYYY-MM-DD)'
    )
    parser.add_argument(
        "--end",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help='End date (format: YYYY-MM-DD, default: today)'
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="tests",
        help="Test directory name (default: tests)"
    )
    parser.add_argument(
        "--markers",
        type=str,
        nargs="+",
        default=["pre_merge", "post_merge", "parallel", "integration", "skip", "unmarked"],
        help="Markers to track (default: pre_merge post_merge parallel integration skip unmarked)"
    )
    parser.add_argument(
        "--json",
        type=str,
        help="Output results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.months and args.start:
        print("Error: Cannot specify both --months and --start", file=sys.stderr)
        sys.exit(1)
    
    if not args.months and not args.start:
        print("Error: Must specify either --months or --start", file=sys.stderr)
        sys.exit(1)
    
    # Calculate start date if using --months
    if args.months:
        today = datetime.now()
        # Go back N months from the 1st of current month
        start = datetime(today.year, today.month, 1)
        for _ in range(args.months):
            if start.month == 1:
                start = datetime(start.year - 1, 12, 1)
            else:
                start = datetime(start.year, start.month - 1, 1)
        args.start = start.strftime("%Y-%m-%d")
    
    repo_path = os.path.abspath(args.repo)
    
    if not os.path.exists(repo_path):
        print(f"Error: Repository path does not exist: {repo_path}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(os.path.join(repo_path, ".git")):
        print(f"Error: Not a git repository: {repo_path}", file=sys.stderr)
        sys.exit(1)
    
    # Save current branch to restore later
    exit_code, current_branch, _ = run_command(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo_path
    )
    current_branch = current_branch.strip()
    
    # Generate date snapshots (1st of each month)
    dates = generate_monthly_dates(args.start, args.end)
    
    print(f"Analyzing test growth from {args.start} to {args.end} (1st of each month)", file=sys.stderr)
    print(f"Repository: {repo_path}", file=sys.stderr)
    print(f"Snapshots: {len(dates)} dates", file=sys.stderr)
    print("", file=sys.stderr)
    
    stats_history = []
    
    try:
        for date in dates:
            print(f"📅 {date}: ", end="", file=sys.stderr, flush=True)
            
            # Get commit at this date
            commit_hash = get_commit_at_date(repo_path, date)
            
            if not commit_hash:
                print(f"No commits found", file=sys.stderr)
                stats_history.append({
                    'date': date,
                    'commit_hash': None,
                    'py_files': 0,
                    'test_files': 0,
                    'total_tests': 0,
                    'markers': {}
                })
                continue
            
            print(f"commit {commit_hash[:8]} ", end="", file=sys.stderr, flush=True)
            
            # Checkout commit
            if not checkout_commit(repo_path, commit_hash):
                print(f"❌ Failed to checkout", file=sys.stderr)
                continue
            
            # Count tests and markers
            stats = count_tests_and_markers_with_pytest(repo_path, args.test_dir)
            
            if not stats:
                print(f" ❌ pytest collection failed, skipping", file=sys.stderr)
                stats_history.append({
                    'date': date,
                    'commit_hash': commit_hash,
                    'py_files': 0,
                    'test_files': 0,
                    'total_tests': 0,
                    'markers': {}
                })
                continue
            
            stats['date'] = date
            stats['commit_hash'] = commit_hash
            
            print(f"✓ {stats['py_files']} py files, {stats['test_files']} test files, {stats['total_tests']} tests", file=sys.stderr)
            
            stats_history.append(stats)
        
    finally:
        # Restore original branch
        print(f"\nRestoring branch: {current_branch}", file=sys.stderr)
        checkout_commit(repo_path, current_branch)
    
    # Print results table
    print("\n")
    print_table(stats_history, args.markers)
    
    # Write JSON if requested
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(stats_history, f, indent=2)
        print(f"\n✓ JSON output written to: {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
