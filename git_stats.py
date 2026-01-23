#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Git repository statistics analyzer.

Analyzes git commit history to provide statistics about contributors,
including number of commits, lines changed, and rankings.
"""

import argparse
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple


@dataclass
class ContributorStats:
    """Statistics for a single contributor."""

    name: str
    email: str
    commits: int = 0
    lines_added: int = 0
    lines_deleted: int = 0

    @property
    def lines_changed(self) -> int:
        """Total lines changed (added + deleted)."""
        return self.lines_added + self.lines_deleted

    @property
    def net_lines(self) -> int:
        """Net lines (added - deleted)."""
        return self.lines_added - self.lines_deleted


def run_git_command(cmd: List[str], cwd: str = None) -> str:
    """Run a git command and return output."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, cwd=cwd
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running git command: {e}", file=sys.stderr)
        print(f"Command: {' '.join(cmd)}", file=sys.stderr)
        print(f"Error output: {e.stderr}", file=sys.stderr)
        sys.exit(1)


def get_time_range_arg(days: int = None, since: str = None, until: str = None) -> List[str]:
    """Generate git time range arguments."""
    args = []
    if days:
        since_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        args.extend(["--since", since_date])
    elif since:
        args.extend(["--since", since])
    if until:
        args.extend(["--until", until])
    return args


def get_contributors(time_args: List[str], cwd: str = None) -> Dict[str, ContributorStats]:
    """Get all contributors and their commit counts."""
    contributors: Dict[str, ContributorStats] = {}

    # Get commit log with author info
    cmd = ["git", "log", "--format=%ae|%an"] + time_args
    output = run_git_command(cmd, cwd=cwd)

    if not output:
        print("No commits found in the specified time range.", file=sys.stderr)
        return contributors

    for line in output.split("\n"):
        if not line:
            continue
        email, name = line.split("|", 1)
        key = email.lower()

        if key not in contributors:
            contributors[key] = ContributorStats(name=name, email=email)
        contributors[key].commits += 1

    return contributors


def get_line_stats(contributors: Dict[str, ContributorStats], time_args: List[str], cwd: str = None) -> None:
    """Update contributors with line change statistics."""
    # Get detailed stats per author
    cmd = ["git", "log", "--numstat", "--format=%ae"] + time_args
    output = run_git_command(cmd, cwd=cwd)

    current_author = None
    for line in output.split("\n"):
        if not line:
            continue

        # Check if line is an email (author line)
        if "@" in line and "\t" not in line:
            current_author = line.lower()
            continue

        # Parse numstat line: "added\tdeleted\tfilename"
        if current_author and "\t" in line:
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    added = int(parts[0]) if parts[0] != "-" else 0
                    deleted = int(parts[1]) if parts[1] != "-" else 0

                    if current_author in contributors:
                        contributors[current_author].lines_added += added
                        contributors[current_author].lines_deleted += deleted
                except ValueError:
                    # Skip binary files or malformed lines
                    continue


def print_statistics(contributors: Dict[str, ContributorStats], time_desc: str) -> None:
    """Print formatted statistics."""
    if not contributors:
        print("No contributors found.")
        return

    stats_list = sorted(contributors.values(), key=lambda x: x.commits, reverse=True)

    total_commits = sum(c.commits for c in stats_list)
    total_lines_added = sum(c.lines_added for c in stats_list)
    total_lines_deleted = sum(c.lines_deleted for c in stats_list)
    total_lines_changed = sum(c.lines_changed for c in stats_list)

    print(f"\n{'=' * 80}")
    print(f"Git Repository Statistics - {time_desc}")
    print(f"{'=' * 80}\n")

    print(f"Number of unique contributors: {len(contributors)}")
    print(f"Total commits: {total_commits}")
    print(f"Total lines added: {total_lines_added:,}")
    print(f"Total lines deleted: {total_lines_deleted:,}")
    print(f"Total lines changed: {total_lines_changed:,}")
    print()

    if len(contributors) > 0:
        print(f"Average commits per person: {total_commits / len(contributors):.1f}")
        print(f"Average lines added per person: {total_lines_added / len(contributors):.1f}")
        print(f"Average lines deleted per person: {total_lines_deleted / len(contributors):.1f}")
        print(f"Average lines changed per person: {total_lines_changed / len(contributors):.1f}")

    print(f"\n{'=' * 80}")
    print("Contributor Rankings (by commits)")
    print(f"{'=' * 80}\n")

    print(f"{'Rank':<6} {'Name':<30} {'Email':<35} {'Commits':<8} {'Added':<10} {'Deleted':<10} {'Changed':<10}")
    print("-" * 110)

    for rank, contributor in enumerate(stats_list, 1):
        print(
            f"{rank:<6} {contributor.name[:29]:<30} {contributor.email[:34]:<35} "
            f"{contributor.commits:<8} {contributor.lines_added:<10,} "
            f"{contributor.lines_deleted:<10,} {contributor.lines_changed:<10,}"
        )

    print("\n" + "=" * 80)
    print("Contributor Rankings (by lines changed)")
    print("=" * 80 + "\n")

    stats_by_lines = sorted(contributors.values(), key=lambda x: x.lines_changed, reverse=True)

    print(f"{'Rank':<6} {'Name':<30} {'Email':<35} {'Commits':<8} {'Added':<10} {'Deleted':<10} {'Changed':<10}")
    print("-" * 110)

    for rank, contributor in enumerate(stats_by_lines, 1):
        print(
            f"{rank:<6} {contributor.name[:29]:<30} {contributor.email[:34]:<35} "
            f"{contributor.commits:<8} {contributor.lines_added:<10,} "
            f"{contributor.lines_deleted:<10,} {contributor.lines_changed:<10,}"
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze git repository contributor statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Statistics for last 30 days
  %(prog)s --days 30

  # Statistics for last 7 days
  %(prog)s --days 7

  # Statistics since a specific date
  %(prog)s --since "2025-01-01"

  # Statistics for a date range
  %(prog)s --since "2025-01-01" --until "2025-01-31"

  # All time statistics
  %(prog)s
        """,
    )

    parser.add_argument(
        "--days", type=int, help="Number of days back from today to analyze"
    )
    parser.add_argument(
        "--since", type=str, help='Start date (format: "YYYY-MM-DD" or "N days ago")'
    )
    parser.add_argument(
        "--until", type=str, help='End date (format: "YYYY-MM-DD" or "N days ago")'
    )
    parser.add_argument(
        "--repo", type=str, help="Path to git repository (defaults to current directory)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.days and args.since:
        print("Error: Cannot specify both --days and --since", file=sys.stderr)
        sys.exit(1)

    # Build time range description
    if args.days:
        time_desc = f"Last {args.days} days"
    elif args.since or args.until:
        parts = []
        if args.since:
            parts.append(f"since {args.since}")
        if args.until:
            parts.append(f"until {args.until}")
        time_desc = " ".join(parts).capitalize()
    else:
        time_desc = "All time"

    # Get time range arguments for git
    time_args = get_time_range_arg(days=args.days, since=args.since, until=args.until)

    # Collect statistics
    repo_path = args.repo if args.repo else None
    print("Analyzing repository...", file=sys.stderr)
    contributors = get_contributors(time_args, cwd=repo_path)

    if contributors:
        print("Calculating line statistics...", file=sys.stderr)
        get_line_stats(contributors, time_args, cwd=repo_path)

    # Print results
    print_statistics(contributors, time_desc)


if __name__ == "__main__":
    main()

