#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Backfill status marker files for existing HTML build reports.

This script traverses all HTML report files in the logs directory and creates
corresponding status marker files (.PASSED, .FAILED, .RUNNING, .KILLED) based
on the HTML content.
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Optional


# Status markers (same as in build_images.py)
MARKER_PASSED = 'PASSED'
MARKER_FAILED = 'FAILED'
MARKER_RUNNING = 'RUNNING'
MARKER_KILLED = 'KILLED'


def parse_html_report_status(html_file: Path) -> Optional[str]:
    """
    Parse HTML report file to determine overall build status.

    Args:
        html_file: Path to HTML report file

    Returns:
        Status marker (PASSED, FAILED, RUNNING, KILLED) or None if cannot determine
    """
    try:
        html_content = html_file.read_text()

        # Look for the overall status in the HTML
        # Format: <title>DynamoDockerBuilder - sha - STATUS</title>
        title_match = re.search(r'<title>.*?\s+-\s+(.*?)</title>', html_content)
        if title_match:
            status_text = title_match.group(1).upper()

            if 'FAILED' in status_text or 'FAIL' in status_text:
                return MARKER_FAILED
            elif 'INTERRUPTED' in status_text or 'KILLED' in status_text:
                return MARKER_KILLED
            elif 'IN PROGRESS' in status_text or 'RUNNING' in status_text or 'BUILDING' in status_text:
                return MARKER_RUNNING
            elif 'PASSED' in status_text or 'PASS' in status_text or 'SUCCESS' in status_text:
                return MARKER_PASSED

        # Fallback: Look for summary boxes in the HTML
        # Check for failed/killed counts
        failed_match = re.search(r'<div class="number">(\d+)</div>\s*<div class="label">Failed</div>', html_content)
        killed_match = re.search(r'<div class="number">(\d+)</div>\s*<div class="label">Killed</div>', html_content)

        if failed_match:
            failed_count = int(failed_match.group(1))
            if failed_count > 0:
                return MARKER_FAILED

        if killed_match:
            killed_count = int(killed_match.group(1))
            if killed_count > 0:
                return MARKER_KILLED

        # If no failures or kills, assume passed
        return MARKER_PASSED

    except Exception as e:
        logging.warning(f"Failed to parse HTML file {html_file}: {e}")
        return None


def create_marker_file(html_file: Path, status_marker: str) -> None:
    """
    Create status marker file for HTML report.

    Args:
        html_file: Path to HTML report file
        status_marker: Status marker (PASSED, FAILED, RUNNING, KILLED)
    """
    # Remove old marker files
    for marker in [MARKER_PASSED, MARKER_FAILED, MARKER_RUNNING, MARKER_KILLED]:
        old_marker = html_file.with_suffix(f'.{marker}')
        if old_marker.exists():
            old_marker.unlink()

    # Create new marker file
    marker_file = html_file.with_suffix(f'.{status_marker}')
    marker_file.touch()
    logging.info(f"Created {marker_file.name} for {html_file.name}")


def backfill_report_markers(logs_dir: Path, dry_run: bool = False) -> int:
    """
    Backfill status marker files for all HTML reports.

    Args:
        logs_dir: Base logs directory to traverse
        dry_run: If True, only report what would be done without creating files

    Returns:
        Number of marker files created
    """
    if not logs_dir.exists():
        logging.error(f"Logs directory does not exist: {logs_dir}")
        return 0

    # Find all HTML report files
    html_files = list(logs_dir.glob('**/*.report.html'))

    if not html_files:
        logging.info(f"No HTML report files found in {logs_dir}")
        return 0

    logging.info(f"Found {len(html_files)} HTML report files")

    created_count = 0
    skipped_count = 0

    for html_file in sorted(html_files):
        # Check if marker already exists
        existing_marker = None
        for marker in [MARKER_PASSED, MARKER_FAILED, MARKER_RUNNING, MARKER_KILLED]:
            marker_file = html_file.with_suffix(f'.{marker}')
            if marker_file.exists():
                existing_marker = marker
                break

        if existing_marker:
            logging.debug(f"Marker already exists for {html_file.name}: {existing_marker}")
            skipped_count += 1
            continue

        # Parse HTML to determine status
        status_marker = parse_html_report_status(html_file)

        if status_marker is None:
            logging.warning(f"Could not determine status for {html_file}")
            skipped_count += 1
            continue

        if dry_run:
            logging.info(f"[DRY RUN] Would create {status_marker} marker for {html_file.name}")
        else:
            create_marker_file(html_file, status_marker)

        created_count += 1

    logging.info(f"Created {created_count} marker files, skipped {skipped_count}")
    return created_count


def main():
    parser = argparse.ArgumentParser(
        description='Backfill status marker files for existing HTML build reports'
    )
    parser.add_argument(
        '--logs-dir',
        type=Path,
        default=Path.home() / 'nvidia' / 'dynamo_ci' / 'logs',
        help='Base logs directory to traverse (default: ~/nvidia/dynamo_ci/logs)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without creating files'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logging.info(f"Backfilling report markers in {args.logs_dir}")

    count = backfill_report_markers(args.logs_dir, dry_run=args.dry_run)

    if args.dry_run:
        logging.info(f"[DRY RUN] Would have created {count} marker files")
    else:
        logging.info(f"Successfully created {count} marker files")

    return 0


if __name__ == '__main__':
    sys.exit(main())
