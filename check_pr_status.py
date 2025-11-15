#!/usr/bin/env python3
"""Show detailed check status breakdown for PRs with required/non-required separation."""

import json
import subprocess
import sys
from collections import defaultdict


def get_pr_check_status(pr_number: int, repo: str = "ai-dynamo/dynamo"):
    """Get check status breakdown for a PR."""
    try:
        result = subprocess.run(
            ['gh', 'pr', 'view', str(pr_number), '--repo', repo, '--json', 'statusCheckRollup'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            return None
        
        data = json.loads(result.stdout)
        checks = data.get('statusCheckRollup', [])
        
        stats = {
            'running': {'required': [], 'not_required': []},
            'passed': {'required': [], 'not_required': []},
            'failed': {'required': [], 'not_required': []},
        }
        
        for check in checks:
            name = check.get('name', '')
            status = check.get('status', '').upper()
            conclusion = check.get('conclusion', '').upper()
            is_required = check.get('isRequired', False)
            
            required_key = 'required' if is_required else 'not_required'
            
            # Determine check state
            if status == 'COMPLETED':
                if conclusion == 'SUCCESS':
                    stats['passed'][required_key].append(name)
                elif conclusion in ('FAILURE', 'TIMED_OUT', 'CANCELLED'):
                    stats['failed'][required_key].append(name)
                # Skip neutral, skipped, etc
            elif status in ('IN_PROGRESS', 'QUEUED', 'PENDING'):
                stats['running'][required_key].append(name)
        
        return stats
        
    except Exception as e:
        print(f"Error getting PR status: {e}", file=sys.stderr)
        return None


def print_pr_stats(pr_number: int, repo: str = "ai-dynamo/dynamo"):
    """Print check status breakdown for a PR."""
    print(f"PR#{pr_number}:")
    
    stats = get_pr_check_status(pr_number, repo)
    if not stats:
        print("  Could not retrieve check status")
        return
    
    # Count totals
    running_req = len(stats['running']['required'])
    running_not_req = len(stats['running']['not_required'])
    passed_req = len(stats['passed']['required'])
    passed_not_req = len(stats['passed']['not_required'])
    failed_req = len(stats['failed']['required'])
    failed_not_req = len(stats['failed']['not_required'])
    
    # Print summary
    print(f"  Running: {running_req + running_not_req} (required: {running_req}, not required: {running_not_req})")
    print(f"  Passed:  {passed_req + passed_not_req} (required: {passed_req}, not required: {passed_not_req})")
    print(f"  Failed:  {failed_req + failed_not_req} (required: {failed_req}, not required: {failed_not_req})")
    
    # Show details if there are running or failed checks
    if running_req > 0:
        print(f"\n  Running (required):")
        for name in stats['running']['required'][:10]:
            print(f"    - {name}")
        if len(stats['running']['required']) > 10:
            print(f"    ... +{len(stats['running']['required']) - 10} more")
    
    if running_not_req > 0:
        print(f"\n  Running (not required):")
        for name in stats['running']['not_required'][:5]:
            print(f"    - {name}")
        if len(stats['running']['not_required']) > 5:
            print(f"    ... +{len(stats['running']['not_required']) - 5} more")
    
    if failed_req > 0:
        print(f"\n  Failed (required):")
        for name in stats['failed']['required']:
            print(f"    - {name}")
    
    if failed_not_req > 0:
        print(f"\n  Failed (not required):")
        for name in stats['failed']['not_required'][:5]:
            print(f"    - {name}")
        if len(stats['failed']['not_required']) > 5:
            print(f"    ... +{len(stats['failed']['not_required']) - 5} more")


def main():
    if len(sys.argv) < 2:
        print("Usage: check_pr_status.py <PR_NUMBER> [<PR_NUMBER> ...]")
        print("\nExample: check_pr_status.py 3687 3804")
        sys.exit(1)
    
    pr_numbers = [int(arg) for arg in sys.argv[1:]]
    
    for i, pr_number in enumerate(pr_numbers):
        if i > 0:
            print()
        print_pr_stats(pr_number)


if __name__ == '__main__':
    main()
