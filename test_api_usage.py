#!/usr/bin/env python3
"""Test script to measure actual API usage of show_commit_history.py"""
import sys
import os
import subprocess

# Add common_github to path
sys.path.insert(0, os.path.dirname(__file__))

from common_github import GitHubAPIClient

def main():
    # Use the same token loading mechanism as GitHubAPIClient
    client = GitHubAPIClient()  # Will auto-load from ~/.config/github-token or ~/.config/gh/hosts.yml
    
    if not client.has_token():
        print("ERROR: No GitHub token found. Run 'gh auth login' or create ~/.config/github-token")
        sys.exit(1)
    
    # Get rate limit BEFORE
    before = client.get_core_rate_limit_info()
    if not before:
        print("ERROR: Could not get rate limit info")
        sys.exit(1)
    
    used_before = before.get('limit', 5000) - before.get('remaining', 0)
    print(f"BEFORE: remaining={before['remaining']}, used={used_before}")
    
    # Run show_commit_history.py
    print("\nRunning show_commit_history.py with debug...")
    result = subprocess.run([
        'python3', 'html_pages/show_commit_history.py',
        '--repo-path', '/home/keivenc/dynamo/dynamo_ci',
        '--output', '/tmp/test_commit_history.html',
        '--max-commits', '5',
        '--skip-gitlab-api',
        '--debug'  # Enable debug mode
    ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
    
    if result.returncode != 0:
        print(f"ERROR: Script failed with code {result.returncode}")
        print("STDERR:", result.stderr)
        print("STDOUT:", result.stdout)
        sys.exit(1)
    
    # Print debug output if available
    if result.stderr:
        print("\n=== Debug output (stderr) ===")
        for line in result.stderr.split('\n')[:50]:  # First 50 lines
            if 'REST GET' in line or 'github' in line.lower():
                print(line)
    
    print("Script completed.")
    
    # Get rate limit AFTER
    after = client.get_core_rate_limit_info()
    if not after:
        print("ERROR: Could not get rate limit info after")
        sys.exit(1)
    
    used_after = after.get('limit', 5000) - after.get('remaining', 0)
    print(f"\nAFTER: remaining={after['remaining']}, used={used_after}")
    
    # Calculate delta
    delta = used_after - used_before
    print(f"\n{'='*60}")
    print(f"ACTUAL API CALLS USED: {delta}")
    print(f"{'='*60}")
    
    # Now extract statistics from the generated HTML
    print("\nExtracting statistics from generated HTML...")
    import re
    with open('/tmp/test_commit_history.html', 'r') as f:
        html = f.read()
    
    # Extract all github.rest metrics
    pattern = r'<td class="k">(github\.rest[^<]*)</td>\s*<td class="v">\s*(?:<[^>]*>)?\s*([^<\n]+)'
    matches = re.findall(pattern, html)
    
    print("\n=== All github.rest metrics ===")
    reported_total = None
    for metric, value in matches:
        value = value.strip()
        if value and not value.startswith('<'):
            print(f'{metric}: {value}')
            if metric == 'github.rest.calls':
                try:
                    reported_total = int(value)
                except:
                    pass
    
    if reported_total is not None:
        print(f"\n{'='*60}")
        print(f"REPORTED TOTAL: {reported_total} API calls")
        print(f"ACTUAL TOTAL: {delta} API calls")
        print(f"DISCREPANCY: {delta - reported_total} API calls not accounted for!")
        print(f"{'='*60}")
    else:
        print("Could not find github.rest.calls in HTML")

if __name__ == '__main__':
    main()
