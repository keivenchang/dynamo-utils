#!/usr/bin/env python3
"""Test script to measure actual API usage without interference"""
import subprocess
import requests
import os
import sys

def get_rate_limit():
    """Get GitHub rate limit using direct API call"""
    token_file = os.path.expanduser('~/.config/github-token')
    if not os.path.exists(token_file):
        # Try gh CLI config
        gh_config = os.path.expanduser('~/.config/gh/hosts.yml')
        if os.path.exists(gh_config):
            import yaml
            with open(gh_config) as f:
                config = yaml.safe_load(f)
                token = config.get('github.com', {}).get('oauth_token')
        else:
            print("No token found")
            sys.exit(1)
    else:
        with open(token_file) as f:
            token = f.read().strip()
    
    resp = requests.get(
        'https://api.github.com/rate_limit',
        headers={'Authorization': f'token {token}'}
    )
    data = resp.json()
    return data['resources']['core']['remaining']

def main():
    # Get quota BEFORE (1 API call)
    before = get_rate_limit()
    print(f'BEFORE: remaining={before}')
    
    # Run subprocess (should make N API calls internally)
    result = subprocess.run([
        'python3', 'html_pages/show_commit_history.py',
        '--repo-path', '/home/keivenc/dynamo/dynamo_ci',
        '--output', '/tmp/test_commits.html',
        '--max-commits', '5',
        '--skip-gitlab-api'
    ], capture_output=True, text=True, cwd='/home/keivenc/dynamo/dynamo-utils.dev')
    
    if result.returncode != 0:
        print(f'ERROR: Script failed')
        print(result.stderr[:500])
        sys.exit(1)
    
    # Get quota AFTER (1 API call)
    after = get_rate_limit()
    print(f'AFTER: remaining={after}')
    
    consumed = before - after
    print(f'\nTOTAL CONSUMED BY TEST: {consumed} (should be subprocess calls + 2 for before/after)')
    print(f'SUBPROCESS CONSUMED: {consumed - 2}')
    
    # Parse HTML to see what subprocess reported
    import re
    with open('/tmp/test_commits.html') as f:
        html = f.read()
    
    pattern = r'<td class="k">github\.rest\.calls</td>\s*<td class="v">\s*(?:<[^>]*>)?\s*(\d+)'
    match = re.search(pattern, html)
    if match:
        reported = int(match.group(1))
        print(f'SUBPROCESS REPORTED: {reported} API calls')
        print(f'\nDISCREPANCY: {consumed - 2 - reported} calls not tracked')
    
    # Show subprocess API calls
    print('\n=== Subprocess API calls logged ===')
    for line in result.stderr.split('\n'):
        if '[API CALL' in line:
            print(line)

if __name__ == '__main__':
    main()
