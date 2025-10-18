#!/usr/bin/env python3
"""
Script to check branches in dynamo* directories and find corresponding PRs on GitHub.

This script:
1. Scans all dynamo* directories in the parent directory
2. Lists all local branches in each directory
3. Checks if branches exist on the upstream GitHub repository
4. Finds and displays PR URLs for branches that have corresponding pull requests

Usage:
    ./check_dynamo_branches.py [--verbose] [--github-token TOKEN]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def get_github_token_from_cli() -> Optional[str]:
    """Get GitHub token from GitHub CLI configuration"""
    if not HAS_YAML:
        return None
        
    try:
        gh_config_path = Path.home() / '.config' / 'gh' / 'hosts.yml'
        if gh_config_path.exists():
            with open(gh_config_path, 'r') as f:
                config = yaml.safe_load(f)
                if config and 'github.com' in config:
                    github_config = config['github.com']
                    # Try to get token from global config first
                    if 'oauth_token' in github_config:
                        return github_config['oauth_token']
                    # Try to get token from users config
                    if 'users' in github_config:
                        for user, user_config in github_config['users'].items():
                            if 'oauth_token' in user_config:
                                return user_config['oauth_token']
    except Exception:
        pass
    return None


class GitHubAPI:
    """Simple GitHub API client for checking branches and PRs"""
    
    def __init__(self, token: Optional[str] = None, verbose: bool = False):
        self.base_url = "https://api.github.com"
        # Token priority: 1) provided token, 2) environment variable, 3) GitHub CLI config
        self.token = token or os.environ.get('GITHUB_TOKEN') or get_github_token_from_cli()
        self.verbose = verbose
        self.api_call_times: List[Tuple[str, float]] = []
        # Cache for branch protection required checks per (repo, base_branch)
        self._protection_required_checks_cache: Dict[Tuple[str, str], Set[str]] = {}
        # Cache for PR details to avoid refetching, keyed by (repo, pr_number)
        self._pr_details_cache: Dict[Tuple[str, int], Dict] = {}
        # Cache for rulesets per repo (rulesets are repo-level, not branch-specific)
        self._rulesets_cache: Dict[str, Set[str]] = {}
        
    def _make_request(self, endpoint: str) -> Optional[Dict]:
        """Make a request to GitHub API
        
        Args:
            endpoint: API endpoint path, e.g., "/repos/owner/repo/pulls/123"
        
        Returns:
            Dict with API response, or None on error
            Example: {
                "id": 123,
                "number": 456,
                "title": "Fix bug",
                ...
            }
        """
        start_time = time.time()
        url = urljoin(self.base_url, endpoint)
        req = Request(url)
        
        if self.token:
            req.add_header('Authorization', f'token {self.token}')
        req.add_header('Accept', 'application/vnd.github.v3+json')
        req.add_header('User-Agent', 'dynamo-branch-checker/1.0')
        
        try:
            with urlopen(req) as response:
                result = json.loads(response.read().decode())
                elapsed = time.time() - start_time
                self.api_call_times.append((endpoint, elapsed))
                if self.verbose:
                    print(f"  API: {endpoint} took {elapsed:.2f}s", file=sys.stderr)
                return result
        except HTTPError as e:
            if e.code == 404:
                return None
            elif e.code == 403:
                print(f"⚠️  GitHub API rate limit exceeded. Consider using --github-token", file=sys.stderr)
                return None
            else:
                print(f"⚠️  GitHub API error {e.code}: {e.reason}", file=sys.stderr)
                return None
        except URLError as e:
            print(f"⚠️  Network error: {e.reason}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"⚠️  Unexpected error: {e}", file=sys.stderr)
            return None
    
    def branch_exists(self, repo: str, branch_name: str) -> bool:
        """Check if a branch exists on GitHub
        
        Args:
            repo: Repository in format "owner/repo", e.g., "ai-dynamo/dynamo"
            branch_name: Branch name, e.g., "main" or "feature/my-branch"
        
        Returns:
            True if branch exists, False otherwise
        
        API Response Example:
            {
                "name": "main",
                "commit": {"sha": "abc123...", "url": "..."},
                "protected": true
            }
        """
        # URL encode the branch name to handle special characters
        encoded_branch = quote(branch_name, safe='')
        endpoint = f"/repos/{repo}/branches/{encoded_branch}"
        result = self._make_request(endpoint)
        return result is not None
    
    def find_prs_for_branch(self, repo: str, branch_name: str) -> List[Dict]:
        """Find PRs associated with a branch
        
        Args:
            repo: Repository in format "owner/repo", e.g., "ai-dynamo/dynamo"
            branch_name: Branch name, e.g., "feature/my-branch"
        
        Returns:
            List of PR dicts, or empty list if none found
        
        API Response Example:
            [
                {
                    "number": 3266,
                    "title": "feat: implement...",
                    "state": "open",
                    "head": {"ref": "feature/my-branch", "sha": "abc123..."},
                    "base": {"ref": "main"},
                    "html_url": "https://github.com/...",
                    ...
                }
            ]
        """
        # Search for PRs with this branch as head
        # URL encode the branch name to handle special characters
        encoded_branch = quote(branch_name, safe='')
        owner = repo.split('/')[0] if '/' in repo else ''
        endpoint = f"/repos/{repo}/pulls?head={owner}:{encoded_branch}&state=all"
        result = self._make_request(endpoint)
        
        if result is None:
            return []
        
        return result
    
    def search_prs_by_branch(self, repo: str, branch_name: str) -> List[Dict]:
        """Search for PRs that might be related to this branch
        
        Args:
            repo: Repository in format "owner/repo", e.g., "ai-dynamo/dynamo"
            branch_name: Branch name to search for
        
        Returns:
            List of PR dicts matching the branch, or empty list
        
        API Response Example (search/issues):
            {
                "total_count": 1,
                "items": [
                    {
                        "number": 3266,
                        "title": "...",
                        "state": "open",
                        "pull_request": {...},
                        ...
                    }
                ]
            }
        """
        # GitHub search API for PRs mentioning the branch
        query = f"repo:{repo} type:pr {branch_name}"
        # URL encode the entire query
        encoded_query = quote(query, safe='')
        endpoint = f"/search/issues?q={encoded_query}"
        result = self._make_request(endpoint)
        
        if result is None or 'items' not in result:
            return []
        
        # Filter to only include PRs where head branch matches.
        # Fetch PR details in parallel to avoid sequential latency.
        items = result['items']
        pr_numbers = [item.get('number') for item in items if item.get('number')]
        pr_details_list: List[Dict] = []

        # Use cache where possible, and fetch the rest concurrently
        remaining_numbers: List[int] = []
        for num in pr_numbers:
            cached = self._pr_details_cache.get((repo, num))
            if cached is not None:
                pr_details_list.append(cached)
            else:
                remaining_numbers.append(num)

        max_workers = min(8, len(remaining_numbers)) or 0
        if max_workers > 0:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {}
                for num in remaining_numbers:
                    pr_endpoint = f"/repos/{repo}/pulls/{num}"
                    future = executor.submit(self._make_request, pr_endpoint)
                    future_map[future] = num
                for future in as_completed(future_map):
                    num = future_map[future]
                    data = future.result()
                    if data is not None:
                        self._pr_details_cache[(repo, num)] = data
                        pr_details_list.append(data)
        
        # Now filter by exact head ref match
        filtered_prs = [d for d in pr_details_list if d and d.get('head', {}).get('ref') == branch_name]
        
        return filtered_prs
    
    def get_pr_reviews(self, repo: str, pr_number: int) -> Optional[Dict]:
        """Get review status for a PR
        
        Args:
            repo: Repository in format "owner/repo", e.g., "ai-dynamo/dynamo"
            pr_number: PR number, e.g., 3266
        
        Returns:
            Dict with review summary, or None on error
            Example: {
                "approved": 2,
                "changes_requested": 0,
                "commented": 1,
                "has_approvals": True,
                "has_changes_requested": False
            }
        
        API Response Example:
            [
                {
                    "id": 123,
                    "user": {"login": "reviewer1"},
                    "state": "APPROVED",
                    "submitted_at": "2025-01-01T00:00:00Z",
                    ...
                },
                {
                    "id": 124,
                    "user": {"login": "reviewer2"},
                    "state": "COMMENTED",
                    ...
                }
            ]
        """
        endpoint = f"/repos/{repo}/pulls/{pr_number}/reviews"
        reviews = self._make_request(endpoint)
        
        if reviews is None:
            return None
        
        # Analyze reviews to get the latest status
        review_summary = {
            'approved': 0,
            'changes_requested': 0,
            'commented': 0,
            'reviewers': set(),
            'has_approvals': False,
            'has_changes_requested': False
        }
        
        # Track latest review state per reviewer
        reviewer_states = {}
        
        for review in reviews:
            if review.get('state') in ['APPROVED', 'CHANGES_REQUESTED', 'COMMENTED']:
                reviewer = review.get('user', {}).get('login', 'unknown')
                state = review.get('state')
                submitted_at = review.get('submitted_at', '')
                
                # Keep only the latest review from each reviewer
                if reviewer not in reviewer_states or submitted_at > reviewer_states[reviewer]['submitted_at']:
                    reviewer_states[reviewer] = {
                        'state': state,
                        'submitted_at': submitted_at
                    }
        
        # Count final states
        for reviewer, data in reviewer_states.items():
            state = data['state']
            review_summary['reviewers'].add(reviewer)
            
            if state == 'APPROVED':
                review_summary['approved'] += 1
                review_summary['has_approvals'] = True
            elif state == 'CHANGES_REQUESTED':
                review_summary['changes_requested'] += 1
                review_summary['has_changes_requested'] = True
            elif state == 'COMMENTED':
                review_summary['commented'] += 1
        
        return review_summary
    
    def _get_required_checks_for_base_branch(self, repo: str, base_branch: str) -> Set[str]:
        """Return required status check contexts for a base branch, with caching.
        
        Args:
            repo: Repository in format "owner/repo", e.g., "ai-dynamo/dynamo"
            base_branch: Base branch name, e.g., "main"
        
        Returns:
            Set of required check names, e.g., {"copyright-checks", "pre-commit", "DCO"}
        
        API Response Examples:
            Rulesets API (/repos/{repo}/rulesets):
                [
                    {
                        "id": 4130136,
                        "name": "Required PR Checks",
                        "target": "branch",
                        ...
                    }
                ]
            
            Ruleset Detail API (/repos/{repo}/rulesets/{id}):
                {
                    "id": 4130136,
                    "rules": [
                        {
                            "type": "required_status_checks",
                            "parameters": {
                                "required_status_checks": [
                                    {"context": "copyright-checks"},
                                    {"context": "pre-commit"}
                                ]
                            }
                        }
                    ]
                }
            
            Branch Protection API (/repos/{repo}/branches/{branch}/protection):
                {
                    "required_status_checks": {
                        "contexts": ["test", "build"],
                        "checks": [{"context": "lint"}]
                    }
                }
        """
        cache_key = (repo, base_branch)
        if cache_key in self._protection_required_checks_cache:
            return self._protection_required_checks_cache[cache_key]

        required_checks: Set[str] = set()
        
        # Try the rulesets API first (newer, more reliable) - cached per repo
        if repo in self._rulesets_cache:
            required_checks = self._rulesets_cache[repo].copy()
        else:
            rulesets_endpoint = f"/repos/{repo}/rulesets"
            rulesets_data = self._make_request(rulesets_endpoint)
            if rulesets_data and isinstance(rulesets_data, list):
                for ruleset in rulesets_data:
                    ruleset_id = ruleset.get('id')
                    if ruleset_id:
                        # Get detailed ruleset
                        detailed_endpoint = f"/repos/{repo}/rulesets/{ruleset_id}"
                        detailed = self._make_request(detailed_endpoint)
                        if detailed and 'rules' in detailed:
                            for rule in detailed['rules']:
                                if rule.get('type') == 'required_status_checks':
                                    params = rule.get('parameters', {})
                                    for check in params.get('required_status_checks', []):
                                        context = check.get('context', '')
                                        if context:
                                            required_checks.add(context)
            # Cache the rulesets result for this repo
            self._rulesets_cache[repo] = required_checks.copy()
        
        # If rulesets didn't work, try the full protection endpoint (branch-specific)
        if not required_checks:
            protection_endpoint = f"/repos/{repo}/branches/{base_branch}/protection"
            protection_data = self._make_request(protection_endpoint)
            if protection_data and 'required_status_checks' in protection_data:
                checks_data = protection_data['required_status_checks'].get('checks', [])
                for check in checks_data:
                    context = check.get('context', '')
                    if context:
                        required_checks.add(context)
                contexts = protection_data['required_status_checks'].get('contexts', [])
                required_checks.update(contexts)
            else:
                # If full protection fails (404), try just the required_status_checks endpoint
                status_checks_endpoint = f"/repos/{repo}/branches/{base_branch}/protection/required_status_checks"
                status_checks_data = self._make_request(status_checks_endpoint)
                if status_checks_data:
                    checks_data = status_checks_data.get('checks', [])
                    for check in checks_data:
                        context = check.get('context', '')
                        if context:
                            required_checks.add(context)
                    contexts = status_checks_data.get('contexts', [])
                    required_checks.update(contexts)

        self._protection_required_checks_cache[cache_key] = required_checks
        return required_checks

    def get_pr_status_checks(self, repo: str, pr_number: int, pr_data: Optional[Dict] = None) -> Optional[Dict]:
        """Get CI status checks for a PR
        
        Args:
            repo: Repository in format "owner/repo", e.g., "ai-dynamo/dynamo"
            pr_number: PR number, e.g., 3266
            pr_data: Optional pre-fetched PR data to avoid extra API call
        
        Returns:
            Dict with check status summary, or None on error
            Example: {
                "total_checks": 25,
                "passed": 23,
                "failed": 1,
                "pending": 1,
                "required_failed": 1,
                "optional_failed": 0,
                "has_failures": True,
                "has_required_failures": True,
                "failed_checks": ["copyright-checks"],
                "required_failed_checks": ["copyright-checks"]
            }
        
        API Response Examples:
            PR Detail (/repos/{repo}/pulls/{number}):
                {
                    "number": 3266,
                    "head": {"sha": "6bef250...", "ref": "my-branch"},
                    "base": {"ref": "main"},
                    ...
                }
            
            Check Runs (/repos/{repo}/commits/{sha}/check-runs):
                {
                    "total_count": 25,
                    "check_runs": [
                        {
                            "name": "copyright-checks",
                            "status": "completed",
                            "conclusion": "failure",
                            "started_at": "...",
                            ...
                        },
                        {
                            "name": "pre-commit",
                            "status": "completed",
                            "conclusion": "success",
                            ...
                        }
                    ]
                }
            
            Combined Status (/repos/{repo}/commits/{sha}/status):
                {
                    "state": "pending",
                    "statuses": [
                        {
                            "context": "ci/test",
                            "state": "success",
                            ...
                        }
                    ]
                }
        """
        # First get the PR details to get the head SHA and base branch (use cache if available)
        if pr_data is None:
            cache_key = (repo, pr_number)
            if cache_key in self._pr_details_cache:
                pr_data = self._pr_details_cache[cache_key]
            else:
                pr_endpoint = f"/repos/{repo}/pulls/{pr_number}"
                pr_data = self._make_request(pr_endpoint)
                if pr_data:
                    self._pr_details_cache[cache_key] = pr_data
        
        if not pr_data or 'head' not in pr_data:
            return None
        
        head_sha = pr_data['head']['sha']
        base_branch = pr_data.get('base', {}).get('ref', 'main')
        
        # Get required checks (cached per base branch)
        required_checks = self._get_required_checks_for_base_branch(repo, base_branch)
        
        # Get combined status for the commit
        status_endpoint = f"/repos/{repo}/commits/{head_sha}/status"
        # Get check runs (GitHub Actions)
        checks_endpoint = f"/repos/{repo}/commits/{head_sha}/check-runs"

        # Fetch both endpoints concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_status = executor.submit(self._make_request, status_endpoint)
            future_checks = executor.submit(self._make_request, checks_endpoint)
            combined_status = future_status.result()
            check_runs = future_checks.result()
        
        status_summary = {
            'total_checks': 0,
            'passed': 0,
            'failed': 0,
            'pending': 0,
            'required_failed': 0,
            'optional_failed': 0,
            'has_failures': False,
            'has_required_failures': False,
            'failed_checks': [],  # List of failed check names
            'required_failed_checks': []  # List of required failed check names
        }
        
        # Process status checks (older CI systems)
        if combined_status and 'statuses' in combined_status:
            for status in combined_status['statuses']:
                state = status.get('state', '')
                context = status.get('context', '')
                is_required = context in required_checks
                
                if state in ['success', 'failure', 'pending', 'error']:
                    status_summary['total_checks'] += 1
                    if state == 'success':
                        status_summary['passed'] += 1
                    elif state in ['failure', 'error']:
                        status_summary['failed'] += 1
                        status_summary['has_failures'] = True
                        status_summary['failed_checks'].append(context)
                        if is_required:
                            status_summary['required_failed'] += 1
                            status_summary['has_required_failures'] = True
                            status_summary['required_failed_checks'].append(context)
                        else:
                            status_summary['optional_failed'] += 1
                    elif state == 'pending':
                        status_summary['pending'] += 1
        
        # Process check runs (GitHub Actions)
        if check_runs and 'check_runs' in check_runs:
            for check in check_runs['check_runs']:
                conclusion = check.get('conclusion')
                status = check.get('status')
                check_name = check.get('name', '')
                is_required = check_name in required_checks
                
                if status == 'completed':
                    status_summary['total_checks'] += 1
                    if conclusion == 'success':
                        status_summary['passed'] += 1
                    elif conclusion in ['failure', 'timed_out', 'action_required']:
                        status_summary['failed'] += 1
                        status_summary['has_failures'] = True
                        status_summary['failed_checks'].append(check_name)
                        if is_required:
                            status_summary['required_failed'] += 1
                            status_summary['has_required_failures'] = True
                            status_summary['required_failed_checks'].append(check_name)
                        else:
                            status_summary['optional_failed'] += 1
                elif status in ['queued', 'in_progress']:
                    status_summary['total_checks'] += 1
                    status_summary['pending'] += 1
        
        return status_summary


class DynamoBranchChecker:
    """Main class for checking dynamo branches and PRs"""
    
    def __init__(self, base_dir: str, github_token: Optional[str] = None, verbose: bool = False, quick: bool = False, max_workers: int = 8, pull_main: bool = False):
        self.base_dir = Path(base_dir)
        self.github = GitHubAPI(github_token, verbose)
        self.verbose = verbose
        self.quick = quick
        self.max_workers = max(1, int(max_workers))
        self.pull_main = pull_main
        
    def find_dynamo_directories(self) -> List[Path]:
        """Find all dynamo* directories"""
        dynamo_dirs = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and item.name.startswith('dynamo'):
                # Check if it's a git repository
                if (item / '.git').exists():
                    dynamo_dirs.append(item)
                elif self.verbose:
                    print(f"Skipping {item.name} (not a git repository)")
        
        return sorted(dynamo_dirs)

    @staticmethod
    def parse_owner_repo_from_remote_url(remote_url: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """Parse host and owner/repo from a git remote URL.
        Supports HTTPS and SSH forms. Returns (host, owner_repo) or (None, None).
        """
        if not remote_url:
            return None, None
        try:
            url = remote_url.strip()
            host = None
            path = None
            if url.startswith('git@'):
                # e.g., git@github.com:owner/repo.git
                # Split 'git@host:path'
                at_idx = url.find('@')
                colon_idx = url.find(':', at_idx + 1)
                if at_idx != -1 and colon_idx != -1:
                    host = url[at_idx + 1:colon_idx]
                    path = url[colon_idx + 1:]
            elif url.startswith('ssh://'):
                # e.g., ssh://git@github.com/owner/repo.git
                # Remove scheme
                rest = url[len('ssh://'):]
                # Remove optional user@
                at_idx = rest.find('@')
                if at_idx != -1:
                    rest = rest[at_idx + 1:]
                slash_idx = rest.find('/')
                if slash_idx != -1:
                    host = rest[:slash_idx]
                    path = rest[slash_idx + 1:]
            elif url.startswith('https://') or url.startswith('http://'):
                # e.g., https://github.com/owner/repo.git
                # Find host between scheme and next '/'
                scheme_end = url.find('://') + 3
                slash_idx = url.find('/', scheme_end)
                if slash_idx != -1:
                    host = url[scheme_end:slash_idx]
                    path = url[slash_idx + 1:]
            # Normalize path to owner/repo
            if path and host:
                if path.endswith('.git'):
                    path = path[:-4]
                # Ensure it has at least owner/repo
                parts = path.split('/')
                if len(parts) >= 2:
                    owner_repo = f"{parts[0]}/{parts[1]}"
                    return host, owner_repo
            return None, None
        except Exception:
            return None, None
    
    def get_local_branches(self, repo_dir: Path) -> List[str]:
        """Get all local branches from a git repository"""
        try:
            result = subprocess.run(
                ['git', 'branch', '--format=%(refname:short)'],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                check=True
            )
            branches = [branch.strip() for branch in result.stdout.strip().split('\n') if branch.strip()]
            # Filter out HEAD and branches with invalid characters (like rebase states)
            valid_branches = []
            for b in branches:
                if b != 'HEAD' and not b.startswith('(') and ' ' not in b:
                    valid_branches.append(b)
            return valid_branches
        except subprocess.CalledProcessError as e:
            if self.verbose:
                print(f"⚠️  Error getting branches from {repo_dir.name}: {e}")
            return []
    
    def get_branch_tracking_status(self, repo_dir: Path, branch: str) -> Dict[str, Optional[str]]:
        """Get tracking information for a branch including if remote is gone"""
        try:
            # Get verbose branch info with tracking details
            result = subprocess.run(
                ['git', 'branch', '-vv'],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                check=True
            )
            
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                # Remove leading * for current branch
                if line.startswith('*'):
                    line = line[1:].strip()
                
                # Parse line: branch_name commit [remote: status] message
                parts = line.split(None, 1)
                if not parts:
                    continue
                
                branch_name = parts[0]
                if branch_name != branch:
                    continue
                
                # Check if remote is gone
                is_gone = ': gone]' in line
                
                # Extract upstream branch name if present
                upstream = None
                if '[' in line and ']' in line:
                    bracket_content = line[line.find('[')+1:line.find(']')]
                    upstream = bracket_content.split(':')[0].strip()
                
                return {
                    'upstream': upstream,
                    'is_gone': is_gone
                }
            
            return {'upstream': None, 'is_gone': False}
            
        except subprocess.CalledProcessError:
            return {'upstream': None, 'is_gone': False}

    def get_all_branch_tracking_statuses(self, repo_dir: Path) -> Dict[str, Dict[str, Optional[str]]]:
        """Get tracking info for all branches in one git call."""
        try:
            result = subprocess.run(
                ['git', 'branch', '-vv'],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                check=True
            )
            tracking: Dict[str, Dict[str, Optional[str]]] = {}
            for raw_line in result.stdout.strip().split('\n'):
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith('*'):
                    line = line[1:].strip()
                parts = line.split(None, 1)
                if not parts:
                    continue
                branch_name = parts[0]
                is_gone = ': gone]' in line
                upstream = None
                if '[' in line and ']' in line:
                    bracket_content = line[line.find('[')+1:line.find(']')]
                    upstream = bracket_content.split(':')[0].strip()
                tracking[branch_name] = {
                    'upstream': upstream,
                    'is_gone': is_gone
                }
            return tracking
        except subprocess.CalledProcessError:
            return {}
    
    def get_current_branch(self, repo_dir: Path) -> Optional[str]:
        """Get the current branch of a git repository"""
        try:
            result = subprocess.run(
                ['git', 'branch', '--show-current'],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip() or None
        except subprocess.CalledProcessError:
            return None
    
    def pull_main_branch(self, repo_dir: Path) -> bool:
        """Pull latest on main branch, returning to the original branch after
        
        Stashes any uncommitted changes before switching branches and reapplies them after.
        
        Returns:
            True if successful, False otherwise
        """
        stashed = False
        try:
            # Get current branch to return to it later
            current_branch = self.get_current_branch(repo_dir)
            
            if self.verbose:
                print(f"  Pulling latest on main for {repo_dir.name}...")
            
            # Check if there are uncommitted changes
            status_result = subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                check=True
            )
            
            has_changes = bool(status_result.stdout.strip())
            
            # Stash changes if any exist
            if has_changes:
                if self.verbose:
                    print(f"  Stashing uncommitted changes...")
                subprocess.run(
                    ['git', 'stash', 'push', '-u', '-m', 'Auto-stash by show_dynamo_branches.py'],
                    cwd=repo_dir,
                    capture_output=True,
                    text=True,
                    check=True
                )
                stashed = True
            
            # Fetch latest from remote
            subprocess.run(
                ['git', 'fetch', 'origin'],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Checkout main
            subprocess.run(
                ['git', 'checkout', 'main'],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Pull latest
            result = subprocess.run(
                ['git', 'pull', '--ff-only'],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                check=True
            )
            
            if self.verbose:
                print(f"  ✓ Updated main: {result.stdout.strip()}")
            
            # Return to original branch if it wasn't main
            if current_branch and current_branch != 'main':
                subprocess.run(
                    ['git', 'checkout', current_branch],
                    cwd=repo_dir,
                    capture_output=True,
                    text=True,
                    check=True
                )
                if self.verbose:
                    print(f"  Returned to branch: {current_branch}")
            
            # Pop stashed changes if we stashed them
            if stashed:
                if self.verbose:
                    print(f"  Restoring stashed changes...")
                subprocess.run(
                    ['git', 'stash', 'pop'],
                    cwd=repo_dir,
                    capture_output=True,
                    text=True,
                    check=True
                )
            
            return True
            
        except subprocess.CalledProcessError as e:
            if self.verbose:
                print(f"  ⚠️  Error pulling main in {repo_dir.name}: {e}")
                if e.stderr:
                    print(f"      {e.stderr.strip()}")
            
            # Try to restore state if something went wrong and we stashed
            if stashed:
                try:
                    if self.verbose:
                        print(f"  Attempting to restore stashed changes after error...")
                    subprocess.run(
                        ['git', 'stash', 'pop'],
                        cwd=repo_dir,
                        capture_output=True,
                        text=True,
                        check=True
                    )
                except subprocess.CalledProcessError:
                    print(f"  ⚠️  Failed to restore stashed changes in {repo_dir.name}")
                    print(f"      You may need to manually run 'git stash pop' in {repo_dir}")
            
            return False
    
    def get_remote_url(self, repo_dir: Path) -> Optional[str]:
        """Get the remote origin URL"""
        try:
            result = subprocess.run(
                ['git', 'remote', 'get-url', 'origin'],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None
    
    def get_all_remote_urls(self, repo_dir: Path) -> Dict[str, str]:
        """Get all remote URLs"""
        try:
            result = subprocess.run(
                ['git', 'remote', '-v'],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                check=True
            )
            remotes = {}
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        remote_name = parts[0]
                        remote_url = parts[1]
                        # Only keep fetch URLs (ignore push URLs)
                        if len(parts) < 3 or parts[2] == '(fetch)':
                            remotes[remote_name] = remote_url
            return remotes
        except subprocess.CalledProcessError:
            return {}
    
    def format_pr_info(self, pr: Dict, branch_name: str = "", is_current: bool = False, is_gone: bool = False, review_summary: Optional[Dict] = None, status_checks: Optional[Dict] = None) -> str:
        """Format PR information for display"""
        number = pr.get('number', 'Unknown')
        title = pr.get('title', 'No title')
        state = pr.get('state', 'unknown')
        merged_at = pr.get('merged_at')
        url = pr.get('html_url', '')
        
        # Truncate long titles to fit on same line with branch name
        max_title_length = 80 - len(branch_name) if branch_name else 60
        if len(title) > max_title_length:
            title = title[:max_title_length-3] + "..."
        
        # GitHub API returns 'closed' for merged PRs, check merged_at field
        if state == 'closed' and merged_at:
            display_state = 'merged'
        else:
            display_state = state
        
        state_emoji = {
            'open': '📖',
            'closed': '❌',
            'merged': '🔀'
        }.get(display_state, '⚪')
        
        # Add gone indicator
        gone_info = " [GONE]" if is_gone else ""
        
        # Add review status indicators
        review_info = ""
        if review_summary and display_state == 'open':
            if review_summary['has_changes_requested']:
                review_info = f" [CHANGES:{review_summary['changes_requested']}]"
            elif review_summary['has_approvals']:
                review_info = f" ✅ {review_summary['approved']}"
            elif review_summary['commented'] > 0:
                review_info = f" [Unresolved Comments:{review_summary['commented']}]"
        
        # Add CI status indicators
        ci_info = ""
        ci_details = ""
        # NOTE: "Required Checks" displayed in the CI badge refers to status checks that
        # are marked as required by GitHub branch protection rules on the PR's base branch.
        # We build and cache the required set per base branch via the branch protection API
        # (see GitHubAPI._get_required_checks_for_base_branch). When summarizing CI in
        # GitHubAPI.get_pr_status_checks, only failures whose context/name is in that
        # required set count toward 'required_failed' and trigger the
        # "[CI: ❌ N required failed]" message below. If no protection is configured,
        # the required set is empty and failures are treated as optional. In --quick mode
        # CI/review enrichment is skipped entirely, so no CI badge is added.
        if status_checks and display_state == 'open':
            if status_checks['has_required_failures']:
                failed = status_checks['required_failed']
                ci_info = f" [CI: ❌ {failed} required failed]"
                # Add details of failed required checks
                if status_checks.get('required_failed_checks'):
                    failed_list = '\n         • '.join(status_checks['required_failed_checks'])
                    ci_details = f"\n       Required checks failed:\n         • {failed_list}"
            elif status_checks['pending'] > 0:
                ci_info = f" [CI: ⏳ pending]"
            elif status_checks['total_checks'] > 0:
                ci_info = f" [CI: ✅ passed]"
        
        # Format branch name (bold if current)
        if branch_name:
            if is_current:
                branch_display = f"\033[1m{branch_name}\033[0m"  # Bold
            else:
                branch_display = branch_name
            return f"     {state_emoji} {branch_display}: PR #{number} - {title}{gone_info}{review_info}{ci_info}\n       {url}{ci_details}"
        else:
            return f"  {state_emoji} PR #{number}: {title}{gone_info}{review_info}{ci_info}\n     {url}{ci_details}"
    
    def check_directory(self, repo_dir: Path) -> Dict:
        """Check a single dynamo directory for branches and PRs"""
        if self.verbose:
            print(f"\nChecking {repo_dir.name}...")
        
        # Pull latest on main if requested
        if self.pull_main:
            self.pull_main_branch(repo_dir)
        
        # Get repository info
        remote_url = self.get_remote_url(repo_dir)
        all_remotes = self.get_all_remote_urls(repo_dir)
        current_branch = self.get_current_branch(repo_dir)
        local_branches = self.get_local_branches(repo_dir)
        
        # Find GitHub remote and determine if it's the ai-dynamo/dynamo repo
        github_remote_url = None
        github_owner_repo = None
        
        # Check all remotes for GitHub URLs, prioritizing common remote names
        remote_priority = ['tracking', 'upstream', 'origin']
        for remote_name in remote_priority:
            if remote_name in all_remotes:
                host, owner_repo = self.parse_owner_repo_from_remote_url(all_remotes[remote_name])
                if host and host.endswith('github.com') and owner_repo == 'ai-dynamo/dynamo':
                    github_remote_url = all_remotes[remote_name]
                    github_owner_repo = owner_repo
                    break
        
        # If no priority remote found, check all other remotes
        if not github_remote_url:
            for remote_name, url in all_remotes.items():
                if remote_name not in remote_priority:
                    host, owner_repo = self.parse_owner_repo_from_remote_url(url)
                    if host and host.endswith('github.com') and owner_repo == 'ai-dynamo/dynamo':
                        github_remote_url = url
                        github_owner_repo = owner_repo
                        break
        
        is_dynamo_repo = bool(github_remote_url and github_owner_repo)
        
        result = {
            'directory': repo_dir.name,
            'path': str(repo_dir),
            'remote_url': remote_url,
            'github_remote_url': github_remote_url,
            'all_remotes': all_remotes,
            'repo': github_owner_repo,
            'current_branch': current_branch,
            'local_branches': local_branches,
            'is_dynamo_repo': is_dynamo_repo,
            'branch_info': {}
        }
        
        if not is_dynamo_repo:
            if self.verbose:
                github_remotes = [f"{name}: {url}" for name, url in all_remotes.items() 
                                if 'github.com' in url]
                if github_remotes:
                    print(f"  ⚠️  Found GitHub remotes but not ai-dynamo/dynamo: {', '.join(github_remotes)}")
                else:
                    print(f"  ⚠️  No GitHub remotes found (origin: {remote_url})")
            return result
        
        # Precompute tracking for all branches in one call
        tracking_map = self.get_all_branch_tracking_statuses(repo_dir)

        def process_branch(branch: str) -> Tuple[str, Dict]:
            tracking_status = tracking_map.get(branch, {'upstream': None, 'is_gone': False})
            branch_info = {
                'exists_on_github': False,
                'prs': [],
                'is_current': branch == current_branch,
                'is_gone': tracking_status['is_gone'],
                'upstream': tracking_status['upstream']
            }

            if branch in ['main', 'master']:
                branch_info['exists_on_github'] = True
                return branch, branch_info

            if self.verbose:
                print(f"  Checking branch: {branch}")

            # Check if branch exists and fetch PRs in parallel
            with ThreadPoolExecutor(max_workers=min(3, self.max_workers)) as exec_branch:
                fut_exists = exec_branch.submit(self.github.branch_exists, github_owner_repo, branch)
                fut_find = exec_branch.submit(self.github.find_prs_for_branch, github_owner_repo, branch)
                fut_search = exec_branch.submit(self.github.search_prs_by_branch, github_owner_repo, branch)
                exists = fut_exists.result()
                find_results = fut_find.result() or []
                search_results = fut_search.result() or []

            # Prefer head-based find results; otherwise fall back to search
            prs = find_results if find_results else search_results
            # If both returned, deduplicate by PR number
            if find_results and search_results:
                seen: Set[int] = set()
                merged: List[Dict] = []
                for pr in (find_results + search_results):
                    num = pr.get('number')
                    if num and num not in seen:
                        seen.add(num)
                        merged.append(pr)
                prs = merged

            # Enrich PRs concurrently across all PRs with a single pool
            prs_with_reviews: List[Dict] = []
            if not self.quick and prs:
                open_prs = [p for p in prs if p.get('state') == 'open']
                review_futures = {}
                status_futures = {}
                max_workers = min(self.max_workers, max(2, len(open_prs) * 2))
                with ThreadPoolExecutor(max_workers=max_workers) as exec_prs:
                    for p in open_prs:
                        num = p.get('number')
                        if not num:
                            continue
                        review_futures[exec_prs.submit(self.github.get_pr_reviews, github_owner_repo, num)] = num
                        status_futures[exec_prs.submit(self.github.get_pr_status_checks, github_owner_repo, num, p)] = num

                    review_results: Dict[int, Optional[Dict]] = {}
                    status_results: Dict[int, Optional[Dict]] = {}
                    for f in as_completed(list(review_futures.keys()) + list(status_futures.keys())):
                        if f in review_futures:
                            review_results[review_futures[f]] = f.result()
                        else:
                            status_results[status_futures[f]] = f.result()

                # Attach results back to PRs
                for p in prs:
                    pr_copy = p.copy()
                    num = p.get('number')
                    if num and p.get('state') == 'open':
                        pr_copy['review_summary'] = review_results.get(num)
                        pr_copy['status_checks'] = status_results.get(num)
                    prs_with_reviews.append(pr_copy)
            else:
                # Quick mode or no PRs: copy as-is
                prs_with_reviews = [p.copy() for p in prs]

            branch_info['exists_on_github'] = exists
            branch_info['prs'] = prs_with_reviews
            return branch, branch_info

        # Process branches in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for branch, info in executor.map(process_branch, local_branches):
                result['branch_info'][branch] = info
        
        return result
    
    def run(self) -> None:
        """Main execution method"""
        print("Dynamo Branch & PR Checker")
        print("=" * 50)
        
        # Show PR status guide
        print("PR Status: 📖 [OPEN] | ❌ [CLOSED] | 🔀 [MERGED]")
        print("Review Status: ✅ Approved | [CHANGES] | [Unresolved Comments]")
        print("Branch Status: [GONE] = Remote branch deleted (likely merged/closed)")
        
        # Show token status
        if self.github.token:
            if self.verbose:
                token_source = "unknown"
                if self.github.token == os.environ.get('GITHUB_TOKEN'):
                    token_source = "environment variable"
                elif self.github.token == get_github_token_from_cli():
                    token_source = "GitHub CLI config"
                else:
                    token_source = "command line argument"
                print(f"Using GitHub token from: {token_source}")
            else:
                print("GitHub token detected - higher rate limits available")
        else:
            print("⚠️  No GitHub token found - limited to 60 requests/hour")
        
        dynamo_dirs = self.find_dynamo_directories()
        
        if not dynamo_dirs:
            print("No dynamo* directories found!")
            return
        
        print(f"Found {len(dynamo_dirs)} dynamo directories:")
        for d in dynamo_dirs:
            print(f"   • {d.name}")
        
        all_results = []
        # Process directories in parallel while preserving order
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for res in executor.map(self.check_directory, dynamo_dirs):
                all_results.append(res)
        
        # Display results
        print("\n" + "=" * 50)
        print("RESULTS")
        print("=" * 50)
        
        for result in all_results:
            self.display_result(result)
        
        # Show timing summary if verbose
        if self.verbose and self.github.api_call_times:
            print("\n" + "=" * 50)
            print("API CALL TIMING SUMMARY")
            print("=" * 50)
            total_time = sum(t for _, t in self.github.api_call_times)
            print(f"Total API calls: {len(self.github.api_call_times)}")
            print(f"Total API time: {total_time:.2f}s")
            print(f"Average per call: {total_time/len(self.github.api_call_times):.2f}s")
            
            # Group by endpoint type
            endpoint_times: Dict[str, List[float]] = {}
            for endpoint, elapsed in self.github.api_call_times:
                # Extract endpoint type (e.g., /repos/.../branches/... -> branches)
                if '/branches/' in endpoint and '/protection' not in endpoint:
                    key = 'branch_exists'
                elif '/pulls/' in endpoint and '/reviews' in endpoint:
                    key = 'get_reviews'
                elif '/commits/' in endpoint and '/status' in endpoint:
                    key = 'ci_combined_status'
                elif '/commits/' in endpoint and '/check-runs' in endpoint:
                    key = 'ci_check_runs'
                elif '/branches/' in endpoint and '/protection' in endpoint:
                    key = 'ci_branch_protection'
                elif '/pulls/' in endpoint and endpoint.count('/') == 5:  # /repos/owner/repo/pulls/number
                    key = 'get_pr_details'
                elif '/pulls?' in endpoint:
                    key = 'find_prs'
                elif '/search/' in endpoint:
                    key = 'search_prs'
                else:
                    key = 'other'
                
                if key not in endpoint_times:
                    endpoint_times[key] = []
                endpoint_times[key].append(elapsed)
            
            print("\nBy endpoint type:")
            for key, times in sorted(endpoint_times.items(), key=lambda x: sum(x[1]), reverse=True):
                total = sum(times)
                avg = total / len(times)
                print(f"  {key}: {len(times)} calls, {total:.2f}s total, {avg:.2f}s avg")
    
    def display_result(self, result: Dict) -> None:
        """Display results for a single directory"""
        dir_name = result['directory']
        current_branch = result['current_branch']
        
        print(f"\n[{dir_name}]")
        print(f"   Current branch: {current_branch or 'Unknown'}")
        
        if not result['is_dynamo_repo']:
            github_remotes = [f"{name}: {url}" for name, url in result['all_remotes'].items() 
                            if 'github.com' in url]
            if github_remotes:
                print(f"   ⚠️  Found GitHub remotes but not ai-dynamo/dynamo: {', '.join(github_remotes)}")
            else:
                print(f"   ⚠️  No GitHub remotes found (origin: {result['remote_url']})")
            return
        
        branches_with_prs = []
        branches_on_github = []
        local_only_branches = []
        gone_branches = []
        
        for branch, info in result['branch_info'].items():
            current_marker = " (current)" if info['is_current'] else ""
            
            if info['prs']:
                branches_with_prs.append((branch, info['prs'], current_marker, info['is_gone']))
            elif info['is_gone']:
                gone_branches.append((branch, current_marker))
            elif info['exists_on_github']:
                branches_on_github.append((branch, current_marker))
            else:
                local_only_branches.append((branch, current_marker))
        
        # Display branches with PRs
        if branches_with_prs:
            print("   Branches with PRs:")
            for branch, prs, marker, is_gone in branches_with_prs:
                is_current = "(current)" in marker
                for pr in prs:
                    review_summary = pr.get('review_summary')
                    status_checks = pr.get('status_checks')
                    print(f"{self.format_pr_info(pr, branch, is_current, is_gone, review_summary, status_checks)}")
        
        # Display branches on GitHub (no PRs found)
        if branches_on_github:
            print("   Branches on GitHub (no PRs found):")
            for branch, marker in branches_on_github:
                is_current = "(current)" in marker
                if is_current:
                    branch_display = f"\033[1m{branch}\033[0m (current)"  # Bold
                else:
                    branch_display = branch
                print(f"     • {branch_display}")
        
        # Display gone branches (remote deleted but no PR found)
        if gone_branches:
            print("   Gone branches (remote deleted):")
            for branch, marker in gone_branches:
                is_current = "(current)" in marker
                if is_current:
                    branch_display = f"\033[1m{branch}\033[0m (current)"  # Bold
                else:
                    branch_display = branch
                print(f"     • {branch_display} [GONE]")
        
        # Display local-only branches
        if local_only_branches:
            print("   Local-only branches:")
            for branch, marker in local_only_branches:
                is_current = "(current)" in marker
                if is_current:
                    branch_display = f"\033[1m{branch}\033[0m (current)"  # Bold
                else:
                    branch_display = branch
                print(f"     • {branch_display}")


def main():
    parser = argparse.ArgumentParser(
        description="Check dynamo* directories for branches and corresponding GitHub PRs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./check_dynamo_branches.py                    # Basic usage
  ./check_dynamo_branches.py --verbose          # Verbose output
  ./check_dynamo_branches.py --github-token TOKEN  # Use GitHub token for higher rate limits

GitHub Token:
  The script automatically detects GitHub tokens in this priority order:
  1. --github-token command line argument
  2. GITHUB_TOKEN environment variable  
  3. GitHub CLI configuration (~/.config/gh/hosts.yml)
  
  If no token is found, you'll be limited to 60 requests/hour.
  Create a token at: https://github.com/settings/tokens
        """
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--github-token',
        help='GitHub API token for higher rate limits'
    )
    
    parser.add_argument(
        '--base-dir',
        default='.',
        help='Base directory to search for dynamo* directories (default: current directory)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: skip CI status and review checks (much faster)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=16,
        help='Max worker threads for parallel API calls (default: 8)'
    )
    
    parser.add_argument(
        '--pull-main',
        action='store_true',
        help='Pull latest on main branch before checking (returns to original branch after)'
    )
    
    args = parser.parse_args()
    
    # Resolve base directory
    base_dir = Path(args.base_dir).resolve()
    
    if not base_dir.exists():
        print(f"❌ Base directory does not exist: {base_dir}")
        sys.exit(1)
    
    if args.verbose:
        print(f"🔍 Searching in: {base_dir}")
    
    checker = DynamoBranchChecker(
        base_dir=str(base_dir),
        github_token=args.github_token,
        verbose=args.verbose,
        quick=args.quick,
        max_workers=args.max_workers,
        pull_main=args.pull_main
    )
    
    try:
        checker.run()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
