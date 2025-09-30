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
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
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
    
    def __init__(self, token: Optional[str] = None):
        self.base_url = "https://api.github.com"
        self.repo = "ai-dynamo/dynamo"
        # Token priority: 1) provided token, 2) environment variable, 3) GitHub CLI config
        self.token = token or os.environ.get('GITHUB_TOKEN') or get_github_token_from_cli()
        
    def _make_request(self, endpoint: str) -> Optional[Dict]:
        """Make a request to GitHub API"""
        url = urljoin(self.base_url, endpoint)
        req = Request(url)
        
        if self.token:
            req.add_header('Authorization', f'token {self.token}')
        req.add_header('Accept', 'application/vnd.github.v3+json')
        req.add_header('User-Agent', 'dynamo-branch-checker/1.0')
        
        try:
            with urlopen(req) as response:
                return json.loads(response.read().decode())
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
    
    def branch_exists(self, branch_name: str) -> bool:
        """Check if a branch exists on GitHub"""
        # URL encode the branch name to handle special characters
        encoded_branch = quote(branch_name, safe='')
        endpoint = f"/repos/{self.repo}/branches/{encoded_branch}"
        result = self._make_request(endpoint)
        return result is not None
    
    def find_prs_for_branch(self, branch_name: str) -> List[Dict]:
        """Find PRs associated with a branch"""
        # Search for PRs with this branch as head
        # URL encode the branch name to handle special characters
        encoded_branch = quote(branch_name, safe='')
        endpoint = f"/repos/{self.repo}/pulls?head=ai-dynamo:{encoded_branch}&state=all"
        result = self._make_request(endpoint)
        
        if result is None:
            return []
        
        return result
    
    def search_prs_by_branch(self, branch_name: str) -> List[Dict]:
        """Search for PRs that might be related to this branch"""
        # GitHub search API for PRs mentioning the branch
        query = f"repo:{self.repo} type:pr {branch_name}"
        # URL encode the entire query
        encoded_query = quote(query, safe='')
        endpoint = f"/search/issues?q={encoded_query}"
        result = self._make_request(endpoint)
        
        if result is None or 'items' not in result:
            return []
        
        # Filter results to only include PRs where head branch matches
        # Search API can return false positives where branch name words appear in PR text
        filtered_prs = []
        for item in result['items']:
            # Need to fetch full PR details to check head branch
            pr_number = item.get('number')
            if pr_number:
                pr_endpoint = f"/repos/{self.repo}/pulls/{pr_number}"
                pr_details = self._make_request(pr_endpoint)
                if pr_details and pr_details.get('head', {}).get('ref') == branch_name:
                    filtered_prs.append(pr_details)
        
        return filtered_prs
    
    def get_pr_reviews(self, pr_number: int) -> Optional[Dict]:
        """Get review status for a PR"""
        endpoint = f"/repos/{self.repo}/pulls/{pr_number}/reviews"
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
    
    def get_pr_status_checks(self, pr_number: int) -> Optional[Dict]:
        """Get CI status checks for a PR"""
        # First get the PR details to get the head SHA and base branch
        pr_endpoint = f"/repos/{self.repo}/pulls/{pr_number}"
        pr_data = self._make_request(pr_endpoint)
        
        if not pr_data or 'head' not in pr_data:
            return None
        
        head_sha = pr_data['head']['sha']
        base_branch = pr_data.get('base', {}).get('ref', 'main')
        
        # Get branch protection rules to know which checks are required
        protection_endpoint = f"/repos/{self.repo}/branches/{base_branch}/protection"
        protection_data = self._make_request(protection_endpoint)
        
        required_checks = set()
        if protection_data and 'required_status_checks' in protection_data:
            checks_data = protection_data['required_status_checks'].get('checks', [])
            for check in checks_data:
                context = check.get('context', '')
                if context:
                    required_checks.add(context)
            # Also check legacy format
            contexts = protection_data['required_status_checks'].get('contexts', [])
            required_checks.update(contexts)
        
        # Get combined status for the commit
        status_endpoint = f"/repos/{self.repo}/commits/{head_sha}/status"
        combined_status = self._make_request(status_endpoint)
        
        # Get check runs (GitHub Actions)
        checks_endpoint = f"/repos/{self.repo}/commits/{head_sha}/check-runs"
        check_runs = self._make_request(checks_endpoint)
        
        status_summary = {
            'total_checks': 0,
            'passed': 0,
            'failed': 0,
            'pending': 0,
            'required_failed': 0,
            'optional_failed': 0,
            'has_failures': False,
            'has_required_failures': False
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
                        if is_required:
                            status_summary['required_failed'] += 1
                            status_summary['has_required_failures'] = True
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
                        if is_required:
                            status_summary['required_failed'] += 1
                            status_summary['has_required_failures'] = True
                        else:
                            status_summary['optional_failed'] += 1
                elif status in ['queued', 'in_progress']:
                    status_summary['total_checks'] += 1
                    status_summary['pending'] += 1
        
        return status_summary


class DynamoBranchChecker:
    """Main class for checking dynamo branches and PRs"""
    
    def __init__(self, base_dir: str, github_token: Optional[str] = None, verbose: bool = False):
        self.base_dir = Path(base_dir)
        self.github = GitHubAPI(github_token)
        self.verbose = verbose
        
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
            'open': '🟢',
            'closed': '🔴',
            'merged': '🟣'
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
        if status_checks and display_state == 'open':
            if status_checks['has_required_failures']:
                failed = status_checks['required_failed']
                ci_info = f" [CI: ❌ {failed} required failed]"
            elif status_checks['pending'] > 0:
                ci_info = f" [CI: pending]"
            elif status_checks['total_checks'] > 0:
                ci_info = f" [CI: ✅ passed]"
        
        # Format branch name (bold if current)
        if branch_name:
            if is_current:
                branch_display = f"\033[1m{branch_name}\033[0m"  # Bold
            else:
                branch_display = branch_name
            return f"     {state_emoji} {branch_display}: PR #{number} - {title}{gone_info}{review_info}{ci_info}\n       {url}"
        else:
            return f"  {state_emoji} PR #{number}: {title}{gone_info}{review_info}{ci_info}\n     {url}"
    
    def check_directory(self, repo_dir: Path) -> Dict:
        """Check a single dynamo directory for branches and PRs"""
        if self.verbose:
            print(f"\nChecking {repo_dir.name}...")
        
        # Get repository info
        remote_url = self.get_remote_url(repo_dir)
        current_branch = self.get_current_branch(repo_dir)
        local_branches = self.get_local_branches(repo_dir)
        
        # Check if this is actually pointing to the ai-dynamo/dynamo repo
        is_dynamo_repo = remote_url and 'ai-dynamo/dynamo' in remote_url
        
        result = {
            'directory': repo_dir.name,
            'path': str(repo_dir),
            'remote_url': remote_url,
            'current_branch': current_branch,
            'local_branches': local_branches,
            'is_dynamo_repo': is_dynamo_repo,
            'branch_info': {}
        }
        
        if not is_dynamo_repo:
            if self.verbose:
                print(f"  ⚠️  Not pointing to ai-dynamo/dynamo (remote: {remote_url})")
            return result
        
        # Check each branch
        for branch in local_branches:
            # Get tracking status (whether remote is gone)
            tracking_status = self.get_branch_tracking_status(repo_dir, branch)
            
            branch_info = {
                'exists_on_github': False,
                'prs': [],
                'is_current': branch == current_branch,
                'is_gone': tracking_status['is_gone'],
                'upstream': tracking_status['upstream']
            }
            
            if branch in ['main', 'master']:
                branch_info['exists_on_github'] = True
            else:
                # Check if branch exists on GitHub
                if self.verbose:
                    print(f"  Checking branch: {branch}")
                
                branch_info['exists_on_github'] = self.github.branch_exists(branch)
                
                # Look for PRs even if branch doesn't exist (it might be gone/merged)
                prs = self.github.find_prs_for_branch(branch)
                if not prs:
                    # Try searching for PRs that mention this branch
                    prs = self.github.search_prs_by_branch(branch)
                
                # Get review status and CI checks for each PR
                prs_with_reviews = []
                for pr in prs:
                    pr_data = pr.copy()
                    pr_number = pr.get('number')
                    if pr_number and pr.get('state') == 'open':
                        review_summary = self.github.get_pr_reviews(pr_number)
                        pr_data['review_summary'] = review_summary
                        
                        # Get CI status checks
                        status_checks = self.github.get_pr_status_checks(pr_number)
                        pr_data['status_checks'] = status_checks
                    prs_with_reviews.append(pr_data)
                
                branch_info['prs'] = prs_with_reviews
            
            result['branch_info'][branch] = branch_info
        
        return result
    
    def run(self) -> None:
        """Main execution method"""
        print("Dynamo Branch & PR Checker")
        print("=" * 50)
        
        # Show PR status guide
        print("PR Status: 🟢 [OPEN] | 🔴 [CLOSED] | 🟣 [MERGED]")
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
        
        for repo_dir in dynamo_dirs:
            result = self.check_directory(repo_dir)
            all_results.append(result)
        
        # Display results
        print("\n" + "=" * 50)
        print("RESULTS")
        print("=" * 50)
        
        for result in all_results:
            self.display_result(result)
    
    def display_result(self, result: Dict) -> None:
        """Display results for a single directory"""
        dir_name = result['directory']
        current_branch = result['current_branch']
        
        print(f"\n[{dir_name}]")
        print(f"   Current branch: {current_branch or 'Unknown'}")
        
        if not result['is_dynamo_repo']:
            print(f"   ⚠️  Not ai-dynamo/dynamo repo (remote: {result['remote_url']})")
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
        verbose=args.verbose
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
