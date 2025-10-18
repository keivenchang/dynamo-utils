#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Simplified script to check branches in dynamo* directories and find corresponding PRs.

This version simplifies the original show_dynamo_branches.py by:
- Removing complex caching mechanisms
- Simplifying GitHub API interactions
- Using requests library instead of urllib
- Clearer class structure with single responsibility
- Less verbose output by default

Usage:
    ./show_dynamo_branches2.py [--verbose] [--token TOKEN]
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: 'requests' library not found. Install with: pip install requests")
    sys.exit(1)

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import git
    HAS_GIT = True
except ImportError:
    HAS_GIT = False
    print("Warning: 'gitpython' library not found. Install with: pip install gitpython")
    sys.exit(1)


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
                    # Try oauth_token first
                    if 'oauth_token' in github_config:
                        return github_config['oauth_token']
                    # Try users config
                    if 'users' in github_config:
                        for user, user_config in github_config['users'].items():
                            if 'oauth_token' in user_config:
                                return user_config['oauth_token']
    except Exception:
        pass
    return None


@dataclass
class GitRepo:
    """Represents a git repository"""
    path: Path
    current_branch: Optional[str]
    branches: List[str]
    remote_url: Optional[str]
    owner_repo: Optional[str]  # e.g., "ai-dynamo/dynamo"


@dataclass
class PullRequest:
    """Represents a GitHub pull request"""
    number: int
    title: str
    state: str  # open, closed, merged
    url: str
    branch: str
    author: str
    review_comments: int = 0
    unresolved_conversations: int = 0
    ci_status: Optional[str] = None  # passed, failed, pending, None
    is_merged: bool = False
    review_decision: Optional[str] = None  # APPROVED, CHANGES_REQUESTED, REVIEW_REQUIRED, None
    has_conflicts: bool = False
    mergeable_state: Optional[str] = None  # clean, dirty, blocked, unstable, etc.
    conflict_message: Optional[str] = None  # Message about conflicts
    blocking_message: Optional[str] = None  # Message about merge being blocked


class GitHelper:
    """Simple helper for git operations using GitPython API"""

    def __init__(self, verbose: bool = False):
        """Initialize with repo cache for better performance"""
        self._repo_cache: Dict[str, git.Repo] = {}
        self.verbose = verbose
        import logging
        self.logger = logging.getLogger(self.__class__.__name__)
        if verbose:
            self.logger.setLevel(logging.DEBUG)

    def _get_repo(self, repo_path: Path) -> Optional[git.Repo]:
        """Get cached git.Repo object or create new one"""
        repo_key = str(repo_path.absolute())
        if repo_key not in self._repo_cache:
            try:
                self._repo_cache[repo_key] = git.Repo(repo_path)
                self.logger.debug(f"Initialized git repo: {repo_path}")
            except (git.exc.InvalidGitRepositoryError, git.exc.NoSuchPathError):
                return None
        return self._repo_cache[repo_key]

    def get_current_branch(self, repo_path: Path) -> Optional[str]:
        """Get current branch name"""
        self.logger.debug(f"Equivalent: git -C {repo_path} branch --show-current")
        try:
            repo = self._get_repo(repo_path)
            if not repo:
                return None
            return repo.active_branch.name
        except (git.exc.GitError, TypeError):
            return None

    def get_all_branches(self, repo_path: Path) -> List[str]:
        """Get all local branch names"""
        self.logger.debug(f"Equivalent: git -C {repo_path} branch --format='%(refname:short)'")
        try:
            repo = self._get_repo(repo_path)
            if not repo:
                return []
            branches = [head.name for head in repo.heads]
            return branches
        except (git.exc.GitError, AttributeError):
            return []

    def get_remote_url(self, repo_path: Path, remote: str = 'origin') -> Optional[str]:
        """Get remote URL for given remote name"""
        self.logger.debug(f"Equivalent: git -C {repo_path} remote get-url {remote}")
        try:
            repo = self._get_repo(repo_path)
            if not repo:
                return None
            if remote in repo.remotes:
                return list(repo.remotes[remote].urls)[0]
            return None
        except (git.exc.GitError, IndexError, AttributeError):
            return None

    def get_all_remotes(self, repo_path: Path) -> Dict[str, str]:
        """Get all remotes as dict of {name: url}"""
        self.logger.debug(f"Equivalent: git -C {repo_path} remote -v")
        try:
            repo = self._get_repo(repo_path)
            if not repo:
                return {}
            remotes = {}
            for remote in repo.remotes:
                urls = list(remote.urls)
                if urls:
                    remotes[remote.name] = urls[0]
            return remotes
        except (git.exc.GitError, AttributeError):
            return {}

    @staticmethod
    def parse_github_repo(remote_url: str) -> Optional[str]:
        """
        Parse GitHub owner/repo from remote URL.

        Examples:
            git@github.com:ai-dynamo/dynamo.git -> ai-dynamo/dynamo
            https://github.com/ai-dynamo/dynamo.git -> ai-dynamo/dynamo
        """
        if not remote_url or 'github.com' not in remote_url:
            return None

        # Remove .git suffix
        url = remote_url.rstrip('/')
        if url.endswith('.git'):
            url = url[:-4]

        # Extract owner/repo
        if url.startswith('git@github.com:'):
            return url.split('git@github.com:')[1]
        elif 'github.com/' in url:
            parts = url.split('github.com/')[1].split('/')
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"

        return None

    def find_github_remote(self, repo_path: Path) -> Optional[str]:
        """
        Find the GitHub remote that points to ai-dynamo/dynamo.
        Check remotes in priority order: tracking, upstream, origin.
        """
        remotes = self.get_all_remotes(repo_path)

        # Priority order
        for remote_name in ['tracking', 'upstream', 'origin']:
            if remote_name in remotes:
                owner_repo = self.parse_github_repo(remotes[remote_name])
                if owner_repo == 'ai-dynamo/dynamo':
                    return owner_repo

        # Check all other remotes
        for url in remotes.values():
            owner_repo = self.parse_github_repo(url)
            if owner_repo == 'ai-dynamo/dynamo':
                return owner_repo

        return None


class GitHubClient:
    """Simple GitHub API client"""

    def __init__(self, token: Optional[str] = None, verbose: bool = False):
        self.base_url = "https://api.github.com"
        # Token priority: 1) provided token, 2) environment variable, 3) GitHub CLI config
        import os
        self.token = token or os.environ.get('GITHUB_TOKEN') or get_github_token_from_cli()
        self.verbose = verbose
        self.session = requests.Session()

        if self.token:
            self.session.headers.update({'Authorization': f'token {self.token}'})
        self.session.headers.update({
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'dynamo-branch-checker/2.0'
        })

    def get(self, endpoint: str) -> Optional[Dict]:
        """Make GET request to GitHub API"""
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.get(url, timeout=10)

            if response.status_code == 404:
                return None
            elif response.status_code == 403:
                print(f"  Warning: Rate limit exceeded. Use --token to increase limit.", file=sys.stderr)
                return None
            elif response.status_code != 200:
                if self.verbose:
                    print(f"  API returned status {response.status_code} for {endpoint}", file=sys.stderr)
                return None

            return response.json()

        except requests.exceptions.RequestException as e:
            if self.verbose:
                print(f"  API error for {endpoint}: {e}", file=sys.stderr)
            return None

    def find_prs_for_branch(self, owner_repo: str, branch: str) -> List[PullRequest]:
        """Find all PRs for a given branch"""
        # Search for PRs with this branch as head
        endpoint = f"/repos/{owner_repo}/pulls?head={owner_repo.split('/')[0]}:{branch}&state=all"
        data = self.get(endpoint)

        if not data:
            return []

        prs = []
        for pr_data in data:
            # Get review comments count (top-level conversations)
            review_comments, unresolved_conversations = self._get_review_comments_count(owner_repo, pr_data['number'])

            # Get CI status
            ci_status = self._get_ci_status(owner_repo, pr_data)

            # Check if merged
            is_merged = pr_data.get('merged_at') is not None

            # Get review decision
            review_decision = self._get_review_decision(owner_repo, pr_data['number'])

            # Get full PR details to check mergeable status (not included in list API)
            pr_details = self._get_pr_details(owner_repo, pr_data['number'])
            mergeable = pr_details.get('mergeable') if pr_details else None
            mergeable_state = pr_details.get('mergeable_state') if pr_details else None
            has_conflicts = (mergeable == False) or (mergeable_state == 'dirty')

            # Generate conflict message
            conflict_message = None
            if has_conflicts:
                base_branch = pr_data.get('base', {}).get('ref', 'main')
                conflict_message = f"This branch has conflicts that must be resolved (merge {base_branch} into this branch)"

            # Generate blocking message based on mergeable_state
            blocking_message = None
            if mergeable_state in ['blocked', 'unstable', 'behind']:
                if mergeable_state == 'unstable':
                    blocking_message = "Merging is blocked - Waiting on code owner review or required status checks"
                elif mergeable_state == 'blocked':
                    blocking_message = "Merging is blocked - Required reviews or checks not satisfied"
                elif mergeable_state == 'behind':
                    blocking_message = "This branch is out of date with the base branch"

            prs.append(PullRequest(
                number=pr_data['number'],
                title=pr_data['title'],
                state=pr_data['state'],
                url=pr_data['html_url'],
                branch=branch,
                author=pr_data['user']['login'],
                review_comments=review_comments,
                unresolved_conversations=unresolved_conversations,
                ci_status=ci_status,
                is_merged=is_merged,
                review_decision=review_decision,
                has_conflicts=has_conflicts,
                mergeable_state=mergeable_state,
                conflict_message=conflict_message,
                blocking_message=blocking_message
            ))

        return prs

    def _get_review_comments_count(self, owner_repo: str, pr_number: int) -> tuple[int, int]:
        """
        Get review comment counts for a PR.

        Returns:
            tuple: (total_comments, unresolved_conversations)
        """
        endpoint = f"/repos/{owner_repo}/pulls/{pr_number}/comments"
        data = self.get(endpoint)

        if not data:
            return 0, 0

        # Count total comments and top-level conversations (unresolved)
        total_comments = len(data)
        unresolved_conversations = sum(1 for comment in data if not comment.get('in_reply_to_id'))

        return total_comments, unresolved_conversations

    def _get_ci_status(self, owner_repo: str, pr_data: Dict) -> Optional[str]:
        """Get CI status for a PR"""
        # Try to get status from commits
        if 'head' in pr_data and 'sha' in pr_data['head']:
            sha = pr_data['head']['sha']
            endpoint = f"/repos/{owner_repo}/commits/{sha}/status"
            status_data = self.get(endpoint)

            if status_data and 'state' in status_data:
                state = status_data['state']
                # Map GitHub status states to our simplified states
                if state == 'success':
                    return 'passed'
                elif state == 'failure':
                    return 'failed'
                elif state in ['pending', 'error']:
                    return 'pending'

        return None

    def _get_pr_details(self, owner_repo: str, pr_number: int) -> Optional[Dict]:
        """Get full PR details (includes mergeable status)"""
        endpoint = f"/repos/{owner_repo}/pulls/{pr_number}"
        return self.get(endpoint)

    def _get_review_decision(self, owner_repo: str, pr_number: int) -> Optional[str]:
        """Get review decision for a PR (APPROVED, CHANGES_REQUESTED, etc.)"""
        endpoint = f"/repos/{owner_repo}/pulls/{pr_number}/reviews"
        reviews = self.get(endpoint)

        if not reviews:
            return None

        # Get the latest review from each reviewer
        latest_reviews = {}
        for review in reviews:
            reviewer = review['user']['login']
            # Keep only the most recent review per reviewer
            if reviewer not in latest_reviews or review['id'] > latest_reviews[reviewer]['id']:
                latest_reviews[reviewer] = review

        # Check review states
        has_approved = False
        has_changes_requested = False

        for review in latest_reviews.values():
            state = review.get('state')
            if state == 'APPROVED':
                has_approved = True
            elif state == 'CHANGES_REQUESTED':
                has_changes_requested = True

        # Return decision based on reviews
        if has_changes_requested:
            return 'CHANGES_REQUESTED'
        elif has_approved:
            return 'APPROVED'

        return None

    def branch_exists(self, owner_repo: str, branch: str) -> bool:
        """Check if branch exists on GitHub"""
        encoded_branch = quote(branch, safe='')
        endpoint = f"/repos/{owner_repo}/branches/{encoded_branch}"
        return self.get(endpoint) is not None


class BranchChecker:
    """Main class to check branches and PRs"""

    def __init__(self, base_dir: Path, github_token: Optional[str] = None, verbose: bool = False):
        self.base_dir = base_dir
        self.github = GitHubClient(github_token, verbose)
        self.git = GitHelper(verbose)  # Initialize GitHelper instance with verbose flag
        self.verbose = verbose

    def find_dynamo_repos(self) -> List[GitRepo]:
        """Find all dynamo* git repositories"""
        repos = []

        for item in self.base_dir.iterdir():
            if not item.is_dir() or not item.name.startswith('dynamo'):
                continue

            if not (item / '.git').exists():
                continue

            current_branch = self.git.get_current_branch(item)
            branches = self.git.get_all_branches(item)
            remote_url = self.git.get_remote_url(item)
            owner_repo = self.git.find_github_remote(item)

            repos.append(GitRepo(
                path=item,
                current_branch=current_branch,
                branches=branches,
                remote_url=remote_url,
                owner_repo=owner_repo
            ))

        return sorted(repos, key=lambda r: r.path.name)

    def check_repo(self, repo: GitRepo) -> Dict:
        """Check a single repository for branches and PRs"""
        result = {
            'repo': repo,
            'branches_with_prs': {},
            'branches_on_github_no_prs': [],
            'local_only_branches': []
        }

        if not repo.owner_repo or repo.owner_repo != 'ai-dynamo/dynamo':
            result['error'] = 'Not ai-dynamo/dynamo repository'
            return result

        # Check all branches in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all branch existence checks in parallel
            branch_futures = {}
            for branch in repo.branches:
                future = executor.submit(self.github.branch_exists, repo.owner_repo, branch)
                branch_futures[future] = branch

            # Wait for all checks to complete
            branch_exists_map = {}
            for future in as_completed(branch_futures):
                branch = branch_futures[future]
                try:
                    exists = future.result()
                    branch_exists_map[branch] = exists
                except Exception:
                    branch_exists_map[branch] = False

        # Now fetch PRs for branches that exist on GitHub (also in parallel)
        branches_on_github = [b for b, exists in branch_exists_map.items() if exists]

        pr_futures = {}
        if branches_on_github:
            with ThreadPoolExecutor(max_workers=4) as executor:
                for branch in branches_on_github:
                    future = executor.submit(self.github.find_prs_for_branch, repo.owner_repo, branch)
                    pr_futures[future] = branch

                # Wait for all PR fetches to complete
                branch_prs_map = {}
                for future in as_completed(pr_futures):
                    branch = pr_futures[future]
                    try:
                        prs = future.result()
                        branch_prs_map[branch] = prs
                    except Exception:
                        branch_prs_map[branch] = []

        # Organize results
        for branch in repo.branches:
            is_current = branch == repo.current_branch

            if not branch_exists_map.get(branch, False):
                # Local-only branch
                result['local_only_branches'].append({
                    'name': branch,
                    'is_current': is_current
                })
                continue

            # Branch exists on GitHub
            prs = branch_prs_map.get(branch, [])

            if prs:
                # Branch with PR
                result['branches_with_prs'][branch] = {
                    'is_current': is_current,
                    'prs': prs
                }
            else:
                # Branch on GitHub but no PR
                result['branches_on_github_no_prs'].append({
                    'name': branch,
                    'is_current': is_current
                })

        return result

    def print_results(self, results: List[Dict]):
        """Print results in a tree structure format"""
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        for result in results:
            repo = result['repo']
            # Root: repository name
            current = f" (on {repo.current_branch})" if repo.current_branch else ""
            print(f"\n{repo.path.name}/{current}")

            if 'error' in result:
                print(f"└─ ⚠️  {result['error']}")
                if repo.remote_url:
                    print(f"  └─ Remote: {repo.remote_url}")
                continue

            has_prs = bool(result['branches_with_prs'])
            has_local = bool(result['local_only_branches'])

            sections = []
            if has_prs:
                sections.append('prs')
            if has_local:
                sections.append('local')

            # If nothing to show, print message
            if not has_prs and not has_local:
                print(f"└─ No branches with PRs or local-only branches")
                continue

            # Branches with PRs
            if has_prs:
                is_last_section = sections[-1] == 'prs'
                section_prefix = "└─" if is_last_section else "├─"
                print(f"{section_prefix} Branches with PRs:")

                branch_items = list(result['branches_with_prs'].items())
                for idx, (branch, info) in enumerate(branch_items):
                    is_last_branch = (idx == len(branch_items) - 1) and is_last_section

                    if is_last_section:
                        branch_prefix = "  └─" if is_last_branch else "  ├─"
                    else:
                        branch_prefix = "│ └─" if is_last_branch else "│ ├─"

                    # Bold current branch
                    branch_display = f"\033[1m{branch}\033[0m" if info['is_current'] else branch
                    current_marker = " ⭐" if info['is_current'] else ""
                    print(f"{branch_prefix} {branch_display}{current_marker}")

                    for pr_idx, pr in enumerate(info['prs']):
                        is_last_pr = pr_idx == len(info['prs']) - 1

                        if is_last_section:
                            if is_last_branch:
                                pr_prefix = "    └─" if is_last_pr else "    ├─"
                                detail_prefix = "      " if is_last_pr else "    │ "
                            else:
                                pr_prefix = "  │ └─" if is_last_pr else "  │ ├─"
                                detail_prefix = "  │   " if is_last_pr else "  │ │ "
                        else:
                            if is_last_branch:
                                pr_prefix = "│   └─" if is_last_pr else "│   ├─"
                                detail_prefix = "│     " if is_last_pr else "│   │ "
                            else:
                                pr_prefix = "│ │ └─" if is_last_pr else "│ │ ├─"
                                detail_prefix = "│ │   " if is_last_pr else "│ │ │ "

                        # State emoji - check merged first
                        if pr.is_merged:
                            state_emoji = '🔀'
                        elif pr.state == 'open':
                            state_emoji = '📖'
                        elif pr.state == 'closed':
                            state_emoji = '❌'
                        else:
                            state_emoji = '📖'

                        # Truncate title if too long
                        title = pr.title[:80] + '...' if len(pr.title) > 80 else pr.title

                        print(f"{pr_prefix} {state_emoji} PR #{pr.number}: {title}")
                        print(f"{detail_prefix}URL: {pr.url}")

                        # Build status line
                        status_parts = []

                        # Review decision
                        if pr.review_decision == 'APPROVED':
                            status_parts.append("Review: ✅ Approved")
                        elif pr.review_decision == 'CHANGES_REQUESTED':
                            status_parts.append("Review: 🔴 Changes Requested")

                        # Unresolved conversations (not necessarily blocking)
                        if pr.unresolved_conversations > 0:
                            status_parts.append(f"💬 Comments: {pr.unresolved_conversations}")

                        # CI status
                        if pr.ci_status:
                            ci_icon = "✅" if pr.ci_status == "passed" else "❌" if pr.ci_status == "failed" else "⏳"
                            status_parts.append(f"CI: {ci_icon} {pr.ci_status}")

                        if status_parts:
                            print(f"{detail_prefix}Status: {', '.join(status_parts)}")

                        # Merge conflicts (show as separate line for visibility)
                        if pr.has_conflicts and pr.conflict_message:
                            print(f"{detail_prefix}⚠️  {pr.conflict_message}")

                        # Blocking message (show as separate line)
                        if pr.blocking_message:
                            print(f"{detail_prefix}🚫 {pr.blocking_message}")

            # Local-only branches
            if has_local:
                print(f"└─ Local-only branches:")

                for idx, branch_info in enumerate(result['local_only_branches']):
                    branch = branch_info['name']
                    is_last = idx == len(result['local_only_branches']) - 1
                    branch_prefix = "  └─" if is_last else "  ├─"

                    # Bold current branch
                    if branch_info['is_current']:
                        branch_display = f"\033[1m{branch}\033[0m ⭐"
                    else:
                        branch_display = branch

                    print(f"{branch_prefix} {branch_display}")


def main():
    parser = argparse.ArgumentParser(
        description="Check branches in dynamo* directories and find corresponding PRs"
    )
    parser.add_argument(
        '--base-dir',
        type=Path,
        default=Path.cwd(),
        help='Base directory to search for dynamo* repos (default: current directory)'
    )
    parser.add_argument(
        '--token',
        help='GitHub personal access token (or set GITHUB_TOKEN env var)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Configure logging if verbose
    if args.verbose:
        import logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(levelname)s - [%(name)s] %(message)s',
            stream=sys.stderr
        )

    # Token will be auto-detected in GitHubClient (from args, env, or gh CLI config)
    checker = BranchChecker(args.base_dir, args.token, args.verbose)

    if args.verbose:
        print(f"Scanning for dynamo* repositories in: {args.base_dir}")

    repos = checker.find_dynamo_repos()

    if not repos:
        print(f"No dynamo* git repositories found in {args.base_dir}")
        return 1

    if args.verbose:
        print(f"Found {len(repos)} repositories")

    results = []
    for repo in repos:
        if args.verbose:
            print(f"\nChecking {repo.path.name}...")
        result = checker.check_repo(repo)
        results.append(result)

    checker.print_results(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
