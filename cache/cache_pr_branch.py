"""
Cache for PR branch lookups.

Stores PR information for branches (both open and closed/merged PRs).
Cache key format: "{owner}/{repo}:{branch}"
Cache value: {"ts": timestamp, "prs": [list of PR dicts]}

TTL:
- Branches with no PRs: shorter TTL (no_pr_ttl_s, default 3600s)
- Branches with closed/merged PRs: longer TTL (closed_ttl_s, default 86400s)
"""

from pathlib import Path
from typing import Any, Dict, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cache.cache_base import BaseDiskCache
from common import dynamo_utils_cache_dir


class PRBranchCache(BaseDiskCache):
    """Cache for PR branch lookups with TTL-based invalidation."""
    
    def __init__(self):
        cache_file = dynamo_utils_cache_dir() / "pr-branches" / "pr_branch_cache.json"
        super().__init__(cache_file=cache_file)
    
    def get_if_fresh(
        self,
        *,
        owner: str,
        repo: str,
        branch: str,
        ttl_s: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached PR branch data if fresh.
        
        Args:
            owner: Repository owner
            repo: Repository name
            branch: Branch name
            ttl_s: TTL in seconds
            
        Returns:
            {"ts": timestamp, "prs": [list of PR dicts]} or None if not found/stale
        """
        cache_key = f"{owner}/{repo}:{branch}"
        return super().get_if_fresh(cache_key, ttl_s=ttl_s)
    
    def put(
        self,
        *,
        owner: str,
        repo: str,
        branch: str,
        value: Dict[str, Any],
    ) -> None:
        """
        Store PR branch data.
        
        Args:
            owner: Repository owner
            repo: Repository name
            branch: Branch name
            value: {"ts": timestamp, "prs": [list of PR dicts]}
        """
        cache_key = f"{owner}/{repo}:{branch}"
        super().put(cache_key, value)


# Global singleton
PR_BRANCH_CACHE = PRBranchCache()
