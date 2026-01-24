"""
Cache for PR branch lookups.

Stores PR information for branches (both open and closed/merged PRs).
Cache key format: "{owner}/{repo}:{branch}"
Cache value: {"ts": timestamp, "prs": [list of PR dicts (includes updated_at)]}

TTL:
- Adaptive TTL based on most recent PR's updated_at:
  - age < 1h -> 2m
  - age < 2h -> 4m
  - age < 4h -> 30m
  - age < 8h -> 60m
  - age < 12h -> 80m
  - age >= 12h -> 120m
- Closed/merged PRs: 60d (longer TTL for immutable PRs)
- Branches with no PRs: fallback TTL (default 3600s)
"""

from pathlib import Path
from typing import Any, Dict, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cache.cache_base import BaseDiskCache
from common import dynamo_utils_cache_dir
from common_github.cache_ttl_utils import adaptive_ttl_s


class PRBranchCache(BaseDiskCache):
    """Cache for PR branch lookups with TTL-based invalidation."""
    
    def __init__(self):
        cache_file = dynamo_utils_cache_dir() / "pr_branches.json"
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
        Get cached PR branch data if fresh (using adaptive TTL if updated_at is available).
        
        Args:
            owner: Repository owner
            repo: Repository name
            branch: Branch name
            ttl_s: Fallback TTL in seconds (used if updated_at is not available or no PRs)
            
        Returns:
            {"ts": timestamp, "prs": [list of PR dicts]} or None if not found/stale
        """
        import time
        from datetime import datetime
        cache_key = f"{owner}/{repo}:{branch}"
        
        with self._mu:
            self._load_once()
            ent = self._check_item(cache_key)
            
            if not isinstance(ent, dict):
                return None
            
            ts = int(ent.get("ts", 0) or 0)
            now = int(time.time())
            
            # Adaptive TTL: use most recent PR's updated_at if available
            prs_list = ent.get("prs")
            if isinstance(prs_list, list) and prs_list:
                # Find most recent PR's updated_at
                most_recent_epoch = None
                for pr in prs_list:
                    if isinstance(pr, dict):
                        upd_str = pr.get("updated_at")
                        if upd_str:
                            try:
                                dt = datetime.fromisoformat(str(upd_str).replace("Z", "+00:00"))
                                upd_epoch = int(dt.timestamp())
                                if most_recent_epoch is None or upd_epoch > most_recent_epoch:
                                    most_recent_epoch = upd_epoch
                            except (ValueError, TypeError):
                                pass
                
                if most_recent_epoch is not None:
                    # Adaptive TTL for active PRs (via shared adaptive_ttl_s)
                    # For closed/merged PRs, use longer TTL (60d)
                    # Check if PR is closed/merged (all PRs in pr_branch are closed/merged)
                    is_closed = all(
                        pr.get("state", "").lower() in ("closed", "merged") or pr.get("is_merged", False)
                        for pr in prs_list if isinstance(pr, dict)
                    )
                    if is_closed:
                        effective_ttl = 60 * 24 * 3600  # 60d for closed/merged
                    else:
                        effective_ttl = adaptive_ttl_s(most_recent_epoch, default_ttl_s=ttl_s)
                else:
                    effective_ttl = ttl_s
            else:
                # No PRs: use fallback TTL
                effective_ttl = ttl_s
            
            if ts and (now - ts) <= max(0, int(effective_ttl)):
                return ent
            
            return None
    
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
        
        with self._mu:
            self._load_once()
            self._set_item(cache_key, value)
            self._persist()


# Global singleton
PR_BRANCH_CACHE = PRBranchCache()
