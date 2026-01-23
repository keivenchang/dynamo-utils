"""Cache for required status checks (from branch protection rules).

Caching strategy:
  - Key: <owner>/<repo>:required_checks:<base_ref>
  - Value: Dict with 'ts', 'val' (set/list of check names), 'ok', 'pr_state', 'pr_updated_at_epoch'
  - Tiered TTL based on PR state and commit age
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

# Ensure imports work from both html_pages/ and parent directory
_module_dir = Path(__file__).resolve().parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

from cache_base import BaseDiskCache


class RequiredChecksCache(BaseDiskCache):
    """Cache for required status checks from branch protection.
    
    Stores the set of required check names for a given branch, along
    with metadata about PR state and age for smart TTL handling.
    
    Stats (hit/miss/write) are tracked automatically by BaseDiskCache.
    """
    
    _SCHEMA_VERSION = 1
    
    def __init__(self, *, cache_file: Path):
        super().__init__(cache_file=cache_file, schema_version=self._SCHEMA_VERSION)
    
    def get_if_valid(self, key: str, *, cache_only_mode: bool = False, check_ttl: bool = True) -> Optional[Dict[str, Any]]:
        """Get cached required checks if valid.
        
        Args:
            key: Cache key (e.g., "owner/repo:required_checks:main")
            cache_only_mode: If True, always return cached value regardless of TTL
            check_ttl: If False, skip TTL validation
            
        Returns:
            Dict with 'val' (set of check names), 'ok', 'pr_state', 'pr_updated_at_epoch'
            or None if not cached/invalid
        """
        with self._mu:
            self._load_once()
            ent = self._check_item(key)  # Automatically tracks hit/miss
            
            if not isinstance(ent, dict):
                return None
            
            # Extract value (stored as list on disk, return as set)
            val = ent.get("val")
            if isinstance(val, list):
                val_set = set(val)
            elif isinstance(val, set):
                val_set = val
            else:
                return None
            
            # In cache_only_mode, always return cached value
            if cache_only_mode or not check_ttl:
                return {
                    "val": val_set,
                    "ok": ent.get("ok", True),
                    "pr_state": ent.get("pr_state", "open"),
                    "pr_updated_at_epoch": ent.get("pr_updated_at_epoch"),
                }
            
            # Otherwise, check if entry is still fresh
            # (Caller should implement TTL logic based on pr_state/age)
            ts = int(ent.get("ts", 0) or 0)
            if ts:
                return {
                    "val": val_set,
                    "ok": ent.get("ok", True),
                    "pr_state": ent.get("pr_state", "open"),
                    "pr_updated_at_epoch": ent.get("pr_updated_at_epoch"),
                    "ts": ts,
                }
            
            return None
    
    def put(
        self,
        key: str,
        val: Set[str],
        *,
        ok: bool = True,
        pr_state: str = "open",
        pr_updated_at_epoch: Optional[int] = None,
    ) -> None:
        """Store required checks.
        
        Args:
            key: Cache key
            val: Set of required check names
            ok: Whether the fetch was successful
            pr_state: PR state (open/closed/merged)
            pr_updated_at_epoch: PR updated_at timestamp (epoch seconds)
        """
        now = int(time.time())
        with self._mu:
            self._load_once()
            # Store set as list for JSON serialization
            entry = {
                "ts": now,
                "val": list(val) if isinstance(val, set) else val,
                "ok": ok,
                "pr_state": pr_state,
                "pr_updated_at_epoch": pr_updated_at_epoch,
            }
            self._set_item(key, entry)  # Automatically tracks write
            self._persist()


# Singleton cache instance
def _get_cache_file() -> Path:
    """Get cache file path, handling imports from different contexts."""
    try:
        parent_dir = _module_dir.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        import common
        return common.dynamo_utils_cache_dir() / "required_checks.json"
    except ImportError:
        return Path.home() / ".cache" / "dynamo-utils" / "required_checks.json"


REQUIRED_CHECKS_CACHE = RequiredChecksCache(cache_file=_get_cache_file())
