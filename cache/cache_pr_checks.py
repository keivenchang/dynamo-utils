"""Cache for PR check runs/statuses (GitHub Checks API).

Caching strategy:
  - Key: <owner>/<repo>#<pr_number>#<head_sha>
  - Value: GHPRChecksCacheEntry.to_disk_dict() format
           {ts, ver, rows: [{name, status_raw, duration, url, run_id, job_id, ...}], 
            check_runs_etag, status_etag, incomplete}
  - Short TTL (status changes frequently)
  
NOTE: This cache stores structured GHPRChecksCacheEntry objects, not raw API responses.
The caller is responsible for deserialization via GHPRChecksCacheEntry.from_disk_dict_strict().
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure imports work from both html_pages/ and parent directory
_module_dir = Path(__file__).resolve().parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

from cache_base import BaseDiskCache


class PRChecksCache(BaseDiskCache):
    """Cache for PR check runs and statuses.
    
    Stores structured GHPRChecksCacheEntry objects (serialized as dicts).
    The cache itself doesn't validate TTL or versioning - that's the caller's responsibility.
    
    Stats (hit/miss/write) are tracked automatically by BaseDiskCache.
    """
    
    _SCHEMA_VERSION = 1  # BaseDiskCache schema, not GHPRChecksCacheEntry.ver
    
    def __init__(self, *, cache_file: Path):
        super().__init__(cache_file=cache_file, schema_version=self._SCHEMA_VERSION)
    
    def _load_once(self) -> None:
        """Load cache with migration from old format.
        
        Old format (pre-migration):
            {
                "version": 7,
                "owner/repo#123:abc123": {...entry...},
                "owner/repo#456:def456": {...entry...}
            }
        
        New format (BaseDiskCache):
            {
                "version": 1,
                "items": {
                    "owner/repo#123:abc123": {...entry...},
                    "owner/repo#456:def456": {...entry...}
                }
            }
        """
        if self._loaded:
            return
        self._loaded = True

        if not self._cache_file.exists():
            self._data = self._create_empty_cache()
            self._initial_disk_count = 0
            return

        try:
            data = json.loads(self._cache_file.read_text() or "{}")
        except Exception:
            data = {}

        if not isinstance(data, dict):
            data = {}

        # Migration: Check for any top-level entries (keys that aren't "version" or "items")
        items_dict = data.get("items")
        if not isinstance(items_dict, dict):
            items_dict = {}
        
        # Find any orphaned top-level entries
        migrated_count = 0
        for key, value in list(data.items()):
            if key not in ["version", "items"]:
                # Move to items dict
                items_dict[key] = value
                migrated_count += 1
        
        # Clean up: remove old top-level entries
        if migrated_count > 0:
            # Create clean new structure
            data = {
                "version": self._schema_version,
                "items": items_dict
            }
            self._dirty = True
        else:
            # No migration needed, just ensure proper structure
            data["items"] = items_dict
            data["version"] = self._schema_version

        # Track initial disk count
        self._initial_disk_count = len(data.get("items", {}))
        
        self._data = data
    
    def get_entry_dict(self, key: str) -> Optional[Dict[str, Any]]:
        """Get raw cache entry dict (GHPRChecksCacheEntry.to_disk_dict() format).
        
        Args:
            key: Cache key (e.g., "owner/repo#123#abc123")
            
        Returns:
            Entry dict, or None if not cached
            
        Note: Caller must validate TTL, version, and deserialize via from_disk_dict_strict()
        """
        with self._mu:
            self._load_once()
            return self._check_item(key)  # Automatically tracks hit/miss
    
    def put_entry_dict(self, key: str, entry_dict: Dict[str, Any]) -> None:
        """Store entry dict (from GHPRChecksCacheEntry.to_disk_dict()).
        
        Args:
            key: Cache key
            entry_dict: Serialized GHPRChecksCacheEntry
        """
        with self._mu:
            self._load_once()
            self._set_item(key, entry_dict)  # Automatically tracks write
            self._persist()


# Singleton cache instance
def _get_cache_file() -> Path:
    """Get cache file path, handling imports from different contexts."""
    try:
        parent_dir = _module_dir.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        import common
        return common.dynamo_utils_cache_dir() / "pr_checks.json"
    except ImportError:
        return Path.home() / ".cache" / "dynamo-utils" / "pr_checks.json"


PR_CHECKS_CACHE = PRChecksCache(cache_file=_get_cache_file())
