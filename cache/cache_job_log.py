"""Cache for extracted error snippets from job logs.

Caching strategy:
  - Key: job_id (string)
  - Value: extracted error snippet text
  - Persistent disk cache with automatic stats tracking
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Ensure imports work from both html_pages/ and parent directory
_module_dir = Path(__file__).resolve().parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

from cache_base import BaseDiskCache


class JobLogCache(BaseDiskCache):
    """Cache extracted error snippets from job logs.
    
    This cache stores the extracted error snippets (not the full logs)
    to avoid re-parsing job logs on every page generation.
    
    Stats (hit/miss/write) are tracked automatically by BaseDiskCache.
    """
    
    _SCHEMA_VERSION = 1
    
    def __init__(self, *, cache_file: Path):
        super().__init__(cache_file=cache_file, schema_version=self._SCHEMA_VERSION)
    
    def get(self, job_id: str) -> Optional[str]:
        """Get cached error snippet for a job.
        
        Args:
            job_id: The job ID (string)
            
        Returns:
            The cached error snippet, or None if not cached
        """
        with self._mu:
            self._load_once()
            value = self._check_item(job_id)  # Automatically tracks hit/miss
            if value is None:
                return None
            return str(value) if isinstance(value, str) else None
    
    def put(self, job_id: str, snippet: str) -> None:
        """Store error snippet for a job.
        
        Args:
            job_id: The job ID (string)
            snippet: The extracted error snippet text
        """
        with self._mu:
            self._load_once()
            self._set_item(job_id, snippet)  # Automatically tracks write
            self._persist()


# Singleton cache instance
def _get_cache_file() -> Path:
    """Get cache file path, handling imports from different contexts."""
    try:
        # Try importing from parent directory
        parent_dir = _module_dir.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        import common
        return common.dynamo_utils_cache_dir() / "job-logs" / "job_log_cache.json"
    except ImportError:
        # Fallback for testing
        return Path.home() / ".cache" / "dynamo-utils" / "job-logs" / "job_log_cache.json"


JOB_LOG_CACHE = JobLogCache(cache_file=_get_cache_file())
