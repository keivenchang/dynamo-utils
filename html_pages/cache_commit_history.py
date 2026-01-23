#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Commit History Cache

Caches commit metadata including:
- Composite SHA (image SHA, hash of container/ contents)
- Author, date, merge date
- Commit message and stats
- Changed files

Cache file: ~/.cache/dynamo-utils/commit-history.json

Cache key format: <full_commit_sha>

Cache entry format:
{
    "composite_docker_sha": "746bc31d05b3",  # image SHA (12 chars)
    "author": "John Doe",
    "author_email": "john@example.com",
    "date": "2025-12-17 20:03:39",
    "merge_date": "2025-12-17 21:15:30",  # null if not merged
    "message": "feat: Add new feature (#1234)",  # first line
    "full_message": "feat: Add new feature (#1234)\n\nDetailed description...",
    "stats": {
        "files": 5,
        "insertions": 100,
        "deletions": 50
    },
    "changed_files": ["path/to/file1.py", "path/to/file2.py"]
}

TTL: None (immutable, entries never expire since commits don't change)
"""

from __future__ import annotations

import sys
from pathlib import Path as PathLib

# Ensure parent directory is in sys.path for imports
_this_dir = PathLib(__file__).resolve().parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))
_parent_dir = _this_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from pathlib import Path
from typing import Any, Dict, Optional

import sys
from pathlib import Path

# Import from parent cache/ directory
_parent_dir = Path(__file__).resolve().parent.parent
_cache_dir = _parent_dir / "cache"
if str(_cache_dir) not in sys.path:
    sys.path.insert(0, str(_cache_dir))

from cache_base import BaseDiskCache
import common


class CommitHistoryCache(BaseDiskCache):
    """Disk-backed cache for commit metadata with automatic persistence.
    
    Stats (hit/miss/write) are tracked automatically by BaseDiskCache.
    """

    def __init__(self, *, cache_file: Path):
        super().__init__(cache_file=cache_file, schema_version=1)
        # self.stats inherited from BaseDiskCache

    def get(self, commit_sha: str) -> Optional[Dict[str, Any]]:
        """
        Get cached commit metadata.

        Args:
            commit_sha: Full commit SHA

        Returns:
            Cached entry dict or None if not found
            
        Note: Stats (hit/miss) are tracked automatically by _check_item()
        """
        with self._mu:
            self._load_once()
            return self._check_item(commit_sha)  # Automatically tracks hit/miss

    def put(self, commit_sha: str, entry: Dict[str, Any]) -> None:
        """
        Store commit metadata in cache.

        Args:
            commit_sha: Full commit SHA
            entry: Commit metadata dict
            
        Note: Stats (write) are tracked automatically by _set_item()
        """
        with self._mu:
            self._load_once()
            self._set_item(commit_sha, entry)  # Automatically tracks write

    def bulk_update(self, entries: Dict[str, Dict[str, Any]]) -> None:
        """
        Update multiple cache entries at once.

        Args:
            entries: Dict mapping commit SHA -> entry dict
        """
        items = self._get_items()
        items.update(entries)
        self._persist()


# Singleton cache used by all dashboard generators
COMMIT_HISTORY_CACHE = CommitHistoryCache(
    cache_file=(common.dynamo_utils_cache_dir() / "commit-history.json")
)
