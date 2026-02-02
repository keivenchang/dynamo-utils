"""Shared TTL calculation utilities for all caches.

This module provides a unified adaptive TTL calculation that is used across
all time-based caches (jobs, PRs, etc.) to ensure consistent behavior.
"""
import time
from typing import Union


def adaptive_ttl_s(timestamp_epoch: Union[int, None], default_ttl_s: int = 180) -> int:
    """Unified adaptive TTL for all time-based caches.

    Schedule (updated 2026-02 to reduce staleness for old PRs/commits):
      - age < 1h   -> 2m (120s)
      - age < 2h   -> 4m (240s)
      - age < 4h   -> 10m (600s)   [reduced from 30m]
      - age < 8h   -> 15m (900s)   [reduced from 60m]
      - age < 12h  -> 20m (1200s)  [reduced from 80m]
      - age >= 12h -> 30m (1800s)  [reduced from 120m to detect new CI faster]

    Args:
        timestamp_epoch: Entity's timestamp (job started_at, PR updated_at, etc.) in epoch seconds
        default_ttl_s: Fallback TTL if timestamp is unknown or invalid

    Returns:
        TTL in seconds
    """
    now = int(time.time())
    try:
        ts = int(timestamp_epoch or 0)
    except (ValueError, TypeError):
        ts = 0
    if ts <= 0 or ts > now:
        return int(default_ttl_s)
    age = int(now) - ts
    if age < 3600:        # < 1h
        return 120        # 2m
    if age < 7200:        # < 2h
        return 240        # 4m
    if age < 14400:       # < 4h
        return 600        # 10m (reduced from 30m)
    if age < 28800:       # < 8h
        return 900        # 15m (reduced from 60m)
    if age < 43200:       # < 12h
        return 1200       # 20m (reduced from 80m)
    return 1800           # 30m (reduced from 120m)


def pulls_list_adaptive_ttl_s(timestamp_epoch: Union[int, None], default_ttl_s: int = 180) -> int:
    """Adaptive TTL schedule for the pulls list cache.

    This is intentionally *more aggressive* than the generic adaptive_ttl_s()
    because open PR lists can change at any time and we want fresher "is PR updated?"
    behavior without forcing other caches (jobs, required checks, etc.) to refresh more often.

    Schedule:
      - age < 1h   -> 1m (60s)
      - age < 2h   -> 2m (120s)
      - age < 4h   -> 4m (240s)
      - age >= 4h  -> 8m (480s)
    """
    now = int(time.time())
    try:
        ts = int(timestamp_epoch or 0)
    except (ValueError, TypeError):
        ts = 0
    if ts <= 0 or ts > now:
        return int(default_ttl_s)
    age = int(now) - ts
    if age < 3600:
        return 60
    if age < 7200:
        return 120
    if age < 14400:
        return 240
    return 480
