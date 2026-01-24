"""Shared TTL calculation utilities for all caches.

This module provides a unified adaptive TTL calculation that is used across
all time-based caches (jobs, PRs, etc.) to ensure consistent behavior.
"""
import time
from typing import Union


def adaptive_ttl_s(timestamp_epoch: Union[int, None], default_ttl_s: int = 180) -> int:
    """Unified adaptive TTL for all time-based caches.

    Schedule:
      - age < 1h   -> 2m (120s)
      - age < 2h   -> 4m (240s)
      - age < 4h   -> 30m (1800s)
      - age < 8h   -> 60m (3600s)
      - age < 12h  -> 80m (4800s)
      - age >= 12h -> 120m (7200s)

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
    if age < 3600:
        return 120
    if age < 7200:
        return 240
    if age < 14400:
        return 1800
    if age < 28800:
        return 3600
    if age < 43200:
        return 4800
    return 7200
