#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remove only old Dynamo images unused by every running or stopped container."""

import argparse
import datetime
import json
import math
import re
import subprocess
from collections import defaultdict


def docker(*args: str) -> str:
    return subprocess.run(
        ["docker", *args], check=True, capture_output=True, text=True, timeout=120
    ).stdout


def container_images() -> set[str]:
    ids = docker("ps", "-aq").split()
    if not ids:
        return set()
    return {row["Image"] for row in json.loads(docker("inspect", *ids))}


def buckets(image: dict) -> set[str] | None:
    tags = image.get("RepoTags") or []
    if not tags:
        return None  # Untagged images have no reliable Dynamo ownership.
    keys = set()
    for reference in tags:
        repo, separator, tag = reference.rpartition(":")
        if (
            not separator
            or repo.rsplit("/", 1)[-1] != "dynamo"
            or tag.startswith("latest")
        ):
            return None
        match = re.search(r"(?:^|-)[0-9a-f]{7,40}-(.+)$", tag)
        if match is None:
            return None
        keys.add(match.group(1))
    return keys


def candidates(
    images: list[dict],
    used: set[str],
    now: datetime.datetime,
    min_age_days: float,
    retain: int,
) -> list[dict]:
    if not math.isfinite(min_age_days) or min_age_days < 30:
        raise ValueError("minimum image age must be finite and at least 30 days")
    owned = [(row, buckets(row)) for row in images]
    groups = defaultdict(list)
    for row, keys in owned:
        if keys is not None:
            for key in keys:
                groups[key].append(row)
    protected = set()
    for rows in groups.values():
        rows.sort(key=lambda row: row["Created"], reverse=True)
        protected.update(row["Id"] for row in rows[:retain])
    cutoff = now - datetime.timedelta(days=min_age_days)
    result = []
    for row, keys in owned:
        created = datetime.datetime.fromisoformat(row["Created"].replace("Z", "+00:00"))
        if keys is not None and row["Id"] not in used | protected and created < cutoff:
            result.append(row)
    return sorted(result, key=lambda row: row["Created"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", "--dryrun", action="store_true")
    parser.add_argument("--retain", type=int, default=2)
    parser.add_argument("--min-age-days", type=float, default=30)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Compatibility option; deletion is never forced",
    )
    args = parser.parse_args()
    if (
        args.retain < 0
        or not math.isfinite(args.min_age_days)
        or args.min_age_days < 30
    ):
        parser.error("retain must be non-negative; minimum image age is 30 days")
    ids = sorted(set(docker("image", "ls", "-aq", "--no-trunc").split()))
    images = json.loads(docker("image", "inspect", *ids)) if ids else []
    now = datetime.datetime.now(datetime.timezone.utc)
    removed = 0
    for row in candidates(
        images, container_images(), now, args.min_age_days, args.retain
    ):
        image_id = row["Id"]
        # Revalidate tags and all container references before each exact-ID deletion.
        current = json.loads(docker("image", "inspect", image_id))[0]
        if current["RepoTags"] != row["RepoTags"] or image_id in container_images():
            print(f"image: skip changed or referenced {image_id}")
            continue
        print(
            f"image: {'would remove' if args.dry_run else 'removing'} "
            f"{image_id} {row['RepoTags']}"
        )
        if not args.dry_run:
            print(docker("image", "rm", "--no-prune", image_id), end="")
        removed += 1
    print(
        f"images: {'would remove' if args.dry_run else 'removed'} {removed}; "
        "containers, volumes, and cache unchanged"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
