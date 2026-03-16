#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Clean up old Docker images from the GitLab Container Registry.

Uses the GitLab API (via python-gitlab) to delete tags older than a
specified retention period from a specific registry repository.

Target registry path:
  gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo/dev/dynamo:<ImageSHA>.<commitSHA>-<framework>-<target>

Prerequisites:
  - pip install python-gitlab
  - GitLab private token at ~/.config/gitlab-token (or pass --token-path)

Usage:
  python3 clean_old_gitlab_images.py --project-path dl/ai-dynamo/dynamo --dry-run
  python3 clean_old_gitlab_images.py --project-path dl/ai-dynamo/dynamo --retain-days 28
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta, timezone

import gitlab

GITLAB_URL = "https://gitlab-master.nvidia.com"
GL_TOKEN_PATH = os.path.expanduser("~/.config/gitlab-token")
RETAIN_DAYS = 28
DEFAULT_REPOSITORY = "dev/dynamo"


def get_gitlab_client(gitlab_url, private_token_path):
    """Initializes and returns a GitLab API client."""
    with open(private_token_path) as f:
        private_token = f.read().strip()
    gl = gitlab.Gitlab(gitlab_url, private_token=private_token)
    gl.auth()
    return gl


def find_repository(project, repo_suffix):
    """Find a specific container registry repository by its path suffix.

    Args:
        project: GitLab project object
        repo_suffix: The sub-path to match (e.g. "dev/dynamo").
                     Matched against the end of each repository's path.

    Returns:
        The matching repository object, or None.
    """
    for repo in project.repositories.list(iterator=True):
        if repo.path.endswith(repo_suffix):
            return repo
    return None


def parse_created_at(created_at_str):
    """Parse GitLab ISO 8601 timestamp to a naive UTC datetime."""
    if not created_at_str:
        return None
    try:
        dt = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        return dt.replace(tzinfo=None)
    except ValueError:
        return None


def cleanup_repository(repo, retain_days, dry_run=True):
    """Delete tags older than retain_days from a single repository.

    In dry-run mode, fetches each tag's metadata to show which tags would
    be kept vs deleted. In non-dry-run mode, uses GitLab's server-side
    bulk-delete API for speed.
    """
    cutoff = datetime.utcnow() - timedelta(days=retain_days)
    older_than_spec = f"{retain_days}d"

    print(f"  Cutoff date: {cutoff.strftime('%Y-%m-%d %H:%M:%S')} UTC", flush=True)
    print(flush=True)

    # List tag names (lightweight, no created_at)
    print(f"  Listing tags ...", flush=True)
    t0 = time.time()
    tag_list = repo.tags.list(get_all=True)
    elapsed = time.time() - t0
    print(f"  Found {len(tag_list)} tag(s) ({elapsed:.1f}s)", flush=True)
    print(flush=True)

    if not tag_list:
        print("  Nothing to do.", flush=True)
        return

    # Fetch full metadata for each tag (needed for created_at)
    to_keep = []
    to_delete = []
    errors = []

    print(f"  Fetching tag details (1 API call per tag) ...", flush=True)
    for i, tag_stub in enumerate(tag_list):
        if (i + 1) % 25 == 0 or (i + 1) == len(tag_list):
            print(f"    [{i + 1}/{len(tag_list)}]", flush=True)

        try:
            tag = repo.tags.get(tag_stub.name)
        except gitlab.exceptions.GitlabGetError as e:
            errors.append((tag_stub.name, str(e)))
            continue

        created_at = parse_created_at(tag.created_at)
        if created_at is None:
            errors.append((tag.name, f"unparseable created_at: {tag.created_at!r}"))
            continue

        if created_at < cutoff:
            to_delete.append((tag.name, created_at))
        else:
            to_keep.append((tag.name, created_at))

    # Sort by date (oldest first for deletes, newest first for keeps)
    to_delete.sort(key=lambda x: x[1])
    to_keep.sort(key=lambda x: x[1], reverse=True)

    print(flush=True)

    if to_keep:
        print(f"  KEEP ({len(to_keep)} tags, newer than {retain_days} days):", flush=True)
        for name, created_at in to_keep:
            print(f"    {name}  ({created_at.strftime('%Y-%m-%d %H:%M')})", flush=True)
        print(flush=True)

    if to_delete:
        print(f"  DELETE ({len(to_delete)} tags, older than {retain_days} days):", flush=True)
        for name, created_at in to_delete:
            print(f"    D {name}  ({created_at.strftime('%Y-%m-%d %H:%M')})", flush=True)
        print(flush=True)

    if errors:
        print(f"  ERRORS ({len(errors)} tags could not be inspected):", flush=True)
        for name, err in errors:
            print(f"    ? {name}: {err}", flush=True)
        print(flush=True)

    # Summary
    print(f"  Summary: keep={len(to_keep)}  delete={len(to_delete)}  errors={len(errors)}", flush=True)

    if not to_delete:
        print("  Nothing to delete.", flush=True)
        return

    if dry_run:
        print(f"  DRY RUN: No tags were deleted.", flush=True)
    else:
        print(f"  Submitting bulk delete (older_than='{older_than_spec}') ...", flush=True)
        repo.tags.delete_in_bulk(older_than=older_than_spec)
        print(f"  Bulk delete request submitted (runs asynchronously on GitLab).", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Clean up old GitLab Container Registry images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s --project-path dl/ai-dynamo/dynamo --dry-run
  %(prog)s --project-path dl/ai-dynamo/dynamo --retain-days 14
  %(prog)s --project-path dl/ai-dynamo/dynamo --repository dev/dynamo --retain-days 28
""",
    )
    parser.add_argument(
        "--project-path", required=True,
        help="GitLab project path with namespace (e.g., 'dl/ai-dynamo/dynamo').",
    )
    parser.add_argument(
        "--repository", default=DEFAULT_REPOSITORY,
        help=f"Registry sub-path to clean (default: '{DEFAULT_REPOSITORY}').",
    )
    parser.add_argument(
        "--retain-days", type=int, default=RETAIN_DAYS,
        help=f"Keep images newer than N days (default: {RETAIN_DAYS}).",
    )
    parser.add_argument(
        "--token-path", default=GL_TOKEN_PATH,
        help=f"Path to GitLab private token file (default: {GL_TOKEN_PATH}).",
    )
    parser.add_argument(
        "--dry-run", "--dryrun", action="store_true",
        help="Print what would be done without deleting.",
    )
    args = parser.parse_args()

    if args.retain_days < 1:
        print("Error: --retain-days must be a positive integer", file=sys.stderr)
        sys.exit(2)

    print(f"GitLab container registry cleanup", flush=True)
    print(f"  retain-days: {args.retain_days}", flush=True)
    print(f"  dry-run:     {args.dry_run}", flush=True)
    print(f"  repository:  {args.repository}", flush=True)
    print(flush=True)

    print(f"Connecting to {GITLAB_URL} ...", flush=True)
    gl_client = get_gitlab_client(GITLAB_URL, args.token_path)
    print(f"Authenticated as: {gl_client.user.username}", flush=True)
    print(flush=True)

    print(f"Fetching project: {args.project_path} ...", flush=True)
    project = gl_client.projects.get(args.project_path)
    print(f"Project: {project.name} (ID: {project.id})", flush=True)
    print(flush=True)

    print(f"Looking for repository: .../{args.repository}", flush=True)
    repo = find_repository(project, args.repository)

    if repo is None:
        print(f"Error: repository ending with '{args.repository}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found: {repo.path} (ID: {repo.id})", flush=True)
    print(flush=True)

    cleanup_repository(repo, args.retain_days, dry_run=args.dry_run)

    print(flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
