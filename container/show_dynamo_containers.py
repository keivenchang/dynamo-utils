#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Show running Dynamo devcontainers with backend, branch, commit, and host path."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


KNOWN_BACKENDS = {"vllm", "sglang", "trtllm"}


def run(cmd: list[str], *, check: bool = True) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=check)
    return result.stdout.strip()


def get_container_ids() -> list[str]:
    out = run(["sudo", "docker", "ps", "-q", "--no-trunc"])
    if not out:
        return []
    return out.splitlines()


def inspect_containers(ids: list[str]) -> list[dict]:
    out = run(["sudo", "docker", "inspect"] + ids)
    return json.loads(out)


def detect_backend(config_file: str) -> str:
    """Detect LLM backend from the devcontainer config path.

    Looks for known backend names in the path components, e.g.:
      .devcontainer/sglang/devcontainer.json -> sglang
      .devcontainer/keivenc_vllm/devcontainer.json -> vllm
    """
    lower = config_file.lower()
    for backend in KNOWN_BACKENDS:
        if backend in lower:
            return backend
    return "unknown"


def git_info(repo_path: str) -> tuple[str, str]:
    """Return (branch, short_sha) for a repo path."""
    branch = run(
        ["git", "-C", repo_path, "branch", "--show-current"], check=False
    )
    if not branch:
        branch = "(detached)"
    sha = run(
        ["git", "-C", repo_path, "rev-parse", "--short=11", "HEAD"], check=False
    )
    return branch, sha


def compose_repo_name(config_files: str) -> str:
    """Extract repo name from docker-compose config path."""
    m = re.search(r"/dynamo/(dynamo[^/]+)/", config_files)
    return m.group(1) if m else "unknown"


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Print a table with auto-sized columns."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    gap = "   "
    header_line = gap.join(f"{h:<{widths[i]}}" for i, h in enumerate(headers))
    total = sum(widths) + len(gap) * (len(headers) - 1)

    print(header_line)
    print("-" * total)
    for row in rows:
        print(gap.join(f"{cell:<{widths[i]}}" for i, cell in enumerate(row)))


def main() -> None:
    if Path("/.dockerenv").exists():
        print(
            "ERROR: This can only run on the host, not inside the container.",
            file=sys.stderr,
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Show running Dynamo devcontainers."
    )
    parser.add_argument(
        "--infra", action="store_true",
        help="Also show infrastructure services (docker-compose).",
    )
    args = parser.parse_args()

    ids = get_container_ids()
    if not ids:
        print("No running Dynamo containers.")
        return

    containers = inspect_containers(ids)

    devcontainers: list[dict] = []
    infra_services: list[dict] = []

    for c in containers:
        labels = c.get("Config", {}).get("Labels", {})
        local_folder = labels.get("devcontainer.local_folder", "")
        config_file = labels.get("devcontainer.config_file", "")
        compose_service = labels.get("com.docker.compose.service", "")
        compose_config = labels.get(
            "com.docker.compose.project.config_files", ""
        )

        if local_folder and config_file:
            backend = detect_backend(config_file)
            repo = Path(local_folder).name
            container_id = c.get("Id", "")[:12]
            container_name = c.get("Name", "").lstrip("/")
            branch, sha = git_info(local_folder)
            user = run(
                ["sudo", "docker", "exec", container_name, "whoami"],
                check=False,
            ) or "unknown"
            dynamo_sha = run(
                ["sudo", "docker", "exec", container_name,
                 "bash", "-c", "echo ${DYNAMO_COMMIT_SHA:-n/a}"],
                check=False,
            ) or "n/a"
            # Show short form (11 chars) to match git commit SHA column
            dynamo_sha_short = dynamo_sha[:11] if dynamo_sha != "n/a" else "n/a"
            devcontainers.append({
                "repo": repo,
                "backend": backend,
                "branch": branch,
                "sha": sha,
                "host_path": local_folder,
                "user": user,
                "container_id": container_id,
                "container_name": container_name,
                "dynamo_sha": dynamo_sha_short,
            })
        elif compose_service and args.infra:
            image = c.get("Config", {}).get("Image", "")
            repo = compose_repo_name(compose_config)
            infra_services.append({
                "service": compose_service,
                "image": image,
                "repo": repo,
            })

    devcontainers.sort(key=lambda x: x["repo"])
    infra_services.sort(key=lambda x: x["service"])

    if devcontainers:
        print_table(
            ["Repo", "Backend", "User", "Container ID", "Container Name",
             "Branch", "Git HEAD", "Dynamo SHA", "Host Path"],
            [
                [dc["repo"], dc["backend"], dc["user"],
                 dc["container_id"], dc["container_name"],
                 dc["branch"], dc["sha"], dc["dynamo_sha"], dc["host_path"]]
                for dc in devcontainers
            ],
        )

    if infra_services:
        if devcontainers:
            print()
        repo_set = sorted({s["repo"] for s in infra_services})
        print(f"Infrastructure services (from {', '.join(repo_set)})")
        print_table(
            ["Service", "Image"],
            [[s["service"], s["image"]] for s in infra_services],
        )

    if not devcontainers and not infra_services:
        print("No running Dynamo containers found.")


if __name__ == "__main__":
    main()
