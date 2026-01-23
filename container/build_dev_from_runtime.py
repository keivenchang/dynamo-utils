#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Build a dev image from a runtime image.

Given a runtime image like:
  gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo:224f63f56a478a7abe85b24b6411ff90101672ab-42221925-vllm-amd64

This script will:
1. Pull it to local as dynamo:224f63f56a478a7abe85b24b6411ff90101672ab-42221925-vllm-amd64
2. Generate a Dockerfile.dev (based on dynamo_ci/container/dev/Dockerfile.dev pattern)
3. Build the dev image using the runtime image as base

Usage:
  ./build_from_runtime.py <runtime-image-url>
  ./build_from_runtime.py gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo:224f63f56a478a7abe85b24b6411ff90101672ab-42221925-vllm-amd64
"""

import argparse
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

_KNOWN_FRAMEWORKS = ("vllm", "trtllm", "sglang", "none")


def infer_framework_from_tag(image_tag: str) -> str | None:
    """Infer framework name from a Dynamo image tag.

    Expected patterns include "-vllm-", "-trtllm-", "-sglang-", "-none-".
    Returns the lowercase framework string, or None if not recognized.
    """
    tag_lower = image_tag.lower()
    for fw in _KNOWN_FRAMEWORKS:
        if f"-{fw}-" in tag_lower:
            return fw
    return None


def parse_image_url(image_url: str) -> tuple[str, str, str]:
    """
    Parse image URL into components.
    
    Args:
        image_url: Full image URL like gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo:tag
    
    Returns:
        (registry, repository, tag) tuple
        e.g., ("gitlab-master.nvidia.com:5005/dl/ai-dynamo", "dynamo", "224f63f56a478a7abe85b24b6411ff90101672ab-42221925-vllm-amd64")
    """
    if ":" not in image_url:
        logger.error(f"Invalid image URL (no tag): {image_url}")
        sys.exit(1)
    
    # Split into image_part and tag
    *image_parts, tag = image_url.rsplit(":", 1)
    image_part = ":".join(image_parts)
    
    # Split image_part into registry and repo
    if "/" not in image_part:
        logger.error(f"Invalid image URL (no registry/repo): {image_url}")
        sys.exit(1)
    
    parts = image_part.split("/")
    repo = parts[-1]
    registry = "/".join(parts[:-1])
    
    return registry, repo, tag


def run_command(
    cmd: list[str],
    description: str,
    check: bool = True,
    stream_output: bool = False,
) -> subprocess.CompletedProcess:
    """Run a shell command and handle errors.

    If stream_output is True, stdout/stderr are not captured and are streamed directly
    to the terminal (useful for `docker pull` progress output).
    """
    logger.info(f"{description}...")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.debug(f"Running: {' '.join(cmd)}")

    try:
        if stream_output:
            # Let the subprocess write directly to our stdout/stderr so progress UIs render.
            return subprocess.run(cmd, check=check)

        result = subprocess.run(
            cmd,
            check=check,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            logger.debug(result.stdout)
        if result.stderr:
            logger.debug(result.stderr)
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {' '.join(cmd)}")
        if getattr(e, "stdout", None):
            logger.error(f"STDOUT:\n{e.stdout}")
        if getattr(e, "stderr", None):
            logger.error(f"STDERR:\n{e.stderr}")
        raise


def pull_and_retag_image(full_image_url: str, local_tag: str, dry_run: bool = False) -> None:
    """Pull image from registry and retag it locally."""
    if dry_run:
        logger.info(f"[DRY RUN] Would pull: {full_image_url}")
        logger.info(f"[DRY RUN] Would tag as: {local_tag}")
        logger.info(f"[DRY RUN] Would remove remote tag: {full_image_url}")
        return
    
    # Pull from registry
    run_command(
        ["docker", "pull", full_image_url],
        f"Pulling {full_image_url}",
        stream_output=True,
    )
    
    # Tag locally
    run_command(
        ["docker", "tag", full_image_url, local_tag],
        f"Tagging as {local_tag}"
    )

    # Drop the remote tag locally so the image is referred to only by our local tag.
    # This does NOT delete layers if they're still referenced (local_tag keeps them alive).
    run_command(
        ["docker", "rmi", full_image_url],
        f"Removing remote tag {full_image_url}",
        check=False,
    )
    
    logger.info(f"✓ Image available locally as: {local_tag}")


def generate_dockerfile_dev(base_image: str, output_path: Path, dry_run: bool = False) -> None:
    """
    Generate a Dockerfile.dev that uses the runtime image as base.
    
    Uses the template from container/Dockerfile.dev.template and replaces
    the FROM runtime AS dev line with the specified base_image.
    Note: dynamo_tools stage uses ubuntu:24.04, not the runtime image.
    """
    if dry_run:
        logger.info(f"[DRY RUN] Would generate Dockerfile.dev at: {output_path}")
        logger.info(f"[DRY RUN] Base image: {base_image}")
        return
    
    # Find the Dockerfile.dev template (should be in same directory as this script)
    script_dir = Path(__file__).parent
    template_path = script_dir / "Dockerfile.dev.template"
    
    if not template_path.exists():
        logger.error(f"Could not find Dockerfile.dev.template at: {template_path}")
        sys.exit(1)
    
    # Read the template
    template_content = template_path.read_text()
    
    # Replace "FROM runtime AS dev" with our base image
    # Note: dynamo_tools uses ubuntu:24.04, so we only replace the dev stage
    modified_content = template_content.replace(
        "FROM runtime AS dev",
        f"FROM {base_image} AS dev"
    )
    
    output_path.write_text(modified_content)
    logger.info(f"✓ Generated Dockerfile.dev at: {output_path}")
    logger.info(f"  Template: {template_path}")
    logger.info(f"  Base image: {base_image}")


def build_dev_image(
    dockerfile_path: Path,
    context_dir: Path,
    base_tag: str,
    output_tag: str,
    target: str = "dev",
    dry_run: bool = False,
) -> None:
    """Build the requested Dockerfile target using docker build."""
    image_tag = output_tag
    
    # Detect architecture from tag (e.g., -amd64 or -arm64)
    arch = "amd64"
    arch_alt = "x86_64"
    if "-arm64" in base_tag or "-aarch64" in base_tag:
        arch = "arm64"
        arch_alt = "aarch64"
    elif "-amd64" in base_tag or "-x86_64" in base_tag:
        arch = "amd64"
        arch_alt = "x86_64"
    
    if dry_run:
        logger.info(f"[DRY RUN] Would build image: {image_tag}")
        logger.info(f"[DRY RUN] Using dockerfile: {dockerfile_path}")
        logger.info(f"[DRY RUN] Context: {context_dir}")
        logger.info(f"[DRY RUN] ARCH={arch}, ARCH_ALT={arch_alt}")
        logger.info(f"[DRY RUN] Target: {target}")
        if target == "local-dev":
            logger.info(f"[DRY RUN] USER_UID={os.getuid()}, USER_GID={os.getgid()}")
        return
    
    logger.info(f"Detected architecture: ARCH={arch}, ARCH_ALT={arch_alt}")
    
    build_cmd = [
            "docker", "build",
            "-f", str(dockerfile_path),
            "-t", image_tag,
            "--build-arg", f"ARCH={arch}",
            "--build-arg", f"ARCH_ALT={arch_alt}",
            "--target", target,
        ]
    if target == "local-dev":
        build_cmd += [
            "--build-arg",
            f"USER_UID={os.getuid()}",
            "--build-arg",
            f"USER_GID={os.getgid()}",
        ]
    build_cmd.append(str(context_dir))

    run_command(
        build_cmd,
        f"Building image: {image_tag} (target={target})",
        stream_output=True,
    )
    
    logger.info(f"✓ Image built: {image_tag}")


def main():
    parser = argparse.ArgumentParser(
        description="Build a dev image from a runtime image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build dev image from a runtime image
  %(prog)s gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo:224f63f56a478a7abe85b24b6411ff90101672ab-42221925-vllm-amd64

  # Specify custom output tag
  %(prog)s --output-tag my-dev:latest gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo:abc123-vllm-amd64

  # Keep generated Dockerfile
  %(prog)s --keep-dockerfile gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo:abc123-vllm-amd64
        """
    )
    parser.add_argument(
        "runtime_image",
        help="Full runtime image URL (e.g., registry.com/repo/image:tag)"
    )
    parser.add_argument(
        "--target",
        choices=["dev", "local-dev"],
        default="dev",
        help="Which Dockerfile target to build (default: dev)"
    )
    parser.add_argument(
        "--output-tag",
        help="Output tag for built image (default: <repo>:<runtime-tag>-<target>)"
    )
    parser.add_argument(
        "--no-tag-latest",
        action="store_true",
        help="Do not add a latest-{framework} tag to the built image (mirrors container/build.sh behavior)",
    )
    parser.add_argument(
        "--keep-dockerfile",
        action="store_true",
        help="Keep the generated Dockerfile.dev (saved to ./Dockerfile.dev.generated)"
    )
    parser.add_argument(
        "--skip-pull",
        action="store_true",
        help="Skip pulling image (assume it's already available locally)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--dry-run", "--dryrun",
        dest="dry_run",
        action="store_true",
        help="Show what would be done without executing"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse image URL
    registry, repo, tag = parse_image_url(args.runtime_image)
    local_tag = f"{repo}:{tag}"
    default_output_tag = f"{repo}:{tag}-{args.target}"
    output_tag = args.output_tag or default_output_tag
    framework = infer_framework_from_tag(tag)
    latest_tag = None
    if not args.no_tag_latest and framework is not None:
        # Match build.sh-style "latest" tags, but always include the target suffix for clarity:
        # - dev:       dynamo:latest-<framework>-dev
        # - local-dev: dynamo:latest-<framework>-local-dev
        if args.target == "local-dev":
            latest_tag = f"{repo}:latest-{framework}-local-dev"
        else:
            latest_tag = f"{repo}:latest-{framework}-dev"
    
    logger.info("=" * 70)
    logger.info("Building dev image from runtime image")
    logger.info("=" * 70)
    logger.info(f"Runtime image: {args.runtime_image}")
    logger.info(f"Local tag:     {local_tag}")
    logger.info(f"Output tag:    {output_tag}")
    if latest_tag is not None:
        logger.info(f"Latest tag:    {latest_tag}")
    if args.dry_run:
        logger.info("Mode:          DRY RUN (no changes will be made)")
    logger.info(f"{'=' * 70}\n")
    
    # Step 1: Pull and retag image
    if not args.skip_pull:
        pull_and_retag_image(args.runtime_image, local_tag, dry_run=args.dry_run)
    else:
        logger.info(f"Skipping pull (using local image: {local_tag})")
    
    # Step 2: Generate Dockerfile.dev
    if args.keep_dockerfile:
        dockerfile_path = Path.cwd() / "Dockerfile.dev.generated"
        context_dir = Path.cwd()
    else:
        # Use temp directory
        temp_dir = Path(tempfile.mkdtemp(prefix="build_from_runtime_"))
        dockerfile_path = temp_dir / "Dockerfile.dev"
        context_dir = temp_dir
    
    generate_dockerfile_dev(local_tag, dockerfile_path, dry_run=args.dry_run)
    
    # Step 3: Build image
    try:
        build_dev_image(dockerfile_path, context_dir, local_tag, output_tag, dry_run=args.dry_run, target=args.target)
        if latest_tag is not None:
            if args.dry_run:
                logger.info(f"[DRY RUN] Would tag latest: {latest_tag}")
            else:
                run_command(
                    ["docker", "tag", output_tag, latest_tag],
                    f"Tagging latest as {latest_tag}",
                )
    finally:
        # Clean up temp directory if not keeping dockerfile
        if not args.keep_dockerfile and dockerfile_path.parent != Path.cwd():
            import shutil
            if not args.dry_run:
                shutil.rmtree(dockerfile_path.parent, ignore_errors=True)
                logger.debug(f"Cleaned up temp directory: {dockerfile_path.parent}")
    
    logger.info(f"\n{'=' * 70}")
    logger.info(f"✓ {'Would succeed!' if args.dry_run else 'Success!'}")
    logger.info(f"{'=' * 70}")
    logger.info(f"Image: {output_tag}")
    if latest_tag is not None:
        logger.info(f"Latest: {latest_tag}")
    logger.info(f"\nTo run:")
    # Prefer the repo's run.sh wrapper (sets sensible defaults: mounts, shm-size, ulimits, etc.).
    # This script lives in dynamo-utils.dev, so we can't assume which Dynamo repo checkout you're in.
    logger.info(f"  ./container/run.sh --image {output_tag} --mount-workspace -it")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
