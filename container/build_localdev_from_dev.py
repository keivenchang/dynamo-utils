#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Build a local-dev image from an existing dev image.

Given a dev image (local or remote), this script adds a UID/GID remapping layer
so the container user matches the host user, avoiding permission issues with
bind-mounted volumes.

Usage:
  ./build_localdev_from_dev.py dynamo:latest-vllm-dev
  ./build_localdev_from_dev.py gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo:abc123-vllm-amd64-dev
  ./build_localdev_from_dev.py --skip-pull dynamo:my-local-dev-image
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

_KNOWN_FRAMEWORKS = ("vllm", "trtllm", "sglang", "none")

# Inline Dockerfile template so this script works when piped via `curl | python3 -`.
# This string MUST be identical to the body of Dockerfile.localdev.template (same directory).
# Any deviation is a bug. After editing either copy, run:
#   diff <(sed -n '/^# =====/,$p' Dockerfile.localdev.template) \
#        <(python3 -c "from build_localdev_from_dev import _DOCKERFILE_TEMPLATE; print(_DOCKERFILE_TEMPLATE, end='')" | sed -n '/^# =====/,$p')
_DOCKERFILE_TEMPLATE = """\
# syntax=docker/dockerfile:1.10.0
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ======================================================================
# TARGET: local-dev (non-root development with UID/GID remapping)
# ======================================================================
# ***** DEVIATED ***** Upstream uses Jinja2: `{% if make_efa != true %}FROM dev AS local-dev{% else %}FROM aws AS local-dev{% endif %}`
# Replaced with a static placeholder that build_localdev_from_dev.py substitutes at build time.
FROM dev_base AS local-dev

ENV USERNAME=dynamo
ARG USER_UID
ARG USER_GID
ARG DEVICE

# rustup is already at /home/dynamo/.rustup from the dev stage (COPY --from=wheel_builder
# with --chown=dynamo:0 --chmod=775), so no re-copy needed here.
ENV RUSTUP_HOME=/home/${USERNAME}/.rustup
ENV CARGO_HOME=/home/${USERNAME}/.cargo
# ${VIRTUAL_ENV}/bin MUST lead: /usr/local/bin has a `python` -> system python3
# symlink (from the runtime stage) that otherwise shadows the venv's `python`, so
# bare `python` would escape the venv (no `dynamo`). Matches upstream local_dev.Dockerfile.
ENV PATH=${VIRTUAL_ENV}/bin:/usr/local/cargo/bin:/usr/local/bin:${CARGO_HOME}/bin:${PATH}

# https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
# Configure user with sudo access for Dev Container workflows
#
# 🚨 PERFORMANCE / PERMISSIONS MEMO (DO NOT VIOLATE)
# NEVER use `chown -R` or `chmod -R` in local-dev images.
# - It can take minutes on large mounts (and makes devcontainers feel "hung")
# - It is unnecessary: permissioning should be done via COPY --chmod/--chown and a few targeted, non-recursive ops.
# If you think you need recursion here, stop and redesign the permissions flow.
RUN mkdir -p /etc/sudoers.d \\
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \\
    && chmod 0440 /etc/sudoers.d/$USERNAME \\
    && mkdir -p /home/$USERNAME \\
    # Handle GID conflicts: if target GID exists and it's not our group, remove it
    && (getent group $USER_GID | grep -v "^$USERNAME:" && groupdel $(getent group $USER_GID | cut -d: -f1) || true) \\
    # Create group if it doesn't exist, otherwise modify existing group
    && (getent group $USERNAME > /dev/null 2>&1 && groupmod -g $USER_GID $USERNAME || groupadd -g $USER_GID $USERNAME) \\
    && usermod -u $USER_UID -g $USER_GID -G 0 $USERNAME \\
    && chown $USERNAME:$USER_GID /home/$USERNAME \\
    && chsh -s /bin/bash $USERNAME

# Set workspace directory variable
ENV WORKSPACE_DIR=${WORKSPACE_DIR}

# Development environment variables for the local-dev target
# Path configuration notes:
# - DYNAMO_HOME: Main project directory (workspace mount point)
# - CARGO_TARGET_DIR: Build artifacts in workspace/target for persistence
# - PATH: Includes cargo binaries for rust tool access
ENV HOME=/home/$USERNAME
ENV DYNAMO_HOME=${WORKSPACE_DIR}
ENV CARGO_TARGET_DIR=${WORKSPACE_DIR}/target
ENV PATH=${CARGO_HOME}/bin:$PATH

# Switch to dynamo user (dev stage has umask 002, so files should already be group-writable)
USER $USERNAME
WORKDIR $HOME

# Create user-level cargo/rustup state dirs as the target user (avoids root-owned caches).
RUN mkdir -p "${CARGO_HOME}" "${RUSTUP_HOME}"

# Ensure Python user site-packages exists and is writable (important for non-venv frameworks like SGLang).
RUN python3 -c 'import os, site; p = site.getusersitepackages(); os.makedirs(p, exist_ok=True); print(p)'

# https://code.visualstudio.com/remote/advancedcontainers/persist-bash-history
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=$HOME/.commandhistory/.bash_history" \\
    && mkdir -p $HOME/.commandhistory \\
    && chmod g+w $HOME/.commandhistory \\
    && touch $HOME/.commandhistory/.bash_history \\
    && echo "$SNIPPET" >> "$HOME/.bashrc"

# Framework runtime bases bake their own uv cache location into the image env
# (vllm/vllm-openai sets UV_CACHE_DIR=/opt/uv/cache), and its entries are root-owned
# 0755. That works while the image runs as root, but this stage remaps to a non-root
# user, so afterwards every uv call dies with:
#   error: Failed to initialize cache at `/opt/uv/cache`
#   Caused by: failed to open file `/opt/uv/cache/sdists-v9/.git`: Permission denied
# which breaks the two commands the dev docs tell you to run (`maturin develop --uv`
# and `uv pip install --no-deps -e /workspace`).
#
# Point the cache at the user's own cache dir — the same /home/dynamo/.cache/uv path
# the runtime, frontend and planner stages already use. `chmod -R g+w /opt/uv/cache`
# would also work but copies a ~780MB tree into a new layer, which the permissions
# memo above rules out.
ENV UV_CACHE_DIR=/home/${USERNAME}/.cache/uv

# `usermod -u` above re-chowns only files owned by the OLD uid, so anything under
# /home/$USERNAME that the base image left root-owned survives the remap — and `chmod`
# on it then fails with EPERM because we are no longer root. The trtllm dev image ships
# /home/dynamo/.cache/uv exactly this way. Group-write is what we actually need, so set
# it where we own the directory and assert the result either way.
RUN set -eux; \\
    for d in /home/$USERNAME/.cache /home/$USERNAME/.cache/pre-commit /home/$USERNAME/.cache/uv; do \\
        mkdir -p "$d"; \\
        if [ "$(stat -c %u "$d")" = "$(id -u)" ]; then chmod g+w "$d"; fi; \\
        test -w "$d"; \\
    done

# ***** DEVIATED ***** Upstream uses Jinja2:
#   {% if device == "xpu" or device == "cpu" %}SHELL/CMD
#   {% elif framework == "vllm" %}ENTRYPOINT []/CMD
#   {% else %}ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]/CMD{% endif %}
# Hardcoded to the CUDA non-vLLM path (no XPU/CPU support in this standalone template).
# build_localdev_from_dev.py rewrites the ENTRYPOINT line below to `ENTRYPOINT []`
# when the dev image is a vLLM one: vllm/vllm-openai does not ship
# /opt/nvidia/nvidia_entrypoint.sh, so leaving it makes plain `docker run` fail with
# "stat /opt/nvidia/nvidia_entrypoint.sh: no such file or directory".
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD []
"""

# ENTRYPOINT/CMD block as it appears in _DOCKERFILE_TEMPLATE, and the replacement used
# for frameworks whose base image does not ship the NVIDIA entrypoint script. The
# replacement is byte-identical to upstream local_dev.Dockerfile's `framework == "vllm"`
# branch. CMD must stay non-empty: with both ENTRYPOINT and CMD empty, a bare
# `docker run <image>` fails with "no command specified".
_NVIDIA_ENTRYPOINT_BLOCK = 'ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]\nCMD []'
_RESET_ENTRYPOINT_BLOCK = (
    'ENTRYPOINT []\nCMD ["bash", "-c", "source /home/$USERNAME/.bashrc && exec bash"]'
)

# Frameworks whose dev image has no /opt/nvidia/nvidia_entrypoint.sh. vLLM dev images
# are built FROM vllm/vllm-openai, which does not ship it; sglang and trtllm dev images
# inherit CUDA/TensorRT-LLM bases that do.
_FRAMEWORKS_WITHOUT_NVIDIA_ENTRYPOINT = ("vllm",)


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


def run_command(
    cmd: list[str],
    description: str,
    check: bool = True,
    stream_output: bool = False,
) -> subprocess.CompletedProcess:
    """Run a shell command and handle errors."""
    logger.info(f"{description}...")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        if stream_output:
            return subprocess.run(cmd, check=check)

        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.stdout:
            logger.debug(result.stdout)
        if result.stderr:
            logger.debug(result.stderr)
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {' '.join(cmd)}")
        if e.stdout:
            logger.error(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            logger.error(f"STDERR:\n{e.stderr}")
        raise


def is_remote_image(image_ref: str) -> bool:
    """Return True if image_ref looks like a remote registry reference (has a slash in the name part)."""
    # Local images: "dynamo:tag", "myimage:latest"
    # Remote images: "registry.com/repo/image:tag", "ghcr.io/org/image:tag"
    name_part = image_ref.rsplit(":", 1)[0] if ":" in image_ref else image_ref
    return "/" in name_part


def pull_image(image_ref: str, dry_run: bool = False) -> None:
    """Pull image from registry."""
    if dry_run:
        logger.info(f"[DRY RUN] Would pull: {image_ref}")
        return

    run_command(
        ["docker", "pull", image_ref],
        f"Pulling {image_ref}",
        stream_output=True,
    )


def generate_dockerfile(
    dev_image: str,
    output_path: Path,
    framework: str | None = None,
    dry_run: bool = False,
) -> None:
    """Generate a Dockerfile that builds local-dev from the given dev image.

    Uses the inline _DOCKERFILE_TEMPLATE so this works even when piped via stdin.

    `framework` selects the ENTRYPOINT the way upstream local_dev.Dockerfile's Jinja2
    conditional does: vLLM dev images have no /opt/nvidia/nvidia_entrypoint.sh, so
    keeping the inherited ENTRYPOINT there makes the resulting image unrunnable.
    """
    reset_entrypoint = framework in _FRAMEWORKS_WITHOUT_NVIDIA_ENTRYPOINT

    if dry_run:
        logger.info(f"[DRY RUN] Would generate Dockerfile at: {output_path}")
        logger.info(f"[DRY RUN] Dev image: {dev_image}")
        logger.info(f"[DRY RUN] ENTRYPOINT: {'[] (reset)' if reset_entrypoint else 'nvidia_entrypoint.sh'}")
        return

    modified_content = _DOCKERFILE_TEMPLATE.replace(
        "FROM dev_base AS local-dev",
        f"FROM {dev_image} AS local-dev",
    )

    if reset_entrypoint:
        # str.replace is a no-op if the block ever drifts, which would silently
        # reintroduce the unrunnable image — fail loudly instead.
        if _NVIDIA_ENTRYPOINT_BLOCK not in modified_content:
            raise RuntimeError(
                f"expected {_NVIDIA_ENTRYPOINT_BLOCK!r} in the local-dev template so it "
                f"could be reset for framework={framework}, but it is not there"
            )
        modified_content = modified_content.replace(
            _NVIDIA_ENTRYPOINT_BLOCK, _RESET_ENTRYPOINT_BLOCK
        )

    output_path.write_text(modified_content)
    logger.info(f"Generated Dockerfile at: {output_path}")
    logger.info(f"  Dev image: {dev_image}")
    if reset_entrypoint:
        logger.info(f"  ENTRYPOINT: reset to [] ({framework} images have no /opt/nvidia/nvidia_entrypoint.sh)")


def build_image(
    dockerfile_path: Path,
    context_dir: Path,
    output_tag: str,
    dry_run: bool = False,
) -> None:
    """Build the local-dev image using docker build."""
    uid = os.getuid()
    gid = os.getgid()

    if dry_run:
        logger.info(f"[DRY RUN] Would build image: {output_tag}")
        logger.info(f"[DRY RUN] Dockerfile: {dockerfile_path}")
        logger.info(f"[DRY RUN] Context: {context_dir}")
        logger.info(f"[DRY RUN] USER_UID={uid}, USER_GID={gid}")
        return

    build_cmd = [
        "docker", "build",
        "-f", str(dockerfile_path),
        "-t", output_tag,
        "--target", "local-dev",
        "--build-arg", f"USER_UID={uid}",
        "--build-arg", f"USER_GID={gid}",
        str(context_dir),
    ]

    run_command(
        build_cmd,
        f"Building local-dev image: {output_tag}",
        stream_output=True,
    )

    logger.info(f"Image built: {output_tag}")


def main():
    parser = argparse.ArgumentParser(
        description="Build a local-dev image from a dev image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From a local dev image
  %(prog)s dynamo:latest-vllm-dev

  # From a remote dev image (will pull first)
  %(prog)s gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo:abc123-vllm-amd64-dev

  # Custom output tag
  %(prog)s --output-tag my-localdev:latest dynamo:latest-vllm-dev

  # Dry run
  %(prog)s --dry-run dynamo:latest-vllm-dev
        """,
    )
    parser.add_argument(
        "dev_image",
        help="Dev image reference (local tag or remote URL)",
    )
    parser.add_argument(
        "--output-tag",
        help="Output tag for built image (default: <dev-image>-local-dev)",
    )
    parser.add_argument(
        "--no-tag-latest",
        action="store_true",
        help="Do not add a latest-{framework}-local-dev tag",
    )
    parser.add_argument(
        "--keep-dockerfile",
        action="store_true",
        help="Keep the generated Dockerfile (saved to ./Dockerfile.localdev.generated)",
    )
    parser.add_argument(
        "--skip-pull",
        action="store_true",
        help="Skip pulling image (assume it's already available locally)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--dry-run", "--dryrun",
        dest="dry_run",
        action="store_true",
        help="Show what would be done without executing",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    dev_image = args.dev_image

    # Derive default output tag
    if args.output_tag:
        output_tag = args.output_tag
    elif ":" in dev_image:
        name, tag = dev_image.rsplit(":", 1)
        # Keep generated tags in the canonical order:
        # <sha>-<framework>-local-dev-<cuda>, not <sha>-<framework>-dev-<cuda>-local-dev.
        short_name = name.rsplit("/", 1)[-1] if "/" in name else name
        if "-dev-" in tag:
            output_tag = f"{short_name}:{tag.replace('-dev-', '-local-dev-', 1)}"
        elif tag.endswith("-dev"):
            output_tag = f"{short_name}:{tag[:-4]}-local-dev"
        else:
            output_tag = f"{short_name}:{tag}-local-dev"
    else:
        output_tag = f"{dev_image}:local-dev"

    # Infer framework for latest tag
    tag_part = dev_image.rsplit(":", 1)[-1] if ":" in dev_image else ""
    framework = infer_framework_from_tag(tag_part)
    latest_tag = None
    if not args.no_tag_latest and framework is not None:
        repo_name = dev_image.rsplit(":", 1)[0].rsplit("/", 1)[-1] if ":" in dev_image else dev_image
        latest_tag = f"{repo_name}:latest-{framework}-local-dev"

    logger.info("=" * 70)
    logger.info("Building local-dev image from dev image")
    logger.info("=" * 70)
    logger.info(f"Dev image:     {dev_image}")
    logger.info(f"Output tag:    {output_tag}")
    logger.info(f"UID/GID:       {os.getuid()}:{os.getgid()}")
    if latest_tag is not None:
        logger.info(f"Latest tag:    {latest_tag}")
    if args.dry_run:
        logger.info("Mode:          DRY RUN (no changes will be made)")
    logger.info(f"{'=' * 70}\n")

    # Step 1: Pull if remote and not skipped
    if not args.skip_pull and is_remote_image(dev_image):
        pull_image(dev_image, dry_run=args.dry_run)
    elif is_remote_image(dev_image):
        logger.info(f"Skipping pull (using local image: {dev_image})")

    # Step 2: Generate Dockerfile
    if args.keep_dockerfile:
        dockerfile_path = Path.cwd() / "Dockerfile.localdev.generated"
        context_dir = Path.cwd()
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="build_localdev_"))
        dockerfile_path = temp_dir / "Dockerfile.localdev"
        context_dir = temp_dir

    generate_dockerfile(dev_image, dockerfile_path, framework=framework, dry_run=args.dry_run)

    # Step 3: Build image
    try:
        build_image(dockerfile_path, context_dir, output_tag, dry_run=args.dry_run)
        if latest_tag is not None:
            if args.dry_run:
                logger.info(f"[DRY RUN] Would tag latest: {latest_tag}")
            else:
                run_command(
                    ["docker", "tag", output_tag, latest_tag],
                    f"Tagging latest as {latest_tag}",
                )
    finally:
        if not args.keep_dockerfile and dockerfile_path.parent != Path.cwd():
            if not args.dry_run:
                shutil.rmtree(dockerfile_path.parent, ignore_errors=True)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"{'Would succeed!' if args.dry_run else 'Success!'}")
    logger.info(f"{'=' * 70}")
    logger.info(f"Image: {output_tag}")
    if latest_tag is not None:
        logger.info(f"Latest: {latest_tag}")
    base = f"./container/run.sh --image {output_tag} --mount-workspace --hf-home ~/.cache/huggingface"
    logger.info(f"\nYou can now use the image. Here are some commands you can try:")
    logger.info(f"  (shell only)         {base} -it")
    logger.info(f"  (with sanity check)  {base} -- bash -c 'python3 /workspace/deploy/sanity_check.py; exec bash'")
    logger.info(f"  (pytest, isolated)   {base} -- python3 -m pytest -xvs --durations=10 tests/")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
