#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Render a standalone local-dev Dockerfile from the Jinja2 template.

Takes a dev image name (e.g. dynamo:bbe82f182-vllm-dev) and produces a plain
Dockerfile in /tmp/ that can be built directly with `docker build`.

If the image is not available locally, pulls it from the GitLab registry
(gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo/dev/), retags it to the
short name, and removes the long registry name.

Usage:
    ./render_local_dev.py dynamo:bbe82f182-vllm-dev
    ./render_local_dev.py dynamo:bbe82f182-none-dev --output /tmp/my.Dockerfile
    ./render_local_dev.py dynamo:bbe82f182-vllm-dev --build
    ./render_local_dev.py dynamo:bbe82f182-vllm-dev --build --dry-run
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


TEMPLATE_REL_PATH = "container/templates/local_dev.Dockerfile"

REGISTRY = "gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo/dev"

# dynamo:<sha>-<framework>[-optional-segments]-dev
IMAGE_PATTERN = re.compile(r"^dynamo:[a-f0-9]+-(none|vllm|sglang|trtllm)(-[a-zA-Z0-9.]+)*-dev$")

# Pre-rendered fallback: used when local_dev.Dockerfile is not found on disk.
# {base_image} is replaced at render time.
# Keep in sync with container/templates/local_dev.Dockerfile.
EMBEDDED_TEMPLATE = """\
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# === BEGIN templates/local_dev.Dockerfile ===
# ======================================================================
# TARGET: local-dev (non-root development with UID/GID remapping)
# ======================================================================
FROM {base_image} AS local-dev

ENV USERNAME=dynamo
ARG USER_UID
ARG USER_GID

# Copy rustup home into a writable per-user location so sanity_check passes.
# (dev target already has rustup/cargo/maturin from concatenated wheel_builder/dynamo_base)
RUN cp -r /usr/local/rustup /home/dynamo/.rustup && \\
    chown -R dynamo:0 /home/dynamo/.rustup

# Put rustup state under the user's home (writable) while still using /usr/local/cargo/bin shims.
ENV RUSTUP_HOME=/home/${{USERNAME}}/.rustup
ENV CARGO_HOME=/home/${{USERNAME}}/.cargo
ENV PATH=/usr/local/cargo/bin:/usr/local/bin:${{CARGO_HOME}}/bin:${{PATH}}

# https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
# Configure user with sudo access for Dev Container workflows
#
# \U0001f6a8 PERFORMANCE / PERMISSIONS MEMO (DO NOT VIOLATE)
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
ENV WORKSPACE_DIR=${{WORKSPACE_DIR}}

# Development environment variables for the local-dev target
# Path configuration notes:
# - DYNAMO_HOME: Main project directory (workspace mount point)
# - CARGO_TARGET_DIR: Build artifacts in workspace/target for persistence
# - PATH: Includes cargo binaries for rust tool access
ENV HOME=/home/$USERNAME
ENV DYNAMO_HOME=${{WORKSPACE_DIR}}
ENV CARGO_TARGET_DIR=${{WORKSPACE_DIR}}/target
ENV PATH=${{CARGO_HOME}}/bin:$PATH

# Switch to dynamo user (dev stage has umask 002, so files should already be group-writable)
USER $USERNAME
WORKDIR $HOME

# Create user-level cargo/rustup state dirs as the target user (avoids root-owned caches).
RUN mkdir -p "${{CARGO_HOME}}" "${{RUSTUP_HOME}}"

# Ensure Python user site-packages exists and is writable (important for non-venv frameworks like SGLang).
RUN python3 -c 'import os, site; p = site.getusersitepackages(); os.makedirs(p, exist_ok=True); print(p)'

# https://code.visualstudio.com/remote/advancedcontainers/persist-bash-history
RUN SNIPPET="export PROMPT_COMMAND=\'history -a\' && export HISTFILE=$HOME/.commandhistory/.bash_history" \\
    && mkdir -p $HOME/.commandhistory \\
    && chmod g+w $HOME/.commandhistory \\
    && touch $HOME/.commandhistory/.bash_history \\
    && echo "$SNIPPET" >> "$HOME/.bashrc"

RUN mkdir -p /home/$USERNAME/.cache/ \\
    && mkdir -p /home/$USERNAME/.cache/pre-commit \\
    && chmod g+w /home/$USERNAME/.cache/ \\
    && chmod g+w /home/$USERNAME/.cache/pre-commit

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD []
"""


def find_template() -> Path | None:
    """Try to locate local_dev.Dockerfile on disk. Returns None if not found."""
    candidate = Path(__file__).resolve().parent.parent
    path = candidate / TEMPLATE_REL_PATH
    if path.exists():
        return path
    fallback = Path("/workspace") / TEMPLATE_REL_PATH
    if fallback.exists():
        return fallback
    return None


def render_from_template(template_text: str, base_image: str) -> str:
    """Strip Jinja2 directives from the on-disk template and substitute the FROM line."""
    lines = template_text.splitlines()
    out: list[str] = []

    # State machine to skip Jinja2 conditionals around the FROM line
    skip = False
    from_emitted = False

    for line in lines:
        stripped = line.strip()

        # Skip Jinja2 comment blocks {# ... #}
        if stripped.startswith("{#") or stripped.endswith("#}"):
            continue

        # Handle Jinja2 if/else/endif blocks (only used for the FROM line in this template)
        if re.match(r"\{%\s*(if|elif)\b", stripped):
            skip = True
            continue
        if re.match(r"\{%\s*else\s*%\}", stripped):
            skip = True
            continue
        if re.match(r"\{%\s*endif\s*%\}", stripped):
            skip = False
            if not from_emitted:
                out.append(f"FROM {base_image} AS local-dev")
                from_emitted = True
            continue

        if skip:
            continue

        out.append(line)

    return "\n".join(out) + "\n"


def render(base_image: str) -> tuple[str, str]:
    """Render the Dockerfile content. Returns (rendered_text, source_description)."""
    template_path = find_template()
    if template_path is not None:
        template_text = template_path.read_text()
        return render_from_template(template_text, base_image), str(template_path)
    # Fallback to embedded copy
    return EMBEDDED_TEMPLATE.format(base_image=base_image), "embedded template"


def image_exists_locally(image: str) -> bool:
    """Check if a Docker image exists in the local daemon."""
    rc = subprocess.call(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return rc == 0


def ensure_image_local(image: str) -> None:
    """Pull from GitLab registry if the image is not available locally, then retag."""
    if image_exists_locally(image):
        print(f"Image {image} found locally.")
        return

    # image is "dynamo:<tag>", remote is "REGISTRY/dynamo:<tag>"
    remote = f"{REGISTRY}/{image}"
    print(f"Image {image} not found locally. Pulling {remote} ...")
    rc = subprocess.call(["docker", "pull", remote])
    if rc != 0:
        print(f"ERROR: docker pull exited with code {rc}", file=sys.stderr)
        sys.exit(rc)

    # Retag to the short name and remove the long name
    print(f"Retagging {remote} -> {image}")
    rc = subprocess.call(["docker", "tag", remote, image])
    if rc != 0:
        print(f"ERROR: docker tag exited with code {rc}", file=sys.stderr)
        sys.exit(rc)
    subprocess.call(["docker", "rmi", remote])
    print(f"Image {image} ready.")


def validate_image(image: str) -> re.Match[str]:
    """Validate the image name matches dynamo:<sha>-<framework>[...]-dev."""
    m = IMAGE_PATTERN.match(image)
    if not m:
        print(
            f"ERROR: Image must match dynamo:<sha>-<framework>[...]-dev\n"
            f"       where framework is one of: none, vllm, sglang, trtllm\n"
            f"       Got: {image}",
            file=sys.stderr,
        )
        sys.exit(1)
    return m


def output_image_name(image: str) -> str:
    """Derive the local-dev output tag: replace trailing '-dev' with '-local-dev'."""
    # Already validated by validate_image, so this is safe.
    return re.sub(r"-dev$", "-local-dev", image)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a standalone local-dev Dockerfile from the Jinja2 template.",
    )
    parser.add_argument(
        "image",
        help="Dev base image, e.g. dynamo:<sha>-vllm-dev or dynamo:<sha>-vllm-cuda12.9-dev",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path (default: /tmp/Dockerfile.local-dev)",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Run docker build after rendering. Output image tag is the input with -dev replaced by -local-dev.",
    )
    parser.add_argument(
        "--dry-run",
        "--dryrun",
        dest="dry_run",
        action="store_true",
        help="With --build: print docker build and suggested run command but do not run build.",
    )
    args = parser.parse_args()

    validate_image(args.image)
    ensure_image_local(args.image)

    rendered, source = render(args.image)

    output_path = args.output or "/tmp/Dockerfile.local-dev"
    Path(output_path).write_text(rendered)
    print(f"Wrote {output_path}  (FROM {args.image}, source: {source})")

    def suggest_run_command(tag: str) -> None:
        base = f"container/run.sh --image {tag} -it --mount-workspace --hf-home ~/.cache/huggingface"
        with_sanity = f"{base} -- bash -c 'python /workspace/deploy/sanity_check.py && exec bash'"
        shell_only = f"{base} -- bash"
        print(f"\nYou can now use the image. Here are some commands you can try:")
        print(f"  (with sanity check)  {with_sanity}")
        print(f"  (shell only)         {shell_only}")

    if args.build:
        out_tag = output_image_name(args.image)
        uid = os.getuid()
        gid = os.getgid()
        cmd = [
            "docker", "build",
            "-f", output_path,
            "-t", out_tag,
            "--build-arg", f"USER_UID={uid}",
            "--build-arg", f"USER_GID={gid}",
            ".",
        ]
        if args.dry_run:
            print(f"\n[--dry-run] Would run: {' '.join(cmd)}")
            suggest_run_command(out_tag)
        else:
            print(f"\nBuilding: {' '.join(cmd)}")
            rc = subprocess.call(cmd)
            if rc != 0:
                print(f"ERROR: docker build exited with code {rc}", file=sys.stderr)
                sys.exit(rc)
            print(f"\nSuccessfully built image: {out_tag}")
            suggest_run_command(out_tag)


if __name__ == "__main__":
    main()
