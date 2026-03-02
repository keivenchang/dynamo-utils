# Container utilities (`dynamo-utils.PRODUCTION/container/`)

Scripts in this directory are focused on **Docker image build/test**, **retagging**, and **cleanup/monitoring** for the Dynamo repos under your workspace (commonly `~/dynamo`).

## Directory structure

```
container/
├── build_all_targets_and_verify.sh
├── build_images.py
├── build_images_report.html.j2
├── cleanup_old_images.sh
├── restart_gpu_containers.sh
└── retag_images.py
```

## Scripts

- **`build_images.py`**: automated Docker build/test pipeline (multi-framework).
- **`build_all_targets_and_verify.sh`**: build runtime/dev/local-dev targets and run verification (run from a Dynamo repo root).
- **`retag_images.py`**: retag images (useful for promoting images across tags).
- **`cleanup_old_images.sh`**: prune old images/build cache. Use `--keep-dev-and-local-dev-only` to remove only runtime images (keeps BOTH dev and local-dev; used by `cleanup_log_and_docker.sh`).
- **`wipe_all_images_and_rebuild.py`**: **DANGEROUS** kill all containers, delete all Docker images, and optionally rebuild from scratch. Supports `--prune-builder-cache`, `--no-prune-volumes`, `--skip-cleanup`.
- **`restart_gpu_containers.sh`**: watchdog-style monitor/restart for GPU containers.
- **`build_images_report.html.j2`**: HTML template used by `build_images.py`.

---

## `build_images.py`

**Overview**: Automated Docker build and test pipeline for multiple inference frameworks.

**Usage summary**

| Goal | Command |
|------|---------|
| Dry run | `--repo-path <path> --dry-run -v` |
| Build at image-SHA origin (HEAD) | `--repo-path <path> --parallel --skip --run-no-matter-what -v` |
| Build specific commit | `--repo-path <path> --commit-sha <commit> --parallel --skip --run-no-matter-what -v` |
| Reuse dev if image exists | `--repo-path <path> --reuse-dev-if-image-exists --parallel --skip -v` |
| Sanity only | `--repo-path <path> --sanity-check-only --framework sglang --run-no-matter-what` |

Common flags: `--repo-path` (required), `--commit-sha`, `--parallel`, `--skip`, `--run-no-matter-what`, `--no-upload`, `--no-compress`, `-v`, `--dry-run`.

### Usage (examples)

```bash
# Quick test (single framework)
python3 container/build_images.py --repo-path ~/dynamo/dynamo_ci --sanity-check-only --framework sglang --run-no-matter-what

# Parallel build with skip
python3 container/build_images.py --repo-path ~/dynamo/dynamo_ci --skip-action-if-already-passed --parallel --run-no-matter-what

# Full build
python3 container/build_images.py --repo-path ~/dynamo/dynamo_ci --parallel --run-no-matter-what

# Without upload and compress (both are on by default)
python3 container/build_images.py --repo-path ~/dynamo/dynamo_ci --parallel --skip --run-no-matter-what --no-upload --no-compress
```

### Commit SHA and image SHA

- **Image SHA** is a 6-char uppercase SHA256 of `git ls-tree -r <commit> -- container/`. Any change in `container/` produces a new hash. Image tags include it (e.g. `dynamo:C62194.6d3e0137c-none-cuda13.0-runtime`).
- **Commit SHA** is the git commit used for the build (logs, report paths, and the first part of the image tag).

**Default (no `--commit-sha`) – build at image-SHA origin**

If you omit `--commit-sha` / `--sha`, the script:

1. Uses the **latest commit** (current HEAD).
2. Computes the **image SHA** for that commit (hash of `container/`).
3. **Traverses back** in history to find the commit that **first introduced** that image SHA (`find_docker_image_sha_origin`).
4. If that commit differs from HEAD, **checks out** that commit.
5. **Builds** that commit (so the image tag is that commit SHA + image SHA).

Use this when you want to build the canonical commit for the current container state (e.g. from `dynamo_ci` HEAD) without specifying a commit yourself.

**Explicit commit (`--commit-sha` / `--sha`)**

If you pass `--commit-sha <commit>` (e.g. `--commit-sha 6d3e0137c`), the script checks out that exact commit and builds it. No traverse-back step: you get that commit’s tree and its image SHA in the tag.

```bash
# Build at image-SHA origin (default: use HEAD, resolve to introducing commit, then build)
python3 container/build_images.py --repo-path ~/dynamo/dynamo_ci --parallel --skip --run-no-matter-what -v

# Build a specific commit as-is (no traverse-back)
python3 container/build_images.py --repo-path ~/dynamo/dynamo_ci --commit-sha 6d3e0137c --parallel --skip --run-no-matter-what -v
```

### Reuse dev images (`--reuse-dev-if-image-exists`)

When you pass **`--reuse-dev-if-image-exists`**, the script:

1. Computes the **image SHA** for the target commit (HEAD or `--commit-sha`).
2. Checks if the **local-dev** image for that image SHA exists locally (per framework).
3. **If it exists**: only runs **local-dev compilation and sanity check** (no build steps).
4. **If the image is missing**: runs the **full build** as usual.

Use this for a fast compile/sanity pass when container/ has not changed and you already have the local-dev image (e.g. after a previous full build).

```bash
# Reuse local-dev if image exists; otherwise full build
python3 container/build_images.py --repo-path ~/dynamo/dynamo_ci --commit-sha fd839b8d5 --reuse-dev-if-image-exists --parallel --skip -v
```

### Dev upload and compress

- **Dev upload** pushes the dev image to `gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo/dev/`. It is **on by default**.
- **Dev compress** squashes the dev image layers to reduce size. It is **on by default**.
- To disable upload, pass **`--no-upload`**. To disable compress, pass **`--no-compress`**.
- **Cron**: Upload and compress run automatically. To disable either, add the corresponding `--no-*` flag.

### Skip behavior (incremental builds)

Two tracking files in the repo root control what gets skipped on re-runs:

- **`.last_docker_image_sha`**: Stores the SHA256 of `container/` contents from the last build. If unchanged, the entire build is skipped (no Docker image rebuild needed).
- **`.last_compilation_sha`**: Stores the 9-char commit SHA from the last successful compilation. If unchanged, all compilation and chown tasks are skipped (artifacts are still valid).

This means a re-run of the same commit does nothing, and a re-run after a code-only change (no `container/` changes) skips Docker builds but re-compiles. Pass `--run-no-matter-what` to bypass both checks.

### Features

- **Frameworks supported**: VLLM, SGLANG, TRTLLM
- **Target environments**: runtime, dev, local-dev
- **Build stages**: build, chown, compilation, sanity checks
- **Process management**:
  - Lock files prevent concurrent runs (`.build_images.lock`)
  - PID tracking with stale lock cleanup
  - Dry-run mode for testing
- **HTML report generation**:
  - Automatic report generation with clickable links
  - Two versions: file paths vs absolute URLs (for email)
  - Log file paths: `~/dynamo/dynamo_ci/logs/YYYY-MM-DD/{image_sha}.{commit_sha}/`
  - Report: `{image_sha}.{commit_sha}/report.html`
- **Email notifications**:
  - SMTP server: `smtp.nvidia.com:25`
  - Subject format: `{SUCC|FAIL}: DynamoDockerBuilder - {sha_short} [{failed_tasks}]`
  - HTML email body with clickable links
  - Not sent in dry-run mode

---

## Renaming Images in GitLab Container Registry

**Web UI Location**: https://gitlab-master.nvidia.com/dl/ai-dynamo/dynamo/container_registry/165910

Or navigate via project:
1. Go to https://gitlab-master.nvidia.com/dl/ai-dynamo/dynamo
2. Click **Packages & Registries** → **Container Registry** in left sidebar
3. Click on **dev/dynamo** repository

### Bulk Rename with crane

Use [crane](https://github.com/google/go-containerregistry/blob/main/cmd/crane/doc/crane.md) to copy tags directly in the registry without pulling images:

```bash
# Install crane
cd /tmp
curl -sL "https://github.com/google/go-containerregistry/releases/latest/download/go-containerregistry_Linux_x86_64.tar.gz" | tar -xz crane
chmod +x crane

# Copy old tag to new tag (no docker pull needed!)
./crane copy \
  gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo/dev/dynamo:OLD_TAG \
  gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo/dev/dynamo:NEW_TAG

# List all tags
./crane ls gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo/dev/dynamo

# Delete old tag using GitLab API (crane delete doesn't work - see note below)
PROJECT_ID="169905"  # dl/ai-dynamo/dynamo
REGISTRY_ID="165910"  # dev/dynamo
TOKEN="your-gitlab-token"  # Must have 'api' permission
TAG="OLD_TAG"

curl --request DELETE \
  --header "PRIVATE-TOKEN: $TOKEN" \
  "https://gitlab-master.nvidia.com/api/v4/projects/${PROJECT_ID}/registry/repositories/${REGISTRY_ID}/tags/${TAG}"
```

**IMPORTANT: Why not use `crane delete`?**
- `crane delete` requires Docker Registry API delete permission
- GitLab tokens with `write_registry` allow **push** but not **delete** via Docker Registry API
- Must use **GitLab REST API** instead (requires `api` permission on token)
- Example: Token with `api` + `write_registry` works for GitLab API delete, but not crane delete

**Token location**:
- Stored in `~/.docker/config.json` (set via `docker login gitlab-master.nvidia.com:5005`)
- Extract token from Docker config:
  ```bash
  python3 -c "
  import json, base64
  from pathlib import Path
  config = json.load(open(Path.home() / '.docker' / 'config.json'))
  auth = config['auths']['gitlab-master.nvidia.com:5005']['auth']
  username, token = base64.b64decode(auth).decode().split(':', 1)
  print(f'Username: {username}')
  print(f'Token: {token}')
  "
  ```
```

**Image naming convention**:
- **Current format**: `AABB11.commit_sha-variant-cuda-type`
- **Example**: `1FA782.9a15730a7-none-cuda12.9-dev`
- **`AABB11`**: uppercase 6-char SHA256 of `git ls-tree -r <commit> -- container/` (identifies container content)
- **`commit_sha`**: lowercase 9-char git commit SHA (identifies code version)

**Calculate image SHA for a commit**:
```bash
cd ~/dynamo
python3 -c "
import sys; sys.path.insert(0, 'dynamo-utils.dev')
from common import DynamoRepositoryUtils
repo = DynamoRepositoryUtils('/home/keivenc/dynamo/dynamo4')
print(repo.generate_docker_image_sha_for_commit('COMMIT_SHA', full_hash=False))
"
```


