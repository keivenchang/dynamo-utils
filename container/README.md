# Container utilities (`dynamo-utils/container/`)

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
- **`cleanup_old_images.sh`**: prune old images/build cache.
- **`dangerously_wipe_local_docker.sh`**: **DANGEROUS** kill all running containers and delete **all local Docker images** (requires `--wipe-out-all-docker-images`).
- **`restart_gpu_containers.sh`**: watchdog-style monitor/restart for GPU containers.
- **`build_images_report.html.j2`**: HTML template used by `build_images.py`.

---

## `build_images.py`

**Overview**: Automated Docker build and test pipeline for multiple inference frameworks.

**Usage summary**

| Goal | Command |
|------|---------|
| Dry run | `--repo-path <path> --dry-run -v` |
| Build at image-SHA origin (HEAD) | `--repo-path <path> --parallel --skip --run-ignore-lock -v` |
| Build specific commit | `--repo-path <path> --repo-sha <commit> --parallel --skip --run-ignore-lock -v` |
| Reuse if images exist | `--repo-path <path> --reuse-if-image-exists --parallel --skip -v` |
| Sanity only | `--repo-path <path> --sanity-check-only --framework sglang --force-run` |

Common flags: `--repo-path` (required), `--repo-sha` / `--sha`, `--parallel`, `--skip`, `--run-ignore-lock`, `--no-upload`, `--no-compress`, `-v`, `--dry-run`.

### Usage (examples)

```bash
# Quick test (single framework)
python3 container/build_images.py --repo-path ~/dynamo/dynamo_ci --sanity-check-only --framework sglang --force-run

# Parallel build with skip
python3 container/build_images.py --repo-path ~/dynamo/dynamo_ci --skip-action-if-already-passed --parallel --force-run

# Full build
python3 container/build_images.py --repo-path ~/dynamo/dynamo_ci --parallel --force-run

# Without upload and compress (both are on by default)
python3 container/build_images.py --repo-path ~/dynamo/dynamo_ci --parallel --skip --run-ignore-lock --no-upload --no-compress
```

### Commit SHA and image SHA

- **Image SHA** is a content hash of the `container/` directory. It identifies when the Dockerfile/context changed. Image tags include it (e.g. `dynamo:6d3e0137c.IMAGE.c62194-none-cuda13.0-runtime`).
- **Commit SHA** is the git commit used for the build (logs, report paths, and the first part of the image tag).

**Default (no `--repo-sha`) – build at image-SHA origin**

If you omit `--repo-sha` / `--sha`, the script:

1. Uses the **latest commit** (current HEAD).
2. Computes the **image SHA** for that commit (hash of `container/`).
3. **Traverses back** in history to find the commit that **first introduced** that image SHA (`find_docker_image_sha_origin`).
4. If that commit differs from HEAD, **checks out** that commit.
5. **Builds** that commit (so the image tag is that commit SHA + image SHA).

Use this when you want to build the canonical commit for the current container state (e.g. from `dynamo_ci` HEAD) without specifying a commit yourself.

**Explicit commit (`--repo-sha` / `--sha`)**

If you pass `--repo-sha <commit>` (e.g. `--repo-sha 6d3e0137c`), the script checks out that exact commit and builds it. No traverse-back step: you get that commit’s tree and its image SHA in the tag.

```bash
# Build at image-SHA origin (default: use HEAD, resolve to introducing commit, then build)
python3 container/build_images.py --repo-path ~/dynamo/dynamo_ci --parallel --skip --run-ignore-lock -v

# Build a specific commit as-is (no traverse-back)
python3 container/build_images.py --repo-path ~/dynamo/dynamo_ci --repo-sha 6d3e0137c --parallel --skip --run-ignore-lock -v
```

### Reuse existing images (`--reuse-if-image-exists`)

When you pass **`--reuse-if-image-exists`**, the script:

1. Goes to the **latest commit** (HEAD; `--repo-sha` is ignored).
2. Computes the **image SHA** and resolves to the commit that introduced it (same as default).
3. Checks if **all** required images for that image SHA exist locally (runtime, dev, local-dev for each framework).
4. **If they exist**: only runs **compilation and sanity checks** with `/workspace` mounted, reusing those images (no build steps).
5. **If any image is missing**: runs the **full build** (runtime, dev, local-dev) as usual.

Use this for a fast compile/sanity pass when container/ has not changed and you already have the images (e.g. after a previous full build or when syncing from another machine).

```bash
# Reuse if images exist; otherwise full build (uses HEAD, ignores --repo-sha)
python3 container/build_images.py --repo-path ~/dynamo/dynamo_ci --reuse-if-image-exists --parallel --skip -v
```

### Dev upload and compress

- **Dev upload** pushes the dev image to `gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo/dev/`. It is **on by default**.
- **Dev compress** squashes the dev image layers to reduce size. It is **on by default**.
- To disable upload, pass **`--no-upload`**. To disable compress, pass **`--no-compress`**.
- **Cron**: Upload and compress run automatically. To disable either, add the corresponding `--no-*` flag.

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
  - Log file paths: `~/dynamo/dynamo_ci/logs/YYYY-MM-DD/`
  - Report: `YYYY-MM-DD.{sha_short}.report.html`
- **Email notifications**:
  - SMTP server: `smtp.nvidia.com:25`
  - Subject format: `{SUCC|FAIL}: DynamoDockerBuilder - {sha_short} [{failed_tasks}]`
  - HTML email body with clickable links
  - Not sent in dry-run mode


