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
- **`restart_gpu_containers.sh`**: watchdog-style monitor/restart for GPU containers.
- **`build_images_report.html.j2`**: HTML template used by `build_images.py`.

---

## `build_images.py`

**Overview**: Automated Docker build and test pipeline for multiple inference frameworks.

### Usage

```bash
# Quick test (single framework)
python3 container/build_images.py --repo-path ~/dynamo/dynamo_ci --sanity-check-only --framework sglang --force-run

# Parallel build with skip
python3 container/build_images.py --repo-path ~/dynamo/dynamo_ci --skip-action-if-already-passed --parallel --force-run

# Full build
python3 container/build_images.py --repo-path ~/dynamo/dynamo_ci --parallel --force-run
```

### Features

- **Frameworks supported**: VLLM, SGLANG, TRTLLM
- **Target environments**: base, dev, local-dev
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


