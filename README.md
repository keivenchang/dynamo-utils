# Dynamo Utils

A collection of utility scripts and configuration files for developing and deploying the NVIDIA Dynamo distributed inference framework.

## ⚠️ DISCLAIMER

**IMPORTANT**: This is an experimental utilities repository and is NOT officially tied to or supported by the ai-dynamo project. These tools are provided as-is without any warranty or official support. Use at your own risk.

This collection of utilities is maintained independently for development convenience and is not part of the official Dynamo project.

## Overview

This repository contains essential development tools, build scripts, and configuration management utilities for working with the Dynamo project. These tools streamline the development workflow, manage container environments, and facilitate testing of the Dynamo inference system.

## Prerequisites

- Docker with GPU support
- Python >= 3.10
- Rust toolchain (for building Dynamo components)
- Git
- jq (for JSON processing in scripts)

## Directory Structure

```
dynamo-utils/
├── compile.sh                    # Build and install Dynamo Python packages
├── curl.sh                       # Test models via chat completions API
├── inference.sh                  # Launch Dynamo inference services
├── devcontainer_sync.sh          # Sync dev configs across projects
├── devcontainer.json             # VS Code Dev Container configuration
├── common.py                     # Shared utilities module
├── show_dynamo_branches.py       # Branch status checker
├── show_commit_history.py        # Commit history with composite SHAs
├── git_stats.py                  # Git repository statistics analyzer
├── update_html_pages.sh          # HTML page update cron script
├── soak_fe.py                    # Frontend soak testing script
├── gpu_reset.sh                  # GPU reset utility
└── docker/                       # Docker-related scripts
    ├── build_images.py               # Automated Docker build/test pipeline
    ├── build_images_report.html.j2   # HTML report template
    ├── retag_images.py               # Docker image retagging utility
    ├── restart_gpu_containers.sh     # GPU error monitoring/restart
    └── cleanup_old_images.sh         # Cleanup old Docker images
```

## Key Scripts

### Build & Development

#### `compile.sh`
Builds and installs Python packages for the Dynamo distributed inference framework.

```bash
# Development mode (fast build, editable install)
./compile.sh --dev

# Release mode (optimized build)
./compile.sh --release

# Clean Python packages
./compile.sh --python-clean

# Clean Rust build artifacts
./compile.sh --cargo-clean
```

**Packages built:**
- `ai-dynamo-runtime`: Core Rust extensions + Python bindings
- `ai-dynamo`: Complete framework with all components

### Testing & Inference

#### `curl.sh`
Convenient script to test models via the chat completions API with retry logic and response validation.

```bash
# Basic test
./curl.sh --port 8000 --prompt "Hello!"

# Streaming with retry
./curl.sh --port 8000 --stream --retry --prompt "Tell me a story"

# Loop testing with metrics
./curl.sh --port 8000 --loop --metrics --random
```

**Options:**
- `--port`: API server port (default: 8000)
- `--prompt`: User prompt
- `--stream`: Enable streaming responses
- `--retry`: Retry until success
- `--loop`: Run in infinite loop
- `--metrics`: Show performance metrics

#### `inference.sh`
Launches Dynamo inference services (frontend and backend).

```bash
# Run with default framework
./inference.sh

# Run with specific backend
./inference.sh --framework vllm

# Dry run to see commands
./inference.sh --dry-run
```

**Environment variables:**
- `DYN_FRONTEND_PORT`: Frontend port (default: 8000)
- `DYN_BACKEND_PORT`: Backend port (default: 8081)

### Configuration Management

#### `devcontainer_sync.sh`
Automatically syncs development configuration files across multiple Dynamo project directories.

```bash
# Sync configurations
./devcontainer_sync.sh

# Dry run to preview changes
./devcontainer_sync.sh --dryrun

# Force sync regardless of changes
./devcontainer_sync.sh --force

# Silent mode for cron jobs
./devcontainer_sync.sh --silent
```

---

## Python Utilities

### docker/build_images.py

Automated Docker build and test pipeline system that builds and tests multiple inference frameworks (VLLM, SGLANG, TRTLLM) across different target environments (base, dev, local-dev), generates HTML reports, and sends email notifications.

#### Usage Examples

**Quick Test (Single Framework)**:
```bash
python3 docker/build_images.py --repo-path ~/nvidia/dynamo_ci --sanity-check-only --framework sglang --force-run
```

**Parallel Build with Skip**:
```bash
python3 docker/build_images.py --repo-path ~/nvidia/dynamo_ci --skip-action-if-already-passed --parallel --force-run
```

**Full Build**:
```bash
python3 docker/build_images.py --repo-path ~/nvidia/dynamo_ci --parallel --force-run
```

**Note**: `--repo-path` is required and specifies the path to the Dynamo repository.

#### HTML Report Generation
- Generate two versions when email is requested:
  - **File version**: relative paths (just filenames)
  - **Email version**: absolute URLs with hostname
- Use helper function `get_log_url()` to abstract URL generation
- Pass `use_absolute_urls` flag to control URL format
- Default hostname: `keivenc-linux`
- Default HTML path: `/nvidia/dynamo_ci/logs`

#### Log File Paths
- Logs stored in: `~/nvidia/dynamo_ci/logs/YYYY-MM-DD/`
- Log filename format: `YYYY-MM-DD.{sha_short}.{framework}-{target}-{type}.log`
- HTML report: `YYYY-MM-DD.{sha_short}.report.html`
- Always use date subdirectories
- Use `log_dir.parent` to get root logs directory

#### Version Extraction
- Automatically extracts version from `build.sh` output (not hardcoded)
- Version format depends on git repository state:
  - If on a tagged commit: `v{tag}` (e.g., `v0.6.1`)
  - Otherwise: `v{latest_tag}.dev.{commit_id}` (e.g., `v0.6.1.dev.d9b674b86`)
- Version is extracted once per framework before creating task graphs
- All image tags use the dynamically extracted version (base, runtime, dev, local-dev)

#### Process Management
- Use lock files to prevent concurrent runs
- Lock file: `.dynamo_builder.lock` (in repository root)
- Store PID in lock file
- Check if process is still running before acquiring lock
- Clean up stale locks automatically

#### Email Notifications
- Uses SMTP server: `smtp.nvidia.com:25`
- Email subject format: `{SUCC|FAIL}: DynamoDockerBuilder - {sha_short} [{failed_tasks}]`
- HTML email body with clickable links (absolute URLs)
- Includes failed task names in subject if any failures occurred
- **Note**: Email notifications are NOT sent in dry-run mode (only shows a note that email would be sent)
- Separate error handling for email failures (errors are logged separately from HTML report generation)
- Email sending continues even if HTML report generation fails

#### Commit History Feature

**Overview**: The `show_commit_history.py` script displays recent commits with their composite SHAs. Supports both terminal and HTML output modes with integrated caching for performance.

**Caching System**:
- Cache file: `.commit_history_cache.json` (in repository root)
- Format: JSON mapping of commit SHA (full) → composite SHA
- Purpose: Avoid expensive git checkout + composite SHA recalculation
- Performance: Cached lookups are nearly instant vs ~1-2 seconds per commit calculation

**Usage Examples**:

Terminal output with caching:
```bash
python3 show_commit_history.py --repo-path ~/nvidia/dynamo_ci --max-commits 50
```

HTML output with caching:
```bash
python3 show_commit_history.py --repo-path ~/nvidia/dynamo_ci --html --max-commits 50 --output ~/nvidia/dynamo_ci/logs/commit-history.html
```

Verbose mode (shows cache hits/misses):
```bash
python3 show_commit_history.py --repo-path ~/nvidia/dynamo_ci --html --max-commits 50 --output ~/nvidia/dynamo_ci/logs/commit-history.html --verbose
```

**HTML Output Features**:
- Clickable commit SHAs linking to GitHub commit page
- Clickable PR numbers extracted from commit messages
- Docker image detection showing expandable list of images containing each commit SHA
- Expandable sections using HTML `<details>` tag
- GitHub-style CSS

**Implementation Details**:
- Commit SHA format: Full SHA for GitHub links, 9-char short SHA for display
- PR extraction regex: `r'\(#(\d+)\)'`
- Docker image query: Uses `docker images --format "{{.Repository}}:{{.Tag}}"` once for all commits
- Cache management: Loaded at start, saved after updates

#### Image Size Population

**Overview**: Docker image sizes are automatically populated in HTML reports for all BUILD tasks, regardless of whether images were just built or were skipped (using `--skip-build-if-image-exists`).

**Implementation**:
- Location: `docker/build_images.py:1810-1851` and line 2003-2004
- Method: `_populate_image_sizes(pipeline: BuildPipeline)`
- Called after task execution completes (but before HTML report generation)
- Only runs when NOT in dry-run mode
- Queries Docker for all BUILD tasks that have an `image_tag`
- Populates `task.image_size` with format "XX.X GB"

**Why This Approach**:
- Centralized: All image size detection happens in one place after task execution
- Works for all scenarios: Handles both newly built images and existing images
- Non-blocking: Uses try-except to gracefully handle missing images
- Efficient: Only queries Docker once per image after all tasks complete

---

### update_html_pages.sh

Automated cron script that runs every 30 minutes to update multiple HTML pages.

**Schedule**:
```cron
*/30 * * * * $HOME/nvidia/dynamo-utils/update_html_pages.sh
```

**Tasks Performed**:
1. **Cleanup old logs** (runs first)
   - Keeps only the last 10 non-empty dated directories in `$LOGS_DIR` (defaults to `$NVIDIA_HOME/logs`)
   - Deletes older directories to save disk space
   - Logs all cleanup actions with directory counts

2. Updates branch status HTML (`$NVIDIA_HOME/index.html` where `NVIDIA_HOME` defaults to parent of script directory)
   - Calls `show_dynamo_branches.py --html --output $BRANCHES_TEMP_FILE`
   - Shows status of all dynamo branches
   - Uses atomic file replacement (temp file → final file)

3. Updates commit history HTML (`$DYNAMO_REPO/index.html` where `DYNAMO_REPO` defaults to `$NVIDIA_HOME/dynamo_latest`)
   - Calls `show_commit_history.py --repo-path . --html --max-commits 200 --output $COMMIT_HISTORY_HTML`
   - Leverages caching for fast updates (only calculates new commits)
   - Shows last 200 commits with composite SHAs and Docker images

**Log file**: `$LOGS_DIR/cron.log` (where `LOGS_DIR` defaults to `$NVIDIA_HOME/logs`)

**Performance Optimization**:
- First run: ~50-100 seconds (calculates all 200 commits)
- Subsequent runs: ~5-10 seconds (only processes new commits, rest from cache)
- Cache file size: ~20-40KB for 200 commits
- No cache invalidation needed: Composite SHAs are deterministic based on file content

---

### common.py

Shared utilities module providing reusable components across all scripts.

**Key Classes**:
- `GitUtils`: GitPython API wrapper (no subprocess calls)
- `GitHubAPIClient`: GitHub API client with auto token detection and rate limit handling
- `DockerUtils`: Docker operations and image management
- `DynamoImageInfo`, `DockerImageInfo`: Dataclasses for image metadata (used by retag script)

**Utility Functions**:
- `get_terminal_width()`: Terminal width detection with fallback
- `normalize_framework()`: Framework name normalization
- `get_framework_display_name()`: Pretty framework names for output

---

### show_dynamo_branches.py

Branch status checker that displays information about dynamo* repository branches with GitHub PR integration.

**Features**:
- Scans all dynamo* directories for branch information
- Queries GitHub API for PR status using `common.GitHubAPIClient`
- Supports both terminal and HTML output modes
- Parallel data gathering for improved performance
- Automatic GitHub token detection (env var, gh CLI config)

---

### git_stats.py

Git repository statistics analyzer that provides detailed contributor metrics and rankings.

**Features**:
- Analyzes git commit history for any time range
- Tracks unique contributors with commit counts
- Calculates lines added/deleted/changed per contributor
- Provides average statistics per person
- Dual ranking views: by commits and by lines changed
- Supports flexible time ranges (days, since/until dates, all-time)

**Usage Examples**:

```bash
# Statistics for last 30 days
python3 git_stats.py --days 30

# Statistics for last 7 days
python3 git_stats.py --days 7

# Statistics since a specific date
python3 git_stats.py --since "2025-01-01"

# Statistics for a date range
python3 git_stats.py --since "2025-01-01" --until "2025-01-31"

# All time statistics
python3 git_stats.py
```

**Output Includes**:
1. Number of unique contributors
2. Total commits and line changes
3. Average commits per person
4. Average lines added/deleted/changed per person
5. Contributor rankings by commits (with full details)
6. Contributor rankings by lines changed

**Sample Output**:
```
================================================================================
Git Repository Statistics - Last 30 days
================================================================================

Number of unique contributors: 59
Total commits: 340
Total lines added: 148,764
Total lines deleted: 61,093
Total lines changed: 209,857

Average commits per person: 5.8
Average lines added per person: 2521.4
Average lines deleted per person: 1035.5
Average lines changed per person: 3556.9

================================================================================
Contributor Rankings (by commits)
================================================================================

Rank   Name                           Email                               Commits  Added      Deleted    Changed   
--------------------------------------------------------------------------------------------------------------
1      John Doe                       john@example.com                    31       4,455      3,512      7,967
...
```

---

## Environment Setup

The typical workflow for setting up a development environment:

1. Clone the Dynamo repository
2. Use `./compile.sh --dev` to build in development mode
3. Test with `./inference.sh` and `./curl.sh`

## Tips & Best Practices

1. **Development Mode**: Use `./compile.sh --dev` for faster iteration during development
2. **Testing**: Always test API endpoints with `./curl.sh` after starting services
3. **Configuration Sync**: Run `devcontainer_sync.sh` periodically or via cron to keep configs updated
4. **Container Development**: Use the Dev Container for a consistent development environment
5. **Port Conflicts**: Check port availability before running inference services

## Troubleshooting

### Port Already in Use
If you encounter port conflicts when running `inference.sh`:
```bash
# Check what's using the port
lsof -i :8000
# Kill the process or use different ports
DYN_FRONTEND_PORT=8090 DYN_BACKEND_PORT=8091 ./inference.sh
```

### Build Failures
For build issues with `compile.sh`:
```bash
# Clean and rebuild
./compile.sh --cargo-clean
./compile.sh --python-clean
./compile.sh --dev
```

### Container Issues
If Docker container fails to start:
```bash
# Check Docker daemon
docker ps
# Verify GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## Contributing

When contributing to this repository:
1. Test scripts thoroughly before committing
2. Update this README if adding new scripts or features
3. Use meaningful commit messages with `--signoff`

## Additional Documentation

- **CLAUDE.md**: Operational procedures, environment setup, and inference server documentation
- **.cursorrules**: Coding conventions, style guidelines, and development practices
