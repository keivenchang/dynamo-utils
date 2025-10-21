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
├── dynamo_docker_builder.py      # Automated Docker build/test pipeline
├── common.py                     # Shared utilities module
├── show_dynamo_branches.py       # Branch status checker
├── update_html_pages.sh          # HTML page update cron script
├── soak_fe.py                    # Frontend soak testing script
├── retag_latest_dynamo_images.py # Docker image retagging utility
├── gpu_reset.sh                  # GPU reset utility
└── rm_old_dynamo_docker_images.sh # Cleanup old Docker images
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
./curl.sh --port 8080 --prompt "Hello!"

# Streaming with retry
./curl.sh --stream --retry --prompt "Tell me a story"

# Loop testing with metrics
./curl.sh --loop --metrics --random
```

**Options:**
- `--port`: API server port (default: 8080)
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
- `DYN_FRONTEND_PORT`: Frontend port (default: 8080)
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

### dynamo_docker_builder.py

Automated Docker build and test pipeline system that builds and tests multiple inference frameworks (VLLM, SGLANG, TRTLLM) across different target environments (base, dev, local-dev), generates HTML reports, and sends email notifications.

#### Usage Examples

**Quick Test (Single Framework)**:
```bash
python3 dynamo_docker_builder.py --sanity-check-only --framework sglang --force-run --email <email>
```

**Parallel Build with Skip**:
```bash
python3 dynamo_docker_builder.py --skip-build-if-image-exists --parallel --force-run --email <email>
```

**Full Build**:
```bash
python3 dynamo_docker_builder.py --parallel --force-run --email <email>
```

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

#### Process Management
- Use lock files to prevent concurrent runs
- Lock file: `.dynamo_docker_builder.py.lock`
- Store PID in lock file
- Check if process is still running before acquiring lock
- Clean up stale locks automatically

#### Email Notifications
- Use SMTP (localhost:25) for emails
- Email subject format: `[DynamoDockerBuilder] {status} - {sha_short}`
- HTML email body with clickable links (absolute URLs)
- Plain text fallback for email clients

#### Commit History Feature

**Overview**: The `--show-commit-history` flag displays recent commits with their composite SHAs. Supports both terminal and HTML output modes with integrated caching for performance.

**Caching System**:
- Cache file: `~/nvidia/dynamo_ci/.commit_history_cache.json`
- Format: JSON mapping of commit SHA (full) → composite SHA
- Purpose: Avoid expensive git checkout + composite SHA recalculation
- Performance: Cached lookups are nearly instant vs ~1-2 seconds per commit calculation

**Usage Examples**:

Terminal output with caching:
```bash
python3 dynamo_docker_builder.py --show-commit-history --max-commits 50 --repo-path ~/nvidia/dynamo_ci
```

HTML output with caching (generates `~/nvidia/dynamo_ci/logs/commit-history.html`):
```bash
python3 dynamo_docker_builder.py --show-commit-history --html --max-commits 50 --repo-path ~/nvidia/dynamo_ci
```

Verbose mode (shows cache hits/misses):
```bash
python3 dynamo_docker_builder.py --show-commit-history --html --max-commits 50 --repo-path ~/nvidia/dynamo_ci --verbose
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
- Location: `dynamo_docker_builder.py:1810-1851` and line 2003-2004
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

Automated cron script that runs every 5 minutes during daytime (9am-9pm) to update multiple HTML pages.

**Schedule**:
```cron
*/5 9-20 * * * $HOME/nvidia/dynamo-utils/update_html_pages.sh
```

**Tasks Performed**:
1. **Cleanup old logs** (runs first)
   - Keeps only the last 10 non-empty dated directories in `~/nvidia/dynamo_ci/logs/`
   - Deletes older directories to save disk space
   - Logs all cleanup actions with directory counts

2. Updates branch status HTML (`$HOME/nvidia/index.html`)
   - Calls `show_dynamo_branches.py --html`
   - Shows status of all dynamo branches

3. Updates commit history HTML (`~/nvidia/dynamo_ci/logs/commit-history.html`)
   - Calls `dynamo_docker_builder.py --show-commit-history --html --max-commits 50`
   - Leverages caching for fast updates (only calculates new commits)
   - Shows last 50 commits with composite SHAs and Docker images

**Log file**: `~/nvidia/dynamo-utils/update_html_pages.log`

**Performance Optimization**:
- First run: ~50-100 seconds (calculates all 50 commits)
- Subsequent runs: ~5-10 seconds (only processes new commits, rest from cache)
- Cache file size: ~5-10KB for 50 commits
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
lsof -i :8080
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
