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
- GitHub CLI (`gh`) - optional, for enhanced features

---

## Directory Structure

```
dynamo-utils/
├── backup.sh                     # Smart backup with versioned history
├── common.py                     # Shared utilities module (API clients, caching, parallelization)
├── compile.sh                    # Build and install Dynamo Python packages
├── curl.sh                       # Test models via chat completions API
├── soak_fe.py                    # Frontend soak testing script
├── devcontainer.json             # VS Code Dev Container configuration
├── devcontainer_sync.sh          # Sync dev configs across projects
├── git_stats.py                  # Git repository statistics analyzer
├── gpu_reset.sh                  # GPU reset utility
├── inference.sh                  # Launch Dynamo inference services
├── resource_monitor.py           # Periodic system + GPU sampler -> SQLite
├── resource_report.py            # Fancy interactive HTML charts from resource_monitor.sqlite
├── show_commit_history.j2        # HTML template for commit history
├── show_commit_history.py        # Commit history with CI status and Docker images
├── show_dynamo_branches.py       # Branch status checker
├── update_html_pages.sh          # HTML page update cron script
└── container/                    # Docker-related scripts
    ├── build_images.py               # Automated Docker build/test pipeline
    ├── build_images_report.html.j2   # HTML report template
    ├── cleanup_old_images.sh         # Cleanup old Docker images
    ├── restart_gpu_containers.sh     # GPU error monitoring/restart
    └── retag_images.py               # Docker image retagging utility
```

---

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

---

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

---

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

### show_commit_history.py

**Overview**: Displays recent commits with CI status, Docker images, and metadata in terminal or HTML format.

#### Features

**Core Functionality:**
- 6-column table: Commit SHA, Composite Docker SHA (CDS), CI Status, Date/Time (PST), Author, Message
- Local build status (not GitHub Actions) with colored indicators
- Per-commit CI status tracking with CDS inheritance fallback
- GitLab Docker registry integration
- GitHub PR merge date integration
- Auto-reload every 15 minutes in HTML mode

**Performance Optimizations:**
- Forever caching for immutable data (merge dates, completed pipelines)
- Parallel API calls with ThreadPoolExecutor (10 workers)
- Early termination for GitLab registry fetching (stops at 8-hour old tags)
- Consolidated subprocess calls (50% reduction)
- Generation time: ~34 seconds (200 commits with optimizations)

#### Quick Start

**Simplest usage** (terminal output, 50 commits):
```bash
cd ~/nvidia/dynamo-utils
python3 show_commit_history.py --repo-path ~/nvidia/dynamo_latest --max-commits 50
```

**Production HTML** (what the cron job runs):
```bash
python3 show_commit_history.py \
  --repo-path ~/nvidia/dynamo_latest \
  --max-commits 200 \
  --html \
  --output ~/nvidia/dynamo_latest/index.html
```

#### Usage Examples

```bash
# Terminal output (default, 50 commits)
python3 show_commit_history.py --repo-path ~/nvidia/dynamo_latest --max-commits 50

# HTML output with all features (200 commits)
python3 show_commit_history.py --repo-path ~/nvidia/dynamo_latest --max-commits 200 --html --output ~/nvidia/dynamo_latest/index.html

# Fast mode: Skip GitLab fetch (use cache only)
python3 show_commit_history.py --repo-path ~/nvidia/dynamo_latest --max-commits 50 --skip-gitlab-fetch

# Debug mode: Show cache hits/misses
python3 show_commit_history.py --repo-path ~/nvidia/dynamo_latest --max-commits 50 --verbose

# Debug mode: Show even more details
python3 show_commit_history.py --repo-path ~/nvidia/dynamo_latest --max-commits 50 --debug
```

#### Command-Line Options

**Required:**
- `--repo-path PATH`: Path to the Dynamo git repository (e.g., `~/nvidia/dynamo_latest`)

**Common Options:**
- `--max-commits N`: Number of commits to display (default: 50, production: 200)
- `--html`: Generate HTML output instead of terminal output
- `--output FILE`: Output file path (required when using `--html`)

**Performance Options:**
- `--skip-gitlab-fetch`: Skip GitLab API calls, use cached data only (faster, ~1 second)
- `--skip-fetch`: Alias for `--skip-gitlab-fetch`

**Debug Options:**
- `--verbose`: Show cache hit/miss information and timing
- `--debug`: Show detailed debug information (very verbose)

**Usage Notes:**
- First run will be slower (~50-100s) as it builds the cache
- Subsequent runs are fast (~5-10s for terminal, ~34s for HTML with API calls)
- Use `--skip-gitlab-fetch` for instant results when you don't need fresh data
- Cache files are stored under `~/.cache/dynamo-utils/` (or override via `DYNAMO_UTILS_CACHE_DIR`)
- HTML auto-reloads every 15 minutes when viewed in browser

#### Caching System

**Cache Files** (5 active, stored in `~/.cache/dynamo-utils/`):
- `commit_history.json` (186 KB) - Full commit metadata
- `github_pr_merge_dates.json` (210 PRs) - GitHub PR merge dates (forever cache)
- `gitlab_commit_sha.json` (2.5 MB, 1349 entries) - Docker registry images
- `gitlab_pipeline_status.json` (61 KB, 458 entries) - CI pipeline status
- `gitlab_pipeline_jobs.json` (13 KB, 87 entries) - Pipeline job counts

**Cache Strategy:**
- Forever cache: Immutable data (merge dates, completed pipelines)
- Fresh data: Running pipelines, recent commits (< 30 min)
- Cache hit rate: 97-98% for typical runs

#### API Usage

**GitHub API** (Minimal usage):
- Single endpoint: `GET /repos/ai-dynamo/dynamo/pulls/{pr_number}`
- Purpose: Fetch PR merge dates only
- Average per run: ~2 API calls (only new PRs)
- First run (cold cache): ~180 API calls (one-time)
- Daily average: ~5-10 calls total (across all runs)
- Rate limit impact: Negligible (< 10/hour vs 5,000/hour limit)

**GitLab API** (Optimized):
- Pipeline status: Parallelized with ThreadPoolExecutor (10 workers)
- Registry tags: Early termination at 8-hour old tags (~90% reduction)
- Job counts: Parallelized fetching

**GitHub CLI** (Optimized):
- `gh pr checks`: Consolidated from 3 calls → 2 calls per PR (33% reduction)
- `gh pr view`: Eliminated (redundant with `gh pr checks`)
- Total subprocess reduction: 50% (4 → 2 calls per PR)

#### CI Status Indicators

**Source:** Local build logs at `/home/keivenc/nvidia/dynamo_ci/logs/YYYY-MM-DD/`

**Status Markers:**
- `.PASS` - All builds/compile/sanity passed (green circle)
- `.FAIL` - Some failed (red circle)
- `.RUNNING` - Building (yellow circle)
- Gray circle - Unknown/no status
- Hollow circle - Inherited from CDS (no own build logs)

**Build Stages:**
- `*-build`, `*-chown`, `*-compilation`, `*-sanity`
- Frameworks: vllm, sglang, trtllm
- Variants: dev, local-dev, runtime

**Priority:** FAIL > RUNNING > PASS

#### HTML Output Features

**Interactive Elements:**
- Clickable commit SHAs linking to GitHub
- Clickable PR numbers
- Expandable dropdowns with full commit details
- Docker image lists with copy buttons
- Legend & Help section

**Layout:**
- 6 columns with proper colspan for dropdowns
- GitHub-style CSS
- Full-width container for legend
- Timezone: All dates in PST (converted from UTC)

#### Performance

**Generation Time:**
- Current (optimized): 34.4 seconds (200 commits)
- Previous (before optimizations): 49.3 seconds
- Original (no optimizations): ~100+ seconds
- **Speed improvement: 66% faster than original**

**Cache Performance:**
- First run: ~50-100 seconds (calculates all)
- Subsequent runs: ~5-10 seconds (only new commits)
- Cache size: ~2.7 MB total (5 cache files)

---

### common.py

**Overview**: Shared utilities module with API clients, caching, and parallelization support.

#### Key Classes

**GitHubAPIClient:**
- GitHub REST API client with token auto-detection
- Methods: `get_pr_details()`, `get_ci_status()`, `get_failed_checks()`, `get_running_checks()`
- Caching: Forever cache for immutable data (merge dates)
- Optimization: Consolidated subprocess calls, parallel fetching

**GitLabAPIClient:**
- GitLab API client for CI/CD and Docker registry
- Methods: `get_cached_pipeline_status()`, `get_cached_pipeline_job_counts()`, `get_cached_registry_images_for_shas()`
- Caching: Forever cache for completed pipelines
- Optimization: Parallel fetching (ThreadPoolExecutor), early termination for registry

**GitUtils:**
- GitPython API wrapper (no subprocess calls)
- Methods: Repository operations, branch management, commit queries

**DockerUtils:**
- Docker operations and image management
- Methods: Image listing, tagging, cleanup

#### Optimizations Applied

**1. Consolidated Subprocess Calls:**
- `gh pr checks`: 3 → 2 calls per PR (33% reduction)
- `gh pr view`: Eliminated completely (100% reduction)
- Total: 4 → 2 calls per PR (50% reduction)

**2. Parallelized API Calls:**
- GitHub PR merge dates: 10 workers (10x faster)
- GitLab pipeline status: 10 workers (8-10x faster)
- GitLab job counts: 10 workers (8-10x faster)

**3. Early Termination:**
- GitLab registry: Stops at 8-hour old tags (~90% reduction)
- Smart batching: 10 pages per batch for better control

**4. Code Quality:**
- Moved 48 duplicate imports to top (PEP 8)
- Simplified GitLab early termination logic (68 → 46 lines)
- Single source of truth for timestamp checking

#### Utility Functions

- `get_terminal_width()`: Terminal width detection with fallback
- `normalize_framework()`: Framework name normalization
- `get_framework_display_name()`: Pretty framework names for output

---

### backup.sh

**Overview**: Smart backup with versioned history using rsync.

#### Usage

```bash
# Basic backup (required parameters)
./backup.sh --input-path ~/nvidia --output-path /mnt/sda/keivenc/nvidia

# Show help
./backup.sh --help
```

**Parameters:**
- `--input-path`: Source directory (required)
- `--output-path`: Destination directory (required)

#### Features

**Versioned History:**
- Format: `backup_history/YYYYMMDD_HHMMSS/`
- Location: Parent directory of destination
- Preserves changed/deleted files before updating
- Shows count and size of backed up files
- Auto-cleanup of empty history directories

**Exclusions:**
- Git metadata: `.git/FETCH_HEAD`, `.git/HEAD`, `.git/index`, `.git/logs/`
- Reduces backup churn from frequently-changing git files

**Logging:**
- Log file: `/tmp/backup.log`
- Captures all backup operations

#### Cron Setup

```bash
# Backup every 6 minutes
*/6 * * * * $HOME/nvidia/dynamo-utils/backup.sh --input-path $HOME/nvidia --output-path /mnt/sda/keivenc/nvidia >> /tmp/backup.log 2>&1
```

---

### update_html_pages.sh

**Overview**: Automated cron script that updates HTML pages every 15 minutes.

#### Schedule

```cron
*/15 * * * * $HOME/nvidia/dynamo-utils/update_html_pages.sh
```

#### Tasks Performed

1. **Cleanup old logs** (runs first)
   - Keeps only the last 10 non-empty dated directories in `$LOGS_DIR`
   - Deletes older directories to save disk space
   - Logs all cleanup actions

2. **Updates commit history HTML**
   - Location: `$DYNAMO_REPO/index.html` (default: `$NVIDIA_HOME/dynamo_latest`)
   - Shows last 200 commits with CI status and Docker images
   - Leverages caching for fast updates (~34 seconds)
   - Uses `--skip-gitlab-fetch` for cache-only mode when appropriate

3. **Updates branch status HTML** (optional)
   - Location: `$NVIDIA_HOME/index.html`
   - Shows status of all dynamo branches
   - Uses atomic file replacement

**Log file:** `$LOGS_DIR/cron.log` (default: `$NVIDIA_HOME/logs`)

---

### container/build_images.py

**Overview**: Automated Docker build and test pipeline for multiple inference frameworks.

#### Usage

```bash
# Quick test (single framework)
python3 container/build_images.py --repo-path ~/nvidia/dynamo_ci --sanity-check-only --framework sglang --force-run

# Parallel build with skip
python3 container/build_images.py --repo-path ~/nvidia/dynamo_ci --skip-action-if-already-passed --parallel --force-run

# Full build
python3 container/build_images.py --repo-path ~/nvidia/dynamo_ci --parallel --force-run
```

#### Features

**Frameworks Supported:**
- VLLM, SGLANG, TRTLLM

**Target Environments:**
- base, dev, local-dev

**Build Stages:**
- Build, chown, compilation, sanity checks

**Process Management:**
- Lock files prevent concurrent runs (`.build_images.lock`)
- PID tracking with stale lock cleanup
- Dry-run mode for testing

**HTML Report Generation:**
- Automatic report generation with clickable links
- Two versions: File paths vs absolute URLs for email
- Log file paths: `~/nvidia/dynamo_ci/logs/YYYY-MM-DD/`
- Report: `YYYY-MM-DD.{sha_short}.report.html`

**Email Notifications:**
- SMTP server: `smtp.nvidia.com:25`
- Subject format: `{SUCC|FAIL}: DynamoDockerBuilder - {sha_short} [{failed_tasks}]`
- HTML email body with clickable links
- Not sent in dry-run mode

---

### resource_monitor.py / resource_report.py

**Overview**:
- `resource_monitor.py` periodically samples CPU/MEM/IO + NVIDIA GPU metrics and appends them to a SQLite DB.
- `resource_report.py` generates a **fancy interactive HTML report** (Plotly, zoom/pan/range buttons) and marks best-effort **top-process CPU spikes**.

#### Usage

```bash
# 7-day report (includes zoom buttons for 1d / 12h / 6h / 1h)
python3 resource_report.py \
  --db-path ~/.cache/dynamo-utils/resource_monitor.sqlite \
  --output ~/nvidia/dynamo_latest/resource_report_7d.html \
  --days 7

# 1-day report (smaller + faster)
python3 resource_report.py \
  --db-path ~/.cache/dynamo-utils/resource_monitor.sqlite \
  --output ~/nvidia/dynamo_latest/resource_report_1d.html \
  --days 1
```

### show_dynamo_branches.py

**Overview**: Branch status checker with GitHub PR integration.

#### Features

- Scans all dynamo* directories for branch information
- Queries GitHub API for PR status
- Supports both terminal and HTML output modes
- Parallel data gathering
- Automatic GitHub token detection

#### Usage

```bash
# Terminal output
python3 show_dynamo_branches.py

# HTML output
python3 show_dynamo_branches.py --html --output ~/nvidia/index.html
```

---

### git_stats.py

**Overview**: Git repository statistics analyzer with contributor metrics.

#### Features

- Analyzes git commit history for any time range
- Tracks unique contributors with commit counts
- Calculates lines added/deleted/changed per contributor
- Provides average statistics per person
- Dual ranking views: by commits and by lines changed

#### Usage

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

---

## Environment Setup

The typical workflow for setting up a development environment:

1. Clone the Dynamo repository
2. Use `./compile.sh --dev` to build in development mode
3. Test with `./inference.sh` and `./curl.sh`
4. Set up cron jobs for automated updates

---

## Tips & Best Practices

1. **Development Mode**: Use `./compile.sh --dev` for faster iteration
2. **Testing**: Always test API endpoints with `./curl.sh` after starting services
3. **Configuration Sync**: Run `devcontainer_sync.sh` periodically or via cron
4. **Container Development**: Use Dev Container for consistent environment
5. **Port Conflicts**: Check port availability before running inference services
6. **Caching**: Leverage caching in scripts for performance (show_commit_history.py)
7. **Backups**: Use `backup.sh` with versioned history for important data
8. **API Usage**: Optimize API calls with caching and parallelization

---

## Troubleshooting

### Port Already in Use
```bash
# Check what's using the port
lsof -i :8000

# Kill the process or use different ports
DYN_FRONTEND_PORT=8090 DYN_BACKEND_PORT=8091 ./inference.sh
```

### Build Failures
```bash
# Clean and rebuild
./compile.sh --cargo-clean
./compile.sh --python-clean
./compile.sh --dev
```

### Container Issues
```bash
# Check Docker daemon
docker ps

# Verify GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Cache Issues
```bash
# Clear specific cache file
rm ~/.cache/dynamo-utils/github_pr_merge_dates.json

# Or clear all cache files
rm -rf ~/.cache/dynamo-utils/

# Regenerate with fresh data
python3 show_commit_history.py --repo-path ~/nvidia/dynamo_latest --max-commits 200 --html --output ~/nvidia/dynamo_latest/index.html
```

---

## Performance Metrics

### show_commit_history.py (200 commits)

**Generation Time:**
- Current (optimized): 34.4 seconds
- Previous: 49.3 seconds
- Original: ~100 seconds
- **Improvement: 66% faster**

**API Calls (average per run):**
- GitHub: ~2 calls (only new PRs)
- GitLab: ~10-20 calls (with caching)
- Cache hit rate: 97-98%

**Subprocess Calls (per PR):**
- Before: 4 calls
- After: 2 calls
- **Reduction: 50%**

---

## Contributing

When contributing to this repository:

1. Test scripts thoroughly before committing
2. Update this README if adding new scripts or features
3. Use meaningful commit messages with `--signoff`
4. Optimize for performance (use caching, parallel calls)
5. Follow PEP 8 for Python code (imports at top, etc.)

---

## Additional Documentation

- **CLAUDE.md**: Operational procedures, environment setup, and inference server documentation
- **.cursorrules**: Coding conventions, style guidelines, and development practices

---

## License

This repository is for internal development purposes only. Not for public distribution.
