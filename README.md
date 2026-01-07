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

## Development Environment Notes (host vs dev container)

- **Host venv**: On the host machine, activate your local venv before running Python tooling (pre-commit, linters, etc.). On the keivenc setup this is typically `~/nvidia/venv/bin/activate`.
- **Dev container**: Inside the dev container, the environment is typically pre-configured/activated.
- **Path mapping (common setup)**:
  - Host: `~/nvidia/dynamo-utils`
  - Dev container: `/workspace/_`

---

## Directory Structure

```
dynamo-utils/
├── backup.sh                     # Smart backup with versioned history
├── common.py                     # Shared utilities module (API clients, caching, parallelization)
├── common_types.py               # Shared enums/types (used by API + dashboards)
├── compile.sh                    # Build and install Dynamo Python packages
├── cron_log.sh                   # Cron wrapper that writes logs to ~/nvidia/logs/YYYY-MM-DD/<job>.log
├── curl.sh                       # Test models via chat completions API
├── soak_fe.py                    # Frontend soak testing script
├── devcontainer.json.j2          # VS Code Dev Container template (Jinja2)
├── devcontainer_sync.py          # Sync dev configs across projects
├── git_stats.py                  # Git repository statistics analyzer
├── gpu_reset.sh                  # GPU reset utility
├── inference.sh                  # Launch Dynamo inference services
├── resource_monitor.py           # Periodic system + GPU sampler -> SQLite
├── html_pages/                   # HTML dashboard generators (see html_pages/README.md)
└── container/                    # Docker build/test/cleanup tools (see container/README.md)
```

---

## Dashboards / log categorization pitfalls (learnings)

Repeated mistakes we hit while iterating on `dynamo-utils/html_pages/*` dashboards and
`dynamo-utils/ci_log_errors/core.py` (shared library + CLI):

- **Golden logs + self-test discipline**
  - After changing categorization/snippet logic, run:
    - `python3 dynamo-utils/ci_log_errors/core.py --self-test-examples`
  - If you update the “Category frequency summary”, ensure every category has at least one golden
    training example in the docstring list.
  - Golden logs must be preserved: keep them **non-writable**; scans/retrain helpers should not
    accidentally make them writable.

- **Category naming + canonicalization**
  - Use canonical category tokens in examples (e.g. `github-lfs-error`, not `github-LFS-error`).
  - If you rename/add categories, update:
    - categorization regexes
    - `CATEGORY_RULES`
    - suppression rules (e.g. “generic timeout” must not appear when any specific timeout matches)
    - the docstring examples list

- **Shared logic drift (full log vs snippet)**
  - Don’t duplicate categorization rules in multiple places. Prefer a single rule table + shared
    application helper.
  - If a category exists but never triggers, check it’s actually wired into the shared rule table.

- **Back-compat / imports**
  - When refactoring shared dashboard libs (e.g. `html_pages/common_dashboard_lib.py`), do NOT delete
    exported symbols without fixing all importers.
  - Symptom: `./html_pages/update_html_pages.sh` “runs too quickly” because a generator crashed early
    (ImportError).
  - Prefer keeping a small back-compat constant/export over breaking runtime imports.

- **HTML table column collapse**
  - Don’t `display:none` `<td>` cells to “collapse” a column — it breaks alignment.
  - Instead, keep the `<td>` and hide its *contents* (wrapper span/div) so the column count stays
    stable.

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

#### `devcontainer_sync.py`
Automatically syncs development configuration files across multiple Dynamo project directories.

```bash
# Sync configurations
./devcontainer_sync.py

# Dry run to preview changes
./devcontainer_sync.py --dryrun

# Force sync regardless of changes
./devcontainer_sync.py --force

# Silent mode for cron jobs
./devcontainer_sync.py --silent
```

---

## Python Utilities

### HTML dashboards (`html_pages/`)

All HTML dashboard documentation (generators, cron wrapper, caching/API budgets, and logs) lives in:

- `html_pages/README.md`

---

### gitlab_pipeline_pr_map.py

**Overview**: Map GitLab pipeline URLs/IDs to the Merge Request IID(s) ("PR#") by resolving:
pipeline → SHA → MR(s).

#### Usage

```bash
# Read pipeline URLs from stdin, output CSV
cat pipelines.txt | python3 gitlab_pipeline_pr_map.py --format csv > out.csv

# Mix URLs and numeric IDs, show a compact table
python3 gitlab_pipeline_pr_map.py 40743226 https://gitlab-master.nvidia.com/dl/ai-dynamo/dynamo/-/pipelines/38895507 --format table
```


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
- Log file: `$NVIDIA_HOME/logs/YYYY-MM-DD/backup.log`
- Captures all backup operations

#### Cron Setup

```bash
# Backup every 6 minutes
*/6 * * * * NVIDIA_HOME=$HOME/nvidia $HOME/nvidia/dynamo-utils/cron_log.sh backup $HOME/nvidia/dynamo-utils/backup.sh --input-path $HOME/nvidia --output-path /mnt/sda/keivenc/nvidia
```

---

### container/build_images.py

Container tooling documentation lives in:

- `container/README.md`

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
3. **Configuration Sync**: Run `devcontainer_sync.py` periodically (or via cron)
4. **Container Development**: Use Dev Container for consistent environment
5. **Port Conflicts**: Check port availability before running inference services
6. **Caching**: Many scripts use `~/.cache/dynamo-utils/` for performance; see tool-specific docs for details.
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
```

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
