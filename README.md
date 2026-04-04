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

- **Host venv**: On the host machine, activate your local venv before running Python tooling (pre-commit, linters, etc.). On common setups this is typically `<workspace>/venv/bin/activate` (e.g. `~/dynamo/venv/bin/activate`).
- **Dev container**: Inside the dev container, the environment is typically pre-configured/activated.
- **Path mapping (common setup)**:
  - Host: `~/dynamo/dynamo-utils.PRODUCTION`
  - Dev container: `/workspace/.utils`

---

## GitHub API Optimizations (2026-01-18)

The `common_github.py` module includes several optimizations that reduce GitHub API usage by 85-98%:

### Key Optimizations

1. **ETag Support (Conditional Requests)**
   - `_rest_get()` supports `If-None-Match` header for conditional requests
   - 304 Not Modified responses **don't count against rate limit**
   - Cache schema v6 stores ETags for `/commits/{sha}/check-runs` and `/commits/{sha}/status`
   - Benefit: 85-95% rate limit reduction on subsequent runs within cache TTL

2. **Batched Workflow Run Fetching**
   - Collects all `run_ids` first, then fetches workflow metadata in batch
   - Implemented in `get_pr_checks_rows()` (lines 3247-3271)
   - Benefit: ~90% reduction (100 individual calls → 10-20 batched calls)

3. **Parallelization Bug Fix**
   - Fixed `get_required_checks_for_base_ref()` being called 100× in parallel loop
   - Now called once before loop and passed as parameter
   - Benefit: 99 redundant API calls eliminated per run

4. **Batched Job Fetching (Infrastructure Ready)**
   - `get_actions_runs_jobs_batched()`: Fetch all jobs for multiple workflow runs
   - Uses `/actions/runs/{run_id}/jobs` (returns all jobs in one call)
   - Status: ✅ Implemented, ⏳ Not yet wired up (requires refactoring lazy materialization)
   - Potential benefit: 95% reduction (500-1000 → 10-20 calls)

### Impact

```
Before: ~2000 API calls per run → Rate limit exhausted after 2-3 runs
After:  ~200-300 calls (first run), ~10-30 calls (subsequent runs with ETags)
Result: 85-98% reduction → 16-500 runs before rate limit exhaustion
```

All optimizations are in `common_github.py`, so **all dashboard scripts benefit uniformly**:
- `html_pages/show_commit_history.py`
- `html_pages/show_local_branches.py`
- `html_pages/show_remote_branches.py`

---

## Directory Structure

```
dynamo-utils/
├── await_output.sh               # Run command, exit on sentinel string (replaces sleep+grep)
├── aws-ecr-setup.sh              # AWS ECR login/logout/list-images for NVIDIA container registry
├── backup.sh                     # Smart backup with versioned history
├── check_pr_status.py            # Show detailed PR check status (required vs non-required)
├── clean_log.sh                  # Delete old YYYY-MM-DD/ log directories (default: >30 days)
├── clean_system.sh               # Orchestrate cleanup of old images, logs, VSC containers
├── common.py                     # Shared utilities module (API clients, caching, parallelization)
├── common_build_report.py        # Typed dataclasses for build report JSON serialization
├── common_types.py               # Shared enums/types (used by API + dashboards)
├── compile.sh                    # Build and install Dynamo Python packages
├── cron_log.sh                   # Cron wrapper that writes logs to ~/dynamo/logs/YYYY-MM-DD/<job>.log
├── curl.sh                       # Test models via chat completions API
├── ddns.py                       # Dynamic DNS updater for *.dyn.nvidia.com hostnames
├── devcontainer.json.j2          # VS Code Dev Container template (Jinja2)
├── devcontainer_sync.py          # Sync dev configs across projects
├── git_stats.py                  # Git repository statistics analyzer
├── gpu_reset.sh                  # GPU reset utility
├── inference.sh                  # Launch Dynamo inference services
├── kill_dynamo_processes.sh      # Safe kill of dynamo/vllm/sglang inference processes
├── py_indent_report.py           # Python indentation gate (py_compile -tt + tabnanny + indent report)
├── read_cursor_transcript.py     # Search, list, and continue Cursor agent transcripts
├── recompress_backups.sh         # Re-compress .tgz backup archives with updated exclusions
├── rerun_github_pr.py            # Monitor PRs and auto-rerun failed infrastructure jobs
├── resource_monitor.py           # Periodic system + GPU sampler -> SQLite
├── soak_fe.py                    # Frontend soak testing script
├── update_cron_tail.sh           # Extract last 135 lines of cron.log for quick inspection
│
├── cache/                        # Cache subsystem (base classes, duration, job log caching)
├── ci_log_errors/                # CI log categorization engine (regex + snippet extraction)
├── common_github/                # GitHub API client with ETags, caching, batched fetching
├── common_gitlab/                # GitLab API client with caching
├── container/                    # Docker build/test/cleanup tools (see container/README.md)
├── html_pages/                   # HTML dashboard generators (see html_pages/README.md)
│   ├── gpu_monitor.py            # Real-time GPU/CPU/disk/network web dashboard (WebSocket + Plotly)
│   └── start_gpu_monitor.sh      # Launcher wrapper for gpu_monitor.py
└── reports/                      # Repository and CI analytics
    ├── analyze_repo.py           # Repository analysis
    ├── contributor_stats.py      # Contributor statistics
    └── pr_ci_report.py           # PR & CI statistics report generator
```

---

## Python indentation checker (`py_indent_report.py`)

If you keep hitting `IndentationError` / `TabError` / “off-by-one indent” mistakes, use this tool as a quick, single-command gate.

What it does:
- Runs `python -tt -m py_compile <file>` (hard syntax/indent gate)
- Runs `tabnanny` (ambiguous/mixed indentation detector)
- Prints a compact indentation report and flags suspicious-but-legal indent patterns (e.g., an `else:` block that indents by 8 instead of 4)

Examples:
```bash
# Show only suspicious lines (recommended)
python3 py_indent_report.py --only-problems html_pages/common_dashboard_lib.py

# Show more context (all non-blank lines)
python3 py_indent_report.py --all html_pages/common_dashboard_lib.py

# Verify the detector catches common mistakes
python3 py_indent_report.py --self-check
```

## Dashboards / log categorization pitfalls (learnings)

Repeated mistakes we hit while iterating on `dynamo-utils.PRODUCTION/html_pages/*` dashboards and
`dynamo-utils.PRODUCTION/ci_log_errors/*` (shared library + CLI):

- **Golden logs + self-test discipline**
  - After changing categorization/snippet logic, run:
    - `cd dynamo-utils && python3 -m ci_log_errors --self-test-examples`
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

### Monitoring

#### `html_pages/gpu_monitor.py`
Real-time GPU/CPU/disk/network monitor served as a web dashboard. Uses multiprocessing to sample
at different rates (PCIe at max speed, GPU/CPU at 5/s, disk at 1/s), WebSocket push to the browser,
and Plotly.js for interactive time-series charts. Tracks per-process GPU memory and CPU usage with
automatic top-N pruning.

```bash
# Start the monitor (default port 8051)
python3 html_pages/gpu_monitor.py --host 0.0.0.0 --port 9999

# Or use the launcher wrapper
html_pages/start_gpu_monitor.sh
```

**Options:**
- `--port`: Web server port (default: 8051)
- `--host`: Bind address (default: `127.0.0.1`)
- `--main-interval`: Main collection interval in ms for CPU + GPU mem/util/temp + network (default: 200, i.e. 5/s)
- `--disk-interval`: Disk I/O collection interval in ms (default: 1000, i.e. 1/s)
- `--window`: Rolling window in seconds (default: 900 = 15 min)
- `--top-n`: Show top N processes per chart, rest grouped as "Other" (default: 12)

**Features:**
- Per-GPU charts: memory by process, %-utilization, temperature, PCIe TX/RX
- System charts: CPU by process, disk I/O (aggregate), network I/O
- Progressive data coarsening on the client (keeps UI responsive at long time ranges)
- Adaptive WebSocket push rate based on client zoom level (75ms at 1m, up to 1s at 15m)
- Saves/restores metrics cache to `~/.cache/gpu_monitor/` on shutdown/startup

#### `resource_monitor.py`
Periodic system + GPU sampler that writes all data to a SQLite database. Tracks CPU, memory, load,
network/disk IO rates, GPU utilization, and top-offender processes. Designed to run as a persistent
background service.

```bash
# Start monitoring (writes to default SQLite DB)
python3 resource_monitor.py

# Custom DB path and interval
python3 resource_monitor.py --db-path /tmp/monitor.db --interval-seconds 5

# Single snapshot
python3 resource_monitor.py --once

# Include Docker container stats and GitHub rate limit tracking
python3 resource_monitor.py --docker-stats --gh-rate-limit --disk-usage
```

**Options:**
- `--db-path`: SQLite database path (default: `~/.cache/dynamo-utils/resource_monitor.db`)
- `--interval-seconds`: Sample interval (default: 10)
- `--once`: Take one sample and exit
- `--top-k`: Number of top processes to track (default: 10)
- `--docker-stats`: Include Docker container resource stats
- `--gh-rate-limit`: Track GitHub API rate limit usage
- `--disk-usage`: Track disk usage for specified paths
- `--net-top`: Enable per-process network bandwidth attribution

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
./backup.sh --input-path ~/dynamo --output-path /mnt/sda/keivenc/dynamo

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
- Log file: `$DYNAMO_HOME/logs/YYYY-MM-DD/backup.log`
- Captures all backup operations

#### Cron Setup

```bash
# Backup every 6 minutes
*/6 * * * * DYNAMO_HOME=$HOME/dynamo $HOME/dynamo/dynamo-utils.PRODUCTION/cron_log.sh backup $HOME/dynamo/dynamo-utils.PRODUCTION/backup.sh --input-path $HOME/dynamo --output-path /mnt/sda/keivenc/dynamo
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

## Operations & Process Management

### `kill_dynamo_processes.sh`
Safely kills dynamo/vllm/sglang inference processes without accidentally killing IDE processes,
pytest runners, or other unrelated processes. Uses narrow process matching patterns.

```bash
# Kill inference server processes
./kill_dynamo_processes.sh

# Force kill (SIGKILL)
./kill_dynamo_processes.sh --force

# Also kill etcd and nats
./kill_dynamo_processes.sh --all

# Kill specific ports
./kill_dynamo_processes.sh --ports 8000,8081
```

### `await_output.sh`
Run a command and exit immediately when a sentinel string appears in the output (or on timeout).
Replaces the inefficient `sleep N && grep` pattern. All output is logged to a file regardless.

```bash
# Wait for model to load (exits as soon as "model loaded" appears, or after 90s)
./await_output.sh -t 90 -s "model loaded" -- python serve.py

# Multiple sentinels (first match wins)
./await_output.sh -t 60 -s "listening on" -s "READY" -s "model loaded" -- ./start_server.sh

# Watch a background PID (exit when build finishes)
./container/build.sh > /tmp/build.log 2>&1 &
./await_output.sh -t 1200 -p $! -- tail -f /tmp/build.log
```

**Options:**
- `-t, --timeout SECONDS`: Maximum wait time (required)
- `-s, --sentinel STRING`: String to watch for (repeatable)
- `-p, --pid PID`: Exit when this PID exits
- `-l, --log FILE`: Log file path (default: `/tmp/await_output.log`)
- `-q, --quiet`: Suppress terminal output (log only)

### `clean_system.sh`
Orchestrates cleanup of old Docker images, log directories, and optionally VS Code containers.
Delegates to specialized sub-scripts (`clean_old_local_dynamo_images.sh`, `clean_log.sh`).

```bash
# Default cleanup
./clean_system.sh

# Keep 14 days of logs, retain 5 dynamo images
./clean_system.sh --keep-days 14 --retain-dynamo-images 5

# Preview what would be deleted
./clean_system.sh --dry-run

# Also clean VS Code dev containers
./clean_system.sh --clean-vsc
```

### `clean_log.sh`
Deletes old `YYYY-MM-DD/` log directories beyond a retention window. Only deletes directories
matching the date regex; never deletes today's directory.

```bash
# Delete logs older than 30 days (default)
./clean_log.sh

# Keep 14 days
./clean_log.sh --keep-days 14

# Preview
./clean_log.sh --dry-run
```

---

## Developer Tools

### `read_cursor_transcript.py`
Search, list, and continue Cursor agent transcripts from previous sessions.

```bash
# List recent transcripts
python3 read_cursor_transcript.py --list

# Show the most recent transcript
python3 read_cursor_transcript.py --latest

# Resume where you left off
python3 read_cursor_transcript.py --latest --continue

# Search by topic
python3 read_cursor_transcript.py --search "gpu monitor"

# Resume a specific transcript (prefix match OK)
python3 read_cursor_transcript.py --id 806f --continue
```

### `check_pr_status.py`
Show detailed check status breakdown for GitHub PRs, separating required from non-required checks.

```bash
python3 check_pr_status.py 1234
```

### `rerun_github_pr.py`
Monitor GitHub Actions workflows on PRs and automatically re-run failed jobs that appear to be
infrastructure failures (not code errors). Detects code-related patterns (SyntaxError, ImportError, etc.)
to avoid re-running genuine failures.

```bash
python3 rerun_github_pr.py --pr 1234
```

### `ddns.py`
Dynamic DNS updater for `*.dyn.nvidia.com` hostnames. Detects the active NIC and VPN interface,
then updates DNS records if the IP has changed.

---

## Infrastructure

### `aws-ecr-setup.sh`
AWS ECR access setup for the `ai-dynamo/dynamo` container registry. Handles AWS CLI authentication
(via nvsec browser login), Docker ECR login, and image listing.

```bash
# Login (AWS + Docker ECR, skips browser if creds still valid)
./aws-ecr-setup.sh

# List image tags (fast)
./aws-ecr-setup.sh --list-images

# Full metadata (size, dates)
./aws-ecr-setup.sh --describe-images

# Clear credentials
./aws-ecr-setup.sh --logout

# One-time install of AWS CLI + nvsec
./aws-ecr-setup.sh --install
```

### `recompress_backups.sh`
Re-compress existing `.tgz` backup archives while applying updated `.rsyncrules` exclusions,
reducing archive sizes by removing files that should have been excluded.

```bash
./recompress_backups.sh --backup-history /mnt/sda/keivenc/backup_history --rsyncrules ~/dynamo/.rsyncrules

# Preview
./recompress_backups.sh --backup-history /mnt/sda/keivenc/backup_history --rsyncrules ~/dynamo/.rsyncrules --dry-run
```

### `update_cron_tail.sh`
Extracts the last 135 lines of `cron.log` into `cron-tail.txt` for quick inspection without
opening the full log.

---

## Python Modules

### `common_github/`
GitHub API client module with ETag-based conditional requests, response caching, and batched
fetching. All dashboard scripts (`show_commit_history.py`, `show_local_branches.py`,
`show_remote_branches.py`) use this module. See the "GitHub API Optimizations" section above
for details on rate limit savings.

Key submodules:
- `api/commit_checks_cached.py`: Cached commit check-run queries
- `api/actions_runs_list_cached.py`: Cached workflow run listings
- `api/pulls_list_cached.py`: Cached PR list queries
- `cache_ttl_utils.py`: TTL-based cache expiration helpers

### `common_gitlab/`
GitLab API client module with caching, mirroring the `common_github` architecture. Supports
pipeline status, MR pipeline queries, pipeline job listing, and container registry image listing.

### `ci_log_errors/`
CI log categorization engine. Classifies GitHub Actions failure logs into categories
(e.g., `github-lfs-error`, `docker-timeout`, `rust-compile-error`) using regex rules and
snippet extraction.

```bash
# Run self-test against golden examples
python3 -m ci_log_errors --self-test-examples

# Categorize a log file
python3 -m ci_log_errors categorize /path/to/log.txt
```

### `cache/`
Cache subsystem providing base classes and helpers for all caching in the project:
- `cache_base.py`: Base cache class with file-backed JSON storage
- `cache_duration.py`: Duration/timing cache for build and test results
- `cache_job_log.py`: CI job log caching

### `reports/`
Repository and CI analytics scripts:
- `analyze_repo.py`: Repository structure analysis
- `pr_ci_report.py`: PR & CI statistics report generator
- `contributor_stats.py`: Contributor commit and line-change statistics

### `common_build_report.py`
Typed dataclasses for build report JSON serialization. Defines the schema for report files
written by `build_images.py` and consumed by `show_commit_history.py`.

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
