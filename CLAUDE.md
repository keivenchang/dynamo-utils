# NVIDIA Dynamo Projects - Instructions

> **Note**: For coding conventions, style guidelines, and development practices, refer to `.cursorrules` in this directory.

---

# PART 1: OPERATIONAL PROCEDURES

## Meta Instructions

### Remember This
When the user says "remember this", document it in this CLAUDE.md file.

### Commit Policy
**NEVER auto-commit changes without explicit user approval.** Always wait for the user to explicitly request a commit before running git commit commands.

### Permission Policy
- No need to ask permission when running any read-only operations, such as `echo`, `cat`, `tail`, `head`, `grep`, `egrep`, `ls`, `uname`, `soak_fe.py` or `curl` commands
- When executing `docker exec ... bash -c "<command> ..." and the <command> is one of the read-only operations, just execute the command, no need to ask for permission.

## Environment Setup

### All Projects Overview

The `~/nvidia/` directory contains multiple projects:

- **dynamo1, dynamo2, dynamo3, dynamo4**: Multiple working branches of the Dynamo repository
- **dynamo_ci**: Main CI/testing repository for Dynamo
- **dynamo-utils**: Build automation scripts and utilities (this directory)

### Docker Container Naming

When running `docker ps`, VS Code/Cursor dev container images follow this naming pattern:

- `vsc-dynamo1-*` → `~/nvidia/dynamo1`
- `vsc-dynamo2-*` → `~/nvidia/dynamo2`
- `vsc-dynamo3-*` → `~/nvidia/dynamo3`
- `vsc-dynamo4-*` → `~/nvidia/dynamo4`
- `vsc-dynamo_ci-*` → `~/nvidia/dynamo_ci`

The `vsc-` prefix indicates VS Code/Cursor dev containers, and the part after it matches the directory name.

**Note**: Container names (like `epic_satoshi`, `distracted_shockley`) are transient and should not be documented.

### Host-Container Directory Mapping

The `dynamo-utils` directory on the host is mapped to the container's `shared` directory:

- **Host**: `~/nvidia/dynamo-utils/`
- **Container**: `~/dynamo/shared/`

Example: A file at `~/nvidia/dynamo-utils/notes/metrics-vllm.log` on the host appears at `~/dynamo/shared/notes/metrics-vllm.log` inside the container.

### Backup File Convention

When creating backup files, use the naming format: `<filename>.<YYYY-MM-DD>.bak`

Example:
```bash
cp dynamo_docker_builder.py dynamo_docker_builder.py.2025-10-18.bak
```

This convention:
- Makes backup dates immediately visible
- Allows multiple backups from different dates
- Is automatically ignored by .gitignore (*.bak pattern)

## Running Inference Servers

### Collecting Prometheus Metrics from Inference Server

This procedure collects Prometheus metrics from a running inference server inside a Docker container. Repeat this process for different frameworks (VLLM, SGLANG, TRTLLM) to compare metrics.

**Prerequisites**:
- Docker container running
- Container must have dynamo project mounted

**Steps**:

1. **Start inference server in background**:
```bash
docker exec <container_name> bash -c "cd ~/dynamo && nohup shared/inference.sh > /tmp/inference-<framework>.log 2>&1 &"
```

2. **Monitor inference server startup** (wait ~30 seconds):
```bash
docker exec <container_name> bash -c "tail -20 /tmp/inference-<framework>.log"
```
Look for: "added model model_name=..." indicating the model is ready.

3. **Run soak test** to generate some load:
```bash
docker exec <container_name> bash -c "cd ~/dynamo && python3 shared/soak_fe.py --max-tokens 1000 --requests_per_worker 5"
```
Wait for: "All requests completed successfully."

4. **Collect metrics and save to shared directory**:
```bash
docker exec <container_name> bash -c "curl -s localhost:8081/metrics > ~/dynamo/shared/notes/metrics-<framework>.log"
```

**Example for VLLM**:
```bash
# Find container name
docker ps --format "table {{.Names}}\t{{.Image}}" | grep dynamo1

# Start inference (example container: epic_satoshi)
docker exec <container_name> bash -c "cd ~/dynamo && nohup shared/inference.sh > /tmp/inference-vllm.log 2>&1 &"

# Wait and check log
sleep 30 && docker exec <container_name> bash -c "tail -20 /tmp/inference-vllm.log"

# Run soak test
docker exec <container_name> bash -c "cd ~/dynamo && python3 shared/soak_fe.py --max-tokens 1000 --requests_per_worker 5"

# Collect metrics
docker exec <container_name> bash -c "mkdir -p ~/dynamo/shared/notes && curl -s localhost:8081/metrics > ~/dynamo/shared/notes/metrics-vllm.log"
```

**Output**:
- Metrics saved to: `~/nvidia/dynamo-utils/notes/metrics-<framework>.log` (on host)
- Typical size: ~200-600 lines

**Repeat for other frameworks**:
- SGLANG: Save to `metrics-sglang.log`
- TRTLLM: Save to `metrics-trtllm.log`

**Cleanup after collection**:
```bash
# Kill inference processes
docker exec <container_name> bash -c "ps aux | grep -E '(inference|sglang|vllm|trtllm|dynamo)' | grep -v grep | awk '{print \$2}' | xargs -r kill -9"

# Verify ports freed
docker exec <container_name> bash -c "ss -tlnp | grep -E '(8000|8081)' || echo 'Ports freed'"
```

### Prometheus Metrics Comparison: VLLM vs SGLANG vs TRTLLM

**Collection Date**: 2025-10-21 (Updated with correct test parameters)

**Test Parameters**: `--max-tokens 1000 --requests_per_worker 5`

#### Summary

Successfully collected and compared Prometheus metrics from three inference frameworks running on the same model (Qwen/Qwen3-0.6B) with 5 requests × 1000 tokens each:

| Framework | Total Lines | Unique Metrics | Dynamo Metrics | Request Duration | TTFT | Throughput |
|-----------|-------------|----------------|----------------|------------------|------|------------|
| **TRTLLM** | 220 | 5 (trtllm:*) | 53 lines | **2.19s** | **10.3ms** | **456 tok/s** |
| **SGLANG** | 224 | 25 (sglang:*) | 35 lines | 2.77s | N/A | 361 tok/s |
| VLLM | 527 | 25 (vllm:*) | 53 lines | 5.94s | 20.6ms | 168 tok/s |

**Key Result**: TRTLLM is 2.7x faster than VLLM and 1.3x faster than SGLANG for this workload.

#### Key Findings

**1. Metrics Structure Differences**

**VLLM** provides detailed metrics:
- **25 unique metrics** (`vllm:*` - excluding `_created` metadata)
- Detailed token-level histograms (prompt, generation, iteration)
- Multiple latency breakdowns (TTFT, inter-token, per-output-token, queue time, prefill time, decode time)
- Cache configuration: KV cache usage %, prefix cache hits/queries
- Comprehensive observability for debugging

**SGLANG** provides moderate metrics:
- **25 unique metrics** (`sglang:*`)
- Focus on queue management (6 different queue types tracked)
- Per-stage request latency, KV transfer speed/latency
- Cache hit rate, token usage, utilization, speculative decoding metrics
- Good balance of observability

**TRTLLM** provides minimal focused metrics:
- **5 unique metrics** (`trtllm:*` - excluding `_created` metadata)
- Core performance only: E2E latency, TTFT, time-per-output-token, queue time, success counter
- Extremely lightweight instrumentation - 5x fewer metrics than VLLM/SGLANG
- Lowest overhead among all frameworks

**Dynamo common metrics** (shared across all):
- `dynamo_component_*` metrics (uptime, NATS client, KV stats, request handling)
- 53 lines for VLLM/TRTLLM, 35 lines for SGLANG

**2. Performance Characteristics**

**TRTLLM** (FASTEST):
- Time to first token (TTFT): 10.3ms avg (2.0x faster than VLLM)
- Time per output token: 2.26ms avg (2.6x faster than VLLM)
- Request processing: 2.19s avg for 1000 tokens (2.7x faster than VLLM)
- Throughput: 456 tokens/sec (5000 tokens / 10.97s)

**SGLANG** (MIDDLE):
- Request processing: 2.77s avg for 1000 tokens (2.1x faster than VLLM)
- Reported generation throughput: 460 tokens/sec
- Actual throughput: 361 tokens/sec (5000 tokens / 13.84s)
- Note: Reported metric (460) vs actual (361) shows 27% discrepancy

**VLLM** (SLOWEST):
- Time to first token (TTFT): 20.6ms avg
- Time per output token: ~5.94ms avg
- Request processing: 5.94s avg for 1000 tokens
- Throughput: 168 tokens/sec (5000 tokens / 29.70s)

**3. Token Processing**

All frameworks processed:
- **Prompt tokens**: 22 per request (110 total across 5 requests)
- **Generation tokens**: 1000 per request (5000 total across 5 requests)
- **Total tokens**: 1022 per request (5110 total)

**VLLM**:
- KV cache blocks available: 4,834
- GPU memory utilization: 20%
- Request finished due to: `length` limit (max_tokens=1000)

**SGLANG**:
- KV cache: Different architecture from VLLM/TRTLLM

**4. Key Observations**

- **Performance ranking**: TRTLLM (456 tok/s) > SGLANG (361 tok/s) > VLLM (168 tok/s)
- **Speed ratios**: TRTLLM is 2.7x faster than VLLM, 1.3x faster than SGLANG
- **TTFT comparison**: TRTLLM 10.3ms < VLLM 20.6ms (SGLANG: no TTFT data)
- **Metrics count**: VLLM and SGLANG both have 25 metrics, TRTLLM has only 5 (5x fewer)
- **Observability vs Performance**: TRTLLM achieves 2.7x better performance with 5x fewer metrics
- **SGLANG throughput**: Reported metric (460 tok/s) is 27% higher than actual (361 tok/s)

**5. Common Dynamo Metrics**

All frameworks showed consistent Dynamo component behavior:
- NATS client connected (state=1)
- Backend uptime: TRTLLM 119s, VLLM 278s, SGLANG 79s
- Request bytes: 1,253 bytes (identical across all frameworks)
- Response bytes: ~18,200 bytes (TRTLLM 18,200, VLLM 18,129, SGLANG 18,219)
- Active endpoints: 2 (VLLM, TRTLLM), 1 (SGLANG)

#### Recommendations

**Use TRTLLM when**:
- **Maximum performance is critical** (2.7x faster than VLLM)
- Ultra-low latency needed (10.3ms TTFT)
- High throughput required (456 tokens/sec)
- Minimal observability overhead acceptable
- Production workloads prioritizing speed

**Use SGLANG when**:
- Good balance of speed and features (2.1x faster than VLLM)
- Moderate throughput needs (361 tokens/sec)
- Alternative to TRTLLM when feature set is needed

**Use VLLM when**:
- Detailed observability and debugging required
- Most comprehensive metrics suite (527 lines)
- Development and troubleshooting scenarios
- Acceptable performance for lower-throughput workloads

#### Performance Summary

**Winner: TRTLLM**
- 2.7x faster than VLLM (2.19s vs 5.94s per 1000 tokens)
- 2.0x faster TTFT (10.3ms vs 20.6ms)
- 2.7x higher throughput (456 vs 168 tokens/sec)
- Best choice for production performance

**Runner-up: SGLANG**
- 2.1x faster than VLLM (2.77s vs 5.94s per 1000 tokens)
- Good middle ground between speed and features
- 361 tokens/sec actual throughput

**Third: VLLM**
- Excellent observability with most detailed metrics
- Best for development and debugging
- 168 tokens/sec throughput

## Testing Commands

> **Note**: General pytest guidelines (including `--basetemp=/tmp/pytest_temp`) are in `.cursorrules`.

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

## Important Reminders

- Never commit sensitive information (credentials, tokens, etc.)
- Always test changes locally before pushing
- Use meaningful commit messages
- Review diffs before committing
- Consider backward compatibility when making changes

═══════════════════════════════════════════════════════════════════════════════
═══════════════════════════════════════════════════════════════════════════════
═══════════════════════════════════════════════════════════════════════════════

# PART 2: PROJECT DOCUMENTATION (dynamo-utils/*)

## dynamo-utils Project Overview

This directory contains utilities and automation scripts for the Dynamo inference framework:

- **dynamo_docker_builder.py**: Automated Docker build and test pipeline system
- **common.py**: Shared utilities (GitUtils, GitHubAPIClient, DockerUtils, etc.)
- **show_dynamo_branches.py**: Branch status checker with GitHub PR integration
- **run_docker_builder.sh**: Shell wrapper for the Docker builder
- **update_html_pages.sh**: Automated cron script for HTML page updates

> **Note**: Git integration conventions (commit message format, PR number parsing, GitHub URLs) are documented in `.cursorrules`.

> **Note**: Common code patterns (Git commit parsing, terminal width detection, etc.) are documented in `.cursorrules`.

---

## dynamo_docker_builder.py

Automated Docker build and test pipeline system that builds and tests multiple inference frameworks (VLLM, SGLANG, TRTLLM) across different target environments (base, dev, local-dev), generates HTML reports, and sends email notifications.

### HTML Report Generation
- Generate two versions when email is requested:
  - **File version**: relative paths (just filenames)
  - **Email version**: absolute URLs with hostname
- Use helper function `get_log_url()` to abstract URL generation
- Pass `use_absolute_urls` flag to control URL format
- Default hostname: `keivenc-linux`
- Default HTML path: `/nvidia/dynamo_ci/logs`

### Log File Paths
- Logs stored in: `~/nvidia/dynamo_ci/logs/YYYY-MM-DD/`
- Log filename format: `YYYY-MM-DD.{sha_short}.{framework}-{target}-{type}.log`
- HTML report: `YYYY-MM-DD.{sha_short}.report.html`
- Always use date subdirectories
- Use `log_dir.parent` to get root logs directory

### Process Management
- Use lock files to prevent concurrent runs
- Lock file: `.dynamo_docker_builder.py.lock`
- Store PID in lock file
- Check if process is still running before acquiring lock
- Clean up stale locks automatically

### Email Notifications
- Use SMTP (localhost:25) for emails
- Email subject format: `[DynamoDockerBuilder] {status} - {sha_short}`
- HTML email body with clickable links (absolute URLs)
- Plain text fallback for email clients

### Important Notes
- Always test with `--force-run` to bypass checks during development
- Use `--dry-run` to see what would happen without executing
- Check process locks before making changes to lock file handling
- HTML reports must work standalone (relative paths for log links)
- Email HTML needs absolute URLs for external viewing
- All GitHub links should be clickable in HTML output
- Commit SHAs in headers should have white underline styling for visibility

### Commit History Feature

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

### Image Size Population

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

## update_html_pages.sh

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

## common.py

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

## show_dynamo_branches.py

Branch status checker that displays information about dynamo* repository branches with GitHub PR integration.

**Features**:
- Scans all dynamo* directories for branch information
- Queries GitHub API for PR status using `common.GitHubAPIClient`
- Supports both terminal and HTML output modes
- Parallel data gathering for improved performance
- Automatic GitHub token detection (env var, gh CLI config)
