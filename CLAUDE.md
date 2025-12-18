# NVIDIA Dynamo Projects - Instructions

> **Note**: For coding conventions, style guidelines, and development practices, refer to `dynamo-utils/.cursorrules`. Periodically review this file so you don't forget the details in .cursorrules.

---

# PART 1: OPERATIONAL PROCEDURES

## Meta Instructions

### Remember This
When the user says "remember this" or "remember how to do this" or something similar, document it in this CLAUDE.md file.

### New Project Protocol

When the user says "new project", always:
1. Re-read `~/nvidia/dynamo-utils/.cursorrules`
2. Re-read `~/nvidia/dynamo-utils/CLAUDE.md`

These are the canonical project instructions. Do NOT read .cursorrules or CLAUDE.md from other project directories (dynamo1, dynamo2, dynamo3, dynamo_ci, etc.) unless explicitly instructed.

### Commit Policy
**NEVER auto-commit changes without explicit user approval.** Always wait for the user to explicitly request a commit before running git commit commands.

### Permission Policy
- No need to ask permission when running any read-only operations, such as `echo`, `cat`, `tail`, `head`, `grep`, `egrep`, `ls`, `uname`, `soak_fe.py` or `curl` commands
- When executing `docker exec ... bash -c "<command> ..." and the <command> is one of the read-only operations, just execute the command, no need to ask for permission.

### Python Virtual Environment
**Always use the dynamo-utils venv**: Before any Python operations in dynamo-utils, activate the virtual environment:
```bash
source dynamo-utils/.venv/bin/activate
```

The venv includes:
- pre-commit (for git hooks, always run this git commit)
- All Python dependencies for dynamo-utils scripts

**Location**: `nvidia/dynamo-utils/.venv/`

## Environment Setup

### All Projects Overview

The `nvidia/` directory contains multiple projects:

- **dynamo1, dynamo2, dynamo3, dynamo4**: Multiple working branches of the Dynamo repository
- **dynamo_ci**: Main CI/testing repository for Dynamo
- **dynamo-utils**: Build automation scripts and utilities (this directory)

### Docker Container Naming

When running `docker ps`, VS Code/Cursor dev container images follow this naming pattern:

- `vsc-dynamo1-*` → `nvidia/dynamo1`
- `vsc-dynamo2-*` → `nvidia/dynamo2`
- `vsc-dynamo3-*` → `nvidia/dynamo3`
- `vsc-dynamo4-*` → `nvidia/dynamo4`
- `vsc-dynamo_ci-*` → `nvidia/dynamo_ci`

The `vsc-` prefix indicates VS Code/Cursor dev containers, and the part after it matches the directory name.

**Note**: Container names (like `epic_satoshi`, `distracted_shockley`) are transient and should not be documented.

### Host-Container Directory Mapping

The `dynamo-utils` directory on the host is mapped to the container's `_` directory:

- **Host**: `~/nvidia/dynamo-utils/`
- **Container**: `~/dynamo/_/`

Example: A file at `~/nvidia/dynamo-utils/notes/metrics-vllm.log` on the host appears at `~/dynamo/_/notes/metrics-vllm.log` inside the container.

### Backup File Convention

When creating backup files, use the naming format: `<filename>.<YYYY-MM-DD>.bak`

Example:
```bash
cp container/build_images.py container/build_images.py.2025-10-18.bak
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
docker exec <container_name> bash -c "cd ~/dynamo && nohup _/inference.sh > /tmp/inference-<framework>.log 2>&1 &"
```

2. **Monitor inference server startup** (wait ~30 seconds):
```bash
docker exec <container_name> bash -c "tail -20 /tmp/inference-<framework>.log"
```
Look for: "added model model_name=..." indicating the model is ready.

3. **Run soak test** to generate some load:
```bash
docker exec <container_name> bash -c "cd ~/dynamo && python3 _/soak_fe.py --max-tokens 1000 --requests_per_worker 5"
```
Wait for: "All requests completed successfully."

4. **Collect metrics and save to _ directory**:
```bash
docker exec <container_name> bash -c "curl -s localhost:8081/metrics > ~/dynamo/_/notes/metrics-<framework>.log"
```

**Example for VLLM**:
```bash
# Find container name
docker ps --format "table {{.Names}}\t{{.Image}}" | grep dynamo1

# Start inference (example container: epic_satoshi)
docker exec <container_name> bash -c "cd ~/dynamo && nohup _/inference.sh > /tmp/inference-vllm.log 2>&1 &"

# Wait and check log
sleep 30 && docker exec <container_name> bash -c "tail -20 /tmp/inference-vllm.log"

# Run soak test
docker exec <container_name> bash -c "cd ~/dynamo && python3 _/soak_fe.py --max-tokens 1000 --requests_per_worker 5"

# Collect metrics
docker exec <container_name> bash -c "mkdir -p ~/dynamo/_/notes && curl -s localhost:8081/metrics > ~/dynamo/_/notes/metrics-vllm.log"
```

**Output**:
- Metrics saved to: `nvidia/dynamo-utils/notes/metrics-<framework>.log` (on host)
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

### Running Pytest with WORKSPACE_DIR

**IMPORTANT**: When using `WORKSPACE_DIR` environment variable for pytest tests, always run pytest from `$WORKSPACE_DIR` instead of `/workspace`:

**Correct** ✅:
```bash
docker exec -u root <container> bash -c "export WORKSPACE_DIR=~/dynamo && cd ~/dynamo && pytest -xvs tests/serve/test_vllm.py::test_serve_deployment -k aggregated --basetemp=/tmp/pytest_test"
```

**Incorrect** ❌:
```bash
docker exec -u root <container> bash -c "export WORKSPACE_DIR=/home/<user>/dynamo && cd /workspace && pytest ..."
```

**Why this matters**:
- pytest writes cache files (`.pytest_cache`) to the **current working directory**, not based on environment variables
- When running from `/workspace`, pytest tries to write to `/workspace/.pytest_cache` (permission denied for non-root)
- When running from `$WORKSPACE_DIR`, pytest writes to `$WORKSPACE_DIR/.pytest_cache` (works correctly)
- The test code uses `WORKSPACE_DIR` to find scripts/models, but pytest's cache location depends on `pwd`

**Solution**: Always `cd $WORKSPACE_DIR` before running pytest when using WORKSPACE_DIR environment variable.

**Alternative**: Use `--cache-dir=/tmp/pytest_cache` flag if you must run from a different directory:
```bash
pytest --cache-dir=/tmp/pytest_cache --basetemp=/tmp/pytest_test ...
```

### Docker Builder Tests

**Quick Test (Single Framework)**:
```bash
python3 container/build_images.py --sanity-check-only --framework sglang --force-run
```

**Parallel Build with Skip**:
```bash
python3 container/build_images.py --skip-build-if-image-exists --parallel --force-run
```

**Full Build**:
```bash
python3 container/build_images.py --parallel --force-run
```

### Go Operator Linting

Before committing Go operator changes, run the linter to verify formatting and code quality (same as CI).

**Command to run**:
```bash
cd /path/to/dynamo/deploy/cloud/operator
docker build --target linter --progress=plain .
```

This runs golangci-lint which includes:
- gofmt (Go formatting)
- Multiple code quality checks
- Same checks that run in CI

**Expected result**: Build completes successfully with no linting errors

**When to run**: Before committing any changes to `deploy/cloud/operator/*.go` files

### Complete Pre-Merge CI Checks

**See `~/nvidia/dynamo-utils/.cursorrules`** for the complete list of pre-merge CI checks including:
- Rust Format Check (cargo fmt)
- Rust Clippy Checks (unused imports, warnings)
- Rust Tests (unit, doc, integration)
- Pre-commit Hooks (ruff, mypy, YAML/JSON validation)
- Copyright Headers
- Cargo Deny (license checks)
- Quick Pre-Commit Checklist

### Documentation Build Test

Test Sphinx documentation build (same as CI) to verify no warnings/errors.

**IMPORTANT**: Since Docker commands cannot be run inside Cursor, you must:
1. Tell the user to run the docker build command in their external terminal
2. Wait for the user to report back the results
3. If build fails, analyze the error and fix the documentation files

**Tell user to run** (in external terminal):
```bash
cd /path/to/dynamo/repo
docker build -t docs-builder -f container/Dockerfile.docs .
```

This builds documentation using:
- **Container**: `container/Dockerfile.docs`
- **Script**: `docs/generate_docs.py`
- **Steps**: `make clean` → preprocess links → `make html` with `-W` (warnings as errors)
- **Output**: `docs/build/html/` (inside container)

**Expected result**: `build succeeded` with no warnings

**Common build failures**:

1. **Invalid JSON in code blocks**:
   - Error: `WARNING: Lexing literal_block as "json" resulted in an error`
   - Cause: Code block marked as ` ```json` contains invalid JSON (ellipsis `...`, comments, etc.)
   - Fix: Either remove invalid syntax OR change to ` ```text`
   - Example: `docs/observability/logging.md:215` had `...` on line 217

2. **Missing toctree entries**:
   - Error: `WARNING: document isn't included in any toctree`
   - Fix: Add to `docs/index.rst` or `docs/hidden_toctree.rst`

3. **Missing images**:
   - Error: `WARNING: image file not found`
   - Fix: Check image path is correct

4. **Broken relative links**:
   - Error: Various link warnings
   - Fix: Verify link paths

**Extract built HTML** (optional, tell user to run):
```bash
docker create --name docs-container docs-builder
docker cp docs-container:/workspace/dynamo/docs/build/html ./dynamo-docs/
docker rm docs-container
cd dynamo-docs
python3 -m http.server 8000  # View at http://localhost:8000
```

**Quick check for warnings without full build**:
```bash
# Look for common issues in recent changes
git diff --name-only HEAD~5 | grep '\.md$' | xargs grep -l 'http.*github.com.*dynamo'  # Check links
find docs -name '*.md' -newer docs/build/html 2>/dev/null  # Find modified docs
```

## Important Reminders

- **NEVER delete `nvidia/dynamo_latest/index.html`** - This is the production HTML page for commit history
- Never commit sensitive information (credentials, tokens, etc.)
- Always test changes locally before pushing
- Use meaningful commit messages
- Review diffs before committing
- Consider backward compatibility when making changes

---

## Additional Documentation

For detailed information about Python utilities and scripts in this directory, see:
- **README.md**: Comprehensive documentation for all Python scripts and tools
- **.cursorrules**: Coding conventions, style guidelines, and development practices

---

## GitHub API Access

**GitHub credentials location**: `~/.config/gh/hosts.yml`

**Environment variable**: `GITHUB_TOKEN` is set in `~/.bashrc` and automatically loaded on login.

To use GitHub API with curl:
```bash
# Using environment variable (preferred)
curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user

# Or read from config file
TOKEN=$(grep oauth_token ~/.config/gh/hosts.yml | head -1 | awk '{print $2}')
curl -H "Authorization: token $TOKEN" https://api.github.com/repos/ai-dynamo/dynamo/...
```

For Python scripts, use `GitHubAPIClient` from `common.py` which automatically reads credentials from:
1. Provided token argument
2. `GITHUB_TOKEN` environment variable (set in ~/.bashrc)
3. `~/.config/gh/hosts.yml` (GitHub CLI config)

## Re-running Failed GitHub Actions

**Quick method using GitHub CLI** (recommended):

```bash
# Step 1: View PR checks to get workflow run ID
gh pr checks <PR_NUMBER> --repo ai-dynamo/dynamo

# Step 2: Re-run failed jobs using the run ID from check URLs
gh run rerun <RUN_ID> --repo ai-dynamo/dynamo --failed

# Step 3: Verify the re-run started
gh run view <RUN_ID> --repo ai-dynamo/dynamo --json status,conclusion,url
```

**Example workflow**:
```bash
# For PR #3688
gh pr checks 3688 --repo ai-dynamo/dynamo
# Output shows: https://github.com/ai-dynamo/dynamo/actions/runs/18690241847/...
# Extract run ID: 18690241847

gh run rerun 18690241847 --repo ai-dynamo/dynamo --failed
# ✅ Successfully triggered re-run

gh run view 18690241847 --repo ai-dynamo/dynamo --json status,conclusion,url
# Shows: {"conclusion":"","status":"queued",...}
```

**Alternative: Using Python API**:

```python
import sys
sys.path.insert(0, 'nvidia/dynamo-utils')
from common import GitHubAPIClient
import requests

client = GitHubAPIClient()

# Re-run failed jobs for a specific workflow run
run_id = 18672015489  # Get from workflow URL or API
rerun_url = f"{client.base_url}/repos/ai-dynamo/dynamo/actions/runs/{run_id}/rerun-failed-jobs"

response = requests.post(rerun_url, headers=client.headers)
if response.status_code == 201:
    print("✅ Successfully triggered re-run of failed jobs")
```

**When to use**:
- Transient infrastructure failures (ARM64 runner issues, network timeouts)
- Failed checks that passed locally
- When all code tests pass but build/deploy steps fail

**Finding workflow run ID**:
- From `gh pr checks <PR_NUMBER>`: Extract from check URLs (e.g., `/runs/18690241847/`)
- From Actions URL: `https://github.com/ai-dynamo/dynamo/actions/runs/RUN_ID`
- From PR check status: Use GitHubAPIClient to get commit check-runs, extract run ID from `html_url`
