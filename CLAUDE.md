CLAUDE.md - NVIDIA Dynamo Projects (Operational Procedures)

Operational playbooks and environment conventions for the `~/nvidia` workspace.
For coding conventions, style guidelines, and development practices, refer to `dynamo-utils/.cursorrules`.

=============================================================================
TABLE OF CONTENTS
=============================================================================
1. META INSTRUCTIONS AND POLICIES
  1.1 Remember This
  1.2 New Project Protocol
  1.3 Commit Policy
  1.4 Permission Policy
  1.5 Python Virtual Environment (Host vs Container)

2. ENVIRONMENT SETUP
  2.1 All Projects Overview
  2.2 Docker Container Naming
  2.3 Host-Container Directory Mapping
  2.4 Backup File Convention

3. OPERATIONAL PROCEDURES
  3.1 Running Inference Servers (Docker)
    3.1.1 Collecting Prometheus Metrics (VLLM/SGLANG/TRTLLM)

4. TESTING AND CI-RELATED COMMANDS
  4.1 Pytest with WORKSPACE_DIR (Docker exec)
  4.2 Docker Builder Tests
  4.3 Go Operator Linting
  4.4 Documentation Build Test
  4.5 Documentation Link Check
  4.6 Analyzing CI Failures (log-grepping workflow)
  4.7 GitHub CI job dependencies (needs:)

5. GITHUB OPERATIONS
  5.1 GitHub API Access
  5.2 Re-running Failed GitHub Actions

6. REMINDERS AND POINTERS
  6.1 Important Reminders
  6.2 Additional Documentation

=============================================================================
1. META INSTRUCTIONS AND POLICIES
=============================================================================

## 1.1 Remember This
When the user says "remember this" or "remember how to do this", document it in this `CLAUDE.md`.

## 1.2 New Project Protocol
When the user says "new project", always:
1. Re-read `~/nvidia/dynamo-utils/.cursorrules`
2. Re-read `~/nvidia/dynamo-utils/CLAUDE.md`

These are the canonical project instructions. Do NOT read `.cursorrules` or `CLAUDE.md` from other project directories (dynamo1, dynamo2, dynamo3, dynamo_ci, etc.) unless explicitly instructed.

## 1.3 Commit Policy
**NEVER auto-commit changes without explicit user approval.**
Always wait for the user to explicitly request a commit before running `git commit`.

## 1.4 Permission Policy
- Full permission to run read-only operations without asking: `wget`, `curl`, `echo`, `cat`, `tail`, `head`, `grep`, `egrep`, `ls`, `uname`.
- Always OK to `curl` without asking.
- When executing `docker exec ... bash -c "<command> ..."` and the `<command>` is read-only, just execute it (no need to ask).

## 1.5 Python Virtual Environment (Host vs Container)
If running on the host (not inside the dev container), activate the host venv before Python operations:
```bash
source ~/nvidia/venv/bin/activate
```

If running inside the dev container, the Python environment is already set up.

=============================================================================
2. ENVIRONMENT SETUP
=============================================================================

## 2.1 All Projects Overview
The `nvidia/` directory contains multiple projects:
- `dynamo1, dynamo2, dynamo3, dynamo4`: multiple working branches of the Dynamo repository
- `dynamo_ci`: main CI/testing repository for Dynamo
- `dynamo-utils`: build automation scripts and utilities

## 2.2 Docker Container Naming
When running `docker ps`, VS Code/Cursor dev container images follow this naming pattern:
- `vsc-dynamo1-*` → `nvidia/dynamo1`
- `vsc-dynamo2-*` → `nvidia/dynamo2`
- `vsc-dynamo3-*` → `nvidia/dynamo3`
- `vsc-dynamo4-*` → `nvidia/dynamo4`
- `vsc-dynamo_ci-*` → `nvidia/dynamo_ci`

The `vsc-` prefix indicates VS Code/Cursor dev containers, and the part after it matches the directory name.
Container names (like `epic_satoshi`, `distracted_shockley`) are transient and should not be documented.

## 2.3 Host-Container Directory Mapping
The `dynamo-utils` directory on the host is mapped into the container at `/workspace/_` (same repo, bind-mounted):
- Host: `~/nvidia/dynamo-utils/`
- Container: `/workspace/_/`

Example: `~/nvidia/dynamo-utils/notes/metrics-vllm.log` on the host appears at `/workspace/_/notes/metrics-vllm.log` inside the container.

## 2.4 Backup File Convention
When creating backups, use: `<filename>.<YYYY-MM-DD>.bak` (ignored by `.gitignore`).

Example:
```bash
cp container/build_images.py container/build_images.py.2025-10-18.bak
```

=============================================================================
3. OPERATIONAL PROCEDURES
=============================================================================

## 3.1 Running Inference Servers (Docker)

### 3.1.1 Collecting Prometheus Metrics (VLLM/SGLANG/TRTLLM)
This procedure collects Prometheus metrics from a running inference server inside a Docker container.
Repeat for different frameworks (VLLM, SGLANG, TRTLLM) to compare metrics.

Note: The metrics comparison table + engine/observability conclusions live in:
- `.cursorrules` → **3.5.3 Prometheus Metrics (VLLM vs SGLANG vs TRTLLM)**

Prerequisites:
- Docker container running
- Container must have dynamo project mounted

Steps:
1. Start inference server in background:
```bash
docker exec <container_name> bash -c "cd ~/dynamo && nohup _/inference.sh > /tmp/inference-<framework>.log 2>&1 &"
```

2. Monitor inference server startup (wait ~30 seconds):
```bash
docker exec <container_name> bash -c "tail -20 /tmp/inference-<framework>.log"
```
Look for: `added model model_name=...` indicating the model is ready.

3. Run soak test to generate some load:
```bash
docker exec <container_name> bash -c "cd ~/dynamo && python3 _/soak_fe.py --max-tokens 1000 --requests_per_worker 5"
```
Wait for: `All requests completed successfully.`

4. Collect metrics and save to the mounted `_` directory:
```bash
docker exec <container_name> bash -c "curl -s localhost:8081/metrics > /workspace/_/notes/metrics-<framework>.log"
```

Example for VLLM:
```bash
# Find container name
docker ps --format "table {{.Names}}\t{{.Image}}" | grep dynamo1

# Start inference
docker exec <container_name> bash -c "cd ~/dynamo && nohup _/inference.sh > /tmp/inference-vllm.log 2>&1 &"

# Wait and check log
sleep 30 && docker exec <container_name> bash -c "tail -20 /tmp/inference-vllm.log"

# Run soak test
docker exec <container_name> bash -c "cd ~/dynamo && python3 _/soak_fe.py --max-tokens 1000 --requests_per_worker 5"

# Collect metrics
docker exec <container_name> bash -c "mkdir -p /workspace/_/notes && curl -s localhost:8081/metrics > /workspace/_/notes/metrics-vllm.log"
```

Output:
- Metrics saved to: `~/nvidia/dynamo-utils/notes/metrics-<framework>.log` (on host)
- Typical size: ~200-600 lines

Repeat for other frameworks:
- SGLANG: save to `metrics-sglang.log`
- TRTLLM: save to `metrics-trtllm.log`

Cleanup after collection:
```bash
# Kill inference processes
docker exec <container_name> bash -c "ps aux | grep -E '(inference|sglang|vllm|trtllm|dynamo)' | grep -v grep | awk '{print \\$2}' | xargs -r kill -9"

# Verify ports freed
docker exec <container_name> bash -c "ss -tlnp | grep -E '(8000|8081)' || echo 'Ports freed'"
```

=============================================================================
4. TESTING AND CI-RELATED COMMANDS
=============================================================================

## 4.1 Pytest with WORKSPACE_DIR (Docker exec)

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

## 4.2 Docker Builder Tests

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

## 4.3 Go Operator Linting

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

## 4.4 Documentation Build Test

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

## 4.5 Documentation Link Check

Test markdown documentation links (same as CI) to verify no broken internal/external links.

**CI Workflow**: `.github/workflows/docs-link-check.yml` runs **two separate jobs**:

1. **`lychee` job** - External URL checker (lines 13-77)
   - Checks external URLs (HTTP/HTTPS links)
   - Uses [lychee](https://github.com/lycheeverse/lychee) tool
   - Offline mode for PRs (internal links only), full check for main branch
   - Caches results to avoid rate limits

2. **`broken-links-check` job** - Internal markdown link checker (lines 79-268)
   - Checks internal relative links between markdown files
   - Uses `.github/workflows/detect_broken_links.py` script
   - Validates file paths, symlinks, anchors
   - Creates GitHub annotations for broken links
   - **This is the most common check that fails**

**To verify broken link fixes locally**:
```bash
cd /path/to/dynamo/repo

# Run the same check as CI
python3 .github/workflows/detect_broken_links.py \
  --verbose \
  --format json \
  --check-symlinks \
  --output /tmp/broken-links-report.json \
  .

# Check exit code
echo $?  # 0 = pass, 1 = broken links found

# View summary
cat /tmp/broken-links-report.json | python3 -m json.tool | head -30
```

**Expected result**: Exit code 0, no broken links in summary

**Common broken link issues**:

1. **Stale relative paths after file moves**:
   - Error: `Broken link: [Pre-Deployment Checks](../../deploy/cloud/pre-deployment/README.md)`
   - Cause: Directory was moved/deleted (`deploy/cloud/` → `deploy/`)
   - Fix: Update relative path to match new location
   - Example: Change `../../deploy/cloud/pre-deployment/README.md` to `../../deploy/pre-deployment/README.md`

2. **Wrong relative path depth**:
   - Error: Link target doesn't exist
   - Cause: Incorrect `../` count in relative path
   - Fix: Count directory levels correctly from source to target

3. **Problematic symlinks**:
   - Warning: Suspicious symlink with many directory traversals
   - Cause: Symlink uses excessive `../../../../` patterns
   - Fix: Consider using direct file copy or shorter path

**Workflow integration**:
- Both checks run on every PR and push to main
- `broken-links-check` creates annotations in "Files Changed" tab
- Failed checks block merge (required status check)

**Difference from Sphinx build**:
- **Link check**: Validates link targets exist (files, URLs)
- **Sphinx build**: Validates documentation can be built as HTML
- Both are important and run independently in CI

## 4.6 Analyzing CI Failures (log-grepping workflow)

**CRITICAL: Always grep for errors when analyzing CI logs**

When investigating CI failures, immediately search for error patterns instead of reading the entire log:

**Command pattern**:
```bash
# ALWAYS save to temp file first, then grep repeatedly (see .cursorrules section 2.3.3)
curl -s "<CI_LOG_URL>" > /tmp/ci-log.txt
grep -E "ERROR|FAIL|Error|error:|failed|Failed" /tmp/ci-log.txt | head -50
```

**Common error patterns to search for**:

1. **Python/pytest errors**:
   ```bash
   grep -E "ERROR at setup|ModuleNotFoundError|ImportError|RuntimeError|AssertionError"
   ```

2. **Build errors**:
   ```bash
   grep -E "error:|ERROR:|fatal error|compilation terminated"
   ```

3. **Docker build errors**:
   ```bash
   grep -E "failed to|ERROR \[|RUN failed"
   ```

4. **Test failures**:
   ```bash
   grep -E "FAILED|ERRORS|tests.*ERROR"
   ```

**Example workflow**:
```bash
# 1. Get the CI log URL from gh pr checks or workflow run
# 2. Search for errors immediately
curl -s "$CI_LOG_URL" > /tmp/CI-12345.log && grep -B 5 -A 10 "ERROR" /tmp/CI-12345.log

# 3. If specific error found, get more context
grep -B 20 -A 20 "ModuleNotFoundError: No module named" /tmp/CI-12345.log
```

**Why this matters**:
- CI logs can be 50,000+ lines long
- Reading sequentially wastes time
- Errors usually have clear error messages
- Context lines (-B/-A) provide surrounding information

**Real example from dynamo PR #5050**:
```bash
# Instead of reading 50k lines, immediately found:
curl -s "$LOG" > /tmp/CI-12345.log && grep "ERROR at setup" /tmp/CI-12345.log
# Found: RuntimeError: Failed to get git HEAD commit
# Root cause: Missing DYNAMO_COMMIT_SHA environment variable
```

## 4.7 GitHub CI job dependencies (needs:)

Two common patterns:

- **Execution dependency (`needs:`)**: the job can’t start until the `needs` jobs finish. GitHub also exposes `needs.<job>.result`.
- **Aggregator / gating job (`needs:`)**: depends on multiple upstream jobs and fails based on their results (often named `*-status-check`).

Below is a tree view of the **pure `needs:` job dependency graph** in `dynamo_latest/.github/workflows/*` (guards/`if:` conditions omitted):

### `dynamo_latest/.github/workflows/container-validation-dynamo.yml`

```
dynamo-status-check
└── build-test — "Build and Test - dynamo"
    └── changed-files
```

### `dynamo_latest/.github/workflows/container-validation-backends.yml`

```
backend-status-check
├── operator — "operator (${{ matrix.platform.arch }})"
│   └── changed-files
├── vllm — "vllm (${{ matrix.platform.arch }})"
│   └── changed-files
├── sglang — "sglang (${{ matrix.platform.arch }})"
│   └── changed-files
└── trtllm — "trtllm (${{ matrix.platform.arch }})"
    └── changed-files

deploy-operator
├── changed-files
├── operator
├── vllm
├── sglang
└── trtllm

deploy-test-vllm — "deploy-test-vllm (${{ matrix.profile }})"
├── changed-files
├── deploy-operator
└── vllm

deploy-test-sglang — "deploy-test-sglang (${{ matrix.profile }})"
├── changed-files
├── deploy-operator
└── sglang

deploy-test-trtllm — "deploy-test-trtllm (${{ matrix.profile }})"
├── changed-files
├── deploy-operator
└── trtllm

cleanup
├── changed-files
├── deploy-operator
├── deploy-test-trtllm
├── deploy-test-sglang
├── deploy-test-vllm
└── deploy-test-vllm-disagg-router
```

### `dynamo_latest/.github/workflows/nightly-ci.yml`

```
notify-slack — "Notify Slack"
└── results-summary — "Results Summary"
    ├── build-amd64 — "Build ${{ matrix.framework }} (amd64)"
    ├── build-arm64 — "Build ${{ matrix.framework }} (arm64)"
    ├── unit-tests — "${{ matrix.framework }}-${{ matrix.arch.arch }}-unit"
    │   ├── build-amd64
    │   └── build-arm64
    ├── integration-tests — "${{ matrix.framework }}-${{ matrix.arch.arch }}-integ"
    │   ├── build-amd64
    │   └── build-arm64
    ├── e2e-single-gpu-tests — "${{ matrix.framework }}-${{ matrix.arch.arch }}-1gpu-e2e"
    │   ├── build-amd64
    │   └── build-arm64
    ├── e2e-multi-gpu-tests — "${{ matrix.framework }}-${{ matrix.arch.arch }}-2gpu-e2e"
    │   ├── build-amd64
    │   └── build-arm64
    └── fault-tolerance-tests — "${{ matrix.framework.name }}-amd64-ft"
        └── build-amd64
```

### `dynamo_latest/.github/workflows/generate-docs.yml`

```
publish-s3 — "Publish docs to S3 and flush Akamai"
└── build-docs — "Build Documentation"
```

### `dynamo_latest/.github/workflows/trigger_ci.yml`

```
trigger-ci — "Trigger CI Pipeline"
└── mirror_repo — "Mirror Repository to GitLab"
```

**Debugging tip**: If an aggregator job fails (e.g., `backend-status-check`), go straight to its upstream jobs listed in `needs:`; the aggregator is typically just reporting their results.

=============================================================================
5. GITHUB OPERATIONS
=============================================================================

## 5.1 GitHub API Access

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

## 5.2 Re-running Failed GitHub Actions

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

=============================================================================
6. REMINDERS AND POINTERS
=============================================================================

## 6.1 Important Reminders
- **NEVER delete `nvidia/dynamo_latest/index.html`** - This is the production HTML page for commit history
- Never commit sensitive information (credentials, tokens, etc.)
- Always test changes locally before pushing
- Use meaningful commit messages
- Review diffs before committing
- Consider backward compatibility when making changes

## 6.2 Additional Documentation
For detailed information about Python utilities and scripts in this directory, see:
- `README.md`: comprehensive documentation for scripts/tools
- `.cursorrules`: coding conventions, style guidelines, and development practices
