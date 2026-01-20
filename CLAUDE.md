CLAUDE.md - NVIDIA Dynamo Projects (Operational Procedures)

Operational playbooks and environment conventions for this workspace.
For coding conventions, style guidelines, and in-container development practices, refer to `.cursorrules`.

Scope / intent:
- This file (`CLAUDE.md`) is for **host-side operations**: Docker commands, process management, CI log triage, and GitHub operations.
- Keep it short and executable: prefer checklists + commands over long narratives or huge pasted graphs.
- Avoid duplicating `.cursorrules` content; link to it when the topic is coding/formatting/linting inside the container.

=============================================================================
TABLE OF CONTENTS
=============================================================================
1. META INSTRUCTIONS AND POLICIES
  1.1 Remember This
  1.2 New Project Protocol
  1.3 Commit Policy
  1.4 Permission Policy
  1.5 Python Virtual Environment (Host vs Container)
  1.6 CRITICAL ANTI-PATTERNS (Review Every 10 Minutes)

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
  4.8 Dynamo-utils dashboards (operational runbook)
    4.8.1 Running `update_html_pages.sh`
    4.8.2 Outputs, logs, and “it ran too quickly”
    4.8.3 Common UI pitfalls (links/buttons inside <details>)

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
1. Re-read `~/dynamo/dynamo-utils/.cursorrules`
2. Re-read `~/dynamo/dynamo-utils/CLAUDE.md`

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
source ~/dynamo/venv/bin/activate  # example; adjust to your machine
```

If running inside the dev container, the Python environment is already set up.

## 1.6 CRITICAL ANTI-PATTERNS (Review Every 10 Minutes)

**REMINDER: Review this section every 10 minutes during active coding.**

These anti-patterns are ABSOLUTELY FORBIDDEN. Violating them wastes debugging time and creates bugs:

### 1.6.1 ALL IMPORTS AT TOP OF FILE - NO try/except

❌ **ABSOLUTELY FORBIDDEN:**
```python
def foo():
    import json  # NEVER import inside functions!
    import re

# ALSO FORBIDDEN - Don't hide import errors! Just make sure the 3rd party imports are in requirements.txt.
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    requests = None
```

✅ **CORRECT:**
```python
# At top of file - import directly, fail fast if not available
import json
import re
import requests
import yaml

def foo():
    # Use imports directly
```

**RULES:**
- ALL imports at module top (never inside functions)
- ⚠️ EXTREMELY IMPORTANT: Import third-party packages DIRECTLY - don't use try/except to hide ImportError
- Fail fast if dependencies are missing (add them to requirements.txt)
- If a package isn't installed, the import will fail immediately with clear error

**ONLY EXCEPTION:** With explicit user permission for lazy imports in rare cases.

### 1.6.2 NO DEFENSIVE getattr() WHEN TYPES ARE KNOWN

❌ **ABSOLUTELY FORBIDDEN:**
```python
# If node has job_name attribute:
getattr(node, "job_name", "")  # REDUNDANT!
str(getattr(node, "job_name", "") or "")  # Hides None bugs!

# If pr is typed PRInfo:
getattr(pr, "number", 0)  # REDUNDANT! Don't ever do this if pr is typed.
```

✅ **CORRECT:**
```python
# Type is known, attribute exists - just use it!
node.job_name
str(node.job_name)
pr.number
```

**WHY:** Using getattr() for known attributes masks AttributeError bugs, makes type checkers useless, and hides None/empty string confusion.

### 1.6.3 FAIL FAST - NEVER HIDE ERRORS

❌ **ABSOLUTELY FORBIDDEN:**
```python
try:
    something()
except Exception:  # TODO: ❌ Fix me. This is an anti-pattern.
    pass  # Silent failure!

try:
    something()
except Exception as e:  # # TODO: ❌ Fix me. ⚠️ This is NOT preferred, and to be avoided
    logging.error(f"Error: {e}")  # Log and hide!
    # Missing raise = hidden error!

try:
    something()
except Exception:  # TODO: ❌ Fix me. ❌ Anti-pattern, horrible style.
    return []  # Return default on error!
```

✅ **CORRECT:**
```python
# Let exceptions propagate:
result = something()  # If it fails, let it crash!

# Only catch SPECIFIC exceptions you can handle:
try:
    result = parse_config(file)
except FileNotFoundError:
    result = DEFAULT_CONFIG  # Specific exception we know how to handle
except ValueError as e:
    logger.warning(f"Invalid config: {e}")
    result = DEFAULT_CONFIG
# Other exceptions propagate up!

# Re-raise after logging if you can't handle it:
try:
    result = something()
except Exception as e:  # TODO: ❌ Fix me. ⚠️ Only do this if you exhausted all possible options.
    logger.error(f"Failed: {e}")
    raise  # Re-raise so caller knows it failed!
```

**CRITICAL: NEVER USE `except Exception:` WITHOUT RE-RAISING**

Using `except Exception:` catches **everything** (ValueError, TypeError, AttributeError, KeyError, etc.) and hides bugs you didn't anticipate.

**THE RULE:**
1. If you can remove the try/except → DO IT (let it fail fast!)
2. If you must catch exceptions → Use SPECIFIC types (FileNotFoundError, ValueError, TypeError, etc.)
3. If you catch Exception → You MUST re-raise it (or have explicit user permission)

**WHY:** Silent failures make debugging impossible, hide bugs until production, waste hours of debugging time, and corrupt data by continuing with invalid state.

**For complete details, see `.cursorrules` section 3.5.3 "Error Handling and Anti-Patterns"**

=============================================================================
2. ENVIRONMENT SETUP
=============================================================================

## 2.1 All Projects Overview
This workspace contains multiple projects:
- `dynamo1, dynamo2, dynamo3, dynamo4`: multiple working branches of the Dynamo repository
- `dynamo_ci`: main CI/testing repository for Dynamo
- `dynamo-utils`: build automation scripts and utilities

## 2.2 Docker Container Naming
When running `docker ps`, VS Code/Cursor dev container images follow this naming pattern:
- `vsc-dynamo1-*` → `<workspace>/dynamo1`
- `vsc-dynamo2-*` → `<workspace>/dynamo2`
- `vsc-dynamo3-*` → `<workspace>/dynamo3`
- `vsc-dynamo4-*` → `<workspace>/dynamo4`
- `vsc-dynamo_ci-*` → `<workspace>/dynamo_ci`

The `vsc-` prefix indicates VS Code/Cursor dev containers, and the part after it matches the directory name.
Container names (like `epic_satoshi`, `distracted_shockley`) are transient and should not be documented.

## 2.3 Host-Container Directory Mapping
The `dynamo-utils` directory on the host is mapped into the container at `/workspace/_` (same repo, bind-mounted):
- Host: `<workspace>/dynamo-utils/` (e.g. `~/dynamo/dynamo-utils/`)
- Container: `/workspace/_/`

Example: `<workspace>/dynamo-utils/notes/metrics-vllm.log` on the host appears at `/workspace/_/notes/metrics-vllm.log` inside the container.

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

Tip: Find the container name with:
```bash
docker ps --format "table {{.Names}}\t{{.Image}}" | grep dynamo
```

Example for VLLM (copy/paste template):
```bash
# Find container name (adjust grep to the repo you’re using: dynamo1/dynamo2/dynamo_ci/...)
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
- Metrics saved to: `<workspace>/dynamo-utils/notes/metrics-<framework>.log` (e.g. `~/dynamo/dynamo-utils/notes/metrics-<framework>.log`) (on host)
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
docker exec -u root <container> bash -c "export WORKSPACE_DIR=~/dynamo && cd ~/dynamo && pytest -xvs tests/serve/test_vllm.py::test_serve_deployment -k aggregated --basetemp=/tmp/pytest_temp"
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

**See `.cursorrules`** for the complete list of pre-merge CI checks including:
- Rust Format Check (cargo fmt)
- Rust Clippy Checks (unused imports, warnings)
- Rust Tests (unit, doc, integration)
- Pre-commit Hooks (ruff, mypy, YAML/JSON validation)
- Copyright Headers
- Cargo Deny (license checks)
- Quick Pre-Commit Checklist

## 4.4 Documentation Build Test

Test Sphinx documentation build (same as CI) to verify no warnings/errors.

**IMPORTANT**: Run this on the host (outside the dev container). If Docker isn’t available, have the user run it and paste the failing log snippet.

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
   - Error: `Broken link: Pre-Deployment Checks -> ../../deploy/pre-deployment/README.md`
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

Practical workflow:
- If an aggregator job fails (often named `*-status-check`), immediately jump to its upstream jobs listed in `needs:`; the aggregator usually just reports their results.
- Open the failing workflow file under `.github/workflows/`, find the job, and inspect `needs:` + any `if:` guards.

Quick commands:
```bash
# Find where a job is defined
grep -n "^[[:space:]]\\{0,2\\}<job_name>:" .github/workflows/*.yml

# Find needs blocks in a workflow
grep -n "needs:" .github/workflows/<workflow>.yml
```

## 4.8 Dynamo-utils dashboards (operational runbook)

For operational procedures including:
- Running `update_html_pages.sh`
- Output verification and troubleshooting
- Common UI pitfalls
- Cache statistics understanding
- GitHub API optimizations

**See:** `html_pages/README.md` → "Operational Runbook" section

=============================================================================
5. GITHUB OPERATIONS
=============================================================================

## 5.1 GitHub API Access

**GitHub credentials location**: `~/.config/gh/hosts.yml`

**Environment variables**:
- Prefer `GH_TOKEN` for host-side GitHub access (matches GitHub CLI conventions).
- Some scripts still use `GITHUB_TOKEN` as a fallback name (common in GitHub Actions).

To use GitHub API with curl:
```bash
# Using environment variable (preferred)
curl -H "Authorization: token ${GH_TOKEN:-$GITHUB_TOKEN}" https://api.github.com/user

# Or read from config file
TOKEN=$(grep oauth_token ~/.config/gh/hosts.yml | head -1 | awk '{print $2}')
curl -H "Authorization: token $TOKEN" https://api.github.com/repos/ai-dynamo/dynamo/...
```

For Python scripts, use `GitHubAPIClient` from `common.py` which automatically reads credentials from:
1. Provided token argument
2. `GH_TOKEN` environment variable (preferred)
3. `GITHUB_TOKEN` environment variable (fallback)
4. `~/.config/gh/hosts.yml` (GitHub CLI config)

**For GitHub API optimizations and cache statistics details:**
- See `html_pages/README.md` → "Operational Runbook" section

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
- **NEVER delete `<workspace>/dynamo_latest/index.html`** (e.g. `~/dynamo/dynamo_latest/index.html`) - This is the production HTML page for commit history
- Never commit sensitive information (credentials, tokens, etc.)
- Always test changes locally before pushing
- Use meaningful commit messages
- Review diffs before committing
- Consider backward compatibility when making changes

## 6.2 Additional Documentation
For detailed information about Python utilities and scripts in this directory, see:
- `README.md`: comprehensive documentation for scripts/tools
- `.cursorrules`: in-container coding conventions, style guidelines, and development practices
