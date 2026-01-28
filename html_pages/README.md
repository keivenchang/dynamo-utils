# HTML Dashboards

HTML dashboard generators and shared UI utilities for monitoring Dynamo CI/CD.

---

## Overview

**Key dashboards:**
- **Local branches** (`show_local_branches.py`) - Scans local repos, shows PRs + CI + workflow status
- **Remote PRs** (`show_remote_branches.py`) - Shows PRs by GitHub username
- **Commit history** (`show_commit_history.py`) - Shows recent commits with CI checks
- **Resource report** (`show_local_resources.py`) - System resource monitoring

**Utilities:**
- `update_html_pages.sh` - Cron-friendly wrapper for atomic updates
- `common_dashboard*.py` - Shared rendering logic (see module docstring for node hierarchy)

**Prerequisites:**
- Python 3.10+
- `pip install jinja2 requests`
- GitHub token: `~/.config/github-token` or `~/.config/gh/hosts.yml`

**Outputs:**
- Local branches: `$HOME/dynamo/speedoflight/dynamo/users/<user>/local.html`
- Commit history: `$DYNAMO_REPO/index.html`  
- Resource report: `$DYNAMO_HOME/resource_report.html`

---

## Local Branches Dashboard

Scans local git repositories and displays branch status with GitHub integration.

### Features

- **Branch discovery** under `--repo-path` (scans direct children)
- **PR integration** with CI checks from GitHub
- **Workflow status** for branches with remotes but no PRs *(NEW)*
- **Client-side sorting** (latest modified/created/branch name) *(NEW)*
- **Failure snippets** with cached raw logs
- **URL state persistence** for view settings
- **Statistics panel** with API usage and timing

### Tree Structure

**Branch with PR:**
```
[copy] [✖] branch-name → base [SHA]
├─ commit message (#PR)
├─ (modified PT, created UTC, age)
└─ ▶ PASSED ✓26 ✗2
   ├─ ✓ check-1 (6m) [log]
   └─ ✗ check-2 (2m) [log] ▶ Snippet
```

**Branch without PR (remote):**
```
[copy] branch-name → base [SHA]
└─ ✅ PASSED ✓5
   ├─ ✓ pre_merge
   ├─ ✓ Rust pre-merge checks
   └─ ✓ Copyright Checks
```

See module docstring in `common_dashboard_lib.py` for complete node hierarchy and creation flow.

### CI Check Details

**Ordering:** Checks sorted lexically by display name (`kind: job name`)

**Required badge:** Derived from GitHub GraphQL `statusCheckRollup.isRequired`
- Cached in `~/.cache/dynamo-utils/github_required_checks.json`
- Persists in `PRInfo` for cache-only mode

**Job steps:** Shown for long-running jobs (≥10 min) and required jobs
- Display steps ≥30s duration + all failing steps
- Fetched via GitHub Actions job details API

**Special handling:** `Build and Test - dynamo` shows phase breakdown
- Uses job `steps[]` for accurate duration and status
- Falls back to simple display if steps unavailable

### API Budget

- Default: `--max-github-api-calls 100`
- Switches to cache-only mode when exhausted
- Statistics panel shows API usage breakdown

### Usage

```bash
python3 html_pages/show_local_branches.py \
  --repo-path ~/dynamo \
  --output ~/dynamo/speedoflight/dynamo/users/keivenchang/local.html \
  --max-github-api-calls 100
```

---

## Remote PRs Dashboard

Shows PRs by GitHub username (not tied to local repos).

### Features

- Fetches PRs via GitHub API
- **IDENTICAL tree structure and UI as local branches** *(FIXED)*
- Sorted by latest activity or branch name
- **Full CI job hierarchy with parent-child relationships** *(FIXED)*
- Reuses all formatting/status helpers from `show_local_branches.py`

### Tree Structure

**Identical to local branches, but with `UserNode` instead of `RepoNode`:**
```
UserNode (github-username)
└─ BranchInfoNode (remote-branch)
   ├─ CommitMessageNode (PR title)
   ├─ MetadataNode (modified, created, age)
   ├─ PRNode (PR link)
   └─ PRStatusNode (PASSED/FAILED pill)
      ├─ CIJobTreeNode (backend-status-check)
      │  ├─ CIJobTreeNode (vllm (amd64))
      │  └─ CIJobTreeNode (trtllm (amd64))
      ├─ CIJobTreeNode (DCO [REQUIRED])
      └─ ...
```

### Key Differences from Local Branches

- Uses PR title (no local `git log` access)
- Shows only PRs created by specified user
- No local-only branches section
- **Root node:** `UserNode` (GitHub user) instead of `RepoNode` (directory)

### Implementation Notes

- Imports `PRStatusNode` and `_build_ci_hierarchy_nodes` from `show_local_branches.py` to ensure **identical rendering logic**
- The versions in `common_branch_nodes.py` are incomplete stubs and should NOT be used

### Usage

```bash
python3 show_remote_branches.py \
  --github-user keivenchang \
  --repo-root ~/dynamo \
  --output speedoflight/users/keivenchang/index.html
```

### Cron Integration

**Working hours (8am-6pm PT):** Every 1 minute
```cron
# Working hours: 16:00-23:59 UTC + 00:00-01:59 UTC
* 16-23 * * * DYNAMO_HOME=$HOME/dynamo REMOTE_GITHUB_USERS="kthui keivenchang" $HOME/dynamo/dynamo-utils.dev/cron_log.sh remote_prs_working $HOME/dynamo/dynamo-utils.dev/html_pages/update_html_pages.sh --show-remote-branches
* 0-1 * * * DYNAMO_HOME=$HOME/dynamo REMOTE_GITHUB_USERS="kthui keivenchang" $HOME/dynamo/dynamo-utils.dev/cron_log.sh remote_prs_working $HOME/dynamo/dynamo-utils.dev/html_pages/update_html_pages.sh --show-remote-branches
```

**Off hours (6pm-8am PT):** Every 20 minutes
```cron
# Off hours: 02:00-15:59 UTC
*/20 2-15 * * * DYNAMO_HOME=$HOME/dynamo REMOTE_GITHUB_USERS="kthui keivenchang" $HOME/dynamo/dynamo-utils.dev/cron_log.sh remote_prs_offhours $HOME/dynamo/dynamo-utils.dev/html_pages/update_html_pages.sh --show-remote-branches
```

**Output locations:**
- `$HOME/dynamo/speedoflight/users/kthui/index.html`
- `$HOME/dynamo/speedoflight/users/keivenchang/index.html`

**Logs:**
- Working hours: `~/dynamo/logs/YYYY-MM-DD/remote_prs_working.log`
- Off hours: `~/dynamo/logs/YYYY-MM-DD/remote_prs_offhours.log`

---

## Commit History Dashboard

Shows recent commits with expandable GitHub checks.

### Features

- Recent commits from local git
- GitHub checks tree per commit
- GitLab pipeline summary (optional)
- Cached raw logs + snippets for failures
- Same job step rules as branches dashboard

### Usage

```bash
# Full refresh
python3 html_pages/show_commit_history.py \
  --repo-path ~/dynamo/commits \
  --max-commits 100 \
  --output ~/dynamo/commits/index.html

# Cache-only (skip GitLab)
python3 html_pages/show_commit_history.py \
  --repo-path ~/dynamo/commits \
  --skip-gitlab-api \
  --output ~/dynamo/commits/index.html
```

---


## Performance & API Usage

### Caching Architecture

All dashboards use aggressive caching to minimize GitHub API usage:

**Cache Types:**
- **Memory cache** - Fast, per-process, short TTL
- **Disk cache** - Persistent JSON files in `~/.cache/dynamo-utils/`
- **ETag support** - 304 responses don't count against rate limit

**Cache TTLs:**
- Open PRs (recent commits <8h): 3 minutes
- Open PRs (older commits ≥8h): 2 hours  
- Closed/merged PRs: 30 days (immutable)
- Completed jobs: ∞ (never changes)
- In-progress jobs: 1 minute

### API Consumption Patterns

**Typical run (show_local_branches with warm cache):**
- Total API calls: ~50-60
- ETag 304 responses: ~4 (FREE)
- Effective cost: ~45-55 calls

**What consumes APIs:**
1. **Batch prefetch (10-20 calls)** - Fetches all jobs for workflow runs to populate cache (optimization that avoids 500+ individual calls)
2. **In-progress job checks (10-15 calls)** - Monitors running jobs for completion
3. **Log downloads (5-10 calls)** - Fetches logs for new failures
4. **Metadata refresh (10-20 calls)** - PR lists, check runs, workflow details

**Cache hit rates (typical):**
- actions_job_details: 97-99% (batch prefetch working)
- pr_checks: 70-100% (depends on PR activity)
- raw_log_text: 95-99% (failure logs cached)

### ETag Optimization

When cache TTL expires, ETag support minimizes actual data transfer:
- Send request with `If-None-Match: <etag>`
- GitHub returns 304 if data unchanged (FREE!)
- Only 200 OK responses count against rate limit

**ETag effectiveness:**
- check_runs: 60-70% return 304
- Completed workflow runs: ~100% return 304 (immutable)
- In-progress jobs: 0% (data changing, can't use ETags)

### Batch Prefetch

Instead of fetching job details individually (500+ calls):
1. Collect all run_ids from check-runs
2. Batch fetch using `/actions/runs/{run_id}/jobs`
3. Populate cache with all jobs (10-20 calls total)
4. Individual lookups hit cache (100% hit rate)

**Result:** 95%+ reduction in API calls

## Cron Wrapper


**Error Handling:** When commands fail, errors are printed to stderr (terminal) in addition to being logged. This makes debugging interactive runs easier while maintaining clean cron behavior.

`update_html_pages.sh` runs generators with atomic file updates.

### Scheduling

```cron
# Full update every 30 minutes
0,30 * * * * DYNAMO_HOME=$HOME/dynamo $HOME/dynamo/dynamo-utils.dev/cron_log.sh update_html_pages_full $HOME/dynamo/dynamo-utils.dev/html_pages/update_html_pages.sh --show-local-branches --show-commit-history

# Cache-heavy between full updates (every 4 minutes)
8-59/4 * * * * DYNAMO_HOME=$HOME/dynamo $HOME/dynamo/dynamo-utils.dev/cron_log.sh update_html_pages_cached $HOME/dynamo/dynamo-utils.dev/html_pages/update_html_pages.sh --show-local-branches --show-commit-history --skip-gitlab-api

# Resource report (every minute)
* * * * * DYNAMO_HOME=$HOME/dynamo $HOME/dynamo/dynamo-utils.dev/cron_log.sh resource_report $HOME/dynamo/dynamo-utils.dev/html_pages/update_html_pages.sh --show-local-resources

# Remote PRs - working hours (8am-6pm PT): every minute
* 16-23 * * * DYNAMO_HOME=$HOME/dynamo REMOTE_GITHUB_USERS="kthui keivenchang" $HOME/dynamo/dynamo-utils.dev/cron_log.sh remote_prs_working $HOME/dynamo/dynamo-utils.dev/html_pages/update_html_pages.sh --show-remote-branches
* 0-1 * * * DYNAMO_HOME=$HOME/dynamo REMOTE_GITHUB_USERS="kthui keivenchang" $HOME/dynamo/dynamo-utils.dev/cron_log.sh remote_prs_working $HOME/dynamo/dynamo-utils.dev/html_pages/update_html_pages.sh --show-remote-branches

# Remote PRs - off hours (6pm-8am PT): every 20 minutes
*/20 2-15 * * * DYNAMO_HOME=$HOME/dynamo REMOTE_GITHUB_USERS="kthui keivenchang" $HOME/dynamo/dynamo-utils.dev/cron_log.sh remote_prs_offhours $HOME/dynamo/dynamo-utils.dev/html_pages/update_html_pages.sh --show-remote-branches
```

### Logs

- Per-run logs: `~/dynamo/logs/YYYY-MM-DD/<job>.log`
- Generator logs: `html_pages/show_*.log`

### Troubleshooting

**Script reports failure?** Errors are now printed to stderr with log file location.

Example output:
```bash
$ ./html_pages/update_html_pages.sh --show-local-branches
ERROR: Failed to update /home/keivenc/dynamo/speedoflight/dynamo/users/keivenchang/local.html
See log for details: /home/keivenc/dynamo/logs/2026-01-23/show_local_branches.log
```

1. Check the log file path shown in the error message
2. Look for Python tracebacks or error messages:
   ```bash
   tail -50 /home/keivenc/dynamo/logs/2026-01-23/show_local_branches.log
   ```

3. Run with --debug-html for faster iteration:
   ```bash
   ./html_pages/update_html_pages.sh --debug-html --show-local-branches
   ```

**Notes:**
- If running from cron, expect most output to go to files under `~/dynamo/logs/<YYYY-MM-DD>/` (not stdout)
- Use `--run-ignore-lock` only if you're sure another run isn't actively writing outputs/caches
- Default behavior (no flags): Runs all tasks (local branches + commit history + resource report + remote PRs)

**Common flags:**
- `--show-local-resources` - Update resource report only
- `--show-local-branches` - Update branches dashboard only
- `--show-remote-branches` - Update remote PRs dashboard only
- `--show-commit-history` - Update commit history dashboard only
- `--debug-html` - Fast mode: smaller commit window (25 commits), outputs to debug.html
- `--skip-gitlab-api` - Skip GitLab fetching (faster, cache-only for registry data)
- `--github-token <token>` - Override GitHub token

### Outputs and Verification

**Quick "did it actually update?" checks:**
```bash
ls -lah ~/dynamo/speedoflight/dynamo/users/keivenchang/local.html       # local branches dashboard
ls -lah ~/dynamo/commits/index.html       # commit history dashboard
ls -lah ~/dynamo/speedoflight/stats/index.html  # stats landing page
```

**Common foot-guns:**
- There are **two repos**: `dynamo-utils.dev/` (dev) and `dynamo-utils/` (prod). If working on dev only, run the dev script: `dynamo-utils.dev/html_pages/update_html_pages.sh`
- `update_html_pages.sh --fast` is intentionally **removed**; use `--debug-html` instead
- Per-component logs are **append-only** and may contain older, non-prefixed lines from previous runs
- If a log message is missing commit SHA context, the caller didn't pass `commit_sha` through to helpers

**If `update_html_pages.sh` appears to "finish instantly":**

Check logs first (common root cause: generator crashed early due to ImportError):
```bash
tail -n 200 ~/dynamo/logs/$(date +%Y-%m-%d)/cron.log
tail -n 200 ~/dynamo/logs/$(date +%Y-%m-%d)/show_commit_history.log
tail -n 200 ~/dynamo/logs/$(date +%Y-%m-%d)/show_local_branches.log
```

### Common UI Pitfalls

**Links/buttons inside `<details>` trees:**

Clicks may toggle the tree unintentionally. Fix pattern: ensure handlers call **both** `event.preventDefault()` and `event.stopPropagation()`.

**Example:**
```javascript
element.addEventListener('click', (event) => {
    event.preventDefault();     // Prevent default link behavior
    event.stopPropagation();    // Stop event from bubbling to <details>
    // Your handler code here
});
```

### Cache Statistics Understanding

**CRITICAL: Misleading statistic names in dashboards**

The dashboard shows `cache.github.required_checks.hits` and `cache.github.required_checks.misses`, but these do NOT track the `get_required_checks()` function!

**What these stats actually track:**
- Located in: `common_github.py` lines 3678, 3707, 3710, 3735
- Function: `get_pr_checks_rows()` (NOT `get_required_checks()`)
- Purpose: Track whether PR check rows with `is_required` flags are cached
- Called: Once per commit (100 times for 100 commits)

**The actual `get_required_checks()` function:**
- File: `common_github.py` lines 4480-4705
- Cache: `~/.cache/dynamo-utils/required-checks/required_checks.json`
- Keys: `required_checks:ai-dynamo/dynamo:pr5478`
- Stats: **NO statistics tracking** (not in dashboard)

**Production behavior (show_commit_history.py):**
- Calls `get_required_checks()` ONCE for a single open PR (line 2080-2084)
- Uses the result as a template for all 100 commits (line 2040)
- Does NOT query individual PR required checks for each commit
- Cached entries for merged PRs exist but aren't used in this workflow

**Why "0 hits / 100 misses" appears in production:**
- First run with 100 new commits → 100 `get_pr_checks_rows()` misses (populating cache)
- Second run with same commits → 100 `get_pr_checks_rows()` hits (from cache)
- This is NORMAL and expected behavior

**Negative caching implementation (2026-01-19):**
- Location: `common_github.py` lines 4565-4571, 4694-4699
- When PR fetch fails (404, timeout), cache the empty result
- Prevents retrying non-existent PRs on every run
- Working correctly (verified with 50/100 commit tests)

### GitHub API Optimizations

**Key optimizations (2026-01-18) that reduce API usage by 85-98%:**

1. **ETag support**: Conditional requests via `If-None-Match` header
   - 304 responses DON'T count against rate limit
   - Cache v6 stores ETags for check-runs and status endpoints
   - Benefit: 85-95% rate limit reduction on subsequent runs

2. **Batched workflow run fetching**: Collects all run_ids first, then batch fetches
   - Benefit: 90% reduction (100 individual → 10-20 batched calls)

3. **Parallelization bug fix**: `get_required_checks_for_base_ref()` called once instead of 100×
   - Benefit: 99 redundant API calls eliminated

**Impact:**
- Before: ~2000 API calls per run → exhausted after 2-3 runs
- After: ~200-300 calls (first run), ~10-30 calls (subsequent runs with ETags)
- All dashboard scripts benefit automatically (no changes needed)

**Verify optimizations:**
```bash
# Check cache version (v6 has ETag support)
python3 << 'EOF'
import json
with open('~/.cache/dynamo-utils/pr-checks/pr_checks_cache.json') as f:
    cache = json.load(f)
v6_count = sum(1 for e in cache.values() if e.get('ver', 0) >= 6)
etag_count = sum(1 for e in cache.values() if e.get('check_runs_etag'))
print(f"v6 entries: {v6_count}/{len(cache)}, with ETags: {etag_count}/{len(cache)}")
EOF
```
