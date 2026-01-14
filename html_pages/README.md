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
- `TREE_NODE_REFERENCE.md` - Complete node type documentation
- `common_dashboard*.py` - Shared rendering logic

**Prerequisites:**
- Python 3.10+
- `pip install jinja2 requests`
- GitHub token: `~/.config/github-token` or `~/.config/gh/hosts.yml`

**Outputs:**
- Branches: `$NVIDIA_HOME/index.html`
- Commit history: `$DYNAMO_REPO/index.html`  
- Resource report: `$NVIDIA_HOME/resource_report.html`

---

## Local Branches Dashboard

Scans local git repositories and displays branch status with GitHub integration.

### Features (Updated 2026-01-07)

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

See `TREE_NODE_REFERENCE.md` for complete node definitions.

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
  --output ~/dynamo/index.html \
  --max-github-api-calls 100
```

---

## Remote PRs Dashboard

Shows PRs by GitHub username (not tied to local repos).

### Features (Updated 2026-01-08)

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

```bash
# Via update_html_pages.sh
REMOTE_GITHUB_USERS="user1 user2" update_html_pages.sh --show-remote-branches
```

See `CRONTAB_REMOTE_BRANCHES.md` for scheduling details.

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
  --repo-path ~/dynamo/dynamo_latest \
  --max-commits 100 \
  --output ~/dynamo/dynamo_latest/index.html

# Cache-only (skip GitLab)
python3 html_pages/show_commit_history.py \
  --repo-path ~/dynamo/dynamo_latest \
  --skip-gitlab-fetch \
  --output ~/dynamo/dynamo_latest/index.html
```

---

## Cron Wrapper

`update_html_pages.sh` runs generators with atomic file updates.

### Scheduling

```cron
# Full update every 30 minutes
0,30 * * * * NVIDIA_HOME=$HOME/dynamo $HOME/dynamo/dynamo-utils/cron_log.sh update_html_pages_full $HOME/dynamo/dynamo-utils/html_pages/update_html_pages.sh --show-local-branches --show-commit-history

# Cache-heavy between full updates (every 4 minutes)
8-59/4 * * * * NVIDIA_HOME=$HOME/dynamo SKIP_GITLAB_FETCH=1 $HOME/dynamo/dynamo-utils/cron_log.sh update_html_pages_cached $HOME/dynamo/dynamo-utils/html_pages/update_html_pages.sh --show-local-branches --show-commit-history

# Resource report (every minute)
* * * * * NVIDIA_HOME=$HOME/dynamo $HOME/dynamo/dynamo-utils/cron_log.sh resource_report $HOME/dynamo/dynamo-utils/html_pages/update_html_pages.sh --show-local-resources

# Remote PRs - working hours (8am-6pm PT): every minute
* 16-23 * * * NVIDIA_HOME=$HOME/dynamo REMOTE_GITHUB_USERS="kthui keivenchang" $HOME/dynamo/dynamo-utils/cron_log.sh remote_prs_working $HOME/dynamo/dynamo-utils/html_pages/update_html_pages.sh --show-remote-branches
* 0-1 * * * NVIDIA_HOME=$HOME/dynamo REMOTE_GITHUB_USERS="kthui keivenchang" $HOME/dynamo/dynamo-utils/cron_log.sh remote_prs_working $HOME/dynamo/dynamo-utils/html_pages/update_html_pages.sh --show-remote-branches

# Remote PRs - off hours (6pm-8am PT): every 20 minutes
*/20 2-15 * * * NVIDIA_HOME=$HOME/dynamo REMOTE_GITHUB_USERS="kthui keivenchang" $HOME/dynamo/dynamo-utils/cron_log.sh remote_prs_offhours $HOME/dynamo/dynamo-utils/html_pages/update_html_pages.sh --show-remote-branches
```

### Logs

- Per-run logs: `~/dynamo/logs/YYYY-MM-DD/<job>.log`
- Generator logs: `html_pages/show_*.log`

### Troubleshooting

**Script exits too quickly?** Usually means a generator crashed.

1. Check per-generator logs:
   ```bash
   tail -200 html_pages/show_local_branches.log
   tail -200 html_pages/show_commit_history.log
   ```

2. Run generator directly:
   ```bash
   # Debug-mode output (debug.html):
   ./html_pages/update_html_pages.sh --fast-debug --show-local-branches
   ```

3. Common cause: Missing/renamed exports in `common_dashboard_lib.py`

---

## Caching

**Location:**
- `~/.cache/dynamo-utils/` (default)
- `$DYNAMO_UTILS_CACHE_DIR/` (override)

**Key caches:**

| Cache | Path | TTL | Purpose |
|-------|------|-----|---------|
| PR list | `pulls/` | 60s | Open PRs per repo |
| PR checks | `pr-checks/` | 300s | Check-runs per PR |
| PR info | `pr-info/pr_info.json` | Keyed by `updated_at` | Full PR details |
| PR updated_at | `search-issues/` | 60s | Batched probe |
| Required checks | `required-checks/` | Long-lived | Branch protection |
| Job details | `actions-jobs/` | 600s | Steps/timings |
| Raw log URLs | `raw-log-urls/` | 3600s | Signed URLs |
| Raw log text | `raw-log-text/` | 30 days | Downloaded logs |

### Zero-API PRInfo Reuse

For unchanged PRs, we skip all per-PR API calls:

1. Batch probe `updated_at` via `/search/issues` (1 call for all PRs)
2. Cache `PRInfo` keyed by `(pr_number, updated_at)`
3. Reuse cached `PRInfo` if `updated_at` matches → **0 API calls**

In cache-only/budget-exhausted mode, stale `PRInfo` is reused even when TTLs expire.

---

## API Call Types

Dashboards report GitHub REST usage by label (see `common.py`):

**Common labels:**
- `rate_limit` - Quota check
- `search_issues` - Batched PR probe
- `pulls_list` - Open PRs per repo
- `pull_request` - PR details
- `check_runs` - Check-runs per commit
- `actions_run` - Workflow run metadata
- `actions_job_status` - Job details
- `actions_job_logs_zip` - Raw log download
- `pr_review_comments` - Conversation count

**Why calls happen "when nothing changed":**
- Short TTL on check-runs (refreshes unsettled CI)
- New workflow reruns (new `run_id`)
- Missing raw logs (triggers download)
- Phase breakdown fetch (job steps API)

---

## API Call Example (One PR)

Detailed walkthrough for branch `keivenchang/DIS-1200__refactor` with PR #5050:

### Step 0: Rate Limit Check
- `GET /rate_limit` (observability only)
- **Calls:** 1-2 per run

### Step 1: List Open PRs
- `GET /repos/ai-dynamo/dynamo/pulls?state=open&per_page=100`
- **Cache:** 60s TTL
- **Calls:** 0 (cached) or 1

### Step 2: Per-PR Enrichment

**2A) Probe updated_at (batched)**
- `GET /search/issues?q=repo:ai-dynamo/dynamo type:pr number:5050 ...`
- Returns `updated_at` for all PRs in one call
- If matches cached `PRInfo` → **skip all remaining per-PR calls**
- **Cache:** 60s TTL
- **Calls:** 0 (cached) or 1

**2B) Fetch PR + checks (if updated)**
- `GET /repos/ai-dynamo/dynamo/pulls/5050`
- `GET /repos/ai-dynamo/dynamo/commits/{sha}/check-runs`
- **Cache:** 300s TTL (checks)
- **Calls:** 0-2

**2C) Required checks (long-lived)**
- GraphQL via `gh api graphql`
- **Cache:** Persistent (rarely changes)
- **Calls:** 0-1

**2D) Review comments**
- `GET /repos/ai-dynamo/dynamo/pulls/5050/comments`
- **Cache:** None (not cached today)
- **Calls:** 0-1

### Step 3: Failed Job Logs (on failures)

**3A) Job status**
- `GET /repos/ai-dynamo/dynamo/actions/jobs/{job_id}`
- **Cache:** 120s TTL (memory)
- **Calls:** 0-2 (per failed job)

**3B) Download logs**
- `GET /repos/ai-dynamo/dynamo/actions/jobs/{job_id}/logs`
- **Cache:** 30 days (persistent)
- **Calls:** 0-2 (download once, reuse forever)

**3C) Job details (for phase breakdown)**
- `GET /repos/ai-dynamo/dynamo/actions/jobs/{job_id}`
- **Cache:** 600s TTL
- **Calls:** 0-1 (for `Build and Test - dynamo` jobs)

### Call Count Summary (One PR)

**Best case (warm caches, nothing changed):**
- Total: **1-2 calls** (just rate_limit)

**Worst case (cold caches):**
- `rate_limit`: 1-2
- `pulls_list`: 1
- `search_issues`: 1
- `pull_request`: 2
- `check_runs`: 2
- `required_status_checks`: 1
- `pr_review_comments`: 1
- `actions_run`: 3 (per unique run_id)
- `actions_job_status`: 2
- `actions_job_logs_zip`: 2
- Total: **15-16 calls**

Multiply by number of PRs to estimate total. Budget (`--max-github-api-calls`) caps total and switches to cache-only mode when exhausted.

---

## Quick Reference

**Node types:** See `TREE_NODE_REFERENCE.md`

**Common modifications:**
- Branch line format → `BranchInfoNode._format_html_content()`
- CI expansion logic → `PRStatusNode.to_tree_vm()` or `CIJobTreeNode._subtree_needs_attention()`
- Repo icon → `RepoNode._format_html_content()`

**Helper functions:** See "Shared Helper Functions" in `TREE_NODE_REFERENCE.md`
