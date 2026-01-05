## HTML dashboards (`dynamo-utils/html_pages/`)

This directory contains the **HTML dashboard generators** and shared dashboard UI utilities.

### Directory structure

```
html_pages/
├── README.md
├── common_dashboard.j2
├── common_dashboard_lib.py
├── common_github_workflow.py
├── show_commit_history.j2
├── show_commit_history.py
├── show_local_branches.j2
├── show_local_branches.py
├── show_local_resources.py
└── update_html_pages.sh
```

- **Branches dashboard**: `show_local_branches.py` → **HTML-only**; writes `index.html` under your “nvidia” workspace root
- **Commit history dashboard**: `show_commit_history.py` → **HTML-only**; writes `index.html` under a Dynamo repo clone (e.g. `dynamo_latest/`)
- **Resource report**: `show_local_resources.py` (generated from `resource_monitor.sqlite`)
- **Cron wrapper**: `update_html_pages.sh` (runs the generators and updates outputs atomically)

### Prerequisites

- **Python**: 3.10+
- **Jinja2**: required (`pip install jinja2`)
- **requests**: required for GitHub API usage (`pip install requests`)
- **GitHub token** (recommended): use `~/.config/github-token` (or `~/.config/gh/hosts.yml`) to avoid anonymous rate limits.

### Outputs (what gets generated)

- **Branches**: `<NVIDIA_HOME>/index.html`
- **Commit history**: `<DYNAMO_REPO>/index.html`
- **Resource report**: `<NVIDIA_HOME>/resource_report.html` (or the path you pass)

These are **build artifacts**; don’t commit them unless you explicitly intend to.

---

## show_local_branches.py (branches dashboard)

### What it shows

- Local repo discovery under `--repo-path` (direct children)
- Branches + linked PRs (GitHub)
- CI checks per PR (GitHub check-runs)
- Failure snippets and stable `[cached raw log]` links (downloaded once, cached)
- A bottom “Statistics” section with timings and API usage breakdown

### CI view (flat tree)

The branches dashboard renders CI as a **flat tree** under the PR status line:

- Each GitHub check-run becomes a node (sorted by job name; see “Ordering” below).
- Some jobs have optional child “subsections” (phases/steps; see below).

- **Required badge (`[REQUIRED]`)**: derived from GitHub’s merge/required-checks data.
  - We do *not* rely on the branch-protection REST endpoint (often 403). We derive required-ness from GitHub’s
    GraphQL merge-box data: `statusCheckRollup … isRequired(pullRequestId: …)` (via `gh api graphql`).
  - This is cached in `~/.cache/dynamo-utils/github_required_checks.json`, and is also copied into cached `PRInfo`
    so the REQUIRED label persists even if the dashboard later runs in cache-only / REST-budget-exhausted mode.

### Ordering (shared logic)

Both dashboards sort check-run / check-row lists by the **lexical display label**:

- `kind: job name` (e.g. `lint: …`, `test: …`, `build: …`), then stable tie-breakers (job id / URL).

- Commit-history additionally disambiguates identical names by appending a stable suffix (e.g. `[job 12345]`).

### Example tree (local branches dashboard)

```text
<branch name> → <base branch>
└─ PR: <title> (#NNNN) → <base branch>
   └─ (tree children match Details 1:1)
      ├─ backend-status-check [REQUIRED] (…)
      ├─ Build and Test - dynamo [REQUIRED] (…)
      │  ├─ build: Build Image (…)
      │  ├─ lint: Rust checks (…)
      │  ├─ test: pytest (parallel) (…)
      │  └─ test: pytest (serial) (…)
      ├─ pre-commit [REQUIRED] (…)
      ├─ CodeRabbit (…)
      └─ …
```

### Special rule: `Build and Test - dynamo` phase breakdown

If a check-run’s name is exactly `Build and Test - dynamo`, we expand it with phase children so you can see
where the time went:

- **Preferred source**: GitHub Actions **job details** (`GET /repos/{owner}/{repo}/actions/jobs/{job_id}`),
  using the job’s `steps[]` to compute per-phase **duration** and **✓/✗**.
- If steps aren’t available, we simply omit the phase breakdown (no raw-log parsing fallback).

### Long-running job subsections (steps)

For GitHub Actions jobs we may show job-step children:

- We always cache the full job `steps[]` payload (job details API).
- We display steps whose duration is **≥ 30 seconds** (to avoid noise), and we **always display failing steps** even if they’re shorter.
- **Required jobs**: steps can be shown even if the job isn’t “long-running” (still using the same ≥30s + failing rule).
- **Non-required jobs**: steps are shown only for long-running jobs (default threshold: 10 minutes) to avoid noise.

### Legend (status icons)

Both dashboards use the same icon rendering (from `common_dashboard_lib.py` → `status_icon_html()`), including:

- **Skipped**: grey circle-slash icon (GitHub-like “skipped/neutral”).

### API budget (per invocation)

The script enforces a hard cap on GitHub REST calls:

- `--max-github-api-calls N` (default: **100**)
- Once exhausted, the client switches into **cache-only mode** (best-effort) instead of failing.

### Example

```bash
python3 html_pages/show_local_branches.py \
  --repo-path ~/nvidia \
  --output ~/nvidia/index.html \
  --max-github-api-calls 100
```

---

## show_commit_history.py (commit history dashboard)

### What it shows

- Recent commits (local git)
- GitHub checks tree per commit (GitHub check-runs)
- GitLab pipeline summary (optional; can be skipped)
- Stable `[cached raw log]` links + snippets for failures
- Timing + API counters at the bottom

### CI expanded view

`show_commit_history.py` renders an expandable GitHub checks section per commit, using the same subsections rules:

- `Build and Test - dynamo` gets phase children (job steps API, with raw-log fallback).
- Steps are displayed using the same rule as branches:
  - steps ≥ 30 seconds, plus any failing steps
  - required jobs can show steps even if the job isn’t “long-running”

### Example tree (commit history dashboard, merge-to-main commit)

```text
<commit subject> (<short sha>)
└─ GitHub checks
   ├─ Build and Test - dynamo [REQUIRED] (…)
   │  ├─ build: Build Image (…)
   │  ├─ lint: Rust checks (…)
   │  ├─ test: pytest (parallel) (…)
   │  └─ test: pytest (serial) (…)
   ├─ clippy (…)
   └─ tests (…)
```

### API budget (per invocation)

- `--max-github-api-calls N` (default: **100**)

### Examples

```bash
# HTML output (typical)
python3 html_pages/show_commit_history.py \
  --repo-path ~/nvidia/dynamo_latest \
  --max-commits 100 \
  --output ~/nvidia/dynamo_latest/index.html \
  --max-github-api-calls 100

# Cache-only style run (skip GitLab network)
python3 html_pages/show_commit_history.py \
  --repo-path ~/nvidia/dynamo_latest \
  --max-commits 100 \
  --skip-gitlab-fetch \
  --output ~/nvidia/dynamo_latest/index.html
```

---

## update_html_pages.sh (cron-friendly wrapper)

This script runs one or more generators and writes outputs via atomic replacement.

### Logs

- `update_html_pages.sh` writes per-run logs under `~/nvidia/logs/YYYY-MM-DD/` (when run with `NVIDIA_HOME=~/nvidia`).
- `dynamo-utils/cron_log.sh` captures stdout/stderr for a job into `~/nvidia/logs/YYYY-MM-DD/<job>.log`.

### Troubleshooting: `update_html_pages.sh` “runs too quickly”

This almost always means **a generator crashed early** (Python exception / ImportError), so the wrapper script exits quickly.

Do this:
- **Check the per-generator logs** (they usually tell you which script crashed):
  - `tail -200 html_pages/show_local_branches.log`
  - `tail -200 html_pages/show_commit_history.log`
  - `tail -200 html_pages/resource_report.log`
- **Run the failing generator directly** to see the traceback:
  - `python3 html_pages/show_local_branches.py --fast`
  - `python3 html_pages/show_commit_history.py --fast`
- **Common root cause**: refactors in `common_dashboard_lib.py` removed/renamed an exported symbol that another script still imports.
  - Fix by updating the importer(s), or by keeping a small back-compat constant/export if appropriate.

### Typical cron schedule

```cron
# Dashboards:
# - Full update every 30 minutes
0,30 * * * * NVIDIA_HOME=$HOME/nvidia $HOME/nvidia/dynamo-utils/cron_log.sh update_html_pages_full $HOME/nvidia/dynamo-utils/html_pages/update_html_pages.sh --show-local-branches --show-commit-history

# - Cache-heavy runs between full updates
8-59/4 * * * * NVIDIA_HOME=$HOME/nvidia SKIP_GITLAB_FETCH=1 $HOME/nvidia/dynamo-utils/cron_log.sh update_html_pages_cached $HOME/nvidia/dynamo-utils/html_pages/update_html_pages.sh --show-local-branches --show-commit-history

# Resource report:
* * * * * NVIDIA_HOME=$HOME/nvidia $HOME/nvidia/dynamo-utils/cron_log.sh resource_report $HOME/nvidia/dynamo-utils/html_pages/update_html_pages.sh --show-local-resources
```

---

## Caching (important)

All persistent caches live under:

- `~/.cache/dynamo-utils/` (default)
- or `$DYNAMO_UTILS_CACHE_DIR/` (override)

Key caches used by the dashboards:

- **PR list**: `pulls/`
- **PR check-runs rows**: `pr-checks/`
- **Enriched PRInfo (per-PR, updated_at-keyed)**: `pr-info/pr_info.json`
- **PR updated_at probe (search/issues)**: `search-issues/search_issues.json`
- **Required checks (branch protection)**: `required-checks/required_checks.json`
- **Actions job details (steps/timings; used for phase breakdown)**: `actions-jobs/actions_jobs.json`
- **Actions job raw log redirect URLs**: `raw-log-urls/`
- **Downloaded raw log text**: `raw-log-text/` (+ `raw-log-text/index.json`)

### PRInfo “0 API” reuse for unchanged PRs

For `show_local_branches.py`, the biggest win is skipping per-PR enrichment when a PR hasn’t changed.

We do this by:

- probing `updated_at` for the target PR list via **one** `search/issues` request (batched)
- caching the fully enriched `PRInfo` object keyed by **(pr_number, updated_at)**

If the PR’s `updated_at` matches what we already cached, we reuse the cached `PRInfo` and do **zero**
per-PR network calls.

In cache-only / budget-exhausted mode, the dashboards will reuse cached `PRInfo` even if TTLs for other
short-lived caches have expired.

---

## Statistics: interpreting “API call types”

Dashboards report GitHub REST usage by **label** (see `common.py` → `_rest_label_for_url`).

Common labels you’ll see:

- `rate_limit`
- `search_issues`
- `pulls_list`
- `pull_request`
- `check_runs`
- `actions_run`
- `actions_run_jobs`
- `actions_job_status`
- `actions_job_logs_zip`
- `pr_review_comments`
- `commit_pulls`
- `required_status_checks`
- `repos_<resource>` (fallback bucket for other `/repos/.../<resource>` endpoints)

If you see calls happening “even though nothing changed”, the usual causes are:

- short TTL caches for check-runs (intentionally refreshes recent/unsettled CI)
- new Actions runs due to reruns (new run_id → new `actions_run` metadata)
- missing raw logs (failure triggers log materialization)
- phase breakdown for `Build and Test - dynamo` (may fetch job details to read `steps[]`)

---

## Concrete API call graph (branches dashboard)

This section is a **step-by-step, concrete example** of what the branches dashboard does for *one* local branch.
It shows:

- which GitHub APIs are called
- what we read from the responses (example keys)
- what follow-on calls happen because of those values
- how many calls happen in best/worst cases
- whether each call is cacheable in this codebase today

### Example setup (REAL: `keivenchang/DIS-1200__refactor-out-dev-from-Dockerfiles`)

This is a real branch/PR from your environment:

- Local branch: `keivenchang/DIS-1200__refactor-out-dev-from-Dockerfiles`
- Matched open PR: `#5050` in `ai-dynamo/dynamo`
- PR URL: `https://github.com/ai-dynamo/dynamo/pull/5050`
- PR head SHA: `5065cf08a1caffbfeb123aa3258271344980af95`
- `--max-github-api-calls 100`

Important nuance: the script does **not** call GitHub “per branch” first. It calls GitHub **per repo** (list open PRs once),
then maps PRs to local branches, then does per-PR enrichment.

### Step 0: quota check (optional / best-effort)

**Call (type `rate_limit`)**

- `GET /rate_limit`

**Used for**

- Showing quota info in the HTML “Statistics” section
- Deciding whether to start in cache-only mode

**Cacheable?**

- Not persisted; called for observability. (Could be cached, but isn’t important.)

**Calls**

- Usually 1–2 per invocation (we may call it more than once for display).

---

### Step 1: map local branches → open PRs (per repo)

**Call (type `pulls_list`)**

- `GET /repos/ai-dynamo/dynamo/pulls?state=open&per_page=100&page=1`

**We read from the response**

Response is a list of PR dicts. We primarily use:

```json
{
  "number": 5050,
  "state": "open",
  "html_url": "https://github.com/ai-dynamo/dynamo/pull/5050",
  "head": {
    "ref": "keivenchang/DIS-1200__refactor-out-dev-from-Dockerfiles",
    "sha": "5065cf08a1caffbfeb123aa3258271344980af95"
  },
  "base": { "ref": "main" }
}
```

**Follow-on effects**

- If `head.ref` matches a local branch name, we treat that branch as “has an open PR”
- We now need per-PR enrichment (next step)

**Cacheable?**

- Yes. Cached in memory + disk by `GitHubAPIClient.list_pull_requests(...)`
- TTL: **60s** (1 minute) (`DEFAULT_OPEN_PRS_TTL_S`) because open PR lists can change

**Calls**

- **Best case**: 0 (served from cache)
- **Worst case**: 1 per page (usually 1; more only if >100 open PRs)

---

### Step 2: per-PR enrichment for one matched PR (#5050)

This happens in `GitHubAPIClient._pr_info_from_pr_data(...)`.

Before we do per-PR enrichment, we try to avoid it entirely:

#### 2A) Probe `updated_at` for PRs we care about (batched)

**Call (type `search_issues`)**

- `GET /search/issues?q=repo:ai-dynamo/dynamo type:pr number:5050 number:4578 number:4790`

**We read from the response**

Each item includes:

```json
{
  "number": 5050,
  "updated_at": "2025-12-25T07:30:00Z",
  "pull_request": { "url": "https://api.github.com/repos/ai-dynamo/dynamo/pulls/5050" }
}
```

For the other PRs in the same call, the items look the same shape:

```json
{
  "number": 4578,
  "updated_at": "2025-11-20T18:12:00Z",
  "pull_request": { "url": "https://api.github.com/repos/ai-dynamo/dynamo/pulls/4578" }
}
```

```json
{
  "number": 4790,
  "updated_at": "2025-10-02T09:41:00Z",
  "pull_request": { "url": "https://api.github.com/repos/ai-dynamo/dynamo/pulls/4790" }
}
```

**Follow-on effects**

- If `updated_at` matches the cached PRInfo entry, we reuse the cached PRInfo and do **0 per-PR network calls**
  (no PR fetch, no check-runs fetch, no comments fetch, etc).

**Cacheable?**

- Yes.
- TTL: **60s** (1 minute) (default `get_pr_updated_at_via_search_issues(..., ttl_s=60)`)
- Cache-only behavior: reuse stale disk cache if present.

#### 2B) Fetch check-runs data (so we can compute CI state)

**Calls**

- (type `pull_request`) `GET /repos/ai-dynamo/dynamo/pulls/5050`
- (type `check_runs`) `GET /repos/ai-dynamo/dynamo/commits/5065cf08a1caffbfeb123aa3258271344980af95/check-runs?per_page=100`

**We read from the responses**

From `/pulls/5050`:

```json
{
  "number": 5050,
  "base": { "ref": "main" },
  "head": { "sha": "5065cf08a1caffbfeb123aa3258271344980af95" }
}
```

From `/commits/<sha>/check-runs`:

```json
{
  "total_count": 31,
  "check_runs": [
    {
      "name": "Validate PR title and add label",
      "status": "completed",
      "conclusion": "success",
      "html_url": "https://github.com/ai-dynamo/dynamo/actions/runs/20661331612/job/59324346738"
    },
    {
      "name": "deploy-test-vllm (disagg_router)",
      "status": "completed",
      "conclusion": "failure",
      "html_url": "https://github.com/ai-dynamo/dynamo/actions/runs/20500122691/job/58907618904"
    }
  ]
}
```

**Cacheable?**

- The *rendered check rows* are cached by `GitHubAPIClient.get_pr_checks_rows(...)` in `pr-checks/`.
- TTL: **300s** (5 minutes) (default `get_pr_checks_rows(..., ttl_s=300)`)
- Cache-only behavior: reuse stale disk cache if present (even if TTL expired).
- However, this particular sub-step uses an internal fetch (`_fetch_pr_checks_data`) and may still hit the network even if the UI is unchanged.

#### 2C) Required checks (best-effort; long-lived cache)

**Calls**

- (type `pull_request`) `GET /repos/ai-dynamo/dynamo/pulls/5050` (to read PR metadata like `node_id`)
- (best-effort) branch protection required checks can be queried, but is often 403 depending on token permissions
- fallback (preferred): GitHub GraphQL “merge box” required-ness (`statusCheckRollup … isRequired(pullRequestId: …)`)
  via `gh api graphql`

**We read from the responses**

From required checks:

```json
{
  "contexts": ["lint", "build"],
  "checks": [{"context": "Build and Test - dynamo", "app_id": 12345}]
}
```

**Cacheable?**

- Yes. Cached per PR number in `github_required_checks.json` (long-lived; required-ness changes rarely).
- We also maintain a branch-protection cache in `required-checks/required_checks.json` when accessible, but we do not
  rely on it.

#### 2D) “Unresolved conversations” approximation (review comments)

**Call (type `pr_review_comments`)**

- `GET /repos/ai-dynamo/dynamo/pulls/5050/comments`

**We read from the response**

We count comments without `in_reply_to_id`:

```json
[
  {"id": 1, "in_reply_to_id": null},
  {"id": 2, "in_reply_to_id": 1}
]
```

**Cacheable?**

- Not cached by this code path today.

---

### Step 3 (only on failures): stable raw log + snippet materialization

If a check is failed and we want a stable `[cached raw log]` link + snippet, we may do:

#### 4A) Ensure job is completed

**Call (type `actions_job_status`)**

- `GET /repos/ai-dynamo/dynamo/actions/jobs/53317461976`

We read:

```json
{"id": 53317461976, "status": "completed"}
```

**Cacheable?**

- Yes (in-memory only).
- TTL: **120s** (default `get_actions_job_status(..., ttl_s=120)`).

#### 4B) Download job logs (zip) and cache the extracted text

**Call (type `actions_job_logs_zip`)**

- `GET /repos/ai-dynamo/dynamo/actions/jobs/53317461976/logs`

We store:

- `raw-log-text/53317461976.log` (text) + `raw-log-text/index.json` metadata

**Cacheable?**

- Yes. Persisted. Once downloaded, subsequent runs reuse the local file and do **not** re-download.
- Raw log redirect URL TTL: **3600s** (1 hour) (`DEFAULT_RAW_LOG_URL_TTL_S`) (signed URLs expire quickly).
- Raw log text TTL: **30 days** (`DEFAULT_RAW_LOG_TEXT_TTL_S`) (whether we consider refreshing the downloaded content).

---

### Step 4C (best-effort): job details for phase breakdown (`Build and Test - dynamo`)

If we want per-phase ✓/✗ + timings (instead of guessing from raw logs), we fetch job details:

**Call (type `actions_job_status`)**

- `GET /repos/ai-dynamo/dynamo/actions/jobs/53317461976`

We read:

```json
{
  "id": 53317461976,
  "status": "completed",
  "conclusion": "failure",
  "steps": [
    {"name": "Build Image", "status": "completed", "conclusion": "success"},
    {"name": "Rust checks", "status": "completed", "conclusion": "failure"}
  ]
}
```

**Cacheable?**

- Yes. Persisted to `actions-jobs/actions_jobs.json` (memory + disk).
- TTL: **600s** (default `get_actions_job_details_cached(..., ttl_s=600)`).
  - (This is intentionally shorter than run metadata; step timings can change during an in-progress run.)

### Call counts summary for this example (single PR)

Assume:

- PR has 25 check-runs
- They reference 3 unique Actions runs (`run_id`s)
- 2 failed jobs need logs/snippets

**Best-case (warm caches, nothing new)**

- `pulls_list`: 0
- `check_runs`: 0
- `actions_run`: 0
- `actions_job_logs_zip`: 0
- `rate_limit`: ~1–2
- (others): 0

Total: **~1–2 calls**

**Worst-case (cold caches)**

- `rate_limit`: 1–2
- `pulls_list`: 1
- `pull_request`: 2 (PR details fetched twice in current code paths)
- `check_runs`: 2 (check-runs fetched via multiple helpers in current code paths)
- `required_status_checks`: 1
- `pr_review_comments`: 1
- `actions_run`: 3 (one per unique run_id)
- `actions_job_status`: 2
- `actions_job_logs_zip`: 2

Total: **~15–16 calls** for one PR.

Multiply by “number of PRs shown” to estimate the run’s ceiling, and note that the **API budget** caps the total
per invocation (`--max-github-api-calls`), switching to cache-only mode when exhausted.


