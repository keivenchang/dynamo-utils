# Weekly CI Stats Generation

How to generate, update, and maintain the weekly `.txt` stat files and the
`2026-Q1-summary.md` rollup.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Weekly Update Checklist](#weekly-update-checklist)
3. [What Each Stat File Contains](#what-each-stat-file-contains)
4. [Data Sources and Caches](#data-sources-and-caches)
5. [Error Categorization](#error-categorization)
6. [Re-run % and Re-run/PR](#re-run--and-re-runpr)
7. [Pre-merge Duration](#pre-merge-duration-pre-merge-min-column)
8. [Table Formatting Convention](#table-formatting-convention-bold--arrows)
9. [Repository Analysis (LoC, Rust Tests, Pytest Growth)](#repository-analysis-loc-rust-tests-pytest-growth)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
cd ~/dynamo/dynamo-utils.dev

# Generate one week (Monday date):
python3 -u ci_log_errors/gen_weekly_stats.py --week 2026-03-23

# Generate multiple weeks at once:
python3 -u ci_log_errors/gen_weekly_stats.py --week 2026-03-09 --week 2026-03-16

# Long runs (>5 min): use nohup + await_output.sh
nohup python3 -u ci_log_errors/gen_weekly_stats.py --week 2026-03-23 \
    > /tmp/gen_weekly_stats.log 2>&1 &
CMD_PID=$!
~/dynamo/dynamo-utils.dev/await_output.sh -t 600 \
    -s "Written to" -s "Traceback" -p $CMD_PID \
    -l /tmp/await_weekly.log -- tail -f /tmp/gen_weekly_stats.log
```

Output: `ci_log_errors/stats/YYYY-MM-DD.txt` (one file per week-start Monday).

---

## Weekly Update Checklist

End-to-end steps to publish a new week's data in `2026-Q1-summary.md`.
Each step references a section below for details.

1. **Generate the `.txt` file** -- run `gen_weekly_stats.py --week YYYY-MM-DD`
   (see [Quick Start](#quick-start)). This produces Development Metrics, error
   distributions, and workflow stats.

2. **Compute Re-run % and Re-run/PR** -- run the one-liner in
   [Re-run % and Re-run/PR](#re-run--and-re-runpr), changing `since`/`until`
   to your week. `gen_weekly_stats.py` undercounts this metric.

3. **Compute Pre-merge (min)** -- run the one-liner in
   [Pre-merge Duration](#pre-merge-duration-pre-merge-min-column), changing
   `since`/`until` to your week. This metric is not in the `.txt` file.

4. **Add a row to the Dev Metrics table** (line ~20 in `2026-Q1-summary.md`):
   - PRs, PRs Merged, Avg Lines/PR, Median Lines/PR, Days to merge, Push/PR,
     PR Fail %, Post-merge (min) -- from the `.txt` file
   - Re-run %, Re-run/PR -- from step 2
   - Pre-merge (min) -- from step 3

5. **Apply bold/arrows** only to the newest completed week (see
   [Table Formatting Convention](#table-formatting-convention-bold--arrows)).

6. **Update Key Takeaways** (line ~7) if metrics shifted significantly.

7. **Update the Period line** (line ~62) to extend the end date.

---

## What Each Stat File Contains

1. **Development Metrics** -- PR counts, merge velocity, re-run rates, CI duration.
2. **Error Rate** -- group error hits as % of total post-merge runs.
3. **Workflow-Level Overview** -- success/failure/cancelled counts.
4. **Failures by Workflow** -- per-workflow failure counts and rates.
5. **Failure Distribution by Group** -- errors bucketed into Tests, Docker/build,
   K8s/Helm, Network, Infra/system, Auth, Gates/policy, Docs, Build.
6. **Failure Distribution Detail** -- individual error categories ranked by hit count.

---

## Data Sources and Caches

All caches live under `~/.cache/dynamo-utils/`.

| Cache file | Key | Written by | TTL | What it stores |
|---|---|---|---|---|
| `actions_runs_list.json` | per-run | dashboard + gen_weekly_stats | 60 days | Workflow runs (run ID, conclusion, SHA, timestamps, event, branch) |
| `pulls_list.json` | bulk list | dashboard + gen_weekly_stats | 1 hour | PR metadata (number, created/merged dates, state, head SHA). Does NOT include `additions`/`deletions`. |
| `weekly_stats_jobs.json` | run_id → [job_ids] | gen_weekly_stats | indefinite | Failed job IDs per run (3-tier: this cache → `actions_jobs.json` → API) |
| `weekly_stats_scan.json` | job_id → [categories] | gen_weekly_stats | indefinite | Categorized error lists per job (avoids re-scanning logs) |
| `raw-log-text/{job_id}.log` | job_id | dashboard + gen_weekly_stats | 180 days | Raw job log text |
| `raw-log-text/index.json` | job_id → metadata | dashboard + gen_weekly_stats | -- | Must have `"completed": true` or cron pruner deletes the `.log` file |

### Key APIs

| Metric | API endpoint | Notes |
|---|---|---|
| Post-merge runs | `GET /repos/ai-dynamo/dynamo/actions/runs?event=push&branch=main` | Cached in `actions_runs_list.json` |
| Pre-merge runs | Same endpoint with `event=pull_request`, `pull_request_target` | Also cached |
| Failed job IDs | `GET /repos/.../actions/runs/{run_id}/jobs?per_page=100` | 3-tier cache lookup |
| Raw job logs | `GET /repos/.../actions/jobs/{job_id}/logs` | 302 redirect to Azure Blob |
| Line counts | `GET /repos/.../pulls/{number}` (individual PR) | Returns `additions`, `deletions`; ~100-250 calls/week |

### Cache structure gotcha

Cache entries in `actions_runs_list.json` are wrapped:
`{"ts": ..., "run": {...actual run data...}}`. Always unwrap with
`v.get("run", v)` -- accessing fields directly on `v` returns `None`.

### Raw log pruning

The production cron runs `prune_partial_raw_log_caches()` every 2-7 minutes
(`html_pages/common_dashboard_runtime.py`). It deletes any `raw-log-text/*.log`
file that either:

1. Has no entry in `index.json` with `completed: true` AND is older than 180 days
2. Is in a dashboard-served dir without a matching completed index entry

**If you download logs outside the dashboard pipeline** (e.g., from
`gen_weekly_stats.py`), you MUST register them in `index.json` with
`{"ts": <epoch>, "bytes": <size>, "completed": true}` or they will be pruned.

`_download_missing_logs()` in `gen_weekly_stats.py` handles this automatically
(flushes `index.json` after every download).

---

## Error Categorization

- **Engine**: `ci_log_errors/engine.py` → `categorize_error_log_lines(lines)`
- **Bucket map**: `gen_weekly_stats.py` → `BUCKET_MAP` (raw category → broad bucket)
- **Group map**: `gen_weekly_stats.py` → `GROUP_MAP` (raw category → display group)
- Groups: Tests, Docker/build, K8s/Helm, Network, Infra/system, Auth,
  Gates/policy, Docs, Build

---

## Re-run % and Re-run/PR

These metrics measure manual workflow re-runs (someone clicking "Re-run jobs"
in the GitHub Actions UI after a failure).

### Detection

GitHub Actions tracks re-runs via the `run_attempt` field:
- `run_attempt=1` → original run
- `run_attempt=2` → first re-run
- `run_attempt=3` → second re-run, etc.

### Which events to include (CRITICAL)

**MUST include `push:pull-request/NNN` events.** This is where the heavy CI
workflows run (NVIDIA Dynamo Github Validation, PR, Test Lab) and where
~95% of manual re-runs happen. The lightweight `pull_request` events (DCO,
copyright, lint) almost never get re-run.

`gen_weekly_stats.py` currently only uses `pull_request` + `pull_request_target`
events -- this **undercounts by 10-20x**. Use the one-liner below instead.

### Formulas

- **Re-run %** = (PRs with at least one `run_attempt > 1` run) / (all PRs with CI) x 100
- **Re-run/PR** = total re-runs / PRs that had re-runs (among affected PRs only)

### One-liner: compute Re-run for a week

Change `since` and `until` to your target week (Monday 00:00 to Sunday 23:59).

```bash
python3 -c "
import json, re
from datetime import datetime, timezone
from collections import defaultdict
d = json.load(open('$HOME/.cache/dynamo-utils/actions_runs_list.json'))
PR_RE = re.compile(r'^pull-request/(\d+)\$')
parse = lambda s: datetime.fromisoformat(s.replace('Z','+00:00')) if s else None
since = datetime(2026, 3, 16, tzinfo=timezone.utc)   # <-- CHANGE: week start (Monday)
until = datetime(2026, 3, 22, 23, 59, 59, tzinfo=timezone.utc)  # <-- CHANGE: week end (Sunday)
prs_ci, pr_reruns = set(), defaultdict(int)
for v in d['items'].values():
    r = v.get('run', v)
    c = parse(r.get('created_at',''))
    if not c or c < since or c > until: continue
    ev, br = r.get('event',''), r.get('head_branch','')
    prn = None
    m = PR_RE.match(br)
    if m: prn = int(m.group(1))
    elif r.get('pr_numbers'): prn = r['pr_numbers'][0]
    if prn is None: continue
    if ev in ('pull_request','pull_request_target') or (ev=='push' and PR_RE.match(br)):
        prs_ci.add(prn)
        att = int(r.get('run_attempt', 1))
        if att > 1: pr_reruns[prn] += att - 1
n = len(prs_ci); nr = len(pr_reruns); tot = sum(pr_reruns.values())
print(f'PRs={n} w/rerun={nr} ({nr/n*100:.0f}%) reruns/affected={tot/nr:.1f}' if nr else f'PRs={n} reruns=0')
"
```

Example output: `PRs=142 w/rerun=24 (17%) reruns/affected=2.4`
→ Re-run % = 17%, Re-run/PR = 2.4

---

## Pre-merge Duration (Pre-merge (min) column)

This metric is **not** produced by `gen_weekly_stats.py`. It requires a separate
computation from the local workflow runs cache.

### Which runs to include

Only `event="push"` runs where `head_branch` matches `^pull-request/\d+$`.

These are the "fork-PR CI" runs: a merge bot pushes PR code to a repo-owned
`pull-request/NNN` branch so CI can run with repo secrets. This is where the
heavy workflows run (NVIDIA Dynamo Github Validation, PR, NVIDIA Test Lab
Validation).

**Do NOT include** lightweight `event="pull_request"` or `event="pull_request_target"`
runs (DCO, Copyright, Docs link check). They run in 1-4 minutes and dilute the
average without reflecting actual developer wait time.

### Duration formula

```
per-run duration = updated_at - created_at    (includes queue time)
cap: exclude runs > 600 minutes               (hung/zombie jobs)
per-commit: max(all run durations for that head_sha)
weekly avg: mean(per-commit max values)
```

Uses `created_at` (not `run_started_at`) because queue time is part of the
developer's actual wait. Grouped by `head_sha` because workflows for the same
commit run in parallel -- the developer waits for the slowest one.

### One-liner: compute Pre-merge for a week

Change `since` and `until` to your target week (Monday 00:00 to Sunday 23:59).

```bash
python3 -c "
import json, re, statistics
from datetime import datetime, timezone
from collections import defaultdict
d = json.load(open('$HOME/.cache/dynamo-utils/actions_runs_list.json'))
PR_RE = re.compile(r'^pull-request/\d+\$')
parse = lambda s: datetime.fromisoformat(s.replace('Z','+00:00')) if s else None
since = datetime(2026, 3, 16, tzinfo=timezone.utc)   # <-- CHANGE: week start (Monday)
until = datetime(2026, 3, 22, 23, 59, 59, tzinfo=timezone.utc)  # <-- CHANGE: week end (Sunday)
sha_durs = defaultdict(list)
for v in d['items'].values():
    r = v.get('run', v)
    if r.get('event')=='push' and PR_RE.match(r.get('head_branch','')):
        c, u = parse(r.get('created_at','')), parse(r.get('updated_at',''))
        if c and u and u>c and since<=c<=until:
            dur=(u-c).total_seconds()/60
            if dur<=600: sha_durs[r['head_sha']].append(dur)
maxes=[max(ds) for ds in sha_durs.values() if ds]
print(f'avg={statistics.mean(maxes):.0f} median={statistics.median(maxes):.0f} commits={len(maxes)}')
"
```

Example output: `avg=57 median=42 commits=187`
→ Pre-merge (min) = 57

### Known anomaly: NVIDIA Test Lab Validation spikes

Test Lab Validation normally runs in 5-19 minutes. In some weeks (e.g., Mar 09
2026) it spiked to 200-600 min/run due to infrastructure issues, inflating the
weekly average. When this happens, note the median alongside the average in the
summary (footnote with `‡‡`).

---

## Table Formatting Convention (bold / arrows)

Bold + arrow annotations (`**value&nbsp;↑**` or `**value&nbsp;↓**`) highlight
notable week-over-week changes in `2026-Q1-summary.md`. Rules:

- **Only apply bold/arrows to the most recent completed week** (the "current"
  week being published). Older weeks that were annotated in a previous publish
  keep their formatting.
- **Do NOT bold/arrow weeks that are still accumulating data** or weeks beyond
  the most recent completed one.
- Use `↑` when a metric improved (e.g., more PRs merged, lower re-run %)
  and `↓` when it worsened, relative to the prior week.
- If there is no meaningful change, leave the value plain.
- The week label cell itself (`Mar&nbsp;02`) gets bolded when the row has
  annotations; otherwise it stays plain.

Example: if Feb 23 is the latest completed week, bold/arrow its notable cells.
Mar 02 and later remain plain until they become the "current" week.

---

## Repository Analysis (LoC, Rust Tests, Pytest Growth)

Script: `reports/analyze_repo.py`

Generates three types of tables from git history using read-only plumbing
commands (no checkout required for LoC/Rust; worktrees needed for pytest collect).

### LoC + Rust Tests + Commits (monthly)

```bash
cd ~/dynamo/dynamo-utils.dev
python3 -u reports/analyze_repo.py \
    --repo ~/dynamo/dynamo3 --start 2025-01-01 \
    --loc --rust --commits
```

Produces: Python LoC, Rust LoC, Docs LoC, Total LoC, Lines Added/Deleted,
Rust Files, Rust Tests, Commits, Cumulative commits, Authors.

### LoC + Rust Tests + Commits (daily)

```bash
python3 -u reports/analyze_repo.py \
    --repo ~/dynamo/dynamo3 --start 2026-02-08 --daily \
    --loc --rust --commits
```

Add `--end YYYY-MM-DD` to cap the range (defaults to today).

### Pytest Growth (monthly, with collect)

Requires a **vllm dev container** with dynamo installed for accurate
`pytest --collect-only` counts. Without it, falls back to AST parsing
which undercounts parametrized tests.

```bash
# 1. Make sure dynamo3 is up to date
cd ~/dynamo/dynamo3 && git checkout main && git pull

# 2. Start a temporary vllm container with the repo mounted read-write
docker run -d --name analyze_vllm \
    -v ~/dynamo/dynamo3:/workspace/dynamo \
    -v ~/dynamo/dynamo-utils.dev:/utils \
    -v ~/.cache/dynamo-utils:/root/.cache/dynamo-utils \
    dynamo:9BEBB9.06f17011b-vllm-local-dev-cuda12.9-amd64 \
    sleep 3600

# 3. Install the project (needed for pytest --collect-only)
docker exec analyze_vllm bash -c \
    "cd /workspace/dynamo && pip install -e . -q"

# 4. Run the analysis
docker exec analyze_vllm bash -c \
    "cd /utils && python3 -u reports/analyze_repo.py \
        --repo /workspace/dynamo --start 2025-01-01 \
        --python --collect"

# 5. Cleanup
docker rm -f analyze_vllm
```

Update the image tag (`dynamo:XXXXX...`) to whatever the latest vllm-local-dev
image is. Find it with:

```bash
docker images --format "{{.Repository}}:{{.Tag}}" | grep vllm-local-dev
```

### Key flags

| Flag | What it does |
|---|---|
| `--loc` | Lines of code by language (Python, Rust, Docs, Total) |
| `--rust` | Rust test counts (`#[test]` / `#[tokio::test]`) and `.rs` file counts |
| `--commits` | Commit volume, cumulative commits, unique authors per period |
| `--python` | Pytest test files, test counts, marker breakdown |
| `--collect` | Use `pytest --collect-only` via git worktrees (falls back to AST) |
| `--daily` | Daily snapshots (default: monthly on 1st of month) |
| `--weekly` | Weekly snapshots (every Monday) |
| `--start DATE` | Start date (YYYY-MM-DD) |
| `--end DATE` | End date (default: today) |
| `--repo PATH` | Path to the git repository |

### Caching

Results are cached in `~/.cache/dynamo-utils/analyze_repo/` keyed by
(analysis type, commit SHA, parameters). Git SHAs are immutable so cached
results never go stale. Re-runs are fast (cache hits shown in output footer).

### Gotchas

- **`pytest --collect-only` fails for old commits**: The container has packages
  pinned to the latest commit. Older commits may have incompatible imports,
  so they fall back to AST (marked with `*`). Historical AST rows are still
  accurate for `def test_*` counts but miss parametrized/dynamically generated
  tests.
- **Repo must be on `main`**: The script uses `git log` to find the closest
  commit to each date. If the repo is on a feature branch, `git pull` will
  pull that branch. Always `git checkout main && git pull` first.
- **Read-write mount required for `--collect`**: Git worktrees need write
  access to the repo's `.git/` directory. Mount without `:ro`.
- **Large Docs LoC swings**: Docs LoC includes `.md`, `.json`, `.yaml`, etc.
  A single large schema file addition/removal can shift totals by hundreds of
  thousands of lines (e.g., Mar 16 2026: -420K Docs LoC from cleanup).

---

## Troubleshooting

**"Logs on disk: X/Y" is much lower than expected**
→ The cron pruner deleted unregistered files. Verify `index.json` entries exist
for downloaded logs. Re-run the script (it will re-download missing logs and
register them).

**"Download issues: N errors"**
→ Transient GitHub API failures (`RemoteDisconnected`). Re-running usually
recovers most. Some job IDs have permanently unavailable logs.

**Script hangs at "Fetching pre-merge CI runs..."**
→ DNS resolution failure or GitHub API rate limit. The script has retry logic
with exponential backoff. Wait or check `~/.config/gh/hosts.yml` token validity.

**Re-runs are slow**
→ First run for a week fetches job IDs and logs via API (5-10 min). Subsequent
re-runs hit the local caches and finish in 1-3 min.

**Re-run % looks suspiciously low (< 5%)**
→ You're probably only counting `pull_request` events. Make sure you include
`push:pull-request/NNN` events -- see [Which events to include](#which-events-to-include-critical).
