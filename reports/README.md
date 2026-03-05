# Reports

Scripts for analyzing the dynamo repository from three angles:
codebase shape, contributor activity, and development process health.

## Scripts

### `analyze_repo.py` — Codebase over time

Snapshots the repo at regular intervals (monthly by default) and tracks:
- Lines of code by language (Python, Rust, docs)
- Rust file and test counts
- Commit volume and author counts
- Lines added/deleted between snapshots

Uses read-only git plumbing (no checkout). Safe to run on bare repos.

```bash
python3 reports/analyze_repo.py \
    --repo-path ~/dynamo/dynamo_ci \
    --loc --rust --commits \
    --start 2024-06-01 --end 2026-03-01 \
    --interval month
```

### `contributor_stats.py` — Contributor rankings

Per-contributor breakdown for a time window:
- Commits, lines added/deleted, net lines
- Rankings by commits and lines changed
- Averages per contributor

```bash
python3 reports/contributor_stats.py --repo ~/dynamo/dynamo_ci --days 90
```

### `pr_ci_report.py` — PR and CI health

Tracks development process metrics using GitHub API:
- PR submission and merge rates (code vs non-code split)
- CI failure rates per PR
- Manual re-trigger frequency (developer pain signal)
- Average time-to-merge
- Top failing workflows

Caches all API responses on disk (60-day TTL for runs, 1h for PR lists).
Second runs for the same window cost zero API calls.

```bash
python3 reports/pr_ci_report.py \
    --repo ai-dynamo/dynamo \
    --days 60 --bucket week \
    --output-dir ~/dynamo/temp \
    --max-github-api-calls 5000
```

Output: terminal text, JSON, and HTML reports.

## What each script answers

| Question | Script |
|----------|--------|
| How is the codebase growing? | `analyze_repo.py` |
| Who are the top contributors? | `contributor_stats.py` |
| Is CI stable? How long do PRs take? | `pr_ci_report.py` |
