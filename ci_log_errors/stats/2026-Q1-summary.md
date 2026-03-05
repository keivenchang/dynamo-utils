Post-Merge CI Failure Analysis — 2026 Q1 Summary
===================================================
Period: 2026-01-24 to 2026-03-03 (6 weeks + 2 partial weeks)
Source: 1,987 raw post-merge job logs + GitHub Actions workflow run data
Methodology:
  - Each failed job log assigned to exactly ONE bucket (no double-counting)
  - Priority: Testing > Infra > Build
  - Catch-alls excluded: vllm-error, sglang-error, trtllm-error, exit-139-sigsegv
  - "% of runs" = failures / total post-merge runs (including successes)
  - Uncategorized = logs where the categorizer found a snippet but could not
    match a known error pattern (~34% of failed logs)

===========================================================================
WEEKLY BREAKDOWN (% of all post-merge runs)
===========================================================================

Week         | Runs  | Testing | Infra   | Build   | Uncat   | WF Fail%
-------------|-------|---------|---------|---------|---------|--------
Jan 19*      |   458 |   1.3%  |   0.0%  |   0.0%  |   0.4%  |   6.8%
Jan 26       |   693 |   8.7%  |  13.4%  |   2.3%  |  11.3%  |  23.1%
Feb 2        |   867 |   4.4%  |  16.8%  |   0.2%  |   7.2%  |  25.0%
Feb 9        |   934 |   5.5%  |  17.7%  |   3.2%  |  11.6%  |  23.6%
Feb 16       |   794 |  41.6%  |  12.7%  |  11.3%  |  29.1%  |  28.2%
Feb 23       | 1,085 |   8.0%  |   4.0%  |   0.2%  |  13.1%  |  15.4%
Mar 2**      |   215 |  11.2%  |  13.0%  |   0.0%  |  24.2%  |  29.8%
-------------|-------|---------|---------|---------|---------|--------
TOTAL        | 5,046 |  11.8%  |  11.4%  |   2.8%  |  13.4%  |  22.1%

*  Jan 19 = partial week (Sat-Sun only, 8 logs)
** Mar 2  = partial week (Mon-Tue, 2 days)

===========================================================================
BEFORE / AFTER FEB 23 (improvement snapshot)
===========================================================================

| Category      | Before Feb 23 (5 wks, 3,746 runs) | After Feb 23 (2 wks, 1,300 runs) | Change |
|---------------|-----------------------------------|----------------------------------|--------|
| Testing       | 12.9% of runs                     | 8.5% of runs                     | -4.4pp |
| Infra         | 13.5% of runs                     | 5.5% of runs                     | -8.0pp |
| Build         | 3.7% of runs                      | 0.2% of runs                     | -3.5pp |
| Uncategorized | 12.8% of runs                     | 15.0% of runs                    | +2.2pp |
| **WF failure rate** | **22.7%**                    | **17.8%**                        | **-4.9pp** |

All three failure categories improved after Feb 23. WF failure rate
dropped from 22.7% to 17.8%.

===========================================================================
KEY IMPROVEMENTS
===========================================================================

Testing Improvements
---------------------
- Testing failures dropped from 12.9% to 8.5% of all runs after Feb 23
  (-4.4 percentage points, a 34% relative reduction).
- The Feb 16 spike (41.6% of runs) was a one-week anomaly driven by an
  etcd-error surge (176 hits, up from 3 the prior week) that cascaded into
  pytest failures. By Feb 23, testing was back to 8.0%.
- pytest-error remains the single largest testing category but has stabilized
  in the 5-8% of runs range post-Feb 23 (vs 41.6% during the spike).
- pytest-timeout-error appeared in Mar 2 (21 hits, 20% of that week's logs),
  suggesting a new slow-test issue worth monitoring.

Infra Improvements
-------------------
- Infra failures dropped from 13.5% to 5.5% of all runs after Feb 23
  (-8.0 percentage points, a 59% relative reduction).
- This is the largest absolute improvement of any category.
- network-error and k8s-error, which were consistently 13-18% of runs
  through Jan 26 - Feb 9, dropped to 4.0% the week of Feb 23.
- etcd-error, which spiked to 176 hits during Feb 16, returned to
  baseline (31 hits Feb 23, 0 hits Mar 2).
- disk-space-error dropped from 77 hits (Feb 16) to 12 (Feb 23) to 0 (Mar 2).
- Caveat: Mar 2 saw infra bounce back to 13.0% of runs, driven by
  NVIDIA Test Lab Validation surging to 48% fail rate. This may be a
  new issue or noise from the small 2-day sample.

Build Improvements
-------------------
- Build failures dropped from 3.7% to 0.2% of all runs after Feb 23
  (-3.5 percentage points, effectively eliminated).
- helm-error dropped from 64 hits (Feb 9) to 49 (Feb 23) to 2 (Mar 2).
- docker-build-error dropped from 30 hits (Feb 9) to 3 (Feb 23) to 0 (Mar 2).
- docker-daemon-error-response, which spiked to 76 hits during Feb 16,
  disappeared entirely by Feb 23.
- Build is the clearest success story of Q1.

Uncategorized Errors
---------------------
- ~34% of all failed job logs could not be classified into Testing, Infra,
  or Build. These logs had error snippets extracted but did not match any
  known categorization pattern.
- Uncategorized rate increased slightly after Feb 23 (12.8% to 15.0% of
  runs), meaning the categorizer's coverage did not improve as overall
  errors dropped.
- Week-by-week uncategorized % of runs:
    Jan 19: 0.4%  |  Jan 26: 11.3%  |  Feb 2: 7.2%  |  Feb 9: 11.6%
    Feb 16: 29.1% |  Feb 23: 13.1%  |  Mar 2: 24.2%
- The Feb 16 and Mar 2 spikes in uncategorized suggest that during high-
  failure periods, new error patterns emerge that the categorizer misses.
- Improving categorizer coverage is a potential area for investment: ~675
  of 1,987 failed logs remain unclassified. Better patterns could shift
  some of these into actionable buckets.

===========================================================================
WORKFLOW-LEVEL HIGHLIGHTS
===========================================================================

Three chronically broken workflows dominated post-merge failures:
  - Docs link check:             84% fail rate (544/650 runs)
  - build-frontend-image.yaml:  100% fail rate (49/49 runs)
  - Post-Merge CI Pipeline:      95% fail rate (87/92 runs)

These 3 workflows account for 680 of 1,056 total workflow failures (64%).
If fixed or disabled, the overall WF failure rate would drop from 22.1%
to approximately 8%.

===========================================================================
METHODOLOGY NOTES
===========================================================================

- "Post-merge" = GitHub Actions runs with event="push", head_branch="main"
- Job logs downloaded via GitHub API, cached at ~/.cache/dynamo-utils/raw-log-text/
  (TTL: 365 days). No API calls needed for re-analysis.
- Error categorization via ci_log_errors --scan-all-logs (local regex-based scan)
- Each log assigned to one bucket by priority (Testing > Infra > Build) to
  avoid double-counting when a log matches multiple categories
- Workflow-level data from actions_runs_list.json cache (81,962 runs)
- Weekly stats files in this directory contain per-week detail including
  full category breakdowns and workflow failure rates
- See also: 2026-01-23.txt for pre-merge baseline (1,698 logs, pre-merge only)
