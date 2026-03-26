# Dynamo PyTest Metrics Q1 2026

**Important note:** This presentation focuses on **pytest stability** — not CI build time, infrastructure, or other aspects of the CI system. Developers write tests, and those tests directly impact merge velocity. When pytest stability suffers, the effects compound quickly: flaky tests trigger failed builds, failures trigger manual re-runs that waste compute and developer time, and developers gradually lose trust in test signals — leading them to ignore legitimate failures altogether.

## Key Takeaways

- **Test failures were the #1 driver of PR merge delays.** At peak (Feb 16), 46% of all post-merge job runs triggered a test failure.
- After targeted reliability fixes (mid-to-late Feb), that dropped to **14% by Feb 23** and continued improving through March.
- **Days to merge cut by 75%**: 4.2 days (Feb 16) → 2.6 days (Feb 23) → 1.1 days (Mar 2) → 1.4 days (Mar 9).
- **Manual job re-run rate cut by half**: 36% (Feb 9) → 16% (Feb 23) → 20% (Mar 2) → 17% (Mar 16). Fewer developers having to manually re-run their CI.
- **~155 developer-hours saved per week from test reliability improvements alone** (holding pre-merge duration constant) — equivalent to ~19 full (8-hour) work days reclaimed per week across the team. Combined impact with infrastructure speed improvements is detailed in [Time Savings Extrapolation](#time-savings-extrapolation).
- **Busiest week on record saw the best metrics**: Feb 23 had 225 PRs, 195 merges, and 64 active authors — while every reliability metric improved.

### Dev Metrics 2026 Q1

Data sources:
- **Pre-merge PR data** (columns: PRs through PR Fail %, Pre-merge Duration): PR lifecycle metrics from GitHub PR API and pre-merge workflow runs. Pre-merge runs include both `pull_request` event workflows (lint, copyright, "Pre Merge" Rust checks) and `push` event workflows on `pull-request/NNN` mirror branches ("NVIDIA Dynamo Github Validation" container build+test). Measures the developer experience during the PR review and feedback loop before merge.
- **Post-merge job data** (column: Post-merge Duration): Measured from GitHub Actions runs on main (event="push", head_branch="main"). Post-merge runs include additional workflows (NVIDIA Validation, Test Lab, Post-Merge Pipeline) not triggered pre-merge. Error distributions in later sections also use this data source.

| Week | PRs | PRs Merged | Avg Lines/PR <sup>⋆</sup> | Median Lines/PR <sup>⋆</sup> | Days to merge/PR <sup>¶</sup> | Push/PR <sup>△</sup> | PR Re-run % <sup>‡</sup> | Re-run/PR <sup>‖</sup> | PR Fail % <sup>†</sup> | Pre-merge <sup>⊕</sup> (min) | Post-merge (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dec&nbsp;29 | 35 | 14 | 395 | 52 | 1.5 | 2.3 | 16% | 1.4 | 89% | 54 | 109 |
| Jan&nbsp;05 | 167 | 133 | 277 | 27 | 2.8 | 2.9 | 21% | 1.4 | 76% | 59 | 108 |
| Jan&nbsp;12 | 138 | 114 | 588 ° | 28 ° | 2.7 | 3.2 | 19% | 1.3 | 62% | 53 | 123 |
| Jan&nbsp;19 | 118 | 76 | 546 | 89 | 3.5 | 3.9 | 16% | 1.6 | 67% | 45 | 66 |
| Jan&nbsp;26 | 198 | 141 | 532 | 56 | 2.2 | 3.4 | 24% | 2.2 | 71% | 50 | 75 |
| Feb&nbsp;02 | 185 | 158 | 853 | 130 | 3.6 | 4.3 | 25% | 3.0 | 71% | 53 | 61 |
| Feb&nbsp;09 | 196 | 144 | 1074 | 109 | 3.4 | 4.3 | 36% | 3.2 | 72% | 55 | 115 |
| Feb&nbsp;16 | 150 | 106 | 834 | 122 | 4.2 | 4.5 | 21% | 2.2 | 73% | 39 | 128 |
| **Feb&nbsp;23** | 225 | **195&nbsp;↑** | 638 | 113 | **2.6&nbsp;↓** | 4.3 | **16%&nbsp;↓** | **1.6&nbsp;↓** | **62%&nbsp;↓** | **35&nbsp;↓** | **72&nbsp;↓** |
| **Mar&nbsp;02** | 306 | **220&nbsp;↑** | 1271 | 73 | **1.1&nbsp;↓** | **3.5&nbsp;↓** | 20% | 2.2 | **45%&nbsp;↓** | 43 | 45 |
| Mar&nbsp;09 | 322 | 253 | 2681 | 93 | 1.4 | 4.6 | 18% | 1.8 | 67% | 126 ‡‡ | 130 |
| Mar&nbsp;16 | 142 | 109 | 6104 | 144 | 2.3 | 5.7 | 17% | 2.4 | 66% | 57 | 71 |

Note: Re-run % means the share of PRs where at least one workflow was manually re-run (GitHub Actions `run_attempt > 1`); Re-run/PR means average manual re-runs among affected PRs. Data source: all pre-merge events including `push:pull-request/NNN` (heavy CI). Mar 02–16 values corrected from earlier version that undercounted by excluding heavy CI runs. ‡‡ Mar 09 pre-merge avg inflated by NVIDIA Test Lab Validation infrastructure issue (avg 216 min/run vs 5–19 min in prior weeks); median pre-merge was 51 min.

### Developer Velocity

$$\text{Developer Velocity} \approx \frac{\text{PRs Merged} \times (1 - \text{PR Fail \%}^†) \times (1 - \text{PR Re\text{-}run \%}^‡)}{\text{Pre-merge Duration}^⊕ \times \text{Push/PR}^△}$$

$$\text{PR Fail \%}^† = f(\text{test failures, docker/build errors, k8s/helm errors, network errors, infra errors, auth errors})$$

$$\text{Pre-merge Duration}^⊕ = f(\text{failed tests that run to max timeout, tests without proper timeout limits, infrastructure speed})$$

<sub>Note: Developer Velocity (DV) is a hypothetical composite metric constructed to illustrate how the measured metrics interact. Days to merge/PR <sup>¶</sup> is the observable output. Push/PR <sup>△</sup> partially captures the review iteration cycle.</sub>

Mission: Improving test reliability to increase developer velocity

- At its peak (Feb 16), **nearly half of all post-merge job runs (46%) triggered a test failure**. This was the single largest source of PR merge delays — every flaky or broken test meant a developer waiting for a re-run, adding hours to merge time.
- Reducing test error rate from 46% to 14% of runs directly contributed to halving days-to-merge and eliminating ~155 hours of developer wait time per week (test reliability alone; ~174 hours combined with infrastructure speed improvements).
- Feb 24 set the all-time single-day merge record.
- At the peak of 1.0 code-freeze activity (week of Feb 23), we observed 225 PRs (busiest week on record), 195 merges, and 64 active authors, while key metrics improved in PR submission failure rate <sup>†</sup>, manual job re-run rate <sup>‡</sup>, and days to merge <sup>¶</sup>.
- Several major changes were made from mid to late Feb (Feb 16 to Feb 27) to improve test stability, harden the workflow, and fix recurring issues — including fixing flaky pytest fixtures, adding timeouts to long-running tests that previously ran to the max timeout, adding retry logic for transient network errors in tests, disabling or quarantining chronically unstable tests, and hardening timeout/resource configurations.
- Before mid Feb and after the stability fixes:
  - Manual job re-run rate <sup>‡</sup> went from **36% to 16%**.
  - Average manual job re-runs per affected PR <sup>‖</sup> went from **3.2 to 1.6**.
  - PR submission failure rate <sup>†</sup> hovered around 70% but went down to 62%.
  - Days to merge <sup>¶</sup> went from **4.2 days down to 2.6 days**.

---

Period: 2026-01-24 to 2026-03-22 (12 weeks)
Source: 4,284 raw post-merge job logs + GitHub Actions workflow run data
Methodology:
  - Catch-alls excluded: vllm-error, sglang-error, trtllm-error, exit-139-sigsegv
  - A single log can match multiple categories (hit-level counting)
  - 9 failure groups: Tests, Docker/build, K8s/Helm, Network,
    Infra/system, Gates/policy, Docs, Build, Auth

---

## Metric Definitions

Pre-merge PR metrics:
- <sup>⋆</sup> **Avg Lines/PR** and **Median Lines/PR** -- average and median lines changed (additions + deletions) per merged PR that week. Median is more representative of a typical PR; averages are skewed by a few large refactoring or auto-generated PRs. ° Jan 12 excludes 4 auto-generated attribution file PRs (150k-239k lines each); raw avg would be 8,103.
- <sup>¶</sup> **Days to merge/PR** -- elapsed time (days) from first PR submission to final merge. Example: 2.6d means a PR took about 2.6 days to merge.
- <sup>△</sup> **Push/PR** -- average number of code pushes per PR before merge.
- <sup>‡</sup> **PR Re-run %** -- the share of PRs that needed at least one manual job re-run after a failed run. Example: 20% means 1 out of every 5 PRs, someone manually triggered at least one re-run (without pushing new code).
- <sup>‖</sup> **Re-run/PR** -- average number of manual job re-runs among PRs that were manually re-runed. Example: 1.6 means each affected PR was manually re-run about 1 to 2 times.
- <sup>†</sup> **PR Fail %** -- the share of PRs that had at least one failed automation run. Example: 33% means 1 out of every 3 PRs failed at least once.
- <sup>⊕</sup> **Pre-merge (min)** -- average of max(workflow durations) per commit, across all pre-merge workflows triggered for that commit (including `pull_request` checks and validation runs on `pull-request/NNN` mirror branches). Since workflows run in parallel, the developer waits for the slowest one.

Post-merge job metrics (code that passed pre-merge validation and was merged to main):
- **Post-merge (min)** -- average wall-clock time (minutes) from the first workflow created to the last workflow completed, per commit push on main. Post-merge runs include additional workflows (NVIDIA Validation, Test Lab, Post-Merge Pipeline) not triggered during pre-merge.
- <sup>§</sup> **PR Avg (min)** -- average "PR" workflow run duration (minutes) from post-merge job data (created_at to updated_at). Reflects how long a single workflow run takes to complete.
- **WF Failure %** -- post-merge workflow failure rate (% of runs with conclusion = failure). Not the same as <sup>†</sup>.

### Time Savings Extrapolation

$$\text{Wasted Wait per PR (hours)} = \text{PR Re\text{-}run \%}^‡ \times \text{Re-run/PR}^‖ \times \text{Pre-merge Duration}^⊕ \text{(hours)}$$

<sub>Developers wait during pre-merge checks. Once merged, post-merge runs happen in the background and don't block the developer.</sub>

**Calculation 1 — Test reliability improvements only** (hold pre-merge duration constant at 55 min / 0.92h):
- Feb 9 baseline: 0.36 × 3.2 × 0.92 = **1.06 hours/PR**.
- Feb 23 (with test fixes, but same pre-merge duration): 0.16 × 1.6 × 0.92 = **0.24 hours/PR**.
- Reduction: 1.06 → 0.24 = **−78% wasted wait per PR**.
- Scaled to actual PR volume: Feb 9 = 1.06 × 196 = 207.8 h/week; Feb 23 = 0.24 × 225 = 53.3 h/week.
- **Savings from test reliability alone: ~155 hours/week** (19.3 work days).

**Calculation 2 — Combined: test reliability + infrastructure speed improvements**:
- Pre-merge duration dropped from 55 min (0.92h) to 35 min (0.58h) between Feb 9 and Feb 23, thanks to infrastructure optimizations (build caching, parallel stages, runner improvements) that happened concurrently with these test stability fixes.
- Feb 9 baseline: 0.36 × 3.2 × 0.92 = **1.06 hours/PR**.
- Feb 23 (test fixes + faster pre-merge): 0.16 × 1.6 × 0.58 = **0.15 hours/PR**.
- Reduction: 1.06 → 0.15 = **−86% wasted wait per PR**.
- Scaled to actual PR volume: Feb 9 = 1.06 × 196 = 207.8 h/week; Feb 23 = 0.15 × 225 = 33.5 h/week.
- **Combined savings: ~174 hours/week** (21.8 work days).

The pre-merge duration improvement (55 → 35 min, −36%) accounts for roughly 20 of those 174 hours — about 11% of the total savings. The remaining 89% comes from fewer re-runs (lower flake rate and fewer re-runs per affected PR).

---

### PR Size vs. Time to Merge

683 merged PRs (Jan 19 – Mar 5) grouped by lines changed (additions + deletions):

| Size | Lines | PRs | Avg&nbsp;Days | Median&nbsp;Days | P75&nbsp;Days |
| --- | ---: | ---: | ---: | ---: | ---: |
| S | 1–50 | 251 (37%) | 2.5 | 0.3 | 1.1 |
| M | 51–100 | 76 (11%) | 1.4 | 0.6 | 1.9 |
| M+ | 101–200 | 86 (13%) | 3.8 | 0.9 | 2.8 |
| L | 201–500 | 103 (15%) | 5.2 | 1.7 | 5.8 |
| L+ | 501–1,000 | 59 (9%) | 5.3 | 1.7 | 7.0 |
| XL | 1,001–5,000 | 90 (13%) | 6.6 | 3.2 | 6.5 |
| XXL | 5,001+ | 17 (2%) | 8.7 | 7.1 | 11.8 |

Larger PRs take significantly longer to merge: XXL PRs (5,001+ lines) take ~24x longer than S PRs at the median (7.1 vs 0.3 days). 61% of all PRs are 200 lines or fewer (S + M) and merge in under a day at the median.

---

## Failure Distribution by Group (post-merge data)

5,652 errors across 4,284 failed job logs (a log can match multiple groups):

| Group           | Errors | % of errors |
|-----------------|------:|----------:|
| Tests           | 2,308 |    40.8%  |
| Docker / build  |   935 |    16.5%  |
| Network         |   862 |    15.3%  |
| Gates / policy  |   611 |    10.8%  |
| K8s / Helm      |   565 |    10.0%  |
| Infra / system  |   356 |     6.3%  |
| Docs            |   156 |     2.8%  |
| Build           |    48 |     0.8%  |
| Auth            |    26 |     0.5%  |

Tests (41%) remain the dominant failure domain. Docker/build (17%) and
Network (15%) follow. Gates/policy (11%) and K8s/Helm (10%) round out
the top five. Docs failures (broken-links) grew from 0.4% to 2.8% in March.

---

## Failure Distribution Detail (post-merge data)

5,652 errors across 4,284 failed job logs (per-category breakdown of the groups above).

| Category | Errors | % | Group |
|---|---:|---:|---|
| pytest-error | 1,234 | 21.8% | Tests |
| python-error | 715 | 12.7% | Tests |
| docker-build-error | 669 | 11.8% | Docker / build |
| ci-status-check-error | 536 | 9.5% | Gates / policy |
| network-timeout-generic | 433 | 7.7% | Network |
| k8s-error | 358 | 6.3% | K8s / Helm |
| pytest-timeout-error | 225 | 4.0% | Tests |
| network-error | 208 | 3.7% | Network |
| etcd-error | 206 | 3.6% | Infra / system |
| helm-error | 174 | 3.1% | K8s / Helm |
| broken-links | 156 | 2.8% | Docs |
| docker-daemon-error-response-error | 125 | 2.2% | Docker / build |
| network-port-conflict-error | 123 | 2.2% | Network |
| backend-failure | 101 | 1.8% | Tests |
| disk-space-error | 92 | 1.6% | Docker / build |
| deploy-test-status-check | 75 | 1.3% | Gates / policy |
| exit-127-cmd-not-found | 62 | 1.1% | Infra / system |
| cuda-error | 52 | 0.9% | Infra / system |
| go-operator-lint-error | 48 | 0.8% | Build |
| network-timeout-gitlab-mirror | 39 | 0.7% | Network |
| network-download-error | 34 | 0.6% | Network |
| rust-error | 33 | 0.6% | Tests |
| oom | 31 | 0.5% | Infra / system |
| docker-registry-error | 25 | 0.4% | Docker / build |
| docker-upload-error | 24 | 0.4% | Docker / build |
| network-timeout-https | 24 | 0.4% | Network |
| timeout-exit-124 | 18 | 0.3% | K8s / Helm |
| k8s-network-timeout-pod | 15 | 0.3% | K8s / Helm |
| huggingface-auth-error | 14 | 0.2% | Auth |
| auth-token-expired | 12 | 0.2% | Auth |

---

## Post-Merge Run Conclusions

| Conclusion | Count | % |
|------------|------:|---:|
| Success    | 7,041 | 78.8% |
| Failure    | 1,871 | 20.9% |
| Cancelled  |    41 |  0.5% |
| **Total**  | **8,935** | |

---

## Weekly Error Distribution (post-merge data)

Column definitions:
- **Total Runs**: all post-merge workflow runs that week (success + failure + cancelled)
- **Tests / Docker / K8s / Net / Infra / Auth**: group error hits as % of total runs

| Week | Total Runs | Tests | Docker | K8s | Net | Infra | Auth |
|----------|-----:|------:|-------:|----:|----:|------:|-----:|
| Dec 29*  |   60 |  8.3% |  0.0% | 0.0%| 6.7%|  3.3% | 0.0% |
| Jan 5    |  536 |  1.3% | 33.4% | 0.7%| 1.9%|  0.7% | 1.1% |
| Jan 12   |  595 | 10.3% |  0.0% | 2.4%| 0.7%|  0.3% | 0.3% |
| Jan 19   |  458 | 37.1% | 31.0% |17.5%|72.3%|  2.0% | 0.0% |
| Jan 26   |  701 | 15.1% |  1.9% | 2.0%| 4.0%|  3.0% | 0.4% |
| Feb 2    |  951 | 34.4% | 28.0% | 1.5%| 5.9%|  5.3% | 0.0% |
| Feb 9    |1,023 | 30.0% | 15.4% |12.6%|22.4%|  3.1% | 0.0% |
| Feb 16   |  858 | 46.3% |  6.4% | 7.0%|10.5%|  5.6% | 0.0% |
| **Feb 23**   |1,105 | **14.3% ↓** |  8.1% |10.1%| **3.2% ↓**|  **1.8% ↓** | 0.0% |
| **Mar 2**    |1,001 | 19.2% |  0.2% |12.2%| 15.1%|  1.2% | 1.2% |
| Mar 9        |1,310 | 36.9% |  0.5% | 2.6%| 10.7%|  8.1% | 0.0% |
| **Mar 16**   |  662 | **16.9% ↓** |  4.2% | **3.6% ↓**|  **5.9% ↓**|  **0.9% ↓** | 0.0% |

\* Dec 29 = low log coverage (31 job logs for 60 runs); other early weeks now have full coverage after bulk log download.
Docs and Build omitted for brevity (<3% of any week).

---

## Workflow-Level Highlights

Three chronically broken workflows dominated post-merge failures:
  - Docs link check:             84% fail rate (544/650 runs)
  - build-frontend-image.yaml:  100% fail rate (49/49 runs)
  - Post-Merge Pipeline:         95% fail rate (87/92 runs)

These 3 workflows account for 680 of 1,056 total workflow failures (64%).
If fixed or disabled, the overall WF failure rate would drop from 22.1%
to approximately 8%.

---

## Before / After Feb 23 (post-merge improvement snapshot)

Error group hits as % of total runs, before and after Feb 23:

| Group          | Before (5 wks, 3,991 runs) | After (2 wks, 1,320 runs) | Change |
|----------------|---------------------------:|------------------------:|--------|
| Tests          | 32.8% | 12.7% | -20.1pp |
| Docker / build | 15.9% |  8.6% |  -7.3pp |
| Network        | 18.4% |  2.9% | -15.5pp |
| K8s / Helm     |  7.5% |  8.6% |  +1.1pp |
| Infra / system |  4.0% |  1.5% |  -2.5pp |
| Other          |  0.1% |  1.1% |  +1.0pp |

WF failure rate dropped from 22.7% to 17.8% (-4.9pp).

Tests (-20.1pp) saw the largest improvement, followed by Network (-15.5pp),
Docker (-7.3pp), and Infra (-2.5pp). K8s/Helm increased slightly (+1.1pp),
driven by k8s-error and helm-error concentrating in Feb 23.

---

## Key Improvements

### Tests
- Test error rate dropped from 32.8% to 12.7% of runs after Feb 23 (-20.1pp).
- The Feb 16 spike (46.3% of runs) was a one-week anomaly driven by an
  etcd-error surge (38 errors, up from 3 the prior week) that cascaded
  into pytest failures. By Feb 23, the test error rate dropped to 14.3% of runs.
- pytest-error (732 errors Q1-wide) is the single largest category overall.
- pytest-timeout-error is small (9 errors) but appeared mainly in Mar 2,
  suggesting a new slow-test issue worth monitoring.

### Docker / build
- Docker/build error rate dropped from 15.9% to 8.6% of runs after Feb 23 (-7.3pp).
- docker-build-error (477 errors) is the largest category in this group,
  peaking at Feb 2 (28.0% of runs) and tapering off.
- disk-space-error (92 errors) spiked during Feb 2 (76 errors) and has
  largely disappeared since.
- docker-daemon-error-response-error (21 errors) was concentrated in
  Jan 19 and largely disappeared afterward.

### K8s / Helm
- K8s/Helm error rate grew from 7.5% to 8.6% of runs after Feb 23 (+1.1pp), driven by
  k8s-error and helm-error concentrating in the Feb 23 week.
- timeout-exit-124 (18 errors) appeared in Feb 16 and Feb 23.

### Network
- Network error rate dropped from 18.4% to 2.9% of runs after Feb 23 (-15.5pp).
- Network was elevated throughout Jan 19 (72.3% of runs, driven by a network-error
  surge) and peaked again at Feb 9 (22.4% of runs, driven by network-timeout-generic).
  After Feb 23, network errors dropped to low single digits.
- network-error (90 errors) peaked at Feb 16 (68 errors) and improved.

### Infra / system
- Infra/system error rate dropped from 4.0% to 1.5% of runs after Feb 23 (-2.5pp).
- etcd-error (82 errors) spiked to 38 during Feb 16 and returned to
  baseline (12 errors Feb 23).
- oom (14 errors) appeared mainly in Feb 16 (8) and Feb 23 (6).
- Note: oom, etcd-error, and cuda-error are ambiguous — they could be
  caused by bad tests (resource leaks, misconfigured GPU tests) or
  actual infrastructure issues. The categorization boundary is not clear.

---

## Methodology Notes

This document uses two data sources:

**Post-merge job data** (error distributions, job duration, workflow conclusions):
- "Post-merge" = GitHub Actions runs with event="push", head_branch="main"
- Job logs downloaded via GitHub API, cached at ~/.cache/dynamo-utils/raw-log-text/
  (TTL: 365 days). No API calls needed for re-analysis.
- Early weeks (Dec 29 – Jan 19) supplemented with bulk log download (1,331 additional
  job logs) to improve coverage for the Weekly Error Distribution table.
- Error categorization via ci_log_errors engine (local regex-based scan)
- Hit-level counting: a log matching categories in multiple groups is
  counted in each group. Group percentages sum to 100% of all errors.
- Workflow-level data from actions_runs_list.json cache (81,962 runs)
- Post-merge runs execute the same workflows as pre-merge PR checks, so
  post-merge failure rates serve as a proxy for the pre-merge developer experience.

**Pre-merge PR data** (Development Metrics table):
- PR lifecycle metrics (PRs, merges, days to merge, push/PR, re-run rates,
  PR fail rates) computed from GitHub PR API data.
- Pre-merge duration computed from workflow runs grouped by commit SHA, including
  both `pull_request` event runs and `push` event runs on `pull-request/NNN` mirror
  branches. Per-commit duration = max(individual workflow durations), since workflows
  run in parallel and the developer waits for the slowest one.
- These measure the developer experience during the PR review and feedback
  loop before a PR is merged.

- Weekly stats files in this directory contain per-week detail including
  full category breakdowns and workflow failure rates
- See also: 2026-01-23.txt for pre-merge baseline (1,698 logs, pre-merge only)
