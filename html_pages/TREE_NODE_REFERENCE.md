# Tree Node Reference Guide

Complete reference for all tree node types in the local/remote branches dashboards.

---

## Node Hierarchy

```
RepoNode (repository directory)
├─ SectionNode ("Branches with PRs", "Branches", "Local-only branches")
│  └─ BranchInfoNode (individual branch)
│     ├─ CommitMessageNode (commit message with PR link)
│     ├─ MetadataNode (timestamps and age)
│     ├─ PRStatusNode (CI status for PRs)
│     │  ├─ CIJobTreeNode (CI check/job)
│     │  │  └─ CIJobTreeNode (nested steps)
│     │  ├─ ConflictWarningNode
│     │  ├─ BlockedMessageNode
│     │  └─ RerunLinkNode
│     └─ WorkflowStatusBranchNode (workflow status for non-PR branches)
│        └─ BranchNode (individual workflow run)
```

---

## Visual Example

```
▼ dynamo/                                      ← RepoNode
│  ├─ Branches with PRs                        ← SectionNode
│  │  ├─ [copy] ✖ branch-1 → main [SHA]      ← BranchInfoNode (closed PR)
│  │  │  ├─ commit message (#1234)            ← CommitMessageNode
│  │  │  ├─ (modified ..., created ..., ago) ← MetadataNode
│  │  │  └─ ▶ PASSED  3 ✓26 ✗2               ← PRStatusNode (collapsed)
│  │  │     ├─ ✓ check-1 (6m) [log]          ← CIJobTreeNode (hidden)
│  │  │     └─ ✗ check-2 (2m) [log] ▶ Snippet
│  │  └─ [copy] branch-2 → release/0.8 [SHA]  ← BranchInfoNode (open PR)
│  │     ├─ fix: memory leak (#2345)          ← CommitMessageNode
│  │     ├─ (modified ..., created ..., ago) ← MetadataNode
│  │     └─ ▼ FAILED  2 ✓24 ✗1               ← PRStatusNode (expanded)
│  │        ├─ ✓ check-1 (5m)                 ← CIJobTreeNode
│  │        └─ ▼ ✗ check-2 (3m) [log] ▶ Snippet
│  │           ├─ ✓ setup (10s)               ← CIJobTreeNode (step)
│  │           ├─ ✗ test (2m 30s)             ← CIJobTreeNode (step)
│  │           └─ ✓ cleanup (20s)             ← CIJobTreeNode (step)
│  ├─ Branches                                 ← SectionNode (non-PR)
│  │  └─ [copy] feature → main [SHA]          ← BranchInfoNode
│  │     └─ ✅ PASSED ✓5                      ← WorkflowStatusBranchNode
│  │        ├─ ✓ pre_merge                    ← BranchNode
│  │        ├─ ✓ Rust checks                  ← BranchNode
│  │        ├─ ✓ Copyright                    ← BranchNode
│  │        ├─ ✓ DCO                          ← BranchNode
│  │        └─ ✓ Docs                         ← BranchNode
│  └─ Local-only branches                      ← SectionNode
│     └─ [copy] local-branch [SHA]            ← BranchInfoNode
```

---

## Core Node Types

### `BranchNode` (base class)
Abstract base for all tree nodes. Not used directly.

**Key methods:**
- `to_tree_vm()` - Convert to `TreeNodeVM` for rendering
- `_format_html_content()` - Generate HTML for the node

---

### `RepoNode`
Represents a repository directory.

**Display:**
- Normal: `▼ dynamo/` (collapsible)
- Symlink: `■ speedoflight/ → ../path` (non-collapsible)

**Children:** `SectionNode` instances

**Example:**
```
▼ dynamo/
   ├─ Branches with PRs
   └─ Local-only branches
```

---

### `SectionNode`
Groups branches by category.

**Common sections:**
- "Branches with PRs" - branches with open/closed PRs
- "Branches" - branches with remotes but no PRs (shows workflow status)
- "Local-only branches" - no remote tracking

**Children:** `BranchInfoNode` instances

---

### `BranchInfoNode`
Individual git branch with metadata.

**Display:**
```
[copy] [✖] branch-name → base [SHA]
├─ commit message (#PR)
├─ (modified PT, created UTC, age)
└─ Status (PR or Workflow)
```

**Special behavior:**
- Copy button strips repo prefixes ("ai-dynamo/")
- Closed PRs show ✖ mark
- Always expanded when has children
- For non-PR branches: fetches GitHub Actions workflow runs

**Example with PR:**
```
[copy] feature/DIS-1200 → main [4afb3fb]
├─ refactor: remove "dev" stage (#5050)
├─ (modified 2026-01-07 10:36 PT, created 2025-12-22 03:49, 8h 39m ago)
└─ ▶ PASSED  3 ✓26 ✗2
```

**Example without PR:**
```
[copy] feature-branch → main [abc123]
└─ ✅ PASSED ✓5
   ├─ ✓ pre_merge
   ├─ ✓ Rust pre-merge checks
   └─ ✓ Copyright Checks
```

---

## Metadata Nodes

### `CommitMessageNode`
First line of commit message with PR link.

**Display:** `commit message first line (#PR_NUMBER)`

**Behavior:**
- Truncates to 100 chars if too long
- PR number is clickable GitHub link
- Grey text for subtle appearance

---

### `MetadataNode`
Branch timestamps and age.

**Display:** `(modified YYYY-MM-DD HH:MM PT, created YYYY-MM-DD HH:MM, Xd Yh ago)`

**Format:** Compact age like "8h 39m ago" or "16d 23h ago"

---

### `PRNode`
Stores PR metadata (not rendered separately).

**Purpose:** Provides PR title, number, state, base branch for tooltips

---

## Status Nodes

### `PRStatusNode`
Aggregate CI status for PRs.

**Display:** `[▼/▶] PASSED/FAILED/RUNNING [icon] ✓count ✗count [reviews] [💬]`

**Expansion:**
- ✅ PASSED: Collapsed (▶) - hides CI children
- ⚠️ FAILED: Expanded (▼) - shows CI children
- 🔄 RUNNING: Expanded (▼) - shows CI children

**Shows:**
- Review status (✅ Approved, 🔴 Changes Requested)
- Unresolved conversation count
- GitHub icon links to commit checks page

**Children:** `CIJobTreeNode` instances

**Example:**
```
▶ PASSED  3 ✓26 ✗2, 💬 Unresolved: 28
├─ ✓ Build and push Dynamo (6m)
└─ ✗ deploy-test-vllm (2m)
```

---

### `WorkflowStatusBranchNode` *(NEW 2026-01-07)*
GitHub Actions workflow status for branches without PRs.

**Display:** `[icon] STATUS ✓count ✗count`

**Status priority:**
1. **❌ FAILED** - any workflow has `conclusion=failure`
2. **⏳ RUNNING** - any workflow `in_progress` or `queued`
3. **✅ PASSED** - at least one `conclusion=success` (no failures/running)
4. **⚪ UNKNOWN** - no matching workflows

**Behavior:**
- Always expanded (non-collapsible)
- Shows up to 5 most recent workflow runs
- Fetched via `/repos/{owner}/{repo}/actions/runs?branch={branch_name}`
- Cached with 5-minute TTL

**Implementation:** Uses generic `BranchNode` with computed label (no dedicated class)

**Children:** `BranchNode` instances for individual workflow runs

---

## CI Nodes

### `CIJobTreeNode`
Individual CI check/job from GitHub Actions.

**Display:** `[▼/▶] [icon] check-name (duration) [log] [▶ Snippet]`

**Icons:**
- ✓ (green circle/checkmark) - Success (required/optional)
- ✗ (red circle/X) - Failure (required/optional)
- ⏳ - In progress
- ⏸ - Pending/queued
- ✖️ - Cancelled
- ? - Unknown/skipped

**Expansion:**
- Success/skipped: Collapsed
- Required failure: Expanded
- Running/pending: Expanded
- Optional failure: Collapsed

**Special behavior:**
- Groups matrix jobs by architecture (amd64, then arm64)
- Shows nested job steps in hierarchy
- Snippet button if log available

**Children:** Nested `CIJobTreeNode` for job steps

**Example:**
```
▼ ✗ deploy-test-vllm (2m) [log] ▶ Snippet
├─ ✓ checkout (5s)
├─ ✗ run-tests (1m 45s)
└─ ✓ cleanup (10s)
```

---

## Utility Nodes

### `PRURLNode`
Clickable GitHub PR link with title.

**Display:** `📖 PR#1234: Title of the pull request`

**Behavior:** Opens in new tab, truncates long titles

---

### `RerunLinkNode`
Link to rerun failed GitHub Actions workflow.

**Display:** `🔄 Rerun workflow [run 123456789]`

**Behavior:**
- Only shown when CI has failures
- Includes copy button for `gh run rerun --failed` command

---

### `BlockedMessageNode`
Shows when PR is blocked.

**Display:** `🚫 Blocked: <reason>`

---

### `ConflictWarningNode`
Shows merge conflict warnings.

**Display:** `⚠️ <message>`

---

## Expansion Policies

| Node Type | Default | Controlled By |
|-----------|---------|---------------|
| `RepoNode` | Expanded | User toggle |
| `SectionNode` | Expanded | User toggle |
| `BranchInfoNode` | Expanded (when has children) | Always shows children |
| `PRStatusNode` | Collapsed if PASSED, expanded if FAILED/RUNNING | CI status |
| `CIJobTreeNode` | Collapsed if success, expanded if failure/running | Job status |
| `WorkflowStatusBranchNode` | Expanded (non-collapsible) | N/A |

---

## TreeNodeVM Rendering

All nodes convert to `TreeNodeVM` via `to_tree_vm()`.

**Fields:**
- `node_key` - Stable DOM id (survives regeneration)
- `label_html` - Full HTML content
- `children` - Child `TreeNodeVM` instances
- `collapsible` - Show expand/collapse triangle
- `default_expanded` - Initial state (▼/▶)
- `triangle_tooltip` - Optional tooltip
- `noncollapsible_icon` - Icon for non-collapsible ("square" = ■)

---

## Helper Functions

Exported from `show_local_branches.py`, shared with `show_remote_branches.py`:

**Formatting:**
- `_format_age_compact(dt)` - "(8h 39m ago)"
- `_format_branch_metadata_suffix(...)` - "(modified ..., created ..., ago)"
- `_format_base_branch_inline(pr)` - "→ main"
- `_format_commit_tooltip(msg)` - Escaped tooltip
- `_format_pr_number_link(pr)` - "#5050" link
- `_strip_repo_prefix_for_clipboard(name)` - Remove "ai-dynamo/"

**Status:**
- `_pr_needs_attention(pr)` - Has running work or required failures

**CI:**
- `_build_ci_hierarchy_nodes(...)` - Build `CIJobTreeNode` tree

---

## Local vs Remote Dashboards

### Local (`show_local_branches.py`)
- Scans git repos on disk
- Shows local modifications and commit times
- Full CI details with error snippets
- **Root hierarchy:** `RepoNode` (directory) → `BranchInfoNode` (branch) → `CommitMessageNode`, `MetadataNode`, `PRNode`, `PRStatusNode` (PASSED/FAILED pill) → `CIJobTreeNode` (CI jobs)

### Remote (`show_remote_branches.py`)
- Fetches PRs by GitHub username
- No local-only branches
- **Same tree structure** as local branches
- **Root hierarchy:** `UserNode` (GitHub user) → `BranchInfoNode` (branch) → `CommitMessageNode`, `MetadataNode`, `PRNode`, `PRStatusNode` (PASSED/FAILED pill) → `CIJobTreeNode` (CI jobs)
- **Key difference:** Uses `UserNode` instead of `RepoNode` as the collapsible root
- **Implementation:** Imports `PRStatusNode` and `_build_ci_hierarchy_nodes` from `show_local_branches.py` to ensure identical rendering logic

---

## Quick Reference

**Node naming shortcuts:**
- "Repo node" → `RepoNode`
- "Section node" → `SectionNode`
- "Branch node/line" → `BranchInfoNode`
- "Commit message line" → `CommitMessageNode`
- "Metadata line" → `MetadataNode`
- "Status node/PR status line" → `PRStatusNode`
- "CI/check/job node" → `CIJobTreeNode`
- "Workflow status node" → `WorkflowStatusBranchNode`

**Common modifications:**
- Update branch line → `BranchInfoNode._format_html_content()`
- Change CI expansion → `PRStatusNode.to_tree_vm()` or `CIJobTreeNode._subtree_needs_attention()`
- Add repo icon → `RepoNode._format_html_content()`
