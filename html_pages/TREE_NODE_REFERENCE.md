# Tree Node Reference Guide

This document defines all tree node types used in the local and remote branches dashboards, their hierarchy, and their behavior.

---

## Node Type Hierarchy

```
RepoNode (repository directory)
├─ SectionNode ("Branches with PRs", "Local-only branches")
│  └─ BranchInfoNode (individual branch)
│     ├─ CommitMessageNode (first line of commit with PR# link)
│     ├─ MetadataNode (modified, created, age timestamps)
│     └─ PRStatusNode (PASSED/FAILED/RUNNING summary line)
│        ├─ CIJobTreeNode (individual CI check/job)
│        │  └─ CIJobTreeNode (nested job steps, optional)
│        ├─ ConflictWarningNode (merge conflict warning, optional)
│        ├─ BlockedMessageNode (blocked status message, optional)
│        └─ RerunLinkNode (GitHub Actions rerun link, optional)
```

---

## Node Type Definitions

### 1. **`BranchNode`** (base class)
- **Purpose:** Abstract base class for all tree nodes
- **Key fields:**
  - `label` - Display text
  - `children` - List of child nodes
- **Methods:**
  - `to_tree_vm()` - Convert to `TreeNodeVM` for rendering
  - `_format_html_content()` - Generate HTML for the node line
- **Used directly:** No (always subclassed)

---

### 2. **`RepoNode`** (extends `BranchNode`)
- **Purpose:** Represents a repository directory
- **Display:** 
  - Normal: `▼ dynamo/` (collapsible triangle)
  - Symlink: `■ speedoflight/ → ../other/path` (square, non-collapsible)
- **Key fields:**
  - `path` - Filesystem path to repository
  - `error` - Optional error message if scanning failed
  - `remote_url` - Git remote URL
- **Children:** `SectionNode` instances
- **Special behavior:**
  - Detects symlinks and shows target path with tooltip
  - Symlink repos are non-expandable (no children scanned)
  - Repository name is clickable (links to directory)

**Example:**
```
▼ dynamo/
   ├─ Branches with PRs
   └─ Local-only branches
```

---

### 3. **`SectionNode`** (extends `BranchNode`)
- **Purpose:** Section header grouping branches by category
- **Display:** Plain text, always collapsible
- **Common sections:**
  - "Branches with PRs" - branches with open or closed pull requests
  - "Branches" - branches with remotes but no PRs (shows workflow status)
  - "Local-only branches" - branches without remote tracking
  - "Merged branches" (optional)
- **Children:** `BranchInfoNode` instances
- **Special behavior:** None (simple grouping node)

**Example:**
```
├─ Branches with PRs
│  ├─ keivenchang/DIS-1200...
│  └─ keivenchang/DIS-1150...
├─ Branches
│  └─ feature-branch (with workflow status)
```

---

### 4. **`BranchInfoNode`** (extends `BranchNode`)
- **Purpose:** Individual git branch with metadata
- **Display format (NEW as of 2026-01-07):**
  ```
  [copy button] [✖ closed mark] branch-name → base-branch [SHA]
  ├─ commit message first line (#PR_NUMBER)
  ├─ (modified PT, created UTC, age)
  └─ PRStatusNode OR WorkflowStatusNode...
  ```
- **Key fields:**
  - `sha` - Commit SHA (short form, e.g., "4afb3fb")
  - `commit_url` - GitHub commit URL (no longer displayed as link)
  - `commit_time_pt` - Last modified time (Pacific Time)
  - `commit_datetime` - Commit timestamp for age calculation
  - `commit_message` - Full commit message (used in `CommitMessageNode`)
  - `is_current` - Whether this is the current checked-out branch
- **Children:** 
  - `CommitMessageNode` (first child, shows commit message + PR#) - if PR exists
  - `MetadataNode` (second child, shows timestamps) - if PR exists
  - `PRStatusNode` (third child, shows CI status) - if PR exists
  - **WorkflowStatusNode** (shows GitHub Actions runs) - **NEW: if remote branch without PR**
- **Expansion policy:**
  - **Always expanded** when children exist (so all metadata is visible)
- **Special behavior:**
  - Base branch shown inline (`→ main`)
  - SHA shown as plain text (no link)
  - PR# shown in `CommitMessageNode` as link
  - Copy button strips repo prefixes like "ai-dynamo/"
  - Closed PRs show a ✖ mark after the copy button
  - **NEW (2026-01-07):** For branches with remotes but no PRs, fetches and displays recent GitHub Actions workflow runs

**Example with PR:**
```
├─ [copy] keivenchang/DIS-1200__refactor... → main [4afb3fb]
│  ├─ refactor: remove "dev" stage from Dockerfile.* (#5050)
│  ├─ (modified 2026-01-07 10:36 PT, created 2025-12-22 03:49, 8h 39m ago)
│  └─ ▶ PASSED  3 ✓26 ✗2
```

**Example without PR (showing workflow runs):**
```
├─ [copy] feature-branch → main [abc123]
│  └─ ✅ PASSED ✓5
│     ├─ ✓ pre_merge
│     ├─ ✓ Rust pre-merge checks
│     ├─ ✓ Copyright Checks
│     ├─ ✓ DCO Commenter
│     └─ ✓ Docs link check
```

---

### 5. **`CommitMessageNode`** (extends `BranchNode`)
- **Purpose:** Shows first line of commit message with PR number link
- **Display format:**
  ```
  commit message first line (#PR_NUMBER)
  ```
- **Key fields:**
  - `commit_message` - Full commit message (only first line displayed)
  - `pr_number` - GitHub PR number for link
- **Children:** None
- **Expansion policy:** Non-collapsible (leaf node)
- **Special behavior:**
  - Truncates message to 100 characters if too long
  - PR number shown as clickable GitHub link
  - Grey text for subtle appearance

**Example:**
```
├─ refactor: remove "dev" stage from Dockerfile.* (#5050)
```

---

### 6. **`MetadataNode`** (extends `BranchNode`)
- **Purpose:** Shows branch timestamps and age
- **Display format:**
  ```
  (modified YYYY-MM-DD HH:MM PT, created YYYY-MM-DD HH:MM, Xd Yh ago)
  ```
- **Key fields:**
  - `commit_time_pt` - Last modified time (Pacific Time)
  - `commit_datetime` - Commit timestamp for age calculation
  - `pr_created_at_iso` - PR creation timestamp (ISO format)
- **Children:** None
- **Expansion policy:** Non-collapsible (leaf node)
- **Special behavior:**
  - Compact age format (e.g., "8h 39m ago", "16d 23h ago")
  - Grey text for subtle appearance

**Example:**
```
├─ (modified 2026-01-07 10:36 PT, created 2025-12-22 03:49, 8h 39m ago)
```

---

### 7. **`PRNode`** (extends `BranchNode`)
- **Purpose:** Stores PR metadata (merged into parent `BranchInfoNode`, never rendered as separate line)
- **Key fields:**
  - `pr` - Full `PRInfo` object with PR details
- **Children:** None (absorbed by parent)
- **Special behavior:**
  - Not rendered in the tree (metadata merged into `BranchInfoNode`)
  - Provides PR title, number, state, base branch for tooltip

**Note:** This node exists in the tree structure but is filtered out during rendering.

---

### 8. **`PRStatusNode`** (extends `BranchNode`)
- **Purpose:** Shows the aggregate PASSED/FAILED/RUNNING status with check counts
- **Display format:**
  ```
  [triangle] PASSED/FAILED/RUNNING [GitHub icon] counts [review status] [conversations]
  ```
- **Key fields:**
  - `pr` - `PRInfo` object for status calculation
  - `context_key` - Stable key for DOM id generation
- **Children:** `CIJobTreeNode` instances (individual checks)
- **Expansion policy:**
  - ✅ **PASSED**: Collapsed (▶) - CI children hidden
  - ⚠️ **FAILED**: Expanded (▼) - CI children visible
  - 🔄 **RUNNING**: Expanded (▼) - CI children visible
- **Special behavior:**
  - Always visible as a line (controlled by parent `BranchInfoNode`)
  - Calculates counts from `gh pr checks` data
  - Shows review status (✅ Approved, 🔴 Changes Requested)
  - Shows unresolved conversation count
  - GitHub icon links to commit checks page

**Example:**
```
└─ ▶ PASSED  3 ✓26 ✗2, 💬 Unresolved: 28
   ├─ ✓ Build and push Dynamo docker images (amd64) (6m)
   └─ ✗ deploy-test-vllm (disagg_router) (2m)
```

---

### 9. **`CIJobTreeNode`** (extends `BranchNode`)
- **Purpose:** Individual CI check/job from GitHub Actions
- **Display format:**
  ```
  [triangle] [icon] check-name (duration) [log link]
  ```
- **Key fields:**
  - `job_id` - Unique job identifier
  - `display_name` - Human-readable check name
  - `status` - `success`, `failure`, `in_progress`, `pending`, `cancelled`, `skipped`, `unknown`
  - `is_required` - Whether this is a required check (branch protection)
  - `duration` - Human-readable duration (e.g., "6m", "2h 15m")
  - `url` - GitHub Actions job URL
  - `raw_log_href` - Local raw log file path (if cached)
  - `error_snippet_text` - Extracted error snippet from log
  - `failed_check` - Optional `FailedCheck` object with details
- **Children:** Nested `CIJobTreeNode` instances (job steps, matrix jobs)
- **Icons:**
  - ✓ (green circle) - Required success
  - ✓ (green checkmark) - Optional success
  - ✗ (red circle) - Required failure
  - ✗ (red X) - Optional failure
  - ⏳ (hourglass) - In progress
  - ⏸ (pause) - Pending/queued
  - ✖️ (multiply) - Cancelled
  - ? (grey) - Unknown/skipped
- **Expansion policy:**
  - **Success-like** (passed, skipped, unknown with 0 duration): Collapsed
  - **Required failure**: Expanded
  - **Running/pending**: Expanded
  - **Optional failure only**: Collapsed
- **Special behavior:**
  - Shows "▶ Snippet" button if log available
  - Groups by architecture (amd64 first, then arm64) when matrix jobs present
  - Shows nested job steps in hierarchical tree

**Example:**
```
├─ ▼ ✗ deploy-test-vllm (disagg_router) (2m) [log] ▶ Snippet
│  ├─ ✓ checkout (5s)
│  ├─ ✗ run-tests (1m 45s)
│  └─ ✓ cleanup (10s)
```

---

### 10. **`PRURLNode`** (extends `BranchNode`)
- **Purpose:** Shows a clickable GitHub PR link with title
- **Display format:**
  ```
  📖 PR#1234: Title of the pull request
  ```
- **Key fields:**
  - `url` - GitHub PR URL
  - `pr_number` - PR number
  - `title` - PR title
- **Children:** None
- **Special behavior:**
  - Icon: 📖 (open book emoji)
  - Title truncated if too long
  - Opens in new tab

**Example:**
```
└─ 📖 PR#5050: refactor: remove "dev" stage from Dockerfile.* and centralize them into Dockerfile.dev
```

---

### 9. **`RerunLinkNode`** (extends `BranchNode`)
- **Purpose:** Shows a link to rerun failed GitHub Actions workflow
- **Display format:**
  ```
  🔄 Rerun workflow [run 123456789]
  ```
- **Key fields:**
  - `url` - GitHub Actions run URL
  - `run_id` - Workflow run ID
- **Children:** None
- **Special behavior:**
  - Only shown when CI has failures
  - Includes copy button for `gh run rerun --failed` command

**Example:**
```
└─ 🔄 Rerun workflow [run 20774607783]
```

---

### 10. **`BlockedMessageNode`** (extends `BranchNode`)
- **Purpose:** Shows when a PR is blocked (e.g., merge conflicts, required checks)
- **Display:** `🚫 Blocked: <reason>`
- **Children:** None

---

### 11. **`ConflictWarningNode`** (extends `BranchNode`)
- **Purpose:** Shows merge conflict warnings
- **Display:** `⚠️ <message>`
- **Children:** None

---

## Tree Rendering: `TreeNodeVM`

All `BranchNode` subclasses convert to `TreeNodeVM` for rendering via `to_tree_vm()`.

### `TreeNodeVM` Fields
- `node_key` - Stable key for DOM id generation (survives regeneration)
- `label_html` - Full HTML content for the line
- `children` - List of child `TreeNodeVM` instances
- `collapsible` - Whether to show expand/collapse triangle
- `default_expanded` - Initial expanded state (True = ▼, False = ▶)
- `triangle_tooltip` - Optional tooltip for the triangle
- `noncollapsible_icon` - Icon for non-collapsible nodes ("square" = ■, "" = blank)

---

## Expansion Policies Summary

| Node Type | Default Expansion | Controlled By |
|-----------|-------------------|---------------|
| `RepoNode` | Always expanded | User (always has triangle) |
| `SectionNode` | Always expanded | User (always has triangle) |
| `BranchInfoNode` | **Always expanded** (when has children) | Shows `PRStatusNode` line |
| `PRStatusNode` | **Collapsed for PASSED**, expanded for FAILED/RUNNING | CI status |
| `CIJobTreeNode` | Collapsed for success, expanded for failure/running | Job status |

---

## Remote vs Local Branches

### Local Branches (`show_local_branches.py`)
- Scans git repositories on disk
- Shows local branch modifications and commit times
- Full CI details with error snippets from local log cache

### Remote Branches (`show_remote_branches.py`)
- Fetches PRs from GitHub API by username
- Shows only PRs (no local-only branches)
- Same tree structure and expansion logic as local
- Uses shared helpers from `show_local_branches.py`:
  - `_format_age_compact()`
  - `_format_branch_metadata_suffix()`
  - `_format_pr_tooltip()`
  - `_pr_needs_attention()`
  - `_strip_repo_prefix_for_clipboard()`

---

## Shared Helper Functions

### Formatting Helpers (exported from `show_local_branches.py`)
- **`_format_age_compact(dt)`** - "(8h 39m ago)" format
- **`_format_branch_metadata_suffix(commit_time_pt, commit_datetime, pr_created_at_iso)`** - "(modified ..., created ..., ... ago)"
- **`_format_base_branch_inline(pr)`** - "→ main" or "→ release/0.8.0"
- **`_format_commit_tooltip(commit_message)`** - Full commit message for tooltip (escaped)
- **`_format_pr_number_link(pr)`** - "#5050" as GitHub PR link
- **`_strip_repo_prefix_for_clipboard(branch_name)`** - Removes "ai-dynamo/" prefixes

### Status Helpers
- **`_pr_needs_attention(pr)`** - True if PR has running work or required failures

### CI Hierarchy Builder
- **`_build_ci_hierarchy_nodes(repo_path, pr, github_api, ...)`** - Builds the `CIJobTreeNode` tree from GitHub check runs

---

## Naming Convention

When discussing these nodes:
- **"Repo node"** or **"repository node"** → `RepoNode`
- **"Section node"** → `SectionNode`
- **"Branch node"** or **"branch line"** → `BranchInfoNode`
- **"Commit message line"** or **"commit node"** → `CommitMessageNode`
- **"Metadata line"** or **"metadata node"** → `MetadataNode`
- **"Status node"** or **"PR status line"** → `PRStatusNode`
- **"CI node"** or **"check node"** or **"job node"** → `CIJobTreeNode`
- **"PR link node"** → `PRURLNode`
- **"Rerun link node"** → `RerunLinkNode`

---

## Visual Example (Full Tree)

```
▼ dynamo/                                          ← RepoNode
│  ├─ Branches with PRs                            ← SectionNode
│  │  ├─ [copy] ✖ branch-name → main [SHA]        ← BranchInfoNode (closed PR)
│  │  │  ├─ commit message (#1234)                 ← CommitMessageNode
│  │  │  ├─ (modified ..., created ..., ago)      ← MetadataNode
│  │  │  └─ ▶ PASSED  3 ✓26 ✗2                    ← PRStatusNode (collapsed)
│  │  │     ├─ ✓ check-1 (6m) [log]               ← CIJobTreeNode (hidden when collapsed)
│  │  │     └─ ✗ check-2 (2m) [log] ▶ Snippet     ← CIJobTreeNode (hidden when collapsed)
│  │  └─ [copy] branch-2 → release/0.8.0 [SHA]    ← BranchInfoNode (open PR)
│  │     ├─ fix: resolve memory leak (#2345)       ← CommitMessageNode
│  │     ├─ (modified ..., created ..., ago)      ← MetadataNode
│  │     └─ ▼ FAILED  2 ✓24 ✗1                    ← PRStatusNode (expanded)
│  │        ├─ ✓ check-1 (5m)                      ← CIJobTreeNode
│  │        └─ ▼ ✗ check-2 (3m) [log] ▶ Snippet   ← CIJobTreeNode (expanded)
│  │           ├─ ✓ setup (10s)                    ← CIJobTreeNode (nested step)
│  │           ├─ ✗ test (2m 30s)                  ← CIJobTreeNode (nested step)
│  │           └─ ✓ cleanup (20s)                  ← CIJobTreeNode (nested step)
│  └─ Local-only branches                          ← SectionNode
│     └─ [copy] feature-branch [SHA]               ← BranchInfoNode (no PR, no children)
```

---

## Questions for Next Time

Use this reference to:
- Identify which node type to modify for specific UI changes
- Understand the tree hierarchy when debugging rendering issues
- Know which helper functions to reuse for new features
- Clarify expansion policy requirements ("should the status node expand?" → "the PRStatusNode controls its CI children")

**Example queries:**
- "Update the branch line to show..." → Modify `BranchInfoNode._format_html_content()`
- "Change when CI checks expand..." → Modify `PRStatusNode.to_tree_vm()` or `CIJobTreeNode._subtree_needs_attention()`
- "Add a new icon to the repo line..." → Modify `RepoNode._format_html_content()`

