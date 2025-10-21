# Overall instructions - Project Instructions

**Note**: For general coding conventions, style guidelines, and development practices, refer to `.cursorrules` in this directory. This file contains project-specific instructions for the Docker builder system.

## Project Overview

This project contains `dynamo_docker_builder.py`, an automated Docker build and test pipeline system for the Dynamo project. It builds and tests multiple inference frameworks (VLLM, SGLANG, TRTLLM) across different target environments (base, dev, local-dev), generates HTML reports, and sends email notifications.

## Key Files

- **dynamo_docker_builder.py**: Main builder script with parallel execution, HTML reporting, email notifications, and commit history
- **common.py**: Shared utilities including terminal width detection, path helpers, and common functions
- **run_docker_builder.sh**: Shell wrapper for running the builder
- **update_html_pages.sh**: Automated cron script that updates branch status and commit history HTML pages

## Code Conventions

> **Note**: General Python style guidelines, error handling patterns, and testing practices are documented in `.cursorrules`. This section covers project-specific conventions.

### Git Integration
- All commit message parsing expects format: `title (#PR_NUMBER)`
- Extract PR numbers using regex: `r'\(#(\d+)\)'`
- GitHub repo is always: `https://github.com/ai-dynamo/dynamo`
- Use full SHA for GitHub URLs, short SHA (7 chars) for display

### HTML Report Generation
- Generate two versions when email is requested:
  - File version: relative paths (just filenames)
  - Email version: absolute URLs with hostname
- Use helper function `get_log_url()` to abstract URL generation
- Pass `use_absolute_urls` flag to control URL format
- Default hostname: `keivenc-linux`
- Default HTML path: `/dynamo_ci/logs`

### Log File Paths
- Logs stored in: `~/nvidia/dynamo_ci/logs/YYYY-MM-DD/`
- Log filename format: `YYYY-MM-DD.{sha_short}.{framework}-{target}-{type}.log`
- HTML report: `YYYY-MM-DD.{sha_short}.report.html`
- Always use date subdirectories
- Use `log_dir.parent` to get root logs directory

### Process Management
- Use lock files to prevent concurrent runs
- Lock file: `.dynamo_docker_builder.py.lock`
- Store PID in lock file
- Check if process is still running before acquiring lock
- Clean up stale locks automatically

### Email Notifications
- Use SMTP (localhost:25) for emails
- Email subject format: `[DynamoDockerBuilder] {status} - {sha_short}`
- HTML email body with clickable links (absolute URLs)
- Plain text fallback for email clients

## Testing Practices

> **Note**: General pytest guidelines (including `--basetemp=/tmp/pytest_temp` parameter) are documented in `.cursorrules`. This section covers Docker builder-specific testing commands.

### Quick Test (Single Framework)
```bash
python3 dynamo_docker_builder.py --sanity-check-only --framework sglang --force-run --email <email>
```

### Parallel Build with Skip
```bash
python3 dynamo_docker_builder.py --skip-build-if-image-exists --parallel --force-run --email <email>
```

### Full Build
```bash
python3 dynamo_docker_builder.py --parallel --force-run --email <email>
```

## Common Patterns

### Reading Git Commit Information
```python
repo = git.Repo(repo_path)
commit = repo.commit(sha)
author_name = commit.author.name
author_email = commit.author.email
commit_date = datetime.fromtimestamp(commit.committed_date)
commit_message = commit.message
```

### Extracting PR Number from Commit Message
```python
import re
first_line = commit.message.split('\n')[0] if commit.message else ""
pr_match = re.search(r'\(#(\d+)\)', first_line)
if pr_match:
    pr_number = pr_match.group(1)
    pr_link = f"https://github.com/ai-dynamo/dynamo/pull/{pr_number}"
```

### Generating GitHub Links
```python
# Commit link (use full SHA)
commit_link = f"https://github.com/ai-dynamo/dynamo/commit/{repo_sha}"

# PR link (use extracted PR number)
pr_link = f"https://github.com/ai-dynamo/dynamo/pull/{pr_number}"
```

### Terminal Width Detection
```python
from common import get_terminal_width

width = get_terminal_width()
# Fallback to 80 if detection fails
```

## Important Notes

- Always test with `--force-run` to bypass checks during development
- Use `--dry-run` to see what would happen without executing
- Check process locks before making changes to lock file handling
- HTML reports must work standalone (relative paths for log links)
- Email HTML needs absolute URLs for external viewing
- All GitHub links should be clickable in HTML output
- Commit SHAs in headers should have white underline styling for visibility

## Commit History Feature

### Overview
The `--show-commit-history` flag displays recent commits with their composite SHAs. It supports both terminal and HTML output modes with integrated caching for performance.

### Caching System
- **Cache file**: `~/nvidia/dynamo_ci/.commit_history_cache.json`
- **Format**: JSON mapping of commit SHA (full) -> composite SHA
- **Purpose**: Avoid expensive git checkout + composite SHA recalculation
- **Cache is updated**: Only when new commits are encountered or composite SHA changes
- **Performance**: Cached lookups are nearly instant vs ~1-2 seconds per commit calculation

### Usage Examples

**Terminal output with caching**:
```bash
python3 dynamo_docker_builder.py --show-commit-history --max-commits 50 --repo-path ~/nvidia/dynamo_ci
```

**HTML output with caching** (generates `~/nvidia/dynamo_ci/logs/commit-history.html`):
```bash
python3 dynamo_docker_builder.py --show-commit-history --html --max-commits 50 --repo-path ~/nvidia/dynamo_ci
```

**Verbose mode** (shows cache hits/misses):
```bash
python3 dynamo_docker_builder.py --show-commit-history --html --max-commits 50 --repo-path ~/nvidia/dynamo_ci --verbose
```

### HTML Output Features
- **Clickable commit SHAs**: Link to GitHub commit page
- **Clickable PR numbers**: Extract `(#1234)` from commit messages and link to GitHub PR
- **Docker image detection**: Shows expandable list of Docker images containing each commit SHA
- **Expandable sections**: Uses HTML `<details>` tag for clean UI
- **GitHub-style CSS**: Familiar look and feel

### Implementation Details
- **Commit SHA format**: Full SHA for GitHub links, 9-char short SHA for display
- **PR extraction regex**: `r'\(#(\d+)\)'`
- **Docker image query**: Uses `docker images --format "{{.Repository}}:{{.Tag}}"` once for all commits
- **Cache management**: Loaded at start, saved after updates
- **Error handling**: Gracefully handles missing cache file, invalid JSON, etc.

## Automated HTML Page Updates

### Cron Script: `update_html_pages.sh`
Runs every 5 minutes during daytime (9am-9pm) via cron to update multiple HTML pages:

**Schedule**:
```cron
*/5 9-20 * * * $HOME/nvidia/dynamo-utils/update_html_pages.sh
```

**Tasks performed**:
1. **Cleanup old logs** (runs first)
   - Keeps only the last 10 non-empty dated directories in `~/nvidia/dynamo_ci/logs/`
   - Deletes older directories to save disk space
   - Logs all cleanup actions with directory counts

2. Updates branch status HTML (`$HOME/nvidia/index.html`)
   - Calls `show_dynamo_branches.py --html`
   - Shows status of all dynamo branches

3. Updates commit history HTML (`~/nvidia/dynamo_ci/logs/commit-history.html`)
   - Calls `dynamo_docker_builder.py --show-commit-history --html --max-commits 50`
   - Leverages caching for fast updates (only calculates new commits)
   - Shows last 50 commits with composite SHAs and Docker images

**Log file**: `~/nvidia/dynamo-utils/update_html_pages.log`

**Cleanup function** (`cleanup_old_logs()`):
- Finds all dated directories matching `202*` pattern
- Checks each directory for non-empty status (has files)
- Sorts by date in descending order (newest first)
- Keeps the 10 most recent non-empty directories
- Deletes all older directories
- Example log: `2025-10-18 20:15:50 - Cleaning up old logs: 18 non-empty directories found, keeping last 10, deleting 8 oldest`

**Log format**:
```
YYYY-MM-DD HH:MM:SS - Updated ~/nvidia/index.html
YYYY-MM-DD HH:MM:SS - Updated commit-history.html
```

### Performance Optimization
- **First run**: ~50-100 seconds (calculates all 50 commits)
- **Subsequent runs**: ~5-10 seconds (only processes new commits, rest from cache)
- **Cache file size**: ~5-10KB for 50 commits
- **No cache invalidation needed**: Composite SHAs are deterministic based on file content

## Image Size Population

### Overview
Docker image sizes are automatically populated in HTML reports for all BUILD tasks, regardless of whether images were just built or were skipped (using `--skip-build-if-image-exists`).

### Implementation
**Location**: `dynamo_docker_builder.py:1810-1851` and `dynamo_docker_builder.py:2003-2004`

**Method**: `_populate_image_sizes(pipeline: BuildPipeline)`
- Called after task execution completes (but before HTML report generation)
- Only runs when NOT in dry-run mode
- Queries Docker for all BUILD tasks that have an `image_tag`
- Populates `task.image_size` with format "XX.X GB"
- Logs each image size as it's detected

**Example output**:
```
Populating image sizes...
  ✅ dynamo-base:v0.1.0.dev.ca3daddc0-vllm: 23.6 GB
  ✅ dynamo:v0.1.0.dev.ca3daddc0-vllm: 18.2 GB
  ✅ dynamo:v0.1.0.dev.ca3daddc0-vllm-local-dev: 31.5 GB
  ...
```

### Why This Approach
- **Centralized**: All image size detection happens in one place after task execution
- **Works for all scenarios**: Handles both newly built images and existing images (when using `--skip-build-if-image-exists`)
- **Non-blocking**: Uses try-except to gracefully handle missing images (e.g., for failed builds)
- **Efficient**: Only queries Docker once per image after all tasks complete

### Previous Approach (Deprecated)
Previously, image size detection was attempted during task execution in `execute_task()`, but this only worked for images that were actively being built. The new approach works universally.

