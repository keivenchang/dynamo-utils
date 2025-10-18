# DynamoDockerBuilder V2 - Project Instructions

## Project Overview

This project contains `dynamo_docker_builder_v2.py`, an automated Docker build and test pipeline system for the Dynamo project. It builds and tests multiple inference frameworks (VLLM, SGLANG, TRTLLM) across different target environments (base, dev, local-dev), generates HTML reports, and sends email notifications.

## Key Files

- **dynamo_docker_builder_v2.py**: Main V2 builder script with parallel execution, HTML reporting, and email notifications
- **common.py**: Shared utilities including terminal width detection, path helpers, and common functions
- **docker_build_orchestrator.py**: Legacy orchestrator (V1 system)
- **run_docker_builder.sh**: Shell wrapper for running the builder

## Code Conventions

### Python Style
- Use Python 3.10+ features (dataclasses, type hints, pathlib)
- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Prefer pathlib.Path over string paths
- Use dataclasses for structured data (Task, BuildPipeline, etc.)

### Function Signatures
- Always include return type hints
- Use Optional[T] for nullable return types
- Default parameters come last
- Boolean flags should default to False

Example:
```python
def generate_html_report(
    pipeline: BuildPipeline,
    repo_path: Path,
    repo_sha: str,
    log_dir: Path,
    date_str: str,
    hostname: str = "keivenc-linux",
    html_path: str = "/dynamo_ci/logs",
    use_absolute_urls: bool = False
) -> str:
```

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
- Lock file: `.dynamo_docker_builder_v2.py.lock`
- Store PID in lock file
- Check if process is still running before acquiring lock
- Clean up stale locks automatically

### Email Notifications
- Use SMTP (localhost:25) for emails
- Email subject format: `[DynamoDockerBuilder] {status} - {sha_short}`
- HTML email body with clickable links (absolute URLs)
- Plain text fallback for email clients

## Testing Practices

### Quick Test (Single Framework)
```bash
python3 dynamo_docker_builder_v2.py --sanity-check-only --framework sglang --force-run --email <email>
```

### Parallel Build with Skip
```bash
python3 dynamo_docker_builder_v2.py --skip-build-if-image-exists --parallel --force-run --email <email>
```

### Full Build
```bash
python3 dynamo_docker_builder_v2.py --parallel --force-run --email <email>
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
