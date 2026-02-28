---
name: analyze-ci
description: Analyze CI failure logs by automatically fetching and grepping for common error patterns
disable-model-invocation: true
allowed-tools: Bash, Read, Grep
---

# Analyze CI Failures Skill

Quickly analyze CI failure logs by fetching and searching for error patterns.

**Input:** CI log URL or file path
**Output:** Error summary with context

## Workflow

### Step 1: Fetch Log
If URL provided:
```bash
# Save to temp file FIRST (CRITICAL - see .cursorrules section 2.3.3)
curl -s "$LOG_URL" > /tmp/CI-<job-id>.log
```

If file path provided:
```bash
# Copy to temp file for consistent handling
cp "$LOG_PATH" /tmp/CI-analysis.log
```

### Step 2: Search for Error Patterns (Sequential)
Search for common error patterns in order of priority:

**Python/pytest errors:**
```bash
grep -E "ERROR at setup|ModuleNotFoundError|ImportError|RuntimeError|AssertionError" /tmp/CI-*.log | head -20
```

**Build errors:**
```bash
grep -E "error:|ERROR:|fatal error|compilation terminated" /tmp/CI-*.log | head -20
```

**Docker build errors:**
```bash
grep -E "failed to|ERROR \[|RUN failed" /tmp/CI-*.log | head -20
```

**Test failures:**
```bash
grep -E "FAILED|ERRORS|tests.*ERROR" /tmp/CI-*.log | head -20
```

**Network errors:**
```bash
grep -E "network is unreachable|dial tcp|failed to do request|connection refused|timeout" /tmp/CI-*.log | head -20
```

### Step 3: Get Context Around Errors
For each error found, get surrounding context:
```bash
# Get 20 lines before and after first error occurrence
grep -B 20 -A 20 "<error pattern>" /tmp/CI-*.log | head -100
```

### Step 4: Analyze and Report
Provide:
1. **Error Type:** What kind of error (import, build, test, network, etc.)
2. **Root Cause:** Most likely cause based on error message
3. **Context:** Relevant surrounding log lines
4. **Suggested Fix:** What to try based on the error

## Common Error Patterns Reference

| Pattern | Meaning | Typical Fix |
|---------|---------|-------------|
| `ModuleNotFoundError` | Missing Python package | Add to requirements.txt |
| `ERROR at setup` | Pytest setup failure | Check fixtures, conftest.py |
| `network is unreachable` | Network connectivity | Retry, check DNS/firewall |
| `failed to build` | Docker build error | Check Dockerfile syntax |
| `FAILED tests/` | Test assertion failure | Fix test or implementation |
| `RuntimeError: CUDA` | GPU/CUDA issue | Check CUDA availability |

## Important Notes

- **ALWAYS** save log to temp file FIRST, then grep repeatedly (never `curl | grep`)
- CI logs can be 50,000+ lines - don't read sequentially
- Focus on first error occurrence (subsequent errors often cascade)
- Check for multiple error types (errors can be mixed)
- Provide specific line numbers when possible

## Arguments

Required: CI log URL or file path
Optional: Specific error pattern to search for

Example invocations:
- `/analyze-ci https://github.com/org/repo/actions/runs/123456/job/789`
- `/analyze-ci /tmp/failed-build.log`
- `/analyze-ci <url> ImportError`
