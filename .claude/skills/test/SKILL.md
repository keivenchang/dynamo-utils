---
name: test
description: Run pytest with proper environment and flags, handling multiple pytest installations
disable-model-invocation: true
allowed-tools: Bash, Read
---

# Pytest Test Execution Skill

Run pytest tests with correct environment setup and flags.

**Based on:** `.cursorrules` Section 0 "Pytest (MUST)"

## Critical Context

**Multiple pytest installations exist:**
- System pytest at `/usr/local/bin/pytest` (OUTSIDE venv, cannot see venv packages)
- Venv pytest (correct one to use)

**ALWAYS use `python3 -m pytest`** to ensure venv pytest runs with correct `sys.path`

## Workflow

### Step 1: Set Environment Variables
```bash
export HF_HUB_OFFLINE=1
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
```

**Why:**
- `HF_HUB_OFFLINE=1`: Prevents Hugging Face from trying to download models during tests
- `HF_TOKEN`: Required for existing tests that need HF authentication

### Step 2: Run Pytest with Correct Flags
```bash
python3 -m pytest -xvv --basetemp=/tmp/pytest_temp --duration=0 <test-path>
```

**Flag explanations:**
- `-x`: Stop at first failure (fast analysis)
- `-vv`: Very verbose output (detailed error info)
- `--basetemp=/tmp/pytest_temp`: Use temp directory for test artifacts
- `--duration=0`: Show timing of all tests (helps detect slow/flaky tests)

### Step 3: Additional Pytest Options

**Run specific test:**
```bash
python3 -m pytest -xvv --basetemp=/tmp/pytest_temp --duration=0 tests/test_file.py::test_function
```

**Run with keyword filter:**
```bash
python3 -m pytest -xvv --basetemp=/tmp/pytest_temp --duration=0 -k "pattern"
```

**Run all tests in directory:**
```bash
python3 -m pytest -xvv --basetemp=/tmp/pytest_temp --duration=0 tests/
```

**Run with coverage:**
```bash
python3 -m pytest -xvv --basetemp=/tmp/pytest_temp --duration=0 --cov=<module> --cov-report=term-missing
```

## Complete Command Pattern

Combine environment setup and pytest in one command:
```bash
export HF_HUB_OFFLINE=1 HF_TOKEN="$(cat ~/.cache/huggingface/token)" && python3 -m pytest -xvv --basetemp=/tmp/pytest_temp --duration=0 <test-path>
```

## Common Issues

### "ModuleNotFoundError: No module named 'dynamo'"
**Cause:** Using system pytest instead of venv pytest
**Fix:** Use `python3 -m pytest` (not bare `pytest`)

### Tests hang or try to download models
**Cause:** Missing `HF_HUB_OFFLINE=1`
**Fix:** Export environment variable before running

### "No such file or directory: .pytest_cache"
**Cause:** Permission issues when using WORKSPACE_DIR
**Fix:** Run pytest from `$WORKSPACE_DIR` or use `--cache-dir=/tmp/pytest_cache`

## Arguments

Required: Test path or pattern
Optional: Additional pytest flags

Examples:
- `/test tests/test_router.py`
- `/test tests/ -k integration`
- `/test tests/test_file.py::test_func --pdb`
