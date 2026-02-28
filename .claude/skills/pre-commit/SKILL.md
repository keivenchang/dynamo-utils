---
name: pre-commit
description: Run pre-commit checks for Rust and Python changes before committing
disable-model-invocation: true
allowed-tools: Bash, Read
---

# Pre-Commit Checks Skill

Run all pre-commit validation checks before creating a commit.

**Based on:** `.cursorrules` Section 7.2.7 Quick Pre-Commit Checklist

## Workflow

### Step 1: Determine What Changed
```bash
git status --short
```

Check if changes include:
- Rust files (`.rs`, `Cargo.toml`, `Cargo.lock`)
- Python files (`.py`)
- Scripts or HTML generators

### Step 2: Run Appropriate Checks

**For Rust-only changes:**
```bash
cargo fmt && echo "=====" && cargo clippy --no-deps --all-targets -- -D warnings && echo "=====" && cargo test --locked --all-targets
```

**For Rust + Python changes:**
```bash
cargo fmt && echo "=====" && cargo clippy --no-deps --all-targets -- -D warnings && echo "=====" && cargo test --locked --all-targets && echo "=====" && pre-commit run --all-files
```

**For Python-only changes:**
```bash
pre-commit run --all-files
```

**If scripts or HTML generators changed:**
```bash
# Syntax check Python files
python3 -m py_compile <touched_py_files>

# Run relevant generator/update script
# Example: python3 update_html_pages.py --show-commit-history
```

### Step 3: Report Results
- ✅ If all checks pass: Report success
- ❌ If checks fail: Show errors and suggest fixes

## Expected Results
- All checks pass with no errors
- Formatting is applied automatically (cargo fmt)
- No clippy warnings
- All tests pass
- Pre-commit hooks pass

## Important Notes
- **cargo fmt** runs first and auto-fixes formatting
- **clippy** checks for code quality issues (treats warnings as errors with `-D warnings`)
- **cargo test** runs with `--locked` to ensure Cargo.lock is up to date
- **pre-commit** runs all configured hooks (ruff, mypy, YAML/JSON validation, etc.)
- Commands are chained with `&&` so they stop at first failure
- Use `echo "====="` separators for readability

## Common Failures and Fixes

| Error | Fix |
|-------|-----|
| Clippy warnings | Fix the warnings shown in output |
| Test failures | Fix failing tests |
| Pre-commit ruff errors | Run `ruff check --fix` |
| Pre-commit mypy errors | Fix type annotations |
| Cargo.lock out of sync | Run `cargo build` or `cargo update` |

## Arguments
Optional: Specify which check to run (rust, python, all)
If no arguments: Auto-detect based on changed files
