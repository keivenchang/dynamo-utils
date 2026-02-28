---
name: review-code
description: Review code changes against project conventions, security, and anti-patterns
disable-model-invocation: true
allowed-tools: Read, Grep, Bash
---

# Code Review Skill

Review code changes for security, performance, error handling, and adherence to project conventions.

**Based on:** `.cursorrules` Section 6.3 Code Review Guidelines + Anti-patterns

## Review Checklist

### 1. Security Vulnerabilities
- [ ] No API keys or secrets committed
- [ ] Environment variables used for config
- [ ] User inputs validated
- [ ] Parameterized queries for databases
- [ ] Principle of least privilege followed

### 2. Performance Issues
- [ ] No unnecessary computations in loops
- [ ] Appropriate data structures used
- [ ] Expensive operations cached when possible
- [ ] No blocking I/O in hot paths

### 3. Error Handling (CRITICAL)
Check against anti-patterns:
- [ ] **NO silent failures:** No `except Exception: pass`
- [ ] **NO hidden errors:** No `except Exception:` without `raise`
- [ ] **Specific exceptions only:** Catch specific types (FileNotFoundError, ValueError), not blanket Exception
- [ ] **Fail fast:** Let exceptions propagate unless you can handle them
- [ ] See `.cursorrules` Section 3.5.3 for complete anti-patterns

### 4. Python-Specific Rules
- [ ] **All imports at top:** No imports inside functions
- [ ] **No try/except for imports:** Import directly, fail fast
- [ ] **No defensive getattr():** Don't use `getattr()` for known typed attributes
- [ ] See `.cursorrules` Section 0 for critical rules

### 5. Rust-Specific Rules
- [ ] **cargo fmt applied:** All .rs files formatted
- [ ] **No clippy warnings:** Code passes clippy checks
- [ ] **Tests pass:** All tests in affected modules pass
- [ ] See `.cursorrules` Section 7.2 for Rust conventions

### 6. Test Coverage
- [ ] New functionality has tests
- [ ] Edge cases covered
- [ ] Error paths tested

### 7. Code Duplication
- [ ] No copy-paste code blocks
- [ ] Shared logic extracted to functions/modules
- [ ] If duplication is intentional, comment why and note files must stay in sync

### 8. Conventions
- [ ] Naming follows project standards (camelCase, PascalCase, snake_case)
- [ ] File structure follows project organization
- [ ] Documentation updated if needed

## Feedback Style

Use short, direct, conversational paragraphs. Output in plain text with triple backticks.

**Good feedback examples:**

```
Hey [Author], thank you for [what they did well]. This [positive outcome].

You may want to [suggestion]. Otherwise, [potential issue].

The tradeoff is [disadvantage if any]. Maybe [alternative approach].

Looking forward to getting this in, it will be good to have [benefit].
```

**For structural changes (refactors, Dockerfiles, shared scripts):**
- Call out what the author did well and why it works
- Mention disadvantages if any (complexity, readability, maintainability)
- For duplicated sections that must stay in sync: suggest adding comments and considering templating

## Anti-Pattern Detection

### CRITICAL: Check for these FORBIDDEN patterns:

**1. Imports inside functions:**
```python
❌ def foo():
    import json  # FORBIDDEN!
```

**2. Try/except hiding ImportError:**
```python
❌ try:
    import requests
except ImportError:
    requests = None  # FORBIDDEN!
```

**3. Defensive getattr() on known types:**
```python
❌ getattr(node, "job_name", "")  # FORBIDDEN if type is known!
```

**4. Silent failures:**
```python
❌ except Exception:  # FORBIDDEN without raise!
    pass
```

## Review Workflow

### Step 1: Get Changes
```bash
git diff  # or git diff <branch>
```

### Step 2: Check Files
For each changed file, check:
- Imports at top
- Error handling patterns
- Type usage (no unnecessary getattr)
- Security issues

### Step 3: Provide Feedback
Use conversational tone from examples above. Be specific about:
- What works well
- What needs fixing (with file:line references)
- Suggested improvements

## Arguments

Optional: File path or git diff reference
If no arguments: Review current git diff
