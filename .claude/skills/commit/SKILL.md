---
name: commit
description: Create a git commit following project conventions with proper message format
disable-model-invocation: true
allowed-tools: Bash, Read, Grep
---

# Git Commit Skill

Create a git commit following the project's conventions and safety protocols.

## Git Safety Protocol

**NEVER:**
- Update git config
- Run destructive commands (push --force, reset --hard, checkout ., restore ., clean -f, branch -D) unless explicitly requested
- Skip hooks (--no-verify, --no-gpg-sign) unless explicitly requested
- Force push to main/master
- Create commits unless user explicitly asks
- Amend commits after hook failures (create NEW commits instead)

**ALWAYS:**
- Stage specific files by name (not `git add -A` or `git add .`)
- Create NEW commits rather than amending (unless explicitly requested)
- Add co-author line: `Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>`

## Workflow

### Step 1: Gather Context (Parallel)
Run these commands in parallel:
```bash
git status  # See untracked files (NEVER use -uall flag)
git diff --staged  # See staged changes
git diff  # See unstaged changes
git log --oneline -10  # See recent commit messages for style
```

### Step 2: Analyze and Draft
- Summarize the nature of changes (new feature, enhancement, bug fix, refactoring, test, docs)
- Ensure accurate description ("add" = new feature, "update" = enhancement, "fix" = bug fix)
- Do NOT commit files with secrets (.env, credentials.json, etc.)
- Draft concise commit message (1-2 sentences) focusing on "why" not "what"
- Match existing commit message style from git log

### Step 3: Stage and Commit (Sequential)
Run these commands sequentially:
```bash
# Stage relevant files by name
git add <file1> <file2> ...

# Create commit with heredoc for proper formatting
git commit -m "$(cat <<'EOF'
<commit message here>
EOF
)"

# Verify success
git status
```

### Step 4: Handle Failures
If commit fails due to pre-commit hook:
- Fix the issue
- Re-stage files
- Create a NEW commit (do NOT use --amend)

## Important Notes

- **NEVER** run commands beyond git operations (no file reads, no code exploration)
- **NEVER** use TodoWrite or Task tools
- **DO NOT** push unless user explicitly asks
- **NEVER** use `-i` flag (interactive mode not supported)
- **DO NOT** use `--no-edit` with git rebase
- If no changes to commit, do NOT create empty commit
- Use heredoc format for commit messages (see example above)

## Arguments

If arguments provided: Use as commit message (still add co-author line)
If no arguments: Follow full workflow above
