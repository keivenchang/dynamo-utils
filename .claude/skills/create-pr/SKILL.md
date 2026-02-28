---
name: create-pr
description: Create a pull request with proper title and description following project conventions
disable-model-invocation: true
allowed-tools: Bash, Read
---

# Create Pull Request Skill

Create a pull request using `gh pr create` following project conventions.

## Workflow

### Step 1: Gather Full Context (Parallel)
Run these commands in parallel to understand the FULL scope of changes:
```bash
git status  # See untracked files (NEVER use -uall flag)
git diff --staged  # See staged changes
git diff  # See unstaged changes
git branch --show-current  # Get current branch name
git log origin/main..HEAD --oneline  # Get ALL commits in this branch
git diff origin/main...HEAD  # Get FULL diff from base branch
```

**CRITICAL:** Analyze ALL commits that will be in the PR, not just the latest commit!

### Step 2: Check Remote Status
```bash
# Check if branch tracks remote and is up to date
git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null || echo "No upstream"
git status -sb  # Check ahead/behind status
```

### Step 3: Draft PR Title and Description
Analyze ALL changes from Step 1:
- Keep title short (under 70 characters)
- Use description/body for details, not the title
- Focus on "why" and user impact, not implementation details

Format:
```markdown
## Summary
<1-3 bullet points covering ALL changes in the PR>

## Test plan
[Bulleted markdown checklist of TODOs for testing the pull request]

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

### Step 4: Create Branch and Push (if needed)
If no upstream branch or not up to date:
```bash
# Create new branch if needed
git checkout -b <branch-name>  # Only if needed

# Push to remote with -u flag
git push -u origin <branch-name>
```

### Step 5: Create PR
Use heredoc for body to ensure correct formatting:
```bash
gh pr create --title "PR title here" --body "$(cat <<'EOF'
## Summary
- Change 1
- Change 2
- Change 3

## Test plan
- [ ] Test scenario 1
- [ ] Test scenario 2
- [ ] Test scenario 3
EOF
)"
```

### Step 6: Report Success
Return the PR URL from `gh pr create` output so user can view it.

## Important Notes

- **CRITICAL:** Analyze ALL commits in the branch, not just the latest one
- **DO NOT** use TodoWrite or Task tools
- Look at FULL `git diff origin/main...HEAD` to understand complete scope
- If unsure about base branch, check `.github/workflows/` or ask user
- Return PR URL at the end

## Arguments

If arguments provided: Use as additional context for PR description
If no arguments: Follow full workflow above
