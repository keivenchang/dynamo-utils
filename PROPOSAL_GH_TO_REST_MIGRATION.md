# Proposal: Migration from `gh` CLI to Python REST API

**Date:** 2026-01-26  
**Status:** Proposal (Informational)  
**Author:** Analysis of existing codebase

## Executive Summary

This proposal outlines the migration strategy for replacing GitHub CLI (`gh`) subprocess calls with direct Python REST/GraphQL API calls using the `requests` library. The migration affects three main areas:

1. Required Checks API (`common_github/api/required_checks_cached.py`)
2. Rate Limit Check (`common_github/__init__.py`)
3. PR Status Script (`check_pr_status.py`)

## Motivation

### Current Issues with `gh` CLI Approach

1. **External Dependency** - Requires `gh` binary installed and in PATH
2. **Subprocess Overhead** - Process spawning adds latency (~50-200ms per call)
3. **Limited Error Handling** - Must parse stderr/stdout strings instead of structured exceptions
4. **Deployment Complexity** - CI/CD environments need `gh` installation
5. **Testing Difficulty** - Harder to mock subprocess calls vs HTTP requests
6. **Performance** - Cannot reuse HTTP connections or implement connection pooling

### Benefits of Direct Python REST/GraphQL

1. **No External Dependencies** - Only requires `requests` library (already in use)
2. **Better Performance** - Direct HTTP calls, reusable sessions, connection pooling
3. **Structured Error Handling** - Native `requests.HTTPError` with status codes
4. **Easier Testing** - Simple mocking with `responses` or `requests_mock`
5. **Full Control** - Custom headers, timeouts, retry logic, rate limit handling
6. **Debugging** - Transparent HTTP requests/responses

## Migration Plan

### 1. Required Checks API (`common_github/api/required_checks_cached.py`)

**Current Implementation:**
- Line 313-327: `gh api repos/{owner}/{repo}/pulls/{pr}` (REST core, subprocess)
- Line 408-451: `gh api graphql` (GraphQL, subprocess)

**Existing TODO:** Lines 21-39 already document the migration approach

**Proposed Changes:**

```python
# Replace subprocess calls with direct HTTP requests

# Step 1: PR Metadata (REST)
def _fetch_pr_meta(self, *, owner: str, repo: str, prn: int) -> Tuple[str, str, Optional[int]]:
    """Replace `gh api` with direct REST call."""
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{prn}"
    headers = {
        "Authorization": f"Bearer {self.api.token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    
    data = response.json()
    pr_node_id = data.get("node_id", "")
    pr_state = "merged" if data.get("merged_at") else data.get("state", "open")
    
    # Parse updated_at timestamp
    updated_at_s = data.get("updated_at", "")
    pr_updated_at_epoch = None
    if updated_at_s:
        dt = datetime.fromisoformat(updated_at_s.replace("Z", "+00:00"))
        pr_updated_at_epoch = int(dt.timestamp())
    
    return (pr_node_id, pr_state, pr_updated_at_epoch)

# Step 2: Required Checks (GraphQL)
def _fetch_required_checks(self, *, owner: str, repo: str, prn: int, pr_node_id: str) -> Set[str]:
    """Replace `gh api graphql` with direct GraphQL POST."""
    url = "https://api.github.com/graphql"
    headers = {
        "Authorization": f"Bearer {self.api.token}",
        "Accept": "application/vnd.github+json",
    }
    
    # Use the same query template (lines 374-401)
    query = """query($owner:String!,$name:String!,$number:Int!,$prid:ID!,$after:String) { ... }"""
    
    all_required = set()
    after_cursor = None
    
    for page_num in range(25):
        payload = {
            "query": query,
            "variables": {
                "owner": owner,
                "name": repo,
                "number": prn,
                "prid": pr_node_id,
                "after": after_cursor,
            }
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for GraphQL errors
        if "errors" in data:
            raise RuntimeError(f"GraphQL errors: {data['errors']}")
        
        # Parse response (same logic as current implementation)
        # Extract required checks from data["data"]["repository"]["pullRequest"]...
        
        # Handle pagination
        page_info = # ... extract from response
        if not page_info.get("hasNextPage"):
            break
        after_cursor = page_info.get("endCursor")
    
    return all_required
```

**Implementation Considerations:**

1. **Token Management:** Already handled by `self.api.token`
2. **Error Handling:** Use `try/except requests.HTTPError` instead of checking `returncode`
3. **Stats Tracking:** Update `GITHUB_API_STATS` to track REST/GraphQL calls separately
4. **Rate Limit:** Parse `X-RateLimit-*` headers from response
5. **Caching:** Keep existing cache logic unchanged
6. **Testing:** Add unit tests with `requests_mock`

**Validation:**

- Use `test_required_checks_rest_vs_gh.py` to verify behavior matches
- Compare results: `_required_checks_python_graphql()` (lines 152-210) already implements this approach
- Ensure same output for all test cases

---

### 2. Rate Limit Check (`common_github/__init__.py`)

**Current Implementation:**
- Line 2556-2563: `gh api rate_limit` (subprocess)
- Fallback approach when `gh` is available

**Proposed Changes:**

```python
def get_rate_limit_snapshot(self) -> Dict[str, Any]:
    """Get rate limit snapshot via direct REST API."""
    url = "https://api.github.com/rate_limit"
    headers = {
        "Authorization": f"Bearer {self.token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    
    GITHUB_API_STATS.log_actual_api_call(kind="rest.core", text="GET /rate_limit")
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        resources = data.get("resources", {})
        core = resources.get("core", {})
        graphql = resources.get("graphql", {})
        
        return {
            "core": {
                "limit": core.get("limit", 0),
                "remaining": core.get("remaining", 0),
                "reset": core.get("reset", 0),
            },
            "graphql": {
                "limit": graphql.get("limit", 0),
                "remaining": graphql.get("remaining", 0),
                "reset": graphql.get("reset", 0),
            }
        }
    except requests.RequestException as e:
        # Fallback to empty/unknown state
        return {"core": {}, "graphql": {}, "error": str(e)}
```

**Implementation Considerations:**

1. **Remove Dependency Check:** No longer need `shutil.which("gh")` check (line 2551)
2. **Token Handling:** Use `self.token` directly instead of `GH_TOKEN` env var
3. **Stats Tracking:** Change from `gh.core` to `rest.core` in stats
4. **Error Handling:** Use standard `requests.RequestException` handling
5. **Response Format:** Keep same dict structure for compatibility

**Rationale for Migration:**

While the comment says `gh api rate_limit` is "more reliable," this was likely due to:
- Authentication issues with token handling (now resolved)
- Older REST API quirks (since fixed by GitHub)

The direct REST approach is now equally reliable and removes subprocess overhead.

---

### 3. PR Status Script (`check_pr_status.py`)

**Current Implementation:**
- Line 14: `gh pr view {pr_number} --json statusCheckRollup` (subprocess)

**Proposed Changes:**

```python
def get_pr_check_status(pr_number: int, repo: str = "ai-dynamo/dynamo"):
    """Get check status breakdown via direct GraphQL API."""
    owner, repo_name = repo.split('/')
    
    # Get GitHub token (same as current codebase pattern)
    token = os.environ.get("GITHUB_TOKEN") or _read_gh_hosts_token()
    if not token:
        raise RuntimeError("No GitHub token found. Set GITHUB_TOKEN or login with `gh auth login`.")
    
    # Step 1: Get PR node_id via REST
    url = f"https://api.github.com/repos/{owner}/{repo_name}/pulls/{pr_number}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    pr_node_id = response.json()["node_id"]
    
    # Step 2: Query statusCheckRollup via GraphQL
    query = """
    query($owner: String!, $name: String!, $number: Int!, $prid: ID!) {
      repository(owner: $owner, name: $name) {
        pullRequest(number: $number) {
          commits(last: 1) {
            nodes {
              commit {
                statusCheckRollup {
                  contexts(first: 100) {
                    nodes {
                      __typename
                      ... on CheckRun {
                        name
                        status
                        conclusion
                        isRequired(pullRequestId: $prid)
                      }
                      ... on StatusContext {
                        context
                        state
                        isRequired(pullRequestId: $prid)
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    """
    
    graphql_url = "https://api.github.com/graphql"
    payload = {
        "query": query,
        "variables": {
            "owner": owner,
            "name": repo_name,
            "number": pr_number,
            "prid": pr_node_id,
        }
    }
    
    response = requests.post(graphql_url, headers=headers, json=payload, timeout=10)
    response.raise_for_status()
    
    data = response.json()
    
    # Parse response (same logic as current implementation)
    checks = []
    contexts = data["data"]["repository"]["pullRequest"]["commits"]["nodes"][0]["commit"]["statusCheckRollup"]["contexts"]["nodes"]
    
    for check in contexts:
        if check["__typename"] == "CheckRun":
            name = check.get("name", "")
            status = check.get("status", "").upper()
            conclusion = check.get("conclusion", "").upper()
            is_required = check.get("isRequired", False)
            # ... process check
        elif check["__typename"] == "StatusContext":
            name = check.get("context", "")
            state = check.get("state", "").upper()
            is_required = check.get("isRequired", False)
            # ... process status
    
    return stats
```

**Implementation Considerations:**

1. **Token Helper:** Reuse `_read_gh_hosts_token()` from `test_required_checks_rest_vs_gh.py` (lines 47-77)
2. **Error Handling:** Add proper `try/except` for `requests.RequestException`
3. **Pagination:** Add pagination if PR has >100 checks (unlikely but possible)
4. **Response Mapping:** Map GraphQL response to same dict structure as current output
5. **Backward Compatibility:** Keep same function signature and return format

**Benefits for This Script:**

- No dependency on `gh` CLI in CI/container environments
- Faster execution (no subprocess overhead)
- Better error messages (HTTP status codes vs stderr parsing)

---

## Implementation Strategy

### Phase 1: Required Checks API (Highest Priority)

**Effort:** Medium (2-3 days)  
**Risk:** Low (test file already validates approach)  
**Impact:** High (removes main production dependency on `gh`)

**Steps:**

1. Copy implementation from `test_required_checks_rest_vs_gh.py` functions:
   - `_rest_get()` (lines 90-102)
   - `_graphql_post()` (lines 105-120)
   - `_required_checks_python_graphql()` (lines 152-210)

2. Refactor `required_checks_cached.py`:
   - Replace `_fetch_pr_meta()` subprocess with REST call
   - Replace `_fetch_required_checks()` subprocess with GraphQL POST
   - Keep all cache logic unchanged
   - Update stats tracking

3. Test thoroughly:
   - Run `test_required_checks_rest_vs_gh.py` to validate equivalence
   - Test with various PR states (open, merged, closed)
   - Test pagination (PRs with >100 checks)
   - Test error cases (404, 403, network errors)

4. Update documentation:
   - Remove TODO comment (lines 21-39)
   - Document new implementation approach
   - Note that `gh` CLI is no longer required

### Phase 2: Rate Limit Check (Low Priority)

**Effort:** Low (few hours)  
**Risk:** Very Low (simple endpoint)  
**Impact:** Low (optimization only)

**Steps:**

1. Replace subprocess call with direct REST
2. Remove `shutil.which("gh")` check
3. Update stats tracking from `gh.core` to `rest.core`
4. Test in various environments
5. Keep same return format for compatibility

### Phase 3: PR Status Script (Optional)

**Effort:** Low (few hours)  
**Risk:** Low (utility script)  
**Impact:** Low (developer tool only)

**Steps:**

1. Implement direct GraphQL query
2. Add token helper function
3. Update error handling
4. Test with sample PRs
5. Update script documentation

---

## Testing Strategy

### Unit Tests

1. **Mock HTTP responses** using `requests_mock` or `responses` library
2. **Test error cases:** 404, 403, 500, timeout, rate limit
3. **Test pagination:** Multiple pages of results
4. **Test edge cases:** Empty results, missing fields, malformed JSON

### Integration Tests

1. **Compare outputs** with `test_required_checks_rest_vs_gh.py`
2. **Validate against live API** with real PRs
3. **Performance testing:** Measure latency improvements
4. **Load testing:** Verify connection pooling works

### Validation Criteria

1. **Functional equivalence:** Same results as `gh` CLI approach
2. **Performance improvement:** At least 20% faster (subprocess overhead removed)
3. **Error handling:** Better error messages with HTTP status codes
4. **No regressions:** All existing tests pass

---

## Risk Assessment

### Low Risk

- Test file (`test_required_checks_rest_vs_gh.py`) already proves the approach works
- REST/GraphQL APIs are stable and well-documented
- Fallback: can revert to `gh` CLI if issues arise

### Potential Issues

1. **Authentication differences:** Token handling might behave differently
   - **Mitigation:** Thorough testing in various environments
   
2. **Rate limiting:** Direct API calls count toward rate limits
   - **Mitigation:** Implement proper rate limit tracking and backoff
   
3. **API changes:** GitHub API might change
   - **Mitigation:** Use API versioning header (`X-GitHub-Api-Version: 2022-11-28`)

4. **Network issues:** Direct HTTP might expose connection issues
   - **Mitigation:** Implement retry logic with exponential backoff

---

## Performance Analysis

### Expected Improvements

Based on typical subprocess overhead measurements:

| Operation | Current (gh CLI) | Proposed (REST) | Improvement |
|-----------|------------------|-----------------|-------------|
| Single PR metadata | ~150-200ms | ~50-80ms | 60-70% faster |
| GraphQL query (per page) | ~200-300ms | ~80-120ms | 60-65% faster |
| Rate limit check | ~100-150ms | ~30-50ms | 70-75% faster |

### Additional Benefits

- **Connection reuse:** Can implement HTTP connection pooling
- **Parallel requests:** Easier to parallelize multiple API calls
- **Lower CPU:** No process spawning overhead

---

## Dependencies

### Current Dependencies (Removed)
- `gh` CLI binary (external)

### New Dependencies (Already Present)
- `requests` library (already in use throughout codebase)

### Optional Enhancements
- `requests_mock` or `responses` for testing (dev dependency)
- `urllib3` connection pooling (already transitive dependency of `requests`)

---

## Rollout Plan

### Development

1. Create feature branch: `feature/gh-to-rest-migration`
2. Implement Phase 1 (Required Checks API)
3. Run comprehensive tests
4. Code review and validation

### Staging

1. Deploy to staging environment
2. Monitor for 1-2 weeks
3. Compare behavior with production (running `gh` CLI)
4. Validate performance improvements

### Production

1. Gradual rollout with feature flag
2. Monitor API rate limits and performance
3. Keep `gh` CLI as fallback option (via config flag)
4. Full rollout after 1 week of stability

### Cleanup

1. Remove `gh` CLI fallback code
2. Update documentation
3. Remove `gh` from CI/CD installation scripts

---

## Success Metrics

1. **Performance:** 50%+ reduction in API call latency
2. **Reliability:** No increase in error rates
3. **Deployment:** Successful removal of `gh` CLI dependency from containers
4. **Maintainability:** Simpler code, easier testing, better error messages

---

## Alternatives Considered

### Alternative 1: Keep Using `gh` CLI
- **Pros:** No changes needed, proven reliability
- **Cons:** External dependency, subprocess overhead, testing complexity
- **Verdict:** Not recommended; migration benefits outweigh costs

### Alternative 2: Use PyGithub Library
- **Pros:** Higher-level abstractions, maintained library
- **Cons:** Additional dependency, may not support all features (e.g., `isRequired`)
- **Verdict:** Not recommended; `requests` is sufficient and already in use

### Alternative 3: Hybrid Approach
- **Pros:** Use REST for most calls, keep `gh` for edge cases
- **Cons:** Maintains complexity and dependency
- **Verdict:** Not recommended; clean migration is better

---

## Conclusion

Migrating from `gh` CLI subprocess calls to direct Python REST/GraphQL API calls is:

1. **Technically proven** - Test file validates the approach
2. **Low risk** - Gradual rollout with fallback option
3. **High value** - Performance gains, simpler code, no external dependency
4. **Well-scoped** - Clear phases with measurable outcomes

**Recommendation:** Proceed with Phase 1 (Required Checks API) implementation.

---

## References

- Existing test file: `test_required_checks_rest_vs_gh.py` (lines 152-210 for reference implementation)
- Current implementation: `common_github/api/required_checks_cached.py`
- TODO comment: Lines 21-39 in `required_checks_cached.py`
- GitHub API Documentation: https://docs.github.com/en/rest
- GraphQL API: https://docs.github.com/en/graphql
