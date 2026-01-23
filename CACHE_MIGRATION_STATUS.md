# Cache Migration Status

## Overview
Migrating `common_github.py` from inline cache implementations to standardized `cache/*.py` modules using `BaseDiskCache` pattern.

## Completed ✅

### 1. Infrastructure Setup
- Created `/cache/` directory with 10 standardized cache modules
- All caches use `BaseDiskCache` with automatic stats tracking
- Import paths updated throughout codebase

### 2. Fully Migrated Caches

#### `JOB_LOG_CACHE` ✅
- **Old**: `self._job_log_cache` dict + `_load_disk_cache()` / `_save_to_disk_cache()`
- **New**: Direct `JOB_LOG_CACHE.get(job_id)` / `JOB_LOG_CACHE.put(job_id, snippet)`
- **File**: Lines ~5100-5225
- **Stats**: Updated `get_cache_stats()` to use `JOB_LOG_CACHE.get_cache_sizes()`

#### `PR_HEAD_SHA_CACHE` ✅
- **Old**: `self._pr_info_mem_cache` for head_sha lookups + `_load_pr_info_head_sha_disk_cache()`
- **New**: Direct `PR_HEAD_SHA_CACHE.get_if_fresh(key, ttl_s)` / `PR_HEAD_SHA_CACHE.put(key, sha, state=state)`
- **File**: Lines ~1896-1933
- **Logic**: Handles closed/merged PR caching automatically

## Remaining ⏳

### Caches to Migrate (6 caches, 51 usages)

| Cache Module | Old Implementation | Usages | Complexity |
|--------------|-------------------|--------|------------|
| `PR_CHECKS_CACHE` | `_pr_checks_mem_cache` | 7 | Medium |
| `PULLS_LIST_CACHE` | `_pulls_list_mem_cache` | 11 | Medium |
| `PR_BRANCH_CACHE` | `_pr_branch_mem_cache` | 5 | Low |
| `SEARCH_ISSUES_CACHE` | `_search_issues_mem_cache` | 7 | Medium |
| `REQUIRED_CHECKS_CACHE` | `_required_checks_pr_mem_cache` | 15 | High |
| `PR_INFO_CACHE` | `_pr_info_mem_cache` (enriched) | 6 | High |

### Additional Tasks
- Remove old `_load_*_disk_cache()` and `_save_*_disk_cache()` helper methods
- Update all `get_cache_stats()` references
- Remove old cache dict initializations
- Test all migrated implementations

## Migration Pattern

### Before:
```python
# Check memory cache
if key in self._cache_mem_cache:
    self._cache_hit("cache.mem")
    return self._cache_mem_cache[key]

# Check disk cache
disk = self._load_cache_disk_cache()
if key in disk:
    self._cache_mem_cache[key] = disk[key]
    self._cache_hit("cache.disk")
    return disk[key]

# ... fetch from API ...

# Save to both caches
self._cache_mem_cache[key] = value
self._save_cache_disk_cache(key, value)
```

### After:
```python
# Check cache (handles memory + disk automatically)
cached = CACHE.get_if_fresh(key, ttl_s)
if cached:
    return cached

# ... fetch from API ...

# Save to cache
CACHE.put(key, value)
```

## Benefits
- **Automatic stats tracking**: hit/miss/write counted by `BaseDiskCache`
- **Thread-safe**: Built-in locking
- **Consistent API**: All caches follow same pattern
- **Less boilerplate**: ~50% code reduction per cache
- **Testable**: Caches are standalone modules

## Next Steps
Continue migrating remaining 6 caches following the established pattern.
