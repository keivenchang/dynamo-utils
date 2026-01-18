# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GitLab API client for dynamo-utils."""

# Standard library imports
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports
import requests

# Local imports
from common import resolve_cache_path

# Module-level logger
_logger = logging.getLogger(__name__)

class GitLabAPIClient:
    """GitLab API client with automatic token detection and error handling.

    Features:
    - Automatic token detection (--token arg > GITLAB_TOKEN env > ~/.config/gitlab-token)
    - Request/response handling with proper error messages
    - Container registry queries

    Example:
        client = GitLabAPIClient()
        # Use get_cached_registry_images_for_shas for fetching Docker images
    """


    @staticmethod
    def get_gitlab_token_from_file() -> Optional[str]:
        """Get GitLab token from ~/.config/gitlab-token file.

        Returns:
            GitLab token string, or None if not found
        """
        try:
            token_file = Path.home() / '.config' / 'gitlab-token'
            if token_file.exists():
                return token_file.read_text().strip()
        except Exception:
            pass
        return None

    def __init__(self, token: Optional[str] = None, base_url: str = "https://gitlab-master.nvidia.com"):
        """Initialize GitLab API client.

        Args:
            token: GitLab personal access token. If not provided, will try:
                   1. GITLAB_TOKEN environment variable
                   2. ~/.config/gitlab-token file
            base_url: GitLab instance URL (default: https://gitlab-master.nvidia.com)
        """
        # Token priority: 1) provided token, 2) environment variable, 3) config file
        self.token = token or os.environ.get('GITLAB_TOKEN') or self.get_gitlab_token_from_file()
        self.base_url = base_url.rstrip('/')
        self.headers = {}

        if self.token:
            self.headers['PRIVATE-TOKEN'] = self.token

        # Best-effort per-run REST stats (mirrors GitHubAPIClient counters; useful for dashboards).
        self._rest_calls_total: int = 0
        self._rest_calls_by_endpoint: Dict[str, int] = {}
        self._rest_time_total_s: float = 0.0
        self._rest_time_by_endpoint_s: Dict[str, float] = {}
        self._rest_errors_by_status: Dict[int, int] = {}

    def has_token(self) -> bool:
        """Check if a GitLab token is configured."""
        return self.token is not None

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, timeout: int = 10) -> Optional[Any]:
        """Make GET request to GitLab API.

        Args:
            endpoint: API endpoint (e.g., "/api/v4/projects/169905")
            params: Query parameters
            timeout: Request timeout in seconds

        Returns:
            JSON response (dict or list), or None if request failed

            Example return value for registry tags endpoint:
            [
                {
                    "name": "21a03b316dc1e5031183965e5798b0d9fe2e64b3-38895507-vllm-amd64",
                    "path": "dl/ai-dynamo/dynamo:21a03b316dc1e5031183965e5798b0d9fe2e64b3-38895507-vllm-amd64",
                    "location": "gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo:21a03b316...",
                    "created_at": "2025-11-20T22:15:32.829+00:00"
                },
                {
                    "name": "5fe0476e605d2564234f00e8123461e1594a9ce7-38888909-sglang-arm64",
                    "path": "dl/ai-dynamo/dynamo:5fe0476e605d2564234f00e8123461e1594a9ce7-38888909-sglang-arm64",
                    "location": "gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo:5fe0476e6...",
                    "created_at": "2025-11-19T10:00:00.000+00:00"
                }
            ]

            Example return value for pipelines endpoint:
            [
                {
                    "id": 38895507,
                    "status": "success",
                    "web_url": "https://gitlab-master.nvidia.com/dl/ai-dynamo/dynamo/-/pipelines/38895507",
                    "ref": "main",
                    "sha": "21a03b316dc1e5031183965e5798b0d9fe2e64b3"
                }
            ]
        """
        ep = str(endpoint or "")
        t0 = time.monotonic()
        status_code: Optional[int] = None
        url = f"{self.base_url}{endpoint}" if endpoint.startswith('/') else f"{self.base_url}/{endpoint}"

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=timeout)
            try:
                status_code = int(response.status_code)
            except Exception:
                status_code = None

            if response.status_code == 401:
                raise Exception("GitLab API returned 401 Unauthorized. Check your token.")
            elif response.status_code == 403:
                raise Exception("GitLab API returned 403 Forbidden. Token may lack permissions.")
            elif response.status_code == 404:
                raise Exception(f"GitLab API returned 404 Not Found for {endpoint}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:  # type: ignore[union-attr]
            raise Exception(f"GitLab API request failed for {endpoint}: {e}")
        finally:
            dt = max(0.0, time.monotonic() - t0)
            try:
                self._rest_calls_total += 1
                self._rest_calls_by_endpoint[ep] = int(self._rest_calls_by_endpoint.get(ep, 0) or 0) + 1
                self._rest_time_total_s += float(dt)
                self._rest_time_by_endpoint_s[ep] = float(self._rest_time_by_endpoint_s.get(ep, 0.0) or 0.0) + float(dt)
                if status_code is not None and int(status_code) >= 400:
                    self._rest_errors_by_status[int(status_code)] = int(self._rest_errors_by_status.get(int(status_code), 0) or 0) + 1
            except Exception:
                pass

    def get_rest_call_stats(self) -> Dict[str, Any]:
        """Return best-effort REST call stats for the current process/run."""
        try:
            return {
                "total": int(self._rest_calls_total),
                "time_total_s": float(self._rest_time_total_s),
                "by_endpoint": dict(sorted(dict(self._rest_calls_by_endpoint or {}).items(), key=lambda kv: (-int(kv[1] or 0), kv[0]))),
                "time_by_endpoint_s": dict(sorted(dict(self._rest_time_by_endpoint_s or {}).items(), key=lambda kv: (-float(kv[1] or 0.0), kv[0]))),
                "errors_by_status": dict(sorted(dict(self._rest_errors_by_status or {}).items(), key=lambda kv: (-int(kv[1] or 0), int(kv[0] or 0)))),
            }
        except Exception:
            return {"total": 0, "time_total_s": 0.0, "by_endpoint": {}, "time_by_endpoint_s": {}, "errors_by_status": {}}

    def get_cached_registry_images_for_shas(self, project_id: str, registry_id: str,
                                           sha_list: List[str],
                                           sha_to_datetime: Optional[Dict[str, datetime]] = None,
                                           cache_file: str = '.gitlab_commit_sha_cache.json',
                                           skip_fetch: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """Get container registry images for commit SHAs with caching.

        Optimized caching logic:
        - If skip_fetch=True: Only return cached data, no API calls
        - If skip_fetch=False: Use binary search to find tags for recent commits (within 8 hours)
          - Only fetches pages needed for recent SHAs
          - Tracks visited pages to avoid redundant API calls
          - Only updates cache for recent SHAs found

        Args:
            project_id: GitLab project ID
            registry_id: Container registry ID
            sha_list: List of full commit SHAs (40 characters)
            sha_to_datetime: Optional dict mapping SHA to committed_datetime for time-based filtering
            cache_file: Path to cache file (default: .gitlab_commit_sha_cache.json)
            skip_fetch: If True, only return cached data without fetching from GitLab

        Returns:
            Dictionary mapping SHA to list of image info dicts

        Cache file format (.gitlab_commit_sha_cache.json):
            {
                "21a03b316dc1e5031183965e5798b0d9fe2e64b3": [
                    {
                        "tag": "21a03b316dc1e5031183965e5798b0d9fe2e64b3-38895507-vllm-amd64",
                        "framework": "vllm",
                        "arch": "amd64",
                        "pipeline_id": "38895507",
                        "location": "gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo:21a03b316...",
                        "total_size": 15000000000,
                        "created_at": "2024-11-20T13:00:00Z"
                    }
                ],
                "5fe0476e605d2564234f00e8123461e1594a9ce7": []
            }
        """

        # Load cache
        cache = {}
        cache_path = resolve_cache_path(cache_file)
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text())
            except Exception:
                pass

        # Initialize result for requested SHAs
        result = {}

        if skip_fetch:
            # Only return cached data - NO API calls
            for sha in sha_list:
                result[sha] = cache.get(sha, [])

            # Warn if no images found in cache
            if not any(result.values()):
                _logger.warning("⚠️  No Docker images found in cache. Consider running without --skip-gitlab-fetch to fetch fresh data.")

            return result
        else:
            # Identify recent SHAs (within stable window)
            now_utc = datetime.now(timezone.utc)
            eight_hours_ago_utc = now_utc - timedelta(hours=DEFAULT_STABLE_AFTER_HOURS)

            recent_shas = set()
            if sha_to_datetime:
                for sha in sha_list:
                    commit_time = sha_to_datetime.get(sha)
                    if commit_time:
                        # Normalize to UTC for comparison
                        if commit_time.tzinfo is None:
                            # Naive datetime, assume UTC
                            commit_time_utc = commit_time.replace(tzinfo=timezone.utc)
                        else:
                            commit_time_utc = commit_time.astimezone(timezone.utc)

                        if commit_time_utc >= eight_hours_ago_utc:
                            recent_shas.add(sha)

            _logger.debug(f"Found {len(recent_shas)} SHAs within 8 hours (out of {len(sha_list)} total)")

            if not recent_shas:
                # No recent SHAs, just return cached data
                for sha in sha_list:
                    result[sha] = cache.get(sha, [])
                return result

            # Fetch ALL pages first, then filter by SHA
            per_page = 100

            if not self.has_token():
                _logger.warning(
                    "No GitLab token found; cannot fetch Docker registry images. "
                    "Create a personal access token and set it via GITLAB_TOKEN or ~/.config/gitlab-token. "
                    "Token URL: https://gitlab-master.nvidia.com/-/profile/personal_access_tokens"
                )
                return {sha: [] for sha in sha_list}

            # Fetch page 1 first to get total pages from headers
            endpoint = f"/api/v4/projects/{project_id}/registry/repositories/{registry_id}/tags"
            params = {
                'per_page': per_page,
                'page': 1,
                'order_by': 'updated_at',
                'sort': 'desc'
            }

            try:
                # Make direct request to get headers
                url = f"{self.base_url}{endpoint}"
                response = requests.get(url, headers=self.headers, params=params, timeout=10)
                response.raise_for_status()
                first_page_tags = response.json()
                total_pages = int(response.headers.get('X-Total-Pages', '1'))

                if first_page_tags is None:
                    first_page_tags = []

                _logger.debug(f"Total pages available: {total_pages}")

            except Exception as e:
                _logger.warning(f"Failed to fetch page 1 to determine total pages: {e}")
                return {sha: [] for sha in sha_list}

            # Collect all tags from all pages
            all_tags = list(first_page_tags)  # Start with page 1 tags
            lock = threading.Lock()

            def fetch_page(page_num: int) -> List[Dict[str, Any]]:
                """Fetch a single page of tags."""
                endpoint = f"/api/v4/projects/{project_id}/registry/repositories/{registry_id}/tags"
                params = {
                    'per_page': per_page,
                    'page': page_num,
                    'order_by': 'updated_at',
                    'sort': 'desc'
                }

                try:
                    tags = self.get(endpoint, params=params)
                    if tags is None:
                        return []
                    return tags
                except Exception as e:
                    _logger.debug(f"Failed to fetch page {page_num}: {e}")
                    return []

            # Helper to check if tag is older than 8 hours
            def is_old_tag(tag: dict) -> bool:
                tag_created = tag.get('created_at', '')
                if not tag_created:
                    return False
                try:
                    tag_time = datetime.fromisoformat(tag_created.replace('Z', '+00:00'))
                    return tag_time < eight_hours_ago_utc
                except Exception:
                    return False

            # Check if first page has old tags
            found_old_tags = any(is_old_tag(tag) for tag in first_page_tags)
            pages_fetched = 1

            # Fetch remaining pages until we hit old tags
            if total_pages > 1 and not found_old_tags:
                _logger.debug(f"Fetching up to {total_pages} pages with early termination...")

                with ThreadPoolExecutor(max_workers=8) as executor:
                    # Submit all remaining pages
                    future_to_page = {executor.submit(fetch_page, page_num): page_num
                                     for page_num in range(2, total_pages + 1)}

                    # Process results as they complete
                    for future in as_completed(future_to_page):
                        page_num = future_to_page[future]
                        tags = future.result()
                        pages_fetched += 1

                        if tags:
                            all_tags.extend(tags)

                            # Stop if we found old tags
                            if any(is_old_tag(tag) for tag in tags):
                                found_old_tags = True
                                _logger.debug(f"Found tags older than 8 hours at page {page_num}, stopping early")
                                # Cancel remaining futures
                                for f in future_to_page:
                                    if not f.done():
                                        f.cancel()
                                break

                        if pages_fetched % 10 == 0:
                            _logger.debug(f"Fetched {pages_fetched}/{total_pages} pages...")

            _logger.debug(f"Fetched {pages_fetched} pages (stopped early: {found_old_tags}), total tags: {len(all_tags)}")

            # Now filter tags by SHA
            sha_to_images = {}
            recent_shas_set = set(recent_shas)

            for tag_info in all_tags:
                tag_name = tag_info.get('name', '')
                # Check if this tag matches any of our recent SHAs
                for sha in recent_shas_set:
                    if tag_name.startswith(sha + '-'):
                        if sha not in sha_to_images:
                            sha_to_images[sha] = []

                        parts = tag_name.split('-')
                        if len(parts) >= 4:
                            sha_to_images[sha].append({
                                'tag': tag_name,
                                'framework': parts[2],
                                'arch': parts[3],
                                'pipeline_id': parts[1],
                                'location': tag_info.get('location', ''),
                                'total_size': tag_info.get('total_size', 0),
                                'created_at': tag_info.get('created_at', '')
                            })

            found_count = len([sha for sha in recent_shas if sha in sha_to_images and sha_to_images[sha]])
            _logger.debug(f"Found tags for {found_count}/{len(recent_shas)} recent SHAs")

            # Update cache only for recent SHAs we found
            for sha, images in sha_to_images.items():
                cache[sha] = images

            # Build result for all requested SHAs (use cache for non-recent ones)
            for sha in sha_list:
                result[sha] = cache.get(sha, [])

            # Warn if no images found for recent SHAs
            if recent_shas and not any(result[sha] for sha in recent_shas):
                _logger.warning(f"⚠️  No Docker images found for any of the {len(recent_shas)} recent SHAs (within 8 hours)")
                _logger.warning("This might mean the commits haven't been built yet or the builds failed.")

            # Save updated cache with timestamp
            try:
                cache_with_metadata = {
                    '_metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'total_shas': len(cache),
                        'recent_shas_updated': len(sha_to_images)
                    }
                }
                cache_with_metadata.update(cache)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(cache_with_metadata, indent=2))
                _logger.debug(f"Updated cache with {len(sha_to_images)} recent SHAs")
            except Exception as e:
                _logger.warning(f"Failed to save cache: {e}")

        return result

    def get_cached_pipeline_status(self, sha_list: List[str],
                                  cache_file: str = '.gitlab_pipeline_status_cache.json',
                                  skip_fetch: bool = False) -> Dict[str, Optional[Dict[str, Any]]]:
        """Get GitLab CI pipeline status for commits with intelligent caching.

        Caching strategy:
        - If skip_fetch=True: Only return cached data, no API calls
        - If skip_fetch=False:
          - "success" status: Cached permanently (won't change)
          - "failed", "running", "pending", etc.: Always refetched (might be re-run)
          - None/missing: Always fetched

        Args:
            sha_list: List of full commit SHAs (40 characters)
            cache_file: Path to cache file (default: .gitlab_pipeline_status_cache.json)
            skip_fetch: If True, only return cached data without fetching from GitLab

        Returns:
            Dictionary mapping SHA to pipeline status dict (or None if no pipeline found)

            Example return value:
            {
                "21a03b316dc1e5031183965e5798b0d9fe2e64b3": {
                    "status": "success",
                    "id": 38895507,
                    "web_url": "https://gitlab-master.nvidia.com/dl/ai-dynamo/dynamo/-/pipelines/38895507"
                },
                "5fe0476e605d2564234f00e8123461e1594a9ce7": None
            }

        Cache file format (.gitlab_pipeline_status_cache.json) - internally used:
        {
            "21a03b316dc1e5031183965e5798b0d9fe2e64b3": {
                "status": "success",
                "id": 38895507,
                "web_url": "https://gitlab-master.nvidia.com/dl/ai-dynamo/dynamo/-/pipelines/38895507"
            },
            "5fe0476e605d2564234f00e8123461e1594a9ce7": {
                "status": "failed",
                "id": 38888909,
                "web_url": "https://gitlab-master.nvidia.com/dl/ai-dynamo/dynamo/-/pipelines/38888909"
            }
        }
        """

        # Load cache
        cache = {}
        pipeline_cache_path = resolve_cache_path(cache_file)
        if pipeline_cache_path.exists():
            try:
                cache = json.loads(pipeline_cache_path.read_text())
            except Exception:
                pass

        # If skip_fetch=True, only return cached data - NO API calls
        if skip_fetch:
            result = {}
            for sha in sha_list:
                result[sha] = cache.get(sha)
            return result

        # Check which SHAs need to be fetched
        # Only cache "success" status permanently; refetch others as they might change
        shas_to_fetch = []
        result = {}

        for sha in sha_list:
            if sha in cache:
                cached_info = cache[sha]
                # If pipeline succeeded, use cached value
                # If pipeline failed/running/pending, refetch as it might have been re-run
                if cached_info and cached_info.get('status') == 'success':
                    result[sha] = cached_info
                else:
                    # Non-success status or None - refetch to check for updates
                    shas_to_fetch.append(sha)
                    result[sha] = cached_info  # Use cached value temporarily
            else:
                shas_to_fetch.append(sha)
                result[sha] = None

        # Fetch missing SHAs and non-success statuses from GitLab in parallel
        if shas_to_fetch and self.has_token():

            def fetch_pipeline_status(sha):
                """Helper function to fetch pipeline status for a single SHA"""
                try:
                    # Get pipelines for this commit
                    endpoint = f"/api/v4/projects/169905/pipelines"
                    params = {'sha': sha, 'per_page': 1}
                    pipelines = self.get(endpoint, params=params)

                    if pipelines and len(pipelines) > 0:
                        pipeline = pipelines[0]  # Most recent pipeline
                        status_info = {
                            'status': pipeline.get('status', 'unknown'),
                            'id': pipeline.get('id'),
                            'web_url': pipeline.get('web_url', ''),
                        }
                        return (sha, status_info)
                    else:
                        return (sha, None)
                except Exception:
                    return (sha, None)

            # Fetch in parallel with 10 workers
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(fetch_pipeline_status, sha) for sha in shas_to_fetch]

                # Collect results as they complete
                for future in futures:
                    try:
                        sha, status_info = future.result()
                        result[sha] = status_info
                        cache[sha] = status_info
                    except Exception:
                        pass

            # Save updated cache
            try:
                pipeline_cache_path.parent.mkdir(parents=True, exist_ok=True)
                pipeline_cache_path.write_text(json.dumps(cache, indent=2))
            except Exception:
                pass

        return result

    def get_cached_pipeline_job_counts(self, pipeline_ids: List[int],
                                      cache_file: str = '.gitlab_pipeline_jobs_cache.json',
                                      skip_fetch: bool = False) -> Dict[int, Optional[Dict[str, int]]]:
        """Get GitLab CI pipeline job counts with intelligent caching.

        Caching strategy:
        - If skip_fetch=True: Only return cached data, no API calls
        - If skip_fetch=False:
          - Completed pipelines (running=0, pending=0): Cached forever, never refetched
          - Active pipelines (running>0 or pending>0): Refetch if older than 30 minutes

        Args:
            pipeline_ids: List of pipeline IDs
            cache_file: Path to cache file (default: .gitlab_pipeline_jobs_cache.json)
            skip_fetch: If True, only return cached data without fetching from GitLab

        Returns:
            Dictionary mapping pipeline ID to job counts dict (or None if fetch failed)

            Example return value:
            {
                40355198: {
                    "success": 16,
                    "failed": 0,
                    "running": 6,
                    "pending": 0
                },
                40341238: {
                    "success": 11,
                    "failed": 13,
                    "running": 0,
                    "pending": 0
                }
            }

        Cache file format (with timestamps):
            {
                "40355198": {
                    "counts": {"success": 16, "failed": 0, "running": 6, "pending": 0},
                    "fetched_at": "2025-12-17T18:15:00Z"
                }
            }
        """
        # Load cache
        cache = {}
        jobs_cache_path = resolve_cache_path(cache_file)
        if jobs_cache_path.exists():
            try:
                cache = json.loads(jobs_cache_path.read_text())
                # Convert string keys back to int
                cache = {int(k): v for k, v in cache.items()}
            except Exception:
                pass

        # Helper function to extract counts from cache entry (handles old and new format)
        def extract_counts(entry):
            if not entry:
                return None
            return entry.get('counts', entry) if isinstance(entry, dict) else entry

        # Helper function to check if pipeline is completed (no running/pending jobs)
        def is_completed(entry):
            counts = extract_counts(entry)
            if not counts:
                return False
            return counts.get('running', 0) == 0 and counts.get('pending', 0) == 0

        # Helper function to check if cache entry is fresh
        # Completed pipelines (no running/pending) are cached forever
        # Active pipelines (running/pending) must be < 30 minutes old
        def is_fresh(entry, now, age_limit):
            if not isinstance(entry, dict) or 'fetched_at' not in entry:
                return False  # Old format or missing timestamp = stale

            # If pipeline is completed, cache forever
            if is_completed(entry):
                return True

            # Otherwise, check if < 30 minutes old
            try:
                fetched_at = datetime.fromisoformat(entry['fetched_at'].replace('Z', '+00:00'))
                return (now - fetched_at) < age_limit
            except Exception:
                return False  # Invalid timestamp = stale

        now = datetime.now(timezone.utc)
        cache_age_limit = timedelta(minutes=30)

        # If skip_fetch=True, only return cached data - NO API calls
        if skip_fetch:
            return {pid: extract_counts(cache.get(pid)) for pid in pipeline_ids}

        # Determine which pipelines need fetching
        pipeline_ids_to_fetch = []
        result = {}

        for pipeline_id in pipeline_ids:
            cached_entry = cache.get(pipeline_id)

            if cached_entry and is_fresh(cached_entry, now, cache_age_limit):
                # Cache is fresh, use it
                result[pipeline_id] = extract_counts(cached_entry)
            else:
                # Not in cache, or cache is stale - refetch
                pipeline_ids_to_fetch.append(pipeline_id)
                result[pipeline_id] = extract_counts(cached_entry)  # Use existing data temporarily if available

        # Fetch missing pipeline job counts from GitLab in parallel
        if pipeline_ids_to_fetch and self.has_token():
            fetch_timestamp = now.isoformat().replace('+00:00', 'Z')

            def fetch_pipeline_jobs(pipeline_id):
                """Helper function to fetch job counts for a single pipeline"""
                try:
                    # Get jobs for this pipeline
                    endpoint = f"/api/v4/projects/169905/pipelines/{pipeline_id}/jobs"
                    params = {'per_page': 100}  # Get up to 100 jobs
                    jobs = self.get(endpoint, params=params)

                    if jobs:
                        # Count jobs by status
                        counts = {
                            'success': 0,
                            'failed': 0,
                            'running': 0,
                            'pending': 0
                        }

                        for job in jobs:
                            status = job.get('status', 'unknown')
                            if status in counts:
                                counts[status] += 1
                            # Map other statuses to main categories
                            elif status in ('skipped', 'manual', 'canceled'):
                                pass  # Don't count these
                            elif status in ('created', 'waiting_for_resource'):
                                counts['pending'] += 1

                        # Return with timestamp
                        return (pipeline_id, counts, fetch_timestamp)
                    else:
                        return (pipeline_id, None, None)
                except Exception:
                    return (pipeline_id, None, None)

            # Fetch in parallel with 10 workers
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(fetch_pipeline_jobs, pid) for pid in pipeline_ids_to_fetch]

                # Collect results as they complete
                for future in futures:
                    try:
                        pipeline_id, counts, timestamp = future.result()
                        result[pipeline_id] = counts
                        if counts is not None and timestamp is not None:
                            cache[pipeline_id] = {
                                'counts': counts,
                                'fetched_at': timestamp
                            }
                        else:
                            cache[pipeline_id] = None
                    except Exception:
                        pass

            # Save updated cache
            try:
                jobs_cache_path.parent.mkdir(parents=True, exist_ok=True)
                # Convert int keys to string for JSON
                cache_str_keys = {str(k): v for k, v in cache.items()}
                jobs_cache_path.write_text(json.dumps(cache_str_keys, indent=2))
            except Exception:
                pass

        return result

    def get_cached_pipeline_job_details(
        self,
        pipeline_ids: List[int],
        cache_file: str = ".gitlab_pipeline_jobs_details_cache.json",
        skip_fetch: bool = False,
    ) -> Dict[int, Optional[Dict[str, Any]]]:
        """Get GitLab CI pipeline job details (counts + job list) with intelligent caching.

        This is a richer variant of `get_cached_pipeline_job_counts` that also returns a
        slim per-job list for UI tooltips.

        Caching strategy:
        - If skip_fetch=True: Only return cached data, no API calls
        - If skip_fetch=False:
          - Completed pipelines (running=0, pending=0): Cached forever, never refetched
          - Active pipelines (running>0 or pending>0): Refetch if older than 30 minutes

        Args:
            pipeline_ids: List of pipeline IDs
            cache_file: Path to cache file
            skip_fetch: If True, only return cached data without fetching from GitLab

        Returns:
            Dictionary mapping pipeline ID -> details dict (or None if unavailable)

            Example return value:
            {
                40118215: {
                    "counts": {"success": 15, "failed": 8, "running": 0, "pending": 0},
                    "jobs": [
                        {"stage": "build", "name": "build-dynamo-image-amd64", "status": "success"},
                        {"stage": "test", "name": "pre-merge-vllm", "status": "failed"},
                    ],
                    "fetched_at": "2025-12-18T02:44:20.118368Z"
                },
                40118216: None
            }
        """
        # Load cache
        cache: Dict[int, Any] = {}
        jobs_cache_path = resolve_cache_path(cache_file)
        if jobs_cache_path.exists():
            try:
                raw = json.loads(jobs_cache_path.read_text())
                # Convert string keys back to int
                cache = {int(k): v for k, v in raw.items()}
            except Exception:
                cache = {}

        def normalize_entry(entry: Any) -> Optional[Dict[str, Any]]:
            """Normalize cache entry to {counts, jobs, fetched_at} (best-effort)."""
            if not entry:
                return None
            if not isinstance(entry, dict):
                return None
            # New format
            if "counts" in entry or "jobs" in entry or "fetched_at" in entry:
                counts = entry.get("counts") if isinstance(entry.get("counts"), dict) else None
                jobs = entry.get("jobs") if isinstance(entry.get("jobs"), list) else []
                fetched_at = entry.get("fetched_at")
                # Ensure expected keys exist even if older cache entries are missing fields.
                base_counts = {"success": 0, "failed": 0, "running": 0, "pending": 0, "canceled": 0}
                if counts:
                    for k, v in base_counts.items():
                        counts.setdefault(k, v)
                return {
                    "counts": counts or base_counts,
                    "jobs": jobs,
                    "fetched_at": fetched_at,
                }
            # Old format fallback: counts-only dict
            if all(k in entry for k in ("success", "failed", "running", "pending")):
                entry.setdefault("canceled", 0)
                return {"counts": entry, "jobs": [], "fetched_at": None}
            return None

        def is_completed(entry_norm: Optional[Dict[str, Any]]) -> bool:
            if not entry_norm:
                return False
            counts = entry_norm.get("counts") or {}
            return counts.get("running", 0) == 0 and counts.get("pending", 0) == 0

        def is_fresh(entry_norm: Optional[Dict[str, Any]]) -> bool:
            if not entry_norm:
                return False
            # Completed pipelines are cached forever.
            #
            # Active pipelines (running/pending > 0) should be refetched aggressively to avoid
            # showing stale status. We intentionally treat them as *not fresh* regardless of age.
            if is_completed(entry_norm):
                return True
                return False

        # NOTE: We intentionally refetch active pipelines on every run, so we no longer
        # use a time-based "freshness" threshold here.

        # skip_fetch => return cached values only
        if skip_fetch:
            return {pid: normalize_entry(cache.get(pid)) for pid in pipeline_ids}

        pipeline_ids_to_fetch: List[int] = []
        result: Dict[int, Optional[Dict[str, Any]]] = {}

        for pipeline_id in pipeline_ids:
            cached_entry_norm = normalize_entry(cache.get(pipeline_id))
            if cached_entry_norm and is_fresh(cached_entry_norm):
                result[pipeline_id] = cached_entry_norm
            else:
                pipeline_ids_to_fetch.append(pipeline_id)
                result[pipeline_id] = cached_entry_norm

        if pipeline_ids_to_fetch and self.has_token():
            fetch_timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

            def fetch_pipeline_jobs_details(pipeline_id: int) -> Tuple[int, Optional[Dict[str, Any]]]:
                try:
                    endpoint = f"/api/v4/projects/169905/pipelines/{pipeline_id}/jobs"
                    params = {"per_page": 100}
                    jobs = self.get(endpoint, params=params)
                    if not jobs:
                        return pipeline_id, None

                    counts = {"success": 0, "failed": 0, "running": 0, "pending": 0, "canceled": 0}
                    slim_jobs: List[Dict[str, Any]] = []

                    for job in jobs:
                        status = job.get("status", "unknown")
                        name = job.get("name", "")
                        stage = job.get("stage", "")

                        # Counts (map GitLab statuses to our buckets)
                        if status in counts:
                            counts[status] += 1
                        elif status in ("created", "waiting_for_resource"):
                            counts["pending"] += 1
                        elif status in ("skipped", "manual"):
                            pass
                        else:
                            # Keep unknown statuses out of counts
                            pass

                        # Tooltip list (keep it light)
                        if name:
                            slim_jobs.append({"name": name, "stage": stage, "status": status})

                    details = {"counts": counts, "jobs": slim_jobs, "fetched_at": fetch_timestamp}
                    return pipeline_id, details
                except Exception:
                    return pipeline_id, None

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(fetch_pipeline_jobs_details, pid) for pid in pipeline_ids_to_fetch]
                for future in futures:
                    try:
                        pid, details = future.result()
                        result[pid] = details
                        cache[pid] = details
                    except Exception:
                        pass

            # Save updated cache
            try:
                jobs_cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_str_keys = {str(k): v for k, v in cache.items()}
                jobs_cache_path.write_text(json.dumps(cache_str_keys, indent=2))
            except Exception:
                pass

        return result

    def get_cached_merge_request_pipelines(
        self,
        mr_numbers: List[int],
        project_id: str = "169905",
        cache_file: str = ".gitlab_mr_pipelines_cache.json",
        skip_fetch: bool = False,
    ) -> Dict[int, Optional[Dict[str, Any]]]:
        """Get most recent pipeline for each Merge Request (MR IID) with caching.

        This helps link a PR/MR to a pipeline even when the final merge commit SHA
        doesn't have a pipeline (e.g. pipeline is created for merge_request_event only).

        Args:
            mr_numbers: List of MR IIDs (internal IDs)
            project_id: GitLab project ID (default: 169905 for dl/ai-dynamo/dynamo)
            cache_file: Cache file name under the dynamo-utils cache dir
            skip_fetch: If True, only return cached data (no API calls)

        Returns:
            Mapping MR IID -> pipeline dict (id, status, web_url, sha, ref), or None.

        Cache format:
            {
              "5063": {"id": 40743226, "status": "success", "web_url": "...", "sha": "...", "ref": "..."},
              "5064": null
            }
        """
        # Load cache
        cache: Dict[int, Optional[Dict[str, Any]]] = {}
        cache_path = resolve_cache_path(cache_file)
        if cache_path.exists():
            try:
                raw = json.loads(cache_path.read_text())
                cache = {int(k): v for k, v in raw.items()}
            except Exception:
                cache = {}

        if skip_fetch:
            return {mr: cache.get(mr) for mr in mr_numbers}

        result: Dict[int, Optional[Dict[str, Any]]] = {}
        cache_updated = False

        # Determine which MRs to fetch (only if missing from cache or cached None).
        to_fetch = [mr for mr in mr_numbers if mr not in cache]
        for mr in mr_numbers:
            if mr in cache:
                result[mr] = cache[mr]
            else:
                result[mr] = None

        if to_fetch and self.has_token():
            logger = logging.getLogger("common")

            def fetch_one(mr_iid: int) -> Tuple[int, Optional[Dict[str, Any]]]:
                try:
                    endpoint = f"/api/v4/projects/{project_id}/merge_requests/{mr_iid}/pipelines"
                    pipelines = self.get(endpoint, params={"per_page": 1, "order_by": "id", "sort": "desc"}, timeout=10)
                    if isinstance(pipelines, list) and pipelines:
                        p = pipelines[0]
                        if isinstance(p, dict):
                            return mr_iid, {
                                "id": p.get("id"),
                                "status": p.get("status", "unknown"),
                                "web_url": p.get("web_url", ""),
                                "sha": p.get("sha", ""),
                                "ref": p.get("ref", ""),
                            }
                    return mr_iid, None
                except Exception as e:
                    logger.debug(f"Failed to fetch MR {mr_iid} pipelines: {e}")
                    return mr_iid, None

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(fetch_one, mr) for mr in to_fetch]
                for fut in futures:
                    try:
                        mr_iid, p = fut.result()
                        result[mr_iid] = p
                        cache[mr_iid] = p
                        cache_updated = True
                    except Exception:
                        pass

        # Save updated cache
        if cache_updated:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_str = {str(k): v for k, v in cache.items()}
                cache_path.write_text(json.dumps(cache_str, indent=2))
            except Exception:
                pass

        return result

    @staticmethod
    def parse_mr_number_from_message(message: str) -> Optional[int]:
        """Parse MR/PR number from commit message (e.g., '... (#1234)')

        Args:
            message: Commit message to parse

        Returns:
            MR number if found, None otherwise

        Example:
            >>> GitLabAPIClient.parse_mr_number_from_message("feat: Add feature (#1234)")
            1234
            >>> GitLabAPIClient.parse_mr_number_from_message("fix: Bug fix")
            None
        """
        match = re.search(r'#(\d+)', message)
        if match:
            return int(match.group(1))
        return None

    def get_cached_mr_merge_dates(self, mr_numbers: List[int],
                                  project_id: str = "169905",
                                  cache_file: str = '.gitlab_mr_merge_dates_cache.json',
                                  skip_fetch: bool = False) -> Dict[int, Optional[str]]:
        """Get merge dates for merge requests with caching.

        Merge dates are cached permanently since they don't change once a MR is merged.

        Args:
            mr_numbers: List of MR IIDs (internal IDs)
            project_id: GitLab project ID (default: 169905 for dynamo)
            cache_file: Path to cache file
            skip_fetch: If True, only return cached data without fetching from GitLab

        Returns:
            Dictionary mapping MR number to merge date string (YYYY-MM-DD HH:MM:SS)
            Returns None for MRs that are not merged or not found

        Example:
            >>> client = GitLabAPIClient()
            >>> merge_dates = client.get_cached_mr_merge_dates([4965, 5009])
            >>> merge_dates
            {4965: "2025-12-18 12:34:56", 5009: None}

        Cache file format (.gitlab_mr_merge_dates_cache.json):
        {
            "4965": "2025-12-18 12:34:56",
            "5009": null
        }
        """

        # Load cache
        cache = {}
        mr_cache_path = resolve_cache_path(cache_file)
        if mr_cache_path.exists():
            try:
                # Keys are stored as strings in JSON, convert back to int
                cache_raw = json.loads(mr_cache_path.read_text())
                cache = {int(k): v for k, v in cache_raw.items()}
            except Exception:
                pass

        # If skip_fetch=True, only return cached data
        if skip_fetch:
            return {mr_num: cache.get(mr_num) for mr_num in mr_numbers}

        # Prepare result and track if cache was updated
        result = {}
        cache_updated = False

        for mr_num in mr_numbers:
            # Check cache first
            if mr_num in cache:
                result[mr_num] = cache[mr_num]
                continue

            # Fetch from GitLab API
            try:
                endpoint = f"/api/v4/projects/{project_id}/merge_requests/{mr_num}"
                response = self.get(endpoint, timeout=5)

                if response and response.get('merged_at'):
                    # Parse ISO timestamp: "2025-12-18T12:34:56.000Z"
                    merged_at = response['merged_at']
                    dt = datetime.fromisoformat(merged_at.replace('Z', '+00:00'))
                    merge_date = dt.strftime('%Y-%m-%d %H:%M:%S')
                    result[mr_num] = merge_date
                    cache[mr_num] = merge_date
                    cache_updated = True
                else:
                    # MR not merged or not found
                    result[mr_num] = None
                    cache[mr_num] = None
                    cache_updated = True
            except Exception as e:
                # Log error but continue with other MRs
                logger = logging.getLogger('common')
                logger.debug(f"Failed to fetch MR {mr_num} merge date: {e}")
                result[mr_num] = None
                cache[mr_num] = None
                cache_updated = True

        # Save updated cache
        if cache_updated:
            try:
                mr_cache_path.parent.mkdir(parents=True, exist_ok=True)
                # Convert int keys to string for JSON
                cache_str_keys = {str(k): v for k, v in cache.items()}
                mr_cache_path.write_text(json.dumps(cache_str_keys, indent=2))
            except Exception:
                pass

        return result


