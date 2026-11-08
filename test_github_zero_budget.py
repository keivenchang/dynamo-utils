"""A cache miss must not make a zero-budget dashboard attempt network access."""

import time

import pytest

from common_github import GitHubAPIClient
from common_github.api import commit_checks_cached, pr_checks_cached
from common_github.pr_checks_types import GHPRCheckRow, GHPRChecksCacheEntry


@pytest.mark.parametrize("budget", [0, -1])
def test_pr_checks_cache_miss_with_zero_budget(monkeypatch, tmp_path, budget):
    monkeypatch.setenv("DYNAMO_UTILS_CACHE_DIR", str(tmp_path))
    client = GitHubAPIClient(token="test-token", max_rest_calls=budget)

    def reject_network(*args, **kwargs):
        pytest.fail("A zero-budget cache miss attempted a network request")

    monkeypatch.setattr(client, "_rest_get", reject_network)
    assert client.get_pr_checks_rows(
        owner="zero-budget-test",
        repo="empty-cache",
        pr_number=99999991,
    ) == []


@pytest.mark.parametrize("resource", ["pr", "commit"])
@pytest.mark.parametrize("etag", ["", 'W/"cached-checks"'])
@pytest.mark.parametrize("age", [1, 3600])
def test_zero_budget_preserves_complete_cached_checks(monkeypatch, tmp_path, resource, etag, age):
    monkeypatch.setenv("DYNAMO_UTILS_CACHE_DIR", str(tmp_path))
    client = GitHubAPIClient(token="test-token", max_rest_calls=0)
    module = pr_checks_cached if resource == "pr" else commit_checks_cached
    row = GHPRCheckRow(name="Rust tests", status_raw="failure")
    entry = GHPRChecksCacheEntry(
        ts=int(time.time()) - age,
        ver=9 if resource == "pr" else 1,
        rows=(row,),
        check_runs_etag=etag,
    )
    monkeypatch.setattr(module._CACHE, "get_entry_dict", lambda key: entry.to_disk_dict())

    def reject_network(*args, **kwargs):
        pytest.fail("Cached checks attempted a network request with a zero budget")

    monkeypatch.setattr(client, "_rest_get", reject_network)
    if resource == "pr":
        rows = client.get_pr_checks_rows(owner="test", repo="repo", pr_number=123, ttl_s=300)
    else:
        rows = client.get_commit_checks_rows(owner="test", repo="repo", commit_sha="abc123", ttl_s=300)
    assert rows == [row]


@pytest.mark.parametrize("resource", ["pr", "commit"])
@pytest.mark.parametrize("invalid", ["old-version", "incomplete"])
def test_zero_budget_rejects_unusable_cached_checks(monkeypatch, tmp_path, resource, invalid):
    monkeypatch.setenv("DYNAMO_UTILS_CACHE_DIR", str(tmp_path))
    client = GitHubAPIClient(token="test-token", max_rest_calls=0)
    module = pr_checks_cached if resource == "pr" else commit_checks_cached
    version = 9 if resource == "pr" else 1
    entry = GHPRChecksCacheEntry(
        ts=int(time.time()),
        ver=version - 1 if invalid == "old-version" else version,
        rows=(GHPRCheckRow(name="Rust tests", status_raw="success"),),
        check_runs_etag='W/"cached-checks"',
        incomplete=invalid == "incomplete",
    )
    monkeypatch.setattr(module._CACHE, "get_entry_dict", lambda key: entry.to_disk_dict())

    def reject_network(*args, **kwargs):
        pytest.fail("An unusable cache entry attempted a zero-budget network request")

    monkeypatch.setattr(client, "_rest_get", reject_network)
    if resource == "pr":
        rows = client.get_pr_checks_rows(owner="test", repo="repo", pr_number=123)
    else:
        rows = client.get_commit_checks_rows(owner="test", repo="repo", commit_sha="abc123")
    assert rows == []


@pytest.mark.parametrize("resource", ["pr", "commit"])
def test_online_checks_still_revalidate_etags(monkeypatch, tmp_path, resource):
    monkeypatch.setenv("DYNAMO_UTILS_CACHE_DIR", str(tmp_path))
    client = GitHubAPIClient(token="test-token", max_rest_calls=10)
    module = pr_checks_cached if resource == "pr" else commit_checks_cached
    entry = GHPRChecksCacheEntry(
        ts=int(time.time()),
        ver=9 if resource == "pr" else 1,
        rows=(GHPRCheckRow(name="Rust tests", status_raw="success"),),
        check_runs_etag='W/"cached-checks"',
    )
    monkeypatch.setattr(module._CACHE, "get_entry_dict", lambda key: entry.to_disk_dict())

    def observe_network(*args, **kwargs):
        assert kwargs["etag"] == entry.check_runs_etag
        raise RuntimeError("Observed online ETag revalidation")

    monkeypatch.setattr(client, "_rest_get", observe_network)
    with pytest.raises(RuntimeError, match="Observed online ETag revalidation"):
        if resource == "pr":
            client.get_pr_checks_rows(owner="test", repo="repo", pr_number=123, head_sha="abc123")
        else:
            client.get_commit_checks_rows(owner="test", repo="repo", commit_sha="abc123")
