"""
Dashboard runtime helpers (file/materialization/pruning) for html_pages/* scripts.

These helpers are intentionally kept out of `dynamo-utils/common.py` because they are
only used by the HTML dashboard generators in `dynamo-utils/html_pages/*`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow importing `common.py` from the parent dynamo-utils directory when executed as a script tool.
_THIS_DIR = Path(__file__).resolve().parent
_UTILS_DIR = _THIS_DIR.parent
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

import common


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Atomically write text to `path` by writing a temp file in the same directory and os.replace().

    This prevents partially-written HTML from being observed by readers (e.g. nginx) during refresh.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp.{os.getpid()}")
    # Best-effort cleanup of stale tmp from prior crashes.
    try:
        if tmp.exists():
            tmp.unlink()
    except Exception:
        pass
    tmp.write_text(content, encoding=encoding)
    os.replace(str(tmp), str(p))


def dashboard_served_raw_log_repo_cache_dir(*, page_root_dir: Path) -> Path:
    """Return the directory used to *serve* raw logs for dashboards.

    Critical policy:
    - All raw log *storage* must live under the single global cache dir:
        `~/.cache/dynamo-utils/raw-log-text`
      (or `$DYNAMO_UTILS_CACHE_DIR/raw-log-text`).
    - Dashboards must link to logs via a URL path that is under the page root:
        `raw-log-text/<job_id>.log`
      (never `/.cache/...`).

    Implementation:
    - We expose `raw-log-text/` under each dashboard root as a *symlink* pointing at the global cache.
      That yields stable URLs while keeping storage in exactly one place.
    """
    try:
        page_root_dir = Path(page_root_dir)
    except Exception:
        page_root_dir = Path(".")

    served = page_root_dir / "raw-log-text"

    # If `served` already exists as a real directory, keep using it (don't delete/replace user data).
    try:
        if served.exists() and not served.is_symlink():
            return served
    except Exception:
        return served

    target = common.dynamo_utils_cache_dir() / "raw-log-text"
    try:
        target.mkdir(parents=True, exist_ok=True)
    except Exception:
        return served

    try:
        served.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        return served

    # Create/refresh symlink if possible.
    try:
        if served.is_symlink():
            try:
                if served.resolve() == target.resolve():
                    return served
            except Exception:
                pass
            try:
                served.unlink()
            except Exception:
                return served

        # Use a relative symlink so the tree is portable.
        rel = os.path.relpath(str(target), start=str(served.parent))
        os.symlink(rel, str(served))
    except Exception:
        # Fall back to per-page directory if symlink creation fails.
        return served

    return served


def prune_dashboard_raw_logs(*, page_root_dir: Path, max_age_days: int = 30) -> int:
    """Prune dashboard-served raw logs under the consolidated repo-local cache.

    Safety: only delete files that look like GitHub Actions job logs:
      - filename must be digits only + ".log" (e.g., "58906138553.log")
    """
    try:
        root = dashboard_served_raw_log_repo_cache_dir(page_root_dir=page_root_dir)

        deleted = 0
        now = time.time()
        cutoff_s = float(max(0, int(max_age_days))) * 24.0 * 3600.0

        try:
            if not root.exists() or not root.is_dir():
                return 0
        except Exception:
            return 0

        for p in root.glob("*.log"):
            try:
                if not p.is_file():
                    continue
                if not re.fullmatch(r"[0-9]+\.log", p.name or ""):
                    continue
                age_s = now - float(p.stat().st_mtime)
                if cutoff_s and age_s > cutoff_s:
                    p.unlink()
                    deleted += 1
            except Exception:
                continue

        return deleted
    except Exception:
        return 0


def prune_partial_raw_log_caches(*, page_root_dirs: List[Path]) -> Dict[str, int]:
    """Remove raw-log cache artifacts that could be partial/unverified.

    Policy:
    - Global cache: delete all legacy `raw-log-text/*.txt` (they predate completed-only metadata).
    - Global cache: delete any `raw-log-text/*.log` whose index entry is missing `completed=true`.
    - Dashboard-served logs: delete any `<page_root_dir>/raw-log-text/*.log` whose job_id
      is not present in the global index with `completed=true`.
    """
    stats: Dict[str, int] = {
        "deleted_global_txt": 0,
        "deleted_global_log_unverified": 0,
        "deleted_page_logs_unverified": 0,
    }
    try:
        cache_dir = common.dynamo_utils_cache_dir() / "raw-log-text"
        index_file = cache_dir / "index.json"
        meta: Dict[str, Any] = {}
        if index_file.exists():
            try:
                meta = json.loads(index_file.read_text() or "{}")
            except Exception:
                meta = {}
        if not isinstance(meta, dict):
            meta = {}

        # 1) Global legacy .txt: always delete (cannot be verified completed=true)
        try:
            for p in cache_dir.glob("*.txt"):
                try:
                    if p.is_file() and re.fullmatch(r"[0-9]+\.txt", p.name or ""):
                        p.unlink()
                        stats["deleted_global_txt"] += 1
                except Exception:
                    continue
        except Exception:
            pass

        # 2) Global .log without completed=true in index: delete
        try:
            for p in cache_dir.glob("*.log"):
                try:
                    if not p.is_file():
                        continue
                    if not re.fullmatch(r"[0-9]+\.log", p.name or ""):
                        continue
                    job_id = p.stem
                    ent = meta.get(job_id) if isinstance(meta, dict) else None
                    if not (isinstance(ent, dict) and bool(ent.get("completed", False))):
                        p.unlink()
                        stats["deleted_global_log_unverified"] += 1
                except Exception:
                    continue
        except Exception:
            pass

        # Build allowlist: completed job_ids
        completed_job_ids: set[str] = set()
        try:
            for k, v in meta.items():
                if isinstance(k, str) and isinstance(v, dict) and bool(v.get("completed", False)):
                    completed_job_ids.add(k)
        except Exception:
            completed_job_ids = set()

        # 3) Dashboard-served logs under each page root: delete unverified
        for base in page_root_dirs or []:
            try:
                d = dashboard_served_raw_log_repo_cache_dir(page_root_dir=Path(base))
                if not d.exists() or not d.is_dir():
                    continue
                for p in d.glob("*.log"):
                    try:
                        if not p.is_file():
                            continue
                        if not re.fullmatch(r"[0-9]+\.log", p.name or ""):
                            continue
                        job_id = p.stem
                        if job_id not in completed_job_ids:
                            p.unlink()
                            stats["deleted_page_logs_unverified"] += 1
                    except Exception:
                        continue
            except Exception:
                continue

        return stats
    except Exception:
        return stats


def materialize_job_raw_log_text_local_link(
    github: "common.GitHubAPIClient",
    *,
    job_url: str,
    job_name: Optional[str] = None,
    owner: str,
    repo: str,
    page_root_dir: Path,
    allow_fetch: bool = True,
    ttl_s: int = common.DEFAULT_RAW_LOG_TEXT_TTL_S,
    assume_completed: bool = False,
) -> Optional[str]:
    """Ensure a stable repo-local raw log file exists and return a relative link.

    The dashboard never links to ephemeral GitHub signed URLs.

    Returned href points into the page-root served cache:
      "raw-log-text/<job_id>.log"   (from <page_root_dir>/index.html)
    """
    def _extract_job_id(u: str) -> str:
        try:
            if "/job/" not in (u or ""):
                return ""
            return str(u.split("/job/")[1].split("?")[0] or "").strip()
        except Exception:
            return ""

    def _extract_run_id(u: str) -> str:
        try:
            s = str(u or "")
            if "/actions/runs/" not in s:
                return ""
            rest = s.split("/actions/runs/", 1)[1]
            run_id = rest.split("/", 1)[0].split("?", 1)[0].strip()
            return run_id if run_id.isdigit() else ""
        except Exception:
            return ""

    def _norm_name(s: str) -> str:
        try:
            x = (s or "").strip().lower()
            x = x.replace("—", "-").replace("–", "-")
            x = re.sub(r"[^a-z0-9]+", " ", x)
            x = re.sub(r"\\s+", " ", x).strip()
            return x
        except Exception:
            return ""

    job_url = str(job_url or "")
    job_id = _extract_job_id(job_url)
    run_id = _extract_run_id(job_url)

    # If we only have a run URL, try to resolve job_id using the Actions "list jobs for a workflow run" API.
    if not job_id and run_id and job_name and allow_fetch:
        try:
            data = github.get(
                f"/repos/{owner}/{repo}/actions/runs/{run_id}/jobs",
                params={"per_page": 100},
                timeout=15,
            )
            jobs = data.get("jobs") if isinstance(data, dict) else None
            want = _norm_name(str(job_name or ""))
            best = ""
            if isinstance(jobs, list) and want:
                for j in jobs:
                    try:
                        jid = str(j.get("id", "") or "").strip()
                        nm = _norm_name(str(j.get("name", "") or ""))
                        if not jid or not jid.isdigit() or not nm:
                            continue
                        if nm == want:
                            best = jid
                            break
                        if want in nm or nm in want:
                            best = jid
                    except Exception:
                        continue
            job_id = best
        except Exception:
            job_id = ""

    if not job_id:
        return None

    # Best-effort: if we're allowed to fetch and not in cache-only mode, also cache the job details
    # (including `steps[]`). This avoids "cached raw log exists but no step breakdown" when later
    # runs are forced into cache-only mode (budget/rate-limit).
    try:
        if allow_fetch and (not bool(github.cache_only_mode)):
            _ = github.get_actions_job_details_cached(owner=owner, repo=repo, job_id=str(job_id), ttl_s=7 * 24 * 3600)
    except Exception:
        pass

    # Ensure we use a canonical /job/<id> URL for fetch APIs that require it.
    job_url_for_fetch = job_url
    if "/job/" not in job_url_for_fetch:
        if run_id:
            job_url_for_fetch = f"https://github.com/{owner}/{repo}/actions/runs/{run_id}/job/{job_id}"
        else:
            # Best-effort fallback (no run_id): still build a canonical URL; it should work for log download.
            job_url_for_fetch = f"https://github.com/{owner}/{repo}/actions/job/{job_id}"

    try:
        page_root_dir = Path(page_root_dir)
    except Exception:
        return None

    # Destination path: consolidated repo-local served cache.
    try:
        dest_dir = dashboard_served_raw_log_repo_cache_dir(page_root_dir=page_root_dir)
        # If dest_dir is a symlink, mkdir would fail; ensure the resolved target exists instead.
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            try:
                Path(dest_dir).resolve().mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        dest_path = dest_dir / f"{job_id}.log"
        # Always link under the dashboard root (never to /.cache/...).
        href = f"raw-log-text/{job_id}.log"
    except Exception:
        return None

    # Enforce "completed-only" policy even if a stale file already exists.
    if not bool(assume_completed):
        try:
            st = str(github.get_actions_job_status(owner=owner, repo=repo, job_id=job_id) or "").lower()
            if not st or st != "completed":
                try:
                    if dest_path.exists():
                        dest_path.unlink()
                except Exception:
                    pass
                return None
        except Exception:
            return None

    # If already materialized, return immediately (no IO / network).
    try:
        if dest_path.exists():
            # Count this as a raw-log cache hit (served page cache already has <job_id>.log).
            try:
                github._cache_hit("raw_log_text.page")
            except Exception:
                pass
            return href
    except Exception:
        pass

    # If the served dir is just a symlink to the global cache, no copying is needed.
    try:
        global_dir = (common.dynamo_utils_cache_dir() / "raw-log-text").resolve()
        if Path(dest_dir).resolve() == global_dir:
            # Ensure the file exists in the global cache (fetch if allowed).
            if dest_path.exists():
                return href
    except Exception:
        pass

    # If we already have the global cached file, just copy it into the served cache (or symlink target).
    try:
        global_txt = github._raw_log_text_cache_dir / f"{job_id}.log"  # type: ignore[attr-defined]
        legacy_txt = github._raw_log_text_cache_dir / f"{job_id}.txt"  # type: ignore[attr-defined]
        src = global_txt if global_txt.exists() else (legacy_txt if legacy_txt.exists() else None)
        if src is not None:
            # Only reuse global cache if it is marked as completed in the cache index.
            try:
                meta = {}
                idx = github._raw_log_text_index_file  # type: ignore[attr-defined]
                if idx.exists():
                    meta = json.loads(idx.read_text() or "{}")
                ent = meta.get(job_id) if isinstance(meta, dict) else None
                if not (isinstance(ent, dict) and bool(ent.get("completed", False))):
                    src = None
            except Exception:
                src = None
        if src is not None:
            tmp = str(dest_path) + ".tmp"
            shutil.copyfile(str(src), tmp)
            os.replace(tmp, dest_path)
            # Count this as a cache hit (copied from global raw-log cache into served cache).
            try:
                github._cache_hit("raw_log_text.global")
            except Exception:
                pass
            return href
    except Exception:
        pass

    if not allow_fetch:
        return None

    # Fetch (or refresh) the cached text, then copy it into page_root_dir.
    try:
        _ = github.get_job_raw_log_text_cached(
            job_url=job_url_for_fetch,
            owner=owner,
            repo=repo,
            ttl_s=int(ttl_s),
            assume_completed=bool(assume_completed),
        )
    except Exception:
        return None

    try:
        global_txt = github._raw_log_text_cache_dir / f"{job_id}.log"  # type: ignore[attr-defined]
        legacy_txt = github._raw_log_text_cache_dir / f"{job_id}.txt"  # type: ignore[attr-defined]
        src = global_txt if global_txt.exists() else (legacy_txt if legacy_txt.exists() else None)
        if src is None:
            return None
        tmp = str(dest_path) + ".tmp"
        shutil.copyfile(str(src), tmp)
        os.replace(tmp, dest_path)
        return href
    except Exception:
        return None


