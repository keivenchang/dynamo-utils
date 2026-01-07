"""Regex catalog for `ci_log_errors`.

Goal: keep regexes *discoverable* and *stable*.

Conventions:
- CAT_*     : categorization (full-log + snippet categorization)
- SNIPPET_* : snippet extraction / command prelude detection
- RED_*     : full-line red highlighting rules

This module is intentionally "boring":
- no side effects
- no imports from other `ci_log_errors` modules (avoid cycles)
- grouped + alphabetized names to reduce grep time
"""

from __future__ import annotations

import re
from typing import List, Pattern, Tuple

#
# =============================================================================
# CAT_* (categorization)
# =============================================================================
#

# Backend result blocks are printed as JSON-ish text in logs, e.g.:
#   "sglang": { ... "result": "failure", ... }
CAT_BACKEND_BLOCK_START_RE: Pattern[str] = re.compile(r"\"(trtllm|sglang|vllm)\"\s*:\s*\{", re.IGNORECASE)
CAT_BACKEND_RESULT_FAILURE_RE: Pattern[str] = re.compile(r"\"result\"\s*:\s*\"failure\"", re.IGNORECASE)

# NOTE: These are intentionally *not* all IGNORECASE because many call sites lowercase the text first.
CAT_BROKEN_LINKS_RE: Pattern[str] = re.compile(r"\bbroken\s+links?\b|\bdead\s+links?\b", re.IGNORECASE)

CAT_BUILD_STATUS_CHECK_ERROR_RE: Pattern[str] = re.compile(
    r"(?:"
    r"\bchecking\s+build\s+status\s+for\b"
    r"|\bbuild\s+status\s+for\s+'Build\b"
    r"|\bError:\s*Failed\s+to\s+query\s+GitHub\s+API\b"
    r"|\bBuild\s+failed\s+or\s+did\s+not\s+complete\s+successfully\.\s*(?:Failing\s+tests|Marking\s+tests\s+as\s+failed)\b"
    r")",
    re.IGNORECASE,
)

CAT_COPYRIGHT_HEADER_ERROR_RE: Pattern[str] = re.compile(
    r"(?:"
    r"\bcopyright-checks\b"
    r"|\bcopyright\s+checkers\s+detected\s+missing\s+or\s+invalid\s+copyright\s+headers\b"
    r"|\bInvalid/Missing\s+Header:\b"
    r"|\binvalid/missing\s+header:\b"
    r")",
    re.IGNORECASE,
)

CAT_CUDA_ERROR_RE: Pattern[str] = re.compile(
    r"(?:"
    r"unsupported\s+cuda\s+version\s+for\s+vllm\s+installation"
    r"|\bcuda\b[^\n]{0,120}\bunsupported\b"
    r"|\bimporterror:\s*libcuda\.so\.1:\s*cannot\s+open\s+shared\s+object\s+file\b"
    r")"
)

CAT_DOCKER_BUILD_ERROR_RE: Pattern[str] = re.compile(r"\berror:\s*failed\s+to\s+build\b|\bfailed\s+to\s+solve\b")
CAT_DOCKER_CLI_ERROR_RE: Pattern[str] = re.compile(r"\bdocker:\s+.*\berror\b", re.IGNORECASE)

CAT_DOCKER_DAEMON_CONNECTION_ERROR_RE: Pattern[str] = re.compile(r"cannot\s+connect\s+to\s+the\s+docker\s+daemon", re.IGNORECASE)
CAT_DOCKER_DAEMON_ERROR_RESPONSE_RE: Pattern[str] = re.compile(
    r"error\s+response\s+from\s+daemon:(?!.*no\s+such\s+container)", re.IGNORECASE
)
CAT_DOCKER_INFRA_ERROR_RE: Pattern[str] = re.compile(
    r"(?:"
    r"cannot\s+connect\s+to\s+the\s+docker\s+daemon"
    r"|error\s+response\s+from\s+daemon:(?!.*no\s+such\s+container)"
    r"|\bdocker:\s+.*\berror\b"
    r")"
)

CAT_DOWNLOAD_ERROR_RE: Pattern[str] = re.compile(r"\bcaused by:\s*failed to download\b|\bfailed to download\b|\bdownload error\b")

CAT_ETCD_ERROR_RE: Pattern[str] = re.compile(
    r"\bunable\s+to\s+create\s+lease\b|\bcheck\s+etcd\s+server\s+status\b|\betcd[^\n]{0,80}\blease\b|\blease\b[^\n]{0,80}\betcd\b"
)

CAT_EXIT_CODE_127_RE: Pattern[str] = re.compile(
    r"process completed with exit code 127\b|exit code:\s*127\b|\bcommand not found\b",
    re.IGNORECASE,
)
CAT_EXIT_CODE_139_RE: Pattern[str] = re.compile(r"process completed with exit code 139\b|exit code:\s*139\b", re.IGNORECASE)

CAT_GITHUB_ACTION_STEP_TIMEOUT_RE: Pattern[str] = re.compile(
    r"##\[error\].{0,40}\bhas\s+timed\s+out\s+after\s+\d+\s+minutes?\b",
    re.IGNORECASE,
)
CAT_GITHUB_API_RE: Pattern[str] = re.compile(
    r"Failed to query GitHub API|secondary rate limit|API rate limit exceeded|HTTP 403|HTTP 429", re.IGNORECASE
)
CAT_GITHUB_FETCH_RE: Pattern[str] = re.compile(r"failed to fetch some objects from|RPC failed|early EOF|remote end hung up|fetch-pack", re.IGNORECASE)
CAT_GITHUB_LFS_RE: Pattern[str] = re.compile(r"/info/lfs|git lfs", re.IGNORECASE)

CAT_GITLAB_MIRROR_TIMEOUT_RE: Pattern[str] = re.compile(r"\bmirror sync failed or timed out\b", re.IGNORECASE)

CAT_HELM_ERROR_RE: Pattern[str] = re.compile(
    r"(?:"
    r"\bUPGRADE\s+FAILED\b"
    r"|\bINSTALLATION\s+FAILED\b"
    r"|\bKubernetes\s+cluster\s+unreachable\b"
    r"|\bhelm\b[^\n]{0,120}\berror\b"
    r"|\berror:\s*flag\s+needs\s+an\s+argument:\s*'n'\s+in\s+-n\b"
    r"|\berror:\s*resource\(s\)\s+were\s+provided,\s+but\s+no\s+name\s+was\s+specified\b"
    r")",
    re.IGNORECASE,
)

CAT_HUGGINGFACE_AUTH_ERROR_RE: Pattern[str] = re.compile(
    r"(?:"
    r"\bHF_TOKEN\s+not\s+found\s+in\s+environment\b"
    r"|\bHfHubHTTPError\b"
    r"|\bGatedRepoError\b"
    r"|\bRepositoryNotFoundError\b"
    r"|\bhuggingface[_-]hub\b[^\n]{0,160}\b(unauthorized|forbidden|invalid|token)\b"
    r"|\b401\b[^\n]{0,120}\bhuggingface\b"
    r"|\b403\b[^\n]{0,120}\bhuggingface\b"
    r")",
    re.IGNORECASE,
)

CAT_HTTP_TIMEOUT_RE: Pattern[str] = re.compile(
    r"(?:"
    r"awaiting\s+response\.\.\.\s*(?:504|503|502)\b"
    r"|gateway\s+time-?out"
    r"|\bhttp\s+(?:504|503|502)\b"
    # Lychee/link-checker timeouts:
    #   [TIMEOUT] https://example.com | Timeout
    r"|\[timeout\]\s+https?://"
    r")"
)

CAT_K8S_ERROR_RE: Pattern[str] = re.compile(
    r"(?:"
    r"CrashLoopBackOff|ImagePullBackOff|ErrImagePull"
    r"|Kubernetes\s+cluster\s+unreachable"
    r"|\bkubectl\b[^\n]{0,200}\b("
    r"error|failed|failure|timed\s*out|timeout|unable|forbidden|unauthorized|refused|i/o\s+timeout|not\s+found|"
    r"context\s+deadline\s+exceeded"
    r")\b"
    r")",
    re.IGNORECASE,
)

CAT_K8S_PODS_TIMED_OUT_RE: Pattern[str] = re.compile(
    r"\btimed\s*out\s+waiting\s+for\s+the\s+condition\s+on\s+pods/",
    re.IGNORECASE,
)

CAT_KUBECTL_PORTFORWARD_TIMEOUT_RE: Pattern[str] = re.compile(
    r"(?:"
    r"\bportforward\.go:\d+\].*\bconnection timed out\b"
    r"|\ban error occurred forwarding\b[^\n]{0,400}\bconnection timed out\b"
    r")",
    re.IGNORECASE,
)

CAT_NETWORK_ERROR_RE: Pattern[str] = re.compile(
    r"\bnetwork\s+error:\s*connection\s+failed\b|\bconnection\s+failed\.\s*check\s+network\s+connectivity\b|\bfirewall\s+settings\b"
)

CAT_OOM_RE: Pattern[str] = re.compile(r"\b(out of memory|CUDA out of memory|Killed process|oom)\b", re.IGNORECASE)

CAT_PYTEST_DETECT_RE: Pattern[str] = re.compile(
    r"(?:"
    # Restrict to pytest-style failing test ids: "FAILED path/to/test_foo.py::test_name[...]"
    r"(?:^|[ \t])FAILED(?:[ \t]+|$)[^\\n]*\\.py::"
    r"|==+\\s*FAILURES\\s*==+"
    r"|==+\\s*ERRORS\\s*==+"
    r"|\berror[ \t]+collecting\b"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

CAT_PYTEST_ERROR_RE: Pattern[str] = re.compile(
    r"(?:^|\s)FAILED(?:\s+|$).*::|\berror\s+collecting\b|==+\s*(?:FAILURES|ERRORS)\s*==+",
    re.IGNORECASE,
)

CAT_RUST_TEST_FAIL_RE: Pattern[str] = re.compile(r"^\s*failures:\s*$|test result:\s*FAILED\.", re.IGNORECASE | re.MULTILINE)

CAT_TIMED_OUT_RE: Pattern[str] = re.compile(r"\b(?:timed\s*out|timedout)\b")

# Shared categorization rules (used by BOTH full-log categorization and snippet categorization).
#
# IMPORTANT: keep the ORDER stable (UX/tests rely on the output order), even though regex *definitions*
# in this module are alphabetized.
CAT_RULES: List[Tuple[str, Pattern[str]]] = [
    ("pytest-timeout-error", re.compile(r"\bE\s+Failed:\s+Timeout\b.*\bpytest-timeout\b", re.IGNORECASE)),
    ("pytest-error", CAT_PYTEST_ERROR_RE),
    ("network-download-error", re.compile(CAT_DOWNLOAD_ERROR_RE.pattern, re.IGNORECASE)),
    ("docker-build-error", re.compile(CAT_DOCKER_BUILD_ERROR_RE.pattern, re.IGNORECASE)),
    ("build-status-check-error", CAT_BUILD_STATUS_CHECK_ERROR_RE),
    ("huggingface-auth-error", CAT_HUGGINGFACE_AUTH_ERROR_RE),
    ("copyright-header-error", CAT_COPYRIGHT_HEADER_ERROR_RE),
    ("helm-error", CAT_HELM_ERROR_RE),
    ("cuda-error", re.compile(CAT_CUDA_ERROR_RE.pattern, re.IGNORECASE)),
    ("network-timeout-https", re.compile(CAT_HTTP_TIMEOUT_RE.pattern, re.IGNORECASE)),
    ("network-timeout-gitlab-mirror", CAT_GITLAB_MIRROR_TIMEOUT_RE),
    ("k8s-network-timeout-pod", CAT_K8S_PODS_TIMED_OUT_RE),
    ("k8s-network-timeout-portfwd", CAT_KUBECTL_PORTFORWARD_TIMEOUT_RE),
    ("network-timeout-github-action", CAT_GITHUB_ACTION_STEP_TIMEOUT_RE),
    ("network-error", re.compile(CAT_NETWORK_ERROR_RE.pattern, re.IGNORECASE)),
    ("etcd-error", re.compile(CAT_ETCD_ERROR_RE.pattern, re.IGNORECASE)),
    ("git-fetch", CAT_GITHUB_FETCH_RE),
    ("github-api", CAT_GITHUB_API_RE),
    ("github-lfs-error", CAT_GITHUB_LFS_RE),
    ("network-timeout-generic", re.compile(CAT_TIMED_OUT_RE.pattern, re.IGNORECASE)),
    ("oom", CAT_OOM_RE),
    ("docker-daemon-connection-error", CAT_DOCKER_DAEMON_CONNECTION_ERROR_RE),
    ("docker-daemon-error-response-error", CAT_DOCKER_DAEMON_ERROR_RESPONSE_RE),
    ("docker-cli-error", CAT_DOCKER_CLI_ERROR_RE),
    # NOTE: docker-image-error regex lives outside "categorization section" today; will be migrated next.
    ("k8s-error", CAT_K8S_ERROR_RE),
    # NOTE: python-error regex lives outside "categorization section" today; will be migrated next.
    ("broken-links", CAT_BROKEN_LINKS_RE),
    ("rust-error", CAT_RUST_TEST_FAIL_RE),
    ("exit-139-sigsegv", CAT_EXIT_CODE_139_RE),
    ("exit-127-cmd-not-found", CAT_EXIT_CODE_127_RE),
]

#
# =============================================================================
# SNIPPET_* (snippet extraction)
# =============================================================================
#

# Lines that should anchor an "error snippet" extraction from raw logs.
# Keep this conservative and high-signal to avoid pulling unrelated noise.
SNIPPET_ANCHOR_LINE_RE: Pattern[str] = re.compile(
    r"(?:"
    r"\b(?:error|failed|failure|exception|traceback|fatal)\b"
    # Pytest failures (the exact failing test id line is the most useful snippet anchor).
    # Example: "FAILED tests/x.py::test_name[param]"
    r"|(?:^|\s)FAILED(?:\s+|$).*::"
    # Timeout: keep this very conservative to avoid false positives from dependency strings like
    # "pytest-timeout==2.4.0" / "timeout-2.4.0" and from prose like "individual test timeouts".
    r"|\b(?:timed\s*out|timedout)\b"
    # Generic error/exception class tokens (CamelCase) like "ModuleNotFoundError:".
    # Avoid false positives from crate/package names like "serde_path_to_error" and from stack traces.
    r"|(?-i:(?<![./])\b[A-Z][A-Za-z0-9]{2,}(?:Error|Exception)(?::|\b$))"
    r"|\b(?:broken\s+links?|broken\s+link|dead\s+links?)\b"
    r"|\b(?:network\s+error|connection\s+failed)\b"
    # Multi-line backend result blocks (JSON-ish) often show:
    #   "trtllm": { ... "result": "failure", ... }
    # Anchor on the high-signal failure field so the snippet includes the surrounding block.
    r"|\"result\"\s*:\s*\"failure\""
    # Copyright header checks
    r"|\bInvalid/Missing\s+Header:\b"
    r"|\binvalid/missing\s+header:\b"
    r"|\bcopyright\s+checkers\s+detected\s+missing\s+or\s+invalid\s+copyright\s+headers\b"
    # HuggingFace auth / missing token warnings
    r"|\bHF_TOKEN\s+not\s+found\s+in\s+environment\b"
    r")",
    re.IGNORECASE,
)

# Snippet command/execution lines (user wants these shown prominently in blue).
SNIPPET_COMMAND_LINE_BLUE_RE: Pattern[str] = re.compile(
    r"(?:"
    r"\bPYTEST_CMD\s*="
    # Treat pytest as a command only when it doesn't look like a Python module/variable dump:
    #   pytest = <module 'pytest' from '...'>
    r"|\bpython\s+-m\s+pytest\b(?!\s*=)"
    r"|\bpytest\b(?!\s*=)(?:\s|$)"
    r"|\bbash\s+-c\s+['\"][^'\"]*\bpytest\b"
    r"|\bcargo\s+(?:test|build|check|clippy|fmt|rustfmt)\b"
    r"|\bdocker\s+(?:run|buildx|build)\b"
    r"|\b(?:\./)?run\.sh\b"
    r"|\b(?:\./)?build\.sh\b"
    r")",
    re.IGNORECASE,
)

SNIPPET_PYTEST_CMD_LINE_RE: Pattern[str] = re.compile(r"\bPYTEST_CMD\s*=", re.IGNORECASE)
SNIPPET_PYTEST_ERROR_FILE_LINE_RE: Pattern[str] = re.compile(
    r"(?:^|\s)\bERROR\s+components/src/dynamo/trtllm/tests/test_trtllm_[^\s]+\.py\b",
    re.IGNORECASE,
)
SNIPPET_PYTEST_FAILED_LINE_RE: Pattern[str] = re.compile(
    r"(?:^|\s)FAILED(?:\s+|$).*::",
    re.IGNORECASE,
)

#
# =============================================================================
# RED_* (full-line red highlighting)
# =============================================================================
#

RED_KEYWORD_HIGHLIGHT_RE: Pattern[str] = re.compile(
    r"(?:"
    r"\b(?:error|failed|failure|exception|traceback|fatal)\b"
    r"|\bno\s+module\s+named\b"
    # Timeout: keep conservative to avoid false positives like "timeout-2.4.0" (pytest plugin) or prose.
    r"|\b(?:timed\s*out|timedout)\b"
    r"|\b(?:gateway\s+time-?out)\b"
    r"|\b(?:http\s*)?(?:502|503|504)\b"
    r"|\b(?:network\s+error|connection\s+failed|check\s+network\s+connectivity|firewall\s+settings)\b"
    # Don't highlight bare "etcd"/"lease" (too many false positives). Highlight the actual error phrases.
    r"|\b(?:unable\s+to\s+create\s+lease|check\s+etcd\s+server\s+status)\b"
    # Generic error/exception class tokens (CamelCase) like "ModuleNotFoundError:".
    # Avoid false positives from crate/package names like "serde_path_to_error" and from stack traces.
    r"|(?-i:(?<![./])\b[A-Z][A-Za-z0-9]{2,}(?:Error|Exception)(?::|\b$))"
    r"|\b(?:broken\s+links?|broken\s+link|dead\s+links?)\b"
    r")",
    re.IGNORECASE,
)

RED_DOCKER_DAEMON_ERROR_LINE_RE: Pattern[str] = re.compile(r"^.*\berror response from daemon:.*$", re.IGNORECASE)
RED_DOCKER_NO_SUCH_CONTAINER_RE: Pattern[str] = re.compile(
    r"\berror response from daemon:\s*no\s+such\s+container\b",
    re.IGNORECASE,
)
RED_NETWORK_ERROR_LINE_RE: Pattern[str] = re.compile(
    r"\bnetwork\s+error:\s*connection\s+failed\b|\bconnection\s+failed\.\s*check\s+network\s+connectivity\b",
    re.IGNORECASE,
)

# This list is ORDERED (not alphabetized): it’s a UX/tuning list, not a catalog.
RED_FULL_LINE_RES: List[Pattern[str]] = [
    # Kubernetes timeouts - highlight explicitly
    re.compile(r"\btimed\s*out\s+waiting\s+for\s+the\s+condition\s+on\s+pods/", re.IGNORECASE),
    re.compile(r"\bconnection timed out\b", re.IGNORECASE),
    # Helm errors
    re.compile(r"\bUPGRADE\s+FAILED\b|\bINSTALLATION\s+FAILED\b", re.IGNORECASE),
    # Exit codes
    re.compile(r"\bprocess completed with exit code 139\b|\bexit code:\s*139\b", re.IGNORECASE),
    re.compile(r"\bprocess completed with exit code 127\b|\bexit code:\s*127\b|\bcommand not found\b", re.IGNORECASE),
    # Network / docker infra errors
    RED_DOCKER_DAEMON_ERROR_LINE_RE,
    RED_DOCKER_NO_SUCH_CONTAINER_RE,
    RED_NETWORK_ERROR_LINE_RE,
    # Pytest file-level ERROR markers
    SNIPPET_PYTEST_ERROR_FILE_LINE_RE,
    # Backend failure field
    re.compile(r"\"result\"\s*:\s*\"failure\"", re.IGNORECASE),
    # Docker registry manifest missing (kept as legacy in engine for now)
    # Timeout markers inserted by snippet extraction / categorization.
    re.compile(r"\[TIMEOUT\]", re.IGNORECASE),
    # Assertion failures
    re.compile(r"\bassertion\s+failed:", re.IGNORECASE),
    # Build failure summary
    re.compile(r"\berror:\s*failed\s+to\s+build\b", re.IGNORECASE),
    # Local policy checks emitting [FAIL]
    re.compile(r"\[FAIL\]\s*incorrect\s+date\s*:", re.IGNORECASE),
    # Git LFS failures
    re.compile(r"\bGit\s+operation\s+failed\b", re.IGNORECASE),
    re.compile(r"\bfailed\s+to\s+fetch\s+LFS\s+objects\b", re.IGNORECASE),
    # Pytest-timeout plugin marker
    re.compile(r"\bE\s+Failed:\s+Timeout\b.*\bpytest-timeout\b", re.IGNORECASE),
    # 100% progress line containing failing F
    re.compile(r"\[100%\].*F", re.IGNORECASE),
    # Rust harness failures
    re.compile(r"^\s*failures:\s*$", re.IGNORECASE),
    re.compile(r"^\s*[A-Za-z0-9_:]+\s*$", re.IGNORECASE),
    re.compile(r"test result:\s*FAILED\.", re.IGNORECASE),
]



