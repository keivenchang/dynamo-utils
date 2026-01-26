# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GitLab API error types.

These are intentionally lightweight so cached API modules can catch specific
error classes (e.g. 404 Not Found) without creating import cycles.
"""

from __future__ import annotations


class GitLabAPIError(Exception):
    def __init__(self, *, status_code: int, endpoint: str, message: str):
        super().__init__(message)
        self.status_code = int(status_code)
        self.endpoint = str(endpoint or "")


class GitLabAuthError(GitLabAPIError):
    pass


class GitLabForbiddenError(GitLabAPIError):
    pass


class GitLabNotFoundError(GitLabAPIError):
    pass


class GitLabRequestError(GitLabAPIError):
    pass

