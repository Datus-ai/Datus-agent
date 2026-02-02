# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Document Fetcher Module

Provides fetchers for retrieving documentation from various sources:
- GitHub repositories (README, docs directories)
- Official websites (HTML documentation)
- Local file system directories (Markdown, HTML files)
"""

from datus.storage.document.fetcher.base_fetcher import BaseFetcher
from datus.storage.document.fetcher.github_fetcher import GitHubFetcher, GitHubFetchMetadata
from datus.storage.document.fetcher.local_fetcher import LocalFetcher
from datus.storage.document.fetcher.rate_limiter import RateLimiter
from datus.storage.document.fetcher.web_fetcher import WebFetcher

__all__ = [
    "BaseFetcher",
    "RateLimiter",
    "GitHubFetcher",
    "GitHubFetchMetadata",
    "WebFetcher",
    "LocalFetcher",
]
