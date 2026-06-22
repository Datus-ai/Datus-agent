# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.
# -*- coding: utf-8 -*-
"""Unified web tool group (``web_tool.web_search`` / ``web_tool.web_fetch``).

The agent always sees a single pair of web capabilities. The backend is chosen
per active provider, but that decision lives outside this module:

* For providers that expose a vendor-native web tool (Codex / OpenAI Responses
  ``web_search``; Claude native ``web_search_20250305`` / ``web_fetch_20250910``),
  the model layer injects the hosted tool and the node suppresses the matching
  *local* function tool here via ``expose_local_search`` / ``expose_local_fetch``.
* For every other provider, the local backends below are used:
  ``web_search`` calls Tavily (needs a key) and ``web_fetch`` pulls the page
  directly with httpx — no third-party API.

This class deliberately does NOT inspect the provider; it only owns the local
backends and exposes whichever ones the node asks for.
"""

import os
from typing import List, Optional

import httpx
from agents import Tool
from bs4 import BeautifulSoup

from datus.configuration.agent_config import AgentConfig
from datus.schemas.web_result import normalize_fetch_result, normalize_search_results
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_NAME = "web_tool"
_NAME_WEB_SEARCH = "web_tool.web_search"
_NAME_WEB_FETCH = "web_tool.web_fetch"

# Tags that hold boilerplate rather than the main content of a page.
_NOISE_TAGS = ("script", "style", "noscript", "nav", "footer", "header", "aside", "form")

_DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; DatusAgent/1.0; +https://datus.ai) Python-httpx web_fetch"


class WebTool:
    """Function-call tool group for web access.

    Exposes up to two LLM-callable functions:
    - web_search: Search the web (local backend = Tavily).
    - web_fetch: Fetch and extract the readable text of a single URL (httpx).

    Which functions are exposed locally is controlled by ``expose_local_search``
    / ``expose_local_fetch`` — the node turns the local backend off when the
    active provider serves that capability through a vendor-native tool.
    """

    permission_category: str = "web_tool"

    def __init__(
        self,
        agent_config: AgentConfig,
        sub_agent_name: Optional[str] = None,
        expose_local_search: bool = True,
        expose_local_fetch: bool = True,
    ):
        self.agent_config = agent_config
        self.sub_agent_name = sub_agent_name
        self.expose_local_search = expose_local_search
        self.expose_local_fetch = expose_local_fetch

    @classmethod
    def create_dynamic(cls, agent_config: AgentConfig, sub_agent_name: Optional[str] = None) -> "WebTool":
        return cls(agent_config, sub_agent_name=sub_agent_name)

    @classmethod
    def create_static(
        cls,
        agent_config: AgentConfig,
        sub_agent_name: Optional[str] = None,
        database_name: Optional[str] = None,
    ) -> "WebTool":
        return cls(agent_config, sub_agent_name=sub_agent_name)

    @staticmethod
    def all_tools_name() -> List[str]:
        # Full surface, independent of runtime availability — used by the node's
        # tool registry so ``web_search`` / ``web_fetch`` always map to the
        # ``web_tool`` permission category even when one is suppressed.
        return ["web_search", "web_fetch"]

    def _tavily_key(self) -> Optional[str]:
        return getattr(self.agent_config, "tavily_api_key", None) or os.environ.get("TAVILY_API_KEY")

    def available_tools(self) -> List[Tool]:
        """Return the locally-served web tools, filtered by backend availability.

        - ``web_search`` is exposed only when ``expose_local_search`` is set
          AND a Tavily key is resolvable. When the provider serves search via a
          hosted tool, ``expose_local_search`` is False and we skip it.
        - ``web_fetch`` is exposed whenever ``expose_local_fetch`` is set; the
          httpx backend needs no key, so it is always available locally.
        """
        tools: List[Tool] = []

        if self.expose_local_search:
            if self._tavily_key():
                tools.append(trans_to_function_tool(self.web_search))
            else:
                logger.info(
                    "Skipping web_tool.web_search: neither agent.document.tavily_api_key "
                    "nor TAVILY_API_KEY env var is set, and no vendor-native search is available."
                )

        if self.expose_local_fetch:
            tools.append(trans_to_function_tool(self.web_fetch))

        return tools

    def web_search(
        self,
        keywords: List[str],
        max_results: int = 5,
        include_domains: Optional[List[str]] = None,
    ) -> FuncToolResult:
        """
        Search the web for up-to-date information or technical documentation.

        Use this when local knowledge is insufficient or you need the latest
        information from official sites, blogs, or community resources. Returns
        ranked results with markdown content; pair it with ``web_fetch`` to pull
        the full text of a specific result URL.

        Args:
            keywords: Search queries (e.g., ["StarRocks materialized view syntax"]).
            max_results: Maximum number of results to return, 1-20 (default: 5).
            include_domains: Restrict the search to specific domains (optional),
                e.g., ["docs.snowflake.com", "docs.starrocks.io"].

        Returns:
            FuncToolResult with the canonical ``web_search`` result
            (``{"query", "result_count", "results": [{"title", "url", "snippet", "age"}]}``).
        """
        try:
            from datus.tools.search_tools.search_tool import search_by_tavily

            tavily_key = self._tavily_key()
            if not tavily_key:
                return FuncToolResult(
                    success=0,
                    error="Web search is unavailable: set agent.document.tavily_api_key or TAVILY_API_KEY.",
                )

            result = search_by_tavily(
                keywords=keywords,
                max_results=max_results,
                search_depth="advanced",
                include_answer="basic",
                include_raw_content="markdown",
                include_domains=include_domains,
                api_key=tavily_key,
                structured=True,
            )

            if not result.success:
                return FuncToolResult(success=0, error=result.error)

            # ``structured=True`` keys results by the tab-joined query; flatten
            # to the canonical schema so every backend renders identically.
            query = "\t".join(keywords)
            items = result.docs.get(query) if result.docs else None
            if items is None and result.docs:
                items = next(iter(result.docs.values()), [])
            return FuncToolResult(
                success=1,
                result=normalize_search_results(", ".join(keywords), items or []),
            )
        except Exception as e:
            logger.error(f"Web search failed for keywords {keywords}: {e}")
            return FuncToolResult(success=0, error=str(e))

    def web_fetch(
        self,
        url: str,
        max_chars: int = 20000,
    ) -> FuncToolResult:
        """
        Fetch a single web page and return its readable text content.

        Pulls the URL directly over HTTP (no third-party API), strips scripts,
        styles, navigation, and other boilerplate, and returns the main text.
        Use this to read a specific page — e.g. a URL surfaced by ``web_search``
        or provided by the user.

        Args:
            url: Absolute http(s) URL to fetch.
            max_chars: Truncate the extracted text to this many characters
                (default: 20000). The ``truncated`` flag reports whether the
                content was cut off.

        Returns:
            FuncToolResult with the canonical ``web_fetch`` result
            (``{"url", "title", "content", "truncated", "char_count"}``).
        """
        if not isinstance(url, str) or not url.lower().startswith(("http://", "https://")):
            return FuncToolResult(success=0, error=f"Invalid URL (must start with http:// or https://): {url!r}")

        try:
            response = httpx.get(
                url,
                follow_redirects=True,
                timeout=30,
                headers={"User-Agent": _DEFAULT_USER_AGENT},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            return FuncToolResult(success=0, error=f"HTTP {e.response.status_code} fetching {url}")
        except httpx.HTTPError as e:
            return FuncToolResult(success=0, error=f"Failed to fetch {url}: {e}")

        content_type = (response.headers.get("content-type") or "").lower()
        if content_type and not any(t in content_type for t in ("text/html", "application/xhtml", "text/plain")):
            return FuncToolResult(
                success=0,
                error=f"Unsupported content-type '{content_type}' for {url}; web_fetch handles HTML/text only.",
            )

        try:
            soup = BeautifulSoup(response.text, "lxml")
            for tag in soup(list(_NOISE_TAGS)):
                tag.decompose()

            title = soup.title.get_text(strip=True) if soup.title else ""
            text = soup.get_text(separator="\n", strip=True)
            # Collapse runs of blank lines left behind by stripped tags.
            lines = [line for line in (ln.strip() for ln in text.splitlines()) if line]
            text = "\n".join(lines)

            truncated = len(text) > max_chars
            if truncated:
                text = text[:max_chars]

            return FuncToolResult(
                success=1,
                result=normalize_fetch_result(
                    url=str(response.url),
                    title=title,
                    content=text,
                    truncated=truncated,
                ),
            )
        except Exception as e:
            logger.error(f"Failed to parse content from {url}: {e}")
            return FuncToolResult(success=0, error=f"Failed to parse {url}: {e}")
