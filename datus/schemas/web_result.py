# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Canonical result schema + summaries for the ``web_tool`` capability group.

Every web backend (local Tavily / httpx, Anthropic server tools, OpenAI hosted
``web_search``) normalizes into ONE shape here so the TUI, API SSE and
``datus -p`` event payloads render identically regardless of which provider
served the call.

Canonical shapes (stored as the tool ``result`` / ``raw_output.result``):

``web_search``::

    {
        "query": "duckdb latest release",
        "result_count": 3,
        "results": [
            {"title": "...", "url": "https://...", "snippet": "...", "age": "2 days ago"},
            ...
        ],
    }

``web_fetch``::

    {
        "url": "https://...",
        "title": "DuckDB 1.5 Release Notes",
        "content": "<full extracted text>",
        "truncated": false,
        "char_count": 12431,
    }

The two ``*_short_summary`` helpers produce the compact one-liner used as the
SSE ``shortDesc`` and the TUI compact ``└─`` line — the only place the short
form is ever used.
"""

from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

# Cap a single result snippet so the full-result payload stays bounded while
# still being far richer than the compact shortDesc.
_SNIPPET_MAX_CHARS = 500


def _domain(url: str) -> str:
    """Best-effort host label from a URL (``https://a.b/c`` -> ``a.b``)."""
    try:
        host = urlparse(url).netloc
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def normalize_search_results(query: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the canonical ``web_search`` result from raw per-result dicts.

    Each ``items`` entry may carry ``title`` / ``url`` / ``snippet`` (or
    ``content``) / ``age``; missing fields normalize to empty. Entries with no
    title, url AND snippet are dropped so empty rows never reach the UI.
    """
    results: List[Dict[str, Any]] = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        title = str(it.get("title") or "").strip()
        url = str(it.get("url") or "").strip()
        snippet = str(it.get("snippet") or it.get("content") or "").strip()
        if len(snippet) > _SNIPPET_MAX_CHARS:
            snippet = snippet[:_SNIPPET_MAX_CHARS].rstrip() + "…"
        if not (title or url or snippet):
            continue
        results.append({"title": title, "url": url, "snippet": snippet, "age": it.get("age")})
    return {"query": str(query or "").strip(), "result_count": len(results), "results": results}


def normalize_fetch_result(
    url: str,
    title: str,
    content: str,
    truncated: bool,
) -> Dict[str, Any]:
    """Build the canonical ``web_fetch`` result."""
    content = content or ""
    return {
        "url": str(url or "").strip(),
        "title": str(title or "").strip(),
        "content": content,
        "truncated": bool(truncated),
        "char_count": len(content),
    }


def _result_label(item: Dict[str, Any]) -> str:
    """Short label for one search result: title, else domain, else raw url."""
    title = str(item.get("title") or "").strip()
    if title:
        return title
    url = str(item.get("url") or "").strip()
    return _domain(url) or url


def web_search_short_summary(result: Any) -> str:
    """Compact one-liner for a ``web_search`` result.

    ``3 web results: Title A; Title B; Title C`` when results carry labels;
    falls back to ``searched: "<query>"`` when the backend exposes only the
    query (OpenAI hosted with no citations yet).
    """
    if not isinstance(result, dict):
        return ""
    results = result.get("results")
    if isinstance(results, list) and results:
        labels = [lbl for lbl in (_result_label(r) for r in results[:3]) if lbl]
        n = result.get("result_count", len(results))
        noun = "web result" if n == 1 else "web results"
        if labels:
            return f"{n} {noun}: " + "; ".join(labels)
        return f"{n} {noun}"
    query = str(result.get("query") or "").strip()
    if query:
        return f'searched: "{query}"'
    return "completed"


def web_fetch_short_summary(result: Any) -> str:
    """Compact one-liner for a ``web_fetch`` result: ``Title (12,431 chars)``."""
    if not isinstance(result, dict):
        return ""
    label = str(result.get("title") or "").strip() or _domain(str(result.get("url") or "")) or "fetched"
    char_count = result.get("char_count")
    if not isinstance(char_count, int):
        content = result.get("content")
        char_count = len(content) if isinstance(content, str) else 0
    suffix = " (truncated)" if result.get("truncated") else ""
    return f"{label} ({char_count:,} chars){suffix}"


def extract_url_citations(content_parts: Any) -> List[Dict[str, Any]]:
    """Collect ``url_citation`` annotations from an OpenAI message's content.

    OpenAI hosted ``web_search`` does not surface results on the
    ``web_search_call`` item; the title+url pairs the model actually used come
    back as ``url_citation`` annotations on the assistant message. Each yields a
    ``{title, url, snippet}`` row (snippet left empty — annotations carry only a
    text span index, not the source text).
    """
    citations: List[Dict[str, Any]] = []
    seen: set = set()
    for part in content_parts or []:
        annotations = getattr(part, "annotations", None)
        if annotations is None and isinstance(part, dict):
            annotations = part.get("annotations")
        for ann in annotations or []:
            atype = getattr(ann, "type", None) if not isinstance(ann, dict) else ann.get("type")
            if atype != "url_citation":
                continue
            url = (getattr(ann, "url", None) if not isinstance(ann, dict) else ann.get("url")) or ""
            title = (getattr(ann, "title", None) if not isinstance(ann, dict) else ann.get("title")) or ""
            url = str(url).strip()
            if not url or url in seen:
                continue
            seen.add(url)
            citations.append({"title": str(title).strip(), "url": url, "snippet": ""})
    return citations


def extract_action_sources(action: Any) -> List[Dict[str, Any]]:
    """Collect source URLs from a hosted ``web_search_call.action.sources`` list.

    Present only when the request set ``include=['web_search_call.action.sources']``.
    These carry a URL but no title, so they are a fallback for / supplement to
    the richer :func:`extract_url_citations` rows.
    """
    sources = getattr(action, "sources", None) if not isinstance(action, dict) else (action or {}).get("sources")
    rows: List[Dict[str, Any]] = []
    for src in sources or []:
        url = (getattr(src, "url", None) if not isinstance(src, dict) else src.get("url")) or ""
        url = str(url).strip()
        if url:
            rows.append({"title": "", "url": url, "snippet": ""})
    return rows


def collect_citations_from_input_list(input_list: Any) -> List[Dict[str, Any]]:
    """Aggregate ``url_citation`` rows across every assistant message in a turn.

    OpenAI's hosted ``web_search`` does not attach results to the
    ``web_search_call`` item; the title+url pairs the model used surface as
    ``url_citation`` annotations on the assistant message(s). After the run we
    read ``RunResultStreaming.to_input_list()`` (which preserves annotations) and
    fold all citations together — they become the canonical results for the
    turn's hosted search call(s).
    """
    citations: List[Dict[str, Any]] = []
    seen: set = set()
    for item in input_list or []:
        role = item.get("role") if isinstance(item, dict) else getattr(item, "role", None)
        itype = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        # Aggregate citations from assistant messages only. Accept across payload
        # variants: when ``role`` is present it must be ``assistant``; when it is
        # omitted upstream, fall back to the message type.
        if role is not None:
            if role != "assistant":
                continue
        elif itype != "message":
            continue
        content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
        for row in extract_url_citations(content):
            if row["url"] in seen:
                continue
            seen.add(row["url"])
            citations.append(row)
    return citations


def hosted_action_query(action: Any) -> Optional[str]:
    """Pull a human-readable query / url out of a hosted ``web_search_call`` action."""
    if action is None:
        return None

    def _get(key: str) -> Any:
        return action.get(key) if isinstance(action, dict) else getattr(action, key, None)

    return _get("query") or _get("url") or _get("queries")
