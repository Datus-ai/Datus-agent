# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.
"""Contract tests for the canonical web_tool result schema + summaries.

These guarantee every backend (Tavily / Anthropic / OpenAI hosted) renders the
same shape, and that the compact one-liners behave across the present/absent
field combinations each backend produces.
"""

from types import SimpleNamespace

from datus.schemas.web_result import (
    collect_citations_from_input_list,
    extract_action_sources,
    extract_url_citations,
    hosted_action_query,
    normalize_fetch_result,
    normalize_search_results,
    web_fetch_short_summary,
    web_search_short_summary,
)

# ── normalize_search_results ────────────────────────────────────────


def test_normalize_search_results_fills_fields_and_drops_empty():
    out = normalize_search_results(
        "duckdb",
        [
            {"title": "A", "url": "https://a.com", "snippet": "s", "age": "1d"},
            {"content": "from-content-key"},  # snippet falls back to ``content``
            {},  # fully empty -> dropped
            {"title": "", "url": "", "snippet": ""},  # blank -> dropped
        ],
    )
    assert out["query"] == "duckdb"
    assert out["result_count"] == 2
    assert out["results"][0] == {"title": "A", "url": "https://a.com", "snippet": "s", "age": "1d"}
    assert out["results"][1]["snippet"] == "from-content-key"
    assert out["results"][1]["age"] is None


def test_normalize_search_results_clips_long_snippet():
    out = normalize_search_results("q", [{"url": "https://x", "snippet": "z" * 999}])
    snippet = out["results"][0]["snippet"]
    assert snippet.endswith("…")
    assert len(snippet) <= 501


# ── normalize_fetch_result ──────────────────────────────────────────


def test_normalize_fetch_result_sets_char_count():
    out = normalize_fetch_result(url="https://x", title="T", content="hello", truncated=True)
    assert out == {"url": "https://x", "title": "T", "content": "hello", "truncated": True, "char_count": 5}


# ── summaries ───────────────────────────────────────────────────────


def test_web_search_summary_lists_titles():
    result = normalize_search_results("q", [{"title": "Alpha", "url": "u1"}, {"title": "Beta", "url": "u2"}])
    assert web_search_short_summary(result) == "2 web results: Alpha; Beta"


def test_web_search_summary_falls_back_to_domain_then_query():
    # No titles -> domain labels.
    result = normalize_search_results("q", [{"url": "https://docs.example.com/x"}])
    assert web_search_short_summary(result) == "1 web result: docs.example.com"
    # No results at all -> the query (OpenAI hosted before citations).
    empty = normalize_search_results("duckdb latest", [])
    assert web_search_short_summary(empty) == 'searched: "duckdb latest"'


def test_web_fetch_summary_uses_title_and_char_count():
    result = normalize_fetch_result(url="https://x", title="Release Notes", content="x" * 12431, truncated=False)
    assert web_fetch_short_summary(result) == "Release Notes (12,431 chars)"


def test_web_fetch_summary_truncated_and_domain_fallback():
    result = normalize_fetch_result(url="https://www.duckdb.org/p", title="", content="abc", truncated=True)
    assert web_fetch_short_summary(result) == "duckdb.org (3 chars) (truncated)"


# ── OpenAI hosted extraction (objects and dicts) ────────────────────


def test_normalize_search_results_skips_non_dict_items():
    out = normalize_search_results("q", ["not-a-dict", {"title": "Ok", "url": "https://o", "snippet": "s"}])
    assert out["result_count"] == 1
    assert out["results"][0]["title"] == "Ok"


def test_web_search_summary_non_dict_returns_empty():
    assert web_search_short_summary("nope") == ""
    assert web_search_short_summary(None) == ""


def test_web_search_summary_results_without_labels():
    # Results present but each lacks title/url -> count only, no label list.
    result = {"result_count": 2, "results": [{"snippet": "a"}, {"snippet": "b"}]}
    assert web_search_short_summary(result) == "2 web results"


def test_web_search_summary_completed_when_empty():
    assert web_search_short_summary({}) == "completed"


def test_web_fetch_summary_non_dict_returns_empty():
    assert web_fetch_short_summary(["x"]) == ""


def test_web_fetch_summary_counts_content_when_char_count_missing():
    # char_count absent -> derived from content length.
    result = {"title": "", "url": "https://duckdb.org", "content": "abcd"}
    assert web_fetch_short_summary(result) == "duckdb.org (4 chars)"


def test_extract_url_citations_dedups_and_skips_non_citation():
    parts = [
        SimpleNamespace(
            annotations=[
                SimpleNamespace(type="url_citation", title="T1", url="https://a"),
                SimpleNamespace(type="url_citation", title="dup", url="https://a"),  # dup url
                SimpleNamespace(type="file_citation", title="ignored", url="https://b"),
            ]
        )
    ]
    rows = extract_url_citations(parts)
    assert rows == [{"title": "T1", "url": "https://a", "snippet": ""}]


def test_extract_action_sources_objects_and_dicts():
    action = {"sources": [{"type": "url", "url": "https://a"}, {"type": "url", "url": ""}]}
    assert extract_action_sources(action) == [{"title": "", "url": "https://a", "snippet": ""}]


def test_hosted_action_query_prefers_query_then_url():
    assert hosted_action_query({"query": "q"}) == "q"
    assert hosted_action_query({"url": "https://x"}) == "https://x"
    assert hosted_action_query(None) is None


def test_collect_citations_from_input_list_folds_assistant_messages():
    input_list = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"type": "web_search_call", "action": {"query": "duckdb"}},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "...",
                    "annotations": [
                        {"type": "url_citation", "title": "DuckDB", "url": "https://duckdb.org"},
                    ],
                }
            ],
        },
    ]
    rows = collect_citations_from_input_list(input_list)
    assert rows == [{"title": "DuckDB", "url": "https://duckdb.org", "snippet": ""}]


def test_collect_citations_ignores_user_message_annotations():
    # A user (non-assistant) message that itself carries url_citation annotations
    # must NOT contribute to the hosted web_search canonical results.
    input_list = [
        {
            "role": "user",
            "type": "message",
            "content": [
                {
                    "type": "output_text",
                    "annotations": [
                        {"type": "url_citation", "title": "Bad", "url": "https://example.com/bad"},
                    ],
                }
            ],
        },
        {
            "role": "assistant",
            "type": "message",
            "content": [
                {
                    "type": "output_text",
                    "annotations": [
                        {"type": "url_citation", "title": "Good", "url": "https://duckdb.org"},
                    ],
                }
            ],
        },
    ]
    assert collect_citations_from_input_list(input_list) == [
        {"title": "Good", "url": "https://duckdb.org", "snippet": ""}
    ]
