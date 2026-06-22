# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.
"""Unit tests for the unified web tool group (web_tool.web_search / web_fetch).

All external calls are mocked (Tavily via ``search_by_tavily``; HTTP via
``httpx.get``). Deterministic, no network — CI tier.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest

from datus.tools.func_tool.web_tool import WebTool


def _cfg(tavily=None):
    return SimpleNamespace(tavily_api_key=tavily)


def _names(tool):
    return {t.name for t in tool.available_tools()}


# --- available_tools backend gating -------------------------------------------------


def test_available_tools_local_both_with_key():
    tool = WebTool(_cfg("k"), expose_local_search=True, expose_local_fetch=True)
    assert _names(tool) == {"web_search", "web_fetch"}


def test_available_tools_no_tavily_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    tool = WebTool(_cfg(None), expose_local_search=True, expose_local_fetch=True)
    # web_search suppressed without a key; web_fetch always available locally.
    assert _names(tool) == {"web_fetch"}


def test_available_tools_builtin_search_suppresses_local():
    tool = WebTool(_cfg("k"), expose_local_search=False, expose_local_fetch=True)
    assert _names(tool) == {"web_fetch"}


def test_available_tools_builtin_both_empty():
    tool = WebTool(_cfg("k"), expose_local_search=False, expose_local_fetch=False)
    assert _names(tool) == set()


def test_all_tools_name_full_surface():
    assert WebTool.all_tools_name() == ["web_search", "web_fetch"]


def test_env_var_key_enables_search(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "envkey")
    tool = WebTool(_cfg(None), expose_local_search=True, expose_local_fetch=False)
    assert _names(tool) == {"web_search"}


# --- web_search (Tavily backend) ----------------------------------------------------


def test_web_search_passes_tavily_params_and_maps_result():
    tool = WebTool(_cfg("key123"))
    # structured=True keys results by the tab-joined query; web_search maps them
    # into the canonical {query, result_count, results:[{title,url,snippet,age}]}.
    query = "foo\tbar"
    fake = SimpleNamespace(
        success=True,
        docs={query: [{"title": "T1", "url": "https://a.com", "snippet": "snip", "raw_content": "raw"}]},
        doc_count=1,
        error=None,
    )
    with patch("datus.tools.search_tools.search_tool.search_by_tavily", return_value=fake) as m:
        res = tool.web_search(["foo", "bar"], max_results=3, include_domains=["x.com"])
    assert res.success == 1
    assert res.result["query"] == "foo, bar"
    assert res.result["result_count"] == 1
    assert res.result["results"] == [{"title": "T1", "url": "https://a.com", "snippet": "snip", "age": None}]
    kwargs = m.call_args.kwargs
    assert kwargs["keywords"] == ["foo", "bar"]
    assert kwargs["max_results"] == 3
    assert kwargs["search_depth"] == "advanced"
    assert kwargs["include_answer"] == "basic"
    assert kwargs["include_raw_content"] == "markdown"
    assert kwargs["include_domains"] == ["x.com"]
    assert kwargs["api_key"] == "key123"
    assert kwargs["structured"] is True


def test_web_search_backend_failure_returns_error():
    tool = WebTool(_cfg("key"))
    fake = SimpleNamespace(success=False, docs={}, doc_count=0, error="rate limited")
    with patch("datus.tools.search_tools.search_tool.search_by_tavily", return_value=fake):
        res = tool.web_search(["q"])
    assert res.success == 0
    assert res.error == "rate limited"


def test_web_search_without_key_fails(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    tool = WebTool(_cfg(None))
    res = tool.web_search(["q"])
    assert res.success == 0
    assert "tavily" in res.error.lower()


# --- web_fetch (httpx backend) ------------------------------------------------------


def _resp(text="", content_type="text/html; charset=utf-8", url="http://e.com"):
    r = Mock()
    r.text = text
    r.headers = {"content-type": content_type}
    r.url = url
    r.raise_for_status = Mock()
    return r


def test_web_fetch_extracts_text_and_strips_noise():
    html = (
        "<html><head><title>My Page</title></head>"
        "<body><nav>menu</nav><script>evil()</script>"
        "<p>Hello world</p><footer>foot</footer></body></html>"
    )
    tool = WebTool(_cfg())
    with patch("datus.tools.func_tool.web_tool.httpx.get", return_value=_resp(html)):
        res = tool.web_fetch("http://e.com/page")
    assert res.success == 1
    assert res.result["title"] == "My Page"
    assert "Hello world" in res.result["content"]
    assert "evil" not in res.result["content"]
    assert "menu" not in res.result["content"]
    assert "foot" not in res.result["content"]
    assert res.result["truncated"] is False


def test_web_fetch_truncates_long_content():
    html = "<html><body><p>" + ("A" * 500) + "</p></body></html>"
    tool = WebTool(_cfg())
    with patch("datus.tools.func_tool.web_tool.httpx.get", return_value=_resp(html)):
        res = tool.web_fetch("http://e.com", max_chars=100)
    assert res.success == 1
    assert len(res.result["content"]) == 100
    assert res.result["truncated"] is True


def test_web_fetch_http_status_error():
    tool = WebTool(_cfg())
    r = _resp()
    r.raise_for_status.side_effect = httpx.HTTPStatusError("boom", request=Mock(), response=Mock(status_code=404))
    with patch("datus.tools.func_tool.web_tool.httpx.get", return_value=r):
        res = tool.web_fetch("http://e.com/missing")
    assert res.success == 0
    assert "404" in res.error


def test_web_fetch_transport_error():
    tool = WebTool(_cfg())
    with patch("datus.tools.func_tool.web_tool.httpx.get", side_effect=httpx.ConnectError("no route")):
        res = tool.web_fetch("http://e.com")
    assert res.success == 0
    assert "Failed to fetch" in res.error


def test_web_fetch_rejects_non_html_content_type():
    tool = WebTool(_cfg())
    with patch("datus.tools.func_tool.web_tool.httpx.get", return_value=_resp("{}", content_type="application/json")):
        res = tool.web_fetch("http://e.com/data.json")
    assert res.success == 0
    assert "content-type" in res.error.lower()


@pytest.mark.parametrize("bad", ["ftp://x", "/local/path", "example.com", ""])
def test_web_fetch_rejects_non_http_url(bad):
    tool = WebTool(_cfg())
    res = tool.web_fetch(bad)
    assert res.success == 0
    assert "URL" in res.error
