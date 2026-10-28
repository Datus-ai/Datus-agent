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

from datus.tools.func_tool.web_tool import WebTool, _is_public_web_target


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


def test_create_dynamic_and_static_factories():
    cfg = _cfg("k")
    dyn = WebTool.create_dynamic(cfg, sub_agent_name="a")
    stat = WebTool.create_static(cfg, sub_agent_name="b", database_name="db")
    assert isinstance(dyn, WebTool) and dyn.sub_agent_name == "a"
    assert isinstance(stat, WebTool) and stat.sub_agent_name == "b"


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://duckdb.org", True),
        ("http://example.com/path", True),
        ("ftp://example.com", False),  # non-http scheme
        ("http://", False),  # no hostname
        ("https://127.0.0.1", False),  # literal loopback
        ("https://10.1.2.3", False),  # literal private
    ],
)
def test_is_public_web_target_direct(url, expected):
    assert _is_public_web_target(url) is expected


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


def test_web_search_falls_back_to_first_docs_value_when_query_key_absent():
    # When the backend keys results by something other than the joined query,
    # web_search falls back to the first docs value.
    tool = WebTool(_cfg("key"))
    fake = SimpleNamespace(
        success=True,
        docs={"some-other-key": [{"title": "T", "url": "https://u", "snippet": "s"}]},
        doc_count=1,
        error=None,
    )
    with patch("datus.tools.search_tools.search_tool.search_by_tavily", return_value=fake):
        res = tool.web_search(["foo", "bar"])
    assert res.success == 1
    assert res.result["results"][0]["title"] == "T"


def test_web_search_unexpected_exception_returns_error():
    tool = WebTool(_cfg("key"))
    with patch("datus.tools.search_tools.search_tool.search_by_tavily", side_effect=RuntimeError("kaboom")):
        res = tool.web_search(["q"])
    assert res.success == 0
    assert "kaboom" in res.error


def test_web_fetch_parse_error_returns_error():
    tool = WebTool(_cfg())
    html = "<html><body><p>hi</p></body></html>"
    with (
        patch("datus.tools.func_tool.web_tool.httpx.get", return_value=_resp(html)),
        patch("datus.tools.func_tool.web_tool.BeautifulSoup", side_effect=ValueError("bad parser")),
    ):
        res = tool.web_fetch("https://duckdb.org/x")
    assert res.success == 0
    assert "Failed to parse" in res.error


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


@pytest.mark.parametrize(
    "blocked",
    [
        "http://127.0.0.1",  # loopback
        "http://127.0.0.1:8080/admin",
        "http://[::1]/",  # IPv6 loopback
        "http://169.254.169.254/latest/meta-data/",  # link-local (cloud metadata)
        "http://10.0.0.1",  # private
        "http://192.168.1.1",  # private
        "http://172.16.0.5",  # private
        "http://0.0.0.0",  # unspecified
    ],
)
def test_web_fetch_rejects_ssrf_targets(blocked):
    # Literal-IP hosts are validated directly (no DNS), so the request must be
    # refused before any httpx call is made.
    tool = WebTool(_cfg())
    with patch("datus.tools.func_tool.web_tool.httpx.get") as m:
        res = tool.web_fetch(blocked)
    assert res.success == 0
    assert "SSRF" in res.error
    m.assert_not_called()


@pytest.mark.parametrize(
    "blocked_host",
    ["http://localhost/x", "http://localhost:9000", "http://db.internal/q", "http://myhost.local/"],
)
def test_web_fetch_rejects_internal_hostnames(blocked_host):
    # Internal hostnames are refused without any DNS resolution.
    tool = WebTool(_cfg())
    with patch("datus.tools.func_tool.web_tool.httpx.get") as m:
        res = tool.web_fetch(blocked_host)
    assert res.success == 0
    assert "SSRF" in res.error
    m.assert_not_called()


def test_web_fetch_allows_public_hostname():
    # Public hostnames are allowed (resolution happens at fetch time, not in the
    # guard) — so the mocked httpx path runs normally.
    html = "<html><head><title>OK</title></head><body><p>hi</p></body></html>"
    tool = WebTool(_cfg())
    with patch("datus.tools.func_tool.web_tool.httpx.get", return_value=_resp(html)):
        res = tool.web_fetch("https://duckdb.org/")
    assert res.success == 1
    assert res.result["title"] == "OK"


@pytest.mark.parametrize("bad_max", [0, -1, -100, 3.5, True, "20"])
def test_web_fetch_rejects_invalid_max_chars(bad_max):
    tool = WebTool(_cfg())
    with patch("datus.tools.func_tool.web_tool.httpx.get") as m:
        res = tool.web_fetch("http://e.com", max_chars=bad_max)
    assert res.success == 0
    assert "max_chars" in res.error
    m.assert_not_called()


def test_web_fetch_rejects_oversized_content_length():
    tool = WebTool(_cfg())
    r = _resp("<html><body><p>hi</p></body></html>")
    r.headers = {"content-type": "text/html", "content-length": str(50 * 1024 * 1024)}
    with patch("datus.tools.func_tool.web_tool.httpx.get", return_value=r):
        res = tool.web_fetch("http://e.com/huge")
    assert res.success == 0
    assert "too large" in res.error.lower()
