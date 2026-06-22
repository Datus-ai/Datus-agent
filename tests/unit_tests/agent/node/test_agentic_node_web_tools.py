# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.
"""Tests for AgenticNode._ensure_web_tools_in_tools backend assembly.

Exercises the node-layer coordination as an unbound method against a Mock
``self`` to avoid full node construction. Verifies the provider-driven choice
between local function tools and vendor-native (suppressed-local) tools, plus
reconciliation on a runtime provider switch.
"""

from types import SimpleNamespace
from unittest.mock import Mock

from datus.agent.node.agentic_node import AgenticNode


def _node(search_builtin, fetch_builtin, tavily="k", model=True):
    n = Mock()
    if model:
        n.model.supports_builtin_web_search.return_value = search_builtin
        n.model.supports_builtin_web_fetch.return_value = fetch_builtin
    else:
        n.model = None
    n.agent_config = SimpleNamespace(tavily_api_key=tavily)
    n.sub_agent_name = None
    n.tools = []
    n.get_node_name.return_value = "t"
    return n


def _names(n):
    return {t.name for t in n.tools}


def test_local_both_when_no_builtin():
    n = _node(False, False)
    AgenticNode._ensure_web_tools_in_tools(n)
    assert n._builtin_web_tools == {"web_search": False, "web_fetch": False}
    assert _names(n) == {"web_search", "web_fetch"}


def test_builtin_search_suppresses_local_search():
    n = _node(True, False)
    AgenticNode._ensure_web_tools_in_tools(n)
    assert n._builtin_web_tools["web_search"] is True
    assert _names(n) == {"web_fetch"}


def test_builtin_both_no_local_tools():
    n = _node(True, True)
    AgenticNode._ensure_web_tools_in_tools(n)
    assert _names(n) == set()


def test_no_tavily_key_drops_local_search(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    n = _node(False, False, tavily=None)
    AgenticNode._ensure_web_tools_in_tools(n)
    assert _names(n) == {"web_fetch"}


def test_model_unavailable_defaults_to_local():
    n = _node(False, False, model=False)
    AgenticNode._ensure_web_tools_in_tools(n)
    assert n._builtin_web_tools == {"web_search": False, "web_fetch": False}
    assert _names(n) == {"web_search", "web_fetch"}


def test_reconcile_drops_stale_web_tool_on_switch():
    # A prior turn mounted a local web_search; provider then switches to builtin
    # search. The stale local web_search must be removed, unrelated tools kept.
    n = _node(True, False)
    stale = Mock()
    stale.name = "web_search"
    other = Mock()
    other.name = "read_query"
    n.tools = [other, stale]
    AgenticNode._ensure_web_tools_in_tools(n)
    assert _names(n) == {"read_query", "web_fetch"}


def test_idempotent_no_duplicate_mounts():
    n = _node(False, False)
    AgenticNode._ensure_web_tools_in_tools(n)
    AgenticNode._ensure_web_tools_in_tools(n)
    web = sorted(t.name for t in n.tools if t.name in {"web_search", "web_fetch"})
    assert web == ["web_fetch", "web_search"]
