# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""ContextSearchTools sub-agent scope tests."""

from unittest.mock import Mock, patch

from datus.tools.func_tool.context_search import ContextSearchTools


def _build_tools(sub_agent_config):
    agent_config = Mock()
    agent_config.sub_agent_config.return_value = sub_agent_config

    metric_rag = Mock()
    metric_rag.storage.subject_tree = Mock()
    metric_rag.get_metrics_size.return_value = 1

    sql_rag = Mock()
    sql_rag.get_reference_sql_size.return_value = 1

    semantic_rag = Mock()
    semantic_rag.get_size.return_value = 1

    reference_template_rag = Mock()
    reference_template_rag.get_reference_template_size.return_value = 0

    with (
        patch("datus.tools.func_tool.context_search.MetricRAG", return_value=metric_rag),
        patch("datus.tools.func_tool.context_search.SemanticModelRAG", return_value=semantic_rag),
        patch("datus.tools.func_tool.context_search.ReferenceSqlRAG", return_value=sql_rag),
        patch("datus.tools.func_tool.context_search.ReferenceTemplateRAG", return_value=reference_template_rag),
    ):
        return ContextSearchTools(agent_config, sub_agent_name="chat")


def test_context_wildcard_includes_all_available_context_tools():
    tools = _build_tools({"tools": "context_search_tools.*,date_parsing_tools.*"})

    tool_names = {tool.name for tool in tools.available_tools()}

    assert tool_names == {
        "list_subject_tree",
        "search_metrics",
        "get_metrics",
        "search_reference_sql",
        "get_reference_sql",
        "search_semantic_objects",
    }


def test_unrelated_tool_scope_does_not_expose_context_tools():
    tools = _build_tools({"tools": "db_tools.*"})

    assert tools.available_tools() == []


def test_list_subject_tree_scope_does_not_expose_query_tools():
    tools = _build_tools({"tools": "context_search_tools.list_subject_tree"})

    tool_names = {tool.name for tool in tools.available_tools()}

    assert tool_names == {"list_subject_tree"}


def test_list_subject_tree_description_does_not_name_disabled_context_tools():
    tools = _build_tools(
        {
            "tools": (
                "context_search_tools.search_metrics,context_search_tools.list_subject_tree,db_tools,date_parsing_tools"
            )
        }
    )

    available_tools = tools.available_tools()
    tool_names = {tool.name for tool in available_tools}
    subject_tree_tool = next(tool for tool in available_tools if tool.name == "list_subject_tree")

    assert tool_names == {"list_subject_tree", "search_metrics", "get_metrics"}
    assert "enabled context retrieval tools" in subject_tree_tool.description
    assert "get_reference_sql" not in subject_tree_tool.description


def test_missing_sub_agent_config_uses_default_context_tools():
    tools = _build_tools({})

    tool_names = {tool.name for tool in tools.available_tools()}

    assert tool_names == {
        "list_subject_tree",
        "search_metrics",
        "get_metrics",
        "search_reference_sql",
        "get_reference_sql",
        "search_semantic_objects",
    }


def test_declared_config_without_tools_key_inherits_default_context_tools():
    # A named node config (e.g. ``agentic_nodes.gen_sql: {max_turns, system_prompt}``)
    # that sets other keys but omits ``tools:`` yields an EMPTY tool_list. An empty
    # list must mean "did not restrict → inherit node defaults (allow)", not
    # "deny everything". Before this was fixed, such a bare node silently lost all
    # context-search tools even though its DEFAULT_TOOLS enabled context_search_tools.*.
    tools = _build_tools({"max_turns": 100, "system_prompt": "gen_sql"})

    tool_names = {tool.name for tool in tools.available_tools()}

    assert tool_names == {
        "list_subject_tree",
        "search_metrics",
        "get_metrics",
        "search_reference_sql",
        "get_reference_sql",
        "search_semantic_objects",
    }
