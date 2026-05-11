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

    ext_knowledge_rag = Mock()
    ext_knowledge_rag.get_knowledge_size.return_value = 1

    reference_template_rag = Mock()
    reference_template_rag.get_reference_template_size.return_value = 0

    with (
        patch("datus.tools.func_tool.context_search.MetricRAG", return_value=metric_rag),
        patch("datus.tools.func_tool.context_search.SemanticModelRAG", return_value=semantic_rag),
        patch("datus.tools.func_tool.context_search.ReferenceSqlRAG", return_value=sql_rag),
        patch("datus.tools.func_tool.context_search.ExtKnowledgeRAG", return_value=ext_knowledge_rag),
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
        "search_knowledge",
        "get_knowledge",
    }


def test_unrelated_tool_scope_does_not_expose_context_tools():
    tools = _build_tools({"tools": "db_tools.*"})

    assert tools.available_tools() == []


def test_list_subject_tree_scope_does_not_expose_query_tools():
    tools = _build_tools({"tools": "context_search_tools.list_subject_tree"})

    tool_names = {tool.name for tool in tools.available_tools()}

    assert tool_names == {"list_subject_tree"}


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
        "search_knowledge",
        "get_knowledge",
    }
