# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/agent/node/node_factory.py
"""

from unittest.mock import MagicMock, patch

from datus.agent.node.node_factory import _resolve_node_class_type, create_interactive_node, create_node_input

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_agent_config(**kwargs):
    config = MagicMock()
    config.agentic_nodes = kwargs.get("agentic_nodes", None)
    return config


# ---------------------------------------------------------------------------
# Tests: _resolve_node_class_type
# ---------------------------------------------------------------------------


class TestResolveNodeClassType:
    def test_no_agentic_nodes(self):
        config = _mock_agent_config(agentic_nodes=None)
        assert _resolve_node_class_type("my_agent", config) is None

    def test_missing_subagent(self):
        config = _mock_agent_config(agentic_nodes={"other": {}})
        assert _resolve_node_class_type("my_agent", config) is None

    def test_returns_node_class(self):
        config = _mock_agent_config(agentic_nodes={"my_agent": {"node_class": "gen_report"}})
        assert _resolve_node_class_type("my_agent", config) == "gen_report"

    def test_pydantic_model_dump(self):
        node_config = MagicMock()
        node_config.model_dump.return_value = {"node_class": "gen_report"}
        config = _mock_agent_config(agentic_nodes={"my_agent": node_config})
        assert _resolve_node_class_type("my_agent", config) == "gen_report"


# ---------------------------------------------------------------------------
# Tests: create_interactive_node
# ---------------------------------------------------------------------------


class TestCreateInteractiveNode:
    @patch("datus.agent.node.chat_agentic_node.ChatAgenticNode.__init__", return_value=None)
    def test_default_chat_node(self, mock_init):
        config = _mock_agent_config()
        create_interactive_node(None, config, node_id_suffix="_test")
        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["node_id"] == "chat_test"
        assert call_kwargs["node_type"] == "chat"

    @patch("datus.agent.node.gen_semantic_model_agentic_node.GenSemanticModelAgenticNode.__init__", return_value=None)
    def test_gen_semantic_model(self, mock_init):
        config = _mock_agent_config()
        create_interactive_node("gen_semantic_model", config)
        mock_init.assert_called_once_with(agent_config=config, execution_mode="interactive")

    @patch("datus.agent.node.gen_metrics_agentic_node.GenMetricsAgenticNode.__init__", return_value=None)
    def test_gen_metrics(self, mock_init):
        config = _mock_agent_config()
        create_interactive_node("gen_metrics", config)
        mock_init.assert_called_once_with(agent_config=config, execution_mode="interactive")

    @patch("datus.agent.node.sql_summary_agentic_node.SqlSummaryAgenticNode.__init__", return_value=None)
    def test_gen_sql_summary(self, mock_init):
        config = _mock_agent_config()
        create_interactive_node("gen_sql_summary", config)
        mock_init.assert_called_once_with(
            node_name="gen_sql_summary", agent_config=config, execution_mode="interactive"
        )

    @patch("datus.agent.node.gen_ext_knowledge_agentic_node.GenExtKnowledgeAgenticNode.__init__", return_value=None)
    def test_gen_ext_knowledge(self, mock_init):
        config = _mock_agent_config()
        create_interactive_node("gen_ext_knowledge", config)
        mock_init.assert_called_once_with(
            node_name="gen_ext_knowledge", agent_config=config, execution_mode="interactive"
        )

    @patch("datus.agent.node.gen_report_agentic_node.GenReportAgenticNode.__init__", return_value=None)
    def test_gen_report(self, mock_init):
        config = _mock_agent_config()
        create_interactive_node("gen_report", config, node_id_suffix="_cli")
        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["node_id"] == "gen_report_cli"
        assert call_kwargs["node_type"] == "gen_report"

    @patch("datus.agent.node.gen_report_agentic_node.GenReportAgenticNode.__init__", return_value=None)
    @patch("datus.agent.node.node_factory._resolve_node_class_type", return_value="gen_report")
    def test_config_driven_gen_report(self, mock_resolve, mock_init):
        config = _mock_agent_config()
        create_interactive_node("custom_agent", config)
        mock_init.assert_called_once()
        assert mock_init.call_args[1]["node_name"] == "custom_agent"

    @patch("datus.agent.node.gen_sql_agentic_node.GenSQLAgenticNode.__init__", return_value=None)
    def test_default_subagent_is_gensql(self, mock_init):
        config = _mock_agent_config()
        create_interactive_node("my_custom_sql", config, node_id_suffix="_cli")
        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["node_id"] == "my_custom_sql_cli"
        assert call_kwargs["node_type"] == "gensql"


# ---------------------------------------------------------------------------
# Tests: create_node_input
# ---------------------------------------------------------------------------


class TestCreateNodeInput:
    def test_chat_node_input(self):
        from datus.agent.node.chat_agentic_node import ChatAgenticNode

        node = MagicMock(spec=ChatAgenticNode)
        result = create_node_input("hello", node, catalog="cat", database="db")
        assert result.user_message == "hello"
        assert result.catalog == "cat"

    def test_gensql_node_input(self):
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        node = MagicMock(spec=GenSQLAgenticNode)
        result = create_node_input("generate SQL", node, catalog="cat", plan_mode=True)
        assert result.user_message == "generate SQL"
        assert result.plan_mode is True

    def test_semantic_node_input(self):
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        node = MagicMock(spec=GenSemanticModelAgenticNode)
        result = create_node_input("build model", node, catalog="cat", prompt_language="zh")
        assert result.user_message == "build model"
        assert result.prompt_language == "zh"

    def test_metrics_node_input(self):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        node = MagicMock(spec=GenMetricsAgenticNode)
        result = create_node_input("gen metrics", node)
        assert result.user_message == "gen metrics"

    def test_sql_summary_node_input(self):
        from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode

        node = MagicMock(spec=SqlSummaryAgenticNode)
        result = create_node_input("summarize", node, database="mydb")
        assert result.user_message == "summarize"
        assert result.database == "mydb"

    def test_ext_knowledge_node_input(self):
        from datus.agent.node.gen_ext_knowledge_agentic_node import GenExtKnowledgeAgenticNode

        node = MagicMock(spec=GenExtKnowledgeAgenticNode)
        result = create_node_input("add knowledge", node)
        assert result.user_message == "add knowledge"
        assert result.catalog is None
        assert result.database is None
        assert result.db_schema is None

    def test_ext_knowledge_node_input_with_db_context(self):
        from datus.agent.node.gen_ext_knowledge_agentic_node import GenExtKnowledgeAgenticNode

        node = MagicMock(spec=GenExtKnowledgeAgenticNode)
        result = create_node_input("add knowledge", node, catalog="cat", database="db", db_schema="sch")
        assert result.user_message == "add knowledge"
        assert result.catalog == "cat"
        assert result.database == "db"
        assert result.db_schema == "sch"

    def test_gen_report_node_input(self):
        from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode

        node = MagicMock(spec=GenReportAgenticNode)
        result = create_node_input("report", node, catalog="cat", database="db")
        assert result.user_message == "report"
        assert result.catalog == "cat"
