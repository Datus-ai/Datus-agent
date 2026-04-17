# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/agent/node/node_factory.py
"""

from unittest.mock import MagicMock, patch

import pytest

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
        mock_init.assert_called_once_with(agent_config=config, execution_mode="interactive", scope=None)

    @patch("datus.agent.node.gen_metrics_agentic_node.GenMetricsAgenticNode.__init__", return_value=None)
    def test_gen_metrics(self, mock_init):
        config = _mock_agent_config()
        create_interactive_node("gen_metrics", config)
        mock_init.assert_called_once_with(agent_config=config, execution_mode="interactive", scope=None)

    @patch("datus.agent.node.sql_summary_agentic_node.SqlSummaryAgenticNode.__init__", return_value=None)
    def test_gen_sql_summary(self, mock_init):
        config = _mock_agent_config()
        create_interactive_node("gen_sql_summary", config)
        mock_init.assert_called_once_with(
            node_name="gen_sql_summary", agent_config=config, execution_mode="interactive", scope=None
        )

    @patch("datus.agent.node.gen_ext_knowledge_agentic_node.GenExtKnowledgeAgenticNode.__init__", return_value=None)
    def test_gen_ext_knowledge(self, mock_init):
        config = _mock_agent_config()
        create_interactive_node("gen_ext_knowledge", config)
        mock_init.assert_called_once_with(
            node_name="gen_ext_knowledge", agent_config=config, execution_mode="interactive", scope=None
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

    @patch("datus.agent.node.gen_table_agentic_node.GenTableAgenticNode.__init__", return_value=None)
    def test_gen_table(self, mock_init):
        config = _mock_agent_config()
        create_interactive_node("gen_table", config)
        mock_init.assert_called_once_with(agent_config=config, execution_mode="interactive", node_name=None)

    @patch("datus.agent.node.gen_sql_agentic_node.GenSQLAgenticNode.__init__", return_value=None)
    def test_default_subagent_is_gensql(self, mock_init):
        config = _mock_agent_config()
        create_interactive_node("my_custom_sql", config, node_id_suffix="_cli")
        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["node_id"] == "my_custom_sql_cli"
        assert call_kwargs["node_type"] == "gensql"

    @patch("datus.agent.node.gen_table_agentic_node.GenTableAgenticNode.__init__", return_value=None)
    @patch("datus.agent.node.node_factory._resolve_node_class_type", return_value="gen_table")
    def test_config_driven_gen_table(self, mock_resolve, mock_init):
        config = _mock_agent_config()
        create_interactive_node("wide_table_builder", config)
        mock_init.assert_called_once_with(
            agent_config=config,
            execution_mode="interactive",
            node_name="wide_table_builder",
        )

    @patch("datus.agent.node.gen_skill_agentic_node.SkillCreatorAgenticNode.__init__", return_value=None)
    @patch("datus.agent.node.node_factory._resolve_node_class_type", return_value="gen_skill")
    def test_config_driven_gen_skill(self, mock_resolve, mock_init):
        config = _mock_agent_config()
        create_interactive_node("skill_designer", config, node_id_suffix="_cli")
        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["node_id"] == "skill_designer_cli"
        assert call_kwargs["node_name"] == "skill_designer"
        assert call_kwargs["node_type"] == "gen_skill"

    @patch("datus.agent.node.gen_dashboard_agentic_node.GenDashboardAgenticNode.__init__", return_value=None)
    @patch("datus.agent.node.node_factory._resolve_node_class_type", return_value="gen_dashboard")
    def test_config_driven_gen_dashboard(self, mock_resolve, mock_init):
        config = _mock_agent_config()
        create_interactive_node("sales_dashboard", config, node_id_suffix="_cli", scope="team-a")
        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["node_id"] == "sales_dashboard_cli"
        assert call_kwargs["node_name"] == "sales_dashboard"
        assert call_kwargs["execution_mode"] == "interactive"
        assert call_kwargs["scope"] == "team-a"

    @patch("datus.agent.node.scheduler_agentic_node.SchedulerAgenticNode.__init__", return_value=None)
    @patch("datus.agent.node.node_factory._resolve_node_class_type", return_value="scheduler")
    def test_config_driven_scheduler(self, mock_resolve, mock_init):
        config = _mock_agent_config()
        create_interactive_node("etl_scheduler", config, node_id_suffix="_cli", scope="team-a")
        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["node_id"] == "etl_scheduler_cli"
        assert call_kwargs["node_name"] == "etl_scheduler"
        assert call_kwargs["execution_mode"] == "interactive"
        assert call_kwargs["scope"] == "team-a"


# ---------------------------------------------------------------------------
# Tests: create_node_input
# ---------------------------------------------------------------------------


def _load_node_class(module_path, class_name):
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class TestCreateNodeInput:
    @pytest.mark.parametrize(
        "node_module,node_class_name,message,kwargs,expected_attrs",
        [
            (
                "datus.agent.node.chat_agentic_node",
                "ChatAgenticNode",
                "hello",
                {"catalog": "cat", "database": "db"},
                {"user_message": "hello", "catalog": "cat"},
            ),
            (
                "datus.agent.node.gen_sql_agentic_node",
                "GenSQLAgenticNode",
                "generate SQL",
                {"catalog": "cat", "plan_mode": True},
                {"user_message": "generate SQL", "plan_mode": True},
            ),
            (
                "datus.agent.node.gen_semantic_model_agentic_node",
                "GenSemanticModelAgenticNode",
                "build model",
                {"catalog": "cat", "prompt_language": "zh"},
                {"user_message": "build model", "prompt_language": "zh"},
            ),
            (
                "datus.agent.node.gen_metrics_agentic_node",
                "GenMetricsAgenticNode",
                "gen metrics",
                {},
                {"user_message": "gen metrics"},
            ),
            (
                "datus.agent.node.sql_summary_agentic_node",
                "SqlSummaryAgenticNode",
                "summarize",
                {"database": "mydb"},
                {"user_message": "summarize", "database": "mydb"},
            ),
            (
                "datus.agent.node.gen_ext_knowledge_agentic_node",
                "GenExtKnowledgeAgenticNode",
                "add knowledge",
                {},
                {"user_message": "add knowledge", "catalog": None, "database": None, "db_schema": None},
            ),
            (
                "datus.agent.node.gen_ext_knowledge_agentic_node",
                "GenExtKnowledgeAgenticNode",
                "add knowledge with context",
                {"catalog": "cat", "database": "db", "db_schema": "sch"},
                {"user_message": "add knowledge with context", "catalog": "cat", "database": "db", "db_schema": "sch"},
            ),
            (
                "datus.agent.node.gen_table_agentic_node",
                "GenTableAgenticNode",
                "create table",
                {"catalog": "cat", "database": "db"},
                {"user_message": "create table", "catalog": "cat", "database": "db"},
            ),
            (
                "datus.agent.node.gen_report_agentic_node",
                "GenReportAgenticNode",
                "report",
                {"catalog": "cat", "database": "db"},
                {"user_message": "report", "catalog": "cat", "database": "db"},
            ),
        ],
    )
    def test_create_node_input(self, node_module, node_class_name, message, kwargs, expected_attrs):
        node_class = _load_node_class(node_module, node_class_name)
        node = MagicMock(spec=node_class)
        result = create_node_input(message, node, **kwargs)
        for attr, expected_value in expected_attrs.items():
            assert getattr(result, attr) == expected_value, (
                f"{node_class_name}: expected {attr}={expected_value!r}, got {getattr(result, attr)!r}"
            )
