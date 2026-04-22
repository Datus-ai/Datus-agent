# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ReflectNode - zero external dependencies."""

from unittest.mock import MagicMock, patch

import pytest

from datus.agent.node import Node
from datus.configuration.node_type import NodeType
from datus.schemas.node_models import ReflectionInput, ReflectionResult, SQLContext, SqlTask


def make_agent_config():
    cfg = MagicMock()
    cfg.datasource_configs = {}
    cfg.current_datasource = "test"
    cfg.nodes = {}
    return cfg


def make_reflection_input():
    task = SqlTask(task="Count rows", database_type="sqlite", database_name="db")
    ctx = SQLContext(sql_query="SELECT COUNT(*) FROM t", sql_error="", row_count=0)
    return ReflectionInput(task_description=task, sql_context=[ctx])


def make_node(input_data=None):
    cfg = make_agent_config()
    return Node.new_instance(
        "reflect_1",
        "Reflect",
        NodeType.TYPE_REFLECT,
        input_data=input_data,
        agent_config=cfg,
    )


class TestReflectNodeExecute:
    """Test execute() / _execute_reflect()."""

    def test_execute_no_model_raises(self):
        node = make_node(make_reflection_input())
        node.model = None
        with pytest.raises(ValueError, match="Model is required"):
            node.execute()

    def test_execute_empty_sql_context_returns_failure(self):
        task = SqlTask(task="no ctx", database_type="sqlite", database_name="db")
        empty_input = ReflectionInput(task_description=task, sql_context=[])
        node = make_node(empty_input)
        node.model = MagicMock()
        node.execute()
        assert node.result.success is False
        assert "No SQL context" in node.result.error

    def test_execute_calls_evaluate_with_model(self):
        node = make_node(make_reflection_input())
        node.model = MagicMock()
        eval_return = {
            "success": True,
            "strategy": "SUCCESS",
            "details": {"explanation": "looks good"},
            "error": "",
        }
        with patch("datus.agent.node.reflect_node.evaluate_with_model", return_value=eval_return):
            node.execute()

        assert node.result.success is True
        assert node.result.strategy == "SUCCESS"

    def test_execute_handles_evaluate_failure(self):
        node = make_node(make_reflection_input())
        node.model = MagicMock()
        eval_return = {
            "success": False,
            "strategy": "SCHEMA_LINKING",
            "details": {},
            "error": "SQL error",
        }
        with patch("datus.agent.node.reflect_node.evaluate_with_model", return_value=eval_return):
            node.execute()

        assert node.result.strategy == "SCHEMA_LINKING"


class TestReflectNodeSetupInput:
    """Test setup_input."""

    def test_setup_input_builds_reflection_input(self):
        node = make_node()
        workflow = MagicMock()
        workflow.task = SqlTask(task="Count rows", database_type="sqlite", database_name="db")
        sql_ctx = SQLContext(sql_query="SELECT 1", sql_error="")
        workflow.context.sql_contexts = [sql_ctx]

        result = node.setup_input(workflow)
        assert result["success"] is True
        assert isinstance(node.input, ReflectionInput)
        assert node.input.task_description.task == "Count rows"


class TestReflectNodeUpdateContext:
    """Test update_context."""

    def test_update_context_success_strategy(self):
        node = make_node(make_reflection_input())
        node.result = ReflectionResult(success=True, strategy="SUCCESS", details={"explanation": "all good"})
        workflow = MagicMock()
        workflow.reflection_round = 0
        sql_ctx = SQLContext(sql_query="SELECT COUNT(*) FROM t")
        workflow.context.sql_contexts = [sql_ctx]

        with patch.object(node, "_execute_reflection_strategy", return_value={"success": True, "message": "done"}):
            result = node.update_context(workflow)

        assert result["success"] is True
        assert workflow.reflection_round == 1

    def test_update_context_sql_query_mismatch(self):
        node = make_node(make_reflection_input())
        node.result = ReflectionResult(success=True, strategy="SUCCESS", details={"explanation": "all good"})
        workflow = MagicMock()
        workflow.reflection_round = 0
        # The last sql context doesn't match input's last context
        workflow.context.sql_contexts = [SQLContext(sql_query="SELECT DIFFERENT")]

        result = node.update_context(workflow)
        assert result["success"] is False
        assert "mismatch" in result["message"]

    def test_update_context_keywords_set_doc_search(self):
        node = make_node(make_reflection_input())
        node.result = ReflectionResult(
            success=True, strategy="DOC_SEARCH", details={"keywords": ["revenue", "order"], "explanation": "need docs"}
        )
        workflow = MagicMock()
        workflow.reflection_round = 0
        workflow.context.sql_contexts = [SQLContext(sql_query="SELECT COUNT(*) FROM t")]

        with patch.object(node, "_execute_reflection_strategy", return_value={"success": True}):
            node.update_context(workflow)

        assert workflow.context.doc_search_keywords == ["revenue", "order"]

    def test_update_context_exception(self):
        node = make_node(make_reflection_input())
        node.result = ReflectionResult(success=True, strategy="SUCCESS", details={})
        workflow = MagicMock()
        workflow.reflection_round = MagicMock()
        workflow.reflection_round.__iadd__ = MagicMock(side_effect=RuntimeError("ctx error"))

        result = node.update_context(workflow)
        assert result["success"] is False


class TestReflectNodeExecuteStrategy:
    """Test _execute_reflection_strategy and _execute_strategy."""

    def test_strategy_success_returns_immediately(self):
        node = make_node(make_reflection_input())
        node.result = ReflectionResult(success=True, strategy="SUCCESS", details={})
        workflow = MagicMock()
        workflow.reflection_round = 1

        result = node._execute_reflection_strategy("SUCCESS", {}, workflow)
        assert result["success"] is True

    def test_strategy_max_round_triggers_reasoning(self):
        node = make_node(make_reflection_input())
        node.result = ReflectionResult(success=True, strategy="DOC_SEARCH", details={})
        workflow = MagicMock()
        workflow.reflection_round = 3  # equals max (default 3)
        workflow.current_node_index = 0
        workflow.tools = []
        workflow._global_config = MagicMock()
        workflow._global_config.reflection_nodes.return_value = ["generate_sql", "execute_sql"]
        workflow.add_node = MagicMock()
        node.agent_config = make_agent_config()

        with patch("datus.agent.node.reflect_node.get_env_int", return_value=3):
            with patch.object(Node, "new_instance") as mock_ni:
                mock_ni.return_value = MagicMock()
                result = node._execute_reflection_strategy("DOC_SEARCH", {}, workflow)

        assert result["success"] is True

    def test_strategy_exceeded_max_round_exits(self):
        node = make_node(make_reflection_input())
        node.result = ReflectionResult(success=True, strategy="SCHEMA_LINKING", details={})
        workflow = MagicMock()
        workflow.reflection_round = 5  # > max (3)

        with patch("datus.agent.node.reflect_node.get_env_int", return_value=3):
            result = node._execute_reflection_strategy("SCHEMA_LINKING", {}, workflow)

        assert result["success"] is True
        assert "exceeded" in result["message"]

    def test_unknown_strategy_returns_failure(self):
        node = make_node(make_reflection_input())
        node.result = ReflectionResult(success=True, strategy="UNKNOWN", details={})
        workflow = MagicMock()
        workflow.reflection_round = 1

        with patch("datus.agent.node.reflect_node.get_env_int", return_value=3):
            result = node._execute_reflection_strategy("BOGUS_STRATEGY", {}, workflow)

        assert result["success"] is False
        assert "Unknown strategy" in result["message"]

    def test_execute_strategy_simple_regenerate_with_sql(self):
        node = make_node(make_reflection_input())
        node.result = ReflectionResult(
            success=True, strategy="SIMPLE_REGENERATE", details={"sql": "SELECT 1", "explanation": "regen"}
        )
        workflow = MagicMock()
        workflow.context.sql_contexts = []
        workflow.current_node_index = 0
        workflow.tools = []
        workflow._global_config = MagicMock()
        workflow._global_config.reflection_nodes.return_value = []
        workflow.add_node = MagicMock()
        node.agent_config = make_agent_config()

        with patch.object(Node, "new_instance") as mock_ni:
            mock_ni.return_value = MagicMock()
            result = node._execute_strategy({"sql": "SELECT 1"}, workflow, "SIMPLE_REGENERATE")

        assert result["success"] is True


class TestReflectNodeStream:
    """Test execute_stream / _reflect_stream."""

    @pytest.mark.asyncio
    async def test_execute_stream_no_model_yields_nothing(self):
        node = make_node(make_reflection_input())
        node.model = None
        actions = []
        async for action in node.execute_stream():
            actions.append(action)
        assert actions == []

    @pytest.mark.asyncio
    async def test_execute_stream_yields_actions(self):
        node = make_node(make_reflection_input())
        node.model = MagicMock()
        eval_return = {"success": True, "strategy": "SUCCESS", "details": {}, "error": ""}
        with patch("datus.agent.node.reflect_node.evaluate_with_model", return_value=eval_return):
            actions = []
            async for action in node.execute_stream():
                actions.append(action)

        assert len(actions) >= 2  # PROCESSING + final
