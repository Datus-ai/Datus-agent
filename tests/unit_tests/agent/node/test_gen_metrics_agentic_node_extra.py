# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Additional unit tests for GenMetricsAgenticNode covering uncovered helper methods.

CI-level: zero external deps, zero network, zero API keys.
"""

import json
from unittest.mock import MagicMock

import pytest

from datus.schemas.action_history import ActionHistoryManager, ActionStatus
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(real_agent_config, mock_llm_create, execution_mode="workflow"):
    from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

    return GenMetricsAgenticNode(
        agent_config=real_agent_config,
        execution_mode=execution_mode,
    )


# ---------------------------------------------------------------------------
# TestExtractMetricAndOutputFromResponse
# ---------------------------------------------------------------------------


class TestExtractMetricAndOutputFromResponse:
    def test_extracts_from_dict_content(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {
            "content": {
                "semantic_model_file": "model.yml",
                "metric_file": "revenue_metrics.yml",
                "output": "Generated successfully",
            }
        }
        sem_model, metric_file, out = node._extract_metric_and_output_from_response(output)
        assert metric_file == "revenue_metrics.yml"
        assert sem_model == "model.yml"
        assert out == "Generated successfully"

    def test_extracts_from_json_string(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        content = json.dumps(
            {
                "semantic_model_file": "model.yml",
                "metric_file": "sales_metrics.yml",
                "output": "Done",
            }
        )
        output = {"content": content}
        sem_model, metric_file, out = node._extract_metric_and_output_from_response(output)
        assert metric_file == "sales_metrics.yml"
        assert out == "Done"

    def test_returns_none_triple_on_empty_content(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {"content": ""}
        sem_model, metric_file, out = node._extract_metric_and_output_from_response(output)
        assert metric_file is None
        assert sem_model is None
        assert out is None

    def test_returns_none_triple_on_dict_missing_metric_file(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {"content": {"some_key": "some_value"}}
        sem_model, metric_file, out = node._extract_metric_and_output_from_response(output)
        assert metric_file is None

    def test_returns_none_triple_on_invalid_json(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {"content": "not json at all !!!"}
        sem_model, metric_file, out = node._extract_metric_and_output_from_response(output)
        assert metric_file is None


# ---------------------------------------------------------------------------
# TestExtractMetricSqlsFromActions
# ---------------------------------------------------------------------------


class TestExtractMetricSqlsFromActions:
    def test_extracts_metric_sqls(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        from datus.schemas.action_history import ActionHistory

        action = MagicMock(spec=ActionHistory)
        action.role = "tool"
        action.output = {"raw_output": {"result": {"metric_sqls": {"revenue": "SELECT SUM(revenue) FROM sales"}}}}

        manager = MagicMock()
        manager.get_actions.return_value = [action]

        sqls = node._extract_metric_sqls_from_actions(manager)
        assert "revenue" in sqls
        assert sqls["revenue"] == "SELECT SUM(revenue) FROM sales"

    def test_returns_empty_when_no_tool_actions(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        manager = MagicMock()
        manager.get_actions.return_value = []

        sqls = node._extract_metric_sqls_from_actions(manager)
        assert sqls == {}

    def test_returns_empty_on_exception(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        manager = MagicMock()
        manager.get_actions.side_effect = RuntimeError("error")

        sqls = node._extract_metric_sqls_from_actions(manager)
        assert sqls == {}


# ---------------------------------------------------------------------------
# TestPrepareTemplateContext
# ---------------------------------------------------------------------------


class TestPrepareTemplateContext:
    def test_prepare_template_context_no_subject_tree(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.subject_tree = None

        # Mock the storage to return empty subject trees
        node.metrics_rag = MagicMock()
        node.metrics_rag.storage = MagicMock()
        node.metrics_rag.storage.get_subject_tree_flat.return_value = []

        user_input = SemanticNodeInput(user_message="Generate metrics")
        context = node._prepare_template_context(user_input)

        assert "semantic_model_dir" in context
        assert context["has_subject_tree"] is False
        assert "existing_subject_trees" in context

    def test_prepare_template_context_with_predefined_subject_tree(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.subject_tree = ["Finance", "Revenue"]

        user_input = SemanticNodeInput(user_message="Generate metrics")
        context = node._prepare_template_context(user_input)

        assert context["has_subject_tree"] is True
        assert context["subject_tree"] == ["Finance", "Revenue"]

    def test_prepare_template_context_includes_tools(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.subject_tree = None
        node.metrics_rag = MagicMock()
        node.metrics_rag.storage.get_subject_tree_flat.return_value = []

        user_input = SemanticNodeInput(user_message="Generate metrics")
        context = node._prepare_template_context(user_input)

        assert "native_tools" in context
        assert "mcp_tools" in context


# ---------------------------------------------------------------------------
# TestGetExistingSubjectTrees
# ---------------------------------------------------------------------------


class TestGetExistingSubjectTrees:
    def test_returns_subject_paths(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        mock_storage = MagicMock()
        mock_storage.get_subject_tree_flat.return_value = ["Finance/Revenue", "Sales/Quarterly"]
        node.metrics_rag = MagicMock()
        node.metrics_rag.storage = mock_storage

        result = node._get_existing_subject_trees()
        assert result == ["Finance/Revenue", "Sales/Quarterly"]

    def test_returns_empty_when_no_storage(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.metrics_rag = MagicMock()
        node.metrics_rag.storage = None

        result = node._get_existing_subject_trees()
        assert result == []

    def test_returns_empty_on_exception(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        node.metrics_rag = MagicMock()
        node.metrics_rag.storage = MagicMock()
        node.metrics_rag.storage.get_subject_tree_flat.side_effect = RuntimeError("storage error")

        result = node._get_existing_subject_trees()
        assert result == []


# ---------------------------------------------------------------------------
# TestGetNodeName
# ---------------------------------------------------------------------------


class TestGetNodeNameGenMetrics:
    def test_get_node_name(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        assert node.get_node_name() == "gen_metrics"


# ---------------------------------------------------------------------------
# TestExecuteStreamError
# ---------------------------------------------------------------------------


class TestExecuteStreamGenMetricsError:
    @pytest.mark.asyncio
    async def test_execute_stream_error_yields_error_action(self, real_agent_config, mock_llm_create):
        """When model raises a generic exception, execute_stream yields error action."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        async def _raise_error(*args, **kwargs):
            raise RuntimeError("LLM error")
            yield  # noqa

        node = GenMetricsAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
        )
        node.input = SemanticNodeInput(user_message="Generate metrics")
        mock_llm_create.generate_with_tools_stream = _raise_error

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        # Should have initial USER action + error action
        assert len(actions) >= 2
        last = actions[-1]
        assert last.status == ActionStatus.FAILED
        assert last.action_type == "error"
