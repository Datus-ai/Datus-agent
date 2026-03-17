# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Additional unit tests for GenReportAgenticNode covering uncovered helper methods.

CI-level: zero external deps, zero network, zero API keys.
"""

import json
from unittest.mock import patch

import pytest

from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistoryManager, ActionStatus
from datus.schemas.gen_report_agentic_node_models import GenReportNodeInput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(real_agent_config, mock_llm_create, node_name="gen_report"):
    from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode

    return GenReportAgenticNode(
        node_id="report_extra",
        description="Extra report test",
        node_type=NodeType.TYPE_GEN_REPORT,
        agent_config=real_agent_config,
        node_name=node_name,
    )


# ---------------------------------------------------------------------------
# TestExtractReportFromResponse
# ---------------------------------------------------------------------------


class TestExtractReportFromResponse:
    def test_extracts_from_dict_with_report_key(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {
            "content": {
                "report": "## Revenue Analysis\n\nRevenue grew 15%.",
                "data_sources": ["revenue_total"],
                "key_findings": ["15% growth"],
            }
        }
        report, metadata = node._extract_report_from_response(output)
        assert "Revenue Analysis" in report
        assert metadata["data_sources"] == ["revenue_total"]
        assert metadata["key_findings"] == ["15% growth"]

    def test_extracts_from_json_string(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        content = json.dumps(
            {
                "report": "## Sales Report\n\nSales data analysis.",
                "data_sources": [],
                "key_findings": [],
            }
        )
        output = {"content": content}
        report, metadata = node._extract_report_from_response(output)
        assert "Sales Report" in report

    def test_returns_empty_string_on_empty_content(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {"content": ""}
        report, metadata = node._extract_report_from_response(output)
        assert report == ""
        assert metadata is None

    def test_returns_raw_string_when_no_json_report_key(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {"content": "plain text response without JSON"}
        report, metadata = node._extract_report_from_response(output)
        assert "plain text response" in report

    def test_returns_dict_as_string_on_missing_report_key(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {"content": {"some_key": "some_value"}}
        report, metadata = node._extract_report_from_response(output)
        # dict without 'report' key should return string representation
        assert isinstance(report, str)

    def test_checks_raw_output_field(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        output = {
            "content": "",
            "raw_output": "Response from raw output field",
        }
        report, metadata = node._extract_report_from_response(output)
        # raw_output fallback should be used when content is empty
        assert isinstance(report, str)


# ---------------------------------------------------------------------------
# TestBuildEnhancedMessage
# ---------------------------------------------------------------------------


class TestBuildEnhancedMessage:
    def test_no_context_returns_user_message(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        user_input = GenReportNodeInput(user_message="Analyze revenue")
        result = node._build_enhanced_message(user_input)
        assert result == "Analyze revenue"

    def test_with_database_context_builds_structured_message(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        user_input = GenReportNodeInput(
            user_message="Analyze revenue",
            database="california_schools",
        )
        result = node._build_enhanced_message(user_input)
        assert "california_schools" in result
        assert "Analyze revenue" in result

    def test_with_schema_context(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        user_input = GenReportNodeInput(
            user_message="Analyze data",
            db_schema="main_schema",
        )
        result = node._build_enhanced_message(user_input)
        assert "main_schema" in result


# ---------------------------------------------------------------------------
# TestExtractReportResult
# ---------------------------------------------------------------------------


class TestExtractReportResult:
    def test_base_implementation_returns_none(self, real_agent_config, mock_llm_create):
        """Base _extract_report_result always returns None."""
        node = _make_node(real_agent_config, mock_llm_create)
        result = node._extract_report_result([])
        assert result is None


# ---------------------------------------------------------------------------
# TestSetupToolPattern
# ---------------------------------------------------------------------------


class TestSetupToolPattern:
    def test_unknown_pattern_logs_warning(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        # Should not raise
        node._setup_tool_pattern("unknown_tool_type.*")

    def test_specific_method_setup(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        # Setup semantic_tools.search_metrics specifically
        with patch.object(node, "_setup_specific_tool_method") as mock_method:
            node._setup_tool_pattern("semantic_tools.search_metrics")
        mock_method.assert_called_once_with("semantic_tools", "search_metrics")

    def test_exact_db_tools_pattern(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        with patch.object(node, "_setup_db_tools") as mock_setup:
            node._setup_tool_pattern("db_tools")
        mock_setup.assert_called_once()

    def test_wildcard_context_search_tools(self, real_agent_config, mock_llm_create):
        node = _make_node(real_agent_config, mock_llm_create)
        with patch.object(node, "_setup_context_search_tools") as mock_setup:
            node._setup_tool_pattern("context_search_tools.*")
        mock_setup.assert_called_once()


# ---------------------------------------------------------------------------
# TestExecuteStreamError
# ---------------------------------------------------------------------------


class TestExecuteStreamGenReportError:
    @pytest.mark.asyncio
    async def test_execute_stream_error_yields_error_action(self, real_agent_config, mock_llm_create):
        """When model raises a generic exception, execute_stream yields error action."""
        from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode

        async def _raise_error(*args, **kwargs):
            raise RuntimeError("LLM error")
            yield  # noqa

        node = GenReportAgenticNode(
            node_id="report_error",
            description="Error test",
            node_type=NodeType.TYPE_GEN_REPORT,
            agent_config=real_agent_config,
            node_name="gen_report",
        )
        node.input = GenReportNodeInput(user_message="Analyze data")
        mock_llm_create.generate_with_tools_stream = _raise_error

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert len(actions) >= 2
        last = actions[-1]
        assert last.status == ActionStatus.FAILED
        assert last.action_type == "error"
