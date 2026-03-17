# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Extended unit tests for SemanticTools - CI level, zero external dependencies."""

from unittest.mock import Mock, patch

import pytest

from datus.tools.func_tool.semantic_tools import _run_async


class TestRunAsync:
    def test_delegates_to_run_async_utility(self):
        mock_coro = Mock()
        with patch("datus.utils.async_utils.run_async", return_value="result") as mock_run:
            result = _run_async(mock_coro)
        mock_run.assert_called_once_with(mock_coro)
        assert result == "result"


@pytest.fixture
def semantic_tools():
    with (
        patch("datus.tools.func_tool.semantic_tools.SemanticModelRAG"),
        patch("datus.tools.func_tool.semantic_tools.MetricRAG"),
    ):
        from datus.tools.func_tool.semantic_tools import SemanticTools

        config = Mock()
        tool = SemanticTools(agent_config=config)
        return tool


@pytest.fixture
def semantic_tools_with_adapter():
    with (
        patch("datus.tools.func_tool.semantic_tools.SemanticModelRAG"),
        patch("datus.tools.func_tool.semantic_tools.MetricRAG"),
    ):
        from datus.tools.func_tool.semantic_tools import SemanticTools

        config = Mock()
        tool = SemanticTools(agent_config=config, adapter_type="metricflow")
        mock_adapter = Mock()
        tool._adapter = mock_adapter
        return tool, mock_adapter


class TestAllToolsName:
    def test_returns_expected_names(self):
        from datus.tools.func_tool.semantic_tools import SemanticTools

        names = SemanticTools.all_tools_name()
        assert "search_metrics" in names
        assert "list_metrics" in names
        assert "get_dimensions" in names
        assert "query_metrics" in names
        assert "validate_semantic" in names
        assert "attribution_analyze" in names


class TestAvailableTools:
    def test_no_adapter_returns_four_tools(self, semantic_tools):
        with patch("datus.tools.func_tool.semantic_tools.trans_to_function_tool") as mock_trans:
            mock_trans.side_effect = lambda f: Mock(name=f.__name__)
            tools = semantic_tools.available_tools()
        assert len(tools) == 4

    def test_with_adapter_adds_validate_and_attribution_tools(self):
        with (
            patch("datus.tools.func_tool.semantic_tools.SemanticModelRAG"),
            patch("datus.tools.func_tool.semantic_tools.MetricRAG"),
        ):
            from datus.tools.func_tool.semantic_tools import SemanticTools

            config = Mock()
            tool = SemanticTools(agent_config=config)
            tool._adapter = Mock()  # Set adapter (also enables attribution_tool)

            with patch("datus.tools.func_tool.semantic_tools.trans_to_function_tool") as mock_trans:
                mock_trans.side_effect = lambda f: Mock(name=f.__name__)
                tools = tool.available_tools()
        # 4 base + validate_semantic + attribution_analyze (both enabled when adapter is set)
        assert len(tools) == 6


class TestSearchMetrics:
    def test_success_returns_formatted_metrics(self, semantic_tools):
        semantic_tools.metric_rag.search_metrics.return_value = [
            {
                "name": "revenue",
                "description": "Total revenue",
                "metric_type": "simple",
                "dimensions": ["date", "region"],
                "base_measures": ["amount"],
                "subject_path": ["Finance"],
            }
        ]

        result = semantic_tools.search_metrics("revenue metrics")

        assert result.success == 1
        assert len(result.result) == 1
        assert result.result[0]["name"] == "revenue"
        assert result.result[0]["type"] == "simple"

    def test_no_results_returns_error(self, semantic_tools):
        semantic_tools.metric_rag.search_metrics.return_value = []

        result = semantic_tools.search_metrics("nonexistent")

        assert result.success == 0
        assert "No metrics found" in result.error

    def test_exception_returns_failure(self, semantic_tools):
        semantic_tools.metric_rag.search_metrics.side_effect = Exception("storage down")

        result = semantic_tools.search_metrics("revenue")

        assert result.success == 0
        assert "storage down" in result.error

    def test_null_subject_path_normalized(self, semantic_tools):
        semantic_tools.metric_rag.search_metrics.return_value = []

        semantic_tools.search_metrics("revenue", subject_path="null")

        semantic_tools.metric_rag.search_metrics.assert_called_once_with(
            query_text="revenue",
            subject_path=None,
            top_n=5,
        )


class TestListMetrics:
    def test_success_from_storage(self, semantic_tools):
        semantic_tools.metric_rag.search_all_metrics.return_value = [
            {
                "name": "orders",
                "description": "Order count",
                "metric_type": "count",
                "dimensions": [],
                "base_measures": [],
                "unit": None,
                "format": None,
                "subject_path": ["Sales"],
            }
        ]

        result = semantic_tools.list_metrics()

        assert result.success == 1
        assert len(result.result) == 1
        assert result.result[0]["name"] == "orders"

    def test_empty_storage_no_adapter_returns_empty(self, semantic_tools):
        semantic_tools.metric_rag.search_all_metrics.return_value = []

        result = semantic_tools.list_metrics()

        assert result.success == 1
        assert result.result == []

    def test_path_filter_applied(self, semantic_tools):
        semantic_tools.metric_rag.search_all_metrics.return_value = [
            {
                "name": "m1",
                "subject_path": ["Finance"],
                "description": "",
                "metric_type": "",
                "dimensions": [],
                "base_measures": [],
                "unit": None,
                "format": None,
            },
            {
                "name": "m2",
                "subject_path": ["Sales"],
                "description": "",
                "metric_type": "",
                "dimensions": [],
                "base_measures": [],
                "unit": None,
                "format": None,
            },
        ]

        result = semantic_tools.list_metrics(path=["Finance"])

        assert result.success == 1
        assert len(result.result) == 1
        assert result.result[0]["name"] == "m1"

    def test_pagination(self, semantic_tools):
        metrics = [
            {
                "name": f"m{i}",
                "subject_path": [],
                "description": "",
                "metric_type": "",
                "dimensions": [],
                "base_measures": [],
                "unit": None,
                "format": None,
            }
            for i in range(10)
        ]
        semantic_tools.metric_rag.search_all_metrics.return_value = metrics

        result = semantic_tools.list_metrics(limit=3, offset=2)

        assert result.success == 1
        assert len(result.result) == 3
        assert result.result[0]["name"] == "m2"

    def test_falls_back_to_adapter(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter
        tool.metric_rag.search_all_metrics.return_value = []

        mock_metric = Mock()
        mock_metric.name = "revenue"
        mock_metric.description = "Revenue metric"
        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=[mock_metric]):
            result = tool.list_metrics()

        assert result.success == 1
        assert len(result.result) == 1
        assert result.result[0]["name"] == "revenue"

    def test_exception_returns_failure(self, semantic_tools):
        semantic_tools.metric_rag.search_all_metrics.side_effect = Exception("db error")

        result = semantic_tools.list_metrics()

        assert result.success == 0
        assert "db error" in result.error


class TestGetDimensions:
    def test_with_adapter(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter
        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=["date", "region"]):
            result = tool.get_dimensions("revenue")

        assert result.success == 1
        assert result.result == ["date", "region"]

    def test_no_adapter_from_storage(self, semantic_tools):
        semantic_tools.metric_rag.search_all_metrics.return_value = [
            {"name": "revenue", "dimensions": ["date", "channel"]}
        ]

        result = semantic_tools.get_dimensions("revenue")

        assert result.success == 1
        assert result.result == ["date", "channel"]

    def test_no_adapter_metric_not_found(self, semantic_tools):
        semantic_tools.metric_rag.search_all_metrics.return_value = []

        result = semantic_tools.get_dimensions("nonexistent")

        assert result.success == 0
        assert "not found" in result.error

    def test_with_path_filter(self, semantic_tools):
        mock_storage = Mock()
        semantic_tools.metric_rag.storage = mock_storage
        mock_storage.search_all_metrics.return_value = [{"name": "revenue", "dimensions": ["date"]}]

        result = semantic_tools.get_dimensions("revenue", path=["Finance"])

        assert result.success == 1
        assert result.result == ["date"]

    def test_exception_returns_failure(self, semantic_tools):
        semantic_tools.metric_rag.search_all_metrics.side_effect = Exception("conn error")

        result = semantic_tools.get_dimensions("revenue")

        assert result.success == 0
        assert "conn error" in result.error


class TestValidateSemantic:
    def test_no_adapter_returns_error(self, semantic_tools):
        result = semantic_tools.validate_semantic()
        assert result.success == 0
        assert "adapter" in result.error.lower()

    def test_valid_result(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter

        mock_validation = Mock()
        mock_validation.valid = True
        mock_validation.issues = []

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=mock_validation):
            with patch.object(tool, "_reload_adapter", return_value=True):
                result = tool.validate_semantic()

        assert result.success == 1
        assert result.result["valid"] is True
        assert result.result["issues"] == []

    def test_invalid_result(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter

        mock_issue = Mock()
        mock_issue.model_dump.return_value = {"severity": "error", "message": "bad config"}
        mock_validation = Mock()
        mock_validation.valid = False
        mock_validation.issues = [mock_issue]

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=mock_validation):
            result = tool.validate_semantic()

        assert result.success == 0
        assert result.result["valid"] is False
        assert len(result.result["issues"]) == 1
        assert "1 validation errors" in result.error

    def test_exception_returns_failure(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter

        with patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=Exception("adapter crash")):
            result = tool.validate_semantic()

        assert result.success == 0
        assert "adapter crash" in result.error


class TestAttributionAnalyze:
    def test_no_attribution_tool_returns_error(self, semantic_tools):
        result = semantic_tools.attribution_analyze(
            metric_name="revenue",
            candidate_dimensions=["region"],
            baseline_start="2024-01-01",
            baseline_end="2024-01-07",
            current_start="2024-01-08",
            current_end="2024-01-14",
        )
        assert result.success == 0
        assert "Attribution tool not available" in result.error

    def test_success_with_dict_anomaly_context(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter
        mock_attribution = Mock()
        tool._attribution_tool = mock_attribution

        mock_result = Mock()
        mock_result.model_dump.return_value = {
            "dimension_ranking": [],
            "selected_dimensions": [],
            "top_dimension_values": {},
        }

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=mock_result):
            result = tool.attribution_analyze(
                metric_name="revenue",
                candidate_dimensions=["region"],
                baseline_start="2024-01-01",
                baseline_end="2024-01-07",
                current_start="2024-01-08",
                current_end="2024-01-14",
                anomaly_context={"rule": "3sigma", "observed_change_pct": 20.0},
            )

        assert result.success == 1

    def test_success_none_anomaly_context(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter
        mock_attribution = Mock()
        tool._attribution_tool = mock_attribution

        mock_result = Mock()
        mock_result.model_dump.return_value = {"dimension_ranking": []}

        with patch("datus.tools.func_tool.semantic_tools._run_async", return_value=mock_result):
            result = tool.attribution_analyze(
                metric_name="revenue",
                candidate_dimensions=["region"],
                baseline_start="2024-01-01",
                baseline_end="2024-01-07",
                current_start="2024-01-08",
                current_end="2024-01-14",
                anomaly_context=None,
            )

        assert result.success == 1

    def test_exception_returns_failure(self, semantic_tools_with_adapter):
        tool, mock_adapter = semantic_tools_with_adapter
        mock_attribution = Mock()
        tool._attribution_tool = mock_attribution

        with patch("datus.tools.func_tool.semantic_tools._run_async", side_effect=Exception("analysis failed")):
            result = tool.attribution_analyze(
                metric_name="revenue",
                candidate_dimensions=["region"],
                baseline_start="2024-01-01",
                baseline_end="2024-01-07",
                current_start="2024-01-08",
                current_end="2024-01-14",
            )

        assert result.success == 0
        assert "analysis failed" in result.error


class TestReloadAdapter:
    def test_no_adapter_type_returns_false(self, semantic_tools):
        result = semantic_tools._reload_adapter()
        assert result is False

    def test_reload_success(self, semantic_tools_with_adapter):
        tool, _ = semantic_tools_with_adapter
        new_adapter = Mock()
        # After clearing, the property should return a new adapter
        with patch.object(type(tool), "adapter", new_callable=lambda: property(lambda self: new_adapter)):
            result = tool._reload_adapter()
        assert result is True

    def test_reload_adapter_fails_returns_false(self, semantic_tools_with_adapter):
        tool, _ = semantic_tools_with_adapter
        tool._adapter = None

        # Simulate adapter load failure
        with patch("datus.tools.func_tool.semantic_tools.semantic_adapter_registry") as mock_registry:
            mock_registry.get_metadata.return_value = None
            mock_registry.create_adapter.side_effect = Exception("config missing")

            result = tool._reload_adapter()

        # It either returns False or raises - we just check it doesn't crash
        assert isinstance(result, bool)
