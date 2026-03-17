# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Unit tests for GenerationTools - CI level, zero external dependencies."""

import json
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def mock_agent_config():
    return Mock()


@pytest.fixture
def generation_tools(mock_agent_config):
    with (
        patch("datus.tools.func_tool.generation_tools.MetricRAG") as mock_metric_rag_cls,
        patch("datus.tools.func_tool.generation_tools.SemanticModelRAG") as mock_semantic_rag_cls,
    ):
        mock_metric_rag = Mock()
        mock_semantic_rag = Mock()
        mock_metric_rag_cls.return_value = mock_metric_rag
        mock_semantic_rag_cls.return_value = mock_semantic_rag

        from datus.tools.func_tool.generation_tools import GenerationTools

        tool = GenerationTools(agent_config=mock_agent_config)
        tool.metric_rag = mock_metric_rag
        tool.semantic_rag = mock_semantic_rag
        return tool


class TestAvailableTools:
    def test_returns_four_tools(self, generation_tools):
        with patch("datus.tools.func_tool.generation_tools.trans_to_function_tool") as mock_trans:
            mock_trans.side_effect = lambda f: Mock(name=f.__name__)
            tools = generation_tools.available_tools()
        assert len(tools) == 4


class TestCheckSemanticObjectExists:
    def test_table_found(self, generation_tools):
        mock_storage = Mock()
        generation_tools.semantic_rag.storage = mock_storage
        mock_storage.search_all.return_value = [{"id": "t1", "name": "orders", "kind": "table"}]

        with patch("datus.tools.func_tool.generation_tools.build_where"), patch(
            "datus.tools.func_tool.generation_tools.And"
        ), patch("datus.tools.func_tool.generation_tools.eq"):
            result = generation_tools.check_semantic_object_exists("orders", kind="table")

        assert result.success == 1
        assert result.result["exists"] is True
        assert result.result["name"] == "orders"

    def test_table_not_found(self, generation_tools):
        mock_storage = Mock()
        generation_tools.semantic_rag.storage = mock_storage
        mock_storage.search_all.return_value = []

        with patch("datus.tools.func_tool.generation_tools.build_where"), patch(
            "datus.tools.func_tool.generation_tools.And"
        ), patch("datus.tools.func_tool.generation_tools.eq"):
            result = generation_tools.check_semantic_object_exists("unknown_table", kind="table")

        assert result.success == 1
        assert result.result["exists"] is False

    def test_metric_found(self, generation_tools):
        mock_storage = Mock()
        generation_tools.metric_rag.storage = mock_storage
        mock_storage.search_all.return_value = [{"id": "m1", "name": "revenue"}]

        with patch("datus.tools.func_tool.generation_tools.build_where"), patch(
            "datus.tools.func_tool.generation_tools.eq"
        ):
            result = generation_tools.check_semantic_object_exists("revenue", kind="metric")

        assert result.success == 1
        assert result.result["exists"] is True

    def test_metric_not_found(self, generation_tools):
        mock_storage = Mock()
        generation_tools.metric_rag.storage = mock_storage
        mock_storage.search_all.return_value = []

        with patch("datus.tools.func_tool.generation_tools.build_where"), patch(
            "datus.tools.func_tool.generation_tools.eq"
        ):
            result = generation_tools.check_semantic_object_exists("unknown_metric", kind="metric")

        assert result.success == 1
        assert result.result["exists"] is False

    def test_column_found_with_table_context(self, generation_tools):
        mock_storage = Mock()
        generation_tools.semantic_rag.storage = mock_storage
        mock_storage.search_objects.return_value = [
            {"id": "c1", "name": "amount", "table_name": "orders", "kind": "column"}
        ]

        result = generation_tools.check_semantic_object_exists("orders.amount", kind="column", table_context="orders")

        assert result.success == 1
        assert result.result["exists"] is True

    def test_column_not_found(self, generation_tools):
        mock_storage = Mock()
        generation_tools.semantic_rag.storage = mock_storage
        mock_storage.search_objects.return_value = []

        result = generation_tools.check_semantic_object_exists("orders.nonexistent", kind="column")

        assert result.success == 1
        assert result.result["exists"] is False

    def test_column_name_match_without_table(self, generation_tools):
        mock_storage = Mock()
        generation_tools.semantic_rag.storage = mock_storage
        mock_storage.search_objects.return_value = [
            {"id": "c1", "name": "amount", "table_name": "orders", "kind": "column"}
        ]

        result = generation_tools.check_semantic_object_exists("amount", kind="column")

        assert result.success == 1
        assert result.result["exists"] is True

    def test_dotted_name_extracts_target(self, generation_tools):
        mock_storage = Mock()
        generation_tools.semantic_rag.storage = mock_storage
        mock_storage.search_all.return_value = [{"id": "t1", "name": "orders", "kind": "table"}]

        with patch("datus.tools.func_tool.generation_tools.build_where"), patch(
            "datus.tools.func_tool.generation_tools.And"
        ), patch("datus.tools.func_tool.generation_tools.eq"):
            result = generation_tools.check_semantic_object_exists("public.orders", kind="table")

        assert result.success == 1

    def test_exception_returns_failure(self, generation_tools):
        mock_storage = Mock()
        generation_tools.semantic_rag.storage = mock_storage
        mock_storage.search_all.side_effect = Exception("storage error")

        with patch("datus.tools.func_tool.generation_tools.build_where"), patch(
            "datus.tools.func_tool.generation_tools.And"
        ), patch("datus.tools.func_tool.generation_tools.eq"):
            result = generation_tools.check_semantic_object_exists("orders", kind="table")

        assert result.success == 0
        assert "storage error" in result.error

    def test_legacy_wrapper(self, generation_tools):
        mock_storage = Mock()
        generation_tools.semantic_rag.storage = mock_storage
        mock_storage.search_all.return_value = []

        with patch("datus.tools.func_tool.generation_tools.build_where"), patch(
            "datus.tools.func_tool.generation_tools.And"
        ), patch("datus.tools.func_tool.generation_tools.eq"):
            result = generation_tools.check_semantic_model_exists("orders")

        assert result.success == 1


class TestEndSemanticModelGeneration:
    def test_success_single_file(self, generation_tools):
        result = generation_tools.end_semantic_model_generation(["/path/to/model.yaml"])
        assert result.success == 1
        assert result.result["semantic_model_files"] == ["/path/to/model.yaml"]
        assert "1 file(s)" in result.result["message"]

    def test_success_multiple_files(self, generation_tools):
        files = ["/path/model1.yaml", "/path/model2.yaml"]
        result = generation_tools.end_semantic_model_generation(files)
        assert result.success == 1
        assert result.result["semantic_model_files"] == files
        assert "2 file(s)" in result.result["message"]

    def test_exception_returns_failure(self, generation_tools):
        with patch.object(generation_tools, "end_semantic_model_generation", side_effect=Exception("disk full")):
            pass  # testing the inner exception path

        # Test directly by triggering exception inside
        with patch("datus.tools.func_tool.generation_tools.logger") as mock_logger:
            mock_logger.info.side_effect = Exception("log failure")
            result = generation_tools.end_semantic_model_generation(["/path/model.yaml"])
            # Even if logger fails, result should fail gracefully
            assert result.success == 0


class TestEndMetricGeneration:
    def test_success_basic(self, generation_tools):
        result = generation_tools.end_metric_generation(metric_file="/path/metric.yaml")
        assert result.success == 1
        assert result.result["metric_file"] == "/path/metric.yaml"
        assert result.result["semantic_model_file"] == ""
        assert result.result["metric_sqls"] == {}

    def test_success_with_semantic_model(self, generation_tools):
        result = generation_tools.end_metric_generation(
            metric_file="/path/metric.yaml", semantic_model_file="/path/model.yaml"
        )
        assert result.success == 1
        assert result.result["semantic_model_file"] == "/path/model.yaml"

    def test_success_with_metric_sqls_json(self, generation_tools):
        metric_sqls_json = json.dumps({"revenue_total": "SELECT SUM(revenue) FROM orders"})
        result = generation_tools.end_metric_generation(
            metric_file="/path/metric.yaml", metric_sqls_json=metric_sqls_json
        )
        assert result.success == 1
        assert result.result["metric_sqls"] == {"revenue_total": "SELECT SUM(revenue) FROM orders"}

    def test_invalid_metric_sqls_json_ignored(self, generation_tools):
        result = generation_tools.end_metric_generation(
            metric_file="/path/metric.yaml", metric_sqls_json="not valid json"
        )
        assert result.success == 1
        assert result.result["metric_sqls"] == {}


class TestGenerateSqlSummaryId:
    def test_success(self, generation_tools):
        with patch("datus.storage.reference_sql.init_utils.gen_reference_sql_id", return_value="abc123"):
            result = generation_tools.generate_sql_summary_id("SELECT * FROM orders")
        assert result.success == 1
        assert result.result == "abc123"

    def test_exception_returns_failure(self, generation_tools):
        with patch(
            "datus.storage.reference_sql.init_utils.gen_reference_sql_id",
            side_effect=Exception("hash error"),
        ):
            result = generation_tools.generate_sql_summary_id("SELECT 1")
        assert result.success == 0
        assert "hash error" in result.error
