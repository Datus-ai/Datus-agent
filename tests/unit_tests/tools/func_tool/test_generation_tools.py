# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Unit tests for GenerationTools - CI level, zero external dependencies."""

import json
import os
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

        with patch("datus.tools.func_tool.generation_tools.And"), patch("datus.tools.func_tool.generation_tools.eq"):
            result = generation_tools.check_semantic_object_exists("orders", kind="table")

        assert result.success == 1
        assert result.result["exists"] is True
        assert result.result["name"] == "orders"

    def test_table_not_found(self, generation_tools):
        mock_storage = Mock()
        generation_tools.semantic_rag.storage = mock_storage
        mock_storage.search_all.return_value = []

        with patch("datus.tools.func_tool.generation_tools.And"), patch("datus.tools.func_tool.generation_tools.eq"):
            result = generation_tools.check_semantic_object_exists("unknown_table", kind="table")

        assert result.success == 1
        assert result.result["exists"] is False

    def test_metric_found(self, generation_tools):
        mock_storage = Mock()
        generation_tools.metric_rag.storage = mock_storage
        mock_storage.search_all.return_value = [{"id": "m1", "name": "revenue"}]

        with patch("datus.tools.func_tool.generation_tools.eq"):
            result = generation_tools.check_semantic_object_exists("revenue", kind="metric")

        assert result.success == 1
        assert result.result["exists"] is True

    def test_metric_not_found(self, generation_tools):
        mock_storage = Mock()
        generation_tools.metric_rag.storage = mock_storage
        mock_storage.search_all.return_value = []

        with patch("datus.tools.func_tool.generation_tools.eq"):
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

        with patch("datus.tools.func_tool.generation_tools.And"), patch("datus.tools.func_tool.generation_tools.eq"):
            result = generation_tools.check_semantic_object_exists("public.orders", kind="table")

        assert result.success == 1

    def test_exception_returns_failure(self, generation_tools):
        mock_storage = Mock()
        generation_tools.semantic_rag.storage = mock_storage
        mock_storage.search_all.side_effect = Exception("storage error")

        with patch("datus.tools.func_tool.generation_tools.And"), patch("datus.tools.func_tool.generation_tools.eq"):
            result = generation_tools.check_semantic_object_exists("orders", kind="table")

        assert result.success == 0
        assert "storage error" in result.error

    def test_legacy_wrapper(self, generation_tools):
        mock_storage = Mock()
        generation_tools.semantic_rag.storage = mock_storage
        mock_storage.search_all.return_value = []

        with patch("datus.tools.func_tool.generation_tools.And"), patch("datus.tools.func_tool.generation_tools.eq"):
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
        # Trigger exception inside the method by making logger.info raise
        with patch("datus.tools.func_tool.generation_tools.logger") as mock_logger:
            mock_logger.info.side_effect = Exception("log failure")
            result = generation_tools.end_semantic_model_generation(["/path/model.yaml"])
        assert result.success == 0
        assert "log failure" in result.error


class TestEndMetricGeneration:
    def _patch_sync(self, generation_tools):
        """Patch get_path_manager, the pre-flight validator (so legacy tests
        can pass synthetic paths), and _sync_metric_to_db."""
        return (
            patch("datus.tools.func_tool.generation_tools.get_path_manager"),
            patch.object(
                type(generation_tools),
                "_validate_metric_file_has_blocks",
                staticmethod(lambda _path: None),
            ),
            patch.object(generation_tools, "_sync_metric_to_db", return_value={"success": True, "message": "ok"}),
        )

    def test_success_basic(self, generation_tools):
        p1, p2, p3 = self._patch_sync(generation_tools)
        with p1, p2, p3:
            result = generation_tools.end_metric_generation(metric_file="/path/metric.yaml")
        assert result.success == 1
        assert result.result["metric_file"] == "/path/metric.yaml"
        assert result.result["semantic_model_file"] == ""
        assert result.result["metric_sqls"] == {}
        assert result.result["sync"]["success"] is True

    def test_success_with_semantic_model(self, generation_tools):
        p1, p2, p3 = self._patch_sync(generation_tools)
        with p1, p2, p3:
            result = generation_tools.end_metric_generation(
                metric_file="/path/metric.yaml", semantic_model_file="/path/model.yaml"
            )
        assert result.success == 1
        assert result.result["semantic_model_file"] == "/path/model.yaml"

    def test_success_with_metric_sqls_json(self, generation_tools):
        metric_sqls_json = json.dumps({"revenue_total": "SELECT SUM(revenue) FROM orders"})
        p1, p2, p3 = self._patch_sync(generation_tools)
        with p1, p2, p3:
            result = generation_tools.end_metric_generation(
                metric_file="/path/metric.yaml", metric_sqls_json=metric_sqls_json
            )
        assert result.success == 1
        assert result.result["metric_sqls"] == {"revenue_total": "SELECT SUM(revenue) FROM orders"}

    def test_invalid_metric_sqls_json_ignored(self, generation_tools):
        p1, p2, p3 = self._patch_sync(generation_tools)
        with p1, p2, p3:
            result = generation_tools.end_metric_generation(
                metric_file="/path/metric.yaml", metric_sqls_json="not valid json"
            )
        assert result.success == 1
        assert result.result["metric_sqls"] == {}


class TestEndMetricGenerationPreflight:
    """Pre-flight validation rejects metric files with no `metric:` blocks
    BEFORE attempting the deeper sync, so the LLM gets an actionable error
    instead of an opaque "No valid objects found to sync"."""

    @staticmethod
    def _patch_path_resolution(tools, kb_root):
        """Make end_metric_generation treat absolute paths as-is."""
        mock_pm = Mock()
        mock_pm.knowledge_base_home = str(kb_root)
        tools.agent_config.current_namespace = "ns"
        return patch(
            "datus.tools.func_tool.generation_tools.get_path_manager",
            return_value=mock_pm,
        )

    def test_rejects_missing_metric_file(self, generation_tools, tmp_path):
        with self._patch_path_resolution(generation_tools, tmp_path):
            result = generation_tools.end_metric_generation(metric_file=str(tmp_path / "missing.yaml"))
        assert result.success == 0
        assert "Metric file not found" in result.error

    def test_rejects_documentation_only_metric_file(self, generation_tools, tmp_path):
        bad = tmp_path / "frpm_metrics.yml"
        bad.write_text(
            "# Generated metric documentation\n\n"
            "## Summary\n\n"
            "- avg_percent_eligible_free_ages_5_17\n"
            "- total_free_meal_count_ages_5_17\n"
        )
        with (
            self._patch_path_resolution(generation_tools, tmp_path),
            patch.object(generation_tools, "_sync_metric_to_db") as sync_mock,
        ):
            result = generation_tools.end_metric_generation(metric_file=str(bad))
        assert result.success == 0
        assert "no `metric:` YAML blocks" in result.error
        assert "create_metric: true" in result.error
        sync_mock.assert_not_called()

    def test_rejects_invalid_yaml(self, generation_tools, tmp_path):
        bad = tmp_path / "broken.yml"
        bad.write_text("name: x\n  bad-indent: : :\n")
        with (
            self._patch_path_resolution(generation_tools, tmp_path),
            patch.object(generation_tools, "_sync_metric_to_db") as sync_mock,
        ):
            result = generation_tools.end_metric_generation(metric_file=str(bad))
        assert result.success == 0
        assert "not valid YAML" in result.error
        sync_mock.assert_not_called()

    def test_accepts_file_with_metric_block(self, generation_tools, tmp_path):
        good = tmp_path / "good_metric.yml"
        good.write_text("metric:\n  name: revenue_total\n  type: measure_proxy\n  type_params:\n    measure: revenue\n")
        with (
            self._patch_path_resolution(generation_tools, tmp_path),
            patch.object(generation_tools, "_sync_metric_to_db", return_value={"success": True, "message": "ok"}),
        ):
            result = generation_tools.end_metric_generation(metric_file=str(good))
        assert result.success == 1


class TestValidateMetricFileHasBlocks:
    """Direct unit tests for the metric-file pre-flight validator."""

    def test_returns_error_for_missing_file(self):
        from datus.tools.func_tool.generation_tools import GenerationTools

        msg = GenerationTools._validate_metric_file_has_blocks("/nonexistent/m.yaml")
        assert msg is not None and "not found" in msg

    def test_returns_error_for_documentation_only(self, tmp_path):
        from datus.tools.func_tool.generation_tools import GenerationTools

        f = tmp_path / "doc.yml"
        f.write_text("# just docs\n- bullet\n- bullet2\n")
        msg = GenerationTools._validate_metric_file_has_blocks(str(f))
        assert msg is not None
        assert "no `metric:` YAML blocks" in msg

    def test_returns_error_for_invalid_yaml(self, tmp_path):
        from datus.tools.func_tool.generation_tools import GenerationTools

        f = tmp_path / "broken.yml"
        f.write_text(": : :\n  - oops\n  not yaml")
        msg = GenerationTools._validate_metric_file_has_blocks(str(f))
        assert msg is not None
        assert "not valid YAML" in msg

    def test_returns_none_for_single_metric_block(self, tmp_path):
        from datus.tools.func_tool.generation_tools import GenerationTools

        f = tmp_path / "ok.yml"
        f.write_text("metric:\n  name: x\n  type: measure_proxy\n")
        assert GenerationTools._validate_metric_file_has_blocks(str(f)) is None

    def test_returns_none_for_multi_metric_yaml(self, tmp_path):
        from datus.tools.func_tool.generation_tools import GenerationTools

        f = tmp_path / "multi.yml"
        f.write_text("metric:\n  name: a\n  type: measure_proxy\n---\nmetric:\n  name: b\n  type: measure_proxy\n")
        assert GenerationTools._validate_metric_file_has_blocks(str(f)) is None

    def test_data_source_only_is_rejected(self, tmp_path):
        """A file with only `data_source:` (no `metric:`) is not a metric file."""
        from datus.tools.func_tool.generation_tools import GenerationTools

        f = tmp_path / "ds.yml"
        f.write_text("data_source:\n  name: orders\n")
        msg = GenerationTools._validate_metric_file_has_blocks(str(f))
        assert msg is not None and "no `metric:` YAML blocks" in msg


class TestSyncMetricToDb:
    """Tests for GenerationTools._sync_metric_to_db() private method."""

    def test_metric_file_not_found(self, generation_tools):
        result = generation_tools._sync_metric_to_db("/nonexistent/metric.yaml")
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_metric_only_sync(self, generation_tools, tmp_path):
        """Sync metric file alone when no semantic model file provided."""
        metric_file = tmp_path / "metric.yaml"
        metric_file.write_text("metric:\n  name: revenue\n  type: simple\n")

        with patch("datus.cli.generation_hooks.GenerationHooks._sync_semantic_to_db") as mock_sync:
            mock_sync.return_value = {"success": True, "message": "synced"}
            result = generation_tools._sync_metric_to_db(str(metric_file))

        assert result["success"] is True
        mock_sync.assert_called_once_with(
            str(metric_file),
            generation_tools.agent_config,
            metric_sqls=None,
        )

    def test_metric_with_semantic_model_combines_files(self, generation_tools, tmp_path):
        """When both metric and semantic model files exist, combine into temp file."""
        metric_file = tmp_path / "metric.yaml"
        metric_file.write_text("metric:\n  name: revenue\n  type: simple\n")
        semantic_file = tmp_path / "model.yaml"
        semantic_file.write_text("semantic_model:\n  name: orders\n")

        with patch("datus.cli.generation_hooks.GenerationHooks._sync_semantic_to_db") as mock_sync:
            mock_sync.return_value = {"success": True, "message": "synced"}
            result = generation_tools._sync_metric_to_db(str(metric_file), str(semantic_file), {"rev": "SELECT 1"})

        assert result["success"] is True
        # Should have been called twice: first for semantic objects, then for metrics
        assert mock_sync.call_count == 2
        # First call: sync semantic objects
        sem_call = mock_sync.call_args_list[0]
        assert sem_call.kwargs.get("include_semantic_objects") is True
        assert sem_call.kwargs.get("include_metrics") is False
        # Second call: sync metrics from combined temp file
        metric_call = mock_sync.call_args_list[1]
        actual_temp_path = metric_call[0][0]
        assert not os.path.exists(actual_temp_path), f"Temp file should be cleaned up: {actual_temp_path}"
        assert metric_call.kwargs.get("include_semantic_objects") is False
        assert metric_call.kwargs.get("include_metrics") is True
        assert metric_call.kwargs.get("metric_sqls") == {"rev": "SELECT 1"}

    def test_semantic_sync_failure_aborts_metric_sync(self, generation_tools, tmp_path):
        """When semantic object sync fails, metric sync is skipped and failure propagated."""
        metric_file = tmp_path / "metric.yaml"
        metric_file.write_text("metric:\n  name: revenue\n  type: simple\n")
        semantic_file = tmp_path / "model.yaml"
        semantic_file.write_text("semantic_model:\n  name: orders\n")

        with patch("datus.cli.generation_hooks.GenerationHooks._sync_semantic_to_db") as mock_sync:
            mock_sync.return_value = {"success": False, "error": "semantic sync failed"}
            result = generation_tools._sync_metric_to_db(str(metric_file), str(semantic_file))

        assert result["success"] is False
        assert result["error"] == "semantic sync failed"
        # Only called once (semantic sync), metric sync was skipped
        assert mock_sync.call_count == 1

    def test_semantic_model_not_exists_falls_through(self, generation_tools, tmp_path):
        """When semantic_model_file path provided but file doesn't exist, sync metric alone."""
        metric_file = tmp_path / "metric.yaml"
        metric_file.write_text("metric:\n  name: revenue\n")

        with patch("datus.cli.generation_hooks.GenerationHooks._sync_semantic_to_db") as mock_sync:
            mock_sync.return_value = {"success": True, "message": "ok"}
            result = generation_tools._sync_metric_to_db(str(metric_file), "/nonexistent/model.yaml")

        assert result["success"] is True
        # Should call with metric file directly (not combined)
        mock_sync.assert_called_once_with(
            str(metric_file),
            generation_tools.agent_config,
            metric_sqls=None,
        )

    def test_sync_failure_propagated(self, generation_tools, tmp_path):
        """Sync failure result is returned as-is."""
        metric_file = tmp_path / "metric.yaml"
        metric_file.write_text("metric:\n  name: revenue\n")

        with patch("datus.cli.generation_hooks.GenerationHooks._sync_semantic_to_db") as mock_sync:
            mock_sync.return_value = {"success": False, "error": "storage unavailable"}
            result = generation_tools._sync_metric_to_db(str(metric_file))

        assert result["success"] is False
        assert result["error"] == "storage unavailable"

    def test_exception_returns_failure(self, generation_tools, tmp_path):
        """Exception during sync is caught and returned as failure dict."""
        metric_file = tmp_path / "metric.yaml"
        metric_file.write_text("metric:\n  name: revenue\n")

        with patch("datus.cli.generation_hooks.GenerationHooks._sync_semantic_to_db") as mock_sync:
            mock_sync.side_effect = RuntimeError("connection lost")
            result = generation_tools._sync_metric_to_db(str(metric_file))

        assert result["success"] is False
        assert "connection lost" in result["error"]

    def test_temp_file_cleaned_on_sync_exception(self, generation_tools, tmp_path):
        """Temp combined file is cleaned up even when sync raises."""
        metric_file = tmp_path / "metric.yaml"
        metric_file.write_text("metric:\n  name: revenue\n")
        semantic_file = tmp_path / "model.yaml"
        semantic_file.write_text("semantic_model:\n  name: orders\n")

        with patch("datus.cli.generation_hooks.GenerationHooks._sync_semantic_to_db") as mock_sync:
            mock_sync.side_effect = RuntimeError("boom")
            result = generation_tools._sync_metric_to_db(str(metric_file), str(semantic_file))

        assert result["success"] is False
        # Temp file should still be cleaned up
        assert not (tmp_path / "model.yaml.combined.tmp").exists()


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
