# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Unit tests for GenerationTools - CI level, zero external dependencies."""

import json
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from datus.tools.func_tool.base import FuncToolResult


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

    def test_accepts_prompt_documented_name_argument(self, generation_tools):
        mock_storage = Mock()
        generation_tools.semantic_rag.storage = mock_storage
        mock_storage.search_all.return_value = [{"id": "t1", "name": "orders", "kind": "table"}]

        with patch("datus.tools.func_tool.generation_tools.And"), patch("datus.tools.func_tool.generation_tools.eq"):
            result = generation_tools.check_semantic_object_exists(name="orders", kind="table")

        assert result.success == 1
        assert result.result["exists"] is True

    def test_accepts_legacy_object_name_argument(self, generation_tools):
        mock_storage = Mock()
        generation_tools.semantic_rag.storage = mock_storage
        mock_storage.search_all.return_value = [{"id": "t1", "name": "orders", "kind": "table"}]

        with patch("datus.tools.func_tool.generation_tools.And"), patch("datus.tools.func_tool.generation_tools.eq"):
            result = generation_tools.check_semantic_object_exists(object_name="orders", kind="table")

        assert result.success == 1
        assert result.result["exists"] is True

    def test_requires_name(self, generation_tools):
        result = generation_tools.check_semantic_object_exists(kind="table")

        assert result.success == 0
        assert "name is required" in result.error

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
    def _mark_validated(self, generation_tools):
        generation_tools.generation_evidence.validation_passed = True

    def test_requires_validation(self, generation_tools):
        result = generation_tools.end_semantic_model_generation(["/path/to/model.yaml"])
        assert result.success == 0
        assert "validate_semantic must pass" in result.error

    def test_success_single_file(self, generation_tools):
        self._mark_validated(generation_tools)
        result = generation_tools.end_semantic_model_generation(["/path/to/model.yaml"])
        assert result.success == 1
        assert result.result["semantic_model_files"] == ["/path/to/model.yaml"]
        assert "1 file(s)" in result.result["message"]

    def test_success_multiple_files(self, generation_tools):
        self._mark_validated(generation_tools)
        files = ["/path/model1.yaml", "/path/model2.yaml"]
        result = generation_tools.end_semantic_model_generation(files)
        assert result.success == 1
        assert result.result["semantic_model_files"] == files
        assert "2 file(s)" in result.result["message"]

    def test_exception_returns_failure(self, generation_tools):
        self._mark_validated(generation_tools)
        # Trigger exception inside the method by making logger.info raise
        with patch("datus.tools.func_tool.generation_tools.logger") as mock_logger:
            mock_logger.info.side_effect = Exception("log failure")
            result = generation_tools.end_semantic_model_generation(["/path/model.yaml"])
        assert result.success == 0
        assert "log failure" in result.error


class TestEndMetricGeneration:
    def _mark_ready_to_publish(self, generation_tools):
        generation_tools.generation_evidence.validation_passed = True
        generation_tools.generation_evidence.metric_dry_run_passed = True

    def _patch_sync(self, generation_tools):
        """Patch get_path_manager, the pre-flight validator (so legacy tests
        can pass synthetic paths), and _sync_metric_to_db."""
        mock_pm = Mock()
        mock_pm.subject_dir = "/path"
        return (
            patch("datus.tools.func_tool.generation_tools.get_path_manager", return_value=mock_pm),
            patch.object(
                type(generation_tools),
                "_validate_metric_file_has_blocks",
                staticmethod(lambda _path: None),
            ),
            patch.object(generation_tools, "_sync_metric_to_db", return_value={"success": True, "message": "ok"}),
        )

    def test_requires_validation(self, generation_tools):
        result = generation_tools.end_metric_generation(metric_file="/path/semantic_models/metric.yaml")
        assert result.success == 0
        assert "validate_semantic must pass" in result.error

    def test_requires_dry_run(self, generation_tools):
        generation_tools.generation_evidence.validation_passed = True
        result = generation_tools.end_metric_generation(metric_file="/path/semantic_models/metric.yaml")
        assert result.success == 0
        assert "query_metrics(dry_run=True) must pass" in result.error

    def test_success_basic(self, generation_tools):
        self._mark_ready_to_publish(generation_tools)
        p1, p2, p3 = self._patch_sync(generation_tools)
        with p1, p2, p3:
            result = generation_tools.end_metric_generation(metric_file="/path/semantic_models/metric.yaml")
        assert result.success == 1
        assert result.result["metric_file"] == "/path/semantic_models/metric.yaml"
        assert result.result["semantic_model_files"] == []
        assert result.result["metric_sqls"] == {}
        assert result.result["sync"]["success"] is True

    def test_success_with_semantic_models(self, generation_tools):
        self._mark_ready_to_publish(generation_tools)
        p1, p2, p3 = self._patch_sync(generation_tools)
        with p1, p2, p3:
            result = generation_tools.end_metric_generation(
                metric_file="/path/semantic_models/metric.yaml",
                semantic_model_files=["/path/semantic_models/model.yaml"],
            )
        assert result.success == 1
        assert result.result["semantic_model_files"] == ["/path/semantic_models/model.yaml"]

    def test_success_with_metric_sqls_json(self, generation_tools):
        self._mark_ready_to_publish(generation_tools)
        metric_sqls_json = json.dumps({"revenue_total": "SELECT SUM(revenue) FROM orders"})
        p1, p2, p3 = self._patch_sync(generation_tools)
        with p1, p2, p3:
            result = generation_tools.end_metric_generation(
                metric_file="/path/semantic_models/metric.yaml", metric_sqls_json=metric_sqls_json
            )
        assert result.success == 1
        assert result.result["metric_sqls"] == {"revenue_total": "SELECT SUM(revenue) FROM orders"}

    def test_invalid_metric_sqls_json_ignored(self, generation_tools):
        self._mark_ready_to_publish(generation_tools)
        p1, p2, p3 = self._patch_sync(generation_tools)
        with p1, p2, p3:
            result = generation_tools.end_metric_generation(
                metric_file="/path/semantic_models/metric.yaml", metric_sqls_json="not valid json"
            )
        assert result.success == 1
        assert result.result["metric_sqls"] == {}

    def test_osi_skips_metricflow_metric_block_preflight(self, generation_tools, tmp_path):
        self._mark_ready_to_publish(generation_tools)
        generation_tools.authoring_format = "osi"
        metric_file = tmp_path / "semantic_models" / "starrocks" / "orders_metrics.yml"
        metric_file.parent.mkdir(parents=True)
        metric_file.write_text(
            "metrics:\n  - name: order_count\n    expression: COUNT(DISTINCT order_id)\n    dataset: orders\n"
        )
        mock_pm = Mock()
        mock_pm.subject_dir = str(tmp_path)
        with (
            patch("datus.tools.func_tool.generation_tools.get_path_manager", return_value=mock_pm),
            patch.object(type(generation_tools), "_validate_metric_file_has_blocks") as preflight_mock,
            patch.object(
                generation_tools,
                "_sync_osi_metric_to_db",
                return_value={"success": True, "message": "synced"},
            ) as sync_mock,
        ):
            result = generation_tools.end_metric_generation(metric_file=str(metric_file))

        assert result.success == 1
        preflight_mock.assert_not_called()
        sync_mock.assert_called_once_with(str(metric_file), [], {}, replace_metric_artifact=False)

    def test_osi_forwards_all_semantic_model_files_to_sync(self, generation_tools, tmp_path):
        self._mark_ready_to_publish(generation_tools)
        generation_tools.authoring_format = "osi"
        semantic_root = tmp_path / "semantic_models" / "starrocks"
        metric_file = semantic_root / "metrics" / "orders_metrics.yml"
        orders_file = semantic_root / "orders.yml"
        customers_file = semantic_root / "customers.yml"
        metric_file.parent.mkdir(parents=True)
        orders_file.parent.mkdir(parents=True, exist_ok=True)
        metric_file.write_text("metrics:\n  - name: order_count\n")
        orders_file.write_text("datasets:\n  - name: orders\n")
        customers_file.write_text("datasets:\n  - name: customers\n")
        mock_pm = Mock()
        mock_pm.subject_dir = str(tmp_path)

        with (
            patch("datus.tools.func_tool.generation_tools.get_path_manager", return_value=mock_pm),
            patch.object(type(generation_tools), "_validate_metric_file_has_blocks") as preflight_mock,
            patch.object(
                generation_tools,
                "_sync_osi_metric_to_db",
                return_value={
                    "success": True,
                    "message": "synced",
                    "semantic_synced": True,
                    "semantic_model_files_synced": [str(orders_file), str(customers_file)],
                },
            ) as sync_mock,
        ):
            result = generation_tools.end_metric_generation(
                metric_file=str(metric_file),
                semantic_model_files=[str(orders_file), str(customers_file)],
            )

        assert result.success == 1
        preflight_mock.assert_not_called()
        sync_mock.assert_called_once_with(
            str(metric_file),
            [str(orders_file), str(customers_file)],
            {},
            replace_metric_artifact=False,
        )
        assert generation_tools.generation_evidence.semantic_kb_sync_passed is True


class TestEndMetricGenerationPreflight:
    """Pre-flight validation rejects metric files with no `metric:` blocks
    BEFORE attempting the deeper sync, so the LLM gets an actionable error
    instead of an opaque "No valid objects found to sync"."""

    @staticmethod
    def _patch_path_resolution(tools, kb_root):
        """Make end_metric_generation resolve paths under a synthetic KB root."""
        mock_pm = Mock()
        mock_pm.subject_dir = str(kb_root)
        return patch(
            "datus.tools.func_tool.generation_tools.get_path_manager",
            return_value=mock_pm,
        )

    @staticmethod
    def _mark_ready_to_publish(generation_tools):
        generation_tools.generation_evidence.validation_passed = True
        generation_tools.generation_evidence.metric_dry_run_passed = True

    def test_rejects_missing_metric_file(self, generation_tools, tmp_path):
        self._mark_ready_to_publish(generation_tools)
        missing = tmp_path / "semantic_models" / "missing.yaml"
        with self._patch_path_resolution(generation_tools, tmp_path):
            result = generation_tools.end_metric_generation(metric_file=str(missing))
        assert result.success == 0
        assert "Metric file not found" in result.error

    def test_rejects_documentation_only_metric_file(self, generation_tools, tmp_path):
        self._mark_ready_to_publish(generation_tools)
        bad = tmp_path / "semantic_models" / "frpm_metrics.yml"
        bad.parent.mkdir(parents=True, exist_ok=True)
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
        self._mark_ready_to_publish(generation_tools)
        bad = tmp_path / "semantic_models" / "broken.yml"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text("name: x\n  bad-indent: : :\n")
        with (
            self._patch_path_resolution(generation_tools, tmp_path),
            patch.object(generation_tools, "_sync_metric_to_db") as sync_mock,
        ):
            result = generation_tools.end_metric_generation(metric_file=str(bad))
        assert result.success == 0
        assert "not valid YAML" in result.error
        sync_mock.assert_not_called()

    def test_rejects_unnamed_metric_block(self, generation_tools, tmp_path):
        self._mark_ready_to_publish(generation_tools)
        bad = tmp_path / "semantic_models" / "unnamed_metric.yml"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text("metric:\n  description: missing name\n  type: measure_proxy\n")
        with (
            self._patch_path_resolution(generation_tools, tmp_path),
            patch.object(generation_tools, "_sync_metric_to_db") as sync_mock,
        ):
            result = generation_tools.end_metric_generation(metric_file=str(bad))
        assert result.success == 0
        assert "non-empty `metric.name`" in result.error
        sync_mock.assert_not_called()

    def test_accepts_file_with_metric_block(self, generation_tools, tmp_path):
        self._mark_ready_to_publish(generation_tools)
        generation_tools.generation_evidence.metric_dry_run_metrics.add("revenue_total")
        good = tmp_path / "semantic_models" / "good_metric.yml"
        good.parent.mkdir(parents=True, exist_ok=True)
        good.write_text("metric:\n  name: revenue_total\n  type: measure_proxy\n  type_params:\n    measure: revenue\n")
        with (
            self._patch_path_resolution(generation_tools, tmp_path),
            patch.object(generation_tools, "_sync_metric_to_db", return_value={"success": True, "message": "ok"}),
        ):
            result = generation_tools.end_metric_generation(metric_file=str(good))
        assert result.success == 1

    def test_metric_sqls_scope_excludes_existing_file_metrics(self, generation_tools, tmp_path):
        self._mark_ready_to_publish(generation_tools)
        generation_tools.generation_evidence.metric_dry_run_metrics.add("new_metric")
        generation_tools.generation_evidence.set_metric_queryability_contracts(
            [
                {
                    "source": "sql_1",
                    "metric_hints": ["existing_metric"],
                    "dimension_hints": ["start_month"],
                    "time_group_hints": [
                        {
                            "alias": "start_month",
                            "base_expr": "created_at",
                            "grain": "month",
                        }
                    ],
                }
            ]
        )
        metric_file = tmp_path / "semantic_models" / "mixed_metrics.yml"
        metric_file.parent.mkdir(parents=True, exist_ok=True)
        metric_file.write_text(
            "metric:\n"
            "  name: existing_metric\n"
            "  type: measure_proxy\n"
            "  type_params:\n"
            "    measure: existing_metric\n"
            "---\n"
            "metric:\n"
            "  name: new_metric\n"
            "  type: measure_proxy\n"
            "  type_params:\n"
            "    measure: new_metric\n"
        )
        metric_sqls_json = json.dumps(
            {
                "existing_metric": "SELECT old_metric",
                "new_metric": "SELECT new_metric",
                "__query_metrics_dry_run__": "SELECT grouped_validation",
            }
        )

        with (
            self._patch_path_resolution(generation_tools, tmp_path),
            patch.object(generation_tools, "_existing_metric_names", return_value={"existing_metric"}),
            patch.object(generation_tools, "_validate_metric_name_conflicts", return_value=None),
            patch.object(
                generation_tools, "_sync_metric_to_db", return_value={"success": True, "message": "ok"}
            ) as sync_mock,
        ):
            result = generation_tools.end_metric_generation(
                metric_file=str(metric_file),
                metric_sqls_json=metric_sqls_json,
            )

        assert result.success == 1
        sync_mock.assert_called_once()
        assert sync_mock.call_args.kwargs["metric_names_to_sync"] == {"new_metric"}

    def test_combined_dry_run_scope_syncs_all_new_metrics(self, generation_tools, tmp_path):
        self._mark_ready_to_publish(generation_tools)
        generation_tools.generation_evidence.metric_dry_run_metrics.update(
            {"existing_metric", "new_metric", "other_new_metric"}
        )
        generation_tools.generation_evidence.set_metric_queryability_contracts(
            [
                {
                    "source": "sql_1",
                    "metric_hints": ["existing_metric"],
                    "dimension_hints": ["start_month"],
                }
            ]
        )
        metric_file = tmp_path / "semantic_models" / "combined_metrics.yml"
        metric_file.parent.mkdir(parents=True, exist_ok=True)
        metric_file.write_text(
            "metric:\n"
            "  name: existing_metric\n"
            "  type: measure_proxy\n"
            "  type_params:\n"
            "    measure: existing_metric\n"
            "---\n"
            "metric:\n"
            "  name: new_metric\n"
            "  type: measure_proxy\n"
            "  type_params:\n"
            "    measure: new_metric\n"
            "---\n"
            "metric:\n"
            "  name: other_new_metric\n"
            "  type: measure_proxy\n"
            "  type_params:\n"
            "    measure: other_new_metric\n"
        )
        metric_sqls_json = json.dumps(
            {
                "existing_metric": "SELECT old_metric",
                "__query_metrics_dry_run__": "SELECT grouped_validation",
            }
        )

        with (
            self._patch_path_resolution(generation_tools, tmp_path),
            patch.object(generation_tools, "_existing_metric_names", return_value={"existing_metric"}),
            patch.object(generation_tools, "_validate_metric_name_conflicts", return_value=None),
            patch.object(
                generation_tools, "_sync_metric_to_db", return_value={"success": True, "message": "ok"}
            ) as sync_mock,
        ):
            result = generation_tools.end_metric_generation(
                metric_file=str(metric_file),
                metric_sqls_json=metric_sqls_json,
            )

        assert result.success == 1
        sync_mock.assert_called_once()
        assert sync_mock.call_args.kwargs["metric_names_to_sync"] == {"new_metric", "other_new_metric"}

    def test_rejects_missing_grouped_queryability_dry_run(self, generation_tools, tmp_path):
        self._mark_ready_to_publish(generation_tools)
        generation_tools.generation_evidence.metric_dry_run_metrics.add("revenue_total")
        generation_tools.generation_evidence.set_metric_queryability_contracts(
            [
                {
                    "source": "sql_1",
                    "metric_hints": ["revenue_total"],
                    "dimension_hints": ["customer_segment"],
                }
            ]
        )
        good = tmp_path / "semantic_models" / "good_metric.yml"
        good.parent.mkdir(parents=True, exist_ok=True)
        good.write_text("metric:\n  name: revenue_total\n  type: measure_proxy\n  type_params:\n    measure: revenue\n")
        with (
            self._patch_path_resolution(generation_tools, tmp_path),
            patch.object(generation_tools, "_sync_metric_to_db") as sync_mock,
        ):
            result = generation_tools.end_metric_generation(metric_file=str(good))
        assert result.success == 0
        assert "source SQL group-by dimensions" in result.error
        assert result.result["queryability_contracts"][0]["dimension_hints"] == ["customer_segment"]
        sync_mock.assert_not_called()

    def test_accepts_grouped_queryability_dry_run(self, generation_tools, tmp_path):
        self._mark_ready_to_publish(generation_tools)
        generation_tools.generation_evidence.set_metric_queryability_contracts(
            [
                {
                    "source": "sql_1",
                    "metric_hints": ["revenue_total"],
                    "dimension_hints": ["customer_segment"],
                }
            ]
        )
        generation_tools.generation_evidence.record_metric_dry_run(
            ["revenue_total"],
            FuncToolResult(success=1, result={"metadata": {"sql": "SELECT 1"}}),
            dimensions=["customer_segment"],
        )
        good = tmp_path / "semantic_models" / "good_metric.yml"
        good.parent.mkdir(parents=True, exist_ok=True)
        good.write_text("metric:\n  name: revenue_total\n  type: measure_proxy\n  type_params:\n    measure: revenue\n")
        with (
            self._patch_path_resolution(generation_tools, tmp_path),
            patch.object(generation_tools, "_sync_metric_to_db", return_value={"success": True, "message": "ok"}),
        ):
            result = generation_tools.end_metric_generation(metric_file=str(good))
        assert result.success == 1

    def test_rejects_metric_not_covered_by_dry_run(self, generation_tools, tmp_path):
        self._mark_ready_to_publish(generation_tools)
        good = tmp_path / "semantic_models" / "good_metric.yml"
        good.parent.mkdir(parents=True, exist_ok=True)
        good.write_text("metric:\n  name: revenue_total\n  type: measure_proxy\n  type_params:\n    measure: revenue\n")
        with (
            self._patch_path_resolution(generation_tools, tmp_path),
            patch.object(generation_tools, "_sync_metric_to_db") as sync_mock,
        ):
            result = generation_tools.end_metric_generation(metric_file=str(good))
        assert result.success == 0
        assert "revenue_total" in result.error
        sync_mock.assert_not_called()

    def test_rejects_out_of_sandbox_metric_path_before_reading(self, generation_tools, tmp_path):
        self._mark_ready_to_publish(generation_tools)
        outside = tmp_path / "outside_metric.yml"
        outside.write_text("metric:\n  name: outside_metric\n")
        with (
            self._patch_path_resolution(generation_tools, tmp_path),
            patch.object(type(generation_tools), "_validate_metric_file_has_blocks") as validate_mock,
            patch.object(generation_tools, "_sync_metric_to_db") as sync_mock,
        ):
            result = generation_tools.end_metric_generation(metric_file=str(outside))
        assert result.success == 0
        assert "metric_file escapes Knowledge Base sandbox" in result.error
        validate_mock.assert_not_called()
        sync_mock.assert_not_called()


class TestValidateMetricFileHasBlocks:
    """Direct unit tests for the metric-file pre-flight validator."""

    def test_returns_error_for_missing_file(self):
        from datus.tools.func_tool.generation_tools import GenerationTools

        msg = GenerationTools._validate_metric_file_has_blocks("/nonexistent/m.yaml")
        assert "not found" in msg

    def test_returns_error_for_documentation_only(self, tmp_path):
        from datus.tools.func_tool.generation_tools import GenerationTools

        f = tmp_path / "doc.yml"
        f.write_text("# just docs\n- bullet\n- bullet2\n")
        msg = GenerationTools._validate_metric_file_has_blocks(str(f))
        assert "no `metric:` YAML blocks" in msg

    def test_returns_error_for_invalid_yaml(self, tmp_path):
        from datus.tools.func_tool.generation_tools import GenerationTools

        f = tmp_path / "broken.yml"
        f.write_text(": : :\n  - oops\n  not yaml")
        msg = GenerationTools._validate_metric_file_has_blocks(str(f))
        assert "not valid YAML" in msg

    def test_returns_error_for_unnamed_metric_block(self, tmp_path):
        from datus.tools.func_tool.generation_tools import GenerationTools

        f = tmp_path / "unnamed.yml"
        f.write_text("metric:\n  description: missing name\n  type: measure_proxy\n")
        msg = GenerationTools._validate_metric_file_has_blocks(str(f))
        assert "non-empty `metric.name`" in msg

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

    def test_current_db_parts_prefers_runtime_context(self):
        from datus.tools.func_tool.generation_tools import GenerationTools

        agent_config = SimpleNamespace(
            current_db_config=lambda: SimpleNamespace(catalog="", database="", schema=""),
            runtime_db_context=lambda: {
                "catalog": "default_catalog",
                "database": "ac_manage",
                "schema": "public",
            },
        )

        assert GenerationTools._current_db_parts(agent_config) == {
            "catalog_name": "default_catalog",
            "database_name": "ac_manage",
            "schema_name": "public",
        }

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
        assert result["semantic_synced"] is False
        mock_sync.assert_called_once_with(
            str(metric_file),
            generation_tools.agent_config,
            include_semantic_objects=False,
            include_metrics=True,
            metric_sqls=None,
            original_yaml_path=str(metric_file),
            replace_metric_artifact=False,
        )

    def test_metric_with_semantic_models_syncs_semantic_then_metric(self, generation_tools, tmp_path):
        """When semantic model files are provided, sync semantic objects before metrics."""
        metric_file = tmp_path / "metric.yaml"
        metric_file.write_text("metric:\n  name: revenue\n  type: simple\n")
        semantic_file = tmp_path / "model.yaml"
        semantic_file.write_text("semantic_model:\n  name: orders\n")

        with patch("datus.cli.generation_hooks.GenerationHooks._sync_semantic_to_db") as mock_sync:
            mock_sync.return_value = {"success": True, "message": "synced"}
            result = generation_tools._sync_metric_to_db(str(metric_file), [str(semantic_file)], {"rev": "SELECT 1"})

        assert result["success"] is True
        assert result["semantic_synced"] is True
        assert result["semantic_model_files_synced"] == [str(semantic_file)]
        # Should have been called twice: first for semantic objects, then for metrics
        assert mock_sync.call_count == 2
        # First call: sync semantic objects
        sem_call = mock_sync.call_args_list[0]
        assert sem_call.kwargs.get("include_semantic_objects") is True
        assert sem_call.kwargs.get("include_metrics") is False
        # Second call: sync metrics from the metric file itself
        metric_call = mock_sync.call_args_list[1]
        assert metric_call[0][0] == str(metric_file)
        assert metric_call.kwargs.get("include_semantic_objects") is False
        assert metric_call.kwargs.get("include_metrics") is True
        assert metric_call.kwargs.get("replace_metric_artifact") is False
        assert metric_call.kwargs.get("metric_sqls") == {"rev": "SELECT 1"}
        assert metric_call.kwargs.get("original_yaml_path") == str(metric_file)

    def test_metric_sync_filters_to_publish_scope(self, generation_tools, tmp_path):
        """When a publish scope is supplied, sync only those metric YAML docs."""
        metric_file = tmp_path / "metric.yaml"
        metric_file.write_text(
            "metric:\n"
            "  name: existing_metric\n"
            "  type: measure_proxy\n"
            "  type_params:\n"
            "    measure: existing_metric\n"
            "---\n"
            "metric:\n"
            "  name: new_metric\n"
            "  type: measure_proxy\n"
            "  type_params:\n"
            "    measure: new_metric\n"
        )
        captured = {}

        def fake_sync(file_path, *args, **kwargs):
            captured["file_path"] = file_path
            with open(file_path, encoding="utf-8") as f:
                captured["content"] = f.read()
            captured["kwargs"] = kwargs
            return {"success": True, "message": "synced"}

        with patch("datus.cli.generation_hooks.GenerationHooks._sync_semantic_to_db", side_effect=fake_sync):
            result = generation_tools._sync_metric_to_db(
                str(metric_file),
                metric_sqls={
                    "existing_metric": "SELECT old_metric",
                    "new_metric": "SELECT new_metric",
                    "__query_metrics_dry_run__": "SELECT grouped_validation",
                },
                metric_names_to_sync={"new_metric"},
            )

        assert result["success"] is True
        assert result["metric_names_synced"] == ["new_metric"]
        assert captured["file_path"] != str(metric_file)
        assert not os.path.exists(captured["file_path"])
        assert "name: new_metric" in captured["content"]
        assert "name: existing_metric" not in captured["content"]
        assert captured["kwargs"]["metric_sqls"] == {"new_metric": "SELECT new_metric"}
        assert captured["kwargs"]["original_yaml_path"] == str(metric_file)
        assert captured["kwargs"]["replace_metric_artifact"] is False

    def test_semantic_sync_failure_aborts_metric_sync(self, generation_tools, tmp_path):
        """When semantic object sync fails, metric sync is skipped and failure propagated."""
        metric_file = tmp_path / "metric.yaml"
        metric_file.write_text("metric:\n  name: revenue\n  type: simple\n")
        semantic_file = tmp_path / "model.yaml"
        semantic_file.write_text("semantic_model:\n  name: orders\n")

        with patch("datus.cli.generation_hooks.GenerationHooks._sync_semantic_to_db") as mock_sync:
            mock_sync.return_value = {"success": False, "error": "semantic sync failed"}
            result = generation_tools._sync_metric_to_db(str(metric_file), [str(semantic_file)])

        assert result["success"] is False
        assert result["error"] == "semantic sync failed"
        # Only called once (semantic sync), metric sync was skipped
        assert mock_sync.call_count == 1

    def test_missing_semantic_model_file_returns_failure(self, generation_tools, tmp_path):
        """When semantic_model_files contains a missing file, return failure before syncing metrics."""
        metric_file = tmp_path / "metric.yaml"
        metric_file.write_text("metric:\n  name: revenue\n")

        with patch("datus.cli.generation_hooks.GenerationHooks._sync_semantic_to_db") as mock_sync:
            mock_sync.return_value = {"success": True, "message": "ok"}
            result = generation_tools._sync_metric_to_db(str(metric_file), ["/nonexistent/model.yaml"])

        assert result["success"] is False
        assert "Semantic model file not found" in result["error"]
        mock_sync.assert_not_called()

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

    def test_exception_during_semantic_sync_returns_failure(self, generation_tools, tmp_path):
        """Exception during semantic sync is caught and returned as failure dict."""
        metric_file = tmp_path / "metric.yaml"
        metric_file.write_text("metric:\n  name: revenue\n")
        semantic_file = tmp_path / "model.yaml"
        semantic_file.write_text("semantic_model:\n  name: orders\n")

        with patch("datus.cli.generation_hooks.GenerationHooks._sync_semantic_to_db") as mock_sync:
            mock_sync.side_effect = RuntimeError("boom")
            result = generation_tools._sync_metric_to_db(str(metric_file), [str(semantic_file)])

        assert result["success"] is False
        assert "boom" in result["error"]


class TestOsiSync:
    def test_sync_osi_metric_to_db_upserts_only_metrics_declared_in_current_file(self, generation_tools, tmp_path):
        generation_tools.agent_config.current_db_config.return_value = SimpleNamespace(
            catalog="default_catalog", database="shop", schema=""
        )
        metric_file = tmp_path / "orders_metrics.yml"
        metric_file.write_text(
            "version: 0.2.0.dev0\n"
            "semantic_model:\n"
            "  - name: shop\n"
            "    datasets:\n"
            "      - name: orders\n"
            "        source: orders\n"
            "    metrics:\n"
            "      - name: order_count\n"
            "        expression:\n"
            "          dialects:\n"
            "            - dialect: ANSI_SQL\n"
            "              expression: COUNT(DISTINCT order_id)\n"
            "        custom_extensions:\n"
            "          - vendor_name: DATUS\n"
            '            data: \'{"dataset":"orders"}\'\n'
        )
        dataset = SimpleNamespace(
            name="orders",
            source=SimpleNamespace(table="orders"),
            primary_key="order_id",
            time_dimension=SimpleNamespace(name="order_date"),
            dimensions=[SimpleNamespace(name="customer_segment")],
        )
        metric = SimpleNamespace(
            name="order_count",
            description="Number of orders",
            expression="COUNT(DISTINCT order_id)",
            dataset="orders",
            subject_path=None,
            kind=None,
        )
        old_metric = SimpleNamespace(
            name="old_metric",
            description="Should not be synced from this file",
            expression="SUM(old_value)",
            dataset="orders",
            subject_path=None,
            kind=None,
        )
        doc = SimpleNamespace(datasets=[dataset], metrics=[metric, old_metric])

        with patch.object(generation_tools, "_load_osi_document", return_value=doc):
            result = generation_tools._sync_osi_metric_to_db(
                str(metric_file),
                metric_sqls={"order_count": "SELECT 1", "old_metric": "SELECT 2"},
            )

        assert result["success"] is True
        generation_tools.metric_rag.delete_artifact_rows.assert_not_called()
        generation_tools.metric_rag.delete_artifact_rows_except.assert_called_once()
        generation_tools.metric_rag.upsert_batch.assert_called_once()
        metric_objects = generation_tools.metric_rag.upsert_batch.call_args.args[0]
        assert len(metric_objects) == 1
        metric_obj = metric_objects[0]
        assert metric_obj["name"] == "order_count"
        assert metric_obj["semantic_model_name"] == "orders"
        assert metric_obj["measure_expr"] == "COUNT(DISTINCT order_id)"
        assert metric_obj["dimensions"] == ["order_date", "customer_segment"]
        assert metric_obj["entities"] == ["order_id"]
        assert metric_obj["sql"] == "SELECT 1"
        assert metric_obj["yaml_path"] == str(metric_file)
        assert result["metric_names"] == ["order_count"]

    def test_sync_osi_metric_to_db_can_upsert_without_replacing_artifact(self, generation_tools, tmp_path):
        generation_tools.agent_config.current_db_config.return_value = SimpleNamespace(
            catalog="default_catalog", database="shop", schema=""
        )
        metric_file = tmp_path / "orders_metrics.yml"
        metric_file.write_text(
            "version: 0.2.0.dev0\n"
            "semantic_model:\n"
            "  - name: shop\n"
            "    metrics:\n"
            "      - name: order_count\n"
            "        expression:\n"
            "          dialects:\n"
            "            - dialect: ANSI_SQL\n"
            "              expression: COUNT(DISTINCT order_id)\n"
        )
        dataset = SimpleNamespace(
            name="orders",
            source=SimpleNamespace(table="orders"),
            primary_key="order_id",
            time_dimension=None,
            dimensions=[],
        )
        metric = SimpleNamespace(
            name="order_count",
            description="Number of orders",
            expression="COUNT(DISTINCT order_id)",
            dataset="orders",
            subject_path=None,
            kind=None,
        )
        doc = SimpleNamespace(datasets=[dataset], metrics=[metric])

        with patch.object(generation_tools, "_load_osi_document", return_value=doc):
            result = generation_tools._sync_osi_metric_to_db(str(metric_file), replace_metric_artifact=False)

        assert result["success"] is True
        generation_tools.metric_rag.delete_artifact_rows.assert_not_called()
        generation_tools.metric_rag.delete_artifact_rows_except.assert_not_called()
        generation_tools.metric_rag.list_artifact_rows.assert_called_once_with(str(metric_file))
        generation_tools.metric_rag.upsert_batch.assert_called_once()

    def test_sync_osi_metric_partial_publish_restores_on_later_failure(self, generation_tools, tmp_path):
        generation_tools.agent_config.current_db_config.return_value = SimpleNamespace(
            catalog="default_catalog", database="shop", schema=""
        )
        metric_file = tmp_path / "orders_metrics.yml"
        metric_file.write_text(
            "version: 0.2.0.dev0\n"
            "semantic_model:\n"
            "  - name: shop\n"
            "    metrics:\n"
            "      - name: order_count\n"
            "        expression:\n"
            "          dialects:\n"
            "            - dialect: ANSI_SQL\n"
            "              expression: COUNT(DISTINCT order_id)\n"
        )
        dataset = SimpleNamespace(
            name="orders",
            source=SimpleNamespace(table="orders"),
            primary_key="order_id",
            time_dimension=None,
            dimensions=[],
        )
        metric = SimpleNamespace(
            name="order_count",
            description="Number of orders",
            expression="COUNT(DISTINCT order_id)",
            dataset="orders",
            subject_path=None,
            kind=None,
        )
        doc = SimpleNamespace(datasets=[dataset], metrics=[metric])
        generation_tools.metric_rag.list_artifact_rows.return_value = [{"id": "old-metric"}]
        generation_tools.metric_rag.create_indices.side_effect = RuntimeError("index failed")

        with patch.object(generation_tools, "_load_osi_document", return_value=doc):
            result = generation_tools._sync_osi_metric_to_db(str(metric_file), replace_metric_artifact=False)

        assert result["success"] is False
        assert "index failed" in result["error"]
        generation_tools.metric_rag.delete_artifact_rows_except.assert_not_called()
        generation_tools.metric_rag.restore_artifact_rows.assert_called_once_with(
            str(metric_file), [{"id": "old-metric"}]
        )

    def test_sync_osi_metric_to_db_includes_derived_and_joined_dimensions(self, generation_tools, tmp_path):
        generation_tools.agent_config.current_db_config.return_value = SimpleNamespace(
            catalog="default_catalog", database="shop", schema=""
        )
        metric_file = tmp_path / "orders_metrics.yml"
        metric_file.write_text(
            "version: 0.2.0.dev0\n"
            "semantic_model:\n"
            "  - name: shop\n"
            "    datasets:\n"
            "      - name: orders\n"
            "        source: orders\n"
            "    metrics:\n"
            "      - name: order_count\n"
            "      - name: order_count_prev\n"
        )
        orders = SimpleNamespace(
            name="orders",
            source=SimpleNamespace(table="orders"),
            primary_key="order_id",
            time_dimension=SimpleNamespace(name="order_date"),
            dimensions=[],
        )
        customers = SimpleNamespace(
            name="customers",
            source=SimpleNamespace(table="customers"),
            primary_key="customer_id",
            time_dimension=None,
            dimensions=[SimpleNamespace(name="region_id")],
        )
        regions = SimpleNamespace(
            name="regions",
            source=SimpleNamespace(table="regions"),
            primary_key="region_id",
            time_dimension=None,
            dimensions=[SimpleNamespace(name="region_name")],
        )
        relationships = [
            SimpleNamespace(
                **{
                    "from": "orders",
                    "to": "customers",
                    "from_columns": ["customer_id"],
                    "to_columns": ["customer_id"],
                },
            ),
            SimpleNamespace(
                **{
                    "from": "customers",
                    "to": "regions",
                    "from_columns": ["region_id"],
                    "to_columns": ["region_id"],
                },
            ),
        ]
        base_metric = SimpleNamespace(
            name="order_count",
            description="Number of orders",
            expression="COUNT(DISTINCT order_id)",
            dataset="orders",
            subject_path=None,
            kind="aggregate",
            inputs=[],
            measures=[],
        )
        derived_metric = SimpleNamespace(
            name="order_count_prev",
            description="Previous-period order count",
            expression="order_count_prev",
            dataset=None,
            subject_path=None,
            kind="derived",
            inputs=[
                SimpleNamespace(
                    name="order_count",
                    alias="order_count_prev",
                    offset_window="1 month",
                )
            ],
            measures=[],
        )
        doc = SimpleNamespace(
            datasets=[orders, customers, regions],
            relationships=relationships,
            metrics=[base_metric, derived_metric],
        )

        with patch.object(generation_tools, "_load_osi_document", return_value=doc):
            result = generation_tools._sync_osi_metric_to_db(str(metric_file))

        assert result["success"] is True
        metric_objects = generation_tools.metric_rag.upsert_batch.call_args.args[0]
        by_name = {obj["name"]: obj for obj in metric_objects}
        assert by_name["order_count"]["dimensions"] == [
            "order_date",
            "customer_id__region_id",
            "customer_id__region_id__region_name",
        ]
        assert by_name["order_count"]["entities"] == ["order_id"]
        assert by_name["order_count_prev"]["semantic_model_name"] == "orders"
        assert by_name["order_count_prev"]["dimensions"] == by_name["order_count"]["dimensions"]
        assert by_name["order_count_prev"]["entities"] == ["order_id"]

    def test_sync_osi_metric_to_db_syncs_every_semantic_file(self, generation_tools, tmp_path):
        generation_tools.agent_config.current_db_config.return_value = SimpleNamespace(
            catalog="default_catalog", database="shop", schema=""
        )
        metric_file = tmp_path / "orders_metrics.yml"
        metric_file.write_text(
            "version: 0.2.0.dev0\nsemantic_model:\n  - name: shop\n    metrics:\n      - name: order_count\n"
        )
        orders_file = tmp_path / "orders.yml"
        customers_file = tmp_path / "customers.yml"
        orders_file.write_text("datasets:\n  - name: orders\n")
        customers_file.write_text("datasets:\n  - name: customers\n")
        dataset = SimpleNamespace(
            name="orders",
            source=SimpleNamespace(table="orders"),
            primary_key="order_id",
            time_dimension=None,
            dimensions=[],
        )
        metric = SimpleNamespace(
            name="order_count",
            description="Number of orders",
            expression="COUNT(DISTINCT order_id)",
            dataset="orders",
            subject_path=None,
            kind="aggregate",
            inputs=[],
            measures=[],
        )
        doc = SimpleNamespace(datasets=[dataset], relationships=[], metrics=[metric])

        with (
            patch.object(generation_tools, "_load_osi_document", return_value=doc),
            patch.object(generation_tools, "sync_osi_semantic_to_db", return_value={"success": True}) as sync_mock,
        ):
            result = generation_tools._sync_osi_metric_to_db(
                str(metric_file),
                [str(orders_file), str(customers_file)],
            )

        assert result["success"] is True
        assert result["semantic_synced"] is True
        assert result["semantic_model_files_synced"] == [str(orders_file), str(customers_file)]
        assert [call.args[0] for call in sync_mock.call_args_list] == [str(orders_file), str(customers_file)]

    def test_sync_osi_metric_to_db_rejects_metric_file_without_metrics(self, generation_tools, tmp_path):
        metric_file = tmp_path / "empty_metrics.yml"
        metric_file.write_text(
            "version: 0.2.0.dev0\n"
            "semantic_model:\n"
            "  - name: empty\n"
            "    datasets:\n"
            "      - name: orders\n"
            "        source: orders\n"
        )

        result = generation_tools._sync_osi_metric_to_db(str(metric_file))

        assert result["success"] is False
        assert "No OSI metrics found in metric file" in result["error"]
        generation_tools.metric_rag.upsert_batch.assert_not_called()

    def test_sync_osi_semantic_to_db_upserts_only_current_dataset_columns(self, generation_tools, tmp_path):
        generation_tools.agent_config.current_db_config.return_value = SimpleNamespace(
            catalog="default_catalog", database="shop", schema=""
        )
        generation_tools.table_semantic_profile_rag = Mock()
        semantic_file = tmp_path / "orders.yml"
        semantic_file.write_text(
            "version: 0.2.0.dev0\n"
            "semantic_model:\n"
            "  - name: shop\n"
            "    datasets:\n"
            "      - name: orders\n"
            "        source: orders\n"
            "        primary_key: [order_id]\n"
        )
        dataset = SimpleNamespace(
            name="orders",
            description="Orders table",
            ai_context={
                "instructions": "Use this dataset for order-level analytics.",
                "synonyms": ["purchases"],
            },
            source=SimpleNamespace(table="orders"),
            primary_key="order_id",
            time_dimension=SimpleNamespace(name="order_date", granularity="day"),
            dimensions=[
                SimpleNamespace(
                    name="customer_segment",
                    expr="customer_segment",
                    type="categorical",
                    description="Customer segment",
                    granularity=None,
                )
            ],
        )
        other_dataset = SimpleNamespace(
            name="customers",
            description="Customers table",
            source=SimpleNamespace(table="customers"),
            primary_key="customer_id",
            time_dimension=None,
            dimensions=[],
        )
        relationship = SimpleNamespace(
            **{
                "from": "orders",
                "to": "customers",
                "from_columns": ["customer_id", "store_id"],
                "to_columns": ["customer_id", "store_id"],
            }
        )
        doc = SimpleNamespace(datasets=[dataset, other_dataset], relationships=[relationship], metrics=[])

        with patch.object(generation_tools, "_load_osi_document", return_value=doc):
            result = generation_tools.sync_osi_semantic_to_db(str(semantic_file))

        assert result["success"] is True
        generation_tools.semantic_rag.delete_artifact_rows.assert_not_called()
        generation_tools.semantic_rag.delete_artifact_rows_except.assert_called_once()
        generation_tools.semantic_rag.upsert_batch.assert_called_once()
        objects = generation_tools.semantic_rag.upsert_batch.call_args.args[0]
        assert [obj["kind"] for obj in objects] == ["table", "column", "column", "column"]
        assert objects[0]["name"] == "orders"
        assert objects[1]["name"] == "order_id"
        assert objects[1]["is_entity_key"] is True
        generation_tools.table_semantic_profile_rag.delete_artifact_rows.assert_not_called()
        generation_tools.table_semantic_profile_rag.delete_artifact_rows_except.assert_called_once()
        generation_tools.table_semantic_profile_rag.upsert_batch.assert_called_once()
        profiles = generation_tools.table_semantic_profile_rag.upsert_batch.call_args.args[0]
        assert profiles[0]["format"] == "osi"
        assert profiles[0]["dataset_name"] == "orders"
        assert profiles[0]["description"] == "Orders table"
        assert "order-level analytics" in profiles[0]["ai_context_json"]
        assert '"name": "customer_segment"' in profiles[0]["columns_json"]
        assert '"from_columns": ["customer_id", "store_id"]' in profiles[0]["relationships_json"]
        assert '"to_columns": ["customer_id", "store_id"]' in profiles[0]["relationships_json"]
        assert result["table_semantic_profiles"] == 1

    def test_sync_osi_semantic_to_db_fails_when_table_profile_sync_fails(self, generation_tools, tmp_path):
        generation_tools.agent_config.current_db_config.return_value = SimpleNamespace(
            catalog="default_catalog", database="shop", schema=""
        )
        generation_tools.table_semantic_profile_rag = Mock()
        generation_tools.table_semantic_profile_rag.upsert_batch.side_effect = RuntimeError("profile sync failed")
        semantic_file = tmp_path / "orders.yml"
        semantic_file.write_text(
            "version: 0.2.0.dev0\n"
            "semantic_model:\n"
            "  - name: shop\n"
            "    datasets:\n"
            "      - name: orders\n"
            "        source: orders\n"
            "        primary_key: [order_id]\n"
        )
        dataset = SimpleNamespace(
            name="orders",
            description="Orders table",
            ai_context={"instructions": "Use this dataset for order-level analytics."},
            source=SimpleNamespace(table="orders"),
            primary_key="order_id",
            time_dimension=None,
            dimensions=[],
        )
        doc = SimpleNamespace(datasets=[dataset], relationships=[], metrics=[])

        with patch.object(generation_tools, "_load_osi_document", return_value=doc):
            result = generation_tools.sync_osi_semantic_to_db(str(semantic_file))

        assert result["success"] is False
        assert "profile sync failed" in result["error"]
        generation_tools.semantic_rag.restore_artifact_rows.assert_called_once()
        generation_tools.table_semantic_profile_rag.restore_artifact_rows.assert_called_once()

    def test_sync_osi_to_db_routes_metric_file_through_metric_sync(self, generation_tools, tmp_path):
        # A metrics-bearing OSI doc: one pass through _sync_osi_metric_to_db (which
        # also syncs the referenced datasets); no separate semantic sync call.
        osi_file = tmp_path / "shop.yml"
        osi_file.write_text("version: 0.2.0.dev0\n")
        with (
            patch.object(generation_tools, "extract_osi_metric_names", return_value=["order_count"]),
            patch.object(
                generation_tools,
                "_sync_osi_metric_to_db",
                return_value={"success": True, "metric_artifact_ids": ["metric:a", "metric:b"]},
            ) as metric_sync,
            patch.object(generation_tools, "sync_osi_semantic_to_db") as semantic_sync,
        ):
            result = generation_tools.sync_osi_to_db(str(osi_file))

        metric_sync.assert_called_once_with(metric_file=str(osi_file), semantic_model_file=str(osi_file))
        semantic_sync.assert_not_called()
        assert result["success"] is True
        assert result["synced"] == 2

    def test_sync_osi_to_db_routes_dataset_only_file_through_semantic_sync(self, generation_tools, tmp_path):
        # A dataset-only OSI doc (no metrics): syncs just the datasets.
        osi_file = tmp_path / "model.yml"
        osi_file.write_text("version: 0.2.0.dev0\n")
        with (
            patch.object(generation_tools, "extract_osi_metric_names", return_value=[]),
            patch.object(
                generation_tools,
                "sync_osi_semantic_to_db",
                return_value={"success": True, "semantic_objects": 3},
            ) as semantic_sync,
            patch.object(generation_tools, "_sync_osi_metric_to_db") as metric_sync,
        ):
            result = generation_tools.sync_osi_to_db(str(osi_file))

        semantic_sync.assert_called_once_with(str(osi_file))
        metric_sync.assert_not_called()
        assert result["success"] is True
        assert result["synced"] == 3

    def test_sync_osi_to_db_returns_error_dict_on_unexpected_failure(self, generation_tools, tmp_path):
        # Consistent with the delegated syncs: an unexpected raise degrades to an
        # error dict rather than propagating out of the public entry.
        osi_file = tmp_path / "shop.yml"
        osi_file.write_text("version: 0.2.0.dev0\n")
        with patch.object(generation_tools, "extract_osi_metric_names", side_effect=RuntimeError("bad yaml")):
            result = generation_tools.sync_osi_to_db(str(osi_file))
        assert result == {"success": False, "error": "bad yaml"}


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


class TestRowsToDicts:
    """Tests for generation_tools._rows_to_dicts helper."""

    def test_none_returns_empty(self):
        from datus.tools.func_tool.generation_tools import _rows_to_dicts

        assert _rows_to_dicts(None) == []

    def test_list_of_dicts_returned_as_is(self):
        from datus.tools.func_tool.generation_tools import _rows_to_dicts

        rows = [{"a": 1}, {"b": 2}]
        assert _rows_to_dicts(rows) == rows

    def test_single_dict_wrapped_in_list(self):
        from datus.tools.func_tool.generation_tools import _rows_to_dicts

        assert _rows_to_dicts({"a": 1}) == [{"a": 1}]

    def test_tuple_of_dicts_returned(self):
        from datus.tools.func_tool.generation_tools import _rows_to_dicts

        rows = ({"a": 1}, {"b": 2})
        result = _rows_to_dicts(rows)
        assert result == [{"a": 1}, {"b": 2}]

    def test_non_dict_items_in_list_filtered_out(self):
        from datus.tools.func_tool.generation_tools import _rows_to_dicts

        rows = [{"a": 1}, "not_a_dict", 42, {"b": 2}]
        assert _rows_to_dicts(rows) == [{"a": 1}, {"b": 2}]

    def test_object_with_to_pylist_called(self):
        from datus.tools.func_tool.generation_tools import _rows_to_dicts

        mock_table = Mock()
        mock_table.to_pylist.return_value = [{"x": 1}]
        assert _rows_to_dicts(mock_table) == [{"x": 1}]

    def test_string_returns_empty(self):
        from datus.tools.func_tool.generation_tools import _rows_to_dicts

        assert _rows_to_dicts("some_string") == []

    def test_bytes_returns_empty(self):
        from datus.tools.func_tool.generation_tools import _rows_to_dicts

        assert _rows_to_dicts(b"bytes") == []

    def test_generator_of_dicts_consumed(self):
        from datus.tools.func_tool.generation_tools import _rows_to_dicts

        def gen():
            yield {"a": 1}
            yield {"b": 2}

        assert _rows_to_dicts(gen()) == [{"a": 1}, {"b": 2}]


class TestIsSupportedRowContainer:
    """Tests for generation_tools._is_supported_row_container helper."""

    def test_none_is_supported(self):
        from datus.tools.func_tool.generation_tools import _is_supported_row_container

        assert _is_supported_row_container(None) is True

    def test_list_is_supported(self):
        from datus.tools.func_tool.generation_tools import _is_supported_row_container

        assert _is_supported_row_container([]) is True

    def test_dict_is_supported(self):
        from datus.tools.func_tool.generation_tools import _is_supported_row_container

        assert _is_supported_row_container({}) is True

    def test_tuple_is_supported(self):
        from datus.tools.func_tool.generation_tools import _is_supported_row_container

        assert _is_supported_row_container(()) is True

    def test_object_with_to_pylist_is_supported(self):
        from datus.tools.func_tool.generation_tools import _is_supported_row_container

        mock_table = Mock()
        mock_table.to_pylist = lambda: []
        assert _is_supported_row_container(mock_table) is True

    def test_string_not_supported(self):
        from datus.tools.func_tool.generation_tools import _is_supported_row_container

        assert _is_supported_row_container("str") is False

    def test_bytes_not_supported(self):
        from datus.tools.func_tool.generation_tools import _is_supported_row_container

        assert _is_supported_row_container(b"bytes") is False

    def test_integer_not_supported(self):
        from datus.tools.func_tool.generation_tools import _is_supported_row_container

        assert _is_supported_row_container(42) is False


class TestRagScopeConditions:
    """Tests for generation_tools._rag_scope_conditions helper."""

    def test_no_method_returns_empty(self):
        from datus.tools.func_tool.generation_tools import _rag_scope_conditions

        class NoMethod:
            pass

        assert _rag_scope_conditions(NoMethod()) == []

    def test_non_callable_attribute_returns_empty(self):
        from datus.tools.func_tool.generation_tools import _rag_scope_conditions

        class WithAttr:
            _sub_agent_conditions = "not callable"

        assert _rag_scope_conditions(WithAttr()) == []

    def test_method_returning_list_returned(self):
        from datus.tools.func_tool.generation_tools import _rag_scope_conditions

        sentinel = object()
        rag = Mock()
        rag._sub_agent_conditions.return_value = [sentinel]
        result = _rag_scope_conditions(rag)
        assert result == [sentinel]

    def test_method_returning_non_list_returns_empty(self):
        from datus.tools.func_tool.generation_tools import _rag_scope_conditions

        rag = Mock()
        rag._sub_agent_conditions.return_value = "not a list"
        assert _rag_scope_conditions(rag) == []

    def test_method_raising_exception_returns_empty(self):
        from datus.tools.func_tool.generation_tools import _rag_scope_conditions

        rag = Mock()
        rag._sub_agent_conditions.side_effect = RuntimeError("boom")
        assert _rag_scope_conditions(rag) == []


class TestCheckSemanticObjectExistsCacheHit:
    """Test that the cache hit path (lines 129-130) is exercised."""

    def test_cache_hit_returns_copy(self, generation_tools):
        mock_storage = Mock()
        generation_tools.semantic_rag.storage = mock_storage
        mock_storage.search_all.return_value = [{"id": "t1", "name": "orders", "kind": "table"}]

        with patch("datus.tools.func_tool.generation_tools.And"), patch("datus.tools.func_tool.generation_tools.eq"):
            # First call populates the cache
            result1 = generation_tools.check_semantic_object_exists("orders", kind="table")
            # Second call should hit the cache
            result2 = generation_tools.check_semantic_object_exists("orders", kind="table")

        assert result1.success == result2.success
        assert result1.result == result2.result
        # Cache should have been populated
        assert len(generation_tools._semantic_object_exists_cache) >= 1


class TestEndSemanticModelGenerationCacheReset:
    """Test that end_semantic_model_generation resets caches (lines 272-273)."""

    def test_clears_caches_on_success(self, generation_tools):
        generation_tools.generation_evidence.validation_passed = True
        # Pre-populate caches
        generation_tools._semantic_object_exists_cache[("table", "orders", "")] = Mock()
        generation_tools._semantic_table_object_index = {"orders": {}}

        result = generation_tools.end_semantic_model_generation(["/path/model.yaml"])

        assert result.success == 1
        assert generation_tools._semantic_object_exists_cache == {}
        assert generation_tools._semantic_table_object_index is None


class TestEndMetricGenerationLegacyParam:
    """Test end_metric_generation with the legacy semantic_model_file param (line 319)."""

    def test_legacy_semantic_model_file_param(self, generation_tools, tmp_path):
        generation_tools.generation_evidence.validation_passed = True
        generation_tools.generation_evidence.metric_dry_run_passed = True
        generation_tools.generation_evidence.metric_dry_run_metrics.add("revenue")

        good = tmp_path / "semantic_models" / "good_metric.yml"
        good.parent.mkdir(parents=True, exist_ok=True)
        good.write_text("metric:\n  name: revenue\n  type: measure_proxy\n  type_params:\n    measure: revenue\n")
        mock_pm = Mock()
        mock_pm.subject_dir = str(tmp_path)

        with (
            patch("datus.tools.func_tool.generation_tools.get_path_manager", return_value=mock_pm),
            patch.object(generation_tools, "_sync_metric_to_db", return_value={"success": True, "message": "ok"}),
        ):
            result = generation_tools.end_metric_generation(
                metric_file=str(good),
                semantic_model_file="/some/model.yaml",  # legacy param
            )
        # The legacy param should be converted to semantic_model_files
        assert result.success == 0  # will fail because /some/model.yaml is outside sandbox


class TestEndMetricGenerationMetricSqlsFromEvidence:
    """Test that metric_sqls from generation_evidence take precedence (line 362-363)."""

    def test_uses_evidence_metric_sqls_when_present(self, generation_tools, tmp_path):
        generation_tools.generation_evidence.validation_passed = True
        generation_tools.generation_evidence.metric_dry_run_passed = True
        generation_tools.generation_evidence.metric_dry_run_metrics.add("revenue")
        generation_tools.generation_evidence.metric_sqls = {"revenue": "SELECT SUM(revenue)"}

        good = tmp_path / "semantic_models" / "good_metric.yml"
        good.parent.mkdir(parents=True, exist_ok=True)
        good.write_text("metric:\n  name: revenue\n  type: measure_proxy\n  type_params:\n    measure: revenue\n")
        mock_pm = Mock()
        mock_pm.subject_dir = str(tmp_path)
        captured = {}

        def fake_sync(metric_file, *args, **kwargs):
            captured["metric_sqls"] = kwargs.get("metric_sqls")
            return {"success": True, "message": "ok"}

        with (
            patch("datus.tools.func_tool.generation_tools.get_path_manager", return_value=mock_pm),
            patch.object(generation_tools, "_sync_metric_to_db", side_effect=fake_sync),
        ):
            result = generation_tools.end_metric_generation(metric_file=str(good))

        assert result.success == 1
        # Evidence SQLs should be visible in the returned result
        assert result.result["metric_sqls"] == {"revenue": "SELECT SUM(revenue)"}


class TestEndMetricGenerationSemanticFileSandbox:
    """Test that semantic_model_files outside sandbox is rejected (line 398)."""

    def test_rejects_semantic_file_outside_sandbox(self, generation_tools, tmp_path):
        generation_tools.generation_evidence.validation_passed = True
        generation_tools.generation_evidence.metric_dry_run_passed = True

        good = tmp_path / "semantic_models" / "good_metric.yml"
        good.parent.mkdir(parents=True, exist_ok=True)
        good.write_text("metric:\n  name: revenue\n  type: measure_proxy\n  type_params:\n    measure: revenue\n")
        mock_pm = Mock()
        mock_pm.subject_dir = str(tmp_path)

        with (
            patch("datus.tools.func_tool.generation_tools.get_path_manager", return_value=mock_pm),
            patch.object(generation_tools, "_sync_metric_to_db") as sync_mock,
        ):
            result = generation_tools.end_metric_generation(
                metric_file=str(good),
                semantic_model_files=["/outside/model.yaml"],
            )

        assert result.success == 0
        assert "Knowledge Base sandbox" in result.error
        sync_mock.assert_not_called()


class TestFilterMetricSqls:
    """Tests for GenerationTools._filter_metric_sqls static method."""

    def test_none_returns_none(self):
        from datus.tools.func_tool.generation_tools import GenerationTools

        assert GenerationTools._filter_metric_sqls(None, {"revenue"}) is None

    def test_filters_to_matching_names(self):
        from datus.tools.func_tool.generation_tools import GenerationTools

        sqls = {"revenue_total": "SELECT 1", "cost": "SELECT 2", "__combined__": "SELECT 3"}
        result = GenerationTools._filter_metric_sqls(sqls, {"revenue_total"})
        assert result == {"revenue_total": "SELECT 1"}

    def test_empty_sync_set_returns_empty(self):
        from datus.tools.func_tool.generation_tools import GenerationTools

        sqls = {"revenue": "SELECT 1"}
        result = GenerationTools._filter_metric_sqls(sqls, set())
        assert result == {}


class TestWriteFilteredMetricFile:
    """Tests for GenerationTools._write_filtered_metric_file static method."""

    def test_writes_only_matching_docs(self, tmp_path):
        from datus.tools.func_tool.generation_tools import GenerationTools

        metric_file = tmp_path / "metrics.yml"
        metric_file.write_text(
            "metric:\n  name: revenue\n  type: measure_proxy\n---\nmetric:\n  name: cost\n  type: measure_proxy\n"
        )
        temp_path = GenerationTools._write_filtered_metric_file(str(metric_file), {"revenue"})
        try:
            import yaml

            with open(temp_path, encoding="utf-8") as f:
                docs = list(yaml.safe_load_all(f))
            names = [d["metric"]["name"] for d in docs if isinstance(d, dict) and "metric" in d]
            assert names == ["revenue"]
        finally:
            import os

            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_raises_when_no_matching_docs(self, tmp_path):
        from datus.tools.func_tool.generation_tools import GenerationTools

        metric_file = tmp_path / "metrics.yml"
        metric_file.write_text("metric:\n  name: revenue\n  type: measure_proxy\n")

        with pytest.raises(ValueError, match="No matching metric definitions"):
            GenerationTools._write_filtered_metric_file(str(metric_file), {"nonexistent"})


class TestValidateMetricNameConflicts:
    """Tests for GenerationTools._validate_metric_name_conflicts."""

    def test_empty_definitions_returns_none(self, generation_tools):
        assert generation_tools._validate_metric_name_conflicts([]) is None

    def test_no_existing_metrics_returns_none(self, generation_tools):
        generation_tools.metric_rag.search_all_metrics.return_value = []
        result = generation_tools._validate_metric_name_conflicts([{"name": "revenue", "metric_type": "measure_proxy"}])
        assert result is None

    def test_non_conflicting_same_definition_returns_none(self, generation_tools):
        from unittest.mock import patch

        generation_tools.metric_rag.search_all_metrics.return_value = [
            {
                "id": "m1",
                "name": "revenue",
                "semantic_model_name": "orders",
                "metric_type": "measure_proxy",
                "measure_expr": "revenue",
                "base_measures": ["revenue"],
            }
        ]
        with patch("datus.tools.func_tool.generation_tools.metric_definition_conflict", return_value=None):
            result = generation_tools._validate_metric_name_conflicts(
                [
                    {
                        "name": "revenue",
                        "metric_type": "measure_proxy",
                        "measure_expr": "revenue",
                        "base_measures": ["revenue"],
                    }
                ]
            )
        assert result is None

    def test_conflicting_definition_returns_error_string(self, generation_tools):
        from unittest.mock import patch

        generation_tools.metric_rag.search_all_metrics.return_value = [
            {
                "id": "m1",
                "name": "revenue",
                "semantic_model_name": "orders",
                "metric_type": "measure_proxy",
                "measure_expr": "revenue",
                "base_measures": ["revenue"],
            }
        ]
        with patch("datus.tools.func_tool.generation_tools.metric_definition_conflict", return_value="metric_type"):
            result = generation_tools._validate_metric_name_conflicts([{"name": "revenue", "metric_type": "ratio"}])
        assert result == (
            "Metric name conflict within this datasource for 'revenue': "
            "existing metric id 'm1' has a different 'metric_type'. "
            "Metric names must be unique within a datasource; choose a more specific name "
            "or update the existing metric explicitly."
        )

    def test_exception_during_search_returns_none(self, generation_tools):
        generation_tools.metric_rag.search_all_metrics.side_effect = RuntimeError("storage error")
        result = generation_tools._validate_metric_name_conflicts([{"name": "revenue"}])
        assert result is None


class TestFilterMetricNames:
    """Tests for GenerationTools._filter_metric_names static method."""

    def test_none_scope_returns_all(self):
        from datus.tools.func_tool.generation_tools import GenerationTools

        assert GenerationTools._filter_metric_names(["a", "b"], None) == ["a", "b"]

    def test_filters_to_scope(self):
        from datus.tools.func_tool.generation_tools import GenerationTools

        result = GenerationTools._filter_metric_names(["revenue_total", "cost"], {"revenue_total"})
        assert result == ["revenue_total"]


class TestPublicMetricSqlNames:
    """Tests for GenerationTools._public_metric_sql_names static method."""

    def test_filters_out_dunder_names(self):
        from datus.tools.func_tool.generation_tools import GenerationTools

        sqls = {"revenue": "SELECT 1", "__query_metrics_dry_run__": "SELECT 2", "__combined__": "SELECT 3"}
        result = GenerationTools._public_metric_sql_names(sqls)
        assert result == {"revenue"}

    def test_none_returns_empty(self):
        from datus.tools.func_tool.generation_tools import GenerationTools

        assert GenerationTools._public_metric_sql_names(None) == set()

    def test_empty_name_skipped(self):
        from datus.tools.func_tool.generation_tools import GenerationTools

        assert GenerationTools._public_metric_sql_names({"": "SELECT 1"}) == set()
