# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Unit tests for GenerationTools - CI level, zero external dependencies."""

import hashlib
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

import pytest


def _bind_osi_target(
    generation_tools,
    target,
    *,
    model_name="orders_model",
    touched_metric_names=None,
    touched_dataset_names=None,
):
    from datus.tools.func_tool.osi_target_tools import OsiSemanticModelTargetState

    state = OsiSemanticModelTargetState()
    state.select(
        {
            "semantic_model_name": model_name,
            "semantic_model_file": f"subject/semantic_models/warehouse/{target.name}",
            "absolute_path": str(target.resolve()),
            "artifact_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "table_references": ["analytics.orders"],
        },
        mode="bound",
    )
    state.touched_metric_names = list(touched_metric_names or [])
    state.touched_dataset_names = list(touched_dataset_names or [])
    generation_tools.authoring_format = "osi"
    generation_tools.osi_target_state = state
    generation_tools.require_bound_osi_target = True
    return state


def _plan_osi_target(generation_tools, target, *, model_name):
    from datus.tools.func_tool.osi_target_tools import OsiSemanticModelTargetState

    state = OsiSemanticModelTargetState()
    state.select(
        {
            "semantic_model_name": model_name,
            "semantic_model_file": str(target),
            "absolute_path": str(target.resolve()),
            "artifact_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        },
        mode="planned",
    )
    generation_tools.authoring_format = "osi"
    generation_tools.osi_target_state = state
    return state


def _record_compiled_evidence(generation_tools, target, metric_names):
    descriptions = [
        {
            "name": name,
            "kind": "aggregate",
            "datasets": ["orders"],
            "measures": [name],
        }
        for name in metric_names
    ]
    assert generation_tools.generation_evidence.record_compiled_validation(
        {
            "contract_digest": "sha256:" + hashlib.sha256(b"contract").hexdigest(),
            "artifact_sha256": {str(target.resolve()): "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()},
            "compiled_metrics": descriptions,
            "compiled_metric_digests": {
                name: "sha256:" + hashlib.sha256(name.encode()).hexdigest() for name in metric_names
            },
        },
        metric_names=metric_names,
    )
    return {item["name"]: item for item in descriptions}


@pytest.fixture
def mock_agent_config():
    return Mock()


@pytest.fixture
def generation_tools(mock_agent_config):
    with (
        patch("datus.tools.func_tool.generation_tools.MetricRAG") as mock_metric_rag_cls,
        patch("datus.tools.func_tool.generation_tools.SemanticDatasetRAG") as mock_semantic_rag_cls,
    ):
        mock_metric_rag = Mock()
        mock_semantic_rag = Mock()
        mock_metric_rag_cls.return_value = mock_metric_rag
        mock_semantic_rag_cls.return_value = mock_semantic_rag

        from datus.tools.func_tool.generation_tools import GenerationTools

        tool = GenerationTools(agent_config=mock_agent_config)
        tool.metric_rag = mock_metric_rag
        tool.semantic_dataset_rag = mock_semantic_rag
        return tool


class TestAvailableTools:
    def test_returns_four_tools(self, generation_tools):
        with patch("datus.tools.func_tool.generation_tools.trans_to_function_tool") as mock_trans:
            mock_trans.side_effect = lambda f: Mock(name=f.__name__)
            tools = generation_tools.available_tools()
        assert len(tools) == 4


class TestCompiledMetricCatalog:
    def test_uses_shared_adapter_catalog_paging(self, generation_tools, tmp_path):
        calls = []

        async def list_metrics(*, limit, offset):
            calls.append((limit, offset))
            return [SimpleNamespace(name="revenue")] if offset == 0 else []

        generation_tools.agent_config.resolve_semantic_adapter.return_value = "dosi"
        generation_tools.agent_config.build_semantic_adapter_config.return_value = None
        adapter = SimpleNamespace(list_metrics=list_metrics)

        with (
            patch("datus.tools.semantic_tools.registry.semantic_adapter_registry.get_metadata", return_value=None),
            patch(
                "datus.tools.semantic_tools.registry.semantic_adapter_registry.create_adapter",
                return_value=adapter,
            ),
            patch("datus.tools.semantic_tools.paging.metric_catalog_paging", return_value=(5000, 2)),
        ):
            catalog = generation_tools._compiled_metric_catalog(str(tmp_path / "model.yml"))

        assert list(catalog) == ["revenue"]
        assert calls == [(5000, 0)]


class TestCheckSemanticObjectExists:
    def test_osi_bound_yaml_takes_precedence_over_stale_rag(self, generation_tools, tmp_path):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "orders.yml"
        target.parent.mkdir(parents=True)
        target.write_text(
            "version: 0.2.0\n"
            "semantic_model:\n"
            "  - name: orders_model\n"
            "    datasets:\n"
            "      - name: orders\n"
            "        source: analytics.orders\n"
            "    metrics:\n"
            "      - name: live_revenue\n",
            encoding="utf-8",
        )
        _bind_osi_target(generation_tools, target)
        generation_tools.metric_rag.storage.search_all.return_value = [{"name": "stale_revenue"}]

        live_result = generation_tools.check_semantic_object_exists("live_revenue", kind="metric")
        stale_result = generation_tools.check_semantic_object_exists("stale_revenue", kind="metric")

        assert live_result.success == 1
        assert live_result.result["exists"] is True
        assert stale_result.success == 1
        assert stale_result.result["exists"] is False
        generation_tools.metric_rag.storage.search_all.assert_not_called()

    def test_table_found(self, generation_tools):
        mock_storage = Mock()
        generation_tools.semantic_dataset_rag.storage = mock_storage
        mock_storage._search_all.return_value = [
            {"id": "t1", "name": "orders", "kind": "dataset", "dataset_name": "orders", "source_table": "orders"}
        ]

        with patch("datus.tools.func_tool.generation_tools.And"), patch("datus.tools.func_tool.generation_tools.eq"):
            result = generation_tools.check_semantic_object_exists("orders", kind="table")

        assert result.success == 1
        assert result.result["exists"] is True
        assert result.result["name"] == "orders"

    def test_accepts_prompt_documented_name_argument(self, generation_tools):
        mock_storage = Mock()
        generation_tools.semantic_dataset_rag.storage = mock_storage
        mock_storage._search_all.return_value = [
            {"id": "t1", "name": "orders", "kind": "dataset", "dataset_name": "orders", "source_table": "orders"}
        ]

        with patch("datus.tools.func_tool.generation_tools.And"), patch("datus.tools.func_tool.generation_tools.eq"):
            result = generation_tools.check_semantic_object_exists(name="orders", kind="table")

        assert result.success == 1
        assert result.result["exists"] is True

    def test_accepts_legacy_object_name_argument(self, generation_tools):
        mock_storage = Mock()
        generation_tools.semantic_dataset_rag.storage = mock_storage
        mock_storage._search_all.return_value = [
            {"id": "t1", "name": "orders", "kind": "dataset", "dataset_name": "orders", "source_table": "orders"}
        ]

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
        generation_tools.semantic_dataset_rag.storage = mock_storage
        mock_storage._search_all.return_value = []

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
        generation_tools.semantic_dataset_rag.storage = mock_storage
        mock_storage._search_all.return_value = [
            {"id": "c1", "name": "amount", "dataset_name": "orders", "kind": "field"}
        ]

        result = generation_tools.check_semantic_object_exists("orders.amount", kind="column", table_context="orders")

        assert result.success == 1
        assert result.result["exists"] is True

    def test_column_not_found(self, generation_tools):
        mock_storage = Mock()
        generation_tools.semantic_dataset_rag.storage = mock_storage
        mock_storage._search_all.return_value = []

        result = generation_tools.check_semantic_object_exists("orders.nonexistent", kind="column")

        assert result.success == 1
        assert result.result["exists"] is False

    def test_column_name_match_without_table(self, generation_tools):
        mock_storage = Mock()
        generation_tools.semantic_dataset_rag.storage = mock_storage
        mock_storage._search_all.return_value = [
            {"id": "c1", "name": "amount", "dataset_name": "orders", "kind": "field"}
        ]

        result = generation_tools.check_semantic_object_exists("amount", kind="column")

        assert result.success == 1
        assert result.result["exists"] is True

    def test_dotted_name_extracts_target(self, generation_tools):
        mock_storage = Mock()
        generation_tools.semantic_dataset_rag.storage = mock_storage
        mock_storage._search_all.return_value = [
            {"id": "t1", "name": "orders", "kind": "dataset", "dataset_name": "orders", "source_table": "orders"}
        ]

        with patch("datus.tools.func_tool.generation_tools.And"), patch("datus.tools.func_tool.generation_tools.eq"):
            result = generation_tools.check_semantic_object_exists("public.orders", kind="table")

        assert result.success == 1

    def test_legacy_wrapper(self, generation_tools):
        mock_storage = Mock()
        generation_tools.semantic_dataset_rag.storage = mock_storage
        mock_storage._search_all.return_value = []

        with patch("datus.tools.func_tool.generation_tools.And"), patch("datus.tools.func_tool.generation_tools.eq"):
            result = generation_tools.check_semantic_model_exists("orders")

        assert result.success == 1


class TestEndSemanticModelGeneration:
    def test_non_dosi_publish_is_rejected_before_any_sync(self, generation_tools):
        """Contract: semantic authoring is Dosi-only — publishing from a
        MetricFlow/plain-OSI project must fail with the query-only message
        without touching the KB."""
        generation_tools.authoring_format = "metricflow"
        generation_tools.generation_evidence.validation_passed = True

        with patch.object(generation_tools, "sync_osi_to_db") as sync_mock:
            result = generation_tools.publish_semantic_model(["/path/to/model.yaml"])

        assert result.success == 0
        assert "query-only" in result.error
        sync_mock.assert_not_called()

    def test_osi_accepts_revised_query_backed_source_before_sync(self, generation_tools, tmp_path):
        model_file = tmp_path / "subject" / "semantic_models" / "warehouse" / "daily_sales.yml"
        model_file.parent.mkdir(parents=True)
        model_file.write_text(
            "version: 0.2.0\n"
            "semantic_model:\n"
            "  - name: daily_sales\n"
            "    datasets:\n"
            "      - name: daily_sales\n"
            "        source: SELECT sale_date, SUM(amount) AS revenue FROM sales GROUP BY sale_date\n",
            encoding="utf-8",
        )
        _plan_osi_target(generation_tools, model_file, model_name="daily_sales")
        generation_tools.generation_evidence.record_semantic_artifact_validation("daily_sales", model_file)
        mock_pm = Mock(
            subject_dir=model_file.parents[2],
            semantic_model_path=Mock(return_value=model_file.parent),
        )

        with (
            patch("datus.tools.func_tool.generation_tools.get_path_manager", return_value=mock_pm),
            patch.object(generation_tools, "sync_osi_to_db", return_value={"success": True}) as sync_mock,
        ):
            result = generation_tools.publish_semantic_model([str(model_file)])

        assert result.success == 1
        sync_mock.assert_called_once_with(str(model_file), include_semantic_objects=True, include_metrics=False)

    def test_osi_requires_validation_for_the_exact_target_artifact(self, generation_tools, tmp_path):
        sales_file = tmp_path / "semantic_models" / "warehouse" / "sales.yml"
        finance_file = tmp_path / "semantic_models" / "warehouse" / "finance.yml"
        sales_file.parent.mkdir(parents=True)
        sales_file.write_text(
            "version: 0.2.0.dev0\nsemantic_model:\n  - name: sales\n    datasets: []\n",
            encoding="utf-8",
        )
        finance_file.write_text(
            "version: 0.2.0.dev0\nsemantic_model:\n  - name: finance\n    datasets: []\n",
            encoding="utf-8",
        )
        generation_tools.generation_evidence.validation_passed = True
        generation_tools.generation_evidence.record_semantic_artifact_validation("sales", sales_file)
        _plan_osi_target(generation_tools, finance_file, model_name="finance")
        mock_pm = Mock(subject_dir=str(tmp_path))

        with (
            patch("datus.tools.func_tool.generation_tools.get_path_manager", return_value=mock_pm),
            patch.object(generation_tools, "sync_osi_to_db") as sync_mock,
        ):
            result = generation_tools.publish_semantic_model([str(finance_file)])

        assert result.success == 0
        assert "exact semantic model artifact" in result.error
        sync_mock.assert_not_called()

    def test_osi_publishes_the_exact_validated_artifact(self, generation_tools, tmp_path):
        model_file = tmp_path / "semantic_models" / "warehouse" / "finance.yml"
        model_file.parent.mkdir(parents=True)
        model_file.write_text(
            "version: 0.2.0.dev0\nsemantic_model:\n  - name: finance\n    datasets: []\n",
            encoding="utf-8",
        )
        _plan_osi_target(generation_tools, model_file, model_name="finance")
        generation_tools.generation_evidence.record_semantic_artifact_validation("finance", model_file)
        mock_pm = Mock(subject_dir=str(tmp_path))

        with (
            patch("datus.tools.func_tool.generation_tools.get_path_manager", return_value=mock_pm),
            patch.object(
                generation_tools,
                "sync_osi_to_db",
                return_value={"success": True},
            ) as sync_mock,
        ):
            result = generation_tools.publish_semantic_model([str(model_file)])

        assert result.success == 1
        sync_mock.assert_called_once_with(str(model_file), include_semantic_objects=True, include_metrics=False)

    def test_osi_rejects_file_whose_model_name_differs_from_plan(self, generation_tools, tmp_path):
        model_file = tmp_path / "semantic_models" / "warehouse" / "orders.yml"
        model_file.parent.mkdir(parents=True)
        model_file.write_text("semantic_model:\n  - name: finance\n    datasets: []\n", encoding="utf-8")
        _plan_osi_target(generation_tools, model_file, model_name="orders")
        generation_tools.generation_evidence.record_semantic_artifact_validation("finance", model_file)
        mock_pm = Mock(subject_dir=str(tmp_path))

        with (
            patch("datus.tools.func_tool.generation_tools.get_path_manager", return_value=mock_pm),
            patch.object(generation_tools, "sync_osi_to_db") as sync_mock,
        ):
            result = generation_tools.publish_semantic_model([str(model_file)])

        assert result.success == 0
        assert "planned OSI target" in result.error
        sync_mock.assert_not_called()


class TestEndMetricGeneration:
    def _mark_ready_to_publish(self, generation_tools):
        generation_tools.generation_evidence.validation_passed = True

    def test_non_dosi_metric_publish_is_rejected_before_any_sync(self, generation_tools):
        """Contract: metric authoring is Dosi-only — publishing from a
        MetricFlow/plain-OSI project must fail with the query-only message
        without touching the KB."""
        generation_tools.authoring_format = "metricflow"
        self._mark_ready_to_publish(generation_tools)

        with patch.object(generation_tools, "_sync_osi_metric_to_db") as sync_mock:
            result = generation_tools.publish_metrics(metric_file="/path/semantic_models/metric.yaml")

        assert result.success == 0
        assert "query-only" in result.error
        sync_mock.assert_not_called()

    def test_osi_rejects_publish_without_bound_target(self, generation_tools):
        from datus.tools.func_tool.osi_target_tools import OsiSemanticModelTargetState

        generation_tools.authoring_format = "osi"
        generation_tools.osi_target_state = OsiSemanticModelTargetState()
        generation_tools.require_bound_osi_target = True

        result = generation_tools.publish_metrics(metric_file="subject/semantic_models/warehouse/orders.yml")

        assert result.success == 0
        assert result.result["code"] == "semantic_model_required"

    def test_osi_rejects_publish_path_other_than_bound_target(self, generation_tools, tmp_path):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "orders.yml"
        wrong_target = target.with_name("customers.yml")
        target.parent.mkdir(parents=True)
        target.write_text("semantic_model:\n  - name: orders_model\n    metrics: []\n", encoding="utf-8")
        wrong_target.write_text("semantic_model:\n  - name: customers_model\n    metrics: []\n", encoding="utf-8")
        _bind_osi_target(generation_tools, target, touched_metric_names=["order_count"])
        mock_pm = Mock(subject_dir=str(tmp_path / "subject"))

        with (
            patch("datus.tools.func_tool.generation_tools.get_path_manager", return_value=mock_pm),
            patch.object(generation_tools, "_sync_osi_metric_to_db") as sync_mock,
        ):
            result = generation_tools.publish_metrics(metric_file=str(wrong_target))

        assert result.success == 0
        assert result.result["code"] == "semantic_model_target_invalid"
        sync_mock.assert_not_called()

    def test_osi_publishes_exact_bound_target(self, generation_tools, tmp_path):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "orders.yml"
        target.parent.mkdir(parents=True)
        target.write_text(
            "semantic_model:\n"
            "  - name: orders_model\n"
            "    datasets:\n"
            "      - name: orders\n"
            "        source: analytics.orders\n"
            "    metrics:\n"
            "      - name: order_count\n",
            encoding="utf-8",
        )
        state = _bind_osi_target(generation_tools, target, touched_metric_names=["order_count"])
        state.record_artifact_snapshot(target, b"pre-authoring")
        generation_tools.generation_evidence.record_semantic_artifact_validation("orders_model", target)
        _record_compiled_evidence(generation_tools, target, ["order_count"])
        mock_pm = Mock(subject_dir=str(tmp_path / "subject"))

        with (
            patch("datus.tools.func_tool.generation_tools.get_path_manager", return_value=mock_pm),
            patch.object(
                generation_tools,
                "_sync_osi_metric_to_db",
                return_value={"success": True, "message": "synced"},
            ) as sync_mock,
        ):
            result = generation_tools.publish_metrics(metric_file=str(target))

        assert result.success == 1
        sync_mock.assert_called_once_with(
            str(target),
            [],
            {},
            metric_names_to_sync={"order_count"},
            metric_names_to_reconcile={"order_count"},
            compiled_metric_catalog=ANY,
        )
        assert result.result["metric_file"] == str(target)
        assert generation_tools.generation_evidence.has_metric_kb_sync(["order_count"])
        assert state.artifact_snapshot_content is None

    def test_osi_publishes_dataset_repairs_with_touched_metrics(self, generation_tools, tmp_path):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "orders.yml"
        target.parent.mkdir(parents=True)
        target.write_text(
            "semantic_model:\n"
            "  - name: orders_model\n"
            "    datasets:\n"
            "      - name: orders\n"
            "        source: analytics.orders\n"
            "    metrics:\n"
            "      - name: order_count\n",
            encoding="utf-8",
        )
        _bind_osi_target(
            generation_tools,
            target,
            touched_metric_names=["order_count"],
            touched_dataset_names=["orders"],
        )
        generation_tools.generation_evidence.record_semantic_artifact_validation("orders_model", target)
        _record_compiled_evidence(generation_tools, target, ["order_count"])
        mock_pm = Mock(subject_dir=str(tmp_path / "subject"))

        with (
            patch("datus.tools.func_tool.generation_tools.get_path_manager", return_value=mock_pm),
            patch.object(
                generation_tools,
                "_sync_osi_metric_to_db",
                return_value={"success": True, "message": "synced", "semantic_synced": True},
            ) as sync_mock,
        ):
            result = generation_tools.publish_metrics(metric_file=str(target))

        assert result.success == 1
        sync_mock.assert_called_once_with(
            str(target),
            [str(target)],
            {},
            metric_names_to_sync={"order_count"},
            metric_names_to_reconcile={"order_count"},
            compiled_metric_catalog=ANY,
        )
        assert generation_tools.generation_evidence.semantic_kb_sync_passed is True

    def test_osi_publishes_pure_deletion(self, generation_tools, tmp_path):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "orders.yml"
        target.parent.mkdir(parents=True)
        target.write_text(
            "semantic_model:\n"
            "  - name: orders_model\n"
            "    datasets:\n"
            "      - name: orders\n"
            "        source: analytics.orders\n"
            "    metrics: []\n",
            encoding="utf-8",
        )
        state = _bind_osi_target(
            generation_tools,
            target,
            touched_metric_names=["old_metric", "already_missing"],
        )
        state.record_artifact_snapshot(target, b"pre-deletion")
        generation_tools.generation_evidence.record_semantic_artifact_validation("orders_model", target)
        _record_compiled_evidence(generation_tools, target, [])
        mock_pm = Mock(subject_dir=str(tmp_path / "subject"))

        with (
            patch("datus.tools.func_tool.generation_tools.get_path_manager", return_value=mock_pm),
            patch.object(
                generation_tools,
                "_sync_osi_metric_to_db",
                return_value={"success": True, "message": "deleted"},
            ) as sync_mock,
        ):
            result = generation_tools.publish_metrics(metric_file=str(target))

        assert result.success == 1
        sync_mock.assert_called_once_with(
            str(target),
            [],
            {},
            metric_names_to_sync=set(),
            metric_names_to_reconcile={"old_metric", "already_missing"},
            compiled_metric_catalog={},
        )
        assert generation_tools.generation_evidence.has_metric_kb_sync(["old_metric", "already_missing"])
        assert state.artifact_snapshot_content is None

    def test_osi_rejects_poisoned_target_after_failed_rebind(self, generation_tools, tmp_path):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "orders.yml"
        target.parent.mkdir(parents=True)
        target.write_text("semantic_model:\n  - name: orders_model\n    metrics:\n      - name: order_count\n")
        state = _bind_osi_target(generation_tools, target, touched_metric_names=["order_count"])
        state.last_error_code = "semantic_model_target_invalid"
        generation_tools.generation_evidence.record_semantic_artifact_validation("orders_model", target)
        mock_pm = Mock(subject_dir=str(tmp_path / "subject"))

        with (
            patch("datus.tools.func_tool.generation_tools.get_path_manager", return_value=mock_pm),
            patch.object(generation_tools, "_sync_osi_metric_to_db") as sync_mock,
        ):
            result = generation_tools.publish_metrics(metric_file=str(target))

        assert result.success == 0
        assert result.result["code"] == "semantic_model_target_invalid"
        assert "failed bind" in result.error
        sync_mock.assert_not_called()

    def test_osi_treats_touched_name_absent_from_final_yaml_as_deletion(self, generation_tools, tmp_path):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "orders.yml"
        target.parent.mkdir(parents=True)
        target.write_text("semantic_model:\n  - name: orders_model\n    metrics:\n      - name: replacement_metric\n")
        _bind_osi_target(generation_tools, target, touched_metric_names=["order_count"])
        generation_tools.generation_evidence.record_semantic_artifact_validation("orders_model", target)
        _record_compiled_evidence(generation_tools, target, [])
        mock_pm = Mock(subject_dir=str(tmp_path / "subject"))

        with (
            patch("datus.tools.func_tool.generation_tools.get_path_manager", return_value=mock_pm),
            patch.object(
                generation_tools,
                "_sync_osi_metric_to_db",
                return_value={"success": True, "message": "reconciled"},
            ) as sync_mock,
        ):
            result = generation_tools.publish_metrics(metric_file=str(target))

        assert result.success == 1
        sync_mock.assert_called_once_with(
            str(target),
            [],
            {},
            metric_names_to_sync=set(),
            metric_names_to_reconcile={"order_count"},
            compiled_metric_catalog={},
        )

    def test_osi_rejects_unrelated_validation_evidence(self, generation_tools, tmp_path):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "orders.yml"
        unrelated = target.with_name("finance.yml")
        target.parent.mkdir(parents=True)
        target.write_text("semantic_model:\n  - name: orders_model\n    metrics:\n      - name: order_count\n")
        unrelated.write_text("semantic_model:\n  - name: finance\n")
        _bind_osi_target(generation_tools, target, touched_metric_names=["order_count"])
        generation_tools.generation_evidence.validation_passed = True
        generation_tools.generation_evidence.record_semantic_artifact_validation("finance", unrelated)
        mock_pm = Mock(subject_dir=str(tmp_path / "subject"))

        with (
            patch("datus.tools.func_tool.generation_tools.get_path_manager", return_value=mock_pm),
            patch.object(generation_tools, "_sync_osi_metric_to_db") as sync_mock,
        ):
            result = generation_tools.publish_metrics(metric_file=str(target))

        assert result.success == 0
        assert "exact bound OSI" in result.error
        sync_mock.assert_not_called()

    def test_osi_publishes_without_bound_target_requirement(self, generation_tools, tmp_path):
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
            patch.object(
                generation_tools,
                "_sync_osi_metric_to_db",
                return_value={"success": True, "message": "synced"},
            ) as sync_mock,
        ):
            result = generation_tools.publish_metrics(metric_file=str(metric_file))

        assert result.success == 1
        sync_mock.assert_called_once_with(
            str(metric_file),
            [],
            {},
            metric_names_to_sync={"order_count"},
        )

    def test_osi_ignores_mutated_semantic_files_to_avoid_duplicate_sync(self, generation_tools, tmp_path):
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
        generation_tools.generation_evidence.record_artifact_mutation(orders_file)
        generation_tools.generation_evidence.record_artifact_mutation(customers_file)
        self._mark_ready_to_publish(generation_tools)
        mock_pm = Mock()
        mock_pm.subject_dir = str(tmp_path)

        with (
            patch("datus.tools.func_tool.generation_tools.get_path_manager", return_value=mock_pm),
            patch.object(
                generation_tools,
                "_sync_osi_metric_to_db",
                return_value={"success": True, "message": "synced"},
            ) as sync_mock,
        ):
            result = generation_tools.publish_metrics(metric_file=str(metric_file))

        assert result.success == 1
        sync_mock.assert_called_once_with(
            str(metric_file),
            [],
            {},
            metric_names_to_sync={"order_count"},
        )
        assert result.result["semantic_model_files"] == []
        assert generation_tools.generation_evidence.semantic_kb_sync_passed is False


class TestOsiSync:
    def test_native_dosi_window_is_projected_as_window_metric(self, generation_tools):
        metric = SimpleNamespace(
            name="running_revenue",
            description="Running revenue",
            expression="SUM(orders.amount)",
            dataset=None,
            subject_path=["sales", "revenue", "running"],
            kind="aggregate",
            inputs=[],
            measures=[],
            window={"type": "cumulative", "function": "sum"},
        )
        doc = SimpleNamespace(name="shop", datasets=[], relationships=[], metrics=[metric])

        (row,) = generation_tools._build_osi_metric_objects(
            doc=doc,
            metric_file="model.yml",
            target_metric_names={"running_revenue"},
        )

        assert row["metric_type"] == "window"

    def test_metric_rows_use_compiled_adapter_semantics(self, generation_tools):
        metric = SimpleNamespace(
            name="average_order_value",
            description="Average order value",
            expression="SUM(orders.amount) / COUNT(*)",
            dataset="guessed_dataset",
            datasets=["guessed_dataset"],
            subject_path=["sales", "orders"],
            kind=None,
            inputs=[],
            measures=[],
            window=None,
        )
        orders = SimpleNamespace(
            name="orders",
            primary_key=["order_id"],
            time_dimension=None,
            dimensions=[SimpleNamespace(name="status")],
        )
        doc = SimpleNamespace(name="shop", datasets=[orders], relationships=[], metrics=[metric])
        compiled = SimpleNamespace(
            name="average_order_value",
            type="expression",
            dimensions=["orders.status", "orders.order_date"],
            measures=["order_amount", "order_count"],
            metadata={"datasets": ["orders"]},
        )

        with patch.object(
            generation_tools,
            "_compiled_metric_catalog",
            return_value={"average_order_value": compiled},
        ):
            (row,) = generation_tools._build_osi_metric_objects(
                doc=doc,
                metric_file="model.yml",
                target_metric_names={"average_order_value"},
            )

        assert row["metric_type"] == "expression"
        assert row["dimensions"] == ["orders.status", "orders.order_date"]
        assert row["entities"] == ["order_id"]
        assert row["base_measures"] == ["order_amount", "order_count"]

    def test_preserve_existing_metric_sql_keeps_only_compatible_sql(self, generation_tools):
        metric_objects = [
            {"name": "revenue", "measure_expr": "SUM(revenue)", "sql": "SELECT fresh"},
            {"name": "order_count", "measure_expr": "COUNT(*)", "sql": ""},
            {"name": "customer_count", "measure_expr": "COUNT(DISTINCT customer_id)", "sql": ""},
        ]

        generation_tools._preserve_existing_metric_sql(
            metric_objects,
            [
                {"name": "Revenue", "measure_expr": "SUM(revenue)", "sql": "SELECT stale"},
                {"name": "ORDER_COUNT", "measure_expr": "COUNT(*)", "sql": "SELECT preserved"},
                {"name": "customer_count", "measure_expr": "COUNT(*)", "sql": "SELECT obsolete"},
            ],
        )

        assert metric_objects == [
            {"name": "revenue", "measure_expr": "SUM(revenue)", "sql": "SELECT fresh"},
            {"name": "order_count", "measure_expr": "COUNT(*)", "sql": "SELECT preserved"},
            {"name": "customer_count", "measure_expr": "COUNT(DISTINCT customer_id)", "sql": ""},
        ]

    def test_sync_osi_metric_to_db_reconciles_absent_touched_names_without_upsert(self, generation_tools, tmp_path):
        metric_file = tmp_path / "orders_metrics.yml"
        metric_file.write_text(
            "version: 0.2.0.dev0\nsemantic_model:\n  - name: shop\n    datasets: []\n    metrics: []\n",
            encoding="utf-8",
        )
        doc = SimpleNamespace(name="shop", datasets=[], metrics=[])

        with patch.object(generation_tools, "_load_osi_document", return_value=doc):
            result = generation_tools._sync_osi_metric_to_db(
                str(metric_file),
                metric_names_to_sync=set(),
                metric_names_to_reconcile={"old_metric"},
            )

        assert result["success"] is True
        assert result["metric_names"] == []
        assert result["deleted_metric_names"] == ["old_metric"]
        generation_tools.metric_rag.upsert_batch.assert_not_called()
        generation_tools.metric_rag.create_indices.assert_not_called()
        generation_tools.metric_rag.delete_artifact_rows_except.assert_called_once_with(str(metric_file), [])

    def test_sync_osi_metric_full_replacement_restores_snapshot_on_failure(self, generation_tools, tmp_path):
        metric_file = tmp_path / "orders_metrics.yml"
        metric_file.write_text(
            "semantic_model:\n  - name: shop\n    datasets: []\n    metrics: []\n",
            encoding="utf-8",
        )
        snapshot = [{"id": "metric:old_metric", "name": "old_metric", "yaml_path": str(metric_file)}]
        generation_tools.metric_rag.list_artifact_rows.return_value = snapshot
        generation_tools.metric_rag.delete_artifact_rows_except.side_effect = RuntimeError("delete failed")
        doc = SimpleNamespace(name="shop", datasets=[], metrics=[])

        with patch.object(generation_tools, "_load_osi_document", return_value=doc):
            result = generation_tools._sync_osi_metric_to_db(str(metric_file))

        assert result["success"] is False
        assert "delete failed" in result["error"]
        generation_tools.metric_rag.restore_artifact_rows.assert_called_once_with(str(metric_file), snapshot)

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
        doc = SimpleNamespace(name="shop", datasets=[dataset], metrics=[metric, old_metric])

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
        assert metric_obj["semantic_model_name"] == "shop"
        assert metric_obj["measure_expr"] == "COUNT(DISTINCT order_id)"
        assert metric_obj["dimensions"] == ["order_date", "customer_segment"]
        assert metric_obj["entities"] == ["order_id"]
        assert metric_obj["sql"] == "SELECT 1"
        assert metric_obj["yaml_path"] == str(metric_file)
        assert result["metric_names"] == ["order_count"]

    def test_sync_osi_metric_scope_does_not_overwrite_existing_metric_sql(self, generation_tools, tmp_path):
        generation_tools.agent_config.current_db_config.return_value = SimpleNamespace(
            catalog="default_catalog", database="shop", schema=""
        )
        metric_file = tmp_path / "orders_metrics.yml"
        metric_file.write_text(
            "version: 0.2.0.dev0\n"
            "semantic_model:\n"
            "  - name: shop\n"
            "    metrics:\n"
            "      - name: existing_metric\n"
            "      - name: new_metric\n"
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
        existing_metric = SimpleNamespace(
            name="existing_metric",
            description="Existing metric",
            expression="SUM(existing_value)",
            dataset="orders",
            subject_path=None,
            kind=None,
        )
        new_metric = SimpleNamespace(
            name="new_metric",
            description="New metric",
            expression="COUNT(DISTINCT order_id)",
            dataset="orders",
            subject_path=None,
            kind=None,
        )
        doc = SimpleNamespace(name="shop", datasets=[dataset], metrics=[existing_metric, new_metric])

        with patch.object(generation_tools, "_load_osi_document", return_value=doc):
            result = generation_tools._sync_osi_metric_to_db(
                str(metric_file),
                metric_sqls={"new_metric": "SELECT new_metric"},
                metric_names_to_sync={"new_metric"},
            )

        assert result["success"] is True
        metric_objects = generation_tools.metric_rag.upsert_batch.call_args.args[0]
        assert [metric["name"] for metric in metric_objects] == ["new_metric"]
        assert metric_objects[0]["sql"] == "SELECT new_metric"
        generation_tools.metric_rag.delete_artifact_rows.assert_not_called()
        generation_tools.metric_rag.delete_artifact_rows_except.assert_called_once()
        assert generation_tools.metric_rag.delete_artifact_rows_except.call_args.args[0] == str(metric_file)
        assert set(generation_tools.metric_rag.delete_artifact_rows_except.call_args.args[1]) == {
            "metric:existing_metric",
            "metric:new_metric",
        }
        generation_tools.metric_rag.list_artifact_rows.assert_called_once_with(str(metric_file))

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
            result = generation_tools._sync_osi_metric_to_db(
                str(metric_file),
                metric_names_to_sync={"order_count"},
            )

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
                    "name": "customer",
                    "from": "orders",
                    "to": "customers",
                    "from_columns": ["customer_id"],
                    "to_columns": ["customer_id"],
                },
            ),
            SimpleNamespace(
                **{
                    "name": "region",
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
            name="shop",
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
            "customer__region_id",
            "customer__region__region_name",
        ]
        assert by_name["order_count"]["entities"] == ["order_id"]
        assert by_name["order_count_prev"]["semantic_model_name"] == "shop"
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
            patch.object(
                generation_tools,
                "_sync_osi_semantic_objects_to_db",
                side_effect=[
                    {
                        "success": True,
                        "semantic_objects": [{"id": "table:shop:orders"}],
                        "semantic_dataset_rows": [],
                    },
                    {
                        "success": True,
                        "semantic_objects": [{"id": "table:shop:customers"}],
                        "semantic_dataset_rows": [],
                    },
                ],
            ) as sync_mock,
        ):
            result = generation_tools._sync_osi_metric_to_db(
                str(metric_file),
                [str(orders_file), str(customers_file)],
            )

        assert result["success"] is True
        assert result["semantic_synced"] is True
        assert result["semantic_model_files_synced"] == [str(orders_file), str(customers_file)]
        assert [call.args[0] for call in sync_mock.call_args_list] == [str(orders_file), str(customers_file)]
        assert all(call.kwargs == {"prepare_only": True} for call in sync_mock.call_args_list)

    def test_sync_osi_metric_to_db_restores_semantic_profile_and_metric_rows_on_metric_failure(
        self, generation_tools, tmp_path
    ):
        metric_file = tmp_path / "orders_metrics.yml"
        metric_file.write_text(
            "version: 0.2.0.dev0\nsemantic_model:\n  - name: shop\n    metrics:\n      - name: order_count\n"
        )
        semantic_file = tmp_path / "orders.yml"
        semantic_file.write_text("datasets:\n  - name: orders\n")
        semantic_snapshot = [{"id": "profile:shop:old_orders", "yaml_path": str(semantic_file)}]
        profile_snapshot = [{"id": "profile:shop:old_orders", "yaml_path": str(semantic_file)}]
        metric_snapshot = [{"id": "metric:old_metric", "name": "old_metric", "yaml_path": str(metric_file)}]
        generation_tools.semantic_dataset_rag = Mock()
        generation_tools.semantic_dataset_rag.list_artifact_rows.return_value = semantic_snapshot
        generation_tools.semantic_dataset_rag.list_artifact_rows.return_value = profile_snapshot
        generation_tools.metric_rag.list_artifact_rows.return_value = metric_snapshot
        generation_tools.metric_rag.delete_artifact_rows_except.side_effect = RuntimeError("metric delete failed")
        metric = SimpleNamespace(
            name="order_count",
            description="Number of orders",
            expression="COUNT(*)",
            dataset="orders",
            subject_path=None,
            kind="aggregate",
            inputs=[],
            measures=[],
        )
        doc = SimpleNamespace(name="shop", datasets=[], relationships=[], metrics=[metric])
        prepared_semantic = {
            "success": True,
            "semantic_objects": [{"id": "table:shop:orders", "yaml_path": str(semantic_file)}],
            "semantic_dataset_rows": [{"id": "profile:shop:orders", "yaml_path": str(semantic_file)}],
        }

        with (
            patch.object(generation_tools, "_load_osi_document", return_value=doc),
            patch.object(
                generation_tools,
                "_sync_osi_semantic_objects_to_db",
                return_value=prepared_semantic,
            ),
        ):
            result = generation_tools._sync_osi_metric_to_db(str(metric_file), str(semantic_file))

        assert result["success"] is False
        assert "metric delete failed" in result["error"]
        generation_tools.semantic_dataset_rag.restore_artifact_rows.assert_called_once_with(
            str(semantic_file), semantic_snapshot
        )
        generation_tools.semantic_dataset_rag.restore_artifact_rows.assert_called_once_with(
            str(semantic_file), profile_snapshot
        )
        generation_tools.metric_rag.restore_artifact_rows.assert_called_once_with(str(metric_file), metric_snapshot)

    def test_sync_osi_metric_to_db_does_not_write_metrics_when_semantic_write_fails(self, generation_tools, tmp_path):
        metric_file = tmp_path / "orders_metrics.yml"
        metric_file.write_text(
            "version: 0.2.0.dev0\nsemantic_model:\n  - name: shop\n    metrics:\n      - name: order_count\n"
        )
        semantic_file = tmp_path / "orders.yml"
        semantic_file.write_text("datasets:\n  - name: orders\n")
        generation_tools.semantic_dataset_rag.upsert_batch.side_effect = RuntimeError("semantic write failed")
        metric = SimpleNamespace(
            name="order_count",
            description="Number of orders",
            expression="COUNT(*)",
            dataset="orders",
            subject_path=None,
            kind="aggregate",
            inputs=[],
            measures=[],
        )
        doc = SimpleNamespace(name="shop", datasets=[], relationships=[], metrics=[metric])
        prepared_semantic = {
            "success": True,
            "semantic_dataset_rows": [{"id": "dataset:shop:orders", "yaml_path": str(semantic_file)}],
        }

        with (
            patch.object(generation_tools, "_load_osi_document", return_value=doc),
            patch.object(
                generation_tools,
                "_sync_osi_semantic_objects_to_db",
                return_value=prepared_semantic,
            ),
        ):
            result = generation_tools._sync_osi_metric_to_db(str(metric_file), str(semantic_file))

        assert result["success"] is False
        assert "semantic write failed" in result["error"]
        generation_tools.metric_rag.upsert_batch.assert_not_called()
        generation_tools.metric_rag.delete_artifact_rows_except.assert_not_called()

    def test_sync_osi_metric_to_db_reconciles_metric_file_without_metrics(self, generation_tools, tmp_path):
        metric_file = tmp_path / "empty_metrics.yml"
        metric_file.write_text(
            "version: 0.2.0.dev0\n"
            "semantic_model:\n"
            "  - name: empty\n"
            "    datasets:\n"
            "      - name: orders\n"
            "        source: orders\n"
        )
        snapshot = [{"id": "metric:old_metric", "name": "old_metric", "yaml_path": str(metric_file)}]
        generation_tools.metric_rag.list_artifact_rows.return_value = snapshot
        doc = SimpleNamespace(name="empty", datasets=[], metrics=[])

        with patch.object(generation_tools, "_load_osi_document", return_value=doc):
            result = generation_tools._sync_osi_metric_to_db(str(metric_file))

        assert result["success"] is True
        assert result["metric_names"] == []
        assert result["deleted_metric_names"] == ["old_metric"]
        generation_tools.metric_rag.upsert_batch.assert_not_called()
        generation_tools.metric_rag.delete_artifact_rows_except.assert_called_once_with(str(metric_file), [])

    def test_sync_osi_metric_to_db_rejects_missing_scoped_metric(self, generation_tools, tmp_path):
        metric_file = tmp_path / "orders.yml"
        metric_file.write_text(
            "version: 0.2.0.dev0\nsemantic_model:\n  - name: shop\n    metrics:\n      - name: order_count\n"
        )

        result = generation_tools._sync_osi_metric_to_db(
            str(metric_file),
            metric_names_to_sync={"missing_metric"},
        )

        assert result["success"] is False
        assert "missing_metric" in result["error"]
        generation_tools.metric_rag.upsert_batch.assert_not_called()

    def test__sync_osi_semantic_objects_to_db_upserts_only_current_dataset_columns(self, generation_tools, tmp_path):
        generation_tools.agent_config.current_db_config.return_value = SimpleNamespace(
            catalog="default_catalog", database="shop", schema=""
        )
        generation_tools.semantic_dataset_rag = Mock()
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
        customer_segment = SimpleNamespace(
            name="customer_segment",
            expr="customer_segment",
            type="categorical",
            description="Customer segment",
            granularity=None,
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
            dimensions=[customer_segment],
            fields=[
                SimpleNamespace(name="order_id", expr="order_id", type="categorical"),
                SimpleNamespace(name="order_date", expr="order_date", type="time"),
                customer_segment,
                SimpleNamespace(
                    name="amount",
                    expr="amount",
                    type="numeric",
                    description="Order amount",
                    granularity=None,
                ),
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
        doc = SimpleNamespace(
            name="shop",
            datasets=[dataset, other_dataset],
            relationships=[relationship],
            metrics=[],
        )

        with patch.object(generation_tools, "_load_osi_document", return_value=doc):
            result = generation_tools._sync_osi_semantic_objects_to_db(str(semantic_file))

        assert result["success"] is True
        generation_tools.semantic_dataset_rag.delete_artifact_rows.assert_not_called()
        generation_tools.semantic_dataset_rag.delete_artifact_rows_except.assert_called_once()
        generation_tools.semantic_dataset_rag.upsert_batch.assert_called_once()
        objects = generation_tools.semantic_dataset_rag.upsert_batch.call_args.args[0]
        assert [obj["kind"] for obj in objects] == ["dataset", "field", "field", "field", "field", "relationship"]
        assert objects[0]["name"] == "orders"
        assert objects[1]["name"] == "order_id"
        assert objects[1]["is_primary_key"] is True
        assert objects[3]["name"] == "customer_segment"
        assert objects[3]["is_dimension"] is True
        assert objects[4]["name"] == "amount"
        assert objects[4]["is_dimension"] is False
        generation_tools.semantic_dataset_rag.delete_artifact_rows.assert_not_called()
        generation_tools.semantic_dataset_rag.delete_artifact_rows_except.assert_called_once()
        generation_tools.semantic_dataset_rag.upsert_batch.assert_called_once()
        rows = generation_tools.semantic_dataset_rag.upsert_batch.call_args.args[0]
        by_kind: dict[str, list] = {}
        for row in rows:
            by_kind.setdefault(row["kind"], []).append(row)

        dataset_row = by_kind["dataset"][0]
        assert dataset_row["dataset_name"] == "orders"
        assert dataset_row["source_table"] == "orders"
        assert dataset_row["source_query"] == ""
        assert dataset_row["description"] == "Orders table"
        assert "order-level analytics" in dataset_row["ai_context_json"]

        fields = {row["name"]: row for row in by_kind["field"]}
        assert fields["customer_segment"]["is_dimension"] is True
        assert fields["amount"]["is_dimension"] is False
        assert fields["order_id"]["is_primary_key"] is True

        relationship = by_kind["relationship"][0]
        assert '["customer_id", "store_id"]' in relationship["from_columns_json"]
        assert '["customer_id", "store_id"]' in relationship["to_columns_json"]
        assert result["semantic_dataset_rows"] == len(rows)

    def test_load_osi_document_selects_only_artifact_model(self, generation_tools, tmp_path):
        (tmp_path / "sales.yml").write_text(
            "version: 0.2.0.dev0\n"
            "semantic_model:\n"
            "  - name: sales\n"
            "    datasets:\n"
            "      - name: orders\n"
            "        source: orders\n"
            "        primary_key: [order_id]\n"
        )
        (tmp_path / "finance.yml").write_text(
            "version: 0.2.0.dev0\n"
            "semantic_model:\n"
            "  - name: finance\n"
            "    datasets:\n"
            "      - name: budgets\n"
            "        source: budgets\n"
            "        primary_key: [budget_id]\n"
        )
        metric_file = tmp_path / "finance_metrics.yml"
        metric_file.write_text(
            "version: 0.2.0.dev0\n"
            "semantic_model:\n"
            "  - name: finance\n"
            "    datasets:\n"
            "      - name: budgets\n"
            "        source: budgets\n"
            "        primary_key: [budget_id]\n"
            "    metrics:\n"
            "      - name: budget_total\n"
            "        expression:\n"
            "          dialects:\n"
            "            - dialect: ANSI_SQL\n"
            "              expression: SUM(amount)\n"
            "        custom_extensions:\n"
            "          - vendor_name: DATUS\n"
            '            data: \'{"dataset":"budgets"}\'\n'
        )

        finance_doc = SimpleNamespace(
            name="finance",
            datasets=[SimpleNamespace(name="budgets")],
            metrics=[SimpleNamespace(name="budget_total")],
        )
        with patch(
            "datus.tools.semantic_tools.osi_document.load_osi_document",
            return_value=finance_doc,
        ) as loader:
            doc = generation_tools._load_osi_document(metric_file=str(metric_file))

        assert doc.name == "finance"
        assert [dataset.name for dataset in doc.datasets] == ["budgets"]
        assert [metric.name for metric in doc.metrics] == ["budget_total"]
        loader.assert_called_once_with(str(metric_file), semantic_model_name="finance")

    def test__sync_osi_semantic_objects_to_db_distinguishes_same_named_tables_across_databases(
        self, generation_tools, tmp_path
    ):
        """Issue #1084 (OSI path): qualified source tables in different databases keep distinct ids."""
        generation_tools.agent_config.current_db_config.return_value = SimpleNamespace(
            catalog="", database="shop", schema=""
        )
        generation_tools.agent_config.db_type = "snowflake"
        generation_tools.semantic_dataset_rag = Mock()
        semantic_file = tmp_path / "orders.yml"
        semantic_file.write_text(
            "version: 0.2.0.dev0\n"
            "semantic_model:\n"
            "  - name: shop\n"
            "    datasets:\n"
            "      - name: orders_db1\n"
            "        source: db1.public.orders\n"
            "      - name: orders_db2\n"
            "        source: db2.sales.orders\n"
        )
        doc = SimpleNamespace(
            name="shop",
            datasets=[
                SimpleNamespace(
                    name="orders_db1",
                    description="",
                    source=SimpleNamespace(table="db1.public.orders"),
                    primary_key="order_id",
                    time_dimension=None,
                    dimensions=[],
                ),
                SimpleNamespace(
                    name="orders_db2",
                    description="",
                    source=SimpleNamespace(table="db2.sales.orders"),
                    primary_key="order_id",
                    time_dimension=None,
                    dimensions=[],
                ),
            ],
            relationships=[],
            metrics=[],
        )

        with patch.object(generation_tools, "_load_osi_document", return_value=doc):
            result = generation_tools._sync_osi_semantic_objects_to_db(str(semantic_file))

        assert result["success"] is True
        objects = generation_tools.semantic_dataset_rag.upsert_batch.call_args.args[0]
        dataset_ids = [obj["id"] for obj in objects if obj["kind"] == "dataset"]
        assert dataset_ids == ["dataset:shop:orders_db1", "dataset:shop:orders_db2"]
        field_ids = [obj["id"] for obj in objects if obj["kind"] == "field"]
        assert field_ids == ["field:shop:orders_db1.order_id", "field:shop:orders_db2.order_id"]
        # Both datasets bind a table called "orders"; their coordinates are what
        # tell the two physical tables apart.
        datasets_by_id = {obj["id"]: obj for obj in objects if obj["kind"] == "dataset"}
        assert datasets_by_id["dataset:shop:orders_db1"]["source_table"] == "orders"
        assert datasets_by_id["dataset:shop:orders_db1"]["database_name"] == "db1"
        assert datasets_by_id["dataset:shop:orders_db1"]["schema_name"] == "public"
        assert datasets_by_id["dataset:shop:orders_db2"]["database_name"] == "db2"
        assert datasets_by_id["dataset:shop:orders_db2"]["schema_name"] == "sales"

    def test__sync_osi_semantic_objects_to_db_fails_when_table_profile_sync_fails(self, generation_tools, tmp_path):
        generation_tools.agent_config.current_db_config.return_value = SimpleNamespace(
            catalog="default_catalog", database="shop", schema=""
        )
        generation_tools.semantic_dataset_rag = Mock()
        generation_tools.semantic_dataset_rag.upsert_batch.side_effect = RuntimeError("profile sync failed")
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
            result = generation_tools._sync_osi_semantic_objects_to_db(str(semantic_file))

        assert result["success"] is False
        assert "profile sync failed" in result["error"]
        generation_tools.semantic_dataset_rag.restore_artifact_rows.assert_called_once()
        generation_tools.semantic_dataset_rag.restore_artifact_rows.assert_called_once()

    def test_sync_osi_to_db_reconciles_semantic_and_metric_rows_together(self, generation_tools, tmp_path):
        osi_file = tmp_path / "shop.yml"
        osi_file.write_text("version: 0.2.0.dev0\n")
        doc = SimpleNamespace()
        semantic_rows = [{"id": "dataset:shop:orders"}]
        metric_rows = [{"id": "metric:order_count", "name": "order_count", "sql": ""}]
        generation_tools.metric_rag.list_artifact_rows.return_value = [
            {
                "id": "metric:order_count",
                "name": "Order_Count",
                "sql": "SELECT COUNT(*) FROM orders",
            }
        ]
        with (
            patch.object(generation_tools, "_load_osi_document", return_value=doc),
            patch.object(generation_tools, "extract_osi_metric_names", return_value=["order_count"]),
            patch.object(
                generation_tools,
                "_sync_osi_semantic_objects_to_db",
                return_value={
                    "success": True,
                    "semantic_dataset_rows": semantic_rows,
                    "synced_items": ["table:orders"],
                },
            ) as prepare_semantic,
            patch.object(generation_tools, "_build_osi_metric_objects", return_value=metric_rows),
        ):
            result = generation_tools.sync_osi_to_db(str(osi_file))

        prepare_semantic.assert_called_once_with(str(osi_file), doc=doc, prepare_only=True)
        generation_tools.semantic_dataset_rag.upsert_batch.assert_called_once_with(semantic_rows)
        generation_tools.metric_rag.upsert_batch.assert_called_once_with(metric_rows)
        assert metric_rows[0]["sql"] == "SELECT COUNT(*) FROM orders"
        generation_tools.semantic_dataset_rag.delete_artifact_rows_except.assert_called_once()
        generation_tools.metric_rag.delete_artifact_rows_except.assert_called_once()
        assert result["success"] is True
        assert result["synced"] == 1
        assert result["semantic_dataset_rows"] == 1
        assert result["metric_names"] == ["order_count"]

    def test_sync_osi_to_db_reconciles_empty_metric_collection(self, generation_tools, tmp_path):
        osi_file = tmp_path / "model.yml"
        osi_file.write_text("version: 0.2.0.dev0\n")
        doc = SimpleNamespace()
        generation_tools.semantic_dataset_rag = Mock()
        generation_tools.semantic_dataset_rag.list_artifact_rows.return_value = [{"id": "profile:model:old"}]
        generation_tools.metric_rag.list_artifact_rows.return_value = [{"id": "metric:old", "name": "old"}]
        with (
            patch.object(generation_tools, "_load_osi_document", return_value=doc),
            patch.object(generation_tools, "extract_osi_metric_names", return_value=[]),
            patch.object(
                generation_tools,
                "_sync_osi_semantic_objects_to_db",
                return_value={
                    "success": True,
                    "semantic_objects": [{"id": "table:model:orders"}],
                    "semantic_dataset_rows": [],
                    "synced_items": [],
                },
            ),
            patch.object(generation_tools, "_build_osi_metric_objects", return_value=[]),
        ):
            result = generation_tools.sync_osi_to_db(str(osi_file))

        assert result["success"] is True
        assert result["synced"] == 0
        assert result["deleted_metric_names"] == ["old"]
        generation_tools.metric_rag.upsert_batch.assert_not_called()
        generation_tools.metric_rag.delete_artifact_rows_except.assert_called_once_with(str(osi_file), [])
        generation_tools.semantic_dataset_rag.delete_artifact_rows_except.assert_called_once_with(str(osi_file), [])

    def test_sync_osi_to_db_reconciles_empty_metrics_without_semantic_sync(self, generation_tools, tmp_path):
        osi_file = tmp_path / "model.yml"
        osi_file.write_text("version: 0.2.0.dev0\n")
        with (
            patch.object(generation_tools, "_load_osi_document", return_value=SimpleNamespace()),
            patch.object(generation_tools, "extract_osi_metric_names", return_value=[]),
            patch.object(generation_tools, "_build_osi_metric_objects", return_value=[]),
            patch.object(generation_tools, "_sync_osi_semantic_objects_to_db") as semantic_sync,
        ):
            result = generation_tools.sync_osi_to_db(
                str(osi_file), include_semantic_objects=False, include_metrics=True
            )

        semantic_sync.assert_not_called()
        assert result["success"] is True
        assert result["synced"] == 0
        generation_tools.metric_rag.delete_artifact_rows_except.assert_called_once_with(str(osi_file), [])

    def test_sync_osi_to_db_restores_all_stores_when_metric_write_fails(self, generation_tools, tmp_path):
        osi_file = tmp_path / "model.yml"
        osi_file.write_text("version: 0.2.0.dev0\n")
        dataset_rows = [{"id": "dataset:model:orders"}]
        metric_rows = [{"id": "metric:new", "name": "new"}]
        generation_tools.semantic_dataset_rag = Mock()
        generation_tools.semantic_dataset_rag.list_artifact_rows.return_value = [{"id": "dataset:old"}]
        generation_tools.metric_rag.list_artifact_rows.return_value = [{"id": "metric:old", "name": "old"}]
        generation_tools.metric_rag.create_indices.side_effect = RuntimeError("metric index failed")
        with (
            patch.object(generation_tools, "_load_osi_document", return_value=SimpleNamespace()),
            patch.object(generation_tools, "extract_osi_metric_names", return_value=["new"]),
            patch.object(
                generation_tools,
                "_sync_osi_semantic_objects_to_db",
                return_value={
                    "success": True,
                    "semantic_dataset_rows": dataset_rows,
                    "synced_items": [],
                },
            ),
            patch.object(generation_tools, "_build_osi_metric_objects", return_value=metric_rows),
        ):
            result = generation_tools.sync_osi_to_db(str(osi_file))

        assert result["success"] is False
        assert "metric index failed" in result["error"]
        generation_tools.semantic_dataset_rag.restore_artifact_rows.assert_called_once_with(
            str(osi_file), [{"id": "dataset:old"}]
        )
        generation_tools.metric_rag.restore_artifact_rows.assert_called_once_with(
            str(osi_file), [{"id": "metric:old", "name": "old"}]
        )

    def test_sync_osi_to_db_returns_error_dict_on_unexpected_failure(self, generation_tools, tmp_path):
        # Consistent with the delegated syncs: an unexpected raise degrades to an
        # error dict rather than propagating out of the public entry.
        osi_file = tmp_path / "shop.yml"
        osi_file.write_text("version: 0.2.0.dev0\n")
        with patch.object(generation_tools, "_load_osi_document", side_effect=RuntimeError("bad yaml")):
            result = generation_tools.sync_osi_to_db(str(osi_file))
        assert result == {"success": False, "error": "bad yaml", "synced": 0}


class TestGenerateSqlSummaryId:
    def test_success(self, generation_tools):
        with patch("datus.storage.reference_sql.init_utils.gen_reference_sql_id", return_value="abc123"):
            result = generation_tools.generate_sql_summary_id("SELECT * FROM orders")
        assert result.success == 1
        assert result.result == "abc123"


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
        generation_tools.semantic_dataset_rag.storage = mock_storage
        mock_storage._search_all.return_value = [
            {"id": "t1", "name": "orders", "kind": "dataset", "dataset_name": "orders", "source_table": "orders"}
        ]

        with patch("datus.tools.func_tool.generation_tools.And"), patch("datus.tools.func_tool.generation_tools.eq"):
            # First call populates the cache
            result1 = generation_tools.check_semantic_object_exists("orders", kind="table")
            # Second call should hit the cache
            result2 = generation_tools.check_semantic_object_exists("orders", kind="table")

        assert result1.success == result2.success
        assert result1.result == result2.result
        # Cache should have been populated
        assert len(generation_tools._semantic_object_exists_cache) >= 1


class TestOsiDatasetColumnRoles:
    """Projection has to keep what the loader resolved, not re-derive it."""

    @staticmethod
    def _dataset(fields, time_dimension=None, dimensions=(), unique_keys=()):
        from types import SimpleNamespace

        return SimpleNamespace(
            name="orders",
            primary_key=[],
            unique_keys=list(unique_keys),
            time_dimension=time_dimension,
            dimensions=list(dimensions),
            fields=fields,
        )

    @staticmethod
    def _field(name, type_="categorical", granularity=""):
        from types import SimpleNamespace

        return SimpleNamespace(
            name=name, expr=name, type=type_, granularity=granularity, description="", ai_context=None
        )

    def test_every_time_field_keeps_its_temporal_role(self):
        """A dataset has one primary time axis but any number of time
        dimensions. Reading only `time_dimension` demoted the others to plain
        fields and lost their granularity."""
        from datus.tools.func_tool.generation_tools import GenerationTools

        start = self._field("start_date", "time", "day")
        end = self._field("end_date", "time", "day")
        dataset = self._dataset([start, end], time_dimension=start, dimensions=[end])

        columns = {c["name"]: c for c in GenerationTools._osi_dataset_columns(dataset)}

        assert columns["start_date"]["role"] == "time_dimension"
        assert columns["end_date"]["role"] == "time_dimension"
        assert columns["end_date"]["granularity"] == "day"

    def test_a_non_temporal_field_is_not_promoted(self):
        from datus.tools.func_tool.generation_tools import GenerationTools

        dataset = self._dataset([self._field("region")])

        columns = {c["name"]: c for c in GenerationTools._osi_dataset_columns(dataset)}

        assert columns["region"]["role"] == "field"
