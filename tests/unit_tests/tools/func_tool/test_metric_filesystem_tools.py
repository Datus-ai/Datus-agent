# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
import yaml

from datus.tools.func_tool.generation_evidence import GenerationEvidence
from datus.tools.func_tool.metric_filesystem_tools import (
    MetricFilesystemFuncTool,
    OsiSemanticModelFilesystemFuncTool,
)
from datus.tools.func_tool.osi_target_tools import OsiSemanticModelTargetState


def _osi_metric(name, expression):
    return {
        "name": name,
        "description": f"Definition for {name}",
        "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": expression}]},
    }


def _bound_state(target, name="sales"):
    state = OsiSemanticModelTargetState()
    state.select(
        {
            "semantic_model_name": name,
            "semantic_model_file": f"subject/semantic_models/warehouse/{target.name}",
            "absolute_path": str(target.resolve()),
            "artifact_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        },
        mode="bound",
    )
    return state


def _planned_state(target, name="sales"):
    state = OsiSemanticModelTargetState()
    state.select(
        {
            "semantic_model_name": name,
            "semantic_model_file": f"subject/semantic_models/warehouse/{target.name}",
            "absolute_path": str(target.resolve()),
            "artifact_sha256": "",
        },
        mode="planned",
    )
    return state


@pytest.fixture
def osi_schema_validator(monkeypatch):
    validator = Mock(return_value=None)
    monkeypatch.setattr(MetricFilesystemFuncTool, "_validate_osi_document", staticmethod(validator))
    return validator


class TestMetricFilesystemFuncTool:
    def test_osi_available_tools_are_metrics_only(self, tmp_path):
        tool = MetricFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_metrics",
            authoring_format="osi",
        )

        tool_names = {tool.name for tool in tool.available_tools()}

        assert tool_names == {"read_file", "upsert_osi_metrics", "glob", "grep"}

    def test_osi_semantic_model_dataset_upsert_preserves_metrics_and_relationships(
        self,
        tmp_path,
        osi_schema_validator,
    ):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "sales.yml"
        target.parent.mkdir(parents=True)
        target.write_text(
            yaml.safe_dump(
                {
                    "version": "0.2.0.dev0",
                    "semantic_model": [
                        {
                            "name": "sales",
                            "datasets": [{"name": "orders", "source": "analytics.orders"}],
                            "relationships": [{"name": "orders_to_customers"}],
                            "metrics": [_osi_metric("revenue", "SUM(orders.amount)")],
                        }
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        state = OsiSemanticModelTargetState()
        state.select(
            {
                "semantic_model_name": "sales",
                "semantic_model_file": str(target.relative_to(tmp_path)),
                "absolute_path": str(target),
                "artifact_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            },
            mode="planned",
        )
        tool = OsiSemanticModelFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_semantic_model",
            mutation_guard=state.require_planned_path,
            mutation_callback=state.record_planned_write,
        )
        query_dataset = {
            "name": "retention_query_dataset",
            "source": "WITH cohort AS (SELECT user_id FROM users) SELECT COUNT(*) AS users FROM cohort",
            "description": "One row containing the retained-user result.",
            "ai_context": {"instructions": "Use for the exact retention result grain."},
            "custom_extensions": [{"vendor_name": "DATUS", "data": '{"source_type":"query"}'}],
        }

        result = tool.upsert_osi_datasets(
            str(target.relative_to(tmp_path)),
            json.dumps([query_dataset]),
        )

        assert result.success == 1
        assert result.result["created"] == ["retention_query_dataset"]
        document = yaml.safe_load(target.read_text(encoding="utf-8"))
        model = document["semantic_model"][0]
        assert model["relationships"] == [{"name": "orders_to_customers"}]
        assert model["metrics"] == [_osi_metric("revenue", "SUM(orders.amount)")]
        assert model["datasets"][-1] == query_dataset
        assert state.target_mutated is True
        assert set(tool.name for tool in tool.available_tools()) == {
            "read_file",
            "edit_file",
            "upsert_osi_datasets",
            "glob",
            "grep",
        }
        osi_schema_validator.assert_called_once()

    def test_osi_dataset_upsert_creates_first_valid_document_without_shell(
        self,
        tmp_path,
        osi_schema_validator,
    ):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "sales.yml"
        state = _planned_state(target)
        tool = OsiSemanticModelFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_semantic_model",
            mutation_guard=state.require_planned_path,
            mutation_callback=state.record_planned_write,
            osi_target_state=state,
        )

        result = tool.upsert_osi_datasets(
            str(target.relative_to(tmp_path)),
            json.dumps([{"name": "orders", "source": "analytics.orders"}]),
        )

        assert result.success == 1
        assert result.result["created"] == ["orders"]
        document = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert document["semantic_model"] == [
            {
                "name": "sales",
                "datasets": [{"name": "orders", "source": "analytics.orders"}],
                "relationships": [],
                "metrics": [],
            }
        ]
        assert state.target_mutated is True
        osi_schema_validator.assert_called_once()

    def test_osi_edit_rejects_invalid_document_without_changing_file(self, tmp_path, monkeypatch):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "sales.yml"
        target.parent.mkdir(parents=True)
        target.write_text(
            "version: 0.2.0.dev0\n"
            "semantic_model:\n"
            "  - name: sales\n"
            "    datasets:\n"
            "      - name: orders\n"
            "        source: analytics.orders\n"
            "    relationships: []\n"
            "    metrics: []\n",
            encoding="utf-8",
        )
        original = target.read_text(encoding="utf-8")
        validator = Mock(return_value="semantic_model[0].datasets must not be empty")
        monkeypatch.setattr(MetricFilesystemFuncTool, "_validate_osi_document", staticmethod(validator))
        tool = OsiSemanticModelFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_semantic_model",
        )

        result = tool.edit_file(
            str(target.relative_to(tmp_path)),
            "    datasets:\n      - name: orders\n        source: analytics.orders\n",
            "    datasets: []\n",
        )

        assert result.success == 0
        assert "must not be empty" in result.error
        assert target.read_text(encoding="utf-8") == original
        validator.assert_called_once()

    def test_query_backed_dataset_upsert_rejects_same_name_with_different_sql(
        self,
        tmp_path,
        osi_schema_validator,
    ):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "sales.yml"
        target.parent.mkdir(parents=True)
        existing_dataset = {
            "name": "retained_users",
            "source": "SELECT user_id FROM retained_users",
            "custom_extensions": [{"vendor_name": "DATUS", "data": '{"source_type":"query"}'}],
        }
        target.write_text(
            yaml.safe_dump(
                {
                    "semantic_model": [
                        {
                            "name": "sales",
                            "datasets": [existing_dataset],
                        }
                    ]
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        original = target.read_text(encoding="utf-8")
        tool = OsiSemanticModelFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_semantic_model",
        )
        conflicting_dataset = {
            **existing_dataset,
            "source": "SELECT user_id FROM newly_retained_users",
        }

        result = tool.upsert_osi_datasets(
            str(target.relative_to(tmp_path)),
            json.dumps([conflicting_dataset]),
        )

        assert result.success == 0
        assert result.result["code"] == "query_dataset_name_conflict"
        assert "choose a new semantic dataset name" in result.error
        assert target.read_text(encoding="utf-8") == original
        osi_schema_validator.assert_not_called()

    def test_query_backed_dataset_source_identity_normalizes_line_endings(
        self,
        tmp_path,
        osi_schema_validator,
    ):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "sales.yml"
        target.parent.mkdir(parents=True)
        existing_dataset = {
            "name": "retained_users",
            "source": "SELECT user_id\r\nFROM retained_users\r\n",
            "custom_extensions": [{"vendor_name": "DATUS", "data": '{"source_type":"query"}'}],
        }
        target.write_text(
            yaml.safe_dump(
                {"semantic_model": [{"name": "sales", "datasets": [existing_dataset]}]},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        tool = OsiSemanticModelFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_semantic_model",
        )
        incoming_dataset = {
            **existing_dataset,
            "source": "SELECT user_id\nFROM retained_users",
        }

        result = tool.upsert_osi_datasets(
            str(target.relative_to(tmp_path)),
            json.dumps([incoming_dataset]),
        )

        assert result.success == 1
        assert result.result["updated"] == ["retained_users"]
        osi_schema_validator.assert_called_once()

    def test_query_backed_dataset_upsert_rejects_source_already_used_by_another_model(
        self,
        tmp_path,
        osi_schema_validator,
    ):
        model_dir = tmp_path / "subject" / "semantic_models" / "warehouse"
        model_dir.mkdir(parents=True)
        source_sql = "SELECT region, COUNT(*) AS order_count\nFROM orders GROUP BY region"
        existing = model_dir / "regional_orders.yml"
        existing.write_text(
            yaml.safe_dump(
                {
                    "semantic_model": [
                        {
                            "name": "regional_orders",
                            "datasets": [
                                {
                                    "name": "orders_by_region",
                                    "source": source_sql.replace("\n", "\r\n"),
                                    "custom_extensions": [{"vendor_name": "DATUS", "data": '{"source_type":"query"}'}],
                                }
                            ],
                        }
                    ]
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        target = model_dir / "sales.yml"
        state = _planned_state(target)
        evidence = GenerationEvidence(required_query_backed_sql={"query_dataset:orders": source_sql})
        tool = OsiSemanticModelFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_semantic_model",
            mutation_guard=state.require_planned_path,
            mutation_callback=state.record_planned_write,
            osi_target_state=state,
            generation_evidence=evidence,
        )

        result = tool.upsert_osi_datasets(
            str(target.relative_to(tmp_path)),
            json.dumps(
                [
                    {
                        "dataset_requirement_id": "query_dataset:orders",
                        "name": "regional_order_counts",
                    }
                ]
            ),
        )

        assert result.success == 0
        assert result.result["code"] == "query_dataset_source_conflict"
        assert result.result["existing_semantic_model_name"] == "regional_orders"
        assert result.result["existing_dataset_name"] == "orders_by_region"
        assert not target.exists()
        osi_schema_validator.assert_not_called()

    def test_query_backed_dataset_upsert_injects_request_local_sql(
        self,
        tmp_path,
        osi_schema_validator,
    ):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "sales.yml"
        target.parent.mkdir(parents=True)
        target.write_text(
            "version: 0.2.0.dev0\nsemantic_model:\n  - name: sales\n    datasets: []\n",
            encoding="utf-8",
        )
        exact_sql = "WITH scoped AS (SELECT * FROM sales)\nSELECT COUNT(*) AS sale_count FROM scoped;"
        evidence = GenerationEvidence(required_query_backed_sql={"query_dataset:abc": exact_sql})
        tool = OsiSemanticModelFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_semantic_model",
            generation_evidence=evidence,
        )

        result = tool.upsert_osi_datasets(
            str(target.relative_to(tmp_path)),
            json.dumps(
                [
                    {
                        "dataset_requirement_id": "query_dataset:abc",
                        "name": "scoped_sales",
                        "description": "One row containing the scoped sale count.",
                        "ai_context": {"instructions": "Use for the exact scoped result."},
                    }
                ]
            ),
        )

        assert result.success == 1
        dataset = yaml.safe_load(target.read_text(encoding="utf-8"))["semantic_model"][0]["datasets"][0]
        assert dataset["source"] == exact_sql
        assert "dataset_requirement_id" not in dataset
        extension_data = json.loads(dataset["custom_extensions"][0]["data"])
        assert extension_data == {"source_type": "query"}

    def test_query_source_extension_always_serializes_data_as_json(self, tmp_path):
        tool = OsiSemanticModelFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_semantic_model",
        )

        extensions = tool._query_source_extensions([{"vendor_name": "DATUS", "data": {"owner": "semantic-authoring"}}])

        assert isinstance(extensions[0]["data"], str)
        assert json.loads(extensions[0]["data"]) == {
            "owner": "semantic-authoring",
            "source_type": "query",
        }

    def test_query_backed_requirement_reuses_first_dataset_name_during_retry(
        self,
        tmp_path,
        osi_schema_validator,
    ):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "sales.yml"
        target.parent.mkdir(parents=True)
        target.write_text(
            "version: 0.2.0.dev0\nsemantic_model:\n  - name: sales\n    datasets: []\n",
            encoding="utf-8",
        )
        requirement_id = "query_dataset:stable"
        evidence = GenerationEvidence(
            required_query_backed_sql={
                requirement_id: "SELECT region, COUNT(*) AS order_count FROM orders GROUP BY region;"
            }
        )
        tool = OsiSemanticModelFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_semantic_model",
            generation_evidence=evidence,
        )
        relative_path = str(target.relative_to(tmp_path))

        first = tool.upsert_osi_datasets(
            relative_path,
            json.dumps(
                [
                    {
                        "dataset_requirement_id": requirement_id,
                        "name": "regional_orders",
                        "fields": [{"name": "region"}],
                    }
                ]
            ),
        )
        retry = tool.upsert_osi_datasets(
            relative_path,
            json.dumps(
                [
                    {
                        "dataset_requirement_id": requirement_id,
                        "name": "regional_orders_v2",
                        "fields": [{"name": "region"}],
                    }
                ]
            ),
        )

        assert first.success == 1
        assert retry.success == 1
        assert retry.result["canonicalized_names"] == {"regional_orders_v2": "regional_orders"}
        datasets = yaml.safe_load(target.read_text(encoding="utf-8"))["semantic_model"][0]["datasets"]
        assert [dataset["name"] for dataset in datasets] == ["regional_orders"]

    def test_query_backed_dataset_upsert_rejects_unknown_requirement(
        self,
        tmp_path,
    ):
        target = tmp_path / "model.yml"
        target.write_text("semantic_model:\n  - name: sales\n    datasets: []\n", encoding="utf-8")
        tool = OsiSemanticModelFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_semantic_model",
            generation_evidence=GenerationEvidence(),
        )

        result = tool.upsert_osi_datasets(
            "model.yml",
            json.dumps([{"dataset_requirement_id": "query_dataset:missing", "name": "missing"}]),
        )

        assert result.success == 0
        assert result.result["code"] == "dataset_requirement_not_found"

    def test_existing_query_backed_dataset_cannot_be_overwritten_by_omitting_extension(
        self,
        tmp_path,
        osi_schema_validator,
    ):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "sales.yml"
        target.parent.mkdir(parents=True)
        existing_dataset = {
            "name": "retained_users",
            "source": "SELECT user_id FROM retained_users",
            "custom_extensions": [{"vendor_name": "DATUS", "data": '{"source_type":"query"}'}],
        }
        target.write_text(
            yaml.safe_dump(
                {"semantic_model": [{"name": "sales", "datasets": [existing_dataset]}]},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        original = target.read_text(encoding="utf-8")
        tool = OsiSemanticModelFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_semantic_model",
        )

        result = tool.upsert_osi_datasets(
            str(target.relative_to(tmp_path)),
            json.dumps(
                [
                    {
                        "name": "retained_users",
                        "source": "analytics.retained_users",
                    }
                ]
            ),
        )

        assert result.success == 0
        assert result.result["code"] == "query_dataset_name_conflict"
        assert target.read_text(encoding="utf-8") == original
        osi_schema_validator.assert_not_called()

    def test_upsert_osi_metrics_preserves_semantic_objects(self, tmp_path, osi_schema_validator):
        project = tmp_path / "project"
        target = project / "subject" / "semantic_models" / "warehouse" / "sales.yml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """
version: 0.2.0.dev0
semantic_model:
  - name: sales
    description: Sales domain
    datasets:
      - name: orders
        source: orders
        fields:
          - name: amount
            expression:
              dialects:
                - dialect: ANSI_SQL
                  expression: amount
            dimension:
              is_time: false
    relationships:
      - name: orders_to_customers
        from: orders
        to: customers
        from_columns: [customer_id]
        to_columns: [customer_id]
    metrics:
      - name: revenue
        description: Old definition
        expression:
          dialects:
            - dialect: ANSI_SQL
              expression: SUM(amount)
""".lstrip(),
            encoding="utf-8",
        )
        before = yaml.safe_load(target.read_text(encoding="utf-8"))
        tool = MetricFilesystemFuncTool(
            root_path=str(project),
            current_node="gen_metrics",
            authoring_format="osi",
            osi_target_state=_bound_state(target),
        )

        result = tool.upsert_osi_metrics(
            "subject/semantic_models/warehouse/sales.yml",
            json.dumps(
                [
                    {
                        "name": "revenue",
                        "description": "Corrected definition",
                        "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "SUM(net_amount)"}]},
                    },
                    _osi_metric("order_count", "COUNT(*)"),
                ]
            ),
        )

        assert result.success == 1
        assert result.result["created"] == ["order_count"]
        assert result.result["updated"] == ["revenue"]
        after = yaml.safe_load(target.read_text(encoding="utf-8"))
        before_model = before["semantic_model"][0]
        after_model = after["semantic_model"][0]
        assert {key: value for key, value in after_model.items() if key != "metrics"} == {
            key: value for key, value in before_model.items() if key != "metrics"
        }
        assert [metric["name"] for metric in after_model["metrics"]] == ["revenue", "order_count"]
        assert after_model["metrics"][0]["description"] == "Corrected definition"
        osi_schema_validator.assert_called_once()

    def test_upsert_invalidates_prior_validation_dry_run_and_sync_evidence(self, tmp_path, osi_schema_validator):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "sales.yml"
        target.parent.mkdir(parents=True)
        target.write_text(
            "version: 0.2.0.dev0\n"
            "semantic_model:\n"
            "  - name: sales\n"
            "    datasets:\n"
            "      - name: orders\n"
            "        source: orders\n",
            encoding="utf-8",
        )
        evidence = GenerationEvidence(
            validation_passed=True,
            metric_dry_run_passed=True,
            metric_dry_run_metrics={"revenue"},
            semantic_kb_sync_passed=True,
            metric_kb_sync_passed=True,
        )
        evidence.record_semantic_artifact_validation("sales", target)
        tool = MetricFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_metrics",
            authoring_format="osi",
            osi_target_state=_bound_state(target),
            mutation_callback=evidence.invalidate_artifact_evidence,
        )

        result = tool.upsert_osi_metrics(
            str(target.relative_to(tmp_path)),
            json.dumps([_osi_metric("revenue", "SUM(amount)")]),
        )

        assert result.success == 1
        assert evidence.validation_passed is False
        assert evidence.metric_dry_run_passed is False
        assert evidence.metric_dry_run_metrics == set()
        assert evidence.validated_semantic_artifacts == {}
        assert evidence.kb_sync_passed is False

    def test_failed_metric_authoring_can_restore_pre_request_artifact(self, tmp_path, osi_schema_validator):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "sales.yml"
        target.parent.mkdir(parents=True)
        target.write_text(
            "version: 0.2.0.dev0\n"
            "semantic_model:\n"
            "  - name: sales\n"
            "    datasets:\n"
            "      - name: orders\n"
            "        source: orders\n",
            encoding="utf-8",
        )
        original = target.read_bytes()
        state = _bound_state(target)
        tool = MetricFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_metrics",
            authoring_format="osi",
            osi_target_state=state,
        )

        result = tool.upsert_osi_metrics(
            str(target.relative_to(tmp_path)),
            json.dumps([_osi_metric("revenue", "SUM(amount)")]),
        )

        assert result.success == 1
        assert target.read_bytes() != original
        assert tool.rollback_failed_metric_authoring() is True
        assert target.read_bytes() == original
        assert state.authored_metric_names == []
        assert tool.rollback_failed_metric_authoring() is False

    def test_failed_metric_authoring_rollback_returns_false_on_invalid_snapshot(self, tmp_path):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "sales.yml"
        target.parent.mkdir(parents=True)
        target.write_text("semantic_model: []\n", encoding="utf-8")
        state = _bound_state(target)
        state.record_metric_snapshot(target, b"\xff")
        tool = MetricFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_metrics",
            authoring_format="osi",
            osi_target_state=state,
        )

        assert tool.rollback_failed_metric_authoring() is False
        assert state.metric_snapshot_content == b"\xff"

    def test_failed_metric_authoring_rollback_returns_false_on_write_error(self, tmp_path):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "sales.yml"
        target.parent.mkdir(parents=True)
        target.write_text("semantic_model: []\n", encoding="utf-8")
        state = _bound_state(target)
        state.record_metric_snapshot(target, target.read_bytes())
        tool = MetricFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_metrics",
            authoring_format="osi",
            osi_target_state=state,
        )
        tool._atomic_write_text = Mock(side_effect=OSError("disk full"))

        assert tool.rollback_failed_metric_authoring() is False
        assert state.metric_snapshot_content == b"semantic_model: []\n"

    def test_identical_upsert_preserves_bytes_and_registers_publish_scope(self, tmp_path, osi_schema_validator):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "sales.yml"
        target.parent.mkdir(parents=True)
        metric = _osi_metric("revenue", "SUM(amount)")
        target.write_text(
            yaml.safe_dump(
                {
                    "version": "0.2.0.dev0",
                    "semantic_model": [
                        {
                            "name": "sales",
                            "datasets": [{"name": "orders", "source": "orders"}],
                            "metrics": [metric],
                        }
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        original_content = target.read_bytes()
        mutation_callback = Mock()
        state = _bound_state(target)
        tool = MetricFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_metrics",
            authoring_format="osi",
            osi_target_state=state,
            mutation_callback=mutation_callback,
        )

        result = tool.upsert_osi_metrics(str(target.relative_to(tmp_path)), json.dumps([metric]))

        assert result.success == 1
        assert result.result["created"] == []
        assert result.result["updated"] == []
        assert result.result["unchanged"] == ["revenue"]
        assert target.read_bytes() == original_content
        assert state.authored_metric_names == ["revenue"]
        mutation_callback.assert_not_called()
        osi_schema_validator.assert_not_called()

    @pytest.mark.parametrize("invalid_metrics", [{}, ""])
    def test_upsert_osi_metrics_rejects_present_invalid_metrics_collection(self, tmp_path, invalid_metrics):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "sales.yml"
        target.parent.mkdir(parents=True)
        document = {
            "version": "0.2.0.dev0",
            "semantic_model": [{"name": "sales", "datasets": [{"name": "orders", "source": "orders"}]}],
        }
        document["semantic_model"][0]["metrics"] = invalid_metrics
        original = yaml.safe_dump(document, sort_keys=False)
        target.write_text(original, encoding="utf-8")
        tool = MetricFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_metrics",
            authoring_format="osi",
            osi_target_state=_bound_state(target),
        )

        result = tool.upsert_osi_metrics(
            str(target.relative_to(tmp_path)), json.dumps([_osi_metric("revenue", "SUM(amount)")])
        )

        assert result.success == 0
        assert "metrics must be a list" in result.error
        assert target.read_text(encoding="utf-8") == original

    def test_upsert_osi_metrics_validates_metric_schema_before_writing(self, tmp_path, osi_schema_validator):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "sales.yml"
        target.parent.mkdir(parents=True)
        original = """version: 0.2.0.dev0
semantic_model:
  - name: sales
    datasets:
      - name: orders
        source: orders
"""
        target.write_text(original, encoding="utf-8")
        tool = MetricFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_metrics",
            authoring_format="osi",
            osi_target_state=_bound_state(target),
        )
        osi_schema_validator.return_value = "metric expression is required"

        result = tool.upsert_osi_metrics(
            str(target.relative_to(tmp_path)),
            json.dumps([{"name": "revenue"}]),
        )

        assert result.success == 0
        assert "Invalid OSI metric update" in result.error
        assert target.read_text(encoding="utf-8") == original

    def test_upsert_osi_metrics_serializes_concurrent_tool_instances(self, tmp_path, osi_schema_validator):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "sales.yml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """version: 0.2.0.dev0
semantic_model:
  - name: sales
    datasets:
      - name: orders
        source: orders
""",
            encoding="utf-8",
        )
        target_state = _bound_state(target)
        tools = [
            MetricFilesystemFuncTool(
                root_path=str(tmp_path),
                current_node="gen_metrics",
                authoring_format="osi",
                osi_target_state=target_state,
            ),
            MetricFilesystemFuncTool(
                root_path=str(tmp_path),
                current_node="gen_metrics",
                authoring_format="osi",
                osi_target_state=target_state,
            ),
        ]
        relative_path = str(target.relative_to(tmp_path))
        shared_lock = tools[0]._osi_metric_path_lock(target)
        assert shared_lock is tools[1]._osi_metric_path_lock(target)
        second_started = threading.Event()

        def upsert_from_second_tool():
            second_started.set()
            return tools[1].upsert_osi_metrics(relative_path, json.dumps([_osi_metric("order_count", "COUNT(*)")]))

        with ThreadPoolExecutor(max_workers=2) as executor:
            with shared_lock:
                second_result = executor.submit(upsert_from_second_tool)
                assert second_started.wait(timeout=1)
                assert not second_result.done()
            results = [
                second_result.result(),
                tools[0].upsert_osi_metrics(relative_path, json.dumps([_osi_metric("revenue", "SUM(amount)")])),
            ]

        assert all(result.success == 1 for result in results)
        metrics = yaml.safe_load(target.read_text(encoding="utf-8"))["semantic_model"][0]["metrics"]
        assert {metric["name"] for metric in metrics} == {"revenue", "order_count"}

    def test_upsert_osi_metrics_requires_existing_model(self, tmp_path):
        tool = MetricFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_metrics",
            authoring_format="osi",
        )

        result = tool.upsert_osi_metrics(
            "subject/semantic_models/warehouse/sales.yml",
            json.dumps([_osi_metric("revenue", "SUM(amount)")]),
        )

        assert result.success == 0
        assert result.result["code"] == "semantic_model_required"

    def test_upsert_osi_metrics_rejects_path_other_than_bound_target(self, tmp_path):
        model_dir = tmp_path / "subject" / "semantic_models" / "warehouse"
        selected = model_dir / "selected.yml"
        other = model_dir / "other.yml"
        for path, name in ((selected, "selected"), (other, "other")):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"semantic_model:\n  - name: {name}\n    datasets: []\n", encoding="utf-8")
        tool = MetricFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_metrics",
            authoring_format="osi",
            osi_target_state=_bound_state(selected, "selected"),
        )

        result = tool.upsert_osi_metrics(
            str(other.relative_to(tmp_path)),
            json.dumps([_osi_metric("revenue", "SUM(amount)")]),
        )

        assert not result.success
        assert result.result["code"] == "semantic_model_target_invalid"
        assert "bound to" in result.error

    def test_upsert_osi_metrics_rejects_target_changed_since_bind(self, tmp_path):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "sales.yml"
        target.parent.mkdir(parents=True)
        target.write_text("semantic_model:\n  - name: sales\n    datasets: []\n", encoding="utf-8")
        state = _bound_state(target)
        tool = MetricFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_metrics",
            authoring_format="osi",
            osi_target_state=state,
        )
        target.write_text(target.read_text(encoding="utf-8") + "# external edit\n", encoding="utf-8")

        result = tool.upsert_osi_metrics(
            str(target.relative_to(tmp_path)),
            json.dumps([_osi_metric("revenue", "SUM(amount)")]),
        )

        assert not result.success
        assert result.result["code"] == "semantic_model_target_invalid"
        assert "changed after selection" in result.error

    def test_write_file_merges_existing_semantic_model(self, tmp_path):
        project = tmp_path / "project"
        target = project / "subject" / "semantic_models" / "ac_manage" / "orders.yml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """
data_source:
  name: orders
  sql_table: ac_manage.orders
  measures:
    - name: order_count
      agg: COUNT
      expr: "1"
  dimensions:
    - name: ds
      type: TIME
""".lstrip(),
            encoding="utf-8",
        )
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics")

        result = tool.write_file(
            "subject/semantic_models/ac_manage/orders.yml",
            """
data_source:
  name: orders
  sql_table: ac_manage.orders
  measures:
    - name: paid_order_count
      agg: SUM
      expr: "CASE WHEN status = 'paid' THEN 1 ELSE 0 END"
""".lstrip(),
        )

        assert result.success == 1
        docs = list(yaml.safe_load_all(target.read_text(encoding="utf-8")))
        data_source = docs[0]["data_source"]
        assert [measure["name"] for measure in data_source["measures"]] == [
            "order_count",
            "paid_order_count",
        ]
        assert data_source["dimensions"][0]["name"] == "ds"

    def test_write_file_rejects_conflicting_measure_overwrite(self, tmp_path):
        project = tmp_path / "project"
        target = project / "subject" / "semantic_models" / "ac_manage" / "orders.yml"
        target.parent.mkdir(parents=True)
        original = """
data_source:
  name: orders
  sql_table: ac_manage.orders
  measures:
    - name: order_count
      agg: COUNT
      expr: "1"
""".lstrip()
        target.write_text(original, encoding="utf-8")
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics")

        result = tool.write_file(
            "subject/semantic_models/ac_manage/orders.yml",
            """
data_source:
  name: orders
  sql_table: ac_manage.orders
  measures:
    - name: order_count
      agg: SUM
      expr: amount
""".lstrip(),
        )

        assert result.success == 0
        assert "Refusing to overwrite measure 'order_count'" in result.error
        assert target.read_text(encoding="utf-8") == original

    def test_write_file_merges_existing_metric_file(self, tmp_path):
        project = tmp_path / "project"
        target = project / "subject" / "semantic_models" / "ac_manage" / "metrics" / "orders_metrics.yml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """
metric:
  name: order_count
  type: measure_proxy
  type_params:
    measure: order_count
""".lstrip(),
            encoding="utf-8",
        )
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics")

        result = tool.write_file(
            "subject/semantic_models/ac_manage/metrics/orders_metrics.yml",
            """
metric:
  name: paid_order_count
  type: measure_proxy
  type_params:
    measure: paid_order_count
""".lstrip(),
        )

        assert result.success == 1
        docs = list(yaml.safe_load_all(target.read_text(encoding="utf-8")))
        assert [doc["metric"]["name"] for doc in docs] == ["order_count", "paid_order_count"]

    def test_write_file_rejects_conflicting_metric_overwrite(self, tmp_path):
        project = tmp_path / "project"
        target = project / "subject" / "semantic_models" / "ac_manage" / "metrics" / "orders_metrics.yml"
        target.parent.mkdir(parents=True)
        original = """
metric:
  name: order_count
  type: measure_proxy
  type_params:
    measure: order_count
""".lstrip()
        target.write_text(original, encoding="utf-8")
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics")

        result = tool.write_file(
            "subject/semantic_models/ac_manage/metrics/orders_metrics.yml",
            """
metric:
  name: order_count
  type: ratio
  type_params:
    numerator: paid_order_count
    denominator: order_count
""".lstrip(),
        )

        assert result.success == 0
        assert "Refusing to overwrite metric 'order_count'" in result.error
        assert target.read_text(encoding="utf-8") == original

    def test_osi_authoring_skips_metricflow_merge(self, tmp_path):
        project = tmp_path / "project"
        target = project / "subject" / "semantic_models" / "ac_manage" / "orders.yml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """
semantic_model:
  - name: ac_manage
    datasets:
      - name: orders
        source:
          table: orders
""".lstrip(),
            encoding="utf-8",
        )
        tool = MetricFilesystemFuncTool(
            root_path=str(project),
            current_node="gen_metrics",
            authoring_format="osi",
        )
        incoming = """
semantic_model:
  - name: ac_manage
    datasets:
      - name: orders
        source:
          table: orders
        fields:
          - name: amount
""".lstrip()

        result = tool.write_file("subject/semantic_models/ac_manage/orders.yml", incoming)

        assert result.success == 1
        assert target.read_text(encoding="utf-8") == incoming

    def test_write_file_strict_external_path_is_rejected_before_merge(self, tmp_path, monkeypatch):
        project = tmp_path / "project"
        project.mkdir()
        target = tmp_path / "outside" / "subject" / "semantic_models" / "ac_manage" / "metrics" / "orders.yml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """
metric:
  name: order_count
  type: measure_proxy
  type_params:
    measure: order_count
""".lstrip(),
            encoding="utf-8",
        )
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics", strict=True)

        def fail_if_called(*_args, **_kwargs):
            pytest.fail("external file was read before strict path rejection")

        monkeypatch.setattr(tool, "_merge_metric_content", fail_if_called)

        result = tool.write_file(
            str(target),
            """
metric:
  name: paid_order_count
  type: measure_proxy
  type_params:
    measure: paid_order_count
""".lstrip(),
        )

        assert result.success == 0
        assert "outside workspace" in result.error

    def test_quote_escape_repair_only_touches_yaml_error_location(self):
        content = """
data_source:
  name: orders
  sql_query: |
    SELECT '\\'' AS literal_value
  measures:
    - name: paid_order_count
      agg: SUM
      expr: "CASE WHEN status = \\'paid\\' THEN 1 END"
""".lstrip()

        repaired = MetricFilesystemFuncTool._repair_invalid_yaml_single_quote_escapes(content)

        docs = list(yaml.safe_load_all(repaired))
        data_source = docs[0]["data_source"]
        assert data_source["measures"][0]["expr"] == "CASE WHEN status = 'paid' THEN 1 END"
        assert "SELECT '\\'' AS literal_value" in data_source["sql_query"]


class TestEditFile:
    """Tests for MetricFilesystemFuncTool.edit_file — covers lines 65-97."""

    def test_edit_file_in_semantic_yaml(self, tmp_path):
        project = tmp_path / "project"
        target = project / "subject" / "semantic_models" / "orders.yml"
        target.parent.mkdir(parents=True)
        target.write_text("data_source:\n  name: orders\n  description: old\n", encoding="utf-8")
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics")
        result = tool.edit_file(
            "subject/semantic_models/orders.yml",
            "description: old",
            "description: new",
        )
        assert result.success == 1
        assert "description: new" in target.read_text(encoding="utf-8")

    def test_edit_file_old_string_not_found(self, tmp_path):
        project = tmp_path / "project"
        target = project / "subject" / "semantic_models" / "orders.yml"
        target.parent.mkdir(parents=True)
        target.write_text("data_source:\n  name: orders\n", encoding="utf-8")
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics")
        result = tool.edit_file(
            "subject/semantic_models/orders.yml",
            "nonexistent string",
            "replacement",
        )
        assert result.success == 0

    def test_edit_file_outside_semantic_yaml_no_postprocess(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir(parents=True)
        target = project / "notes.txt"
        target.write_text("hello world\n", encoding="utf-8")
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics")
        result = tool.edit_file("notes.txt", "hello", "goodbye")
        assert result.success == 1

    def test_edit_file_postprocess_restores_on_invalid_yaml(self, tmp_path):
        project = tmp_path / "project"
        target = project / "subject" / "semantic_models" / "orders.yml"
        target.parent.mkdir(parents=True)
        original = "data_source:\n  name: orders\n"
        target.write_text(original, encoding="utf-8")
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics")
        # Replace valid YAML with something that becomes invalid after edit
        result = tool.edit_file(
            "subject/semantic_models/orders.yml",
            "name: orders",
            "name: orders\n  bad: : :",
        )
        # Should fail due to invalid YAML and restore original
        assert result.success == 0
        # Original content should be restored
        assert target.read_text(encoding="utf-8") == original


class TestMergeSemanticModelContentErrors:
    """Tests for _merge_semantic_model_content error paths — lines 213-216, 220-228."""

    def test_rejects_missing_existing_data_source(self, tmp_path):
        project = tmp_path / "project"
        target = project / "subject" / "semantic_models" / "orders.yml"
        target.parent.mkdir(parents=True)
        target.write_text("semantic_model:\n  name: orders\n", encoding="utf-8")
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics")

        result = tool.write_file(
            "subject/semantic_models/orders.yml",
            "data_source:\n  name: orders\n  measures:\n    - name: revenue\n      agg: SUM\n      expr: revenue\n",
        )
        assert result.success == 0
        assert "data_source" in result.error.lower()

    def test_rejects_missing_incoming_data_source(self, tmp_path):
        project = tmp_path / "project"
        target = project / "subject" / "semantic_models" / "orders.yml"
        target.parent.mkdir(parents=True)
        target.write_text(
            "data_source:\n  name: orders\n  measures:\n    - name: revenue\n      agg: SUM\n      expr: revenue\n",
            encoding="utf-8",
        )
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics")

        result = tool.write_file(
            "subject/semantic_models/orders.yml",
            "semantic_model:\n  name: orders\n",
        )
        assert result.success == 0
        assert "data_source" in result.error.lower()


class TestMergeMetricContentErrors:
    """Tests for _merge_metric_content error paths — lines 245-248, 256-257, 290-291."""

    def test_rejects_when_existing_has_no_metrics(self, tmp_path):
        project = tmp_path / "project"
        target = project / "subject" / "semantic_models" / "ds" / "metrics" / "orders_metrics.yml"
        target.parent.mkdir(parents=True)
        target.write_text("some_key:\n  name: something\n", encoding="utf-8")
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics")

        result = tool.write_file(
            "subject/semantic_models/ds/metrics/orders_metrics.yml",
            "metric:\n  name: new_metric\n  type: measure_proxy\n",
        )
        assert result.success == 0
        assert "metric" in result.error.lower()

    def test_rejects_when_incoming_has_no_metrics(self, tmp_path):
        project = tmp_path / "project"
        target = project / "subject" / "semantic_models" / "ds" / "metrics" / "orders_metrics.yml"
        target.parent.mkdir(parents=True)
        target.write_text("metric:\n  name: order_count\n  type: measure_proxy\n", encoding="utf-8")
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics")

        result = tool.write_file(
            "subject/semantic_models/ds/metrics/orders_metrics.yml",
            "some_key:\n  name: something\n",
        )
        assert result.success == 0
        assert "metric" in result.error.lower()


class TestNormalizeMetricSubjectTreeTags:
    """Tests for _normalize_metric_subject_tree_tags — lines 296-327."""

    def test_normalizes_subject_tree_tag_in_metric(self, tmp_path):
        project = tmp_path / "project"
        target = project / "subject" / "semantic_models" / "myds" / "metrics" / "orders_metrics.yml"
        target.parent.mkdir(parents=True)
        # Write a metric with a subject_tree tag that will be normalized
        target.write_text(
            "metric:\n"
            "  name: order_count\n"
            "  type: measure_proxy\n"
            "  type_params:\n"
            "    measure: order_count\n"
            "  locked_metadata:\n"
            "    tags:\n"
            "      - 'subject_tree: metrics/OrderCount'\n",
            encoding="utf-8",
        )
        tool = MetricFilesystemFuncTool(root_path=str(project), current_node="gen_metrics")
        result = tool.write_file(
            "subject/semantic_models/myds/metrics/orders_metrics.yml",
            "metric:\n  name: new_metric\n  type: measure_proxy\n  type_params:\n    measure: new_metric\n",
        )
        assert result.success == 1


class TestStaticHelpers:
    """Tests for static methods: _metric_scope_from_path, _merge_metric_fields, etc."""

    def test_metric_scope_from_path_with_metrics_suffix(self, tmp_path):
        path = tmp_path / "subject" / "semantic_models" / "myds" / "metrics" / "orders_metrics.yml"
        datasource, table_name = MetricFilesystemFuncTool._metric_scope_from_path(path)
        assert datasource == "myds"
        assert table_name == "orders"

    def test_metric_scope_from_path_without_suffix(self, tmp_path):
        path = tmp_path / "subject" / "semantic_models" / "myds" / "metrics" / "revenue.yml"
        _, table_name = MetricFilesystemFuncTool._metric_scope_from_path(path)
        assert table_name == "revenue"

    def test_metric_from_doc_non_dict_returns_none(self):
        assert MetricFilesystemFuncTool._metric_from_doc("not_a_dict") is None
        assert MetricFilesystemFuncTool._metric_from_doc(None) is None

    def test_metric_from_doc_no_metric_key_returns_none(self):
        assert MetricFilesystemFuncTool._metric_from_doc({"other": "value"}) is None

    def test_metric_from_doc_valid_returns_dict(self):
        result = MetricFilesystemFuncTool._metric_from_doc({"metric": {"name": "m1"}})
        assert result == {"name": "m1"}

    def test_merge_metric_fields_fills_empty(self):
        existing = {"name": "m1", "type": "measure_proxy", "description": ""}
        incoming = {"name": "m1", "description": "new description", "extra": "value"}
        merged = MetricFilesystemFuncTool._merge_metric_fields(existing, incoming)
        assert merged["description"] == "new description"
        assert merged["extra"] == "value"
        assert merged["type"] == "measure_proxy"

    def test_merge_metric_fields_preserves_existing_non_empty(self):
        existing = {"name": "m1", "description": "existing"}
        incoming = {"name": "m1", "description": "new"}
        merged = MetricFilesystemFuncTool._merge_metric_fields(existing, incoming)
        assert merged["description"] == "existing"

    def test_metric_definition_conflict_detects_type_change(self):
        existing = {"name": "m1", "type": "measure_proxy"}
        incoming = {"name": "m1", "type": "ratio"}
        assert MetricFilesystemFuncTool._metric_definition_conflict(existing, incoming) == "type"

    def test_metric_definition_conflict_no_conflict(self):
        existing = {"name": "m1", "type": "measure_proxy"}
        incoming = {"name": "m1", "type": "measure_proxy"}
        assert MetricFilesystemFuncTool._metric_definition_conflict(existing, incoming) == ""

    def test_metric_definition_conflict_missing_value_skipped(self):
        existing = {"name": "m1", "type": None}
        incoming = {"name": "m1", "type": "measure_proxy"}
        assert MetricFilesystemFuncTool._metric_definition_conflict(existing, incoming) == ""

    def test_find_data_source_doc_found(self):
        docs = [{"other": "value"}, {"data_source": {"name": "orders"}}]
        idx, doc, ds = MetricFilesystemFuncTool._find_data_source_doc(docs)
        assert idx == 1
        assert ds == {"name": "orders"}

    def test_find_data_source_doc_not_found(self):
        docs = [{"other": "value"}]
        idx, doc, ds = MetricFilesystemFuncTool._find_data_source_doc(docs)
        assert idx == -1
        assert ds is None

    def test_named_item_conflict_detects_field_diff(self):
        existing = {"name": "m1", "agg": "SUM"}
        incoming = {"name": "m1", "agg": "COUNT"}
        assert MetricFilesystemFuncTool._named_item_conflict(existing, incoming, ("agg",)) == "agg"

    def test_named_item_conflict_no_conflict(self):
        existing = {"name": "m1", "agg": "SUM"}
        incoming = {"name": "m1", "agg": "SUM"}
        assert MetricFilesystemFuncTool._named_item_conflict(existing, incoming, ("agg",)) == ""

    def test_named_item_conflict_empty_values_skipped(self):
        existing = {"name": "m1", "agg": None}
        incoming = {"name": "m1", "agg": "SUM"}
        assert MetricFilesystemFuncTool._named_item_conflict(existing, incoming, ("agg",)) == ""

    def test_stable_yaml_value_is_deterministic(self):
        val = {"b": 2, "a": 1}
        s1 = MetricFilesystemFuncTool._stable_yaml_value(val)
        s2 = MetricFilesystemFuncTool._stable_yaml_value(val)
        assert s1 == s2

    def test_merge_stable_scalar_fills_empty(self):
        merged = {"name": "orders"}
        incoming = {"sql_table": "orders_table"}
        err = MetricFilesystemFuncTool._merge_stable_scalar(merged, incoming, "sql_table", "orders")
        assert err == ""
        assert merged["sql_table"] == "orders_table"

    def test_merge_stable_scalar_conflict_returns_error(self):
        merged = {"sql_table": "original_table"}
        incoming = {"sql_table": "different_table"}
        err = MetricFilesystemFuncTool._merge_stable_scalar(merged, incoming, "sql_table", "orders")
        assert err != ""
        assert "original_table" in err
        assert "different_table" in err

    def test_merge_data_sources_name_conflict_returns_error(self):
        tool = MetricFilesystemFuncTool.__new__(MetricFilesystemFuncTool)
        existing_ds = {"name": "orders"}
        incoming_ds = {"name": "customers"}
        _, error = tool._merge_data_sources(existing_ds, incoming_ds)
        assert error != ""
        assert "orders" in error

    def test_merge_named_items_conflict_returns_error(self):
        tool = MetricFilesystemFuncTool.__new__(MetricFilesystemFuncTool)
        existing = [{"name": "order_count", "agg": "COUNT", "expr": "1"}]
        incoming = [{"name": "order_count", "agg": "SUM", "expr": "amount"}]
        _, error = tool._merge_named_items("measures", existing, incoming, ("agg", "expr"), "orders")
        assert error != ""
        assert "order_count" in error
