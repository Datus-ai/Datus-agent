# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import pytest
import yaml

from datus.storage.semantic_model.artifact_file import semantic_artifact_lock
from datus.tools.func_tool.generation_evidence import GenerationEvidence
from datus.tools.func_tool.metric_filesystem_tools import (
    MetricFilesystemFuncTool,
    OsiSemanticModelFilesystemFuncTool,
    SemanticModelingFilesystemFuncTool,
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
    def test_osi_available_tools_include_narrow_dataset_mutations(self, tmp_path):
        tool = MetricFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_metrics",
        )

        tool_names = {tool.name for tool in tool.available_tools()}

        assert tool_names == {
            "read_file",
            "upsert_osi_metrics",
            "delete_osi_metrics",
            "upsert_osi_datasets",
            "delete_osi_datasets",
            "glob",
            "grep",
        }

    def test_bound_dataset_repair_can_be_followed_by_metric_upsert(self, tmp_path, osi_schema_validator):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "sales.yml"
        target.parent.mkdir(parents=True)
        original_document = {
            "version": "0.2.0.dev0",
            "semantic_model": [
                {
                    "name": "sales",
                    "datasets": [
                        {
                            "name": "orders",
                            "source": "analytics.orders",
                            "fields": [
                                {
                                    "name": "ordered_at",
                                    "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "ordered_at"}]},
                                    "dimension": {"is_time": True},
                                }
                            ],
                        },
                        {"name": "obsolete_orders_query", "source": "SELECT 1 AS unused"},
                    ],
                    "relationships": [],
                    "metrics": [],
                }
            ],
        }
        target.write_text(yaml.safe_dump(original_document, sort_keys=False), encoding="utf-8")
        original_content = target.read_bytes()
        state = _bound_state(target)
        tool = MetricFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_metrics",
            mutation_guard=state.require_bound_path,
            osi_target_state=state,
        )
        repaired_dataset = dict(original_document["semantic_model"][0]["datasets"][0])
        repaired_dataset["fields"] = [
            *repaired_dataset["fields"],
            {
                "name": "order_id",
                "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "order_id"}]},
            },
        ]

        dataset_result = tool.upsert_osi_datasets(str(target.relative_to(tmp_path)), json.dumps([repaired_dataset]))
        delete_result = tool.delete_osi_datasets(
            str(target.relative_to(tmp_path)),
            ["obsolete_orders_query"],
        )
        metric_result = tool.upsert_osi_metrics(
            str(target.relative_to(tmp_path)),
            json.dumps([_osi_metric("order_count", "COUNT(DISTINCT orders.order_id)")]),
        )

        assert dataset_result.success == 1
        assert delete_result.success == 1
        assert metric_result.success == 1
        assert state.touched_dataset_names == ["orders", "obsolete_orders_query"]
        assert state.touched_metric_names == ["order_count"]
        assert state.artifact_snapshot_content == original_content
        model = yaml.safe_load(target.read_text(encoding="utf-8"))["semantic_model"][0]
        assert [dataset["name"] for dataset in model["datasets"]] == ["orders"]
        assert [field["name"] for field in model["datasets"][0]["fields"]] == ["ordered_at", "order_id"]
        assert [metric["name"] for metric in model["metrics"]] == ["order_count"]
        assert osi_schema_validator.call_count == 3

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
            osi_target_state=state,
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
        authored_query_dataset = model["datasets"][-1]
        assert {key: value for key, value in authored_query_dataset.items() if key != "custom_extensions"} == {
            key: value for key, value in query_dataset.items() if key != "custom_extensions"
        }
        assert json.loads(authored_query_dataset["custom_extensions"][0]["data"]) == {"source_type": "query"}
        assert state.target_mutated is True
        assert state.planned_dataset_names == ["retention_query_dataset"]
        assert set(tool.name for tool in tool.available_tools()) == {
            "read_file",
            "edit_file",
            "upsert_osi_datasets",
            "delete_osi_datasets",
            "glob",
            "grep",
        }
        osi_schema_validator.assert_called_once()

    def test_planned_dataset_update_is_recorded(self, tmp_path, osi_schema_validator):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "sales.yml"
        target.parent.mkdir(parents=True)
        target.write_text(
            yaml.safe_dump(
                {
                    "semantic_model": [
                        {
                            "name": "sales",
                            "datasets": [{"name": "orders", "source": "analytics.orders"}],
                        }
                    ]
                }
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
            osi_target_state=state,
        )

        result = tool.upsert_osi_datasets(
            str(target.relative_to(tmp_path)),
            json.dumps([{"name": "orders", "source": "SELECT * FROM analytics.orders"}]),
        )

        assert result.success == 1
        assert result.result["updated"] == ["orders"]
        assert state.planned_dataset_names == ["orders"]

    def test_osi_edit_records_changed_dataset(self, tmp_path, osi_schema_validator):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "sales.yml"
        target.parent.mkdir(parents=True)
        target.write_text(
            "semantic_model:\n  - name: sales\n    datasets:\n      - name: orders\n        source: analytics.orders\n",
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
            osi_target_state=state,
        )

        result = tool.edit_file(
            str(target.relative_to(tmp_path)),
            "source: analytics.orders",
            "source: SELECT * FROM analytics.orders",
        )

        assert result.success == 1
        assert state.planned_dataset_names == ["orders"]

    def test_delete_osi_datasets_is_scoped_tolerant_and_retryable(self, tmp_path, osi_schema_validator):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "sales.yml"
        target.parent.mkdir(parents=True)
        target.write_text(
            yaml.safe_dump(
                {
                    "version": "0.2.0.dev0",
                    "semantic_model": [
                        {
                            "name": "sales",
                            "description": "Sales domain",
                            "datasets": [
                                {"name": "orders", "source": "analytics.orders"},
                                {"name": "Customers", "source": "analytics.customers"},
                            ],
                            "relationships": [
                                {
                                    "name": "orders_to_customers",
                                    "from": "orders",
                                    "to": "Customers",
                                    "from_columns": ["customer_id"],
                                    "to_columns": ["customer_id"],
                                }
                            ],
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
            osi_target_state=state,
        )

        result = tool.delete_osi_datasets(
            str(target.relative_to(tmp_path)),
            [" Customers ", "customers", "missing_dataset"],
        )

        assert result.success == 1
        assert result.result["requested"] == ["Customers", "missing_dataset"]
        assert result.result["deleted"] == ["Customers"]
        assert result.result["already_absent"] == ["missing_dataset"]
        assert result.result["remaining"] == ["orders"]
        document = yaml.safe_load(target.read_text(encoding="utf-8"))
        model = document["semantic_model"][0]
        assert model["description"] == "Sales domain"
        assert model["relationships"][0]["to"] == "Customers"
        assert model["metrics"] == [_osi_metric("revenue", "SUM(orders.amount)")]
        assert state.target_mutated is True

        after_delete = target.read_bytes()
        retry = tool.delete_osi_datasets(str(target.relative_to(tmp_path)), ["customers"])

        assert retry.success == 1
        assert retry.result["deleted"] == []
        assert retry.result["already_absent"] == ["customers"]
        assert target.read_bytes() == after_delete

        restored = tool.upsert_osi_datasets(
            str(target.relative_to(tmp_path)),
            json.dumps([{"name": "Customers", "source": "analytics.customers"}]),
        )

        assert restored.success == 1
        assert restored.result["created"] == ["Customers"]
        restored_model = yaml.safe_load(target.read_text(encoding="utf-8"))["semantic_model"][0]
        assert [dataset["name"] for dataset in restored_model["datasets"]] == ["orders", "Customers"]
        assert osi_schema_validator.call_count == 2

    def test_delete_osi_datasets_rejects_invalid_empty_model_without_changing_file(self, tmp_path, monkeypatch):
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
        original = target.read_bytes()
        validator = Mock(return_value="semantic_model[0].datasets: [] should be non-empty")
        monkeypatch.setattr(MetricFilesystemFuncTool, "_validate_osi_document", staticmethod(validator))
        state = OsiSemanticModelTargetState()
        state.select(
            {
                "semantic_model_name": "sales",
                "semantic_model_file": str(target.relative_to(tmp_path)),
                "absolute_path": str(target),
                "artifact_sha256": hashlib.sha256(original).hexdigest(),
            },
            mode="planned",
        )
        tool = OsiSemanticModelFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_semantic_model",
            mutation_guard=state.require_planned_path,
            mutation_callback=state.record_planned_write,
        )

        result = tool.delete_osi_datasets(str(target.relative_to(tmp_path)), ["orders"])

        assert result.success == 0
        assert "should be non-empty" in result.error
        assert target.read_bytes() == original
        assert state.target_mutated is False

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

    def test_query_backed_dataset_upsert_updates_same_name_with_revised_sql(
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

        assert result.success == 1
        assert result.result["updated"] == ["retained_users"]
        authored = yaml.safe_load(target.read_text(encoding="utf-8"))["semantic_model"][0]["datasets"][0]
        assert authored["source"] == "SELECT user_id FROM newly_retained_users"
        assert json.loads(authored["custom_extensions"][0]["data"])["source_type"] == "query"
        osi_schema_validator.assert_called_once()

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

    def test_query_backed_dataset_upsert_allows_source_used_by_another_model(
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
        evidence = GenerationEvidence()
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
                        "name": "regional_order_counts",
                        "source": source_sql,
                    }
                ]
            ),
        )

        assert result.success == 1
        assert result.result["created"] == ["regional_order_counts"]
        authored = yaml.safe_load(target.read_text(encoding="utf-8"))["semantic_model"][0]["datasets"][0]
        assert authored["source"] == source_sql
        assert json.loads(authored["custom_extensions"][0]["data"])["source_type"] == "query"
        osi_schema_validator.assert_called_once()

    @pytest.mark.parametrize(
        "exact_sql",
        [
            "WITH\r\nscoped AS (SELECT * FROM sales)\nSELECT COUNT(*) AS sale_count FROM scoped;",
            "SELECT\tCOUNT(*) AS sale_count FROM sales;",
        ],
    )
    def test_query_backed_dataset_upsert_accepts_generated_sql(
        self,
        tmp_path,
        osi_schema_validator,
        exact_sql,
    ):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "sales.yml"
        target.parent.mkdir(parents=True)
        target.write_text(
            "version: 0.2.0.dev0\nsemantic_model:\n  - name: sales\n    datasets: []\n",
            encoding="utf-8",
        )
        evidence = GenerationEvidence()
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
                        "name": "scoped_sales",
                        "source": exact_sql,
                        "description": "One row containing the scoped sale count.",
                        "ai_context": {"instructions": "Use for the generated scoped result."},
                    }
                ]
            ),
        )

        assert result.success == 1
        dataset = yaml.safe_load(target.read_text(encoding="utf-8"))["semantic_model"][0]["datasets"][0]
        assert dataset["source"] == exact_sql
        extension_data = json.loads(dataset["custom_extensions"][0]["data"])
        assert extension_data == {"source_type": "query"}

    def test_generated_query_can_replace_same_named_dataset_source(
        self,
        tmp_path,
        osi_schema_validator,
    ):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "sales.yml"
        target.parent.mkdir(parents=True)
        target.write_text(
            yaml.safe_dump(
                {
                    "semantic_model": [
                        {
                            "name": "sales",
                            "datasets": [
                                {
                                    "name": "retained_users",
                                    "source": "SELECT user_id FROM retained_users WHERE cohort = 'old'",
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
        exact_sql = "SELECT user_id FROM retained_users WHERE cohort = 'current'"
        evidence = GenerationEvidence()
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
                        "name": "retained_users",
                        "source": exact_sql,
                        "description": "Current retained-user cohort.",
                    }
                ]
            ),
        )

        assert result.success == 1
        assert result.result["updated"] == ["retained_users"]
        dataset = yaml.safe_load(target.read_text(encoding="utf-8"))["semantic_model"][0]["datasets"][0]
        assert dataset["source"] == exact_sql
        assert json.loads(dataset["custom_extensions"][0]["data"])["source_type"] == "query"
        osi_schema_validator.assert_called_once()

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

    def test_existing_query_backed_dataset_can_be_replaced_with_physical_source(
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

        assert result.success == 1
        assert result.result["updated"] == ["retained_users"]
        authored = yaml.safe_load(target.read_text(encoding="utf-8"))["semantic_model"][0]["datasets"][0]
        assert authored == {"name": "retained_users", "source": "analytics.retained_users"}
        osi_schema_validator.assert_called_once()

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

    def test_delete_osi_metrics_is_scoped_and_tolerates_absent_names(self, tmp_path, osi_schema_validator):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "sales.yml"
        target.parent.mkdir(parents=True)
        target.write_text(
            yaml.safe_dump(
                {
                    "version": "0.2.0.dev0",
                    "semantic_model": [
                        {
                            "name": "sales",
                            "description": "Sales domain",
                            "datasets": [{"name": "orders", "source": "orders"}],
                            "relationships": [{"name": "orders_to_customers"}],
                            "metrics": [
                                _osi_metric("revenue", "SUM(orders.amount)"),
                                _osi_metric("order_count", "COUNT(*)"),
                            ],
                        }
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        before = yaml.safe_load(target.read_text(encoding="utf-8"))
        evidence = GenerationEvidence(validation_passed=True, metric_kb_sync_passed=True)
        evidence.record_semantic_artifact_validation("sales", target)
        state = _bound_state(target)
        tool = MetricFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_metrics",
            osi_target_state=state,
            mutation_callback=evidence.invalidate_artifact_evidence,
        )

        result = tool.delete_osi_metrics(
            str(target.relative_to(tmp_path)),
            ["revenue", "missing_metric", "revenue", ""],
        )

        assert result.success == 1
        assert result.result["requested"] == ["revenue", "missing_metric"]
        assert result.result["deleted"] == ["revenue"]
        assert result.result["already_absent"] == ["missing_metric"]
        assert result.result["remaining"] == ["order_count"]
        after = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert {key: value for key, value in after["semantic_model"][0].items() if key != "metrics"} == {
            key: value for key, value in before["semantic_model"][0].items() if key != "metrics"
        }
        assert state.touched_metric_names == ["revenue", "missing_metric"]
        assert evidence.validation_passed is False
        assert evidence.kb_sync_passed is False
        osi_schema_validator.assert_called_once()

    def test_delete_and_upsert_can_reverse_each_other_in_one_run(self, tmp_path, osi_schema_validator):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "sales.yml"
        target.parent.mkdir(parents=True)
        metric = _osi_metric("revenue", "SUM(orders.amount)")
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
        original = target.read_bytes()
        state = _bound_state(target)
        tool = MetricFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_metrics",
            osi_target_state=state,
        )
        path = str(target.relative_to(tmp_path))

        assert tool.delete_osi_metrics(path, [" Revenue "]).success == 1
        assert state.touched_metric_names == ["Revenue"]

        assert tool.upsert_osi_metrics(path, json.dumps([metric])).success == 1
        assert state.touched_metric_names == ["Revenue"]

        assert tool.delete_osi_metrics(path, ["revenue"]).success == 1
        assert state.touched_metric_names == ["Revenue"]
        assert tool.rollback_failed_authoring() is True
        assert target.read_bytes() == original
        assert state.touched_metric_names == []

    def test_absent_metric_delete_preserves_bytes_but_records_cleanup_scope(self, tmp_path, osi_schema_validator):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "sales.yml"
        target.parent.mkdir(parents=True)
        target.write_text(
            "version: 0.2.0.dev0\nsemantic_model:\n  - name: sales\n    datasets: []\n",
            encoding="utf-8",
        )
        original = target.read_bytes()
        mutation_callback = Mock()
        state = _bound_state(target)
        tool = MetricFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_metrics",
            osi_target_state=state,
            mutation_callback=mutation_callback,
        )

        result = tool.delete_osi_metrics(str(target.relative_to(tmp_path)), ["stale_metric"])

        assert result.success == 1
        assert result.result["already_absent"] == ["stale_metric"]
        assert target.read_bytes() == original
        assert state.touched_metric_names == ["stale_metric"]
        mutation_callback.assert_called_once_with(target.resolve())
        osi_schema_validator.assert_not_called()

    def test_upsert_invalidates_prior_validation_query_sql_and_sync_evidence(self, tmp_path, osi_schema_validator):
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
            metric_sqls={"revenue": "SELECT SUM(amount) FROM orders"},
            semantic_kb_sync_passed=True,
            metric_kb_sync_passed=True,
        )
        evidence.record_semantic_artifact_validation("sales", target)
        tool = MetricFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_metrics",
            osi_target_state=_bound_state(target),
            mutation_callback=evidence.invalidate_artifact_evidence,
        )

        result = tool.upsert_osi_metrics(
            str(target.relative_to(tmp_path)),
            json.dumps([_osi_metric("revenue", "SUM(amount)")]),
        )

        assert result.success == 1
        assert evidence.validation_passed is False
        assert evidence.metric_sqls == {}
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
        state.planned_dataset_names = ["orders"]
        tool = MetricFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_metrics",
            osi_target_state=state,
        )

        result = tool.upsert_osi_metrics(
            str(target.relative_to(tmp_path)),
            json.dumps([_osi_metric("revenue", "SUM(amount)")]),
        )

        assert result.success == 1
        assert target.read_bytes() != original
        assert tool.rollback_failed_authoring() is True
        assert target.read_bytes() == original
        assert state.touched_metric_names == []
        assert state.planned_dataset_names == []
        assert tool.rollback_failed_authoring() is False

    def test_full_authoring_failure_restores_artifact_before_dataset_changes(self, tmp_path, osi_schema_validator):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "sales.yml"
        target.parent.mkdir(parents=True)
        target.write_text(
            "version: 0.2.0.dev0\n"
            "semantic_model:\n"
            "  - name: sales\n"
            "    datasets:\n"
            "      - name: orders\n"
            "        source: orders\n"
            "    metrics: []\n",
            encoding="utf-8",
        )
        original = target.read_bytes()
        state = OsiSemanticModelTargetState()
        state.select(
            {
                "semantic_model_name": "sales",
                "semantic_model_file": str(target.relative_to(tmp_path)),
                "absolute_path": str(target),
                "artifact_sha256": hashlib.sha256(original).hexdigest(),
            },
            mode="planned",
        )

        def mutation_guard(path):
            return state.require_bound_path(path) if state.bound is not None else state.require_planned_path(path)

        def mutation_callback(_path):
            if state.planned is not None:
                state.record_planned_write()

        tool = SemanticModelingFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="semantic_modeling",
            osi_target_state=state,
            mutation_guard=mutation_guard,
            mutation_callback=mutation_callback,
        )
        path = str(target.relative_to(tmp_path))

        assert (
            tool.upsert_osi_datasets(
                path,
                json.dumps(
                    [
                        {"name": "orders", "source": "orders"},
                        {"name": "customers", "source": "customers"},
                    ]
                ),
            ).success
            == 1
        )
        state.select(
            {
                "semantic_model_name": "sales",
                "semantic_model_file": path,
                "absolute_path": str(target),
                "artifact_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            },
            mode="bound",
        )
        assert tool.upsert_osi_metrics(path, json.dumps([_osi_metric("revenue", "SUM(orders.amount)")])).success == 1

        assert tool.rollback_failed_authoring() is True
        assert target.read_bytes() == original
        assert state.touched_metric_names == []
        assert state.touched_dataset_names == []
        assert state.planned_dataset_names == []

    def test_failed_new_model_authoring_removes_created_artifact(self, tmp_path, osi_schema_validator):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "sales.yml"
        state = _planned_state(target)
        tool = SemanticModelingFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="semantic_modeling",
            osi_target_state=state,
            mutation_guard=state.require_planned_path,
            mutation_callback=lambda _path: state.record_planned_write(),
        )

        result = tool.upsert_osi_datasets(
            str(target.relative_to(tmp_path)),
            json.dumps([{"name": "orders", "source": "orders"}]),
        )

        assert result.success == 1
        assert target.is_file()
        assert tool.rollback_failed_authoring() is True
        assert not target.exists()

    def test_semantic_modeling_stamps_query_source_with_runtime_dosi_version(self, tmp_path):
        pytest.importorskip("datus_semantic_dosi")
        tool = SemanticModelingFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="semantic_modeling",
        )

        with patch("datus_semantic_dosi.engine.datus_extension_version", return_value="1.3"):
            extensions = tool._query_source_extensions([])

        assert json.loads(extensions[0]["data"]) == {"source_type": "query", "v": "1.3"}

    def test_failed_metric_authoring_rollback_returns_false_on_invalid_snapshot(self, tmp_path):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "sales.yml"
        target.parent.mkdir(parents=True)
        target.write_text("semantic_model: []\n", encoding="utf-8")
        state = _bound_state(target)
        state.record_artifact_snapshot(target, b"\xff")
        tool = MetricFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_metrics",
            osi_target_state=state,
        )

        assert tool.rollback_failed_authoring() is False
        assert state.artifact_snapshot_content == b"\xff"

    def test_failed_metric_authoring_rollback_returns_false_on_write_error(self, tmp_path, monkeypatch):
        target = tmp_path / "subject" / "semantic_models" / "warehouse" / "sales.yml"
        target.parent.mkdir(parents=True)
        target.write_text("semantic_model: []\n", encoding="utf-8")
        state = _bound_state(target)
        state.record_artifact_snapshot(target, target.read_bytes())
        tool = MetricFilesystemFuncTool(
            root_path=str(tmp_path),
            current_node="gen_metrics",
            osi_target_state=state,
        )
        monkeypatch.setattr(
            "datus.tools.func_tool.metric_filesystem_tools.atomic_write_text",
            Mock(side_effect=OSError("disk full")),
        )

        assert tool.rollback_failed_authoring() is False
        assert state.artifact_snapshot_content == b"semantic_model: []\n"

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
            osi_target_state=state,
            mutation_callback=mutation_callback,
        )

        result = tool.upsert_osi_metrics(str(target.relative_to(tmp_path)), json.dumps([metric]))

        assert result.success == 1
        assert result.result["created"] == []
        assert result.result["updated"] == []
        assert result.result["unchanged"] == ["revenue"]
        assert target.read_bytes() == original_content
        assert state.touched_metric_names == ["revenue"]
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
                osi_target_state=target_state,
            ),
            MetricFilesystemFuncTool(
                root_path=str(tmp_path),
                current_node="gen_metrics",
                osi_target_state=target_state,
            ),
        ]
        relative_path = str(target.relative_to(tmp_path))
        second_started = threading.Event()

        def upsert_from_second_tool():
            second_started.set()
            return tools[1].upsert_osi_metrics(relative_path, json.dumps([_osi_metric("order_count", "COUNT(*)")]))

        with ThreadPoolExecutor(max_workers=2) as executor:
            with semantic_artifact_lock(target):
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
