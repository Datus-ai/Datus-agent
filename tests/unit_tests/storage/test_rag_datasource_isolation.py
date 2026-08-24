# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Datasource row-scope tests for RAG stores.

Physical storage remains project-scoped: every datasource in a project shares
the same vector tables. RAG classes must therefore inject and filter by
``datasource_id`` on every read/write/delete, while ``storage_key`` prevents
same business ids from colliding across datasources.
"""

from typing import Any, Dict

import pyarrow as pa

from datus.storage.metric.store import MetricRAG
from datus.storage.semantic_dataset.store import SemanticDatasetRAG


def _make_table_object(suffix: str = "a", description: str = "") -> Dict[str, Any]:
    """Return a minimal SemanticDatasetRAG-compatible table object."""

    return {
        "id": "dataset:orders:orders",
        "kind": "dataset",
        "semantic_model_name": "orders",
        "dataset_name": "orders",
        "name": "orders",
        "source_table": "orders",
        "source_query": "",
        "catalog_name": "default",
        "database_name": "analytics",
        "schema_name": "public",
        "description": description or f"Order table {suffix}",
        "ai_context_json": "",
        "search_text": description or f"Order table {suffix}",
        "expr": "",
        "field_type": "",
        "is_dimension": False,
        "is_time": False,
        "is_primary_key": False,
        "time_granularity": "",
        "from_dataset": "",
        "to_dataset": "",
        "from_columns_json": "",
        "to_columns_json": "",
        "rel_type": "",
        "join_type": "",
        "yaml_path": "",
        "updated_at": pa.scalar(0, type=pa.timestamp("ms")),
    }


def _make_metric(description: str = "Total revenue metric") -> Dict[str, Any]:
    """Return a minimal MetricRAG-compatible metric object."""

    return {
        "id": "metric:Finance/Revenue.total_revenue",
        "subject_path": ["Finance", "Revenue"],
        "name": "total_revenue",
        "semantic_model_name": "orders",
        "description": description,
    }


def _rag_pair(real_agent_config, rag_cls):
    rag_a = rag_cls(real_agent_config, datasource_id="ds_a")
    rag_b = rag_cls(real_agent_config, datasource_id="ds_b")
    return rag_a, rag_b


class TestRAGDatasourceRowScope:
    def test_semantic_model_same_business_id_isolated_by_datasource(self, real_agent_config):
        rag_a, rag_b = _rag_pair(real_agent_config, SemanticDatasetRAG)

        rag_a.upsert_batch([_make_table_object(description="orders from ds_a")])
        rag_b.upsert_batch([_make_table_object(description="orders from ds_b")])

        results_a = rag_a.list_datasets(table_name="orders")
        results_b = rag_b.list_datasets(table_name="orders")

        assert [row["description"] for row in results_a] == ["orders from ds_a"]
        assert [row["description"] for row in results_b] == ["orders from ds_b"]

    def test_semantic_model_truncate_deletes_only_current_datasource(self, real_agent_config):
        rag_a, rag_b = _rag_pair(real_agent_config, SemanticDatasetRAG)

        rag_a.upsert_batch([_make_table_object(description="orders from ds_a")])
        rag_b.upsert_batch([_make_table_object(description="orders from ds_b")])

        rag_a.truncate()

        assert rag_a.get_size() == 0
        assert [row["description"] for row in rag_b.list_datasets(table_name="orders")] == ["orders from ds_b"]

    def test_metric_same_business_id_isolated_by_datasource(self, real_agent_config):
        rag_a, rag_b = _rag_pair(real_agent_config, MetricRAG)

        rag_a.upsert_batch([_make_metric("metric from ds_a")])
        rag_b.upsert_batch([_make_metric("metric from ds_b")])

        results_a = rag_a.search_all_metrics()
        results_b = rag_b.search_all_metrics()

        assert [row["description"] for row in results_a] == ["metric from ds_a"]
        assert [row["description"] for row in results_b] == ["metric from ds_b"]

    def test_metric_upsert_updates_only_current_datasource(self, real_agent_config):
        rag_a, rag_b = _rag_pair(real_agent_config, MetricRAG)

        rag_a.upsert_batch([_make_metric("metric from ds_a")])
        rag_b.upsert_batch([_make_metric("metric from ds_b")])
        rag_a.upsert_batch([_make_metric("updated metric from ds_a")])

        assert [row["description"] for row in rag_a.search_all_metrics()] == ["updated metric from ds_a"]
        assert [row["description"] for row in rag_b.search_all_metrics()] == ["metric from ds_b"]

    def test_metric_truncate_deletes_only_current_datasource(self, real_agent_config):
        rag_a, rag_b = _rag_pair(real_agent_config, MetricRAG)

        rag_a.upsert_batch([_make_metric("metric from ds_a")])
        rag_b.upsert_batch([_make_metric("metric from ds_b")])

        rag_a.truncate()

        assert rag_a.get_metrics_size() == 0
        assert [row["description"] for row in rag_b.search_all_metrics()] == ["metric from ds_b"]
