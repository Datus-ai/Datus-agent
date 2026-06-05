# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datasource-scoped RAG storage namespaces."""

from typing import Any, Dict

import pyarrow as pa

from datus.storage.metric.store import MetricRAG, build_metric_id
from datus.storage.semantic_model.store import SemanticModelRAG

# ---------------------------------------------------------------------------
# Sample data helpers
# ---------------------------------------------------------------------------


def _make_table_object(suffix: str = "a") -> Dict[str, Any]:
    """Return a minimal SemanticModelRAG-compatible table object."""
    return {
        "id": f"table:orders_{suffix}",
        "kind": "table",
        "name": f"orders_{suffix}",
        "fq_name": f"analytics.public.orders_{suffix}",
        "semantic_model_name": f"orders_{suffix}",
        "catalog_name": "default",
        "database_name": "analytics",
        "schema_name": "public",
        "table_name": f"orders_{suffix}",
        "description": f"Order table {suffix}",
        "is_dimension": False,
        "is_measure": False,
        "is_entity_key": False,
        "is_deprecated": False,
        "expr": "",
        "column_type": "",
        "agg": "",
        "create_metric": False,
        "agg_time_dimension": "",
        "is_partition": False,
        "time_granularity": "",
        "entity": "",
        "yaml_path": "",
        "updated_at": pa.scalar(0, type=pa.timestamp("ms")),
    }


def _make_metric(suffix: str = "a") -> Dict[str, Any]:
    """Return a minimal MetricRAG-compatible metric object."""
    return {
        "id": f"metric:total_revenue_{suffix}",
        "subject_path": ["Finance", "Revenue"],
        "name": f"total_revenue_{suffix}",
        "semantic_model_name": "orders",
        "description": f"Total revenue metric {suffix}",
    }


# ---------------------------------------------------------------------------
# PHYSICAL mode tests — shared storage within one process
# ---------------------------------------------------------------------------


class TestRAGPhysicalModeSharedStorage:
    """RAGs can store and retrieve data inside one datasource namespace."""

    def test_semantic_model_store_and_search(self, real_agent_config):
        """SemanticModelRAG can store and retrieve data."""
        rag = SemanticModelRAG(real_agent_config)
        rag.store_batch([_make_table_object("x1"), _make_table_object("x2")])
        results = rag.search_all()
        assert len(results) == 2

    def test_semantic_model_truncate(self, real_agent_config):
        """SemanticModelRAG truncate clears data."""
        rag = SemanticModelRAG(real_agent_config)
        rag.store_batch([_make_table_object("trunc")])
        assert rag.get_size() >= 1
        rag.truncate()
        assert rag.get_size() == 0

    def test_metric_store_and_search(self, real_agent_config):
        """MetricRAG can store and retrieve metrics."""
        rag = MetricRAG(real_agent_config)
        rag.store_batch([_make_metric("m1"), _make_metric("m2")])
        results = rag.search_all_metrics()
        assert len(results) >= 2

    def test_metric_get_size(self, real_agent_config):
        """MetricRAG.get_metrics_size returns correct count."""
        rag = MetricRAG(real_agent_config)
        rag.store_batch([_make_metric("sz1"), _make_metric("sz2")])
        assert rag.get_metrics_size() >= 2

    def test_metric_upsert_batch(self, real_agent_config):
        """MetricRAG.upsert_batch updates existing and inserts new."""
        rag = MetricRAG(real_agent_config)
        m = _make_metric("ups")
        rag.store_batch([m])
        initial_size = rag.get_metrics_size()

        # Upsert same id — should not increase count
        rag.upsert_batch([m])
        assert rag.get_metrics_size() == initial_size


class TestDatasourceScopedNamespaces:
    """Datasource-bound KB stores must not share project-level storage."""

    def _add_datasource_alias(self, agent_config, datasource: str) -> None:
        agent_config.services.datasources[datasource] = agent_config.services.datasources[
            agent_config.current_datasource
        ]

    def test_namespace_helper_sanitizes_project_and_datasource(self, real_agent_config):
        from datus.storage.scope import datasource_storage_namespace

        self._add_datasource_alias(real_agent_config, "Sales DB/2026")
        namespace = datasource_storage_namespace(real_agent_config, "Sales DB/2026")

        assert "__ds__" in namespace
        assert "/" not in namespace
        assert " " not in namespace
        assert "-" not in namespace

    def test_metric_id_does_not_depend_on_subject_path(self):
        assert build_metric_id(["Finance"], "total_revenue") == build_metric_id(["Sales"], "total_revenue")

    def test_rag_wrappers_use_datasource_scoped_namespace(self, real_agent_config):
        from datus.storage.scope import datasource_storage_namespace, project_storage_namespace

        expected_namespace = datasource_storage_namespace(real_agent_config)
        project_namespace = project_storage_namespace(real_agent_config)

        metric_rag = MetricRAG(real_agent_config)
        semantic_rag = SemanticModelRAG(real_agent_config)

        assert metric_rag.storage_namespace == expected_namespace
        assert semantic_rag.storage_namespace == expected_namespace
        assert metric_rag.storage_namespace != project_namespace
        assert semantic_rag.storage_namespace != project_namespace

    def test_metric_storage_isolated_by_datasource(self, real_agent_config):
        ds_a = real_agent_config.current_datasource
        ds_b = "other_datasource"
        self._add_datasource_alias(real_agent_config, ds_b)

        rag_a = MetricRAG(real_agent_config)
        rag_a.store_batch([_make_metric("a")])
        assert rag_a.get_metrics_size() == 1
        assert rag_a.storage.get_subject_tree_flat() == ["Finance", "Finance/Revenue"]

        real_agent_config.current_datasource = ds_b
        rag_b = MetricRAG(real_agent_config)
        assert rag_b.search_all_metrics() == []
        assert rag_b.storage.get_subject_tree_flat() == []

        rag_b.store_batch([_make_metric("b")])
        assert rag_b.get_metrics_size() == 1

        real_agent_config.current_datasource = ds_a
        rag_a_again = MetricRAG(real_agent_config)
        assert [row["name"] for row in rag_a_again.search_all_metrics()] == ["total_revenue_a"]
        rag_a_again.truncate()
        assert rag_a_again.search_all_metrics() == []

        real_agent_config.current_datasource = ds_b
        assert [row["name"] for row in MetricRAG(real_agent_config).search_all_metrics()] == ["total_revenue_b"]

    def test_same_metric_name_can_exist_in_different_datasources(self, real_agent_config):
        ds_a = real_agent_config.current_datasource
        ds_b = "same_metric_other_datasource"
        self._add_datasource_alias(real_agent_config, ds_b)

        metric_a = _make_metric("a")
        metric_a["name"] = "total_revenue"
        metric_a["id"] = build_metric_id(metric_a["subject_path"], metric_a["name"])
        metric_a["measure_expr"] = "SUM(revenue_a)"
        metric_a["base_measures"] = ["revenue_a"]
        metric_b = _make_metric("b")
        metric_b["name"] = "total_revenue"
        metric_b["id"] = build_metric_id(metric_b["subject_path"], metric_b["name"])
        metric_b["measure_expr"] = "SUM(net_revenue)"
        metric_b["base_measures"] = ["net_revenue"]

        MetricRAG(real_agent_config).store_batch([metric_a])

        real_agent_config.current_datasource = ds_b
        MetricRAG(real_agent_config).store_batch([metric_b])

        assert [row["measure_expr"] for row in MetricRAG(real_agent_config).search_all_metrics()] == [
            "SUM(net_revenue)"
        ]

        real_agent_config.current_datasource = ds_a
        assert [row["measure_expr"] for row in MetricRAG(real_agent_config).search_all_metrics()] == ["SUM(revenue_a)"]

    def test_semantic_model_storage_isolated_by_datasource(self, real_agent_config):
        ds_a = real_agent_config.current_datasource
        ds_b = "semantic_other_datasource"
        self._add_datasource_alias(real_agent_config, ds_b)

        rag_a = SemanticModelRAG(real_agent_config)
        rag_a.store_batch([_make_table_object("a")])
        assert [row["table_name"] for row in rag_a.search_all()] == ["orders_a"]

        real_agent_config.current_datasource = ds_b
        rag_b = SemanticModelRAG(real_agent_config)
        assert rag_b.search_all() == []
        rag_b.store_batch([_make_table_object("b")])

        real_agent_config.current_datasource = ds_a
        assert [row["table_name"] for row in SemanticModelRAG(real_agent_config).search_all()] == ["orders_a"]

        real_agent_config.current_datasource = ds_b
        assert [row["table_name"] for row in SemanticModelRAG(real_agent_config).search_all()] == ["orders_b"]
